from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import ActionDecision, CommandResult, ContextPacket, StepRecord, TaskSpec

_SOFTWARE_WORK_PHASES = ("implementation", "migration", "test", "follow_up_fix")


def _canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())


def _software_work_phase_rank(phase: str) -> int:
    normalized = str(phase).strip()
    try:
        return _SOFTWARE_WORK_PHASES.index(normalized)
    except ValueError:
        return len(_SOFTWARE_WORK_PHASES)


def _software_work_objective_phase(
    objective: str,
    *,
    test_attempted: bool = False,
    regression_signaled: bool = False,
) -> str:
    normalized = str(objective).strip().lower()
    if not normalized:
        return "implementation"
    if normalized.startswith("run workflow test "):
        return "test"
    if normalized.startswith(("accept required branch ", "prepare workflow branch ", "regenerate generated artifact ")):
        return "migration"
    if normalized.startswith("write workflow report "):
        return "follow_up_fix"
    if normalized.startswith(("revise implementation for ", "remove forbidden artifact ", "preserve required artifact ")):
        if test_attempted or regression_signaled:
            return "follow_up_fix"
        return "implementation"
    if normalized.startswith(
        (
            "materialize expected artifact ",
            "apply planned edit ",
            "complete implementation for ",
            "update workflow path ",
        )
    ):
        return "implementation"
    if regression_signaled and any(
        token in normalized for token in ("fix", "repair", "revise", "restore", "cleanup", "follow-up")
    ):
        return "follow_up_fix"
    return "implementation"


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

    def refresh_plan_progress(self, world_model_summary: dict[str, object]) -> None:
        remaining = [goal for goal in self.plan if not self._subgoal_satisfied(goal, world_model_summary)]
        if self.world_horizon() == "long_horizon":
            remaining = self._reconcile_long_horizon_plan(remaining, world_model_summary)
        self.plan = remaining
        self.active_subgoal = remaining[0] if remaining else ""
        self.subgoal_diagnoses = {
            goal: dict(diagnosis)
            for goal, diagnosis in self.subgoal_diagnoses.items()
            if goal in remaining and isinstance(diagnosis, dict)
        }
        self._refresh_planner_recovery_progress(remaining, world_model_summary)

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
        if self.world_horizon() != "long_horizon":
            return []
        base_objectives = self._software_work_base_objectives()
        stage_state = self.software_work_stage_overview()
        phase_gate = self.software_work_phase_gate_state()
        gate_active = bool(phase_gate.get("active", False))
        gate_phase = str(phase_gate.get("gate_phase", "")).strip()
        gate_status = str(phase_gate.get("gate_status", "")).strip()
        gate_kind = str(phase_gate.get("gate_kind", "")).strip()
        gate_objectives = {
            str(item).strip()
            for item in phase_gate.get("gate_objectives", [])
            if str(item).strip()
        }
        objective_states = stage_state.get("objective_states", {})
        objective_states = objective_states if isinstance(objective_states, dict) else {}
        attempt_counts = stage_state.get("attempt_counts", {})
        attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}
        scored: list[tuple[int, int, str]] = []
        for index, item in enumerate(base_objectives):
            status = str(objective_states.get(item, "pending")).strip() or "pending"
            attempts = max(0, int(attempt_counts.get(item, 0) or 0))
            score = 100 - (index * 5)
            if item == self.active_subgoal.strip():
                score += -12 if status in {"stalled", "regressed"} else 12
            if status == "completed":
                score -= 1000
            elif status == "advanced":
                score += 18
            elif status == "pending":
                score += 8
            elif status == "stalled":
                score -= 18 + (min(3, attempts) * 4)
            elif status == "regressed":
                score -= 26 + (min(3, attempts) * 5)
            if gate_active:
                objective_phase = _software_work_objective_phase(item)
                if item in gate_objectives:
                    if gate_kind == "merge_acceptance" or gate_status != "stalled":
                        score += 36 - (index * 2)
                    else:
                        score -= 8
                elif (
                    gate_phase
                    and _software_work_phase_rank(objective_phase) > _software_work_phase_rank(gate_phase)
                    and (gate_kind == "merge_acceptance" or gate_status != "stalled")
                ):
                    score -= 52
            scored.append((score, index, item))
        ordered: list[str] = []
        for _, _, item in sorted(scored, key=lambda entry: (-entry[0], entry[1], entry[2])):
            if item and item not in ordered:
                ordered.append(item)
        return ordered[:6]

    def campaign_contract_state(self) -> dict[str, object]:
        if self.world_horizon() != "long_horizon":
            return {}
        stage_state = self.software_work_stage_overview()
        phase_gate = self.software_work_phase_gate_state()
        objective_states = stage_state.get("objective_states", {})
        objective_states = objective_states if isinstance(objective_states, dict) else {}
        recent_outcomes = stage_state.get("recent_outcomes", [])
        recent_outcomes = recent_outcomes if isinstance(recent_outcomes, list) else []
        attempt_counts = stage_state.get("attempt_counts", {})
        attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}

        regressed_objectives: list[str] = []
        stalled_objectives: list[str] = []
        for objective in self.software_work_plan_update()[:8]:
            normalized = str(objective).strip()
            if not normalized or self._subgoal_satisfied(normalized, self.world_model_summary):
                continue
            status = str(objective_states.get(normalized, "pending")).strip() or "pending"
            if status == "regressed" and normalized not in regressed_objectives:
                regressed_objectives.append(normalized)
            elif status == "stalled" and normalized not in stalled_objectives:
                stalled_objectives.append(normalized)
        for item in recent_outcomes[-6:]:
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective", "")).strip()
            status = str(item.get("status", "")).strip()
            if not objective or self._subgoal_satisfied(objective, self.world_model_summary):
                continue
            if status == "regressed" and objective not in regressed_objectives:
                regressed_objectives.append(objective)
            elif status == "stalled" and objective not in stalled_objectives:
                stalled_objectives.append(objective)

        current_objective = self.current_software_work_objective()
        gate_objectives = [
            str(item).strip()
            for item in phase_gate.get("gate_objectives", [])
            if str(item).strip() and not self._subgoal_satisfied(str(item).strip(), self.world_model_summary)
        ] if isinstance(phase_gate, dict) else []

        anchor_objectives: list[str] = []
        for objective in [
            *gate_objectives,
            *regressed_objectives,
            current_objective,
            *stalled_objectives,
            *self.software_work_plan_update(),
        ]:
            normalized = str(objective).strip()
            if not normalized or normalized in anchor_objectives:
                continue
            if self._subgoal_satisfied(normalized, self.world_model_summary):
                continue
            anchor_objectives.append(normalized)

        recent_regressions = [
            str(item).strip()
            for item in self.latest_state_transition.get("regressions", [])
            if str(item).strip()
        ]
        required_paths = self._campaign_contract_required_paths(anchor_objectives)
        drift_signals: list[str] = []
        if self.consecutive_failures > 0:
            drift_signals.append("failure_pressure")
        if self.consecutive_no_progress_steps > 0:
            drift_signals.append("no_progress")
        if self.repeated_action_count > 1:
            drift_signals.append("repeat_pressure")
        if regressed_objectives:
            drift_signals.append("regressed_obligation")
        if stalled_objectives:
            drift_signals.append("stalled_obligation")
        if recent_regressions:
            drift_signals.append("state_regression")
        drift_pressure = (
            min(3, int(self.consecutive_failures > 0) + max(0, self.repeated_action_count - 1))
            + min(3, int(self.consecutive_no_progress_steps))
            + min(2, len(regressed_objectives))
            + min(1, len(recent_regressions))
        )
        return {
            "current_objective": current_objective,
            "anchor_objectives": anchor_objectives[:6],
            "regressed_objectives": regressed_objectives[:4],
            "stalled_objectives": stalled_objectives[:4],
            "required_paths": required_paths[:6],
            "phase_gate_active": bool(phase_gate.get("active", False)) if isinstance(phase_gate, dict) else False,
            "gate_phase": str(phase_gate.get("gate_phase", "")).strip() if isinstance(phase_gate, dict) else "",
            "recent_regressions": recent_regressions[:4],
            "drift_signals": drift_signals,
            "drift_pressure": drift_pressure,
            "attempt_counts": {
                objective: max(0, int(attempt_counts.get(objective, 0) or 0))
                for objective in anchor_objectives[:6]
            },
        }

    def software_work_phase_gate_state(self) -> dict[str, object]:
        if self.world_horizon() != "long_horizon":
            return {}
        phase_state = self.software_work_phase_state()
        if not isinstance(phase_state, dict) or not phase_state:
            return {}
        current_phase = str(phase_state.get("current_phase", "")).strip()
        current_phase_status = str(phase_state.get("current_phase_status", "")).strip()
        if not current_phase or current_phase_status in {"", "absent", "completed", "handoff_ready"}:
            return {}
        gate_objectives: list[str] = []
        for objective in self._software_work_base_objectives():
            normalized = str(objective).strip()
            if not normalized:
                continue
            if _software_work_objective_phase(normalized) != current_phase:
                continue
            if self._subgoal_satisfied(normalized, self.world_model_summary):
                continue
            gate_objectives.append(normalized)
        if not gate_objectives:
            return {}
        prioritized: list[str] = []
        for prefix in (
            "accept required branch ",
            "prepare workflow branch ",
            "regenerate generated artifact ",
            "update workflow path ",
        ):
            for objective in gate_objectives:
                if objective.startswith(prefix) and objective not in prioritized:
                    prioritized.append(objective)
        for objective in gate_objectives:
            if objective not in prioritized:
                prioritized.append(objective)
        blocked_phases = [
            phase
            for phase in _SOFTWARE_WORK_PHASES
            if _software_work_phase_rank(phase) > _software_work_phase_rank(current_phase)
        ]
        gate_kind = "merge_acceptance" if any(
            objective.startswith("accept required branch ")
            for objective in prioritized
        ) else "phase_progression"
        gate_reason = (
            "Required branch acceptance remains unresolved before later workflow phases."
            if gate_kind == "merge_acceptance"
            else f"{current_phase} phase objectives remain unresolved before later workflow phases."
        )
        return {
            "active": True,
            "gate_kind": gate_kind,
            "gate_phase": current_phase,
            "gate_status": current_phase_status,
            "gate_reason": gate_reason,
            "gate_objectives": prioritized[:4],
            "blocked_phases": blocked_phases,
        }

    def current_software_work_objective(self) -> str:
        if self.world_horizon() != "long_horizon":
            return ""
        artifact = dict(self.planner_recovery_artifact) if isinstance(self.planner_recovery_artifact, dict) else {}
        next_stage = str(artifact.get("next_stage_objective", "")).strip()
        if next_stage and not self._subgoal_satisfied(next_stage, self.world_model_summary):
            return next_stage
        if self.active_subgoal.strip():
            return self.active_subgoal.strip()
        objectives = self._software_work_base_objectives()
        return objectives[0] if objectives else ""

    def software_work_stage_overview(self) -> dict[str, object]:
        state = dict(self.software_work_stage_state) if isinstance(self.software_work_stage_state, dict) else {}
        objective_states = state.get("objective_states", {})
        attempt_counts = state.get("attempt_counts", {})
        recent_outcomes = state.get("recent_outcomes", [])
        return {
            "current_objective": str(state.get("current_objective", "")).strip(),
            "last_status": str(state.get("last_status", "")).strip(),
            "objective_states": {
                str(key): str(value).strip()
                for key, value in objective_states.items()
                if str(key).strip() and str(value).strip()
            }
            if isinstance(objective_states, dict)
            else {},
            "attempt_counts": {
                str(key): int(value or 0)
                for key, value in attempt_counts.items()
                if str(key).strip()
            }
            if isinstance(attempt_counts, dict)
            else {},
            "recent_outcomes": [
                dict(item)
                for item in recent_outcomes[-6:]
                if isinstance(item, dict)
            ]
            if isinstance(recent_outcomes, list)
            else [],
        }

    def software_work_phase_state(self) -> dict[str, object]:
        if self.world_horizon() != "long_horizon":
            return {}
        overview = self.software_work_stage_overview()
        objective_states = overview.get("objective_states", {})
        objective_states = objective_states if isinstance(objective_states, dict) else {}
        attempt_counts = overview.get("attempt_counts", {})
        attempt_counts = attempt_counts if isinstance(attempt_counts, dict) else {}
        recent_outcomes = overview.get("recent_outcomes", [])
        recent_outcomes = recent_outcomes if isinstance(recent_outcomes, list) else []
        test_attempted = any(
            _software_work_objective_phase(str(item.get("objective", ""))) == "test"
            for item in recent_outcomes
            if isinstance(item, dict)
        )
        regression_signaled = any(
            str(item.get("status", "")).strip() == "regressed"
            for item in recent_outcomes
            if isinstance(item, dict)
        ) or bool(self.latest_state_transition.get("regressed", False))
        phase_objectives: dict[str, list[str]] = {phase: [] for phase in _SOFTWARE_WORK_PHASES}
        for objective in self._software_work_base_objectives():
            phase = _software_work_objective_phase(
                objective,
                test_attempted=test_attempted,
                regression_signaled=regression_signaled,
            )
            if objective and objective not in phase_objectives[phase]:
                phase_objectives[phase].append(objective)
        phase_states: dict[str, dict[str, object]] = {}
        for phase in _SOFTWARE_WORK_PHASES:
            objectives = phase_objectives[phase]
            status_counts = {"pending": 0, "advanced": 0, "stalled": 0, "regressed": 0, "completed": 0}
            phase_attempts = 0
            for objective in objectives:
                status = str(objective_states.get(objective, "pending")).strip() or "pending"
                if status not in status_counts:
                    status = "pending"
                status_counts[status] += 1
                phase_attempts += max(0, int(attempt_counts.get(objective, 0) or 0))
            objective_count = len(objectives)
            if objective_count == 0:
                phase_status = "absent"
            elif status_counts["completed"] >= objective_count:
                phase_status = "completed"
            elif status_counts["completed"] + status_counts["advanced"] >= objective_count:
                phase_status = "handoff_ready"
            elif status_counts["regressed"] > 0:
                phase_status = "regressed"
            elif status_counts["stalled"] > 0:
                phase_status = "stalled"
            elif status_counts["advanced"] > 0:
                phase_status = "advanced"
            else:
                phase_status = "pending"
            phase_states[phase] = {
                "status": phase_status,
                "objective_count": objective_count,
                "attempt_count": phase_attempts,
                "objectives": objectives[:3],
                **status_counts,
            }
        current_phase = ""
        for phase in _SOFTWARE_WORK_PHASES:
            status = str(phase_states[phase].get("status", "")).strip()
            if status and status not in {"absent", "completed"}:
                current_phase = phase
                break
        next_phase = ""
        if current_phase:
            for phase in _SOFTWARE_WORK_PHASES[_software_work_phase_rank(current_phase) + 1 :]:
                status = str(phase_states[phase].get("status", "")).strip()
                if status and status not in {"absent", "completed"}:
                    next_phase = phase
                    break
        current_phase_status = str(phase_states.get(current_phase, {}).get("status", "")).strip() if current_phase else ""
        suggested_phase = current_phase
        if current_phase_status == "handoff_ready" and next_phase:
            suggested_phase = next_phase
        return {
            "current_phase": current_phase,
            "current_phase_status": current_phase_status,
            "next_phase": next_phase,
            "suggested_phase": suggested_phase,
            "handoff_ready": bool(current_phase and current_phase_status == "handoff_ready" and next_phase),
            "phase_states": phase_states,
        }

    def _software_work_base_objectives(self) -> list[str]:
        artifact = dict(self.planner_recovery_artifact) if isinstance(self.planner_recovery_artifact, dict) else {}
        staged = artifact.get("staged_plan_update", [])
        prioritized = [str(item).strip() for item in staged if str(item).strip()] if isinstance(staged, list) else []
        derived: list[str] = []
        if self.active_subgoal.strip():
            derived.append(self.active_subgoal.strip())
        derived.extend(str(item).strip() for item in self.plan if str(item).strip())
        derived.extend(self._pending_synthetic_edit_objectives())
        derived.extend(self._pending_world_software_objectives())
        ordered: list[str] = []
        for item in [*prioritized, *derived]:
            if item and item not in ordered:
                ordered.append(item)
        return ordered

    def _reconcile_long_horizon_plan(
        self,
        remaining: list[str],
        world_model_summary: dict[str, object],
    ) -> list[str]:
        concrete_remaining = [
            goal
            for goal in remaining
            if not self._is_generic_contract_subgoal(goal)
        ]
        generic_remaining = [
            goal
            for goal in remaining
            if self._is_generic_contract_subgoal(goal)
        ]
        gate_state = self.software_work_phase_gate_state()
        gate_objectives = [
            str(item).strip()
            for item in gate_state.get("gate_objectives", [])
            if (
                str(item).strip()
                and not self._is_generic_contract_subgoal(str(item).strip())
                and not self._subgoal_satisfied(str(item).strip(), world_model_summary)
            )
        ] if isinstance(gate_state, dict) else []
        stage_overview = self.software_work_stage_overview()
        objective_states = stage_overview.get("objective_states", {})
        objective_states = objective_states if isinstance(objective_states, dict) else {}
        stalled_or_regressed = [
            objective
            for objective, status in objective_states.items()
            if str(objective).strip()
            and not self._is_generic_contract_subgoal(str(objective).strip())
            and status in {"stalled", "regressed"}
            and not self._subgoal_satisfied(str(objective).strip(), world_model_summary)
        ]
        reconciled: list[str] = []
        for objective in [
            *gate_objectives,
            *stalled_or_regressed,
            *self._artifact_stage_objectives(world_model_summary),
            *self._pending_world_software_objectives(),
            *concrete_remaining,
            *generic_remaining,
        ]:
            normalized = str(objective).strip()
            if (
                not normalized
                or normalized in reconciled
                or self._subgoal_satisfied(normalized, world_model_summary)
                or not self._is_plan_trackable_subgoal(normalized)
            ):
                continue
            reconciled.append(normalized)
        return reconciled

    def _artifact_stage_objectives(self, world_model_summary: dict[str, object]) -> list[str]:
        artifact = dict(self.planner_recovery_artifact) if isinstance(self.planner_recovery_artifact, dict) else {}
        staged = artifact.get("staged_plan_update", [])
        if not isinstance(staged, list):
            return []
        return [
            str(item).strip()
            for item in staged
            if str(item).strip() and not self._subgoal_satisfied(str(item).strip(), world_model_summary)
        ]

    def _refresh_planner_recovery_progress(
        self,
        remaining: list[str],
        world_model_summary: dict[str, object],
    ) -> None:
        artifact = dict(self.planner_recovery_artifact) if isinstance(self.planner_recovery_artifact, dict) else {}
        if not artifact:
            self.planner_recovery_artifact = {}
            return
        staged = artifact.get("staged_plan_update", [])
        unresolved_staged = [
            str(item).strip()
            for item in staged
            if str(item).strip() and not self._subgoal_satisfied(str(item).strip(), world_model_summary)
        ] if isinstance(staged, list) else []
        source_subgoal = str(artifact.get("source_subgoal", "")).strip()
        if not remaining and not unresolved_staged:
            self.planner_recovery_artifact = {}
            return
        if source_subgoal and source_subgoal not in remaining and unresolved_staged:
            artifact["source_subgoal"] = unresolved_staged[0]
        elif source_subgoal and source_subgoal not in remaining:
            self.planner_recovery_artifact = {}
            return
        artifact["staged_plan_update"] = unresolved_staged[:4]
        next_stage = str(artifact.get("next_stage_objective", "")).strip()
        if next_stage and self._subgoal_satisfied(next_stage, world_model_summary):
            next_stage = ""
        if not next_stage and unresolved_staged:
            next_stage = unresolved_staged[0]
        if next_stage:
            artifact["next_stage_objective"] = next_stage
        else:
            artifact.pop("next_stage_objective", None)
        self.planner_recovery_artifact = artifact

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
        normalized = str(objective).strip()
        if self.world_horizon() != "long_horizon" or not normalized:
            return
        status = "stalled"
        if verification_passed or self._subgoal_satisfied(normalized, self.world_model_summary):
            status = "completed"
        elif state_regressed or bool(state_transition.get("regressed", False)) or list(state_transition.get("regressions", [])):
            status = "regressed"
        elif float(progress_delta or 0.0) > 0.0:
            status = "advanced"
        state = dict(self.software_work_stage_state) if isinstance(self.software_work_stage_state, dict) else {}
        objective_states = dict(state.get("objective_states", {})) if isinstance(state.get("objective_states", {}), dict) else {}
        attempt_counts = dict(state.get("attempt_counts", {})) if isinstance(state.get("attempt_counts", {}), dict) else {}
        recent_outcomes = [
            dict(item)
            for item in state.get("recent_outcomes", [])
            if isinstance(item, dict)
        ]
        objective_states[normalized] = status
        attempt_counts[normalized] = int(attempt_counts.get(normalized, 0) or 0) + 1
        recent_outcomes.append(
            {
                "objective": normalized,
                "status": status,
                "step_index": int(step_index),
                "command": str(command).strip(),
                "progress_delta": round(float(progress_delta or 0.0), 4),
                "regressed": bool(state_regressed or state_transition.get("regressed", False)),
            }
        )
        self.software_work_stage_state = {
            "current_objective": normalized,
            "last_status": status,
            "objective_states": objective_states,
            "attempt_counts": attempt_counts,
            "recent_outcomes": recent_outcomes[-8:],
        }

    def _pending_synthetic_edit_objectives(self) -> list[str]:
        metadata = dict(getattr(self.task, "metadata", {}) or {})
        edit_plan = metadata.get("synthetic_edit_plan", [])
        if not isinstance(edit_plan, list):
            return []
        executed = self.all_executed_command_signatures()
        pending: list[str] = []
        for step in edit_plan:
            if not isinstance(step, dict):
                continue
            path = str(step.get("path", "")).strip()
            if not path:
                continue
            if any(path.lower() in signature.lower() for signature in executed):
                continue
            pending.append(f"apply planned edit {path}")
        return pending

    def _pending_world_software_objectives(self) -> list[str]:
        summary = dict(self.world_model_summary or {})
        pending: list[str] = []
        for path in summary.get("missing_expected_artifacts", []):
            normalized = str(path).strip()
            if normalized:
                pending.append(f"complete implementation for {normalized}")
        for path in summary.get("unsatisfied_expected_contents", []):
            normalized = str(path).strip()
            if normalized:
                pending.append(f"revise implementation for {normalized}")
        updated_reports = {str(item).strip() for item in summary.get("updated_report_paths", []) if str(item).strip()}
        for path in summary.get("workflow_report_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in updated_reports:
                pending.append(f"write workflow report {normalized}")
        updated_generated = {str(item).strip() for item in summary.get("updated_generated_paths", []) if str(item).strip()}
        for path in summary.get("workflow_generated_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in updated_generated:
                pending.append(f"regenerate generated artifact {normalized}")
        for branch in summary.get("workflow_required_merges", []):
            normalized = str(branch).strip()
            if normalized:
                pending.append(f"accept required branch {normalized}")
        for branch in summary.get("workflow_branch_targets", []):
            normalized = str(branch).strip()
            if normalized:
                pending.append(f"prepare workflow branch {normalized}")
        for label in summary.get("workflow_required_tests", []):
            normalized = str(label).strip()
            if normalized:
                pending.append(f"run workflow test {normalized}")
        return pending

    def _campaign_contract_required_paths(self, objectives: list[str]) -> list[str]:
        required: list[str] = []
        for objective in objectives:
            normalized = str(objective).strip()
            if not normalized:
                continue
            target = ""
            for prefix in (
                "apply planned edit ",
                "complete implementation for ",
                "revise implementation for ",
                "materialize expected artifact ",
                "remove forbidden artifact ",
                "preserve required artifact ",
                "update workflow path ",
                "regenerate generated artifact ",
                "write workflow report ",
                "accept required branch ",
                "prepare workflow branch ",
                "run workflow test ",
            ):
                if normalized.startswith(prefix):
                    target = normalized.removeprefix(prefix).strip()
                    break
            if target and target not in required:
                required.append(target)
        for key in (
            "missing_expected_artifacts",
            "unsatisfied_expected_contents",
            "present_forbidden_artifacts",
            "changed_preserved_artifacts",
            "workflow_report_paths",
            "workflow_generated_paths",
            "workflow_required_merges",
            "workflow_branch_targets",
            "workflow_required_tests",
        ):
            for value in self.world_model_summary.get(key, []):
                normalized = str(value).strip()
                if normalized and normalized not in required:
                    required.append(normalized)
        return required

    @staticmethod
    def _is_plan_trackable_subgoal(goal: str) -> bool:
        normalized = str(goal).strip()
        if not normalized:
            return False
        return normalized.startswith(
            (
                "prepare workflow branch ",
                "accept required branch ",
                "materialize expected artifact ",
                "complete implementation for ",
                "revise implementation for ",
                "remove forbidden artifact ",
                "preserve required artifact ",
                "update workflow path ",
                "regenerate generated artifact ",
                "write workflow report ",
                "run workflow test ",
                "satisfy verifier contract",
                "check verifier contract before terminating",
                "validate expected artifacts and forbidden artifacts before termination",
                "verify preserved artifacts remain unchanged before termination",
            )
        )

    @staticmethod
    def _is_generic_contract_subgoal(goal: str) -> bool:
        return str(goal).strip() in {
            "satisfy verifier contract",
            "check verifier contract before terminating",
            "validate expected artifacts and forbidden artifacts before termination",
            "verify preserved artifacts remain unchanged before termination",
        }

    def _subgoal_satisfied(self, goal: str, world_model_summary: dict[str, object]) -> bool:
        normalized = str(goal).strip()
        if not normalized:
            return True
        if normalized in {
            "satisfy verifier contract",
            "check verifier contract before terminating",
        }:
            return self._verifier_contract_satisfied(world_model_summary)
        if normalized == "validate expected artifacts and forbidden artifacts before termination":
            return self._validation_contract_satisfied(world_model_summary)
        if normalized == "verify preserved artifacts remain unchanged before termination":
            return self._preservation_contract_satisfied(world_model_summary)
        if normalized.startswith("prepare workflow branch "):
            branch = normalized.removeprefix("prepare workflow branch ").strip()
            return self._recent_command_mentions(branch, prefixes=("git switch", "git checkout"))
        if normalized.startswith("accept required branch "):
            branch = normalized.removeprefix("accept required branch ").strip()
            return self._recent_command_mentions(branch, prefixes=("git merge", "git cherry-pick", "git rebase"))
        if normalized.startswith(
            (
                "materialize expected artifact ",
                "complete implementation for ",
                "revise implementation for ",
            )
        ):
            path = normalized
            for prefix in (
                "materialize expected artifact ",
                "complete implementation for ",
                "revise implementation for ",
            ):
                if path.startswith(prefix):
                    path = path.removeprefix(prefix).strip()
                    break
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
            label = normalized.removeprefix("run workflow test ").strip()
            return self._workflow_test_satisfied(label)
        return False

    @staticmethod
    def _validation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
        return not any(
            list(world_model_summary.get(key, []))
            for key in (
                "missing_expected_artifacts",
                "unsatisfied_expected_contents",
                "present_forbidden_artifacts",
            )
        )

    @staticmethod
    def _preservation_contract_satisfied(world_model_summary: dict[str, object]) -> bool:
        return not any(
            list(world_model_summary.get(key, []))
            for key in (
                "changed_preserved_artifacts",
                "missing_preserved_artifacts",
            )
        )

    def _verifier_contract_satisfied(self, world_model_summary: dict[str, object]) -> bool:
        if not self._validation_contract_satisfied(world_model_summary):
            return False
        if not self._preservation_contract_satisfied(world_model_summary):
            return False
        updated_workflow_paths = {
            str(item).strip() for item in world_model_summary.get("updated_workflow_paths", []) if str(item).strip()
        }
        updated_generated_paths = {
            str(item).strip() for item in world_model_summary.get("updated_generated_paths", []) if str(item).strip()
        }
        updated_report_paths = {
            str(item).strip() for item in world_model_summary.get("updated_report_paths", []) if str(item).strip()
        }
        pending_workflow_paths = [
            str(item).strip()
            for item in world_model_summary.get("workflow_expected_changed_paths", [])
            if str(item).strip() and str(item).strip() not in updated_workflow_paths
        ]
        pending_generated_paths = [
            str(item).strip()
            for item in world_model_summary.get("workflow_generated_paths", [])
            if str(item).strip() and str(item).strip() not in updated_generated_paths
        ]
        pending_report_paths = [
            str(item).strip()
            for item in world_model_summary.get("workflow_report_paths", [])
            if str(item).strip() and str(item).strip() not in updated_report_paths
        ]
        pending_merges = [
            str(item).strip()
            for item in world_model_summary.get("workflow_required_merges", [])
            if str(item).strip() and not self._subgoal_satisfied(f"accept required branch {str(item).strip()}", world_model_summary)
        ]
        pending_branches = [
            str(item).strip()
            for item in world_model_summary.get("workflow_branch_targets", [])
            if str(item).strip() and not self._subgoal_satisfied(f"prepare workflow branch {str(item).strip()}", world_model_summary)
        ]
        pending_tests = [
            str(item).strip()
            for item in world_model_summary.get("workflow_required_tests", [])
            if str(item).strip() and not self._subgoal_satisfied(f"run workflow test {str(item).strip()}", world_model_summary)
        ]
        return not any(
            (
                pending_workflow_paths,
                pending_generated_paths,
                pending_report_paths,
                pending_merges,
                pending_branches,
                pending_tests,
            )
        )

    def _workflow_test_satisfied(self, label: str) -> bool:
        normalized_label = str(label).strip().lower()
        if not normalized_label:
            return False
        verifier = self.task.metadata.get("semantic_verifier", {})
        verifier = verifier if isinstance(verifier, dict) else {}
        tokens = self._workflow_test_match_tokens(normalized_label, verifier)
        for step in reversed(self.history):
            if not bool(step.verification.get("passed", False)):
                continue
            command = " ".join(
                value.strip().lower()
                for value in (
                    str(step.content),
                    str(step.command_result.get("command", "")) if isinstance(step.command_result, dict) else "",
                )
                if value.strip()
            )
            if not command:
                continue
            if normalized_label in command:
                return True
            if tokens and all(token in command for token in tokens):
                return True
        return False

    @staticmethod
    def _workflow_test_match_tokens(label: str, verifier: dict[str, object]) -> list[str]:
        for rule in verifier.get("test_commands", []):
            if not isinstance(rule, dict):
                continue
            candidate_label = str(rule.get("label", "")).strip().lower()
            if candidate_label != label:
                continue
            argv = rule.get("argv", [])
            if not isinstance(argv, list):
                return []
            tokens = [
                normalized
                for value in argv
                if (normalized := str(value).strip().lower()) and not normalized.startswith("-")
            ]
            return tokens
        return []

    def _recent_command_mentions(self, token: str, *, prefixes: tuple[str, ...] = ()) -> bool:
        normalized_token = str(token).strip().lower()
        if not normalized_token:
            return False
        normalized_prefixes = tuple(str(prefix).strip().lower() for prefix in prefixes if str(prefix).strip())
        for step in reversed(self.history[-6:]):
            command = str(step.content).strip().lower()
            if not command or normalized_token not in command:
                continue
            if normalized_prefixes and not any(command.startswith(prefix) for prefix in normalized_prefixes):
                continue
            return True
        return False

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
