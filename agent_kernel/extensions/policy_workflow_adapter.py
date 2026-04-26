from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
from typing import Protocol

from ..actions import CODE_EXECUTE
from ..extensions.policy_command_utils import (
    canonicalize_command as _canonicalize_command,
    normalize_command_for_workspace as _normalize_command_for_workspace,
)
from ..extensions.runtime_modeling_adapter import render_structured_edit_command
from ..extensions.syntax_motor import summarize_python_edit_step
from ..schemas import ActionDecision
from ..state import AgentState
from ..tasking.task_bank import TaskBank


class WorkflowPolicyHost(Protocol):
    def _command_control_score(self, state: AgentState, command: str) -> int:
        ...

    def _command_matches_any_software_work_objective(self, command: str, objectives: list[str]) -> bool:
        ...

    def _first_step_command_covers_required_artifacts(self, state: AgentState, command: str) -> bool:
        ...

    def _software_work_phase_gate_command_score(self, state: AgentState, command: str) -> int:
        ...

    def _subgoal_alignment_score(self, state: AgentState, command: str) -> int:
        ...

    def _normalized_role(self, state: AgentState) -> str:
        ...

    def _policy_control_int(self, field: str) -> int:
        ...

    def _recovery_loop_penalty(self, state: AgentState, command: str) -> int:
        ...


class PolicyWorkflowAdapter:
    def __init__(self, host: WorkflowPolicyHost) -> None:
        self._host = host

    def adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if state.history:
            return None
        if str(state.task.metadata.get("curriculum_kind", "")).strip() != "adjacent_success":
            return None
        for command in state.task.suggested_commands[:2]:
            normalized = str(command).strip()
            if not normalized:
                continue
            if self._host._command_control_score(state, normalized) < 0:
                continue
            if not self._host._first_step_command_covers_required_artifacts(state, normalized):
                continue
            return ActionDecision(
                thought="Use adjacent-success task contract command.",
                action=CODE_EXECUTE,
                content=_normalize_command_for_workspace(normalized, state.task.workspace_subdir),
                done=False,
                decision_source="adjacent_success_direct",
                retrieval_influenced=False,
            )
        return None

    def synthetic_edit_plan_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        edit_plan = state.task.metadata.get("synthetic_edit_plan", [])
        if not isinstance(edit_plan, list) or not edit_plan:
            return None
        if not self.synthetic_edit_plan_direct_active(state):
            return None
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        executed_commands = state.all_executed_command_signatures()
        ranked_steps: list[tuple[tuple[int, int, int, int], int, dict[str, object] | None, dict[str, object]]] = []
        for index, step in enumerate(edit_plan):
            if not isinstance(step, dict):
                continue
            syntax_motor = summarize_python_edit_step(
                step,
                expected_file_contents=state.task.expected_file_contents,
            )
            ranked_steps.append(
                (
                    self.synthetic_edit_step_priority_key(step, syntax_motor=syntax_motor),
                    index,
                    syntax_motor,
                    step,
                )
            )
        for _, _, syntax_motor, step in sorted(ranked_steps, key=lambda item: (item[0], -item[1]), reverse=True):
            command = self.render_synthetic_edit_step_command(step)
            if not command:
                continue
            normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked or canonical in executed_commands:
                continue
            if self._host._command_control_score(state, normalized) < 0:
                continue
            if not state.history and not self._host._first_step_command_covers_required_artifacts(state, normalized):
                continue
            edit_kind = str(step.get("edit_kind", "rewrite")).strip() or "rewrite"
            return ActionDecision(
                thought="Advance the task using the next synthetic edit-plan step.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="synthetic_edit_plan_direct",
                proposal_source=f"structured_edit:{edit_kind}",
                proposal_novel=False,
                proposal_metadata={
                    "edit_kind": edit_kind,
                    "edit_source": "synthetic_edit_plan",
                    "path": str(step.get("path", "")).strip(),
                    "edit_score": int(step.get("edit_score", 0) or 0),
                    "syntax_motor": syntax_motor,
                    "target_symbol_fqn": (
                        str(syntax_motor.get("edited_symbol_fqn", "")).strip()
                        if isinstance(syntax_motor, dict)
                        else ""
                    ),
                    "enclosing_symbol_qualname": (
                        str(syntax_motor.get("enclosing_symbol_qualname", "")).strip()
                        if isinstance(syntax_motor, dict)
                        else ""
                    ),
                    "import_change_risk": bool(syntax_motor.get("import_change_risk", False))
                    if isinstance(syntax_motor, dict)
                    else False,
                    "signature_change_risk": bool(syntax_motor.get("signature_change_risk", False))
                    if isinstance(syntax_motor, dict)
                    else False,
                    "call_targets": list(syntax_motor.get("call_targets_after", []))[:8]
                    if isinstance(syntax_motor, dict)
                    else [],
                },
                retrieval_influenced=False,
            )
        return None

    @staticmethod
    def synthetic_edit_step_priority_key(
        step: dict[str, object],
        *,
        syntax_motor: dict[str, object] | None,
    ) -> tuple[int, int, int, int]:
        edit_score = int(step.get("edit_score", 0) or 0)
        targeted_symbol_bonus = 1 if isinstance(syntax_motor, dict) and syntax_motor.get("edited_symbol_fqn") else 0
        import_risk_penalty = -1 if isinstance(syntax_motor, dict) and syntax_motor.get("import_change_risk") else 0
        signature_risk_penalty = -1 if isinstance(syntax_motor, dict) and syntax_motor.get("signature_change_risk") else 0
        return (
            edit_score + (targeted_symbol_bonus * 4) + (import_risk_penalty * 3) + (signature_risk_penalty * 2),
            targeted_symbol_bonus,
            -1 if import_risk_penalty else 0,
            -1 if signature_risk_penalty else 0,
        )

    def shared_repo_integrator_segment_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if bool(metadata.get("synthetic_worker", False)):
            return None
        if not str(metadata.get("workflow_guard", {})).strip() and not verifier:
            return None
        if int(metadata.get("shared_repo_order", 0) or 0) <= 0:
            return None
        required_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if not required_branches:
            return None
        if not state.task.suggested_commands:
            return None
        primary_command = str(state.task.suggested_commands[0]).strip()
        if " && " not in primary_command:
            return None
        segments = self.shared_repo_integrator_grouped_segments(primary_command)
        if len(segments) < 2:
            return None
        premerge_segment = self.shared_repo_integrator_premerge_segment(
            state,
            segments=segments,
            required_branches=required_branches,
            blocked_commands=blocked_commands,
        )
        if premerge_segment is not None:
            normalized = _normalize_command_for_workspace(
                premerge_segment,
                state.task.workspace_subdir,
            )
            return ActionDecision(
                thought="Prepare the shared-repo integrator workspace before accepting the required worker branch.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="shared_repo_integrator_segment_direct",
                retrieval_influenced=False,
            )
        required_branch_segment = self.shared_repo_required_branch_segment(
            state,
            segments=segments,
            required_branches=required_branches,
        )
        if required_branch_segment is not None:
            normalized = _normalize_command_for_workspace(
                required_branch_segment,
                state.task.workspace_subdir,
            )
            return ActionDecision(
                thought="Advance the shared-repo integrator by accepting the next unresolved required branch.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="shared_repo_integrator_segment_direct",
                retrieval_influenced=False,
            )
        sequential_segment = self.shared_repo_integrator_next_segment(state, segments)
        if sequential_segment is not None:
            normalized = _normalize_command_for_workspace(
                sequential_segment,
                state.task.workspace_subdir,
            )
            control_score = self._host._command_control_score(state, normalized)
            adjusted_control_score = control_score + max(
                0,
                self._host._software_work_phase_gate_command_score(state, normalized),
            )
            if adjusted_control_score >= -8:
                return ActionDecision(
                    thought="Continue the shared-repo integrator through its next workflow segment.",
                    action=CODE_EXECUTE,
                    content=normalized,
                    done=False,
                    decision_source="shared_repo_integrator_segment_direct",
                    retrieval_influenced=False,
                )
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        executed = state.all_successful_command_signatures()
        ranked_segments: list[tuple[tuple[int, int, str], str]] = []
        for segment in segments:
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in blocked or canonical in executed:
                continue
            control_score = self._host._command_control_score(state, normalized)
            adjusted_control_score = control_score + max(
                0,
                self._host._software_work_phase_gate_command_score(state, normalized),
            )
            phase_rank = self.shared_repo_integrator_segment_phase_rank(normalized)
            if adjusted_control_score < -4:
                continue
            ranked_segments.append(((phase_rank, -adjusted_control_score, normalized), normalized))
        if not ranked_segments:
            return None
        ranked_segments.sort(key=lambda item: item[0])
        normalized = ranked_segments[0][1]
        return ActionDecision(
            thought="Advance the shared-repo integrator one workflow segment at a time.",
            action=CODE_EXECUTE,
            content=normalized,
            done=False,
            decision_source="shared_repo_integrator_segment_direct",
            retrieval_influenced=False,
        )

    def shared_repo_worker_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        if not str(workflow_guard.get("worker_branch", "")).strip():
            return None
        if not str(workflow_guard.get("shared_repo_id", "")).strip():
            return None
        if not state.task.suggested_commands:
            return None
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        exhausted = state.all_successful_command_signatures() | state.all_failed_command_signatures()
        for command in state.task.suggested_commands[:2]:
            normalized = _normalize_command_for_workspace(
                str(command).strip(),
                state.task.workspace_subdir,
            )
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in blocked or canonical in exhausted:
                continue
            if self._host._command_control_score(state, normalized) < 0:
                continue
            return ActionDecision(
                thought="Use the shared-repo worker task-contract command.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="shared_repo_worker_direct",
                retrieval_influenced=False,
            )
        return None

    def workspace_contract_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        if not self.workspace_contract_direct_active(state):
            return None
        if not state.task.suggested_commands:
            return None
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        exhausted = state.all_successful_command_signatures() | state.all_failed_command_signatures()
        for command in state.task.suggested_commands[:8]:
            normalized = _normalize_command_for_workspace(
                str(command).strip(),
                state.task.workspace_subdir,
            )
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in blocked or canonical in exhausted:
                continue
            if not self.workspace_contract_command_advances_contract(state, normalized):
                continue
            return ActionDecision(
                thought="Use the exact workspace task-contract command before freeform synthesis.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="workspace_contract_direct",
                retrieval_influenced=False,
            )
        return None

    @staticmethod
    def workspace_contract_direct_active(state: AgentState) -> bool:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        if bool(metadata.get("requires_retrieval", False)):
            return False
        if str(metadata.get("memory_source", "")).strip() == "transition_pressure":
            return False
        if str(metadata.get("curriculum_kind", "")).strip() == "adjacent_success":
            return False
        if bool(metadata.get("synthetic_worker", False)):
            return False
        if isinstance(metadata.get("synthetic_edit_plan"), list) and metadata.get("synthetic_edit_plan"):
            return False
        if int(metadata.get("shared_repo_order", 0) or 0) > 0:
            return False
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        if str(workflow_guard.get("worker_branch", "")).strip():
            return False
        if str(workflow_guard.get("shared_repo_id", "")).strip():
            return False
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if str(verifier.get("kind", "")).strip() == "git_repo_review":
            return False
        if verifier.get("required_merged_branches"):
            return False
        return bool(
            state.task.expected_files
            or state.task.expected_file_contents
            or state.task.forbidden_files
        )

    def workspace_contract_command_advances_contract(
        self,
        state: AgentState,
        command: str,
    ) -> bool:
        normalized = str(command).strip()
        if not normalized or self.workspace_contract_observation_command(normalized):
            return False
        control_score = self._host._command_control_score(state, normalized)
        adjusted_control_score = control_score + max(
            0,
            self._host._software_work_phase_gate_command_score(state, normalized),
        )
        if adjusted_control_score < 0:
            return False
        if self._host._first_step_command_covers_required_artifacts(state, normalized):
            return True
        task_paths = self.workspace_contract_task_paths(state)
        if any(path in normalized for path in task_paths):
            return True
        return self.workspace_contract_preparation_command_mentions_parent(state, normalized)

    @staticmethod
    def workspace_contract_observation_command(command: str) -> bool:
        normalized = str(command).lstrip()
        observation_prefixes = (
            "cat ",
            "grep ",
            "rg ",
            "ls ",
            "find ",
            "test ",
            "[ ",
        )
        return normalized.startswith(observation_prefixes)

    @staticmethod
    def workspace_contract_task_paths(state: AgentState) -> list[str]:
        paths: list[str] = []
        for path in (
            list(state.task.expected_files)
            + list(state.task.expected_file_contents.keys())
            + list(state.task.forbidden_files)
        ):
            normalized = str(path).strip()
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

    def workspace_contract_preparation_command_mentions_parent(
        self,
        state: AgentState,
        command: str,
    ) -> bool:
        normalized = str(command).strip()
        if not normalized.startswith(("mkdir ", "mkdir -p ", "rm ", "rm -f ", "chmod ")):
            return False
        parent_dirs = {
            str(Path(path).parent)
            for path in self.workspace_contract_task_paths(state)
            if str(Path(path).parent) not in {"", "."}
        }
        return any(parent in normalized for parent in parent_dirs)

    @staticmethod
    def shared_repo_unresolved_required_branches(
        state: AgentState,
        *,
        required_branches: list[str],
    ) -> set[str]:
        unresolved = {branch for branch in required_branches if branch}
        if not state.history:
            return unresolved
        unresolved = set()
        verification = state.history[-1].verification if isinstance(state.history[-1].verification, dict) else {}
        for reason in verification.get("reasons", []):
            text = str(reason).strip()
            if not text.startswith("required worker branch not accepted into "):
                continue
            branch = text.rsplit(": ", 1)[-1].strip()
            if branch:
                unresolved.add(branch)
        return unresolved

    def shared_repo_integrator_premerge_segment(
        self,
        state: AgentState,
        *,
        segments: list[str],
        required_branches: list[str],
        blocked_commands: list[str],
    ) -> str | None:
        if not segments or not required_branches:
            return None
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        if not (
            verifier.get("resolved_conflict_paths")
            or verifier.get("generated_paths")
            or bool(workflow_guard.get("touches_generated_paths", False))
        ):
            return None
        unresolved = self.shared_repo_unresolved_required_branches(
            state,
            required_branches=required_branches,
        )
        if not unresolved:
            return None
        blocked = {_canonicalize_command(command) for command in blocked_commands}
        successful = state.all_successful_command_signatures()
        prioritized_paths = [
            str(path).strip()
            for path in (
                list(verifier.get("resolved_conflict_paths", []))
                + list(verifier.get("expected_changed_paths", []))
                + list(verifier.get("generated_paths", []))
                + list(workflow_guard.get("claimed_paths", []))
            )
            if str(path).strip()
        ]
        first_unresolved_merge_index: int | None = None
        for index, segment in enumerate(segments):
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            if "git merge" in normalized and any(branch in normalized for branch in unresolved):
                first_unresolved_merge_index = index
                break
        if first_unresolved_merge_index in {None, 0}:
            return None
        for segment in segments[:first_unresolved_merge_index]:
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in blocked or canonical in successful:
                continue
            if prioritized_paths and not any(path in normalized for path in prioritized_paths):
                continue
            control_score = self._host._command_control_score(state, normalized)
            adjusted_control_score = control_score + max(
                0,
                self._host._software_work_phase_gate_command_score(state, normalized),
            )
            if adjusted_control_score < -8:
                continue
            return segment
        return None

    @staticmethod
    def shared_repo_required_branch_segment(
        state: AgentState,
        *,
        segments: list[str],
        required_branches: list[str],
    ) -> str | None:
        if not segments or not required_branches:
            return None
        unresolved = PolicyWorkflowAdapter.shared_repo_unresolved_required_branches(
            state,
            required_branches=required_branches,
        )
        if not unresolved:
            return None
        successful = state.all_successful_command_signatures()
        for segment in segments:
            normalized = _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            canonical = _canonicalize_command(normalized)
            if canonical in successful:
                continue
            if "git merge" not in normalized:
                continue
            if any(branch in normalized for branch in unresolved):
                return segment
        return None

    @staticmethod
    def shared_repo_integrator_next_segment(
        state: AgentState,
        segments: list[str],
    ) -> str | None:
        if not segments:
            return None
        successful = state.all_successful_command_signatures()
        for segment in segments:
            canonical = _canonicalize_command(
                _normalize_command_for_workspace(segment, state.task.workspace_subdir)
            )
            if canonical not in successful:
                return segment
        return None

    @staticmethod
    def shared_repo_integrator_segment_phase_rank(command: str) -> int:
        normalized = str(command).strip()
        if normalized.startswith("git merge "):
            return 0
        if "reports/" in normalized or normalized.startswith("mkdir -p reports"):
            return 3
        if normalized.startswith("printf ") or normalized.startswith("scripts/") or normalized.startswith("./scripts/"):
            return 1
        if normalized.startswith("tests/") or " tests/" in normalized:
            return 2
        if normalized.startswith("git add ") or normalized.startswith("git commit "):
            if "reports/" in normalized:
                return 3
            return 4
        if normalized.startswith("git diff ") or normalized.startswith("cat "):
            return 5
        if normalized.startswith("git branch "):
            return 6
        return 7

    @classmethod
    def shared_repo_integrator_grouped_segments(cls, command: str) -> list[str]:
        raw_segments = [segment.strip() for segment in str(command).split(" && ") if segment.strip()]
        if len(raw_segments) < 2:
            return raw_segments
        grouped: list[str] = []
        buffer: list[str] = []
        buffer_phase: int | None = None
        for segment in raw_segments:
            phase = cls.shared_repo_integrator_segment_phase_rank(segment)
            starts_new_group = False
            if phase == 0:
                starts_new_group = True
            elif buffer_phase is None:
                starts_new_group = True
            elif phase != buffer_phase:
                if {phase, buffer_phase} <= {3, 4}:
                    starts_new_group = False
                elif (
                    phase == 4
                    and buffer_phase == 1
                    and buffer
                    and str(segment).strip().startswith("git add ")
                ):
                    starts_new_group = False
                else:
                    starts_new_group = True
            elif phase in {3, 4, 5, 6}:
                starts_new_group = False
            if starts_new_group and buffer:
                grouped.append(" && ".join(buffer))
                buffer = []
                buffer_phase = None
            buffer.append(segment)
            buffer_phase = phase if buffer_phase is None else max(buffer_phase, phase)
        if buffer:
            grouped.append(" && ".join(buffer))
        return grouped

    @staticmethod
    def synthetic_edit_plan_direct_active(state: AgentState) -> bool:
        metadata = dict(state.task.metadata)
        if bool(metadata.get("synthetic_worker", False)):
            return True
        difficulty = str(
            metadata.get("difficulty", metadata.get("task_difficulty", ""))
        ).strip().lower()
        if difficulty == "long_horizon":
            return True
        if str(state.world_model_summary.get("horizon", "")).strip() == "long_horizon":
            return True
        return bool(state.active_subgoal) or bool(state.history)

    @staticmethod
    def render_synthetic_edit_step_command(step: dict[str, object]) -> str:
        command = render_structured_edit_command(step)
        if command:
            return command
        path = str(step.get("path", "")).strip()
        target_content = step.get("target_content")
        if not path or target_content is None:
            return ""
        write_commands: list[str] = []
        parent = Path(path).parent
        if str(parent) not in {"", "."}:
            write_commands.append(f"mkdir -p {shlex.quote(str(parent))}")
        write_commands.append(f"printf %s {shlex.quote(str(target_content))} > {shlex.quote(path)}")
        return " && ".join(write_commands)

    def plan_progress_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        ranked = self.rank_plan_progress_candidates(
            state,
            blocked_commands=blocked_commands,
        )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        command = ranked[0][2]
        return ActionDecision(
            thought="Use task-contract command to advance the current subgoal.",
            action=CODE_EXECUTE,
            content=_normalize_command_for_workspace(command, state.task.workspace_subdir),
            done=False,
            decision_source="plan_progress_direct",
            retrieval_influenced=False,
        )

    def rank_plan_progress_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> list[tuple[int, int, str]]:
        if not state.history:
            return []
        if not state.task.suggested_commands:
            return []
        active_subgoal = str(state.active_subgoal or "").strip()
        if (
            not active_subgoal
            and state.consecutive_no_progress_steps <= 0
            and state.consecutive_failures <= 0
            and str(state.world_model_summary.get("horizon", "")).strip() != "long_horizon"
        ):
            return []
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        failed_commands = state.all_failed_command_signatures()
        ranked: list[tuple[int, int, str]] = []
        phase_gate = state.software_work_phase_gate_state()
        gate_objectives = [
            str(item).strip()
            for item in phase_gate.get("gate_objectives", [])
            if str(item).strip()
        ] if isinstance(phase_gate, dict) else []
        concrete_gate_objectives = [
            objective
            for objective in gate_objectives
            if not state._is_generic_contract_subgoal(objective)
        ]
        for command in state.task.suggested_commands[:5]:
            normalized = str(command).strip()
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked:
                continue
            if concrete_gate_objectives and not self._host._command_matches_any_software_work_objective(
                normalized,
                concrete_gate_objectives,
            ):
                continue
            if canonical in failed_commands and (state.consecutive_failures > 0 or state.consecutive_no_progress_steps > 0):
                continue
            control_score = self._host._command_control_score(state, normalized)
            alignment_score = self._host._subgoal_alignment_score(state, normalized)
            alignment_score += self._host._software_work_phase_gate_command_score(state, normalized)
            if active_subgoal and alignment_score <= 0:
                continue
            combined_score = control_score + alignment_score
            if combined_score <= 0:
                continue
            retry_penalty = self._host._recovery_loop_penalty(state, normalized)
            ranked.append((combined_score, -retry_penalty, normalized))
        return ranked

    def recovery_contract_exhausted(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        if self.rank_plan_progress_candidates(state, blocked_commands=blocked_commands):
            return False
        if not state.history or not state.task.suggested_commands:
            return False
        active_subgoal = str(state.active_subgoal or "").strip()
        horizon = str(state.world_model_summary.get("horizon", "")).strip()
        if not active_subgoal and state.consecutive_failures <= 0 and state.consecutive_no_progress_steps <= 0 and horizon != "long_horizon":
            return False
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        failed_commands = state.all_failed_command_signatures()
        last_command = _canonicalize_command(str(state.last_action_signature).partition(":")[2])
        recovery_seen = False
        for command in state.task.suggested_commands[:5]:
            normalized = str(command).strip()
            canonical = _canonicalize_command(normalized)
            if not canonical:
                continue
            recovery_seen = True
            if canonical in normalized_blocked:
                continue
            if canonical in failed_commands:
                continue
            if canonical == last_command and state.repeated_action_count > 1:
                continue
            return False
        return recovery_seen

    def pre_context_adjacent_success_direct_decision(self, state: AgentState) -> ActionDecision | None:
        if state.history:
            return None
        if str(state.task.metadata.get("curriculum_kind", "")).strip() != "adjacent_success":
            return None
        forbidden = {
            str(path).strip()
            for path in state.task.forbidden_files
            if str(path).strip()
        }
        for command in state.task.suggested_commands[:2]:
            normalized = str(command).strip()
            if not normalized:
                continue
            if forbidden and any(path in normalized for path in forbidden):
                continue
            if not self._host._first_step_command_covers_required_artifacts(state, normalized):
                continue
            return ActionDecision(
                thought="Use adjacent-success task contract command before context compilation.",
                action=CODE_EXECUTE,
                content=_normalize_command_for_workspace(normalized, state.task.workspace_subdir),
                done=False,
                decision_source="adjacent_success_direct",
                retrieval_influenced=False,
            )
        return None

    def pre_context_synthetic_edit_plan_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.synthetic_edit_plan_direct_decision(state, blocked_commands=[])

    def pre_context_shared_repo_worker_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.shared_repo_worker_direct_decision(state, blocked_commands=[])

    def pre_context_workspace_contract_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.workspace_contract_direct_decision(
            state,
            blocked_commands=sorted(state.all_failed_command_signatures()),
        )

    def pre_context_shared_repo_integrator_direct_decision(self, state: AgentState) -> ActionDecision | None:
        decision = self.shared_repo_integrator_segment_direct_decision(state, blocked_commands=[])
        if decision is None:
            return None
        if state.history:
            return decision
        return decision if self.shared_repo_integrator_first_segment_safe_without_context(state, decision.content) else None

    def git_repo_review_direct_decision(
        self,
        state: AgentState,
        *,
        blocked_commands: list[str],
    ) -> ActionDecision | None:
        if state.history:
            return None
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        if str(metadata.get("curriculum_kind", "")).strip() == "adjacent_success":
            return None
        if int(metadata.get("shared_repo_order", 0) or 0) > 0:
            return None
        if bool(metadata.get("synthetic_worker", False)):
            return None
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if str(verifier.get("kind", "")).strip() != "git_repo_review":
            return None
        if verifier.get("required_merged_branches"):
            return None
        if not state.task.suggested_commands:
            return None
        normalized_blocked = {_canonicalize_command(command) for command in blocked_commands}
        for command in state.task.suggested_commands[:2]:
            normalized = _normalize_command_for_workspace(
                str(command).strip(),
                state.task.workspace_subdir,
            )
            canonical = _canonicalize_command(normalized)
            if not canonical or canonical in normalized_blocked:
                continue
            if self._host._command_control_score(state, normalized) < 0:
                continue
            if not self.git_repo_review_first_step_safe_without_context(state, normalized):
                continue
            return ActionDecision(
                thought="Use the git repo review task contract command.",
                action=CODE_EXECUTE,
                content=normalized,
                done=False,
                decision_source="git_repo_review_direct",
                retrieval_influenced=False,
            )
        return None

    def pre_context_git_repo_review_direct_decision(self, state: AgentState) -> ActionDecision | None:
        return self.git_repo_review_direct_decision(state, blocked_commands=[])

    def pre_context_plan_progress_direct_decision(self, state: AgentState) -> ActionDecision | None:
        role = self._host._normalized_role(state)
        if role not in {"planner", "critic"}:
            return None
        blocked_commands = sorted(state.all_failed_command_signatures())
        if role == "planner" and self.planner_recovery_rewrite_required(
            state,
            blocked_commands=blocked_commands,
        ):
            return None
        return self.plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )

    def pre_context_trusted_retrieval_carryover_decision(self, state: AgentState) -> ActionDecision | None:
        role = self._host._normalized_role(state)
        if role not in {"planner", "critic", "executor"}:
            return None
        if role == "executor" and not self.trusted_retrieval_executor_direct_allowed(state):
            return None
        if not self.trusted_retrieval_carryover_active(state):
            return None
        candidates = self.trusted_retrieval_carryover_candidates(
            state,
            blocked_commands=state.all_failed_command_signatures(),
        )
        if not candidates:
            return None
        best = candidates[0]
        command = str(best.get("command", "")).strip()
        if not command or int(best.get("control_score", 0) or 0) <= 0:
            return None
        return ActionDecision(
            thought=(
                "Continue a previously trusted retrieval-backed repair sequence."
                if bool(best.get("procedure", False))
                else (
                    "Reuse a previously trusted retrieval-backed write pattern."
                    if bool(best.get("generated", False))
                    else "Reuse a previously trusted retrieval-backed repair command."
                )
            ),
            action=CODE_EXECUTE,
            content=command,
            done=False,
            selected_retrieval_span_id=str(best.get("span_id", "")).strip() or None,
            retrieval_influenced=True,
            decision_source="trusted_retrieval_carryover_direct",
        )

    @staticmethod
    def trusted_retrieval_executor_direct_allowed(state: AgentState) -> bool:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        if bool(metadata.get("requires_retrieval", False)):
            return True
        active_subgoal = str(getattr(state, "active_subgoal", "") or "").strip().lower()
        return active_subgoal.startswith("materialize expected artifact ")

    def pre_context_recovery_exhaustion_decision(self, state: AgentState) -> ActionDecision | None:
        if self._host._normalized_role(state) != "critic":
            return None
        blocked_commands = sorted(state.all_failed_command_signatures())
        if not self.recovery_contract_exhausted(state, blocked_commands=blocked_commands):
            return None
        return ActionDecision(
            thought="No safe task-contract recovery command remains.",
            action="respond",
            content="No safe deterministic recovery command remains.",
            done=True,
        )

    def planner_recovery_rewrite_required(self, state: AgentState, *, blocked_commands: list[str]) -> bool:
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if artifact:
            if str(artifact.get("kind", "")).strip() != "planner_recovery_rewrite":
                return False
            if str(artifact.get("source_subgoal", "")).strip() != str(state.active_subgoal).strip():
                return False
            return True
        if self._host._normalized_role(state) != "planner":
            return False
        diagnosis = state.active_subgoal_diagnosis()
        if str(diagnosis.get("source_role", "")).strip().lower() != "critic":
            return False
        return self.recovery_contract_exhausted(state, blocked_commands=blocked_commands)

    def planner_recovery_rewrite_brief(self, state: AgentState, *, blocked_commands: list[str]) -> str:
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if artifact and self.planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
            rewritten_subgoal = str(artifact.get("rewritten_subgoal", "")).strip()
            next_stage_objective = str(artifact.get("next_stage_objective", "")).strip()
            summary = str(artifact.get("summary", "")).strip()
            contract_outline = [
                str(item).strip()
                for item in artifact.get("contract_outline", [])
                if str(item).strip()
            ]
            stale_commands = [
                str(item).strip()
                for item in artifact.get("stale_commands", [])
                if str(item).strip()
            ]
            related_objectives = [
                str(item).strip()
                for item in artifact.get("related_objectives", [])
                if str(item).strip()
            ]
            parts = []
            if rewritten_subgoal:
                parts.append(f"Critic has exhausted the bounded repair set. New planner objective: {rewritten_subgoal}.")
            if next_stage_objective:
                parts.append(f"Current staged recovery objective: {next_stage_objective}.")
            if summary:
                parts.append(f"Diagnosis: {summary}.")
            if related_objectives:
                parts.append(f"Related verifier obligations: {', '.join(related_objectives[:4])}.")
            if stale_commands:
                parts.append(f"Do not reuse stale repair commands: {', '.join(stale_commands[:3])}.")
            if contract_outline:
                parts.append(f"Rewrite outline: {'; '.join(contract_outline[:3])}.")
            return " ".join(parts).strip()
        if not self.planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
            return ""
        diagnosis = state.active_subgoal_diagnosis()
        active_subgoal = str(state.active_subgoal or "").strip() or "the current verifier-facing subgoal"
        diagnosis_summary = str(diagnosis.get("summary", "")).strip()
        path = str(diagnosis.get("path", "")).strip()
        failed_candidates = [
            str(command).strip()
            for command in state.task.suggested_commands[:5]
            if _canonicalize_command(str(command).strip()) in state.all_failed_command_signatures()
        ]
        exhausted_surface = ", ".join(failed_candidates[:3]) if failed_candidates else "current task-contract repair commands"
        location_clause = f" around {path}" if path else ""
        summary_clause = f" Critic diagnosis: {diagnosis_summary}." if diagnosis_summary else ""
        return (
            f"Critic has exhausted the bounded recovery set for {active_subgoal}{location_clause}.{summary_clause} "
            f"Do not reuse or lightly re-rank the stale repair surface ({exhausted_surface}). "
            "Synthesize a fresh verifier-relevant subgoal or rewrite the recovery contract before choosing the next command."
        )

    @staticmethod
    def software_work_phase_gate_brief(state: AgentState) -> str:
        gate_state = state.software_work_phase_gate_state()
        if not isinstance(gate_state, dict) or not gate_state:
            return ""
        gate_phase = str(gate_state.get("gate_phase", "")).strip()
        gate_reason = str(gate_state.get("gate_reason", "")).strip()
        gate_objectives = [
            str(item).strip()
            for item in gate_state.get("gate_objectives", [])
            if str(item).strip()
        ]
        blocked_phases = [
            str(item).strip()
            for item in gate_state.get("blocked_phases", [])
            if str(item).strip()
        ]
        parts: list[str] = []
        if gate_phase:
            parts.append(f"Current workflow gate phase: {gate_phase}.")
        if gate_reason:
            parts.append(gate_reason)
        if gate_objectives:
            parts.append(f"Resolve these obligations first: {', '.join(gate_objectives[:3])}.")
        if blocked_phases:
            parts.append(f"Do not advance into later phases yet: {', '.join(blocked_phases)}.")
        return " ".join(parts).strip()

    @staticmethod
    def campaign_contract_brief(state: AgentState) -> str:
        contract = state.campaign_contract_state()
        if not isinstance(contract, dict) or not contract:
            return ""
        current_objective = str(contract.get("current_objective", "")).strip()
        anchor_objectives = [
            str(item).strip()
            for item in contract.get("anchor_objectives", [])
            if str(item).strip()
        ]
        regressed_objectives = [
            str(item).strip()
            for item in contract.get("regressed_objectives", [])
            if str(item).strip()
        ]
        stalled_objectives = [
            str(item).strip()
            for item in contract.get("stalled_objectives", [])
            if str(item).strip()
        ]
        parts: list[str] = []
        if current_objective:
            parts.append(f"Current campaign objective: {current_objective}.")
        if anchor_objectives:
            parts.append(f"Unresolved campaign obligations: {', '.join(anchor_objectives[:3])}.")
        if regressed_objectives:
            parts.append(f"Regressed obligations to restore first: {', '.join(regressed_objectives[:2])}.")
        elif stalled_objectives:
            parts.append(f"Stalled obligations still anchoring the run: {', '.join(stalled_objectives[:2])}.")
        return " ".join(parts).strip()

    def shared_repo_integrator_first_segment_safe_without_context(self, state: AgentState, command: str) -> bool:
        normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
        adjusted_control_score = self._host._command_control_score(state, normalized) + max(
            0,
            self._host._software_work_phase_gate_command_score(state, normalized),
        )
        if adjusted_control_score < 0:
            return False
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_branches = {
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        }
        if required_branches and any(branch in normalized for branch in required_branches):
            return True
        prioritized_paths = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if not prioritized_paths:
            prioritized_paths = [
                str(path).strip()
                for path in workflow_guard.get("claimed_paths", [])
                if str(path).strip()
            ]
        if prioritized_paths and any(path in normalized for path in prioritized_paths):
            return True
        return self._host._first_step_command_covers_required_artifacts(state, normalized)

    def git_repo_review_first_step_safe_without_context(self, state: AgentState, command: str) -> bool:
        normalized = _normalize_command_for_workspace(command, state.task.workspace_subdir)
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        if str(verifier.get("kind", "")).strip() != "git_repo_review":
            return False
        expected_branch = str(verifier.get("expected_branch", "")).strip()
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        worker_branch = str(workflow_guard.get("worker_branch", "")).strip()
        if expected_branch and expected_branch != worker_branch and expected_branch not in normalized:
            return False
        prioritized_paths = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if not prioritized_paths:
            prioritized_paths = [
                str(path).strip()
                for path in workflow_guard.get("claimed_paths", [])
                if str(path).strip()
            ]
        if prioritized_paths:
            matched_paths = sum(1 for path in prioritized_paths if path in normalized)
            required_artifact_bias = max(0, self._host._policy_control_int("required_artifact_first_step_bias"))
            minimum_matches = max(1, len(prioritized_paths) - required_artifact_bias)
            if matched_paths < minimum_matches:
                return False
        required_test_commands: list[str] = []
        for test_command in verifier.get("test_commands", []):
            if not isinstance(test_command, dict):
                continue
            argv = test_command.get("argv", [])
            if not isinstance(argv, list) or not argv:
                continue
            entry = str(argv[0]).strip()
            if entry:
                required_test_commands.append(entry)
        if required_test_commands and not all(entry in normalized for entry in required_test_commands):
            return False
        return True

    @staticmethod
    def transition_pressure_task_active(state: AgentState) -> bool:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        benchmark_family = str(metadata.get("benchmark_family", "")).strip().lower()
        memory_source = str(metadata.get("memory_source", "")).strip().lower()
        return benchmark_family == "transition_pressure" or memory_source == "transition_pressure"

    @staticmethod
    def trusted_retrieval_carryover_disabled(state: AgentState) -> bool:
        return PolicyWorkflowAdapter.transition_pressure_task_active(state)

    @staticmethod
    def deterministic_fallback_disabled(state: AgentState) -> bool:
        return PolicyWorkflowAdapter.transition_pressure_task_active(state)

    @staticmethod
    def trusted_retrieval_carryover_active(state: AgentState) -> bool:
        if PolicyWorkflowAdapter.trusted_retrieval_carryover_disabled(state):
            return False
        if not state.history:
            return False
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip().lower()
        horizon = str(state.world_model_summary.get("horizon", "")).strip().lower()
        if difficulty == "long_horizon" or horizon == "long_horizon":
            return True
        return bool(state.active_subgoal or state.subgoal_diagnoses or state.consecutive_failures > 0)

    def trusted_retrieval_carryover_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        if self.trusted_retrieval_carryover_disabled(state):
            return []
        trusted_commands = state.graph_summary.get("trusted_retrieval_command_counts", {})
        if not isinstance(trusted_commands, dict):
            return []
        candidates: list[tuple[tuple[int, int, str], dict[str, object]]] = []
        seen: set[str] = set()
        for command, raw_count in trusted_commands.items():
            command_text = _normalize_command_for_workspace(str(command).strip(), state.task.workspace_subdir)
            canonical = _canonicalize_command(command_text)
            if not canonical or canonical in blocked_commands or canonical in seen:
                continue
            count = self._safe_int(raw_count, 0)
            if count <= 0:
                continue
            if not self.command_matches_current_repair_surface(state, command_text):
                continue
            if self.trusted_retrieval_direct_requires_task_path_match(state) and not self.command_mentions_current_task_paths(
                state, command_text
            ):
                continue
            control_score = self._host._command_control_score(state, command_text)
            total_score = control_score + self.trusted_retrieval_carryover_match_bonus(state, command_text)
            total_score += min(4, count * 2)
            candidates.append(
                (
                    (-total_score, -count, command_text),
                    {
                        "command": command_text,
                        "count": count,
                        "control_score": control_score,
                        "total_score": total_score,
                        "span_id": self.trusted_retrieval_carryover_span_id(command_text),
                        "generated": False,
                    },
                )
            )
            seen.add(canonical)
        for source_candidate in self.trusted_retrieval_source_task_bundle_candidates(
            state,
            blocked_commands=blocked_commands,
        ):
            canonical = _canonicalize_command(str(source_candidate.get("command", "")).strip())
            if canonical and canonical not in seen and canonical not in blocked_commands:
                candidates.append(
                    (
                        (
                            -int(source_candidate.get("total_score", 0) or 0),
                            -int(source_candidate.get("count", 0) or 0),
                            str(source_candidate.get("command", "")),
                        ),
                        source_candidate,
                    )
                )
                seen.add(canonical)
        generated_materialize = self.trusted_retrieval_carryover_materialize_candidate(
            state,
            trusted_commands=trusted_commands,
            blocked_commands=blocked_commands,
        )
        if generated_materialize is not None:
            canonical = _canonicalize_command(str(generated_materialize.get("command", "")).strip())
            if canonical and canonical not in seen and canonical not in blocked_commands:
                candidates.append(
                    (
                        (
                            -int(generated_materialize.get("total_score", 0) or 0),
                            -int(generated_materialize.get("count", 0) or 0),
                            str(generated_materialize.get("command", "")),
                        ),
                        generated_materialize,
                    )
                )
        for procedure_candidate in self.trusted_retrieval_carryover_procedure_candidates(
            state,
            blocked_commands=blocked_commands,
        ):
            canonical = _canonicalize_command(str(procedure_candidate.get("command", "")).strip())
            if canonical and canonical not in seen and canonical not in blocked_commands:
                candidates.append(
                    (
                        (
                            -int(procedure_candidate.get("total_score", 0) or 0),
                            -int(procedure_candidate.get("count", 0) or 0),
                            str(procedure_candidate.get("command", "")),
                        ),
                        procedure_candidate,
                    )
                )
                seen.add(canonical)
        candidates.sort(key=lambda item: item[0])
        return [payload for _, payload in candidates]

    def trusted_retrieval_source_task_bundle_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        source_task_id = str(metadata.get("source_task", "")).strip()
        if not bool(metadata.get("requires_retrieval", False)) or not source_task_id:
            return []
        expected_paths = [
            str(value).strip()
            for value in [*state.task.expected_files, *state.task.expected_file_contents.keys()]
            if str(value).strip()
        ]
        forbidden_paths = [str(value).strip() for value in state.task.forbidden_files if str(value).strip()]
        if len(expected_paths) + len(forbidden_paths) < 4:
            return []
        try:
            host_config = getattr(self._host, "config", None)
            bank = TaskBank(config=host_config) if host_config is not None else TaskBank()
            source_task = bank.get(source_task_id)
        except Exception:
            return []
        candidates: list[tuple[tuple[int, int, str], dict[str, object]]] = []
        seen: set[str] = set()
        source_commands: list[tuple[str, bool]] = [
            (str(command).strip(), False) for command in getattr(source_task, "suggested_commands", [])[:8]
        ]
        source_commands.extend(
            (command, True)
            for command in self.trusted_retrieval_source_task_report_commands(source_task_id)
        )
        for raw_command, runtime_backed in source_commands:
            if not raw_command:
                continue
            command_text = _normalize_command_for_workspace(raw_command, state.task.workspace_subdir)
            canonical = _canonicalize_command(command_text)
            if not canonical or canonical in blocked_commands or canonical in seen:
                continue
            if not self.command_matches_current_repair_surface(state, command_text):
                continue
            expected_hits = sum(1 for path in expected_paths if path and path.lower() in command_text.lower())
            forbidden_hits = sum(1 for path in forbidden_paths if path and path.lower() in command_text.lower())
            cleanup_candidate = self.trusted_retrieval_source_task_cleanup_candidate_allowed(
                state,
                command_text,
                expected_hits=expected_hits,
                forbidden_hits=forbidden_hits,
                runtime_backed=runtime_backed,
            )
            if expected_hits + forbidden_hits < 3 and not cleanup_candidate:
                continue
            if expected_hits <= 1 and self.verification_or_report_command(command_text):
                continue
            control_score = self._host._command_control_score(state, command_text)
            if control_score <= 0:
                continue
            total_score = control_score + 10 + (expected_hits * 2) + (forbidden_hits * 2)
            if runtime_backed:
                total_score += 2
            if cleanup_candidate:
                total_score += 4
            candidates.append(
                (
                    (-total_score, -expected_hits, command_text),
                    {
                        "command": command_text,
                        "count": max(1, expected_hits + forbidden_hits),
                        "control_score": control_score,
                        "total_score": total_score,
                        "span_id": self.trusted_retrieval_source_task_span_id(source_task_id, command_text),
                        "generated": False,
                        "source_task_bundle": True,
                        "source_task_runtime": runtime_backed,
                    },
                )
            )
            seen.add(canonical)
        candidates.sort(key=lambda item: item[0])
        return [payload for _, payload in candidates]

    def trusted_retrieval_source_task_report_commands(self, source_task_id: str) -> list[str]:
        host_config = getattr(self._host, "config", None)
        commands: list[str] = []
        seen: set[str] = set()
        pattern = f"task_report_{source_task_id}_*.json"
        roots = self.trusted_retrieval_source_task_report_roots(host_config)
        for root in roots:
            for path in sorted(root.glob(pattern), reverse=True)[:6]:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(payload.get("report_kind", "")).strip() != "unattended_task_report":
                    continue
                if str(payload.get("task_id", "")).strip() != source_task_id:
                    continue
                if not bool(payload.get("success", False)):
                    continue
                report_commands = payload.get("commands", [])
                if not isinstance(report_commands, list):
                    continue
                for entry in report_commands:
                    if not isinstance(entry, dict):
                        continue
                    command = str(entry.get("command", "")).strip()
                    canonical = _canonicalize_command(command)
                    if not canonical or canonical in seen:
                        continue
                    commands.append(command)
                    seen.add(canonical)
        return commands

    @staticmethod
    def trusted_retrieval_source_task_report_roots(host_config: object) -> list[Path]:
        if host_config is None:
            return []
        roots: list[Path] = []
        seen: set[str] = set()
        run_reports_dir = getattr(host_config, "run_reports_dir", None)
        candidates = [run_reports_dir]
        if run_reports_dir is not None:
            run_reports_path = Path(run_reports_dir)
            if run_reports_path.name != "reports":
                candidates.append(run_reports_path.parent)
        trajectories_root = getattr(host_config, "trajectories_root", None)
        if trajectories_root is not None:
            candidates.append(Path(trajectories_root) / "reports")
        for candidate in candidates:
            if candidate is None:
                continue
            root = Path(candidate)
            if not root.exists():
                continue
            marker = str(root.resolve())
            if marker in seen:
                continue
            roots.append(root)
            seen.add(marker)
        return roots

    @staticmethod
    def trusted_retrieval_source_task_cleanup_candidate_allowed(
        state: AgentState,
        command: str,
        *,
        expected_hits: int,
        forbidden_hits: int,
        runtime_backed: bool,
    ) -> bool:
        if not runtime_backed or forbidden_hits <= 0 or expected_hits > 0:
            return False
        unresolved_expected = {
            str(item).strip()
            for item in state.world_model_summary.get("missing_expected_artifacts", [])
            if str(item).strip()
        }
        unresolved_expected.update(
            str(item).strip()
            for item in state.world_model_summary.get("unsatisfied_expected_contents", [])
            if str(item).strip()
        )
        if unresolved_expected:
            return False
        normalized = str(command).strip().lower()
        return any(str(path).strip().lower() in normalized for path in state.task.forbidden_files)

    def trusted_retrieval_carryover_match_bonus(self, state: AgentState, command: str) -> int:
        trusted_commands = state.graph_summary.get("trusted_retrieval_command_counts", {})
        if not isinstance(trusted_commands, dict):
            return 0
        canonical = _canonicalize_command(command)
        if not canonical:
            return 0
        for candidate, raw_count in trusted_commands.items():
            if canonical != _canonicalize_command(str(candidate)):
                continue
            return min(4, self._safe_int(raw_count, 0) * 2)
        return 0

    def trusted_retrieval_carryover_materialize_candidate(
        self,
        state: AgentState,
        *,
        trusted_commands: dict[str, object],
        blocked_commands: set[str],
    ) -> dict[str, object] | None:
        target = self.trusted_retrieval_materialize_target(state)
        if target is None:
            return None
        path, target_content = target
        if self.trusted_retrieval_materialize_requires_structured_edit(state, path):
            return None
        source_command = ""
        count = 0
        for command, raw_count in sorted(
            trusted_commands.items(),
            key=lambda item: (-self._safe_int(item[1], 0), str(item[0])),
        ):
            normalized_command = _normalize_command_for_workspace(str(command).strip(), state.task.workspace_subdir)
            if not self.trusted_retrieval_supports_materialize_write(normalized_command):
                continue
            candidate_count = self._safe_int(raw_count, 0)
            if candidate_count <= 0:
                continue
            source_command = normalized_command
            count = candidate_count
            break
        if not source_command or count <= 0:
            return None
        command = self.trusted_retrieval_materialize_write_command(path, target_content)
        canonical = _canonicalize_command(command)
        if not canonical or canonical in blocked_commands:
            return None
        if not self.command_matches_current_repair_surface(state, command):
            return None
        control_score = self._host._command_control_score(state, command)
        total_score = control_score + min(3, count * 2) + 2
        return {
            "command": command,
            "count": count,
            "control_score": control_score,
            "total_score": total_score,
            "span_id": self.trusted_retrieval_materialize_span_id(source_command, path),
            "generated": True,
            "source_command": source_command,
        }

    def trusted_retrieval_carryover_procedure_candidates(
        self,
        state: AgentState,
        *,
        blocked_commands: set[str],
    ) -> list[dict[str, object]]:
        procedures = state.graph_summary.get("trusted_retrieval_procedures", {})
        if not isinstance(procedures, list):
            return []
        recent_commands = self.recent_successful_command_suffix(state)
        if not recent_commands:
            return []
        candidates: list[tuple[tuple[int, int, str], dict[str, object]]] = []
        for item in procedures[:4]:
            if not isinstance(item, dict):
                continue
            commands = [
                _normalize_command_for_workspace(str(value).strip(), state.task.workspace_subdir)
                for value in item.get("commands", [])
                if str(value).strip()
            ]
            if len(commands) < 2:
                continue
            matched_prefix = self.trusted_retrieval_procedure_prefix_match(recent_commands, commands)
            if matched_prefix <= 0 or matched_prefix >= len(commands):
                continue
            next_command = commands[matched_prefix]
            canonical = _canonicalize_command(next_command)
            if not canonical or canonical in blocked_commands:
                continue
            control_score = self._host._command_control_score(state, next_command)
            if control_score <= 0 and self.verification_or_report_command(next_command):
                control_score = 1
            if control_score <= 0:
                continue
            if not (
                self.command_matches_current_repair_surface(state, next_command)
                or self.verification_or_report_command(next_command)
            ):
                continue
            count = self._safe_int(item.get("count", 0), 0)
            total_score = control_score + min(6, matched_prefix * 2) + min(4, count * 2)
            candidates.append(
                (
                    (-total_score, -count, next_command),
                    {
                        "command": next_command,
                        "count": count,
                        "control_score": control_score,
                        "total_score": total_score,
                        "span_id": self.trusted_retrieval_procedure_span_id(commands),
                        "generated": False,
                        "procedure": True,
                        "matched_prefix": matched_prefix,
                    },
                )
            )
        candidates.sort(key=lambda item: item[0])
        return [payload for _, payload in candidates]

    @staticmethod
    def trusted_retrieval_materialize_target(state: AgentState) -> tuple[str, str] | None:
        path = ""
        active_subgoal = str(state.active_subgoal or "").strip()
        if active_subgoal.lower().startswith("materialize expected artifact "):
            path = active_subgoal[len("materialize expected artifact ") :].strip()
        if not path:
            diagnosis = state.active_subgoal_diagnosis()
            diagnosis_path = str(diagnosis.get("path", "")).strip()
            if diagnosis_path and diagnosis_path in state.task.expected_file_contents:
                path = diagnosis_path
        if path and path not in state.task.expected_file_contents:
            path = ""
        unsatisfied = {
            str(item).strip()
            for item in state.world_model_summary.get("unsatisfied_expected_contents", [])
            if str(item).strip()
        }
        missing = {
            str(item).strip()
            for item in state.world_model_summary.get("missing_expected_artifacts", [])
            if str(item).strip()
        }
        if not path:
            ordered_expected_paths: list[str] = []
            for value in list(state.task.expected_files) + list(state.task.expected_file_contents.keys()):
                candidate = str(value).strip()
                if not candidate or candidate in ordered_expected_paths:
                    continue
                ordered_expected_paths.append(candidate)
            for candidate in ordered_expected_paths:
                if candidate not in state.task.expected_file_contents:
                    continue
                if (unsatisfied or missing) and candidate not in unsatisfied and candidate not in missing:
                    continue
                path = candidate
                break
        if not path or path not in state.task.expected_file_contents:
            return None
        if (unsatisfied or missing) and path not in unsatisfied and path not in missing:
            return None
        return path, str(state.task.expected_file_contents.get(path, ""))

    @staticmethod
    def trusted_retrieval_materialize_requires_structured_edit(state: AgentState, path: str) -> bool:
        previews = state.world_model_summary.get("workspace_file_previews", {})
        return isinstance(previews, dict) and isinstance(previews.get(path), dict)

    @staticmethod
    def trusted_retrieval_supports_materialize_write(command: str) -> bool:
        normalized = str(command).strip()
        return "printf " in normalized and " > " in normalized and " >> " not in normalized

    @staticmethod
    def trusted_retrieval_materialize_write_command(path: str, target_content: str) -> str:
        commands: list[str] = []
        parent = Path(path).parent
        if str(parent) not in {"", "."}:
            commands.append(f"mkdir -p {shlex.quote(str(parent))}")
        commands.append(f"printf %s {shlex.quote(str(target_content))} > {shlex.quote(path)}")
        return " && ".join(commands)

    @staticmethod
    def trusted_retrieval_materialize_span_id(source_command: str, path: str) -> str:
        digest = hashlib.sha1(f"{source_command}|{path}".encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:materialize:{digest}"

    def command_matches_current_repair_surface(self, state: AgentState, command: str) -> bool:
        normalized = str(command).strip().lower()
        if not normalized:
            return False
        diagnosis = state.active_subgoal_diagnosis()
        path = str(diagnosis.get("path", "")).strip().lower()
        if path and path in normalized:
            return True
        if self._host._subgoal_alignment_score(state, command) > 0:
            return True
        if self.command_mentions_current_task_paths(state, command):
            return True
        return False

    @staticmethod
    def command_mentions_current_task_paths(state: AgentState, command: str) -> bool:
        normalized = str(command).strip().lower()
        if not normalized:
            return False
        for value in (
            list(state.task.expected_files)
            + list(state.task.expected_file_contents.keys())
            + list(state.task.forbidden_files)
        ):
            candidate_path = str(value).strip().lower()
            if candidate_path and candidate_path in normalized:
                return True
        return False

    @staticmethod
    def trusted_retrieval_direct_requires_task_path_match(state: AgentState) -> bool:
        metadata = dict(getattr(state.task, "metadata", {}) or {})
        requires_retrieval = bool(metadata.get("requires_retrieval", False))
        difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip().lower()
        horizon = str(
            getattr(state, "world_model_summary", {}).get(
                "horizon",
                metadata.get("horizon", metadata.get("task_horizon", "")),
            )
        ).strip().lower()
        return requires_retrieval and (horizon == "long_horizon" or difficulty in {"cross_component", "multi_system"})

    @staticmethod
    def recent_successful_command_suffix(state: AgentState) -> list[str]:
        commands: list[str] = []
        for step in state.history[-6:]:
            if str(getattr(step, "action", "")).strip() != CODE_EXECUTE:
                continue
            command = str(getattr(step, "content", "")).strip()
            if not command:
                continue
            command_result = getattr(step, "command_result", {})
            if isinstance(command_result, dict) and not bool(command_result.get("timed_out", False)):
                try:
                    if int(command_result.get("exit_code", 1)) == 0:
                        commands.append(_canonicalize_command(command))
                        continue
                except (TypeError, ValueError):
                    pass
            verification = getattr(step, "verification", {})
            if isinstance(verification, dict) and bool(verification.get("passed", False)):
                commands.append(_canonicalize_command(command))
        return [command for command in commands if command]

    @staticmethod
    def trusted_retrieval_procedure_prefix_match(
        recent_commands: list[str],
        procedure_commands: list[str],
    ) -> int:
        canonical_procedure = [_canonicalize_command(command) for command in procedure_commands]
        canonical_procedure = [command for command in canonical_procedure if command]
        if len(canonical_procedure) < 2:
            return 0
        max_prefix = min(len(recent_commands), len(canonical_procedure) - 1)
        for prefix_len in range(max_prefix, 0, -1):
            if recent_commands[-prefix_len:] == canonical_procedure[:prefix_len]:
                return prefix_len
        return 0

    @staticmethod
    def verification_or_report_command(command: str) -> bool:
        normalized = str(command).strip().lower()
        if not normalized:
            return False
        return any(
            token in normalized
            for token in ("pytest", "unittest", "tox", "nose", " test", "report", "summary", "status")
        )

    @staticmethod
    def trusted_retrieval_carryover_span_id(command: str) -> str:
        digest = hashlib.sha1(str(command).encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:{digest}"

    @staticmethod
    def trusted_retrieval_procedure_span_id(commands: list[str]) -> str:
        digest = hashlib.sha1("||".join(commands).encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:procedure:{digest}"

    @staticmethod
    def trusted_retrieval_source_task_span_id(source_task_id: str, command: str) -> str:
        digest = hashlib.sha1(f"{source_task_id}|{command}".encode("utf-8")).hexdigest()[:12]
        return f"graph:trusted_retrieval:source_task:{digest}"

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
