from __future__ import annotations

from typing import Any

from ..extensions.policy_command_utils import canonicalize_command as _canonicalize_command
from ..extensions.runtime_modeling_adapter import latent_command_bias
from ..extensions.improvement.state_estimation_improvement import state_estimation_policy_bias
from ..extensions.improvement.transition_model_improvement import transition_model_command_pattern
from ..state import AgentState


def command_control_score(policy: Any, state: AgentState, command: str) -> int:
    if policy._is_prohibited_null_command(state, command):
        return -1000
    score = policy._universe_command_score(state, command)
    score += policy.world_model.score_command(state.world_model_summary, command)
    score += policy._transition_model_command_score(state, command)
    score += policy._graph_memory_environment_command_score(state, command)
    if policy._tolbert_model_surfaces().get("latent_state", False):
        score += latent_command_bias(state.latent_state_summary, command)
    score += state_estimation_policy_bias(
        state.latent_state_summary,
        command,
        policy._state_estimation_policy_controls(),
    )
    verifier_alignment_bias = policy._policy_control_int("verifier_alignment_bias")
    role = str(state.current_role or "executor")
    active_subgoal = str(state.active_subgoal or "")
    active_diagnosis = state.active_subgoal_diagnosis()
    score += policy._graph_memory_failure_signal_command_score(
        state,
        command,
        role=role,
    )
    if verifier_alignment_bias:
        if any(path and path in command for path in state.task.expected_files):
            score += verifier_alignment_bias
        if any(path and path in command for path in state.task.expected_file_contents):
            score += verifier_alignment_bias
        if any(path and path in command for path in state.task.forbidden_files):
            score -= verifier_alignment_bias * 2
    score += policy._active_subgoal_diagnosis_command_score(
        state,
        command,
        role=role,
        active_subgoal=active_subgoal,
        diagnosis=active_diagnosis,
    )
    score += policy._planner_recovery_stage_command_score(
        state,
        command,
        role=role,
    )
    score += policy._campaign_contract_command_score(
        state,
        command,
        role=role,
    )
    if role == "planner":
        planner_subgoal_command_bias = policy._policy_control_int("planner_subgoal_command_bias")
        if active_subgoal and any(token in command for token in active_subgoal.split()):
            score += max(2, planner_subgoal_command_bias)
        if "mkdir -p " in command or "cp " in command:
            score += 1
        score -= policy._recovery_loop_penalty(state, command)
    elif role == "critic":
        critic_repeat_failure_bias = policy._policy_control_int("critic_repeat_failure_bias")
        score -= max(
            policy._recovery_loop_penalty(state, command),
            policy._failed_command_attempts(state, command) * max(4, critic_repeat_failure_bias * 2),
        )
        if "rm -rf" in command:
            score -= 4
    elif role == "executor":
        if "printf " in command or "> " in command:
            score += 1
    return score


def campaign_contract_command_score(
    policy: Any,
    state: AgentState,
    command: str,
    *,
    role: str,
) -> int:
    if role not in {"planner", "critic"}:
        return 0
    contract = state.campaign_contract_state()
    if not isinstance(contract, dict) or not contract:
        return 0
    anchors = [
        str(item).strip()
        for item in contract.get("anchor_objectives", [])
        if str(item).strip()
    ]
    if not anchors:
        return 0
    regressed = {
        str(item).strip()
        for item in contract.get("regressed_objectives", [])
        if str(item).strip()
    }
    stalled = {
        str(item).strip()
        for item in contract.get("stalled_objectives", [])
        if str(item).strip()
    }
    for index, objective in enumerate(anchors[:4], start=1):
        if not policy._command_matches_software_work_objective(command, objective):
            continue
        bonus = max(2, 8 - (index * 2))
        if objective in regressed:
            bonus += 2
        elif objective in stalled:
            bonus += 1
        if index == 1 and role == "planner":
            bonus += 1
        return bonus
    governance = policy._simulate_command_governance(state.universe_summary, command)
    action_categories = {
        str(item).strip()
        for item in governance.get("action_categories", [])
        if str(item).strip()
    }
    verification_aligned = bool(governance.get("verification_aligned", False))
    required_paths = {
        str(item).strip().lower()
        for item in contract.get("required_paths", [])
        if str(item).strip()
    }
    normalized_command = str(command).strip().lower()
    if required_paths and any(path in normalized_command for path in required_paths):
        return 2
    if "read_only_discovery" in action_categories or verification_aligned:
        return 0
    drift_pressure = max(0, int(contract.get("drift_pressure", 0) or 0))
    if drift_pressure <= 0:
        return 0
    return -min(8, 2 + drift_pressure)


def graph_memory_environment_command_score(policy: Any, state: AgentState, command: str) -> int:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    universe_summary = state.universe_summary if isinstance(state.universe_summary, dict) else {}
    if not graph_summary or not universe_summary:
        return 0
    governance = policy._simulate_command_governance(universe_summary, command)
    action_categories = {
        str(item).strip()
        for item in governance.get("action_categories", [])
        if str(item).strip()
    }
    risk_flags = {
        str(item).strip()
        for item in governance.get("risk_flags", [])
        if str(item).strip()
    }
    verification_aligned = bool(governance.get("verification_aligned", False))
    novelty = policy._historical_environment_novelty(graph_summary, universe_summary)
    alignment_failures = policy._graph_environment_alignment_failures(graph_summary)
    score = 0

    if novelty > 0:
        if "read_only_discovery" in action_categories:
            score += novelty * 2
        if verification_aligned:
            score += novelty
        score -= novelty * sum(
            1
            for label in risk_flags
            if label
            in {
                "destructive_mutation",
                "git_mutation",
                "network_fetch",
                "remote_execution",
                "workspace_scope_escape",
                "git_write_conflict",
                "network_access_conflict",
                "path_scope_conflict",
            }
        )

    if alignment_failures.get("network_access_aligned", 0) > 0 and (
        "network_fetch" in risk_flags or "network_access_conflict" in risk_flags
    ):
        score -= min(6, alignment_failures["network_access_aligned"] + novelty)
    if alignment_failures.get("git_write_aligned", 0) > 0 and (
        "git_mutation" in risk_flags or "git_write_conflict" in risk_flags
    ):
        score -= min(6, alignment_failures["git_write_aligned"] + novelty)
    if alignment_failures.get("workspace_scope_aligned", 0) > 0 and (
        "workspace_scope_escape" in risk_flags or "path_scope_conflict" in risk_flags
    ):
        score -= min(6, alignment_failures["workspace_scope_aligned"] + novelty)

    if (novelty > 0 or any(alignment_failures.values())) and "read_only_discovery" in action_categories:
        score += 1
    if any(alignment_failures.values()) and verification_aligned:
        score += 1
    return score


def graph_memory_failure_signal_command_score(
    policy: Any,
    state: AgentState,
    command: str,
    *,
    role: str,
) -> int:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    if not graph_summary:
        return 0
    failure_signals = graph_summary.get("failure_signals", {})
    failure_signals = failure_signals if isinstance(failure_signals, dict) else {}
    failure_types = graph_summary.get("failure_types", graph_summary.get("failure_type_counts", {}))
    failure_types = failure_types if isinstance(failure_types, dict) else {}
    no_progress = policy._safe_int(failure_signals.get("no_state_progress", 0), 0)
    regressions = policy._safe_int(failure_signals.get("state_regression", 0), 0)
    command_failures = policy._safe_int(failure_types.get("command_failure", 0), 0)
    alignment_failures = policy._graph_environment_alignment_failures(graph_summary)
    if no_progress <= 0 and regressions <= 0 and command_failures <= 0 and not any(alignment_failures.values()):
        return 0
    governance = policy._simulate_command_governance(state.universe_summary, command)
    action_categories = {
        str(item).strip()
        for item in governance.get("action_categories", [])
        if str(item).strip()
    }
    risk_flags = {
        str(item).strip()
        for item in governance.get("risk_flags", [])
        if str(item).strip()
    }
    verification_aligned = bool(governance.get("verification_aligned", False))
    score = 0
    if role in {"planner", "critic"} and no_progress > 0:
        if "read_only_discovery" in action_categories:
            score += min(4, no_progress)
        if verification_aligned:
            score += min(3, no_progress)
    if command_failures > 0 and verification_aligned:
        score += min(2, command_failures)
    if any(alignment_failures.values()) and (
        "read_only_discovery" in action_categories or verification_aligned
    ):
        score += min(3, sum(alignment_failures.values()))
    if regressions > 0 and risk_flags & {
        "destructive_mutation",
        "git_mutation",
        "network_fetch",
        "remote_execution",
        "workspace_scope_escape",
        "git_write_conflict",
        "network_access_conflict",
        "path_scope_conflict",
    }:
        score -= min(6, regressions * 2)
    return score


def active_subgoal_diagnosis_command_score(
    policy: Any,
    state: AgentState,
    command: str,
    *,
    role: str,
    active_subgoal: str,
    diagnosis: dict[str, object],
) -> int:
    if role not in {"planner", "critic"} or not diagnosis:
        return 0
    path = str(diagnosis.get("path", "")).strip()
    if not path:
        return 0
    normalized_command = str(command).strip().lower()
    normalized_path = path.lower()
    signals = {
        str(signal).strip()
        for signal in diagnosis.get("signals", [])
        if str(signal).strip()
    }
    touches_path = normalized_path in normalized_command
    score = 0
    if touches_path:
        score += 3
        if signals.intersection({"state_regression", "no_state_progress"}):
            score += 3
        if signals.intersection({"command_failure", "command_timeout", "inference_failure", "retrieval_failure"}):
            score += 2
        score += policy._active_subgoal_repair_shape_bonus(
            active_subgoal=active_subgoal,
            command=normalized_command,
            path=normalized_path,
        )
    elif role == "critic" and signals.intersection({"state_regression", "command_failure", "command_timeout"}):
        score -= 2
    failed_commands = state.all_failed_command_signatures()
    canonical = _canonicalize_command(command)
    if role == "critic" and canonical and canonical in failed_commands:
        score -= max(3, policy._policy_control_int("critic_repeat_failure_bias") * 2)
    return score


def transition_model_command_score(policy: Any, state: AgentState, command: str) -> int:
    if not policy.config.use_transition_model_proposals:
        return 0
    normalized = _canonicalize_command(command)
    if not normalized:
        return 0
    task_benchmark_family = str(state.task.metadata.get("benchmark_family", "")).strip()
    task_difficulty = str(
        state.task.metadata.get("difficulty", state.task.metadata.get("task_difficulty", ""))
    ).strip()
    task_horizon = str(state.world_model_summary.get("horizon", "")).strip()
    if not task_difficulty and task_horizon:
        task_difficulty = task_horizon
    is_long_horizon = task_difficulty == "long_horizon" or task_horizon == "long_horizon"
    normalized_pattern = transition_model_command_pattern(normalized) or normalized
    controls = policy._transition_model_controls()
    if not controls:
        return 0
    score = 0
    latest_transition = state.latest_state_transition if isinstance(state.latest_state_transition, dict) else {}
    regressed_paths = {
        str(path).strip()
        for path in latest_transition.get("regressions", [])
        if str(path).strip()
    }
    base_repeat_penalty = policy._transition_model_control_int("repeat_command_penalty", 4)
    base_progress_bonus = policy._transition_model_control_int("progress_command_bonus", 2)
    long_horizon_repeat_penalty = policy._transition_model_control_int("long_horizon_repeat_command_penalty", 1)
    long_horizon_progress_bonus = policy._transition_model_control_int("long_horizon_progress_command_bonus", 1)
    last_command = ""
    if state.history:
        last_command = _canonicalize_command(str(state.history[-1].content))
    last_pattern = transition_model_command_pattern(last_command) if last_command else ""
    cleanup_command = "rm " in normalized or "rm -f " in normalized or "unlink " in normalized
    for signature in policy._transition_model_signatures():
        signature_command = _canonicalize_command(str(signature.get("command", "")))
        if not signature_command:
            continue
        signature_family = str(signature.get("benchmark_family", "")).strip()
        if signature_family and task_benchmark_family and signature_family != task_benchmark_family:
            continue
        signature_difficulty = str(signature.get("difficulty", "")).strip()
        if signature_difficulty and task_difficulty and signature_difficulty != task_difficulty:
            continue
        signature_pattern = (
            transition_model_command_pattern(str(signature.get("command_pattern", "")))
            or transition_model_command_pattern(signature_command)
            or signature_command
        )
        support = max(1, int(signature.get("support", 1)))
        signal = str(signature.get("signal", "")).strip()
        signature_regressions = {
            str(path).strip()
            for path in signature.get("regressions", [])
            if str(path).strip()
        }
        signature_touched_paths = {
            str(path).strip()
            for path in signature.get("touched_paths", [])
            if str(path).strip()
        }
        signature_problem_paths = signature_regressions | signature_touched_paths
        repeat_penalty = base_repeat_penalty
        progress_bonus = base_progress_bonus
        if is_long_horizon:
            repeat_penalty += long_horizon_repeat_penalty
            progress_bonus += long_horizon_progress_bonus
        if normalized == signature_command:
            penalty = repeat_penalty + min(3, support - 1)
            if signal == "state_regression":
                penalty += policy._transition_model_control_int("regressed_path_command_penalty", 3)
            score -= penalty
            continue
        if normalized_pattern == signature_pattern:
            penalty = max(1, repeat_penalty - 2) + min(2, support - 1)
            if signal == "state_regression":
                penalty += max(1, policy._transition_model_control_int("regressed_path_command_penalty", 3) - 1)
            score -= penalty
            continue
        if signature_problem_paths and any(path in normalized for path in signature_problem_paths):
            score -= policy._transition_model_control_int("regressed_path_command_penalty", 3)
        if regressed_paths and signature_problem_paths and regressed_paths.intersection(signature_problem_paths):
            if any(path in normalized for path in regressed_paths):
                score += policy._transition_model_control_int("recovery_command_bonus", 2)
                if cleanup_command:
                    score += 1
    if latest_transition.get("no_progress", False):
        latest_repeat_penalty = base_repeat_penalty + (long_horizon_repeat_penalty if is_long_horizon else 0)
        latest_progress_bonus = base_progress_bonus + (long_horizon_progress_bonus if is_long_horizon else 0)
        if last_command and normalized == last_command:
            score -= latest_repeat_penalty
        elif last_pattern and normalized_pattern == last_pattern:
            score -= max(1, latest_repeat_penalty // 2)
        elif normalized != last_command:
            score += latest_progress_bonus
    return score
