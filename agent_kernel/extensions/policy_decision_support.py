from __future__ import annotations

from ..actions import CODE_EXECUTE
from ..extensions.policy_runtime_support import SkillLibrary
from ..schemas import ActionDecision


def deterministic_role_decision(
    policy,
    state,
    *,
    role: str,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    tolbert_mode: str,
    retrieval_has_signal: bool,
    tolbert_route_mode: str,
) -> ActionDecision | None:
    if role == "planner":
        decision = planner_direct_decision(
            policy,
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
        )
        if decision is not None:
            return decision
    elif role == "critic":
        decision = critic_direct_decision(
            policy,
            state,
            top_skill=top_skill,
            retrieval_guidance=retrieval_guidance,
            blocked_commands=blocked_commands,
            tolbert_mode=tolbert_mode,
            retrieval_has_signal=retrieval_has_signal,
            tolbert_route_mode=tolbert_route_mode,
        )
        if decision is not None:
            return decision
    return executor_direct_decision(
        policy,
        state,
        top_skill=top_skill,
        retrieval_guidance=retrieval_guidance,
        blocked_commands=blocked_commands,
        tolbert_mode=tolbert_mode,
        retrieval_has_signal=retrieval_has_signal,
        tolbert_route_mode=tolbert_route_mode,
    )


def planner_direct_decision(
    policy,
    state,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    tolbert_mode: str,
    retrieval_has_signal: bool,
    tolbert_route_mode: str,
) -> ActionDecision | None:
    del tolbert_mode, retrieval_has_signal
    integrator_segment_decision = policy._shared_repo_integrator_segment_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if integrator_segment_decision is not None:
        return integrator_segment_decision
    git_repo_review_decision = policy._git_repo_review_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if git_repo_review_decision is not None:
        return git_repo_review_decision
    adjacent_success_decision = policy._adjacent_success_direct_decision(state)
    if adjacent_success_decision is not None:
        return adjacent_success_decision
    synthetic_edit_plan_decision = policy._synthetic_edit_plan_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if synthetic_edit_plan_decision is not None:
        return synthetic_edit_plan_decision
    if policy._planner_recovery_rewrite_required(state, blocked_commands=blocked_commands):
        return None
    plan_progress_decision = policy._plan_progress_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if plan_progress_decision is not None:
        return plan_progress_decision
    decision = policy._tolbert_direct_decision(
        state,
        top_skill=top_skill,
        recommended_commands=retrieval_guidance.get("recommended_command_spans", []),
        blocked_commands=blocked_commands,
        route_mode=tolbert_route_mode,
    )
    if decision is None:
        return None
    if policy._command_control_score(state, decision.content) <= 0 and state.active_subgoal:
        return None
    return decision


def critic_direct_decision(
    policy,
    state,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    tolbert_mode: str,
    retrieval_has_signal: bool,
    tolbert_route_mode: str,
) -> ActionDecision | None:
    del top_skill, retrieval_guidance, tolbert_mode, retrieval_has_signal, tolbert_route_mode
    integrator_segment_decision = policy._shared_repo_integrator_segment_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if integrator_segment_decision is not None:
        return integrator_segment_decision
    git_repo_review_decision = policy._git_repo_review_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if git_repo_review_decision is not None:
        return git_repo_review_decision
    horizon = str(
        state.world_model_summary.get(
            "horizon",
            state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
        )
    ).strip()
    if state.active_subgoal_diagnosis() or (state.active_subgoal and horizon == "long_horizon"):
        plan_progress_decision = policy._plan_progress_direct_decision(
            state,
            blocked_commands=blocked_commands,
        )
        if plan_progress_decision is not None:
            return plan_progress_decision
    if state.context_packet is not None and policy._has_retrieval_signal(state) and state.history and state.consecutive_failures > 0:
        return ActionDecision(
            thought="Pause unsafe repetition and hand control back to synthesis.",
            action="respond",
            content="No safe deterministic command remains.",
            done=True,
        )
    if policy._recovery_contract_exhausted(state, blocked_commands=blocked_commands):
        return ActionDecision(
            thought="No safe task-contract recovery command remains.",
            action="respond",
            content="No safe deterministic recovery command remains.",
            done=True,
        )
    return None


def executor_direct_decision(
    policy,
    state,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    tolbert_mode: str,
    retrieval_has_signal: bool,
    tolbert_route_mode: str,
) -> ActionDecision | None:
    del tolbert_mode, retrieval_has_signal
    integrator_segment_decision = policy._shared_repo_integrator_segment_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if integrator_segment_decision is not None:
        return integrator_segment_decision
    git_repo_review_decision = policy._git_repo_review_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if git_repo_review_decision is not None:
        return git_repo_review_decision
    adjacent_success_decision = policy._adjacent_success_direct_decision(state)
    if adjacent_success_decision is not None:
        return adjacent_success_decision
    synthetic_edit_plan_decision = policy._synthetic_edit_plan_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if synthetic_edit_plan_decision is not None:
        return synthetic_edit_plan_decision
    plan_progress_decision = policy._plan_progress_direct_decision(
        state,
        blocked_commands=blocked_commands,
    )
    if plan_progress_decision is not None:
        return plan_progress_decision
    return policy._tolbert_direct_decision(
        state,
        top_skill=top_skill,
        recommended_commands=retrieval_guidance.get("recommended_command_spans", []),
        blocked_commands=blocked_commands,
        route_mode=tolbert_route_mode,
    )


def followup_skill_decision(
    policy,
    state,
    *,
    role: str,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    tolbert_mode: str,
    retrieval_has_signal: bool,
) -> ActionDecision | None:
    if role == "critic":
        return None
    if role == "planner" and policy._planner_recovery_rewrite_required(
        state,
        blocked_commands=policy._blocked_commands(state),
    ):
        return None
    require_active_skill_ranking = (
        not state.history
        and retrieval_has_signal
        and policy._tolbert_skill_ranking_enabled()
        and not policy._tolbert_skill_ranking_active(state)
    )
    if not policy._skill_is_safe_for_task(state, top_skill):
        top_skill = None
    if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
        skill_commands = policy.skill_library._commands_for_skill(top_skill)
        if skill_commands and policy._command_control_score(state, skill_commands[0]) < 0:
            top_skill = None
    if require_active_skill_ranking:
        top_skill = None
    if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
        skill_commands = policy.skill_library._commands_for_skill(top_skill)
        if skill_commands and not first_step_command_covers_required_artifacts(policy, state, skill_commands[0]):
            top_skill = None
    if not state.history and top_skill is not None and tolbert_mode in {"full", "skill_ranking", "disabled"}:
        matched_span_id = policy._matching_retrieval_span_id(
            policy.skill_library._commands_for_skill(top_skill)[0],
            retrieval_guidance.get("recommended_command_spans", []),
        )
        return policy._skill_action_decision(
            state,
            top_skill,
            thought_prefix="Use Tolbert-ranked skill" if policy._tolbert_skill_ranking_enabled() else "Use skill",
            retrieval_influenced=retrieval_has_signal and policy._tolbert_influence_enabled(),
            retrieval_ranked_skill=retrieval_has_signal and policy._tolbert_skill_ranking_enabled(),
            selected_retrieval_span_id=matched_span_id,
        )
    return None


def best_deterministic_fallback_decision(
    policy,
    state,
    *,
    top_skill: dict[str, object] | None,
    retrieval_guidance: dict[str, list[str]],
    blocked_commands: list[str],
    allow_partial_first_step: bool = False,
) -> ActionDecision | None:
    metadata = dict(state.task.metadata) if isinstance(getattr(state.task, "metadata", {}), dict) else {}
    if (
        not allow_partial_first_step
        and not state.history
        and str(metadata.get("curriculum_kind", "")).strip() == "failure_recovery"
        and bool(metadata.get("contract_clean_failure_recovery_origin", False))
        and bool(metadata.get("decision_yield_contract_candidate", False))
    ):
        return None
    candidates = policy._tolbert_ranked_candidates(
        state,
        top_skill=top_skill if policy._skill_is_safe_for_task(state, top_skill) else None,
        retrieval_guidance=retrieval_guidance,
        blocked_commands=blocked_commands,
    )
    for candidate in candidates:
        command = str(candidate.get("command", "")).strip()
        if not command:
            continue
        if int(candidate.get("score", 0)) <= 0:
            continue
        if (
            not allow_partial_first_step
            and not first_step_command_covers_required_artifacts(policy, state, command)
        ):
            continue
        decision_source = (
            "trusted_retrieval_carryover_direct"
            if str(candidate.get("reason", "")).strip() == "trusted retrieval carryover"
            else "deterministic_fallback"
        )
        return ActionDecision(
            thought="Use deterministic fallback after inference failure.",
            action=CODE_EXECUTE,
            content=command,
            done=False,
            selected_retrieval_span_id=str(candidate.get("span_id", "")).strip() or None,
            retrieval_influenced=bool(candidate.get("retrieval_influenced", False)),
            retrieval_ranked_skill=bool(candidate.get("retrieval_ranked_skill", False)),
            decision_source=decision_source,
        )
    return None


def first_step_command_covers_required_artifacts(policy, state, command: str) -> bool:
    if state.history:
        return True
    required_paths = required_first_step_artifacts(state)
    if not required_paths:
        return True
    normalized = SkillLibrary._normalize_command(command)
    if all(path in normalized for path in required_paths):
        return True
    required_artifact_bias = policy._policy_control_int("required_artifact_first_step_bias")
    if required_artifact_bias <= 0:
        return False
    matched_paths = sum(1 for path in required_paths if path in normalized)
    return matched_paths >= max(1, len(required_paths) - required_artifact_bias)


def subgoal_alignment_score(state, command: str) -> int:
    active_subgoal = str(state.active_subgoal or "").strip().lower()
    normalized_command = str(command).strip().lower()
    score = 0
    if active_subgoal:
        for token in active_subgoal.split():
            cleaned = token.strip(".,:;()[]{}\"'")
            if len(cleaned) < 3:
                continue
            if cleaned in normalized_command:
                score += 2
    diagnosis = state.active_subgoal_diagnosis()
    path = str(diagnosis.get("path", "")).strip().lower()
    if path and path in normalized_command:
        score += 4
        signals = {
            str(signal).strip()
            for signal in diagnosis.get("signals", [])
            if str(signal).strip()
        }
        if signals.intersection({"state_regression", "command_failure", "command_timeout", "no_state_progress"}):
            score += 2
    return score


def required_first_step_artifacts(state) -> list[str]:
    metadata = dict(getattr(state.task, "metadata", {}) or {})
    if bool(metadata.get("synthetic_worker", False)):
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        workflow_guard = metadata.get("workflow_guard", {})
        workflow_guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
        prioritized = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
        if not prioritized:
            prioritized = [
                str(path).strip()
                for path in workflow_guard.get("claimed_paths", [])
                if str(path).strip()
            ]
        if prioritized:
            return prioritized
    setup_commands = [str(command) for command in state.task.setup_commands]
    required: list[str] = []
    for path in state.task.expected_files:
        normalized = str(path).strip()
        if not normalized:
            continue
        if any(normalized in command for command in setup_commands):
            continue
        required.append(normalized)
    return required


def transition_preview(policy, state) -> dict[str, object]:
    preview_summary = dict(state.world_model_summary)
    if not preview_summary.get("semantic_episodes"):
        graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
        semantic_episodes = graph_summary.get("semantic_episodes", [])
        if isinstance(semantic_episodes, list) and semantic_episodes:
            preview_summary["semantic_episodes"] = [
                dict(item)
                for item in semantic_episodes[:4]
                if isinstance(item, dict)
            ]
    if not preview_summary.get("semantic_prototypes"):
        graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
        semantic_prototypes = graph_summary.get("semantic_prototypes", [])
        if isinstance(semantic_prototypes, list) and semantic_prototypes:
            preview_summary["semantic_prototypes"] = [
                dict(item)
                for item in semantic_prototypes[:4]
                if isinstance(item, dict)
            ]
    candidates: list[str] = []
    subgoal_plan = _hierarchical_subgoal_plan(policy, state, preview_summary)
    retrieval_guidance = policy._retrieval_guidance(state)
    for item in retrieval_guidance.get("recommended_command_spans", [])[:4]:
        command = str(item.get("command", "")).strip()
        if command and command not in candidates:
            candidates.append(command)
    for command in state.task.suggested_commands[:5]:
        command = str(command).strip()
        if command and command not in candidates:
            candidates.append(command)
    for command in _semantic_memory_preview_candidates(state):
        if command and command not in candidates:
            candidates.append(command)
    for command in _semantic_prototype_preview_candidates(state):
        if command and command not in candidates:
            candidates.append(command)
    scored_candidates: list[tuple[tuple[int, int, str], str]] = []
    previews = []
    for command in candidates:
        effect = policy.world_model.simulate_command_effect(preview_summary, command)
        governance = policy._simulate_command_governance(state.universe_summary, command)
        preview_score = int(effect.get("score", 0) or 0)
        preview_score += int(effect.get("predicted_progress_gain", 0) or 0)
        preview_score += int(effect.get("predicted_verifier_delta", 0) or 0)
        preview_score += int(round(float(effect.get("latent_transition_score", 0.0) or 0.0)))
        preview_score -= _semantic_failure_repeat_penalty(effect)
        if str(governance.get("decision", "")).strip().lower() == "allow":
            preview_score += 1
        else:
            preview_score -= 3
        subgoal_bonus, best_subgoal = _best_matching_subgoal(command, subgoal_plan)
        preview_score += subgoal_bonus
        scored_candidates.append(((-preview_score, -len(command), command), command))
        previews.append(
            {
                "command": command,
                **effect,
                "governance": governance,
                "preview_score": preview_score,
                "hierarchical_bonus": subgoal_bonus,
                "best_subgoal": best_subgoal,
            }
        )
    preview_limit = 5 if _preview_frontier_should_expand(state) else 3
    search_branches = _adaptive_transition_search(
        policy,
        state,
        preview_summary=preview_summary,
        previews=previews,
        max_depth=3 if _preview_frontier_should_expand(state) else 2,
        subgoal_plan=subgoal_plan,
    )
    branch_command_scores: dict[str, float] = {}
    for branch in search_branches:
        sequence = branch.get("sequence", [])
        if not isinstance(sequence, list) or not sequence:
            continue
        command = str(sequence[0]).strip()
        if not command:
            continue
        branch_command_scores[command] = max(
            branch_command_scores.get(command, float("-inf")),
            float(branch.get("normalized_score", branch.get("cumulative_score", 0.0)) or 0.0),
        )
    for preview in previews:
        command = str(preview.get("command", "")).strip()
        failure_repeat_penalty = float(_semantic_failure_repeat_penalty(preview))
        if command in branch_command_scores:
            ranked_search_score = branch_command_scores[command] - failure_repeat_penalty
            preview["branch_rank_score"] = round(ranked_search_score, 4)
            preview["search_score"] = round(
                max(
                    ranked_search_score,
                    float(preview.get("preview_score", 0.0) or 0.0),
                ),
                4,
            )
        else:
            preview["branch_rank_score"] = float("-inf")
            preview["search_score"] = float(preview.get("preview_score", 0) or 0)
    selected_commands = {
        str(preview.get("command", "")).strip()
        for preview in sorted(
            previews,
            key=lambda item: (
                -float(item.get("branch_rank_score", float("-inf")) or float("-inf")),
                -float(item.get("search_score", 0.0) or 0.0),
                -int(item.get("preview_score", 0) or 0),
                str(item.get("command", "")),
            ),
        )[:preview_limit]
    }
    previews = [
        preview
        for preview in sorted(
            previews,
            key=lambda item: (
                -float(item.get("search_score", 0.0) or 0.0),
                -int(item.get("preview_score", 0) or 0),
                str(item.get("command", "")),
            ),
        )
        if str(preview.get("command", "")).strip() in selected_commands
    ][:preview_limit]
    return {
        "candidates": previews,
        "search_branches": search_branches,
        "subgoal_plan": subgoal_plan,
        "planning_mode": "hierarchical" if subgoal_plan else "bounded_sequence",
        "universe_id": state.universe_summary.get("universe_id", ""),
        "completion_ratio": preview_summary.get("completion_ratio", 0.0),
        "missing_expected_artifacts": list(preview_summary.get("missing_expected_artifacts", []))[:4],
        "present_forbidden_artifacts": list(preview_summary.get("present_forbidden_artifacts", []))[:4],
        "changed_preserved_artifacts": list(preview_summary.get("changed_preserved_artifacts", []))[:4],
    }


def _semantic_memory_preview_candidates(state) -> list[str]:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    semantic_episodes = graph_summary.get("semantic_episodes", [])
    if not isinstance(semantic_episodes, list):
        return []
    commands: list[str] = []
    for episode in semantic_episodes[:4]:
        if not isinstance(episode, dict):
            continue
        recovery_trace = episode.get("recovery_trace", {})
        recovery_trace = dict(recovery_trace) if isinstance(recovery_trace, dict) else {}
        for raw in (
            recovery_trace.get("recovery_command", ""),
            recovery_trace.get("failed_command", ""),
        ):
            command = str(raw).strip()
            if command and command not in commands:
                commands.append(command)
    return commands


def _semantic_prototype_preview_candidates(state) -> list[str]:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    semantic_prototypes = graph_summary.get("semantic_prototypes", [])
    if not isinstance(semantic_prototypes, list):
        return []
    commands: list[str] = []
    for prototype in semantic_prototypes[:4]:
        if not isinstance(prototype, dict):
            continue
        for key in ("instantiated_application_commands", "application_commands"):
            values = prototype.get(key, [])
            if not isinstance(values, list):
                continue
            for raw in values:
                command = str(raw).strip()
                if command and command not in commands:
                    commands.append(command)
        for raw in dict(prototype.get("application_command_templates", {})).keys():
            command = str(raw).strip()
            if command and command not in commands:
                commands.append(command)
    return commands


def _preview_frontier_should_expand(state) -> bool:
    graph_summary = state.graph_summary if isinstance(state.graph_summary, dict) else {}
    semantic_episodes = graph_summary.get("semantic_episodes", [])
    if isinstance(semantic_episodes, list) and semantic_episodes:
        return True
    horizon = str(
        state.world_model_summary.get(
            "horizon",
            state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
        )
    ).strip()
    if horizon == "long_horizon":
        return True
    active_subgoal = str(state.active_subgoal or "").strip()
    if active_subgoal and state.active_subgoal_diagnosis():
        return True
    return False


def _hierarchical_subgoal_plan(policy, state, preview_summary: dict[str, object]) -> list[dict[str, object]]:
    hotspots = policy.world_model.prioritized_long_horizon_hotspots(
        state.task,
        preview_summary,
        latest_transition=state.latest_state_transition,
        latent_state_summary=state.latent_state_summary,
        active_subgoal=str(state.active_subgoal or "").strip(),
        max_items=4,
    )
    if hotspots:
        return [
            {
                "subgoal": str(item.get("subgoal", "")).strip(),
                "path": str(item.get("path", "")).strip(),
                "priority": int(item.get("priority", 0) or 0),
                "signals": [
                    str(signal).strip()
                    for signal in item.get("signals", [])
                    if str(signal).strip()
                ],
            }
            for item in hotspots
            if isinstance(item, dict) and str(item.get("subgoal", "")).strip()
        ]
    active_subgoal = str(state.active_subgoal or "").strip()
    if not active_subgoal:
        return []
    diagnosis = state.active_subgoal_diagnosis()
    return [
        {
            "subgoal": active_subgoal,
            "path": str(diagnosis.get("path", "")).strip(),
            "priority": 100,
            "signals": [
                str(signal).strip()
                for signal in diagnosis.get("signals", [])
                if str(signal).strip()
            ],
        }
    ]


def _hierarchical_subgoal_bonus(command: str, subgoal: dict[str, object]) -> int:
    normalized_command = str(command).strip().lower()
    if not normalized_command:
        return 0
    score = 0
    path = str(subgoal.get("path", "")).strip().lower()
    if path and path in normalized_command:
        score += 6
    for token in str(subgoal.get("subgoal", "")).strip().lower().split():
        cleaned = token.strip(".,:;()[]{}\"'")
        if len(cleaned) >= 4 and cleaned in normalized_command:
            score += 1
    if "state_regression" in {str(signal).strip() for signal in subgoal.get("signals", [])} and path and path in normalized_command:
        score += 2
    return score


def _best_matching_subgoal(command: str, subgoal_plan: list[dict[str, object]]) -> tuple[int, str]:
    best_score = 0
    best_subgoal = ""
    for subgoal in subgoal_plan:
        if not isinstance(subgoal, dict):
            continue
        score = _hierarchical_subgoal_bonus(command, subgoal)
        if score > best_score:
            best_score = score
            best_subgoal = str(subgoal.get("subgoal", "")).strip()
    return best_score, best_subgoal


def _adaptive_transition_search(policy, state, *, preview_summary: dict[str, object], previews: list[dict[str, object]], max_depth: int, subgoal_plan: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    allowed_previews = [
        dict(item)
        for item in previews
        if str(dict(item).get("governance", {}).get("decision", "")).strip().lower() not in {"deny", "block"}
    ]
    if not allowed_previews:
        return []
    normalized_subgoal_plan = [dict(item) for item in (subgoal_plan or []) if isinstance(item, dict)]
    beam_width = 3 if max_depth >= 3 else 2
    frontier: list[dict[str, object]] = []
    selected_seed_previews: list[dict[str, object]] = []
    seen_seed_commands: set[str] = set()
    if normalized_subgoal_plan:
        for subgoal in normalized_subgoal_plan[:beam_width]:
            ranked = sorted(
                allowed_previews,
                key=lambda item: (
                    -_hierarchical_subgoal_bonus(str(item.get("command", "")), subgoal),
                    -int(item.get("preview_score", 0) or 0),
                    str(item.get("command", "")),
                ),
            )
            if not ranked:
                continue
            selected = dict(ranked[0])
            command = str(selected.get("command", "")).strip()
            if not command or command in seen_seed_commands:
                continue
            seen_seed_commands.add(command)
            selected_seed_previews.append(selected)
    for preview in sorted(
        allowed_previews,
        key=lambda item: (-int(item.get("preview_score", 0) or 0), str(item.get("command", ""))),
    ):
        command = str(preview.get("command", "")).strip()
        if not command or command in seen_seed_commands:
            continue
        seen_seed_commands.add(command)
        selected_seed_previews.append(dict(preview))
        if len(selected_seed_previews) >= beam_width:
            break
    for preview in selected_seed_previews[:beam_width]:
        summary_after = _rolled_summary_after_effect(preview_summary, preview)
        sequence_effect = policy.world_model.simulate_command_sequence_effect(
            preview_summary,
            [str(preview.get("command", "")).strip()],
        )
        subgoal_bonus, best_subgoal = _best_matching_subgoal(str(preview.get("command", "")), normalized_subgoal_plan)
        failure_repeat_penalty = float(_semantic_failure_repeat_penalty(preview))
        cumulative_score = float(sequence_effect.get("score", preview.get("preview_score", 0)) or 0.0)
        cumulative_score += float(sequence_effect.get("latent_transition_score", 0.0) or 0.0)
        cumulative_score += float(subgoal_bonus)
        cumulative_score -= failure_repeat_penalty
        frontier.append(
            {
                "sequence": [str(preview.get("command", "")).strip()],
                "cumulative_score": cumulative_score,
                "normalized_score": cumulative_score,
                "depth": 1,
                "summary": summary_after,
                "subgoal": best_subgoal,
                "failure_repeat_penalty": failure_repeat_penalty,
            }
        )
    completed = [dict(item) for item in frontier]
    for _ in range(1, max_depth):
        next_frontier: list[dict[str, object]] = []
        for branch in frontier:
            branch_summary = dict(branch.get("summary", {})) if isinstance(branch.get("summary", {}), dict) else dict(preview_summary)
            seen = {str(value).strip() for value in branch.get("sequence", []) if str(value).strip()}
            candidates = []
            for preview in allowed_previews:
                command = str(preview.get("command", "")).strip()
                if not command or command in seen:
                    continue
                effect = policy.world_model.simulate_command_effect(branch_summary, command)
                governance = policy._simulate_command_governance(state.universe_summary, command)
                score = float(effect.get("score", 0) or 0)
                score += float(effect.get("predicted_progress_gain", 0.0) or 0.0)
                score += float(effect.get("predicted_verifier_delta", 0.0) or 0.0)
                score += float(effect.get("latent_transition_score", 0.0) or 0.0)
                score -= float(_semantic_failure_repeat_penalty(effect))
                if str(governance.get("decision", "")).strip().lower() == "allow":
                    score += 1.0
                else:
                    score -= 3.0
                subgoal_bonus, best_subgoal = _best_matching_subgoal(command, normalized_subgoal_plan)
                score += float(subgoal_bonus)
                candidates.append((score, command, effect, best_subgoal))
            for score, command, effect, best_subgoal in sorted(candidates, key=lambda item: (-item[0], item[1]))[:beam_width]:
                rolled_summary = _rolled_summary_after_effect(branch_summary, effect)
                sequence = [*branch.get("sequence", []), command]
                sequence_effect = policy.world_model.simulate_command_sequence_effect(
                    preview_summary,
                    sequence,
                )
                failure_repeat_penalty = float(branch.get("failure_repeat_penalty", 0.0) or 0.0)
                failure_repeat_penalty += float(_semantic_failure_repeat_penalty(effect))
                cumulative_score = float(sequence_effect.get("score", 0.0) or 0.0)
                cumulative_score += float(sequence_effect.get("latent_transition_score", 0.0) or 0.0)
                cumulative_score -= failure_repeat_penalty
                next_branch = {
                    "sequence": sequence,
                    "cumulative_score": cumulative_score,
                    "normalized_score": cumulative_score
                    / float(max(1, len(sequence))),
                    "depth": int(branch.get("depth", 1) or 1) + 1,
                    "summary": rolled_summary,
                    "subgoal": best_subgoal or str(branch.get("subgoal", "")).strip(),
                    "failure_repeat_penalty": failure_repeat_penalty,
                }
                next_frontier.append(next_branch)
                completed.append(dict(next_branch))
        frontier = sorted(
            next_frontier,
            key=lambda item: (
                -float(item.get("cumulative_score", 0.0) or 0.0),
                [str(value).strip() for value in item.get("sequence", [])],
            ),
        )[:beam_width]
        if not frontier:
            break
    return [
        {
            "sequence": [str(value).strip() for value in item.get("sequence", []) if str(value).strip()],
            "cumulative_score": round(float(item.get("cumulative_score", 0.0) or 0.0), 4),
            "normalized_score": round(float(item.get("normalized_score", 0.0) or 0.0), 4),
            "depth": int(item.get("depth", 1) or 1),
            "subgoal": str(item.get("subgoal", "")).strip(),
        }
        for item in sorted(
            completed,
            key=lambda payload: (
                -float(payload.get("cumulative_score", 0.0) or 0.0),
                -float(payload.get("normalized_score", 0.0) or 0.0),
                [str(value).strip() for value in payload.get("sequence", []) if str(value).strip()],
            ),
        )[:3]
    ]


def _rolled_summary_after_effect(summary: dict[str, object], effect: dict[str, object]) -> dict[str, object]:
    rolled = dict(summary)
    existing_expected = {
        str(value).strip()
        for value in rolled.get("existing_expected_artifacts", [])
        if str(value).strip()
    }
    missing_expected = [
        str(value).strip()
        for value in rolled.get("missing_expected_artifacts", [])
        if str(value).strip()
    ]
    predicted_outputs = {
        str(value).strip()
        for value in effect.get("predicted_outputs", [])
        if str(value).strip()
    }
    predicted_changed = {
        str(value).strip()
        for value in effect.get("predicted_changed_paths", [])
        if str(value).strip()
    }
    rolled["existing_expected_artifacts"] = sorted(existing_expected | predicted_outputs)
    rolled["missing_expected_artifacts"] = [
        path for path in missing_expected if path not in predicted_outputs
    ]
    for key in ("workflow_expected_changed_paths", "workflow_generated_paths", "workflow_report_paths"):
        rolled[key] = [
            str(value).strip()
            for value in rolled.get(key, [])
            if str(value).strip() and str(value).strip() not in predicted_changed
        ]
    completion_ratio = float(rolled.get("completion_ratio", 0.0) or 0.0)
    predicted_progress_gain = float(effect.get("predicted_progress_gain", 0.0) or 0.0)
    rolled["completion_ratio"] = min(1.0, round(completion_ratio + 0.15 * predicted_progress_gain, 4))
    semantic_episodes = rolled.get("semantic_episodes", [])
    if isinstance(semantic_episodes, list) and predicted_changed:
        rolled["semantic_episodes"] = [
            dict(item)
            for item in semantic_episodes
            if isinstance(item, dict)
        ]
    return rolled


def _semantic_failure_repeat_penalty(effect: dict[str, object]) -> int:
    empirical_prior = effect.get("empirical_prior", {})
    empirical_prior = dict(empirical_prior) if isinstance(empirical_prior, dict) else {}
    prediction_source = str(empirical_prior.get("prediction_source", "")).strip().lower()
    semantic_prior = empirical_prior.get("semantic_prior", {})
    semantic_prior = dict(semantic_prior) if isinstance(semantic_prior, dict) else {}
    prototype_prior = empirical_prior.get("prototype_prior", {})
    prototype_prior = dict(prototype_prior) if isinstance(prototype_prior, dict) else {}
    active_prior = semantic_prior
    if prediction_source == "semantic_memory":
        active_prior = empirical_prior
    elif prediction_source == "bootstrap_prior" and prototype_prior:
        active_prior = prototype_prior
    elif prediction_source == "learned_state_model" and prototype_prior:
        active_prior = prototype_prior
    if int(active_prior.get("sample_count", 0) or 0) <= 0:
        return 0
    if float(active_prior.get("regression_rate", 0.0) or 0.0) < 0.5:
        return 0
    if float(active_prior.get("pass_rate", 0.0) or 0.0) > 0.0:
        return 0
    return 16
