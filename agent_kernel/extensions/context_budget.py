from __future__ import annotations

from dataclasses import dataclass

from ..config import KernelConfig
from ..state import AgentState, _software_work_objective_phase, _software_work_phase_rank


@dataclass(slots=True)
class _BudgetChunk:
    source: str
    key: str
    text: str
    score: int
    payload: object


class ContextBudgeter:
    def __init__(self, config: KernelConfig) -> None:
        self.config = config

    def build_payload(
        self,
        *,
        state: AgentState,
        task_payload: dict[str, object],
        history_payload: list[dict[str, object]],
        history_archive: dict[str, object],
        llm_context_packet: dict[str, object] | None,
        retrieval_plan: dict[str, object],
        transition_preview: dict[str, object] | None,
        available_skills: list[dict[str, object]],
        prompt_adjustments: list[dict[str, object]],
        allowed_actions: list[str],
        graph_summary: dict[str, object],
        universe_summary: dict[str, object],
        world_model_summary: dict[str, object],
        plan: list[str],
        active_subgoal: str,
    ) -> dict[str, object]:
        selected = self._select_chunks(
            state=state,
            llm_context_packet=llm_context_packet,
            graph_summary=graph_summary,
            universe_summary=universe_summary,
            world_model_summary=world_model_summary,
            plan=plan,
            active_subgoal=active_subgoal,
        )
        selected_by_source: dict[str, list[_BudgetChunk]] = {}
        for chunk in selected:
            selected_by_source.setdefault(chunk.source, []).append(chunk)
        context_packet = self._budget_context_packet(llm_context_packet, selected_by_source)
        payload = {
            "task": task_payload,
            "history": history_payload,
            "history_archive": history_archive,
            "recent_workspace_summary": state.recent_workspace_summary,
            "context_packet": context_packet,
            "retrieval_plan": retrieval_plan,
            "transition_preview": transition_preview or {},
            "available_skills": available_skills,
            "prompt_adjustments": prompt_adjustments,
            "graph_summary": self._budget_graph_summary(graph_summary, selected_by_source),
            "universe_summary": self._budget_universe_summary(universe_summary, selected_by_source),
            "world_model_summary": self._budget_world_model_summary(world_model_summary, selected_by_source),
            "latest_state_transition": dict(state.latest_state_transition),
            "latent_state_summary": dict(state.latent_state_summary),
            "active_subgoal_diagnosis": state.active_subgoal_diagnosis(),
            "planner_recovery_artifact": dict(state.planner_recovery_artifact),
            "planner_recovery_plan_update": list(
                dict(state.planner_recovery_artifact).get("staged_plan_update", [])
            )
            if isinstance(state.planner_recovery_artifact, dict)
            else [],
            "software_work_plan_update": self._planner_software_work_plan_update(state),
            "software_work_stage_state": state.software_work_stage_overview(),
            "software_work_phase_state": state.software_work_phase_state(),
            "software_work_phase_gate_state": state.software_work_phase_gate_state(),
            "campaign_contract_state": state.campaign_contract_state(),
            "plan": [str(chunk.payload) for chunk in selected_by_source.get("plan", [])],
            "active_subgoal": str(selected_by_source.get("subgoal", [])[0].payload)
            if selected_by_source.get("subgoal")
            else "",
            "acting_role": state.current_role,
            "allowed_actions": allowed_actions,
            "state_context_chunks": [
                {
                    "source": chunk.source,
                    "key": chunk.key,
                    "score": chunk.score,
                    "text": chunk.text,
                }
                for chunk in selected
            ],
        }
        if payload["planner_recovery_plan_update"] and str(state.current_role or "") == "planner":
            staged = [
                str(item).strip()
                for item in payload["planner_recovery_plan_update"]
                if str(item).strip()
            ]
            merged_plan: list[str] = []
            for item in [*staged, *payload["plan"]]:
                if item and item not in merged_plan:
                    merged_plan.append(item)
            payload["plan"] = merged_plan[: self.config.llm_plan_max_items]
        if payload["software_work_plan_update"] and str(state.current_role or "") == "planner":
            staged = [
                str(item).strip()
                for item in payload["software_work_plan_update"]
                if str(item).strip()
            ]
            merged_plan: list[str] = []
            for item in [*staged, *payload["plan"]]:
                if item and item not in merged_plan:
                    merged_plan.append(item)
            payload["plan"] = merged_plan[: self.config.llm_plan_max_items]
        if not payload["plan"]:
            payload["plan"] = plan[: self.config.llm_plan_max_items]
        if not payload["active_subgoal"]:
            payload["active_subgoal"] = active_subgoal
        return payload

    @staticmethod
    def _planner_software_work_plan_update(state: AgentState) -> list[str]:
        objectives = [str(item).strip() for item in state.software_work_plan_update() if str(item).strip()]
        if not objectives or str(state.current_role or "").strip() != "planner":
            return objectives
        phase_state = state.software_work_phase_state()
        suggested_phase = str(phase_state.get("suggested_phase", "")).strip()
        current_phase = str(phase_state.get("current_phase", "")).strip()
        overview = state.software_work_stage_overview()
        objective_states = overview.get("objective_states", {})
        objective_states = objective_states if isinstance(objective_states, dict) else {}
        objective_index = {objective: index for index, objective in enumerate(objectives)}

        def prefix_priority(objective: str) -> int:
            normalized = str(objective).strip()
            for index, prefix in enumerate(
                (
                    "apply planned edit ",
                    "complete implementation for ",
                    "materialize expected artifact ",
                    "revise implementation for ",
                    "remove forbidden artifact ",
                    "preserve required artifact ",
                    "update workflow path ",
                    "prepare workflow branch ",
                    "accept required branch ",
                    "regenerate generated artifact ",
                    "run workflow test ",
                    "write workflow report ",
                )
            ):
                if normalized.startswith(prefix):
                    return index
            return 99

        def phase_distance(phase: str) -> int:
            if suggested_phase:
                return abs(_software_work_phase_rank(phase) - _software_work_phase_rank(suggested_phase))
            if current_phase:
                return abs(_software_work_phase_rank(phase) - _software_work_phase_rank(current_phase))
            return _software_work_phase_rank(phase)

        ranked = sorted(
            objectives,
            key=lambda objective: (
                phase_distance(_software_work_objective_phase(objective)),
                prefix_priority(objective),
                1 if str(objective_states.get(objective, "pending")).strip() == "completed" else 0,
                objective_index.get(objective, 0),
                objective,
            ),
        )
        ordered: list[str] = []
        for objective in ranked:
            if objective not in ordered:
                ordered.append(objective)
        return ordered[:6]

    def _select_chunks(
        self,
        *,
        state: AgentState,
        llm_context_packet: dict[str, object] | None,
        graph_summary: dict[str, object],
        universe_summary: dict[str, object],
        world_model_summary: dict[str, object],
        plan: list[str],
        active_subgoal: str,
    ) -> list[_BudgetChunk]:
        chunks = self._collect_chunks(
            state=state,
            llm_context_packet=llm_context_packet,
            graph_summary=graph_summary,
            universe_summary=universe_summary,
            world_model_summary=world_model_summary,
            plan=plan,
            active_subgoal=active_subgoal,
        )
        packet_budget = {}
        if llm_context_packet is not None:
            control = llm_context_packet.get("control", {})
            if isinstance(control, dict):
                raw_budget = control.get("context_chunk_budget", {})
                if isinstance(raw_budget, dict):
                    packet_budget = raw_budget
        budget = max(
            256,
            _safe_int(packet_budget.get("char_budget"), self.config.tolbert_context_char_budget),
        )
        max_chunks = max(
            1,
            _safe_int(packet_budget.get("max_chunks"), self.config.tolbert_context_max_chunks),
        )
        selected: list[_BudgetChunk] = []
        used_chars = 0
        for chunk in sorted(chunks, key=lambda item: (-item.score, item.source, item.key, item.text)):
            cost = len(chunk.text)
            if selected and used_chars + cost > budget:
                continue
            selected.append(chunk)
            used_chars += cost
            if len(selected) >= max_chunks or used_chars >= budget:
                break
        return selected

    def _collect_chunks(
        self,
        *,
        state: AgentState,
        llm_context_packet: dict[str, object] | None,
        graph_summary: dict[str, object],
        universe_summary: dict[str, object],
        world_model_summary: dict[str, object],
        plan: list[str],
        active_subgoal: str,
    ) -> list[_BudgetChunk]:
        chunks: list[_BudgetChunk] = []
        role = str(state.current_role or "executor")
        subgoal_tokens = {token for token in active_subgoal.split() if len(token) > 2}
        expected = {str(value) for value in world_model_summary.get("expected_artifacts", [])}
        forbidden = {str(value) for value in world_model_summary.get("forbidden_artifacts", [])}
        unsatisfied = {str(value) for value in world_model_summary.get("unsatisfied_expected_contents", [])}

        if llm_context_packet is not None:
            for item in llm_context_packet.get("control", {}).get("selected_context_chunks", []):
                text = str(item.get("text", "")).strip()
                if text:
                    score = 9
                    span_id = str(item.get("span_id", ""))
                    span_type = str(item.get("span_type", ""))
                    if span_id == "research:applied_guidance" or span_type == "research_library:applied_guidance":
                        score += 8
                    elif span_id == "research:paper_hits" or span_type == "research_library:paper_hits":
                        score += 3
                    score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                    if role == "executor":
                        score += 2
                    chunks.append(
                        _BudgetChunk(
                            source="retrieval",
                            key=span_id,
                            text=text,
                            score=score,
                            payload=item,
                        )
                    )

        for item in plan:
            text = str(item).strip()
            if text:
                score = 6 if role == "planner" else 4
                score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                chunks.append(_BudgetChunk("plan", text, text, score, text))

        if active_subgoal.strip():
            score = 8 if role == "planner" else 5
            chunks.append(_BudgetChunk("subgoal", "active", active_subgoal.strip(), score, active_subgoal.strip()))
        artifact = dict(state.planner_recovery_artifact) if isinstance(state.planner_recovery_artifact, dict) else {}
        if artifact:
            next_stage_objective = str(artifact.get("next_stage_objective", "")).strip()
            if next_stage_objective:
                chunks.append(
                    _BudgetChunk(
                        "planner_recovery",
                        "next_stage",
                        next_stage_objective,
                        12 if role == "planner" else 8,
                        next_stage_objective,
                    )
                )
            rewritten_subgoal = str(artifact.get("rewritten_subgoal", "")).strip()
            if rewritten_subgoal:
                chunks.append(
                    _BudgetChunk(
                        "planner_recovery",
                        "rewritten_subgoal",
                        rewritten_subgoal,
                        11 if role == "planner" else 7,
                        rewritten_subgoal,
                    )
                )
            for index, objective in enumerate(list(artifact.get("related_objectives", []))[:4], start=1):
                normalized = str(objective).strip()
                if not normalized:
                    continue
                chunks.append(
                    _BudgetChunk(
                        "planner_recovery",
                        f"related#{index}",
                        normalized,
                        10 if role == "planner" else 6,
                        normalized,
                    )
                )
            for index, item in enumerate(list(artifact.get("ranked_objectives", []))[:3], start=1):
                if not isinstance(item, dict):
                    continue
                objective = str(item.get("objective", "")).strip()
                if not objective:
                    continue
                score = int(item.get("score", 0) or 0)
                reason = str(item.get("reason", "")).strip()
                text = f"stage {index}: {objective} [score={score}]"
                if reason:
                    text += f" {reason}"
                chunks.append(
                    _BudgetChunk(
                        "planner_recovery",
                        f"stage#{index}",
                        text,
                        11 - index if role == "planner" else 7 - min(index, 2),
                        {"objective": objective, "score": score, "reason": reason},
                    )
                )
        for index, objective in enumerate(state.software_work_plan_update()[:4], start=1):
            normalized = str(objective).strip()
            if not normalized:
                continue
            chunks.append(
                _BudgetChunk(
                    "software_work",
                    f"stage#{index}",
                    normalized,
                    10 if role == "planner" else 6,
                        normalized,
                    )
                )
        software_work_stage = state.software_work_stage_overview()
        recent_outcomes = software_work_stage.get("recent_outcomes", [])
        if isinstance(recent_outcomes, list):
            for index, item in enumerate(recent_outcomes[-3:], start=1):
                if not isinstance(item, dict):
                    continue
                objective = str(item.get("objective", "")).strip()
                status = str(item.get("status", "")).strip()
                if not objective or not status:
                    continue
                text = f"{objective}: {status}"
                chunks.append(
                    _BudgetChunk(
                        "software_work_outcome",
                        f"recent#{index}",
                        text,
                        10 if role == "planner" else 6,
                        {"objective": objective, "status": status},
                    )
                )
        software_work_phase = state.software_work_phase_state()
        if isinstance(software_work_phase, dict) and software_work_phase:
            suggested_phase = str(software_work_phase.get("suggested_phase", "")).strip()
            current_phase = str(software_work_phase.get("current_phase", "")).strip()
            current_phase_status = str(software_work_phase.get("current_phase_status", "")).strip()
            if suggested_phase:
                text = f"suggested phase: {suggested_phase}"
                if current_phase and current_phase_status:
                    text += f" (from {current_phase}:{current_phase_status})"
                chunks.append(
                    _BudgetChunk(
                        "software_work_phase",
                        "suggested_phase",
                        text,
                        11 if role == "planner" else 7,
                        {
                            "suggested_phase": suggested_phase,
                            "current_phase": current_phase,
                            "current_phase_status": current_phase_status,
                        },
                    )
                )
            phase_states = software_work_phase.get("phase_states", {})
            if isinstance(phase_states, dict):
                for phase in ("implementation", "migration", "test", "follow_up_fix"):
                    state_payload = phase_states.get(phase, {})
                    if not isinstance(state_payload, dict):
                        continue
                    objective_count = int(state_payload.get("objective_count", 0) or 0)
                    if objective_count <= 0:
                        continue
                    status = str(state_payload.get("status", "")).strip()
                    text = f"{phase}: {status} ({objective_count} objectives)"
                    chunks.append(
                        _BudgetChunk(
                            "software_work_phase",
                            phase,
                            text,
                            10 if role == "planner" else 6,
                            {"phase": phase, **state_payload},
                        )
                    )
        software_work_gate = state.software_work_phase_gate_state()
        if isinstance(software_work_gate, dict) and software_work_gate:
            gate_phase = str(software_work_gate.get("gate_phase", "")).strip()
            gate_reason = str(software_work_gate.get("gate_reason", "")).strip()
            gate_objectives = [
                str(item).strip()
                for item in software_work_gate.get("gate_objectives", [])
                if str(item).strip()
            ]
            if gate_phase:
                text = f"phase gate: {gate_phase}"
                if gate_reason:
                    text += f" ({gate_reason})"
                chunks.append(
                    _BudgetChunk(
                        "software_work_gate",
                        "phase_gate",
                        text,
                        12 if role == "planner" else 8,
                        {
                            "gate_phase": gate_phase,
                            "gate_reason": gate_reason,
                        },
                    )
                )
            for index, objective in enumerate(gate_objectives[:3], start=1):
                chunks.append(
                    _BudgetChunk(
                        "software_work_gate",
                        f"objective#{index}",
                        objective,
                        12 - min(index, 2) if role == "planner" else 8 - min(index, 2),
                        objective,
                    )
                )
        campaign_contract = state.campaign_contract_state()
        if isinstance(campaign_contract, dict) and campaign_contract:
            current_objective = str(campaign_contract.get("current_objective", "")).strip()
            if current_objective:
                chunks.append(
                    _BudgetChunk(
                        "campaign_contract",
                        "current_objective",
                        f"campaign anchor: {current_objective}",
                        13 if role == "planner" else 8,
                        current_objective,
                    )
                )
            for index, objective in enumerate(campaign_contract.get("anchor_objectives", [])[:4], start=1):
                normalized = str(objective).strip()
                if not normalized:
                    continue
                chunks.append(
                    _BudgetChunk(
                        "campaign_contract",
                        f"anchor#{index}",
                        normalized,
                        12 - min(index, 2) if role == "planner" else 7 - min(index, 2),
                        normalized,
                    )
                )
            for index, objective in enumerate(campaign_contract.get("regressed_objectives", [])[:2], start=1):
                normalized = str(objective).strip()
                if not normalized:
                    continue
                chunks.append(
                    _BudgetChunk(
                        "campaign_contract",
                        f"regressed#{index}",
                        f"regressed obligation: {normalized}",
                        13 if role in {"planner", "critic"} else 8,
                        {"objective": normalized, "status": "regressed"},
                    )
                )
            for index, objective in enumerate(campaign_contract.get("stalled_objectives", [])[:2], start=1):
                normalized = str(objective).strip()
                if not normalized:
                    continue
                chunks.append(
                    _BudgetChunk(
                        "campaign_contract",
                        f"stalled#{index}",
                        f"stalled obligation: {normalized}",
                        11 if role in {"planner", "critic"} else 7,
                        {"objective": normalized, "status": "stalled"},
                    )
                )
            for index, path in enumerate(campaign_contract.get("required_paths", [])[:3], start=1):
                normalized = str(path).strip()
                if not normalized:
                    continue
                chunks.append(
                    _BudgetChunk(
                        "campaign_contract",
                        f"path#{index}",
                        f"required path: {normalized}",
                        10 if role == "planner" else 6,
                        normalized,
                    )
                )
        for goal, diagnosis in list(state.subgoal_diagnoses.items())[:6]:
            if not isinstance(diagnosis, dict):
                continue
            summary = str(diagnosis.get("summary", "")).strip()
            if not summary:
                continue
            text = f"{goal}: {summary}"
            score = 10 if role == "planner" else (9 if role == "critic" else 5)
            if str(goal).strip() == active_subgoal.strip():
                score += 2
            score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
            chunks.append(
                _BudgetChunk(
                    "subgoal_diagnosis",
                    str(goal).strip(),
                    text,
                    score,
                    {"subgoal": str(goal).strip(), **dict(diagnosis)},
                )
            )

        for family, count in list((graph_summary.get("benchmark_families") or {}).items())[:6]:
            text = f"{family}:{count}"
            chunks.append(_BudgetChunk("graph_family", str(family), text, 2, (family, count)))
        for failure, count in list((graph_summary.get("failure_types") or {}).items())[:6]:
            text = f"{failure}:{count}"
            score = 6 if role == "critic" else 4
            chunks.append(_BudgetChunk("graph_failure", str(failure), text, score, (failure, count)))
        for related in list(graph_summary.get("related_tasks") or [])[:6]:
            text = str(related)
            score = 4 + self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
            chunks.append(_BudgetChunk("graph_related", text, text, score, text))
        trusted_commands = graph_summary.get("trusted_retrieval_command_counts", {})
        if isinstance(trusted_commands, dict):
            for command, count in list(trusted_commands.items())[:4]:
                command_text = str(command).strip()
                if not command_text:
                    continue
                text = f"trusted retrieval x{_safe_int(count, 0)}: {command_text}"
                score = 7 if role == "executor" else (8 if role == "planner" else 6)
                score += self._token_overlap_bonus(command_text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_trusted_retrieval_command",
                        command_text,
                        text,
                        score,
                        (command_text, _safe_int(count, 0)),
                    )
                )
        trusted_procedures = graph_summary.get("trusted_retrieval_procedures", {})
        if isinstance(trusted_procedures, list):
            for item in trusted_procedures[:2]:
                if not isinstance(item, dict):
                    continue
                commands = [str(value).strip() for value in item.get("commands", []) if str(value).strip()]
                if len(commands) < 2:
                    continue
                count = _safe_int(item.get("count", 0), 0)
                text = f"trusted retrieval sequence x{count}: " + " -> ".join(commands[:4])
                score = 8 if role == "planner" else (7 if role == "critic" else 5)
                score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_trusted_retrieval_procedure",
                        commands[0],
                        text,
                        score,
                        {"commands": commands, "count": count},
                    )
                )
        verifier_obligations = graph_summary.get("verifier_obligation_counts", {})
        if isinstance(verifier_obligations, dict):
            for obligation, count in list(verifier_obligations.items())[:4]:
                obligation_text = str(obligation).strip()
                if not obligation_text:
                    continue
                text = f"verifier obligation x{_safe_int(count, 0)}: {obligation_text}"
                score = 8 if role == "planner" else (7 if role == "critic" else 5)
                score += self._token_overlap_bonus(obligation_text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_verifier_obligation",
                        obligation_text,
                        text,
                        score,
                        (obligation_text, _safe_int(count, 0)),
                    )
                )
        changed_paths = graph_summary.get("changed_path_counts", {})
        if isinstance(changed_paths, dict):
            for changed_path, count in list(changed_paths.items())[:4]:
                path_text = str(changed_path).strip()
                if not path_text:
                    continue
                text = f"historically changed path x{_safe_int(count, 0)}: {path_text}"
                score = 7 if role == "planner" else (6 if role == "critic" else 5)
                score += self._token_overlap_bonus(path_text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_changed_path",
                        path_text,
                        text,
                        score,
                        (path_text, _safe_int(count, 0)),
                    )
                )
        edit_patch_paths = graph_summary.get("edit_patch_path_counts", {})
        if isinstance(edit_patch_paths, dict):
            for edit_path, count in list(edit_patch_paths.items())[:4]:
                path_text = str(edit_path).strip()
                if not path_text:
                    continue
                text = f"historical edit patch x{_safe_int(count, 0)}: {path_text}"
                score = 8 if role == "planner" else (7 if role == "critic" else 5)
                score += self._token_overlap_bonus(path_text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_edit_patch_path",
                        path_text,
                        text,
                        score,
                        (path_text, _safe_int(count, 0)),
                    )
                )
        recovery_commands = graph_summary.get("recovery_command_counts", {})
        if isinstance(recovery_commands, dict):
            for command, count in list(recovery_commands.items())[:3]:
                command_text = str(command).strip()
                if not command_text:
                    continue
                text = f"recovery trace x{_safe_int(count, 0)}: {command_text}"
                score = 7 if role == "planner" else (8 if role == "critic" else 5)
                score += self._token_overlap_bonus(command_text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_recovery_command",
                        command_text,
                        text,
                        score,
                        (command_text, _safe_int(count, 0)),
                    )
                )
        semantic_episodes = graph_summary.get("semantic_episodes", [])
        if isinstance(semantic_episodes, list):
            for item in semantic_episodes[:2]:
                if not isinstance(item, dict):
                    continue
                task_id = str(item.get("task_id", "")).strip()
                obligations = [
                    str(value).strip()
                    for value in item.get("verifier_obligations", [])
                    if str(value).strip()
                ]
                changed_paths = [
                    str(value).strip()
                    for value in item.get("changed_paths", [])
                    if str(value).strip()
                ]
                recovery_trace = dict(item.get("recovery_trace", {})) if isinstance(item.get("recovery_trace", {}), dict) else {}
                parts = [task_id] if task_id else []
                if obligations:
                    parts.append("obligations: " + "; ".join(obligations[:2]))
                if changed_paths:
                    parts.append("changed: " + ", ".join(changed_paths[:3]))
                edit_patches = [dict(value) for value in item.get("edit_patches", []) if isinstance(value, dict)]
                if edit_patches:
                    patch_descriptions: list[str] = []
                    for patch in edit_patches[:2]:
                        patch_summary = str(patch.get("patch_summary", "")).strip()
                        patch_excerpt = str(patch.get("patch_excerpt", "")).strip()
                        if patch_summary:
                            patch_descriptions.append(patch_summary)
                        if patch_excerpt:
                            patch_descriptions.append(patch_excerpt[:120])
                    if patch_descriptions:
                        parts.append("patches: " + " | ".join(patch_descriptions[:3]))
                recovery_command = str(recovery_trace.get("recovery_command", "")).strip()
                if recovery_command:
                    parts.append(f"recovered via {recovery_command}")
                text = ". ".join(part for part in parts if part).strip()
                if not text:
                    continue
                score = 8 if role == "planner" else 6
                score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                chunks.append(
                    _BudgetChunk(
                        "graph_semantic_episode",
                        task_id or text,
                        text,
                        score,
                        dict(item),
                    )
                )

        for invariant in list(universe_summary.get("invariants") or [])[:4]:
            text = str(invariant)
            score = 6 if role == "critic" else 4
            chunks.append(_BudgetChunk("universe_invariant", text, text, score, text))
        for pattern in list(universe_summary.get("forbidden_command_patterns") or [])[:4]:
            text = str(pattern)
            score = 7 if role == "critic" else 5
            chunks.append(_BudgetChunk("universe_forbidden_command", text, text, score, text))
        for prefix in list(universe_summary.get("preferred_command_prefixes") or [])[:4]:
            text = str(prefix)
            score = 4
            chunks.append(_BudgetChunk("universe_preferred_command", text, text, score, text))

        for key, source_name, base in (
            ("expected_artifacts", "world_expected", 7),
            ("forbidden_artifacts", "world_forbidden", 8 if role == "critic" else 6),
            ("preserved_artifacts", "world_preserved", 5),
            ("missing_expected_artifacts", "world_missing_expected", 8),
            ("unsatisfied_expected_contents", "world_unsatisfied_expected", 9),
            ("present_forbidden_artifacts", "world_present_forbidden", 9 if role == "critic" else 7),
            ("changed_preserved_artifacts", "world_changed_preserved", 8 if role == "critic" else 6),
            ("updated_workflow_paths", "world_updated_workflow", 6),
        ):
            for item in list(world_model_summary.get(key) or [])[:6]:
                text = str(item)
                score = base + self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                chunks.append(_BudgetChunk(source_name, text, text, score, text))
        previews = world_model_summary.get("workspace_file_previews", {})
        if isinstance(previews, dict):
            for path, preview in list(previews.items())[:4]:
                preview_windows = self._preview_windows(preview)
                multi_window = len(preview_windows) > 1
                for window_index, preview_window in enumerate(preview_windows, start=1):
                    if not isinstance(preview_window, dict):
                        continue
                    content = str(preview_window.get("content", ""))
                    if not content:
                        continue
                    line_start = _safe_int(preview_window.get("line_start", 1), 1)
                    line_end = _safe_int(preview_window.get("line_end", line_start), line_start)
                    text = f"{path}:{line_start}-{line_end}\n{content}".strip()
                    score = 8 + self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                    if str(path) in unsatisfied:
                        score += 2
                    chunks.append(
                        _BudgetChunk(
                            "world_preview",
                            f"{path}#{window_index}" if multi_window else str(path),
                            text,
                            score,
                            {
                                "content": content,
                                "truncated": bool(preview_window.get("truncated", False)),
                                "path": str(path),
                                "window_index": window_index,
                                "line_start": line_start,
                                "line_end": line_end,
                            },
                        )
                    )
        return chunks

    @staticmethod
    def _token_overlap_bonus(
        text: str,
        subgoal_tokens: set[str],
        expected: set[str],
        forbidden: set[str],
    ) -> int:
        score = 0
        lower = text.lower()
        if any(token.lower() in lower for token in subgoal_tokens):
            score += 2
        if any(token and token in text for token in expected):
            score += 2
        if any(token and token in text for token in forbidden):
            score += 1
        return score

    @staticmethod
    def _budget_context_packet(
        llm_context_packet: dict[str, object] | None,
        selected_by_source: dict[str, list[_BudgetChunk]],
    ) -> dict[str, object] | None:
        if llm_context_packet is None:
            return None
        context_packet = dict(llm_context_packet)
        control = dict(context_packet.get("control", {}))
        control["selected_context_chunks"] = [chunk.payload for chunk in selected_by_source.get("retrieval", [])]
        context_packet["control"] = control
        return context_packet

    @staticmethod
    def _budget_graph_summary(
        graph_summary: dict[str, object],
        selected_by_source: dict[str, list[_BudgetChunk]],
    ) -> dict[str, object]:
        if not graph_summary:
            return {}
        summary: dict[str, object] = {}
        if "document_count" in graph_summary:
            summary["document_count"] = graph_summary["document_count"]
        for key in (
            "retrieval_backed_successes",
            "retrieval_influenced_successes",
            "trusted_retrieval_successes",
        ):
            if key in graph_summary:
                summary[key] = graph_summary[key]
        family_items = selected_by_source.get("graph_family", [])
        if family_items:
            summary["benchmark_families"] = {str(key): value for key, value in (chunk.payload for chunk in family_items)}
        elif isinstance(graph_summary.get("benchmark_families"), dict):
            summary["benchmark_families"] = dict(list(graph_summary.get("benchmark_families", {}).items())[:6])
        failure_items = selected_by_source.get("graph_failure", [])
        if failure_items:
            summary["failure_types"] = {str(key): value for key, value in (chunk.payload for chunk in failure_items)}
        elif isinstance(graph_summary.get("failure_types"), dict):
            summary["failure_types"] = dict(list(graph_summary.get("failure_types", {}).items())[:6])
        related_items = selected_by_source.get("graph_related", [])
        if related_items:
            summary["related_tasks"] = [str(chunk.payload) for chunk in related_items]
        elif isinstance(graph_summary.get("related_tasks"), list):
            summary["related_tasks"] = [str(item) for item in graph_summary.get("related_tasks", [])[:6]]
        retrieval_items = selected_by_source.get("graph_trusted_retrieval_command", [])
        if retrieval_items:
            summary["trusted_retrieval_command_counts"] = {
                str(command): count for command, count in (chunk.payload for chunk in retrieval_items)
            }
        elif isinstance(graph_summary.get("trusted_retrieval_command_counts"), dict):
            summary["trusted_retrieval_command_counts"] = dict(
                list(graph_summary.get("trusted_retrieval_command_counts", {}).items())[:4]
            )
        procedure_items = selected_by_source.get("graph_trusted_retrieval_procedure", [])
        if procedure_items:
            summary["trusted_retrieval_procedures"] = [
                {
                    "commands": [str(value) for value in dict(chunk.payload).get("commands", [])[:4] if str(value).strip()],
                    "count": _safe_int(dict(chunk.payload).get("count", 0), 0),
                }
                for chunk in procedure_items
                if isinstance(chunk.payload, dict)
            ]
        elif isinstance(graph_summary.get("trusted_retrieval_procedures"), list):
            summary["trusted_retrieval_procedures"] = [
                {
                    "commands": [str(value) for value in dict(item).get("commands", [])[:4] if str(value).strip()],
                    "count": _safe_int(dict(item).get("count", 0), 0),
                }
                for item in graph_summary.get("trusted_retrieval_procedures", [])[:2]
                if isinstance(item, dict)
            ]
        obligation_items = selected_by_source.get("graph_verifier_obligation", [])
        if obligation_items:
            summary["verifier_obligation_counts"] = {
                str(text): count for text, count in (chunk.payload for chunk in obligation_items)
            }
        elif isinstance(graph_summary.get("verifier_obligation_counts"), dict):
            summary["verifier_obligation_counts"] = dict(
                list(graph_summary.get("verifier_obligation_counts", {}).items())[:4]
            )
        changed_path_items = selected_by_source.get("graph_changed_path", [])
        if changed_path_items:
            summary["changed_path_counts"] = {
                str(path): count for path, count in (chunk.payload for chunk in changed_path_items)
            }
        elif isinstance(graph_summary.get("changed_path_counts"), dict):
            summary["changed_path_counts"] = dict(list(graph_summary.get("changed_path_counts", {}).items())[:4])
        edit_patch_items = selected_by_source.get("graph_edit_patch_path", [])
        if edit_patch_items:
            summary["edit_patch_path_counts"] = {
                str(path): count for path, count in (chunk.payload for chunk in edit_patch_items)
            }
        elif isinstance(graph_summary.get("edit_patch_path_counts"), dict):
            summary["edit_patch_path_counts"] = dict(list(graph_summary.get("edit_patch_path_counts", {}).items())[:4])
        recovery_items = selected_by_source.get("graph_recovery_command", [])
        if recovery_items:
            summary["recovery_command_counts"] = {
                str(command): count for command, count in (chunk.payload for chunk in recovery_items)
            }
        elif isinstance(graph_summary.get("recovery_command_counts"), dict):
            summary["recovery_command_counts"] = dict(
                list(graph_summary.get("recovery_command_counts", {}).items())[:3]
            )
        semantic_episode_items = selected_by_source.get("graph_semantic_episode", [])
        if semantic_episode_items:
            summary["semantic_episodes"] = [
                dict(chunk.payload)
                for chunk in semantic_episode_items
                if isinstance(chunk.payload, dict)
            ]
        elif isinstance(graph_summary.get("semantic_episodes"), list):
            summary["semantic_episodes"] = [
                dict(item)
                for item in graph_summary.get("semantic_episodes", [])[:2]
                if isinstance(item, dict)
            ]
        observed_modes = graph_summary.get("observed_environment_modes", {})
        if isinstance(observed_modes, dict) and observed_modes:
            summary["observed_environment_modes"] = {
                str(key): str(value)
                for key, value in list(observed_modes.items())[:3]
                if str(key).strip() and str(value).strip()
            }
        alignment_failures = graph_summary.get("environment_alignment_failures", {})
        if isinstance(alignment_failures, dict) and alignment_failures:
            summary["environment_alignment_failures"] = {
                str(key): value
                for key, value in list(alignment_failures.items())[:3]
                if str(key).strip()
            }
        return summary

    @staticmethod
    def _budget_universe_summary(
        universe_summary: dict[str, object],
        selected_by_source: dict[str, list[_BudgetChunk]],
    ) -> dict[str, object]:
        if not universe_summary:
            return {}
        summary: dict[str, object] = {}
        for key in ("universe_id", "stability", "governance_mode"):
            if key in universe_summary:
                summary[key] = universe_summary[key]
        if "requires_verification" in universe_summary:
            summary["requires_verification"] = universe_summary["requires_verification"]
        if "requires_bounded_steps" in universe_summary:
            summary["requires_bounded_steps"] = universe_summary["requires_bounded_steps"]
        if "prefer_reversible_actions" in universe_summary:
            summary["prefer_reversible_actions"] = universe_summary["prefer_reversible_actions"]
        for source, key in (
            ("universe_invariant", "invariants"),
            ("universe_forbidden_command", "forbidden_command_patterns"),
            ("universe_preferred_command", "preferred_command_prefixes"),
        ):
            items = selected_by_source.get(source, [])
            if items:
                summary[key] = [str(chunk.payload) for chunk in items]
        action_risk_controls = universe_summary.get("action_risk_controls", {})
        if isinstance(action_risk_controls, dict) and action_risk_controls:
            summary["action_risk_controls"] = {
                str(key): int(value)
                for key, value in action_risk_controls.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        for key in (
            "environment_assumptions",
            "environment_alignment",
            "envelope_alignment",
            "constitutional_compliance",
            "runtime_attestation",
            "plan_risk_summary",
        ):
            value = universe_summary.get(key, {})
            if isinstance(value, dict) and value:
                summary[key] = dict(value)
        autonomy_scope = universe_summary.get("autonomy_scope", {})
        if isinstance(autonomy_scope, dict):
            summary["autonomy_scope"] = dict(autonomy_scope)
        return summary

    @staticmethod
    def _budget_world_model_summary(
        world_model_summary: dict[str, object],
        selected_by_source: dict[str, list[_BudgetChunk]],
    ) -> dict[str, object]:
        if not world_model_summary:
            return {}
        summary: dict[str, object] = {}
        for key in ("benchmark_family", "horizon"):
            if key in world_model_summary:
                summary[key] = world_model_summary[key]
        for source, key in (
            ("world_expected", "expected_artifacts"),
            ("world_forbidden", "forbidden_artifacts"),
            ("world_preserved", "preserved_artifacts"),
            ("world_missing_expected", "missing_expected_artifacts"),
            ("world_unsatisfied_expected", "unsatisfied_expected_contents"),
            ("world_present_forbidden", "present_forbidden_artifacts"),
            ("world_changed_preserved", "changed_preserved_artifacts"),
            ("world_updated_workflow", "updated_workflow_paths"),
        ):
            items = selected_by_source.get(source, [])
            if items:
                summary[key] = [str(chunk.payload) for chunk in items]
        preview_items = selected_by_source.get("world_preview", [])
        if preview_items:
            previews: dict[str, dict[str, object]] = {}
            for chunk in preview_items:
                if not isinstance(chunk.payload, dict):
                    continue
                path = str(chunk.payload.get("path", "")).strip()
                if not path:
                    continue
                window = {
                    "content": str(chunk.payload.get("content", "")),
                    "truncated": bool(chunk.payload.get("truncated", False)),
                    "line_start": _safe_int(chunk.payload.get("line_start", 1), 1),
                    "line_end": _safe_int(chunk.payload.get("line_end", 1), 1),
                }
                existing = previews.get(path)
                if existing is None:
                    previews[path] = window
                    continue
                windows = existing.get("edit_windows")
                if not isinstance(windows, list):
                    windows = [dict(existing)]
                windows.append(window)
                merged = dict(windows[0])
                merged["edit_windows"] = windows
                previews[path] = merged
            if previews:
                summary["workspace_file_previews"] = previews
        if "completion_ratio" in world_model_summary:
            summary["completion_ratio"] = world_model_summary["completion_ratio"]
        return summary

    @staticmethod
    def _preview_windows(preview: object) -> list[dict[str, object]]:
        if not isinstance(preview, dict):
            return []
        windows = preview.get("edit_windows", [])
        if isinstance(windows, list) and windows:
            return [window for window in windows if isinstance(window, dict)]
        return [preview]


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
