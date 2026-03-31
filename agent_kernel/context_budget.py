from __future__ import annotations

from dataclasses import dataclass

from .config import KernelConfig
from .state import AgentState


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
        if not payload["plan"]:
            payload["plan"] = plan[: self.config.llm_plan_max_items]
        if not payload["active_subgoal"]:
            payload["active_subgoal"] = active_subgoal
        return payload

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
        budget = max(256, self.config.tolbert_context_char_budget)
        max_chunks = max(1, self.config.tolbert_context_max_chunks)
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

        if llm_context_packet is not None:
            for item in llm_context_packet.get("control", {}).get("selected_context_chunks", []):
                text = str(item.get("text", "")).strip()
                if text:
                    score = 9
                    score += self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                    if role == "executor":
                        score += 2
                    chunks.append(
                        _BudgetChunk(
                            source="retrieval",
                            key=str(item.get("span_id", "")),
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
            ("present_forbidden_artifacts", "world_present_forbidden", 9 if role == "critic" else 7),
            ("changed_preserved_artifacts", "world_changed_preserved", 8 if role == "critic" else 6),
            ("updated_workflow_paths", "world_updated_workflow", 6),
        ):
            for item in list(world_model_summary.get(key) or [])[:6]:
                text = str(item)
                score = base + self._token_overlap_bonus(text, subgoal_tokens, expected, forbidden)
                chunks.append(_BudgetChunk(source_name, text, text, score, text))
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
            ("world_present_forbidden", "present_forbidden_artifacts"),
            ("world_changed_preserved", "changed_preserved_artifacts"),
            ("world_updated_workflow", "updated_workflow_paths"),
        ):
            items = selected_by_source.get(source, [])
            if items:
                summary[key] = [str(chunk.payload) for chunk in items]
        if "completion_ratio" in world_model_summary:
            summary["completion_ratio"] = world_model_summary["completion_ratio"]
        return summary
