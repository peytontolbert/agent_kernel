from __future__ import annotations

from typing import Any

from evals.metrics import EvalMetrics

from ...improvement_engine import ImprovementExperiment
from .improvement_common import normalized_control_mapping


def select_portfolio_campaign(
    planner: Any,
    metrics: EvalMetrics,
    *,
    max_candidates: int = 2,
    relative_score_floor: float = 0.75,
    absolute_score_margin: float = 0.04,
    recent_cycle_window: int = 6,
    observe_hypothesis_bonus: dict[str, float] | None = None,
) -> list[ImprovementExperiment]:
    planner_controls = planner._improvement_planner_controls()
    relative_score_floor = planner._planner_guardrail_float(
        planner_controls,
        scope="campaign",
        field="relative_score_floor",
        legacy_field="campaign_relative_score_floor",
        default=relative_score_floor,
        min_value=0.5,
        max_value=0.99,
    )
    absolute_score_margin = planner._planner_guardrail_float(
        planner_controls,
        scope="campaign",
        field="absolute_score_margin",
        legacy_field="campaign_absolute_score_margin",
        default=absolute_score_margin,
        min_value=0.0,
        max_value=0.2,
    )
    ranked = planner.rank_experiments(metrics)
    if not ranked:
        return []
    top_score = float(ranked[0].score)
    recent_activity = {
        candidate.subsystem: (
            planner.recent_campaign_surface_activity_summary(
                subsystem=candidate.subsystem,
                recent_cycle_window=recent_cycle_window,
            )
            if planner._campaign_surface_key(candidate.subsystem) == "retrieval_stack"
            else planner.recent_subsystem_activity_summary(
                subsystem=candidate.subsystem,
                recent_cycle_window=recent_cycle_window,
            )
        )
        for candidate in ranked
    }
    broad_observe_signal = planner._broad_coding_observe_diversification_signal(metrics)
    lead_recent_activity = recent_activity.get(ranked[0].subsystem, {})
    breadth_pressure = planner._campaign_breadth_pressure(lead_recent_activity)
    observe_bonus = {
        str(subsystem).strip(): float(value)
        for subsystem, value in dict(observe_hypothesis_bonus or {}).items()
        if str(subsystem).strip()
    }
    selected: list[ImprovementExperiment] = []
    selected_surfaces: set[str] = set()
    available = list(ranked)
    while available and len(selected) < max(1, max_candidates):
        candidate_pool = list(available)
        if bool(broad_observe_signal.get("active", False)) and not bool(
            broad_observe_signal.get("retrieval_emergency", False)
        ):
            non_retrieval_available = [
                candidate
                for candidate in candidate_pool
                if planner._base_subsystem(candidate.subsystem) not in {"retrieval", "tolbert_model", "qwen_adapter"}
            ]
            if non_retrieval_available:
                candidate_pool = non_retrieval_available
            elif not selected:
                return []
        best_candidate: ImprovementExperiment | None = None
        best_sort_key: tuple[float, int, str] = (float("-inf"), -1, "")
        for candidate in candidate_pool:
            if planner._campaign_surface_key(candidate.subsystem) in selected_surfaces:
                continue
            relative_score = 0.0 if top_score <= 0.0 else float(candidate.score) / top_score
            score_margin = top_score - float(candidate.score)
            adjusted_score, reasons = planner._portfolio_adjusted_experiment_score(
                candidate,
                recent_activity=recent_activity.get(candidate.subsystem, {}),
                planner_controls=planner_controls,
            )
            strategy_choice = planner._strategy_portfolio_choice(
                candidate,
                metrics=metrics,
                recent_activity=recent_activity.get(candidate.subsystem, {}),
                broad_observe_signal=broad_observe_signal,
            )
            strategy_candidate = dict(strategy_choice.get("strategy_candidate", {}))
            strategy_recent_activity = dict(strategy_choice.get("recent_activity", {}))
            strategy_memory_summary = dict(strategy_choice.get("strategy_memory_summary", {}))
            strategy_score_delta = float(strategy_choice.get("score_delta", 0.0) or 0.0)
            adjusted_score += strategy_score_delta
            hypothesis_score_delta = float(observe_bonus.get(candidate.subsystem, 0.0) or 0.0)
            adjusted_score += hypothesis_score_delta
            if breadth_pressure > 0.0:
                reasons.append(f"campaign_breadth_pressure={breadth_pressure:.4f}")
            if strategy_score_delta != 0.0:
                reasons.append(f"strategy_history_score_delta={strategy_score_delta:.4f}")
            reasons.extend(
                [
                    str(reason).strip()
                    for reason in list(strategy_choice.get("reasons", []))
                    if str(reason).strip()
                ]
            )
            if hypothesis_score_delta != 0.0:
                reasons.append(f"observe_hypothesis_score_delta={hypothesis_score_delta:.4f}")
            if float(strategy_memory_summary.get("score_delta", 0.0) or 0.0) != 0.0:
                reasons.append(
                    "strategy_memory_score_delta="
                    f"{float(strategy_memory_summary.get('score_delta', 0.0) or 0.0):.4f}"
                )
            if bool(broad_observe_signal.get("active", False)):
                if bool(broad_observe_signal.get("retrieval_emergency", False)):
                    reasons.append("broad_observe_diversification_blocked_by_retrieval_emergency")
                elif planner._base_subsystem(candidate.subsystem) not in {"retrieval", "tolbert_model", "qwen_adapter"}:
                    reasons.append("broad_observe_diversification_preferred")
            if (
                planner._campaign_surface_key(candidate.subsystem) == "retrieval_stack"
                and bool(strategy_memory_summary.get("avoid_reselection", False))
                and any(
                    planner._campaign_surface_key(other.subsystem)
                    != planner._campaign_surface_key(candidate.subsystem)
                    for other in candidate_pool
                )
            ):
                continue
            if selected and relative_score < relative_score_floor and score_margin > absolute_score_margin:
                continue
            sort_key = (adjusted_score, candidate.priority, candidate.subsystem)
            if best_candidate is None or sort_key > best_sort_key:
                evidence = dict(candidate.evidence)
                strategy_candidate_payload = {
                    **strategy_candidate,
                    "parent_strategy_node_ids": list(
                        strategy_memory_summary.get("selected_parent_strategy_node_ids", [])
                    ),
                    "best_retained_strategy_node_id": str(
                        strategy_memory_summary.get("best_retained_strategy_node_id", "")
                    ).strip(),
                    "continuation_parent_node_id": str(
                        strategy_memory_summary.get("continuation_parent_node_id", "")
                    ).strip(),
                    "continuation_artifact_path": str(
                        strategy_memory_summary.get("continuation_artifact_path", "")
                    ).strip(),
                    "continuation_workspace_ref": str(
                        strategy_memory_summary.get("continuation_workspace_ref", "")
                    ).strip(),
                    "continuation_branch": str(
                        strategy_memory_summary.get("continuation_branch", "")
                    ).strip(),
                    "selected_parent_nodes": list(strategy_memory_summary.get("selected_parent_nodes", []) or []),
                    "best_retained_snapshot": dict(strategy_memory_summary.get("best_retained_snapshot", {}) or {}),
                    "parent_control_surface": dict(strategy_memory_summary.get("parent_control_surface", {}) or {}),
                    "recent_activity": dict(strategy_recent_activity),
                    "score_delta": round(strategy_score_delta, 4),
                    "strategy_memory_score_delta": round(
                        float(strategy_memory_summary.get("score_delta", 0.0) or 0.0), 4
                    ),
                    "strategy_memory_recent_rejects": int(
                        strategy_memory_summary.get("recent_rejects", 0) or 0
                    ),
                    "strategy_memory_recent_retains": int(
                        strategy_memory_summary.get("recent_retains", 0) or 0
                    ),
                    "strategy_memory_avoid_reselection": bool(
                        strategy_memory_summary.get("avoid_reselection", False)
                    ),
                }
                retention_inputs = (
                    dict(strategy_candidate_payload.get("retention_inputs", {}))
                    if isinstance(strategy_candidate_payload.get("retention_inputs", {}), dict)
                    else {}
                )
                retention_inputs.update(
                    {
                        "selected_subsystem": candidate.subsystem,
                        "portfolio_adjusted_score": round(adjusted_score, 4),
                        "relative_score": round(relative_score, 4),
                        "score_margin": round(score_margin, 4),
                        "recent_cycle_window": max(1, recent_cycle_window),
                        "parent_strategy_node_ids": list(
                            strategy_memory_summary.get("selected_parent_strategy_node_ids", [])
                        ),
                        "best_retained_strategy_node_id": str(
                            strategy_memory_summary.get("best_retained_strategy_node_id", "")
                        ).strip(),
                        "selected_parent_nodes": list(strategy_memory_summary.get("selected_parent_nodes", []) or []),
                        "best_retained_snapshot": dict(
                            strategy_memory_summary.get("best_retained_snapshot", {}) or {}
                        ),
                        "parent_control_surface": dict(
                            strategy_memory_summary.get("parent_control_surface", {}) or {}
                        ),
                    }
                )
                strategy_candidate_payload["retention_inputs"] = retention_inputs
                evidence["portfolio"] = {
                    "adjusted_score": round(adjusted_score, 4),
                    "relative_score": round(relative_score, 4),
                    "score_margin": round(score_margin, 4),
                    "campaign_breadth_pressure": round(breadth_pressure, 4),
                    "broad_observe_diversification": dict(broad_observe_signal),
                    "recent_activity": dict(recent_activity.get(candidate.subsystem, {})),
                    "reasons": reasons,
                }
                evidence["strategy_candidate"] = strategy_candidate_payload
                best_candidate = ImprovementExperiment(
                    subsystem=candidate.subsystem,
                    reason=candidate.reason,
                    priority=candidate.priority,
                    expected_gain=candidate.expected_gain,
                    estimated_cost=candidate.estimated_cost,
                    score=candidate.score,
                    evidence=evidence,
                )
                best_sort_key = sort_key
        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_surfaces.add(planner._campaign_surface_key(best_candidate.subsystem))
        available = [
            candidate
            for candidate in available
            if planner._campaign_surface_key(candidate.subsystem) != planner._campaign_surface_key(best_candidate.subsystem)
        ]
    if selected:
        return selected
    return [ranked[0]]


def strategy_portfolio_choice(
    planner: Any,
    candidate: ImprovementExperiment,
    *,
    metrics: EvalMetrics,
    recent_activity: dict[str, object] | None = None,
    broad_observe_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    options = planner._strategy_candidate_options(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
    best_choice: dict[str, object] | None = None
    best_sort_key: tuple[float, int, str] = (float("-inf"), -1, "")
    for option in options:
        strategy_candidate = planner._normalized_strategy_candidate(
            candidate_subsystem=candidate.subsystem,
            strategy_candidate=option,
        )
        strategy_recent_activity = planner.recent_strategy_activity_summary(
            strategy_candidate_id=str(strategy_candidate.get("strategy_candidate_id", "")).strip()
        )
        history_score_delta = planner._strategy_history_score_delta(
            strategy_candidate,
            recent_activity=strategy_recent_activity,
        )
        strategy_memory = planner._strategy_memory_summary(
            candidate_subsystem=candidate.subsystem,
            strategy_candidate=strategy_candidate,
        )
        strategy_candidate = planner._apply_strategy_memory_control_surface(
            candidate_subsystem=candidate.subsystem,
            strategy_candidate=strategy_candidate,
            strategy_memory_summary=strategy_memory,
        )
        memory_score_delta = float(strategy_memory.get("score_delta", 0.0) or 0.0)
        selection_bonus = float(strategy_candidate.get("selection_bonus", 0.0) or 0.0)
        retained_reuse_bonus = 0.0
        if (
            int(strategy_recent_activity.get("retained_cycles", 0) or 0) > 0
            and str(strategy_candidate.get("origin", "")).strip() != "discovered_strategy"
        ):
            retained_reuse_bonus = round(
                min(0.02, 0.015 * int(strategy_recent_activity.get("retained_cycles", 0) or 0)),
                4,
            )
        total_score_delta = round(
            selection_bonus + history_score_delta + memory_score_delta + retained_reuse_bonus,
            4,
        )
        choice_reasons = [
            str(reason).strip()
            for reason in list(strategy_candidate.get("portfolio_reasons", []))
            if str(reason).strip()
        ]
        if selection_bonus != 0.0:
            choice_reasons.append(f"strategy_selection_bonus={selection_bonus:.4f}")
        if retained_reuse_bonus != 0.0:
            choice_reasons.append(f"retained_strategy_reuse_bonus={retained_reuse_bonus:.4f}")
        origin_rank = 1 if str(strategy_candidate.get("origin", "")).strip() == "discovered_strategy" else 0
        sort_key = (
            total_score_delta,
            origin_rank,
            str(strategy_candidate.get("strategy_candidate_id", "")).strip(),
        )
        if best_choice is None or sort_key > best_sort_key:
            best_choice = {
                "strategy_candidate": strategy_candidate,
                "recent_activity": strategy_recent_activity,
                "strategy_memory_summary": strategy_memory,
                "score_delta": total_score_delta,
                "reasons": choice_reasons,
            }
            best_sort_key = sort_key
    if best_choice is not None:
        return best_choice
    fallback = planner._normalized_strategy_candidate(
        candidate_subsystem=candidate.subsystem,
        strategy_candidate={},
    )
    return {
        "strategy_candidate": fallback,
        "recent_activity": {},
        "strategy_memory_summary": {},
        "score_delta": 0.0,
        "reasons": [],
    }


def strategy_memory_summary(
    planner: Any,
    *,
    candidate_subsystem: str,
    strategy_candidate: dict[str, object] | None,
) -> dict[str, object]:
    return planner._strategy_prior_store.summarize(
        runtime_config=planner.runtime_config,
        candidate_subsystem=str(candidate_subsystem).strip(),
        strategy_candidate=strategy_candidate,
    )


def apply_strategy_memory_control_surface(
    planner: Any,
    *,
    candidate_subsystem: str,
    strategy_candidate: dict[str, object] | None,
    strategy_memory_summary: dict[str, object] | None,
) -> dict[str, object]:
    candidate = dict(strategy_candidate or {})
    summary = strategy_memory_summary if isinstance(strategy_memory_summary, dict) else {}
    control_surface = (
        dict(summary.get("parent_control_surface", {}))
        if isinstance(summary.get("parent_control_surface", {}), dict)
        else {}
    )
    if not control_surface:
        return candidate
    generation_basis = (
        dict(candidate.get("generation_basis", {}))
        if isinstance(candidate.get("generation_basis", {}), dict)
        else {}
    )
    target_conditions = (
        dict(candidate.get("target_conditions", {}))
        if isinstance(candidate.get("target_conditions", {}), dict)
        else {}
    )
    controls = dict(candidate.get("controls", {})) if isinstance(candidate.get("controls", {}), dict) else {}
    retention_inputs = (
        dict(candidate.get("retention_inputs", {}))
        if isinstance(candidate.get("retention_inputs", {}), dict)
        else {}
    )
    expected_signals = normalized_control_mapping(
        {
            "items": [
                *list(candidate.get("expected_signals", []) or []),
                "family_breadth_gain" if bool(control_surface.get("prefer_family_breadth", False)) else "",
                "unattended_closeout"
                if bool(control_surface.get("prefer_unattended_closeout", False))
                else "",
            ]
        },
        list_fields=("items",),
    ).get("items", [])
    semantic_hypotheses = normalized_control_mapping(
        {
            "items": [
                *list(candidate.get("semantic_hypotheses", []) or []),
                *list(control_surface.get("semantic_hypotheses", []) or []),
                *list(control_surface.get("reuse_conditions", []) or []),
            ]
        },
        list_fields=("items",),
    ).get("items", [])[:16]
    generation_basis["strategy_memory_parent_strategy_ids"] = list(
        summary.get("selected_parent_strategy_node_ids", []) or []
    )
    generation_basis["strategy_memory_closeout_modes"] = list(control_surface.get("closeout_modes", []) or [])
    generation_basis["strategy_memory_required_family_gains"] = list(
        control_surface.get("required_family_gains", []) or []
    )
    generation_basis["strategy_memory_semantic_hypotheses"] = list(
        control_surface.get("semantic_hypotheses", []) or []
    )
    if control_surface.get("required_family_gains", []):
        target_conditions["required_family_gains"] = list(control_surface.get("required_family_gains", []) or [])
    if control_surface.get("reuse_conditions", []):
        target_conditions["reuse_conditions"] = list(control_surface.get("reuse_conditions", []) or [])
    if control_surface.get("avoid_conditions", []):
        target_conditions["avoid_conditions"] = list(control_surface.get("avoid_conditions", []) or [])
    if bool(control_surface.get("prefer_family_breadth", False)):
        target_conditions["prefer_family_breadth"] = True
    if bool(control_surface.get("prefer_unattended_closeout", False)):
        target_conditions["prefer_unattended_closeout"] = True
    controls["strategy_memory_parent_control_surface"] = control_surface
    controls["strategy_memory_best_parent_strategy_node_id"] = str(
        control_surface.get("best_parent_strategy_node_id", "")
    ).strip()
    retention_inputs["strategy_memory_parent_control_surface"] = control_surface
    retention_inputs["strategy_memory_selected_parent_nodes"] = list(summary.get("selected_parent_nodes", []) or [])
    retention_inputs["strategy_memory_best_retained_snapshot"] = dict(summary.get("best_retained_snapshot", {}) or {})
    selection_bonus = float(candidate.get("selection_bonus", 0.0) or 0.0)
    if bool(control_surface.get("prefer_family_breadth", False)):
        selection_bonus += 0.01
    if bool(control_surface.get("prefer_unattended_closeout", False)):
        selection_bonus += 0.01
    return planner._normalized_strategy_candidate(
        candidate_subsystem=candidate_subsystem,
        strategy_candidate={
            **candidate,
            "generation_basis": generation_basis,
            "target_conditions": target_conditions,
            "controls": controls,
            "retention_inputs": retention_inputs,
            "expected_signals": expected_signals,
            "semantic_hypotheses": semantic_hypotheses,
            "selection_bonus": round(selection_bonus, 4),
            "parent_control_surface": control_surface,
        },
    )


def portfolio_adjusted_experiment_score(
    planner: Any,
    candidate: ImprovementExperiment,
    *,
    recent_activity: dict[str, object],
    planner_controls: dict[str, object] | None = None,
) -> tuple[float, list[str]]:
    adjusted = float(candidate.score)
    reasons = [f"base_score={candidate.score:.4f}"]
    resolved_planner_controls = planner_controls or {}
    exploration_bonus = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_exploration_bonus",
        0.01,
        min_value=0.0,
        max_value=0.05,
    )
    saturation_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_selection_saturation_penalty_per_cycle",
        0.01,
        min_value=0.0,
        max_value=0.05,
    )
    retention_multiplier = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_recent_retention_bonus_multiplier",
        1.0,
        min_value=0.0,
        max_value=2.0,
    )
    rejection_multiplier = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_recent_rejection_penalty_multiplier",
        1.0,
        min_value=0.0,
        max_value=2.0,
    )
    no_yield_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_no_yield_penalty_per_cycle",
        0.02,
        min_value=0.0,
        max_value=0.08,
    )
    regression_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_regression_penalty_per_cycle",
        0.0125,
        min_value=0.0,
        max_value=0.06,
    )
    phase_gate_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_phase_gate_penalty_per_cycle",
        0.01,
        min_value=0.0,
        max_value=0.05,
    )
    incomplete_cycle_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_incomplete_cycle_penalty_per_cycle",
        0.025,
        min_value=0.0,
        max_value=0.08,
    )
    reconciled_failure_penalty_per_cycle = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_reconciled_failure_penalty_per_cycle",
        0.03,
        min_value=0.0,
        max_value=0.1,
    )
    repeated_phase_gate_reason_penalty = planner._planner_control_float(
        resolved_planner_controls,
        "portfolio_repeated_phase_gate_reason_penalty",
        0.03,
        min_value=0.0,
        max_value=0.12,
    )
    selected_cycles = int(recent_activity.get("selected_cycles", 0))
    retained_cycles = int(recent_activity.get("retained_cycles", 0))
    rejected_cycles = int(recent_activity.get("rejected_cycles", 0))
    no_yield_cycles = int(recent_activity.get("no_yield_cycles", 0))
    recent_incomplete_cycles = int(recent_activity.get("recent_incomplete_cycles", 0))
    recent_regression_cycles = int(recent_activity.get("recent_regression_cycles", 0))
    recent_phase_gate_failure_cycles = int(recent_activity.get("recent_phase_gate_failure_cycles", 0))
    recent_reconciled_failure_cycles = int(recent_activity.get("recent_reconciled_failure_cycles", 0))
    repeated_phase_gate_reason_count = int(recent_activity.get("repeated_phase_gate_reason_count", 0) or 0)
    if selected_cycles == 0:
        adjusted += exploration_bonus
        reasons.append(f"exploration_bonus={exploration_bonus:.4f}")
    else:
        saturation_penalty = min(0.03, selected_cycles * saturation_penalty_per_cycle)
        adjusted -= saturation_penalty
        reasons.append(f"selection_saturation_penalty={saturation_penalty:.4f}")
    if no_yield_cycles > 0:
        no_yield_penalty = min(0.08, no_yield_cycles * no_yield_penalty_per_cycle)
        adjusted -= no_yield_penalty
        reasons.append(f"no_yield_penalty={no_yield_penalty:.4f}")
    if recent_incomplete_cycles > 0:
        incomplete_cycle_penalty = min(0.1, recent_incomplete_cycles * incomplete_cycle_penalty_per_cycle)
        adjusted -= incomplete_cycle_penalty
        reasons.append(f"recent_incomplete_cycle_penalty={incomplete_cycle_penalty:.4f}")
    retained_delta = max(0.0, float(recent_activity.get("average_retained_pass_rate_delta", 0.0)))
    if retained_cycles > 0 and retained_delta > 0.0:
        retention_bonus = min(0.015, (retained_delta * 0.1 + retained_cycles * 0.0025) * retention_multiplier)
        adjusted += retention_bonus
        reasons.append(f"recent_retention_bonus={retention_bonus:.4f}")
    rejected_delta = max(0.0, -float(recent_activity.get("average_rejected_pass_rate_delta", 0.0)))
    if rejected_cycles > 0:
        rejection_penalty = min(0.03, (rejected_cycles * 0.0125 + rejected_delta) * rejection_multiplier)
        adjusted -= rejection_penalty
        reasons.append(f"recent_rejection_penalty={rejection_penalty:.4f}")
    if recent_regression_cycles > 0:
        regression_penalty = min(0.06, recent_regression_cycles * regression_penalty_per_cycle)
        adjusted -= regression_penalty
        reasons.append(f"recent_regression_penalty={regression_penalty:.4f}")
    if recent_phase_gate_failure_cycles > 0:
        phase_gate_penalty = min(0.05, recent_phase_gate_failure_cycles * phase_gate_penalty_per_cycle)
        adjusted -= phase_gate_penalty
        reasons.append(f"recent_phase_gate_penalty={phase_gate_penalty:.4f}")
    if recent_reconciled_failure_cycles > 0:
        reconciled_failure_penalty = min(
            0.12,
            recent_reconciled_failure_cycles * reconciled_failure_penalty_per_cycle,
        )
        adjusted -= reconciled_failure_penalty
        reasons.append(f"recent_reconciled_failure_penalty={reconciled_failure_penalty:.4f}")
    if repeated_phase_gate_reason_count > 0:
        repeated_reason_penalty = min(
            0.12,
            repeated_phase_gate_reason_count * repeated_phase_gate_reason_penalty,
        )
        adjusted -= repeated_reason_penalty
        reasons.append(f"repeated_phase_gate_reason_penalty={repeated_reason_penalty:.4f}")
    observation_timeout_penalty, observation_timeout_reasons = planner._recent_observation_timeout_penalty(
        recent_activity,
        planner_controls=resolved_planner_controls,
    )
    if observation_timeout_penalty > 0.0:
        adjusted -= observation_timeout_penalty
        reasons.extend(observation_timeout_reasons)
    promotion_failure_penalty, promotion_failure_reasons = planner._recent_promotion_failure_penalty(
        recent_activity,
        planner_controls=resolved_planner_controls,
    )
    if promotion_failure_penalty > 0.0:
        adjusted -= promotion_failure_penalty
        reasons.extend(promotion_failure_reasons)
    decision_quality_score = float(recent_activity.get("decision_quality_score", 0.0))
    if decision_quality_score != 0.0:
        adjusted += decision_quality_score * 0.35
        reasons.append(f"decision_quality_adjustment={decision_quality_score * 0.35:.4f}")
    return round(adjusted, 4), reasons


__all__ = [
    "apply_strategy_memory_control_surface",
    "portfolio_adjusted_experiment_score",
    "select_portfolio_campaign",
    "strategy_memory_summary",
    "strategy_portfolio_choice",
]
