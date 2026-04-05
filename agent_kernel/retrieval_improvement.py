from __future__ import annotations

from copy import deepcopy

from evals.metrics import EvalMetrics
from .improvement_common import (
    build_standard_proposal_artifact,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)


def build_retrieval_proposal_artifact(
    metrics: EvalMetrics,
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    proposals = _proposals(metrics, focus=generation_focus)
    base_overrides = retained_retrieval_overrides(current_payload)
    asset_controls = _asset_controls(
        metrics,
        focus=generation_focus,
        baseline=retained_retrieval_asset_controls(current_payload),
    )
    merged_overrides = deepcopy(base_overrides)
    merged_overrides.update(_merge_overrides(proposals))
    return build_standard_proposal_artifact(
        artifact_kind="retrieval_policy_set",
        generation_focus=generation_focus,
        retention_gate=_retrieval_retention_gate(metrics),
        proposals=proposals,
        extra_sections={
            "asset_strategy": _asset_strategy(generation_focus),
            "overrides": merged_overrides,
            "asset_controls": asset_controls,
            "asset_rebuild_plan": _asset_rebuild_plan(generation_focus, asset_controls),
            "preview_controls": _preview_controls(metrics, focus=generation_focus),
        },
    )


def _retrieval_retention_gate(metrics: EvalMetrics) -> dict[str, object]:
    gate = retention_gate_preset("retrieval")
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    if carryover_repair_rate < 0.1:
        return gate
    gate["require_trusted_carryover_repair_improvement"] = True
    gate["min_trusted_carryover_repair_rate"] = round(carryover_repair_rate, 2)
    gate["min_trusted_carryover_verified_step_delta"] = 1
    gate["max_low_confidence_episode_regression"] = 0
    return gate


def retained_retrieval_overrides(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="retrieval_policy_set", section="overrides")


def retained_retrieval_asset_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="retrieval_policy_set", section="asset_controls")


def _proposals(metrics: EvalMetrics, *, focus: str) -> list[dict[str, object]]:
    low_confidence_rate = 0.0 if metrics.total == 0 else metrics.low_confidence_episodes / max(1, metrics.total)
    trusted_retrieval_rate = 0.0 if metrics.total == 0 else min(1.0, metrics.trusted_retrieval_steps / max(1, metrics.total))
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    confidence_pressure = _confidence_pressure(metrics)
    stalled_without_trust_rate = _stalled_without_trust_rate(metrics)
    successful_without_guidance_rate = _successful_without_guidance_rate(metrics)
    bootstrap_guidance = _bootstrap_guidance(metrics)
    selected_without_influence_rate = _selected_without_influence_rate(metrics)
    activation_gap_bootstrap = bootstrap_guidance and selected_without_influence_rate >= 0.2
    confidence_relaxation = (
        min(
            0.1,
            successful_without_guidance_rate * 0.1
            + selected_without_influence_rate * (0.12 if activation_gap_bootstrap else 0.08),
        )
        if bootstrap_guidance
        else 0.0
    )
    tighten_context = confidence_pressure >= 0.12 or stalled_without_trust_rate >= 0.1
    widen_branch_routing = low_confidence_rate >= 0.2 or stalled_without_trust_rate >= 0.1 or bootstrap_guidance
    proposals: list[dict[str, object]] = []

    if focus in {"balanced", "confidence"}:
        if bootstrap_guidance:
            tighten_context = False
        deterministic_confidence = round(
            max(
                0.74 if activation_gap_bootstrap else (0.76 if bootstrap_guidance else 0.78),
                min(
                    0.8 if activation_gap_bootstrap else (0.82 if bootstrap_guidance else 0.9),
                    0.78
                    + confidence_pressure * (0.06 if activation_gap_bootstrap else (0.08 if bootstrap_guidance else 0.12))
                    + (1.0 - trusted_retrieval_rate) * (0.02 if bootstrap_guidance else 0.04)
                    - selected_without_influence_rate * (0.04 if activation_gap_bootstrap else 0.0),
                ),
            ),
            2,
        )
        proposals.append(
            {
                "proposal_id": "retrieval:confidence_gating",
                "area": "confidence",
                "reason": (
                    "trusted retrieval is already converting into verified long-horizon repairs, so confidence gating should preserve precise repair reuse instead of staying in blind bootstrap mode"
                    if carryover_repair_rate >= 0.2
                    else (
                    "retrieval is being selected but still not converting into trusted guidance across most successful runs, so confidence gating should keep bootstrap activation pressure on"
                    if activation_gap_bootstrap
                    else (
                    "successful runs still complete without retrieval guidance, so confidence gating should bootstrap retrieval activation"
                    if bootstrap_guidance
                    else "low-confidence retrieval remains common relative to trusted retrieval usage"
                    )
                    )
                ),
                "overrides": {
                    "tolbert_confidence_threshold": round(
                        max(
                            0.15,
                            min(
                                0.22 if activation_gap_bootstrap else (0.24 if bootstrap_guidance else 0.55),
                                0.2 + confidence_pressure * 0.5 - confidence_relaxation,
                            ),
                        ),
                        2,
                    ),
                    "tolbert_skill_ranking_min_confidence": round(
                        max(
                            0.34 if activation_gap_bootstrap else (0.36 if bootstrap_guidance else 0.4),
                            min(
                                0.4 if activation_gap_bootstrap else (0.42 if bootstrap_guidance else 0.7),
                                0.35 + confidence_pressure * 0.4 - confidence_relaxation,
                            ),
                        ),
                        2,
                    ),
                    "tolbert_deterministic_command_confidence": deterministic_confidence,
                    "tolbert_first_step_direct_command_confidence": round(
                        max(
                            0.88 if activation_gap_bootstrap else (0.9 if bootstrap_guidance else 0.94),
                            min(
                                0.98,
                                deterministic_confidence
                                + (
                                    0.12
                                    if activation_gap_bootstrap
                                    else (0.1 if bootstrap_guidance else 0.14)
                                ),
                            ),
                        ),
                        2,
                    ),
                    "tolbert_direct_command_min_score": 1 if bootstrap_guidance else (2 if confidence_pressure >= 0.15 else 1),
                    "tolbert_branch_confidence_margin": round(
                        max(
                            0.07 if activation_gap_bootstrap else (0.08 if bootstrap_guidance else 0.1),
                            min(
                                0.18 if activation_gap_bootstrap else 0.22,
                                0.11 + confidence_pressure * 0.12 - confidence_relaxation * 0.4,
                            ),
                        ),
                        2,
                    ),
                    "tolbert_low_confidence_widen_threshold": round(
                        max(
                            0.48 if activation_gap_bootstrap else (0.5 if bootstrap_guidance else 0.55),
                            min(
                                0.68 if activation_gap_bootstrap else 0.75,
                                0.58 + confidence_pressure * 0.18 - confidence_relaxation * 0.5,
                            ),
                        ),
                        2,
                    ),
                    "tolbert_distractor_penalty": round(
                        max(5.0, min(8.0, 5.5 + confidence_pressure * 2.5)),
                        1,
                    ),
                    "tolbert_branch_results": 4 if activation_gap_bootstrap else (3 if bootstrap_guidance else (2 if tighten_context else 3)),
                    "tolbert_global_results": 2 if bootstrap_guidance else (1 if tighten_context else 2),
                    "tolbert_context_max_chunks": 7 if activation_gap_bootstrap else (6 if bootstrap_guidance else (4 if confidence_pressure >= 0.3 else (5 if tighten_context else 8))),
                    "tolbert_max_spans_per_source": 2 if bootstrap_guidance else (1 if tighten_context else 2),
                    "tolbert_top_branches": 6 if activation_gap_bootstrap else (5 if bootstrap_guidance else (4 if widen_branch_routing else 3)),
                    "tolbert_ancestor_branch_levels": 5 if activation_gap_bootstrap else (4 if bootstrap_guidance else (3 if widen_branch_routing else 2)),
                },
            }
        )
    if focus in {"balanced", "breadth"}:
        proposals.append(
            {
                "proposal_id": "retrieval:breadth_rebalance",
                "area": "breadth",
                "reason": "retrieval should narrow broad noisy context while preserving branch coverage",
                "overrides": {
                    "tolbert_branch_results": 2,
                    "tolbert_global_results": 1,
                    "tolbert_context_max_chunks": 4,
                    "tolbert_max_spans_per_source": 1,
                    "tolbert_distractor_penalty": 5.0,
                },
            }
        )
    if focus in {"balanced", "routing"}:
        proposals.append(
            {
                "proposal_id": "retrieval:routing_depth",
                "area": "routing",
                "reason": "branch routing should widen under uncertainty but stay deeper when confidence is stable",
                "overrides": {
                    "tolbert_top_branches": 4 if low_confidence_rate >= 0.2 else 3,
                    "tolbert_ancestor_branch_levels": 3 if low_confidence_rate >= 0.2 else 2,
                    "tolbert_branch_confidence_margin": round(max(0.08, min(0.2, 0.12 + low_confidence_rate * 0.1)), 2),
                    "tolbert_low_confidence_widen_threshold": round(max(0.45, min(0.7, 0.55 + low_confidence_rate * 0.2)), 2),
                },
            }
        )
    if focus in {"balanced", "safety"}:
        proposals.append(
            {
                "proposal_id": "retrieval:direct_command_safety",
                "area": "safety",
                "reason": "direct retrieval commands should only trigger when retrieval is both confident and sufficiently aligned",
                "overrides": {
                    "tolbert_direct_command_min_score": 2,
                    "tolbert_first_step_direct_command_confidence": round(max(0.92, min(0.98, 0.93 + (1.0 - trusted_retrieval_rate) * 0.03)), 2),
                    "tolbert_deterministic_command_confidence": round(max(0.75, min(0.9, 0.78 + low_confidence_rate * 0.15)), 2),
                    "tolbert_distractor_penalty": round(max(5.0, min(8.0, 5.5 + low_confidence_rate * 3.0)), 1),
                },
            }
        )
    return proposals


def _merge_overrides(proposals: list[dict[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for proposal in proposals:
        overrides = proposal.get("overrides", {})
        if isinstance(overrides, dict):
            merged.update(deepcopy(overrides))
    return merged


def _asset_strategy(focus: str) -> str:
    if focus == "confidence":
        return "confidence_hardening"
    if focus == "breadth":
        return "breadth_pruning"
    if focus == "routing":
        return "routing_expansion"
    if focus == "safety":
        return "negative_signal_enrichment"
    return "balanced_rebuild"


def _asset_controls(
    metrics: EvalMetrics,
    *,
    focus: str,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    confidence_pressure = _confidence_pressure(metrics)
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    bootstrap_guidance = focus == "confidence" and _bootstrap_guidance(metrics)
    activation_gap_bootstrap = (
        bootstrap_guidance and _selected_without_influence_rate(metrics) >= 0.2
    )
    controls: dict[str, object] = {
        "include_failure_catalog_spans": True,
        "include_negative_episode_spans": True,
        "include_tool_candidate_spans": True,
        "include_episode_step_spans": True,
        "include_skill_procedure_spans": True,
        "max_episode_step_spans_per_task": 3,
        "max_failure_spans_per_task": 2,
        "prefer_failure_alignment": confidence_pressure >= 0.2,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if carryover_repair_rate >= 0.1:
        controls["prefer_successful_carryover_repairs"] = True
        controls["max_episode_step_spans_per_task"] = max(
            int(controls.get("max_episode_step_spans_per_task", 3) or 0),
            4,
        )
    if focus == "confidence":
        controls["prefer_failure_alignment"] = True
        controls["max_failure_spans_per_task"] = 4 if confidence_pressure >= 0.25 else 3
        if carryover_repair_rate >= 0.1:
            controls["max_episode_step_spans_per_task"] = max(
                int(controls.get("max_episode_step_spans_per_task", 3) or 0),
                4,
            )
        elif activation_gap_bootstrap:
            controls["max_episode_step_spans_per_task"] = 3
            controls["prefer_tool_candidate_bootstrap"] = True
        elif bootstrap_guidance:
            controls["max_episode_step_spans_per_task"] = 2
    elif focus == "breadth":
        controls["max_episode_step_spans_per_task"] = 2
        controls["max_failure_spans_per_task"] = 1
    elif focus == "routing":
        controls["include_tool_candidate_spans"] = True
        controls["include_episode_step_spans"] = True
        controls["max_episode_step_spans_per_task"] = 4
    elif focus == "safety":
        controls["include_negative_episode_spans"] = True
        controls["prefer_failure_alignment"] = True
        controls["max_failure_spans_per_task"] = 4
    return controls


def _asset_rebuild_plan(focus: str, controls: dict[str, object]) -> dict[str, object]:
    source_priorities = [
        "native_repo_structure",
        "successful_trace_procedures",
        "failure_catalog",
        "negative_episode_spans",
        "tool_candidates",
    ]
    if bool(controls.get("prefer_successful_carryover_repairs", False)):
        source_priorities = [
            "native_repo_structure",
            "successful_trace_procedures",
            "tool_candidates",
            "failure_catalog",
            "negative_episode_spans",
        ]
    elif focus == "confidence" and (
        int(controls.get("max_episode_step_spans_per_task", 3) or 0) <= 2
        or bool(controls.get("prefer_tool_candidate_bootstrap", False))
    ):
        source_priorities = [
            "native_repo_structure",
            "tool_candidates",
            "successful_trace_procedures",
            "failure_catalog",
            "negative_episode_spans",
        ]
    return {
        "rebuild_required": True,
        "strategy": _asset_strategy(focus),
        "source_priorities": source_priorities,
        "controls": deepcopy(controls),
    }


def _preview_controls(metrics: EvalMetrics, *, focus: str) -> dict[str, object]:
    pass_rate = _pass_rate(metrics)
    low_signal_preview = (
        pass_rate >= 0.9
        and int(metrics.novel_valid_command_steps or 0) <= 0
        and int(metrics.tolbert_primary_episodes or 0) <= 0
    )
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    if not low_signal_preview and carryover_repair_rate < 0.1:
        return {}

    comparison_task_limit_floor = 12 if low_signal_preview else 8
    if carryover_repair_rate >= 0.1:
        comparison_task_limit_floor = max(comparison_task_limit_floor, 24)
    elif focus == "confidence":
        comparison_task_limit_floor = max(comparison_task_limit_floor, 16)

    priority_families: list[str] = []
    priority_weights: dict[str, float] = {}

    def _push_family(family: str, weight: float) -> None:
        normalized = str(family).strip()
        if not normalized:
            return
        if normalized not in priority_families:
            priority_families.append(normalized)
        priority_weights[normalized] = max(weight, float(priority_weights.get(normalized, 0.0) or 0.0))

    _push_family("benchmark_candidate", 4.0)
    if metrics.total_by_benchmark_family.get("integration", 0) > 0:
        _push_family("integration", 3.0)
    if metrics.total_by_benchmark_family.get("repository", 0) > 0:
        _push_family("repository", 2.0)
    if metrics.total_by_benchmark_family.get("project", 0) > 0:
        _push_family("project", 1.5)
    if not priority_families:
        _push_family("integration", 3.0)
        _push_family("repository", 2.0)
        _push_family("project", 1.5)

    return {
        "comparison_task_limit_floor": comparison_task_limit_floor,
        "priority_benchmark_families": priority_families,
        "priority_benchmark_family_weights": priority_weights,
        "prefer_family_discrimination_probe": low_signal_preview,
        "bounded_comparison_required": True,
    }


def _bootstrap_guidance(metrics: EvalMetrics) -> bool:
    if metrics.total <= 0:
        return False
    total = max(1, metrics.total)
    low_confidence_rate = metrics.low_confidence_episodes / total
    pass_rate = _pass_rate(metrics)
    trusted_retrieval_rate = min(1.0, metrics.trusted_retrieval_steps / total)
    retrieval_influenced_rate = min(1.0, metrics.retrieval_influenced_steps / total)
    successful_without_effective_guidance_rate = _successful_without_effective_guidance_rate(metrics)
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    if carryover_repair_rate >= 0.2:
        return False
    if trusted_retrieval_rate < 0.1 and (
        (low_confidence_rate < 0.2 and successful_without_effective_guidance_rate >= 0.25)
        or (pass_rate >= 0.8 and successful_without_effective_guidance_rate >= 0.6)
    ):
        return True
    return (
        pass_rate >= 0.8
        and successful_without_effective_guidance_rate >= 0.5
        and trusted_retrieval_rate <= 0.25
        and retrieval_influenced_rate <= 0.25
    )


def _confidence_pressure(metrics: EvalMetrics) -> float:
    if metrics.total == 0:
        return 0.0
    low_confidence_rate = metrics.low_confidence_episodes / max(1, metrics.total)
    trusted_retrieval_rate = min(1.0, metrics.trusted_retrieval_steps / max(1, metrics.total))
    stalled_without_trust_rate = _stalled_without_trust_rate(metrics)
    carryover_repair_rate = _trusted_carryover_repair_rate(metrics)
    trust_gap = max(0.0, min(1.0, (0.3 - trusted_retrieval_rate) / 0.3))
    pressure = min(
        0.45,
        max(
            low_confidence_rate,
            low_confidence_rate + stalled_without_trust_rate * 0.6 + trust_gap * 0.12,
        ),
    )
    if carryover_repair_rate <= 0.0:
        return pressure
    return max(low_confidence_rate * 0.5, round(pressure - carryover_repair_rate * 0.12, 4))


def _stalled_without_trust_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    stalled = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        trusted_steps = int(outcome.get("trusted_retrieval_steps", 0) or 0)
        signals = {str(signal).strip() for signal in list(outcome.get("failure_signals", [])) if str(signal).strip()}
        termination_reason = str(outcome.get("termination_reason", "")).strip()
        if trusted_steps > 0:
            continue
        if (
            "no_state_progress" in signals
            or "state_regression" in signals
            or termination_reason in {"failed", "max_steps_reached"}
        ):
            stalled += 1
    return stalled / max(1, metrics.total)


def _successful_without_guidance_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    successful_without_guidance = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        termination_reason = str(outcome.get("termination_reason", "")).strip()
        if termination_reason != "success":
            continue
        trusted_steps = int(outcome.get("trusted_retrieval_steps", 0) or 0)
        influenced_steps = int(outcome.get("retrieval_influenced_steps", 0) or 0)
        retrieval_selected_steps = int(outcome.get("retrieval_selected_steps", 0) or 0)
        proposal_selected_steps = int(outcome.get("proposal_selected_steps", 0) or 0)
        if trusted_steps > 0 or influenced_steps > 0 or retrieval_selected_steps > 0 or proposal_selected_steps > 0:
            continue
        successful_without_guidance += 1
    return successful_without_guidance / max(1, metrics.total)


def _successful_without_effective_guidance_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    successful_without_effective_guidance = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        termination_reason = str(outcome.get("termination_reason", "")).strip()
        if termination_reason != "success":
            continue
        trusted_steps = int(outcome.get("trusted_retrieval_steps", 0) or 0)
        influenced_steps = int(outcome.get("retrieval_influenced_steps", 0) or 0)
        if trusted_steps > 0 or influenced_steps > 0:
            continue
        successful_without_effective_guidance += 1
    return successful_without_effective_guidance / max(1, metrics.total)


def _selected_without_influence_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    selected_without_influence = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        termination_reason = str(outcome.get("termination_reason", "")).strip()
        if termination_reason != "success":
            continue
        trusted_steps = int(outcome.get("trusted_retrieval_steps", 0) or 0)
        influenced_steps = int(outcome.get("retrieval_influenced_steps", 0) or 0)
        if trusted_steps > 0 or influenced_steps > 0:
            continue
        retrieval_selected_steps = int(outcome.get("retrieval_selected_steps", 0) or 0)
        proposal_selected_steps = int(outcome.get("proposal_selected_steps", 0) or 0)
        if retrieval_selected_steps <= 0 and proposal_selected_steps <= 0:
            continue
        selected_without_influence += 1
    return selected_without_influence / max(1, metrics.total)


def _trusted_carryover_repair_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0 or not isinstance(metrics.task_outcomes, dict):
        return 0.0
    eligible_successes = 0
    converted_repairs = 0
    for outcome in metrics.task_outcomes.values():
        if not isinstance(outcome, dict):
            continue
        if str(outcome.get("termination_reason", "")).strip() != "success":
            continue
        if str(outcome.get("difficulty", "")).strip() != "long_horizon":
            continue
        eligible_successes += 1
        if int(outcome.get("trusted_retrieval_carryover_verified_steps", 0) or 0) <= 0:
            continue
        converted_repairs += 1
    return converted_repairs / max(1, eligible_successes)


def _pass_rate(metrics: EvalMetrics) -> float:
    if metrics.total == 0:
        return 0.0
    return metrics.passed / max(1, metrics.total)
