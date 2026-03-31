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
        retention_gate=retention_gate_preset("retrieval"),
        proposals=proposals,
        extra_sections={
            "asset_strategy": _asset_strategy(generation_focus),
            "overrides": merged_overrides,
            "asset_controls": asset_controls,
            "asset_rebuild_plan": _asset_rebuild_plan(generation_focus, asset_controls),
        },
    )


def retained_retrieval_overrides(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="retrieval_policy_set", section="overrides")


def retained_retrieval_asset_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="retrieval_policy_set", section="asset_controls")


def _proposals(metrics: EvalMetrics, *, focus: str) -> list[dict[str, object]]:
    low_confidence_rate = 0.0 if metrics.total == 0 else metrics.low_confidence_episodes / max(1, metrics.total)
    trusted_retrieval_rate = 0.0 if metrics.total == 0 else metrics.trusted_retrieval_steps / max(1, metrics.total)
    proposals: list[dict[str, object]] = []

    if focus in {"balanced", "confidence"}:
        proposals.append(
            {
                "proposal_id": "retrieval:confidence_gating",
                "area": "confidence",
                "reason": "low-confidence retrieval remains common relative to trusted retrieval usage",
                "overrides": {
                    "tolbert_confidence_threshold": round(max(0.15, min(0.55, 0.2 + low_confidence_rate * 0.5)), 2),
                    "tolbert_skill_ranking_min_confidence": round(max(0.4, min(0.7, 0.35 + low_confidence_rate * 0.4)), 2),
                    "tolbert_deterministic_command_confidence": 0.8,
                    "tolbert_first_step_direct_command_confidence": 0.95,
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
    low_confidence_rate = 0.0 if metrics.total == 0 else metrics.low_confidence_episodes / max(1, metrics.total)
    controls: dict[str, object] = {
        "include_failure_catalog_spans": True,
        "include_negative_episode_spans": True,
        "include_tool_candidate_spans": True,
        "include_episode_step_spans": True,
        "include_skill_procedure_spans": True,
        "max_episode_step_spans_per_task": 3,
        "max_failure_spans_per_task": 2,
        "prefer_failure_alignment": low_confidence_rate >= 0.2,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if focus == "confidence":
        controls["prefer_failure_alignment"] = True
        controls["max_failure_spans_per_task"] = 3
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
    return {
        "rebuild_required": True,
        "strategy": _asset_strategy(focus),
        "source_priorities": [
            "native_repo_structure",
            "successful_trace_procedures",
            "failure_catalog",
            "negative_episode_spans",
            "tool_candidates",
        ],
        "controls": deepcopy(controls),
    }
