from __future__ import annotations

from copy import deepcopy
import re

from evals.metrics import EvalMetrics
from .improvement_common import (
    retained_artifact_payload,
    build_standard_proposal_artifact,
    ensure_proposals,
    filter_proposals_by_area,
    normalized_generation_focus,
    retention_gate_preset,
)


def _default_search_guardrails() -> dict[str, dict[str, float]]:
    return {
        "campaign": {
            "relative_score_floor": 0.75,
            "absolute_score_margin": 0.04,
            "close_score_relative_threshold": 0.9,
            "close_score_margin_threshold": 0.01,
            "history_relative_threshold": 0.8,
        },
        "variant": {
            "close_score_relative_threshold": 0.92,
            "close_score_margin_threshold": 0.003,
            "history_relative_threshold": 0.85,
        },
    }


def _default_priority_family_allocation_confidence() -> dict[str, float | int]:
    return {
        "minimum_runs": 3,
        "target_priority_tasks": 12,
        "target_family_tasks": 0,
        "history_window_runs": 3,
        "history_weight": 0.5,
        "bonus_history_weight": 0.75,
        "normalization_history_weight": 0.25,
    }


def _normalize_search_guardrails(controls: dict[str, object]) -> dict[str, dict[str, float]]:
    defaults = _default_search_guardrails()
    guardrails = controls.get("search_guardrails", {})
    normalized = deepcopy(defaults)
    if isinstance(guardrails, dict):
        for scope, scope_defaults in defaults.items():
            incoming = guardrails.get(scope, {})
            if not isinstance(incoming, dict):
                continue
            for field in scope_defaults:
                if field in incoming:
                    normalized[scope][field] = incoming[field]
    legacy_fields = {
        "campaign_relative_score_floor": ("campaign", "relative_score_floor"),
        "campaign_absolute_score_margin": ("campaign", "absolute_score_margin"),
        "campaign_close_score_relative_threshold": ("campaign", "close_score_relative_threshold"),
        "campaign_close_score_margin_threshold": ("campaign", "close_score_margin_threshold"),
        "campaign_history_relative_threshold": ("campaign", "history_relative_threshold"),
        "variant_close_score_relative_threshold": ("variant", "close_score_relative_threshold"),
        "variant_close_score_margin_threshold": ("variant", "close_score_margin_threshold"),
        "variant_history_relative_threshold": ("variant", "history_relative_threshold"),
    }
    for legacy_field, (scope, field) in legacy_fields.items():
        if field not in normalized[scope] or normalized[scope][field] == defaults[scope][field]:
            if legacy_field in controls:
                normalized[scope][field] = controls[legacy_field]
        controls.pop(legacy_field, None)
    controls["search_guardrails"] = normalized
    return normalized


def _normalize_priority_family_allocation_confidence(controls: dict[str, object]) -> dict[str, float | int]:
    defaults = _default_priority_family_allocation_confidence()
    normalized = deepcopy(defaults)
    incoming = controls.get("priority_family_allocation_confidence", {})
    if isinstance(incoming, dict):
        for field in defaults:
            if field in incoming:
                normalized[field] = incoming[field]
    normalized["minimum_runs"] = max(1, int(normalized.get("minimum_runs", defaults["minimum_runs"]) or 0))
    normalized["target_priority_tasks"] = max(
        1,
        int(normalized.get("target_priority_tasks", defaults["target_priority_tasks"]) or 0),
    )
    normalized["target_family_tasks"] = max(
        0,
        int(normalized.get("target_family_tasks", defaults["target_family_tasks"]) or 0),
    )
    normalized["history_window_runs"] = max(
        1,
        int(normalized.get("history_window_runs", defaults["history_window_runs"]) or 0),
    )
    normalized["history_weight"] = max(0.0, min(1.0, float(normalized.get("history_weight", defaults["history_weight"]) or 0.0)))
    normalized["bonus_history_weight"] = max(
        0.0,
        min(1.0, float(normalized.get("bonus_history_weight", defaults["bonus_history_weight"]) or 0.0)),
    )
    normalized["normalization_history_weight"] = max(
        0.0,
        min(1.0, float(normalized.get("normalization_history_weight", defaults["normalization_history_weight"]) or 0.0)),
    )
    controls["priority_family_allocation_confidence"] = normalized
    return normalized


def resolve_improvement_planner_controls(controls: dict[str, object] | None = None) -> dict[str, object]:
    resolved: dict[str, object] = {
        "subsystem_expected_gain_multiplier": {},
        "subsystem_cost_multiplier": {},
        "subsystem_score_bias": {},
        "bootstrap_penalty_multiplier": {},
        "variant_expected_gain_multiplier": {},
        "variant_cost_multiplier": {},
        "variant_score_bias": {},
        "priority_family_retained_gain_multiplier": {},
        "priority_family_cost_multiplier": {},
        "priority_family_score_bias": {},
        "priority_family_exploration_bonus": 0.05,
        "priority_family_allocation_confidence": _default_priority_family_allocation_confidence(),
        "variant_expansions": {},
        "search_guardrails": _default_search_guardrails(),
        "portfolio_exploration_bonus": 0.01,
        "portfolio_selection_saturation_penalty_per_cycle": 0.01,
        "portfolio_recent_retention_bonus_multiplier": 1.0,
        "portfolio_recent_rejection_penalty_multiplier": 1.0,
    }
    if isinstance(controls, dict):
        resolved.update(deepcopy(controls))
    _normalize_search_guardrails(resolved)
    _normalize_priority_family_allocation_confidence(resolved)
    return resolved


def policy_behavior_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "direct_command_confidence_boost": 0.0,
        "skill_ranking_confidence_boost": 0.0,
        "verifier_alignment_bias": 0,
        "planner_subgoal_command_bias": 0,
        "critic_repeat_failure_bias": 0,
        "required_artifact_first_step_bias": 0,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if metrics.low_confidence_episodes > 0:
        controls["direct_command_confidence_boost"] = 0.1
        controls["skill_ranking_confidence_boost"] = 0.05
    if focus == "retrieval_caution":
        controls["direct_command_confidence_boost"] = max(float(controls["direct_command_confidence_boost"]), 0.2)
        controls["skill_ranking_confidence_boost"] = max(float(controls["skill_ranking_confidence_boost"]), 0.1)
        controls["verifier_alignment_bias"] = max(int(controls["verifier_alignment_bias"]), 1)
        controls["planner_subgoal_command_bias"] = max(int(controls["planner_subgoal_command_bias"]), 1)
        controls["critic_repeat_failure_bias"] = max(int(controls["critic_repeat_failure_bias"]), 2)
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "verifier_alignment":
        controls["verifier_alignment_bias"] = 2
        controls["required_artifact_first_step_bias"] = 2
        controls["planner_subgoal_command_bias"] = 2
    if focus == "verifier_alignment":
        controls["critic_repeat_failure_bias"] = max(int(controls["critic_repeat_failure_bias"]), 1)
    if focus == "long_horizon_success":
        controls["direct_command_confidence_boost"] = max(float(controls["direct_command_confidence_boost"]), 0.05)
        controls["skill_ranking_confidence_boost"] = max(float(controls["skill_ranking_confidence_boost"]), 0.05)
        controls["verifier_alignment_bias"] = max(int(controls["verifier_alignment_bias"]), 1)
        controls["planner_subgoal_command_bias"] = max(int(controls["planner_subgoal_command_bias"]), 2)
        controls["critic_repeat_failure_bias"] = max(int(controls["critic_repeat_failure_bias"]), 2)
        controls["required_artifact_first_step_bias"] = max(int(controls["required_artifact_first_step_bias"]), 2)
    return controls


def planner_mutation_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "max_initial_subgoals": 5,
        "prepend_verifier_contract_check": False,
        "append_validation_subgoal": True,
        "prefer_expected_artifacts_first": True,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if metrics.low_confidence_episodes > 0 or focus == "retrieval_caution":
        controls["max_initial_subgoals"] = 4
    if focus == "retrieval_caution":
        controls["prepend_verifier_contract_check"] = True
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "verifier_alignment":
        controls["prepend_verifier_contract_check"] = True
    if focus == "verifier_alignment":
        controls["max_initial_subgoals"] = 6
    if focus == "long_horizon_success":
        try:
            max_initial_subgoals = int(controls.get("max_initial_subgoals", 5))
        except (TypeError, ValueError):
            max_initial_subgoals = 5
        controls["max_initial_subgoals"] = max(max_initial_subgoals, 6)
        controls["prepend_verifier_contract_check"] = True
        controls["append_validation_subgoal"] = True
        controls["append_preservation_subgoal"] = True
        controls["prefer_expected_artifacts_first"] = True
        controls["include_preserved_artifact_steps"] = True
        controls["prefer_preserved_artifacts_first"] = True
    return controls


def improvement_planner_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls = resolve_improvement_planner_controls(baseline)
    search_guardrails = _normalize_search_guardrails(controls)
    allocation_confidence = _normalize_priority_family_allocation_confidence(controls)
    gain_multiplier = controls["subsystem_expected_gain_multiplier"]
    cost_multiplier = controls["subsystem_cost_multiplier"]
    score_bias = controls["subsystem_score_bias"]
    bootstrap_multiplier = controls["bootstrap_penalty_multiplier"]
    variant_gain_multiplier = controls["variant_expected_gain_multiplier"]
    variant_cost_multiplier = controls["variant_cost_multiplier"]
    variant_score_bias = controls["variant_score_bias"]
    variant_expansions = controls["variant_expansions"]
    assert isinstance(gain_multiplier, dict)
    assert isinstance(cost_multiplier, dict)
    assert isinstance(score_bias, dict)
    assert isinstance(bootstrap_multiplier, dict)
    assert isinstance(variant_gain_multiplier, dict)
    assert isinstance(variant_cost_multiplier, dict)
    assert isinstance(variant_score_bias, dict)
    assert isinstance(variant_expansions, dict)
    assert isinstance(search_guardrails, dict)
    assert isinstance(allocation_confidence, dict)

    if metrics.low_confidence_episodes > 0:
        gain_multiplier["retrieval"] = 1.2
        score_bias["retrieval"] = 0.006
        allocation_confidence["minimum_runs"] = max(int(allocation_confidence.get("minimum_runs", 0) or 0), 4)
        allocation_confidence["target_priority_tasks"] = max(
            int(allocation_confidence.get("target_priority_tasks", 0) or 0),
            16,
        )
        allocation_confidence["target_family_tasks"] = max(
            int(allocation_confidence.get("target_family_tasks", 0) or 0),
            3,
        )
        allocation_confidence["history_window_runs"] = max(
            int(allocation_confidence.get("history_window_runs", 0) or 0),
            4,
        )
        allocation_confidence["history_weight"] = max(float(allocation_confidence.get("history_weight", 0.0) or 0.0), 0.6)
        allocation_confidence["bonus_history_weight"] = max(
            float(allocation_confidence.get("bonus_history_weight", 0.0) or 0.0),
            0.8,
        )
        variant_expansions["retrieval"] = [
            {
                "variant_id": "routing_depth",
                "description": "widen branch routing depth under persistent uncertainty",
                "expected_gain": 0.024,
                "estimated_cost": 3,
                "controls": {"focus": "routing"},
            }
        ]
        variant_score_bias["retrieval"] = {"routing_depth": 0.002}
    if failure_counts.get("missing_expected_file", 0) > 0:
        gain_multiplier["policy"] = 1.25
        score_bias["policy"] = 0.008
        bootstrap_multiplier["verifier"] = 0.75
        variant_score_bias["policy"] = {"verifier_alignment": 0.002}
    if failure_counts.get("command_failure", 0) > 0:
        gain_multiplier["tooling"] = 1.15
        score_bias["tooling"] = 0.004
        allocation_confidence["target_family_tasks"] = max(
            int(allocation_confidence.get("target_family_tasks", 0) or 0),
            3,
        )
        allocation_confidence["normalization_history_weight"] = min(
            float(allocation_confidence.get("normalization_history_weight", 1.0) or 0.0),
            0.2,
        )
        variant_gain_multiplier["tooling"] = {"script_hardening": 1.1}
    if focus == "retrieval_caution":
        gain_multiplier["retrieval"] = max(float(gain_multiplier.get("retrieval", 1.0)), 1.35)
        cost_multiplier["retrieval"] = 0.85
        score_bias["retrieval"] = max(float(score_bias.get("retrieval", 0.0)), 0.01)
        allocation_confidence["minimum_runs"] = max(int(allocation_confidence.get("minimum_runs", 0) or 0), 5)
        allocation_confidence["target_priority_tasks"] = max(
            int(allocation_confidence.get("target_priority_tasks", 0) or 0),
            20,
        )
        allocation_confidence["target_family_tasks"] = max(
            int(allocation_confidence.get("target_family_tasks", 0) or 0),
            4,
        )
        allocation_confidence["history_window_runs"] = max(
            int(allocation_confidence.get("history_window_runs", 0) or 0),
            4,
        )
        allocation_confidence["history_weight"] = max(float(allocation_confidence.get("history_weight", 0.0) or 0.0), 0.65)
        allocation_confidence["bonus_history_weight"] = max(
            float(allocation_confidence.get("bonus_history_weight", 0.0) or 0.0),
            0.9,
        )
        allocation_confidence["normalization_history_weight"] = min(
            float(allocation_confidence.get("normalization_history_weight", 1.0) or 0.0),
            0.15,
        )
        variant_expansions["retrieval"] = [
            {
                "variant_id": "routing_depth",
                "description": "widen branch routing depth under persistent uncertainty",
                "expected_gain": 0.024,
                "estimated_cost": 3,
                "controls": {"focus": "routing"},
            },
            {
                "variant_id": "direct_command_safety",
                "description": "tighten direct-command safety under low-confidence retrieval",
                "expected_gain": 0.022,
                "estimated_cost": 2,
                "controls": {"focus": "safety"},
            },
        ]
        variant_score_bias["retrieval"] = {
            "routing_depth": max(
                0.002,
                float(dict(variant_score_bias.get("retrieval", {})).get("routing_depth", 0.0)),
            ),
            "direct_command_safety": 0.003,
        }
        search_guardrails["campaign"]["close_score_relative_threshold"] = 0.84
        search_guardrails["campaign"]["history_relative_threshold"] = 0.75
        controls["portfolio_recent_rejection_penalty_multiplier"] = 1.15
    if focus == "verifier_alignment":
        gain_multiplier["policy"] = max(float(gain_multiplier.get("policy", 1.0)), 1.35)
        gain_multiplier["verifier"] = 1.1
        score_bias["policy"] = max(float(score_bias.get("policy", 0.0)), 0.012)
        bootstrap_multiplier["verifier"] = min(float(bootstrap_multiplier.get("verifier", 1.0)), 0.5)
        variant_gain_multiplier["policy"] = {"verifier_alignment": 1.15}
        search_guardrails["campaign"]["relative_score_floor"] = 0.7
        controls["portfolio_exploration_bonus"] = 0.012
    if focus == "long_horizon_success":
        gain_multiplier["policy"] = max(float(gain_multiplier.get("policy", 1.0)), 1.25)
        gain_multiplier["verifier"] = max(float(gain_multiplier.get("verifier", 1.0)), 1.1)
        score_bias["policy"] = max(float(score_bias.get("policy", 0.0)), 0.009)
        bootstrap_multiplier["verifier"] = min(float(bootstrap_multiplier.get("verifier", 1.0)), 0.75)
        variant_gain_multiplier["policy"] = {
            **dict(variant_gain_multiplier.get("policy", {})),
            "long_horizon_orientation": 1.12,
        }
        allocation_confidence["history_window_runs"] = max(
            int(allocation_confidence.get("history_window_runs", 0) or 0),
            5,
        )
        allocation_confidence["history_weight"] = max(float(allocation_confidence.get("history_weight", 0.0) or 0.0), 0.65)
        allocation_confidence["bonus_history_weight"] = max(
            float(allocation_confidence.get("bonus_history_weight", 0.0) or 0.0),
            0.85,
        )
        search_guardrails["campaign"]["history_relative_threshold"] = min(
            float(search_guardrails["campaign"].get("history_relative_threshold", 0.8)),
            0.78,
        )
        controls["portfolio_recent_retention_bonus_multiplier"] = max(
            float(controls.get("portfolio_recent_retention_bonus_multiplier", 1.0) or 0.0),
            1.1,
        )
        controls["portfolio_recent_rejection_penalty_multiplier"] = max(
            float(controls.get("portfolio_recent_rejection_penalty_multiplier", 1.0) or 0.0),
            1.05,
        )
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        allocation_confidence["target_priority_tasks"] = max(
            int(allocation_confidence.get("target_priority_tasks", 0) or 0),
            18,
        )
        allocation_confidence["target_family_tasks"] = max(
            int(allocation_confidence.get("target_family_tasks", 0) or 0),
            4,
        )
        allocation_confidence["history_window_runs"] = max(
            int(allocation_confidence.get("history_window_runs", 0) or 0),
            4,
        )
        allocation_confidence["history_weight"] = max(float(allocation_confidence.get("history_weight", 0.0) or 0.0), 0.6)
    weakest_family = _weakest_generated_family(metrics)
    if weakest_family:
        variant_expansions["curriculum"] = [
            {
                "variant_id": f"family_pressure_{weakest_family}",
                "description": f"raise generated pressure on weakest family {weakest_family}",
                "expected_gain": 0.02,
                "estimated_cost": 2,
                "controls": {"focus": "benchmark_family", "family": weakest_family},
            }
        ]
        variant_score_bias["curriculum"] = {f"family_pressure_{weakest_family}": 0.001}
    return controls


def role_directive_overrides(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, str] | None = None,
) -> dict[str, str]:
    directives: dict[str, str] = dict(baseline) if isinstance(baseline, dict) else {}
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "verifier_alignment":
        directives["planner"] = (
            "Bias the next subgoal toward expected artifacts and explicit verifier-visible validation before termination."
        )
    if focus == "retrieval_caution":
        directives["planner"] = (
            "Bias the next subgoal toward verifier-visible probes, expected artifacts, and short repo-grounding checks before larger edits when retrieval remains weak."
        )
    if failure_counts.get("command_failure", 0) > 0 or focus == "retrieval_caution":
        directives["critic"] = (
            "Escalate repeated command-shape failures quickly and require a verifier-facing reason before approving execution."
        )
    if metrics.low_confidence_episodes > 0 or focus == "retrieval_caution":
        directives["executor"] = (
            "Prefer shorter verifier-relevant commands, starting with a read, search, or verifier-visible probe when retrieval remains weak, and avoid committing to a brittle path too early."
        )
    if focus == "long_horizon_success":
        directives["planner"] = (
            "Sequence multi-file repo work toward verifier-visible milestones that preserve working artifacts, maintain rollback room, ground on existing repo state, and avoid spending the whole step budget on a brittle rewrite."
        )
        directives["critic"] = (
            "Reject commands that create avoidable regressions, skip targeted validation, ignore concrete failure evidence, or trade immediate completion for a weaker future repo state."
        )
        directives["executor"] = (
            "Prefer the smallest reversible edit or verifier-relevant local check that advances expected artifacts, keeps preserved files stable, reads the relevant file or failure before a rewrite, and earns the right to stop through validation on the touched paths."
        )
    return directives


def _normalized_prompt_proposal_text(value: object) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return " ".join(tokens)


def _prompt_proposal_topic(proposal: dict[str, object]) -> str:
    reason = str(proposal.get("reason", "")).strip()
    suggestion = str(proposal.get("suggestion", "")).strip()
    text = f"{reason} {suggestion}".lower()
    if "retrieval confidence" in text or "low-confidence retrieval" in text or (
        "retrieval" in text and "confidence" in text
    ):
        return "retrieval_confidence_caution"
    if (
        ("missing expected file" in text or "expected artifact" in text or "expected artifacts" in text)
        or ("expected" in text and ("file" in text or "artifact" in text))
    ) and any(token in text for token in ("confirm", "created", "terminate", "terminating")):
        return "expected_artifact_confirmation"
    if "command failure" in text or "failing command" in text:
        return "command_failure_avoidance"
    if "generated-task performance trails" in text or "one-shot synthesis" in text:
        return "generated_lane_adaptation"
    if "long-horizon" in text or ("repository" in text and "validation-backed" in text):
        return "long_horizon_success"
    return _normalized_prompt_proposal_text(suggestion or reason)


def dedupe_prompt_adjustments(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    winners: dict[tuple[str, str], dict[str, object]] = {}
    winner_order: dict[tuple[str, str], int] = {}
    for index, raw_proposal in enumerate(proposals):
        if not isinstance(raw_proposal, dict):
            continue
        proposal = deepcopy(raw_proposal)
        area = str(proposal.get("area", "")).strip()
        key = (area, _prompt_proposal_topic(proposal))
        current = winners.get(key)
        if current is None:
            winners[key] = proposal
            winner_order[key] = index
            continue
        current_priority = int(current.get("priority", 0) or 0)
        proposal_priority = int(proposal.get("priority", 0) or 0)
        current_signal = len(str(current.get("suggestion", "")).strip()) + len(str(current.get("reason", "")).strip())
        proposal_signal = len(str(proposal.get("suggestion", "")).strip()) + len(str(proposal.get("reason", "")).strip())
        if (proposal_priority, proposal_signal) > (current_priority, current_signal):
            winners[key] = proposal
            winner_order[key] = index
    return [deepcopy(winners[key]) for key in sorted(winners, key=lambda item: winner_order[item])]


def propose_prompt_adjustments(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    if focus == "retrieval_caution":
        proposals.append(
            {
                "area": "decision",
                "priority": 6,
                "reason": "selected variant targets low-confidence retrieval caution",
                "suggestion": "When retrieval confidence is weak, prefer additional search, verifier checks, or a verifier-visible repo-grounding command before committing to a command path.",
            }
        )
    if focus == "verifier_alignment":
        proposals.append(
            {
                "area": "system",
                "priority": 6,
                "reason": "selected variant targets verifier alignment",
                "suggestion": "Bias action selection toward commands that directly satisfy expected files and verifier-visible artifacts.",
            }
        )
    if focus == "long_horizon_success":
        proposals.extend(
            [
                {
                    "area": "system",
                    "priority": 6,
                    "reason": "selected variant targets durable long-horizon coding progress",
                    "suggestion": "Favor small reversible edits, preservation of working artifacts, repo-state grounding, and validation-backed stopping on repository, project, tooling, and integration tasks over optimistic large rewrites.",
                },
                {
                    "area": "decision",
                    "priority": 6,
                    "reason": "selected variant targets durable long-horizon coding progress",
                    "suggestion": "Before executing or terminating, check expected artifacts, preserved paths, concrete failure evidence, and whether the next step should be a targeted edit or local verification step on the touched files.",
                },
                {
                    "area": "reflection",
                    "priority": 5,
                    "reason": "selected variant targets durable long-horizon coding progress",
                    "suggestion": "Treat single easy-task wins as weak evidence and use repo-scale regressions, preserved-artifact damage, and repeated repository failure modes as stronger signals for the next change.",
                },
            ]
        )
    if metrics.low_confidence_episodes > 0:
        proposals.append(
            {
                "area": "decision",
                "priority": 5,
                "reason": "low-confidence retrieval remains common",
                "suggestion": "Tell the model to widen search and avoid overcommitting when retrieval confidence is low.",
            }
        )
    if failure_counts.get("command_failure", 0) > 0:
        proposals.append(
            {
                "area": "system",
                "priority": 4,
                "reason": "command failures are still present in memory",
                "suggestion": "Emphasize verifier-compatible file targets and avoidance of previously failing command shapes.",
            }
        )
    if failure_counts.get("missing_expected_file", 0) > 0:
        proposals.append(
            {
                "area": "decision",
                "priority": 4,
                "reason": "missing expected files recur in memory",
                "suggestion": "Require explicit confirmation that expected artifacts were created before terminating.",
            }
        )
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        proposals.append(
            {
                "area": "reflection",
                "priority": 3,
                "reason": "generated-task performance trails base task performance",
                "suggestion": "Bias reflection prompts toward adaptation from prior traces instead of one-shot synthesis.",
            }
        )
    if focus == "retrieval_caution":
        return dedupe_prompt_adjustments(
            filter_proposals_by_area(proposals, allowed_areas={"decision", "reflection"})
        )
    if focus == "verifier_alignment":
        return dedupe_prompt_adjustments(
            filter_proposals_by_area(proposals, allowed_areas={"system", "decision"})
        )
    if focus == "long_horizon_success":
        return dedupe_prompt_adjustments(
            filter_proposals_by_area(proposals, allowed_areas={"system", "decision", "reflection"})
        )
    return dedupe_prompt_adjustments(
        ensure_proposals(
            proposals,
            fallback={
                "area": "system",
                "priority": 3,
                "reason": "policy behavior should remain explicit as a retained runtime surface",
                "suggestion": "Preserve bounded prompt and planner controls even when recent failures are sparse.",
            },
        )
    )


def build_prompt_proposal_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    return build_standard_proposal_artifact(
        artifact_kind="prompt_proposal_set",
        generation_focus=generation_focus,
        control_schema="policy_behavior_controls_v3",
        retention_gate=retention_gate_preset("policy"),
        controls=policy_behavior_controls(
            metrics,
            failure_counts,
            focus=focus,
            baseline=retained_policy_controls(current_payload),
        ),
        proposals=propose_prompt_adjustments(metrics, failure_counts, focus=focus),
        extra_sections={
            "planner_controls": planner_mutation_controls(
                metrics,
                failure_counts,
                focus=focus,
                baseline=retained_planner_controls(current_payload),
            ),
            "improvement_planner_controls": improvement_planner_controls(
                metrics,
                failure_counts,
                focus=focus,
                baseline=retained_improvement_planner_controls(current_payload),
            ),
            "role_directives": role_directive_overrides(
                metrics,
                failure_counts,
                focus=focus,
                baseline=retained_role_directives(current_payload),
            ),
        },
    )


def retained_policy_controls(payload: object) -> dict[str, object]:
    retained = _retained_prompt_payload(payload)
    if retained is None:
        return {}
    controls = retained.get("controls", {})
    return dict(controls) if isinstance(controls, dict) else {}


def retained_planner_controls(payload: object) -> dict[str, object]:
    retained = _retained_prompt_payload(payload)
    if retained is None:
        return {}
    controls = retained.get("planner_controls", {})
    return dict(controls) if isinstance(controls, dict) else {}


def retained_improvement_planner_controls(payload: object) -> dict[str, object]:
    retained = _retained_prompt_payload(payload)
    if retained is None:
        return {}
    controls = retained.get("improvement_planner_controls", {})
    return resolve_improvement_planner_controls(dict(controls) if isinstance(controls, dict) else {})


def retained_role_directives(payload: object) -> dict[str, str]:
    retained = _retained_prompt_payload(payload)
    if retained is None:
        return {}
    directives = retained.get("role_directives", {})
    if not isinstance(directives, dict):
        return {}
    return {
        str(role).strip().lower(): str(text).strip()
        for role, text in directives.items()
        if str(role).strip() and str(text).strip()
    }


def _retained_prompt_payload(payload: object) -> dict[str, object] | None:
    return retained_artifact_payload(payload, artifact_kind="prompt_proposal_set")


def _weakest_generated_family(metrics: EvalMetrics) -> str:
    if not metrics.generated_by_benchmark_family:
        return ""
    return min(
        metrics.generated_by_benchmark_family,
        key=lambda family: 0.0
        if metrics.generated_by_benchmark_family[family] == 0
        else metrics.generated_passed_by_benchmark_family.get(family, 0)
        / metrics.generated_by_benchmark_family[family],
    )
