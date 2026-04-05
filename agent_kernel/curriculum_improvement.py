from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from evals.metrics import EvalMetrics
from .improvement_common import (
    build_standard_proposal_artifact,
    filter_proposals_by_area,
    merged_string_lists,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)

_CODING_FRONTIER_FAMILIES = (
    "repo_sandbox",
    "repository",
    "integration",
    "tooling",
    "project",
    "workflow",
)


def _family_pass_rate(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(passed) / float(total)


def _merge_ranked_tokens(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*primary, *secondary]:
        token = str(value).strip().lower()
        if token and token not in merged:
            merged.append(token)
    return merged


def _task_signal_metadata(payload: dict[str, object]) -> dict[str, object]:
    metadata = payload.get("task_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _task_signal_family(payload: dict[str, object]) -> str:
    metadata = _task_signal_metadata(payload)
    family = str(payload.get("benchmark_family", "")).strip().lower()
    if family:
        return family
    return str(metadata.get("benchmark_family", "")).strip().lower()


def _task_signal_success(payload: dict[str, object]) -> bool:
    if "success" in payload:
        return bool(payload.get("success", False))
    termination_reason = str(payload.get("termination_reason", "")).strip().lower()
    return termination_reason == "success"


def _task_signal_failure_motifs(payload: dict[str, object]) -> list[str]:
    summary = payload.get("summary", {})
    summary = dict(summary) if isinstance(summary, dict) else {}
    motifs: list[str] = []
    for values in (
        summary.get("failure_types", []),
        summary.get("transition_failures", []),
        payload.get("failure_types", []),
        payload.get("transition_failures", []),
        payload.get("failure_signals", []),
    ):
        if not isinstance(values, list):
            continue
        for value in values:
            token = str(value).strip().lower()
            if token and token not in motifs:
                motifs.append(token)
    return motifs


def _task_signal_repo_setting_signatures(payload: dict[str, object]) -> list[str]:
    metadata = _task_signal_metadata(payload)
    family = _task_signal_family(payload)
    difficulty = str(
        payload.get("difficulty", metadata.get("difficulty", metadata.get("task_difficulty", "")))
    ).strip().lower()
    lineage_families = payload.get("lineage_families", metadata.get("lineage_families", []))
    lineage_surfaces = payload.get("lineage_surfaces", metadata.get("lineage_surfaces", []))
    lineage_branch_kinds = payload.get("lineage_branch_kinds", metadata.get("lineage_branch_kinds", []))
    long_horizon_surface = str(
        payload.get("long_horizon_coding_surface", metadata.get("long_horizon_coding_surface", ""))
    ).strip().lower()
    surfaces = [
        str(value).strip().lower()
        for value in [long_horizon_surface, *(lineage_surfaces if isinstance(lineage_surfaces, list) else [])]
        if str(value).strip()
    ]
    branch_kinds = [
        str(value).strip().lower()
        for value in (lineage_branch_kinds if isinstance(lineage_branch_kinds, list) else [])
        if str(value).strip()
    ]
    families = [
        str(value).strip().lower()
        for value in (lineage_families if isinstance(lineage_families, list) else [])
        if str(value).strip()
    ]
    signatures: list[str] = []
    if difficulty == "long_horizon":
        signatures.append("long_horizon")
    if family == "repo_sandbox" or "repo_sandbox" in families:
        signatures.append("repo_sandbox")
    if any("shared_repo" in surface for surface in surfaces):
        signatures.append("shared_repo")
    if any("worker" in surface for surface in surfaces):
        signatures.append("worker_handoff")
    if any("integrator" in surface for surface in surfaces):
        signatures.append("integrator_handoff")
    if any("validation" in surface for surface in surfaces):
        signatures.append("validation_lane")
    if "cleanup" in branch_kinds or any("cleanup" in surface for surface in surfaces):
        signatures.append("cleanup_lane")
    if "audit" in branch_kinds or any("audit" in surface for surface in surfaces):
        signatures.append("audit_lane")
    return signatures


def _coding_frontier_signal_records(metrics: EvalMetrics) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for source in (metrics.task_outcomes, metrics.task_trajectories):
        if not isinstance(source, dict):
            continue
        for task_id, raw_payload in source.items():
            if not isinstance(raw_payload, dict):
                continue
            key = str(raw_payload.get("task_id", task_id)).strip() or str(task_id).strip()
            if not key:
                continue
            current = merged.setdefault(key, {"task_id": key})
            current.update(raw_payload)
            if isinstance(current.get("task_metadata"), dict) and isinstance(raw_payload.get("task_metadata"), dict):
                current["task_metadata"] = {
                    **dict(current.get("task_metadata", {})),
                    **dict(raw_payload.get("task_metadata", {})),
                }
    return list(merged.values())


def _coding_frontier_failure_motif_priority_pairs(metrics: EvalMetrics) -> list[str]:
    pair_scores: dict[tuple[str, str], float] = {}
    for payload in _coding_frontier_signal_records(metrics):
        family = _task_signal_family(payload)
        if family not in _CODING_FRONTIER_FAMILIES or _task_signal_success(payload):
            continue
        motifs = _task_signal_failure_motifs(payload)
        if not motifs:
            continue
        difficulty = str(
            payload.get("difficulty", _task_signal_metadata(payload).get("difficulty", ""))
        ).strip().lower()
        weight = 1.0
        if difficulty == "long_horizon":
            weight += 0.5
        if _task_signal_repo_setting_signatures(payload):
            weight += 0.25
        for motif in motifs:
            pair_scores[(family, motif)] = pair_scores.get((family, motif), 0.0) + weight
    ranked = sorted(pair_scores.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [f"{family}:{motif}" for (family, motif), score in ranked if score >= 1.0]


def _coding_frontier_repo_setting_priority_pairs(metrics: EvalMetrics) -> list[str]:
    pair_scores: dict[tuple[str, str], float] = {}
    for payload in _coding_frontier_signal_records(metrics):
        family = _task_signal_family(payload)
        if family not in _CODING_FRONTIER_FAMILIES or _task_signal_success(payload):
            continue
        signatures = _task_signal_repo_setting_signatures(payload)
        if not signatures:
            continue
        motifs = _task_signal_failure_motifs(payload)
        weight = 1.0 + min(0.5, 0.15 * float(len(motifs)))
        for signature in signatures:
            pair_scores[(family, signature)] = pair_scores.get((family, signature), 0.0) + weight
    ranked = sorted(pair_scores.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [f"{family}:{signature}" for (family, signature), score in ranked if score >= 1.0]


def _coding_frontier_generalization_priority_families(metrics: EvalMetrics) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for frontier_family in _CODING_FRONTIER_FAMILIES:
        observed_total = int(metrics.total_by_benchmark_family.get(frontier_family, 0) or 0)
        observed_passed = int(metrics.passed_by_benchmark_family.get(frontier_family, 0) or 0)
        if observed_total <= 0:
            continue
        observed_pass_rate = _family_pass_rate(observed_passed, observed_total)
        generated_total = int(metrics.generated_by_benchmark_family.get(frontier_family, 0) or 0)
        generated_passed = int(metrics.generated_passed_by_benchmark_family.get(frontier_family, 0) or 0)
        generated_pass_rate = _family_pass_rate(generated_passed, generated_total)
        if generated_total <= 0:
            gap = observed_pass_rate + 0.2
        else:
            gap = observed_pass_rate - generated_pass_rate
        coverage_bonus = 1.0 / float(generated_total + 1)
        score = gap + (coverage_bonus * 0.25)
        if score <= 0.15:
            continue
        ranked.append((-score, frontier_family))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [family for _, family in ranked]


def _coding_frontier_priority_state(
    metrics: EvalMetrics,
    *,
    family: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    missing_families = [
        frontier_family
        for frontier_family in _CODING_FRONTIER_FAMILIES
        if metrics.total_by_benchmark_family.get(frontier_family, 0) <= 0
        and metrics.generated_by_benchmark_family.get(frontier_family, 0) <= 0
    ]
    weakness_rank: list[tuple[float, str]] = []
    retention_priority_families: list[str] = []
    for frontier_family in _CODING_FRONTIER_FAMILIES:
        generated_total = int(metrics.generated_by_benchmark_family.get(frontier_family, 0) or 0)
        generated_passed = int(metrics.generated_passed_by_benchmark_family.get(frontier_family, 0) or 0)
        observed_total = int(metrics.total_by_benchmark_family.get(frontier_family, 0) or 0)
        observed_passed = int(metrics.passed_by_benchmark_family.get(frontier_family, 0) or 0)
        total = generated_total if generated_total > 0 else observed_total
        passed = generated_passed if generated_total > 0 else observed_passed
        if total <= 0:
            continue
        pass_rate = _family_pass_rate(passed, total)
        coverage_penalty = 1.0 / float(total + 1)
        weakness_rank.append((-(1.0 - pass_rate + coverage_penalty), frontier_family))
        if generated_total > 0 and pass_rate < max(0.25, metrics.pass_rate):
            retention_priority_families.append(frontier_family)
    weakness_rank.sort(key=lambda item: (item[0], item[1]))
    priority_families = [
        *missing_families,
        *[
            frontier_family
            for _, frontier_family in weakness_rank
            if frontier_family not in missing_families
        ],
    ]
    if family and family in _CODING_FRONTIER_FAMILIES:
        priority_families = [family, *[value for value in priority_families if value != family]]
        if family not in retention_priority_families:
            retention_priority_families = [family, *retention_priority_families]
    return priority_families, missing_families, retention_priority_families


def _recent_curriculum_retention_feedback(cycles_path: Path | None) -> dict[str, object]:
    if cycles_path is None or not cycles_path.exists():
        return {
            "retained_family_delta": {},
            "promotion_risk_family_delta": {},
            "retained_gain_families": [],
            "promotion_risk_families": [],
        }
    try:
        raw_lines = cycles_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raw_lines = []
    decision_records: list[dict[str, object]] = []
    for raw_line in raw_lines:
        text = raw_line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        state = str(payload.get("state", "")).strip()
        if state in {"retain", "reject"}:
            decision_records.append(payload)
    retained_totals: dict[str, float] = {}
    retained_counts: dict[str, int] = {}
    risk_totals: dict[str, float] = {}
    for payload in decision_records[-24:]:
        state = str(payload.get("state", "")).strip()
        metrics_summary = payload.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            metrics_summary = {}
        for field, weight in (
            ("family_pass_rate_delta", 1.0),
            ("generated_family_pass_rate_delta", 0.75),
        ):
            family_deltas = metrics_summary.get(field, {})
            if not isinstance(family_deltas, dict):
                continue
            for family, raw_delta in family_deltas.items():
                family_name = str(family).strip().lower()
                if not family_name:
                    continue
                try:
                    weighted_delta = float(raw_delta) * weight
                except (TypeError, ValueError):
                    continue
                if state == "retain" and weighted_delta > 0.0:
                    retained_totals[family_name] = retained_totals.get(family_name, 0.0) + weighted_delta
                    retained_counts[family_name] = retained_counts.get(family_name, 0) + 1
                elif state == "reject" or weighted_delta < 0.0:
                    risk_totals[family_name] = risk_totals.get(family_name, 0.0) + abs(weighted_delta)
        if state == "reject":
            family_deltas = metrics_summary.get("family_pass_rate_delta", {})
            if isinstance(family_deltas, dict) and int(metrics_summary.get("regressed_family_count", 0) or 0) > 0:
                for family in family_deltas:
                    family_name = str(family).strip().lower()
                    if family_name:
                        risk_totals[family_name] = risk_totals.get(family_name, 0.0) + 0.02
    retained_family_delta = {
        family: round(total / max(1, retained_counts.get(family, 1)), 4)
        for family, total in retained_totals.items()
        if total > 0.0
    }
    promotion_risk_family_delta = {
        family: round(total, 4)
        for family, total in risk_totals.items()
        if total > 0.0
    }
    retained_gain_families = [
        family
        for family, _ in sorted(retained_family_delta.items(), key=lambda item: (-float(item[1]), item[0]))
    ]
    promotion_risk_families = [
        family
        for family, _ in sorted(promotion_risk_family_delta.items(), key=lambda item: (-float(item[1]), item[0]))
    ]
    return {
        "retained_family_delta": retained_family_delta,
        "promotion_risk_family_delta": promotion_risk_family_delta,
        "retained_gain_families": retained_gain_families,
        "promotion_risk_families": promotion_risk_families,
    }


def curriculum_behavior_controls(
    metrics: EvalMetrics,
    *,
    focus: str | None = None,
    family: str | None = None,
    baseline: dict[str, object] | None = None,
    cycles_path: Path | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "success_reference_limit": 3,
        "failure_reference_family_only": False,
        "failure_recovery_anchor_min_matches": 1,
        "failure_recovery_command_cap": 4,
        "adjacent_reference_limit": 3,
        "max_generated_adjacent_tasks": 4,
        "max_generated_failure_recovery_tasks": 4,
        "frontier_priority_family_bonus": 2,
        "frontier_missing_family_bonus": 4,
        "frontier_retention_priority_bonus": 2,
        "frontier_generalization_bonus": 3,
        "frontier_outward_branch_bonus": 2,
        "frontier_lineage_breadth_bonus": 1,
        "frontier_failure_motif_bonus": 2,
        "frontier_repo_setting_bonus": 3,
        "frontier_harder_task_bonus": 2,
        "frontier_min_lineage_depth": 3,
        "frontier_retained_gain_bonus": 2,
        "frontier_promotion_risk_penalty": 2,
        "frontier_priority_families": list(_CODING_FRONTIER_FAMILIES),
        "frontier_missing_families": [],
        "frontier_retention_priority_families": [],
        "frontier_generalization_priority_families": [],
        "frontier_retained_gain_families": [],
        "frontier_promotion_risk_families": [],
        "frontier_failure_motif_priority_pairs": [],
        "frontier_repo_setting_priority_pairs": [],
        "frontier_retained_family_delta": {},
        "frontier_promotion_risk_family_delta": {},
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    priority_families, missing_families, retention_priority_families = _coding_frontier_priority_state(
        metrics,
        family=family if focus == "benchmark_family" else None,
    )
    generalization_priority_families = _coding_frontier_generalization_priority_families(metrics)
    failure_motif_priority_pairs = _coding_frontier_failure_motif_priority_pairs(metrics)
    repo_setting_priority_pairs = _coding_frontier_repo_setting_priority_pairs(metrics)
    retention_feedback = _recent_curriculum_retention_feedback(cycles_path)
    controls["frontier_priority_families"] = priority_families
    controls["frontier_missing_families"] = missing_families
    controls["frontier_retention_priority_families"] = retention_priority_families
    controls["frontier_generalization_priority_families"] = generalization_priority_families
    controls["frontier_retained_gain_families"] = list(retention_feedback["retained_gain_families"])
    controls["frontier_promotion_risk_families"] = list(retention_feedback["promotion_risk_families"])
    controls["frontier_retained_family_delta"] = dict(retention_feedback["retained_family_delta"])
    controls["frontier_promotion_risk_family_delta"] = dict(retention_feedback["promotion_risk_family_delta"])
    controls["frontier_failure_motif_priority_pairs"] = _merge_ranked_tokens(
        failure_motif_priority_pairs,
        merged_string_lists(controls.get("frontier_failure_motif_priority_pairs", []), [], lowercase=True),
    )
    controls["frontier_repo_setting_priority_pairs"] = _merge_ranked_tokens(
        repo_setting_priority_pairs,
        merged_string_lists(controls.get("frontier_repo_setting_priority_pairs", []), [], lowercase=True),
    )
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        controls["failure_reference_family_only"] = True
    if metrics.generated_passed_by_kind.get("failure_recovery", 0) < metrics.generated_by_kind.get("failure_recovery", 0):
        controls["failure_recovery_anchor_min_matches"] = 2
        controls["failure_recovery_command_cap"] = 3
    if missing_families:
        controls["max_generated_adjacent_tasks"] = max(int(controls["max_generated_adjacent_tasks"]), 5)
    if focus == "failure_recovery":
        controls["failure_reference_family_only"] = True
        controls["failure_recovery_anchor_min_matches"] = max(int(controls["failure_recovery_anchor_min_matches"]), 2)
        controls["failure_recovery_command_cap"] = min(int(controls["failure_recovery_command_cap"]), 3)
        controls["max_generated_adjacent_tasks"] = min(int(controls["max_generated_adjacent_tasks"]), 2)
        controls["max_generated_failure_recovery_tasks"] = max(int(controls["max_generated_failure_recovery_tasks"]), 6)
    if focus == "benchmark_family" and family:
        controls["preferred_benchmark_family"] = family
        controls["frontier_priority_families"] = [family, *[value for value in priority_families if value != family]]
        controls["adjacent_reference_limit"] = 2
        controls["max_generated_adjacent_tasks"] = min(int(controls["max_generated_adjacent_tasks"]), 3)
        controls["max_generated_failure_recovery_tasks"] = min(int(controls["max_generated_failure_recovery_tasks"]), 3)
    return controls


def build_curriculum_proposal_artifact(
    metrics: EvalMetrics,
    *,
    focus: str | None = None,
    family: str | None = None,
    current_payload: object | None = None,
    cycles_path: Path | None = None,
) -> dict[str, object]:
    proposals: list[dict[str, object]] = []
    if focus == "failure_recovery":
        proposals.append(
            {
                "area": "failure_recovery",
                "priority": 6,
                "reason": "selected variant targets failure-recovery specificity",
                "suggestion": "Constrain generated recovery tasks to one localized repair that preserves the original verifier contract.",
            }
        )
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        proposals.append(
            {
                "area": "failure_recovery",
                "priority": 5,
                "reason": "generated-task pass rate trails the base pass rate",
                "suggestion": "Increase recovery-task specificity around the weakest generated benchmark families.",
            }
        )
    if metrics.generated_passed_by_kind.get("failure_recovery", 0) < metrics.generated_by_kind.get("failure_recovery", 0):
        proposals.append(
            {
                "area": "failure_recovery",
                "priority": 5,
                "reason": "failure-recovery generation is weaker than adjacent-success generation",
                "suggestion": "Bias synthesis toward verifier-preserving recovery variants and explicit avoided-command histories.",
            }
        )
    priority_families, missing_families, retention_priority_families = _coding_frontier_priority_state(
        metrics,
        family=family if focus == "benchmark_family" else None,
    )
    if missing_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 6,
                "reason": f"curriculum has no live coding evidence yet for {', '.join(missing_families[:3])}",
                "suggestion": "Generate bounded repository-scale tasks that open those missing coding families without dropping verifier-preserving constraints.",
            }
        )
    if priority_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": f"coding frontier should ratchet toward {priority_families[0]} before repeating saturated families",
                "suggestion": "Spend adjacent-success budget on under-covered coding families first, then deepen the strongest retained lineages by one harder step.",
            }
        )
    if retention_priority_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": f"generated coding tasks remain weak on {', '.join(retention_priority_families[:2])}",
                "suggestion": "Prefer harder followups that stay inside the weakest generated families until the gains survive promotion and retained-baseline comparison.",
            }
        )
    retention_feedback = _recent_curriculum_retention_feedback(cycles_path)
    retained_gain_families = list(retention_feedback["retained_gain_families"])
    promotion_risk_families = list(retention_feedback["promotion_risk_families"])
    if retained_gain_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 4,
                "reason": f"recent retained-baseline wins compound most in {', '.join(retained_gain_families[:2])}",
                "suggestion": "Spend harder-task budget on those retained winner families so the open-ended curriculum compounds on coding surfaces that already survive promotion.",
            }
        )
    if promotion_risk_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": f"recent promotion or retained-baseline regressions cluster around {', '.join(promotion_risk_families[:2])}",
                "suggestion": "Throttle frontier escalation on those families until family-level retained-baseline deltas recover, then reopen the harder adjacent ladder.",
            }
        )
    generalization_priority_families = _coding_frontier_generalization_priority_families(metrics)
    if generalization_priority_families:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": (
                    "generated curriculum is not yet generalizing retained coding evidence into "
                    f"{', '.join(generalization_priority_families[:2])}"
                ),
                "suggestion": (
                    "Use self-generated followups to branch successful lineages outward into adjacent weak families, "
                    "so retained wins prove broader repo and workflow transfer instead of only repeating one lane."
                ),
            }
        )
    failure_motif_priority_pairs = _coding_frontier_failure_motif_priority_pairs(metrics)
    if failure_motif_priority_pairs:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": (
                    "retained coding failures keep clustering around "
                    f"{', '.join(failure_motif_priority_pairs[:2])}"
                ),
                "suggestion": (
                    "Generate harder followups that aim directly at those family-level bottlenecks, "
                    "so the next pressure comes from repeated verifier-visible failure motifs instead of generic family gaps."
                ),
            }
        )
    repo_setting_priority_pairs = _coding_frontier_repo_setting_priority_pairs(metrics)
    if repo_setting_priority_pairs:
        proposals.append(
            {
                "area": "coding_frontier",
                "priority": 5,
                "reason": (
                    "retained weakness is concentrated in repo settings such as "
                    f"{', '.join(repo_setting_priority_pairs[:2])}"
                ),
                "suggestion": (
                    "Spend outward long-horizon branches on those worker, integrator, shared-repo, or validation settings "
                    "so self-generated curriculum expands into the concrete repo configurations that are still failing."
                ),
            }
        )
    if metrics.generated_by_benchmark_family:
        weakest_family = family or min(
            metrics.generated_by_benchmark_family,
            key=lambda family: 0.0
            if metrics.generated_by_benchmark_family[family] == 0
            else metrics.generated_passed_by_benchmark_family.get(family, 0)
            / metrics.generated_by_benchmark_family[family],
        )
        proposals.append(
            {
                "area": "benchmark_family",
                "priority": 4,
                "reason": f"generated family {weakest_family} is the weakest current curriculum slice",
                "suggestion": f"Add harder adjacent and recovery tasks centered on {weakest_family}.",
            }
        )
    if focus == "benchmark_family" and family:
        proposals = filter_proposals_by_area(proposals, allowed_areas={"benchmark_family", "failure_recovery"})
    elif focus == "failure_recovery":
        proposals = filter_proposals_by_area(proposals, allowed_areas={"failure_recovery"})

    return build_standard_proposal_artifact(
        artifact_kind="curriculum_proposal_set",
        generation_focus=normalized_generation_focus(focus),
        control_schema="curriculum_behavior_controls_v3",
        retention_gate=retention_gate_preset("curriculum"),
        controls=curriculum_behavior_controls(
            metrics,
            focus=focus,
            family=family,
            baseline=retained_curriculum_controls(current_payload),
            cycles_path=cycles_path,
        ),
        proposals=proposals,
    )


def retained_curriculum_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="curriculum_proposal_set", section="controls")
