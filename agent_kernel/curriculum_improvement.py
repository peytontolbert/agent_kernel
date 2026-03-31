from __future__ import annotations

from copy import deepcopy

from evals.metrics import EvalMetrics
from .improvement_common import (
    build_standard_proposal_artifact,
    filter_proposals_by_area,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)


def curriculum_behavior_controls(
    metrics: EvalMetrics,
    *,
    focus: str | None = None,
    family: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "success_reference_limit": 3,
        "failure_reference_family_only": False,
        "failure_recovery_anchor_min_matches": 1,
        "failure_recovery_command_cap": 4,
        "adjacent_reference_limit": 3,
        "max_generated_adjacent_tasks": 4,
        "max_generated_failure_recovery_tasks": 4,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        controls["failure_reference_family_only"] = True
    if metrics.generated_passed_by_kind.get("failure_recovery", 0) < metrics.generated_by_kind.get("failure_recovery", 0):
        controls["failure_recovery_anchor_min_matches"] = 2
        controls["failure_recovery_command_cap"] = 3
    if focus == "failure_recovery":
        controls["failure_reference_family_only"] = True
        controls["failure_recovery_anchor_min_matches"] = max(int(controls["failure_recovery_anchor_min_matches"]), 2)
        controls["failure_recovery_command_cap"] = min(int(controls["failure_recovery_command_cap"]), 3)
        controls["max_generated_adjacent_tasks"] = min(int(controls["max_generated_adjacent_tasks"]), 2)
        controls["max_generated_failure_recovery_tasks"] = max(int(controls["max_generated_failure_recovery_tasks"]), 6)
    if focus == "benchmark_family" and family:
        controls["preferred_benchmark_family"] = family
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
        control_schema="curriculum_behavior_controls_v2",
        retention_gate=retention_gate_preset("curriculum"),
        controls=curriculum_behavior_controls(
            metrics,
            focus=focus,
            family=family,
            baseline=retained_curriculum_controls(current_payload),
        ),
        proposals=proposals,
    )


def retained_curriculum_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="curriculum_proposal_set", section="controls")
