from __future__ import annotations

from copy import deepcopy

from evals.metrics import EvalMetrics
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    normalized_generation_focus,
    retained_mapping_section,
    retention_gate_preset,
)


def world_model_behavior_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "expected_artifact_score_weight": 3,
        "preserved_artifact_score_weight": 2,
        "forbidden_artifact_penalty": 5,
        "forbidden_cleanup_score_weight": 4,
        "workflow_branch_target_score_weight": 4,
        "workflow_changed_path_score_weight": 3,
        "workflow_generated_path_score_weight": 3,
        "workflow_report_path_score_weight": 2,
        "workflow_preserved_path_score_weight": 2,
        "required_tests_score_weight": 2,
        "required_merges_score_weight": 4,
        "long_horizon_scaffold_bonus": 1,
        "retrieved_expected_artifact_score_weight": 3,
        "retrieved_forbidden_artifact_penalty": 4,
        "retrieved_preserved_artifact_score_weight": 2,
        "retrieved_workflow_changed_path_score_weight": 2,
        "retrieved_workflow_report_path_score_weight": 1,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "workflow_alignment":
        controls["expected_artifact_score_weight"] = 4
        controls["workflow_changed_path_score_weight"] = 4
        controls["workflow_report_path_score_weight"] = 3
        controls["retrieved_expected_artifact_score_weight"] = 4
        controls["retrieved_workflow_changed_path_score_weight"] = 3
    if failure_counts.get("no_state_progress", 0) > 0:
        controls["expected_artifact_score_weight"] = max(int(controls["expected_artifact_score_weight"]), 4)
        controls["workflow_changed_path_score_weight"] = max(int(controls["workflow_changed_path_score_weight"]), 4)
        controls["forbidden_cleanup_score_weight"] = max(int(controls["forbidden_cleanup_score_weight"]), 5)
    if failure_counts.get("command_failure", 0) > 0 or focus == "conflict_avoidance":
        controls["forbidden_artifact_penalty"] = 6
        controls["workflow_preserved_path_score_weight"] = 3
        controls["retrieved_forbidden_artifact_penalty"] = 5
        controls["retrieved_preserved_artifact_score_weight"] = 3
    if failure_counts.get("state_regression", 0) > 0:
        controls["preserved_artifact_score_weight"] = max(int(controls["preserved_artifact_score_weight"]), 4)
        controls["workflow_preserved_path_score_weight"] = max(int(controls["workflow_preserved_path_score_weight"]), 4)
        controls["forbidden_artifact_penalty"] = max(int(controls["forbidden_artifact_penalty"]), 6)
    if focus == "preservation_bias":
        controls["preserved_artifact_score_weight"] = 4
        controls["workflow_preserved_path_score_weight"] = 4
        controls["retrieved_preserved_artifact_score_weight"] = 4
    if metrics.low_confidence_episodes > 0:
        controls["workflow_branch_target_score_weight"] = max(
            int(controls["workflow_branch_target_score_weight"]),
            5,
        )
    return controls


def world_model_planning_controls(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "include_preserved_artifact_steps": True,
        "prefer_preserved_artifacts_first": False,
        "append_preservation_subgoal": False,
        "max_preserved_artifacts": 3,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "workflow_alignment":
        controls["append_preservation_subgoal"] = True
    if failure_counts.get("no_state_progress", 0) > 0:
        controls["append_preservation_subgoal"] = True
    if failure_counts.get("command_failure", 0) > 0 or focus == "conflict_avoidance":
        controls["max_preserved_artifacts"] = 4
    if failure_counts.get("state_regression", 0) > 0:
        controls["prefer_preserved_artifacts_first"] = True
        controls["append_preservation_subgoal"] = True
        controls["max_preserved_artifacts"] = max(int(controls["max_preserved_artifacts"]), 4)
    if focus == "preservation_bias":
        controls["prefer_preserved_artifacts_first"] = True
        controls["append_preservation_subgoal"] = True
        controls["max_preserved_artifacts"] = 5
    if metrics.low_confidence_episodes > 0 and focus != "preservation_bias":
        controls["max_preserved_artifacts"] = max(int(controls["max_preserved_artifacts"]), 4)
    return controls


def build_world_model_proposal_artifact(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    return build_standard_proposal_artifact(
        artifact_kind="world_model_policy_set",
        generation_focus=generation_focus,
        control_schema="world_model_behavior_controls_v1",
        retention_gate=retention_gate_preset("world_model", min_pass_rate_delta_abs=0.01),
        controls=world_model_behavior_controls(
            metrics,
            failure_counts,
            focus=None if generation_focus == "balanced" else generation_focus,
            baseline=retained_world_model_controls(current_payload),
        ),
        proposals=_proposals(
            metrics,
            failure_counts,
            focus=None if generation_focus == "balanced" else generation_focus,
        ),
        extra_sections={
            "planning_controls": world_model_planning_controls(
                metrics,
                failure_counts,
                focus=None if generation_focus == "balanced" else generation_focus,
                baseline=retained_world_model_planning_controls(current_payload),
            ),
        },
    )


def retained_world_model_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="world_model_policy_set", section="controls")


def retained_world_model_planning_controls(payload: object) -> dict[str, object]:
    return retained_mapping_section(payload, artifact_kind="world_model_policy_set", section="planning_controls")


def _proposals(
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    if failure_counts.get("missing_expected_file", 0) > 0 or focus == "workflow_alignment":
        proposals.append(
            {
                "area": "workflow_alignment",
                "priority": 5,
                "reason": "expected workflow paths remain underweighted relative to command drift",
                "suggestion": "Bias command scoring toward verifier-declared changed paths, reports, and branch targets.",
            }
        )
    if failure_counts.get("no_state_progress", 0) > 0:
        proposals.append(
            {
                "area": "workflow_alignment",
                "priority": 5,
                "reason": "episodes are stalling without measurable state progress",
                "suggestion": "Raise weighting for commands that clear missing expected artifacts, cleanup forbidden paths, and advance workflow paths measurably.",
            }
        )
    if failure_counts.get("command_failure", 0) > 0 or focus == "conflict_avoidance":
        proposals.append(
            {
                "area": "conflict_avoidance",
                "priority": 5,
                "reason": "command failures suggest the world model under-penalizes preserved or forbidden path collisions",
                "suggestion": "Increase penalties for forbidden artifacts and raise preserved-path protection in scoring and planning.",
            }
        )
    if failure_counts.get("state_regression", 0) > 0:
        proposals.append(
            {
                "area": "preservation_bias",
                "priority": 6,
                "reason": "recorded state regressions show preserved or previously-safe paths are being destabilized",
                "suggestion": "Increase preserved-path protection, prefer preservation subgoals earlier, and penalize commands that are likely to regress intermediate state.",
            }
        )
    if focus == "preservation_bias":
        proposals.append(
            {
                "area": "preservation_bias",
                "priority": 6,
                "reason": "selected variant targets verifier-preserved artifacts",
                "suggestion": "Front-load preserved-path obligations in the initial plan and prefer commands that avoid collateral edits.",
            }
        )
    if metrics.low_confidence_episodes > 0:
        proposals.append(
            {
                "area": "branch_targeting",
                "priority": 4,
                "reason": "low-confidence episodes indicate weak routing toward workflow structure",
                "suggestion": "Increase branch-target and workflow-path weighting so the model commits to clearer structure earlier.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "workflow_alignment",
            "priority": 3,
            "reason": "world-model controls should remain explicit even when repo-workflow failures are currently sparse",
            "suggestion": "Keep workflow path, preserved artifact, and report-path scoring explicit so later retained mutations have a stable baseline surface.",
        },
    )
