from __future__ import annotations

from pathlib import Path

from evals.metrics import EvalMetrics

from . import artifact_support_evidence as artifact_support_evidence_ext
from . import control_evidence
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER
from . import planner_runtime_state as planner_runtime_state_ext
from . import transition_model_improvement as transition_model_improvement_ext


def _generated_task_summary(
    metrics: EvalMetrics,
    *,
    curriculum_kind: str | None = None,
) -> dict[str, object]:
    summaries = getattr(metrics, "generated_task_summaries", {})
    normalized_kind = "" if curriculum_kind is None else str(curriculum_kind).strip()
    if not isinstance(summaries, dict) or not summaries:
        if normalized_kind:
            task_count = int(metrics.generated_by_kind.get(normalized_kind, 0) or 0)
            success_count = int(metrics.generated_passed_by_kind.get(normalized_kind, 0) or 0)
        else:
            task_count = int(metrics.generated_total or 0)
            success_count = int(metrics.generated_passed or 0)
        return {
            "task_count": task_count,
            "success_count": success_count,
            "clean_success_count": 0,
            "pass_rate": round(0.0 if task_count <= 0 else success_count / task_count, 4),
            "clean_success_rate": 0.0,
            "average_steps": 0.0,
            "clean_success_benchmark_families": [],
            "distinct_clean_success_benchmark_families": 0,
        }
    task_count = 0
    success_count = 0
    clean_success_count = 0
    total_steps = 0
    clean_success_families: set[str] = set()
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        summary_kind = str(summary.get("curriculum_kind", "")).strip()
        if normalized_kind and summary_kind != normalized_kind:
            continue
        task_count += 1
        success = bool(summary.get("success", False))
        clean_success = bool(summary.get("clean_success", False))
        success_count += int(success)
        clean_success_count += int(clean_success)
        try:
            total_steps += max(0, int(summary.get("steps", 0) or 0))
        except (TypeError, ValueError):
            pass
        if clean_success:
            family = str(summary.get("benchmark_family", "bounded")).strip() or "bounded"
            clean_success_families.add(family)
    return {
        "task_count": task_count,
        "success_count": success_count,
        "clean_success_count": clean_success_count,
        "pass_rate": round(0.0 if task_count <= 0 else success_count / task_count, 4),
        "clean_success_rate": round(0.0 if task_count <= 0 else clean_success_count / task_count, 4),
        "average_steps": round(0.0 if task_count <= 0 else total_steps / task_count, 4),
        "clean_success_benchmark_families": sorted(clean_success_families),
        "distinct_clean_success_benchmark_families": len(clean_success_families),
    }


def _clean_success_family_gain_count(
    baseline_summary: dict[str, object],
    candidate_summary: dict[str, object],
) -> int:
    baseline_families = {
        str(value).strip()
        for value in baseline_summary.get("clean_success_benchmark_families", [])
        if str(value).strip()
    }
    candidate_families = {
        str(value).strip()
        for value in candidate_summary.get("clean_success_benchmark_families", [])
        if str(value).strip()
    }
    return len(candidate_families - baseline_families)


def _contract_clean_failure_recovery_family_gain_count(
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> int:
    baseline_summary = getattr(baseline_metrics, "contract_clean_failure_recovery_by_origin_benchmark_family", {})
    candidate_summary = getattr(candidate_metrics, "contract_clean_failure_recovery_by_origin_benchmark_family", {})
    if not isinstance(baseline_summary, dict):
        baseline_summary = {}
    if not isinstance(candidate_summary, dict):
        candidate_summary = {}
    gained = 0
    for family in sorted(set(baseline_summary) | set(candidate_summary)):
        baseline_row = baseline_summary.get(family, {})
        candidate_row = candidate_summary.get(family, {})
        if not isinstance(baseline_row, dict):
            baseline_row = {}
        if not isinstance(candidate_row, dict):
            candidate_row = {}
        baseline_clean_successes = int(baseline_row.get("clean_success_count", 0) or 0)
        candidate_clean_successes = int(candidate_row.get("clean_success_count", 0) or 0)
        if candidate_clean_successes > baseline_clean_successes:
            gained += 1
    return gained


def retention_evidence(
    subsystem: str,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
    *,
    payload: dict[str, object] | None = None,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    from ... import improvement as core

    subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem, capability_modules_path)
    core._ensure_retention_plugin_registry()
    evidence: dict[str, object] = {
        "pass_rate_delta": candidate_metrics.pass_rate - baseline_metrics.pass_rate,
        "average_step_delta": candidate_metrics.average_steps - baseline_metrics.average_steps,
        "generated_pass_rate_delta": candidate_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
        "proposal_selected_steps_delta": (
            candidate_metrics.proposal_selected_steps - baseline_metrics.proposal_selected_steps
        ),
        "novel_command_steps_delta": (
            candidate_metrics.novel_command_steps - baseline_metrics.novel_command_steps
        ),
        "novel_valid_command_rate_delta": (
            candidate_metrics.novel_valid_command_rate - baseline_metrics.novel_valid_command_rate
        ),
        "tolbert_primary_episodes_delta": (
            candidate_metrics.tolbert_primary_episodes - baseline_metrics.tolbert_primary_episodes
        ),
        "tolbert_shadow_episodes_delta": (
            candidate_metrics.tolbert_shadow_episodes - baseline_metrics.tolbert_shadow_episodes
        ),
    }
    evidence = core.engine_attach_learning_evidence(
        evidence=evidence,
        subsystem=subsystem,
        learning_artifacts_path=core._default_runtime_learning_artifacts_path(),
        load_learning_candidates_fn=core.load_learning_candidates,
        runtime_config=None,
        capability_modules_path=capability_modules_path,
        base_subsystem_fn=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem,
        learning_evidence_adapter=core.DEFAULT_LEARNING_EVIDENCE_ADAPTER,
    )
    baseline_generated_summary = _generated_task_summary(baseline_metrics)
    candidate_generated_summary = _generated_task_summary(candidate_metrics)
    evidence["generated_task_count_delta"] = (
        int(candidate_generated_summary.get("task_count", 0) or 0)
        - int(baseline_generated_summary.get("task_count", 0) or 0)
    )
    evidence["generated_clean_success_delta"] = (
        int(candidate_generated_summary.get("clean_success_count", 0) or 0)
        - int(baseline_generated_summary.get("clean_success_count", 0) or 0)
    )
    evidence["generated_average_steps_delta"] = round(
        float(candidate_generated_summary.get("average_steps", 0.0) or 0.0)
        - float(baseline_generated_summary.get("average_steps", 0.0) or 0.0),
        4,
    )
    evidence["generated_clean_success_family_gain_count"] = _clean_success_family_gain_count(
        baseline_generated_summary,
        candidate_generated_summary,
    )
    baseline_adjacent_success_summary = _generated_task_summary(
        baseline_metrics,
        curriculum_kind="adjacent_success",
    )
    candidate_adjacent_success_summary = _generated_task_summary(
        candidate_metrics,
        curriculum_kind="adjacent_success",
    )
    evidence["adjacent_success_clean_success_delta"] = (
        int(candidate_adjacent_success_summary.get("clean_success_count", 0) or 0)
        - int(baseline_adjacent_success_summary.get("clean_success_count", 0) or 0)
    )
    evidence["adjacent_success_average_steps_delta"] = round(
        float(candidate_adjacent_success_summary.get("average_steps", 0.0) or 0.0)
        - float(baseline_adjacent_success_summary.get("average_steps", 0.0) or 0.0),
        4,
    )
    evidence["adjacent_success_clean_success_family_gain_count"] = _clean_success_family_gain_count(
        baseline_adjacent_success_summary,
        candidate_adjacent_success_summary,
    )
    family_pass_rate_delta = core._family_pass_rate_delta_map(baseline_metrics, candidate_metrics)
    if family_pass_rate_delta:
        evidence["family_pass_rate_delta"] = family_pass_rate_delta
        evidence["regressed_family_count"] = core._family_regression_count(baseline_metrics, candidate_metrics)
        evidence["worst_family_delta"] = core._family_worst_delta(baseline_metrics, candidate_metrics)
    difficulty_pass_rate_delta = core._difficulty_pass_rate_delta_map(baseline_metrics, candidate_metrics)
    if difficulty_pass_rate_delta:
        evidence["difficulty_pass_rate_delta"] = difficulty_pass_rate_delta
    generated_family_pass_rate_delta = core._generated_family_pass_rate_delta_map(baseline_metrics, candidate_metrics)
    if generated_family_pass_rate_delta:
        evidence["generated_family_pass_rate_delta"] = generated_family_pass_rate_delta
        evidence["generated_regressed_family_count"] = core._generated_family_regression_count(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["generated_worst_family_delta"] = core._generated_family_worst_delta(
            baseline_metrics,
            candidate_metrics,
        )
    if core._has_generated_kind(baseline_metrics, "failure_recovery") or core._has_generated_kind(
        candidate_metrics, "failure_recovery"
    ):
        evidence["failure_recovery_pass_rate_delta"] = core._generated_kind_pass_rate(
            candidate_metrics,
            "failure_recovery",
        ) - core._generated_kind_pass_rate(baseline_metrics, "failure_recovery")
        baseline_failure_recovery_summary = _generated_task_summary(
            baseline_metrics,
            curriculum_kind="failure_recovery",
        )
        candidate_failure_recovery_summary = _generated_task_summary(
            candidate_metrics,
            curriculum_kind="failure_recovery",
        )
        evidence["failure_recovery_clean_success_delta"] = (
            int(candidate_failure_recovery_summary.get("clean_success_count", 0) or 0)
            - int(baseline_failure_recovery_summary.get("clean_success_count", 0) or 0)
        )
        evidence["failure_recovery_average_steps_delta"] = round(
            float(candidate_failure_recovery_summary.get("average_steps", 0.0) or 0.0)
            - float(baseline_failure_recovery_summary.get("average_steps", 0.0) or 0.0),
            4,
        )
        evidence["failure_recovery_clean_success_family_gain_count"] = _clean_success_family_gain_count(
            baseline_failure_recovery_summary,
            candidate_failure_recovery_summary,
        )
    baseline_contract_clean_failure_recovery_summary = getattr(
        baseline_metrics,
        "contract_clean_failure_recovery_summary",
        {},
    )
    candidate_contract_clean_failure_recovery_summary = getattr(
        candidate_metrics,
        "contract_clean_failure_recovery_summary",
        {},
    )
    if not isinstance(baseline_contract_clean_failure_recovery_summary, dict):
        baseline_contract_clean_failure_recovery_summary = {}
    if not isinstance(candidate_contract_clean_failure_recovery_summary, dict):
        candidate_contract_clean_failure_recovery_summary = {}
    evidence["contract_clean_failure_recovery_clean_success_delta"] = (
        int(candidate_contract_clean_failure_recovery_summary.get("clean_success_count", 0) or 0)
        - int(baseline_contract_clean_failure_recovery_summary.get("clean_success_count", 0) or 0)
    )
    evidence["contract_clean_failure_recovery_average_steps_delta"] = round(
        float(candidate_contract_clean_failure_recovery_summary.get("average_steps", 0.0) or 0.0)
        - float(baseline_contract_clean_failure_recovery_summary.get("average_steps", 0.0) or 0.0),
        4,
    )
    evidence["contract_clean_failure_recovery_family_gain_count"] = (
        _contract_clean_failure_recovery_family_gain_count(
            baseline_metrics,
            candidate_metrics,
        )
    )
    validation_family_summary = core._benchmark_family_summary(
        baseline_metrics,
        candidate_metrics,
        family="validation",
    )
    if validation_family_summary:
        evidence["validation_family_summary"] = validation_family_summary
    if subsystem in {"skills", "tooling"}:
        retrieval_reuse_summary = artifact_support_evidence_ext.artifact_retrieval_reuse_comparison(
            payload,
            subsystem=subsystem,
            active_artifact_payload_from_generation_context_fn=planner_runtime_state_ext.active_artifact_payload_from_generation_context,
        )
        if retrieval_reuse_summary:
            evidence["retrieval_reuse_summary"] = retrieval_reuse_summary
    if subsystem == "retrieval":
        evidence["baseline_trusted_carryover_repair_rate"] = core._trusted_carryover_repair_rate(baseline_metrics)
        benchmark_candidate_total = candidate_metrics.total_by_benchmark_family.get("benchmark_candidate", 0)
        evidence["family_discrimination_gain"] = core._family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["false_failure_rate"] = (
            0.0
            if benchmark_candidate_total == 0
            else core._candidate_family_failure_rate(candidate_metrics, "benchmark_candidate")
        )
        evidence["trusted_retrieval_delta"] = (
            candidate_metrics.trusted_retrieval_steps - baseline_metrics.trusted_retrieval_steps
        )
        evidence["retrieval_influenced_steps_delta"] = (
            candidate_metrics.retrieval_influenced_steps - baseline_metrics.retrieval_influenced_steps
        )
        evidence["retrieval_selected_steps_delta"] = (
            candidate_metrics.retrieval_selected_steps - baseline_metrics.retrieval_selected_steps
        )
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
        evidence["trusted_carryover_repair_rate"] = core._trusted_carryover_repair_rate(candidate_metrics)
        evidence["trusted_carryover_repair_rate_delta"] = (
            core._trusted_carryover_repair_rate(candidate_metrics)
            - core._trusted_carryover_repair_rate(baseline_metrics)
        )
        evidence["baseline_trusted_carryover_verified_steps"] = core._trusted_carryover_verified_steps(baseline_metrics)
        evidence["trusted_carryover_verified_steps"] = core._trusted_carryover_verified_steps(candidate_metrics)
        evidence["trusted_carryover_verified_step_delta"] = (
            core._trusted_carryover_verified_steps(candidate_metrics)
            - core._trusted_carryover_verified_steps(baseline_metrics)
        )
    if subsystem == "tolbert_model":
        runtime_paths = payload.get("runtime_paths", {}) if isinstance(payload, dict) else {}
        checkpoint_path = (
            core.resolve_tolbert_runtime_checkpoint_path(runtime_paths) if isinstance(runtime_paths, dict) else ""
        )
        cache_paths = runtime_paths.get("cache_paths", []) if isinstance(runtime_paths, dict) else []
        evidence["family_discrimination_gain"] = core._family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["trusted_retrieval_delta"] = (
            candidate_metrics.trusted_retrieval_steps - baseline_metrics.trusted_retrieval_steps
        )
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
        proposal_metrics_by_family = core._proposal_metrics_delta_by_benchmark_family(
            baseline_metrics,
            candidate_metrics,
        )
        if proposal_metrics_by_family:
            evidence["proposal_metrics_by_benchmark_family"] = proposal_metrics_by_family
        proposal_metrics_by_difficulty = core._proposal_metrics_delta_by_difficulty(
            baseline_metrics,
            candidate_metrics,
        )
        if proposal_metrics_by_difficulty:
            evidence["proposal_metrics_by_difficulty"] = proposal_metrics_by_difficulty
        world_feedback_by_difficulty = core._world_feedback_delta_by_difficulty(
            baseline_metrics,
            candidate_metrics,
        )
        if world_feedback_by_difficulty:
            evidence["world_feedback_by_difficulty"] = world_feedback_by_difficulty
        evidence["long_horizon_summary"] = core._long_horizon_summary(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["first_step_confidence_delta"] = (
            candidate_metrics.average_first_step_path_confidence
            - baseline_metrics.average_first_step_path_confidence
        )
        runtime_policy = payload.get("runtime_policy", {}) if isinstance(payload, dict) else {}
        hybrid_runtime = payload.get("hybrid_runtime", {}) if isinstance(payload, dict) else {}
        if isinstance(runtime_policy, dict):
            evidence["primary_benchmark_family_count"] = len(
                [str(value).strip() for value in runtime_policy.get("primary_benchmark_families", []) if str(value).strip()]
            )
            evidence["shadow_benchmark_family_count"] = len(
                [str(value).strip() for value in runtime_policy.get("shadow_benchmark_families", []) if str(value).strip()]
            )
        if isinstance(hybrid_runtime, dict):
            evidence["primary_takeover_enabled"] = bool(hybrid_runtime.get("primary_enabled", False))
            evidence["shadow_takeover_enabled"] = bool(hybrid_runtime.get("shadow_enabled", False))
        evidence["checkpoint_exists"] = bool(checkpoint_path and Path(checkpoint_path).exists())
        evidence["cache_count"] = (
            len([str(path).strip() for path in cache_paths if str(path).strip()])
            if isinstance(cache_paths, list)
            else 0
        )
    if subsystem == "qwen_adapter":
        runtime_paths = payload.get("runtime_paths", {}) if isinstance(payload, dict) else {}
        dataset_manifest = payload.get("training_dataset_manifest", {}) if isinstance(payload, dict) else {}
        runtime_policy = payload.get("runtime_policy", {}) if isinstance(payload, dict) else {}
        evidence["family_discrimination_gain"] = core._family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        if isinstance(dataset_manifest, dict):
            evidence["dataset_total_examples"] = int(dataset_manifest.get("total_examples", 0) or 0)
            evidence["long_horizon_example_count"] = int(dataset_manifest.get("long_horizon_example_count", 0) or 0)
        runtime_target = ""
        if isinstance(runtime_paths, dict):
            runtime_target = str(
                runtime_paths.get("served_model_name")
                or runtime_paths.get("merged_output_dir")
                or runtime_paths.get("adapter_output_dir")
            ).strip()
        evidence["runtime_target_declared"] = bool(runtime_target)
        evidence["runtime_target"] = runtime_target
        evidence["base_model_name"] = str(payload.get("base_model_name", "")).strip() if isinstance(payload, dict) else ""
        evidence["support_runtime_only"] = not bool(
            runtime_policy.get("allow_primary_routing", False)
        ) if isinstance(runtime_policy, dict) else True
        evidence["teacher_generation_enabled"] = bool(
            runtime_policy.get("allow_teacher_generation", False)
        ) if isinstance(runtime_policy, dict) else False
    if subsystem == "verifier":
        verifier_candidate_total = candidate_metrics.total_by_benchmark_family.get("verifier_candidate", 0)
        verifier_candidate_pass_rate = candidate_metrics.benchmark_family_pass_rate("verifier_candidate")
        false_failure_rate = core._candidate_family_failure_rate(candidate_metrics, "verifier_candidate")
        evidence["verifier_candidate_total"] = verifier_candidate_total
        evidence["verifier_candidate_pass_rate"] = verifier_candidate_pass_rate
        evidence["false_failure_rate"] = false_failure_rate
        evidence["proposal_discrimination_estimate"] = core._verifier_discrimination_gain(payload)
        evidence["discrimination_gain"] = max(0.0, verifier_candidate_pass_rate * (1.0 - false_failure_rate))
        evidence["require_contract_strictness_satisfied"] = core._verifier_contracts_are_strict(payload)
    if subsystem == "benchmark":
        benchmark_candidate_total = candidate_metrics.total_by_benchmark_family.get("benchmark_candidate", 0)
        evidence["benchmark_candidate_total"] = benchmark_candidate_total
        evidence["family_discrimination_gain"] = core._family_discrimination_gain(
            baseline_metrics,
            candidate_metrics,
        )
        evidence["false_failure_rate"] = (
            0.0
            if benchmark_candidate_total == 0
            else core._candidate_family_failure_rate(candidate_metrics, "benchmark_candidate")
        )
    if subsystem == "operators":
        evidence["transfer_pass_rate"] = candidate_metrics.memory_source_pass_rate("operator")
        evidence["baseline_transfer_pass_rate"] = baseline_metrics.memory_source_pass_rate("skill_transfer")
        evidence["transfer_pass_rate_delta"] = (
            candidate_metrics.memory_source_pass_rate("operator")
            - baseline_metrics.memory_source_pass_rate("skill_transfer")
        )
        evidence["support_count"] = core._operator_support_count(payload)
    if subsystem == "tooling":
        evidence["replay_verified"] = core._tool_candidates_have_stage(payload, "replay_verified")
        evidence.update(artifact_support_evidence_ext.tool_shared_repo_bundle_evidence(payload))
        shared_repo_bundle_summary = artifact_support_evidence_ext.tool_shared_repo_bundle_comparison(
            payload,
            active_artifact_payload_from_generation_context_fn=planner_runtime_state_ext.active_artifact_payload_from_generation_context,
        )
        if shared_repo_bundle_summary:
            evidence["shared_repo_bundle_summary"] = shared_repo_bundle_summary
    if subsystem == "trust":
        baseline_controls = core._trust_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_controls = core._trust_controls_from_payload(payload)
        evidence.update(core._trust_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "universe":
        baseline_payload = planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        evidence.update(
            core._universe_control_evidence(
                baseline_governance=core._universe_governance_from_payload(baseline_payload),
                candidate_governance=core._universe_governance_from_payload(payload),
                baseline_action_risk_controls=core._universe_action_risk_controls_from_payload(baseline_payload),
                candidate_action_risk_controls=core._universe_action_risk_controls_from_payload(payload),
                baseline_environment_assumptions=core._universe_environment_assumptions_from_payload(baseline_payload),
                candidate_environment_assumptions=core._universe_environment_assumptions_from_payload(payload),
                baseline_invariants=core._universe_invariants_from_payload(baseline_payload),
                candidate_invariants=core._universe_invariants_from_payload(payload),
                baseline_forbidden_patterns=core._universe_forbidden_patterns_from_payload(baseline_payload),
                candidate_forbidden_patterns=core._universe_forbidden_patterns_from_payload(payload),
                baseline_preferred_prefixes=core._universe_preferred_prefixes_from_payload(baseline_payload),
                candidate_preferred_prefixes=core._universe_preferred_prefixes_from_payload(payload),
            )
        )
        evidence["universe_change_scope"] = core._universe_change_scope(payload)
        evidence["cross_family_support"] = core._universe_cross_family_support(evidence)
        evidence["outcome_weighted_support"] = core._universe_outcome_weighted_support(evidence)
        retention_context = payload.get("retention_context", {}) if isinstance(payload, dict) else {}
        if isinstance(retention_context, dict):
            for key in ("prior_retained_universe_cycle_count", "constitution_cooldown_cycles_remaining"):
                if key in retention_context:
                    evidence[key] = retention_context.get(key)
    if subsystem == "recovery":
        baseline_controls = core._recovery_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_controls = core._recovery_controls_from_payload(payload)
        evidence.update(core._recovery_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "delegation":
        baseline_controls = core._delegation_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_controls = core._delegation_controls_from_payload(payload)
        evidence.update(core._delegation_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "operator_policy":
        baseline_controls = core._operator_policy_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_controls = core._operator_policy_controls_from_payload(payload)
        evidence.update(core._operator_policy_control_evidence(baseline_controls, candidate_controls))
    if subsystem == "transition_model":
        baseline_payload = planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        baseline_controls = control_evidence.transition_model_controls_from_payload(baseline_payload)
        candidate_controls = control_evidence.transition_model_controls_from_payload(payload)
        baseline_signatures = core._transition_model_signatures_from_payload(baseline_payload)
        candidate_signatures = core._transition_model_signatures_from_payload(payload)
        evidence.update(
            core._transition_model_evidence(
                baseline_controls,
                candidate_controls,
                baseline_signatures=baseline_signatures,
                candidate_signatures=candidate_signatures,
            )
        )
    if subsystem == "capabilities":
        baseline_summary = core.capability_surface_summary(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_summary = core.capability_surface_summary(payload)
        evidence.update(core._capability_surface_evidence(baseline_summary, candidate_summary))
    if subsystem == "world_model":
        evidence["low_confidence_episode_delta"] = (
            candidate_metrics.low_confidence_episodes - baseline_metrics.low_confidence_episodes
        )
        evidence["first_step_success_delta"] = (
            candidate_metrics.first_step_successes - baseline_metrics.first_step_successes
        )
    if subsystem == "state_estimation":
        baseline_transition_controls = control_evidence.state_estimation_transition_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_transition_controls = control_evidence.state_estimation_transition_controls_from_payload(payload)
        baseline_latent_controls = control_evidence.state_estimation_latent_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_latent_controls = control_evidence.state_estimation_latent_controls_from_payload(payload)
        baseline_policy_controls = control_evidence.state_estimation_policy_controls_from_payload(
            planner_runtime_state_ext.active_artifact_payload_from_generation_context(payload)
        )
        candidate_policy_controls = control_evidence.state_estimation_policy_controls_from_payload(payload)
        evidence.update(
            core._state_estimation_evidence(
                baseline_transition_controls=baseline_transition_controls,
                candidate_transition_controls=candidate_transition_controls,
                baseline_latent_controls=baseline_latent_controls,
                candidate_latent_controls=candidate_latent_controls,
                baseline_policy_controls=baseline_policy_controls,
                candidate_policy_controls=candidate_policy_controls,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
            )
        )
    return evidence


__all__ = ["retention_evidence"]
