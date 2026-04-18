from __future__ import annotations

from evals.metrics import EvalMetrics

from ...improvement_engine import ImprovementExperiment, sort_experiments as engine_sort_experiments


def rank_experiments(planner, metrics: EvalMetrics) -> list[ImprovementExperiment]:
    failure_counts = planner.failure_counts()
    transition_failure_counts = planner.transition_failure_counts()
    transition_summary = planner.transition_summary()
    trust_summary = planner.trust_ledger_summary()
    candidates: list[ImprovementExperiment] = []
    if (failure_counts or transition_failure_counts) and (
        metrics.total_by_benchmark_family.get("benchmark_candidate", 0) == 0
        or metrics.generated_pass_rate < metrics.pass_rate
    ):
        benchmark_gain_raw = max(
            0.02,
            (
                failure_counts.get("missing_expected_file", 0)
                + transition_failure_counts.get("no_state_progress", 0)
                + transition_failure_counts.get("state_regression", 0)
            )
            / max(1, sum(failure_counts.values())),
        )
        benchmark_gain = round(max(0.02, min(0.05, benchmark_gain_raw)), 4)
        candidates.append(
            ImprovementExperiment(
                subsystem="benchmark",
                reason="failure clusters, stalled transitions, and environment patterns can be turned into benchmark proposals, but those proposals must expand operator-relevant coverage instead of only matching the current validator shape",
                priority=5,
                expected_gain=benchmark_gain,
                estimated_cost=3,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "benchmark_candidate_total": metrics.total_by_benchmark_family.get("benchmark_candidate", 0),
                    "generated_pass_rate": metrics.generated_pass_rate,
                    "pass_rate": metrics.pass_rate,
                    "benchmark_expected_gain_raw": round(float(benchmark_gain_raw), 4),
                    "benchmark_expected_gain_capped": benchmark_gain,
                    "benchmark_candidate_share": round(
                        metrics.total_by_benchmark_family.get("benchmark_candidate", 0) / max(1, metrics.total),
                        4,
                    ),
                },
            )
        )
    low_confidence_state_signal = 0
    if metrics.low_confidence_episodes > 0 or metrics.trusted_retrieval_steps < metrics.total // 2:
        confidence_gap = max(
            metrics.low_confidence_episodes / max(1, metrics.total),
            0.0 if metrics.total == 0 else 0.5 - (metrics.trusted_retrieval_steps / max(1, metrics.total)),
        )
        retrieval_deficit = max(0, (metrics.total // 2) - int(metrics.trusted_retrieval_steps or 0))
        if confidence_gap >= 0.1 or retrieval_deficit > 0:
            low_confidence_state_signal = max(
                int(metrics.low_confidence_episodes or 0),
                retrieval_deficit,
            )
        allow_qwen_support_runtime, coding_strength = planner._allow_qwen_adapter_support_runtime(metrics)
        candidates.append(
            ImprovementExperiment(
                subsystem="retrieval",
                reason="low-confidence retrieval remains common relative to trusted retrieval usage",
                priority=5,
                expected_gain=round(max(0.02, confidence_gap), 4),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "retrieval_selected_steps": metrics.retrieval_selected_steps,
                    "retrieval_influenced_steps": metrics.retrieval_influenced_steps,
                    "proposal_selected_steps": metrics.proposal_selected_steps,
                    "total": metrics.total,
                },
            )
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="tolbert_model",
                reason="persistent retrieval weakness may still need a learned Tolbert checkpoint, but runtime retrieval and state controls should be repaired first",
                priority=4,
                expected_gain=round(max(0.02, confidence_gap), 4),
                estimated_cost=4,
                score=0.0,
                evidence={
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                    "total": metrics.total,
                    "coding_strength": coding_strength,
                    "cross_cycle_weight_update": True,
                },
            )
        )
        if allow_qwen_support_runtime:
            candidates.append(
                ImprovementExperiment(
                    subsystem="qwen_adapter",
                    reason="current coding weakness may eventually need a stronger adapted Qwen support runtime, but prompt and state controls should lead the low-confidence repair loop",
                    priority=4,
                    expected_gain=round(max(0.015, confidence_gap * 0.6), 4),
                    estimated_cost=3,
                    score=0.0,
                    evidence={
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                        "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                        "total": metrics.total,
                        "support_runtime_only": True,
                        "coding_strength": coding_strength,
                        "cross_cycle_weight_update": True,
                    },
                )
            )
    if metrics.total_by_memory_source.get("verifier", 0) == 0 or transition_failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="verifier",
                reason="verifier-memory lane is not yet populated enough to discriminate stalled or regressive intermediate states",
                priority=5,
                expected_gain=round(max(0.03, min(0.05, 0.01 * sum(transition_failure_counts.values()))), 4),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "verifier_memory_total": metrics.total_by_memory_source.get("verifier", 0),
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if failure_counts.get("command_failure", 0) >= failure_counts.get("missing_expected_file", 0) and failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="tooling",
                reason="command failures dominate stored failures, suggesting missing reusable procedures or tools",
                priority=4,
                expected_gain=round(
                    max(0.01, failure_counts.get("command_failure", 0) / max(1, sum(failure_counts.values()))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "command_failure_count": failure_counts.get("command_failure", 0),
                    "failure_counts": failure_counts,
                },
            )
        )
    if metrics.retrieval_ranked_skill_steps < max(1, metrics.skill_selected_steps // 2):
        candidates.append(
            ImprovementExperiment(
                subsystem="skills",
                reason="skill usage is high but retrieval-ranked skill selection is comparatively weak",
                priority=4,
                expected_gain=round(
                    max(
                        0.01,
                        (metrics.skill_selected_steps - metrics.retrieval_ranked_skill_steps) / max(1, metrics.skill_selected_steps),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "skill_selected_steps": metrics.skill_selected_steps,
                    "retrieval_ranked_skill_steps": metrics.retrieval_ranked_skill_steps,
                },
            )
        )
    if metrics.total_by_memory_source.get("skill_transfer", 0) == 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="operators",
                reason="cross-task operator transfer is not yet populated or measured against raw skill transfer",
                priority=4,
                expected_gain=0.03,
                estimated_cost=4,
                score=0.0,
                evidence={
                    "skill_transfer_total": metrics.total_by_memory_source.get("skill_transfer", 0),
                    "operator_total": metrics.total_by_memory_source.get("operator", 0),
                },
            )
        )
    if (
        (metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate)
        or transition_failure_counts.get("no_state_progress", 0) > 0
        or transition_failure_counts.get("state_regression", 0) > 0
    ):
        generated_gap = 0.0
        if metrics.generated_total:
            generated_gap = max(0.0, metrics.pass_rate - metrics.generated_pass_rate)
        candidates.append(
            ImprovementExperiment(
                subsystem="curriculum",
                reason="generated-task pressure should target repeated no-progress and regression transitions, not only coarse task failures",
                priority=4,
                expected_gain=round(
                    max(
                        0.02,
                        generated_gap,
                        min(
                            0.05,
                            (
                                transition_failure_counts.get("no_state_progress", 0)
                                + transition_failure_counts.get("state_regression", 0)
                            )
                            / max(1, sum(transition_failure_counts.values())),
                        ),
                    ),
                    4,
                ),
                estimated_cost=3,
                score=0.0,
                evidence={
                    "pass_rate": metrics.pass_rate,
                    "generated_pass_rate": metrics.generated_pass_rate,
                    "generated_total": metrics.generated_total,
                    "generated_pass_rate_gap": round(generated_gap, 4),
                    "transition_failure_counts": transition_failure_counts,
                },
            )
        )
    repo_world_model_total = sum(
        int(metrics.total_by_benchmark_family.get(family, 0))
        for family in ("repo_sandbox", "repo_chore", "repository", "project", "integration")
    )
    world_model_failure_signal = (
        failure_counts.get("missing_expected_file", 0)
        + failure_counts.get("command_failure", 0)
        + transition_failure_counts.get("no_state_progress", 0)
        + transition_failure_counts.get("state_regression", 0)
    )
    if repo_world_model_total > 0 and world_model_failure_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="world_model",
                reason="repo-workflow failures and bad transitions suggest command scoring, progress estimation, and preserved-path modeling are still weak",
                priority=3,
                expected_gain=round(
                    max(0.01, min(0.03, world_model_failure_signal / max(1, sum(failure_counts.values())))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "repo_world_model_total": repo_world_model_total,
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if (
        transition_failure_counts
        or int(transition_summary.get("state_regression_steps", 0) or 0) > 0
        or low_confidence_state_signal > 0
    ):
        state_estimation_signal = (
            transition_failure_counts.get("no_state_progress", 0)
            + transition_failure_counts.get("state_regression", 0)
            + int(transition_summary.get("state_regression_steps", 0) or 0)
            + low_confidence_state_signal
        )
        state_estimation_reason = (
            "low-confidence routing should first be repaired through runtime state summarization and recovery cues before cross-cycle weight updates"
            if low_confidence_state_signal > 0
            else "state summarization should separate stalls, regressions, and recovery opportunities more explicitly before policy scoring"
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="state_estimation",
                reason=state_estimation_reason,
                priority=4,
                expected_gain=round(
                    max(
                        0.015,
                        min(
                            0.05,
                            state_estimation_signal
                            / max(1, (sum(transition_failure_counts.values()) or 0) + low_confidence_state_signal),
                        ),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "average_first_step_path_confidence": metrics.average_first_step_path_confidence,
                    "total": metrics.total,
                    "runtime_learning_priority": low_confidence_state_signal > 0,
                },
            )
        )
    universe_constitution_signal = (
        failure_counts.get("command_failure", 0)
        + transition_failure_counts.get("no_state_progress", 0)
        + transition_failure_counts.get("state_regression", 0)
        + metrics.low_confidence_episodes
    )
    environment_violation_summary = planner.environment_violation_summary()
    universe_cycle_feedback = planner.universe_cycle_feedback_summary()
    operating_envelope_signal = (
        int(environment_violation_summary.get("violation_total", 0))
        + int(environment_violation_summary.get("alignment_failure_total", 0))
    )
    broad_support_bonus = min(
        2,
        int(
            universe_cycle_feedback.get(
                "broad_support_cycle_count",
                universe_cycle_feedback.get("retained_cycle_count", 0),
            )
        ),
    )
    universe_constitution_signal += min(
        1,
        int(universe_cycle_feedback.get("constitution_retained_cycle_count", 0) or 0),
    )
    operating_envelope_signal += broad_support_bonus
    operating_envelope_signal += min(
        1,
        int(universe_cycle_feedback.get("operating_envelope_retained_cycle_count", 0) or 0),
    )
    if universe_constitution_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="universe_constitution",
                reason="constitutional machine-law should tighten verifier, bounded-action, and destructive-reset rules above task-local world state",
                priority=4,
                expected_gain=round(
                    max(0.012, min(0.025, universe_constitution_signal / max(1, metrics.total or 1))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "universe_cycle_feedback": universe_cycle_feedback,
                    "total": metrics.total,
                },
            )
        )
    if operating_envelope_signal > 0:
        candidates.append(
            ImprovementExperiment(
                subsystem="operating_envelope",
                reason="the retained operating envelope should calibrate to repeated environment conflicts and attested runtime mismatch",
                priority=4,
                expected_gain=round(
                    max(0.015, min(0.03, operating_envelope_signal / max(1, metrics.total or 1))),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "failure_counts": failure_counts,
                    "transition_failure_counts": transition_failure_counts,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "environment_violation_summary": environment_violation_summary,
                    "universe_cycle_feedback": universe_cycle_feedback,
                    "total": metrics.total,
                },
            )
        )
    if transition_failure_counts:
        candidates.append(
            ImprovementExperiment(
                subsystem="transition_model",
                reason="retained bad-transition signatures should directly penalize repeated stalled or regressive commands",
                priority=5,
                expected_gain=round(
                    max(
                        0.02,
                        min(
                            0.05,
                            (
                                transition_failure_counts.get("no_state_progress", 0)
                                + transition_failure_counts.get("state_regression", 0)
                            )
                            / max(1, sum(transition_failure_counts.values())),
                        ),
                    ),
                    4,
                ),
                estimated_cost=2,
                score=0.0,
                evidence={
                    "transition_failure_counts": transition_failure_counts,
                    "transition_summary": transition_summary,
                },
            )
        )
    if trust_summary and (
        str(trust_summary.get("overall_status", "")).strip() in {"bootstrap", "restricted"}
        or float(trust_summary.get("unsafe_ambiguous_rate", 0.0)) > 0.0
        or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("false_pass_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("unexpected_change_report_rate", 0.0)) > 0.0
        or int(trust_summary.get("distinct_benchmark_families", 0)) < 2
        or int(trust_summary.get("distinct_family_gap", 0)) > 0
        or list(trust_summary.get("missing_required_families", []))
        or list(trust_summary.get("missing_required_family_clean_task_root_breadth", []))
    ):
        clean_success_rate = float(trust_summary.get("clean_success_rate", 0.0))
        false_pass_risk_rate = float(trust_summary.get("false_pass_risk_rate", 0.0))
        hidden_side_effect_risk_rate = float(trust_summary.get("hidden_side_effect_risk_rate", 0.0))
        unexpected_change_report_rate = float(trust_summary.get("unexpected_change_report_rate", 0.0))
        missing_required_task_root_breadth = list(
            trust_summary.get("missing_required_family_clean_task_root_breadth", [])
        )
        breadth_threshold = int(trust_summary.get("family_breadth_min_distinct_task_roots", 0) or 0)
        breadth_gap = 0.0
        if breadth_threshold > 0 and missing_required_task_root_breadth:
            counts = trust_summary.get("required_family_clean_task_root_counts", {})
            if not isinstance(counts, dict):
                counts = {}
            deficits = [
                max(0, breadth_threshold - int(counts.get(family, 0) or 0))
                for family in missing_required_task_root_breadth
            ]
            breadth_gap = min(0.08, sum(deficits) * 0.02)
        false_pass_contamination = max(0.0, false_pass_risk_rate - max(0.0, clean_success_rate))
        trust_risk_signal = max(
            float(trust_summary.get("unsafe_ambiguous_rate", 0.0)),
            hidden_side_effect_risk_rate,
            float(trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)),
            false_pass_risk_rate,
            unexpected_change_report_rate,
            false_pass_contamination,
            breadth_gap,
            min(
                0.08,
                (int(trust_summary.get("distinct_family_gap", 0) or 0) * 0.01)
                + (len(list(trust_summary.get("missing_required_families", []))) * 0.005),
            ),
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="trust",
                reason="unattended trust gating remains restricted, coverage-misaligned, or exposed to hidden-risk and false-pass outcomes",
                priority=4,
                expected_gain=round(max(0.01, trust_risk_signal or 0.02), 4),
                estimated_cost=2,
                score=0.0,
                evidence=trust_summary,
            )
        )
    if trust_summary and (
        float(trust_summary.get("rollback_performed_rate", 0.0)) > 0.0
        or float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("false_pass_risk_rate", 0.0)) > 0.0
        or float(trust_summary.get("unexpected_change_report_rate", 0.0)) > 0.0
    ):
        rollback_performed_rate = float(trust_summary.get("rollback_performed_rate", 0.0))
        hidden_side_effect_risk_rate = float(trust_summary.get("hidden_side_effect_risk_rate", 0.0))
        success_hidden_side_effect_risk_rate = float(
            trust_summary.get("success_hidden_side_effect_risk_rate", 0.0)
        )
        false_pass_risk_rate = float(trust_summary.get("false_pass_risk_rate", 0.0))
        unexpected_change_report_rate = float(trust_summary.get("unexpected_change_report_rate", 0.0))
        recovery_signal = max(
            rollback_performed_rate,
            hidden_side_effect_risk_rate,
            success_hidden_side_effect_risk_rate,
            false_pass_risk_rate,
            unexpected_change_report_rate,
        )
        candidates.append(
            ImprovementExperiment(
                subsystem="recovery",
                reason="unattended runs still depend on rollback or leave residual side-effect uncertainty after restore paths",
                priority=4,
                expected_gain=round(max(0.01, recovery_signal), 4),
                estimated_cost=2,
                score=0.0,
                evidence=trust_summary,
            )
        )
    delegation_summary = planner.delegation_policy_summary()
    if delegation_summary and (
        int(delegation_summary.get("delegated_job_max_concurrency", 1)) < 3
        or int(delegation_summary.get("delegated_job_max_active_per_budget_group", 0)) < 2
        or int(delegation_summary.get("delegated_job_max_queued_per_budget_group", 0)) < 8
        or int(delegation_summary.get("delegated_job_max_subprocesses_per_job", 1)) < 2
        or int(delegation_summary.get("max_steps", 5)) < 12
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="delegation",
                reason="delegated execution remains throttled by shallow worker budgets or narrow queue policy",
                priority=4,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=delegation_summary,
            )
        )
    operator_policy_summary = planner.operator_policy_summary()
    if operator_policy_summary and (
        len(list(operator_policy_summary.get("unattended_allowed_benchmark_families", []))) < 5
        or not bool(operator_policy_summary.get("unattended_allow_git_commands", False))
        or not bool(operator_policy_summary.get("unattended_allow_http_requests", False))
        or not bool(operator_policy_summary.get("unattended_allow_generated_path_mutations", False))
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="operator_policy",
                reason="unattended operator-boundary policy still limits family breadth or critical execution scopes",
                priority=4,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=operator_policy_summary,
            )
        )
    capability_summary = planner.capability_surface_summary()
    if capability_summary and (
        int(capability_summary.get("enabled_module_count", 0)) == 0
        or int(capability_summary.get("improvement_surface_count", 0)) == 0
    ):
        candidates.append(
            ImprovementExperiment(
                subsystem="capabilities",
                reason="capability registry does not yet expose retained autonomous improvement surfaces",
                priority=3,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.0,
                evidence=capability_summary,
            )
        )
    external_subsystems = {candidate.subsystem for candidate in candidates}
    for external in planner._plugin_layer.external_experiments(planner.capability_modules_path):
        subsystem = str(external.get("subsystem", "")).strip()
        if not subsystem or subsystem in external_subsystems:
            continue
        candidates.append(
            ImprovementExperiment(
                subsystem=subsystem,
                reason=str(external.get("reason", "")).strip() or "module-defined improvement surface",
                priority=int(external.get("priority", 3)),
                expected_gain=float(external.get("expected_gain", 0.01)),
                estimated_cost=int(external.get("estimated_cost", 2)),
                score=0.0,
                evidence=dict(external.get("evidence", {}))
                if isinstance(external.get("evidence", {}), dict)
                else {},
            )
        )
        external_subsystems.add(subsystem)
    candidates.append(
        ImprovementExperiment(
            subsystem="policy",
            reason="no sharper subsystem deficit dominates current metrics",
            priority=3,
            expected_gain=0.01,
            estimated_cost=2,
            score=0.0,
            evidence={"default_fallback": True},
        )
    )
    planner_controls = planner._improvement_planner_controls()
    learning_candidate_summary = planner._learning_candidate_summary()
    scored_candidates = [
        planner._score_experiment(
            candidate,
            metrics=metrics,
            planner_controls=planner_controls,
            learning_candidate_summary=learning_candidate_summary,
            trust_summary=trust_summary,
        )
        for candidate in candidates
    ]
    return engine_sort_experiments(scored_candidates)


__all__ = ["rank_experiments"]
