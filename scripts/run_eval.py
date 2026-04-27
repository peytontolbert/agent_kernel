from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from evals.harness import (
    compare_abstraction_transfer_modes,
    compare_research_library_modes,
    compare_skill_modes,
    compare_tolbert_feature_modes,
    compare_tolbert_modes,
    run_eval,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-tolbert-context", choices=("0", "1"), default=None)
    parser.add_argument("--use-research-library-context", choices=("0", "1"), default=None)
    parser.add_argument("--research-library-standalone-context", choices=("0", "1"), default=None)
    parser.add_argument("--require-live-llm-coding-control", choices=("0", "1"), default=None)
    parser.add_argument(
        "--tolbert-mode",
        choices=("full", "path_only", "retrieval_only", "deterministic_command", "skill_ranking"),
        default=None,
    )
    parser.add_argument("--use-skills", choices=("0", "1"), default=None)
    parser.add_argument("--tolbert-python", default=None)
    parser.add_argument("--tolbert-config", default=None)
    parser.add_argument("--tolbert-checkpoint", default=None)
    parser.add_argument("--tolbert-nodes", default=None)
    parser.add_argument("--tolbert-cache", action="append", default=None)
    parser.add_argument("--tolbert-label-map", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--priority-benchmark-family", action="append", default=None)
    parser.add_argument("--restrict-to-priority-benchmark-families", action="store_true")
    parser.add_argument("--prefer-retrieval-tasks", action="store_true")
    parser.add_argument("--prefer-low-cost-tasks", action="store_true")
    parser.add_argument("--prefer-long-horizon-tasks", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--include-discovered-tasks", action="store_true")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-benchmark-candidates", action="store_true")
    parser.add_argument("--include-verifier-candidates", action="store_true")
    parser.add_argument("--use-prompt-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-curriculum-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-retrieval-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-state-estimation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-tolbert-model-artifacts", choices=("0", "1"), default=None)
    parser.add_argument("--use-trust-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-recovery-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-delegation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-operator-policy-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-transition-model-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--allow-git-commands", choices=("0", "1"), default=None)
    parser.add_argument("--allow-http-requests", choices=("0", "1"), default=None)
    parser.add_argument("--allow-generated-path-mutations", choices=("0", "1"), default=None)
    parser.add_argument("--compare-skills", action="store_true")
    parser.add_argument("--compare-abstractions", action="store_true")
    parser.add_argument("--compare-tolbert", action="store_true")
    parser.add_argument("--compare-research-library", action="store_true")
    parser.add_argument("--compare-tolbert-features", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.use_tolbert_context is not None:
        config.use_tolbert_context = args.use_tolbert_context == "1"
    if args.use_research_library_context is not None:
        config.use_research_library_context = args.use_research_library_context == "1"
    if args.research_library_standalone_context is not None:
        config.research_library_standalone_context = args.research_library_standalone_context == "1"
    if args.require_live_llm_coding_control is not None:
        config.asi_coding_require_live_llm = args.require_live_llm_coding_control == "1"
    if args.tolbert_mode:
        config.tolbert_mode = args.tolbert_mode
    if args.use_skills is not None:
        config.use_skills = args.use_skills == "1"
    if args.use_prompt_proposals is not None:
        config.use_prompt_proposals = args.use_prompt_proposals == "1"
    if args.use_curriculum_proposals is not None:
        config.use_curriculum_proposals = args.use_curriculum_proposals == "1"
    if args.use_retrieval_proposals is not None:
        config.use_retrieval_proposals = args.use_retrieval_proposals == "1"
    if args.use_state_estimation_proposals is not None:
        config.use_state_estimation_proposals = args.use_state_estimation_proposals == "1"
    if args.use_tolbert_model_artifacts is not None:
        config.use_tolbert_model_artifacts = args.use_tolbert_model_artifacts == "1"
    if args.use_trust_proposals is not None:
        config.use_trust_proposals = args.use_trust_proposals == "1"
    if args.use_recovery_proposals is not None:
        config.use_recovery_proposals = args.use_recovery_proposals == "1"
    if args.use_delegation_proposals is not None:
        config.use_delegation_proposals = args.use_delegation_proposals == "1"
    if args.use_operator_policy_proposals is not None:
        config.use_operator_policy_proposals = args.use_operator_policy_proposals == "1"
    if args.use_transition_model_proposals is not None:
        config.use_transition_model_proposals = args.use_transition_model_proposals == "1"
    if args.allow_git_commands is not None:
        config.unattended_allow_git_commands = args.allow_git_commands == "1"
    if args.allow_http_requests is not None:
        config.unattended_allow_http_requests = args.allow_http_requests == "1"
    if args.allow_generated_path_mutations is not None:
        config.unattended_allow_generated_path_mutations = args.allow_generated_path_mutations == "1"
    if args.tolbert_python:
        config.tolbert_python_bin = args.tolbert_python
    if args.tolbert_config:
        config.tolbert_config_path = args.tolbert_config
    if args.tolbert_checkpoint:
        config.tolbert_checkpoint_path = args.tolbert_checkpoint
    if args.tolbert_nodes:
        config.tolbert_nodes_path = args.tolbert_nodes
    if args.tolbert_cache:
        config.tolbert_cache_paths = tuple(args.tolbert_cache)
    if args.tolbert_label_map:
        config.tolbert_label_map_path = args.tolbert_label_map
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device

    if args.compare_skills:
        comparison = compare_skill_modes(
            config=config,
            include_discovered_tasks=args.include_discovered_tasks,
            include_episode_memory=args.include_episode_memory,
            include_skill_memory=args.include_skill_memory,
            include_skill_transfer=args.include_skill_transfer,
            include_operator_memory=args.include_operator_memory,
            include_tool_memory=args.include_tool_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_benchmark_candidates=args.include_benchmark_candidates,
            include_verifier_candidates=args.include_verifier_candidates,
            include_generated=args.include_curriculum,
            include_failure_generated=args.include_failure_curriculum,
        )
        print(
            "skill_compare "
            f"pass_rate_delta={comparison.pass_rate_delta:.2f} "
            f"average_steps_delta={comparison.average_steps_delta:.2f}"
        )
        for capability in sorted(comparison.capability_pass_rate_delta):
            print(
                "skill_compare "
                f"capability={capability} "
                f"pass_rate_delta={comparison.capability_pass_rate_delta[capability]:.2f}"
            )
        for family in sorted(comparison.benchmark_family_pass_rate_delta):
            print(
                "skill_compare "
                f"benchmark_family={family} "
                f"pass_rate_delta={comparison.benchmark_family_pass_rate_delta[family]:.2f}"
            )
        return

    if args.compare_research_library:
        comparison = compare_research_library_modes(
            config=config,
            include_discovered_tasks=args.include_discovered_tasks,
            include_episode_memory=args.include_episode_memory,
            include_skill_memory=args.include_skill_memory,
            include_skill_transfer=args.include_skill_transfer,
            include_operator_memory=args.include_operator_memory,
            include_tool_memory=args.include_tool_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_benchmark_candidates=args.include_benchmark_candidates,
            include_verifier_candidates=args.include_verifier_candidates,
            include_generated=args.include_curriculum,
            include_failure_generated=args.include_failure_curriculum,
            task_limit=args.task_limit,
            priority_benchmark_families=args.priority_benchmark_family,
            prefer_low_cost_tasks=args.prefer_low_cost_tasks,
            prefer_long_horizon_tasks=args.prefer_long_horizon_tasks,
            prefer_retrieval_tasks=args.prefer_retrieval_tasks,
            restrict_to_priority_benchmark_families=args.restrict_to_priority_benchmark_families,
            progress_label_prefix="research_library",
        )
        print(
            "research_library_compare "
            f"pass_rate_delta={comparison.pass_rate_delta:.2f} "
            f"average_steps_delta={comparison.average_steps_delta:.2f} "
            f"average_retrieval_evidence_delta={comparison.average_retrieval_evidence_delta:.2f} "
            f"average_research_context_chunks_delta={comparison.average_research_context_chunks_delta:.2f} "
            f"average_llm_visible_research_context_chunks_delta="
            f"{comparison.average_llm_visible_research_context_chunks_delta:.2f} "
            f"average_research_retrieval_evidence_delta="
            f"{comparison.average_research_retrieval_evidence_delta:.2f} "
            f"average_research_model_assets_delta={comparison.average_research_model_assets_delta:.2f} "
            f"average_research_repository_matches_delta="
            f"{comparison.average_research_repository_matches_delta:.2f} "
            f"average_research_algorithm_matches_delta="
            f"{comparison.average_research_algorithm_matches_delta:.2f} "
            f"retrieval_influenced_steps_delta={comparison.retrieval_influenced_steps_delta} "
            f"trusted_retrieval_steps_delta={comparison.trusted_retrieval_steps_delta}"
        )
        print(
            "research_library_with "
            f"passed={comparison.with_research_library.passed} "
            f"total={comparison.with_research_library.total} "
            f"pass_rate={comparison.with_research_library.pass_rate:.2f} "
            f"average_steps={comparison.with_research_library.average_steps:.2f} "
            f"average_retrieval_evidence={comparison.with_research_library.average_retrieval_evidence:.2f} "
            f"average_research_context_chunks="
            f"{comparison.with_research_library.average_research_context_chunks:.2f} "
            f"average_llm_visible_research_context_chunks="
            f"{comparison.with_research_library.average_llm_visible_research_context_chunks:.2f} "
            f"average_research_retrieval_evidence="
            f"{comparison.with_research_library.average_research_retrieval_evidence:.2f}"
        )
        print(
            "research_library_without "
            f"passed={comparison.without_research_library.passed} "
            f"total={comparison.without_research_library.total} "
            f"pass_rate={comparison.without_research_library.pass_rate:.2f} "
            f"average_steps={comparison.without_research_library.average_steps:.2f} "
            f"average_retrieval_evidence={comparison.without_research_library.average_retrieval_evidence:.2f} "
            f"average_research_context_chunks="
            f"{comparison.without_research_library.average_research_context_chunks:.2f} "
            f"average_llm_visible_research_context_chunks="
            f"{comparison.without_research_library.average_llm_visible_research_context_chunks:.2f} "
            f"average_research_retrieval_evidence="
            f"{comparison.without_research_library.average_research_retrieval_evidence:.2f}"
        )
        for capability in sorted(comparison.capability_pass_rate_delta):
            print(
                "research_library_compare "
                f"capability={capability} "
                f"pass_rate_delta={comparison.capability_pass_rate_delta[capability]:.2f}"
            )
        for family in sorted(comparison.benchmark_family_pass_rate_delta):
            print(
                "research_library_compare "
                f"benchmark_family={family} "
                f"pass_rate_delta={comparison.benchmark_family_pass_rate_delta[family]:.2f}"
            )
        return

    if args.compare_abstractions:
        comparison = compare_abstraction_transfer_modes(
            config=config,
            include_discovered_tasks=args.include_discovered_tasks,
            include_episode_memory=args.include_episode_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_benchmark_candidates=args.include_benchmark_candidates,
            include_verifier_candidates=args.include_verifier_candidates,
            include_generated=args.include_curriculum,
            include_failure_generated=args.include_failure_curriculum,
            task_limit=args.task_limit,
            progress_label_prefix="abstraction",
        )
        print(
            "abstraction_compare "
            f"pass_rate_delta={comparison.pass_rate_delta:.2f} "
            f"average_steps_delta={comparison.average_steps_delta:.2f} "
            f"transfer_pass_rate_delta={comparison.transfer_pass_rate_delta:.2f}"
        )
        return

    if args.compare_tolbert:
        comparison = compare_tolbert_modes(
            config=config,
            include_discovered_tasks=args.include_discovered_tasks,
            include_episode_memory=args.include_episode_memory,
            include_skill_memory=args.include_skill_memory,
            include_skill_transfer=args.include_skill_transfer,
            include_operator_memory=args.include_operator_memory,
            include_tool_memory=args.include_tool_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_benchmark_candidates=args.include_benchmark_candidates,
            include_verifier_candidates=args.include_verifier_candidates,
            include_generated=args.include_curriculum,
            include_failure_generated=args.include_failure_curriculum,
        )
        print(
            "tolbert_compare "
            f"pass_rate_delta={comparison.pass_rate_delta:.2f} "
            f"average_steps_delta={comparison.average_steps_delta:.2f}"
        )
        for capability in sorted(comparison.capability_pass_rate_delta):
            print(
                "tolbert_compare "
                f"capability={capability} "
                f"pass_rate_delta={comparison.capability_pass_rate_delta[capability]:.2f}"
            )
        for family in sorted(comparison.benchmark_family_pass_rate_delta):
            print(
                "tolbert_compare "
                f"benchmark_family={family} "
                f"pass_rate_delta={comparison.benchmark_family_pass_rate_delta[family]:.2f}"
            )
        return

    if args.compare_tolbert_features:
        comparison = compare_tolbert_feature_modes(
            config=config,
            include_discovered_tasks=args.include_discovered_tasks,
            include_episode_memory=args.include_episode_memory,
            include_skill_memory=args.include_skill_memory,
            include_skill_transfer=args.include_skill_transfer,
            include_operator_memory=args.include_operator_memory,
            include_tool_memory=args.include_tool_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_benchmark_candidates=args.include_benchmark_candidates,
            include_verifier_candidates=args.include_verifier_candidates,
            include_generated=args.include_curriculum,
            include_failure_generated=args.include_failure_curriculum,
        )
        for mode, metrics in comparison.mode_metrics.items():
            print(
                "tolbert_mode "
                f"mode={mode} "
                f"passed={metrics.passed} total={metrics.total} "
                f"pass_rate={metrics.pass_rate:.2f} "
                f"average_steps={metrics.average_steps:.2f} "
                f"retrieval_influenced_steps={metrics.retrieval_influenced_steps} "
                f"retrieval_ranked_skill_steps={metrics.retrieval_ranked_skill_steps} "
                f"retrieval_selected_steps={metrics.retrieval_selected_steps} "
                f"proposal_selected_steps={metrics.proposal_selected_steps} "
                f"novel_command_steps={metrics.novel_command_steps} "
                f"novel_valid_command_rate={metrics.novel_valid_command_rate:.2f}"
            )
        return

    metrics = run_eval(
        config=config,
        include_discovered_tasks=args.include_discovered_tasks,
        include_generated=args.include_curriculum,
        include_failure_generated=args.include_failure_curriculum,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_benchmark_candidates=args.include_benchmark_candidates,
        include_verifier_candidates=args.include_verifier_candidates,
        task_limit=args.task_limit,
        priority_benchmark_families=args.priority_benchmark_family,
        prefer_low_cost_tasks=args.prefer_low_cost_tasks,
        prefer_long_horizon_tasks=args.prefer_long_horizon_tasks,
        prefer_retrieval_tasks=args.prefer_retrieval_tasks,
        restrict_to_priority_benchmark_families=args.restrict_to_priority_benchmark_families,
        progress_label="eval" if args.task_limit else None,
    )
    print(f"passed={metrics.passed} total={metrics.total} pass_rate={metrics.pass_rate:.2f}")
    print(
        "average_steps="
        f"{metrics.average_steps:.2f} average_success_steps={metrics.average_success_steps:.2f}"
    )
    print(
        "skill_selected_steps="
        f"{metrics.skill_selected_steps} "
        f"episodes_with_skill_use={metrics.episodes_with_skill_use} "
        f"skill_use_rate={metrics.skill_use_rate:.2f} "
        f"available_but_unused_skill_steps={metrics.available_but_unused_skill_steps} "
        f"excess_available_skill_slots={metrics.excess_available_skill_slots}"
    )
    print(
        "average_available_skills="
        f"{metrics.average_available_skills:.2f} "
        f"average_retrieval_candidates={metrics.average_retrieval_candidates:.2f} "
        f"average_retrieval_evidence={metrics.average_retrieval_evidence:.2f} "
        f"average_retrieval_direct_candidates={metrics.average_retrieval_direct_candidates:.2f} "
        f"retrieval_guided_steps={metrics.retrieval_guided_steps} "
        f"retrieval_selected_steps={metrics.retrieval_selected_steps} "
        f"retrieval_influenced_steps={metrics.retrieval_influenced_steps} "
        f"retrieval_ranked_skill_steps={metrics.retrieval_ranked_skill_steps} "
        f"proposal_selected_steps={metrics.proposal_selected_steps} "
        f"novel_command_steps={metrics.novel_command_steps} "
        f"novel_valid_command_steps={metrics.novel_valid_command_steps} "
        f"novel_valid_command_rate={metrics.novel_valid_command_rate:.2f} "
        f"trusted_retrieval_steps={metrics.trusted_retrieval_steps} "
        f"low_confidence_steps={metrics.low_confidence_steps} "
        f"first_step_successes={metrics.first_step_successes}"
    )
    print(
        "average_first_step_path_confidence="
        f"{metrics.average_first_step_path_confidence:.2f} "
        f"success_first_step_path_confidence={metrics.success_first_step_path_confidence:.2f} "
        f"failed_first_step_path_confidence={metrics.failed_first_step_path_confidence:.2f} "
        f"low_confidence_episodes={metrics.low_confidence_episodes} "
        f"low_confidence_episode_pass_rate={metrics.low_confidence_episode_pass_rate:.2f}"
    )
    print(
        "memory_documents="
        f"{metrics.memory_documents} "
        f"reusable_skills={metrics.reusable_skills}"
    )
    world_feedback_summary = (
        dict(metrics.world_feedback_summary)
        if isinstance(metrics.world_feedback_summary, dict)
        else {}
    )
    if int(world_feedback_summary.get("step_count", 0) or 0) > 0:
        print(
            "world_feedback "
            f"step_count={int(world_feedback_summary.get('step_count', 0) or 0)} "
            f"progress_calibration_mae={float(world_feedback_summary.get('progress_calibration_mae', 0.0) or 0.0):.2f} "
            f"risk_calibration_mae={float(world_feedback_summary.get('risk_calibration_mae', 0.0) or 0.0):.2f} "
            f"decoder_progress_calibration_mae={float(world_feedback_summary.get('decoder_progress_calibration_mae', 0.0) or 0.0):.2f} "
            f"decoder_risk_calibration_mae={float(world_feedback_summary.get('decoder_risk_calibration_mae', 0.0) or 0.0):.2f}"
        )
    for capability in sorted(metrics.total_by_capability):
        passed = metrics.passed_by_capability.get(capability, 0)
        total = metrics.total_by_capability[capability]
        rate = metrics.capability_pass_rate(capability)
        print(f"capability={capability} passed={passed} total={total} pass_rate={rate:.2f}")
    for difficulty in sorted(metrics.total_by_difficulty):
        passed = metrics.passed_by_difficulty.get(difficulty, 0)
        total = metrics.total_by_difficulty[difficulty]
        rate = metrics.difficulty_pass_rate(difficulty)
        print(f"difficulty={difficulty} passed={passed} total={total} pass_rate={rate:.2f}")
    for family in sorted(metrics.total_by_benchmark_family):
        passed = metrics.passed_by_benchmark_family.get(family, 0)
        total = metrics.total_by_benchmark_family[family]
        rate = metrics.benchmark_family_pass_rate(family)
        print(f"benchmark_family={family} passed={passed} total={total} pass_rate={rate:.2f}")
    for source in sorted(metrics.total_by_memory_source):
        passed = metrics.passed_by_memory_source.get(source, 0)
        total = metrics.total_by_memory_source[source]
        rate = metrics.memory_source_pass_rate(source)
        print(f"memory_source={source} passed={passed} total={total} pass_rate={rate:.2f}")
    for family in sorted(metrics.total_by_origin_benchmark_family):
        passed = metrics.passed_by_origin_benchmark_family.get(family, 0)
        total = metrics.total_by_origin_benchmark_family[family]
        rate = metrics.origin_benchmark_family_pass_rate(family)
        print(f"origin_benchmark_family={family} passed={passed} total={total} pass_rate={rate:.2f}")
    for family in sorted(metrics.world_feedback_by_benchmark_family):
        summary = metrics.world_feedback_by_benchmark_family[family]
        if int(summary.get("step_count", 0) or 0) <= 0:
            continue
        print(
            "world_feedback_benchmark_family "
            f"benchmark_family={family} "
            f"step_count={int(summary.get('step_count', 0) or 0)} "
            f"progress_calibration_mae={float(summary.get('progress_calibration_mae', 0.0) or 0.0):.2f} "
            f"risk_calibration_mae={float(summary.get('risk_calibration_mae', 0.0) or 0.0):.2f}"
        )
    if metrics.generated_total:
        print(
            "generated_passed="
            f"{metrics.generated_passed} generated_total={metrics.generated_total} "
            f"generated_pass_rate={metrics.generated_pass_rate:.2f}"
        )
        for kind in sorted(metrics.generated_by_kind):
            passed = metrics.generated_passed_by_kind.get(kind, 0)
            total = metrics.generated_by_kind[kind]
            rate = 0.0 if total == 0 else passed / total
            print(f"generated_kind={kind} passed={passed} total={total} pass_rate={rate:.2f}")
        for family in sorted(metrics.generated_by_benchmark_family):
            passed = metrics.generated_passed_by_benchmark_family.get(family, 0)
            total = metrics.generated_by_benchmark_family[family]
            rate = 0.0 if total == 0 else passed / total
            print(f"generated_benchmark_family={family} passed={passed} total={total} pass_rate={rate:.2f}")
    for reason, count in sorted(metrics.termination_reasons.items()):
        print(f"termination_reason={reason} count={count}")


if __name__ == "__main__":
    main()
