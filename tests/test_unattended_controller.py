import json
from pathlib import Path
from types import SimpleNamespace

from agent_kernel.llm import MockLLMClient
from agent_kernel.modeling.policy import decoder as decoder_module
from agent_kernel.policy import LLMDecisionPolicy
from agent_kernel.schemas import TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.subsystems import external_subsystem_specs
from agent_kernel.unattended_controller import (
    action_key_for_policy,
    build_failure_observation,
    build_round_observation,
    controller_state_summary,
    default_controller_state,
    observation_reward,
    plan_next_policy,
    predict_next_observation,
    update_controller_state,
)
from scripts import run_job_queue as run_job_queue_module


def test_update_controller_state_learns_transition_and_reward():
    state = default_controller_state(exploration_bonus=0.0)
    policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    start = build_round_observation(campaign_signal={})
    end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.08,
            "average_retained_step_delta": -0.5,
        },
        liftoff_signal={"state": "shadow_only", "allow_kernel_autobuild": True},
    )

    state, diagnostics = update_controller_state(
        state,
        start_observation=start,
        action_policy=policy,
        end_observation=end,
    )

    action_key = action_key_for_policy(policy)
    action_stats = state["action_models"][action_key]
    assert state["updates"] == 1
    assert diagnostics["reward"] > 0.0
    assert action_stats["count"] == 1
    assert action_stats["reward_mean"] > 0.0
    assert action_stats["transition_mean"]["retained_cycles"] > 0.0


def test_controller_state_summary_preserves_strategy_memory_priors():
    state = default_controller_state()
    state["strategy_memory_priors"] = {
        "retained_count": 2,
        "recent_rejects": 1,
        "best_retained_gain": 0.25,
        "retained_family_gains": {"integration": 0.25},
        "recent_rejects_by_family": {"integration": 1},
    }

    summary = controller_state_summary(state)

    assert summary["strategy_memory_priors"]["retained_count"] == 2
    assert summary["strategy_memory_priors"]["best_retained_gain"] == 0.25
    assert summary["strategy_memory_priors"]["retained_family_gains"]["integration"] == 0.25
    assert summary["strategy_memory_priors"]["recent_rejects_by_family"]["integration"] == 1


def test_predict_next_observation_depends_on_current_state_and_action():
    state = default_controller_state(exploration_bonus=0.0)
    policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    pressured_start = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "max_low_confidence_episode_delta": 5,
            "average_retained_pass_rate_delta": -0.03,
        }
    )
    pressured_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.08,
            "max_low_confidence_episode_delta": 1,
        }
    )
    calm_start = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "max_low_confidence_episode_delta": 0,
            "average_retained_pass_rate_delta": 0.02,
        }
    )
    calm_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.01,
            "max_low_confidence_episode_delta": 0,
        }
    )
    for _ in range(5):
        state, _ = update_controller_state(
            state,
            start_observation=pressured_start,
            action_policy=policy,
            end_observation=pressured_end,
        )
        state, _ = update_controller_state(
            state,
            start_observation=calm_start,
            action_policy=policy,
            end_observation=calm_end,
        )

    pressured_prediction = predict_next_observation(state, pressured_start, policy)
    calm_prediction = predict_next_observation(state, calm_start, policy)

    assert pressured_prediction["features"]["pass_rate_delta"] > calm_prediction["features"]["pass_rate_delta"]
    assert pressured_prediction["features"]["low_confidence_pressure"] < calm_prediction["features"]["low_confidence_pressure"]


def test_plan_next_policy_prefers_learned_high_value_action():
    state = default_controller_state(exploration_bonus=0.0)
    strong_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    weak_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    neutral = build_round_observation(campaign_signal={})
    strong_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.06,
            "average_retained_step_delta": -0.25,
        },
        liftoff_signal={"state": "shadow_only", "allow_kernel_autobuild": True},
    )
    weak_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "average_retained_pass_rate_delta": -0.03,
            "average_retained_step_delta": 1.0,
            "failed_decisions": 1,
            "worst_family_delta": -0.1,
        },
        liftoff_signal={"state": "reject", "allow_kernel_autobuild": False},
    )
    for _ in range(3):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=strong_policy,
            end_observation=strong_end,
        )
    for _ in range(2):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=weak_policy,
            end_observation=weak_end,
        )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[weak_policy, strong_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    assert diagnostics["selected_action_key"] == action_key_for_policy(strong_policy)


def test_plan_next_policy_penalizes_high_uncertainty_action():
    state = default_controller_state(exploration_bonus=0.0, uncertainty_penalty=2.0)
    stable_policy = {
        "focus": "balanced",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    noisy_policy = {
        "focus": "recovery_alignment",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    neutral = build_round_observation(campaign_signal={})
    stable_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.05,
            "average_retained_step_delta": -0.1,
        }
    )
    noisy_good = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.12,
            "average_retained_step_delta": -0.1,
        }
    )
    noisy_bad = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "average_retained_pass_rate_delta": -0.08,
            "average_retained_step_delta": 1.5,
            "failed_decisions": 1,
        }
    )
    for _ in range(4):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=stable_policy,
            end_observation=stable_end,
        )
    state, _ = update_controller_state(
        state,
        start_observation=neutral,
        action_policy=noisy_policy,
        end_observation=noisy_good,
    )
    state, _ = update_controller_state(
        state,
        start_observation=neutral,
        action_policy=noisy_policy,
        end_observation=noisy_bad,
    )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[stable_policy, noisy_policy],
    )

    assert selected["focus"] == "balanced"
    noisy_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(noisy_policy))
    stable_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(stable_policy))
    assert noisy_diag["reward_stddev"] > stable_diag["reward_stddev"]
    assert stable_diag["score"] > noisy_diag["score"]


def test_build_round_observation_tracks_priority_family_yield_feedback():
    observation = build_round_observation(
        campaign_signal={
            "priority_families_with_retained_gain": ["project"],
            "priority_families_without_signal": ["integration"],
            "priority_families_with_signal_but_no_retained_gain": ["repository"],
        }
    )

    assert observation["features"]["priority_family_retained_gain"] == 1.0 / 3.0
    assert observation["features"]["priority_family_yield_gap"] == 2.0 / 3.0

    productive_observation = build_round_observation(
        campaign_signal={"priority_families_with_retained_gain": ["project", "repository"]}
    )
    stalled_observation = build_round_observation(
        campaign_signal={
            "priority_families_without_signal": ["project"],
            "priority_families_with_signal_but_no_retained_gain": ["repository"],
        }
    )

    assert observation_reward(productive_observation) > observation_reward(stalled_observation)


def test_build_round_observation_tracks_generalization_pressure():
    broad = build_round_observation(
        campaign_signal={
            "priority_families": ["repository", "workflow", "integration"],
            "priority_families_with_retained_gain": ["repository", "workflow"],
            "priority_families_with_signal_but_no_retained_gain": ["integration"],
        }
    )
    narrow = build_round_observation(
        campaign_signal={
            "priority_families": ["repository", "workflow", "integration"],
            "priority_families_with_retained_gain": ["repository"],
            "priority_families_with_signal_but_no_retained_gain": ["workflow"],
            "priority_families_without_signal": ["integration"],
        }
    )

    assert broad["features"]["generalization_gain"] > narrow["features"]["generalization_gain"]
    assert broad["features"]["generalization_gap"] < narrow["features"]["generalization_gap"]
    assert observation_reward(broad) > observation_reward(narrow)


def test_build_round_observation_tracks_no_retained_gain_pressure():
    pressured = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 2,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_without_signal": ["project", "repository", "integration"],
        }
    )
    recovered = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_with_retained_gain": ["project"],
        }
    )

    assert pressured["features"]["no_retained_gain_pressure"] == 1.0
    assert pressured["features"]["priority_family_yield_gap"] == 1.0
    assert observation_reward(recovered) > observation_reward(pressured)


def test_build_round_observation_tracks_retained_gain_conversion_pressure_after_first_win():
    converting = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_with_retained_gain": ["project"],
            "priority_families_needing_retained_gain_conversion": ["repository", "integration"],
        }
    )
    settled = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_with_retained_gain": ["project", "repository", "integration"],
        }
    )

    assert converting["features"]["no_retained_gain_pressure"] > 0.0
    assert converting["features"]["retained_gain_conversion_pressure"] == 2.0 / 3.0
    assert observation_reward(settled) > observation_reward(converting)


def test_build_round_observation_penalizes_broad_observe_then_retrieval_first():
    pressured = build_round_observation(
        campaign_signal={
            "priority_families": ["project", "repository", "integration", "repo_chore"],
            "priority_families_with_retained_gain": ["project"],
            "priority_families_needing_retained_gain_conversion": ["repository", "integration", "repo_chore"],
        },
        round_signal={
            "broad_observe_then_retrieval_first": True,
            "live_decision_credit_gap": True,
        },
    )
    settled = build_round_observation(
        campaign_signal={
            "priority_families": ["project", "repository", "integration", "repo_chore"],
            "priority_families_with_retained_gain": ["project", "repository", "integration", "repo_chore"],
        },
    )

    assert pressured["features"]["broad_observe_then_retrieval_first"] == 1.0
    assert pressured["features"]["live_decision_credit_gap"] == 1.0
    assert observation_reward(settled) > observation_reward(pressured)


def test_build_round_observation_tracks_observability_and_robustness_pressure():
    pressured = build_round_observation(
        campaign_signal={
            "partial_productive_without_decision_runs": 1,
            "incomplete_cycle_count": 1,
            "failed_decisions": 1,
            "recent_failed_run": {"run_index": 2, "returncode": 1},
            "recent_failed_decision": {"cycle_id": "cycle:2", "subsystem": "retrieval"},
        },
        round_signal={
            "semantic_progress_drift": True,
            "live_decision_credit_gap": True,
        },
    )
    settled = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
        },
    )

    assert pressured["features"]["observability_pressure"] > 0.0
    assert pressured["features"]["subsystem_robustness_pressure"] > 0.0
    assert observation_reward(settled) > observation_reward(pressured)


def test_build_round_observation_tracks_frontier_motif_and_repo_setting_pressure():
    pressured = build_round_observation(
        campaign_signal={
            "priority_families_with_retained_gain": ["project"],
        },
        planner_pressure_signal={
            "frontier_failure_motif_families": ["workflow", "integration"],
            "frontier_repo_setting_families": ["integration"],
        },
    )
    productive = build_round_observation(
        campaign_signal={
            "priority_families_with_retained_gain": ["project", "workflow", "integration"],
        },
        planner_pressure_signal={
            "frontier_failure_motif_families": ["workflow", "integration"],
            "frontier_repo_setting_families": ["integration"],
        },
    )

    assert pressured["features"]["frontier_failure_motif_pressure"] == 2.0 / 3.0
    assert pressured["features"]["frontier_repo_setting_pressure"] == 1.0 / 3.0
    assert productive["features"]["frontier_failure_motif_gain"] == 2.0 / 3.0
    assert productive["features"]["frontier_repo_setting_gain"] == 1.0 / 3.0
    assert observation_reward(productive) > observation_reward(pressured)


def test_build_round_observation_carries_discovered_structural_classes_and_gap_signal():
    pressured = build_round_observation(
        campaign_signal={
            "priority_families": ["integration", "repository"],
            "priority_families_with_retained_gain": ["repository"],
            "priority_families_with_signal_but_no_retained_gain": ["integration"],
        },
        round_signal={
            "repo_semantics": ["repository", "shared_repo"],
            "artifact_kinds": ["source", "test"],
            "workflow_shape": "multi_branch_validation",
            "contract_shape": "semantic_verifier",
            "workflow_guard": {"shared_repo_id": "repo:shared"},
            "semantic_verifier": {"required_merged_branches": ["lane/repo-generalization"]},
        },
    )
    aligned = build_round_observation(
        campaign_signal={
            "priority_families": ["integration", "repository"],
            "priority_families_with_retained_gain": ["integration", "repository"],
        },
        round_signal={
            "repo_semantics": ["repository", "shared_repo"],
            "artifact_kinds": ["source", "test"],
            "workflow_shape": "multi_branch_validation",
            "contract_shape": "semantic_verifier",
            "workflow_guard": {"shared_repo_id": "repo:shared"},
            "semantic_verifier": {"required_merged_branches": ["lane/repo-generalization"]},
        },
    )

    assert {entry["class_kind"] for entry in pressured["discovered_structural_classes"]} == {
        "repo_topology",
        "verifier_shape",
        "edit_pattern",
        "workflow_shape",
    }
    assert pressured["discovered_structural_family_aliases"] == ["repository", "integration"]
    assert pressured["features"]["structural_class_coverage"] == 1.0
    assert pressured["features"]["structural_class_alignment"] == 1.0 / 3.0
    assert pressured["features"]["structural_class_gap"] == 1.0 / 3.0
    assert observation_reward(aligned) > observation_reward(pressured)


def test_decoder_action_generation_falls_back_to_discovered_structural_family_templates():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="repo_generalization_decoder_task",
            prompt="Repair the repository config.",
            workspace_subdir="repo_generalization_decoder_task",
            expected_files=["configs/app.yaml"],
            expected_file_contents={"configs/app.yaml": "version: 2\n"},
            metadata={
                "benchmark_family": "repo_sandbox",
                "repo_semantics": ["repository"],
                "artifact_kinds": ["config"],
                "workflow_shape": "single_workspace",
            },
        )
    )
    state.world_model_summary = {
        "missing_expected_artifacts": ["configs/app.yaml"],
        "unsatisfied_expected_contents": ["configs/app.yaml"],
    }

    candidates = decoder_module.decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {
                "repository": [
                    {
                        "template_kind": "expected_file_content",
                        "support": 9,
                        "provenance": ["policy:repository"],
                    }
                ]
            },
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    candidate = next(item for item in candidates if item.proposal_source == "expected_file_content")
    assert candidate.proposal_metadata["support"] == 9
    assert candidate.proposal_metadata["routing_benchmark_families"] == ["repo_sandbox", "repository"]
    assert candidate.proposal_metadata["provenance"] == ["policy:repository"]


def test_run_job_queue_structural_summary_and_rollup_surface_discovered_classes():
    structural_summary = run_job_queue_module._job_structural_summary(
        {
            "benchmark_family": "repo_sandbox",
            "repo_semantics": ["repository", "shared_repo"],
            "artifact_kinds": ["source", "test"],
            "workflow_shape": "multi_branch_validation",
            "contract_shape": "semantic_verifier",
            "workflow_guard": {"shared_repo_id": "repo:shared"},
            "semantic_verifier": {"required_merged_branches": ["lane/repo-generalization"]},
        }
    )

    assert set(structural_summary["discovered_structural_class_kinds"]) == {
        "repo_topology",
        "verifier_shape",
        "edit_pattern",
        "workflow_shape",
    }
    assert structural_summary["discovered_structural_family_aliases"] == ["repository", "integration"]

    rollups = run_job_queue_module._queue_structural_class_rollups(
        [
            (
                SimpleNamespace(job_id="job-1"),
                {
                    "benchmark_family": "repo_sandbox",
                    "runnable": True,
                    "blocked": False,
                    "promotable": False,
                    **structural_summary,
                },
            )
        ]
    )

    assert "repo_topology:repository+shared_repo" in rollups
    assert rollups["repo_topology:repository+shared_repo"]["benchmark_families"] == ["repo_sandbox"]
    assert rollups["repo_topology:repository+shared_repo"]["family_aliases"] == ["integration", "repository"]


def test_observation_reward_penalizes_subsystem_reject_pressure():
    reject_only = build_round_observation(
        campaign_signal={"retained_cycles": 0, "rejected_cycles": 1},
        subsystem_signal={"rejected_by_subsystem": {"retrieval": 4}, "retained_by_subsystem": {}},
    )
    mixed = build_round_observation(
        campaign_signal={"retained_cycles": 1, "rejected_cycles": 0},
        subsystem_signal={"rejected_by_subsystem": {"retrieval": 1}, "retained_by_subsystem": {"repository": 2}},
    )

    assert reject_only["features"]["subsystem_reject_pressure"] == 4.0
    assert mixed["features"]["subsystem_retain_pressure"] == 2.0
    assert observation_reward(mixed) > observation_reward(reject_only)


def test_build_round_observation_includes_declared_discovered_controller_features():
    baseline = build_round_observation(campaign_signal={})
    enriched = build_round_observation(
        campaign_signal={
            "controller_features": {
                "novel_subsystem_capability_gain": 0.8,
                "strategy_hook_coverage": 0.5,
            }
        }
    )

    assert enriched["features"]["novel_subsystem_capability_gain"] == 0.8
    assert enriched["features"]["strategy_hook_coverage"] == 0.5
    assert observation_reward(enriched) > observation_reward(baseline)


def test_build_round_observation_scores_intermediate_non_runtime_decisions_as_positive_signal():
    no_credit = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 0,
            "intermediate_decision_evidence": False,
        },
        subsystem_signal={"rejected_by_subsystem": {"retrieval": 1}, "retained_by_subsystem": {}},
    )
    intermediate_credit = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 2,
            "non_runtime_managed_runs": 2,
            "intermediate_decision_evidence": True,
        },
        subsystem_signal={"rejected_by_subsystem": {"retrieval": 1}, "retained_by_subsystem": {}},
    )

    assert no_credit["features"]["intermediate_decision_gain"] == 0.0
    assert intermediate_credit["features"]["intermediate_decision_gain"] == 2.0 / 3.0
    assert observation_reward(intermediate_credit) > observation_reward(no_credit)


def test_build_failure_observation_preserves_supplemental_live_signal():
    supplemental = build_round_observation(
        campaign_signal={
            "non_runtime_managed_decisions": 2,
            "non_runtime_managed_runs": 2,
            "intermediate_decision_evidence": True,
            "partial_productive_without_decision_runs": 1,
        },
        round_signal={
            "broad_observe_then_retrieval_first": True,
            "live_decision_credit_gap": True,
            "semantic_progress_drift": True,
        },
    )

    failure = build_failure_observation(
        phase="campaign",
        reason="controller intervention after repeated verification loop",
        subsystem="retrieval",
        supplemental_observation=supplemental,
    )

    assert failure["features"]["intermediate_decision_gain"] == 2.0 / 3.0
    assert failure["features"]["live_decision_credit_gap"] == 1.0
    assert failure["features"]["observability_pressure"] > 0.0
    assert failure["features"]["subsystem_robustness_pressure"] > 0.0
    assert failure["features"]["rejected_cycles"] == 1.0


def test_update_controller_state_preserves_discovered_feature_transitions():
    state = default_controller_state(exploration_bonus=0.0)
    policy = {
        "focus": "balanced",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    start = build_round_observation(
        campaign_signal={"controller_features": {"novel_subsystem_capability_gain": 0.1}}
    )
    end = build_round_observation(
        campaign_signal={"controller_features": {"novel_subsystem_capability_gain": 0.7}}
    )

    updated_state, diagnostics = update_controller_state(
        state,
        start_observation=start,
        action_policy=policy,
        end_observation=end,
    )
    predicted = predict_next_observation(updated_state, start, policy)

    action_key = action_key_for_policy(policy)
    assert diagnostics["reward"] > 0.0
    assert (
        updated_state["action_models"][action_key]["transition_mean"]["novel_subsystem_capability_gain"] > 0.0
    )
    assert predicted["features"]["novel_subsystem_capability_gain"] > start["features"]["novel_subsystem_capability_gain"]


def test_external_subsystem_specs_preserve_capability_tags_and_strategy_hooks(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "capability_tags": ["github_read", "issue_triage"],
                                    "strategy_hooks": {
                                        "selection_bias": "prefer_repo_issue_pressure",
                                        "reward_signal": "issue_resolution_delta"
                                    }
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = external_subsystem_specs(modules_path)

    assert specs["github_policy"].base_subsystem == "policy"
    assert specs["github_policy"].capability_tags == ("github_read", "issue_triage")
    assert specs["github_policy"].strategy_hooks == {
        "selection_bias": "prefer_repo_issue_pressure",
        "reward_signal": "issue_resolution_delta",
    }


def test_plan_next_policy_generalizes_to_unseen_related_action():
    state = default_controller_state(exploration_bonus=0.0, uncertainty_penalty=1.0)
    trained_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    nearby_unseen_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 3,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    unrelated_unseen_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    neutral = build_round_observation(campaign_signal={})
    strong_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.07,
            "average_retained_step_delta": -0.2,
        },
        liftoff_signal={"state": "shadow_only", "allow_kernel_autobuild": True},
    )
    for _ in range(4):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=trained_policy,
            end_observation=strong_end,
        )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[nearby_unseen_policy, unrelated_unseen_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    nearby_diag = next(
        item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(nearby_unseen_policy)
    )
    unrelated_diag = next(
        item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(unrelated_unseen_policy)
    )
    assert nearby_diag["policy_prior"] > unrelated_diag["policy_prior"]


def test_plan_next_policy_uses_multi_step_rollout_horizon():
    prep_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    exploit_policy = {
        "focus": "recovery_alignment",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    greedy_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        rollout_depth=1,
        rollout_beam_width=2,
        repeat_action_penalty=5.0,
    )
    state["value_weights"] = {"retained_cycles": 1.0, "pass_rate_delta": 5.0}
    state["action_models"] = {
        action_key_for_policy(prep_policy): {
            "count": 10,
            "reward_mean": 0.5,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 1.0, "pass_rate_delta": 1.0},
            "transition_error_ema": {"retained_cycles": 0.0, "pass_rate_delta": 0.0},
            "policy_template": dict(prep_policy),
            "last_reward": 0.5,
        },
        action_key_for_policy(exploit_policy): {
            "count": 10,
            "reward_mean": 4.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 0.0, "pass_rate_delta": 0.0},
            "transition_error_ema": {"retained_cycles": 0.0, "pass_rate_delta": 0.0},
            "policy_template": dict(exploit_policy),
            "last_reward": 4.0,
        },
        action_key_for_policy(greedy_policy): {
            "count": 10,
            "reward_mean": 6.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 0.0, "pass_rate_delta": 0.0},
            "transition_error_ema": {"retained_cycles": 0.0, "pass_rate_delta": 0.0},
            "policy_template": dict(greedy_policy),
            "last_reward": 6.0,
        },
    }
    neutral = build_round_observation(campaign_signal={})

    depth_one_selected, _ = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[prep_policy, exploit_policy, greedy_policy],
    )

    state["rollout_depth"] = 2
    depth_two_selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[prep_policy, exploit_policy, greedy_policy],
    )

    assert depth_one_selected["focus"] == "balanced"
    assert depth_two_selected["focus"] == "discovered_task_adaptation"
    prep_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(prep_policy))
    assert prep_diag["future_action_key"] == action_key_for_policy(greedy_policy)


def test_plan_next_policy_penalizes_repeating_last_action_at_root():
    repeated_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    alternate_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    repeated_key = action_key_for_policy(repeated_policy)
    alternate_key = action_key_for_policy(alternate_policy)
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        rollout_depth=1,
        repeat_action_penalty=3.0,
    )
    state["last_action_key"] = repeated_key
    state["action_models"] = {
        repeated_key: {
            "count": 10,
            "reward_mean": 5.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(repeated_policy),
            "last_reward": 5.0,
        },
        alternate_key: {
            "count": 10,
            "reward_mean": 4.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(alternate_policy),
            "last_reward": 4.0,
        },
    }
    neutral = build_round_observation(campaign_signal={})

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[repeated_policy, alternate_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    assert diagnostics["last_action_key"] == repeated_key
    repeated_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == repeated_key)
    alternate_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == alternate_key)
    assert repeated_diag["repeat_penalty"] == 3.0
    assert alternate_diag["repeat_penalty"] == 0.0
    assert alternate_diag["score"] > repeated_diag["score"]


def test_update_controller_state_tracks_recent_state_feature_memory():
    state = default_controller_state(exploration_bonus=0.0)
    policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    start = build_round_observation(campaign_signal={})
    end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.04,
            "average_retained_step_delta": -0.25,
        }
    )

    state, _ = update_controller_state(
        state,
        start_observation=start,
        action_policy=policy,
        end_observation=end,
    )

    assert len(state["recent_state_features"]) == 1
    snapshot = state["recent_state_features"][0]
    assert snapshot["retained_cycles"] == 1.0
    assert snapshot["pass_rate_delta"] == 0.04
    assert snapshot["step_delta"] == -0.25


def test_plan_next_policy_uses_state_repeat_and_novelty_terms_beyond_action_memory():
    repeated_state_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    novel_state_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    repeated_key = action_key_for_policy(repeated_state_policy)
    novel_key = action_key_for_policy(novel_state_policy)
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        rollout_depth=1,
        repeat_action_penalty=0.0,
        state_repeat_penalty=3.0,
        state_novelty_bonus=2.0,
    )
    neutral = build_round_observation(campaign_signal={})
    neutral_snapshot = {
        key: float(value)
        for key, value in neutral["features"].items()
        if key != "bias"
    }
    state["updates"] = 2
    state["recent_state_features"] = [dict(neutral_snapshot), dict(neutral_snapshot)]
    state["action_models"] = {
        repeated_key: {
            "count": 10,
            "reward_mean": 1.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(repeated_state_policy),
            "last_reward": 1.0,
        },
        novel_key: {
            "count": 10,
            "reward_mean": 1.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 1.0, "pass_rate_delta": 0.4},
            "transition_error_ema": {},
            "policy_template": dict(novel_state_policy),
            "last_reward": 1.0,
        },
    }

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[repeated_state_policy, novel_state_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    repeated_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == repeated_key)
    novel_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == novel_key)
    assert repeated_diag["repeat_penalty"] == 0.0
    assert repeated_diag["state_repeat_penalty"] > novel_diag["state_repeat_penalty"]
    assert novel_diag["state_novelty_bonus"] > repeated_diag["state_novelty_bonus"]
    assert novel_diag["score"] > repeated_diag["score"]


def test_plan_next_policy_penalizes_thin_evidence_exploitation():
    thin_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    proven_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    thin_key = action_key_for_policy(thin_policy)
    proven_key = action_key_for_policy(proven_policy)
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        min_action_support=3,
        thin_evidence_penalty=2.0,
        rollout_depth=1,
    )
    state["action_models"] = {
        thin_key: {
            "count": 1,
            "reward_mean": 8.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 2.0, "pass_rate_delta": 0.2},
            "transition_error_ema": {},
            "policy_template": dict(thin_policy),
            "last_reward": 8.0,
        },
        proven_key: {
            "count": 4,
            "reward_mean": 6.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 0.5, "pass_rate_delta": 0.05},
            "transition_error_ema": {},
            "policy_template": dict(proven_policy),
            "last_reward": 6.0,
        },
    }
    neutral = build_round_observation(campaign_signal={})

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[thin_policy, proven_policy],
    )

    assert selected["focus"] == "balanced"
    thin_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == thin_key)
    proven_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == proven_key)
    assert (1.0 / 3.0) < thin_diag["support_confidence"] < 1.0
    assert thin_diag["support_penalty"] > 0.0
    assert thin_diag["supported_reward_lcb"] < thin_diag["reward_lcb"]
    assert proven_diag["support_confidence"] == 1.0
    assert proven_diag["support_penalty"] == 0.0
    assert proven_diag["score"] > thin_diag["score"]


def test_controller_state_summary_reports_support_controls():
    summary = controller_state_summary(
        default_controller_state(
            min_action_support=4,
            thin_evidence_penalty=1.25,
            support_confidence_power=0.6,
        )
    )

    assert summary["min_action_support"] == 4
    assert summary["thin_evidence_penalty"] == 1.25
    assert summary["support_confidence_power"] == 0.6


def test_plan_next_policy_graduates_nearly_supported_actions_faster():
    nearly_supported_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
    }
    proven_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
    }
    nearly_supported_key = action_key_for_policy(nearly_supported_policy)
    proven_key = action_key_for_policy(proven_policy)
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        min_action_support=3,
        thin_evidence_penalty=2.0,
        support_confidence_power=0.5,
        rollout_depth=1,
    )
    state["action_models"] = {
        nearly_supported_key: {
            "count": 2,
            "reward_mean": 8.5,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 1.5, "pass_rate_delta": 0.15},
            "transition_error_ema": {},
            "policy_template": dict(nearly_supported_policy),
            "last_reward": 8.5,
        },
        proven_key: {
            "count": 4,
            "reward_mean": 6.0,
            "reward_m2": 0.0,
            "transition_mean": {"retained_cycles": 0.5, "pass_rate_delta": 0.05},
            "transition_error_ema": {},
            "policy_template": dict(proven_policy),
            "last_reward": 6.0,
        },
    }

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=build_round_observation(campaign_signal={}),
        candidate_policies=[nearly_supported_policy, proven_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    nearly_supported_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == nearly_supported_key)
    proven_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == proven_key)
    assert nearly_supported_diag["count"] == 2
    assert nearly_supported_diag["support_confidence"] > (2.0 / 3.0)
    assert nearly_supported_diag["support_penalty"] > 0.0
    assert nearly_supported_diag["score"] > proven_diag["score"]


def test_observation_reward_prefers_productive_depth_over_depth_drift():
    productive = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.05,
            "average_retained_step_delta": 3.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 3.0,
            "long_horizon_retained_cycles": 1,
        }
    )
    drifting = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.01,
            "average_retained_step_delta": 3.0,
            "depth_drift_cycles": 1,
            "average_depth_drift_step_delta": 3.0,
        }
    )

    assert productive["features"]["productive_depth_gain"] > 0.0
    assert productive["features"]["depth_drift_pressure"] == 0.0
    assert observation_reward(productive) > observation_reward(drifting)


def test_plan_next_policy_can_learn_task_step_floor_as_policy_dimension():
    state = default_controller_state(exploration_bonus=0.0)
    deeper_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 128,
        "task_step_floor": 120,
    }
    shallow_policy = {
        **deeper_policy,
        "task_step_floor": 40,
    }
    neutral = build_round_observation(campaign_signal={})
    productive_depth_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.06,
            "average_retained_step_delta": 4.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 4.0,
            "long_horizon_retained_cycles": 1,
        }
    )
    for _ in range(3):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=deeper_policy,
            end_observation=productive_depth_end,
        )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[shallow_policy, deeper_policy],
    )

    assert selected["task_step_floor"] == 120
    deep_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(deeper_policy))
    shallow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(shallow_policy))
    assert deep_diag["policy_prior"] > shallow_diag["policy_prior"]


def test_plan_next_policy_prefers_broader_generalization_yield():
    state = default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0)
    broad_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    narrow_policy = {
        "focus": "balanced",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    neutral = build_round_observation(campaign_signal={})
    broad_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.05,
            "priority_families": ["repository", "workflow", "integration"],
            "priority_families_with_retained_gain": ["repository", "workflow"],
            "priority_families_with_signal_but_no_retained_gain": ["integration"],
        }
    )
    narrow_end = build_round_observation(
        campaign_signal={
            "retained_cycles": 1,
            "average_retained_pass_rate_delta": 0.05,
            "priority_families": ["repository", "workflow", "integration"],
            "priority_families_with_retained_gain": ["repository"],
            "priority_families_with_signal_but_no_retained_gain": ["workflow"],
            "priority_families_without_signal": ["integration"],
        }
    )
    for _ in range(3):
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=broad_policy,
            end_observation=broad_end,
        )
        state, _ = update_controller_state(
            state,
            start_observation=neutral,
            action_policy=narrow_policy,
            end_observation=narrow_end,
        )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=neutral,
        candidate_policies=[narrow_policy, broad_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    broad_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(broad_policy))
    narrow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(narrow_policy))
    assert broad_diag["score"] > narrow_diag["score"]


def test_plan_next_policy_prefers_broader_policy_under_subsystem_reject_pressure():
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        repeat_action_penalty=0.0,
    )
    narrow_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project"],
    }
    broad_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 3,
        "campaign_width": 4,
        "variant_width": 3,
        "task_limit": 256,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
    }
    pressured = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 2,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_without_signal": ["project", "repository"],
            "priority_families_with_signal_but_no_retained_gain": ["integration"],
            "max_low_confidence_episode_delta": 2,
            "min_trusted_retrieval_step_delta": -1,
        },
        subsystem_signal={
            "rejected_by_subsystem": {"retrieval": 4},
            "retained_by_subsystem": {},
        },
        planner_pressure_signal={
            "campaign_breadth_pressure_cycles": 2,
            "variant_breadth_pressure_cycles": 1,
        },
    )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=pressured,
        candidate_policies=[narrow_policy, broad_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    broad_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(broad_policy))
    narrow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == action_key_for_policy(narrow_policy))
    assert broad_diag["pressure_alignment_bonus"] > narrow_diag["pressure_alignment_bonus"]
    assert broad_diag["score"] > narrow_diag["score"]


def test_plan_next_policy_prefers_broader_policy_when_no_retained_gain_pressure_is_present():
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        repeat_action_penalty=0.0,
    )
    narrow_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project"],
    }
    broad_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 3,
        "campaign_width": 4,
        "variant_width": 3,
        "task_limit": 256,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
    }
    narrow_key = action_key_for_policy(narrow_policy)
    broad_key = action_key_for_policy(broad_policy)
    state["action_models"] = {
        narrow_key: {
            "count": 6,
            "reward_mean": 7.5,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(narrow_policy),
            "last_reward": 7.5,
        },
        broad_key: {
            "count": 6,
            "reward_mean": 4.5,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(broad_policy),
            "last_reward": 4.5,
        },
    }
    pressured = build_round_observation(
        campaign_signal={
            "retained_cycles": 0,
            "rejected_cycles": 2,
            "priority_families": ["project", "repository", "integration"],
            "priority_families_without_signal": ["project", "repository", "integration"],
        }
    )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=pressured,
        candidate_policies=[narrow_policy, broad_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    broad_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == broad_key)
    narrow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == narrow_key)
    assert broad_diag["pressure_alignment_bonus"] > narrow_diag["pressure_alignment_bonus"]


def test_plan_next_policy_prefers_robust_observable_policy_under_runtime_pressure():
    state = default_controller_state(exploration_bonus=0.0)
    narrow_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project"],
    }
    broad_policy = {
        "focus": "recovery_alignment",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
    }
    narrow_key = action_key_for_policy(narrow_policy)
    broad_key = action_key_for_policy(broad_policy)
    state["action_models"] = {
        narrow_key: {
            "count": 4,
            "reward_mean": 6.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(narrow_policy),
            "last_reward": 6.0,
        },
        broad_key: {
            "count": 4,
            "reward_mean": 6.0,
            "reward_m2": 0.0,
            "transition_mean": {},
            "transition_error_ema": {},
            "policy_template": dict(broad_policy),
            "last_reward": 6.0,
        },
    }
    pressured = build_round_observation(
        campaign_signal={
            "partial_productive_without_decision_runs": 1,
            "incomplete_cycle_count": 1,
            "failed_decisions": 1,
            "recent_failed_run": {"run_index": 2, "returncode": 1},
            "recent_failed_decision": {"cycle_id": "cycle:2", "subsystem": "retrieval"},
        },
        round_signal={
            "semantic_progress_drift": True,
            "live_decision_credit_gap": True,
        },
    )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=pressured,
        candidate_policies=[narrow_policy, broad_policy],
    )

    assert selected["focus"] == "recovery_alignment"
    broad_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == broad_key)
    narrow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == narrow_key)
    assert broad_diag["pressure_alignment_bonus"] > narrow_diag["pressure_alignment_bonus"]
    assert broad_diag["score"] > narrow_diag["score"]


def test_plan_next_policy_prefers_broader_policy_when_broad_observe_retrieval_first_pressure_is_present():
    state = default_controller_state(
        exploration_bonus=0.0,
        uncertainty_penalty=0.0,
        repeat_action_penalty=0.0,
    )
    narrow_policy = {
        "focus": "balanced",
        "adaptive_search": False,
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project"],
    }
    broad_policy = {
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 3,
        "campaign_width": 4,
        "variant_width": 3,
        "task_limit": 256,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
    }
    narrow_key = action_key_for_policy(narrow_policy)
    broad_key = action_key_for_policy(broad_policy)
    state["action_models"] = {
        narrow_key: {"count": 4, "reward_mean": 7.0, "reward_m2": 0.0, "transition_mean": {}, "transition_error_ema": {}, "policy_template": dict(narrow_policy), "last_reward": 7.0},
        broad_key: {"count": 4, "reward_mean": 7.0, "reward_m2": 0.0, "transition_mean": {}, "transition_error_ema": {}, "policy_template": dict(broad_policy), "last_reward": 7.0},
    }
    pressured = build_round_observation(
        campaign_signal={
            "priority_families": ["project", "repository", "integration", "repo_chore"],
            "priority_families_with_retained_gain": ["project"],
            "priority_families_needing_retained_gain_conversion": ["repository", "integration", "repo_chore"],
        },
        round_signal={
            "broad_observe_then_retrieval_first": True,
            "live_decision_credit_gap": True,
        },
    )

    selected, diagnostics = plan_next_policy(
        state,
        current_observation=pressured,
        candidate_policies=[narrow_policy, broad_policy],
    )

    assert selected["focus"] == "discovered_task_adaptation"
    broad_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == broad_key)
    narrow_diag = next(item for item in diagnostics["candidates"] if item["action_key"] == narrow_key)
    assert broad_diag["pressure_alignment_bonus"] > narrow_diag["pressure_alignment_bonus"]
