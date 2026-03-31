from agent_kernel.unattended_controller import (
    action_key_for_policy,
    build_round_observation,
    default_controller_state,
    observation_reward,
    plan_next_policy,
    predict_next_observation,
    update_controller_state,
)


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
