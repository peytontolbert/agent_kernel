from agent_kernel.extensions.multi_agent import RoleCoordinator
from agent_kernel.schemas import TaskSpec
from agent_kernel.state import AgentState


def test_role_coordinator_prefers_planner_for_first_long_horizon_recovery_failure():
    coordinator = RoleCoordinator()
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_recovery",
            prompt="recover from a risky long-horizon state",
            workspace_subdir="long_horizon_recovery",
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        ),
        history=[],
    )
    state.world_model_summary = {"horizon": "long_horizon"}
    state.latest_state_transition = {"regressed": False}
    state.latent_state_summary = {
        "learned_world_state": {
            "progress_signal": 0.21,
            "risk_signal": 0.88,
            "world_risk_score": 0.84,
        }
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 1
    state.repeated_action_count = 1
    state.history = [object()]

    assert coordinator.role_before_decision(state) == "planner"
    assert coordinator.role_after_step(state, verification_passed=False) == "planner"


def test_role_coordinator_prefers_critic_after_repeated_long_horizon_recovery_failures():
    coordinator = RoleCoordinator()
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_recovery_repeat",
            prompt="recover from repeated risky long-horizon failure",
            workspace_subdir="long_horizon_recovery_repeat",
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        ),
        history=[],
    )
    state.world_model_summary = {"horizon": "long_horizon"}
    state.latest_state_transition = {"regressed": False}
    state.latent_state_summary = {
        "learned_world_state": {
            "progress_signal": 0.19,
            "risk_signal": 0.91,
            "decoder_world_risk_score": 0.89,
        }
    }
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.history = [object()]

    assert coordinator.role_before_decision(state) == "critic"
    assert coordinator.role_after_step(state, verification_passed=False) == "critic"


def test_role_coordinator_escalates_stalled_long_horizon_recovery_to_critic():
    coordinator = RoleCoordinator()
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_recovery_stall",
            prompt="recover from a stalled risky long-horizon state",
            workspace_subdir="long_horizon_recovery_stall",
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        ),
        history=[],
    )
    state.world_model_summary = {"horizon": "long_horizon"}
    state.latest_state_transition = {"regressed": False}
    state.latent_state_summary = {
        "learned_world_state": {
            "progress_signal": 0.18,
            "risk_signal": 0.86,
            "transition_regression_score": 0.71,
        }
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 1
    state.history = [object()]

    assert coordinator.role_before_decision(state) == "critic"
    assert coordinator.role_after_step(state, verification_passed=False) == "critic"
