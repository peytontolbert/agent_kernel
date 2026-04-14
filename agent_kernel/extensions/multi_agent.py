from __future__ import annotations

from ..state import AgentState


class RoleCoordinator:
    def initial_role(self, state: AgentState) -> str:
        if state.plan:
            return "planner"
        return "executor"

    def role_before_decision(self, state: AgentState) -> str:
        if not state.history:
            return self.initial_role(state)
        if bool(state.latest_state_transition.get("regressed", False)):
            return "critic"
        if self._long_horizon_recovery_mode(state):
            if self._long_horizon_recovery_pressure(state) >= 2:
                return "critic"
            if self._long_horizon_recovery_pressure(state) >= 1:
                return "planner"
        if state.consecutive_failures > 0:
            return "critic"
        if state.consecutive_no_progress_steps > 0:
            return "planner"
        if state.active_subgoal:
            return "executor"
        return "planner"

    def role_after_step(self, state: AgentState, *, verification_passed: bool) -> str:
        if bool(state.latest_state_transition.get("regressed", False)):
            return "critic"
        if self._long_horizon_recovery_mode(state):
            if not verification_passed:
                if self._long_horizon_recovery_pressure(state) >= 2:
                    return "critic"
                if self._long_horizon_recovery_pressure(state) >= 1:
                    return "planner"
            if self._long_horizon_recovery_pressure(state) >= 1:
                return "planner"
        if not verification_passed:
            return "critic"
        if state.consecutive_no_progress_steps > 0:
            return "planner"
        if state.active_subgoal:
            return "executor"
        return "planner"

    @staticmethod
    def _long_horizon_recovery_mode(state: AgentState) -> bool:
        horizon = str(
            state.world_model_summary.get(
                "horizon",
                state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
            )
        ).strip()
        if horizon != "long_horizon":
            return False
        learned = state.latent_state_summary.get("learned_world_state", {})
        learned = learned if isinstance(learned, dict) else {}
        learned_progress_signal = max(
            RoleCoordinator._float_value(learned.get("progress_signal"), 0.0),
            RoleCoordinator._float_value(learned.get("world_progress_score"), 0.0),
            RoleCoordinator._float_value(learned.get("decoder_world_progress_score"), 0.0),
            RoleCoordinator._float_value(learned.get("transition_progress_score"), 0.0),
        )
        learned_risk_signal = max(
            RoleCoordinator._float_value(learned.get("risk_signal"), 0.0),
            RoleCoordinator._float_value(learned.get("world_risk_score"), 0.0),
            RoleCoordinator._float_value(learned.get("decoder_world_risk_score"), 0.0),
            RoleCoordinator._float_value(learned.get("transition_regression_score"), 0.0),
        )
        return learned_risk_signal >= 0.55 and learned_risk_signal > learned_progress_signal

    @staticmethod
    def _long_horizon_recovery_pressure(state: AgentState) -> int:
        return max(
            int(state.consecutive_failures > 0),
            max(0, int(state.repeated_action_count) - 1),
            int(state.consecutive_no_progress_steps),
        )

    @staticmethod
    def _float_value(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
