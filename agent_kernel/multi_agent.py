from __future__ import annotations

from .state import AgentState


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
        if not verification_passed:
            return "critic"
        if state.consecutive_no_progress_steps > 0:
            return "planner"
        if state.active_subgoal:
            return "executor"
        return "planner"
