"""Actor contracts for bounded kernel-controlled execution roles."""

from .coding import (
    coding_actor_applicable,
    coding_actor_episode_summary,
    CodingActorMode,
    CodingActorOutcome,
    CodingActorPlan,
    CodingActorPolicy,
    CodingActorResult,
    coding_actor_plan_for_task,
    coding_actor_kernel_summary,
    default_coding_actor_policy,
)

__all__ = [
    "coding_actor_applicable",
    "coding_actor_episode_summary",
    "CodingActorMode",
    "CodingActorOutcome",
    "CodingActorPlan",
    "CodingActorPolicy",
    "CodingActorResult",
    "coding_actor_plan_for_task",
    "coding_actor_kernel_summary",
    "default_coding_actor_policy",
]
