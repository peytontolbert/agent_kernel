"""Minimal verifier-driven coding agent kernel."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentKernel",
    "CodingActorMode",
    "CodingActorOutcome",
    "CodingActorPlan",
    "CodingActorPolicy",
    "CodingActorResult",
    "coding_actor_applicable",
    "coding_actor_episode_summary",
    "coding_actor_plan_for_task",
    "DelegatedJob",
    "DelegatedJobLease",
    "DelegatedJobQueue",
    "DelegatedResourcePolicy",
    "DelegatedRuntimeController",
    "KernelConfig",
    "TaskBank",
    "coding_actor_kernel_summary",
    "coding_actor_applicable",
    "coding_actor_episode_summary",
    "coding_actor_plan_for_task",
    "default_coding_actor_policy",
    "drain_delegated_jobs",
    "run_next_delegated_job",
]

_MODULE_MAP = {
    "KernelConfig": ".config",
    "AgentKernel": ".loop",
    "TaskBank": ".tasking.task_bank",
    "CodingActorMode": ".actors",
    "CodingActorOutcome": ".actors",
    "CodingActorPlan": ".actors",
    "CodingActorPolicy": ".actors",
    "CodingActorResult": ".actors",
    "coding_actor_applicable": ".actors",
    "coding_actor_episode_summary": ".actors",
    "coding_actor_plan_for_task": ".actors",
    "coding_actor_kernel_summary": ".actors",
    "default_coding_actor_policy": ".actors",
    "DelegatedJob": ".ops.job_queue",
    "DelegatedJobLease": ".ops.job_queue",
    "DelegatedJobQueue": ".ops.job_queue",
    "DelegatedResourcePolicy": ".ops.job_queue",
    "DelegatedRuntimeController": ".ops.job_queue",
    "drain_delegated_jobs": ".ops.job_queue",
    "run_next_delegated_job": ".ops.job_queue",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
