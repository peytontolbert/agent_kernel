"""Minimal verifier-driven coding agent kernel."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentKernel",
    "DelegatedJob",
    "DelegatedJobLease",
    "DelegatedJobQueue",
    "DelegatedResourcePolicy",
    "DelegatedRuntimeController",
    "KernelConfig",
    "TaskBank",
    "drain_delegated_jobs",
    "run_next_delegated_job",
]

_MODULE_MAP = {
    "KernelConfig": ".config",
    "AgentKernel": ".loop",
    "TaskBank": ".task_bank",
    "DelegatedJob": ".job_queue",
    "DelegatedJobLease": ".job_queue",
    "DelegatedJobQueue": ".job_queue",
    "DelegatedResourcePolicy": ".job_queue",
    "DelegatedRuntimeController": ".job_queue",
    "drain_delegated_jobs": ".job_queue",
    "run_next_delegated_job": ".job_queue",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
