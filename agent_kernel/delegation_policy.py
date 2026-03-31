from __future__ import annotations

import json

from .config import KernelConfig
from .delegation_improvement import retained_delegation_controls


def delegation_policy_snapshot(config: KernelConfig) -> dict[str, int]:
    policy: dict[str, int] = {
        "delegated_job_max_concurrency": max(1, int(config.delegated_job_max_concurrency)),
        "delegated_job_max_active_per_budget_group": max(0, int(config.delegated_job_max_active_per_budget_group)),
        "delegated_job_max_queued_per_budget_group": max(0, int(config.delegated_job_max_queued_per_budget_group)),
        "delegated_job_max_artifact_bytes": max(0, int(config.delegated_job_max_artifact_bytes)),
        "delegated_job_max_subprocesses_per_job": max(1, int(config.delegated_job_max_subprocesses_per_job)),
        "delegated_job_max_consecutive_selections_per_budget_group": max(
            0, int(config.delegated_job_max_consecutive_selections_per_budget_group)
        ),
        "command_timeout_seconds": max(1, int(config.command_timeout_seconds)),
        "llm_timeout_seconds": max(1, int(config.llm_timeout_seconds)),
        "max_steps": max(1, int(config.max_steps)),
    }
    retained = _retained_delegation_controls(config)
    for key in policy:
        if key not in retained:
            continue
        try:
            value = int(retained[key])
        except (TypeError, ValueError):
            continue
        if key in {
            "delegated_job_max_active_per_budget_group",
            "delegated_job_max_queued_per_budget_group",
            "delegated_job_max_artifact_bytes",
            "delegated_job_max_consecutive_selections_per_budget_group",
        }:
            policy[key] = max(0, value)
        else:
            policy[key] = max(1, value)
    return policy


def _retained_delegation_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(config.use_delegation_proposals):
        return {}
    path = config.delegation_proposals_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_delegation_controls(payload)
