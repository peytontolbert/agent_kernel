from __future__ import annotations

import json

from ...config import KernelConfig
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    normalized_generation_focus,
    normalized_control_mapping,
    overlay_control_mapping,
    retained_mapping_section,
    retention_gate_preset,
)

_DELEGATED_RUNTIME_BASELINE_CONCURRENCY = 3
_DELEGATED_RUNTIME_BASELINE_ACTIVE_PER_BUDGET_GROUP = 2
_DELEGATED_RUNTIME_BASELINE_QUEUED_PER_BUDGET_GROUP = 8
_DELEGATED_RUNTIME_BASELINE_ARTIFACT_BYTES = 8 * 1024 * 1024
_DELEGATED_RUNTIME_BASELINE_SUBPROCESSES_PER_JOB = 2
_DELEGATED_RUNTIME_BASELINE_MAX_STEPS = 12


def delegation_behavior_controls(
    config: KernelConfig,
    *,
    focus: str | None = None,
) -> dict[str, object]:
    current = _current_delegation_controls(config)
    controls: dict[str, object] = {
        "delegated_job_max_concurrency": max(1, int(current["delegated_job_max_concurrency"])),
        "delegated_job_max_active_per_budget_group": max(
            0,
            int(current["delegated_job_max_active_per_budget_group"]),
        ),
        "delegated_job_max_queued_per_budget_group": max(
            0,
            int(current["delegated_job_max_queued_per_budget_group"]),
        ),
        "delegated_job_max_artifact_bytes": max(0, int(current["delegated_job_max_artifact_bytes"])),
        "delegated_job_max_subprocesses_per_job": max(
            1,
            int(current["delegated_job_max_subprocesses_per_job"]),
        ),
        "command_timeout_seconds": max(1, int(current["command_timeout_seconds"])),
        "llm_timeout_seconds": max(1, int(current["llm_timeout_seconds"])),
        "max_steps": max(1, int(current["max_steps"])),
    }
    normalized_focus = (focus or "").strip()
    if normalized_focus == "throughput_balance":
        controls["delegated_job_max_concurrency"] = min(
            8,
            max(
                _DELEGATED_RUNTIME_BASELINE_CONCURRENCY,
                int(controls["delegated_job_max_concurrency"]) + 1,
            ),
        )
        active = int(controls["delegated_job_max_active_per_budget_group"])
        controls["delegated_job_max_active_per_budget_group"] = min(
            4,
            max(
                _DELEGATED_RUNTIME_BASELINE_ACTIVE_PER_BUDGET_GROUP,
                max(1, active) + 1,
            ),
        )
        controls["delegated_job_max_queued_per_budget_group"] = min(
            16,
            max(
                _DELEGATED_RUNTIME_BASELINE_QUEUED_PER_BUDGET_GROUP,
                int(controls["delegated_job_max_queued_per_budget_group"]) + 2,
            ),
        )
    elif normalized_focus == "queue_elasticity":
        queued = int(controls["delegated_job_max_queued_per_budget_group"])
        controls["delegated_job_max_queued_per_budget_group"] = min(
            32,
            max(_DELEGATED_RUNTIME_BASELINE_QUEUED_PER_BUDGET_GROUP, queued + 2),
        )
        controls["delegated_job_max_artifact_bytes"] = min(
            50 * 1024 * 1024,
            max(
                _DELEGATED_RUNTIME_BASELINE_ARTIFACT_BYTES,
                int(controls["delegated_job_max_artifact_bytes"]) * 2,
            ),
        )
    elif normalized_focus == "worker_depth":
        controls["delegated_job_max_subprocesses_per_job"] = min(
            8,
            max(
                _DELEGATED_RUNTIME_BASELINE_SUBPROCESSES_PER_JOB,
                int(controls["delegated_job_max_subprocesses_per_job"]) + 1,
            ),
        )
        controls["command_timeout_seconds"] = min(300, int(controls["command_timeout_seconds"]) + 10)
        controls["llm_timeout_seconds"] = min(300, int(controls["llm_timeout_seconds"]) + 10)
        controls["max_steps"] = _scaled_worker_depth_max_steps(
            max(_DELEGATED_RUNTIME_BASELINE_MAX_STEPS, int(controls["max_steps"])),
            config=config,
        )
    return controls


def build_delegation_proposal_artifact(
    config: KernelConfig,
    *,
    focus: str | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    controls = delegation_behavior_controls(
        config,
        focus=None if generation_focus == "balanced" else generation_focus,
    )
    return build_standard_proposal_artifact(
        artifact_kind="delegated_runtime_policy_set",
        generation_focus=generation_focus,
        control_schema="delegated_resource_controls_v1",
        retention_gate=retention_gate_preset("delegation"),
        controls=controls,
        proposals=_proposals(config, generation_focus),
    )


def retained_delegation_controls(payload: object) -> dict[str, object]:
    controls = retained_mapping_section(payload, artifact_kind="delegated_runtime_policy_set", section="controls")
    return normalized_control_mapping(
        controls,
        int_fields=(
            "delegated_job_max_concurrency",
            "delegated_job_max_active_per_budget_group",
            "delegated_job_max_queued_per_budget_group",
            "delegated_job_max_artifact_bytes",
            "delegated_job_max_subprocesses_per_job",
            "command_timeout_seconds",
            "llm_timeout_seconds",
            "max_steps",
        ),
    )


def _proposals(config: KernelConfig, focus: str) -> list[dict[str, object]]:
    current = _current_delegation_controls(config)
    proposals: list[dict[str, object]] = []
    if (
        int(current["delegated_job_max_concurrency"]) < _DELEGATED_RUNTIME_BASELINE_CONCURRENCY
        or int(current["delegated_job_max_active_per_budget_group"])
        < _DELEGATED_RUNTIME_BASELINE_ACTIVE_PER_BUDGET_GROUP
        or focus == "throughput_balance"
    ):
        proposals.append(
            {
                "area": "throughput_balance",
                "priority": 5,
                "reason": "delegated execution remains bottlenecked by low concurrency or narrow active budget-group capacity",
                "suggestion": "Increase concurrent delegated workers while keeping queue-level fairness explicit.",
            }
        )
    if (
        int(current["delegated_job_max_artifact_bytes"]) < _DELEGATED_RUNTIME_BASELINE_ARTIFACT_BYTES
        or int(current["delegated_job_max_queued_per_budget_group"])
        < _DELEGATED_RUNTIME_BASELINE_QUEUED_PER_BUDGET_GROUP
        or focus == "queue_elasticity"
    ):
        proposals.append(
            {
                "area": "queue_elasticity",
                "priority": 4,
                "reason": "delegated runs remain constrained by small artifact budgets or narrow queue elasticity",
                "suggestion": "Broaden delegated artifact and queue budgets so retained campaigns can sustain more parallel work.",
            }
        )
    if (
        int(current["max_steps"]) < _DELEGATED_RUNTIME_BASELINE_MAX_STEPS
        or int(current["delegated_job_max_subprocesses_per_job"])
        < _DELEGATED_RUNTIME_BASELINE_SUBPROCESSES_PER_JOB
        or focus == "worker_depth"
    ):
        proposals.append(
            {
                "area": "worker_depth",
                "priority": 4,
                "reason": "delegated workers still operate under shallow step, subprocess, or timeout ceilings",
                "suggestion": "Increase worker depth so delegated runs can complete longer bounded improvement tasks autonomously.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "throughput_balance",
            "priority": 3,
            "reason": "delegated runtime governance should remain explicit and retained even when current worker policy is acceptable",
            "suggestion": "Preserve delegated resource controls as a retained autonomous runtime surface.",
        },
    )


def _current_delegation_controls(config: KernelConfig) -> dict[str, int]:
    controls: dict[str, int] = {
        "delegated_job_max_concurrency": max(1, int(config.delegated_job_max_concurrency)),
        "delegated_job_max_active_per_budget_group": max(
            0,
            int(config.delegated_job_max_active_per_budget_group),
        ),
        "delegated_job_max_queued_per_budget_group": max(
            0,
            int(config.delegated_job_max_queued_per_budget_group),
        ),
        "delegated_job_max_artifact_bytes": max(0, int(config.delegated_job_max_artifact_bytes)),
        "delegated_job_max_subprocesses_per_job": max(
            1,
            int(config.delegated_job_max_subprocesses_per_job),
        ),
        "command_timeout_seconds": max(1, int(config.command_timeout_seconds)),
        "llm_timeout_seconds": max(1, int(config.llm_timeout_seconds)),
        "max_steps": max(1, int(config.max_steps)),
    }
    if not bool(config.use_delegation_proposals):
        return controls
    path = config.delegation_proposals_path
    if not path.exists():
        return controls
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return controls
    return overlay_control_mapping(
        controls,
        retained_delegation_controls(payload),
        int_fields=(
            "delegated_job_max_concurrency",
            "delegated_job_max_active_per_budget_group",
            "delegated_job_max_queued_per_budget_group",
            "delegated_job_max_artifact_bytes",
            "delegated_job_max_subprocesses_per_job",
            "command_timeout_seconds",
            "llm_timeout_seconds",
            "max_steps",
        ),
    )


def _scaled_worker_depth_max_steps(current_max_steps: int, *, config: KernelConfig) -> int:
    hard_cap = max(1, int(config.max_task_steps_hard_cap))
    frontier_floor = min(hard_cap, max(1, int(config.frontier_task_step_floor)))
    grown_max_steps = max(current_max_steps + 2, current_max_steps * 2)
    if current_max_steps < frontier_floor:
        grown_max_steps = max(grown_max_steps, frontier_floor)
    return min(hard_cap, max(1, grown_max_steps))
