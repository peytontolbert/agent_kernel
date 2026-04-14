from __future__ import annotations

import json
from typing import Any

from ..config import KernelConfig
from .improvement.improvement_common import overlay_control_mapping
from .improvement.operator_policy_improvement import retained_operator_policy_controls


def operator_policy_snapshot(config: KernelConfig) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "unattended_allowed_benchmark_families": [
            str(value).strip() for value in config.unattended_allowed_benchmark_families if str(value).strip()
        ],
        "unattended_allow_git_commands": bool(config.unattended_allow_git_commands),
        "unattended_allow_http_requests": bool(config.unattended_allow_http_requests),
        "unattended_http_allowed_hosts": [
            str(value).strip().lower() for value in config.unattended_http_allowed_hosts if str(value).strip()
        ],
        "unattended_http_timeout_seconds": max(1, int(config.unattended_http_timeout_seconds)),
        "unattended_http_max_body_bytes": max(1, int(config.unattended_http_max_body_bytes)),
        "unattended_allow_generated_path_mutations": bool(config.unattended_allow_generated_path_mutations),
        "unattended_generated_path_prefixes": [
            str(value).strip() for value in config.unattended_generated_path_prefixes if str(value).strip()
        ],
    }
    retained = _retained_operator_policy_controls(config)
    return overlay_control_mapping(
        policy,
        retained,
        bool_fields=(
            "unattended_allow_git_commands",
            "unattended_allow_http_requests",
            "unattended_allow_generated_path_mutations",
        ),
        positive_int_fields=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        list_fields=("unattended_allowed_benchmark_families", "unattended_generated_path_prefixes"),
        lowercase_list_fields=("unattended_http_allowed_hosts",),
    )


def _retained_operator_policy_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(config.use_operator_policy_proposals):
        return {}
    path = config.operator_policy_proposals_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_operator_policy_controls(payload)
