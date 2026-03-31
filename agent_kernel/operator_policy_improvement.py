from __future__ import annotations

import json

from .config import KernelConfig
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    merged_string_lists,
    normalized_generation_focus,
    normalized_control_mapping,
    normalized_string_list,
    overlay_control_mapping,
    retained_mapping_section,
    retention_gate_preset,
)


_DEFAULT_FAMILIES = [
    "bounded",
    "micro",
    "workflow",
    "project",
    "repository",
    "repo_chore",
    "repo_sandbox",
    "tooling",
    "integration",
]
_DEFAULT_GENERATED_PREFIXES = ["build", "dist", "generated", "reports", "tmp"]


def operator_policy_controls(
    config: KernelConfig,
    *,
    focus: str | None = None,
) -> dict[str, object]:
    current = _current_operator_policy_controls(config)
    controls: dict[str, object] = {
        "unattended_allowed_benchmark_families": normalized_string_list(
            current.get("unattended_allowed_benchmark_families", [])
        ),
        "unattended_allow_git_commands": bool(current.get("unattended_allow_git_commands", False)),
        "unattended_allow_http_requests": bool(current.get("unattended_allow_http_requests", False)),
        "unattended_http_allowed_hosts": normalized_string_list(
            current.get("unattended_http_allowed_hosts", []),
            lowercase=True,
        ),
        "unattended_http_timeout_seconds": max(
            1,
            int(current.get("unattended_http_timeout_seconds", 1)),
        ),
        "unattended_http_max_body_bytes": max(
            1,
            int(current.get("unattended_http_max_body_bytes", 1)),
        ),
        "unattended_allow_generated_path_mutations": bool(
            current.get("unattended_allow_generated_path_mutations", False)
        ),
        "unattended_generated_path_prefixes": normalized_string_list(
            current.get("unattended_generated_path_prefixes", [])
        ),
    }
    normalized_focus = (focus or "").strip()
    if normalized_focus == "family_breadth":
        controls["unattended_allowed_benchmark_families"] = merged_string_lists(
            controls["unattended_allowed_benchmark_families"],
            _DEFAULT_FAMILIES,
        )
    elif normalized_focus == "git_http_scope":
        controls["unattended_allow_git_commands"] = True
        controls["unattended_allow_http_requests"] = True
        controls["unattended_http_allowed_hosts"] = merged_string_lists(
            controls["unattended_http_allowed_hosts"],
            ["api.github.com"],
            lowercase=True,
        )
        controls["unattended_http_timeout_seconds"] = max(int(controls["unattended_http_timeout_seconds"]), 15)
        controls["unattended_http_max_body_bytes"] = max(
            int(controls["unattended_http_max_body_bytes"]),
            128 * 1024,
        )
    elif normalized_focus == "generated_path_scope":
        controls["unattended_allow_generated_path_mutations"] = True
        controls["unattended_generated_path_prefixes"] = merged_string_lists(
            controls["unattended_generated_path_prefixes"],
            _DEFAULT_GENERATED_PREFIXES,
        )
    return controls


def build_operator_policy_proposal_artifact(
    config: KernelConfig,
    *,
    focus: str | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    controls = operator_policy_controls(
        config,
        focus=None if generation_focus == "balanced" else generation_focus,
    )
    return build_standard_proposal_artifact(
        artifact_kind="operator_policy_set",
        generation_focus=generation_focus,
        control_schema="unattended_operator_controls_v1",
        retention_gate=retention_gate_preset("operator_policy"),
        controls=controls,
        proposals=_proposals(config, generation_focus),
    )


def retained_operator_policy_controls(payload: object) -> dict[str, object]:
    controls = retained_mapping_section(payload, artifact_kind="operator_policy_set", section="controls")
    return normalized_control_mapping(
        controls,
        bool_fields=(
            "unattended_allow_git_commands",
            "unattended_allow_http_requests",
            "unattended_allow_generated_path_mutations",
        ),
        positive_int_fields=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        list_fields=("unattended_allowed_benchmark_families", "unattended_generated_path_prefixes"),
        lowercase_list_fields=("unattended_http_allowed_hosts",),
    )


def _proposals(config: KernelConfig, focus: str) -> list[dict[str, object]]:
    current = _current_operator_policy_controls(config)
    proposals: list[dict[str, object]] = []
    families = set(normalized_string_list(current.get("unattended_allowed_benchmark_families", [])))
    if len(families) < len(_DEFAULT_FAMILIES) or focus == "family_breadth":
        proposals.append(
            {
                "area": "family_breadth",
                "priority": 5,
                "reason": "unattended execution is still bounded to a narrow benchmark-family subset",
                "suggestion": "Expand unattended benchmark-family coverage so autonomous improvement can operate across more task families.",
            }
        )
    if (
        not bool(current.get("unattended_allow_git_commands", False))
        or not bool(current.get("unattended_allow_http_requests", False))
        or focus == "git_http_scope"
    ):
        proposals.append(
            {
                "area": "git_http_scope",
                "priority": 4,
                "reason": "git or http capabilities remain disabled or under-scoped for unattended execution",
                "suggestion": "Enable bounded git/http execution and pin an explicit allowed-host set for unattended runs.",
            }
        )
    if not bool(current.get("unattended_allow_generated_path_mutations", False)) or focus == "generated_path_scope":
        proposals.append(
            {
                "area": "generated_path_scope",
                "priority": 4,
                "reason": "generated-path mutation policy still blocks a class of unattended repair and build tasks",
                "suggestion": "Enable bounded generated-path mutations with an explicit retained path-prefix allowlist.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "family_breadth",
            "priority": 3,
            "reason": "operator-boundary policy should remain explicit and retained even when current unattended policy is acceptable",
            "suggestion": "Preserve unattended operator-boundary controls as a retained runtime surface.",
        },
    )
def _current_operator_policy_controls(config: KernelConfig) -> dict[str, object]:
    controls: dict[str, object] = {
        "unattended_allowed_benchmark_families": normalized_string_list(config.unattended_allowed_benchmark_families),
        "unattended_allow_git_commands": bool(config.unattended_allow_git_commands),
        "unattended_allow_http_requests": bool(config.unattended_allow_http_requests),
        "unattended_http_allowed_hosts": normalized_string_list(
            config.unattended_http_allowed_hosts,
            lowercase=True,
        ),
        "unattended_http_timeout_seconds": max(1, int(config.unattended_http_timeout_seconds)),
        "unattended_http_max_body_bytes": max(1, int(config.unattended_http_max_body_bytes)),
        "unattended_allow_generated_path_mutations": bool(config.unattended_allow_generated_path_mutations),
        "unattended_generated_path_prefixes": normalized_string_list(config.unattended_generated_path_prefixes),
    }
    if not bool(config.use_operator_policy_proposals):
        return controls
    path = config.operator_policy_proposals_path
    if not path.exists():
        return controls
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return controls
    return overlay_control_mapping(
        controls,
        retained_operator_policy_controls(payload),
        bool_fields=(
            "unattended_allow_git_commands",
            "unattended_allow_http_requests",
            "unattended_allow_generated_path_mutations",
        ),
        positive_int_fields=("unattended_http_timeout_seconds", "unattended_http_max_body_bytes"),
        list_fields=("unattended_allowed_benchmark_families", "unattended_generated_path_prefixes"),
        lowercase_list_fields=("unattended_http_allowed_hosts",),
    )
