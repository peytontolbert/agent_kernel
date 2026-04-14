from __future__ import annotations

from typing import Any


def compact_graph_summary(policy: Any, graph_summary: dict[str, object]) -> dict[str, object]:
    if not graph_summary:
        return {}

    def _compact_count_mapping(key: str, *, limit: int) -> dict[str, int]:
        values = graph_summary.get(key, {})
        if not isinstance(values, dict):
            return {}
        compact = {
            str(name).strip(): policy._safe_int(count, 0)
            for name, count in values.items()
            if str(name).strip() and policy._safe_int(count, 0) > 0
        }
        return dict(list(sorted(compact.items(), key=lambda item: (-item[1], item[0])))[:limit])

    summary: dict[str, object] = {}
    if "document_count" in graph_summary:
        summary["document_count"] = graph_summary["document_count"]
    if "benchmark_families" in graph_summary:
        families = graph_summary.get("benchmark_families", {})
        if isinstance(families, dict):
            summary["benchmark_families"] = dict(list(families.items())[:6])
    if "failure_types" in graph_summary:
        failures = graph_summary.get("failure_types", {})
        if isinstance(failures, dict):
            summary["failure_types"] = dict(list(failures.items())[:6])
    elif "failure_type_counts" in graph_summary:
        failures = graph_summary.get("failure_type_counts", {})
        if isinstance(failures, dict):
            summary["failure_types"] = dict(list(failures.items())[:6])
    if "related_tasks" in graph_summary:
        related = graph_summary.get("related_tasks", [])
        if isinstance(related, list):
            summary["related_tasks"] = [policy._truncate_text(str(item), limit=80) for item in related[:6]]
    elif "neighbors" in graph_summary:
        related = graph_summary.get("neighbors", [])
        if isinstance(related, list):
            summary["related_tasks"] = [policy._truncate_text(str(item), limit=80) for item in related[:6]]
    for key in (
        "retrieval_backed_successes",
        "retrieval_influenced_successes",
        "trusted_retrieval_successes",
    ):
        if key in graph_summary:
            summary[key] = graph_summary[key]
    observed_modes = graph_summary.get("observed_environment_modes", {})
    if isinstance(observed_modes, dict):
        compact_modes: dict[str, object] = {}
        for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
            value = dominant_graph_environment_mode(policy, graph_summary, field)
            if value:
                compact_modes[field] = value
        if compact_modes:
            summary["observed_environment_modes"] = compact_modes
    alignment_failures = graph_summary.get("environment_alignment_failures", {})
    if isinstance(alignment_failures, dict) and alignment_failures:
        summary["environment_alignment_failures"] = {
            str(key): int(value)
            for key, value in list(
                sorted(
                    (
                        (str(key).strip(), policy._safe_int(value, 0))
                        for key, value in alignment_failures.items()
                        if str(key).strip()
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            )[:3]
            if value > 0
        }
    trusted_commands = graph_summary.get("trusted_retrieval_command_counts", {})
    if isinstance(trusted_commands, dict):
        summary["trusted_retrieval_command_counts"] = dict(list(trusted_commands.items())[:4])
    for key, limit in (
        ("verifier_obligation_counts", 4),
        ("changed_path_counts", 4),
        ("edit_patch_path_counts", 4),
        ("recovery_command_counts", 3),
    ):
        compact = _compact_count_mapping(key, limit=limit)
        if compact:
            summary[key] = compact
    semantic_episodes = graph_summary.get("semantic_episodes", [])
    if isinstance(semantic_episodes, list):
        compact_semantic_episodes: list[dict[str, object]] = []
        for item in semantic_episodes[:2]:
            if not isinstance(item, dict):
                continue
            compact_semantic_episodes.append(
                {
                    "task_id": policy._truncate_text(str(item.get("task_id", "")).strip(), limit=80),
                    "benchmark_family": policy._truncate_text(
                        str(item.get("benchmark_family", "bounded")).strip() or "bounded",
                        limit=40,
                    ),
                    "memory_source": policy._truncate_text(str(item.get("memory_source", "")).strip(), limit=40),
                    "success": bool(item.get("success", False)),
                    "verifier_obligations": [
                        policy._truncate_text(str(value).strip(), limit=120)
                        for value in item.get("verifier_obligations", [])
                        if str(value).strip()
                    ][:4],
                    "changed_paths": [
                        policy._truncate_text(str(value).strip(), limit=100)
                        for value in item.get("changed_paths", [])
                        if str(value).strip()
                    ][:4],
                    "edit_patches": [
                        {
                            "path": policy._truncate_text(str(dict(value).get("path", "")).strip(), limit=100),
                            "status": policy._truncate_text(str(dict(value).get("status", "")).strip(), limit=20),
                            "patch_summary": policy._truncate_text(
                                str(dict(value).get("patch_summary", "")).strip(),
                                limit=120,
                            ),
                            "patch_excerpt": policy._truncate_text(
                                str(dict(value).get("patch_excerpt", dict(value).get("patch", ""))).strip(),
                                limit=160,
                            ),
                        }
                        for value in item.get("edit_patches", [])[:2]
                        if isinstance(value, dict) and str(dict(value).get("path", "")).strip()
                    ],
                    "recovery_trace": {
                        "failed_command": policy._truncate_text(
                            str(dict(item.get("recovery_trace", {})).get("failed_command", "")).strip(),
                            limit=120,
                        ),
                        "recovery_command": policy._truncate_text(
                            str(dict(item.get("recovery_trace", {})).get("recovery_command", "")).strip(),
                            limit=120,
                        ),
                        "failure_signals": [
                            policy._truncate_text(str(value).strip(), limit=60)
                            for value in dict(item.get("recovery_trace", {})).get("failure_signals", [])
                            if str(value).strip()
                        ][:3],
                    }
                    if isinstance(item.get("recovery_trace", {}), dict)
                    else {},
                }
            )
        if compact_semantic_episodes:
            summary["semantic_episodes"] = compact_semantic_episodes
    trusted_procedures = graph_summary.get("trusted_retrieval_procedures", {})
    if isinstance(trusted_procedures, list):
        summary["trusted_retrieval_procedures"] = [
            {
                "commands": [str(value) for value in dict(item).get("commands", [])[:4] if str(value).strip()],
                "count": policy._safe_int(dict(item).get("count", 0), 0),
            }
            for item in trusted_procedures[:2]
            if isinstance(item, dict)
        ]
    return summary


def historical_environment_novelty(policy_or_cls: Any, graph_summary: dict[str, object], universe_summary: dict[str, object]) -> int:
    snapshot = universe_summary.get("environment_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    novelty = 0
    for field in ("network_access_mode", "git_write_mode", "workspace_write_scope"):
        current = str(snapshot.get(field, "")).strip().lower()
        dominant = dominant_graph_environment_mode(policy_or_cls, graph_summary, field)
        if current and dominant and current != dominant:
            novelty += 1
    return novelty


def graph_environment_alignment_failures(policy_or_cls: Any, graph_summary: dict[str, object]) -> dict[str, int]:
    failures = graph_summary.get("environment_alignment_failures", {})
    if not isinstance(failures, dict):
        return {}
    return {
        str(key).strip(): value
        for key, value in (
            (str(key).strip(), policy_or_cls._safe_int(raw_value, 0))
            for key, raw_value in failures.items()
        )
        if key and value > 0
    }


def dominant_graph_environment_mode(policy_or_cls: Any, graph_summary: dict[str, object], field: str) -> str:
    observed_modes = graph_summary.get("observed_environment_modes", {})
    if not isinstance(observed_modes, dict):
        return ""
    values = observed_modes.get(field, {})
    if not isinstance(values, dict):
        return ""
    ranked = sorted(
        (
            (str(mode).strip().lower(), policy_or_cls._safe_int(count, 0))
            for mode, count in values.items()
            if str(mode).strip()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0] if ranked and ranked[0][1] > 0 else ""


def compact_world_model_summary(policy: Any, world_model_summary: dict[str, object]) -> dict[str, object]:
    if not world_model_summary:
        return {}
    summary: dict[str, object] = {}
    for key in ("benchmark_family", "horizon", "completion_ratio"):
        if key in world_model_summary:
            summary[key] = world_model_summary[key]
    for key in ("expected_artifacts", "forbidden_artifacts", "preserved_artifacts"):
        values = world_model_summary.get(key, [])
        if isinstance(values, list):
            summary[key] = [policy._truncate_text(str(item), limit=80) for item in values[:6]]
    for key in (
        "missing_expected_artifacts",
        "unsatisfied_expected_contents",
        "present_forbidden_artifacts",
        "changed_preserved_artifacts",
        "intact_preserved_artifacts",
        "workflow_report_paths",
        "workflow_generated_paths",
        "workflow_required_merges",
        "workflow_branch_targets",
        "workflow_required_tests",
        "updated_workflow_paths",
        "updated_generated_paths",
        "updated_report_paths",
    ):
        values = world_model_summary.get(key, [])
        if isinstance(values, list) and values:
            summary[key] = [policy._truncate_text(str(item), limit=80) for item in values[:6]]
    previews = world_model_summary.get("workspace_file_previews", {})
    if isinstance(previews, dict) and previews:
        summary["workspace_file_previews"] = {
            str(path): compact_workspace_preview(policy, preview)
            for path, preview in list(previews.items())[:4]
            if isinstance(preview, dict) and str(path).strip()
        }
    return summary


def compact_workspace_preview(policy: Any, preview: dict[str, object]) -> dict[str, object]:
    windows = preview.get("edit_windows", [])
    if isinstance(windows, list) and windows:
        compact_windows = [
            {
                "content": truncate_preview_content(window.get("content", "")),
                "truncated": bool(window.get("truncated", False)),
                "line_start": int(window.get("line_start", 1) or 1),
                "line_end": int(window.get("line_end", 1) or 1),
            }
            for window in windows[:3]
            if isinstance(window, dict)
        ]
        if compact_windows:
            compact = dict(compact_windows[0])
            compact["edit_windows"] = compact_windows
            return compact
    return {
        "content": truncate_preview_content(preview.get("content", "")),
        "truncated": bool(preview.get("truncated", False)),
    }


def truncate_preview_content(value: object) -> str:
    content = str(value)
    return content[:157] + "..." if len(content) > 160 else content


def compact_universe_summary(policy: Any, universe_summary: dict[str, object]) -> dict[str, object]:
    if not universe_summary:
        return {}
    summary: dict[str, object] = {}
    for key in ("universe_id", "stability", "governance_mode"):
        if key in universe_summary:
            summary[key] = universe_summary[key]
    for key in ("requires_verification", "requires_bounded_steps", "prefer_reversible_actions"):
        if key in universe_summary:
            summary[key] = bool(universe_summary[key])
    for key in ("invariants", "forbidden_command_patterns", "preferred_command_prefixes"):
        values = universe_summary.get(key, [])
        if isinstance(values, list) and values:
            summary[key] = [policy._truncate_text(str(item), limit=80) for item in values[:6]]
    action_risk_controls = universe_summary.get("action_risk_controls", {})
    if isinstance(action_risk_controls, dict) and action_risk_controls:
        summary["action_risk_controls"] = {
            str(key): int(value)
            for key, value in action_risk_controls.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    for key in (
        "environment_assumptions",
        "environment_alignment",
        "envelope_alignment",
        "constitutional_compliance",
        "runtime_attestation",
        "plan_risk_summary",
    ):
        value = universe_summary.get(key, {})
        if isinstance(value, dict) and value:
            summary[key] = {
                str(item_key): item_value
                for item_key, item_value in list(value.items())[:8]
            }
    autonomy_scope = universe_summary.get("autonomy_scope", {})
    if isinstance(autonomy_scope, dict) and autonomy_scope:
        summary["autonomy_scope"] = {
            str(key): policy._truncate_text(str(value), limit=80)
            for key, value in list(autonomy_scope.items())[:8]
        }
    return summary
