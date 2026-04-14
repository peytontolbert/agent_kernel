from __future__ import annotations

from typing import Any


def _retrieval_reuse_records(payload: dict[str, object]) -> list[dict[str, object]]:
    for key in ("procedures", "skills", "candidates"):
        records = payload.get(key, [])
        if isinstance(records, list) and records:
            return [record for record in records if isinstance(record, dict)]
    return []


def artifact_retrieval_reuse_evidence(payload: dict[str, object] | None, *, subsystem: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    procedures = _retrieval_reuse_records(payload)
    if not procedures:
        return {}
    procedure_count = 0
    retrieval_backed_procedure_count = 0
    trusted_retrieval_procedure_count = 0
    retrieval_selected_step_count = 0
    retrieval_influenced_step_count = 0
    trusted_retrieval_step_count = 0
    selected_retrieval_span_ids: set[str] = set()
    verified_retrieval_commands: set[str] = set()
    for record in procedures:
        if not isinstance(record, dict):
            continue
        if subsystem == "retrieval" and str(record.get("subsystem", "")).strip() not in {"", "retrieval"}:
            continue
        procedure_count += 1
        retrieval_selected_steps = int(record.get("retrieval_selected_steps", 0) or 0)
        retrieval_influenced_steps = int(record.get("retrieval_influenced_steps", 0) or 0)
        trusted_retrieval_steps = int(record.get("trusted_retrieval_steps", 0) or 0)
        retrieval_backed = bool(record.get("retrieval_backed", False)) or bool(
            retrieval_selected_steps > 0
            or retrieval_influenced_steps > 0
            or trusted_retrieval_steps > 0
            or any(str(value).strip() for value in record.get("retrieval_backed_commands", []))
        )
        if retrieval_backed:
            retrieval_backed_procedure_count += 1
        if trusted_retrieval_steps > 0:
            trusted_retrieval_procedure_count += 1
        retrieval_selected_step_count += retrieval_selected_steps
        retrieval_influenced_step_count += retrieval_influenced_steps
        trusted_retrieval_step_count += trusted_retrieval_steps
        selected_retrieval_span_ids.update(
            str(value).strip()
            for value in record.get("selected_retrieval_span_ids", [])
            if str(value).strip()
        )
        verified_retrieval_commands.update(
            str(value).strip()
            for value in record.get("retrieval_backed_commands", [])
            if str(value).strip()
        )
    if procedure_count <= 0:
        return {}
    return {
        "procedure_count": procedure_count,
        "retrieval_backed_procedure_count": retrieval_backed_procedure_count,
        "trusted_retrieval_procedure_count": trusted_retrieval_procedure_count,
        "retrieval_selected_step_count": retrieval_selected_step_count,
        "retrieval_influenced_step_count": retrieval_influenced_step_count,
        "trusted_retrieval_step_count": trusted_retrieval_step_count,
        "selected_retrieval_span_count": len(selected_retrieval_span_ids),
        "verified_retrieval_command_count": len(verified_retrieval_commands),
    }


def artifact_retrieval_reuse_comparison(
    payload: dict[str, object] | None,
    *,
    subsystem: str,
    active_artifact_payload_from_generation_context_fn,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    baseline_payload = active_artifact_payload_from_generation_context_fn(payload)
    candidate_summary = artifact_retrieval_reuse_evidence(payload, subsystem=subsystem)
    baseline_summary = artifact_retrieval_reuse_evidence(baseline_payload, subsystem=subsystem)
    if not candidate_summary and not baseline_summary:
        return {}
    keys = (
        "procedure_count",
        "retrieval_backed_procedure_count",
        "trusted_retrieval_procedure_count",
        "retrieval_selected_step_count",
        "retrieval_influenced_step_count",
        "trusted_retrieval_step_count",
        "selected_retrieval_span_count",
        "verified_retrieval_command_count",
    )
    comparison: dict[str, object] = {}
    for key in keys:
        baseline_value = int(baseline_summary.get(key, 0) or 0)
        candidate_value = int(candidate_summary.get(key, 0) or 0)
        comparison[f"baseline_{key}"] = baseline_value
        comparison[f"candidate_{key}"] = candidate_value
        comparison[f"{key}_delta"] = candidate_value - baseline_value
    comparison["retrieval_reuse_delta"] = int(
        comparison.get("retrieval_backed_procedure_count_delta", 0) or 0
    ) + int(comparison.get("trusted_retrieval_procedure_count_delta", 0) or 0)
    return comparison


def tool_shared_repo_bundle_evidence(payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return {}
    shared_repo_candidate_count = 0
    shared_repo_worker_candidate_count = 0
    shared_repo_complete_integrator_candidate_count = 0
    shared_repo_incomplete_integrator_candidate_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bundle = tool_candidate_shared_repo_bundle(candidate)
        if not bundle:
            continue
        shared_repo_candidate_count += 1
        role = str(bundle.get("role", "")).strip()
        if role == "worker":
            shared_repo_worker_candidate_count += 1
            continue
        if bool(bundle.get("bundle_complete", False)):
            shared_repo_complete_integrator_candidate_count += 1
        else:
            shared_repo_incomplete_integrator_candidate_count += 1
    return {
        "shared_repo_candidate_count": shared_repo_candidate_count,
        "shared_repo_worker_candidate_count": shared_repo_worker_candidate_count,
        "shared_repo_complete_integrator_candidate_count": shared_repo_complete_integrator_candidate_count,
        "shared_repo_incomplete_integrator_candidate_count": shared_repo_incomplete_integrator_candidate_count,
        "shared_repo_complete_candidate_count": (
            shared_repo_worker_candidate_count + shared_repo_complete_integrator_candidate_count
        ),
    }


def tool_shared_repo_bundle_comparison(
    payload: dict[str, object] | None,
    *,
    active_artifact_payload_from_generation_context_fn,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    baseline_payload = active_artifact_payload_from_generation_context_fn(payload)
    candidate_summary = tool_shared_repo_bundle_evidence(payload)
    baseline_summary = tool_shared_repo_bundle_evidence(baseline_payload)
    if not candidate_summary and not baseline_summary:
        return {}
    keys = (
        "shared_repo_candidate_count",
        "shared_repo_worker_candidate_count",
        "shared_repo_complete_integrator_candidate_count",
        "shared_repo_incomplete_integrator_candidate_count",
        "shared_repo_complete_candidate_count",
    )
    comparison: dict[str, object] = {}
    for key in keys:
        baseline_value = int(baseline_summary.get(key, 0) or 0)
        candidate_value = int(candidate_summary.get(key, 0) or 0)
        comparison[f"baseline_{key}"] = baseline_value
        comparison[f"candidate_{key}"] = candidate_value
        comparison[f"{key}_delta"] = candidate_value - baseline_value
    comparison["candidate_bundle_coherence_delta"] = int(
        comparison.get("shared_repo_complete_candidate_count_delta", 0) or 0
    ) - int(comparison.get("shared_repo_incomplete_integrator_candidate_count_delta", 0) or 0)
    return comparison


def tool_candidate_shared_repo_bundle(candidate: dict[str, object]) -> dict[str, object]:
    if not isinstance(candidate, dict):
        return {}
    procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
    commands = [str(command).strip() for command in procedure.get("commands", []) if str(command).strip()]
    bundle = dict(candidate.get("shared_repo_bundle", {})) if isinstance(candidate.get("shared_repo_bundle", {}), dict) else {}
    task_contract = dict(candidate.get("task_contract", {})) if isinstance(candidate.get("task_contract", {}), dict) else {}
    metadata = dict(task_contract.get("metadata", {})) if isinstance(task_contract.get("metadata", {}), dict) else {}
    workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
    verifier = dict(metadata.get("semantic_verifier", {})) if isinstance(metadata.get("semantic_verifier", {}), dict) else {}
    repo_id = str(bundle.get("shared_repo_id", workflow_guard.get("shared_repo_id", ""))).strip()
    worker_branch = str(bundle.get("worker_branch", workflow_guard.get("worker_branch", ""))).strip()
    role = str(bundle.get("role", "")).strip()
    try:
        shared_repo_order = int(bundle.get("shared_repo_order", metadata.get("shared_repo_order", 0)) or 0)
    except (TypeError, ValueError):
        shared_repo_order = 0
    required_merged_branches = [
        str(value).strip()
        for value in (
            bundle.get("required_merged_branches", verifier.get("required_merged_branches", []))
            if isinstance(bundle.get("required_merged_branches", verifier.get("required_merged_branches", [])), list)
            else verifier.get("required_merged_branches", [])
        )
        if str(value).strip()
    ]
    observed_merged_branches = [
        branch for branch in required_merged_branches if any(branch in command for command in commands)
    ]
    if not role:
        if shared_repo_order > 0 or required_merged_branches:
            role = "integrator"
        elif repo_id or worker_branch:
            role = "worker"
    if not role:
        return {}
    bundle_complete = bool(bundle.get("bundle_complete", False))
    if role == "worker":
        bundle_complete = True
    elif required_merged_branches:
        bundle_complete = len(observed_merged_branches) >= len(required_merged_branches)
    return {
        "shared_repo_id": repo_id,
        "worker_branch": worker_branch,
        "role": role,
        "shared_repo_order": shared_repo_order,
        "required_merged_branches": required_merged_branches,
        "observed_merged_branches": observed_merged_branches,
        "bundle_complete": bundle_complete,
    }


def tool_candidate_retention_sort_key(candidate: dict[str, object]) -> tuple[int, float, str]:
    bundle = tool_candidate_shared_repo_bundle(candidate)
    role = str(bundle.get("role", "")).strip()
    if role == "worker":
        priority = 0
    elif role == "integrator" and bool(bundle.get("bundle_complete", False)):
        priority = 1
    elif role == "integrator":
        priority = 3
    else:
        priority = 2
    quality = float(candidate.get("quality", 0.0) or 0.0)
    tool_id = str(candidate.get("tool_id", "")).strip()
    return (priority, -quality, tool_id)


def normalized_tool_candidates_for_retention(candidates: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bundle = tool_candidate_shared_repo_bundle(candidate)
        if str(bundle.get("role", "")).strip() == "integrator" and not bool(bundle.get("bundle_complete", False)):
            continue
        updated = dict(candidate)
        if bundle:
            updated["shared_repo_bundle"] = bundle
        normalized.append(updated)
    return sorted(normalized, key=tool_candidate_retention_sort_key)


def operator_support_count(payload: dict[str, object] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    operators = payload.get("operators", [])
    if not isinstance(operators, list) or not operators:
        return 0
    counts = [
        len([str(value) for value in operator.get("source_task_ids", []) if str(value).strip()])
        for operator in operators
        if isinstance(operator, dict)
    ]
    return min(counts) if counts else 0
