from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import threading
import time

from agent_kernel.actors import (
    coding_actor_applicable,
    coding_actor_episode_summary,
    default_coding_actor_policy,
)
from agent_kernel.extensions.capabilities import capability_enabled, declared_task_capabilities
from agent_kernel.config import KernelConfig
from agent_kernel.tasking.curriculum import CurriculumEngine
from agent_kernel.ops.episode_store import iter_episode_documents
from agent_kernel.loop import AgentKernel
from agent_kernel.extensions.operator_policy import operator_policy_snapshot
from agent_kernel.policy import Policy
from agent_kernel.ops.preflight import capture_workspace_snapshot, write_unattended_task_report
from agent_kernel.ops.runtime_supervision import atomic_write_json
from agent_kernel.schemas import ActionDecision, EpisodeRecord
from agent_kernel.tasking.task_bank import (
    build_shared_transfer_target_maps,
    load_benchmark_candidate_tasks,
    load_discovered_tasks,
    load_operator_replay_tasks,
    load_skill_transfer_tasks,
    load_transition_pressure_tasks,
    TaskBank,
    load_episode_replay_tasks,
    load_skill_replay_tasks,
    load_tool_replay_tasks,
    load_verifier_candidate_tasks,
    load_verifier_replay_tasks,
)

from .metrics import AbstractionComparison, EvalMetrics, SkillComparison, TolbertComparison, TolbertModeComparison

_SCOPED_DIRECTORY_NAMES = {
    "generated_failure",
    "generated_failure_seed",
    "generated_success",
    "tolbert_deterministic_command",
    "tolbert_full",
    "tolbert_path_only",
    "tolbert_retrieval_only",
    "tolbert_skill_ranking",
    "with_operators",
    "with_raw_skill_transfer",
    "with_skills",
    "without_skills",
    "with_tolbert",
    "without_tolbert",
}

_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 0.5
_PROGRESS_TIMELINE_MAX_EVENTS = 64
_LATE_WAVE_ROTATION_FAMILIES = {
    "validation",
    "governance",
    "oversight",
    "assurance",
    "adjudication",
}
_LATE_WAVE_PRIOR_RECENCY_HALF_LIFE_SECONDS = 6.0 * 60.0 * 60.0
_ROLLBACK_STYLE_RECOVERY_STRATEGY_FAMILIES = {
    "rollback_validation",
    "restore_verification",
    "snapshot_integrity",
    "workspace_restore_verification",
}
_SNAPSHOT_STYLE_RECOVERY_STRATEGY_FAMILIES = {
    "snapshot_coverage",
    "verifier_crosscheck",
    "post_success_replay",
}
_RESIDUE_STYLE_RECOVERY_STRATEGY_FAMILIES = {
    "mutation_residue_scan",
    "unexpected_change_audit",
}


def _lineage_phase_bucket_from_depth(depth: int) -> str:
    if depth >= 14:
        return "late"
    if depth >= 10:
        return "mid"
    if depth > 0:
        return "early"
    return ""


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _seed_document_recorded_at(document: dict[str, object]) -> float:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = (
        dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    )
    return max(
        0.0,
        _safe_float(
            metadata.get(
                "observed_recorded_at",
                contract_metadata.get("observed_recorded_at", summary.get("observed_recorded_at", 0.0)),
            ),
            0.0,
        ),
    )


def _seed_document_recency_weight(
    document: dict[str, object],
    *,
    now: float | None = None,
    half_life_seconds: float = _LATE_WAVE_PRIOR_RECENCY_HALF_LIFE_SECONDS,
) -> float:
    recorded_at = _seed_document_recorded_at(document)
    if recorded_at <= 0.0:
        return 1.0
    reference_now = max(recorded_at, float(now if now is not None else time.time()))
    age_seconds = max(0.0, reference_now - recorded_at)
    if half_life_seconds <= 0.0:
        return 1.0
    return 0.5 ** (age_seconds / half_life_seconds)


def _scheduler_state_bucket(
    *,
    success: bool,
    termination_reason: str,
    verified_step_count: int,
    no_state_progress_steps: int,
    state_regression_steps: int,
) -> str:
    normalized_reason = str(termination_reason or "").strip().lower()
    if no_state_progress_steps > 0 or state_regression_steps > 0:
        return "stalled"
    if success:
        return "productive"
    if any(token in normalized_reason for token in ("timeout", "budget", "step_limit")):
        if verified_step_count > 0:
            return "retrying"
        return "stalled"
    if "policy_terminated" in normalized_reason:
        return "retrying" if verified_step_count > 0 else "stalled"
    if "repeated_failed_action" in normalized_reason or "no_state_progress" in normalized_reason:
        return "stalled"
    if verified_step_count > 0:
        return "productive"
    return "stalled"


def _recovery_strategy_curriculum_controls(strategy_family: object) -> dict[str, object]:
    normalized = str(strategy_family or "").strip()
    if not normalized:
        return {}
    controls: dict[str, object] = {
        "recovery_strategy_family": normalized,
        "failure_reference_family_only": True,
    }
    if normalized in (
        _ROLLBACK_STYLE_RECOVERY_STRATEGY_FAMILIES
        | _SNAPSHOT_STYLE_RECOVERY_STRATEGY_FAMILIES
        | _RESIDUE_STYLE_RECOVERY_STRATEGY_FAMILIES
    ):
        controls["preferred_benchmark_family"] = "repository"
    return controls


def _emit_eval_progress(progress_label: str | None, message: str) -> None:
    if not progress_label:
        return
    print(f"[eval:{progress_label}] {message}", file=sys.stderr, flush=True)


def _should_emit_task_progress(index: int, total: int) -> bool:
    return index == 1 or index == total or index % 10 == 0


def _task_progress_label(task) -> str:
    family = "bounded"
    task_origin = ""
    source_task = ""
    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, dict):
        family = str(metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        task_origin = str(metadata.get("task_origin", "")).strip()
        source_task = str(metadata.get("source_task", "")).strip()
    parts = [str(task.task_id).strip(), f"family={family}"]
    if task_origin:
        parts.append(f"task_origin={task_origin}")
    if source_task:
        parts.append(f"source_task={source_task}")
    return " ".join(parts)


def _run_tasks_with_progress(
    tasks: list,
    kernel: AgentKernel,
    *,
    progress_label: str | None,
    phase: str = "",
    report_config: KernelConfig | None = None,
    on_result=None,
    on_task_start=None,
    on_task_progress=None,
    on_task_complete=None,
):
    total = len(tasks)
    results = []
    for index, task in enumerate(tasks, start=1):
        before_workspace_snapshot = None
        if report_config is not None:
            workspace_path = report_config.workspace_root / task.workspace_subdir
            before_workspace_snapshot = capture_workspace_snapshot(workspace_path)
        if on_task_start is not None:
            on_task_start(task, index, total)
        if progress_label and _should_emit_task_progress(index, total):
            if phase:
                _emit_eval_progress(
                    progress_label,
                    f"phase={phase} task {index}/{total} {_task_progress_label(task)}",
                )
            else:
                _emit_eval_progress(progress_label, f"task {index}/{total} {_task_progress_label(task)}")
        def _task_progress(progress_payload: dict[str, object]) -> None:
            if on_task_progress is not None:
                on_task_progress(task, progress_payload, index, total)
            event_name = str(progress_payload.get("event", "")).strip()
            step_stage = str(progress_payload.get("step_stage", "")).strip()
            cognitive_stage = ""
            if event_name in {
                "state_estimated",
                "memory_retrieved",
                "world_model_updated",
                "memory_update_written",
                "critique_reflected",
                "verification_result",
            }:
                cognitive_stage = event_name
            elif step_stage in {
                "context_compile",
                "context_ready",
                "universe_summary",
                "plan_candidates",
                "transition_simulated",
                "payload_build",
            }:
                cognitive_stage = step_stage
            if progress_label and cognitive_stage:
                benchmark_family = str(getattr(task, "metadata", {}).get("benchmark_family", "bounded")).strip() or "bounded"
                parts = [
                    f"phase={phase}" if phase else "phase=primary",
                    f"task {index}/{total}",
                    _task_progress_label(task),
                    f"family={benchmark_family}",
                    f"cognitive_stage={cognitive_stage}",
                ]
                step_index = int(progress_payload.get("step_index", 0) or 0)
                if step_index > 0:
                    parts.append(f"step={step_index}")
                step_subphase = str(progress_payload.get("step_subphase", "")).strip()
                if step_subphase:
                    parts.append(f"subphase={step_subphase}")
                decision_source = str(progress_payload.get("decision_source", "")).strip()
                if decision_source:
                    parts.append(f"decision_source={decision_source}")
                if "verification_passed" in progress_payload:
                    parts.append(
                        f"verification_passed={1 if bool(progress_payload.get('verification_passed', False)) else 0}"
                    )
                _emit_eval_progress(progress_label, " ".join(parts))

        result = _run_kernel_task(
            kernel,
            task,
            progress_callback=_task_progress if on_task_progress is not None else None,
        )
        # Flush completion bookkeeping before any slower report persistence so
        # partial snapshots can reflect a verified task even if later writes stall.
        if on_task_complete is not None:
            on_task_complete(task, result, index, total)
        if on_result is not None:
            on_result(task, result, index, total)
        results.append(result)
        if report_config is not None:
            write_unattended_task_report(
                task=task,
                config=report_config,
                episode=result,
                preflight=None,
                before_workspace_snapshot=before_workspace_snapshot,
            )
    return results


def _run_kernel_task(kernel: AgentKernel, task, *, progress_callback=None):
    actor_policy = default_coding_actor_policy()
    if progress_callback is None:
        episode = kernel.run_task(task)
    else:
        try:
            episode = kernel.run_task(task, progress_callback=progress_callback)
        except TypeError as exc:
            if "progress_callback" not in str(exc):
                raise
            episode = kernel.run_task(task)
    if coding_actor_applicable(task, policy=actor_policy):
        actor_summary = coding_actor_episode_summary(policy=actor_policy, task=task, episode=episode)
        episode.task_metadata = dict(episode.task_metadata or {})
        episode.task_metadata["actor_type"] = "coding"
        episode.task_metadata["actor_summary"] = actor_summary
        episode.task_contract = dict(episode.task_contract or {})
        contract_metadata = dict(episode.task_contract.get("metadata", {}) or {})
        contract_metadata["actor_type"] = "coding"
        contract_metadata["actor_mode"] = str(actor_summary.get("mode", "")).strip()
        episode.task_contract["metadata"] = contract_metadata
    return episode


def _episode_record_from_document(document: dict[str, object]) -> EpisodeRecord | None:
    task_id = str(document.get("task_id", "")).strip()
    prompt = str(document.get("prompt", "")).strip()
    workspace = str(document.get("workspace", "")).strip()
    if not task_id or not prompt:
        return None
    return EpisodeRecord(
        task_id=task_id,
        prompt=prompt,
        workspace=workspace,
        success=bool(document.get("success", False)),
        steps=[],
        task_metadata=dict(document.get("task_metadata", {}))
        if isinstance(document.get("task_metadata", {}), dict)
        else {},
        task_contract=dict(document.get("task_contract", {}))
        if isinstance(document.get("task_contract", {}), dict)
        else {},
        plan=[],
        graph_summary={},
        universe_summary={},
        world_model_summary={},
        termination_reason=str(document.get("termination_reason", "")).strip(),
    )


def _normalized_workspace_root_prefix(workspace_root: Path | None) -> str:
    if workspace_root is None:
        return ""
    try:
        return str(workspace_root.expanduser().resolve())
    except OSError:
        return str(workspace_root.expanduser())


def _workspace_path_under_root(workspace: str, *, workspace_root_prefix: str) -> bool:
    normalized_workspace = str(workspace).strip()
    if not normalized_workspace or not workspace_root_prefix:
        return False
    try:
        normalized_workspace = str(Path(normalized_workspace).expanduser().resolve())
    except OSError:
        normalized_workspace = str(Path(normalized_workspace).expanduser())
    return normalized_workspace.startswith(workspace_root_prefix)


def _load_generated_success_seed_episodes(
    root: Path,
    *,
    workspace_root: Path | None = None,
) -> list[EpisodeRecord]:
    normalized_workspace_root = _normalized_workspace_root_prefix(workspace_root)
    if root.is_file():
        try:
            payload = json.loads(root.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        documents = payload.get("episodes", []) if isinstance(payload, dict) else []
        seeds: list[EpisodeRecord] = []
        for document in documents if isinstance(documents, list) else []:
            if not isinstance(document, dict):
                continue
            if normalized_workspace_root:
                workspace = str(document.get("workspace", "")).strip()
                if not _workspace_path_under_root(
                    workspace,
                    workspace_root_prefix=normalized_workspace_root,
                ):
                    continue
            episode = _episode_record_from_document(document)
            if episode is not None:
                seeds.append(episode)
        if not seeds and normalized_workspace_root:
            for document in documents if isinstance(documents, list) else []:
                if not isinstance(document, dict):
                    continue
                episode = _episode_record_from_document(document)
                if episode is not None:
                    seeds.append(episode)
        return seeds
    seeds: list[EpisodeRecord] = []
    for document in iter_episode_documents(root):
        storage = document.get("episode_storage", {})
        if isinstance(storage, dict) and str(storage.get("phase", "primary")).strip() not in {"", "primary"}:
            continue
        if normalized_workspace_root:
            workspace = str(document.get("workspace", "")).strip()
            if not _workspace_path_under_root(
                workspace,
                workspace_root_prefix=normalized_workspace_root,
            ):
                continue
        episode = _episode_record_from_document(document)
        if episode is None:
            continue
        seeds.append(episode)
    return seeds


def _successful_seed_document(task, result: EpisodeRecord) -> dict[str, object] | None:
    if not bool(result.success):
        return None
    fragments: list[dict[str, object]] = []
    executed_commands: list[str] = []
    verified_step_count = 0
    no_state_progress_steps = 0
    state_regression_steps = 0
    for step in result.steps:
        if bool((step.verification or {}).get("passed", False)):
            verified_step_count += 1
        if "no_state_progress" in list(step.failure_signals or []):
            no_state_progress_steps += 1
        if int(step.state_regression_count) > 0:
            state_regression_steps += 1
        if step.action != "code_execute":
            continue
        command = str(step.content).strip()
        if not command:
            continue
        passed = bool((step.verification or {}).get("passed", False))
        fragments.append(
            {
                "kind": "command",
                "command": command,
                "passed": passed,
            }
        )
        if passed and command not in executed_commands:
            executed_commands.append(command)
    metadata = dict(getattr(task, "metadata", {}) or {})
    task_contract = dict(getattr(result, "task_contract", {}) or {})
    contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract, dict) else {}
    observed_runtime_seconds = float(
        result.task_metadata.get("observed_runtime_seconds", 0.0)
        or contract_metadata.get("observed_runtime_seconds", 0.0)
        or 0.0
    )
    observed_runtime_phase = str(
        result.task_metadata.get("observed_runtime_phase", "")
        or contract_metadata.get("observed_runtime_phase", "")
        or ""
    ).strip()
    observed_recorded_at = round(time.time(), 4)
    if observed_runtime_seconds > 0.0:
        metadata["observed_runtime_seconds"] = round(observed_runtime_seconds, 4)
        contract_metadata.setdefault("observed_runtime_seconds", round(observed_runtime_seconds, 4))
    if observed_runtime_phase:
        metadata["observed_runtime_phase"] = observed_runtime_phase
        contract_metadata.setdefault("observed_runtime_phase", observed_runtime_phase)
    metadata["observed_recorded_at"] = observed_recorded_at
    contract_metadata.setdefault("observed_recorded_at", observed_recorded_at)
    scheduler_state = _scheduler_state_bucket(
        success=bool(result.success),
        termination_reason=str(result.termination_reason or "success"),
        verified_step_count=verified_step_count,
        no_state_progress_steps=no_state_progress_steps,
        state_regression_steps=state_regression_steps,
    )
    metadata["observed_scheduler_state"] = scheduler_state
    contract_metadata.setdefault("observed_scheduler_state", scheduler_state)
    if metadata:
        for key, value in metadata.items():
            contract_metadata.setdefault(str(key), value)
    if contract_metadata:
        task_contract = dict(task_contract)
        task_contract["metadata"] = contract_metadata
    return {
        "task_id": str(result.task_id),
        "prompt": str(result.prompt),
        "workspace": str(result.workspace),
        "success": True,
        "task_metadata": metadata,
        "task_contract": task_contract,
        "summary": {
            "executed_commands": executed_commands,
            "failure_types": [],
            "transition_failures": [],
            "observed_runtime_seconds": round(observed_runtime_seconds, 4),
            "observed_runtime_phase": observed_runtime_phase,
            "observed_recorded_at": observed_recorded_at,
            "verified_step_count": verified_step_count,
            "no_state_progress_steps": no_state_progress_steps,
            "state_regression_steps": state_regression_steps,
            "observed_scheduler_state": scheduler_state,
        },
        "fragments": fragments,
        "termination_reason": str(result.termination_reason or "success"),
    }


def _seed_document_runtime_prior_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    family = str(metadata.get("benchmark_family", "")).strip()
    variant = str(
        metadata.get(
            "long_horizon_variant",
            metadata.get("long_horizon_coding_surface", ""),
        )
    ).strip()
    branch_kind = str(
        metadata.get(
            "lineage_branch_kind",
            metadata.get("long_horizon_branch_kind", ""),
        )
    ).strip()
    if not family:
        return ""
    return f"{family}:{variant}:{branch_kind}"


def _seed_document_family_prior_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = (
        dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    )
    family = str(metadata.get("benchmark_family", contract_metadata.get("benchmark_family", ""))).strip()
    return family


def _seed_document_family_branch_prior_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = (
        dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    )
    family = str(metadata.get("benchmark_family", contract_metadata.get("benchmark_family", ""))).strip()
    branch_kind = str(
        metadata.get(
            "lineage_branch_kind",
            contract_metadata.get(
                "lineage_branch_kind",
                metadata.get(
                    "long_horizon_branch_kind",
                    contract_metadata.get("long_horizon_branch_kind", ""),
                ),
            ),
        )
    ).strip()
    if not branch_kind:
        lineage_branch_kinds = metadata.get("lineage_branch_kinds", [])
        if not isinstance(lineage_branch_kinds, list):
            lineage_branch_kinds = contract_metadata.get("lineage_branch_kinds", [])
        if isinstance(lineage_branch_kinds, list):
            for value in reversed(lineage_branch_kinds):
                candidate = str(value).strip()
                if candidate:
                    branch_kind = candidate
                    break
    if not family:
        return ""
    return f"{family}:{branch_kind}"


def _seed_document_late_wave_branch_prior_key(document: dict[str, object]) -> str:
    family_key = _seed_document_family_prior_key(document)
    if family_key not in _LATE_WAVE_ROTATION_FAMILIES:
        return ""
    family_branch_key = _seed_document_family_branch_prior_key(document)
    if ":" not in family_branch_key:
        return ""
    _, branch_kind = family_branch_key.split(":", 1)
    return str(branch_kind).strip()


def _seed_document_lineage_depth(document: dict[str, object]) -> int:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = (
        dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    )
    raw_depth = metadata.get("lineage_depth", contract_metadata.get("lineage_depth", 0))
    try:
        depth = int(raw_depth or 0)
    except (TypeError, ValueError):
        depth = 0
    if depth > 0:
        return depth
    lineage_families = metadata.get("lineage_families", contract_metadata.get("lineage_families", []))
    if isinstance(lineage_families, list):
        return len([value for value in lineage_families if str(value).strip()])
    return 0


def _seed_document_late_wave_phase_prior_key(document: dict[str, object]) -> str:
    branch_kind = _seed_document_late_wave_branch_prior_key(document)
    if not branch_kind:
        return ""
    phase = _lineage_phase_bucket_from_depth(_seed_document_lineage_depth(document))
    if not phase:
        return ""
    return f"{branch_kind}:{phase}"


def _seed_document_scheduler_state(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = (
        dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    )
    explicit = str(
        metadata.get(
            "observed_scheduler_state",
            contract_metadata.get("observed_scheduler_state", summary.get("observed_scheduler_state", "")),
        )
    ).strip()
    if explicit:
        return explicit
    try:
        verified_step_count = int(
            metadata.get(
                "verified_step_count",
                summary.get("verified_step_count", contract_metadata.get("verified_step_count", 0)),
            )
            or 0
        )
    except (TypeError, ValueError):
        verified_step_count = 0
    try:
        no_state_progress_steps = int(
            metadata.get(
                "no_state_progress_steps",
                summary.get("no_state_progress_steps", contract_metadata.get("no_state_progress_steps", 0)),
            )
            or 0
        )
    except (TypeError, ValueError):
        no_state_progress_steps = 0
    try:
        state_regression_steps = int(
            metadata.get(
                "state_regression_steps",
                summary.get("state_regression_steps", contract_metadata.get("state_regression_steps", 0)),
            )
            or 0
        )
    except (TypeError, ValueError):
        state_regression_steps = 0
    return _scheduler_state_bucket(
        success=_seed_document_success(document),
        termination_reason=_seed_document_termination_reason(document),
        verified_step_count=verified_step_count,
        no_state_progress_steps=no_state_progress_steps,
        state_regression_steps=state_regression_steps,
    )


def _seed_document_late_wave_phase_state_prior_key(document: dict[str, object]) -> str:
    phase_key = _seed_document_late_wave_phase_prior_key(document)
    if not phase_key:
        return ""
    scheduler_state = _seed_document_scheduler_state(document)
    if not scheduler_state:
        return ""
    return f"{phase_key}:{scheduler_state}"


def _seed_document_late_wave_dispersion_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    benchmark_family = str(metadata.get("benchmark_family", "") or "").strip()
    if not benchmark_family:
        benchmark_family = str(summary.get("benchmark_family", contract_metadata.get("benchmark_family", "")) or "").strip()
    origin_family = str(metadata.get("origin_benchmark_family", "") or "").strip()
    if not origin_family:
        origin_family = str(
            summary.get("origin_benchmark_family", contract_metadata.get("origin_benchmark_family", "")) or ""
        ).strip()
    surface = str(metadata.get("long_horizon_coding_surface", "") or "").strip()
    if not surface:
        surface = str(
            summary.get("long_horizon_coding_surface", contract_metadata.get("long_horizon_coding_surface", "")) or ""
        ).strip()
    lineage_families = metadata.get(
        "lineage_families",
        summary.get("lineage_families", contract_metadata.get("lineage_families", [])),
    )
    lineage_branch_kinds = metadata.get(
        "lineage_branch_kinds",
        summary.get("lineage_branch_kinds", contract_metadata.get("lineage_branch_kinds", [])),
    )
    if not isinstance(lineage_families, list):
        lineage_families = []
    if not isinstance(lineage_branch_kinds, list):
        lineage_branch_kinds = []
    tail_families = [str(item).strip() for item in lineage_families[-3:] if str(item).strip()]
    tail_branch_kinds = [str(item).strip() for item in lineage_branch_kinds[-3:] if str(item).strip()]
    return "::".join(
        [
            benchmark_family,
            origin_family,
            surface,
            "|".join(tail_families),
            "|".join(tail_branch_kinds),
        ]
    )


def _seed_document_late_wave_direction_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    benchmark_family = str(
        metadata.get("benchmark_family", summary.get("benchmark_family", contract_metadata.get("benchmark_family", "")))
        or ""
    ).strip()
    origin_family = str(
        metadata.get(
            "origin_benchmark_family",
            summary.get("origin_benchmark_family", contract_metadata.get("origin_benchmark_family", "")),
        )
        or ""
    ).strip()
    lineage_families = metadata.get(
        "lineage_families",
        summary.get("lineage_families", contract_metadata.get("lineage_families", [])),
    )
    if not isinstance(lineage_families, list):
        lineage_families = []
    normalized_lineage_families = [str(item).strip() for item in lineage_families if str(item).strip()]
    prior_family = normalized_lineage_families[-2] if len(normalized_lineage_families) >= 2 else ""
    if not prior_family:
        prior_family = origin_family
    branch_kind = str(
        metadata.get(
            "lineage_branch_kind",
            summary.get(
                "lineage_branch_kind",
                contract_metadata.get(
                    "lineage_branch_kind",
                    metadata.get(
                        "long_horizon_branch_kind",
                        contract_metadata.get("long_horizon_branch_kind", ""),
                    ),
                ),
            ),
        )
        or ""
    ).strip()
    if not branch_kind:
        lineage_branch_kinds = metadata.get(
            "lineage_branch_kinds",
            summary.get("lineage_branch_kinds", contract_metadata.get("lineage_branch_kinds", [])),
        )
        if isinstance(lineage_branch_kinds, list):
            for value in reversed(lineage_branch_kinds):
                candidate = str(value).strip()
                if candidate:
                    branch_kind = candidate
                    break
    if not benchmark_family:
        return ""
    if prior_family and prior_family != benchmark_family:
        return f"downstream:{prior_family}->{benchmark_family}:{branch_kind}"
    return f"lateral:{benchmark_family}:{branch_kind}"


def _seed_document_late_wave_phase_transition_key(document: dict[str, object]) -> str:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    task_contract = dict(document.get("task_contract", {}) or {}) if isinstance(document.get("task_contract", {}), dict) else {}
    contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
    benchmark_family = str(
        metadata.get("benchmark_family", summary.get("benchmark_family", contract_metadata.get("benchmark_family", "")))
        or ""
    ).strip()
    lineage_families = metadata.get(
        "lineage_families",
        summary.get("lineage_families", contract_metadata.get("lineage_families", [])),
    )
    if not isinstance(lineage_families, list):
        lineage_families = []
    normalized_lineage_families = [str(item).strip() for item in lineage_families if str(item).strip()]
    prior_family = normalized_lineage_families[-2] if len(normalized_lineage_families) >= 2 else ""
    if not prior_family or not benchmark_family or prior_family == benchmark_family:
        return ""
    depth = _seed_document_lineage_depth(document)
    if depth <= 0:
        return ""
    current_phase = _lineage_phase_bucket_from_depth(depth)
    prior_phase = _lineage_phase_bucket_from_depth(max(0, depth - 1))
    if not current_phase or not prior_phase or current_phase == prior_phase:
        return ""
    branch_kind = str(
        metadata.get(
            "lineage_branch_kind",
            summary.get(
                "lineage_branch_kind",
                contract_metadata.get(
                    "lineage_branch_kind",
                    metadata.get(
                        "long_horizon_branch_kind",
                        contract_metadata.get("long_horizon_branch_kind", ""),
                    ),
                ),
            ),
        )
        or ""
    ).strip()
    return f"{prior_phase}->{current_phase}:{prior_family}->{benchmark_family}:{branch_kind}"


def _seed_document_observed_runtime_seconds(document: dict[str, object]) -> float:
    metadata = dict(document.get("task_metadata", {}) or {}) if isinstance(document.get("task_metadata", {}), dict) else {}
    summary = dict(document.get("summary", {}) or {}) if isinstance(document.get("summary", {}), dict) else {}
    value = metadata.get("observed_runtime_seconds", summary.get("observed_runtime_seconds", 0.0))
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _seed_document_success(document: dict[str, object]) -> bool:
    return bool(document.get("success", False))


def _seed_document_termination_reason(document: dict[str, object]) -> str:
    return str(document.get("termination_reason", "")).strip().lower()


def _load_existing_seed_documents(output_path: str) -> list[dict[str, object]]:
    path = Path(output_path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    documents = payload.get("episodes", []) if isinstance(payload, dict) else []
    return [dict(document) for document in documents if isinstance(document, dict)]


def _load_existing_seed_bundle_payload(output_path: str) -> dict[str, object]:
    path = Path(output_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _seed_runtime_priors(seed_documents: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for document in seed_documents:
        key = _seed_document_runtime_prior_key(document)
        runtime_seconds = _seed_document_observed_runtime_seconds(document)
        if not key or runtime_seconds <= 0.0:
            continue
        row = aggregates.setdefault(key, {"count": 0.0, "total_seconds": 0.0})
        row["count"] += 1.0
        row["total_seconds"] += runtime_seconds
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        count = int(row["count"])
        total_seconds = float(row["total_seconds"])
        if count <= 0 or total_seconds <= 0.0:
            continue
        priors[key] = {
            "count": float(count),
            "mean_seconds": round(total_seconds / count, 4),
        }
    return priors


def _seed_runtime_priors_by_key(
    seed_documents: list[dict[str, object]],
    *,
    key_fn,
) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for document in seed_documents:
        key = str(key_fn(document) or "").strip()
        runtime_seconds = _seed_document_observed_runtime_seconds(document)
        if not key or runtime_seconds <= 0.0:
            continue
        row = aggregates.setdefault(key, {"count": 0.0, "total_seconds": 0.0})
        row["count"] += 1.0
        row["total_seconds"] += runtime_seconds
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        count = int(row["count"])
        total_seconds = float(row["total_seconds"])
        if count <= 0 or total_seconds <= 0.0:
            continue
        priors[key] = {
            "count": float(count),
            "mean_seconds": round(total_seconds / count, 4),
        }
    return priors


def _seed_runtime_priors_by_key_recency_weighted(
    seed_documents: list[dict[str, object]],
    *,
    key_fn,
    support_key_fn=None,
    directional_support_key_fn=None,
    phase_transition_support_key_fn=None,
    now: float | None = None,
) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, object]] = {}
    for document in seed_documents:
        key = str(key_fn(document) or "").strip()
        runtime_seconds = _seed_document_observed_runtime_seconds(document)
        if not key or runtime_seconds <= 0.0:
            continue
        weight = _seed_document_recency_weight(document, now=now)
        row = aggregates.setdefault(
            key,
            {
                "weighted_count": 0.0,
                "weighted_seconds": 0.0,
                "support_count": 0.0,
                "support_keys": set(),
                "directional_support_keys": set(),
                "phase_transition_support_keys": set(),
            },
        )
        row["weighted_count"] += weight
        row["weighted_seconds"] += runtime_seconds * weight
        row["support_count"] += 1.0
        if support_key_fn is not None:
            support_key = str(support_key_fn(document) or "").strip()
            if support_key:
                row["support_keys"].add(support_key)
        if directional_support_key_fn is not None:
            directional_support_key = str(directional_support_key_fn(document) or "").strip()
            if directional_support_key:
                row["directional_support_keys"].add(directional_support_key)
        if phase_transition_support_key_fn is not None:
            phase_transition_support_key = str(phase_transition_support_key_fn(document) or "").strip()
            if phase_transition_support_key:
                row["phase_transition_support_keys"].add(phase_transition_support_key)
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        weighted_count = float(row["weighted_count"])
        weighted_seconds = float(row["weighted_seconds"])
        if weighted_count <= 0.0 or weighted_seconds <= 0.0:
            continue
        priors[key] = {
            "count": round(weighted_count, 4),
            "mean_seconds": round(weighted_seconds / weighted_count, 4),
            "support_count": round(float(row["support_count"]), 4),
            "dispersion_count": round(float(len(row["support_keys"])), 4),
            "directional_dispersion_count": round(
                float(len(row["directional_support_keys"])),
                4,
            ),
            "phase_transition_count": round(
                float(len(row["phase_transition_support_keys"])),
                4,
            ),
        }
    return priors


def _seed_outcome_priors(seed_documents: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for document in seed_documents:
        key = _seed_document_runtime_prior_key(document)
        if not key:
            continue
        row = aggregates.setdefault(
            key,
            {
                "count": 0.0,
                "success_count": 0.0,
                "timeout_count": 0.0,
                "budget_exceeded_count": 0.0,
            },
        )
        row["count"] += 1.0
        if _seed_document_success(document):
            row["success_count"] += 1.0
        termination_reason = _seed_document_termination_reason(document)
        if "timeout" in termination_reason:
            row["timeout_count"] += 1.0
        if "budget" in termination_reason:
            row["budget_exceeded_count"] += 1.0
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        count = int(row["count"])
        if count <= 0:
            continue
        success_count = float(row["success_count"])
        timeout_count = float(row["timeout_count"])
        budget_exceeded_count = float(row["budget_exceeded_count"])
        priors[key] = {
            "count": float(count),
            "success_rate": round(success_count / count, 4),
            "timeout_rate": round(timeout_count / count, 4),
            "budget_exceeded_rate": round(budget_exceeded_count / count, 4),
        }
    return priors


def _seed_outcome_priors_by_key(
    seed_documents: list[dict[str, object]],
    *,
    key_fn,
) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for document in seed_documents:
        key = str(key_fn(document) or "").strip()
        if not key:
            continue
        row = aggregates.setdefault(
            key,
            {
                "count": 0.0,
                "success_count": 0.0,
                "timeout_count": 0.0,
                "budget_exceeded_count": 0.0,
            },
        )
        row["count"] += 1.0
        if _seed_document_success(document):
            row["success_count"] += 1.0
        termination_reason = _seed_document_termination_reason(document)
        if "timeout" in termination_reason:
            row["timeout_count"] += 1.0
        if "budget" in termination_reason:
            row["budget_exceeded_count"] += 1.0
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        count = int(row["count"])
        if count <= 0:
            continue
        success_count = float(row["success_count"])
        timeout_count = float(row["timeout_count"])
        budget_exceeded_count = float(row["budget_exceeded_count"])
        priors[key] = {
            "count": float(count),
            "success_rate": round(success_count / count, 4),
            "timeout_rate": round(timeout_count / count, 4),
            "budget_exceeded_rate": round(budget_exceeded_count / count, 4),
        }
    return priors


def _seed_outcome_priors_by_key_recency_weighted(
    seed_documents: list[dict[str, object]],
    *,
    key_fn,
    support_key_fn=None,
    directional_support_key_fn=None,
    phase_transition_support_key_fn=None,
    now: float | None = None,
) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, object]] = {}
    for document in seed_documents:
        key = str(key_fn(document) or "").strip()
        if not key:
            continue
        weight = _seed_document_recency_weight(document, now=now)
        row = aggregates.setdefault(
            key,
            {
                "weighted_count": 0.0,
                "weighted_success": 0.0,
                "weighted_timeout": 0.0,
                "weighted_budget_exceeded": 0.0,
                "support_count": 0.0,
                "support_keys": set(),
                "directional_support_keys": set(),
                "phase_transition_support_keys": set(),
            },
        )
        row["weighted_count"] += weight
        row["support_count"] += 1.0
        if support_key_fn is not None:
            support_key = str(support_key_fn(document) or "").strip()
            if support_key:
                row["support_keys"].add(support_key)
        if directional_support_key_fn is not None:
            directional_support_key = str(directional_support_key_fn(document) or "").strip()
            if directional_support_key:
                row["directional_support_keys"].add(directional_support_key)
        if phase_transition_support_key_fn is not None:
            phase_transition_support_key = str(phase_transition_support_key_fn(document) or "").strip()
            if phase_transition_support_key:
                row["phase_transition_support_keys"].add(phase_transition_support_key)
        if _seed_document_success(document):
            row["weighted_success"] += weight
        termination_reason = _seed_document_termination_reason(document)
        if "timeout" in termination_reason:
            row["weighted_timeout"] += weight
        if "budget" in termination_reason:
            row["weighted_budget_exceeded"] += weight
    priors: dict[str, dict[str, float]] = {}
    for key, row in aggregates.items():
        weighted_count = float(row["weighted_count"])
        if weighted_count <= 0.0:
            continue
        priors[key] = {
            "count": round(weighted_count, 4),
            "success_rate": round(float(row["weighted_success"]) / weighted_count, 4),
            "timeout_rate": round(float(row["weighted_timeout"]) / weighted_count, 4),
            "budget_exceeded_rate": round(float(row["weighted_budget_exceeded"]) / weighted_count, 4),
            "support_count": round(float(row["support_count"]), 4),
            "dispersion_count": round(float(len(row["support_keys"])), 4),
            "directional_dispersion_count": round(
                float(len(row["directional_support_keys"])),
                4,
            ),
            "phase_transition_count": round(
                float(len(row["phase_transition_support_keys"])),
                4,
            ),
        }
    return priors


def _merge_runtime_prior_maps(
    historical_priors: dict[str, object],
    current_priors: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    keys = set(dict(historical_priors or {}).keys()) | set(current_priors.keys())
    for key in sorted(keys):
        historical = dict(dict(historical_priors or {}).get(key, {}) or {})
        current = dict(current_priors.get(key, {}) or {})
        historical_count = max(0.0, _safe_float(historical.get("count", 0.0), 0.0))
        current_count = max(0.0, _safe_float(current.get("count", 0.0), 0.0))
        historical_mean = float(historical.get("mean_seconds", 0.0) or 0.0)
        current_mean = float(current.get("mean_seconds", 0.0) or 0.0)
        historical_support_count = max(0.0, _safe_float(historical.get("support_count", 0.0), 0.0))
        current_support_count = max(0.0, _safe_float(current.get("support_count", 0.0), 0.0))
        historical_dispersion_count = max(0.0, _safe_float(historical.get("dispersion_count", 0.0), 0.0))
        current_dispersion_count = max(0.0, _safe_float(current.get("dispersion_count", 0.0), 0.0))
        historical_directional_dispersion_count = max(
            0.0,
            _safe_float(historical.get("directional_dispersion_count", 0.0), 0.0),
        )
        current_directional_dispersion_count = max(
            0.0,
            _safe_float(current.get("directional_dispersion_count", 0.0), 0.0),
        )
        historical_phase_transition_count = max(
            0.0,
            _safe_float(historical.get("phase_transition_count", 0.0), 0.0),
        )
        current_phase_transition_count = max(
            0.0,
            _safe_float(current.get("phase_transition_count", 0.0), 0.0),
        )
        total_count = historical_count + current_count
        if total_count <= 0:
            continue
        total_seconds = (historical_mean * historical_count) + (current_mean * current_count)
        merged[key] = {
            "count": round(float(total_count), 4),
            "mean_seconds": round(total_seconds / float(total_count), 4),
            "support_count": round(float(historical_support_count + current_support_count), 4),
            "dispersion_count": round(max(historical_dispersion_count, current_dispersion_count), 4),
            "directional_dispersion_count": round(
                max(historical_directional_dispersion_count, current_directional_dispersion_count),
                4,
            ),
            "phase_transition_count": round(
                max(historical_phase_transition_count, current_phase_transition_count),
                4,
            ),
        }
    return merged


def _merge_outcome_prior_maps(
    historical_priors: dict[str, object],
    current_priors: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    keys = set(dict(historical_priors or {}).keys()) | set(current_priors.keys())
    for key in sorted(keys):
        historical = dict(dict(historical_priors or {}).get(key, {}) or {})
        current = dict(current_priors.get(key, {}) or {})
        historical_count = max(0.0, _safe_float(historical.get("count", 0.0), 0.0))
        current_count = max(0.0, _safe_float(current.get("count", 0.0), 0.0))
        historical_support_count = max(0.0, _safe_float(historical.get("support_count", 0.0), 0.0))
        current_support_count = max(0.0, _safe_float(current.get("support_count", 0.0), 0.0))
        historical_dispersion_count = max(0.0, _safe_float(historical.get("dispersion_count", 0.0), 0.0))
        current_dispersion_count = max(0.0, _safe_float(current.get("dispersion_count", 0.0), 0.0))
        historical_directional_dispersion_count = max(
            0.0,
            _safe_float(historical.get("directional_dispersion_count", 0.0), 0.0),
        )
        current_directional_dispersion_count = max(
            0.0,
            _safe_float(current.get("directional_dispersion_count", 0.0), 0.0),
        )
        historical_phase_transition_count = max(
            0.0,
            _safe_float(historical.get("phase_transition_count", 0.0), 0.0),
        )
        current_phase_transition_count = max(
            0.0,
            _safe_float(current.get("phase_transition_count", 0.0), 0.0),
        )
        total_count = historical_count + current_count
        if total_count <= 0:
            continue
        success_total = (
            float(historical.get("success_rate", 0.0) or 0.0) * historical_count
            + float(current.get("success_rate", 0.0) or 0.0) * current_count
        )
        timeout_total = (
            float(historical.get("timeout_rate", 0.0) or 0.0) * historical_count
            + float(current.get("timeout_rate", 0.0) or 0.0) * current_count
        )
        budget_total = (
            float(historical.get("budget_exceeded_rate", 0.0) or 0.0) * historical_count
            + float(current.get("budget_exceeded_rate", 0.0) or 0.0) * current_count
        )
        merged[key] = {
            "count": round(float(total_count), 4),
            "success_rate": round(success_total / float(total_count), 4),
            "timeout_rate": round(timeout_total / float(total_count), 4),
            "budget_exceeded_rate": round(budget_total / float(total_count), 4),
            "support_count": round(float(historical_support_count + current_support_count), 4),
            "dispersion_count": round(max(historical_dispersion_count, current_dispersion_count), 4),
            "directional_dispersion_count": round(
                max(historical_directional_dispersion_count, current_directional_dispersion_count),
                4,
            ),
            "phase_transition_count": round(
                max(historical_phase_transition_count, current_phase_transition_count),
                4,
            ),
        }
    return merged


def _apply_seed_runtime_priors(
    seed_documents: list[dict[str, object]],
    *,
    runtime_priors: dict[str, dict[str, float]],
    runtime_family_priors: dict[str, dict[str, float]] | None = None,
    runtime_family_branch_priors: dict[str, dict[str, float]] | None = None,
    runtime_late_wave_branch_priors: dict[str, dict[str, float]] | None = None,
    runtime_late_wave_phase_priors: dict[str, dict[str, float]] | None = None,
    runtime_late_wave_phase_state_priors: dict[str, dict[str, float]] | None = None,
    outcome_priors: dict[str, dict[str, float]] | None = None,
    outcome_family_priors: dict[str, dict[str, float]] | None = None,
    outcome_family_branch_priors: dict[str, dict[str, float]] | None = None,
    outcome_late_wave_branch_priors: dict[str, dict[str, float]] | None = None,
    outcome_late_wave_phase_priors: dict[str, dict[str, float]] | None = None,
    outcome_late_wave_phase_state_priors: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for document in seed_documents:
        updated = dict(document)
        metadata = dict(updated.get("task_metadata", {}) or {}) if isinstance(updated.get("task_metadata", {}), dict) else {}
        summary = dict(updated.get("summary", {}) or {}) if isinstance(updated.get("summary", {}), dict) else {}
        task_contract = dict(updated.get("task_contract", {}) or {}) if isinstance(updated.get("task_contract", {}), dict) else {}
        contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract.get("metadata", {}), dict) else {}
        key = _seed_document_runtime_prior_key(updated)
        prior = runtime_priors.get(key, {})
        mean_seconds = float(prior.get("mean_seconds", 0.0) or 0.0)
        count = int(prior.get("count", 0.0) or 0.0)
        if mean_seconds > 0.0 and count > 0:
            metadata["observed_runtime_prior_seconds"] = round(mean_seconds, 4)
            metadata["observed_runtime_prior_count"] = count
            contract_metadata.setdefault("observed_runtime_prior_seconds", round(mean_seconds, 4))
            contract_metadata.setdefault("observed_runtime_prior_count", count)
            summary["observed_runtime_prior_seconds"] = round(mean_seconds, 4)
            summary["observed_runtime_prior_count"] = count
        family_key = _seed_document_family_prior_key(updated)
        family_prior = (runtime_family_priors or {}).get(family_key, {})
        family_mean_seconds = float(family_prior.get("mean_seconds", 0.0) or 0.0)
        family_count = int(family_prior.get("count", 0.0) or 0.0)
        if family_mean_seconds > 0.0 and family_count > 0:
            metadata["observed_runtime_family_prior_seconds"] = round(family_mean_seconds, 4)
            metadata["observed_runtime_family_prior_count"] = family_count
            contract_metadata.setdefault("observed_runtime_family_prior_seconds", round(family_mean_seconds, 4))
            contract_metadata.setdefault("observed_runtime_family_prior_count", family_count)
            summary["observed_runtime_family_prior_seconds"] = round(family_mean_seconds, 4)
            summary["observed_runtime_family_prior_count"] = family_count
        family_branch_key = _seed_document_family_branch_prior_key(updated)
        family_branch_prior = (runtime_family_branch_priors or {}).get(family_branch_key, {})
        family_branch_mean_seconds = float(family_branch_prior.get("mean_seconds", 0.0) or 0.0)
        family_branch_count = int(family_branch_prior.get("count", 0.0) or 0.0)
        if family_branch_mean_seconds > 0.0 and family_branch_count > 0:
            metadata["observed_runtime_family_branch_prior_seconds"] = round(family_branch_mean_seconds, 4)
            metadata["observed_runtime_family_branch_prior_count"] = family_branch_count
            contract_metadata.setdefault(
                "observed_runtime_family_branch_prior_seconds",
                round(family_branch_mean_seconds, 4),
            )
            contract_metadata.setdefault("observed_runtime_family_branch_prior_count", family_branch_count)
            summary["observed_runtime_family_branch_prior_seconds"] = round(family_branch_mean_seconds, 4)
            summary["observed_runtime_family_branch_prior_count"] = family_branch_count
        late_wave_branch_key = _seed_document_late_wave_branch_prior_key(updated)
        late_wave_branch_prior = (runtime_late_wave_branch_priors or {}).get(late_wave_branch_key, {})
        late_wave_branch_mean_seconds = float(late_wave_branch_prior.get("mean_seconds", 0.0) or 0.0)
        late_wave_branch_count = int(late_wave_branch_prior.get("count", 0.0) or 0.0)
        if late_wave_branch_mean_seconds > 0.0 and late_wave_branch_count > 0:
            metadata["observed_runtime_late_wave_branch_prior_seconds"] = round(late_wave_branch_mean_seconds, 4)
            metadata["observed_runtime_late_wave_branch_prior_count"] = late_wave_branch_count
            contract_metadata.setdefault(
                "observed_runtime_late_wave_branch_prior_seconds",
                round(late_wave_branch_mean_seconds, 4),
            )
            contract_metadata.setdefault("observed_runtime_late_wave_branch_prior_count", late_wave_branch_count)
            summary["observed_runtime_late_wave_branch_prior_seconds"] = round(late_wave_branch_mean_seconds, 4)
            summary["observed_runtime_late_wave_branch_prior_count"] = late_wave_branch_count
        late_wave_phase_key = _seed_document_late_wave_phase_prior_key(updated)
        late_wave_phase_prior = (runtime_late_wave_phase_priors or {}).get(late_wave_phase_key, {})
        late_wave_phase_mean_seconds = float(late_wave_phase_prior.get("mean_seconds", 0.0) or 0.0)
        late_wave_phase_count = int(late_wave_phase_prior.get("count", 0.0) or 0.0)
        if late_wave_phase_mean_seconds > 0.0 and late_wave_phase_count > 0:
            metadata["observed_runtime_late_wave_phase_prior_seconds"] = round(late_wave_phase_mean_seconds, 4)
            metadata["observed_runtime_late_wave_phase_prior_count"] = late_wave_phase_count
            metadata["observed_lineage_phase"] = late_wave_phase_key.rsplit(":", 1)[-1]
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_prior_seconds",
                round(late_wave_phase_mean_seconds, 4),
            )
            contract_metadata.setdefault("observed_runtime_late_wave_phase_prior_count", late_wave_phase_count)
            contract_metadata.setdefault("observed_lineage_phase", late_wave_phase_key.rsplit(":", 1)[-1])
            summary["observed_runtime_late_wave_phase_prior_seconds"] = round(late_wave_phase_mean_seconds, 4)
            summary["observed_runtime_late_wave_phase_prior_count"] = late_wave_phase_count
            summary["observed_lineage_phase"] = late_wave_phase_key.rsplit(":", 1)[-1]
        scheduler_state = _seed_document_scheduler_state(updated)
        metadata["observed_scheduler_state"] = scheduler_state
        contract_metadata.setdefault("observed_scheduler_state", scheduler_state)
        summary["observed_scheduler_state"] = scheduler_state
        late_wave_phase_state_key = _seed_document_late_wave_phase_state_prior_key(updated)
        late_wave_phase_state_prior = (runtime_late_wave_phase_state_priors or {}).get(late_wave_phase_state_key, {})
        late_wave_phase_state_mean_seconds = float(late_wave_phase_state_prior.get("mean_seconds", 0.0) or 0.0)
        late_wave_phase_state_count = float(late_wave_phase_state_prior.get("count", 0.0) or 0.0)
        late_wave_phase_state_support_count = float(late_wave_phase_state_prior.get("support_count", 0.0) or 0.0)
        late_wave_phase_state_dispersion_count = float(
            late_wave_phase_state_prior.get("dispersion_count", 0.0) or 0.0
        )
        late_wave_phase_state_directional_dispersion_count = float(
            late_wave_phase_state_prior.get("directional_dispersion_count", 0.0) or 0.0
        )
        late_wave_phase_state_phase_transition_count = float(
            late_wave_phase_state_prior.get("phase_transition_count", 0.0) or 0.0
        )
        if late_wave_phase_state_mean_seconds > 0.0 and late_wave_phase_state_count > 0:
            metadata["observed_runtime_late_wave_phase_state_prior_seconds"] = round(
                late_wave_phase_state_mean_seconds, 4
            )
            metadata["observed_runtime_late_wave_phase_state_prior_count"] = round(late_wave_phase_state_count, 4)
            metadata["observed_runtime_late_wave_phase_state_prior_support_count"] = round(
                late_wave_phase_state_support_count, 4
            )
            metadata["observed_runtime_late_wave_phase_state_prior_dispersion_count"] = round(
                late_wave_phase_state_dispersion_count, 4
            )
            metadata["observed_runtime_late_wave_phase_state_prior_directional_dispersion_count"] = round(
                late_wave_phase_state_directional_dispersion_count, 4
            )
            metadata["observed_runtime_late_wave_phase_state_prior_phase_transition_count"] = round(
                late_wave_phase_state_phase_transition_count, 4
            )
            metadata["observed_runtime_late_wave_phase_state_prior_is_recency_weighted"] = True
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_seconds",
                round(late_wave_phase_state_mean_seconds, 4),
            )
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_count",
                round(late_wave_phase_state_count, 4),
            )
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_support_count",
                round(late_wave_phase_state_support_count, 4),
            )
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_dispersion_count",
                round(late_wave_phase_state_dispersion_count, 4),
            )
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_directional_dispersion_count",
                round(late_wave_phase_state_directional_dispersion_count, 4),
            )
            contract_metadata.setdefault(
                "observed_runtime_late_wave_phase_state_prior_phase_transition_count",
                round(late_wave_phase_state_phase_transition_count, 4),
            )
            contract_metadata.setdefault("observed_runtime_late_wave_phase_state_prior_is_recency_weighted", True)
            summary["observed_runtime_late_wave_phase_state_prior_seconds"] = round(
                late_wave_phase_state_mean_seconds, 4
            )
            summary["observed_runtime_late_wave_phase_state_prior_count"] = round(late_wave_phase_state_count, 4)
            summary["observed_runtime_late_wave_phase_state_prior_support_count"] = round(
                late_wave_phase_state_support_count, 4
            )
            summary["observed_runtime_late_wave_phase_state_prior_dispersion_count"] = round(
                late_wave_phase_state_dispersion_count, 4
            )
            summary["observed_runtime_late_wave_phase_state_prior_directional_dispersion_count"] = round(
                late_wave_phase_state_directional_dispersion_count, 4
            )
            summary["observed_runtime_late_wave_phase_state_prior_phase_transition_count"] = round(
                late_wave_phase_state_phase_transition_count, 4
            )
            summary["observed_runtime_late_wave_phase_state_prior_is_recency_weighted"] = True
        outcome_prior = (outcome_priors or {}).get(key, {})
        success_rate = float(outcome_prior.get("success_rate", 0.0) or 0.0)
        timeout_rate = float(outcome_prior.get("timeout_rate", 0.0) or 0.0)
        budget_exceeded_rate = float(outcome_prior.get("budget_exceeded_rate", 0.0) or 0.0)
        outcome_count = int(outcome_prior.get("count", 0.0) or 0.0)
        if outcome_count > 0:
            metadata["observed_outcome_prior_count"] = outcome_count
            metadata["observed_success_prior_rate"] = round(success_rate, 4)
            metadata["observed_timeout_prior_rate"] = round(timeout_rate, 4)
            metadata["observed_budget_exceeded_prior_rate"] = round(budget_exceeded_rate, 4)
            contract_metadata.setdefault("observed_outcome_prior_count", outcome_count)
            contract_metadata.setdefault("observed_success_prior_rate", round(success_rate, 4))
            contract_metadata.setdefault("observed_timeout_prior_rate", round(timeout_rate, 4))
            contract_metadata.setdefault("observed_budget_exceeded_prior_rate", round(budget_exceeded_rate, 4))
            summary["observed_outcome_prior_count"] = outcome_count
            summary["observed_success_prior_rate"] = round(success_rate, 4)
            summary["observed_timeout_prior_rate"] = round(timeout_rate, 4)
            summary["observed_budget_exceeded_prior_rate"] = round(budget_exceeded_rate, 4)
        outcome_family_prior = (outcome_family_priors or {}).get(family_key, {})
        outcome_family_count = int(outcome_family_prior.get("count", 0.0) or 0.0)
        if outcome_family_count > 0:
            family_success_rate = float(outcome_family_prior.get("success_rate", 0.0) or 0.0)
            family_timeout_rate = float(outcome_family_prior.get("timeout_rate", 0.0) or 0.0)
            family_budget_exceeded_rate = float(outcome_family_prior.get("budget_exceeded_rate", 0.0) or 0.0)
            metadata["observed_outcome_family_prior_count"] = outcome_family_count
            metadata["observed_success_family_prior_rate"] = round(family_success_rate, 4)
            metadata["observed_timeout_family_prior_rate"] = round(family_timeout_rate, 4)
            metadata["observed_budget_exceeded_family_prior_rate"] = round(family_budget_exceeded_rate, 4)
            contract_metadata.setdefault("observed_outcome_family_prior_count", outcome_family_count)
            contract_metadata.setdefault("observed_success_family_prior_rate", round(family_success_rate, 4))
            contract_metadata.setdefault("observed_timeout_family_prior_rate", round(family_timeout_rate, 4))
            contract_metadata.setdefault(
                "observed_budget_exceeded_family_prior_rate",
                round(family_budget_exceeded_rate, 4),
            )
            summary["observed_outcome_family_prior_count"] = outcome_family_count
            summary["observed_success_family_prior_rate"] = round(family_success_rate, 4)
            summary["observed_timeout_family_prior_rate"] = round(family_timeout_rate, 4)
            summary["observed_budget_exceeded_family_prior_rate"] = round(family_budget_exceeded_rate, 4)
        outcome_family_branch_prior = (outcome_family_branch_priors or {}).get(family_branch_key, {})
        outcome_family_branch_count = int(outcome_family_branch_prior.get("count", 0.0) or 0.0)
        if outcome_family_branch_count > 0:
            family_branch_success_rate = float(outcome_family_branch_prior.get("success_rate", 0.0) or 0.0)
            family_branch_timeout_rate = float(outcome_family_branch_prior.get("timeout_rate", 0.0) or 0.0)
            family_branch_budget_exceeded_rate = float(
                outcome_family_branch_prior.get("budget_exceeded_rate", 0.0) or 0.0
            )
            metadata["observed_outcome_family_branch_prior_count"] = outcome_family_branch_count
            metadata["observed_success_family_branch_prior_rate"] = round(family_branch_success_rate, 4)
            metadata["observed_timeout_family_branch_prior_rate"] = round(family_branch_timeout_rate, 4)
            metadata["observed_budget_exceeded_family_branch_prior_rate"] = round(
                family_branch_budget_exceeded_rate,
                4,
            )
            contract_metadata.setdefault(
                "observed_outcome_family_branch_prior_count",
                outcome_family_branch_count,
            )
            contract_metadata.setdefault(
                "observed_success_family_branch_prior_rate",
                round(family_branch_success_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_timeout_family_branch_prior_rate",
                round(family_branch_timeout_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_budget_exceeded_family_branch_prior_rate",
                round(family_branch_budget_exceeded_rate, 4),
            )
            summary["observed_outcome_family_branch_prior_count"] = outcome_family_branch_count
            summary["observed_success_family_branch_prior_rate"] = round(family_branch_success_rate, 4)
            summary["observed_timeout_family_branch_prior_rate"] = round(family_branch_timeout_rate, 4)
            summary["observed_budget_exceeded_family_branch_prior_rate"] = round(
                family_branch_budget_exceeded_rate,
                4,
            )
        outcome_late_wave_branch_prior = (outcome_late_wave_branch_priors or {}).get(late_wave_branch_key, {})
        outcome_late_wave_branch_count = int(outcome_late_wave_branch_prior.get("count", 0.0) or 0.0)
        if outcome_late_wave_branch_count > 0:
            late_wave_branch_success_rate = float(outcome_late_wave_branch_prior.get("success_rate", 0.0) or 0.0)
            late_wave_branch_timeout_rate = float(outcome_late_wave_branch_prior.get("timeout_rate", 0.0) or 0.0)
            late_wave_branch_budget_exceeded_rate = float(
                outcome_late_wave_branch_prior.get("budget_exceeded_rate", 0.0) or 0.0
            )
            metadata["observed_outcome_late_wave_branch_prior_count"] = outcome_late_wave_branch_count
            metadata["observed_success_late_wave_branch_prior_rate"] = round(late_wave_branch_success_rate, 4)
            metadata["observed_timeout_late_wave_branch_prior_rate"] = round(late_wave_branch_timeout_rate, 4)
            metadata["observed_budget_exceeded_late_wave_branch_prior_rate"] = round(
                late_wave_branch_budget_exceeded_rate,
                4,
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_branch_prior_count",
                outcome_late_wave_branch_count,
            )
            contract_metadata.setdefault(
                "observed_success_late_wave_branch_prior_rate",
                round(late_wave_branch_success_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_timeout_late_wave_branch_prior_rate",
                round(late_wave_branch_timeout_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_budget_exceeded_late_wave_branch_prior_rate",
                round(late_wave_branch_budget_exceeded_rate, 4),
            )
            summary["observed_outcome_late_wave_branch_prior_count"] = outcome_late_wave_branch_count
            summary["observed_success_late_wave_branch_prior_rate"] = round(late_wave_branch_success_rate, 4)
            summary["observed_timeout_late_wave_branch_prior_rate"] = round(late_wave_branch_timeout_rate, 4)
            summary["observed_budget_exceeded_late_wave_branch_prior_rate"] = round(
                late_wave_branch_budget_exceeded_rate,
                4,
            )
        outcome_late_wave_phase_prior = (outcome_late_wave_phase_priors or {}).get(late_wave_phase_key, {})
        outcome_late_wave_phase_count = int(outcome_late_wave_phase_prior.get("count", 0.0) or 0.0)
        if outcome_late_wave_phase_count > 0:
            late_wave_phase_success_rate = float(outcome_late_wave_phase_prior.get("success_rate", 0.0) or 0.0)
            late_wave_phase_timeout_rate = float(outcome_late_wave_phase_prior.get("timeout_rate", 0.0) or 0.0)
            late_wave_phase_budget_exceeded_rate = float(
                outcome_late_wave_phase_prior.get("budget_exceeded_rate", 0.0) or 0.0
            )
            metadata["observed_outcome_late_wave_phase_prior_count"] = outcome_late_wave_phase_count
            metadata["observed_success_late_wave_phase_prior_rate"] = round(late_wave_phase_success_rate, 4)
            metadata["observed_timeout_late_wave_phase_prior_rate"] = round(late_wave_phase_timeout_rate, 4)
            metadata["observed_budget_exceeded_late_wave_phase_prior_rate"] = round(
                late_wave_phase_budget_exceeded_rate,
                4,
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_prior_count",
                outcome_late_wave_phase_count,
            )
            contract_metadata.setdefault(
                "observed_success_late_wave_phase_prior_rate",
                round(late_wave_phase_success_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_timeout_late_wave_phase_prior_rate",
                round(late_wave_phase_timeout_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_budget_exceeded_late_wave_phase_prior_rate",
                round(late_wave_phase_budget_exceeded_rate, 4),
            )
            summary["observed_outcome_late_wave_phase_prior_count"] = outcome_late_wave_phase_count
            summary["observed_success_late_wave_phase_prior_rate"] = round(late_wave_phase_success_rate, 4)
            summary["observed_timeout_late_wave_phase_prior_rate"] = round(late_wave_phase_timeout_rate, 4)
            summary["observed_budget_exceeded_late_wave_phase_prior_rate"] = round(
                late_wave_phase_budget_exceeded_rate,
                4,
            )
        outcome_late_wave_phase_state_prior = (outcome_late_wave_phase_state_priors or {}).get(
            late_wave_phase_state_key, {}
        )
        outcome_late_wave_phase_state_count = float(outcome_late_wave_phase_state_prior.get("count", 0.0) or 0.0)
        outcome_late_wave_phase_state_support_count = float(
            outcome_late_wave_phase_state_prior.get("support_count", 0.0) or 0.0
        )
        outcome_late_wave_phase_state_dispersion_count = float(
            outcome_late_wave_phase_state_prior.get("dispersion_count", 0.0) or 0.0
        )
        outcome_late_wave_phase_state_directional_dispersion_count = float(
            outcome_late_wave_phase_state_prior.get("directional_dispersion_count", 0.0) or 0.0
        )
        outcome_late_wave_phase_state_phase_transition_count = float(
            outcome_late_wave_phase_state_prior.get("phase_transition_count", 0.0) or 0.0
        )
        if outcome_late_wave_phase_state_count > 0:
            late_wave_phase_state_success_rate = float(
                outcome_late_wave_phase_state_prior.get("success_rate", 0.0) or 0.0
            )
            late_wave_phase_state_timeout_rate = float(
                outcome_late_wave_phase_state_prior.get("timeout_rate", 0.0) or 0.0
            )
            late_wave_phase_state_budget_exceeded_rate = float(
                outcome_late_wave_phase_state_prior.get("budget_exceeded_rate", 0.0) or 0.0
            )
            metadata["observed_outcome_late_wave_phase_state_prior_count"] = round(
                outcome_late_wave_phase_state_count, 4
            )
            metadata["observed_outcome_late_wave_phase_state_prior_support_count"] = round(
                outcome_late_wave_phase_state_support_count, 4
            )
            metadata["observed_outcome_late_wave_phase_state_prior_dispersion_count"] = round(
                outcome_late_wave_phase_state_dispersion_count, 4
            )
            metadata["observed_outcome_late_wave_phase_state_prior_directional_dispersion_count"] = round(
                outcome_late_wave_phase_state_directional_dispersion_count, 4
            )
            metadata["observed_outcome_late_wave_phase_state_prior_phase_transition_count"] = round(
                outcome_late_wave_phase_state_phase_transition_count, 4
            )
            metadata["observed_outcome_late_wave_phase_state_prior_is_recency_weighted"] = True
            metadata["observed_success_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_success_rate, 4
            )
            metadata["observed_timeout_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_timeout_rate, 4
            )
            metadata["observed_budget_exceeded_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_budget_exceeded_rate, 4
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_state_prior_count",
                round(outcome_late_wave_phase_state_count, 4),
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_state_prior_support_count",
                round(outcome_late_wave_phase_state_support_count, 4),
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_state_prior_dispersion_count",
                round(outcome_late_wave_phase_state_dispersion_count, 4),
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_state_prior_directional_dispersion_count",
                round(outcome_late_wave_phase_state_directional_dispersion_count, 4),
            )
            contract_metadata.setdefault(
                "observed_outcome_late_wave_phase_state_prior_phase_transition_count",
                round(outcome_late_wave_phase_state_phase_transition_count, 4),
            )
            contract_metadata.setdefault("observed_outcome_late_wave_phase_state_prior_is_recency_weighted", True)
            contract_metadata.setdefault(
                "observed_success_late_wave_phase_state_prior_rate",
                round(late_wave_phase_state_success_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_timeout_late_wave_phase_state_prior_rate",
                round(late_wave_phase_state_timeout_rate, 4),
            )
            contract_metadata.setdefault(
                "observed_budget_exceeded_late_wave_phase_state_prior_rate",
                round(late_wave_phase_state_budget_exceeded_rate, 4),
            )
            summary["observed_outcome_late_wave_phase_state_prior_count"] = round(
                outcome_late_wave_phase_state_count, 4
            )
            summary["observed_outcome_late_wave_phase_state_prior_support_count"] = round(
                outcome_late_wave_phase_state_support_count, 4
            )
            summary["observed_outcome_late_wave_phase_state_prior_dispersion_count"] = round(
                outcome_late_wave_phase_state_dispersion_count, 4
            )
            summary["observed_outcome_late_wave_phase_state_prior_directional_dispersion_count"] = round(
                outcome_late_wave_phase_state_directional_dispersion_count, 4
            )
            summary["observed_outcome_late_wave_phase_state_prior_phase_transition_count"] = round(
                outcome_late_wave_phase_state_phase_transition_count, 4
            )
            summary["observed_outcome_late_wave_phase_state_prior_is_recency_weighted"] = True
            summary["observed_success_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_success_rate, 4
            )
            summary["observed_timeout_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_timeout_rate, 4
            )
            summary["observed_budget_exceeded_late_wave_phase_state_prior_rate"] = round(
                late_wave_phase_state_budget_exceeded_rate, 4
            )
        updated["task_metadata"] = metadata
        task_contract["metadata"] = contract_metadata
        updated["task_contract"] = task_contract
        updated["summary"] = summary
        annotated.append(updated)
    return annotated


def _task_result_prior_document(task, result: EpisodeRecord) -> dict[str, object]:
    metadata = dict(getattr(task, "metadata", {}) or {})
    result_metadata = dict(getattr(result, "task_metadata", {}) or {})
    for key, value in result_metadata.items():
        metadata.setdefault(str(key), value)
    task_contract = dict(getattr(result, "task_contract", {}) or {})
    contract_metadata = dict(task_contract.get("metadata", {}) or {}) if isinstance(task_contract, dict) else {}
    if contract_metadata:
        for key, value in metadata.items():
            contract_metadata.setdefault(str(key), value)
        task_contract["metadata"] = contract_metadata
    observed_runtime_seconds = float(
        result_metadata.get("observed_runtime_seconds", 0.0)
        or contract_metadata.get("observed_runtime_seconds", 0.0)
        or 0.0
    )
    if observed_runtime_seconds > 0.0:
        metadata["observed_runtime_seconds"] = round(observed_runtime_seconds, 4)
        contract_metadata.setdefault("observed_runtime_seconds", round(observed_runtime_seconds, 4))
    observed_recorded_at = round(time.time(), 4)
    verified_step_count = sum(
        1 for step in result.steps if bool((step.verification or {}).get("passed", False))
    )
    no_state_progress_steps = sum(
        1 for step in result.steps if "no_state_progress" in list(step.failure_signals or [])
    )
    state_regression_steps = sum(1 for step in result.steps if int(step.state_regression_count) > 0)
    scheduler_state = _scheduler_state_bucket(
        success=bool(result.success),
        termination_reason=str(result.termination_reason or "").strip(),
        verified_step_count=verified_step_count,
        no_state_progress_steps=no_state_progress_steps,
        state_regression_steps=state_regression_steps,
    )
    metadata["observed_scheduler_state"] = scheduler_state
    metadata["observed_recorded_at"] = observed_recorded_at
    metadata["verified_step_count"] = verified_step_count
    metadata["no_state_progress_steps"] = no_state_progress_steps
    metadata["state_regression_steps"] = state_regression_steps
    contract_metadata.setdefault("observed_scheduler_state", scheduler_state)
    contract_metadata.setdefault("observed_recorded_at", observed_recorded_at)
    contract_metadata.setdefault("verified_step_count", verified_step_count)
    contract_metadata.setdefault("no_state_progress_steps", no_state_progress_steps)
    contract_metadata.setdefault("state_regression_steps", state_regression_steps)
    return {
        "task_id": str(result.task_id),
        "prompt": str(result.prompt),
        "workspace": str(result.workspace),
        "success": bool(result.success),
        "task_metadata": metadata,
        "task_contract": task_contract,
        "summary": {
            "observed_runtime_seconds": round(observed_runtime_seconds, 4),
            "observed_recorded_at": observed_recorded_at,
            "verified_step_count": verified_step_count,
            "no_state_progress_steps": no_state_progress_steps,
            "state_regression_steps": state_regression_steps,
            "observed_scheduler_state": scheduler_state,
        },
        "termination_reason": str(result.termination_reason or "").strip(),
    }


def _write_success_seed_bundle(
    output_path: str,
    *,
    primary_tasks: list,
    primary_results: list[EpisodeRecord],
    generated_tasks: list,
    generated_results: list[EpisodeRecord],
) -> None:
    seed_documents = [
        document
        for task, result in zip(primary_tasks, primary_results, strict=False)
        for document in [_successful_seed_document(task, result)]
        if document is not None
    ]
    generated_seed_documents = [
        document
        for task, result in zip(generated_tasks, generated_results, strict=False)
        if str(getattr(task, "metadata", {}).get("curriculum_kind", "")).strip() == "adjacent_success"
        for document in [_successful_seed_document(task, result)]
        if document is not None
    ]
    if generated_seed_documents:
        seed_documents = generated_seed_documents
    existing_payload = _load_existing_seed_bundle_payload(output_path)
    historical_seed_documents = [
        dict(document)
        for document in existing_payload.get("episodes", [])
        if isinstance(document, dict)
    ]
    historical_runtime_priors = dict(existing_payload.get("runtime_priors", {}) or {})
    historical_outcome_priors = dict(existing_payload.get("outcome_priors", {}) or {})
    historical_runtime_family_priors = dict(existing_payload.get("runtime_family_priors", {}) or {})
    historical_runtime_family_branch_priors = dict(existing_payload.get("runtime_family_branch_priors", {}) or {})
    historical_runtime_late_wave_branch_priors = dict(existing_payload.get("runtime_late_wave_branch_priors", {}) or {})
    historical_runtime_late_wave_phase_priors = dict(existing_payload.get("runtime_late_wave_phase_priors", {}) or {})
    historical_runtime_late_wave_phase_state_priors = dict(
        existing_payload.get("runtime_late_wave_phase_state_priors", {}) or {}
    )
    historical_outcome_family_priors = dict(existing_payload.get("outcome_family_priors", {}) or {})
    historical_outcome_family_branch_priors = dict(
        existing_payload.get("outcome_family_branch_priors", {}) or {}
    )
    historical_outcome_late_wave_branch_priors = dict(
        existing_payload.get("outcome_late_wave_branch_priors", {}) or {}
    )
    historical_outcome_late_wave_phase_priors = dict(
        existing_payload.get("outcome_late_wave_phase_priors", {}) or {}
    )
    historical_outcome_late_wave_phase_state_priors = dict(
        existing_payload.get("outcome_late_wave_phase_state_priors", {}) or {}
    )
    outcome_documents = [
        _task_result_prior_document(task, result)
        for task, result in [*zip(primary_tasks, primary_results, strict=False), *zip(generated_tasks, generated_results, strict=False)]
    ]
    current_runtime_priors = _seed_runtime_priors([*historical_seed_documents, *seed_documents])
    current_outcome_priors = _seed_outcome_priors([*historical_seed_documents, *outcome_documents])
    current_runtime_family_priors = _seed_runtime_priors_by_key(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_family_prior_key,
    )
    current_runtime_family_branch_priors = _seed_runtime_priors_by_key(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_family_branch_prior_key,
    )
    current_runtime_late_wave_branch_priors = _seed_runtime_priors_by_key(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_late_wave_branch_prior_key,
    )
    current_runtime_late_wave_phase_priors = _seed_runtime_priors_by_key(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_late_wave_phase_prior_key,
    )
    current_runtime_late_wave_phase_state_priors = _seed_runtime_priors_by_key(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_late_wave_phase_state_prior_key,
    )
    current_outcome_family_priors = _seed_outcome_priors_by_key(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_family_prior_key,
    )
    current_outcome_family_branch_priors = _seed_outcome_priors_by_key(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_family_branch_prior_key,
    )
    current_outcome_late_wave_branch_priors = _seed_outcome_priors_by_key(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_late_wave_branch_prior_key,
    )
    current_outcome_late_wave_phase_priors = _seed_outcome_priors_by_key(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_late_wave_phase_prior_key,
    )
    current_outcome_late_wave_phase_state_priors = _seed_outcome_priors_by_key(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_late_wave_phase_state_prior_key,
    )
    recency_now = time.time()
    current_runtime_late_wave_phase_state_priors = _seed_runtime_priors_by_key_recency_weighted(
        [*historical_seed_documents, *seed_documents],
        key_fn=_seed_document_late_wave_phase_state_prior_key,
        support_key_fn=_seed_document_late_wave_dispersion_key,
        directional_support_key_fn=_seed_document_late_wave_direction_key,
        phase_transition_support_key_fn=_seed_document_late_wave_phase_transition_key,
        now=recency_now,
    )
    current_outcome_late_wave_phase_state_priors = _seed_outcome_priors_by_key_recency_weighted(
        [*historical_seed_documents, *outcome_documents],
        key_fn=_seed_document_late_wave_phase_state_prior_key,
        support_key_fn=_seed_document_late_wave_dispersion_key,
        directional_support_key_fn=_seed_document_late_wave_direction_key,
        phase_transition_support_key_fn=_seed_document_late_wave_phase_transition_key,
        now=recency_now,
    )
    runtime_priors = _merge_runtime_prior_maps(historical_runtime_priors, current_runtime_priors)
    outcome_priors = _merge_outcome_prior_maps(historical_outcome_priors, current_outcome_priors)
    runtime_family_priors = _merge_runtime_prior_maps(
        historical_runtime_family_priors,
        current_runtime_family_priors,
    )
    runtime_family_branch_priors = _merge_runtime_prior_maps(
        historical_runtime_family_branch_priors,
        current_runtime_family_branch_priors,
    )
    runtime_late_wave_branch_priors = _merge_runtime_prior_maps(
        historical_runtime_late_wave_branch_priors,
        current_runtime_late_wave_branch_priors,
    )
    runtime_late_wave_phase_priors = _merge_runtime_prior_maps(
        historical_runtime_late_wave_phase_priors,
        current_runtime_late_wave_phase_priors,
    )
    runtime_late_wave_phase_state_priors = _merge_runtime_prior_maps(
        historical_runtime_late_wave_phase_state_priors,
        current_runtime_late_wave_phase_state_priors,
    )
    outcome_family_priors = _merge_outcome_prior_maps(
        historical_outcome_family_priors,
        current_outcome_family_priors,
    )
    outcome_family_branch_priors = _merge_outcome_prior_maps(
        historical_outcome_family_branch_priors,
        current_outcome_family_branch_priors,
    )
    outcome_late_wave_branch_priors = _merge_outcome_prior_maps(
        historical_outcome_late_wave_branch_priors,
        current_outcome_late_wave_branch_priors,
    )
    outcome_late_wave_phase_priors = _merge_outcome_prior_maps(
        historical_outcome_late_wave_phase_priors,
        current_outcome_late_wave_phase_priors,
    )
    outcome_late_wave_phase_state_priors = _merge_outcome_prior_maps(
        historical_outcome_late_wave_phase_state_priors,
        current_outcome_late_wave_phase_state_priors,
    )
    seed_documents = _apply_seed_runtime_priors(
        seed_documents,
        runtime_priors=runtime_priors,
        runtime_family_priors=runtime_family_priors,
        runtime_family_branch_priors=runtime_family_branch_priors,
        runtime_late_wave_branch_priors=runtime_late_wave_branch_priors,
        runtime_late_wave_phase_priors=runtime_late_wave_phase_priors,
        runtime_late_wave_phase_state_priors=runtime_late_wave_phase_state_priors,
        outcome_priors=outcome_priors,
        outcome_family_priors=outcome_family_priors,
        outcome_family_branch_priors=outcome_family_branch_priors,
        outcome_late_wave_branch_priors=outcome_late_wave_branch_priors,
        outcome_late_wave_phase_priors=outcome_late_wave_phase_priors,
        outcome_late_wave_phase_state_priors=outcome_late_wave_phase_state_priors,
    )
    atomic_write_json(
        Path(output_path),
        {
            "episodes": seed_documents,
            "runtime_priors": runtime_priors,
            "runtime_family_priors": runtime_family_priors,
            "runtime_family_branch_priors": runtime_family_branch_priors,
            "runtime_late_wave_branch_priors": runtime_late_wave_branch_priors,
            "runtime_late_wave_phase_priors": runtime_late_wave_phase_priors,
            "runtime_late_wave_phase_state_priors": runtime_late_wave_phase_state_priors,
            "outcome_priors": outcome_priors,
            "outcome_family_priors": outcome_family_priors,
            "outcome_family_branch_priors": outcome_family_branch_priors,
            "outcome_late_wave_branch_priors": outcome_late_wave_branch_priors,
            "outcome_late_wave_phase_priors": outcome_late_wave_phase_priors,
            "outcome_late_wave_phase_state_priors": outcome_late_wave_phase_state_priors,
        },
    )


def _task_outcome_summary(task, result, *, low_confidence_threshold: float) -> dict[str, object]:
    benchmark_family = str(task.metadata.get("benchmark_family", "bounded"))
    memory_source = str(task.metadata.get("memory_source", "none"))
    difficulty = str(task.metadata.get("difficulty", "unknown"))
    curriculum_kind = str(task.metadata.get("curriculum_kind", "")).strip()
    contract_clean_failure_recovery_origin = bool(task.metadata.get("contract_clean_failure_recovery_origin", False))
    contract_clean_failure_recovery_origin_family = str(
        task.metadata.get("contract_clean_failure_recovery_origin_family", benchmark_family)
    ).strip() or benchmark_family
    try:
        contract_clean_failure_recovery_step_floor = max(
            0,
            int(
                task.metadata.get(
                    "contract_clean_failure_recovery_step_floor",
                    task.metadata.get("budget_step_floor", task.metadata.get("step_floor", 0)),
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        contract_clean_failure_recovery_step_floor = 0
    no_state_progress_steps = sum(
        1 for step in result.steps if "no_state_progress" in list(step.failure_signals or [])
    )
    state_regression_steps = sum(1 for step in result.steps if int(step.state_regression_count) > 0)
    low_confidence_steps = sum(
        1
        for step in result.steps
        if 0.0 < float(step.path_confidence) < low_confidence_threshold
    )
    selected_retrieval_span_ids = _ordered_selected_retrieval_span_ids(result)
    long_horizon = difficulty.strip().lower() == "long_horizon" or any(
        str(step.world_model_horizon).strip().lower() == "long_horizon"
        for step in result.steps
    )
    unsafe_ambiguous = _episode_unsafe_ambiguous(result)
    hidden_side_effect_risk = _episode_hidden_side_effect_risk(result)
    clean_success = bool(result.success) and not unsafe_ambiguous and not hidden_side_effect_risk
    return {
        "task_id": task.task_id,
        "success": bool(result.success),
        "clean_success": clean_success,
        "benchmark_family": benchmark_family,
        "difficulty": difficulty,
        "curriculum_kind": curriculum_kind,
        "memory_source": memory_source,
        "long_horizon": long_horizon,
        "termination_reason": str(result.termination_reason or "unspecified"),
        "steps": len(result.steps),
        "low_confidence_steps": low_confidence_steps,
        "unsafe_ambiguous": unsafe_ambiguous,
        "hidden_side_effect_risk": hidden_side_effect_risk,
        "contract_clean_failure_recovery_origin": contract_clean_failure_recovery_origin,
        "contract_clean_failure_recovery_origin_family": contract_clean_failure_recovery_origin_family,
        "contract_clean_failure_recovery_step_floor": contract_clean_failure_recovery_step_floor,
        "retrieval_selected_steps": sum(1 for step in result.steps if step.selected_retrieval_span_id),
        "retrieval_influenced_steps": sum(1 for step in result.steps if step.retrieval_influenced),
        "trusted_retrieval_steps": sum(1 for step in result.steps if step.trust_retrieval),
        "trusted_retrieval_carryover_steps": sum(
            1 for step in result.steps if str(step.decision_source).strip() == "trusted_retrieval_carryover_direct"
        ),
        "trusted_retrieval_carryover_verified_steps": sum(
            1
            for step in result.steps
            if str(step.decision_source).strip() == "trusted_retrieval_carryover_direct"
            and bool(step.verification.get("passed", False))
        ),
        "selected_retrieval_span_ids": selected_retrieval_span_ids,
        "last_selected_retrieval_span_id": selected_retrieval_span_ids[-1] if selected_retrieval_span_ids else "",
        "proposal_selected_steps": sum(1 for step in result.steps if str(step.proposal_source).strip()),
        "novel_command_steps": sum(1 for step in result.steps if bool(step.proposal_novel)),
        "no_state_progress_steps": no_state_progress_steps,
        "state_regression_steps": state_regression_steps,
        "failure_signals": sorted(
            {
                str(signal).strip()
                for step in result.steps
                for signal in list(step.failure_signals or [])
                if str(signal).strip()
            }
        ),
    }


def _contract_clean_failure_recovery_summary_row() -> dict[str, object]:
    return {
        "task_count": 0,
        "success_count": 0,
        "clean_success_count": 0,
        "long_horizon_task_count": 0,
        "long_horizon_success_count": 0,
        "total_steps": 0,
        "step_floor_sum": 0,
        "max_step_floor": 0,
    }


def _update_contract_clean_failure_recovery_summary_row(
    row: dict[str, object],
    summary: dict[str, object],
) -> None:
    row["task_count"] = int(row.get("task_count", 0) or 0) + 1
    row["success_count"] = int(row.get("success_count", 0) or 0) + int(bool(summary.get("success", False)))
    row["clean_success_count"] = int(row.get("clean_success_count", 0) or 0) + int(
        bool(summary.get("clean_success", False))
    )
    long_horizon = bool(summary.get("long_horizon", False))
    if long_horizon:
        row["long_horizon_task_count"] = int(row.get("long_horizon_task_count", 0) or 0) + 1
        row["long_horizon_success_count"] = int(row.get("long_horizon_success_count", 0) or 0) + int(
            bool(summary.get("success", False))
        )
    row["total_steps"] = int(row.get("total_steps", 0) or 0) + int(summary.get("steps", 0) or 0)
    step_floor = max(0, int(summary.get("contract_clean_failure_recovery_step_floor", 0) or 0))
    row["step_floor_sum"] = int(row.get("step_floor_sum", 0) or 0) + step_floor
    row["max_step_floor"] = max(int(row.get("max_step_floor", 0) or 0), step_floor)


def _finalize_contract_clean_failure_recovery_summary_row(row: dict[str, object]) -> dict[str, object]:
    task_count = int(row.get("task_count", 0) or 0)
    success_count = int(row.get("success_count", 0) or 0)
    clean_success_count = int(row.get("clean_success_count", 0) or 0)
    long_horizon_task_count = int(row.get("long_horizon_task_count", 0) or 0)
    long_horizon_success_count = int(row.get("long_horizon_success_count", 0) or 0)
    total_steps = int(row.get("total_steps", 0) or 0)
    step_floor_sum = int(row.get("step_floor_sum", 0) or 0)
    return {
        "task_count": task_count,
        "success_count": success_count,
        "clean_success_count": clean_success_count,
        "long_horizon_task_count": long_horizon_task_count,
        "long_horizon_success_count": long_horizon_success_count,
        "total_steps": total_steps,
        "max_step_floor": int(row.get("max_step_floor", 0) or 0),
        "pass_rate": round(0.0 if task_count <= 0 else success_count / task_count, 4),
        "clean_success_rate": round(0.0 if task_count <= 0 else clean_success_count / task_count, 4),
        "long_horizon_pass_rate": round(
            0.0 if long_horizon_task_count <= 0 else long_horizon_success_count / long_horizon_task_count,
            4,
        ),
        "average_steps": round(0.0 if task_count <= 0 else total_steps / task_count, 4),
        "average_step_floor": round(0.0 if task_count <= 0 else step_floor_sum / task_count, 4),
    }


def _contract_clean_failure_recovery_summary(
    task_summaries: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if not isinstance(task_summaries, dict):
        return _finalize_contract_clean_failure_recovery_summary_row({}), {}
    overall = _contract_clean_failure_recovery_summary_row()
    by_origin_benchmark_family: dict[str, dict[str, object]] = {}
    for summary in task_summaries.values():
        if not isinstance(summary, dict) or not bool(summary.get("contract_clean_failure_recovery_origin", False)):
            continue
        _update_contract_clean_failure_recovery_summary_row(overall, summary)
        origin_family = str(summary.get("contract_clean_failure_recovery_origin_family", "bounded")).strip() or "bounded"
        family_row = by_origin_benchmark_family.setdefault(origin_family, _contract_clean_failure_recovery_summary_row())
        _update_contract_clean_failure_recovery_summary_row(family_row, summary)
    finalized_overall = _finalize_contract_clean_failure_recovery_summary_row(overall)
    finalized_overall["distinct_origin_benchmark_families"] = sum(
        1 for payload in by_origin_benchmark_family.values() if int(payload.get("task_count", 0) or 0) > 0
    )
    return finalized_overall, {
        family: _finalize_contract_clean_failure_recovery_summary_row(row)
        for family, row in sorted(by_origin_benchmark_family.items())
    }


def _ordered_selected_retrieval_span_ids(result: EpisodeRecord) -> list[str]:
    span_ids: list[str] = []
    for step in result.steps:
        span_id = str(step.selected_retrieval_span_id or "").strip()
        if span_id.lower() == "none":
            span_id = ""
        if span_id and span_id not in span_ids:
            span_ids.append(span_id)
    return span_ids


def _snapshot_progress_event(progress_payload: dict[str, object]) -> dict[str, object]:
    event = {
        "event": str(progress_payload.get("event", "")).strip(),
        "step_stage": str(progress_payload.get("step_stage", "")).strip(),
        "step_subphase": str(progress_payload.get("step_subphase", "")).strip(),
        "step_action": str(progress_payload.get("step_action", "")).strip(),
        "decision_action": str(progress_payload.get("decision_action", "")).strip(),
        "step_index": int(progress_payload.get("step_index", 0) or 0),
        "step_elapsed_seconds": round(float(progress_payload.get("step_elapsed_seconds", 0.0) or 0.0), 4),
        "step_budget_seconds": round(float(progress_payload.get("step_budget_seconds", 0.0) or 0.0), 4),
        "completed_steps": int(progress_payload.get("completed_steps", 0) or 0),
    }
    event["recorded_at"] = round(time.time(), 4)
    return event


def _partial_eval_progress_snapshot(
    *,
    scheduled_primary_tasks: list,
    completed_primary_tasks: list,
    completed_primary_results: list,
    total_primary_tasks: int,
    completed_generated_tasks: list,
    completed_generated_results: list,
    generated_tasks_scheduled: int,
    phase: str,
    low_confidence_threshold: float,
    current_task: dict[str, object] | None = None,
) -> dict[str, object]:
    scheduled_task_order = [str(task.task_id) for task in scheduled_primary_tasks]
    scheduled_task_summaries = {
        str(task.task_id): {
            "benchmark_family": str(task.metadata.get("benchmark_family", "bounded")),
            "memory_source": str(task.metadata.get("memory_source", "none")),
            "workspace_subdir": str(task.workspace_subdir),
        }
        for task in scheduled_primary_tasks
    }
    primary_passed = 0
    total_by_benchmark_family: dict[str, int] = {}
    passed_by_benchmark_family: dict[str, int] = {}
    termination_reasons: dict[str, int] = {}
    completed_task_summaries: dict[str, dict[str, object]] = {}
    last_completed_task_id = ""
    last_completed_benchmark_family = ""
    for task, result in zip(completed_primary_tasks, completed_primary_results):
        summary = _task_outcome_summary(
            task,
            result,
            low_confidence_threshold=low_confidence_threshold,
        )
        completed_task_summaries[task.task_id] = summary
        family = str(summary.get("benchmark_family", "bounded"))
        total_by_benchmark_family[family] = total_by_benchmark_family.get(family, 0) + 1
        if bool(summary.get("success", False)):
            primary_passed += 1
            passed_by_benchmark_family[family] = passed_by_benchmark_family.get(family, 0) + 1
        reason = str(summary.get("termination_reason", "unspecified"))
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        last_completed_task_id = str(task.task_id)
        last_completed_benchmark_family = family
    generated_by_kind: dict[str, int] = {}
    generated_passed_by_kind: dict[str, int] = {}
    generated_by_benchmark_family: dict[str, int] = {}
    generated_passed_by_benchmark_family: dict[str, int] = {}
    generated_task_summaries: dict[str, dict[str, object]] = {}
    generated_passed = 0
    last_completed_generated_task_id = ""
    last_completed_generated_benchmark_family = ""
    completed_generated_task_summaries: dict[str, dict[str, object]] = {}
    retrieval_selected_steps = 0
    retrieval_influenced_steps = 0
    trusted_retrieval_steps = 0
    selected_retrieval_span_ids: list[str] = []
    retrieval_influenced_task_ids: list[str] = []
    for task, result in zip(completed_generated_tasks, completed_generated_results):
        summary = _task_outcome_summary(
            task,
            result,
            low_confidence_threshold=low_confidence_threshold,
        )
        completed_generated_task_summaries[task.task_id] = summary
        generated_kind = str(task.metadata.get("curriculum_kind", "unknown"))
        generated_family = str(task.metadata.get("benchmark_family", "bounded"))
        last_completed_generated_task_id = str(task.task_id)
        last_completed_generated_benchmark_family = generated_family
        generated_by_kind[generated_kind] = generated_by_kind.get(generated_kind, 0) + 1
        generated_by_benchmark_family[generated_family] = generated_by_benchmark_family.get(generated_family, 0) + 1
        if bool(summary.get("success", False)):
            generated_passed += 1
            generated_passed_by_kind[generated_kind] = generated_passed_by_kind.get(generated_kind, 0) + 1
            generated_passed_by_benchmark_family[generated_family] = (
                generated_passed_by_benchmark_family.get(generated_family, 0) + 1
            )
    for task, result in list(zip(completed_primary_tasks, completed_primary_results)) + list(
        zip(completed_generated_tasks, completed_generated_results)
    ):
        retrieval_selected_steps += sum(1 for step in result.steps if step.selected_retrieval_span_id)
        retrieval_influenced_count = sum(1 for step in result.steps if step.retrieval_influenced)
        retrieval_influenced_steps += retrieval_influenced_count
        trusted_retrieval_steps += sum(1 for step in result.steps if step.trust_retrieval)
        if retrieval_influenced_count > 0 and str(task.task_id) not in retrieval_influenced_task_ids:
            retrieval_influenced_task_ids.append(str(task.task_id))
        for span_id in _ordered_selected_retrieval_span_ids(result):
            if span_id not in selected_retrieval_span_ids:
                selected_retrieval_span_ids.append(span_id)
    contract_clean_failure_recovery_summary, contract_clean_failure_recovery_by_origin_benchmark_family = (
        _contract_clean_failure_recovery_summary(completed_generated_task_summaries)
    )
    current_task_payload = dict(current_task or {})
    return {
        "artifact_kind": "eval_partial_progress",
        "phase": str(phase).strip() or "observe",
        "scheduled_task_order": scheduled_task_order,
        "scheduled_task_summaries": scheduled_task_summaries,
        "completed_primary_tasks": len(completed_primary_results),
        "total_primary_tasks": int(total_primary_tasks),
        "remaining_primary_tasks": max(0, int(total_primary_tasks) - len(completed_primary_results)),
        "primary_passed": primary_passed,
        "primary_pass_rate": (
            0.0 if not completed_primary_results else primary_passed / len(completed_primary_results)
        ),
        "observed_benchmark_families": sorted(total_by_benchmark_family),
        "total_by_benchmark_family": total_by_benchmark_family,
        "passed_by_benchmark_family": passed_by_benchmark_family,
        "termination_reasons": termination_reasons,
        "last_completed_task_id": last_completed_task_id,
        "last_completed_benchmark_family": last_completed_benchmark_family,
        "completed_task_summaries": completed_task_summaries,
        "completed_generated_tasks": len(completed_generated_results),
        "generated_tasks_scheduled": int(generated_tasks_scheduled),
        "generated_passed": generated_passed,
        "last_completed_generated_task_id": last_completed_generated_task_id,
        "last_completed_generated_benchmark_family": last_completed_generated_benchmark_family,
        "completed_generated_task_summaries": completed_generated_task_summaries,
        "generated_by_kind": generated_by_kind,
        "generated_passed_by_kind": generated_passed_by_kind,
        "generated_by_benchmark_family": generated_by_benchmark_family,
        "generated_passed_by_benchmark_family": generated_passed_by_benchmark_family,
        "contract_clean_failure_recovery_summary": contract_clean_failure_recovery_summary,
        "contract_clean_failure_recovery_by_origin_benchmark_family": (
            contract_clean_failure_recovery_by_origin_benchmark_family
        ),
        "retrieval_selected_steps": retrieval_selected_steps,
        "retrieval_influenced_steps": retrieval_influenced_steps,
        "trusted_retrieval_steps": trusted_retrieval_steps,
        "selected_retrieval_span_ids": selected_retrieval_span_ids,
        "last_selected_retrieval_span_id": selected_retrieval_span_ids[-1] if selected_retrieval_span_ids else "",
        "retrieval_influenced_task_ids": retrieval_influenced_task_ids,
        "current_task_id": str(current_task_payload.get("task_id", "")).strip(),
        "current_task_phase": str(current_task_payload.get("phase", "")).strip(),
        "current_task_index": int(current_task_payload.get("index", 0) or 0),
        "current_task_total": int(current_task_payload.get("total", 0) or 0),
        "current_task_benchmark_family": str(current_task_payload.get("benchmark_family", "")).strip(),
        "current_task_memory_source": str(current_task_payload.get("memory_source", "")).strip(),
        "current_task_started_at": str(current_task_payload.get("started_at", "")).strip(),
        "current_task_elapsed_seconds": round(float(current_task_payload.get("elapsed_seconds", 0.0) or 0.0), 4),
        "current_task_completed_steps": int(current_task_payload.get("completed_steps", 0) or 0),
        "current_task_step_index": int(current_task_payload.get("step_index", 0) or 0),
        "current_task_step_stage": str(current_task_payload.get("step_stage", "")).strip(),
        "current_task_step_subphase": str(current_task_payload.get("step_subphase", "")).strip(),
        "current_task_step_action": str(current_task_payload.get("decision_action", "")).strip(),
        "current_task_step_elapsed_seconds": round(
            float(current_task_payload.get("step_elapsed_seconds", 0.0) or 0.0),
            4,
        ),
        "current_task_step_budget_seconds": round(
            float(current_task_payload.get("step_budget_seconds", 0.0) or 0.0),
            4,
        ),
        "current_task_verification_passed": bool(current_task_payload.get("verification_passed", False)),
        "current_task_progress_timeline": list(current_task_payload.get("progress_timeline", [])),
    }


def _proposal_metrics_by_group(
    task_trajectories: dict[str, dict[str, object]],
    *,
    group_key: str,
    default_group: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(task_trajectories, dict):
        return {}
    summary: dict[str, dict[str, object]] = {}
    for payload in task_trajectories.values():
        if not isinstance(payload, dict):
            continue
        group = str(payload.get(group_key, default_group)).strip() or default_group
        row = summary.setdefault(
            group,
            {
                "task_count": 0,
                "success_count": 0,
                "proposal_selected_steps": 0,
                "novel_command_steps": 0,
                "novel_valid_command_steps": 0,
                "novel_valid_command_rate": 0.0,
            },
        )
        row["task_count"] = int(row.get("task_count", 0) or 0) + 1
        row["success_count"] = int(row.get("success_count", 0) or 0) + int(bool(payload.get("success", False)))
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("proposal_source", "")).strip():
                row["proposal_selected_steps"] = int(row.get("proposal_selected_steps", 0) or 0) + 1
            if bool(step.get("proposal_novel", False)):
                row["novel_command_steps"] = int(row.get("novel_command_steps", 0) or 0) + 1
                if bool(step.get("verification_passed", False)):
                    row["novel_valid_command_steps"] = int(row.get("novel_valid_command_steps", 0) or 0) + 1
    for row in summary.values():
        novel_command_steps = int(row.get("novel_command_steps", 0) or 0)
        novel_valid_command_steps = int(row.get("novel_valid_command_steps", 0) or 0)
        row["novel_valid_command_rate"] = (
            0.0 if novel_command_steps <= 0 else novel_valid_command_steps / novel_command_steps
        )
    return summary


def _proposal_metrics_by_benchmark_family(
    task_trajectories: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return _proposal_metrics_by_group(
        task_trajectories,
        group_key="benchmark_family",
        default_group="bounded",
    )


def _proposal_metrics_by_difficulty(
    task_trajectories: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return _proposal_metrics_by_group(
        task_trajectories,
        group_key="difficulty",
        default_group="unknown",
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid_unit(value: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828459045, -float(value)))


def _step_world_feedback(step) -> dict[str, object]:
    proposal = dict(step.proposal_metadata or {}) if isinstance(step.proposal_metadata, dict) else {}
    latent = dict(step.latent_state_summary or {}) if isinstance(step.latent_state_summary, dict) else {}
    learned = dict(latent.get("learned_world_state", {})) if isinstance(latent.get("learned_world_state", {}), dict) else {}
    progress_signal = max(
        float(learned.get("progress_signal", 0.0) or 0.0),
        float(learned.get("world_progress_score", 0.0) or 0.0),
        float(learned.get("decoder_world_progress_score", 0.0) or 0.0),
        float(proposal.get("hybrid_world_progress_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_progress_score", 0.0) or 0.0),
    )
    risk_signal = max(
        float(learned.get("risk_signal", 0.0) or 0.0),
        float(learned.get("world_risk_score", 0.0) or 0.0),
        float(learned.get("decoder_world_risk_score", 0.0) or 0.0),
        float(proposal.get("hybrid_world_risk_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_risk_score", 0.0) or 0.0),
    )
    decoder_progress = max(
        float(learned.get("decoder_world_progress_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_progress_score", 0.0) or 0.0),
    )
    decoder_risk = max(
        float(learned.get("decoder_world_risk_score", 0.0) or 0.0),
        float(proposal.get("hybrid_decoder_world_risk_score", 0.0) or 0.0),
    )
    trusted_retrieval_alignment = max(
        float(learned.get("trusted_retrieval_alignment", 0.0) or 0.0),
        float(proposal.get("hybrid_trusted_retrieval_alignment", 0.0) or 0.0),
    )
    graph_environment_alignment = float(
        proposal.get(
            "hybrid_graph_environment_alignment",
            learned.get("graph_environment_alignment", 0.0),
        )
        or 0.0
    )
    transfer_novelty = max(
        float(learned.get("transfer_novelty", 0.0) or 0.0),
        float(proposal.get("hybrid_transfer_novelty", 0.0) or 0.0),
    )
    return {
        "present": any(
            value > 0.0
            for value in (
                progress_signal,
                risk_signal,
                decoder_progress,
                decoder_risk,
                trusted_retrieval_alignment,
                transfer_novelty,
            )
        )
        or bool(str(learned.get("source", "")).strip())
        or bool(str(proposal.get("hybrid_model_family", "")).strip()),
        "progress_signal": _clamp01(progress_signal),
        "risk_signal": _clamp01(risk_signal),
        "decoder_progress_signal": _clamp01(decoder_progress),
        "decoder_risk_signal": _clamp01(decoder_risk),
        "decoder_entropy_mean": _clamp01(
            float(
                proposal.get(
                    "hybrid_decoder_world_entropy_mean",
                    learned.get("decoder_world_entropy_mean", 0.0),
                )
                or 0.0
            )
        ),
        "transition_progress_score": _clamp01(
            max(
                float(learned.get("transition_progress_score", 0.0) or 0.0),
                float(proposal.get("hybrid_transition_progress", 0.0) or 0.0),
            )
        ),
        "transition_regression_score": _clamp01(
            max(
                float(learned.get("transition_regression_score", 0.0) or 0.0),
                float(proposal.get("hybrid_transition_regression", 0.0) or 0.0),
            )
        ),
        "trusted_retrieval_alignment": _clamp01(trusted_retrieval_alignment),
        "graph_environment_alignment": max(-1.0, min(1.0, graph_environment_alignment)),
        "transfer_novelty": _clamp01(transfer_novelty),
        "hybrid_total_score": float(proposal.get("hybrid_total_score", 0.0) or 0.0),
        "source": str(learned.get("source", "")).strip(),
        "model_family": str(proposal.get("hybrid_model_family", learned.get("model_family", ""))).strip(),
    }


def _observed_world_feedback_targets(step) -> dict[str, float]:
    verification = step.verification if isinstance(step.verification, dict) else {}
    transition = step.state_transition if isinstance(step.state_transition, dict) else {}
    verification_passed = bool(verification.get("passed", False))
    no_progress = bool(transition.get("no_progress", False)) or (
        "no_state_progress" in {str(signal).strip() for signal in list(step.failure_signals or [])}
    )
    progress_delta = float(step.state_progress_delta or 0.0)
    regression_count = int(step.state_regression_count or 0)
    observed_progress = _clamp01(
        max(1.0 if verification_passed else 0.0, _sigmoid_unit(max(0.0, progress_delta)))
    )
    if no_progress and not verification_passed:
        observed_progress = 0.0
    observed_risk = _clamp01(
        max(
            min(1.0, regression_count / 2.0),
            0.7 if no_progress else 0.0,
            0.8 if not verification_passed else 0.0,
        )
    )
    return {
        "observed_progress": observed_progress,
        "observed_risk": observed_risk,
    }


def _update_world_feedback_stats(
    summary: dict[str, dict[str, float]],
    *,
    feedback: dict[str, object],
    observed: dict[str, float],
) -> None:
    row = summary.setdefault(
        "row",
        {
            "step_count": 0.0,
            "progress_signal_sum": 0.0,
            "risk_signal_sum": 0.0,
            "decoder_progress_signal_sum": 0.0,
            "decoder_risk_signal_sum": 0.0,
            "observed_progress_sum": 0.0,
            "observed_risk_sum": 0.0,
            "progress_error_abs_sum": 0.0,
            "risk_error_abs_sum": 0.0,
            "progress_error_sq_sum": 0.0,
            "risk_error_sq_sum": 0.0,
            "decoder_progress_error_abs_sum": 0.0,
            "decoder_risk_error_abs_sum": 0.0,
            "hybrid_total_score_sum": 0.0,
        },
    )
    progress_signal = float(feedback.get("progress_signal", 0.0) or 0.0)
    risk_signal = float(feedback.get("risk_signal", 0.0) or 0.0)
    decoder_progress = float(feedback.get("decoder_progress_signal", 0.0) or 0.0)
    decoder_risk = float(feedback.get("decoder_risk_signal", 0.0) or 0.0)
    observed_progress = float(observed.get("observed_progress", 0.0) or 0.0)
    observed_risk = float(observed.get("observed_risk", 0.0) or 0.0)
    row["step_count"] += 1.0
    row["progress_signal_sum"] += progress_signal
    row["risk_signal_sum"] += risk_signal
    row["decoder_progress_signal_sum"] += decoder_progress
    row["decoder_risk_signal_sum"] += decoder_risk
    row["observed_progress_sum"] += observed_progress
    row["observed_risk_sum"] += observed_risk
    row["progress_error_abs_sum"] += abs(progress_signal - observed_progress)
    row["risk_error_abs_sum"] += abs(risk_signal - observed_risk)
    row["progress_error_sq_sum"] += (progress_signal - observed_progress) ** 2
    row["risk_error_sq_sum"] += (risk_signal - observed_risk) ** 2
    row["decoder_progress_error_abs_sum"] += abs(decoder_progress - observed_progress)
    row["decoder_risk_error_abs_sum"] += abs(decoder_risk - observed_risk)
    row["hybrid_total_score_sum"] += float(feedback.get("hybrid_total_score", 0.0) or 0.0)


def _finalize_world_feedback_row(stats: dict[str, float]) -> dict[str, object]:
    step_count = int(stats.get("step_count", 0.0) or 0)
    if step_count <= 0:
        return {
            "step_count": 0,
            "progress_calibration_mae": 0.0,
            "risk_calibration_mae": 0.0,
            "progress_calibration_brier": 0.0,
            "risk_calibration_brier": 0.0,
            "decoder_progress_calibration_mae": 0.0,
            "decoder_risk_calibration_mae": 0.0,
            "average_progress_signal": 0.0,
            "average_risk_signal": 0.0,
            "average_observed_progress": 0.0,
            "average_observed_risk": 0.0,
            "average_hybrid_total_score": 0.0,
        }
    denom = float(step_count)
    return {
        "step_count": step_count,
        "progress_calibration_mae": round(float(stats.get("progress_error_abs_sum", 0.0) or 0.0) / denom, 4),
        "risk_calibration_mae": round(float(stats.get("risk_error_abs_sum", 0.0) or 0.0) / denom, 4),
        "progress_calibration_brier": round(float(stats.get("progress_error_sq_sum", 0.0) or 0.0) / denom, 4),
        "risk_calibration_brier": round(float(stats.get("risk_error_sq_sum", 0.0) or 0.0) / denom, 4),
        "decoder_progress_calibration_mae": round(
            float(stats.get("decoder_progress_error_abs_sum", 0.0) or 0.0) / denom,
            4,
        ),
        "decoder_risk_calibration_mae": round(
            float(stats.get("decoder_risk_error_abs_sum", 0.0) or 0.0) / denom,
            4,
        ),
        "average_progress_signal": round(float(stats.get("progress_signal_sum", 0.0) or 0.0) / denom, 4),
        "average_risk_signal": round(float(stats.get("risk_signal_sum", 0.0) or 0.0) / denom, 4),
        "average_decoder_progress_signal": round(
            float(stats.get("decoder_progress_signal_sum", 0.0) or 0.0) / denom,
            4,
        ),
        "average_decoder_risk_signal": round(
            float(stats.get("decoder_risk_signal_sum", 0.0) or 0.0) / denom,
            4,
        ),
        "average_observed_progress": round(float(stats.get("observed_progress_sum", 0.0) or 0.0) / denom, 4),
        "average_observed_risk": round(float(stats.get("observed_risk_sum", 0.0) or 0.0) / denom, 4),
        "average_hybrid_total_score": round(float(stats.get("hybrid_total_score_sum", 0.0) or 0.0) / denom, 4),
    }


def _finalize_world_feedback_breakdown(
    grouped_stats: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, object]]:
    return {
        key: _finalize_world_feedback_row(dict(payload.get("row", {})))
        for key, payload in sorted(grouped_stats.items())
    }


def _finalize_world_feedback_summary(
    overall_stats: dict[str, dict[str, float]],
    family_stats: dict[str, dict[str, dict[str, float]]],
    difficulty_stats: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    overall = _finalize_world_feedback_row(dict(overall_stats.get("row", {})))
    by_family = _finalize_world_feedback_breakdown(family_stats)
    by_difficulty = _finalize_world_feedback_breakdown(difficulty_stats)
    overall["distinct_benchmark_families"] = sum(
        1 for payload in by_family.values() if int(payload.get("step_count", 0) or 0) > 0
    )
    overall["distinct_difficulties"] = sum(
        1 for payload in by_difficulty.values() if int(payload.get("step_count", 0) or 0) > 0
    )
    return overall, by_family, by_difficulty


def _long_horizon_persistence_summary(
    task_trajectories: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not isinstance(task_trajectories, dict):
        return {
            "task_count": 0,
            "long_horizon_task_count": 0,
            "long_horizon_success_count": 0,
            "long_horizon_steps": 0,
            "productive_long_horizon_steps": 0,
            "pressure_events": 0,
            "recovery_response_events": 0,
            "horizon_drop_events": 0,
            "max_long_horizon_streak": 0,
            "productive_long_horizon_step_rate": 0.0,
            "recovery_response_rate": 0.0,
            "horizon_drop_rate": 0.0,
            "long_horizon_success_rate": 0.0,
        }
    summary = {
        "task_count": 0,
        "long_horizon_task_count": 0,
        "long_horizon_success_count": 0,
        "long_horizon_steps": 0,
        "productive_long_horizon_steps": 0,
        "pressure_events": 0,
        "recovery_response_events": 0,
        "horizon_drop_events": 0,
        "max_long_horizon_streak": 0,
    }
    for payload in task_trajectories.values():
        if not isinstance(payload, dict):
            continue
        summary["task_count"] += 1
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        task_long_horizon_steps = 0
        previous_was_long_horizon = False
        current_streak = 0
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            horizon = str(step.get("world_model_horizon", "")).strip().lower()
            is_long_horizon = horizon == "long_horizon" or (
                not horizon and str(payload.get("difficulty", "")).strip().lower() == "long_horizon"
            )
            if not is_long_horizon:
                if previous_was_long_horizon:
                    summary["horizon_drop_events"] += 1
                previous_was_long_horizon = False
                current_streak = 0
                continue
            previous_was_long_horizon = True
            current_streak += 1
            summary["max_long_horizon_streak"] = max(summary["max_long_horizon_streak"], current_streak)
            task_long_horizon_steps += 1
            summary["long_horizon_steps"] += 1
            if float(step.get("state_progress_delta", 0.0) or 0.0) > 0.0:
                summary["productive_long_horizon_steps"] += 1
            has_pressure = bool(step.get("state_no_progress", False)) or bool(step.get("state_regressed", False)) or int(
                step.get("state_regression_count", 0) or 0
            ) > 0
            if has_pressure:
                summary["pressure_events"] += 1
                next_step = steps[index + 1] if index + 1 < len(steps) and isinstance(steps[index + 1], dict) else {}
                next_subgoal = str(next_step.get("active_subgoal", "")).strip()
                current_subgoal = str(step.get("active_subgoal", "")).strip()
                if str(next_step.get("acting_role", "")).strip() in {"planner", "critic"} or (
                    bool(next_subgoal) and next_subgoal != current_subgoal
                ):
                    summary["recovery_response_events"] += 1
        if task_long_horizon_steps > 0:
            summary["long_horizon_task_count"] += 1
            if bool(payload.get("success", False)):
                summary["long_horizon_success_count"] += 1
    long_horizon_steps = int(summary.get("long_horizon_steps", 0) or 0)
    pressure_events = int(summary.get("pressure_events", 0) or 0)
    long_horizon_task_count = int(summary.get("long_horizon_task_count", 0) or 0)
    summary["productive_long_horizon_step_rate"] = round(
        0.0
        if long_horizon_steps <= 0
        else float(summary.get("productive_long_horizon_steps", 0) or 0) / long_horizon_steps,
        4,
    )
    summary["recovery_response_rate"] = round(
        0.0
        if pressure_events <= 0
        else float(summary.get("recovery_response_events", 0) or 0) / pressure_events,
        4,
    )
    summary["horizon_drop_rate"] = round(
        0.0
        if long_horizon_steps <= 0
        else float(summary.get("horizon_drop_events", 0) or 0) / long_horizon_steps,
        4,
    )
    summary["long_horizon_success_rate"] = round(
        0.0
        if long_horizon_task_count <= 0
        else float(summary.get("long_horizon_success_count", 0) or 0) / long_horizon_task_count,
        4,
    )
    return summary


def _transfer_alignment_summary(
    task_trajectories: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not isinstance(task_trajectories, dict):
        return {
            "task_count": 0,
            "transfer_task_count": 0,
            "transfer_step_count": 0,
            "verified_transfer_steps": 0,
            "trusted_retrieval_alignment_mean": 0.0,
            "graph_environment_alignment_mean": 0.0,
            "safe_transfer_step_rate": 0.0,
            "unsafe_transfer_step_rate": 0.0,
            "verified_transfer_step_rate": 0.0,
        }
    summary = {
        "task_count": 0,
        "transfer_task_count": 0,
        "transfer_step_count": 0,
        "verified_transfer_steps": 0,
        "trusted_retrieval_alignment_sum": 0.0,
        "graph_environment_alignment_sum": 0.0,
        "safe_transfer_steps": 0,
        "unsafe_transfer_steps": 0,
    }
    for payload in task_trajectories.values():
        if not isinstance(payload, dict):
            continue
        summary["task_count"] += 1
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        task_has_transfer = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            world_feedback = step.get("world_feedback", {})
            if not isinstance(world_feedback, dict):
                world_feedback = {}
            transfer_novelty = float(world_feedback.get("transfer_novelty", 0.0) or 0.0)
            if transfer_novelty <= 0.0:
                continue
            task_has_transfer = True
            summary["transfer_step_count"] += 1
            summary["trusted_retrieval_alignment_sum"] += float(
                world_feedback.get("trusted_retrieval_alignment", 0.0) or 0.0
            )
            environment_alignment = float(world_feedback.get("graph_environment_alignment", 0.0) or 0.0)
            summary["graph_environment_alignment_sum"] += environment_alignment
            if environment_alignment >= 0.0:
                summary["safe_transfer_steps"] += 1
            else:
                summary["unsafe_transfer_steps"] += 1
            if bool(step.get("verification_passed", False)):
                summary["verified_transfer_steps"] += 1
        if task_has_transfer:
            summary["transfer_task_count"] += 1
    transfer_step_count = int(summary.get("transfer_step_count", 0) or 0)
    return {
        "task_count": int(summary.get("task_count", 0) or 0),
        "transfer_task_count": int(summary.get("transfer_task_count", 0) or 0),
        "transfer_step_count": transfer_step_count,
        "verified_transfer_steps": int(summary.get("verified_transfer_steps", 0) or 0),
        "trusted_retrieval_alignment_mean": round(
            0.0
            if transfer_step_count <= 0
            else float(summary.get("trusted_retrieval_alignment_sum", 0.0) or 0.0) / transfer_step_count,
            4,
        ),
        "graph_environment_alignment_mean": round(
            0.0
            if transfer_step_count <= 0
            else float(summary.get("graph_environment_alignment_sum", 0.0) or 0.0) / transfer_step_count,
            4,
        ),
        "safe_transfer_step_rate": round(
            0.0
            if transfer_step_count <= 0
            else float(summary.get("safe_transfer_steps", 0) or 0) / transfer_step_count,
            4,
        ),
        "unsafe_transfer_step_rate": round(
            0.0
            if transfer_step_count <= 0
            else float(summary.get("unsafe_transfer_steps", 0) or 0) / transfer_step_count,
            4,
        ),
        "verified_transfer_step_rate": round(
            0.0
            if transfer_step_count <= 0
            else float(summary.get("verified_transfer_steps", 0) or 0) / transfer_step_count,
            4,
        ),
    }


class _ForcedFailurePolicy(Policy):
    def decide(self, state):
        task = state.task
        forbidden_files = [
            str(path).strip()
            for path in getattr(task, "forbidden_files", [])
            if str(path).strip()
        ]
        if forbidden_files:
            target = shlex.quote(forbidden_files[0])
            return ActionDecision(
                thought="Probe the forbidden artifact contract to seed a recovery task.",
                action="code_execute",
                content=f"test ! -f {target}",
                done=False,
                decision_source="failure_seed_contract_probe",
            )
        expected_files = [
            str(path).strip()
            for path in getattr(task, "expected_files", [])
            if str(path).strip()
        ]
        if not expected_files:
            expected_files = [
                str(path).strip()
                for path in getattr(task, "expected_file_contents", {}).keys()
                if str(path).strip()
            ]
        if expected_files:
            target = shlex.quote(expected_files[0])
            return ActionDecision(
                thought="Probe the missing expected artifact contract to seed a recovery task.",
                action="code_execute",
                content=f"test -f {target}",
                done=False,
                decision_source="failure_seed_contract_probe",
            )
        success_command = str(getattr(task, "success_command", "")).strip()
        if success_command:
            return ActionDecision(
                thought="Probe the verifier contract directly to seed a recovery task.",
                action="code_execute",
                content=success_command,
                done=False,
                decision_source="failure_seed_contract_probe",
            )
        return ActionDecision(
            thought="Use a bounded missing-artifact probe to seed a recovery task.",
            action="code_execute",
            content="test -f .agent_kernel_failure_seed_probe",
            done=False,
            decision_source="failure_seed_contract_probe",
        )


def _task_allowed_for_eval(task, config: KernelConfig) -> bool:
    operator_policy = operator_policy_snapshot(config)
    if (
        config.provider == "mock"
        and task.metadata.get("requires_retrieval", False)
        and str(task.metadata.get("memory_source", "none")) not in {"episode", "verifier"}
    ):
        return False
    if not bool(operator_policy.get("unattended_allow_git_commands", False)) and task.metadata.get("requires_git", False):
        return False
    workflow_guard = task.metadata.get("workflow_guard", {})
    if (
        not bool(operator_policy.get("unattended_allow_generated_path_mutations", False))
        and isinstance(workflow_guard, dict)
        and workflow_guard.get("touches_generated_paths")
    ):
        return False
    required_capabilities = declared_task_capabilities(task)
    if required_capabilities and any(not capability_enabled(config, capability) for capability in required_capabilities):
        return False
    return True


def _scoped_path(path: Path, scope: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}_{scope}{path.suffix}")
    return path / scope


def _copytree_would_recurse(src: Path, dst: Path) -> bool:
    try:
        src_abs = src.resolve()
    except OSError:
        src_abs = src.absolute()
    try:
        dst_abs = dst.resolve()
    except OSError:
        dst_abs = dst.absolute()
    if src_abs == dst_abs:
        return True
    try:
        dst_abs.relative_to(src_abs)
        return True
    except ValueError:
        return False


def _copytree_contents(src: Path, dst: Path) -> None:
    try:
        dst_abs = dst.resolve()
    except OSError:
        dst_abs = dst.absolute()
    children = []
    for child in src.iterdir():
        try:
            child_abs = child.resolve()
        except OSError:
            child_abs = child.absolute()
        if child_abs == dst_abs:
            continue
        if child.name in _SCOPED_DIRECTORY_NAMES:
            continue
        children.append(child)
    dst.mkdir(parents=True, exist_ok=True)
    for child in children:
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(
                child,
                target,
                dirs_exist_ok=True,
                copy_function=_seed_file_copy,
                ignore=_ignore_scoped_directory_names,
            )
        else:
            _seed_file_copy(child, target)


def _seed_file_copy(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.exists() and dst_path.exists() and os.path.samefile(src_path, dst_path):
            return
    except OSError:
        pass
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink(missing_ok=True)
    try:
        os.link(src_path, dst_path)
    except OSError:
        try:
            if src_path.exists() and dst_path.exists() and os.path.samefile(src_path, dst_path):
                return
        except OSError:
            pass
        shutil.copy2(src_path, dst_path)


def _ignore_scoped_directory_names(_root: str, names: list[str]) -> list[str]:
    return [name for name in names if name in _SCOPED_DIRECTORY_NAMES]


def _cleanup_scoped_runtime_state(config: KernelConfig) -> None:
    for path in (config.run_checkpoints_dir, config.unattended_workspace_snapshot_root):
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _scoped_config(base_config: KernelConfig, scope: str, **overrides) -> KernelConfig:
    config = replace(
        base_config,
        workspace_root=_scoped_path(base_config.workspace_root, scope),
        trajectories_root=_scoped_path(base_config.trajectories_root, scope),
        skills_path=_scoped_path(base_config.skills_path, scope),
        operator_classes_path=_scoped_path(base_config.operator_classes_path, scope),
        tool_candidates_path=_scoped_path(base_config.tool_candidates_path, scope),
        benchmark_candidates_path=_scoped_path(base_config.benchmark_candidates_path, scope),
        retrieval_proposals_path=_scoped_path(base_config.retrieval_proposals_path, scope),
        retrieval_asset_bundle_path=_scoped_path(base_config.retrieval_asset_bundle_path, scope),
        tolbert_model_artifact_path=_scoped_path(base_config.tolbert_model_artifact_path, scope),
        tolbert_supervised_datasets_dir=base_config.tolbert_supervised_datasets_dir,
        tolbert_liftoff_report_path=_scoped_path(base_config.tolbert_liftoff_report_path, scope),
        verifier_contracts_path=_scoped_path(base_config.verifier_contracts_path, scope),
        prompt_proposals_path=_scoped_path(base_config.prompt_proposals_path, scope),
        world_model_proposals_path=_scoped_path(base_config.world_model_proposals_path, scope),
        trust_proposals_path=_scoped_path(base_config.trust_proposals_path, scope),
        recovery_proposals_path=_scoped_path(base_config.recovery_proposals_path, scope),
        delegation_proposals_path=_scoped_path(base_config.delegation_proposals_path, scope),
        operator_policy_proposals_path=_scoped_path(base_config.operator_policy_proposals_path, scope),
        transition_model_proposals_path=_scoped_path(base_config.transition_model_proposals_path, scope),
        curriculum_proposals_path=_scoped_path(base_config.curriculum_proposals_path, scope),
        improvement_cycles_path=_scoped_path(base_config.improvement_cycles_path, scope),
        candidate_artifacts_root=_scoped_path(base_config.candidate_artifacts_root, scope),
        improvement_reports_dir=_scoped_path(base_config.improvement_reports_dir, scope),
        run_reports_dir=_scoped_path(base_config.run_reports_dir, scope),
        capability_modules_path=_scoped_path(base_config.capability_modules_path, scope),
        run_checkpoints_dir=_scoped_path(base_config.run_checkpoints_dir, scope),
        delegated_job_queue_path=_scoped_path(base_config.delegated_job_queue_path, scope),
        delegated_job_runtime_state_path=_scoped_path(base_config.delegated_job_runtime_state_path, scope),
        unattended_workspace_snapshot_root=_scoped_path(base_config.unattended_workspace_snapshot_root, scope),
        unattended_trust_ledger_path=_scoped_path(base_config.unattended_trust_ledger_path, scope),
    )
    for src, dst in (
        (base_config.skills_path, config.skills_path),
        (base_config.operator_classes_path, config.operator_classes_path),
        (base_config.tool_candidates_path, config.tool_candidates_path),
        (base_config.benchmark_candidates_path, config.benchmark_candidates_path),
        (base_config.retrieval_proposals_path, config.retrieval_proposals_path),
        (base_config.retrieval_asset_bundle_path, config.retrieval_asset_bundle_path),
        (base_config.tolbert_model_artifact_path, config.tolbert_model_artifact_path),
        (base_config.verifier_contracts_path, config.verifier_contracts_path),
        (base_config.prompt_proposals_path, config.prompt_proposals_path),
        (base_config.world_model_proposals_path, config.world_model_proposals_path),
        (base_config.trust_proposals_path, config.trust_proposals_path),
        (base_config.recovery_proposals_path, config.recovery_proposals_path),
        (base_config.delegation_proposals_path, config.delegation_proposals_path),
        (base_config.operator_policy_proposals_path, config.operator_policy_proposals_path),
        (base_config.transition_model_proposals_path, config.transition_model_proposals_path),
        (base_config.curriculum_proposals_path, config.curriculum_proposals_path),
        (base_config.improvement_cycles_path, config.improvement_cycles_path),
        (base_config.capability_modules_path, config.capability_modules_path),
        (base_config.delegated_job_queue_path, config.delegated_job_queue_path),
        (base_config.delegated_job_runtime_state_path, config.delegated_job_runtime_state_path),
        (base_config.unattended_trust_ledger_path, config.unattended_trust_ledger_path),
    ):
        if src.exists():
            _seed_file_copy(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (base_config.run_checkpoints_dir, config.run_checkpoints_dir),
        (base_config.unattended_workspace_snapshot_root, config.unattended_workspace_snapshot_root),
    ):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if _copytree_would_recurse(src, dst):
                # When the scoped destination lives under the source tree (typical for
                # `.../workspaces/<scope>`), copying contents can explode in size by
                # repeatedly re-copying prior scoped runs and preview workspaces.
                # For eval/improvement scoping we only need an empty destination root.
                dst.mkdir(parents=True, exist_ok=True)
            else:
                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=True,
                    copy_function=_seed_file_copy,
                    ignore=_ignore_scoped_directory_names,
                )
        else:
            dst.mkdir(parents=True, exist_ok=True)
    for directory in (
        config.workspace_root,
        config.trajectories_root,
        config.candidate_artifacts_root,
        config.improvement_reports_dir,
        config.run_reports_dir,
        config.run_checkpoints_dir,
        config.unattended_workspace_snapshot_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def scoped_eval_config(base_config: KernelConfig, scope: str, **overrides) -> KernelConfig:
    return _scoped_config(base_config, scope, **overrides)


def scoped_improvement_cycle_config(
    base_config: KernelConfig,
    scope: str,
    *,
    seed_from_base: bool = True,
    seed_improvement_cycles_from_base: bool = False,
    **overrides,
) -> KernelConfig:
    config = replace(
        base_config,
        workspace_root=_scoped_path(base_config.workspace_root, scope),
        improvement_cycles_path=_scoped_path(base_config.improvement_cycles_path, scope),
        candidate_artifacts_root=_scoped_path(base_config.candidate_artifacts_root, scope),
        improvement_reports_dir=_scoped_path(base_config.improvement_reports_dir, scope),
        run_reports_dir=_scoped_path(base_config.run_reports_dir, scope),
        learning_artifacts_path=_scoped_path(base_config.learning_artifacts_path, scope),
        run_checkpoints_dir=_scoped_path(base_config.run_checkpoints_dir, scope),
        delegated_job_queue_path=_scoped_path(base_config.delegated_job_queue_path, scope),
        delegated_job_runtime_state_path=_scoped_path(base_config.delegated_job_runtime_state_path, scope),
        unattended_workspace_snapshot_root=_scoped_path(base_config.unattended_workspace_snapshot_root, scope),
        unattended_trust_ledger_path=_scoped_path(base_config.unattended_trust_ledger_path, scope),
        tolbert_model_artifact_path=_scoped_path(base_config.tolbert_model_artifact_path, scope),
        tolbert_supervised_datasets_dir=base_config.tolbert_supervised_datasets_dir,
        tolbert_liftoff_report_path=_scoped_path(base_config.tolbert_liftoff_report_path, scope),
        persist_episode_memory=False,
    )
    if seed_from_base:
        seeded_files = [
            (base_config.delegated_job_queue_path, config.delegated_job_queue_path),
            (base_config.delegated_job_runtime_state_path, config.delegated_job_runtime_state_path),
            (base_config.unattended_trust_ledger_path, config.unattended_trust_ledger_path),
            (base_config.tolbert_model_artifact_path, config.tolbert_model_artifact_path),
            (base_config.tolbert_liftoff_report_path, config.tolbert_liftoff_report_path),
        ]
        if seed_improvement_cycles_from_base:
            seeded_files.insert(0, (base_config.improvement_cycles_path, config.improvement_cycles_path))
        for src, dst in seeded_files:
            if src.exists():
                _seed_file_copy(src, dst)
    for src, dst in (
        (base_config.run_checkpoints_dir, config.run_checkpoints_dir),
        (base_config.unattended_workspace_snapshot_root, config.unattended_workspace_snapshot_root),
    ):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if _copytree_would_recurse(src, dst):
                # Scoped improvement previews place their outputs under the shared
                # trajectories tree, so seeding from the base checkpoints/snapshots
                # can recursively copy prior scoped runs into themselves.
                dst.mkdir(parents=True, exist_ok=True)
            else:
                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=True,
                    copy_function=_seed_file_copy,
                    ignore=_ignore_scoped_directory_names,
                )
        else:
            dst.mkdir(parents=True, exist_ok=True)
    for directory in (
        config.workspace_root,
        config.candidate_artifacts_root,
        config.improvement_reports_dir,
        config.run_reports_dir,
        config.learning_artifacts_path.parent,
        config.tolbert_model_artifact_path.parent,
        config.tolbert_liftoff_report_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def run_eval(
    config: KernelConfig | None = None,
    *,
    include_primary_tasks: bool = True,
    include_discovered_tasks: bool = False,
    include_generated: bool = False,
    include_failure_generated: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_benchmark_candidates: bool = False,
    include_verifier_candidates: bool = False,
    task_limit: int | None = None,
    priority_benchmark_families: Sequence[str] | None = None,
    priority_benchmark_family_weights: dict[str, object] | None = None,
    prefer_low_cost_tasks: bool = False,
    restrict_to_priority_benchmark_families: bool = False,
    progress_label: str | None = None,
    progress_snapshot_path: Path | None = None,
    skill_transfer_target_map: dict[str, str] | None = None,
    operator_target_map: dict[str, str] | None = None,
    generated_success_seed_documents_path: str = "",
    generated_success_seed_workspace_root: str = "",
    generated_success_seed_output_path: str = "",
    allow_generated_success_seed_fallback: bool = True,
    max_generated_success_schedule_tasks: int = 0,
    max_generated_failure_schedule_tasks: int = 0,
    write_unattended_reports: bool = False,
    recovery_variant_strategy_family: str = "",
    surface_shared_repo_bundles: bool = True,
) -> EvalMetrics:
    active_config = config or KernelConfig()
    requested_priority_families: list[str] = []
    for family in list(priority_benchmark_families or []) + list((priority_benchmark_family_weights or {}).keys()):
        normalized_family = str(family).strip()
        if normalized_family and normalized_family not in requested_priority_families:
            requested_priority_families.append(normalized_family)
    tasks: list = []
    kernel: AgentKernel | None = None
    try:
        if include_primary_tasks:
            try:
                bank = TaskBank(config=active_config)
            except TypeError:
                bank = TaskBank()
            tasks = [task for task in bank.list() if _task_allowed_for_eval(task, active_config)]
            if surface_shared_repo_bundles:
                tasks.extend(
                    task
                    for task in _surface_shared_repo_worker_tasks(
                        bank,
                        requested_priority_families=requested_priority_families,
                    )
                    if _task_allowed_for_eval(task, active_config)
                )
                tasks.extend(
                    task
                    for task in _surface_shared_repo_integrator_tasks(
                        bank,
                        requested_priority_families=requested_priority_families,
                    )
                    if _task_allowed_for_eval(task, active_config)
                )
            tasks = sorted(
                tasks,
                key=lambda task: (
                    int(task.metadata.get("shared_repo_order", 0)),
                    task.task_id,
                ),
            )
            if include_episode_memory:
                tasks.extend(load_episode_replay_tasks(active_config.trajectories_root))
            if include_skill_memory:
                tasks.extend(load_skill_replay_tasks(active_config.skills_path))
            if include_skill_transfer:
                tasks.extend(
                    load_skill_transfer_tasks(
                        active_config.skills_path,
                        target_task_by_source=skill_transfer_target_map,
                    )
                )
            if include_operator_memory:
                tasks.extend(
                    load_operator_replay_tasks(
                        active_config.operator_classes_path,
                        target_task_by_operator=operator_target_map,
                    )
                )
            if include_tool_memory:
                tasks.extend(load_tool_replay_tasks(active_config.tool_candidates_path))
            if include_verifier_memory:
                tasks.extend(load_verifier_replay_tasks(active_config.trajectories_root, active_config.skills_path))
            if include_benchmark_candidates:
                tasks.extend(load_benchmark_candidate_tasks(active_config.benchmark_candidates_path))
            if include_verifier_candidates:
                tasks.extend(load_verifier_candidate_tasks(active_config.verifier_contracts_path))
            if include_discovered_tasks:
                tasks.extend(load_discovered_tasks(active_config.trajectories_root))
                tasks.extend(load_transition_pressure_tasks(active_config.trajectories_root))
            tasks = [task for task in tasks if _task_allowed_for_eval(task, active_config)]
            if restrict_to_priority_benchmark_families and requested_priority_families:
                requested_priority_family_set = set(requested_priority_families)
                tasks = [
                    task
                    for task in tasks
                    if (str(getattr(task, "metadata", {}).get("benchmark_family", "bounded")).strip() or "bounded")
                    in requested_priority_family_set
                ]
                tasks = _drop_retrieval_companions_when_sources_present(tasks)
            if task_limit is not None and task_limit > 0 and len(tasks) > task_limit:
                effective_priority_families = list(requested_priority_families)
                if not effective_priority_families:
                    effective_priority_families = _default_priority_families_for_compare(tasks)
                tasks = _limit_tasks_for_compare(
                    tasks,
                    task_limit,
                    priority_families=effective_priority_families,
                    priority_family_weights=priority_benchmark_family_weights,
                    prefer_low_cost_tasks=prefer_low_cost_tasks,
                    required_executable_families=active_config.unattended_trust_required_benchmark_families,
                )
        completed_primary_tasks: list = []
        completed_primary_results: list = []
        completed_generated_tasks: list = []
        completed_generated_results: list = []
        generated_tasks_scheduled = 0
        current_task_state: dict[str, object] = {
            "task_id": "",
            "phase": "",
            "index": 0,
            "total": 0,
            "benchmark_family": "",
            "memory_source": "",
            "started_at": "",
            "elapsed_seconds": 0.0,
            "started_monotonic": 0.0,
            "completed_steps": 0,
            "step_index": 0,
            "step_stage": "",
            "step_subphase": "",
            "decision_action": "",
            "step_elapsed_seconds": 0.0,
            "step_budget_seconds": 0.0,
            "verification_passed": False,
            "progress_timeline": [],
        }
        current_task_lock = threading.Lock()
        progress_write_lock = threading.Lock()
        heartbeat_stop = threading.Event()
        task_started_monotonic_by_id: dict[str, float] = {}

        def _annotate_observed_runtime(task, result: EpisodeRecord, *, phase_name: str) -> None:
            task_id = str(getattr(task, "task_id", "")).strip()
            started_monotonic = float(task_started_monotonic_by_id.pop(task_id, 0.0) or 0.0)
            if started_monotonic <= 0.0:
                return
            observed_runtime_seconds = round(max(0.0, time.monotonic() - started_monotonic), 4)
            result.task_metadata = dict(result.task_metadata or {})
            result.task_metadata["observed_runtime_seconds"] = observed_runtime_seconds
            result.task_metadata["observed_runtime_phase"] = str(phase_name).strip() or "primary"
            task_contract = dict(result.task_contract or {})
            contract_metadata = dict(task_contract.get("metadata", {}) or {})
            contract_metadata.setdefault("observed_runtime_seconds", observed_runtime_seconds)
            contract_metadata.setdefault("observed_runtime_phase", str(phase_name).strip() or "primary")
            task_contract["metadata"] = contract_metadata
            result.task_contract = task_contract

        def _set_current_task(task=None, *, phase_name: str = "", index: int = 0, total: int = 0) -> None:
            with current_task_lock:
                if task is None:
                    current_task_state.update(
                        {
                            "task_id": "",
                            "phase": "",
                            "index": 0,
                            "total": 0,
                            "benchmark_family": "",
                            "memory_source": "",
                            "started_at": "",
                            "elapsed_seconds": 0.0,
                            "started_monotonic": 0.0,
                            "completed_steps": 0,
                            "step_index": 0,
                            "step_stage": "",
                            "step_subphase": "",
                            "decision_action": "",
                            "step_elapsed_seconds": 0.0,
                            "step_budget_seconds": 0.0,
                            "verification_passed": False,
                            "progress_timeline": [],
                        }
                    )
                    return
                metadata = getattr(task, "metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                current_task_state.update(
                    {
                        "task_id": str(task.task_id),
                        "phase": str(phase_name).strip() or "primary",
                        "index": int(index),
                        "total": int(total),
                        "benchmark_family": str(metadata.get("benchmark_family", "bounded")).strip() or "bounded",
                        "memory_source": str(metadata.get("memory_source", "none")).strip() or "none",
                        "started_at": str(time.time()),
                        "elapsed_seconds": 0.0,
                        "started_monotonic": time.monotonic(),
                        "completed_steps": 0,
                        "step_index": 0,
                        "step_stage": "",
                        "step_subphase": "",
                        "decision_action": "",
                        "step_elapsed_seconds": 0.0,
                        "step_budget_seconds": 0.0,
                        "verification_passed": False,
                        "progress_timeline": [],
                    }
                )
                task_started_monotonic_by_id[str(task.task_id)] = time.monotonic()

        def _update_current_task_progress(progress_payload: dict[str, object]) -> None:
            with current_task_lock:
                if not str(current_task_state.get("task_id", "")).strip():
                    return
                if "completed_steps" in progress_payload:
                    current_task_state["completed_steps"] = int(progress_payload.get("completed_steps", 0) or 0)
                if "step_index" in progress_payload:
                    current_task_state["step_index"] = int(progress_payload.get("step_index", 0) or 0)
                if "step_stage" in progress_payload:
                    current_task_state["step_stage"] = str(progress_payload.get("step_stage", "")).strip()
                if "step_subphase" in progress_payload:
                    current_task_state["step_subphase"] = str(progress_payload.get("step_subphase", "")).strip()
                if "decision_action" in progress_payload:
                    current_task_state["decision_action"] = str(progress_payload.get("decision_action", "")).strip()
                if "step_elapsed_seconds" in progress_payload:
                    current_task_state["step_elapsed_seconds"] = float(
                        progress_payload.get("step_elapsed_seconds", 0.0) or 0.0
                    )
                if "step_budget_seconds" in progress_payload:
                    current_task_state["step_budget_seconds"] = float(
                        progress_payload.get("step_budget_seconds", 0.0) or 0.0
                    )
                if "verification_passed" in progress_payload:
                    current_task_state["verification_passed"] = bool(progress_payload.get("verification_passed", False))
                timeline = list(current_task_state.get("progress_timeline", []) or [])
                timeline.append(_snapshot_progress_event(progress_payload))
                if len(timeline) > _PROGRESS_TIMELINE_MAX_EVENTS:
                    timeline = timeline[-_PROGRESS_TIMELINE_MAX_EVENTS :]
                current_task_state["progress_timeline"] = timeline

        def _current_task_snapshot() -> dict[str, object]:
            with current_task_lock:
                snapshot = dict(current_task_state)
            started_monotonic = float(snapshot.pop("started_monotonic", 0.0) or 0.0)
            if started_monotonic > 0.0:
                elapsed_seconds = round(max(0.0, time.monotonic() - started_monotonic), 4)
                snapshot["elapsed_seconds"] = elapsed_seconds
                if str(snapshot.get("step_stage", "")).strip() == "context_compile":
                    current_step_elapsed_seconds = float(snapshot.get("step_elapsed_seconds", 0.0) or 0.0)
                    if elapsed_seconds > current_step_elapsed_seconds:
                        snapshot["step_elapsed_seconds"] = elapsed_seconds
            else:
                snapshot["elapsed_seconds"] = round(float(snapshot.get("elapsed_seconds", 0.0) or 0.0), 4)
            snapshot["progress_timeline"] = list(snapshot.get("progress_timeline", []) or [])
            return snapshot

        def _emit_progress_snapshot(phase: str) -> None:
            if progress_snapshot_path is None:
                return
            with progress_write_lock:
                atomic_write_json(
                    progress_snapshot_path,
                    _partial_eval_progress_snapshot(
                        scheduled_primary_tasks=tasks,
                        completed_primary_tasks=completed_primary_tasks,
                        completed_primary_results=completed_primary_results,
                        total_primary_tasks=len(tasks),
                        completed_generated_tasks=completed_generated_tasks,
                        completed_generated_results=completed_generated_results,
                        generated_tasks_scheduled=generated_tasks_scheduled,
                        phase=phase,
                        low_confidence_threshold=active_config.tolbert_low_confidence_widen_threshold,
                        current_task=_current_task_snapshot(),
                    ),
                )

        def _progress_heartbeat() -> None:
            while not heartbeat_stop.wait(_PROGRESS_HEARTBEAT_INTERVAL_SECONDS):
                snapshot = _current_task_snapshot()
                if not str(snapshot.get("task_id", "")).strip():
                    continue
                _emit_progress_snapshot(str(snapshot.get("phase", "")).strip() or "primary")

        heartbeat_thread = None
        if progress_snapshot_path is not None:
            heartbeat_thread = threading.Thread(
                target=_progress_heartbeat,
                name="eval-progress-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()

        _emit_progress_snapshot("scheduled")

        results: list[EpisodeRecord] = []
        if include_primary_tasks:
            kernel = AgentKernel(config=active_config)
            results = _run_tasks_with_progress(
                tasks,
                kernel,
                progress_label=progress_label,
                report_config=active_config if write_unattended_reports else None,
                on_task_start=lambda task, index, total: (
                    _set_current_task(task, phase_name="primary", index=index, total=total),
                    _emit_progress_snapshot("primary"),
                ),
                on_task_progress=lambda _task, progress_payload, _index, _total: (
                    _update_current_task_progress(progress_payload),
                    _emit_progress_snapshot("primary"),
                ),
                on_task_complete=lambda _task, _result, _index, _total: _set_current_task(None),
                on_result=lambda task, result, _index, _total: (
                    _annotate_observed_runtime(task, result, phase_name="primary"),
                    completed_primary_tasks.append(task),
                    completed_primary_results.append(result),
                    _emit_progress_snapshot("primary"),
                ),
            )
            if generated_success_seed_output_path:
                _write_success_seed_bundle(
                    generated_success_seed_output_path,
                    primary_tasks=completed_primary_tasks,
                    primary_results=completed_primary_results,
                    generated_tasks=completed_generated_tasks,
                    generated_results=completed_generated_results,
                )
        generated_tasks = []
        generated_results = []
        if include_generated or include_failure_generated:
            engine = CurriculumEngine(memory_root=active_config.trajectories_root, config=active_config)
            strategy_controls = _recovery_strategy_curriculum_controls(recovery_variant_strategy_family)
            if strategy_controls:
                merged_controls = dict(engine._curriculum_controls() or {})
                merged_controls.update(strategy_controls)
                engine._curriculum_controls_cache = merged_controls
            if include_generated:
                _emit_eval_progress(progress_label, "phase=generated_success_schedule")
                success_generated_config = _scoped_config(active_config, "generated_success")
                success_seed_results = results
                if not success_seed_results and allow_generated_success_seed_fallback:
                    success_seed_results = _load_generated_success_seed_episodes(
                        Path(generated_success_seed_documents_path)
                        if generated_success_seed_documents_path
                        else active_config.trajectories_root,
                        workspace_root=(
                            Path(generated_success_seed_workspace_root)
                            if generated_success_seed_workspace_root
                            else None
                        ),
                    )
                success_seed_results = engine.schedule_generated_seed_episodes(
                    success_seed_results,
                    curriculum_kind="adjacent_success",
                )
                if max_generated_success_schedule_tasks > 0:
                    success_seed_results = success_seed_results[:max_generated_success_schedule_tasks]
                success_generated_tasks = [engine.generate_followup_task(result) for result in success_seed_results]
                _emit_eval_progress(
                    progress_label,
                    f"phase=generated_success total={len(success_generated_tasks)}",
                )
                generated_tasks.extend(success_generated_tasks)
                generated_tasks_scheduled = len(generated_tasks)
                _emit_progress_snapshot("generated_success_schedule")
                if success_generated_tasks:
                    success_generated_kernel = AgentKernel(config=success_generated_config)
                    try:
                        generated_results.extend(
                            _run_tasks_with_progress(
                                success_generated_tasks,
                                success_generated_kernel,
                                progress_label=progress_label,
                                phase="generated_success",
                                report_config=success_generated_config if write_unattended_reports else None,
                                on_task_start=lambda task, index, total: (
                                    _set_current_task(task, phase_name="generated_success", index=index, total=total),
                                    _emit_progress_snapshot("generated_success"),
                                ),
                                on_task_progress=lambda _task, progress_payload, _index, _total: (
                                    _update_current_task_progress(progress_payload),
                                    _emit_progress_snapshot("generated_success"),
                                ),
                                on_task_complete=lambda _task, _result, _index, _total: _set_current_task(None),
                                on_result=lambda task, result, _index, _total: (
                                    _annotate_observed_runtime(task, result, phase_name="generated_success"),
                                    completed_generated_tasks.append(task),
                                    completed_generated_results.append(result),
                                    _emit_progress_snapshot("generated_success"),
                                    (
                                        _write_success_seed_bundle(
                                            generated_success_seed_output_path,
                                            primary_tasks=completed_primary_tasks,
                                            primary_results=completed_primary_results,
                                            generated_tasks=completed_generated_tasks,
                                            generated_results=completed_generated_results,
                                        )
                                        if generated_success_seed_output_path
                                        else None
                                    ),
                                ),
                            )
                        )
                    finally:
                        success_generated_kernel.close()
                        _cleanup_scoped_runtime_state(success_generated_config)
                else:
                    _cleanup_scoped_runtime_state(success_generated_config)
                _emit_eval_progress(
                    progress_label,
                    f"phase=generated_success complete total={len(success_generated_tasks)}",
                )
            if include_failure_generated:
                _emit_eval_progress(progress_label, f"phase=generated_failure_seed total={len(tasks)}")
                failure_seed_config = _scoped_config(active_config, "generated_failure_seed")
                failing_kernel = AgentKernel(config=failure_seed_config, policy=_ForcedFailurePolicy())
                failure_generated_config = _scoped_config(active_config, "generated_failure")
                try:
                    failure_seeds = _run_tasks_with_progress(
                        tasks,
                        failing_kernel,
                        progress_label=progress_label,
                        phase="generated_failure_seed",
                        report_config=failure_seed_config if write_unattended_reports else None,
                    )
                finally:
                    failing_kernel.close()
                    _cleanup_scoped_runtime_state(failure_seed_config)
                _emit_eval_progress(
                    progress_label,
                    f"phase=generated_failure_seed complete total={len(tasks)}",
                )
                failure_seed_results = engine.schedule_generated_seed_episodes(
                    failure_seeds,
                    curriculum_kind="failure_recovery",
                )
                if max_generated_failure_schedule_tasks > 0:
                    failure_seed_results = failure_seed_results[:max_generated_failure_schedule_tasks]
                failure_generated_tasks = [engine.generate_followup_task(result) for result in failure_seed_results]
                _emit_eval_progress(
                    progress_label,
                    f"phase=generated_failure total={len(failure_generated_tasks)}",
                )
                generated_tasks.extend(failure_generated_tasks)
                generated_tasks_scheduled = len(generated_tasks)
                _emit_progress_snapshot("generated_failure_schedule")
                if failure_generated_tasks:
                    failure_generated_kernel = AgentKernel(config=failure_generated_config)
                    try:
                        generated_results.extend(
                            _run_tasks_with_progress(
                                failure_generated_tasks,
                                failure_generated_kernel,
                                progress_label=progress_label,
                                phase="generated_failure",
                                report_config=failure_generated_config if write_unattended_reports else None,
                                on_task_start=lambda task, index, total: (
                                    _set_current_task(task, phase_name="generated_failure", index=index, total=total),
                                    _emit_progress_snapshot("generated_failure"),
                                ),
                                on_task_progress=lambda _task, progress_payload, _index, _total: (
                                    _update_current_task_progress(progress_payload),
                                    _emit_progress_snapshot("generated_failure"),
                                ),
                                on_task_complete=lambda _task, _result, _index, _total: _set_current_task(None),
                                on_result=lambda task, result, _index, _total: (
                                    _annotate_observed_runtime(task, result, phase_name="generated_failure"),
                                    completed_generated_tasks.append(task),
                                    completed_generated_results.append(result),
                                    _emit_progress_snapshot("generated_failure"),
                                ),
                            )
                        )
                    finally:
                        failure_generated_kernel.close()
                        _cleanup_scoped_runtime_state(failure_generated_config)
                else:
                    _cleanup_scoped_runtime_state(failure_generated_config)
                _emit_eval_progress(
                    progress_label,
                    f"phase=generated_failure complete total={len(failure_generated_tasks)}",
                )
        if generated_success_seed_output_path:
            _write_success_seed_bundle(
                generated_success_seed_output_path,
                primary_tasks=completed_primary_tasks,
                primary_results=completed_primary_results,
                generated_tasks=completed_generated_tasks,
                generated_results=completed_generated_results,
            )
        _emit_eval_progress(progress_label, "phase=metrics_finalize start")
        _emit_progress_snapshot("metrics_finalize")
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=max(0.0, _PROGRESS_HEARTBEAT_INTERVAL_SECONDS))
        _set_current_task(None)
        _emit_progress_snapshot("complete")
        if kernel is not None:
            kernel.close()
    passed = sum(1 for result in results if result.success)
    total_by_capability: dict[str, int] = {}
    passed_by_capability: dict[str, int] = {}
    total_by_difficulty: dict[str, int] = {}
    passed_by_difficulty: dict[str, int] = {}
    total_by_benchmark_family: dict[str, int] = {}
    passed_by_benchmark_family: dict[str, int] = {}
    total_by_memory_source: dict[str, int] = {}
    passed_by_memory_source: dict[str, int] = {}
    total_by_origin_benchmark_family: dict[str, int] = {}
    passed_by_origin_benchmark_family: dict[str, int] = {}
    termination_reasons: dict[str, int] = {}
    skill_selected_steps = 0
    episodes_with_skill_use = 0
    available_but_unused_skill_steps = 0
    excess_available_skill_slots = 0
    total_step_count = 0
    total_available_skills = 0
    total_retrieval_candidates = 0
    total_retrieval_evidence = 0
    total_retrieval_direct_candidates = 0
    retrieval_guided_steps = 0
    retrieval_selected_steps = 0
    retrieval_influenced_steps = 0
    retrieval_ranked_skill_steps = 0
    proposal_selected_steps = 0
    novel_command_steps = 0
    novel_valid_command_steps = 0
    trusted_retrieval_steps = 0
    low_confidence_steps = 0
    first_step_successes = 0
    first_step_confidence_total = 0.0
    success_first_step_confidence_total = 0.0
    failed_first_step_confidence_total = 0.0
    success_first_step_confidence_count = 0
    failed_first_step_confidence_count = 0
    low_confidence_episodes = 0
    low_confidence_passed = 0
    generated_by_kind: dict[str, int] = {}
    generated_passed_by_kind: dict[str, int] = {}
    generated_by_benchmark_family: dict[str, int] = {}
    generated_passed_by_benchmark_family: dict[str, int] = {}
    generated_task_summaries: dict[str, dict[str, object]] = {}
    decision_source_counts: dict[str, int] = {}
    tolbert_route_mode_counts: dict[str, int] = {}
    tolbert_shadow_episodes = 0
    tolbert_primary_episodes = 0
    tolbert_shadow_episodes_by_benchmark_family: dict[str, int] = {}
    tolbert_primary_episodes_by_benchmark_family: dict[str, int] = {}
    task_outcomes: dict[str, dict[str, object]] = {}
    task_trajectories: dict[str, dict[str, object]] = {}
    world_feedback_overall_stats: dict[str, dict[str, float]] = {}
    world_feedback_family_stats: dict[str, dict[str, dict[str, float]]] = {}
    world_feedback_difficulty_stats: dict[str, dict[str, dict[str, float]]] = {}
    unsafe_ambiguous_episodes = 0
    hidden_side_effect_risk_episodes = 0
    success_hidden_side_effect_risk_episodes = 0
    for task, result in zip(tasks, results):
        capability = str(task.metadata.get("capability", "unknown"))
        difficulty = str(task.metadata.get("difficulty", "unknown"))
        benchmark_family = str(task.metadata.get("benchmark_family", "bounded"))
        memory_source = str(task.metadata.get("memory_source", "none"))
        origin_benchmark_family = str(
            task.metadata.get("origin_benchmark_family", benchmark_family)
        )
        total_by_capability[capability] = total_by_capability.get(capability, 0) + 1
        total_by_difficulty[difficulty] = total_by_difficulty.get(difficulty, 0) + 1
        total_by_benchmark_family[benchmark_family] = total_by_benchmark_family.get(benchmark_family, 0) + 1
        total_by_memory_source[memory_source] = total_by_memory_source.get(memory_source, 0) + 1
        total_by_origin_benchmark_family[origin_benchmark_family] = (
            total_by_origin_benchmark_family.get(origin_benchmark_family, 0) + 1
        )
        if result.success:
            passed_by_capability[capability] = passed_by_capability.get(capability, 0) + 1
            passed_by_difficulty[difficulty] = passed_by_difficulty.get(difficulty, 0) + 1
            passed_by_benchmark_family[benchmark_family] = (
                passed_by_benchmark_family.get(benchmark_family, 0) + 1
            )
            passed_by_memory_source[memory_source] = passed_by_memory_source.get(memory_source, 0) + 1
            passed_by_origin_benchmark_family[origin_benchmark_family] = (
                passed_by_origin_benchmark_family.get(origin_benchmark_family, 0) + 1
            )
        termination_reason = result.termination_reason or "unspecified"
        termination_reasons[termination_reason] = termination_reasons.get(termination_reason, 0) + 1
        hidden_side_effect_risk = _episode_hidden_side_effect_risk(result)
        unsafe_ambiguous = _episode_unsafe_ambiguous(result)
        if hidden_side_effect_risk:
            hidden_side_effect_risk_episodes += 1
            if result.success:
                success_hidden_side_effect_risk_episodes += 1
        if unsafe_ambiguous:
            unsafe_ambiguous_episodes += 1
        selected_steps = sum(1 for step in result.steps if step.selected_skill_id)
        skill_selected_steps += selected_steps
        if selected_steps:
            episodes_with_skill_use += 1
        available_but_unused_skill_steps += sum(
            1 for step in result.steps if step.available_skill_count > 0 and not step.selected_skill_id
        )
        excess_available_skill_slots += sum(
            max(0, step.available_skill_count - 1) for step in result.steps
        )
        total_step_count += len(result.steps)
        total_available_skills += sum(step.available_skill_count for step in result.steps)
        total_retrieval_candidates += sum(step.retrieval_candidate_count for step in result.steps)
        total_retrieval_evidence += sum(step.retrieval_evidence_count for step in result.steps)
        total_retrieval_direct_candidates += sum(
            step.retrieval_direct_candidate_count for step in result.steps
        )
        retrieval_guided_steps += sum(1 for step in result.steps if step.retrieval_command_match)
        retrieval_selected_steps += sum(1 for step in result.steps if step.selected_retrieval_span_id)
        retrieval_influenced_steps += sum(1 for step in result.steps if step.retrieval_influenced)
        retrieval_ranked_skill_steps += sum(1 for step in result.steps if step.retrieval_ranked_skill)
        proposal_selected_steps += sum(1 for step in result.steps if str(step.proposal_source).strip())
        novel_command_steps += sum(1 for step in result.steps if bool(step.proposal_novel))
        novel_valid_command_steps += sum(
            1
            for step in result.steps
            if bool(step.proposal_novel) and bool(step.verification.get("passed", False))
        )
        trusted_retrieval_steps += sum(1 for step in result.steps if step.trust_retrieval)
        low_confidence_steps += sum(
            1
            for step in result.steps
            if 0.0 < step.path_confidence < active_config.tolbert_low_confidence_widen_threshold
        )
        episode_route_modes = {
            str(step.tolbert_route_mode).strip()
            for step in result.steps
            if str(step.tolbert_route_mode).strip()
        }
        for step in result.steps:
            source = str(step.decision_source).strip() or "llm"
            decision_source_counts[source] = decision_source_counts.get(source, 0) + 1
            route_mode = str(step.tolbert_route_mode).strip()
            if route_mode:
                tolbert_route_mode_counts[route_mode] = tolbert_route_mode_counts.get(route_mode, 0) + 1
        if "shadow" in episode_route_modes:
            tolbert_shadow_episodes += 1
            tolbert_shadow_episodes_by_benchmark_family[benchmark_family] = (
                tolbert_shadow_episodes_by_benchmark_family.get(benchmark_family, 0) + 1
            )
        if "primary" in episode_route_modes:
            tolbert_primary_episodes += 1
            tolbert_primary_episodes_by_benchmark_family[benchmark_family] = (
                tolbert_primary_episodes_by_benchmark_family.get(benchmark_family, 0) + 1
            )
        if result.steps and result.steps[0].verification.get("passed", False):
            first_step_successes += 1
        if result.steps:
            first_step_confidence = float(result.steps[0].path_confidence)
            first_step_confidence_total += first_step_confidence
            if result.success:
                success_first_step_confidence_total += first_step_confidence
                success_first_step_confidence_count += 1
            else:
                failed_first_step_confidence_total += first_step_confidence
                failed_first_step_confidence_count += 1
        if 0.0 < first_step_confidence < active_config.tolbert_low_confidence_widen_threshold:
                low_confidence_episodes += 1
                if result.success:
                    low_confidence_passed += 1
        world_summary = result.world_model_summary if isinstance(result.world_model_summary, dict) else {}
        base_task_outcome = _task_outcome_summary(
            task,
            result,
            low_confidence_threshold=active_config.tolbert_low_confidence_widen_threshold,
        )
        no_state_progress_steps = sum(
            1 for step in result.steps if "no_state_progress" in list(step.failure_signals or [])
        )
        state_regression_steps = sum(1 for step in result.steps if int(step.state_regression_count) > 0)
        total_state_regression_count = sum(int(step.state_regression_count) for step in result.steps)
        low_confidence_step_count = sum(
            1
            for step in result.steps
            if 0.0 < float(step.path_confidence) < active_config.tolbert_low_confidence_widen_threshold
        )
        episode_world_feedback_overall_stats: dict[str, dict[str, float]] = {}
        task_outcomes[task.task_id] = {
            **base_task_outcome,
            "first_step_verified": bool(result.steps and result.steps[0].verification.get("passed", False)),
            "novel_valid_command_steps": sum(
                1
                for step in result.steps
                if bool(step.proposal_novel) and bool(step.verification.get("passed", False))
            ),
            "total_state_regression_count": total_state_regression_count,
            "max_state_regression_count": max(
                (int(step.state_regression_count) for step in result.steps),
                default=0,
            ),
            "completion_ratio": float(world_summary.get("completion_ratio", 0.0) or 0.0),
            "present_forbidden_artifact_count": len(list(world_summary.get("present_forbidden_artifacts", []))),
            "changed_preserved_artifact_count": len(list(world_summary.get("changed_preserved_artifacts", []))),
            "missing_expected_artifact_count": len(list(world_summary.get("missing_expected_artifacts", []))),
        }
        task_trajectories[task.task_id] = {
            "task_id": task.task_id,
            "benchmark_family": benchmark_family,
            "difficulty": difficulty,
            "success": bool(result.success),
            "termination_reason": termination_reason,
            "steps": [
                {
                    "index": int(step.index),
                    "action": str(step.action).strip(),
                    "content": str(step.content).strip(),
                    "command": str((step.command_result or {}).get("command", "")).strip()
                    if isinstance(step.command_result, dict)
                    else "",
                    "exit_code": int((step.command_result or {}).get("exit_code", 0) or 0)
                    if isinstance(step.command_result, dict)
                    else 0,
                    "timed_out": bool((step.command_result or {}).get("timed_out", False))
                    if isinstance(step.command_result, dict)
                    else False,
                    "verification_passed": bool(step.verification.get("passed", False))
                    if isinstance(step.verification, dict)
                    else False,
                    "verification_reasons": [
                        str(reason).strip()
                        for reason in (step.verification.get("reasons", []) if isinstance(step.verification, dict) else [])
                        if str(reason).strip()
                    ][:4],
                    "failure_signals": [
                        str(signal).strip()
                        for signal in list(step.failure_signals or [])
                        if str(signal).strip()
                    ],
                    "state_regression_count": int(step.state_regression_count),
                    "state_progress_delta": float(step.state_progress_delta),
                    "state_no_progress": bool(
                        (step.state_transition or {}).get("no_progress", False)
                    )
                    if isinstance(step.state_transition, dict)
                    else False,
                    "state_regressed": bool(
                        (step.state_transition or {}).get("regressed", False)
                    )
                    if isinstance(step.state_transition, dict)
                    else False,
                    "world_model_horizon": str(step.world_model_horizon).strip(),
                    "active_subgoal": str(step.active_subgoal).strip(),
                    "acting_role": str(step.acting_role).strip(),
                    "decision_source": str(step.decision_source).strip(),
                    "tolbert_route_mode": str(step.tolbert_route_mode).strip(),
                    "proposal_source": str(step.proposal_source).strip(),
                    "proposal_novel": bool(step.proposal_novel),
                    "selected_retrieval_span_id": (
                        ""
                        if str(step.selected_retrieval_span_id or "").strip().lower() == "none"
                        else str(step.selected_retrieval_span_id or "").strip()
                    ),
                    "retrieval_influenced": bool(step.retrieval_influenced),
                    "trust_retrieval": bool(step.trust_retrieval),
                    "world_feedback": {},
                }
                for step in result.steps
            ],
        }
        for step_payload, step in zip(task_trajectories[task.task_id]["steps"], result.steps):
            feedback = _step_world_feedback(step)
            if not bool(feedback.get("present", False)):
                continue
            observed = _observed_world_feedback_targets(step)
            _update_world_feedback_stats(world_feedback_overall_stats, feedback=feedback, observed=observed)
            family_payload = world_feedback_family_stats.setdefault(benchmark_family, {})
            _update_world_feedback_stats(family_payload, feedback=feedback, observed=observed)
            difficulty_payload = world_feedback_difficulty_stats.setdefault(difficulty, {})
            _update_world_feedback_stats(difficulty_payload, feedback=feedback, observed=observed)
            _update_world_feedback_stats(episode_world_feedback_overall_stats, feedback=feedback, observed=observed)
            step_payload["world_feedback"] = {
                "progress_signal": float(feedback.get("progress_signal", 0.0) or 0.0),
                "risk_signal": float(feedback.get("risk_signal", 0.0) or 0.0),
                "decoder_progress_signal": float(feedback.get("decoder_progress_signal", 0.0) or 0.0),
                "decoder_risk_signal": float(feedback.get("decoder_risk_signal", 0.0) or 0.0),
                "trusted_retrieval_alignment": float(feedback.get("trusted_retrieval_alignment", 0.0) or 0.0),
                "graph_environment_alignment": float(feedback.get("graph_environment_alignment", 0.0) or 0.0),
                "transfer_novelty": float(feedback.get("transfer_novelty", 0.0) or 0.0),
                "observed_progress": float(observed.get("observed_progress", 0.0) or 0.0),
                "observed_risk": float(observed.get("observed_risk", 0.0) or 0.0),
                "source": str(feedback.get("source", "")).strip(),
                "model_family": str(feedback.get("model_family", "")).strip(),
            }
        task_outcomes[task.task_id]["world_feedback_summary"] = _finalize_world_feedback_row(
            dict(episode_world_feedback_overall_stats.get("row", {}))
        )
    generated_passed = 0
    for task, result in zip(generated_tasks, generated_results):
        generated_task_summaries[task.task_id] = _task_outcome_summary(
            task,
            result,
            low_confidence_threshold=active_config.tolbert_low_confidence_widen_threshold,
        )
        generated_kind = str(task.metadata.get("curriculum_kind", "unknown"))
        generated_family = str(task.metadata.get("benchmark_family", "bounded"))
        generated_by_kind[generated_kind] = generated_by_kind.get(generated_kind, 0) + 1
        generated_by_benchmark_family[generated_family] = (
            generated_by_benchmark_family.get(generated_family, 0) + 1
        )
        if result.success:
            generated_passed += 1
            generated_passed_by_kind[generated_kind] = (
                generated_passed_by_kind.get(generated_kind, 0) + 1
            )
            generated_passed_by_benchmark_family[generated_family] = (
                generated_passed_by_benchmark_family.get(generated_family, 0) + 1
            )

    average_steps = sum(len(result.steps) for result in results) / len(results) if results else 0.0
    success_steps = [len(result.steps) for result in results if result.success]
    average_success_steps = sum(success_steps) / len(success_steps) if success_steps else 0.0
    for family in requested_priority_families:
        total_by_benchmark_family.setdefault(family, 0)
        passed_by_benchmark_family.setdefault(family, 0)
    reusable_skills = 0
    if active_config.skills_path.exists():
        payload = json.loads(active_config.skills_path.read_text(encoding="utf-8"))
        reusable_skills = len(payload.get("skills", payload)) if isinstance(payload, dict) else len(payload)
    memory_documents = len(list(active_config.trajectories_root.glob("*.json")))
    (
        world_feedback_summary,
        world_feedback_by_benchmark_family,
        world_feedback_by_difficulty,
    ) = _finalize_world_feedback_summary(
        world_feedback_overall_stats,
        world_feedback_family_stats,
        world_feedback_difficulty_stats,
    )
    long_horizon_persistence_summary = _long_horizon_persistence_summary(task_trajectories)
    (
        contract_clean_failure_recovery_summary,
        contract_clean_failure_recovery_by_origin_benchmark_family,
    ) = _contract_clean_failure_recovery_summary(generated_task_summaries)
    transfer_alignment_summary = _transfer_alignment_summary(task_trajectories)
    metrics = EvalMetrics(
        total=len(results),
        passed=passed,
        average_steps=average_steps,
        average_success_steps=average_success_steps,
        unsafe_ambiguous_episodes=unsafe_ambiguous_episodes,
        hidden_side_effect_risk_episodes=hidden_side_effect_risk_episodes,
        success_hidden_side_effect_risk_episodes=success_hidden_side_effect_risk_episodes,
        total_by_capability=total_by_capability,
        passed_by_capability=passed_by_capability,
        total_by_difficulty=total_by_difficulty,
        passed_by_difficulty=passed_by_difficulty,
        total_by_benchmark_family=total_by_benchmark_family,
        passed_by_benchmark_family=passed_by_benchmark_family,
        total_by_memory_source=total_by_memory_source,
        passed_by_memory_source=passed_by_memory_source,
        total_by_origin_benchmark_family=total_by_origin_benchmark_family,
        passed_by_origin_benchmark_family=passed_by_origin_benchmark_family,
        termination_reasons=termination_reasons,
        skill_selected_steps=skill_selected_steps,
        episodes_with_skill_use=episodes_with_skill_use,
        available_but_unused_skill_steps=available_but_unused_skill_steps,
        excess_available_skill_slots=excess_available_skill_slots,
        average_available_skills=0.0 if total_step_count == 0 else total_available_skills / total_step_count,
        average_retrieval_candidates=0.0
        if total_step_count == 0
        else total_retrieval_candidates / total_step_count,
        average_retrieval_evidence=0.0
        if total_step_count == 0
        else total_retrieval_evidence / total_step_count,
        average_retrieval_direct_candidates=0.0
        if total_step_count == 0
        else total_retrieval_direct_candidates / total_step_count,
        retrieval_guided_steps=retrieval_guided_steps,
        retrieval_selected_steps=retrieval_selected_steps,
        retrieval_influenced_steps=retrieval_influenced_steps,
        retrieval_ranked_skill_steps=retrieval_ranked_skill_steps,
        proposal_selected_steps=proposal_selected_steps,
        novel_command_steps=novel_command_steps,
        novel_valid_command_steps=novel_valid_command_steps,
        trusted_retrieval_steps=trusted_retrieval_steps,
        low_confidence_steps=low_confidence_steps,
        first_step_successes=first_step_successes,
        average_first_step_path_confidence=0.0
        if not results
        else first_step_confidence_total / len(results),
        success_first_step_path_confidence=0.0
        if success_first_step_confidence_count == 0
        else success_first_step_confidence_total / success_first_step_confidence_count,
        failed_first_step_path_confidence=0.0
        if failed_first_step_confidence_count == 0
        else failed_first_step_confidence_total / failed_first_step_confidence_count,
        low_confidence_episodes=low_confidence_episodes,
        low_confidence_passed=low_confidence_passed,
        memory_documents=memory_documents,
        reusable_skills=reusable_skills,
        generated_total=len(generated_results),
        generated_passed=generated_passed,
        generated_by_kind=generated_by_kind,
        generated_passed_by_kind=generated_passed_by_kind,
        generated_by_benchmark_family=generated_by_benchmark_family,
        generated_passed_by_benchmark_family=generated_passed_by_benchmark_family,
        decision_source_counts=decision_source_counts,
        tolbert_route_mode_counts=tolbert_route_mode_counts,
        tolbert_shadow_episodes=tolbert_shadow_episodes,
        tolbert_primary_episodes=tolbert_primary_episodes,
        tolbert_shadow_episodes_by_benchmark_family=tolbert_shadow_episodes_by_benchmark_family,
        tolbert_primary_episodes_by_benchmark_family=tolbert_primary_episodes_by_benchmark_family,
        proposal_metrics_by_benchmark_family=_proposal_metrics_by_benchmark_family(task_trajectories),
        proposal_metrics_by_difficulty=_proposal_metrics_by_difficulty(task_trajectories),
        world_feedback_summary=world_feedback_summary,
        world_feedback_by_benchmark_family=world_feedback_by_benchmark_family,
        world_feedback_by_difficulty=world_feedback_by_difficulty,
        long_horizon_persistence_summary=long_horizon_persistence_summary,
        contract_clean_failure_recovery_summary=contract_clean_failure_recovery_summary,
        contract_clean_failure_recovery_by_origin_benchmark_family=(
            contract_clean_failure_recovery_by_origin_benchmark_family
        ),
        transfer_alignment_summary=transfer_alignment_summary,
        task_outcomes=task_outcomes,
        task_trajectories=task_trajectories,
    )
    _emit_eval_progress(progress_label, "phase=metrics_finalize complete")
    return metrics


def _episode_hidden_side_effect_risk(result) -> bool:
    summary = result.world_model_summary if isinstance(result.world_model_summary, dict) else {}
    return bool(
        list(summary.get("present_forbidden_artifacts", []))
        or list(summary.get("changed_preserved_artifacts", []))
    )


def _episode_unsafe_ambiguous(result) -> bool:
    if bool(result.success):
        return False
    if not any(str(step.action).strip() == "code_execute" for step in result.steps):
        return False
    reason = str(result.termination_reason or "").strip()
    if reason in {"policy_terminated", "setup_failed", "setup_pending"}:
        return False
    return True


def compare_skill_modes(
    config: KernelConfig | None = None,
    *,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_benchmark_candidates: bool = False,
    include_verifier_candidates: bool = False,
    include_generated: bool = False,
    include_failure_generated: bool = False,
) -> SkillComparison:
    base_config = config or KernelConfig()
    with_skills_config = _scoped_config(base_config, "with_skills", use_skills=True)
    try:
        with_skills = run_eval(
            config=with_skills_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_memory=include_skill_memory,
            include_skill_transfer=include_skill_transfer,
            include_operator_memory=include_operator_memory,
            include_tool_memory=include_tool_memory,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
        )
    finally:
        _cleanup_scoped_runtime_state(with_skills_config)
    without_skills_config = _scoped_config(base_config, "without_skills", use_skills=False)
    try:
        without_skills = run_eval(
            config=without_skills_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_memory=include_skill_memory,
            include_skill_transfer=include_skill_transfer,
            include_operator_memory=include_operator_memory,
            include_tool_memory=include_tool_memory,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
        )
    finally:
        _cleanup_scoped_runtime_state(without_skills_config)
    capabilities = sorted(
        set(with_skills.total_by_capability) | set(without_skills.total_by_capability)
    )
    benchmark_families = sorted(
        set(with_skills.total_by_benchmark_family) | set(without_skills.total_by_benchmark_family)
    )
    return SkillComparison(
        with_skills=with_skills,
        without_skills=without_skills,
        pass_rate_delta=with_skills.pass_rate - without_skills.pass_rate,
        average_steps_delta=with_skills.average_steps - without_skills.average_steps,
        capability_pass_rate_delta={
            capability: with_skills.capability_pass_rate(capability)
            - without_skills.capability_pass_rate(capability)
            for capability in capabilities
        },
        benchmark_family_pass_rate_delta={
            family: with_skills.benchmark_family_pass_rate(family)
            - without_skills.benchmark_family_pass_rate(family)
            for family in benchmark_families
        },
    )


def compare_tolbert_modes(
    config: KernelConfig | None = None,
    *,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_benchmark_candidates: bool = False,
    include_verifier_candidates: bool = False,
    include_generated: bool = False,
    include_failure_generated: bool = False,
) -> TolbertComparison:
    base_config = config or KernelConfig()
    with_tolbert_config = _scoped_config(base_config, "with_tolbert", use_tolbert_context=True)
    try:
        with_tolbert = run_eval(
            config=with_tolbert_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_memory=include_skill_memory,
            include_skill_transfer=include_skill_transfer,
            include_operator_memory=include_operator_memory,
            include_tool_memory=include_tool_memory,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
        )
    finally:
        _cleanup_scoped_runtime_state(with_tolbert_config)
    without_tolbert_config = _scoped_config(base_config, "without_tolbert", use_tolbert_context=False)
    try:
        without_tolbert = run_eval(
            config=without_tolbert_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_memory=include_skill_memory,
            include_skill_transfer=include_skill_transfer,
            include_operator_memory=include_operator_memory,
            include_tool_memory=include_tool_memory,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
        )
    finally:
        _cleanup_scoped_runtime_state(without_tolbert_config)
    capabilities = sorted(
        set(with_tolbert.total_by_capability) | set(without_tolbert.total_by_capability)
    )
    benchmark_families = sorted(
        set(with_tolbert.total_by_benchmark_family) | set(without_tolbert.total_by_benchmark_family)
    )
    return TolbertComparison(
        with_tolbert=with_tolbert,
        without_tolbert=without_tolbert,
        pass_rate_delta=with_tolbert.pass_rate - without_tolbert.pass_rate,
        average_steps_delta=with_tolbert.average_steps - without_tolbert.average_steps,
        capability_pass_rate_delta={
            capability: with_tolbert.capability_pass_rate(capability)
            - without_tolbert.capability_pass_rate(capability)
            for capability in capabilities
        },
        benchmark_family_pass_rate_delta={
            family: with_tolbert.benchmark_family_pass_rate(family)
            - without_tolbert.benchmark_family_pass_rate(family)
            for family in benchmark_families
        },
    )


def compare_tolbert_feature_modes(
    config: KernelConfig | None = None,
    *,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_skill_memory: bool = False,
    include_skill_transfer: bool = False,
    include_operator_memory: bool = False,
    include_tool_memory: bool = False,
    include_verifier_memory: bool = False,
    include_benchmark_candidates: bool = False,
    include_verifier_candidates: bool = False,
    include_generated: bool = False,
    include_failure_generated: bool = False,
) -> TolbertModeComparison:
    base_config = config or KernelConfig()
    modes = ("path_only", "retrieval_only", "deterministic_command", "skill_ranking", "full")
    task_limit = None
    if base_config.provider != "mock":
        task_limit = base_config.compare_feature_max_tasks
    mode_metrics = {}
    for mode in modes:
        print(f"[compare_tolbert_features] mode={mode}", file=sys.stderr, flush=True)
        scoped_config = _scoped_config(base_config, f"tolbert_{mode}", use_tolbert_context=True, tolbert_mode=mode)
        try:
            mode_metrics[mode] = run_eval(
                config=scoped_config,
                include_discovered_tasks=include_discovered_tasks,
                include_episode_memory=include_episode_memory,
                include_skill_memory=include_skill_memory,
                include_skill_transfer=include_skill_transfer,
                include_operator_memory=include_operator_memory,
                include_tool_memory=include_tool_memory,
                include_verifier_memory=include_verifier_memory,
                include_benchmark_candidates=include_benchmark_candidates,
                include_verifier_candidates=include_verifier_candidates,
                include_generated=include_generated,
                include_failure_generated=include_failure_generated,
                task_limit=task_limit,
                progress_label=f"tolbert:{mode}",
            )
        finally:
            _cleanup_scoped_runtime_state(scoped_config)
    return TolbertModeComparison(mode_metrics=mode_metrics)


def compare_abstraction_transfer_modes(
    config: KernelConfig | None = None,
    *,
    include_discovered_tasks: bool = False,
    include_episode_memory: bool = False,
    include_verifier_memory: bool = False,
    include_benchmark_candidates: bool = False,
    include_verifier_candidates: bool = False,
    include_generated: bool = False,
    include_failure_generated: bool = False,
    task_limit: int | None = None,
    progress_label_prefix: str | None = None,
) -> AbstractionComparison:
    base_config = config or KernelConfig()
    skill_target_map, operator_target_map = build_shared_transfer_target_maps(
        base_config.skills_path,
        base_config.operator_classes_path,
    )
    operator_config = _scoped_config(base_config, "with_operators")
    try:
        operator_metrics = run_eval(
            config=operator_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_operator_memory=True,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
            operator_target_map=operator_target_map,
            task_limit=task_limit,
            progress_label=(
                None
                if not progress_label_prefix
                else f"{progress_label_prefix}_with_operators"
            ),
        )
    finally:
        _cleanup_scoped_runtime_state(operator_config)
    raw_skill_config = _scoped_config(base_config, "with_raw_skill_transfer")
    try:
        raw_skill_metrics = run_eval(
            config=raw_skill_config,
            include_discovered_tasks=include_discovered_tasks,
            include_episode_memory=include_episode_memory,
            include_skill_transfer=True,
            include_verifier_memory=include_verifier_memory,
            include_benchmark_candidates=include_benchmark_candidates,
            include_verifier_candidates=include_verifier_candidates,
            include_generated=include_generated,
            include_failure_generated=include_failure_generated,
            skill_transfer_target_map=skill_target_map,
            task_limit=task_limit,
            progress_label=(
                None
                if not progress_label_prefix
                else f"{progress_label_prefix}_with_raw_skill_transfer"
            ),
        )
    finally:
        _cleanup_scoped_runtime_state(raw_skill_config)
    return AbstractionComparison(
        operator_metrics=operator_metrics,
        raw_skill_metrics=raw_skill_metrics,
        pass_rate_delta=operator_metrics.pass_rate - raw_skill_metrics.pass_rate,
        average_steps_delta=operator_metrics.average_steps - raw_skill_metrics.average_steps,
        transfer_pass_rate_delta=(
            operator_metrics.memory_source_pass_rate("operator")
            - raw_skill_metrics.memory_source_pass_rate("skill_transfer")
        ),
    )


def _normalize_priority_families(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return normalized


def _normalize_priority_family_weights(values: dict[str, object] | None) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, float] = {}
    for family, raw_weight in values.items():
        token = str(family).strip()
        if not token:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight > 0.0:
            normalized[token] = weight
    return normalized


_DEFAULT_REAL_WORLD_PRIORITY_FAMILIES = (
    "repo_sandbox",
    "repository",
    "integration",
    "tooling",
    "project",
    "workflow",
    "repo_chore",
)
_LONG_HORIZON_CODING_PRIORITY_FAMILIES = {"project", "repository"}


def _default_priority_families_for_compare(tasks: Sequence[object]) -> list[str]:
    present = {
        str(getattr(task, "metadata", {}).get("benchmark_family", "bounded")).strip() or "bounded"
        for task in tasks
    }
    return [family for family in _DEFAULT_REAL_WORLD_PRIORITY_FAMILIES if family in present]


def _should_surface_shared_repo_workers(
    requested_priority_families: Sequence[str] | None,
) -> bool:
    return bool(
        _LONG_HORIZON_CODING_PRIORITY_FAMILIES.intersection(
            _normalize_priority_families(requested_priority_families)
        )
    )


def _shared_repo_eval_benchmark_family(requested_priority_families: Sequence[str] | None) -> str:
    normalized = _normalize_priority_families(requested_priority_families)
    if "project" in normalized:
        return "project"
    if "repository" in normalized:
        return "repository"
    return "project"


def _surface_shared_repo_worker_tasks(bank: TaskBank, *, requested_priority_families: Sequence[str]) -> list:
    if not _should_surface_shared_repo_workers(requested_priority_families):
        return []
    surfaced: list = []
    seen_task_ids: set[str] = set()
    surfaced_family = _shared_repo_eval_benchmark_family(requested_priority_families)
    for task in bank.list():
        metadata = getattr(task, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_merged_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if not required_merged_branches:
            continue
        candidate_workers = list(bank.parallel_worker_tasks(task.task_id))
        if not any(
            bool(dict(getattr(worker, "metadata", {}) or {}).get("synthetic_worker", False))
            and isinstance(dict(getattr(worker, "metadata", {}) or {}).get("synthetic_edit_plan", []), list)
            and dict(getattr(worker, "metadata", {}) or {}).get("synthetic_edit_plan")
            for worker in candidate_workers
        ):
            synthesize_workers = getattr(bank, "_synthesized_parallel_worker_tasks", None)
            if callable(synthesize_workers):
                candidate_workers.extend(
                    synthesize_workers(task, required_branches=required_merged_branches)
                )
        for worker in candidate_workers:
            worker_metadata = dict(getattr(worker, "metadata", {}) or {})
            edit_plan = worker_metadata.get("synthetic_edit_plan", [])
            if not bool(worker_metadata.get("synthetic_worker", False)) or not isinstance(edit_plan, list) or not edit_plan:
                continue
            if worker.task_id in seen_task_ids:
                continue
            seen_task_ids.add(worker.task_id)
            origin_family = str(worker_metadata.get("benchmark_family", "repo_sandbox")).strip() or "repo_sandbox"
            worker_metadata.update(
                {
                    "benchmark_family": surfaced_family,
                    "origin_benchmark_family": origin_family,
                    "difficulty": "long_horizon",
                    "long_horizon_coding_surface": "shared_repo_synthetic_worker",
                    "memory_source": str(worker_metadata.get("memory_source", "none")).strip() or "none",
                }
            )
            surfaced.append(replace(worker, metadata=worker_metadata))
    return surfaced


def _surface_shared_repo_integrator_tasks(bank: TaskBank, *, requested_priority_families: Sequence[str]) -> list:
    if not _should_surface_shared_repo_workers(requested_priority_families):
        return []
    surfaced: list = []
    seen_task_ids: set[str] = set()
    surfaced_family = _shared_repo_eval_benchmark_family(requested_priority_families)
    for task in bank.list():
        metadata = getattr(task, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        required_merged_branches = [
            str(value).strip()
            for value in verifier.get("required_merged_branches", [])
            if str(value).strip()
        ]
        if not required_merged_branches:
            continue
        if int(metadata.get("shared_repo_order", 0) or 0) <= 0:
            continue
        if task.task_id in seen_task_ids:
            continue
        seen_task_ids.add(task.task_id)
        integrator_metadata = dict(metadata)
        origin_family = str(integrator_metadata.get("benchmark_family", "repo_sandbox")).strip() or "repo_sandbox"
        integrator_metadata.update(
            {
                "benchmark_family": surfaced_family,
                "origin_benchmark_family": origin_family,
                "difficulty": "long_horizon",
                "long_horizon_coding_surface": "shared_repo_integrator",
                "memory_source": str(integrator_metadata.get("memory_source", "none")).strip() or "none",
            }
        )
        surfaced.append(replace(task, metadata=integrator_metadata))
    return surfaced


def _priority_family_slot_targets(
    prioritized: list[str],
    task_limit: int,
    *,
    priority_weights: dict[str, float],
) -> dict[str, int]:
    if task_limit <= 0 or not prioritized:
        return {}
    if len(prioritized) == 1:
        return {prioritized[0]: task_limit}
    priority_budget = min(task_limit, max(len(prioritized), (task_limit + 1) // 2))
    if priority_budget <= 0:
        return {}
    if sum(priority_weights.get(family, 0.0) for family in prioritized) <= 0.0:
        priority_weights = {family: 1.0 for family in prioritized}
    total_weight = sum(priority_weights.get(family, 0.0) for family in prioritized)
    raw_targets = {
        family: (priority_weights.get(family, 0.0) / total_weight) * priority_budget
        for family in prioritized
    }
    slot_targets = {family: int(raw_targets[family]) for family in prioritized}
    assigned = sum(slot_targets.values())
    remainder_order = sorted(
        prioritized,
        key=lambda family: (
            raw_targets[family] - slot_targets[family],
            priority_weights.get(family, 0.0),
            -prioritized.index(family),
        ),
        reverse=True,
    )
    index = 0
    while assigned < priority_budget and remainder_order:
        family = remainder_order[index % len(remainder_order)]
        slot_targets[family] += 1
        assigned += 1
        index += 1
    return slot_targets


def _low_cost_task_key(task) -> tuple[int, int, int, int, int, int, int, int, str]:
    metadata = getattr(task, "metadata", {}) or {}
    light_supervision_rank = int(not bool(metadata.get("light_supervision_candidate", False)))
    difficulty = str(metadata.get("difficulty", "")).strip() or str(getattr(task, "difficulty", "")).strip()
    if str(metadata.get("horizon", "")).strip() == "long_horizon":
        difficulty = "long_horizon"
    difficulty_rank = {
        "seed": 0,
        "bounded": 1,
        "": 2,
        "git_worker_branch_synthesized": 2,
        "long_horizon": 3,
        "retrieval": 4,
    }.get(difficulty, 2)
    long_horizon_surface_rank = 3
    surface = str(metadata.get("long_horizon_coding_surface", "")).strip()
    if surface == "shared_repo_synthetic_worker":
        long_horizon_surface_rank = 0
    elif surface == "shared_repo_integrator":
        long_horizon_surface_rank = 1
    elif difficulty == "long_horizon":
        long_horizon_surface_rank = 2
    synthetic_edit_plan_rank = 1
    if bool(metadata.get("synthetic_worker", False)) and isinstance(metadata.get("synthetic_edit_plan", []), list):
        synthetic_edit_plan_rank = 0 if metadata.get("synthetic_edit_plan") else 1
    requires_retrieval_rank = 1 if bool(metadata.get("requires_retrieval", False)) else 0
    repo_sandbox_rank = _repo_sandbox_low_cost_rank(task)
    return (
        light_supervision_rank,
        difficulty_rank,
        long_horizon_surface_rank,
        synthetic_edit_plan_rank,
        requires_retrieval_rank,
        repo_sandbox_rank,
        int(getattr(task, "max_steps", 0) or 0),
        len(getattr(task, "suggested_commands", []) or []),
        len(str(getattr(task, "prompt", "") or "")),
        str(getattr(task, "task_id", "")),
    )


def _repo_sandbox_low_cost_rank(task) -> int:
    metadata = getattr(task, "metadata", {}) or {}
    if str(metadata.get("benchmark_family", "")).strip() != "repo_sandbox":
        return 0
    task_id = str(getattr(task, "task_id", "") or "").strip()
    if task_id in {
        "git_conflict_worker_status_task",
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
        "git_generated_conflict_resolution_task",
    }:
        return 1
    if task_id in {
        "git_release_train_acceptance_task",
        "git_release_train_conflict_acceptance_task",
    }:
        return 2
    return 0


def _repo_sandbox_required_family_rank(task) -> int:
    metadata = getattr(task, "metadata", {}) or {}
    if str(metadata.get("benchmark_family", "")).strip() != "repo_sandbox":
        return 0
    task_id = str(getattr(task, "task_id", "") or "").strip()
    if task_id in {
        "git_release_train_worker_api_task",
        "git_release_train_worker_docs_task",
        "git_release_train_worker_ops_task",
        "git_release_train_conflict_worker_docs_task",
        "git_release_train_conflict_worker_api_task",
        "git_release_train_conflict_worker_ops_task",
        "git_parallel_merge_acceptance_task",
        "git_repo_test_repair_task",
        "git_repo_status_review_task",
        "git_conflict_worker_status_task",
    }:
        return 0
    if task_id in {
        "git_repo_test_repair_retrieval_task",
        "git_repo_status_review_retrieval_task",
    }:
        return 1
    if task_id in {
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
    }:
        return 2
    if task_id == "git_generated_conflict_resolution_task":
        return 3
    if task_id in {
        "git_release_train_acceptance_task",
        "git_release_train_conflict_acceptance_task",
    }:
        return 4
    if task_id in {
        "git_release_train_acceptance_retrieval_task",
        "git_release_train_conflict_acceptance_retrieval_task",
    }:
        return 5
    return 0


def _required_executable_family_sort_key(task) -> tuple[object, ...]:
    metadata = getattr(task, "metadata", {}) or {}
    if str(metadata.get("benchmark_family", "")).strip() == "repo_sandbox":
        return (
            _repo_sandbox_required_family_rank(task),
            _low_cost_task_key(task),
        )
    return (
        int(
            bool(getattr(task, "metadata", {}).get("requires_retrieval", False))
            and bool(str(getattr(task, "metadata", {}).get("source_task", "")).strip())
        ),
        _low_cost_task_key(task),
    )


def _drop_retrieval_companions_when_sources_present(tasks: Sequence) -> list:
    task_ids = {
        str(getattr(task, "task_id", "") or "").strip()
        for task in tasks
        if str(getattr(task, "task_id", "") or "").strip()
    }
    if not task_ids:
        return list(tasks)
    return [
        task
        for task in tasks
        if not (
            bool(getattr(task, "metadata", {}).get("requires_retrieval", False))
            and bool(str(getattr(task, "metadata", {}).get("source_task", "")).strip())
            and str(getattr(task, "metadata", {}).get("source_task", "")).strip() in task_ids
        )
    ]


def _shared_repo_long_horizon_surface(task) -> str:
    metadata = getattr(task, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("long_horizon_coding_surface", "")).strip()


def _shared_repo_workflow_guard(task) -> dict[str, object]:
    metadata = getattr(task, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    workflow_guard = metadata.get("workflow_guard", {})
    return dict(workflow_guard) if isinstance(workflow_guard, dict) else {}


def _shared_repo_semantic_verifier(task) -> dict[str, object]:
    metadata = getattr(task, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    verifier = metadata.get("semantic_verifier", {})
    return dict(verifier) if isinstance(verifier, dict) else {}


def _shared_repo_id(task) -> str:
    return str(_shared_repo_workflow_guard(task).get("shared_repo_id", "")).strip()


def _shared_repo_worker_branch(task) -> str:
    return str(_shared_repo_workflow_guard(task).get("worker_branch", "")).strip()


def _shared_repo_required_merged_branches(task) -> list[str]:
    return [
        str(value).strip()
        for value in _shared_repo_semantic_verifier(task).get("required_merged_branches", [])
        if str(value).strip()
    ]


def _reorder_shared_repo_bundle_tasks(selected: list) -> list:
    if len(selected) <= 1:
        return selected
    positions_by_repo: dict[str, list[int]] = {}
    for index, task in enumerate(selected):
        repo_id = _shared_repo_id(task)
        if repo_id:
            positions_by_repo.setdefault(repo_id, []).append(index)
    for repo_id, positions in positions_by_repo.items():
        if len(positions) <= 1:
            continue
        repo_tasks = [selected[index] for index in positions]
        integrators = [
            task
            for task in repo_tasks
            if _shared_repo_long_horizon_surface(task) == "shared_repo_integrator"
        ]
        if not integrators:
            continue
        branch_order: dict[str, int] = {}
        for integrator in integrators:
            for branch_index, branch in enumerate(_shared_repo_required_merged_branches(integrator)):
                if branch and branch not in branch_order:
                    branch_order[branch] = branch_index
        repo_tasks.sort(
            key=lambda task: (
                int(_shared_repo_long_horizon_surface(task) == "shared_repo_integrator"),
                branch_order.get(_shared_repo_worker_branch(task), len(branch_order)),
                int(getattr(task, "metadata", {}).get("shared_repo_order", 0) or 0),
                str(getattr(task, "task_id", "")),
            )
        )
        for position, task in zip(positions, repo_tasks):
            selected[position] = task
    return selected


def _reserve_shared_repo_integrator_slot(
    selected: list,
    grouped: dict[str, list],
    *,
    priority_families: Sequence[str] | None = None,
) -> list:
    if len(selected) <= 1 or not _should_surface_shared_repo_workers(priority_families):
        return selected
    if any(_shared_repo_long_horizon_surface(task) == "shared_repo_integrator" for task in selected):
        return selected
    integrator_candidates = [
        task
        for family_tasks in grouped.values()
        for task in family_tasks
        if _shared_repo_long_horizon_surface(task) == "shared_repo_integrator"
    ]
    if not integrator_candidates:
        return selected
    selected_worker_branches_by_repo: dict[str, set[str]] = {}
    for task in selected:
        if _shared_repo_long_horizon_surface(task) != "shared_repo_synthetic_worker":
            continue
        repo_id = _shared_repo_id(task)
        worker_branch = _shared_repo_worker_branch(task)
        if not repo_id or not worker_branch:
            continue
        selected_worker_branches_by_repo.setdefault(repo_id, set()).add(worker_branch)
    if not selected_worker_branches_by_repo:
        return selected
    eligible_integrators = []
    for task in integrator_candidates:
        repo_id = _shared_repo_id(task)
        required_branches = {
            branch for branch in _shared_repo_required_merged_branches(task) if branch
        }
        if not repo_id or not required_branches:
            continue
        selected_branches = selected_worker_branches_by_repo.get(repo_id, set())
        if required_branches.issubset(selected_branches):
            eligible_integrators.append(task)
    if not eligible_integrators:
        return selected
    eligible_integrators.sort(
        key=lambda task: (
            -len(selected_worker_branches_by_repo.get(_shared_repo_id(task), set())),
            -sum(
                1
                for selected_task in selected
                if _shared_repo_long_horizon_surface(selected_task) == "shared_repo_synthetic_worker"
                and _shared_repo_id(selected_task) == _shared_repo_id(task)
                and _shared_repo_worker_branch(selected_task) in set(_shared_repo_required_merged_branches(task))
            ),
            int(getattr(task, "metadata", {}).get("shared_repo_order", 0) or 0),
            _low_cost_task_key(task),
        )
    )
    chosen_integrator = eligible_integrators[0]
    chosen_repo_id = _shared_repo_id(chosen_integrator)
    chosen_required_branches = {
        branch for branch in _shared_repo_required_merged_branches(chosen_integrator) if branch
    }
    replacement_index = next(
        (
            index
            for index in range(len(selected) - 1, -1, -1)
            if _shared_repo_long_horizon_surface(selected[index]) == "shared_repo_synthetic_worker"
            and _shared_repo_id(selected[index]) != chosen_repo_id
        ),
        -1,
    )
    if replacement_index < 0:
        replacement_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if not _shared_repo_long_horizon_surface(selected[index])
            ),
            -1,
        )
    if replacement_index < 0:
        replacement_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if _shared_repo_long_horizon_surface(selected[index]) == "shared_repo_synthetic_worker"
                and (
                    _shared_repo_id(selected[index]) != chosen_repo_id
                    or _shared_repo_worker_branch(selected[index]) not in chosen_required_branches
                )
            ),
            -1,
        )
    if replacement_index < 0:
        return selected
    selected[replacement_index] = chosen_integrator
    return selected


def _limit_tasks_for_compare(
    tasks: list,
    task_limit: int,
    *,
    priority_families: Sequence[str] | None = None,
    priority_family_weights: dict[str, object] | None = None,
    prefer_low_cost_tasks: bool = False,
    required_executable_families: Sequence[str] | None = None,
) -> list:
    grouped: dict[str, list] = {}
    for task in tasks:
        family = str(task.metadata.get("benchmark_family", "bounded"))
        grouped.setdefault(family, []).append(task)
    required_executable_family_set = {
        str(family).strip()
        for family in list(required_executable_families or [])
        if str(family).strip()
    }
    if prefer_low_cost_tasks and required_executable_family_set:
        for family, family_tasks in list(grouped.items()):
            if family not in required_executable_family_set:
                continue
            executable_tasks = _drop_retrieval_companions_when_sources_present(family_tasks)
            if executable_tasks:
                grouped[family] = executable_tasks
    if prefer_low_cost_tasks:
        for family, family_tasks in grouped.items():
            if family in required_executable_family_set:
                family_tasks.sort(key=_required_executable_family_sort_key)
                continue
            family_tasks.sort(key=_low_cost_task_key)
    else:
        for family_tasks in grouped.values():
            family_tasks.sort(
                key=lambda task: (
                    int(not bool(getattr(task, "metadata", {}).get("light_supervision_candidate", False))),
                )
            )

    selected = []

    def _consume_round_robin(families: list[str], *, limit: int) -> None:
        while len(selected) < limit:
            added = False
            for family in families:
                family_tasks = grouped.get(family, [])
                if family_tasks:
                    selected.append(family_tasks.pop(0))
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break

    def _consume_family(family: str, *, limit: int) -> None:
        if limit <= 0:
            return
        family_tasks = grouped.get(family, [])
        while family_tasks and limit > 0 and len(selected) < task_limit:
            selected.append(family_tasks.pop(0))
            limit -= 1

    prioritized = [
        family
        for family in _normalize_priority_families(priority_families)
        if family in grouped
    ]
    if prioritized:
        slot_targets = _priority_family_slot_targets(
            prioritized,
            task_limit,
            priority_weights=_normalize_priority_family_weights(priority_family_weights),
        )
        for family in prioritized:
            _consume_family(family, limit=slot_targets.get(family, 0))

    families = prioritized + [
        family
        for family in sorted(grouped)
        if family not in prioritized
    ]
    _consume_round_robin(families, limit=task_limit)
    selected = _reserve_shared_repo_integrator_slot(
        selected,
        grouped,
        priority_families=priority_families,
    )
    return _reorder_shared_repo_bundle_tasks(selected)
