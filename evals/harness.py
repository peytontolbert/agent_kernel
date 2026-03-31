from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

from agent_kernel.capabilities import capability_enabled, declared_task_capabilities
from agent_kernel.config import KernelConfig
from agent_kernel.curriculum import CurriculumEngine
from agent_kernel.loop import AgentKernel
from agent_kernel.operator_policy import operator_policy_snapshot
from agent_kernel.policy import Policy
from agent_kernel.runtime_supervision import atomic_write_json
from agent_kernel.schemas import ActionDecision
from agent_kernel.task_bank import (
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


def _emit_eval_progress(progress_label: str | None, message: str) -> None:
    if not progress_label:
        return
    print(f"[eval:{progress_label}] {message}", file=sys.stderr, flush=True)


def _should_emit_task_progress(index: int, total: int) -> bool:
    return index == 1 or index == total or index % 10 == 0


def _task_progress_label(task) -> str:
    family = "bounded"
    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, dict):
        family = str(metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    return f"{task.task_id} family={family}"


def _run_tasks_with_progress(
    tasks: list,
    kernel: AgentKernel,
    *,
    progress_label: str | None,
    phase: str = "",
    on_result=None,
    on_task_start=None,
    on_task_progress=None,
    on_task_complete=None,
):
    total = len(tasks)
    results = []
    for index, task in enumerate(tasks, start=1):
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

        result = _run_kernel_task(
            kernel,
            task,
            progress_callback=_task_progress if on_task_progress is not None else None,
        )
        results.append(result)
        if on_task_complete is not None:
            on_task_complete(task, result, index, total)
        if on_result is not None:
            on_result(task, result, index, total)
    return results


def _run_kernel_task(kernel: AgentKernel, task, *, progress_callback=None):
    if progress_callback is None:
        return kernel.run_task(task)
    try:
        return kernel.run_task(task, progress_callback=progress_callback)
    except TypeError as exc:
        if "progress_callback" not in str(exc):
            raise
        return kernel.run_task(task)


def _task_outcome_summary(task, result, *, low_confidence_threshold: float) -> dict[str, object]:
    benchmark_family = str(task.metadata.get("benchmark_family", "bounded"))
    memory_source = str(task.metadata.get("memory_source", "none"))
    no_state_progress_steps = sum(
        1 for step in result.steps if "no_state_progress" in list(step.failure_signals or [])
    )
    state_regression_steps = sum(1 for step in result.steps if int(step.state_regression_count) > 0)
    low_confidence_steps = sum(
        1
        for step in result.steps
        if 0.0 < float(step.path_confidence) < low_confidence_threshold
    )
    return {
        "task_id": task.task_id,
        "success": bool(result.success),
        "benchmark_family": benchmark_family,
        "memory_source": memory_source,
        "termination_reason": str(result.termination_reason or "unspecified"),
        "steps": len(result.steps),
        "low_confidence_steps": low_confidence_steps,
        "retrieval_influenced_steps": sum(1 for step in result.steps if step.retrieval_influenced),
        "trusted_retrieval_steps": sum(1 for step in result.steps if step.trust_retrieval),
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
    generated_passed = 0
    for task, result in zip(completed_generated_tasks, completed_generated_results):
        generated_kind = str(task.metadata.get("curriculum_kind", "unknown"))
        generated_by_kind[generated_kind] = generated_by_kind.get(generated_kind, 0) + 1
        if bool(result.success):
            generated_passed += 1
            generated_passed_by_kind[generated_kind] = generated_passed_by_kind.get(generated_kind, 0) + 1
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
        "generated_by_kind": generated_by_kind,
        "generated_passed_by_kind": generated_passed_by_kind,
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


def _proposal_metrics_by_benchmark_family(
    task_trajectories: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(task_trajectories, dict):
        return {}
    summary: dict[str, dict[str, object]] = {}
    for payload in task_trajectories.values():
        if not isinstance(payload, dict):
            continue
        family = str(payload.get("benchmark_family", "bounded")).strip() or "bounded"
        row = summary.setdefault(
            family,
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
    return {
        "present": any(
            value > 0.0
            for value in (progress_signal, risk_signal, decoder_progress, decoder_risk)
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


def _finalize_world_feedback_summary(
    overall_stats: dict[str, dict[str, float]],
    family_stats: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    overall = _finalize_world_feedback_row(dict(overall_stats.get("row", {})))
    by_family = {
        family: _finalize_world_feedback_row(dict(payload.get("row", {})))
        for family, payload in sorted(family_stats.items())
    }
    overall["distinct_benchmark_families"] = sum(
        1 for payload in by_family.values() if int(payload.get("step_count", 0) or 0) > 0
    )
    return overall, by_family


class _ForcedFailurePolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="force deterministic failure for curriculum evaluation",
            action="code_execute",
            content="false",
            done=False,
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
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
    except OSError:
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
    progress_label: str | None = None,
    progress_snapshot_path: Path | None = None,
    skill_transfer_target_map: dict[str, str] | None = None,
    operator_target_map: dict[str, str] | None = None,
) -> EvalMetrics:
    active_config = config or KernelConfig()
    try:
        bank = TaskBank(config=active_config)
    except TypeError:
        bank = TaskBank()
    kernel = AgentKernel(config=active_config)
    try:
        tasks = [task for task in bank.list() if _task_allowed_for_eval(task, active_config)]
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
        if task_limit is not None and task_limit > 0 and len(tasks) > task_limit:
            tasks = _limit_tasks_for_compare(
                tasks,
                task_limit,
                priority_families=priority_benchmark_families,
                priority_family_weights=priority_benchmark_family_weights,
                prefer_low_cost_tasks=prefer_low_cost_tasks,
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

        results = _run_tasks_with_progress(
            tasks,
            kernel,
            progress_label=progress_label,
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
                completed_primary_tasks.append(task),
                completed_primary_results.append(result),
                _emit_progress_snapshot("primary"),
            ),
        )
        generated_tasks = []
        generated_results = []
        if include_generated or include_failure_generated:
            engine = CurriculumEngine(memory_root=active_config.trajectories_root, config=active_config)
            if include_generated:
                _emit_eval_progress(progress_label, "phase=generated_success_schedule")
                success_generated_config = _scoped_config(active_config, "generated_success")
                success_generated_kernel = AgentKernel(config=success_generated_config)
                try:
                    success_seed_results = engine.schedule_generated_seed_episodes(
                        results,
                        curriculum_kind="adjacent_success",
                    )
                    success_generated_tasks = [engine.generate_followup_task(result) for result in success_seed_results]
                    _emit_eval_progress(
                        progress_label,
                        f"phase=generated_success total={len(success_generated_tasks)}",
                    )
                    generated_tasks.extend(success_generated_tasks)
                    generated_tasks_scheduled = len(generated_tasks)
                    _emit_progress_snapshot("generated_success_schedule")
                    generated_results.extend(
                        _run_tasks_with_progress(
                            success_generated_tasks,
                            success_generated_kernel,
                            progress_label=progress_label,
                            phase="generated_success",
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
                                completed_generated_tasks.append(task),
                                completed_generated_results.append(result),
                                _emit_progress_snapshot("generated_success"),
                            ),
                        )
                    )
                finally:
                    success_generated_kernel.close()
                    _cleanup_scoped_runtime_state(success_generated_config)
            if include_failure_generated:
                _emit_eval_progress(progress_label, f"phase=generated_failure_seed total={len(tasks)}")
                failure_seed_config = _scoped_config(active_config, "generated_failure_seed")
                failing_kernel = AgentKernel(config=failure_seed_config, policy=_ForcedFailurePolicy())
                failure_generated_config = _scoped_config(active_config, "generated_failure")
                failure_generated_kernel = AgentKernel(config=failure_generated_config)
                try:
                    failure_seeds = _run_tasks_with_progress(
                        tasks,
                        failing_kernel,
                        progress_label=progress_label,
                        phase="generated_failure_seed",
                    )
                finally:
                    failing_kernel.close()
                    _cleanup_scoped_runtime_state(failure_seed_config)
                try:
                    failure_seed_results = engine.schedule_generated_seed_episodes(
                        failure_seeds,
                        curriculum_kind="failure_recovery",
                    )
                    failure_generated_tasks = [engine.generate_followup_task(result) for result in failure_seed_results]
                    _emit_eval_progress(
                        progress_label,
                        f"phase=generated_failure total={len(failure_generated_tasks)}",
                    )
                    generated_tasks.extend(failure_generated_tasks)
                    generated_tasks_scheduled = len(generated_tasks)
                    _emit_progress_snapshot("generated_failure_schedule")
                    generated_results.extend(
                        _run_tasks_with_progress(
                            failure_generated_tasks,
                            failure_generated_kernel,
                            progress_label=progress_label,
                            phase="generated_failure",
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
                                completed_generated_tasks.append(task),
                                completed_generated_results.append(result),
                                _emit_progress_snapshot("generated_failure"),
                            ),
                        )
                    )
                finally:
                    failure_generated_kernel.close()
                    _cleanup_scoped_runtime_state(failure_generated_config)
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=max(0.0, _PROGRESS_HEARTBEAT_INTERVAL_SECONDS))
        _set_current_task(None)
        _emit_progress_snapshot("complete")
    finally:
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
            "task_id": task.task_id,
            "success": bool(result.success),
            "steps": len(result.steps),
            "benchmark_family": benchmark_family,
            "memory_source": memory_source,
            "termination_reason": termination_reason,
            "unsafe_ambiguous": unsafe_ambiguous,
            "hidden_side_effect_risk": hidden_side_effect_risk,
            "first_step_verified": bool(result.steps and result.steps[0].verification.get("passed", False)),
            "low_confidence_steps": low_confidence_step_count,
            "trusted_retrieval_steps": sum(1 for step in result.steps if step.trust_retrieval),
            "retrieval_influenced_steps": sum(1 for step in result.steps if step.retrieval_influenced),
            "proposal_selected_steps": sum(1 for step in result.steps if str(step.proposal_source).strip()),
            "novel_command_steps": sum(1 for step in result.steps if bool(step.proposal_novel)),
            "novel_valid_command_steps": sum(
                1
                for step in result.steps
                if bool(step.proposal_novel) and bool(step.verification.get("passed", False))
            ),
            "no_state_progress_steps": no_state_progress_steps,
            "state_regression_steps": state_regression_steps,
            "total_state_regression_count": total_state_regression_count,
            "max_state_regression_count": max(
                (int(step.state_regression_count) for step in result.steps),
                default=0,
            ),
            "failure_signals": sorted(
                {
                    str(signal).strip()
                    for step in result.steps
                    for signal in list(step.failure_signals or [])
                    if str(signal).strip()
                }
            ),
            "completion_ratio": float(world_summary.get("completion_ratio", 0.0) or 0.0),
            "present_forbidden_artifact_count": len(list(world_summary.get("present_forbidden_artifacts", []))),
            "changed_preserved_artifact_count": len(list(world_summary.get("changed_preserved_artifacts", []))),
            "missing_expected_artifact_count": len(list(world_summary.get("missing_expected_artifacts", []))),
        }
        task_trajectories[task.task_id] = {
            "task_id": task.task_id,
            "benchmark_family": benchmark_family,
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
                    "decision_source": str(step.decision_source).strip(),
                    "tolbert_route_mode": str(step.tolbert_route_mode).strip(),
                    "proposal_source": str(step.proposal_source).strip(),
                    "proposal_novel": bool(step.proposal_novel),
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
            _update_world_feedback_stats(episode_world_feedback_overall_stats, feedback=feedback, observed=observed)
            step_payload["world_feedback"] = {
                "progress_signal": float(feedback.get("progress_signal", 0.0) or 0.0),
                "risk_signal": float(feedback.get("risk_signal", 0.0) or 0.0),
                "decoder_progress_signal": float(feedback.get("decoder_progress_signal", 0.0) or 0.0),
                "decoder_risk_signal": float(feedback.get("decoder_risk_signal", 0.0) or 0.0),
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
    reusable_skills = 0
    if active_config.skills_path.exists():
        payload = json.loads(active_config.skills_path.read_text(encoding="utf-8"))
        reusable_skills = len(payload.get("skills", payload)) if isinstance(payload, dict) else len(payload)
    memory_documents = len(list(active_config.trajectories_root.glob("*.json")))
    world_feedback_summary, world_feedback_by_benchmark_family = _finalize_world_feedback_summary(
        world_feedback_overall_stats,
        world_feedback_family_stats,
    )
    return EvalMetrics(
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
        world_feedback_summary=world_feedback_summary,
        world_feedback_by_benchmark_family=world_feedback_by_benchmark_family,
        task_outcomes=task_outcomes,
        task_trajectories=task_trajectories,
    )


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


def _priority_family_slot_targets(
    prioritized: list[str],
    task_limit: int,
    *,
    priority_weights: dict[str, float],
) -> dict[str, int]:
    if task_limit <= 0 or not prioritized:
        return {}
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


def _low_cost_task_key(task) -> tuple[int, int, int, int, int, str]:
    metadata = getattr(task, "metadata", {}) or {}
    difficulty = str(metadata.get("difficulty", "")).strip()
    difficulty_rank = {
        "seed": 0,
        "bounded": 1,
        "": 2,
        "retrieval": 3,
    }.get(difficulty, 2)
    requires_retrieval_rank = 1 if bool(metadata.get("requires_retrieval", False)) else 0
    return (
        difficulty_rank,
        requires_retrieval_rank,
        int(getattr(task, "max_steps", 0) or 0),
        len(getattr(task, "suggested_commands", []) or []),
        len(str(getattr(task, "prompt", "") or "")),
        str(getattr(task, "task_id", "")),
    )


def _limit_tasks_for_compare(
    tasks: list,
    task_limit: int,
    *,
    priority_families: Sequence[str] | None = None,
    priority_family_weights: dict[str, object] | None = None,
    prefer_low_cost_tasks: bool = False,
) -> list:
    grouped: dict[str, list] = {}
    for task in tasks:
        family = str(task.metadata.get("benchmark_family", "bounded"))
        grouped.setdefault(family, []).append(task)
    if prefer_low_cost_tasks:
        for family_tasks in grouped.values():
            family_tasks.sort(key=_low_cost_task_key)

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

    families = sorted(grouped)
    _consume_round_robin(families, limit=task_limit)
    return selected
