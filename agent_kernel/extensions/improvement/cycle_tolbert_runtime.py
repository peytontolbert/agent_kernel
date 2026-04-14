from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Callable, Mapping

from ...config import KernelConfig


def is_retryable_tolbert_startup_failure(error_text: str) -> bool:
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return (
        "tolbert service failed to become ready" in normalized
        or "tolbert service exited before startup ready" in normalized
    )


def new_tolbert_runtime_summary(*, use_tolbert_context: bool) -> dict[str, object]:
    return {
        "configured_to_use_tolbert": bool(use_tolbert_context),
        "stages_attempted": [],
        "successful_tolbert_stages": [],
        "startup_failure_stages": [],
        "recovered_without_tolbert_stages": [],
        "bypassed_stages": [],
        "startup_failure_count": 0,
        "outcome": "pending" if bool(use_tolbert_context) else "bypassed",
    }


def ordered_unique_stage_names(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def normalize_tolbert_runtime_summary(
    summary: Mapping[str, object] | None,
    *,
    use_tolbert_context: bool | None = None,
) -> dict[str, object]:
    payload = dict(summary) if isinstance(summary, Mapping) else {}
    configured = (
        bool(payload.get("configured_to_use_tolbert", False))
        if use_tolbert_context is None
        else bool(use_tolbert_context)
    )
    normalized = new_tolbert_runtime_summary(use_tolbert_context=configured)
    for key in (
        "stages_attempted",
        "successful_tolbert_stages",
        "startup_failure_stages",
        "recovered_without_tolbert_stages",
        "bypassed_stages",
    ):
        normalized[key] = ordered_unique_stage_names(payload.get(key, []))
    normalized["startup_failure_count"] = len(normalized["startup_failure_stages"])
    if payload.get("outcome"):
        normalized["outcome"] = str(payload.get("outcome", "")).strip()
    return finalize_tolbert_runtime_summary(normalized)


def mark_tolbert_stage(summary: dict[str, object] | None, key: str, stage_name: str) -> None:
    if not isinstance(summary, dict):
        return
    token = str(stage_name).strip()
    if not token:
        return
    values = summary.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        summary[key] = values
    if token not in values:
        values.append(token)


def finalize_tolbert_runtime_summary(summary: Mapping[str, object] | None) -> dict[str, object]:
    normalized = new_tolbert_runtime_summary(
        use_tolbert_context=bool(dict(summary or {}).get("configured_to_use_tolbert", False))
    )
    normalized.update(dict(summary or {}))
    for key in (
        "stages_attempted",
        "successful_tolbert_stages",
        "startup_failure_stages",
        "recovered_without_tolbert_stages",
        "bypassed_stages",
    ):
        normalized[key] = ordered_unique_stage_names(normalized.get(key, []))
    normalized["startup_failure_count"] = len(normalized["startup_failure_stages"])
    configured = bool(normalized.get("configured_to_use_tolbert", False))
    if normalized["startup_failure_stages"]:
        outcome = "failed_recovered"
    elif normalized["successful_tolbert_stages"]:
        outcome = "succeeded"
    elif normalized["bypassed_stages"] or not configured:
        outcome = "bypassed"
    elif normalized["stages_attempted"]:
        outcome = "pending"
    else:
        outcome = "not_exercised"
    normalized["outcome"] = outcome
    normalized["used_tolbert_successfully"] = bool(normalized["successful_tolbert_stages"])
    normalized["recovered_without_tolbert"] = bool(normalized["recovered_without_tolbert_stages"])
    normalized["bypassed"] = outcome == "bypassed"
    return normalized


def evaluate_subsystem_metrics_with_tolbert_startup_retry(
    *,
    evaluate_subsystem_metrics: Callable[..., object],
    config: KernelConfig,
    subsystem: str,
    flags: dict[str, object],
    progress_label: str | None,
    phase_name: str,
    progress: Callable[[str], None] | None,
    tolbert_runtime_summary: dict[str, object] | None = None,
):
    if isinstance(tolbert_runtime_summary, dict):
        tolbert_runtime_summary["configured_to_use_tolbert"] = bool(config.use_tolbert_context)
        mark_tolbert_stage(tolbert_runtime_summary, "stages_attempted", phase_name)
    try:
        result = evaluate_subsystem_metrics(
            config=config,
            subsystem=subsystem,
            flags=flags,
            progress_label=progress_label,
        )
        if bool(config.use_tolbert_context):
            mark_tolbert_stage(tolbert_runtime_summary, "successful_tolbert_stages", phase_name)
        else:
            mark_tolbert_stage(tolbert_runtime_summary, "bypassed_stages", phase_name)
        return result
    except RuntimeError as exc:
        if not bool(config.use_tolbert_context) or not is_retryable_tolbert_startup_failure(str(exc)):
            raise
        mark_tolbert_stage(tolbert_runtime_summary, "startup_failure_stages", phase_name)
        if progress is not None:
            progress(
                f"finalize phase={phase_name}_retry subsystem={subsystem} "
                "reason=tolbert_startup_failure use_tolbert_context=0"
            )
        mark_tolbert_stage(tolbert_runtime_summary, "recovered_without_tolbert_stages", phase_name)
        mark_tolbert_stage(tolbert_runtime_summary, "bypassed_stages", phase_name)
        return evaluate_subsystem_metrics(
            config=replace(config, use_tolbert_context=False),
            subsystem=subsystem,
            flags=flags,
            progress_label=progress_label,
        )


def call_tolbert_preview_eval(
    *,
    retry_evaluator: Callable[..., object],
    config: KernelConfig,
    subsystem: str,
    flags: dict[str, object],
    progress_label: str | None,
    phase_name: str,
    progress: Callable[[str], None] | None,
    tolbert_runtime_summary: dict[str, object] | None = None,
):
    parameters = inspect.signature(retry_evaluator).parameters
    kwargs = {
        "config": config,
        "subsystem": subsystem,
        "flags": flags,
        "progress_label": progress_label,
        "phase_name": phase_name,
        "progress": progress,
    }
    if "tolbert_runtime_summary" in parameters:
        kwargs["tolbert_runtime_summary"] = tolbert_runtime_summary
    return retry_evaluator(**kwargs)
