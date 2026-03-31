from __future__ import annotations

import importlib.util
from dataclasses import asdict, fields, replace
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from datetime import datetime, timezone
import re
import tempfile
import time
from uuid import uuid4

from agent_kernel.cycle_runner import active_artifact_path_for_subsystem, finalize_cycle
from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner, staged_candidate_artifact_path
from agent_kernel.runtime_supervision import atomic_write_json, terminate_process_tree
from evals.harness import run_eval, scoped_improvement_cycle_config
from evals.metrics import EvalMetrics


def _load_autonomous_cycle_module():
    module_name = "_agentkernel_run_improvement_cycle"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    module_path = SCRIPTS_ROOT / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load improvement cycle module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


autonomous_cycle = _load_autonomous_cycle_module()

_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES = (
    "bounded",
    "episode_memory",
    "tool_memory",
)
_OBSERVATION_PROFILE_DEFAULT = "default"
_OBSERVATION_PROFILE_SMOKE = "smoke"
_SMOKE_OBSERVATION_TASK_LIMIT = 1
_SMOKE_OBSERVATION_BUDGET_SECONDS = 24.0
_SMOKE_CURRENT_TASK_DECISION_BUDGET_SECONDS = 8.0
_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS = 12.0
_CURRENT_TASK_PRESTEP_BUDGET_STAGES = {"context_compile", "decision_pending"}
_CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS = {
    "query_build": 2.0,
    "tolbert_query": 3.0,
    "guidance_build": 1.0,
    "retrieval_query": 1.5,
    "skill_query": 1.5,
    "skill_rank": 1.0,
    "tool_query": 1.5,
    "tool_plan": 1.0,
    "paper_research_query": 2.0,
    "paper_research_merge": 1.0,
    "retrieval_normalize": 1.5,
    "chunk_select": 1.5,
    "complete": 0.5,
    "verifier_query": 1.0,
}
_CURRENT_TASK_CONTEXT_COMPILE_UNKNOWN_SUBPHASE_BUDGET_SECONDS = 1.0


def _progress(progress_label: str | None, message: str) -> None:
    if not progress_label:
        return
    print(f"[supervised:{progress_label}] {message}", file=sys.stderr, flush=True)


def _cycle_id_for_experiment(subsystem: str) -> str:
    return (
        f"cycle:{subsystem}:"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}:"
        f"{uuid4().hex[:8]}"
    )


def _sanitize_scope_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or f"supervised_scope_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _select_guided_experiment(ranked_experiments):
    if not ranked_experiments:
        raise SystemExit("no ranked experiments are available for the guided cycle")
    top_score = float(ranked_experiments[0].score)
    score_floor = top_score * 0.8
    candidates = [experiment for experiment in ranked_experiments if float(experiment.score) >= score_floor]
    candidates.sort(
        key=lambda experiment: (
            -int(experiment.priority),
            int(experiment.estimated_cost),
            -float(experiment.expected_gain),
            -float(experiment.score),
            str(experiment.subsystem),
        )
    )
    return candidates[0]


def _select_guided_variant(ranked_variants):
    if not ranked_variants:
        raise SystemExit("no ranked variants are available for the guided cycle")
    top_score = float(ranked_variants[0].score)
    score_floor = top_score * 0.8
    candidates = [variant for variant in ranked_variants if float(variant.score) >= score_floor]
    candidates.sort(
        key=lambda variant: (
            int(variant.estimated_cost),
            -float(variant.expected_gain),
            -float(variant.score),
            str(variant.variant_id),
        )
    )
    return candidates[0]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _kernel_config_snapshot(config: KernelConfig) -> dict[str, object]:
    return {field.name: _json_ready(getattr(config, field.name)) for field in fields(KernelConfig)}


def _kernel_config_from_snapshot(snapshot: object) -> KernelConfig:
    defaults = KernelConfig()
    if not isinstance(snapshot, dict):
        return defaults
    kwargs: dict[str, object] = {}
    for field in fields(KernelConfig):
        if field.name not in snapshot:
            continue
        value = snapshot[field.name]
        current = getattr(defaults, field.name)
        if isinstance(current, Path):
            kwargs[field.name] = Path(value) if value is not None else current
        elif isinstance(current, tuple):
            if isinstance(value, (list, tuple)):
                kwargs[field.name] = tuple(value)
            elif value is None:
                kwargs[field.name] = tuple()
            else:
                kwargs[field.name] = (value,)
        else:
            kwargs[field.name] = value
    return KernelConfig(**kwargs)


def _observation_child_entry(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = _kernel_config_from_snapshot(payload.get("config"))
    config.ensure_directories()
    result_path = Path(str(payload.get("result_path", "")).strip())
    progress_path_token = str(payload.get("progress_path", "")).strip()
    progress_path = Path(progress_path_token) if progress_path_token else None
    eval_kwargs = payload.get("eval_kwargs", {})
    if not isinstance(eval_kwargs, dict):
        raise SystemExit("invalid observation child payload: eval_kwargs")
    progress_label = str(payload.get("progress_label", "")).strip() or None
    try:
        metrics = run_eval(
            config=config,
            progress_label=progress_label,
            progress_snapshot_path=progress_path,
            **eval_kwargs,
        )
    except Exception as exc:
        if result_path:
            atomic_write_json(result_path, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
    if result_path:
        atomic_write_json(result_path, {"ok": True, "metrics": _json_ready(asdict(metrics))})


def _metrics_from_child_payload(payload: object) -> EvalMetrics | None:
    if not isinstance(payload, dict):
        return None
    metrics_payload = payload.get("metrics", {})
    if not isinstance(metrics_payload, dict):
        return None
    try:
        return EvalMetrics(**metrics_payload)
    except TypeError:
        return None


def _read_text_tail(path: Path, *, max_lines: int = 20) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _read_progress_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_task_prestep_budget_stage(partial_summary: dict[str, object]) -> str:
    current_task_id = str(partial_summary.get("current_task_id", "")).strip()
    if not current_task_id:
        return ""
    current_task_step_stage = str(partial_summary.get("current_task_step_stage", "")).strip()
    current_task_step_subphase = str(partial_summary.get("current_task_step_subphase", "")).strip()
    if current_task_step_stage in _CURRENT_TASK_PRESTEP_BUDGET_STAGES:
        return current_task_step_stage
    if current_task_step_stage in _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS:
        return current_task_step_stage
    current_task_completed_steps = int(partial_summary.get("current_task_completed_steps", 0) or 0)
    current_task_step_index = int(partial_summary.get("current_task_step_index", 0) or 0)
    current_task_step_action = str(partial_summary.get("current_task_step_action", "")).strip()
    if (
        not current_task_step_stage
        and current_task_step_subphase
        and current_task_step_subphase in _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS
        and current_task_completed_steps <= 0
        and current_task_step_index <= 1
        and not current_task_step_action
    ):
        return current_task_step_subphase
    if (
        not current_task_step_stage
        and current_task_completed_steps <= 0
        and current_task_step_index <= 1
        and not current_task_step_action
    ):
        return "prestep_unclassified"
    return ""


def _resolve_current_task_prestep_subphase_budget_seconds(
    *,
    partial_summary: dict[str, object],
    decision_budget_seconds: float,
) -> float:
    current_task_step_stage = str(partial_summary.get("current_task_step_stage", "")).strip()
    progress_step_budget_seconds = float(partial_summary.get("current_task_step_budget_seconds", 0.0) or 0.0)
    if progress_step_budget_seconds > 0.0:
        if decision_budget_seconds > 0.0:
            return min(decision_budget_seconds, progress_step_budget_seconds)
        return progress_step_budget_seconds
    current_task_step_subphase = str(partial_summary.get("current_task_step_subphase", "")).strip()
    subphase_budget = 0.0
    if (
        current_task_step_stage == "context_compile"
        and current_task_step_subphase
        and current_task_step_subphase in _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS
    ):
        subphase_budget = float(
            _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS.get(current_task_step_subphase, 0.0) or 0.0
        )
    elif current_task_step_stage in _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS:
        subphase_budget = float(_CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS.get(current_task_step_stage, 0.0) or 0.0)
    elif (
        not current_task_step_stage
        and current_task_step_subphase
        and current_task_step_subphase in _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS
    ):
        subphase_budget = float(
            _CURRENT_TASK_CONTEXT_COMPILE_SUBPHASE_BUDGETS.get(current_task_step_subphase, 0.0) or 0.0
        )
    elif current_task_step_stage == "context_compile":
        subphase_budget = _CURRENT_TASK_CONTEXT_COMPILE_UNKNOWN_SUBPHASE_BUDGET_SECONDS
    if subphase_budget <= 0.0:
        return 0.0
    if decision_budget_seconds > 0.0:
        return min(decision_budget_seconds, subphase_budget)
    return subphase_budget


def _observation_progress_snapshot(stderr_tail: str, *, progress_label: str | None) -> dict[str, str]:
    lines = [line.strip() for line in str(stderr_tail).splitlines() if line.strip()]
    if not lines:
        return {
            "last_progress_line": "",
            "last_progress_phase": "",
            "last_progress_task_id": "",
            "last_progress_benchmark_family": "",
        }
    preferred_prefix = f"[eval:{progress_label}]" if progress_label else "[eval:"
    progress_lines = [line for line in lines if line.startswith(preferred_prefix)]
    if not progress_lines and progress_label:
        progress_lines = [line for line in lines if line.startswith("[eval:")]
    last_line = progress_lines[-1] if progress_lines else lines[-1]
    phase_match = re.search(r"phase=([A-Za-z0-9_.-]+)", last_line)
    task_match = re.search(r"task\s+\d+/\d+\s+([^\s]+)", last_line)
    benchmark_family_match = re.search(r"family=([A-Za-z0-9_.-]+)", last_line)
    return {
        "last_progress_line": last_line,
        "last_progress_phase": phase_match.group(1) if phase_match else "",
        "last_progress_task_id": task_match.group(1) if task_match else "",
        "last_progress_benchmark_family": benchmark_family_match.group(1) if benchmark_family_match else "",
    }


def _apply_curriculum_flags(eval_kwargs: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    enriched = dict(eval_kwargs)
    enriched["include_generated"] = bool(args.include_curriculum)
    enriched["include_failure_generated"] = bool(args.include_failure_curriculum)
    return enriched


def _apply_bounded_primary_priority_defaults(
    eval_kwargs: dict[str, object],
    *,
    max_observation_seconds: float,
) -> tuple[dict[str, object], str]:
    enriched = dict(eval_kwargs)
    if max(0.0, float(max_observation_seconds or 0.0)) <= 0.0:
        return enriched, ""
    task_limit = int(enriched.get("task_limit", 0) or 0)
    if task_limit <= 0:
        return enriched, ""
    existing = [
        str(value).strip()
        for value in list(enriched.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if existing:
        enriched["priority_benchmark_families"] = existing
        return enriched, "explicit"
    enriched["priority_benchmark_families"] = list(_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES)
    enriched["prefer_low_cost_tasks"] = True
    return enriched, "bounded_default"


def _apply_observation_profile_defaults(args: argparse.Namespace) -> dict[str, object]:
    profile = str(getattr(args, "observation_profile", "") or "").strip().lower() or _OBSERVATION_PROFILE_DEFAULT
    applied: dict[str, object] = {"profile": profile}
    if profile != _OBSERVATION_PROFILE_SMOKE:
        return applied
    task_limit = max(0, int(getattr(args, "task_limit", 0) or 0))
    if task_limit <= 0:
        args.task_limit = _SMOKE_OBSERVATION_TASK_LIMIT
        applied["task_limit"] = _SMOKE_OBSERVATION_TASK_LIMIT
    max_observation_seconds = max(0.0, float(getattr(args, "max_observation_seconds", 0.0) or 0.0))
    if max_observation_seconds <= 0.0:
        args.max_observation_seconds = _SMOKE_OBSERVATION_BUDGET_SECONDS
        applied["max_observation_seconds"] = _SMOKE_OBSERVATION_BUDGET_SECONDS
    current_task_decision_budget_seconds = max(
        0.0,
        float(getattr(args, "max_current_task_decision_seconds", 0.0) or 0.0),
    )
    if current_task_decision_budget_seconds <= 0.0:
        args.max_current_task_decision_seconds = _SMOKE_CURRENT_TASK_DECISION_BUDGET_SECONDS
        applied["max_current_task_decision_seconds"] = _SMOKE_CURRENT_TASK_DECISION_BUDGET_SECONDS
    return applied


def _resolve_current_task_decision_budget_seconds(
    args: argparse.Namespace,
    *,
    max_observation_seconds: float,
    observation_profile: str,
) -> tuple[float, str]:
    explicit = max(0.0, float(getattr(args, "max_current_task_decision_seconds", 0.0) or 0.0))
    if explicit > 0.0:
        if observation_profile == _OBSERVATION_PROFILE_SMOKE and explicit == _SMOKE_CURRENT_TASK_DECISION_BUDGET_SECONDS:
            return explicit, "smoke_default"
        return explicit, "explicit"
    task_limit = max(0, int(getattr(args, "task_limit", 0) or 0))
    if task_limit > 0 and max_observation_seconds > 0.0:
        if task_limit == 1:
            return min(_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS, max_observation_seconds), "one_task_default"
        multi_task_limit = max(1, task_limit)
        multi_task_budget = max_observation_seconds / multi_task_limit
        if multi_task_budget > 0.0:
            return min(_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS, multi_task_budget), "multi_task_default"
    return 0.0, ""


def _without_generated_curriculum(eval_kwargs: dict[str, object]) -> dict[str, object]:
    degraded = dict(eval_kwargs)
    degraded["include_generated"] = False
    degraded["include_failure_generated"] = False
    return degraded


def _with_generated_curriculum(
    eval_kwargs: dict[str, object],
    *,
    include_generated: bool,
    include_failure_generated: bool,
) -> dict[str, object]:
    enriched = dict(eval_kwargs)
    enriched["include_generated"] = bool(include_generated)
    enriched["include_failure_generated"] = bool(include_failure_generated)
    return enriched


def _merge_int_maps(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[str(key)] = int(merged.get(str(key), 0) or 0) + int(value or 0)
    return merged


def _merge_max_int_maps(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for source in (left, right):
        for key, value in source.items():
            normalized = str(key)
            merged[normalized] = max(int(merged.get(normalized, 0) or 0), int(value or 0))
    return merged


def _progress_phase_rank(phase: str) -> int:
    normalized = str(phase).strip()
    order = {
        "": 0,
        "primary": 1,
        "generated_success_schedule": 2,
        "generated_success": 3,
        "generated_failure_seed": 4,
        "generated_failure": 5,
    }
    return order.get(normalized, 0)


def _merge_progress_context(
    current: dict[str, str],
    initial: dict[str, str],
) -> dict[str, str]:
    current_phase_rank = _progress_phase_rank(current.get("phase", ""))
    initial_phase_rank = _progress_phase_rank(initial.get("phase", ""))
    if initial_phase_rank > current_phase_rank:
        return dict(initial)
    if current_phase_rank > initial_phase_rank:
        return dict(current)
    if current.get("task_id") or current.get("line"):
        return dict(current)
    return dict(initial)


def _merge_partial_summary(
    current: dict[str, object],
    initial: dict[str, object],
) -> dict[str, object]:
    if not initial:
        return dict(current)
    if not current:
        return dict(initial)
    merged = dict(current)
    current_phase = str(current.get("phase", "")).strip()
    initial_phase = str(initial.get("phase", "")).strip()
    if _progress_phase_rank(initial_phase) > _progress_phase_rank(current_phase):
        merged["phase"] = initial_phase
    merged["observed_benchmark_families"] = sorted(
        {
            str(value).strip()
            for source in (
                list(current.get("observed_benchmark_families", []) or []),
                list(initial.get("observed_benchmark_families", []) or []),
            )
            for value in source
            if str(value).strip()
        }
    )
    for key in (
        "generated_tasks_scheduled",
        "completed_generated_tasks",
        "generated_passed",
        "remaining_primary_tasks",
        "completed_primary_tasks",
        "total_primary_tasks",
        "primary_passed",
    ):
        merged[key] = max(int(current.get(key, 0) or 0), int(initial.get(key, 0) or 0))
    for key in ("primary_pass_rate",):
        merged[key] = max(float(current.get(key, 0.0) or 0.0), float(initial.get(key, 0.0) or 0.0))
    merged["generated_by_kind"] = _merge_max_int_maps(
        dict(initial.get("generated_by_kind", {}) or {}),
        dict(current.get("generated_by_kind", {}) or {}),
    )
    merged["generated_passed_by_kind"] = _merge_max_int_maps(
        dict(initial.get("generated_passed_by_kind", {}) or {}),
        dict(current.get("generated_passed_by_kind", {}) or {}),
    )
    merged["total_by_benchmark_family"] = _merge_max_int_maps(
        dict(initial.get("total_by_benchmark_family", {}) or {}),
        dict(current.get("total_by_benchmark_family", {}) or {}),
    )
    merged["passed_by_benchmark_family"] = _merge_max_int_maps(
        dict(initial.get("passed_by_benchmark_family", {}) or {}),
        dict(current.get("passed_by_benchmark_family", {}) or {}),
    )
    merged["scheduled_task_order"] = [
        str(value).strip()
        for value in list(current.get("scheduled_task_order", []) or [])
        if str(value).strip()
    ] or [
        str(value).strip()
        for value in list(initial.get("scheduled_task_order", []) or [])
        if str(value).strip()
    ]
    merged["scheduled_task_summaries"] = dict(initial.get("scheduled_task_summaries", {}) or {})
    merged["scheduled_task_summaries"].update(
        {
            str(key): dict(value)
            for key, value in dict(current.get("scheduled_task_summaries", {}) or {}).items()
            if str(key).strip() and isinstance(value, dict)
        }
    )
    if not str(merged.get("last_completed_task_id", "")).strip():
        merged["last_completed_task_id"] = str(initial.get("last_completed_task_id", "")).strip()
    if not str(merged.get("last_completed_benchmark_family", "")).strip():
        merged["last_completed_benchmark_family"] = str(initial.get("last_completed_benchmark_family", "")).strip()
    return merged


def _current_partial_task_benchmark_family(summary: dict[str, object], *, current_task_id: str) -> str:
    task_id = str(current_task_id).strip()
    if not task_id:
        return ""
    scheduled = summary.get("scheduled_task_summaries", {})
    if not isinstance(scheduled, dict):
        return ""
    payload = scheduled.get(task_id, {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("benchmark_family", "")).strip()


def _partial_summary_metric_fields(summary: dict[str, object], *, current_task_id: str = "") -> dict[str, object]:
    return {
        "observation_partial_phase": str(summary.get("phase", "")).strip(),
        "observation_partial_tasks_completed": int(summary.get("completed_primary_tasks", 0) or 0),
        "observation_partial_current_task_completed_steps": int(summary.get("current_task_completed_steps", 0) or 0),
        "observation_partial_current_task_step_index": int(summary.get("current_task_step_index", 0) or 0),
        "observation_partial_current_task_step_stage": str(summary.get("current_task_step_stage", "")).strip(),
        "observation_partial_current_task_step_subphase": str(summary.get("current_task_step_subphase", "")).strip(),
        "observation_partial_current_task_step_action": str(summary.get("current_task_step_action", "")).strip(),
        "observation_partial_current_task_step_elapsed_seconds": float(
            summary.get("current_task_step_elapsed_seconds", 0.0) or 0.0
        ),
        "observation_partial_current_task_step_budget_seconds": float(
            summary.get("current_task_step_budget_seconds", 0.0) or 0.0
        ),
        "observation_partial_generated_tasks_scheduled": int(summary.get("generated_tasks_scheduled", 0) or 0),
        "observation_partial_generated_tasks_completed": int(summary.get("completed_generated_tasks", 0) or 0),
        "observation_partial_last_completed_task_id": str(summary.get("last_completed_task_id", "")).strip(),
        "observation_partial_last_completed_benchmark_family": str(
            summary.get("last_completed_benchmark_family", "")
        ).strip(),
        "observation_partial_observed_benchmark_families": [
            str(value).strip()
            for value in list(summary.get("observed_benchmark_families", []) or [])
            if str(value).strip()
        ],
        "observation_partial_current_task_benchmark_family": _current_partial_task_benchmark_family(
            summary,
            current_task_id=current_task_id,
        ),
        "observation_partial_current_task_progress_timeline": list(
            summary.get("current_task_progress_timeline", []) or []
        ),
    }


def _followup_partial_summary_fields(summary: dict[str, object], *, current_task_id: str = "") -> dict[str, object]:
    observation_fields = _partial_summary_metric_fields(summary, current_task_id=current_task_id)
    return {
        "partial_phase": str(observation_fields["observation_partial_phase"]),
        "partial_tasks_completed": int(observation_fields["observation_partial_tasks_completed"]),
        "partial_current_task_completed_steps": int(
            observation_fields["observation_partial_current_task_completed_steps"]
        ),
        "partial_current_task_step_index": int(observation_fields["observation_partial_current_task_step_index"]),
        "partial_current_task_step_stage": str(observation_fields["observation_partial_current_task_step_stage"]),
        "partial_current_task_step_subphase": str(
            observation_fields["observation_partial_current_task_step_subphase"]
        ),
        "partial_current_task_step_action": str(observation_fields["observation_partial_current_task_step_action"]),
        "partial_current_task_step_elapsed_seconds": float(
            observation_fields["observation_partial_current_task_step_elapsed_seconds"]
        ),
        "partial_current_task_step_budget_seconds": float(
            observation_fields["observation_partial_current_task_step_budget_seconds"]
        ),
        "partial_generated_tasks_scheduled": int(
            observation_fields["observation_partial_generated_tasks_scheduled"]
        ),
        "partial_generated_tasks_completed": int(
            observation_fields["observation_partial_generated_tasks_completed"]
        ),
        "partial_last_completed_task_id": str(
            observation_fields["observation_partial_last_completed_task_id"]
        ),
        "partial_last_completed_benchmark_family": str(
            observation_fields["observation_partial_last_completed_benchmark_family"]
        ),
        "partial_observed_benchmark_families": list(
            observation_fields["observation_partial_observed_benchmark_families"]
        ),
        "partial_current_task_progress_timeline": list(
            observation_fields["observation_partial_current_task_progress_timeline"]
        ),
    }


def _merge_generated_metrics(base: EvalMetrics, supplement: EvalMetrics) -> EvalMetrics:
    payload = asdict(base)
    payload["generated_total"] = int(base.generated_total or 0) + int(supplement.generated_total or 0)
    payload["generated_passed"] = int(base.generated_passed or 0) + int(supplement.generated_passed or 0)
    payload["generated_by_kind"] = _merge_int_maps(base.generated_by_kind, supplement.generated_by_kind)
    payload["generated_passed_by_kind"] = _merge_int_maps(
        base.generated_passed_by_kind,
        supplement.generated_passed_by_kind,
    )
    payload["generated_by_benchmark_family"] = _merge_int_maps(
        base.generated_by_benchmark_family,
        supplement.generated_by_benchmark_family,
    )
    payload["generated_passed_by_benchmark_family"] = _merge_int_maps(
        base.generated_passed_by_benchmark_family,
        supplement.generated_passed_by_benchmark_family,
    )
    return EvalMetrics(**payload)


def _run_observation_eval(
    *,
    config: KernelConfig,
    eval_kwargs: dict[str, object],
    progress_label: str | None,
    max_observation_seconds: float,
    current_task_decision_budget_seconds: float = 0.0,
) -> dict[str, object]:
    budget_seconds = max(0.0, float(max_observation_seconds))
    decision_budget_seconds = max(0.0, float(current_task_decision_budget_seconds or 0.0))
    active_config = config
    if (
        budget_seconds > 0.0
        and float(config.tolbert_context_compile_budget_seconds or 0.0) <= 0.0
    ):
        stage_budget_seconds = decision_budget_seconds if decision_budget_seconds > 0.0 else budget_seconds
        active_config = replace(
            config,
            tolbert_context_compile_budget_seconds=max(0.25, stage_budget_seconds),
        )
    if budget_seconds <= 0.0:
        return {
            "mode": "in_process",
            "metrics": run_eval(config=active_config, progress_label=progress_label, **eval_kwargs),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "current_task_decision_budget_exceeded": False,
            "current_task_decision_budget_seconds": decision_budget_seconds,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    with tempfile.TemporaryDirectory(prefix="agentkernel_supervised_observe_") as tmp_dir:
        scratch = Path(tmp_dir)
        payload_path = scratch / "payload.json"
        result_path = scratch / "result.json"
        progress_path = scratch / "progress.json"
        stdout_path = scratch / "stdout.log"
        stderr_path = scratch / "stderr.log"
        atomic_write_json(
            payload_path,
            {
                "config": _kernel_config_snapshot(active_config),
                "eval_kwargs": _json_ready(eval_kwargs),
                "progress_label": progress_label or "",
                "result_path": str(result_path),
                "progress_path": str(progress_path),
            },
        )
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_observation-child-payload", str(payload_path)]
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                start_new_session=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            returncode = None
            timeout_reason = ""
            current_task_timeout_budget_seconds = 0.0
            current_task_timeout_budget_source = "none"
            decision_budget_exceeded = False
            prestep_stage = ""
            prestep_subphase = ""
            prestep_subphase_budget_seconds = 0.0
            prestep_subphase_start_elapsed = 0.0
            deadline = time.monotonic() + budget_seconds
            while returncode is None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0.0:
                    timeout_reason = f"observation child exceeded max runtime of {budget_seconds:.1f} seconds"
                    break
                try:
                    returncode = process.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
                partial_summary = _read_progress_snapshot(progress_path)
                current_task_id = str(partial_summary.get("current_task_id", "")).strip()
                current_task_budget_stage = _current_task_prestep_budget_stage(partial_summary)
                current_task_step_subphase = str(partial_summary.get("current_task_step_subphase", "")).strip()
                current_task_elapsed_seconds = float(partial_summary.get("current_task_elapsed_seconds", 0.0) or 0.0)
                current_task_step_elapsed_seconds = float(
                    partial_summary.get("current_task_step_elapsed_seconds", current_task_elapsed_seconds) or 0.0
                )
                if current_task_id and current_task_budget_stage:
                    if (
                        current_task_budget_stage != prestep_stage
                        or current_task_step_subphase != prestep_subphase
                    ):
                        prestep_stage = current_task_budget_stage
                        prestep_subphase = current_task_step_subphase
                        prestep_subphase_budget_seconds = _resolve_current_task_prestep_subphase_budget_seconds(
                            partial_summary=partial_summary,
                            decision_budget_seconds=decision_budget_seconds,
                        )
                        prestep_subphase_start_elapsed = current_task_step_elapsed_seconds
                if (
                    decision_budget_seconds > 0.0
                    and current_task_id
                    and current_task_budget_stage
                    and current_task_elapsed_seconds >= decision_budget_seconds
                ):
                    decision_budget_exceeded = True
                    current_task_timeout_budget_seconds = decision_budget_seconds
                    current_task_timeout_budget_source = "decision_budget"
                    timeout_reason = (
                        "observation current task stage "
                        f"{current_task_budget_stage} exceeded max runtime of "
                        f"{decision_budget_seconds:.1f} seconds"
                    )
                    break
                if (
                    current_task_id
                    and current_task_budget_stage
                    and prestep_subphase_budget_seconds > 0.0
                    and current_task_step_elapsed_seconds - prestep_subphase_start_elapsed
                    >= prestep_subphase_budget_seconds
                ):
                    decision_budget_exceeded = True
                    current_task_timeout_budget_seconds = prestep_subphase_budget_seconds
                    current_task_timeout_budget_source = f"prestep_subphase:{prestep_subphase or current_task_budget_stage}"
                    timeout_reason = (
                        "observation current task stage "
                        f"{current_task_budget_stage}"
                        + (
                            f" subphase {prestep_subphase}"
                            if prestep_subphase
                            else ""
                        )
                        + " exceeded max runtime of "
                        f"{prestep_subphase_budget_seconds:.1f} seconds"
                    )
                    break
            if returncode is None:
                terminate_process_tree(process)
                stdout_handle.flush()
                stderr_handle.flush()
                stdout_tail = _read_text_tail(stdout_path)
                stderr_tail = _read_text_tail(stderr_path)
                progress_snapshot = _observation_progress_snapshot(stderr_tail, progress_label=progress_label)
                partial_summary = _read_progress_snapshot(progress_path)
                return {
                    "mode": "child_process",
                    "metrics": None,
                    "timed_out": True,
                    "timeout_reason": timeout_reason,
                    "returncode": -9,
                    "error": "",
                    "stdout": stdout_tail,
                    "stderr": stderr_tail,
                    "last_progress_line": progress_snapshot["last_progress_line"],
                    "last_progress_phase": progress_snapshot["last_progress_phase"],
                    "last_progress_task_id": progress_snapshot["last_progress_task_id"],
                    "last_progress_benchmark_family": progress_snapshot["last_progress_benchmark_family"],
                    "partial_summary": partial_summary,
                    "current_task_decision_budget_exceeded": decision_budget_exceeded,
                    "current_task_decision_budget_seconds": decision_budget_seconds,
                    "current_task_timeout_budget_seconds": current_task_timeout_budget_seconds,
                    "current_task_timeout_budget_source": current_task_timeout_budget_source,
                }
        payload = None
        partial_summary = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        if progress_path.exists():
            try:
                partial_summary = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                partial_summary = {}
        stdout_tail = _read_text_tail(stdout_path)
        stderr_tail = _read_text_tail(stderr_path)
        progress_snapshot = _observation_progress_snapshot(stderr_tail, progress_label=progress_label)
        metrics = _metrics_from_child_payload(payload)
        child_error = ""
        if isinstance(payload, dict) and payload.get("ok") is False:
            child_error = str(payload.get("error", "")).strip()
        if returncode != 0 and not child_error:
            child_error = f"observation child exited with returncode {returncode}"
        if (returncode != 0) or metrics is None or child_error:
            return {
                "mode": "child_process",
                "metrics": metrics,
                "timed_out": False,
                "timeout_reason": "",
                "returncode": int(returncode or 0),
                "error": child_error,
                "stdout": stdout_tail,
                "stderr": stderr_tail,
                "last_progress_line": progress_snapshot["last_progress_line"],
                "last_progress_phase": progress_snapshot["last_progress_phase"],
                "last_progress_task_id": progress_snapshot["last_progress_task_id"],
                "last_progress_benchmark_family": progress_snapshot["last_progress_benchmark_family"],
                "partial_summary": partial_summary,
                "current_task_decision_budget_exceeded": False,
                "current_task_decision_budget_seconds": decision_budget_seconds,
                "current_task_timeout_budget_seconds": current_task_timeout_budget_seconds,
                "current_task_timeout_budget_source": current_task_timeout_budget_source,
            }
        return {
            "mode": "child_process",
            "metrics": metrics,
            "timed_out": False,
            "timeout_reason": "",
            "returncode": int(returncode or 0),
            "error": child_error,
            "stdout": stdout_tail,
            "stderr": stderr_tail,
            "last_progress_line": progress_snapshot["last_progress_line"],
            "last_progress_phase": progress_snapshot["last_progress_phase"],
            "last_progress_task_id": progress_snapshot["last_progress_task_id"],
            "last_progress_benchmark_family": progress_snapshot["last_progress_benchmark_family"],
            "partial_summary": partial_summary,
            "current_task_decision_budget_exceeded": False,
            "current_task_decision_budget_seconds": decision_budget_seconds,
            "current_task_timeout_budget_seconds": current_task_timeout_budget_seconds,
            "current_task_timeout_budget_source": current_task_timeout_budget_source,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem", default=None)
    parser.add_argument("--variant-id", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--observation-profile",
        choices=[_OBSERVATION_PROFILE_DEFAULT, _OBSERVATION_PROFILE_SMOKE],
        default=_OBSERVATION_PROFILE_DEFAULT,
    )
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-observation-seconds", type=float, default=0.0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--priority-benchmark-family-weight", action="append", default=[])
    parser.add_argument("--progress-label", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--protocol-match-id", default="")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--generated-curriculum-budget-seconds", type=float, default=0.0)
    parser.add_argument("--failure-curriculum-budget-seconds", type=float, default=0.0)
    parser.add_argument("--max-current-task-decision-seconds", type=float, default=0.0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--scope-id", default="")
    parser.add_argument("--_observation-child-payload", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    child_payload = Path(str(args._observation_child_payload).strip()) if str(args._observation_child_payload).strip() else None
    if child_payload is not None:
        _observation_child_entry(child_payload)
        return

    observation_profile_applied = _apply_observation_profile_defaults(args)
    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    scope_id = str(args.scope_id).strip()
    if scope_id and not args.generate_only:
        raise SystemExit("--scope-id currently requires --generate-only so scoped supervised runs remain isolated previews")
    if scope_id:
        config = scoped_improvement_cycle_config(config, _sanitize_scope_id(scope_id))
    config.ensure_directories()

    eval_kwargs = _apply_curriculum_flags(autonomous_cycle._observation_eval_kwargs(config, args), args)
    progress_label = str(args.progress_label).strip() or None
    observation_started_at = datetime.now(timezone.utc)
    observation_started_monotonic = time.monotonic()
    max_observation_seconds = max(0.0, float(args.max_observation_seconds or 0.0))
    observation_profile = str(observation_profile_applied.get("profile", _OBSERVATION_PROFILE_DEFAULT))
    observation_profile_defaults_applied = {
        str(key): value
        for key, value in observation_profile_applied.items()
        if str(key) != "profile"
    }
    current_task_decision_budget_seconds, current_task_decision_budget_source = (
        _resolve_current_task_decision_budget_seconds(
            args,
            max_observation_seconds=max_observation_seconds,
            observation_profile=observation_profile,
        )
    )
    if observation_profile != _OBSERVATION_PROFILE_DEFAULT:
        _progress(
            progress_label,
            "phase=observe profile="
            f"{observation_profile} defaults={json.dumps(observation_profile_defaults_applied, sort_keys=True)}",
        )
    if current_task_decision_budget_seconds > 0.0:
        _progress(
            progress_label,
            "phase=observe current_task_decision_budget_seconds="
            f"{current_task_decision_budget_seconds:.1f} source={current_task_decision_budget_source or 'none'}",
        )
    eval_kwargs, observation_priority_source = _apply_bounded_primary_priority_defaults(
        eval_kwargs,
        max_observation_seconds=max_observation_seconds,
    )
    observation_priority_benchmark_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    observation_task_selection_mode = "low_cost" if bool(eval_kwargs.get("prefer_low_cost_tasks", False)) else "default"
    if observation_priority_benchmark_families:
        _progress(
            progress_label,
            "phase=observe priority_families="
            f"{','.join(observation_priority_benchmark_families)} source="
            f"{observation_priority_source or 'none'}",
        )
    generated_curriculum_enabled = bool(eval_kwargs.get("include_generated", False))
    failure_curriculum_enabled = bool(eval_kwargs.get("include_failure_generated", False))
    generated_curriculum_budget_seconds = max(0.0, float(args.generated_curriculum_budget_seconds or 0.0))
    failure_curriculum_budget_seconds = max(0.0, float(args.failure_curriculum_budget_seconds or 0.0))
    observation_staged_curriculum = max_observation_seconds > 0.0 and (
        generated_curriculum_enabled or failure_curriculum_enabled
    )
    observation_eval_kwargs = (
        _without_generated_curriculum(eval_kwargs)
        if observation_staged_curriculum
        else eval_kwargs
    )
    _progress(progress_label, "phase=observe start")
    observation_result = _run_observation_eval(
        config=config,
        eval_kwargs=observation_eval_kwargs,
        progress_label=progress_label,
        max_observation_seconds=max_observation_seconds,
        current_task_decision_budget_seconds=current_task_decision_budget_seconds,
    )
    primary_observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
    observation_elapsed_seconds = primary_observation_elapsed_seconds
    observation_completed_at = datetime.now(timezone.utc)
    metrics = observation_result.get("metrics")
    observation_timed_out = bool(observation_result.get("timed_out"))
    observation_timeout_reason = str(observation_result.get("timeout_reason", "")).strip()
    observation_timeout_budget_seconds = float(
        observation_result.get("current_task_timeout_budget_seconds", 0.0) or 0.0
    )
    observation_timeout_budget_source = str(
        observation_result.get("current_task_timeout_budget_source", "")
    ).strip()
    observation_error = str(observation_result.get("error", "")).strip()
    observation_last_progress_line = str(observation_result.get("last_progress_line", "")).strip()
    observation_last_progress_phase = str(observation_result.get("last_progress_phase", "")).strip()
    observation_last_progress_task_id = str(observation_result.get("last_progress_task_id", "")).strip()
    observation_last_progress_benchmark_family = str(
        observation_result.get("last_progress_benchmark_family", "")
    ).strip()
    observation_partial_summary = (
        dict(observation_result.get("partial_summary", {}))
        if isinstance(observation_result.get("partial_summary", {}), dict)
        else {}
    )
    observation_progress_context = {
        "line": observation_last_progress_line,
        "phase": observation_last_progress_phase,
        "task_id": observation_last_progress_task_id,
        "benchmark_family": observation_last_progress_benchmark_family,
    }
    observation_retried_without_generated_curriculum = False
    observation_retry_warning = ""
    observation_current_task_decision_budget_exceeded = bool(
        observation_result.get("current_task_decision_budget_exceeded", False)
    )
    initial_observation_timeout_reason = observation_timeout_reason
    initial_observation_last_progress_line = observation_last_progress_line
    initial_observation_last_progress_phase = observation_last_progress_phase
    initial_observation_last_progress_task_id = observation_last_progress_task_id
    initial_observation_last_progress_benchmark_family = observation_last_progress_benchmark_family
    initial_observation_progress_context = dict(observation_progress_context)
    initial_observation_partial_summary = dict(observation_partial_summary)
    observation_curriculum_followups: list[dict[str, object]] = []
    remaining_observation_seconds = (
        max(0.0, float(max_observation_seconds) - float(primary_observation_elapsed_seconds))
        if max_observation_seconds > 0.0
        else 0.0
    )
    if (
        metrics is None
        and observation_timed_out
        and (generated_curriculum_enabled or failure_curriculum_enabled)
        and not observation_staged_curriculum
        and remaining_observation_seconds > 0.0
    ):
        observation_retried_without_generated_curriculum = True
        observation_retry_warning = (
            "retrying observation without generated curriculum after timeout "
            f"within remaining budget {remaining_observation_seconds:.1f}s"
        )
        _progress(progress_label, f"phase=observe retry={observation_retry_warning}")
        observation_result = _run_observation_eval(
            config=config,
            eval_kwargs=_without_generated_curriculum(eval_kwargs),
            progress_label=progress_label,
            max_observation_seconds=remaining_observation_seconds,
            current_task_decision_budget_seconds=current_task_decision_budget_seconds,
        )
        primary_observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
        observation_elapsed_seconds = primary_observation_elapsed_seconds
        observation_completed_at = datetime.now(timezone.utc)
        metrics = observation_result.get("metrics")
        observation_timed_out = bool(observation_result.get("timed_out"))
        observation_timeout_reason = str(observation_result.get("timeout_reason", "")).strip()
        observation_error = str(observation_result.get("error", "")).strip()
        observation_last_progress_line = str(observation_result.get("last_progress_line", "")).strip()
        observation_last_progress_phase = str(observation_result.get("last_progress_phase", "")).strip()
        observation_last_progress_task_id = str(observation_result.get("last_progress_task_id", "")).strip()
        observation_last_progress_benchmark_family = str(
            observation_result.get("last_progress_benchmark_family", "")
        ).strip()
        observation_partial_summary = (
            dict(observation_result.get("partial_summary", {}))
            if isinstance(observation_result.get("partial_summary", {}), dict)
            else {}
        )
        observation_progress_context = {
            "line": observation_last_progress_line,
            "phase": observation_last_progress_phase,
            "task_id": observation_last_progress_task_id,
            "benchmark_family": observation_last_progress_benchmark_family,
        }
        observation_current_task_decision_budget_exceeded = bool(
            observation_result.get("current_task_decision_budget_exceeded", False)
        )
        observation_progress_context = _merge_progress_context(
            observation_progress_context,
            initial_observation_progress_context,
        )
        observation_last_progress_line = observation_progress_context["line"]
        observation_last_progress_phase = observation_progress_context["phase"]
        observation_last_progress_task_id = observation_progress_context["task_id"]
        observation_last_progress_benchmark_family = observation_progress_context["benchmark_family"]
        observation_partial_summary = _merge_partial_summary(
            observation_partial_summary,
            initial_observation_partial_summary,
        )
    if metrics is not None and observation_staged_curriculum:
        followup_specs = [
            (
                "generated_success",
                generated_curriculum_enabled,
                generated_curriculum_budget_seconds,
                _with_generated_curriculum(
                    eval_kwargs,
                    include_generated=True,
                    include_failure_generated=False,
                ),
            ),
            (
                "generated_failure",
                failure_curriculum_enabled,
                failure_curriculum_budget_seconds,
                _with_generated_curriculum(
                    eval_kwargs,
                    include_generated=False,
                    include_failure_generated=True,
                ),
            ),
        ]
        for followup_kind, requested, budget_seconds, followup_eval_kwargs in followup_specs:
            followup_summary: dict[str, object] = {
                "kind": followup_kind,
                "requested": bool(requested),
                "budget_seconds": float(budget_seconds),
                "ran": False,
                "merged_generated_metrics": False,
                "timed_out": False,
                "warning": "",
                "skipped_reason": "",
                "generated_total": 0,
                "generated_passed": 0,
                "last_progress_line": "",
                "last_progress_phase": "",
                "last_progress_task_id": "",
                "last_progress_benchmark_family": "",
                "partial_summary": {},
                "partial_phase": "",
                "partial_tasks_completed": 0,
                "partial_current_task_completed_steps": 0,
                "partial_current_task_step_index": 0,
                "partial_current_task_step_stage": "",
                "partial_current_task_step_action": "",
                "partial_generated_tasks_scheduled": 0,
                "partial_generated_tasks_completed": 0,
                "partial_last_completed_task_id": "",
                "partial_last_completed_benchmark_family": "",
                "partial_observed_benchmark_families": [],
            }
            if not requested:
                followup_summary["skipped_reason"] = "not requested"
                _progress(
                    progress_label,
                    f"phase=observe curriculum={followup_kind} skipped reason=not_requested",
                )
                observation_curriculum_followups.append(followup_summary)
                continue
            if budget_seconds <= 0.0:
                followup_summary["skipped_reason"] = "no separate curriculum budget configured"
                _progress(
                    progress_label,
                    f"phase=observe curriculum={followup_kind} skipped reason=no_budget_configured",
                )
                observation_curriculum_followups.append(followup_summary)
                continue
            _progress(
                progress_label,
                f"phase=observe curriculum={followup_kind} start budget_seconds={budget_seconds:.1f}",
            )
            followup_result = _run_observation_eval(
                config=config,
                eval_kwargs=followup_eval_kwargs,
                progress_label=progress_label,
                max_observation_seconds=budget_seconds,
                current_task_decision_budget_seconds=current_task_decision_budget_seconds,
            )
            observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
            observation_completed_at = datetime.now(timezone.utc)
            followup_summary["ran"] = True
            followup_summary["timed_out"] = bool(followup_result.get("timed_out"))
            followup_summary["warning"] = (
                str(followup_result.get("timeout_reason", "")).strip()
                or str(followup_result.get("error", "")).strip()
            )
            followup_summary["last_progress_line"] = str(followup_result.get("last_progress_line", "")).strip()
            followup_summary["last_progress_phase"] = str(followup_result.get("last_progress_phase", "")).strip()
            followup_summary["last_progress_task_id"] = str(followup_result.get("last_progress_task_id", "")).strip()
            followup_summary["last_progress_benchmark_family"] = str(
                followup_result.get("last_progress_benchmark_family", "")
            ).strip()
            followup_partial_summary = (
                dict(followup_result.get("partial_summary", {}))
                if isinstance(followup_result.get("partial_summary", {}), dict)
                else {}
            )
            followup_summary["partial_summary"] = followup_partial_summary
            followup_summary.update(
                _followup_partial_summary_fields(
                    followup_partial_summary,
                    current_task_id=followup_summary["last_progress_task_id"],
                )
            )
            followup_metrics = followup_result.get("metrics")
            if isinstance(followup_metrics, EvalMetrics):
                followup_summary["generated_total"] = int(followup_metrics.generated_total or 0)
                followup_summary["generated_passed"] = int(followup_metrics.generated_passed or 0)
                if followup_metrics.generated_total > 0:
                    metrics = _merge_generated_metrics(metrics, followup_metrics)
                    followup_summary["merged_generated_metrics"] = True
            observation_curriculum_followups.append(followup_summary)
            _progress(
                progress_label,
                f"phase=observe curriculum={followup_kind} complete timed_out={followup_summary['timed_out']} "
                f"merged_generated_metrics={followup_summary['merged_generated_metrics']}",
            )
    observation_budget_exceeded = observation_timed_out or (
        max_observation_seconds > 0.0 and primary_observation_elapsed_seconds > max_observation_seconds
    )
    observation_warning = observation_timeout_reason or observation_error
    if metrics is not None and observation_retried_without_generated_curriculum:
        observation_warning = (
            f"{initial_observation_timeout_reason}; recovered by retrying without generated curriculum"
            if initial_observation_timeout_reason
            else "recovered by retrying without generated curriculum"
        )
    followup_warnings = [
        str(summary.get("warning", "")).strip()
        for summary in observation_curriculum_followups
        if str(summary.get("warning", "")).strip()
    ]
    if metrics is not None and followup_warnings:
        followup_warning_text = "; ".join(followup_warnings)
        observation_warning = (
            f"{observation_warning}; supplemental curriculum follow-up warning: {followup_warning_text}"
            if observation_warning
            else f"supplemental curriculum follow-up warning: {followup_warning_text}"
        )
    if not observation_warning and observation_budget_exceeded:
        observation_warning = (
            f"observation exceeded budget {max_observation_seconds:.1f}s "
            f"with elapsed {primary_observation_elapsed_seconds:.1f}s"
        )
    if observation_warning:
        _progress(progress_label, f"phase=observe warning={observation_warning}")
    _progress(
        progress_label,
        f"phase=observe complete elapsed_seconds={observation_elapsed_seconds:.4f} "
        f"budget_exceeded={observation_budget_exceeded}",
    )

    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        runtime_config=config,
    )
    requested_subsystem = str(args.subsystem or "").strip()
    protocol_match_id = str(args.protocol_match_id).strip()

    if metrics is None:
        cycle_id = _cycle_id_for_experiment(requested_subsystem or "observation")
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem=requested_subsystem or "observation",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason=observation_warning or "human-guided observation failed",
                metrics_summary={
                    "protocol": "human_guided",
                    "protocol_strategy": "careful_guided",
                    "protocol_match_id": protocol_match_id,
                    "guidance_notes": str(args.notes).strip(),
                    "scoped_run": bool(scope_id),
                    "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
                    "workspace_root": str(config.workspace_root),
                    "trajectories_root": str(config.trajectories_root),
                    "observation_started_at": observation_started_at.isoformat(),
                    "observation_completed_at": observation_completed_at.isoformat(),
                    "observation_profile": observation_profile,
                    "observation_profile_defaults_applied": observation_profile_defaults_applied,
                    "observation_current_task_decision_budget_seconds": current_task_decision_budget_seconds,
                    "observation_current_task_decision_budget_source": current_task_decision_budget_source,
                    "observation_current_task_decision_budget_exceeded": observation_current_task_decision_budget_exceeded,
                    "observation_current_task_timeout_budget_seconds": observation_timeout_budget_seconds,
                    "observation_current_task_timeout_budget_source": observation_timeout_budget_source,
                    "observation_elapsed_seconds": observation_elapsed_seconds,
                    "observation_primary_elapsed_seconds": primary_observation_elapsed_seconds,
                    "observation_budget_seconds": max_observation_seconds,
                    "observation_budget_exceeded": observation_budget_exceeded,
                    "observation_warning": observation_warning,
                    "observation_mode": str(observation_result.get("mode", "in_process")),
                    "observation_child_bounded": max_observation_seconds > 0.0,
                    "observation_timed_out": observation_timed_out,
                    "observation_timeout_reason": observation_timeout_reason,
                    "observation_error": observation_error,
                    "observation_returncode": int(observation_result.get("returncode", 0) or 0),
                    "observation_last_progress_line": observation_last_progress_line,
                    "observation_last_progress_phase": observation_last_progress_phase,
                    "observation_last_progress_task_id": observation_last_progress_task_id,
                    "observation_last_progress_benchmark_family": observation_last_progress_benchmark_family,
                    "observation_retried_without_generated_curriculum": observation_retried_without_generated_curriculum,
                    "observation_retry_warning": observation_retry_warning,
                    "observation_staged_curriculum": observation_staged_curriculum,
                    "observation_priority_source": observation_priority_source,
                    "observation_priority_benchmark_families": observation_priority_benchmark_families,
                    "observation_task_selection_mode": observation_task_selection_mode,
                    "observation_generated_curriculum_budget_seconds": generated_curriculum_budget_seconds,
                    "observation_failure_curriculum_budget_seconds": failure_curriculum_budget_seconds,
                    "observation_curriculum_followups": observation_curriculum_followups,
                    "observation_initial_timeout_reason": initial_observation_timeout_reason,
                    "observation_initial_last_progress_line": initial_observation_last_progress_line,
                    "observation_initial_last_progress_phase": initial_observation_last_progress_phase,
                    "observation_initial_last_progress_task_id": initial_observation_last_progress_task_id,
                    "observation_initial_last_progress_benchmark_family": initial_observation_last_progress_benchmark_family,
                    "observation_partial_summary": observation_partial_summary,
                    "observation_initial_partial_summary": initial_observation_partial_summary,
                    **_partial_summary_metric_fields(
                        observation_partial_summary,
                        current_task_id=observation_last_progress_task_id,
                    ),
                },
            ),
        )
        raise SystemExit(observation_warning or "observation failed before metrics were produced")

    ranked_experiments = planner.rank_experiments(metrics)
    if requested_subsystem:
        matching = [experiment for experiment in ranked_experiments if experiment.subsystem == requested_subsystem]
        if matching:
            target = matching[0]
        else:
            raise SystemExit(f"unsupported or unavailable subsystem for guided cycle: {requested_subsystem}")
    else:
        target = _select_guided_experiment(ranked_experiments)
    ranked_variants = planner.rank_variants(target, metrics)
    if args.variant_id:
        matching_variants = [variant for variant in ranked_variants if variant.variant_id == args.variant_id]
        if not matching_variants:
            raise SystemExit(f"variant_id={args.variant_id} is not available for subsystem={args.subsystem}")
        variant = matching_variants[0]
    else:
        variant = _select_guided_variant(ranked_variants)

    cycle_id = _cycle_id_for_experiment(target.subsystem)
    active_artifact_path_obj = active_artifact_path_for_subsystem(config, target.subsystem)
    candidate_artifact_path_obj = staged_candidate_artifact_path(
        active_artifact_path_obj,
        candidates_root=config.candidate_artifacts_root,
        subsystem=target.subsystem,
        cycle_id=f"{cycle_id}:{variant.variant_id}",
    )
    candidate_artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)

    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="observe",
            subsystem=target.subsystem,
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason=target.reason,
            metrics_summary={
                "total": metrics.total,
                "passed": metrics.passed,
                "pass_rate": metrics.pass_rate,
                "generated_pass_rate": metrics.generated_pass_rate,
                "selected_experiment_score": target.score,
                "selected_experiment_expected_gain": target.expected_gain,
                "selected_experiment_estimated_cost": target.estimated_cost,
                "selected_variant_id": variant.variant_id,
                "selected_variant_score": variant.score,
                "protocol": "human_guided",
                "protocol_strategy": "careful_guided",
                "protocol_match_id": protocol_match_id,
                "guidance_notes": str(args.notes).strip(),
                "scoped_run": bool(scope_id),
                "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
                "workspace_root": str(config.workspace_root),
                "trajectories_root": str(config.trajectories_root),
                "observation_started_at": observation_started_at.isoformat(),
                "observation_completed_at": observation_completed_at.isoformat(),
                "observation_profile": observation_profile,
                "observation_profile_defaults_applied": observation_profile_defaults_applied,
                "observation_current_task_decision_budget_seconds": current_task_decision_budget_seconds,
                "observation_current_task_decision_budget_source": current_task_decision_budget_source,
                "observation_current_task_decision_budget_exceeded": observation_current_task_decision_budget_exceeded,
                "observation_current_task_timeout_budget_seconds": observation_timeout_budget_seconds,
                "observation_current_task_timeout_budget_source": observation_timeout_budget_source,
                "observation_elapsed_seconds": observation_elapsed_seconds,
                "observation_primary_elapsed_seconds": primary_observation_elapsed_seconds,
                "observation_budget_seconds": max_observation_seconds,
                "observation_budget_exceeded": observation_budget_exceeded,
                "observation_warning": observation_warning,
                "observation_mode": str(observation_result.get("mode", "in_process")),
                "observation_child_bounded": max_observation_seconds > 0.0,
                "observation_timed_out": observation_timed_out,
                "observation_timeout_reason": observation_timeout_reason,
                "observation_error": observation_error,
                "observation_returncode": int(observation_result.get("returncode", 0) or 0),
                "observation_last_progress_line": observation_last_progress_line,
                "observation_last_progress_phase": observation_last_progress_phase,
                "observation_last_progress_task_id": observation_last_progress_task_id,
                "observation_last_progress_benchmark_family": observation_last_progress_benchmark_family,
                "observation_retried_without_generated_curriculum": observation_retried_without_generated_curriculum,
                "observation_retry_warning": observation_retry_warning,
            "observation_staged_curriculum": observation_staged_curriculum,
            "observation_priority_source": observation_priority_source,
            "observation_priority_benchmark_families": observation_priority_benchmark_families,
            "observation_task_selection_mode": observation_task_selection_mode,
            "observation_generated_curriculum_budget_seconds": generated_curriculum_budget_seconds,
            "observation_failure_curriculum_budget_seconds": failure_curriculum_budget_seconds,
            "observation_curriculum_followups": observation_curriculum_followups,
                "observation_initial_timeout_reason": initial_observation_timeout_reason,
                "observation_initial_last_progress_line": initial_observation_last_progress_line,
                "observation_initial_last_progress_phase": initial_observation_last_progress_phase,
                "observation_initial_last_progress_task_id": initial_observation_last_progress_task_id,
                "observation_initial_last_progress_benchmark_family": initial_observation_last_progress_benchmark_family,
                "observation_partial_summary": observation_partial_summary,
                "observation_initial_partial_summary": initial_observation_partial_summary,
                **_partial_summary_metric_fields(
                    observation_partial_summary,
                    current_task_id=observation_last_progress_task_id,
                ),
            },
        ),
    )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="select",
            subsystem=target.subsystem,
            action="human_guided_target_selection",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="human-guided subsystem and variant selection",
            metrics_summary={
                "selected_subsystem": target.subsystem,
                "selected_variant": {
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "expected_gain": variant.expected_gain,
                    "estimated_cost": variant.estimated_cost,
                    "score": variant.score,
                    "controls": variant.controls,
                },
                "protocol": "human_guided",
                "protocol_strategy": "careful_guided",
                "protocol_match_id": protocol_match_id,
                "guidance_notes": str(args.notes).strip(),
                "scoped_run": bool(scope_id),
                "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
            },
        ),
    )

    prior_retained_record = planner.prior_retained_artifact_record(
        config.improvement_cycles_path,
        target.subsystem,
        before_cycle_id=cycle_id,
    )
    prior_retained_cycle_id = ""
    prior_retained_snapshot_path: Path | None = None
    if prior_retained_record is not None:
        prior_retained_cycle_id = str(prior_retained_record.get("cycle_id", "")).strip()
        snapshot_value = str(prior_retained_record.get("artifact_snapshot_path", "")).strip()
        if snapshot_value:
            prior_retained_snapshot_path = Path(snapshot_value)

    generated = autonomous_cycle._generate_candidate_artifact(
        config=config,
        planner=planner,
        target=target,
        metrics=metrics,
        variant=variant,
        cycle_id=cycle_id,
        active_artifact_path_obj=active_artifact_path_obj,
        candidate_artifact_path_obj=candidate_artifact_path_obj,
        prior_retained_cycle_id=prior_retained_cycle_id,
        prior_retained_snapshot_path=prior_retained_snapshot_path,
    )
    artifact = str(generated["artifact"])
    action = str(generated["action"])
    artifact_kind = str(generated["artifact_kind"])

    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="generate",
            subsystem=target.subsystem,
            action=action,
            artifact_path=str(active_artifact_path_obj),
            artifact_kind=artifact_kind,
            reason="human-guided candidate generation",
            metrics_summary={
                "protocol": "human_guided",
                "protocol_strategy": "careful_guided",
                "protocol_match_id": protocol_match_id,
                "guidance_notes": str(args.notes).strip(),
                "scoped_run": bool(scope_id),
                "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
                "selected_variant": {
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "expected_gain": variant.expected_gain,
                    "estimated_cost": variant.estimated_cost,
                    "score": variant.score,
                    "controls": variant.controls,
                },
            },
            candidate_artifact_path=artifact,
            active_artifact_path=str(active_artifact_path_obj),
        ),
    )

    if artifact and not args.generate_only:
        state, reason = finalize_cycle(
            config=config,
            subsystem=target.subsystem,
            cycle_id=cycle_id,
            artifact_path=Path(artifact),
            active_artifact_path=active_artifact_path_obj,
            include_episode_memory=args.include_episode_memory,
            include_skill_memory=args.include_skill_memory,
            include_skill_transfer=eval_kwargs["include_skill_transfer"],
            include_operator_memory=eval_kwargs["include_operator_memory"],
            include_tool_memory=args.include_tool_memory,
            include_verifier_memory=args.include_verifier_memory,
            include_curriculum=args.include_curriculum,
            include_failure_curriculum=args.include_failure_curriculum,
        )
        print(
            f"protocol=human_guided subsystem={target.subsystem} variant_id={variant.variant_id} "
            f"artifact={artifact} final_state={state} final_reason={reason}"
        )
        return

    print(
        f"protocol=human_guided subsystem={target.subsystem} variant_id={variant.variant_id} artifact={artifact}"
    )


if __name__ == "__main__":
    main()
