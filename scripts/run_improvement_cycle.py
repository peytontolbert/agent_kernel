from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
import subprocess
import tempfile
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone
import time
from uuid import uuid4

from agent_kernel.cycle_runner import autonomous_runtime_eval_flags, finalize_cycle, preview_candidate_retention
from agent_kernel.config import KernelConfig
from agent_kernel.improvement import (
    ImprovementCycleRecord,
    ImprovementPlanner,
    staged_candidate_artifact_path,
    snapshot_artifact_state,
    stamp_artifact_experiment_variant,
    stamp_artifact_generation_context,
)
from agent_kernel.runtime_supervision import atomic_write_json, terminate_process_tree
from agent_kernel.subsystems import active_artifact_path_for_subsystem, base_subsystem_for, generate_candidate_artifact
from evals.harness import run_eval, scoped_improvement_cycle_config
from evals.metrics import EvalMetrics

_CURRENT_TASK_PRESTEP_BUDGET_STAGES = {"context_compile", "decision_pending"}
_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES = (
    "bounded",
    "episode_memory",
    "tool_memory",
)
_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS = 12.0
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
    print(f"[cycle:{progress_label}] {message}", file=sys.stderr, flush=True)


def _variant_generation_kwargs(variant, *, capability_modules_path: Path | None = None) -> dict[str, object]:
    subsystem = base_subsystem_for(variant.subsystem, capability_modules_path)
    controls = dict(variant.controls)
    if subsystem == "benchmark":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "skills":
        return {
            "min_quality": float(controls.get("min_quality", 0.85 if variant.variant_id == "cross_task_transfer" else 0.75)),
            "transfer_only": bool(controls.get("transfer_only", variant.variant_id == "cross_task_transfer")),
        }
    if subsystem == "tooling":
        return {
            "min_quality": float(controls.get("min_quality", 0.8 if variant.variant_id == "script_hardening" else 0.75)),
            "replay_hardening": bool(controls.get("replay_hardening", variant.variant_id == "script_hardening")),
        }
    if subsystem == "operators":
        return {
            "min_support": int(controls.get("min_support", 2)),
            "cross_family_only": bool(controls.get("cross_family_only", variant.variant_id == "cross_family_operator")),
        }
    if subsystem == "verifier":
        return {
            "strategy": str(controls.get("strategy", variant.variant_id)).strip() or variant.variant_id,
        }
    if subsystem == "retrieval":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "tolbert_model":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "policy":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "universe":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "world_model":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "state_estimation":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "trust":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "recovery":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "delegation":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "operator_policy":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "transition_model":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    if subsystem == "curriculum":
        focus = str(controls.get("focus", "")).strip() or None
        return {
            "focus": "benchmark_family" if focus == "family" else focus,
            "family": str(controls.get("family", "")).strip() or None,
        }
    if subsystem == "capabilities":
        return {
            "focus": str(controls.get("focus", "")).strip() or None,
        }
    return {}


def _generate_candidate_artifact(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    target,
    metrics,
    variant,
    cycle_id: str,
    active_artifact_path_obj: Path,
    candidate_artifact_path_obj: Path,
    prior_retained_cycle_id: str,
    prior_retained_snapshot_path: Path | None,
) -> dict[str, object]:
    generation_kwargs = _variant_generation_kwargs(variant, capability_modules_path=config.capability_modules_path)
    artifact = ""
    action = "observe"
    artifact_kind = ""
    prior_active_artifact_path: Path | None = None
    if active_artifact_path_obj.exists():
        prior_active_artifact_path = snapshot_artifact_state(
            active_artifact_path_obj,
            cycle_id=cycle_id,
            stage="pre_generate_active",
        )

    artifact, action, artifact_kind = generate_candidate_artifact(
        config=config,
        planner=planner,
        subsystem=target.subsystem,
        metrics=metrics,
        generation_kwargs=generation_kwargs,
        candidate_artifact_path=candidate_artifact_path_obj,
    )

    if artifact:
        stamp_artifact_experiment_variant(Path(artifact), variant)
        stamp_artifact_generation_context(
            Path(artifact),
            cycle_id=cycle_id,
            active_artifact_path=active_artifact_path_obj,
            candidate_artifact_path=Path(artifact),
            prior_active_artifact_path=prior_active_artifact_path,
            prior_retained_cycle_id=prior_retained_cycle_id or None,
            prior_retained_artifact_snapshot_path=prior_retained_snapshot_path,
        )
    return {
        "artifact": artifact,
        "action": action,
        "artifact_kind": artifact_kind,
        "candidate_artifact_path": candidate_artifact_path_obj,
        "prior_active_artifact_path": prior_active_artifact_path,
    }


def _preview_selection_key(entry: dict[str, object]) -> tuple[int, int, float, float, float]:
    preview = entry.get("preview", {})
    if not isinstance(preview, dict):
        return (0, 0, -1.0, -1.0, 0.0)
    state = str(preview.get("state", "reject"))
    phase_gate_report = preview.get("phase_gate_report", {})
    phase_gate_passed = (
        bool(phase_gate_report.get("passed", False))
        if isinstance(phase_gate_report, dict)
        else False
    )
    baseline = preview.get("baseline")
    candidate = preview.get("candidate")
    pass_rate_delta = 0.0
    step_delta = 0.0
    if baseline is not None and candidate is not None:
        pass_rate_delta = float(candidate.pass_rate - baseline.pass_rate)
        step_delta = float(baseline.average_steps - candidate.average_steps)
    variant = entry.get("variant")
    variant_score = 0.0 if variant is None else float(getattr(variant, "score", 0.0))
    return (
        1 if state == "retain" else 0,
        1 if phase_gate_passed else 0,
        pass_rate_delta,
        step_delta,
        variant_score,
    )


def _comparison_task_limit(config: KernelConfig, args: argparse.Namespace) -> int | None:
    requested = max(0, int(getattr(args, "task_limit", 0) or 0))
    default_cap = max(0, int(getattr(config, "compare_feature_max_tasks", 0) or 0))
    positive_limits = [value for value in (requested, default_cap) if value > 0]
    if not positive_limits:
        return None
    return min(positive_limits)


def _variant_ids_summary(selected_variants) -> str:
    return ",".join(str(getattr(variant, "variant_id", "")).strip() for variant in selected_variants if str(getattr(variant, "variant_id", "")).strip())


def _variant_preview_progress_label(cycle_id: str, subsystem: str, variant_id: str) -> str:
    return f"{cycle_id}_{subsystem}_{variant_id}_preview"


def _normalize_excluded_subsystems(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    normalized: set[str] = set()
    for value in values:
        token = str(value).strip()
        if token:
            normalized.add(token)
    return normalized


def _filter_experiments_by_subsystem(experiments, *, excluded_subsystems: set[str]):
    if not excluded_subsystems:
        return list(experiments)
    filtered = [experiment for experiment in experiments if experiment.subsystem not in excluded_subsystems]
    return filtered if filtered else list(experiments)


def _observation_eval_kwargs(config: KernelConfig, args: argparse.Namespace) -> dict[str, object]:
    task_limit = max(0, int(getattr(args, "task_limit", 0) or 0))
    bounded_preview = task_limit > 0 and max(0.0, float(getattr(args, "max_observation_seconds", 0.0) or 0.0)) > 0.0
    flags = autonomous_runtime_eval_flags(
        config,
        {
        "include_discovered_tasks": True,
        "include_episode_memory": args.include_episode_memory,
        "include_skill_memory": args.include_skill_memory,
        "include_skill_transfer": args.include_skill_transfer,
        "include_operator_memory": args.include_operator_memory,
        "include_tool_memory": args.include_tool_memory,
        "include_verifier_memory": args.include_verifier_memory,
        "include_benchmark_candidates": False,
        "include_verifier_candidates": False,
        "include_generated": bool(args.include_curriculum) if bounded_preview else True,
        "include_failure_generated": bool(args.include_failure_curriculum) if bounded_preview else True,
        },
    )
    if task_limit > 0:
        flags["task_limit"] = task_limit
    priority_benchmark_families = [
        str(value).strip()
        for value in getattr(args, "priority_benchmark_family", [])
        if str(value).strip()
    ]
    if priority_benchmark_families:
        flags["priority_benchmark_families"] = priority_benchmark_families
    elif bounded_preview:
        flags["priority_benchmark_families"] = list(_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES)
        flags["prefer_low_cost_tasks"] = True
    priority_benchmark_family_weights: dict[str, float] = {}
    for value in getattr(args, "priority_benchmark_family_weight", []):
        token = str(value).strip()
        if "=" not in token:
            continue
        family, raw_weight = token.split("=", 1)
        family = family.strip()
        try:
            weight = float(raw_weight)
        except ValueError:
            continue
        if family and weight > 0.0:
            priority_benchmark_family_weights[family] = weight
    if priority_benchmark_family_weights:
        flags["priority_benchmark_family_weights"] = priority_benchmark_family_weights
    return flags


def _priority_family_allocation_summary(metrics, eval_kwargs: dict[str, object]) -> dict[str, object]:
    priority_families = [
        str(value).strip()
        for value in eval_kwargs.get("priority_benchmark_families", [])
        if str(value).strip()
    ]
    if not priority_families:
        return {}
    raw_weights = eval_kwargs.get("priority_benchmark_family_weights", {})
    normalized_weights: dict[str, float] = {}
    if isinstance(raw_weights, dict):
        for family in priority_families:
            try:
                weight = float(raw_weights.get(family, 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0.0:
                normalized_weights[family] = weight
    if not normalized_weights:
        normalized_weights = {family: 1.0 for family in priority_families}
    total_weight = sum(normalized_weights.values())
    planned_weight_shares = {
        family: 0.0 if total_weight <= 0.0 else round(normalized_weights[family] / total_weight, 6)
        for family in priority_families
    }
    task_limit = int(eval_kwargs.get("task_limit", 0) or 0)
    planned_priority_budget = 0
    if task_limit > 0:
        planned_priority_budget = min(task_limit, max(len(priority_families), (task_limit + 1) // 2))
    actual_task_counts = {
        family: int(getattr(metrics, "total_by_benchmark_family", {}).get(family, 0) or 0)
        for family in priority_families
    }
    actual_passed_counts = {
        family: int(getattr(metrics, "passed_by_benchmark_family", {}).get(family, 0) or 0)
        for family in priority_families
    }
    actual_priority_tasks = sum(actual_task_counts.values())
    actual_task_shares = {
        family: 0.0 if actual_priority_tasks <= 0 else round(actual_task_counts[family] / actual_priority_tasks, 6)
        for family in priority_families
    }
    actual_pass_rates = {
        family: 0.0 if actual_task_counts[family] <= 0 else round(actual_passed_counts[family] / actual_task_counts[family], 6)
        for family in priority_families
    }
    return {
        "task_limit": task_limit,
        "planned_priority_budget": planned_priority_budget,
        "priority_benchmark_families": priority_families,
        "priority_benchmark_family_weights": normalized_weights,
        "planned_weight_shares": planned_weight_shares,
        "actual_task_counts": actual_task_counts,
        "actual_task_shares": actual_task_shares,
        "actual_pass_rates": actual_pass_rates,
        "actual_priority_tasks": actual_priority_tasks,
        "unsampled_priority_families": [
            family for family in priority_families if actual_task_counts[family] <= 0
        ],
        "top_planned_family": max(
            priority_families,
            key=lambda family: (planned_weight_shares[family], normalized_weights[family], family),
        ),
        "top_sampled_family": max(
            priority_families,
            key=lambda family: (actual_task_counts[family], actual_task_shares[family], family),
        ) if actual_priority_tasks > 0 else "",
    }


def _cycle_id_for_experiment(subsystem: str) -> str:
    return (
        f"cycle:{subsystem}:"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}:"
        f"{uuid4().hex[:8]}"
    )


def _sanitize_scope_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or f"improvement_scope_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


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
) -> float:
    current_task_step_stage = str(partial_summary.get("current_task_step_stage", "")).strip()
    progress_step_budget_seconds = float(partial_summary.get("current_task_step_budget_seconds", 0.0) or 0.0)
    if progress_step_budget_seconds > 0.0:
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
    return max(0.0, subphase_budget)


def _observation_child_entry(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = _kernel_config_from_snapshot(payload.get("config"))
    config.ensure_directories()
    result_path = Path(str(payload.get("result_path", "")).strip())
    progress_path = Path(str(payload.get("progress_path", "")).strip()) if str(payload.get("progress_path", "")).strip() else None
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
    return EvalMetrics(**metrics_payload)


def _read_text_tail(path: Path, *, max_lines: int = 20) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


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


def _without_generated_curriculum(eval_kwargs: dict[str, object]) -> dict[str, object]:
    degraded = dict(eval_kwargs)
    degraded["include_generated"] = False
    degraded["include_failure_generated"] = False
    return degraded


def _run_observation_eval(
    *,
    config: KernelConfig,
    eval_kwargs: dict[str, object],
    progress_label: str | None,
    max_observation_seconds: float,
) -> dict[str, object]:
    budget_seconds = max(0.0, float(max_observation_seconds))
    active_config = config
    if budget_seconds > 0.0 and float(config.tolbert_context_compile_budget_seconds or 0.0) <= 0.0:
        task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
        stage_budget_seconds = budget_seconds
        if task_limit > 0:
            if task_limit == 1:
                stage_budget_seconds = min(_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS, budget_seconds)
            else:
                stage_budget_seconds = min(
                    _ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS,
                    budget_seconds / max(1, task_limit),
                )
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
            "partial_summary": {},
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    with tempfile.TemporaryDirectory(prefix="agentkernel_autonomous_observe_") as tmp_dir:
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
                cwd=Path(__file__).resolve().parents[1],
                start_new_session=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            returncode = None
            timeout_reason = ""
            current_task_timeout_budget_seconds = 0.0
            current_task_timeout_budget_source = "none"
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
                current_task_step_elapsed_seconds = float(
                    partial_summary.get(
                        "current_task_step_elapsed_seconds",
                        partial_summary.get("current_task_elapsed_seconds", 0.0),
                    )
                    or 0.0
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
                        )
                        prestep_subphase_start_elapsed = current_task_step_elapsed_seconds
                if (
                    current_task_id
                    and current_task_budget_stage
                    and prestep_subphase_budget_seconds > 0.0
                    and current_task_step_elapsed_seconds - prestep_subphase_start_elapsed
                    >= prestep_subphase_budget_seconds
                ):
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
        if returncode != 0 and metrics is None and not child_error:
            child_error = f"observation child exited with returncode {returncode}"
        return {
            "mode": "child_process",
            "metrics": metrics,
            "timed_out": False,
            "timeout_reason": "",
            "returncode": returncode,
            "error": child_error,
            "stdout": stdout_tail,
            "stderr": stderr_tail,
            "last_progress_line": progress_snapshot["last_progress_line"],
            "last_progress_phase": progress_snapshot["last_progress_phase"],
            "last_progress_task_id": progress_snapshot["last_progress_task_id"],
            "last_progress_benchmark_family": progress_snapshot["last_progress_benchmark_family"],
            "partial_summary": partial_summary,
            "current_task_timeout_budget_seconds": current_task_timeout_budget_seconds,
            "current_task_timeout_budget_source": current_task_timeout_budget_source,
        }


def _reconcile_incomplete_cycles(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    progress_label: str | None,
) -> list[dict[str, object]]:
    reconciled: list[dict[str, object]] = []
    for summary in planner.incomplete_cycle_summaries(config.improvement_cycles_path, protocol="autonomous"):
        cycle_id = str(summary.get("cycle_id", "")).strip()
        subsystem = str(summary.get("subsystem", "")).strip()
        if not cycle_id or not subsystem:
            continue
        active_artifact_path = (
            str(summary.get("active_artifact_path", "")).strip()
            or str(summary.get("artifact_path", "")).strip()
        )
        candidate_artifact_path = str(summary.get("candidate_artifact_path", "")).strip()
        artifact_kind = str(summary.get("artifact_kind", "")).strip() or "retention_decision"
        reason = (
            "cycle ended before retention finalization; recorded fail-closed rejection "
            "for stale incomplete autonomous cycle"
        )
        metrics_summary = {
            "protocol": "autonomous",
            "incomplete_cycle": True,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
            "last_state": str(summary.get("last_state", "")).strip(),
            "last_action": str(summary.get("last_action", "")).strip(),
            "selected_variant_id": str(summary.get("selected_variant_id", "")).strip(),
            "record_count": int(summary.get("record_count", 0) or 0),
            "selected_cycles": int(summary.get("selected_cycles", 0) or 0),
        }
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem=subsystem,
                action="finalize_cycle",
                artifact_path=active_artifact_path,
                artifact_kind=artifact_kind,
                reason=reason,
                metrics_summary=metrics_summary,
                candidate_artifact_path=candidate_artifact_path,
                active_artifact_path=active_artifact_path,
            ),
        )
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="record",
                subsystem=subsystem,
                action="persist_retention_outcome",
                artifact_path=active_artifact_path,
                artifact_kind=artifact_kind,
                reason="persisted fail-closed outcome for incomplete autonomous cycle",
                metrics_summary=metrics_summary,
                candidate_artifact_path=candidate_artifact_path,
                active_artifact_path=active_artifact_path,
            ),
        )
        reconciled.append(summary)
        _progress(
            progress_label,
            f"reconciled incomplete cycle cycle_id={cycle_id} subsystem={subsystem} state=reject",
        )
    return reconciled


def _record_finalize_failure(
    *,
    config: KernelConfig,
    planner: ImprovementPlanner,
    cycle_id: str,
    subsystem: str,
    artifact_path: Path,
    active_artifact_path: Path,
    artifact_kind: str,
    exc: Exception,
    progress_label: str | None,
    protocol_match_id: str,
) -> tuple[str, str]:
    reason = (
        f"finalize_cycle_exception:{exc.__class__.__name__}:"
        f"{str(exc).strip() or exc.__class__.__name__}"
    )
    metrics_summary = {
        "protocol": "autonomous",
        "protocol_match_id": str(protocol_match_id).strip(),
        "finalize_exception": True,
        "exception_class": exc.__class__.__name__,
        "exception_message": str(exc).strip(),
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
    }
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="reject",
            subsystem=subsystem,
            action="finalize_cycle",
            artifact_path=str(active_artifact_path),
            artifact_kind=artifact_kind or "retention_decision",
            reason=reason,
            metrics_summary=metrics_summary,
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(active_artifact_path),
        ),
    )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=cycle_id,
            state="record",
            subsystem=subsystem,
            action="persist_retention_outcome",
            artifact_path=str(active_artifact_path),
            artifact_kind=artifact_kind or "retention_record",
            reason="persisted fail-closed outcome after finalize_cycle exception",
            metrics_summary=metrics_summary,
            candidate_artifact_path=str(artifact_path),
            active_artifact_path=str(active_artifact_path),
        ),
    )
    _progress(
        progress_label,
        f"finalize failed closed subsystem={subsystem} exception={exc.__class__.__name__}",
    )
    return "reject", reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--campaign-width", type=int, default=2)
    parser.add_argument("--variant-width", type=int, default=1)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-observation-seconds", type=float, default=0.0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--priority-benchmark-family-weight", action="append", default=[])
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--progress-label", default="")
    parser.add_argument("--protocol-match-id", default="")
    parser.add_argument("--exclude-subsystem", action="append", default=[])
    parser.add_argument("--scope-id", default="")
    parser.add_argument("--_observation-child-payload", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    child_payload = Path(str(args._observation_child_payload).strip()) if str(args._observation_child_payload).strip() else None
    if child_payload is not None:
        _observation_child_entry(child_payload)
        return

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device
    scope_id = str(args.scope_id).strip()
    if scope_id and not args.generate_only:
        raise SystemExit("--scope-id currently requires --generate-only so scoped autonomous runs remain isolated previews")
    if scope_id:
        config = scoped_improvement_cycle_config(config, _sanitize_scope_id(scope_id))
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    _reconcile_incomplete_cycles(
        config=config,
        planner=planner,
        progress_label=str(args.progress_label).strip() or None,
    )
    eval_kwargs = _observation_eval_kwargs(config, args)
    comparison_task_limit = _comparison_task_limit(config, args)

    progress_label = str(args.progress_label).strip() or None
    protocol_match_id = str(args.protocol_match_id).strip()
    observation_started_at = datetime.now(timezone.utc)
    observation_started_monotonic = time.monotonic()
    max_observation_seconds = max(0.0, float(args.max_observation_seconds or 0.0))
    _progress(progress_label, "phase=observe start")
    observation_result = _run_observation_eval(
        config=config,
        eval_kwargs=eval_kwargs,
        progress_label=progress_label,
        max_observation_seconds=max_observation_seconds,
    )
    observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
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
    observation_current_task_timeout_budget_seconds = float(
        observation_result.get("current_task_timeout_budget_seconds", 0.0) or 0.0
    )
    observation_current_task_timeout_budget_source = str(
        observation_result.get("current_task_timeout_budget_source", "none")
    ).strip() or "none"
    observation_retried_without_generated_curriculum = False
    observation_retry_warning = ""
    initial_observation_timeout_reason = observation_timeout_reason
    initial_observation_last_progress_line = observation_last_progress_line
    initial_observation_last_progress_phase = observation_last_progress_phase
    initial_observation_last_progress_task_id = observation_last_progress_task_id
    initial_observation_last_progress_benchmark_family = observation_last_progress_benchmark_family
    initial_observation_partial_summary = observation_partial_summary
    remaining_observation_seconds = (
        max(0.0, float(max_observation_seconds) - float(observation_elapsed_seconds))
        if max_observation_seconds > 0.0
        else 0.0
    )
    if (
        metrics is None
        and observation_timed_out
        and (bool(eval_kwargs.get("include_generated", False)) or bool(eval_kwargs.get("include_failure_generated", False)))
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
        )
        observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
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
        observation_current_task_timeout_budget_seconds = float(
            observation_result.get("current_task_timeout_budget_seconds", 0.0) or 0.0
        )
        observation_current_task_timeout_budget_source = str(
            observation_result.get("current_task_timeout_budget_source", "none")
        ).strip() or "none"
    observation_budget_exceeded = observation_timed_out or (
        max_observation_seconds > 0.0 and observation_elapsed_seconds > max_observation_seconds
    )
    observation_warning = observation_timeout_reason or observation_error
    if metrics is not None and observation_retried_without_generated_curriculum:
        observation_warning = (
            f"{initial_observation_timeout_reason}; recovered by retrying without generated curriculum"
            if initial_observation_timeout_reason
            else "recovered by retrying without generated curriculum"
        )
    if not observation_warning and observation_budget_exceeded:
        observation_warning = (
            f"observation exceeded budget {max_observation_seconds:.1f}s "
            f"with elapsed {observation_elapsed_seconds:.1f}s"
        )
    if observation_warning:
        _progress(progress_label, f"phase=observe warning={observation_warning}")
    _progress(
        progress_label,
        f"phase=observe complete elapsed_seconds={observation_elapsed_seconds:.4f} "
        f"budget_exceeded={observation_budget_exceeded}",
    )
    if metrics is None:
        cycle_id = _cycle_id_for_experiment("observation")
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem="observation",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason=observation_warning or "autonomous observation failed",
                metrics_summary={
                    "protocol": "autonomous",
                    "protocol_strategy": "planner_scored",
                    "protocol_match_id": protocol_match_id,
                    "scoped_run": bool(scope_id),
                    "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
                    "workspace_root": str(config.workspace_root),
                    "trajectories_root": str(config.trajectories_root),
                    "improvement_cycles_path": str(config.improvement_cycles_path),
                    "observation_started_at": observation_started_at.isoformat(),
                    "observation_completed_at": observation_completed_at.isoformat(),
                    "observation_elapsed_seconds": observation_elapsed_seconds,
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
                    "observation_initial_timeout_reason": initial_observation_timeout_reason,
                    "observation_initial_last_progress_line": initial_observation_last_progress_line,
                    "observation_initial_last_progress_phase": initial_observation_last_progress_phase,
                    "observation_initial_last_progress_task_id": initial_observation_last_progress_task_id,
                    "observation_initial_last_progress_benchmark_family": initial_observation_last_progress_benchmark_family,
                    "observation_current_task_timeout_budget_seconds": observation_current_task_timeout_budget_seconds,
                    "observation_current_task_timeout_budget_source": observation_current_task_timeout_budget_source,
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
    priority_family_allocation_summary = _priority_family_allocation_summary(metrics, eval_kwargs)
    _progress(
        progress_label,
        f"observe complete passed={metrics.passed}/{metrics.total} "
        f"pass_rate={metrics.pass_rate:.4f} generated_pass_rate={metrics.generated_pass_rate:.4f}",
    )
    excluded_subsystems = _normalize_excluded_subsystems(args.exclude_subsystem)
    ranked_experiments = _filter_experiments_by_subsystem(
        planner.rank_experiments(metrics),
        excluded_subsystems=excluded_subsystems,
    )
    campaign_budget = planner.recommend_campaign_budget(metrics, max_width=max(1, args.campaign_width))
    campaign = _filter_experiments_by_subsystem(
        planner.select_portfolio_campaign(
        metrics,
        max_candidates=campaign_budget.width if args.adaptive_search else max(1, args.campaign_width),
        ),
        excluded_subsystems=excluded_subsystems,
    )
    outputs: list[str] = []

    for index, target in enumerate(campaign, start=1):
        _progress(progress_label, f"campaign {index}/{len(campaign)} select subsystem={target.subsystem}")
        cycle_id = _cycle_id_for_experiment(target.subsystem)
        active_artifact_path_obj = active_artifact_path_for_subsystem(config, target.subsystem)
        target_portfolio = dict(target.evidence.get("portfolio", {})) if isinstance(target.evidence, dict) else {}
        campaign_breadth_pressure = float(target_portfolio.get("campaign_breadth_pressure", 0.0) or 0.0)
        variant_budget = planner.recommend_variant_budget(target, metrics, max_width=max(1, args.variant_width))
        resolved_variant_width = variant_budget.width if args.adaptive_search else max(1, args.variant_width)
        if resolved_variant_width <= 1:
            selected_variants = [planner.choose_variant(target, metrics)]
        else:
            ranked_variants = planner.rank_variants(target, metrics)
            selected_variants = ranked_variants[: resolved_variant_width]
            if not selected_variants:
                selected_variants = [planner.choose_variant(target, metrics)]
        _progress(
            progress_label,
            f"variant_search start subsystem={target.subsystem} "
            f"selected_variants={len(selected_variants)} variant_ids={_variant_ids_summary(selected_variants)}",
        )
        primary_variant = selected_variants[0]
        primary_variant_recent_activity = planner.recent_variant_activity_summary(
            subsystem=target.subsystem,
            variant_id=primary_variant.variant_id,
            output_path=config.improvement_cycles_path,
        )
        primary_variant_breadth_pressure = float(planner._variant_breadth_pressure(primary_variant_recent_activity))

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
                    "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                    "low_confidence_episodes": metrics.low_confidence_episodes,
                    "selected_experiment_score": target.score,
                    "selected_experiment_expected_gain": target.expected_gain,
                    "selected_experiment_estimated_cost": target.estimated_cost,
                    "selected_variant_id": primary_variant.variant_id,
                    "selected_variant_score": primary_variant.score,
                    "campaign_index": index,
                    "campaign_width": len(campaign),
                    "variant_width": len(selected_variants),
                    "search_strategy": "adaptive_history" if args.adaptive_search else "fixed_width",
                    "campaign_strategy": "portfolio_history",
                    "requested_campaign_width": max(1, args.campaign_width),
                    "requested_variant_width": max(1, args.variant_width),
                    "priority_family_allocation_summary": priority_family_allocation_summary,
                    "campaign_budget": {
                        "width": campaign_budget.width,
                        "max_width": campaign_budget.max_width,
                        "top_score": campaign_budget.top_score,
                        "selected_ids": campaign_budget.selected_ids,
                        "reasons": campaign_budget.reasons,
                    },
                    "variant_budget": {
                        "width": variant_budget.width,
                        "max_width": variant_budget.max_width,
                        "top_score": variant_budget.top_score,
                        "selected_ids": variant_budget.selected_ids,
                        "reasons": variant_budget.reasons,
                    },
                    "campaign_breadth_pressure": campaign_breadth_pressure,
                    "campaign_recent_activity": dict(target_portfolio.get("recent_activity", {}))
                    if isinstance(target_portfolio.get("recent_activity", {}), dict)
                    else {},
                    "selected_variant_breadth_pressure": primary_variant_breadth_pressure,
                    "selected_variant_recent_activity": primary_variant_recent_activity,
                    "selected_campaign_pressures": [
                        {
                            "subsystem": candidate.subsystem,
                            "campaign_breadth_pressure": float(
                                dict(candidate.evidence.get("portfolio", {})).get("campaign_breadth_pressure", 0.0)
                                if isinstance(candidate.evidence, dict)
                                else 0.0
                            ),
                            "portfolio_reasons": list(
                                dict(candidate.evidence.get("portfolio", {})).get("reasons", [])
                                if isinstance(candidate.evidence, dict)
                                else []
                            ),
                        }
                        for candidate in campaign
                    ],
                    "campaign_subsystems": [candidate.subsystem for candidate in campaign],
                    "excluded_subsystems": sorted(excluded_subsystems),
                    "include_skill_transfer": eval_kwargs["include_skill_transfer"],
                    "include_operator_memory": eval_kwargs["include_operator_memory"],
                    "protocol": "autonomous",
                    "protocol_strategy": "planner_scored",
                    "protocol_match_id": protocol_match_id,
                    "scoped_run": bool(scope_id),
                    "scope_id": _sanitize_scope_id(scope_id) if scope_id else "",
                    "workspace_root": str(config.workspace_root),
                    "trajectories_root": str(config.trajectories_root),
                    "improvement_cycles_path": str(config.improvement_cycles_path),
                    "observation_started_at": observation_started_at.isoformat(),
                    "observation_completed_at": observation_completed_at.isoformat(),
                    "observation_elapsed_seconds": observation_elapsed_seconds,
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
                    "observation_initial_timeout_reason": initial_observation_timeout_reason,
                    "observation_initial_last_progress_line": initial_observation_last_progress_line,
                    "observation_initial_last_progress_phase": initial_observation_last_progress_phase,
                    "observation_initial_last_progress_task_id": initial_observation_last_progress_task_id,
                    "observation_initial_last_progress_benchmark_family": initial_observation_last_progress_benchmark_family,
                    "observation_current_task_timeout_budget_seconds": observation_current_task_timeout_budget_seconds,
                    "observation_current_task_timeout_budget_source": observation_current_task_timeout_budget_source,
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
                action="choose_target",
                artifact_path="",
                artifact_kind="improvement_target",
                reason=target.reason,
                metrics_summary={
                    "priority": target.priority,
                    "candidate_subsystems": [candidate.subsystem for candidate in ranked_experiments],
                    "excluded_subsystems": sorted(excluded_subsystems),
                    "campaign_breadth_pressure": campaign_breadth_pressure,
                    "campaign_recent_activity": dict(target_portfolio.get("recent_activity", {}))
                    if isinstance(target_portfolio.get("recent_activity", {}), dict)
                    else {},
                    "candidate_experiments": [
                        {
                            "subsystem": candidate.subsystem,
                            "priority": candidate.priority,
                            "expected_gain": candidate.expected_gain,
                            "estimated_cost": candidate.estimated_cost,
                            "score": candidate.score,
                            "portfolio": candidate.evidence.get("portfolio", {}),
                        }
                        for candidate in ranked_experiments
                    ],
                    "selected_campaign": [
                        {
                            "subsystem": candidate.subsystem,
                            "priority": candidate.priority,
                            "expected_gain": candidate.expected_gain,
                            "estimated_cost": candidate.estimated_cost,
                            "score": candidate.score,
                            "portfolio": candidate.evidence.get("portfolio", {}),
                        }
                        for candidate in campaign
                    ],
                    "selected_experiment_score": target.score,
                    "selected_experiment_expected_gain": target.expected_gain,
                    "selected_experiment_estimated_cost": target.estimated_cost,
                    "selected_variant": {
                        "variant_id": primary_variant.variant_id,
                        "description": primary_variant.description,
                        "expected_gain": primary_variant.expected_gain,
                        "estimated_cost": primary_variant.estimated_cost,
                        "score": primary_variant.score,
                        "controls": primary_variant.controls,
                        "recent_activity": primary_variant_recent_activity,
                        "variant_breadth_pressure": primary_variant_breadth_pressure,
                    },
                    "candidate_variants": [
                        {
                            "variant_id": candidate.variant_id,
                            "description": candidate.description,
                            "expected_gain": candidate.expected_gain,
                            "estimated_cost": candidate.estimated_cost,
                            "score": candidate.score,
                            "controls": candidate.controls,
                            "recent_activity": planner.recent_variant_activity_summary(
                                subsystem=target.subsystem,
                                variant_id=candidate.variant_id,
                                output_path=config.improvement_cycles_path,
                            ),
                            "variant_breadth_pressure": float(
                                planner._variant_breadth_pressure(
                                    planner.recent_variant_activity_summary(
                                        subsystem=target.subsystem,
                                        variant_id=candidate.variant_id,
                                        output_path=config.improvement_cycles_path,
                                    )
                                )
                            ),
                        }
                        for candidate in selected_variants
                    ],
                    "search_strategy": "adaptive_history" if args.adaptive_search else "fixed_width",
                    "campaign_strategy": "portfolio_history",
                    "requested_campaign_width": max(1, args.campaign_width),
                    "requested_variant_width": max(1, args.variant_width),
                    "campaign_budget": {
                        "width": campaign_budget.width,
                        "max_width": campaign_budget.max_width,
                        "top_score": campaign_budget.top_score,
                        "selected_ids": campaign_budget.selected_ids,
                        "reasons": campaign_budget.reasons,
                    },
                    "variant_budget": {
                        "width": variant_budget.width,
                        "max_width": variant_budget.max_width,
                        "top_score": variant_budget.top_score,
                        "selected_ids": variant_budget.selected_ids,
                        "reasons": variant_budget.reasons,
                    },
                    "protocol": "autonomous",
                    "protocol_strategy": "planner_scored",
                    "protocol_match_id": protocol_match_id,
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
        generated_variants: list[dict[str, object]] = []
        for variant_rank, variant in enumerate(selected_variants, start=1):
            _progress(
                progress_label,
                f"variant generate start subsystem={target.subsystem} "
                f"variant={variant.variant_id} rank={variant_rank}/{len(selected_variants)} "
                f"expected_gain={variant.expected_gain:.4f} score={variant.score:.4f}",
            )
            variant_candidate_artifact_path_obj = staged_candidate_artifact_path(
                active_artifact_path_obj,
                candidates_root=config.candidate_artifacts_root,
                subsystem=target.subsystem,
                cycle_id=f"{cycle_id}:{variant.variant_id}",
            )
            variant_candidate_artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)
            generated = _generate_candidate_artifact(
                config=config,
                planner=planner,
                target=target,
                metrics=metrics,
                variant=variant,
                cycle_id=cycle_id,
                active_artifact_path_obj=active_artifact_path_obj,
                candidate_artifact_path_obj=variant_candidate_artifact_path_obj,
                prior_retained_cycle_id=prior_retained_cycle_id,
                prior_retained_snapshot_path=prior_retained_snapshot_path,
            )
            artifact = str(generated["artifact"])
            action = str(generated["action"])
            artifact_kind = str(generated["artifact_kind"])
            _progress(
                progress_label,
                f"variant generate complete subsystem={target.subsystem} "
                f"variant={variant.variant_id} artifact={artifact or '<none>'}",
            )
            planner.append_cycle_record(
                config.improvement_cycles_path,
                ImprovementCycleRecord(
                    cycle_id=cycle_id,
                    state="generate",
                    subsystem=target.subsystem,
                    action=action,
                    artifact_path=str(active_artifact_path_obj),
                    artifact_kind=artifact_kind,
                    reason=target.reason,
                    metrics_summary={
                        "total": metrics.total,
                        "passed": metrics.passed,
                        "pass_rate": metrics.pass_rate,
                        "generated_pass_rate": metrics.generated_pass_rate,
                        "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
                        "low_confidence_episodes": metrics.low_confidence_episodes,
                        "campaign_index": index,
                        "campaign_width": len(campaign),
                        "variant_width": len(selected_variants),
                        "variant_rank": variant_rank,
                        "search_strategy": "adaptive_history" if args.adaptive_search else "fixed_width",
                        "prior_retained_cycle_id": prior_retained_cycle_id,
                        "prior_retained_artifact_snapshot_path": "" if prior_retained_snapshot_path is None else str(prior_retained_snapshot_path),
                        "protocol": "autonomous",
                        "protocol_strategy": "planner_scored",
                        "protocol_match_id": protocol_match_id,
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
            preview = None
            if artifact and len(selected_variants) > 1:
                preview_progress_label = _variant_preview_progress_label(cycle_id, target.subsystem, variant.variant_id)
                _progress(
                    progress_label,
                    f"preview start subsystem={target.subsystem} "
                    f"variant={variant.variant_id} rank={variant_rank}/{len(selected_variants)} "
                    f"task_limit={comparison_task_limit or 0}",
                )
                preview = preview_candidate_retention(
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
                    task_limit=comparison_task_limit,
                    priority_benchmark_families=eval_kwargs.get("priority_benchmark_families"),
                    priority_benchmark_family_weights=eval_kwargs.get("priority_benchmark_family_weights"),
                    progress_label_prefix=preview_progress_label,
                    progress=lambda message: _progress(progress_label, message),
                )
                baseline = preview["baseline"]
                candidate_preview = preview["candidate"]
                phase_gate_report = preview.get("phase_gate_report", {})
                _progress(
                    progress_label,
                    f"preview complete subsystem={target.subsystem} "
                    f"variant={variant.variant_id} state={preview['state']} "
                    f"baseline_pass_rate={baseline.pass_rate:.4f} "
                    f"candidate_pass_rate={candidate_preview.pass_rate:.4f} "
                    f"phase_gate_passed={bool(phase_gate_report.get('passed', False)) if isinstance(phase_gate_report, dict) else False}",
                )
                planner.append_cycle_record(
                    config.improvement_cycles_path,
                    ImprovementCycleRecord(
                        cycle_id=cycle_id,
                        state="evaluate",
                        subsystem=target.subsystem,
                        action="preview_sibling_candidate",
                        artifact_path=str(active_artifact_path_obj),
                        artifact_kind="sibling_variant_preview",
                        reason=str(preview["reason"]),
                        metrics_summary={
                            "variant_id": variant.variant_id,
                            "variant_rank": variant_rank,
                            "variant_width": len(selected_variants),
                            "preview_state": str(preview["state"]),
                            "baseline_pass_rate": baseline.pass_rate,
                            "candidate_pass_rate": candidate_preview.pass_rate,
                            "baseline_average_steps": baseline.average_steps,
                            "candidate_average_steps": candidate_preview.average_steps,
                            "preview_pass_rate_delta": candidate_preview.pass_rate - baseline.pass_rate,
                            "preview_average_step_delta": candidate_preview.average_steps - baseline.average_steps,
                            "preview_task_limit": comparison_task_limit or 0,
                            "preview_phase_gate_passed": bool(
                                phase_gate_report.get("passed", False)
                                if isinstance(phase_gate_report, dict)
                                else False
                            ),
                            "preview_phase_gate_failures": (
                                [
                                    str(failure)
                                    for failure in phase_gate_report.get("failures", [])
                                    if str(failure).strip()
                                ]
                                if isinstance(phase_gate_report, dict)
                                else []
                            ),
                        },
                        candidate_artifact_path=artifact,
                        active_artifact_path=str(active_artifact_path_obj),
                    ),
                )
            generated_variants.append(
                {
                    "variant": variant,
                    "artifact": artifact,
                    "action": action,
                    "artifact_kind": artifact_kind,
                    "preview": preview,
                }
            )

        selected_variant_entry = generated_variants[0]
        if len(generated_variants) > 1:
            selected_variant_entry = max(generated_variants, key=_preview_selection_key)
            _progress(
                progress_label,
                f"variant_search selected subsystem={target.subsystem} "
                f"variant={selected_variant_entry['variant'].variant_id} from={len(generated_variants)}",
            )
            planner.append_cycle_record(
                config.improvement_cycles_path,
                ImprovementCycleRecord(
                    cycle_id=cycle_id,
                    state="select",
                    subsystem=target.subsystem,
                    action="choose_sibling_variant",
                    artifact_path=str(active_artifact_path_obj),
                    artifact_kind="sibling_variant_selection",
                        reason="selected best measured sibling variant before final retention",
                        metrics_summary={
                            "variant_width": len(generated_variants),
                            "comparison_task_limit": comparison_task_limit or 0,
                            "selected_variant_id": selected_variant_entry["variant"].variant_id,
                            "candidate_variants": [
                                {
                                    "variant_id": entry["variant"].variant_id,
                                    "score": entry["variant"].score,
                                    "artifact": entry["artifact"],
                                    "preview_state": "" if not isinstance(entry.get("preview"), dict) else str(entry["preview"].get("state", "")),
                                    "phase_gate_passed": (
                                        bool(entry["preview"].get("phase_gate_report", {}).get("passed", False))
                                        if isinstance(entry.get("preview"), dict)
                                        and isinstance(entry["preview"].get("phase_gate_report", {}), dict)
                                        else False
                                    ),
                                }
                                for entry in generated_variants
                            ],
                    },
                    candidate_artifact_path=str(selected_variant_entry["artifact"]),
                    active_artifact_path=str(active_artifact_path_obj),
                ),
            )

        artifact = str(selected_variant_entry["artifact"])
        action = str(selected_variant_entry["action"])
        variant = selected_variant_entry["variant"]

        if artifact and not args.generate_only:
            _progress(
                progress_label,
                f"finalize start subsystem={target.subsystem} variant={variant.variant_id} artifact={artifact}",
            )
            _progress(progress_label, f"finalize subsystem={target.subsystem} artifact={artifact}")
            try:
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
                    include_curriculum=True,
                    include_failure_curriculum=True,
                    comparison_task_limit=comparison_task_limit,
                    progress=lambda message: _progress(progress_label, message),
                    protocol_match_id=protocol_match_id,
                )
            except Exception as exc:
                state, reason = _record_finalize_failure(
                    config=config,
                    planner=planner,
                    cycle_id=cycle_id,
                    subsystem=target.subsystem,
                    artifact_path=Path(artifact),
                    active_artifact_path=active_artifact_path_obj,
                    artifact_kind=selected_variant_entry["artifact_kind"],
                    exc=exc,
                    progress_label=progress_label,
                    protocol_match_id=protocol_match_id,
                )
            outputs.append(
                f"subsystem={target.subsystem} priority={target.priority} campaign_index={index}/{len(campaign)} action={action} artifact={artifact} reason={target.reason} final_state={state} final_reason={reason}"
            )
            _progress(progress_label, f"finalized subsystem={target.subsystem} state={state}")
            continue

        outputs.append(
            f"subsystem={target.subsystem} priority={target.priority} campaign_index={index}/{len(campaign)} action={action} artifact={artifact} reason={target.reason}"
        )

    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
