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
from agent_kernel.task_bank import TaskBank
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
_LATE_WAVE_BUDGET_FLOOR_SECONDS = 14.0
_GENERATED_SUCCESS_FOLLOWUP_DECISION_BUDGET_SECONDS = 8.0
_LATE_WAVE_GENERATED_SUCCESS_DECISION_BUDGET_SECONDS = 6.0
_MAX_GENERATED_SUCCESS_FOLLOWUP_WAVES = 32
_MULTI_TASK_DECISION_BUDGET_SLACK_TASKS = 1
_GENERATED_SUCCESS_BALANCED_PRIMARY_TASKS_PER_FAMILY = 2
_TOOLING_HIGH_FANOUT_TASK_LIMIT_MIN = 10
_TOOLING_HIGH_FANOUT_DECISION_BUDGET_SECONDS = 8.0
_TOLBERT_HIGHER_TASK_LIMIT_PRESERVE_BUDGET_SECONDS = 10.0
_LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS = 80.0
_LONG_HORIZON_PROJECT_OLLAMA_PRIMARY_TASK_LIMIT = 2
_CODING_BOUNDED_TOLBERT_DISABLE_PRIORITY_FAMILIES = (
    "repository",
    "project",
)
_TOOLING_CODING_FANOUT_PRIORITY_FAMILIES = (
    "repository",
    "project",
    "workflow",
    "repo_chore",
    "integration",
    "repo_sandbox",
    "tooling",
)
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
_OBSERVATION_SUMMARY_RETRIEVAL_LIST_LIMIT = 8


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


def _resolve_requested_experiment(
    ranked_experiments: list,
    requested_subsystem: str,
    *,
    allow_fallback: bool,
):
    normalized_requested_subsystem = str(requested_subsystem).strip()
    if normalized_requested_subsystem:
        matching = [
            experiment
            for experiment in ranked_experiments
            if experiment.subsystem == normalized_requested_subsystem
        ]
        if matching:
            return matching[0], True, ""
        if not allow_fallback:
            raise SystemExit(
                f"unsupported or unavailable subsystem for guided cycle: {normalized_requested_subsystem}"
            )
        fallback = _select_guided_experiment(ranked_experiments)
        return (
            fallback,
            False,
            f"requested subsystem {normalized_requested_subsystem} unavailable; "
            f"falling back to ranked subsystem {fallback.subsystem}",
        )
    return _select_guided_experiment(ranked_experiments), True, ""


def _variant_strategy_family_preferred_variant_id(strategy_family: str) -> str:
    normalized = str(strategy_family).strip()
    if normalized in {
        "rollback_validation",
        "restore_verification",
        "snapshot_integrity",
        "workspace_restore_verification",
    }:
        return "rollback_safety"
    if normalized in {
        "snapshot_coverage",
        "verifier_crosscheck",
        "post_success_replay",
        "mutation_residue_scan",
        "unexpected_change_audit",
    }:
        return "snapshot_coverage"
    return ""


def _sanitize_scope_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or f"supervised_scope_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _trimmed_unique_strings(values: object, *, limit: int = _OBSERVATION_SUMMARY_RETRIEVAL_LIST_LIMIT) -> list[str]:
    unique: list[str] = []
    for value in list(values or []):
        normalized = str(value).strip()
        if normalized.lower() == "none":
            normalized = ""
        if normalized and normalized not in unique:
            unique.append(normalized)
    if limit > 0 and len(unique) > limit:
        return unique[-limit:]
    return unique


def _retrieval_summary_fields(metrics: EvalMetrics) -> dict[str, object]:
    selected_retrieval_span_ids = _trimmed_unique_strings(
        step.get("selected_retrieval_span_id", "")
        for trajectory in dict(metrics.task_trajectories or {}).values()
        if isinstance(trajectory, dict)
        for step in list(trajectory.get("steps", []) or [])
        if isinstance(step, dict)
    )
    retrieval_influenced_task_ids = _trimmed_unique_strings(
        task_id
        for task_id, outcome in dict(metrics.task_outcomes or {}).items()
        if str(task_id).strip()
        and isinstance(outcome, dict)
        and int(outcome.get("retrieval_influenced_steps", 0) or 0) > 0
    )
    return {
        "retrieval_selected_steps": int(metrics.retrieval_selected_steps or 0),
        "retrieval_influenced_steps": int(metrics.retrieval_influenced_steps or 0),
        "trusted_retrieval_steps": int(metrics.trusted_retrieval_steps or 0),
        "selected_retrieval_span_ids": selected_retrieval_span_ids,
        "last_selected_retrieval_span_id": selected_retrieval_span_ids[-1] if selected_retrieval_span_ids else "",
        "retrieval_influenced_task_ids": retrieval_influenced_task_ids,
    }


def _completed_task_summaries(summary: object) -> dict[str, dict[str, object]]:
    if not isinstance(summary, dict):
        return {}
    payload = summary.get("completed_task_summaries", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(task_id).strip(): dict(task_summary)
        for task_id, task_summary in payload.items()
        if str(task_id).strip() and isinstance(task_summary, dict)
    }


def _enrich_metrics_task_outcomes_from_partial_summaries(
    metrics: EvalMetrics,
    *summaries: object,
) -> EvalMetrics:
    if not isinstance(metrics, EvalMetrics):
        return metrics
    payload = asdict(metrics)
    task_outcomes = {
        str(task_id).strip(): dict(outcome)
        for task_id, outcome in dict(metrics.task_outcomes or {}).items()
        if str(task_id).strip() and isinstance(outcome, dict)
    }
    changed = False
    for summary in summaries:
        for task_id, task_summary in _completed_task_summaries(summary).items():
            merged = dict(task_outcomes.get(task_id, {}))
            baseline = dict(merged)
            numeric_fields = (
                "trusted_retrieval_steps",
                "retrieval_influenced_steps",
                "retrieval_selected_steps",
                "proposal_selected_steps",
                "low_confidence_steps",
                "no_state_progress_steps",
                "novel_command_steps",
                "state_regression_steps",
                "steps",
            )
            string_fields = (
                "termination_reason",
                "benchmark_family",
                "memory_source",
                "last_selected_retrieval_span_id",
            )
            for field in numeric_fields:
                merged[field] = max(
                    int(merged.get(field, 0) or 0),
                    int(task_summary.get(field, 0) or 0),
                )
            for field in string_fields:
                value = str(task_summary.get(field, "")).strip()
                if value and not str(merged.get(field, "")).strip():
                    merged[field] = value
            existing_signals = [
                str(value).strip()
                for value in list(merged.get("failure_signals", []) or [])
                if str(value).strip()
            ]
            for value in list(task_summary.get("failure_signals", []) or []):
                normalized = str(value).strip()
                if normalized and normalized not in existing_signals:
                    existing_signals.append(normalized)
            if existing_signals:
                merged["failure_signals"] = existing_signals
            if merged != baseline:
                task_outcomes[task_id] = merged
                changed = True
    if not changed:
        return metrics
    payload["task_outcomes"] = task_outcomes
    return EvalMetrics(**payload)


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
            atomic_write_json(result_path, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, config=config)
        raise
    if result_path:
        atomic_write_json(result_path, {"ok": True, "metrics": _json_ready(asdict(metrics))}, config=config)


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
        enriched.setdefault("prefer_low_cost_tasks", True)
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
    task_limit = max(0, int(getattr(args, "task_limit", 0) or 0))
    return _resolve_current_task_decision_budget_seconds_for_task_limit(
        explicit_budget_seconds=explicit,
        task_limit=task_limit,
        max_observation_seconds=max_observation_seconds,
        observation_profile=observation_profile,
    )


def _resolve_current_task_decision_budget_seconds_for_task_limit(
    *,
    explicit_budget_seconds: float,
    task_limit: int,
    max_observation_seconds: float,
    observation_profile: str,
) -> tuple[float, str]:
    explicit = max(0.0, float(explicit_budget_seconds or 0.0))
    if explicit > 0.0:
        if observation_profile == _OBSERVATION_PROFILE_SMOKE and explicit == _SMOKE_CURRENT_TASK_DECISION_BUDGET_SECONDS:
            return explicit, "smoke_default"
        return explicit, "explicit"
    task_limit = max(0, int(task_limit or 0))
    if task_limit > 0 and max_observation_seconds > 0.0:
        if task_limit == 1:
            return min(_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS, max_observation_seconds), "one_task_default"
        multi_task_limit = max(1, task_limit + _MULTI_TASK_DECISION_BUDGET_SLACK_TASKS)
        multi_task_budget = max_observation_seconds / multi_task_limit
        if multi_task_budget > 0.0:
            return min(_ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS, multi_task_budget), "multi_task_default"
    return 0.0, ""


def _resolve_long_horizon_project_observation_budget_seconds(
    *,
    eval_kwargs: dict[str, object],
    max_observation_seconds: float,
) -> tuple[float, str]:
    budget_seconds = max(0.0, float(max_observation_seconds or 0.0))
    if budget_seconds <= 0.0 or budget_seconds > 60.0:
        return budget_seconds, ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return budget_seconds, ""
    task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if task_limit < 5:
        return budget_seconds, ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if "project" not in priority_families:
        return budget_seconds, ""
    return _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS, "shared_repo_integrator_budget_guard"


def _resolve_staged_curriculum_budget_defaults(
    *,
    remaining_observation_seconds: float,
    generated_curriculum_enabled: bool,
    failure_curriculum_enabled: bool,
    generated_curriculum_budget_seconds: float,
    failure_curriculum_budget_seconds: float,
    observation_priority_source: str,
    primary_passed: int,
    requested_task_limit: int,
    current_task_decision_budget_seconds: float,
) -> tuple[float, float]:
    if str(observation_priority_source).strip() != "explicit":
        return generated_curriculum_budget_seconds, failure_curriculum_budget_seconds
    if (
        generated_curriculum_enabled
        and not failure_curriculum_enabled
        and generated_curriculum_budget_seconds <= 0.0
    ):
        success_seed_target = min(
            max(0, int(primary_passed or 0)),
            max(0, int(requested_task_limit or 0)),
        )
        if success_seed_target > 0:
            per_task_budget_seconds = _GENERATED_SUCCESS_FOLLOWUP_DECISION_BUDGET_SECONDS
            if current_task_decision_budget_seconds > 0.0:
                per_task_budget_seconds = min(
                    per_task_budget_seconds,
                    current_task_decision_budget_seconds,
                )
            generated_curriculum_budget_seconds = round(
                max(0.25, per_task_budget_seconds) * success_seed_target,
                4,
            )
            return generated_curriculum_budget_seconds, failure_curriculum_budget_seconds
    if remaining_observation_seconds <= 0.0:
        return generated_curriculum_budget_seconds, failure_curriculum_budget_seconds
    missing_budget_count = 0
    if generated_curriculum_enabled and generated_curriculum_budget_seconds <= 0.0:
        missing_budget_count += 1
    if failure_curriculum_enabled and failure_curriculum_budget_seconds <= 0.0:
        missing_budget_count += 1
    if missing_budget_count <= 0:
        return generated_curriculum_budget_seconds, failure_curriculum_budget_seconds
    allocated_budget = round(max(0.25, remaining_observation_seconds / missing_budget_count), 4)
    resolved_generated = generated_curriculum_budget_seconds
    resolved_failure = failure_curriculum_budget_seconds
    if generated_curriculum_enabled and resolved_generated <= 0.0:
        resolved_generated = allocated_budget
    if failure_curriculum_enabled and resolved_failure <= 0.0:
        resolved_failure = allocated_budget
    return resolved_generated, resolved_failure


def _resolve_generated_success_balanced_primary_task_limit(
    *,
    subsystem: str,
    eval_kwargs: dict[str, object],
    generated_curriculum_enabled: bool,
    failure_curriculum_enabled: bool,
    generated_curriculum_budget_seconds: float,
    explicit_current_task_decision_budget_seconds: float,
    max_observation_seconds: float,
    observation_profile: str,
) -> tuple[int, str]:
    requested_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if requested_task_limit <= 0:
        return 0, ""
    if not generated_curriculum_enabled or failure_curriculum_enabled:
        return 0, ""
    if generated_curriculum_budget_seconds <= 0.0:
        return 0, ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return 0, ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if len(priority_families) < 2:
        return 0, ""
    balanced_task_limit = max(
        len(priority_families),
        len(priority_families) * _GENERATED_SUCCESS_BALANCED_PRIMARY_TASKS_PER_FAMILY,
    )
    if requested_task_limit <= balanced_task_limit:
        return 0, ""
    replay_decision_budget_seconds = _GENERATED_SUCCESS_FOLLOWUP_DECISION_BUDGET_SECONDS
    if explicit_current_task_decision_budget_seconds > 0.0:
        replay_decision_budget_seconds = min(
            replay_decision_budget_seconds,
            explicit_current_task_decision_budget_seconds,
        )
    replay_capacity = int(
        generated_curriculum_budget_seconds / max(0.25, replay_decision_budget_seconds)
    )
    normalized_subsystem = str(subsystem).strip()
    tooling_priority_families = list(_CODING_BOUNDED_TOLBERT_DISABLE_PRIORITY_FAMILIES)
    if (
        normalized_subsystem == "tooling"
        and priority_families == tooling_priority_families
        and requested_task_limit == balanced_task_limit + 1
        and max_observation_seconds <= 60.0
        and replay_capacity < requested_task_limit
    ):
        return balanced_task_limit, "tooling_repository_project_fanout_guard"
    if replay_capacity < requested_task_limit:
        return 0, ""
    requested_decision_budget_seconds, _ = _resolve_current_task_decision_budget_seconds_for_task_limit(
        explicit_budget_seconds=explicit_current_task_decision_budget_seconds,
        task_limit=requested_task_limit,
        max_observation_seconds=max_observation_seconds,
        observation_profile=observation_profile,
    )
    if (
        requested_task_limit >= _TOOLING_HIGH_FANOUT_TASK_LIMIT_MIN
        and requested_decision_budget_seconds >= _TOOLING_HIGH_FANOUT_DECISION_BUDGET_SECONDS
        and (
            len(priority_families) > len(_CODING_BOUNDED_TOLBERT_DISABLE_PRIORITY_FAMILIES)
            or bool(eval_kwargs.get("include_discovered_tasks", False))
        )
    ):
        return 0, ""
    if (
        requested_task_limit == balanced_task_limit + 1
        and requested_decision_budget_seconds >= _TOLBERT_HIGHER_TASK_LIMIT_PRESERVE_BUDGET_SECONDS
    ):
        return 0, ""
    return balanced_task_limit, "generated_success_family_balance"


def _task_bank_low_cost_family_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        bank = TaskBank()
        tasks = bank.list()
    except Exception:
        return counts
    for task in tasks:
        metadata = dict(getattr(task, "metadata", {}) or {})
        family = str(metadata.get("benchmark_family", "")).strip()
        if not family:
            continue
        if bool(metadata.get("requires_retrieval", False)):
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def _expand_tooling_coding_priority_families(
    *,
    subsystem: str,
    eval_kwargs: dict[str, object],
) -> tuple[dict[str, object], str]:
    enriched = dict(eval_kwargs)
    if str(subsystem).strip() != "tooling":
        return enriched, ""
    requested_task_limit = max(0, int(enriched.get("task_limit", 0) or 0))
    if requested_task_limit < _TOOLING_HIGH_FANOUT_TASK_LIMIT_MIN:
        return enriched, ""
    if not bool(enriched.get("prefer_low_cost_tasks", False)):
        return enriched, ""
    priority_families = [
        str(value).strip()
        for value in list(enriched.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if priority_families != list(_CODING_BOUNDED_TOLBERT_DISABLE_PRIORITY_FAMILIES):
        return enriched, ""
    family_counts = _task_bank_low_cost_family_counts()
    if not family_counts:
        return enriched, ""
    expanded_families: list[str] = []
    expanded_capacity = 0
    for family in _TOOLING_CODING_FANOUT_PRIORITY_FAMILIES:
        expanded_families.append(family)
        expanded_capacity += max(0, int(family_counts.get(family, 0) or 0))
        if expanded_capacity >= requested_task_limit:
            break
    if expanded_families != priority_families:
        enriched["priority_benchmark_families"] = expanded_families
    if expanded_capacity < requested_task_limit and not bool(enriched.get("include_discovered_tasks", False)):
        enriched["include_discovered_tasks"] = True
        return enriched, "tooling_coding_family_scale+discovered"
    if expanded_families != priority_families:
        return enriched, "tooling_coding_family_scale"
    return enriched, ""


def _resolve_tooling_high_fanout_decision_budget_seconds(
    *,
    subsystem: str,
    eval_kwargs: dict[str, object],
    requested_task_limit: int,
    priority_expansion_source: str,
    explicit_budget_seconds: float,
    resolved_decision_budget_seconds: float,
) -> tuple[float, str]:
    if max(0.0, float(explicit_budget_seconds or 0.0)) > 0.0:
        return resolved_decision_budget_seconds, ""
    if str(subsystem).strip() != "tooling":
        return resolved_decision_budget_seconds, ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return resolved_decision_budget_seconds, ""
    active_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if active_task_limit <= 0:
        active_task_limit = max(0, int(requested_task_limit or 0))
    if active_task_limit < _TOOLING_HIGH_FANOUT_TASK_LIMIT_MIN and not str(priority_expansion_source).strip():
        return resolved_decision_budget_seconds, ""
    if resolved_decision_budget_seconds <= 0.0:
        return resolved_decision_budget_seconds, ""
    capped_budget_seconds = min(
        resolved_decision_budget_seconds,
        _GENERATED_SUCCESS_FOLLOWUP_DECISION_BUDGET_SECONDS,
    )
    if capped_budget_seconds >= resolved_decision_budget_seconds:
        return resolved_decision_budget_seconds, ""
    return capped_budget_seconds, "tooling_high_fanout_default"


def _resolve_subsystem_bounded_primary_task_limit(
    *,
    subsystem: str,
    eval_kwargs: dict[str, object],
    max_observation_seconds: float,
    provider: str = "",
) -> tuple[int, str]:
    requested_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if requested_task_limit <= 0:
        return 0, ""
    if max(0.0, float(max_observation_seconds or 0.0)) <= 0.0:
        return 0, ""
    normalized_subsystem = str(subsystem).strip()
    normalized_provider = str(provider).strip().lower()
    if normalized_subsystem == "tooling":
        if (
            normalized_provider == "ollama"
            and requested_task_limit >= 5
            and max_observation_seconds <= _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS
            and bool(eval_kwargs.get("prefer_low_cost_tasks", False))
        ):
            priority_families = [
                str(value).strip()
                for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
                if str(value).strip()
            ]
            if priority_families == ["project"]:
                return (
                    _LONG_HORIZON_PROJECT_OLLAMA_PRIMARY_TASK_LIMIT,
                    "tooling_long_horizon_project_ollama_guard",
                )
        return 0, ""
    if normalized_subsystem != "transition_model":
        return 0, ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return 0, ""
    if not bool(eval_kwargs.get("include_discovered_tasks", False)):
        return 0, ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if priority_families != list(_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES):
        return 0, ""
    if requested_task_limit <= 4:
        return 0, ""
    if max_observation_seconds > 60.0:
        return 0, ""
    return 4, "transition_model_discovered_tail_guard"


def _observation_tolbert_context_disabled_reason(
    *,
    subsystem: str,
    eval_kwargs: dict[str, object],
    max_observation_seconds: float,
) -> str:
    if (
        max(0.0, float(max_observation_seconds or 0.0)) <= 0.0
        or max_observation_seconds > _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS
    ):
        return ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    normalized_subsystem = str(subsystem).strip()
    if normalized_subsystem == "transition_model":
        if not bool(eval_kwargs.get("include_discovered_tasks", False)):
            return ""
        if priority_families == list(_DEFAULT_BOUNDED_OBSERVATION_PRIORITY_FAMILIES):
            return "transition_model_bounded_startup_guard"
        return ""
    if normalized_subsystem != "tooling":
        return ""
    task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if task_limit <= 0:
        return ""
    if (
        task_limit <= 5
        and priority_families == ["project"]
        and bool(eval_kwargs.get("prefer_low_cost_tasks", False))
        and max_observation_seconds <= _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS
    ):
        return "shared_repo_integrator_context_guard"
    if task_limit > 4:
        return ""
    if priority_families != list(_CODING_BOUNDED_TOLBERT_DISABLE_PRIORITY_FAMILIES):
        return ""
    return "tooling_repository_project_bounded_startup_guard"


def _observation_operator_policy_override_reason(
    *,
    eval_kwargs: dict[str, object],
    max_observation_seconds: float,
) -> str:
    if (
        max(0.0, float(max_observation_seconds or 0.0)) <= 0.0
        or max_observation_seconds > _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS
    ):
        return ""
    task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if task_limit <= 0 or task_limit > 5:
        return ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if "repo_sandbox" in priority_families:
        return "repo_sandbox_git_workflow_guard"
    if "project" not in priority_families:
        return ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return ""
    return "shared_repo_long_horizon_project_guard"


def _resolve_reduced_primary_retry_task_limit(
    *,
    eval_kwargs: dict[str, object],
    partial_summary: dict[str, object],
) -> int:
    requested_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if requested_task_limit <= 1:
        return 0
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if not priority_families or not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return 0
    phase = str(partial_summary.get("phase", "")).strip()
    if phase and phase != "primary":
        return 0
    completed_primary_tasks = max(0, int(partial_summary.get("completed_primary_tasks", 0) or 0))
    if completed_primary_tasks <= 0 or completed_primary_tasks >= requested_task_limit:
        return 0
    return max(1, min(requested_task_limit - 1, completed_primary_tasks))


def _resolve_reduced_followup_retry_task_limit(
    *,
    followup_kind: str,
    eval_kwargs: dict[str, object],
    partial_summary: dict[str, object],
) -> int:
    if str(followup_kind).strip() != "generated_success":
        return 0
    requested_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    if requested_task_limit <= 1:
        return 0
    phase = str(partial_summary.get("phase", "")).strip()
    if phase not in {"generated_success_schedule", "generated_success"}:
        return 0
    completed_generated_tasks = max(0, int(partial_summary.get("completed_generated_tasks", 0) or 0))
    if completed_generated_tasks > 0:
        return 0
    generated_tasks_scheduled = max(0, int(partial_summary.get("generated_tasks_scheduled", 0) or 0))
    current_task_id = str(partial_summary.get("current_task_id", "")).strip()
    if generated_tasks_scheduled <= 0 and not current_task_id:
        return 0
    return 1


def _staged_curriculum_followup_workload_units(followup_kind: str) -> int:
    return 2 if str(followup_kind).strip() == "generated_failure" else 1


def _seed_bundle_episode_count(seed_bundle_path: str) -> int:
    path = Path(str(seed_bundle_path or "").strip())
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        return 0
    return len(episodes)


def _is_late_wave_generated_success(eval_kwargs: dict[str, object]) -> bool:
    seed_bundle_path = str(eval_kwargs.get("generated_success_seed_documents_path", "")).strip().lower()
    if not seed_bundle_path:
        return False
    return any(
        marker in seed_bundle_path
        for marker in (
            "generated_success_wave3",
            "generated_success_wave4",
            "generated_success_wave5",
            "generated_success_wave_extra",
        )
        )


def _resolve_long_horizon_generated_success_followup_budget_seconds(
    *,
    followup_kind: str,
    eval_kwargs: dict[str, object],
    budget_seconds: float,
) -> tuple[float, str]:
    resolved_budget_seconds = max(0.0, float(budget_seconds or 0.0))
    if resolved_budget_seconds <= 0.0:
        return resolved_budget_seconds, ""
    if str(followup_kind).strip() != "generated_success":
        return resolved_budget_seconds, ""
    if resolved_budget_seconds >= _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS:
        return resolved_budget_seconds, ""
    if not bool(eval_kwargs.get("prefer_low_cost_tasks", False)):
        return resolved_budget_seconds, ""
    if not str(eval_kwargs.get("generated_success_seed_documents_path", "")).strip():
        return resolved_budget_seconds, ""
    priority_families = [
        str(value).strip()
        for value in list(eval_kwargs.get("priority_benchmark_families", []) or [])
        if str(value).strip()
    ]
    if "project" not in priority_families:
        return resolved_budget_seconds, ""
    return _LONG_HORIZON_PROJECT_OBSERVATION_BUDGET_SECONDS, "shared_repo_integrator_followup_budget_guard"


def _resolve_staged_followup_execution_controls(
    *,
    followup_kind: str,
    eval_kwargs: dict[str, object],
    budget_seconds: float,
    current_task_decision_budget_seconds: float,
) -> tuple[dict[str, object], float]:
    resolved_eval_kwargs = dict(eval_kwargs)
    resolved_budget_seconds = max(0.0, float(budget_seconds or 0.0))
    resolved_decision_budget_seconds = max(0.0, float(current_task_decision_budget_seconds or 0.0))
    if resolved_budget_seconds <= 0.0:
        return resolved_eval_kwargs, resolved_decision_budget_seconds
    workload_units = _staged_curriculum_followup_workload_units(followup_kind)
    baseline_decision_budget_seconds = resolved_decision_budget_seconds
    if baseline_decision_budget_seconds <= 0.0:
        baseline_decision_budget_seconds = min(
            _ONE_TASK_BOUNDED_DECISION_BUDGET_SECONDS,
            resolved_budget_seconds / max(1, workload_units),
        )
    if str(followup_kind).strip() == "generated_success":
        baseline_decision_budget_seconds = min(
            baseline_decision_budget_seconds,
            _GENERATED_SUCCESS_FOLLOWUP_DECISION_BUDGET_SECONDS,
        )
        if _is_late_wave_generated_success(resolved_eval_kwargs):
            baseline_decision_budget_seconds = min(
                baseline_decision_budget_seconds,
                _LATE_WAVE_GENERATED_SUCCESS_DECISION_BUDGET_SECONDS,
            )
    budget_task_limit = max(
        1,
        int(
            resolved_budget_seconds
            / max(0.25, baseline_decision_budget_seconds * max(1, workload_units))
        ),
    )
    requested_task_limit = max(0, int(resolved_eval_kwargs.get("task_limit", 0) or 0))
    resolved_task_limit = budget_task_limit
    if requested_task_limit > 0:
        resolved_task_limit = min(requested_task_limit, budget_task_limit)
    resolved_eval_kwargs["task_limit"] = max(1, resolved_task_limit)
    if str(followup_kind).strip() == "generated_success":
        resolved_eval_kwargs["max_generated_success_schedule_tasks"] = max(1, resolved_eval_kwargs["task_limit"])
    effective_decision_budget_seconds = resolved_budget_seconds / max(
        1,
        resolved_eval_kwargs["task_limit"] * max(1, workload_units),
    )
    if resolved_decision_budget_seconds > 0.0:
        effective_decision_budget_seconds = min(
            resolved_decision_budget_seconds,
            effective_decision_budget_seconds,
        )
    effective_decision_budget_seconds = round(max(0.25, effective_decision_budget_seconds), 4)
    return resolved_eval_kwargs, effective_decision_budget_seconds


def _resolve_staged_followup_runtime_controls(
    *,
    subsystem: str,
    followup_kind: str,
    config: KernelConfig,
    eval_kwargs: dict[str, object],
    budget_seconds: float,
) -> tuple[KernelConfig, float, str, str, str]:
    resolved_budget_seconds, budget_reason = _resolve_long_horizon_generated_success_followup_budget_seconds(
        followup_kind=followup_kind,
        eval_kwargs=eval_kwargs,
        budget_seconds=budget_seconds,
    )
    followup_run_config = config
    tolbert_context_disabled_reason = _observation_tolbert_context_disabled_reason(
        subsystem=subsystem,
        eval_kwargs=eval_kwargs,
        max_observation_seconds=resolved_budget_seconds,
    )
    operator_policy_override_reason = _observation_operator_policy_override_reason(
        eval_kwargs=eval_kwargs,
        max_observation_seconds=resolved_budget_seconds,
    )
    if tolbert_context_disabled_reason:
        followup_run_config = replace(followup_run_config, use_tolbert_context=False)
    if operator_policy_override_reason:
        followup_run_config = replace(
            followup_run_config,
            unattended_allow_git_commands=True,
            unattended_allow_generated_path_mutations=True,
        )
    return (
        followup_run_config,
        resolved_budget_seconds,
        budget_reason,
        tolbert_context_disabled_reason,
        operator_policy_override_reason,
    )


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


def _normalized_positive_int_map(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): int(value or 0)
        for key, value in payload.items()
        if str(key).strip() and int(value or 0) > 0
    }


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
        "retrieval_selected_steps",
        "retrieval_influenced_steps",
        "trusted_retrieval_steps",
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
    merged["selected_retrieval_span_ids"] = _trimmed_unique_strings(
        list(initial.get("selected_retrieval_span_ids", []) or [])
        + list(current.get("selected_retrieval_span_ids", []) or [])
    )
    merged["retrieval_influenced_task_ids"] = _trimmed_unique_strings(
        list(initial.get("retrieval_influenced_task_ids", []) or [])
        + list(current.get("retrieval_influenced_task_ids", []) or [])
    )
    if not str(merged.get("last_selected_retrieval_span_id", "")).strip():
        merged["last_selected_retrieval_span_id"] = str(initial.get("last_selected_retrieval_span_id", "")).strip()
    if not str(merged.get("last_completed_task_id", "")).strip():
        merged["last_completed_task_id"] = str(initial.get("last_completed_task_id", "")).strip()
    if not str(merged.get("last_completed_benchmark_family", "")).strip():
        merged["last_completed_benchmark_family"] = str(initial.get("last_completed_benchmark_family", "")).strip()
    if not str(merged.get("last_completed_generated_task_id", "")).strip():
        merged["last_completed_generated_task_id"] = str(initial.get("last_completed_generated_task_id", "")).strip()
    if not str(merged.get("last_completed_generated_benchmark_family", "")).strip():
        merged["last_completed_generated_benchmark_family"] = str(
            initial.get("last_completed_generated_benchmark_family", "")
        ).strip()
    if not str(merged.get("current_task_id", "")).strip() and str(initial.get("current_task_id", "")).strip():
        for key in (
            "current_task_id",
            "current_task_phase",
            "current_task_index",
            "current_task_total",
            "current_task_benchmark_family",
            "current_task_memory_source",
            "current_task_started_at",
            "current_task_elapsed_seconds",
            "current_task_completed_steps",
            "current_task_step_index",
            "current_task_step_stage",
            "current_task_step_subphase",
            "current_task_step_action",
            "current_task_step_elapsed_seconds",
            "current_task_step_budget_seconds",
            "current_task_verification_passed",
        ):
            if key in initial:
                merged[key] = initial.get(key)
        merged["current_task_progress_timeline"] = list(initial.get("current_task_progress_timeline", []) or [])
    return merged


def _is_retryable_tolbert_startup_failure(error_text: str) -> bool:
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return "tolbert service failed to become ready" in normalized or "tolbert service exited before startup ready" in normalized


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
        "observation_partial_retrieval_selected_steps": int(summary.get("retrieval_selected_steps", 0) or 0),
        "observation_partial_retrieval_influenced_steps": int(summary.get("retrieval_influenced_steps", 0) or 0),
        "observation_partial_trusted_retrieval_steps": int(summary.get("trusted_retrieval_steps", 0) or 0),
        "observation_partial_selected_retrieval_span_ids": _trimmed_unique_strings(
            summary.get("selected_retrieval_span_ids", [])
        ),
        "observation_partial_last_selected_retrieval_span_id": str(
            summary.get("last_selected_retrieval_span_id", "")
        ).strip(),
        "observation_partial_retrieval_influenced_task_ids": _trimmed_unique_strings(
            summary.get("retrieval_influenced_task_ids", [])
        ),
        "observation_partial_last_completed_task_id": str(summary.get("last_completed_task_id", "")).strip(),
        "observation_partial_last_completed_benchmark_family": str(
            summary.get("last_completed_benchmark_family", "")
        ).strip(),
        "observation_partial_last_completed_generated_task_id": str(
            summary.get("last_completed_generated_task_id", "")
        ).strip(),
        "observation_partial_last_completed_generated_benchmark_family": str(
            summary.get("last_completed_generated_benchmark_family", "")
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
        "partial_retrieval_selected_steps": int(observation_fields["observation_partial_retrieval_selected_steps"]),
        "partial_retrieval_influenced_steps": int(
            observation_fields["observation_partial_retrieval_influenced_steps"]
        ),
        "partial_trusted_retrieval_steps": int(observation_fields["observation_partial_trusted_retrieval_steps"]),
        "partial_selected_retrieval_span_ids": list(
            observation_fields["observation_partial_selected_retrieval_span_ids"]
        ),
        "partial_last_selected_retrieval_span_id": str(
            observation_fields["observation_partial_last_selected_retrieval_span_id"]
        ),
        "partial_retrieval_influenced_task_ids": list(
            observation_fields["observation_partial_retrieval_influenced_task_ids"]
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


def _staged_followup_summary_template(
    *,
    kind: str,
    budget_seconds: float,
    eval_kwargs: dict[str, object],
    current_task_decision_budget_seconds: float,
) -> dict[str, object]:
    return {
        "kind": str(kind).strip(),
        "requested": True,
        "budget_seconds": float(budget_seconds),
        "ran": False,
        "retried_without_tolbert_context": False,
        "tolbert_retry_warning": "",
        "retried_with_reduced_task_limit": False,
        "reduced_task_limit_retry_applied": 0,
        "reduced_task_limit_retry_warning": "",
        "salvaged_partial_generated_metrics": False,
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
        "partial_current_task_step_subphase": "",
        "partial_current_task_step_action": "",
        "partial_generated_tasks_scheduled": 0,
        "partial_generated_tasks_completed": 0,
        "partial_last_completed_task_id": "",
        "partial_last_completed_benchmark_family": "",
        "partial_observed_benchmark_families": [],
        "applied_task_limit": max(0, int(eval_kwargs.get("task_limit", 0) or 0)),
        "applied_current_task_decision_budget_seconds": float(current_task_decision_budget_seconds),
        "applied_max_observation_seconds": float(budget_seconds),
        "max_observation_seconds_source": "",
        "tolbert_context_disabled_reason": "",
        "operator_policy_override_reason": "",
    }


def _followup_generated_metrics_from_result(
    *,
    summary: dict[str, object],
    followup_kind: str,
    result_metrics: object,
    partial_summary: dict[str, object],
) -> EvalMetrics | None:
    summary["generated_total"] = 0
    summary["generated_passed"] = 0
    if isinstance(result_metrics, EvalMetrics):
        summary["generated_total"] = int(result_metrics.generated_total or 0)
        summary["generated_passed"] = int(result_metrics.generated_passed or 0)
        if result_metrics.generated_total > 0:
            summary["merged_generated_metrics"] = True
            return result_metrics
        return None
    partial_generated_metrics = _generated_metrics_from_partial_summary(
        followup_kind=followup_kind,
        partial_summary=partial_summary,
    )
    if partial_generated_metrics is None or partial_generated_metrics.generated_total <= 0:
        return None
    summary["generated_total"] = int(partial_generated_metrics.generated_total or 0)
    summary["generated_passed"] = int(partial_generated_metrics.generated_passed or 0)
    summary["merged_generated_metrics"] = True
    summary["salvaged_partial_generated_metrics"] = True
    return partial_generated_metrics


def _apply_followup_result_to_summary(
    *,
    summary: dict[str, object],
    result: dict[str, object],
) -> dict[str, object]:
    summary["timed_out"] = bool(result.get("timed_out"))
    summary["warning"] = (
        str(result.get("timeout_reason", "")).strip()
        or str(result.get("error", "")).strip()
    )
    summary["last_progress_line"] = str(result.get("last_progress_line", "")).strip()
    summary["last_progress_phase"] = str(result.get("last_progress_phase", "")).strip()
    summary["last_progress_task_id"] = str(result.get("last_progress_task_id", "")).strip()
    summary["last_progress_benchmark_family"] = str(result.get("last_progress_benchmark_family", "")).strip()
    partial_summary = (
        dict(result.get("partial_summary", {}))
        if isinstance(result.get("partial_summary", {}), dict)
        else {}
    )
    summary["partial_summary"] = partial_summary
    summary.update(
        _followup_partial_summary_fields(
            partial_summary,
            current_task_id=summary["last_progress_task_id"],
        )
    )
    return partial_summary


def _run_staged_followup_with_runtime_retries(
    *,
    summary_kind: str,
    followup_kind: str,
    subsystem: str,
    config: KernelConfig,
    eval_kwargs: dict[str, object],
    progress_label: str | None,
    budget_seconds: float,
    current_task_decision_budget_seconds: float,
) -> tuple[dict[str, object], EvalMetrics | None]:
    summary = _staged_followup_summary_template(
        kind=summary_kind,
        budget_seconds=budget_seconds,
        eval_kwargs=eval_kwargs,
        current_task_decision_budget_seconds=current_task_decision_budget_seconds,
    )
    resolved_eval_kwargs, resolved_decision_budget_seconds = _resolve_staged_followup_execution_controls(
        followup_kind=followup_kind,
        eval_kwargs=eval_kwargs,
        budget_seconds=budget_seconds,
        current_task_decision_budget_seconds=current_task_decision_budget_seconds,
    )
    summary["applied_task_limit"] = max(0, int(resolved_eval_kwargs.get("task_limit", 0) or 0))
    summary["applied_current_task_decision_budget_seconds"] = float(resolved_decision_budget_seconds)
    (
        followup_run_config,
        followup_max_observation_seconds,
        followup_budget_source,
        followup_tolbert_context_disabled_reason,
        followup_operator_policy_override_reason,
    ) = _resolve_staged_followup_runtime_controls(
        subsystem=subsystem,
        followup_kind=followup_kind,
        config=config,
        eval_kwargs=resolved_eval_kwargs,
        budget_seconds=budget_seconds,
    )
    summary["applied_max_observation_seconds"] = float(followup_max_observation_seconds)
    summary["max_observation_seconds_source"] = str(followup_budget_source).strip()
    summary["tolbert_context_disabled_reason"] = str(followup_tolbert_context_disabled_reason).strip()
    summary["operator_policy_override_reason"] = str(followup_operator_policy_override_reason).strip()
    if followup_budget_source:
        _progress(
            progress_label,
            "phase=observe curriculum="
            f"{summary_kind} max_observation_seconds={followup_max_observation_seconds:.1f} "
            f"source={followup_budget_source}",
        )
    if followup_tolbert_context_disabled_reason:
        _progress(
            progress_label,
            "phase=observe curriculum="
            f"{summary_kind} tolbert_context=disabled "
            f"reason={followup_tolbert_context_disabled_reason}",
        )
    if followup_operator_policy_override_reason:
        _progress(
            progress_label,
            "phase=observe curriculum="
            f"{summary_kind} operator_policy=expanded "
            f"reason={followup_operator_policy_override_reason}",
        )
    _progress(
        progress_label,
        "phase=observe curriculum="
        f"{summary_kind} start budget_seconds={budget_seconds:.1f} "
        f"max_observation_seconds={followup_max_observation_seconds:.1f} "
        f"task_limit={summary['applied_task_limit']} "
        f"current_task_decision_budget_seconds={resolved_decision_budget_seconds:.1f}",
    )
    followup_result = _run_observation_eval(
        config=followup_run_config,
        eval_kwargs=resolved_eval_kwargs,
        progress_label=progress_label,
        max_observation_seconds=followup_max_observation_seconds,
        current_task_decision_budget_seconds=resolved_decision_budget_seconds,
    )
    summary["ran"] = True
    partial_summary = _apply_followup_result_to_summary(summary=summary, result=followup_result)
    generated_metrics = _followup_generated_metrics_from_result(
        summary=summary,
        followup_kind=followup_kind,
        result_metrics=followup_result.get("metrics"),
        partial_summary=partial_summary,
    )
    if (
        generated_metrics is None
        and not summary["timed_out"]
        and bool(followup_run_config.use_tolbert_context)
        and _is_retryable_tolbert_startup_failure(str(summary["warning"]).strip())
    ):
        summary["retried_without_tolbert_context"] = True
        summary["tolbert_retry_warning"] = (
            "retrying curriculum "
            f"{summary_kind} without tolbert context after startup failure "
            f"with fresh followup budget {followup_max_observation_seconds:.1f}s"
        )
        _progress(progress_label, f"phase=observe curriculum={summary_kind} retry={summary['tolbert_retry_warning']}")
        previous_followup_progress_context = {
            "line": str(summary["last_progress_line"]).strip(),
            "phase": str(summary["last_progress_phase"]).strip(),
            "task_id": str(summary["last_progress_task_id"]).strip(),
            "benchmark_family": str(summary["last_progress_benchmark_family"]).strip(),
        }
        previous_followup_partial_summary = dict(partial_summary)
        retry_followup_run_config = replace(followup_run_config, use_tolbert_context=False)
        followup_result = _run_observation_eval(
            config=retry_followup_run_config,
            eval_kwargs=resolved_eval_kwargs,
            progress_label=progress_label,
            max_observation_seconds=followup_max_observation_seconds,
            current_task_decision_budget_seconds=resolved_decision_budget_seconds,
        )
        retry_progress_context = {
            "line": str(followup_result.get("last_progress_line", "")).strip(),
            "phase": str(followup_result.get("last_progress_phase", "")).strip(),
            "task_id": str(followup_result.get("last_progress_task_id", "")).strip(),
            "benchmark_family": str(followup_result.get("last_progress_benchmark_family", "")).strip(),
        }
        retry_progress_context = _merge_progress_context(
            retry_progress_context,
            previous_followup_progress_context,
        )
        summary["last_progress_line"] = retry_progress_context["line"]
        summary["last_progress_phase"] = retry_progress_context["phase"]
        summary["last_progress_task_id"] = retry_progress_context["task_id"]
        summary["last_progress_benchmark_family"] = retry_progress_context["benchmark_family"]
        partial_summary = (
            dict(followup_result.get("partial_summary", {}))
            if isinstance(followup_result.get("partial_summary", {}), dict)
            else {}
        )
        partial_summary = _merge_partial_summary(
            partial_summary,
            previous_followup_partial_summary,
        )
        summary["partial_summary"] = partial_summary
        summary.update(
            _followup_partial_summary_fields(
                partial_summary,
                current_task_id=summary["last_progress_task_id"],
            )
        )
        summary["timed_out"] = bool(followup_result.get("timed_out"))
        summary["warning"] = (
            str(followup_result.get("timeout_reason", "")).strip()
            or str(followup_result.get("error", "")).strip()
        )
        generated_metrics = _followup_generated_metrics_from_result(
            summary=summary,
            followup_kind=followup_kind,
            result_metrics=followup_result.get("metrics"),
            partial_summary=partial_summary,
        )
    reduced_followup_task_limit = _resolve_reduced_followup_retry_task_limit(
        followup_kind=followup_kind,
        eval_kwargs=resolved_eval_kwargs,
        partial_summary=partial_summary,
    )
    if summary["timed_out"] and generated_metrics is None and reduced_followup_task_limit > 0:
        summary["retried_with_reduced_task_limit"] = True
        summary["reduced_task_limit_retry_applied"] = reduced_followup_task_limit
        summary["reduced_task_limit_retry_warning"] = (
            "retrying curriculum "
            f"{summary_kind} with reduced task_limit "
            f"{reduced_followup_task_limit}/{int(resolved_eval_kwargs.get('task_limit', 0) or 0)} "
            f"after timeout with fresh followup budget {followup_max_observation_seconds:.1f}s"
        )
        retry_followup_eval_kwargs = dict(resolved_eval_kwargs)
        retry_followup_eval_kwargs["task_limit"] = reduced_followup_task_limit
        if followup_kind == "generated_success":
            retry_followup_eval_kwargs["max_generated_success_schedule_tasks"] = reduced_followup_task_limit
        retry_followup_eval_kwargs, retry_followup_decision_budget_seconds = _resolve_staged_followup_execution_controls(
            followup_kind=followup_kind,
            eval_kwargs=retry_followup_eval_kwargs,
            budget_seconds=budget_seconds,
            current_task_decision_budget_seconds=current_task_decision_budget_seconds,
        )
        (
            retry_followup_run_config,
            followup_max_observation_seconds,
            followup_budget_source,
            followup_tolbert_context_disabled_reason,
            followup_operator_policy_override_reason,
        ) = _resolve_staged_followup_runtime_controls(
            subsystem=subsystem,
            followup_kind=followup_kind,
            config=config,
            eval_kwargs=retry_followup_eval_kwargs,
            budget_seconds=budget_seconds,
        )
        summary["applied_task_limit"] = max(0, int(retry_followup_eval_kwargs.get("task_limit", 0) or 0))
        summary["applied_current_task_decision_budget_seconds"] = float(retry_followup_decision_budget_seconds)
        summary["applied_max_observation_seconds"] = float(followup_max_observation_seconds)
        summary["max_observation_seconds_source"] = str(followup_budget_source).strip()
        summary["tolbert_context_disabled_reason"] = str(followup_tolbert_context_disabled_reason).strip()
        summary["operator_policy_override_reason"] = str(followup_operator_policy_override_reason).strip()
        _progress(progress_label, f"phase=observe curriculum={summary_kind} retry={summary['reduced_task_limit_retry_warning']}")
        previous_followup_progress_context = {
            "line": str(summary["last_progress_line"]).strip(),
            "phase": str(summary["last_progress_phase"]).strip(),
            "task_id": str(summary["last_progress_task_id"]).strip(),
            "benchmark_family": str(summary["last_progress_benchmark_family"]).strip(),
        }
        previous_followup_partial_summary = dict(partial_summary)
        followup_result = _run_observation_eval(
            config=retry_followup_run_config,
            eval_kwargs=retry_followup_eval_kwargs,
            progress_label=progress_label,
            max_observation_seconds=followup_max_observation_seconds,
            current_task_decision_budget_seconds=retry_followup_decision_budget_seconds,
        )
        retry_progress_context = {
            "line": str(followup_result.get("last_progress_line", "")).strip(),
            "phase": str(followup_result.get("last_progress_phase", "")).strip(),
            "task_id": str(followup_result.get("last_progress_task_id", "")).strip(),
            "benchmark_family": str(followup_result.get("last_progress_benchmark_family", "")).strip(),
        }
        retry_progress_context = _merge_progress_context(
            retry_progress_context,
            previous_followup_progress_context,
        )
        summary["last_progress_line"] = retry_progress_context["line"]
        summary["last_progress_phase"] = retry_progress_context["phase"]
        summary["last_progress_task_id"] = retry_progress_context["task_id"]
        summary["last_progress_benchmark_family"] = retry_progress_context["benchmark_family"]
        partial_summary = (
            dict(followup_result.get("partial_summary", {}))
            if isinstance(followup_result.get("partial_summary", {}), dict)
            else {}
        )
        partial_summary = _merge_partial_summary(
            partial_summary,
            previous_followup_partial_summary,
        )
        summary["partial_summary"] = partial_summary
        summary.update(
            _followup_partial_summary_fields(
                partial_summary,
                current_task_id=summary["last_progress_task_id"],
            )
        )
        summary["timed_out"] = bool(followup_result.get("timed_out"))
        summary["warning"] = (
            str(followup_result.get("timeout_reason", "")).strip()
            or str(followup_result.get("error", "")).strip()
        )
        generated_metrics = _followup_generated_metrics_from_result(
            summary=summary,
            followup_kind=followup_kind,
            result_metrics=followup_result.get("metrics"),
            partial_summary=partial_summary,
        )
    _progress(
        progress_label,
        f"phase=observe curriculum={summary_kind} complete "
        f"timed_out={summary['timed_out']} "
        f"merged_generated_metrics={summary['merged_generated_metrics']}",
    )
    return summary, generated_metrics


def _seed_bundle_has_episodes(path: str) -> bool:
    bundle_path = Path(str(path).strip())
    if not bundle_path.exists():
        return False
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    episodes = payload.get("episodes", [])
    return isinstance(episodes, list) and any(isinstance(episode, dict) for episode in episodes)


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


def _generated_metrics_from_partial_summary(
    *,
    followup_kind: str,
    partial_summary: dict[str, object],
) -> EvalMetrics | None:
    if not partial_summary:
        return None
    completed_generated_tasks = max(0, int(partial_summary.get("completed_generated_tasks", 0) or 0))
    generated_tasks_scheduled = max(0, int(partial_summary.get("generated_tasks_scheduled", 0) or 0))
    generated_by_kind = _normalized_positive_int_map(partial_summary.get("generated_by_kind", {}))
    generated_passed_by_kind = _normalized_positive_int_map(partial_summary.get("generated_passed_by_kind", {}))
    generated_by_benchmark_family = _normalized_positive_int_map(
        partial_summary.get("generated_by_benchmark_family", {})
    )
    generated_passed_by_benchmark_family = _normalized_positive_int_map(
        partial_summary.get("generated_passed_by_benchmark_family", {})
    )
    generated_total = max(
        int(partial_summary.get("generated_total", 0) or 0),
        completed_generated_tasks,
        sum(generated_by_kind.values()),
        sum(generated_by_benchmark_family.values()),
    )
    generated_passed = max(
        0,
        int(partial_summary.get("generated_passed", 0) or 0),
        sum(generated_passed_by_kind.values()),
        sum(generated_passed_by_benchmark_family.values()),
    )
    phase = str(partial_summary.get("phase", "") or "").strip()
    current_task_total = max(0, int(partial_summary.get("current_task_total", 0) or 0))
    current_task_completed_steps = max(0, int(partial_summary.get("current_task_completed_steps", 0) or 0))
    current_task_verification_passed = bool(partial_summary.get("current_task_verification_passed", False))
    current_task_id = str(partial_summary.get("current_task_id", "") or "").strip()
    last_completed_generated_task_id = str(partial_summary.get("last_completed_generated_task_id", "") or "").strip()
    inferred_kind = "adjacent_success" if str(followup_kind).strip() == "generated_success" else "failure_recovery"
    inferred_family = str(partial_summary.get("current_task_benchmark_family", "") or "").strip() or str(
        partial_summary.get("last_completed_generated_benchmark_family", "") or ""
    ).strip() or str(partial_summary.get("last_completed_benchmark_family", "") or "").strip()
    inferred_verified_generated_success = False
    # Some bounded followups can time out after verifier success but before generated-task
    # bookkeeping is flushed into completed_generated_tasks.
    if (
        str(followup_kind).strip() == "generated_success"
        and phase == "generated_success"
        and (current_task_id or current_task_total == 1)
        and current_task_completed_steps > 0
        and current_task_verification_passed
        and (not current_task_id or current_task_id != last_completed_generated_task_id)
    ):
        expected_total_after_success = completed_generated_tasks + 1
        if generated_tasks_scheduled > 0:
            expected_total_after_success = min(expected_total_after_success, generated_tasks_scheduled)
        if expected_total_after_success > generated_total:
            inferred_verified_generated_success = True
            generated_total = expected_total_after_success
            generated_passed = min(generated_total, generated_passed + 1)
            generated_by_kind[inferred_kind] = int(generated_by_kind.get(inferred_kind, 0) or 0) + 1
            generated_passed_by_kind[inferred_kind] = int(generated_passed_by_kind.get(inferred_kind, 0) or 0) + 1
            if inferred_family:
                generated_by_benchmark_family[inferred_family] = (
                    int(generated_by_benchmark_family.get(inferred_family, 0) or 0) + 1
                )
                generated_passed_by_benchmark_family[inferred_family] = (
                    int(generated_passed_by_benchmark_family.get(inferred_family, 0) or 0) + 1
                )
    if not generated_by_kind and generated_total > 0:
        generated_by_kind = {inferred_kind: generated_total}
    if not generated_passed_by_kind and generated_passed > 0:
        generated_passed_by_kind = {inferred_kind: generated_passed}
    if not generated_by_benchmark_family and generated_total > 0 and inferred_family:
        generated_by_benchmark_family = {inferred_family: generated_total}
    if not generated_passed_by_benchmark_family and generated_passed > 0 and inferred_family:
        generated_passed_by_benchmark_family = {inferred_family: generated_passed}
    if generated_total <= 0 and generated_by_kind:
        generated_total = sum(int(value or 0) for value in generated_by_kind.values())
    if generated_passed <= 0 and generated_passed_by_kind:
        generated_passed = sum(int(value or 0) for value in generated_passed_by_kind.values())
    if generated_tasks_scheduled > 0:
        generated_total = min(generated_total, generated_tasks_scheduled)
        generated_passed = min(generated_passed, generated_total)
    elif inferred_verified_generated_success and current_task_total > 0:
        generated_total = min(generated_total, max(1, current_task_total))
        generated_passed = min(generated_passed, generated_total)
    generated_total = max(generated_total, generated_passed)
    if generated_total <= 0 and generated_passed <= 0:
        return None
    return EvalMetrics(
        total=0,
        passed=0,
        average_steps=0.0,
        generated_total=generated_total,
        generated_passed=min(generated_total, generated_passed),
        generated_by_kind=generated_by_kind,
        generated_passed_by_kind=generated_passed_by_kind,
        generated_by_benchmark_family=generated_by_benchmark_family,
        generated_passed_by_benchmark_family=generated_passed_by_benchmark_family,
    )


def _primary_metrics_from_partial_summary(partial_summary: dict[str, object]) -> EvalMetrics | None:
    if not partial_summary:
        return None
    completed_primary_tasks = max(0, int(partial_summary.get("completed_primary_tasks", 0) or 0))
    total_primary_tasks = max(
        completed_primary_tasks,
        int(partial_summary.get("total_primary_tasks", 0) or 0),
        sum(int(value or 0) for value in dict(partial_summary.get("total_by_benchmark_family", {}) or {}).values()),
    )
    remaining_primary_tasks = max(0, int(partial_summary.get("remaining_primary_tasks", 0) or 0))
    if total_primary_tasks <= 0:
        return None
    if completed_primary_tasks < total_primary_tasks or remaining_primary_tasks > 0:
        return None
    primary_passed = max(
        0,
        int(partial_summary.get("primary_passed", 0) or 0),
        sum(int(value or 0) for value in dict(partial_summary.get("passed_by_benchmark_family", {}) or {}).values()),
    )
    primary_passed = min(total_primary_tasks, primary_passed)
    completed_task_summaries = {
        str(task_id).strip(): dict(task_summary)
        for task_id, task_summary in dict(partial_summary.get("completed_task_summaries", {}) or {}).items()
        if str(task_id).strip() and isinstance(task_summary, dict)
    }
    total_steps = sum(int(task_summary.get("steps", 0) or 0) for task_summary in completed_task_summaries.values())
    success_steps = sum(
        int(task_summary.get("steps", 0) or 0)
        for task_summary in completed_task_summaries.values()
        if bool(task_summary.get("success", False))
    )
    success_count = sum(1 for task_summary in completed_task_summaries.values() if bool(task_summary.get("success", False)))
    total_by_benchmark_family = {
        str(key): int(value or 0)
        for key, value in dict(partial_summary.get("total_by_benchmark_family", {}) or {}).items()
        if str(key).strip() and int(value or 0) > 0
    }
    passed_by_benchmark_family = {
        str(key): int(value or 0)
        for key, value in dict(partial_summary.get("passed_by_benchmark_family", {}) or {}).items()
        if str(key).strip() and int(value or 0) > 0
    }
    if not total_by_benchmark_family:
        inferred_family = str(partial_summary.get("last_completed_benchmark_family", "") or "").strip()
        if inferred_family:
            total_by_benchmark_family = {inferred_family: total_primary_tasks}
            if primary_passed > 0:
                passed_by_benchmark_family = {inferred_family: primary_passed}
    task_outcomes = {
        task_id: {
            "task_id": task_id,
            **task_summary,
        }
        for task_id, task_summary in completed_task_summaries.items()
    }
    return EvalMetrics(
        total=total_primary_tasks,
        passed=primary_passed,
        average_steps=(total_steps / total_primary_tasks) if total_primary_tasks > 0 else 0.0,
        average_success_steps=(success_steps / success_count) if success_count > 0 else 0.0,
        total_by_benchmark_family=total_by_benchmark_family,
        passed_by_benchmark_family=passed_by_benchmark_family,
        termination_reasons={
            str(key): int(value or 0)
            for key, value in dict(partial_summary.get("termination_reasons", {}) or {}).items()
            if str(key).strip() and int(value or 0) > 0
        },
        retrieval_selected_steps=int(partial_summary.get("retrieval_selected_steps", 0) or 0),
        retrieval_influenced_steps=int(partial_summary.get("retrieval_influenced_steps", 0) or 0),
        trusted_retrieval_steps=int(partial_summary.get("trusted_retrieval_steps", 0) or 0),
        task_outcomes=task_outcomes,
    )


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
            config=active_config,
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
    parser.add_argument("--variant-strategy-family", default="")
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
    if str(args.subsystem or "").strip() == "recovery":
        requested_variant_strategy_family = str(args.variant_strategy_family or "").strip()
        if requested_variant_strategy_family:
            eval_kwargs["recovery_variant_strategy_family"] = requested_variant_strategy_family
    progress_label = str(args.progress_label).strip() or None
    observation_started_at = datetime.now(timezone.utc)
    observation_started_monotonic = time.monotonic()
    generated_success_seed_bundle_path = ""
    generated_success_wave_seed_bundle_path = ""
    generated_success_wave3_seed_bundle_path = ""
    generated_success_wave4_seed_bundle_path = ""
    generated_success_wave5_seed_bundle_path = ""
    generated_success_extra_wave_seed_bundle_path_a = ""
    generated_success_extra_wave_seed_bundle_path_b = ""
    if args.include_curriculum or args.include_failure_curriculum:
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_seed_bundle_path = handle.name
        eval_kwargs["generated_success_seed_output_path"] = generated_success_seed_bundle_path
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_wave_seed_bundle_path = handle.name
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave3_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_wave3_seed_bundle_path = handle.name
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave4_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_wave4_seed_bundle_path = handle.name
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave5_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_wave5_seed_bundle_path = handle.name
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave_extra_a_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_extra_wave_seed_bundle_path_a = handle.name
        with tempfile.NamedTemporaryFile(
            prefix="generated_success_wave_extra_b_seeds_",
            suffix=".json",
            dir=config.run_reports_dir,
            delete=False,
        ) as handle:
            generated_success_extra_wave_seed_bundle_path_b = handle.name
    max_observation_seconds = max(0.0, float(args.max_observation_seconds or 0.0))
    observation_profile = str(observation_profile_applied.get("profile", _OBSERVATION_PROFILE_DEFAULT))
    observation_profile_defaults_applied = {
        str(key): value
        for key, value in observation_profile_applied.items()
        if str(key) != "profile"
    }
    eval_kwargs, observation_priority_source = _apply_bounded_primary_priority_defaults(
        eval_kwargs,
        max_observation_seconds=max_observation_seconds,
    )
    eval_kwargs, observation_priority_expansion_source = _expand_tooling_coding_priority_families(
        subsystem=str(args.subsystem or "").strip(),
        eval_kwargs=eval_kwargs,
    )
    max_observation_seconds, observation_budget_guard_source = (
        _resolve_long_horizon_project_observation_budget_seconds(
            eval_kwargs=eval_kwargs,
            max_observation_seconds=max_observation_seconds,
        )
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
    if observation_priority_expansion_source:
        _progress(
            progress_label,
            "phase=observe priority_family_expansion_source="
            f"{observation_priority_expansion_source}",
        )
    if observation_budget_guard_source:
        _progress(
            progress_label,
            "phase=observe observation_budget_seconds="
            f"{max_observation_seconds:.1f} source={observation_budget_guard_source}",
        )
    generated_curriculum_enabled = bool(eval_kwargs.get("include_generated", False))
    failure_curriculum_enabled = bool(eval_kwargs.get("include_failure_generated", False))
    generated_curriculum_budget_seconds = max(0.0, float(args.generated_curriculum_budget_seconds or 0.0))
    failure_curriculum_budget_seconds = max(0.0, float(args.failure_curriculum_budget_seconds or 0.0))
    requested_primary_task_limit = max(0, int(eval_kwargs.get("task_limit", 0) or 0))
    primary_task_limit = requested_primary_task_limit
    primary_task_limit_source = ""
    explicit_current_task_decision_budget_seconds = max(
        0.0,
        float(getattr(args, "max_current_task_decision_seconds", 0.0) or 0.0),
    )
    balanced_primary_task_limit, balanced_primary_task_limit_source = (
        _resolve_generated_success_balanced_primary_task_limit(
            subsystem=str(args.subsystem or "").strip(),
            eval_kwargs=eval_kwargs,
            generated_curriculum_enabled=generated_curriculum_enabled,
            failure_curriculum_enabled=failure_curriculum_enabled,
            generated_curriculum_budget_seconds=generated_curriculum_budget_seconds,
            explicit_current_task_decision_budget_seconds=explicit_current_task_decision_budget_seconds,
            max_observation_seconds=max_observation_seconds,
            observation_profile=observation_profile,
        )
    )
    if balanced_primary_task_limit > 0:
        primary_task_limit = min(primary_task_limit, balanced_primary_task_limit)
        primary_task_limit_source = balanced_primary_task_limit_source
        eval_kwargs["task_limit"] = primary_task_limit
    subsystem_primary_task_limit, subsystem_primary_task_limit_source = (
        _resolve_subsystem_bounded_primary_task_limit(
            subsystem=str(args.subsystem or "").strip(),
            eval_kwargs=eval_kwargs,
            max_observation_seconds=max_observation_seconds,
            provider=str(config.provider),
        )
    )
    if subsystem_primary_task_limit > 0:
        primary_task_limit = min(primary_task_limit, subsystem_primary_task_limit)
        primary_task_limit_source = subsystem_primary_task_limit_source
        eval_kwargs["task_limit"] = primary_task_limit
    current_task_decision_budget_seconds, current_task_decision_budget_source = (
        _resolve_current_task_decision_budget_seconds_for_task_limit(
            explicit_budget_seconds=explicit_current_task_decision_budget_seconds,
            task_limit=primary_task_limit,
            max_observation_seconds=max_observation_seconds,
            observation_profile=observation_profile,
        )
    )
    current_task_decision_budget_seconds, tooling_high_fanout_decision_budget_source = (
        _resolve_tooling_high_fanout_decision_budget_seconds(
            subsystem=str(args.subsystem or "").strip(),
            eval_kwargs=eval_kwargs,
            requested_task_limit=requested_primary_task_limit,
            priority_expansion_source=observation_priority_expansion_source,
            explicit_budget_seconds=explicit_current_task_decision_budget_seconds,
            resolved_decision_budget_seconds=current_task_decision_budget_seconds,
        )
    )
    if tooling_high_fanout_decision_budget_source:
        current_task_decision_budget_source = tooling_high_fanout_decision_budget_source
    if observation_profile != _OBSERVATION_PROFILE_DEFAULT:
        _progress(
            progress_label,
            "phase=observe profile="
            f"{observation_profile} defaults={json.dumps(observation_profile_defaults_applied, sort_keys=True)}",
        )
    if primary_task_limit_source:
        _progress(
            progress_label,
            "phase=observe primary_task_limit="
            f"{primary_task_limit}/{requested_primary_task_limit} source={primary_task_limit_source}",
        )
    if current_task_decision_budget_seconds > 0.0:
        _progress(
            progress_label,
            "phase=observe current_task_decision_budget_seconds="
            f"{current_task_decision_budget_seconds:.1f} source={current_task_decision_budget_source or 'none'}",
        )
    observation_staged_curriculum = max_observation_seconds > 0.0 and (
        generated_curriculum_enabled or failure_curriculum_enabled
    )
    observation_eval_kwargs = (
        _without_generated_curriculum(eval_kwargs)
        if observation_staged_curriculum
        else eval_kwargs
    )
    observation_run_config = config
    observation_tolbert_context_disabled_reason = _observation_tolbert_context_disabled_reason(
        subsystem=str(args.subsystem or "").strip(),
        eval_kwargs=observation_eval_kwargs,
        max_observation_seconds=max_observation_seconds,
    )
    observation_operator_policy_override_reason = _observation_operator_policy_override_reason(
        eval_kwargs=observation_eval_kwargs,
        max_observation_seconds=max_observation_seconds,
    )
    if observation_tolbert_context_disabled_reason:
        observation_run_config = replace(config, use_tolbert_context=False)
        _progress(
            progress_label,
            f"phase=observe tolbert_context=disabled reason={observation_tolbert_context_disabled_reason}",
        )
    if observation_operator_policy_override_reason:
        observation_run_config = replace(
            observation_run_config,
            unattended_allow_git_commands=True,
            unattended_allow_generated_path_mutations=True,
        )
        _progress(
            progress_label,
            "phase=observe operator_policy=expanded reason="
            f"{observation_operator_policy_override_reason}",
        )
    _progress(progress_label, "phase=observe start")
    observation_result = _run_observation_eval(
        config=observation_run_config,
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
    observation_retried_without_tolbert_context = False
    observation_tolbert_retry_warning = ""
    observation_retried_with_reduced_primary_task_limit = False
    observation_primary_task_limit_retry_warning = ""
    observation_primary_task_limit_retry_applied = 0
    observation_primary_metrics_salvaged_from_partial_summary = False
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
    generated_curriculum_budget_seconds, failure_curriculum_budget_seconds = (
        _resolve_staged_curriculum_budget_defaults(
            remaining_observation_seconds=remaining_observation_seconds,
            generated_curriculum_enabled=generated_curriculum_enabled,
            failure_curriculum_enabled=failure_curriculum_enabled,
            generated_curriculum_budget_seconds=generated_curriculum_budget_seconds,
            failure_curriculum_budget_seconds=failure_curriculum_budget_seconds,
            observation_priority_source=observation_priority_source,
            primary_passed=int(getattr(metrics, "passed", 0) or 0) if metrics is not None else 0,
            requested_task_limit=int(eval_kwargs.get("task_limit", 0) or 0),
            current_task_decision_budget_seconds=current_task_decision_budget_seconds,
        )
    )
    if (
        metrics is None
        and not observation_timed_out
        and bool(observation_run_config.use_tolbert_context)
        and _is_retryable_tolbert_startup_failure(observation_error)
    ):
        retry_observation_seconds = max(0.0, float(max_observation_seconds or 0.0))
        if retry_observation_seconds > 0.0:
            observation_retried_without_tolbert_context = True
            observation_tolbert_retry_warning = (
                "retrying observation without tolbert context after startup failure "
                f"with fresh observation budget {retry_observation_seconds:.1f}s"
            )
            _progress(progress_label, f"phase=observe retry={observation_tolbert_retry_warning}")
            observation_result = _run_observation_eval(
                config=replace(observation_run_config, use_tolbert_context=False),
                eval_kwargs=observation_eval_kwargs,
                progress_label=progress_label,
                max_observation_seconds=retry_observation_seconds,
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
    if metrics is None and observation_timed_out:
        reduced_primary_task_limit = _resolve_reduced_primary_retry_task_limit(
            eval_kwargs=observation_eval_kwargs,
            partial_summary=observation_partial_summary,
        )
        if reduced_primary_task_limit > 0 and max_observation_seconds > 0.0:
            observation_retried_with_reduced_primary_task_limit = True
            observation_primary_task_limit_retry_applied = reduced_primary_task_limit
            observation_primary_task_limit_retry_warning = (
                "retrying observation with reduced primary task_limit "
                f"{reduced_primary_task_limit}/{int(observation_eval_kwargs.get('task_limit', 0) or 0)} "
                f"after timeout with fresh observation budget {max_observation_seconds:.1f}s"
            )
            retry_eval_kwargs = dict(observation_eval_kwargs)
            retry_eval_kwargs["task_limit"] = reduced_primary_task_limit
            retry_decision_budget_seconds, retry_decision_budget_source = (
                _resolve_current_task_decision_budget_seconds_for_task_limit(
                    explicit_budget_seconds=max(
                        0.0,
                        float(getattr(args, "max_current_task_decision_seconds", 0.0) or 0.0),
                    ),
                    task_limit=reduced_primary_task_limit,
                    max_observation_seconds=max_observation_seconds,
                    observation_profile=observation_profile,
                )
            )
            retry_eval_kwargs, retry_priority_expansion_source = _expand_tooling_coding_priority_families(
                subsystem=str(args.subsystem or "").strip(),
                eval_kwargs=retry_eval_kwargs,
            )
            retry_decision_budget_seconds, retry_tooling_high_fanout_decision_budget_source = (
                _resolve_tooling_high_fanout_decision_budget_seconds(
                    subsystem=str(args.subsystem or "").strip(),
                    eval_kwargs=retry_eval_kwargs,
                    requested_task_limit=requested_primary_task_limit,
                    priority_expansion_source=retry_priority_expansion_source or observation_priority_expansion_source,
                    explicit_budget_seconds=max(
                        0.0,
                        float(getattr(args, "max_current_task_decision_seconds", 0.0) or 0.0),
                    ),
                    resolved_decision_budget_seconds=retry_decision_budget_seconds,
                )
            )
            if retry_tooling_high_fanout_decision_budget_source:
                retry_decision_budget_source = retry_tooling_high_fanout_decision_budget_source
            _progress(progress_label, f"phase=observe retry={observation_primary_task_limit_retry_warning}")
            previous_observation_progress_context = dict(observation_progress_context)
            previous_observation_partial_summary = dict(observation_partial_summary)
            observation_result = _run_observation_eval(
                config=observation_run_config,
                eval_kwargs=retry_eval_kwargs,
                progress_label=progress_label,
                max_observation_seconds=max_observation_seconds,
                current_task_decision_budget_seconds=retry_decision_budget_seconds,
            )
            current_task_decision_budget_seconds = retry_decision_budget_seconds
            current_task_decision_budget_source = retry_decision_budget_source
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
            observation_current_task_decision_budget_exceeded = bool(
                observation_result.get("current_task_decision_budget_exceeded", False)
            )
            observation_progress_context = _merge_progress_context(
                observation_progress_context,
                previous_observation_progress_context,
            )
            observation_last_progress_line = observation_progress_context["line"]
            observation_last_progress_phase = observation_progress_context["phase"]
            observation_last_progress_task_id = observation_progress_context["task_id"]
            observation_last_progress_benchmark_family = observation_progress_context["benchmark_family"]
            observation_partial_summary = _merge_partial_summary(
                observation_partial_summary,
                previous_observation_partial_summary,
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
            config=observation_run_config,
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
    if metrics is None:
        salvaged_primary_metrics = _primary_metrics_from_partial_summary(observation_partial_summary)
        if salvaged_primary_metrics is not None:
            metrics = salvaged_primary_metrics
            observation_primary_metrics_salvaged_from_partial_summary = True
            _progress(
                progress_label,
                "phase=observe recovered_primary_metrics_from_partial_summary=true",
            )
            generated_curriculum_budget_seconds, failure_curriculum_budget_seconds = (
                _resolve_staged_curriculum_budget_defaults(
                    remaining_observation_seconds=remaining_observation_seconds,
                    generated_curriculum_enabled=generated_curriculum_enabled,
                    failure_curriculum_enabled=failure_curriculum_enabled,
                    generated_curriculum_budget_seconds=generated_curriculum_budget_seconds,
                    failure_curriculum_budget_seconds=failure_curriculum_budget_seconds,
                    observation_priority_source=observation_priority_source,
                    primary_passed=int(getattr(metrics, "passed", 0) or 0),
                    requested_task_limit=int(eval_kwargs.get("task_limit", 0) or 0),
                    current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                )
            )
    if metrics is not None and observation_staged_curriculum:
        followup_specs = [
            (
                "generated_success",
                generated_curriculum_enabled,
                generated_curriculum_budget_seconds,
                {
                    **_with_generated_curriculum(
                        eval_kwargs,
                        include_generated=True,
                        include_failure_generated=False,
                    ),
                    "include_primary_tasks": False,
                    "generated_success_seed_documents_path": generated_success_seed_bundle_path
                    or str(config.trajectories_root),
                    "generated_success_seed_workspace_root": str(config.workspace_root),
                    "allow_generated_success_seed_fallback": True,
                    "generated_success_seed_output_path": generated_success_wave_seed_bundle_path,
                },
            ),
            (
                "generated_failure",
                failure_curriculum_enabled,
                failure_curriculum_budget_seconds,
                {
                    **_with_generated_curriculum(
                        eval_kwargs,
                        include_generated=False,
                        include_failure_generated=True,
                    ),
                    "include_primary_tasks": False,
                },
            ),
        ]
        for followup_kind, requested, budget_seconds, followup_eval_kwargs in followup_specs:
            followup_summary: dict[str, object] = {
                "kind": followup_kind,
                "requested": bool(requested),
                "budget_seconds": float(budget_seconds),
                "applied_max_observation_seconds": float(budget_seconds),
                "max_observation_seconds_source": "",
                "ran": False,
                "tolbert_context_disabled_reason": "",
                "operator_policy_override_reason": "",
                "retried_without_tolbert_context": False,
                "tolbert_retry_warning": "",
                "retried_with_reduced_task_limit": False,
                "reduced_task_limit_retry_applied": 0,
                "reduced_task_limit_retry_warning": "",
                "salvaged_partial_generated_metrics": False,
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
                "applied_task_limit": max(0, int(followup_eval_kwargs.get("task_limit", 0) or 0)),
                "applied_current_task_decision_budget_seconds": float(current_task_decision_budget_seconds),
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
            followup_summary, followup_generated_metrics = _run_staged_followup_with_runtime_retries(
                summary_kind=followup_kind,
                followup_kind=followup_kind,
                subsystem=str(args.subsystem or "").strip(),
                config=config,
                eval_kwargs=followup_eval_kwargs,
                progress_label=progress_label,
                budget_seconds=budget_seconds,
                current_task_decision_budget_seconds=current_task_decision_budget_seconds,
            )
            observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
            observation_completed_at = datetime.now(timezone.utc)
            if followup_generated_metrics is not None and followup_generated_metrics.generated_total > 0:
                metrics = _merge_generated_metrics(metrics, followup_generated_metrics)
            observation_curriculum_followups.append(followup_summary)
            _progress(
                progress_label,
                f"phase=observe curriculum={followup_kind} complete timed_out={followup_summary['timed_out']} "
                f"merged_generated_metrics={followup_summary['merged_generated_metrics']}",
            )
            if (
                followup_kind == "generated_success"
                and followup_summary["ran"]
                and not followup_summary["timed_out"]
                and followup_summary["merged_generated_metrics"]
                and int(followup_summary["generated_total"] or 0) > 0
                and str(followup_summary["last_progress_benchmark_family"]).strip() == "repository"
                and generated_success_wave_seed_bundle_path
            ):
                wave2_budget_seconds = min(float(budget_seconds), max(10.0, round(float(budget_seconds) / 2.0, 4)))
                wave2_eval_kwargs = {
                    **_with_generated_curriculum(
                        eval_kwargs,
                        include_generated=True,
                        include_failure_generated=False,
                    ),
                    "include_primary_tasks": False,
                    "generated_success_seed_documents_path": generated_success_wave_seed_bundle_path,
                    "generated_success_seed_workspace_root": str(config.workspace_root),
                    "allow_generated_success_seed_fallback": True,
                    "generated_success_seed_output_path": generated_success_wave3_seed_bundle_path,
                }
                wave2_summary, wave2_generated_metrics = _run_staged_followup_with_runtime_retries(
                    summary_kind="generated_success_wave2",
                    followup_kind="generated_success",
                    subsystem=str(args.subsystem or "").strip(),
                    config=config,
                    eval_kwargs=wave2_eval_kwargs,
                    progress_label=progress_label,
                    budget_seconds=wave2_budget_seconds,
                    current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                )
                observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
                observation_completed_at = datetime.now(timezone.utc)
                if wave2_generated_metrics is not None and wave2_generated_metrics.generated_total > 0:
                    metrics = _merge_generated_metrics(metrics, wave2_generated_metrics)
                observation_curriculum_followups.append(wave2_summary)
                if (
                    wave2_summary["ran"]
                    and wave2_summary["merged_generated_metrics"]
                    and int(wave2_summary["generated_total"] or 0) > 0
                    and str(wave2_summary["last_progress_benchmark_family"]).strip() == "workflow"
                    and generated_success_wave3_seed_bundle_path
                    and _seed_bundle_has_episodes(generated_success_wave3_seed_bundle_path)
                ):
                    wave3_budget_seconds = min(
                        float(wave2_budget_seconds),
                        max(_LATE_WAVE_BUDGET_FLOOR_SECONDS, round(float(wave2_budget_seconds) / 2.0, 4)),
                    )
                    wave3_eval_kwargs = {
                        **_with_generated_curriculum(
                            eval_kwargs,
                            include_generated=True,
                            include_failure_generated=False,
                        ),
                        "include_primary_tasks": False,
                        "generated_success_seed_documents_path": generated_success_wave3_seed_bundle_path,
                        "generated_success_seed_workspace_root": str(config.workspace_root),
                        "allow_generated_success_seed_fallback": True,
                        "generated_success_seed_output_path": generated_success_wave4_seed_bundle_path,
                        "task_limit": min(
                            2,
                            max(1, _seed_bundle_episode_count(generated_success_wave3_seed_bundle_path)),
                        ),
                    }
                    wave3_summary, wave3_generated_metrics = _run_staged_followup_with_runtime_retries(
                        summary_kind="generated_success_wave3",
                        followup_kind="generated_success",
                        subsystem=str(args.subsystem or "").strip(),
                        config=config,
                        eval_kwargs=wave3_eval_kwargs,
                        progress_label=progress_label,
                        budget_seconds=wave3_budget_seconds,
                        current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                    )
                    observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
                    observation_completed_at = datetime.now(timezone.utc)
                    if wave3_generated_metrics is not None and wave3_generated_metrics.generated_total > 0:
                        metrics = _merge_generated_metrics(metrics, wave3_generated_metrics)
                    observation_curriculum_followups.append(wave3_summary)
                    if (
                        wave3_summary["ran"]
                        and wave3_summary["merged_generated_metrics"]
                        and int(wave3_summary["generated_total"] or 0) > 0
                        and str(wave3_summary["last_progress_benchmark_family"]).strip() == "tooling"
                        and generated_success_wave4_seed_bundle_path
                        and _seed_bundle_has_episodes(generated_success_wave4_seed_bundle_path)
                    ):
                        wave4_budget_seconds = min(
                            float(wave3_budget_seconds),
                            max(_LATE_WAVE_BUDGET_FLOOR_SECONDS, round(float(wave3_budget_seconds), 4)),
                        )
                        wave4_eval_kwargs = {
                            **_with_generated_curriculum(
                                eval_kwargs,
                                include_generated=True,
                                include_failure_generated=False,
                            ),
                            "include_primary_tasks": False,
                            "generated_success_seed_documents_path": generated_success_wave4_seed_bundle_path,
                            "generated_success_seed_workspace_root": str(config.workspace_root),
                            "allow_generated_success_seed_fallback": True,
                            "generated_success_seed_output_path": generated_success_wave5_seed_bundle_path,
                            "task_limit": min(
                                2,
                                max(1, _seed_bundle_episode_count(generated_success_wave4_seed_bundle_path)),
                            ),
                        }
                        wave4_summary, wave4_generated_metrics = _run_staged_followup_with_runtime_retries(
                            summary_kind="generated_success_wave4",
                            followup_kind="generated_success",
                            subsystem=str(args.subsystem or "").strip(),
                            config=config,
                            eval_kwargs=wave4_eval_kwargs,
                            progress_label=progress_label,
                            budget_seconds=wave4_budget_seconds,
                            current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                        )
                        observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
                        observation_completed_at = datetime.now(timezone.utc)
                        if wave4_generated_metrics is not None and wave4_generated_metrics.generated_total > 0:
                            metrics = _merge_generated_metrics(metrics, wave4_generated_metrics)
                        observation_curriculum_followups.append(wave4_summary)
                        if (
                            wave4_summary["ran"]
                            and wave4_summary["merged_generated_metrics"]
                            and int(wave4_summary["generated_total"] or 0) > 0
                            and str(wave4_summary["last_progress_benchmark_family"]).strip() == "integration"
                            and generated_success_wave5_seed_bundle_path
                            and _seed_bundle_has_episodes(generated_success_wave5_seed_bundle_path)
                        ):
                            wave5_budget_seconds = min(
                                float(wave4_budget_seconds),
                                max(_LATE_WAVE_BUDGET_FLOOR_SECONDS, round(float(wave4_budget_seconds), 4)),
                            )
                            wave5_eval_kwargs = {
                                **_with_generated_curriculum(
                                    eval_kwargs,
                                    include_generated=True,
                                    include_failure_generated=False,
                                ),
                                "include_primary_tasks": False,
                                "generated_success_seed_documents_path": generated_success_wave5_seed_bundle_path,
                                "generated_success_seed_workspace_root": str(config.workspace_root),
                                "allow_generated_success_seed_fallback": True,
                                "generated_success_seed_output_path": generated_success_extra_wave_seed_bundle_path_a,
                                "task_limit": min(
                                    2,
                                    max(1, _seed_bundle_episode_count(generated_success_wave5_seed_bundle_path)),
                                ),
                            }
                            wave5_summary, wave5_generated_metrics = _run_staged_followup_with_runtime_retries(
                                summary_kind="generated_success_wave5",
                                followup_kind="generated_success",
                                subsystem=str(args.subsystem or "").strip(),
                                config=config,
                                eval_kwargs=wave5_eval_kwargs,
                                progress_label=progress_label,
                                budget_seconds=wave5_budget_seconds,
                                current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                            )
                            observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
                            observation_completed_at = datetime.now(timezone.utc)
                            if wave5_generated_metrics is not None and wave5_generated_metrics.generated_total > 0:
                                metrics = _merge_generated_metrics(metrics, wave5_generated_metrics)
                            observation_curriculum_followups.append(wave5_summary)
                            extra_wave_index = 6
                            extra_input_seed_bundle_path = generated_success_extra_wave_seed_bundle_path_a
                            extra_output_seed_bundle_path = generated_success_extra_wave_seed_bundle_path_b
                            previous_wave_summary = wave5_summary
                            previous_wave_budget_seconds = float(wave5_budget_seconds)
                            while (
                                extra_wave_index <= _MAX_GENERATED_SUCCESS_FOLLOWUP_WAVES
                                and previous_wave_summary["ran"]
                                and previous_wave_summary["merged_generated_metrics"]
                                and int(previous_wave_summary["generated_total"] or 0) > 0
                                and extra_input_seed_bundle_path
                                and _seed_bundle_has_episodes(extra_input_seed_bundle_path)
                            ):
                                extra_wave_budget_seconds = min(
                                    float(previous_wave_budget_seconds),
                                    max(_LATE_WAVE_BUDGET_FLOOR_SECONDS, round(float(previous_wave_budget_seconds), 4)),
                                )
                                next_output_path = ""
                                if extra_wave_index < _MAX_GENERATED_SUCCESS_FOLLOWUP_WAVES:
                                    next_output_path = extra_output_seed_bundle_path
                                extra_wave_eval_kwargs = {
                                    **_with_generated_curriculum(
                                        eval_kwargs,
                                        include_generated=True,
                                        include_failure_generated=False,
                                    ),
                                    "include_primary_tasks": False,
                                    "generated_success_seed_documents_path": extra_input_seed_bundle_path,
                                    "generated_success_seed_workspace_root": str(config.workspace_root),
                                    "allow_generated_success_seed_fallback": True,
                                    "generated_success_seed_output_path": next_output_path,
                                    "task_limit": min(
                                        2,
                                        max(1, _seed_bundle_episode_count(extra_input_seed_bundle_path)),
                                    ),
                                }
                                extra_wave_summary, extra_wave_generated_metrics = _run_staged_followup_with_runtime_retries(
                                    summary_kind=f"generated_success_wave{extra_wave_index}",
                                    followup_kind="generated_success",
                                    subsystem=str(args.subsystem or "").strip(),
                                    config=config,
                                    eval_kwargs=extra_wave_eval_kwargs,
                                    progress_label=progress_label,
                                    budget_seconds=extra_wave_budget_seconds,
                                    current_task_decision_budget_seconds=current_task_decision_budget_seconds,
                                )
                                observation_elapsed_seconds = round(time.monotonic() - observation_started_monotonic, 4)
                                observation_completed_at = datetime.now(timezone.utc)
                                if (
                                    extra_wave_generated_metrics is not None
                                    and extra_wave_generated_metrics.generated_total > 0
                                ):
                                    metrics = _merge_generated_metrics(metrics, extra_wave_generated_metrics)
                                observation_curriculum_followups.append(extra_wave_summary)
                                previous_wave_summary = extra_wave_summary
                                previous_wave_budget_seconds = extra_wave_budget_seconds
                                extra_wave_index += 1
                                extra_input_seed_bundle_path, extra_output_seed_bundle_path = (
                                    extra_output_seed_bundle_path,
                                    extra_input_seed_bundle_path,
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
    if metrics is not None and observation_primary_metrics_salvaged_from_partial_summary:
        salvage_warning = "recovered primary metrics from timed-out observation partial summary"
        observation_warning = (
            f"{observation_warning}; {salvage_warning}" if observation_warning else salvage_warning
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
                    "observation_requested_primary_task_limit": requested_primary_task_limit,
                    "observation_primary_task_limit": primary_task_limit,
                    "observation_primary_task_limit_source": primary_task_limit_source,
                    "observation_current_task_decision_budget_seconds": current_task_decision_budget_seconds,
                    "observation_current_task_decision_budget_source": current_task_decision_budget_source,
                    "observation_current_task_decision_budget_exceeded": observation_current_task_decision_budget_exceeded,
                    "observation_current_task_timeout_budget_seconds": observation_timeout_budget_seconds,
                    "observation_current_task_timeout_budget_source": observation_timeout_budget_source,
                    "observation_elapsed_seconds": observation_elapsed_seconds,
                    "observation_primary_elapsed_seconds": primary_observation_elapsed_seconds,
                    "observation_budget_seconds": max_observation_seconds,
                    "observation_budget_source": observation_budget_guard_source,
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
                    "observation_retried_without_tolbert_context": observation_retried_without_tolbert_context,
                    "observation_tolbert_retry_warning": observation_tolbert_retry_warning,
                    "observation_retried_with_reduced_primary_task_limit": observation_retried_with_reduced_primary_task_limit,
                    "observation_primary_task_limit_retry_warning": observation_primary_task_limit_retry_warning,
                    "observation_primary_task_limit_retry_applied": observation_primary_task_limit_retry_applied,
                    "observation_tolbert_context_disabled_reason": observation_tolbert_context_disabled_reason,
                    "observation_operator_policy_override_reason": observation_operator_policy_override_reason,
                    "observation_staged_curriculum": observation_staged_curriculum,
                    "observation_priority_source": observation_priority_source,
                    "observation_priority_expansion_source": observation_priority_expansion_source,
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

    metrics = _enrich_metrics_task_outcomes_from_partial_summaries(
        metrics,
        initial_observation_partial_summary,
        observation_partial_summary,
    )

    ranked_experiments = planner.rank_experiments(metrics)
    requested_subsystem_fallback_allowed = bool(args.generate_only or scope_id)
    (
        target,
        requested_subsystem_available,
        requested_subsystem_fallback_warning,
    ) = _resolve_requested_experiment(
        ranked_experiments,
        requested_subsystem,
        allow_fallback=requested_subsystem_fallback_allowed,
    )
    if requested_subsystem_fallback_warning:
        _progress(progress_label, f"phase=selection warning={requested_subsystem_fallback_warning}")
    ranked_variants = planner.rank_variants(target, metrics)
    requested_variant_strategy_family = str(args.variant_strategy_family or "").strip()
    preferred_variant_id = ""
    if target.subsystem == "recovery" and requested_variant_strategy_family:
        preferred_variant_id = _variant_strategy_family_preferred_variant_id(requested_variant_strategy_family)
    if args.variant_id:
        matching_variants = [variant for variant in ranked_variants if variant.variant_id == args.variant_id]
        if not matching_variants:
            raise SystemExit(f"variant_id={args.variant_id} is not available for subsystem={args.subsystem}")
        variant = matching_variants[0]
    elif preferred_variant_id:
        matching_variants = [variant for variant in ranked_variants if variant.variant_id == preferred_variant_id]
        variant = matching_variants[0] if matching_variants else _select_guided_variant(ranked_variants)
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
                "generated_total": metrics.generated_total,
                "generated_passed": metrics.generated_passed,
                "generated_pass_rate": metrics.generated_pass_rate,
                "generated_by_kind": dict(metrics.generated_by_kind),
                "generated_passed_by_kind": dict(metrics.generated_passed_by_kind),
                "generated_by_benchmark_family": dict(metrics.generated_by_benchmark_family),
                "generated_passed_by_benchmark_family": dict(metrics.generated_passed_by_benchmark_family),
                **_retrieval_summary_fields(metrics),
                "selected_experiment_score": target.score,
                "selected_experiment_expected_gain": target.expected_gain,
                "selected_experiment_estimated_cost": target.estimated_cost,
                "selected_experiment_subsystem": target.subsystem,
                "selected_variant_id": variant.variant_id,
                "selected_variant_score": variant.score,
                "requested_variant_strategy_family": requested_variant_strategy_family,
                "selected_variant_strategy_family": requested_variant_strategy_family,
                "requested_subsystem": requested_subsystem,
                "requested_subsystem_available": requested_subsystem_available,
                "requested_subsystem_fallback_allowed": requested_subsystem_fallback_allowed,
                "requested_subsystem_fallback_warning": requested_subsystem_fallback_warning,
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
                "observation_requested_primary_task_limit": requested_primary_task_limit,
                "observation_primary_task_limit": primary_task_limit,
                "observation_primary_task_limit_source": primary_task_limit_source,
                "observation_current_task_decision_budget_seconds": current_task_decision_budget_seconds,
                "observation_current_task_decision_budget_source": current_task_decision_budget_source,
                "observation_current_task_decision_budget_exceeded": observation_current_task_decision_budget_exceeded,
                "observation_current_task_timeout_budget_seconds": observation_timeout_budget_seconds,
                "observation_current_task_timeout_budget_source": observation_timeout_budget_source,
                "observation_elapsed_seconds": observation_elapsed_seconds,
                "observation_primary_elapsed_seconds": primary_observation_elapsed_seconds,
                "observation_budget_seconds": max_observation_seconds,
                "observation_budget_source": observation_budget_guard_source,
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
                "observation_retried_without_tolbert_context": observation_retried_without_tolbert_context,
                "observation_tolbert_retry_warning": observation_tolbert_retry_warning,
                "observation_retried_with_reduced_primary_task_limit": observation_retried_with_reduced_primary_task_limit,
                "observation_primary_task_limit_retry_warning": observation_primary_task_limit_retry_warning,
                "observation_primary_task_limit_retry_applied": observation_primary_task_limit_retry_applied,
                "observation_tolbert_context_disabled_reason": observation_tolbert_context_disabled_reason,
                "observation_operator_policy_override_reason": observation_operator_policy_override_reason,
                "observation_staged_curriculum": observation_staged_curriculum,
                "observation_priority_source": observation_priority_source,
                "observation_priority_expansion_source": observation_priority_expansion_source,
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
                "requested_variant_strategy_family": requested_variant_strategy_family,
                "selected_variant_strategy_family": requested_variant_strategy_family,
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
                "requested_variant_strategy_family": requested_variant_strategy_family,
                "selected_variant_strategy_family": requested_variant_strategy_family,
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
