from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from evals.harness import run_eval


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _step_is_long_horizon(step: dict[str, object], task_payload: dict[str, object]) -> bool:
    horizon = str(step.get("world_model_horizon", "")).strip().lower()
    if horizon:
        return horizon == "long_horizon"
    return str(task_payload.get("difficulty", "")).strip().lower() == "long_horizon"


def _step_is_productive(step: dict[str, object]) -> bool:
    return _safe_float(step.get("state_progress_delta", 0.0)) > 0.0


def _step_has_pressure(step: dict[str, object]) -> bool:
    return bool(step.get("state_no_progress", False)) or bool(step.get("state_regressed", False)) or _safe_int(
        step.get("state_regression_count", 0)
    ) > 0


def _step_is_recovery(step: dict[str, object]) -> bool:
    return str(step.get("acting_role", "")).strip() in {"planner", "critic"}


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def summarize_long_horizon_persistence(metrics) -> dict[str, object]:
    trajectories = metrics.task_trajectories if isinstance(metrics.task_trajectories, dict) else {}
    summary = {
        "task_count": 0,
        "long_horizon_task_count": 0,
        "long_horizon_success_count": 0,
        "total_steps": 0,
        "long_horizon_steps": 0,
        "productive_long_horizon_steps": 0,
        "pressure_long_horizon_steps": 0,
        "regressed_long_horizon_steps": 0,
        "recovery_role_long_horizon_steps": 0,
        "active_subgoal_long_horizon_steps": 0,
        "subgoal_refresh_count": 0,
        "pressure_events": 0,
        "recovery_response_events": 0,
        "horizon_drop_events": 0,
        "max_long_horizon_streak": 0,
        "long_horizon_task_ids": [],
        "tasks": [],
    }
    long_horizon_task_ids: list[str] = []
    for task_id, payload in trajectories.items():
        if not isinstance(payload, dict):
            continue
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        task_summary = {
            "task_id": str(task_id),
            "benchmark_family": str(payload.get("benchmark_family", "")).strip() or "bounded",
            "difficulty": str(payload.get("difficulty", "")).strip() or "unknown",
            "success": bool(payload.get("success", False)),
            "termination_reason": str(payload.get("termination_reason", "")).strip(),
            "step_count": len(steps),
            "long_horizon_steps": 0,
            "productive_long_horizon_steps": 0,
            "pressure_events": 0,
            "recovery_response_events": 0,
            "subgoal_refresh_count": 0,
            "horizon_drop_events": 0,
            "max_long_horizon_streak": 0,
        }
        summary["task_count"] = _safe_int(summary.get("task_count", 0)) + 1
        previous_subgoal = ""
        previous_was_long_horizon = False
        current_streak = 0
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, dict):
                continue
            summary["total_steps"] = _safe_int(summary.get("total_steps", 0)) + 1
            is_long_horizon = _step_is_long_horizon(raw_step, payload)
            if not is_long_horizon:
                if previous_was_long_horizon:
                    task_summary["horizon_drop_events"] = _safe_int(task_summary.get("horizon_drop_events", 0)) + 1
                    summary["horizon_drop_events"] = _safe_int(summary.get("horizon_drop_events", 0)) + 1
                previous_was_long_horizon = False
                current_streak = 0
                continue
            previous_was_long_horizon = True
            current_streak += 1
            task_summary["max_long_horizon_streak"] = max(
                _safe_int(task_summary.get("max_long_horizon_streak", 0)),
                current_streak,
            )
            summary["max_long_horizon_streak"] = max(
                _safe_int(summary.get("max_long_horizon_streak", 0)),
                current_streak,
            )
            task_summary["long_horizon_steps"] = _safe_int(task_summary.get("long_horizon_steps", 0)) + 1
            summary["long_horizon_steps"] = _safe_int(summary.get("long_horizon_steps", 0)) + 1
            if _step_is_productive(raw_step):
                task_summary["productive_long_horizon_steps"] = _safe_int(
                    task_summary.get("productive_long_horizon_steps", 0)
                ) + 1
                summary["productive_long_horizon_steps"] = _safe_int(summary.get("productive_long_horizon_steps", 0)) + 1
            if bool(raw_step.get("active_subgoal", False)):
                summary["active_subgoal_long_horizon_steps"] = _safe_int(
                    summary.get("active_subgoal_long_horizon_steps", 0)
                ) + 1
            current_subgoal = str(raw_step.get("active_subgoal", "")).strip()
            if current_subgoal and previous_subgoal and current_subgoal != previous_subgoal:
                task_summary["subgoal_refresh_count"] = _safe_int(task_summary.get("subgoal_refresh_count", 0)) + 1
                summary["subgoal_refresh_count"] = _safe_int(summary.get("subgoal_refresh_count", 0)) + 1
            if current_subgoal:
                previous_subgoal = current_subgoal
            has_pressure = _step_has_pressure(raw_step)
            if has_pressure:
                task_summary["pressure_events"] = _safe_int(task_summary.get("pressure_events", 0)) + 1
                summary["pressure_events"] = _safe_int(summary.get("pressure_events", 0)) + 1
                summary["pressure_long_horizon_steps"] = _safe_int(summary.get("pressure_long_horizon_steps", 0)) + 1
                if bool(raw_step.get("state_regressed", False)) or _safe_int(raw_step.get("state_regression_count", 0)) > 0:
                    summary["regressed_long_horizon_steps"] = _safe_int(summary.get("regressed_long_horizon_steps", 0)) + 1
                next_step = steps[index + 1] if index + 1 < len(steps) and isinstance(steps[index + 1], dict) else {}
                next_subgoal = str(next_step.get("active_subgoal", "")).strip()
                next_is_response = _step_is_recovery(next_step) or (
                    bool(next_subgoal) and next_subgoal != current_subgoal
                )
                if next_is_response:
                    task_summary["recovery_response_events"] = _safe_int(
                        task_summary.get("recovery_response_events", 0)
                    ) + 1
                    summary["recovery_response_events"] = _safe_int(summary.get("recovery_response_events", 0)) + 1
            if _step_is_recovery(raw_step):
                summary["recovery_role_long_horizon_steps"] = _safe_int(
                    summary.get("recovery_role_long_horizon_steps", 0)
                ) + 1
        if _safe_int(task_summary.get("long_horizon_steps", 0)) > 0:
            long_horizon_task_ids.append(str(task_id))
            summary["long_horizon_task_count"] = _safe_int(summary.get("long_horizon_task_count", 0)) + 1
            if bool(task_summary.get("success", False)):
                summary["long_horizon_success_count"] = _safe_int(summary.get("long_horizon_success_count", 0)) + 1
            summary["tasks"].append(task_summary)
    summary["long_horizon_task_ids"] = long_horizon_task_ids
    summary["rates"] = {
        "long_horizon_step_share": round(_rate(summary["long_horizon_steps"], summary["total_steps"]), 4),
        "productive_long_horizon_step_rate": round(
            _rate(summary["productive_long_horizon_steps"], summary["long_horizon_steps"]), 4
        ),
        "pressure_long_horizon_step_rate": round(_rate(summary["pressure_long_horizon_steps"], summary["long_horizon_steps"]), 4),
        "recovery_role_long_horizon_step_rate": round(
            _rate(summary["recovery_role_long_horizon_steps"], summary["long_horizon_steps"]), 4
        ),
        "active_subgoal_long_horizon_step_rate": round(
            _rate(summary["active_subgoal_long_horizon_steps"], summary["long_horizon_steps"]), 4
        ),
        "subgoal_refresh_per_100_long_horizon_steps": round(
            100.0 * _rate(summary["subgoal_refresh_count"], summary["long_horizon_steps"]),
            2,
        ),
        "recovery_response_rate": round(_rate(summary["recovery_response_events"], summary["pressure_events"]), 4),
        "long_horizon_success_rate": round(
            _rate(summary["long_horizon_success_count"], summary["long_horizon_task_count"]),
            4,
        ),
    }
    return summary


def _print_summary(payload: dict[str, object]) -> None:
    rates = payload.get("rates", {})
    if not isinstance(rates, dict):
        rates = {}
    print(
        "long_horizon_persistence "
        f"tasks={_safe_int(payload.get('long_horizon_task_count', 0))} "
        f"steps={_safe_int(payload.get('long_horizon_steps', 0))} "
        f"max_streak={_safe_int(payload.get('max_long_horizon_streak', 0))} "
        f"productive_rate={_safe_float(rates.get('productive_long_horizon_step_rate', 0.0)):.2f} "
        f"recovery_response_rate={_safe_float(rates.get('recovery_response_rate', 0.0)):.2f} "
        f"subgoal_refresh_per_100_steps={_safe_float(rates.get('subgoal_refresh_per_100_long_horizon_steps', 0.0)):.2f}"
    )
    print(
        "long_horizon_support "
        f"active_subgoal_rate={_safe_float(rates.get('active_subgoal_long_horizon_step_rate', 0.0)):.2f} "
        f"recovery_role_rate={_safe_float(rates.get('recovery_role_long_horizon_step_rate', 0.0)):.2f} "
        f"pressure_rate={_safe_float(rates.get('pressure_long_horizon_step_rate', 0.0)):.2f} "
        f"horizon_drop_events={_safe_int(payload.get('horizon_drop_events', 0))}"
    )
    for task in payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        print(
            "long_horizon_task "
            f"task_id={str(task.get('task_id', '')).strip()} "
            f"family={str(task.get('benchmark_family', '')).strip()} "
            f"success={int(bool(task.get('success', False)))} "
            f"long_horizon_steps={_safe_int(task.get('long_horizon_steps', 0))} "
            f"productive_steps={_safe_int(task.get('productive_long_horizon_steps', 0))} "
            f"pressure_events={_safe_int(task.get('pressure_events', 0))} "
            f"recovery_response_events={_safe_int(task.get('recovery_response_events', 0))} "
            f"subgoal_refresh_count={_safe_int(task.get('subgoal_refresh_count', 0))} "
            f"max_streak={_safe_int(task.get('max_long_horizon_streak', 0))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-tolbert-context", choices=("0", "1"), default=None)
    parser.add_argument("--use-skills", choices=("0", "1"), default=None)
    parser.add_argument("--use-prompt-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-curriculum-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-retrieval-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-state-estimation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-trust-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-recovery-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-delegation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-operator-policy-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-transition-model-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--include-discovered-tasks", action="store_true")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-benchmark-candidates", action="store_true")
    parser.add_argument("--include-verifier-candidates", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.use_tolbert_context is not None:
        config.use_tolbert_context = args.use_tolbert_context == "1"
    if args.use_skills is not None:
        config.use_skills = args.use_skills == "1"
    if args.use_prompt_proposals is not None:
        config.use_prompt_proposals = args.use_prompt_proposals == "1"
    if args.use_curriculum_proposals is not None:
        config.use_curriculum_proposals = args.use_curriculum_proposals == "1"
    if args.use_retrieval_proposals is not None:
        config.use_retrieval_proposals = args.use_retrieval_proposals == "1"
    if args.use_state_estimation_proposals is not None:
        config.use_state_estimation_proposals = args.use_state_estimation_proposals == "1"
    if args.use_trust_proposals is not None:
        config.use_trust_proposals = args.use_trust_proposals == "1"
    if args.use_recovery_proposals is not None:
        config.use_recovery_proposals = args.use_recovery_proposals == "1"
    if args.use_delegation_proposals is not None:
        config.use_delegation_proposals = args.use_delegation_proposals == "1"
    if args.use_operator_policy_proposals is not None:
        config.use_operator_policy_proposals = args.use_operator_policy_proposals == "1"
    if args.use_transition_model_proposals is not None:
        config.use_transition_model_proposals = args.use_transition_model_proposals == "1"

    metrics = run_eval(
        config=config,
        include_discovered_tasks=args.include_discovered_tasks,
        include_generated=args.include_curriculum,
        include_failure_generated=args.include_failure_curriculum,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_benchmark_candidates=args.include_benchmark_candidates,
        include_verifier_candidates=args.include_verifier_candidates,
    )
    summary = summarize_long_horizon_persistence(metrics)
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    _print_summary(summary)


if __name__ == "__main__":
    main()
