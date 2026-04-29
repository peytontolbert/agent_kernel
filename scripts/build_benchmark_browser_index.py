from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

from agent_kernel.extensions.artifact_repair_contracts import classify_artifact_contract_failure_report


DEFAULT_OUTPUT = Path("web/benchmark_browser/benchmark_index.json")
DEFAULT_LIVE_OUTPUT = Path("web/benchmark_browser/benchmark_live_status.json")
DEFAULT_TARGET_PACKET = Path("docs/evidence/a8_coding_superhuman_target_packet_20260426.json")
DEFAULT_SOURCE_MANIFEST = Path("config/a8_benchmark_dataset_sources.json")
DEFAULT_SOURCE_STATUS = Path("benchmarks/a8_dataset_sources/status.json")


A8_BENCHMARK_GATES = [
    {
        "benchmark": "codeforces",
        "label": "Codeforces",
        "metric": "rating_equivalent",
        "threshold_key": "codeforces_rating_equivalent",
        "target": "rating equivalent >= 3000",
        "kind": "score_at_least",
    },
    {
        "benchmark": "mle_bench",
        "label": "MLE-Bench",
        "metric": "gold_medal_rate",
        "threshold_key": "mle_bench_gold_medal_rate",
        "target": "gold medal rate >= 20%",
        "kind": "rate_at_least",
    },
    {
        "benchmark": "swe_bench_verified",
        "label": "SWE-Bench Verified",
        "metric": "resolve_rate",
        "threshold_key": "swe_bench_verified_resolve_rate",
        "target": "resolve rate >= 80%",
        "kind": "rate_at_least",
        "dataset_name": "SWE-Bench Verified",
    },
    {
        "benchmark": "swe_rebench",
        "label": "SWE-ReBench",
        "metric": "resolve_rate",
        "threshold_key": "swe_rebench_resolve_rate",
        "target": "resolve rate >= 60%",
        "kind": "rate_at_least",
        "dataset_name": "SWE-ReBench",
    },
    {
        "benchmark": "re_bench",
        "label": "RE-Bench",
        "metric": "human_expert_win_rate",
        "threshold_key": "re_bench_human_expert_win_rate",
        "target": "human expert win rate >= 50%",
        "kind": "rate_at_least",
    },
]

A8_SUPPORT_GATES = [
    {
        "benchmark": "sustained_coding_window",
        "label": "Sustained Coding Window",
        "metric": "task_count",
        "threshold_key": "superhuman_coding_task_count",
        "secondary_threshold_key": "superhuman_coding_window_count",
        "target": ">= 100 tasks across >= 3 windows",
        "kind": "count_at_least",
    },
    {
        "benchmark": "recursive_compounding",
        "label": "Recursive Compounding",
        "metric": "retained_gain_runs",
        "threshold_key": "recursive_compounding_retained_gain_runs",
        "secondary_threshold_key": "recursive_compounding_window_count",
        "target": ">= 5 retained-gain runs across >= 3 windows",
        "kind": "count_at_least",
    },
]

STANDALONE_LEADERBOARD_GATES = [
    {
        "benchmark": "swe_bench_live",
        "label": "SWE-bench Live Verified",
        "metric": "resolve_rate",
        "threshold_key": "swe_bench_live_resolve_rate",
        "target": "official leaderboard submission package",
        "kind": "standalone_leaderboard",
        "dataset_name": "SWE-bench Live Verified",
    }
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("instances", "data", "rows", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"expected dataset list or object with instances/data/rows/tasks at {path}")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return []


def _shorten(value: str, limit: int = 7000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + f"\n...[truncated {len(value) - limit} chars]"


def _repo_from_instance(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    owner, rest = instance_id.split("__", 1)
    repo = rest.split("-", 1)[0]
    return f"{owner}/{repo}" if owner and repo else ""


def _dataset_name(path: Path) -> str:
    parts = path.parts
    if "swe_bench_live" in parts:
        return "SWE-bench Live Verified"
    if "swe_rebench" in parts:
        return "SWE-ReBench"
    if "swe_bench" in parts:
        return "SWE-Bench Full"
    if "swe_bench_lite_probe" in parts:
        return "SWE-Bench Lite"
    if "swe_bench_verified" in parts and "selected_lite_overlap" in parts:
        return "SWE-Bench Verified Lite Overlap"
    if "swe_bench_verified" in parts:
        return "SWE-Bench Verified"
    return path.stem


def _result_name(path: Path) -> str:
    parent = path.parent.name
    if parent == "evaluation_results":
        return path.parent.parent.name
    return parent.removeprefix("evaluation_results_") or path.stem


def _summary_name(path: Path) -> str:
    if path.name == "summary.json":
        return path.parent.name
    return path.stem.removeprefix("summary_")


def _prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append({"line": line_number, "error": str(exc)})
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _build_dataset(path: Path, root: Path) -> dict[str, Any]:
    records = _read_dataset(path)
    repos: dict[str, int] = {}
    years: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    for item in records:
        instance_id = _text(item.get("instance_id") or item.get("id"))
        repo = _text(item.get("repo")) or _repo_from_instance(instance_id)
        repos[repo or "unknown"] = repos.get(repo or "unknown", 0) + 1
        created_at = _text(item.get("created_at"))
        if len(created_at) >= 4 and created_at[:4].isdigit():
            years[created_at[:4]] = years.get(created_at[:4], 0) + 1
        instances.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "version": _text(item.get("version")),
                "created_at": created_at,
                "base_commit": _text(item.get("base_commit")),
                "problem_statement": _shorten(_text(item.get("problem_statement"))),
                "hints_text": _shorten(_text(item.get("hints_text")), 3000),
                "fail_to_pass": _json_list(item.get("FAIL_TO_PASS")),
                "pass_to_pass_count": len(_json_list(item.get("PASS_TO_PASS"))),
                "has_reference_patch": bool(_text(item.get("patch"))),
                "has_test_patch": bool(_text(item.get("test_patch"))),
            }
        )
    return {
        "name": _dataset_name(path),
        "path": str(path.relative_to(root)),
        "task_count": len(instances),
        "repo_counts": dict(sorted(repos.items())),
        "year_counts": dict(sorted(years.items())),
        "instances": instances,
    }


def _build_result(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    total = int(payload.get("total_instances") or payload.get("task_count") or payload.get("submitted") or 0)
    resolved_ids = [
        str(item)
        for item in payload.get("resolved_ids", payload.get("success_ids", []))
        if isinstance(item, str)
    ]
    completed_ids = [str(item) for item in payload.get("completed_ids", []) if isinstance(item, str)]
    unresolved_ids = [
        str(item)
        for item in payload.get("unresolved_ids", payload.get("failure_ids", []))
        if isinstance(item, str)
    ]
    error_ids = [str(item) for item in payload.get("error_ids", []) if isinstance(item, str)]
    incomplete_ids = [
        str(item)
        for item in payload.get("incomplete_ids", payload.get("empty_patch_ids", []))
        if isinstance(item, str)
    ]
    return {
        "name": _result_name(path),
        "path": str(path.relative_to(root)),
        "total": total,
        "completed": len(completed_ids),
        "resolved": len(resolved_ids),
        "unresolved": len(unresolved_ids),
        "errors": len(error_ids),
        "incomplete": len(incomplete_ids),
        "resolve_rate": (len(resolved_ids) / total) if total else 0.0,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "incomplete_ids": incomplete_ids,
    }


def _build_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return {
        "name": _summary_name(path),
        "path": str(path.relative_to(root)),
        "report_kind": _text(payload.get("report_kind")),
        "created_at": _text(payload.get("created_at")),
        "metrics": metrics,
        "task_count": metrics.get("task_count"),
        "resolved_count": metrics.get("resolved_count"),
        "resolve_rate": metrics.get("resolve_rate"),
        "benchmark": _text(payload.get("benchmark")),
        "repo_slices": payload.get("repo_slices", []),
    }


def _build_prediction_file(path: Path, root: Path) -> dict[str, Any]:
    records = _prediction_records(path)
    repos: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    for record in records:
        instance_id = _text(record.get("instance_id"))
        repo = _repo_from_instance(instance_id)
        repos[repo or "unknown"] = repos.get(repo or "unknown", 0) + 1
        patch = _text(record.get("model_patch") or record.get("patch"))
        instances.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "model_name_or_path": _text(record.get("model_name_or_path")),
                "patch_chars": len(patch),
                "has_patch": bool(patch.strip()),
            }
        )
    return {
        "name": path.stem,
        "path": str(path.relative_to(root)),
        "prediction_count": len(records),
        "repo_counts": dict(sorted(repos.items())),
        "instances": instances,
    }


def _build_run_spec(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    runner = payload.get("runner") if isinstance(payload.get("runner"), dict) else {}
    adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    return {
        "name": path.stem,
        "path": str(path.relative_to(root)),
        "benchmark": _text(payload.get("benchmark")),
        "benchmark_role": _text(payload.get("benchmark_role")),
        "ready_to_run": bool(payload.get("ready_to_run")),
        "runner_kind": _text(runner.get("kind")),
        "dataset_name": _text(runner.get("dataset_name")),
        "predictions_path": _text(runner.get("predictions_path")),
        "results_json": _text(runner.get("results_json")),
        "summary_json": _text(adapter.get("summary_json")),
        "open_limits": [str(item) for item in payload.get("open_limits", [])],
    }


def _score_from_summary(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = _build_summary(path, root)
    resolved = _number(summary.get("resolved_count"))
    total = _number(summary.get("task_count"))
    rate = _summary_metric(summary, "resolve_rate")
    if total is None:
        return None
    return {
        "status": "available",
        "resolved_count": int(resolved or 0),
        "task_count": int(total),
        "resolve_rate": float(rate or 0.0),
        "score_source": "summary_json",
        "summary_json": summary.get("path", ""),
    }


def _score_from_results(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json_object(path)
    result = _build_result(path, root)
    total = int(result.get("total") or 0)
    if total <= 0:
        return None
    passed_ids = [str(item) for item in payload.get("success_ids", payload.get("resolved_ids", [])) if isinstance(item, str)]
    failed_ids = [str(item) for item in payload.get("failure_ids", payload.get("unresolved_ids", [])) if isinstance(item, str)]
    return {
        "status": "available",
        "resolved_count": int(result.get("resolved") or 0),
        "task_count": total,
        "failed_count": len(failed_ids),
        "resolve_rate": float(result.get("resolve_rate") or 0.0),
        "score_source": "results_json",
        "results_json": result.get("path", ""),
        "passed_instance_ids": passed_ids[:50],
        "failed_instance_ids": failed_ids[:50],
    }


def _build_official_scorecards(root: Path, run_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scorecards: dict[str, dict[str, Any]] = {}
    for spec in run_specs:
        benchmark = _text(spec.get("benchmark"))
        if not benchmark:
            continue
        summary_json = _text(spec.get("summary_json"))
        results_json = _text(spec.get("results_json"))
        summary_path = _resolve_index_path(root, summary_json) if summary_json else None
        results_path = _resolve_index_path(root, results_json) if results_json else None
        score = (
            _score_from_summary(summary_path, root)
            if summary_path is not None
            else None
        ) or (
            _score_from_results(results_path, root)
            if results_path is not None
            else None
        )
        if score is None:
            score = {
                "status": "pending",
                "resolved_count": None,
                "task_count": None,
                "resolve_rate": None,
                "score_source": "",
            }
        score.update(
            {
                "benchmark": benchmark,
                "benchmark_role": _text(spec.get("benchmark_role")),
                "run_spec_path": _text(spec.get("path")),
                "results_json": results_json,
                "summary_json": summary_json,
                "leaderboard_submission_ready": False,
            }
        )
        if score.get("status") == "available":
            score["leaderboard_submission_ready"] = bool(results_json and summary_json)
        prior = scorecards.get(benchmark)
        if prior is None or (prior.get("status") != "available" and score.get("status") == "available"):
            scorecards[benchmark] = score
    return scorecards


def _prediction_count_from_json(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return None


def _build_rolling_scorecards(root: Path, harness_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scorecards: dict[str, dict[str, Any]] = {}
    for harness in harness_specs:
        benchmark = _text(harness.get("benchmark"))
        run_config = harness.get("run_config") if isinstance(harness.get("run_config"), dict) else {}
        score_kind = _text(run_config.get("score_kind"))
        if not benchmark or not score_kind:
            continue
        artifacts = harness.get("artifacts") if isinstance(harness.get("artifacts"), dict) else {}
        summary_json = _text(artifacts.get("summary_json"))
        results_json = _text(artifacts.get("results_json"))
        preds_json = _text(artifacts.get("predictions_patch_json"))
        summary_path = _resolve_index_path(root, summary_json) if summary_json else None
        results_path = _resolve_index_path(root, results_json) if results_json else None
        preds_path = _resolve_index_path(root, preds_json) if preds_json else None
        score = (
            _score_from_summary(summary_path, root)
            if summary_path is not None
            else None
        ) or (
            _score_from_results(results_path, root)
            if results_path is not None
            else None
        )
        if score is None:
            score = _score_from_partial_reports(results_path, root) if results_path is not None else None
        if score is None:
            score = {
                "status": "pending",
                "resolved_count": None,
                "task_count": None,
                "resolve_rate": None,
                "score_source": "",
            }
        score.update(
            {
                "benchmark": benchmark,
                "score_kind": score_kind,
                "run_id": _text(run_config.get("run_id")),
                "results_json": results_json,
                "summary_json": summary_json,
                "predictions_patch_json": preds_json,
                "prediction_count": _prediction_count_from_json(preds_path) if preds_path is not None else None,
                "label": "Rolling completed-subset official score",
                "final_leaderboard_score": False,
            }
        )
        scorecards[f"{benchmark}:{score_kind}"] = score
    return scorecards


def _score_from_partial_reports(results_path: Path, root: Path) -> dict[str, Any] | None:
    report_dir = results_path.parent
    if not report_dir.exists():
        return None
    reports: list[dict[str, Any]] = []
    for report_path in sorted(report_dir.glob("*/report.json")):
        try:
            payload = _read_json_object(report_path)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload.get("resolved"), bool):
            reports.append(payload)
    if not reports:
        return None
    resolved = sum(1 for report in reports if report.get("resolved") is True)
    total = len(reports)
    passed_ids = [
        _text(report.get("instance_id"))
        for report in reports
        if report.get("resolved") is True and _text(report.get("instance_id"))
    ]
    failed_ids = [
        _text(report.get("instance_id"))
        for report in reports
        if report.get("resolved") is False and _text(report.get("instance_id"))
    ]
    return {
        "status": "partial",
        "resolved_count": resolved,
        "task_count": total,
        "failed_count": len(failed_ids),
        "resolve_rate": resolved / total if total else 0.0,
        "score_source": "partial_report_json",
        "results_json": str(results_path.relative_to(root)) if results_path.is_relative_to(root) else str(results_path),
        "partial": True,
        "passed_instance_ids": passed_ids[:50],
        "failed_instance_ids": failed_ids[:50],
        "remaining_prediction_count": None,
    }


def _build_harness_run(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        payload = _read_json_object(path)
    except (json.JSONDecodeError, ValueError):
        return None
    if payload.get("report_kind") != "autonomous_benchmark_harness_run_log":
        return None
    active_phase = payload.get("active_phase") if isinstance(payload.get("active_phase"), dict) else {}
    phase_results = [item for item in payload.get("phase_results", []) if isinstance(item, dict)]
    progress = _latest_prediction_task_progress(path.parent, root)
    return {
        "path": str(path.relative_to(root)),
        "benchmark": _text(payload.get("benchmark")),
        "success": bool(payload.get("success")),
        "failed_phase": _text(payload.get("failed_phase")),
        "active_phase": {
            "name": _text(active_phase.get("name")),
            "pid": active_phase.get("pid"),
            "elapsed_seconds": active_phase.get("elapsed_seconds"),
            "heartbeat_at": _text(active_phase.get("heartbeat_at")),
            "started_at": _text(active_phase.get("started_at")),
        }
        if active_phase
        else {},
        "completed_phase_count": len(phase_results),
        "completed_phases": [
            {
                "name": _text(item.get("name")),
                "returncode": item.get("returncode"),
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
            for item in phase_results
        ],
        "phase_progress": progress or {},
    }


def _latest_prediction_task_progress(directory: Path, root: Path) -> dict[str, Any] | None:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for progress_path in directory.glob("prediction_task_progress*.json"):
        if not progress_path.is_file():
            continue
        try:
            payload = _read_json_object(progress_path)
        except (json.JSONDecodeError, ValueError):
            continue
        if payload.get("report_kind") != "swe_bench_prediction_task_progress":
            continue
        candidates.append((progress_path.stat().st_mtime, progress_path, payload))
    if not candidates:
        return None
    _, progress_path, payload = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return {
        "path": str(progress_path.relative_to(root)) if progress_path.is_relative_to(root) else str(progress_path),
        "status": _text(payload.get("status")),
        "updated_at": _text(payload.get("updated_at")),
        "processed_items": int(_number(payload.get("processed_items")) or 0),
        "total_items": int(_number(payload.get("total_items")) or 0),
        "selected_tasks": int(_number(payload.get("selected_tasks")) or 0),
        "progress_rate": _number(payload.get("progress_rate")) or 0.0,
        "current_instance_id": _text(payload.get("current_instance_id")),
        "current_repo": _text(payload.get("current_repo")),
        "output_manifest_json": _text(payload.get("output_manifest_json")),
    }


def _resolve_index_path(root: Path, value: str) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else root / raw


def _human_benchmark(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title() if value else "Benchmark"


def _read_job_report(reports_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    if not reports_dir.exists():
        return {}
    job_id = _text(job.get("job_id")) or _text(job.get("id"))
    task_id = _text(job.get("task_id"))
    candidates: list[Path] = []
    if job_id:
        safe_id = job_id.replace(":", "_")
        candidates.extend(sorted(reports_dir.glob(f"*{safe_id}.json")))
    if not candidates and task_id:
        candidates.extend(sorted(reports_dir.glob(f"*{task_id}*.json")))
    if not candidates:
        return {}
    try:
        payload = _read_json_object(candidates[-1])
    except (json.JSONDecodeError, ValueError):
        return {}
    return payload


def _job_artifact_failure_summary(reports_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    report = _read_job_report(reports_dir, job)
    if not report:
        return {
            "mode": "report_missing",
            "repairable": False,
            "last_decision_source": "",
            "evidence": [],
        }
    classification = classify_artifact_contract_failure_report(report)
    return {
        "mode": _text(classification.get("mode")) or "artifact_contract_unknown",
        "repairable": bool(classification.get("repairable")),
        "last_decision_source": _text(classification.get("last_decision_source")),
        "evidence": [str(item) for item in classification.get("evidence", [])],
    }


def _event(
    *,
    at: str,
    benchmark: str,
    kind: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "at": at,
        "benchmark": benchmark,
        "kind": kind,
        "message": message,
        "detail": detail or {},
    }


def _build_queue_snapshot_from_harness(harness: dict[str, Any], root: Path) -> dict[str, Any] | None:
    artifacts = harness.get("artifacts") if isinstance(harness.get("artifacts"), dict) else {}
    queue_root = _text(artifacts.get("queue_root"))
    if not queue_root:
        return None
    queue_path = _resolve_index_path(root, str(Path(queue_root) / "queue.json"))
    if not queue_path.exists():
        return None
    try:
        payload = _read_json_object(queue_path)
    except (json.JSONDecodeError, ValueError):
        return None
    jobs = [job for job in payload.get("jobs", []) if isinstance(job, dict)]
    state_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    artifact_failure_mode_counts: dict[str, int] = {}
    recent_artifact_failures: list[dict[str, Any]] = []
    recent_events: list[dict[str, Any]] = []
    reports_dir = queue_path.parent / "reports"
    for job in jobs:
        state = _text(job.get("state")) or "unknown"
        outcome = _text(job.get("outcome"))
        state_counts[state] = state_counts.get(state, 0) + 1
        if outcome:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        artifact_failure: dict[str, Any] = {}
        if state in {"failed", "safe_stop"}:
            artifact_failure = _job_artifact_failure_summary(reports_dir, job)
            mode = _text(artifact_failure.get("mode")) or "artifact_contract_unknown"
            artifact_failure_mode_counts[mode] = artifact_failure_mode_counts.get(mode, 0) + 1
            recent_artifact_failures.append(
                {
                    "at": _text(job.get("finished_at")),
                    "task_id": _text(job.get("task_id")),
                    "state": state,
                    "outcome": outcome,
                    "mode": mode,
                    "repairable": bool(artifact_failure.get("repairable")),
                    "last_decision_source": _text(artifact_failure.get("last_decision_source")),
                    "evidence": [str(item) for item in artifact_failure.get("evidence", [])][:12],
                }
            )
        history = [item for item in job.get("history", []) if isinstance(item, dict)]
        if history:
            latest = history[-1]
            recent_events.append(
                {
                    "at": _text(latest.get("recorded_at")),
                    "event": _text(latest.get("event")),
                    "task_id": _text(job.get("task_id")),
                    "state": state,
                    "outcome": outcome,
                    "artifact_failure_mode": _text(artifact_failure.get("mode")) if artifact_failure else "",
                }
            )
    recent_events = sorted(recent_events, key=lambda item: item.get("at", ""))[-20:]
    recent_artifact_failures = sorted(recent_artifact_failures, key=lambda item: item.get("at", ""))[-20:]
    completed = state_counts.get("completed", 0)
    safe_stop = state_counts.get("safe_stop", 0)
    failed = state_counts.get("failed", 0)
    terminal = completed + safe_stop + failed + state_counts.get("cancelled", 0) + state_counts.get("expired", 0)
    return {
        "benchmark": _text(harness.get("benchmark")),
        "queue_path": str(queue_path.relative_to(root)) if queue_path.is_relative_to(root) else str(queue_path),
        "total_jobs": len(jobs),
        "terminal_jobs": terminal,
        "active_jobs": state_counts.get("in_progress", 0),
        "queued_jobs": state_counts.get("queued", 0),
        "completed_jobs": completed,
        "safe_stop_jobs": safe_stop,
        "failed_jobs": failed,
        "state_counts": dict(sorted(state_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "artifact_failure_mode_counts": dict(sorted(artifact_failure_mode_counts.items())),
        "recent_artifact_failures": recent_artifact_failures,
        "progress_rate": (terminal / len(jobs)) if jobs else 0.0,
        "recent_events": recent_events,
    }


def _build_live_events(
    *,
    active_runs: dict[str, dict[str, Any]],
    queue_snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for benchmark, run in active_runs.items():
        phase = run.get("active_phase") if isinstance(run.get("active_phase"), dict) else {}
        if phase:
            elapsed = _number(phase.get("elapsed_seconds"))
            elapsed_text = f"{int(elapsed)}s" if elapsed is not None else "unknown elapsed"
            events.append(
                _event(
                    at=_text(phase.get("heartbeat_at")) or now,
                    benchmark=benchmark,
                    kind="harness_heartbeat",
                    message=(
                        f"{_human_benchmark(benchmark)} is running {phase.get('name', 'active phase')} "
                        f"on PID {phase.get('pid', '?')} ({elapsed_text} elapsed)."
                    ),
                    detail={"path": run.get("path", ""), "active_phase": phase},
                )
            )
        progress = run.get("phase_progress") if isinstance(run.get("phase_progress"), dict) else {}
        if progress:
            events.append(
                _event(
                    at=_text(progress.get("updated_at")) or now,
                    benchmark=benchmark,
                    kind="prediction_task_progress",
                    message=(
                        f"{_human_benchmark(benchmark)} prediction-task prep "
                        f"{progress.get('status', 'running')}: "
                        f"{progress.get('processed_items', 0)}/{progress.get('total_items', 0)} rows scanned, "
                        f"{progress.get('selected_tasks', 0)} tasks selected."
                    ),
                    detail=progress,
                )
            )
        for phase_result in run.get("completed_phases", []):
            if not isinstance(phase_result, dict):
                continue
            events.append(
                _event(
                    at=now,
                    benchmark=benchmark,
                    kind="phase_completed",
                    message=(
                        f"{_human_benchmark(benchmark)} completed harness phase "
                        f"{phase_result.get('name', 'unknown')} with return code {phase_result.get('returncode')}."
                    ),
                    detail=phase_result,
                )
            )
    for benchmark, snapshot in queue_snapshots.items():
        events.append(
            _event(
                at=now,
                benchmark=benchmark,
                kind="queue_summary",
                message=(
                    f"{_human_benchmark(benchmark)} queue: {snapshot.get('completed_jobs', 0)} completed, "
                    f"{snapshot.get('safe_stop_jobs', 0)} safe-stop, {snapshot.get('queued_jobs', 0)} queued "
                    f"of {snapshot.get('total_jobs', 0)} jobs."
                ),
                detail=snapshot,
            )
        )
        for item in snapshot.get("recent_events", []):
            event_name = item.get("event") or item.get("state") or "job_event"
            task_id = item.get("task_id") or "job"
            state = item.get("state") or "unknown"
            outcome = item.get("outcome") or ""
            events.append(
                _event(
                    at=item.get("at") or now,
                    benchmark=benchmark,
                    kind=f"job_{event_name}",
                    message=(
                        f"{_human_benchmark(benchmark)} {event_name}: {task_id} "
                        f"state={state}{f' outcome={outcome}' if outcome else ''}."
                    ),
                    detail=item,
                )
            )
        for item in snapshot.get("recent_artifact_failures", [])[-10:]:
            mode = item.get("mode") or "artifact_contract_unknown"
            task_id = item.get("task_id") or "job"
            events.append(
                _event(
                    at=item.get("at") or now,
                    benchmark=benchmark,
                    kind="artifact_contract_failure",
                    message=(
                        f"{_human_benchmark(benchmark)} artifact-contract failure: {task_id} "
                        f"mode={mode}."
                    ),
                    detail=item,
                )
            )
    return sorted(events, key=lambda item: item.get("at", ""), reverse=True)[:80]


def _build_targets(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = _read_json_object(path)
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return {
        "path": str(path),
        "thresholds": target.get("thresholds", {}),
        "benchmark_sources": target.get("benchmark_sources", {}),
        "acceptance_policy": target.get("acceptance_policy", {}),
        "current_status": payload.get("current_status", {}),
    }


def _build_dataset_sources(root: Path, source_manifest: Path | None, source_status: Path | None) -> dict[str, Any]:
    manifest_payload: dict[str, Any] = {}
    status_payload: dict[str, Any] = {}
    if source_manifest and source_manifest.exists():
        manifest_payload = _read_json_object(source_manifest)
    if source_status and source_status.exists():
        status_payload = _read_json_object(source_status)
    status_by_source = {
        (
            str(item.get("benchmark", "")),
            str(item.get("local_path", "")),
        ): item
        for item in status_payload.get("sources", [])
        if isinstance(item, dict)
    }
    sources: list[dict[str, Any]] = []
    for item in manifest_payload.get("sources", []):
        if not isinstance(item, dict):
            continue
        benchmark = str(item.get("benchmark", ""))
        local_path = Path(str(item.get("local_path", "")).strip())
        target = local_path if local_path.is_absolute() else root / local_path
        relative_target = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
        status = status_by_source.get(
            (
                benchmark,
                relative_target,
            ),
            {},
        )
        exists = bool(status.get("exists", target.exists()))
        sources.append(
            {
                **item,
                "local_path": relative_target,
                "status": status.get("status", "available" if exists else "missing"),
                "exists": exists,
                "rows": status.get("rows"),
                "size_bytes": status.get("size_bytes", target.stat().st_size if target.exists() and target.is_file() else 0),
                "last_action": status.get("last_action", {}),
                "error": status.get("error", ""),
            }
        )
    return {
        "source_manifest": str(source_manifest.relative_to(root)) if source_manifest and source_manifest.exists() and source_manifest.is_relative_to(root) else str(source_manifest or ""),
        "status_path": str(source_status.relative_to(root)) if source_status and source_status.exists() and source_status.is_relative_to(root) else str(source_status or ""),
        "generated_at": status_payload.get("generated_at", ""),
        "sources": sources,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _summary_metric(summary: dict[str, Any], metric: str) -> float | None:
    value = _number(summary.get(metric))
    if value is not None:
        return value
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    value = _number(metrics.get(metric))
    if value is not None:
        return value
    if metric == "resolve_rate":
        resolved = _number(summary.get("resolved_count"))
        total = _number(summary.get("task_count"))
        if resolved is not None and total:
            return resolved / total
    return None


def _best_summary_for_benchmark(summaries: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    candidates = [
        summary
        for summary in summaries
        if summary.get("benchmark") == benchmark or benchmark in str(summary.get("path", ""))
    ]
    if benchmark == "swe_bench_verified":
        candidates.extend(
            summary
            for summary in summaries
            if "swe_bench_verified" in str(summary.get("path", "")) or summary.get("name") == "selected_lite_overlap"
        )
    if not candidates:
        return None
    if benchmark == "swe_bench_verified":
        return sorted(
            candidates,
            key=lambda item: (
                int(_number(item.get("task_count")) or 0),
                str(item.get("created_at") or item.get("path") or ""),
            ),
            reverse=True,
        )[0]
    return sorted(candidates, key=lambda item: str(item.get("created_at") or item.get("path") or ""), reverse=True)[0]


def _dataset_count(datasets: list[dict[str, Any]], dataset_name: str | None) -> int | None:
    if not dataset_name:
        return None
    for dataset in datasets:
        if dataset.get("name") == dataset_name:
            return int(dataset.get("task_count") or 0)
    return None


def _run_spec_for_benchmark(run_specs: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    candidates = [spec for spec in run_specs if spec.get("benchmark") == benchmark]
    config_candidates = [spec for spec in candidates if str(spec.get("path", "")).startswith("config/a8_benchmark_run_specs/")]
    for spec in config_candidates or candidates:
        if spec.get("benchmark") == benchmark:
            return spec
    return None


def _active_run_for_benchmark(harness_runs: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    candidates = [
        run
        for run in harness_runs
        if run.get("benchmark") == benchmark
        and not run.get("success")
        and isinstance(run.get("active_phase"), dict)
        and run.get("active_phase")
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: str(item.get("active_phase", {}).get("heartbeat_at") or item.get("path") or ""),
        reverse=True,
    )[0]


def _required_count(threshold: float | None, dataset_total: int | None) -> int | None:
    if threshold is None or dataset_total is None:
        return None
    return int(-(-threshold * dataset_total // 1))


def _gate_status(current: float | None, threshold: float | None, evidence: dict[str, Any] | None) -> str:
    if current is None:
        return "no_evidence"
    if threshold is None:
        return "tracked"
    if current >= threshold:
        return "met"
    return "partial" if evidence else "no_evidence"


def _count_aware_gate_status(
    *,
    current: float | None,
    threshold: float | None,
    evidence: dict[str, Any] | None,
    numerator: float | None,
    required: int | None,
) -> str:
    status = _gate_status(current, threshold, evidence)
    if status != "met" or required is None:
        return status
    if numerator is None or numerator < required:
        return "partial"
    return status


def _build_gate_progress(
    *,
    gate: dict[str, Any],
    thresholds: dict[str, Any],
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    harness_runs: list[dict[str, Any]],
    support_gate: bool = False,
) -> dict[str, Any]:
    benchmark = str(gate["benchmark"])
    summary = _best_summary_for_benchmark(summaries, benchmark)
    threshold = _number(thresholds.get(str(gate["threshold_key"])))
    dataset_total = _dataset_count(datasets, gate.get("dataset_name"))
    required = _required_count(threshold, dataset_total)
    current = _summary_metric(summary, str(gate["metric"])) if summary else None
    numerator = _number(summary.get("resolved_count")) if summary else None
    denominator = _number(summary.get("task_count")) if summary else None
    if gate["kind"] == "count_at_least" and numerator is None:
        numerator = current
    if gate["kind"] == "score_at_least":
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    elif gate["kind"] == "count_at_least":
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    else:
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    run_spec = _run_spec_for_benchmark(run_specs, benchmark)
    active_run = _active_run_for_benchmark(harness_runs, benchmark)
    caveats: list[str] = []
    if benchmark == "swe_bench_verified" and summary and denominator and dataset_total and denominator < dataset_total:
        caveats.append(
            f"Current evidence covers a {int(denominator)} task slice, not the full {dataset_total} task benchmark."
        )
    if active_run:
        active_phase = active_run.get("active_phase", {})
        caveats.append(
            "A full benchmark harness is currently active at phase "
            f"{active_phase.get('name', 'unknown')}; it is not completed evidence until it writes a summary packet."
        )
    if not summary:
        caveats.append("No local evidence summary has been produced for this gate.")
    if run_spec and (
        not run_spec.get("ready_to_run") or run_spec.get("benchmark_role") == "standalone_leaderboard"
    ):
        caveats.extend(str(item) for item in run_spec.get("open_limits", []))
    return {
        "benchmark": benchmark,
        "label": gate["label"],
        "metric": gate["metric"],
        "target": gate["target"],
        "threshold": threshold,
        "secondary_threshold": _number(thresholds.get(str(gate.get("secondary_threshold_key")))),
        "dataset_total": dataset_total,
        "required_count": required,
        "current_value": current,
        "current_numerator": numerator,
        "current_denominator": denominator,
        "progress_to_gate": min(max(progress_to_gate, 0.0), 1.0),
        "status": _count_aware_gate_status(
            current=current,
            threshold=threshold,
            evidence=summary,
            numerator=numerator,
            required=required,
        ),
        "support_gate": support_gate,
        "evidence_path": summary.get("path") if summary else "",
        "run_spec_path": run_spec.get("path") if run_spec else "",
        "active_run": active_run or {},
        "ready_to_run": bool(run_spec.get("ready_to_run")) if run_spec else False,
        "caveats": caveats,
    }


def _build_a8_progress(
    targets: dict[str, Any],
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    harness_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = targets.get("thresholds") if isinstance(targets.get("thresholds"), dict) else {}
    benchmark_gates = [
        _build_gate_progress(
            gate=gate,
            thresholds=thresholds,
            summaries=summaries,
            datasets=datasets,
            run_specs=run_specs,
            harness_runs=harness_runs,
        )
        for gate in A8_BENCHMARK_GATES
    ]
    support_gates = [
        _build_gate_progress(
            gate=gate,
            thresholds=thresholds,
            summaries=summaries,
            datasets=datasets,
            run_specs=run_specs,
            harness_runs=harness_runs,
            support_gate=True,
        )
        for gate in A8_SUPPORT_GATES
    ]
    all_gates = [*benchmark_gates, *support_gates]
    met_count = sum(1 for gate in all_gates if gate["status"] == "met")
    return {
        "level": "A8",
        "domain": "coding",
        "claim_ready": all(gate["status"] == "met" for gate in all_gates),
        "met_gate_count": met_count,
        "gate_count": len(all_gates),
        "benchmark_gates": benchmark_gates,
        "support_gates": support_gates,
        "acceptance_policy": targets.get("acceptance_policy", {}),
        "current_status": targets.get("current_status", {}),
    }


def _build_standalone_leaderboard_progress(
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    harness_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = [
        _build_gate_progress(
            gate=gate,
            thresholds={},
            summaries=summaries,
            datasets=datasets,
            run_specs=run_specs,
            harness_runs=harness_runs,
        )
        for gate in STANDALONE_LEADERBOARD_GATES
    ]
    return {
        "role": "standalone_online_leaderboard",
        "description": "Online leaderboard benchmarks tracked separately from the A8 lane.",
        "gates": gates,
    }


def build_benchmark_live_status(root: Path) -> dict[str, Any]:
    harness_runs = [
        run
        for run in (
            _build_harness_run(path, root)
            for path in sorted(root.glob("benchmarks/**/harness*_log.json"))
            if path.is_file()
        )
        if run is not None
    ]
    active_runs = {
        benchmark: run
        for benchmark in sorted({str(run.get("benchmark", "")) for run in harness_runs if str(run.get("benchmark", ""))})
        for run in [_active_run_for_benchmark(harness_runs, benchmark)]
        if run is not None
    }
    harness_specs = [
        payload
        for payload in (
            _read_json_object(path)
            for path in sorted(root.glob("config/autonomous_benchmark_harnesses/*.json"))
            if path.is_file()
        )
        if payload.get("report_kind") == "autonomous_benchmark_harness_spec"
    ]
    run_specs = [
        _build_run_spec(path, root)
        for path in sorted(
            [
                *root.glob("config/a8_benchmark_run_specs/*.json"),
                *root.glob("config/standalone_benchmark_run_specs/*.json"),
            ]
        )
        if path.is_file()
    ]
    queue_snapshots = {
        snapshot["benchmark"]: snapshot
        for snapshot in (
            _build_queue_snapshot_from_harness(harness, root)
            for harness in harness_specs
        )
        if snapshot is not None and snapshot.get("benchmark")
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "harness_runs": harness_runs,
        "active_runs_by_benchmark": active_runs,
        "queue_snapshots_by_benchmark": queue_snapshots,
        "official_scores_by_benchmark": _build_official_scorecards(root, run_specs),
        "rolling_scores": _build_rolling_scorecards(root, harness_specs),
        "semantic_events": _build_live_events(active_runs=active_runs, queue_snapshots=queue_snapshots),
    }


def build_benchmark_browser_index(
    root: Path,
    target_packet: Path | None = DEFAULT_TARGET_PACKET,
    source_manifest: Path | None = DEFAULT_SOURCE_MANIFEST,
    source_status: Path | None = DEFAULT_SOURCE_STATUS,
) -> dict[str, Any]:
    datasets = [
        _build_dataset(path, root)
        for path in sorted(
            [
                *root.glob("benchmarks/**/swe_bench*_test_dataset.json"),
                *root.glob("benchmarks/**/swe_bench_live*_dataset.json"),
                *root.glob("benchmarks/**/swe_rebench*_dataset.json"),
            ]
        )
        if path.is_file()
    ]
    selected = root / "benchmarks/swe_bench_verified/selected_lite_overlap/dataset_selected_lite_overlap.json"
    if selected.exists():
        datasets.append(_build_dataset(selected, root))

    results = [
        _build_result(path, root)
        for path in sorted(root.glob("benchmarks/**/results.json"))
        if path.is_file()
    ]
    summaries = [
        _build_summary(path, root)
        for path in sorted(root.glob("benchmarks/**/summary*.json"))
        if path.is_file()
    ]
    predictions = [
        _build_prediction_file(path, root)
        for path in sorted(root.glob("benchmarks/**/*.jsonl"))
        if "repo_cache" not in path.parts
    ]
    run_specs = [
        _build_run_spec(path, root)
        for path in sorted(
            [
                *root.glob("benchmarks/**/*run_spec.json"),
                *root.glob("config/a8_benchmark_run_specs/*.json"),
                *root.glob("config/standalone_benchmark_run_specs/*.json"),
            ]
        )
        if path.is_file()
    ]
    harness_runs = [
        run
        for run in (
            _build_harness_run(path, root)
            for path in sorted(root.glob("benchmarks/**/harness*_log.json"))
            if path.is_file()
        )
        if run is not None
    ]
    targets = _build_targets(target_packet if target_packet and target_packet.is_absolute() else root / target_packet if target_packet else None)
    resolved_source_manifest = (
        source_manifest if source_manifest and source_manifest.is_absolute() else root / source_manifest if source_manifest else None
    )
    resolved_source_status = source_status if source_status and source_status.is_absolute() else root / source_status if source_status else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "targets": targets,
        "a8_progress": _build_a8_progress(targets, summaries, datasets, run_specs, harness_runs),
        "standalone_leaderboards": _build_standalone_leaderboard_progress(
            summaries,
            datasets,
            run_specs,
            harness_runs,
        ),
        "dataset_sources": _build_dataset_sources(root, resolved_source_manifest, resolved_source_status),
        "datasets": datasets,
        "results": results,
        "summaries": summaries,
        "predictions": predictions,
        "run_specs": run_specs,
        "harness_runs": harness_runs,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--live-output", default=str(DEFAULT_LIVE_OUTPUT))
    parser.add_argument("--target-packet", default=str(DEFAULT_TARGET_PACKET))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--source-status", default=str(DEFAULT_SOURCE_STATUS))
    parser.add_argument("--watch-live", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    live_output = Path(args.live_output)
    if not live_output.is_absolute():
        live_output = root / live_output
    if args.watch_live:
        interval_seconds = max(1.0, float(args.interval_seconds))
        while True:
            live_status = build_benchmark_live_status(root)
            _write_json(live_output, live_status)
            print(
                f"live_generated_at={live_status['generated_at']} "
                f"active_runs={len(live_status['active_runs_by_benchmark'])} output={live_output}",
                flush=True,
            )
            time.sleep(interval_seconds)
    target_packet = Path(args.target_packet)
    index = build_benchmark_browser_index(root, target_packet, Path(args.source_manifest), Path(args.source_status))
    _write_json(output, index)
    live_status = build_benchmark_live_status(root)
    _write_json(live_output, live_status)
    print(
        f"datasets={len(index['datasets'])} results={len(index['results'])} "
        f"predictions={len(index['predictions'])} run_specs={len(index['run_specs'])} "
        f"output={output} live_output={live_output}"
    )


if __name__ == "__main__":
    main()
