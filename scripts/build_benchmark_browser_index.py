from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.artifact_repair_contracts import classify_artifact_contract_failure_report
from agent_kernel.neural_controller import (
    neural_controller_shadow_promotion_readiness,
    summarize_neural_controller_shadow_documents,
)
from agent_kernel.ops.episode_store import iter_episode_documents
from scripts.report_neural_controller_runtime_contract_metrics import summarize_runtime_contract_metrics


DEFAULT_OUTPUT = Path("web/benchmark_browser/benchmark_index.json")
DEFAULT_LIVE_OUTPUT = Path("web/benchmark_browser/benchmark_live_status.json")
DEFAULT_TARGET_PACKET = Path("docs/evidence/a8_coding_superhuman_target_packet_20260426.json")
DEFAULT_SOURCE_MANIFEST = Path("config/a8_benchmark_dataset_sources.json")
DEFAULT_SOURCE_STATUS = Path("benchmarks/a8_dataset_sources/status.json")
DEFAULT_NEURAL_CONTROLLER_SHADOW_METRICS = Path("trajectories/neural_controller/shadow_metrics.json")
DEFAULT_NEURAL_CONTROLLER_RUNTIME_CONTRACT_METRICS = Path(
    "trajectories/neural_controller/runtime_contract_metrics_current.json"
)
DEFAULT_NEURAL_CONTROLLER_SELECTOR_ACTIVATION_GATE = Path(
    "trajectories/neural_controller/v64_guarded_rowwise_selector_contract_activation_gate.json"
)


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
    source_path = _summary_source_path(path, root)
    if source_path is not None and not source_path.exists():
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
        "score_mtime": path.stat().st_mtime,
        "score_updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
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
        "score_mtime": path.stat().st_mtime,
        "score_updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
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


def _prediction_count_from_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _prediction_instance_ids_from_json(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return {str(key) for key in payload if str(key)}
    if isinstance(payload, list):
        instance_ids: set[str] = set()
        for item in payload:
            if isinstance(item, dict):
                instance_id = _text(item.get("instance_id"))
                if instance_id:
                    instance_ids.add(instance_id)
        return instance_ids
    return None


def _prediction_instance_ids_from_jsonl(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    instance_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            instance_id = _text(payload.get("instance_id"))
            if instance_id:
                instance_ids.add(instance_id)
    return instance_ids


def _pid_is_alive(pid: Any) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _summary_source_path(path: Path, root: Path) -> Path | None:
    try:
        payload = _read_json_object(path)
    except (json.JSONDecodeError, ValueError):
        return None
    source_path = _text(payload.get("source_path"))
    if not source_path:
        return None
    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _verification_summary_from_json(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _read_json_object(path)
    except (json.JSONDecodeError, ValueError):
        return None

    abstained_jobs = [job for job in payload.get("abstained_jobs", []) if isinstance(job, dict)]
    reason_counts: dict[str, int] = {}
    for job in abstained_jobs:
        reasons = job.get("verification_reasons")
        if isinstance(reasons, list):
            reason = str(reasons[0]) if reasons else _text(job.get("reason")) or "unspecified"
        else:
            reason = _text(job.get("reason")) or "unspecified"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top_abstain_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "created_at": _text(payload.get("created_at")),
        "verified_patch_count": int(_number(payload.get("verified_patch_count")) or 0),
        "abstained_patch_count": int(_number(payload.get("abstained_patch_count")) or 0),
        "failed_patch_count": int(_number(payload.get("failed_patch_count")) or 0),
        "skipped_nonterminal_count": int(_number(payload.get("skipped_nonterminal_count")) or 0),
        "skipped_pre_epoch_count": int(_number(payload.get("skipped_pre_epoch_count")) or 0),
        "skipped_missing_count": int(_number(payload.get("skipped_missing_count")) or 0),
        "verification_task_count": int(_number(payload.get("task_count")) or 0),
        "successful_instance_ids": _json_list(payload.get("successful_instance_ids")),
        "retry_instance_ids": _json_list(payload.get("retry_instance_ids")),
        "top_abstain_reasons": top_abstain_reasons,
    }


def _build_rolling_scorecards(root: Path, harness_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scorecards: dict[str, dict[str, Any]] = {}
    for harness in harness_specs:
        benchmark = _text(harness.get("benchmark"))
        run_config = harness.get("run_config") if isinstance(harness.get("run_config"), dict) else {}
        score_kind = _text(run_config.get("score_kind"))
        score_mode = _text(run_config.get("score_mode")) or score_kind
        if not benchmark or not score_kind:
            continue
        artifacts = harness.get("artifacts") if isinstance(harness.get("artifacts"), dict) else {}
        summary_json = _text(artifacts.get("summary_json"))
        results_json = _text(artifacts.get("results_json"))
        preds_json = _text(artifacts.get("predictions_patch_json"))
        verification_json = _text(artifacts.get("patch_job_verification_json"))
        summary_path = _resolve_index_path(root, summary_json) if summary_json else None
        results_path = _resolve_index_path(root, results_json) if results_json else None
        preds_path = _resolve_index_path(root, preds_json) if preds_json else None
        verification_path = _resolve_index_path(root, verification_json) if verification_json else None
        prediction_instance_ids = _prediction_instance_ids_from_json(preds_path) if preds_path is not None else None
        summary_score = _score_from_summary(summary_path, root) if summary_path is not None else None
        results_score = _score_from_results(results_path, root) if results_path is not None else None
        partial_score = (
            _score_from_partial_reports(results_path, root, allowed_instance_ids=prediction_instance_ids)
            if results_path is not None
            else None
        )
        verification_summary = (
            _verification_summary_from_json(verification_path, root) if verification_path is not None else None
        )
        score = _freshest_rolling_score(summary_score, results_score, partial_score)
        if score is None:
            score = {
                "status": "pending",
                "resolved_count": None,
                "task_count": None,
                "resolve_rate": None,
                "score_source": "",
            }
        prediction_count = _prediction_count_from_json(preds_path) if preds_path is not None else None
        if prediction_count == 0:
            score = {
                "status": "no_predictions",
                "resolved_count": 0,
                "task_count": 0,
                "failed_count": 0,
                "resolve_rate": None,
                "score_source": "no_current_predictions",
                "remaining_prediction_count": 0,
            }
        if score.get("status") == "partial" and prediction_count is not None and score.get("task_count") is not None:
            score["remaining_prediction_count"] = max(0, int(prediction_count) - int(score.get("task_count") or 0))
        if preds_path is not None and preds_path.exists() and prediction_count != 0:
            preds_mtime = preds_path.stat().st_mtime
            score_mtime = float(score.get("score_mtime") or 0.0)
            score_task_count = int(score.get("task_count") or 0)
            score_is_complete_for_predictions = (
                score.get("status") == "available"
                and prediction_count is not None
                and score_task_count >= int(prediction_count)
            )
            if not score_mtime:
                score = {
                    "status": "running",
                    "resolved_count": None,
                    "task_count": prediction_count,
                    "failed_count": None,
                    "resolve_rate": None,
                    "score_source": "official_evaluator_running",
                    "remaining_prediction_count": prediction_count,
                }
            elif preds_mtime > score_mtime and not score_is_complete_for_predictions:
                score = {
                    "status": "running",
                    "resolved_count": None,
                    "task_count": prediction_count,
                    "failed_count": None,
                    "resolve_rate": None,
                    "score_source": "official_evaluator_running",
                    "score_mtime": score_mtime,
                    "score_updated_at": _text(score.get("score_updated_at")),
                    "previous_resolved_count": score.get("resolved_count"),
                    "previous_task_count": score.get("task_count"),
                    "previous_resolve_rate": score.get("resolve_rate"),
                    "remaining_prediction_count": prediction_count,
                }
        score.update(
            {
                "benchmark": benchmark,
                "score_kind": score_kind,
                "run_id": _text(run_config.get("run_id")),
                "score_mode": score_mode,
                "results_json": results_json,
                "summary_json": summary_json,
                "predictions_patch_json": preds_json,
                "patch_job_verification_json": verification_json,
                "prediction_count": prediction_count,
                "label": "Rolling raw completed-success official score"
                if "raw" in score_mode
                else "Rolling verified-subset official score",
                "final_leaderboard_score": False,
                "trusted_leaderboard_evidence": "raw" not in score_mode,
            }
        )
        if verification_summary is not None:
            score["verification"] = verification_summary
            score["verified_patch_count"] = verification_summary["verified_patch_count"]
            score["abstained_patch_count"] = verification_summary["abstained_patch_count"]
            score["failed_patch_count"] = verification_summary["failed_patch_count"]
            score["skipped_nonterminal_count"] = verification_summary["skipped_nonterminal_count"]
            score["skipped_pre_epoch_count"] = verification_summary["skipped_pre_epoch_count"]
            score["skipped_missing_count"] = verification_summary["skipped_missing_count"]
        scorecards[f"{benchmark}:{score_kind}"] = score
    scorecards.update(_discover_swe_live_rolling_scorecards(root))
    return scorecards


def _discover_swe_live_rolling_scorecards(root: Path) -> dict[str, dict[str, Any]]:
    scorecards: dict[str, dict[str, Any]] = {}
    rolling_root = root / "benchmarks/swe_bench_live/rolling_score"
    if not rolling_root.exists():
        return scorecards
    for score_dir in sorted(path for path in rolling_root.iterdir() if path.is_dir()):
        results_path = score_dir / "evaluation_results/results.json"
        summary_path = score_dir / "summary.json"
        preds_path = score_dir / "preds.json"
        predictions_jsonl_path = score_dir / "predictions.jsonl"
        verification_path = score_dir / "patch_jobs_selection.json"
        if not any(path.exists() for path in (results_path, summary_path, preds_path, predictions_jsonl_path, verification_path)):
            continue
        prediction_instance_ids = (
            _prediction_instance_ids_from_json(preds_path)
            if preds_path.exists()
            else _prediction_instance_ids_from_jsonl(predictions_jsonl_path)
        )
        summary_score = _score_from_summary(summary_path, root) if summary_path.exists() else None
        results_score = _score_from_results(results_path, root) if results_path.exists() else None
        partial_score = (
            _score_from_partial_reports(results_path, root, allowed_instance_ids=prediction_instance_ids)
            if results_path.exists()
            else None
        )
        score = _freshest_rolling_score(summary_score, results_score, partial_score)
        if score is None:
            score = {
                "status": "pending",
                "resolved_count": None,
                "task_count": None,
                "failed_count": None,
                "resolve_rate": None,
                "score_source": "",
            }
        prediction_count = (
            _prediction_count_from_json(preds_path)
            if preds_path.exists()
            else _prediction_count_from_jsonl(predictions_jsonl_path)
        )
        prediction_mtime = max(
            [path.stat().st_mtime for path in (preds_path, predictions_jsonl_path) if path.exists()],
            default=0.0,
        )
        verification_summary = _verification_summary_from_json(verification_path, root) if verification_path.exists() else None
        if prediction_count == 0:
            freshness_mtime = max(
                prediction_mtime,
                verification_path.stat().st_mtime if verification_path.exists() else 0.0,
                results_path.stat().st_mtime if results_path.exists() else 0.0,
            )
            score = {
                "status": "no_predictions",
                "resolved_count": 0,
                "task_count": 0,
                "failed_count": 0,
                "resolve_rate": None,
                "score_source": "no_current_predictions",
                "remaining_prediction_count": 0,
                "score_mtime": freshness_mtime,
                "score_updated_at": datetime.fromtimestamp(freshness_mtime, tz=timezone.utc).isoformat()
                if freshness_mtime
                else "",
            }
        score_mode = "raw_completed_success" if "raw" in score_dir.name else "verified_subset"
        score.update(
            {
                "benchmark": "swe_bench_live",
                "score_kind": f"discovered_{score_dir.name}",
                "run_id": score_dir.name,
                "score_mode": score_mode,
                "results_json": str(results_path.relative_to(root)) if results_path.exists() else "",
                "summary_json": str(summary_path.relative_to(root)) if summary_path.exists() else "",
                "predictions_patch_json": str(preds_path.relative_to(root)) if preds_path.exists() else "",
                "predictions_jsonl": str(predictions_jsonl_path.relative_to(root)) if predictions_jsonl_path.exists() else "",
                "patch_job_verification_json": str(verification_path.relative_to(root)) if verification_path.exists() else "",
                "prediction_count": prediction_count,
                "prediction_mtime": prediction_mtime,
                "prediction_updated_at": datetime.fromtimestamp(prediction_mtime, tz=timezone.utc).isoformat()
                if prediction_mtime
                else "",
                "label": "Discovered rolling SWE-live score",
                "final_leaderboard_score": False,
                "trusted_leaderboard_evidence": False,
                "discovered_from_rolling_score_dir": True,
            }
        )
        if verification_summary is not None:
            score["verification"] = verification_summary
            score["verified_patch_count"] = verification_summary["verified_patch_count"]
            score["abstained_patch_count"] = verification_summary["abstained_patch_count"]
            score["failed_patch_count"] = verification_summary["failed_patch_count"]
            score["skipped_nonterminal_count"] = verification_summary["skipped_nonterminal_count"]
            score["skipped_pre_epoch_count"] = verification_summary["skipped_pre_epoch_count"]
            score["skipped_missing_count"] = verification_summary["skipped_missing_count"]
        scorecards[f"swe_bench_live:discovered:{score_dir.name}"] = score
    return scorecards


def _freshest_rolling_score(
    summary_score: dict[str, Any] | None,
    results_score: dict[str, Any] | None,
    partial_score: dict[str, Any] | None,
) -> dict[str, Any] | None:
    candidates = [score for score in (summary_score, results_score, partial_score) if score]
    if not candidates:
        return None
    return max(candidates, key=lambda score: float(score.get("score_mtime") or 0.0))


def _score_from_partial_reports(
    results_path: Path,
    root: Path,
    allowed_instance_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    report_dir = results_path.parent
    if not report_dir.exists():
        return None
    reports: list[tuple[Path, dict[str, Any]]] = []
    for report_path in sorted(report_dir.glob("*/report.json")):
        try:
            payload = _read_json_object(report_path)
        except (json.JSONDecodeError, ValueError):
            continue
        instance_id = _text(payload.get("instance_id"))
        if allowed_instance_ids is not None and instance_id not in allowed_instance_ids:
            continue
        if isinstance(payload.get("resolved"), bool):
            reports.append((report_path, payload))
    if not reports:
        return None
    resolved = sum(1 for _path, report in reports if report.get("resolved") is True)
    total = len(reports)
    passed_ids = [
        _text(report.get("instance_id"))
        for _path, report in reports
        if report.get("resolved") is True and _text(report.get("instance_id"))
    ]
    failed_ids = [
        _text(report.get("instance_id"))
        for _path, report in reports
        if report.get("resolved") is False and _text(report.get("instance_id"))
    ]
    score_mtime = max(path.stat().st_mtime for path, _report in reports)
    return {
        "status": "partial",
        "resolved_count": resolved,
        "task_count": total,
        "failed_count": len(failed_ids),
        "resolve_rate": resolved / total if total else 0.0,
        "score_source": "partial_report_json",
        "score_mtime": score_mtime,
        "score_updated_at": datetime.fromtimestamp(score_mtime, tz=timezone.utc).isoformat(),
        "results_json": str(results_path.relative_to(root)) if results_path.is_relative_to(root) else str(results_path),
        "partial": True,
        "passed_instance_ids": passed_ids[:50],
        "failed_instance_ids": failed_ids[:50],
        "remaining_prediction_count": None,
        "filtered_to_current_predictions": allowed_instance_ids is not None,
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
    active_pid = active_phase.get("pid")
    active_pid_alive = _pid_is_alive(active_pid)
    return {
        "path": str(path.relative_to(root)),
        "benchmark": _text(payload.get("benchmark")),
        "success": bool(payload.get("success")),
        "completed_at": _text(payload.get("completed_at")),
        "failed_phase": _text(payload.get("failed_phase")),
        "active_phase": {
            "name": _text(active_phase.get("name")),
            "pid": active_pid,
            "pid_alive": active_pid_alive,
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
        queue_mtime = queue_path.stat().st_mtime
    except OSError:
        queue_mtime = 0.0
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
    latest_event_at = _text(recent_events[-1].get("at")) if recent_events else ""
    recent_activity = False
    if latest_event_at:
        try:
            parsed_latest = datetime.fromisoformat(latest_event_at.replace("Z", "+00:00"))
            if parsed_latest.tzinfo is None:
                parsed_latest = parsed_latest.replace(tzinfo=timezone.utc)
            recent_activity = (datetime.now(timezone.utc) - parsed_latest).total_seconds() <= 900
        except ValueError:
            recent_activity = False

    raw_active = state_counts.get("in_progress", 0)
    active_jobs = raw_active if recent_activity else 0
    completed = state_counts.get("completed", 0)
    safe_stop = state_counts.get("safe_stop", 0)
    failed = state_counts.get("failed", 0)
    terminal = completed + safe_stop + failed + state_counts.get("cancelled", 0) + state_counts.get("expired", 0)
    return {
        "benchmark": _text(harness.get("benchmark")),
        "queue_path": str(queue_path.relative_to(root)) if queue_path.is_relative_to(root) else str(queue_path),
        "queue_mtime": queue_mtime,
        "total_jobs": len(jobs),
        "terminal_jobs": terminal,
        "active_jobs": active_jobs,
        "stale_active_jobs": raw_active if raw_active and not recent_activity else 0,
        "queued_jobs": state_counts.get("queued", 0),
        "completed_jobs": completed,
        "safe_stop_jobs": safe_stop,
        "failed_jobs": failed,
        "latest_event_at": latest_event_at,
        "recent_activity": recent_activity,
        "state_counts": dict(sorted(state_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "artifact_failure_mode_counts": dict(sorted(artifact_failure_mode_counts.items())),
        "recent_artifact_failures": recent_artifact_failures,
        "progress_rate": (terminal / len(jobs)) if jobs else 0.0,
        "recent_events": recent_events,
    }


def _queue_snapshot_rank(snapshot: dict[str, Any]) -> tuple[int, float, int, int, int]:
    return (
        1 if int(snapshot.get("active_jobs", 0) or 0) > 0 else 0,
        float(snapshot.get("queue_mtime", 0.0) or 0.0),
        int(snapshot.get("terminal_jobs", 0) or 0),
        int(snapshot.get("completed_jobs", 0) or 0),
        int(snapshot.get("total_jobs", 0) or 0),
    )


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


def _build_neural_controller_shadow_status(root: Path) -> dict[str, Any]:
    path = root / DEFAULT_NEURAL_CONTROLLER_SHADOW_METRICS
    relative = str(DEFAULT_NEURAL_CONTROLLER_SHADOW_METRICS)
    if not path.exists():
        return {
            "status": "missing",
            "path": relative,
            "summary": {},
            "promotion_readiness": {},
        }
    try:
        payload = _read_json_object(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "unreadable",
            "path": relative,
            "summary": {},
            "promotion_readiness": {},
        }
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    readiness = payload.get("promotion_readiness") if isinstance(payload.get("promotion_readiness"), dict) else {}
    return {
        "status": "available",
        "path": relative,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "summary": summary,
        "promotion_readiness": readiness,
        "manifest_breakdown": payload.get("manifest_breakdown", []),
        "shadow_compare_ready": bool(readiness.get("shadow_compare_ready", False)),
        "kernel_guarded_content_ready": bool(readiness.get("kernel_guarded_content_ready", False)),
        "content_authority_ready": bool(readiness.get("content_authority_ready", False)),
        "pure_content_authority_ready": bool(readiness.get("pure_content_authority_ready", False)),
        "primary_authority_ready": bool(readiness.get("primary_authority_ready", False)),
        "episodes_with_shadow": summary.get("episodes_with_shadow", 0),
        "ready_steps": summary.get("ready_steps", 0),
        "content_comparison_steps": summary.get("content_comparison_steps", 0),
        "error_rate": summary.get("error_rate", 0.0),
        "warning_rate": summary.get("warning_rate", 0.0),
        "verified_action_agreement_rate": summary.get("verified_action_agreement_rate", 0.0),
        "content_exact_agreement_rate": summary.get("content_exact_agreement_rate", 0.0),
        "unrepaired_content_exact_agreement_rate": summary.get(
            "unrepaired_content_exact_agreement_rate",
            0.0,
        ),
        "command_copy_target_repaired_rate": summary.get("command_copy_target_repaired_rate", 0.0),
    }


def _build_neural_controller_runtime_contract_status(root: Path) -> dict[str, Any]:
    path = root / DEFAULT_NEURAL_CONTROLLER_RUNTIME_CONTRACT_METRICS
    relative = str(DEFAULT_NEURAL_CONTROLLER_RUNTIME_CONTRACT_METRICS)
    if not path.exists():
        return {
            "status": "missing",
            "path": relative,
            "summary": {},
        }
    try:
        payload = _read_json_object(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "unreadable",
            "path": relative,
            "summary": {},
        }
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "status": "available",
        "path": relative,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "summary": summary,
        "shadow_steps": summary.get("shadow_steps", 0),
        "runtime_contract_steps": summary.get("runtime_contract_steps", 0),
        "runtime_contract_task_count": summary.get("runtime_contract_task_count", 0),
        "runtime_contract_task_ids": summary.get("runtime_contract_task_ids", []),
        "runtime_contract_success_steps": summary.get("runtime_contract_success_steps", 0),
        "runtime_contract_coverage_rate": summary.get("runtime_contract_coverage_rate", 0.0),
        "runtime_contract_success_rate": summary.get("runtime_contract_success_rate", 0.0),
        "selector_signal_ready": bool(summary.get("selector_signal_ready", False)),
        "runtime_artifact_failure_mode_counts": summary.get("runtime_artifact_failure_mode_counts", {}),
        "rowwise_selector_source_counts": summary.get("rowwise_selector_source_counts", {}),
        "rowwise_selector_policy_counts": summary.get("rowwise_selector_policy_counts", {}),
    }


def _build_neural_controller_selector_activation_status(root: Path) -> dict[str, Any]:
    path = root / DEFAULT_NEURAL_CONTROLLER_SELECTOR_ACTIVATION_GATE
    relative = str(DEFAULT_NEURAL_CONTROLLER_SELECTOR_ACTIVATION_GATE)
    if not path.exists():
        return {
            "status": "missing",
            "path": relative,
        }
    try:
        payload = _read_json_object(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "unreadable",
            "path": relative,
        }
    return {
        "status": "available",
        "path": relative,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "guarded_selector_activation_ready": bool(payload.get("guarded_selector_activation_ready", False)),
        "production_guarded_selector_activation_ready": bool(
            payload.get("production_guarded_selector_activation_ready", False)
        ),
        "primary_authority_ready": bool(payload.get("primary_authority_ready", False)),
        "recommended_runtime_mode": str(payload.get("recommended_runtime_mode", "")).strip(),
        "selector_policy": str(payload.get("selector_policy", "")).strip(),
        "runtime_contract_steps": int(payload.get("runtime_contract_steps", 0) or 0),
        "runtime_contract_task_count": int(payload.get("runtime_contract_task_count", 0) or 0),
        "blockers": list(payload.get("blockers", []) or []),
        "production_blockers": list(payload.get("production_blockers", []) or []),
    }


def _refresh_neural_controller_shadow_metrics(root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1].resolve()
    use_runtime_config = root.resolve() == repo_root
    config = KernelConfig() if use_runtime_config else None
    episodes_root = Path(config.trajectories_root) if config is not None else root / "trajectories"
    if not episodes_root.is_absolute():
        episodes_root = root / episodes_root
    output_path = (
        Path(config.neural_controller_shadow_metrics_path)
        if config is not None
        else root / DEFAULT_NEURAL_CONTROLLER_SHADOW_METRICS
    )
    if not output_path.is_absolute():
        output_path = root / output_path
    documents = iter_episode_documents(episodes_root, config=config)
    reports_dir = Path(config.run_reports_dir) if config is not None else root / "trajectories" / "reports"
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir
    documents.extend(_iter_neural_controller_report_documents(reports_dir))
    summary = summarize_neural_controller_shadow_documents(documents)
    report = {
        "report_kind": "neural_controller_shadow_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episodes_root": str(episodes_root),
        "summary": summary,
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
    }
    _write_json(output_path, report)
    return report


def _refresh_neural_controller_runtime_contract_metrics(root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1].resolve()
    use_runtime_config = root.resolve() == repo_root
    config = KernelConfig() if use_runtime_config else None
    episodes_root = Path(config.trajectories_root) if config is not None else root / "trajectories"
    if not episodes_root.is_absolute():
        episodes_root = root / episodes_root
    output_path = root / DEFAULT_NEURAL_CONTROLLER_RUNTIME_CONTRACT_METRICS
    documents = iter_episode_documents(episodes_root, config=config)
    reports_dir = Path(config.run_reports_dir) if config is not None else root / "trajectories" / "reports"
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir
    documents.extend(_iter_neural_controller_report_documents(reports_dir))
    report = {
        "report_kind": "neural_controller_runtime_contract_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episodes_root": str(episodes_root),
        "summary": summarize_runtime_contract_metrics(documents),
    }
    _write_json(output_path, report)
    return report


def _iter_neural_controller_report_documents(reports_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    if not reports_dir.exists():
        return documents
    for path in sorted(reports_dir.glob("*.json")):
        try:
            payload = _read_json_object(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("report_kind") == "neural_controller_shadow_dataset_eval" and isinstance(
            payload.get("documents"),
            list,
        ):
            documents.extend(item for item in payload["documents"] if isinstance(item, dict))
            continue
        if payload.get("policy_trace"):
            documents.append(payload)
    return documents


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
    now = datetime.now(timezone.utc)

    def fresh_active_phase(run: dict[str, Any]) -> bool:
        phase = run.get("active_phase", {}) if isinstance(run.get("active_phase"), dict) else {}
        if phase.get("pid_alive") is False:
            return False
        heartbeat = _text(phase.get("heartbeat_at"))
        if not heartbeat:
            return False
        try:
            parsed = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (now - parsed).total_seconds() <= 900

    candidates = [
        run
        for run in harness_runs
        if run.get("benchmark") == benchmark
        and not run.get("success")
        and not run.get("completed_at")
        and isinstance(run.get("active_phase"), dict)
        and run.get("active_phase")
        and fresh_active_phase(run)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: str(item.get("active_phase", {}).get("heartbeat_at") or item.get("path") or ""),
        reverse=True,
    )[0]


def _queue_activity_run(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if not snapshot:
        return None
    active_jobs = int(snapshot.get("active_jobs", 0) or 0)
    recent_events = [item for item in snapshot.get("recent_events", []) if isinstance(item, dict)]
    latest_event = sorted(recent_events, key=lambda item: str(item.get("at", "")))[-1] if recent_events else {}
    if active_jobs <= 0 and not latest_event:
        return None
    benchmark = _text(snapshot.get("benchmark")) or "queue"
    event_name = _text(latest_event.get("event")) or "queue_activity"
    task_id = _text(latest_event.get("task_id")) or "delegated_job"
    heartbeat = _text(latest_event.get("at")) or datetime.now(timezone.utc).isoformat()
    return {
        "benchmark": benchmark,
        "path": snapshot.get("queue_path", ""),
        "active_phase": {
            "name": f"Queue activity: {event_name}",
            "pid": "queue",
            "elapsed_seconds": None,
            "heartbeat_at": heartbeat,
            "active_jobs": active_jobs,
            "current_task_id": task_id,
        },
        "phase_progress": {
            "status": "running" if active_jobs > 0 else "recent_activity",
            "processed_items": snapshot.get("terminal_jobs", 0),
            "total_items": snapshot.get("total_jobs", 0),
            "selected_tasks": snapshot.get("total_jobs", 0),
            "updated_at": heartbeat,
            "current_instance_id": task_id,
        },
        "completed_phase_count": 0,
        "completed_phases": [],
        "queue_derived": True,
    }


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
    queue_snapshots: dict[str, dict[str, Any]] = {}
    for snapshot in (
        _build_queue_snapshot_from_harness(harness, root)
        for harness in harness_specs
    ):
        if snapshot is None or not snapshot.get("benchmark"):
            continue
        benchmark = str(snapshot["benchmark"])
        previous = queue_snapshots.get(benchmark)
        if previous is None or _queue_snapshot_rank(snapshot) > _queue_snapshot_rank(previous):
            queue_snapshots[benchmark] = snapshot
    for benchmark, snapshot in queue_snapshots.items():
        if benchmark not in active_runs:
            queue_run = _queue_activity_run(snapshot)
            if queue_run is not None:
                active_runs[benchmark] = queue_run
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "harness_runs": harness_runs,
        "active_runs_by_benchmark": active_runs,
        "queue_snapshots_by_benchmark": queue_snapshots,
        "official_scores_by_benchmark": _build_official_scorecards(root, run_specs),
        "rolling_scores": _build_rolling_scorecards(root, harness_specs),
        "neural_controller_shadow": _build_neural_controller_shadow_status(root),
        "neural_controller_runtime_contract": _build_neural_controller_runtime_contract_status(root),
        "neural_controller_selector_activation": _build_neural_controller_selector_activation_status(root),
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
        "neural_controller_shadow": _build_neural_controller_shadow_status(root),
        "neural_controller_runtime_contract": _build_neural_controller_runtime_contract_status(root),
        "neural_controller_selector_activation": _build_neural_controller_selector_activation_status(root),
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
    parser.add_argument("--refresh-neural-controller-shadow-metrics", action="store_true")
    parser.add_argument("--refresh-neural-controller-runtime-contract-metrics", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    live_output = Path(args.live_output)
    if not live_output.is_absolute():
        live_output = root / live_output
    if args.refresh_neural_controller_shadow_metrics:
        report = _refresh_neural_controller_shadow_metrics(root)
        summary = report["summary"]
        print(
            "refreshed_neural_controller_shadow_metrics "
            f"episodes={summary.get('episode_count', 0)} "
            f"shadow_episodes={summary.get('episodes_with_shadow', 0)}",
            flush=True,
        )
    if args.refresh_neural_controller_runtime_contract_metrics:
        report = _refresh_neural_controller_runtime_contract_metrics(root)
        summary = report["summary"]
        print(
            "refreshed_neural_controller_runtime_contract_metrics "
            f"shadow_steps={summary.get('shadow_steps', 0)} "
            f"runtime_contract_steps={summary.get('runtime_contract_steps', 0)} "
            f"runtime_contract_tasks={summary.get('runtime_contract_task_count', 0)} "
            f"selector_signal_ready={str(summary.get('selector_signal_ready', False)).lower()}",
            flush=True,
        )
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
