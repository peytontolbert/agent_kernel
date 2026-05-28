#!/usr/bin/env python3
"""Apply official SWE-bench Live evaluator results back to a delegated queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_kernel.ops.job_queue import DelegatedJobQueue, _utcnow


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _instance_id(task_id: str) -> str:
    normalized = str(task_id).strip()
    return normalized.removeprefix("swe_patch_")


def _report_path(queue_json: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute() or path.exists():
        return path
    return queue_json.parent / path


def _official_failure_report_path(results_json: Path) -> Path:
    return results_json.parent.parent / "official_failure_retry_report.json"


def _official_feedback_by_instance(results_json: Path) -> dict[str, dict[str, object]]:
    report_path = _official_failure_report_path(results_json)
    report = _load_json(report_path)
    failed_jobs = report.get("failed_jobs", [])
    if not isinstance(failed_jobs, list):
        return {}
    feedback: dict[str, dict[str, object]] = {}
    for item in failed_jobs:
        if not isinstance(item, dict):
            continue
        instance_id = str(item.get("instance_id", "")).strip()
        if not instance_id:
            continue
        feedback[instance_id] = dict(item)
    return feedback


def _attach_official_feedback(job: object, feedback: dict[str, object]) -> None:
    if not feedback:
        return
    runtime_overrides = (
        dict(getattr(job, "runtime_overrides", {}) or {})
        if isinstance(getattr(job, "runtime_overrides", {}) or {}, dict)
        else {}
    )
    task_payload = (
        dict(runtime_overrides.get("task_payload", {}))
        if isinstance(runtime_overrides.get("task_payload", {}), dict)
        else {}
    )
    metadata = (
        dict(task_payload.get("metadata", {}))
        if isinstance(task_payload.get("metadata", {}), dict)
        else {}
    )
    previous = metadata.get("swe_official_feedback_history", [])
    history = [dict(item) for item in previous if isinstance(item, dict)] if isinstance(previous, list) else []
    metadata["swe_official_feedback"] = dict(feedback)
    history.append(dict(feedback))
    metadata["swe_official_feedback_history"] = history[-5:]
    metadata["swe_executable_edit_windows"] = ""
    metadata["swe_suggested_patch_commands"] = []
    task_payload["metadata"] = metadata
    task_payload["prompt"] = _prompt_with_compact_official_feedback(
        str(task_payload.get("prompt", "") or ""),
        feedback,
    )
    runtime_overrides["task_payload"] = task_payload
    job.runtime_overrides = runtime_overrides


def _prompt_with_compact_official_feedback(prompt: str, feedback: dict[str, object]) -> str:
    text = str(prompt or "")
    if not text.strip():
        return text
    sections = text.split("\n\n")
    compact_sections: list[str] = []
    skip_next = 0
    for section in sections:
        if skip_next:
            skip_next -= 1
            continue
        stripped = section.strip()
        if stripped == "Pass-to-pass tests:":
            skip_next = 1
            p2p = feedback.get("pass_to_pass_failures", [])
            p2p_list = [str(item).strip() for item in p2p if str(item).strip()] if isinstance(p2p, list) else []
            compact_sections.extend(
                [
                    "Pass-to-pass tests:",
                    (
                        "Omitted from retry prompt after official scoring. Preserve existing behavior; "
                        f"focus on listed official pass-to-pass regressions only. Regression count: "
                        f"{int(feedback.get('pass_to_pass_failure_count', len(p2p_list)) or 0)}."
                    ),
                ]
            )
            continue
        if stripped == "High-value executable edit windows:":
            # Official failures invalidate the previous static edit-window ranking. Keeping
            # this section can pin retries to the same non-causal source span.
            skip_next = 2
            compact_sections.extend(
                [
                    "Official retry orientation:",
                    (
                        "Ignore stale pre-official edit-window rankings. Re-orient from the "
                        "official failed test source, symbols referenced by that test, and the "
                        "prior rejected patch before choosing a production edit."
                    ),
                ]
            )
            continue
        compact_sections.append(section)
    marker = "Prior official evaluator feedback:"
    if marker not in "\n\n".join(compact_sections):
        failed = feedback.get("failed_tests", [])
        f2p = feedback.get("fail_to_pass_failures", [])
        p2p = feedback.get("pass_to_pass_failures", [])
        compact_sections.extend(
            [
                marker,
                "A previous patch for this instance was unresolved under the official evaluator. Treat this as verifier evidence; do not repeat the same patch blindly.",
                "Official failure mode:",
                str(feedback.get("official_failure_mode", "")).strip(),
                "Official repair directive:",
                str(feedback.get("official_repair_directive", "")).strip(),
                "Official failed tests:",
                json.dumps([str(item).strip() for item in failed if str(item).strip()][:30], sort_keys=True)
                if isinstance(failed, list)
                else "[]",
                "Official fail-to-pass failures to satisfy:",
                json.dumps([str(item).strip() for item in f2p if str(item).strip()][:30], sort_keys=True)
                if isinstance(f2p, list)
                else "[]",
                "Official pass-to-pass regressions introduced by the previous patch:",
                json.dumps([str(item).strip() for item in p2p if str(item).strip()][:30], sort_keys=True)
                if isinstance(p2p, list)
                else "[]",
            ]
        )
    return "\n\n".join(compact_sections)


def apply_results(queue_json: Path, results_json: Path, *, dry_run: bool) -> dict[str, list[str]]:
    results = _load_json(results_json)
    success_ids = {str(value).strip() for value in results.get("success_ids", []) if str(value).strip()}
    failed_ids = {
        str(value).strip()
        for key in ("failure_ids", "error_ids")
        for value in (results.get(key, []) if isinstance(results.get(key, []), list) else [])
        if str(value).strip()
    }
    changed: dict[str, list[str]] = {"official_passed": [], "official_failed": []}
    official_feedback = _official_feedback_by_instance(results_json)
    queue = DelegatedJobQueue(queue_json)
    with queue._locked_jobs() as jobs:  # Maintenance script; use the queue lock/export path.
        for job in jobs:
            instance_id = _instance_id(job.task_id)
            if instance_id not in success_ids and instance_id not in failed_ids:
                continue
            passed = instance_id in success_ids
            outcome = "official_passed" if passed else "official_failed"
            changed[outcome].append(job.task_id)
            if dry_run:
                continue
            job.state = "completed" if passed else "failed"
            job.outcome = outcome
            job.outcome_reasons = [
                outcome,
                "official_swe_live_evaluator",
                f"results_json:{results_json}",
            ]
            job.last_error = "" if passed else "official SWE-bench Live evaluator failed"
            job.finished_at = job.finished_at or _utcnow()
            if not passed:
                _attach_official_feedback(job, official_feedback.get(instance_id, {}))
            job.history.append(
                {
                    "event": "official_score_applied",
                    "state": job.state,
                    "recorded_at": _utcnow(),
                    "detail": f"{instance_id} outcome={outcome}",
                }
            )
            report_path = _report_path(queue_json, job.report_path)
            report = _load_json(report_path)
            if report:
                report["outcome"] = outcome
                report["success"] = passed
                report["termination_reason"] = outcome
                report["outcome_reasons"] = list(job.outcome_reasons)
                report["official_swe_live_result"] = {
                    "instance_id": instance_id,
                    "passed": passed,
                    "results_json": str(results_json),
                    "applied_at": _utcnow(),
                }
                _write_json(report_path, report)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", required=True, type=Path)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changed = apply_results(args.queue_json, args.results_json, dry_run=bool(args.dry_run))
    print(json.dumps({"queue_json": str(args.queue_json), "results_json": str(args.results_json), "changed": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
