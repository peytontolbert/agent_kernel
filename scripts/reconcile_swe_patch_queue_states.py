#!/usr/bin/env python3
"""Reconcile SWE patch queue success states with official-scoring semantics."""

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


def _retire_path(path: Path, suffix: str) -> str:
    if not path.exists():
        return ""
    target = path.with_name(f"{path.name}.{suffix}")
    counter = 2
    while target.exists():
        target = path.with_name(f"{path.name}.{suffix}{counter}")
        counter += 1
    path.replace(target)
    return str(target)


def _resolve_stored_path(raw_path: str, *, queue_json: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute() or path.exists():
        return path
    return queue_json.parent / path


def _is_swe_patch_report(payload: dict[str, object]) -> bool:
    metadata = payload.get("task_metadata")
    if not isinstance(metadata, dict):
        contract = payload.get("task_contract")
        metadata = contract.get("metadata") if isinstance(contract, dict) else {}
    if not isinstance(metadata, dict):
        return False
    verifier = metadata.get("semantic_verifier")
    if not isinstance(verifier, dict):
        return False
    return str(verifier.get("kind", "")).strip() == "swe_patch_apply_check"


def _officially_passed(payload: dict[str, object]) -> bool:
    outcome = str(payload.get("outcome", "")).strip()
    if outcome == "official_passed":
        return True
    reasons = payload.get("outcome_reasons", [])
    return isinstance(reasons, list) and "official_passed" in {str(reason).strip() for reason in reasons}


def _report_patch_path(payload: dict[str, object]) -> Path | None:
    workspace = str(payload.get("workspace", "")).strip()
    if not workspace:
        return None
    metadata = payload.get("task_metadata")
    if not isinstance(metadata, dict):
        contract = payload.get("task_contract")
        metadata = contract.get("metadata") if isinstance(contract, dict) else {}
    verifier = metadata.get("semantic_verifier") if isinstance(metadata, dict) else {}
    patch_path = "patch.diff"
    if isinstance(verifier, dict):
        patch_path = str(verifier.get("patch_path", "patch.diff")).strip() or "patch.diff"
    return Path(workspace) / patch_path


def _semantic_abstentions(selection_json: Path | None) -> dict[str, dict[str, object]]:
    if selection_json is None:
        return {}
    payload = _load_json(selection_json)
    abstained = payload.get("abstained_jobs", [])
    if not isinstance(abstained, list):
        return {}
    records: dict[str, dict[str, object]] = {}
    for item in abstained:
        if not isinstance(item, dict):
            continue
        if str(item.get("reason", "")).strip() != "semantic_artifact_failure":
            continue
        task_id = str(item.get("task_id", "")).strip()
        if task_id:
            records[task_id] = dict(item)
    return records


def reconcile_queue(queue_json: Path, *, dry_run: bool, selection_json: Path | None = None) -> list[str]:
    changed: list[str] = []
    semantic_abstentions = _semantic_abstentions(selection_json)
    queue = DelegatedJobQueue(queue_json)
    with queue._locked_jobs() as jobs:  # The script is a maintenance tool; use the queue's normal lock/export path.
        for job in jobs:
            if (
                job.state == "queued"
                and "missing SWE patch artifact" in str(job.last_error)
                and _resolve_stored_path(job.checkpoint_path, queue_json=queue_json).exists()
            ):
                changed.append(job.task_id)
                if dry_run:
                    continue
                retired_checkpoint = _retire_path(
                    _resolve_stored_path(job.checkpoint_path, queue_json=queue_json),
                    "stale_missing_patch",
                )
                job.history.append(
                    {
                        "event": "checkpoint_reconciled",
                        "state": job.state,
                        "recorded_at": _utcnow(),
                        "detail": "queued missing-patch retry had stale checkpoint retired",
                        "retired_checkpoint_path": retired_checkpoint,
                    }
                )
                continue
            if job.state != "completed" or job.outcome not in {"success", "semantic_unverified", "artifact_ready"}:
                continue
            report_path = _resolve_stored_path(job.report_path, queue_json=queue_json)
            report = _load_json(report_path)
            if not report or not _is_swe_patch_report(report) or _officially_passed(report):
                continue
            patch_path = _report_patch_path(report)
            if patch_path is not None and not patch_path.exists():
                changed.append(job.task_id)
                if dry_run:
                    continue
                job.state = "queued"
                job.outcome = ""
                job.outcome_reasons = []
                job.started_at = ""
                job.finished_at = ""
                job.last_error = "requeued missing SWE patch artifact after semantic_unverified completion"
                retired_checkpoint = _retire_path(
                    _resolve_stored_path(job.checkpoint_path, queue_json=queue_json),
                    "stale_missing_patch",
                )
                job.history.append(
                    {
                        "event": "outcome_reconciled",
                        "state": job.state,
                        "recorded_at": _utcnow(),
                        "detail": "semantic_unverified SWE patch completion had no patch.diff and was requeued",
                        "missing_patch_path": str(patch_path),
                        "retired_checkpoint_path": retired_checkpoint,
                    }
                )
                report["outcome"] = "requeued_missing_patch"
                report["success"] = False
                report["termination_reason"] = "requeued_missing_patch"
                report["outcome_reasons"] = [
                    "missing_patch",
                    "requeued_for_artifact_materialization",
                ]
                _write_json(report_path, report)
                continue
            semantic_failure = semantic_abstentions.get(job.task_id)
            if semantic_failure is not None:
                changed.append(job.task_id)
                if dry_run:
                    continue
                job.state = "queued"
                job.outcome = "official_failed"
                job.outcome_reasons = [
                    "semantic_artifact_failure",
                    "requeued_for_patch_repair",
                ]
                job.started_at = ""
                job.finished_at = ""
                job.last_error = "; ".join(
                    str(reason).strip()
                    for reason in semantic_failure.get("verification_reasons", [])
                    if str(reason).strip()
                )[:1000]
                retired_checkpoint = _retire_path(
                    _resolve_stored_path(job.checkpoint_path, queue_json=queue_json),
                    "stale_semantic_artifact_failure",
                )
                job.history.append(
                    {
                        "event": "outcome_reconciled",
                        "state": job.state,
                        "recorded_at": _utcnow(),
                        "detail": "semantic-artifact failure requeued with rejected patch memory",
                        "retired_checkpoint_path": retired_checkpoint,
                        "patch_path": str(patch_path) if patch_path is not None else "",
                        "verification_reasons": list(semantic_failure.get("verification_reasons", []) or []),
                    }
                )
                report["outcome"] = "semantic_artifact_failed"
                report["success"] = False
                report["termination_reason"] = "semantic_artifact_failed"
                report["outcome_reasons"] = list(job.outcome_reasons)
                uncertainties = report.get("uncertainties", [])
                if not isinstance(uncertainties, list):
                    uncertainties = []
                message = "semantic artifact verifier rejected patch; requeued for autonomous repair"
                if message not in uncertainties:
                    uncertainties.append(message)
                report["uncertainties"] = uncertainties
                _write_json(report_path, report)
                continue
            if job.outcome != "success":
                continue
            changed.append(job.task_id)
            if dry_run:
                continue
            job.outcome = "semantic_unverified"
            job.outcome_reasons = [
                "artifact_ready",
                "semantic_unverified",
                "official_scoring_required",
                "reconciled_from_artifact_success",
            ]
            job.history.append(
                {
                    "event": "outcome_reconciled",
                    "state": job.state,
                    "recorded_at": _utcnow(),
                    "detail": "artifact-only SWE patch success downgraded to semantic_unverified",
                }
            )
            report["outcome"] = "semantic_unverified"
            report["success"] = False
            report["termination_reason"] = "semantic_unverified"
            report["outcome_reasons"] = list(job.outcome_reasons)
            uncertainties = report.get("uncertainties", [])
            if not isinstance(uncertainties, list):
                uncertainties = []
            message = "patch.diff passed local artifact checks but still requires official benchmark scoring"
            if message not in uncertainties:
                uncertainties.append(message)
            report["uncertainties"] = uncertainties
            _write_json(report_path, report)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", required=True, type=Path)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changed = reconcile_queue(args.queue_json, dry_run=bool(args.dry_run), selection_json=args.selection_json)
    print(json.dumps({"queue_json": str(args.queue_json), "dry_run": bool(args.dry_run), "changed": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
