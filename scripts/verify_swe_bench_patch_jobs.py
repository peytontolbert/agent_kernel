from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def verify_swe_patch_jobs(
    *,
    queue_json: Path,
    queue_manifest: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    queue_payload = _read_json(queue_json)
    manifest_payload = _read_json(queue_manifest)
    jobs = queue_payload.get("jobs", [])
    tasks = manifest_payload.get("tasks", [])
    if not isinstance(jobs, list):
        raise ValueError("queue JSON jobs must be a list")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("queue manifest tasks must be a non-empty list")
    latest_by_task_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        task_id = str(job.get("task_id", "")).strip()
        if task_id:
            latest_by_task_id[task_id] = job
    verified: list[dict[str, str]] = []
    failed_jobs: list[dict[str, str]] = []
    failures: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            failures.append("queue manifest contains non-object task")
            continue
        task_id = str(task.get("task_id", "")).strip()
        workspace_subdir = str(task.get("workspace_subdir", "")).strip()
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        instance_id = str(metadata.get("swe_instance_id", "")).strip()
        if not task_id:
            failures.append("queue manifest task missing task_id")
            continue
        job = latest_by_task_id.get(task_id)
        if not job:
            failures.append(f"{task_id} has no queued job")
            failed_jobs.append(
                {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "state": "",
                    "outcome": "",
                    "reason": "missing_job",
                }
            )
            continue
        state = str(job.get("state", "")).strip()
        outcome = str(job.get("outcome", "")).strip()
        if state != "completed" or outcome != "success":
            failures.append(f"{task_id} state={state or '-'} outcome={outcome or '-'}")
            failed_jobs.append(
                {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "job_id": str(job.get("job_id", "")).strip(),
                    "state": state,
                    "outcome": outcome,
                    "reason": "job_not_successful",
                }
            )
            continue
        patch_path = workspace_root / workspace_subdir / "patch.diff"
        if not patch_path.exists() or patch_path.stat().st_size <= 0:
            failures.append(f"{task_id} missing patch.diff at {patch_path}")
            failed_jobs.append(
                {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "job_id": str(job.get("job_id", "")).strip(),
                    "state": state,
                    "outcome": outcome,
                    "reason": "missing_patch",
                }
            )
            continue
        verified.append(
            {
                "task_id": task_id,
                "instance_id": instance_id,
                "job_id": str(job.get("job_id", "")).strip(),
                "patch_path": str(patch_path),
            }
        )
    return {
        "report_kind": "swe_bench_patch_job_verification",
        "created_at": datetime.now(UTC).isoformat(),
        "queue_json": str(queue_json),
        "queue_manifest": str(queue_manifest),
        "workspace_root": str(workspace_root),
        "task_count": len(tasks),
        "verified_patch_count": len(verified),
        "failed_patch_count": len(failed_jobs),
        "success": not failures,
        "failures": failures,
        "retry_instance_ids": [
            item["instance_id"]
            for item in failed_jobs
            if item.get("instance_id")
        ],
        "successful_instance_ids": [
            item["instance_id"]
            for item in verified
            if item.get("instance_id")
        ],
        "failed_jobs": failed_jobs,
        "verified_patches": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = verify_swe_patch_jobs(
        queue_json=Path(args.queue_json),
        queue_manifest=Path(args.queue_manifest),
        workspace_root=Path(args.workspace_root),
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["success"]:
        raise SystemExit("SWE patch job verification failed: " + "; ".join(result["failures"]))
    print(
        f"verified_patch_count={result['verified_patch_count']} "
        f"task_count={result['task_count']} "
        f"output_json={output_path}"
    )


if __name__ == "__main__":
    main()
