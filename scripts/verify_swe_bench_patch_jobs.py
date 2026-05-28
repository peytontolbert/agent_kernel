from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import hashlib
import json
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.schemas import CommandResult, TaskSpec
from agent_kernel.verifier import Verifier


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _read_json_object_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _job_attempt_id(job: dict[str, Any]) -> str:
    raw = job.get("attempt_count", "")
    if raw == "":
        return ""
    return str(raw)


def _patch_provenance(
    patch_path: Path,
    *,
    task_id: str,
    instance_id: str,
    job: dict[str, Any],
    report_hash: str,
    verified_at: str,
) -> dict[str, Any]:
    stat = patch_path.stat()
    return {
        "task_id": task_id,
        "instance_id": instance_id,
        "job_id": str(job.get("job_id", "")).strip(),
        "attempt_id": _job_attempt_id(job),
        "attempt_count": job.get("attempt_count", ""),
        "state": str(job.get("state", "")).strip(),
        "outcome": str(job.get("outcome", "")).strip(),
        "finished_at": str(job.get("finished_at", "")).strip(),
        "report_path": str(job.get("report_path", "")).strip(),
        "report_sha256": report_hash,
        "patch_path": str(patch_path),
        "patch_sha256": _sha256_file(patch_path),
        "patch_size": stat.st_size,
        "patch_mtime": stat.st_mtime,
        "patch_mtime_ns": stat.st_mtime_ns,
        "verified_at": verified_at,
    }


def _resolve_job_sidecar_path(path_text: str, *, queue_json: Path) -> Path:
    path_text = str(path_text).strip()
    if not path_text:
        return Path()
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    queue_relative = queue_json.parent / path
    if queue_relative.exists():
        return queue_relative
    return path


def _candidate_workspace_patch_paths(
    *,
    workspace_root: Path,
    workspace_subdir: str,
    job: dict[str, Any],
    queue_json: Path,
) -> list[Path]:
    candidates: list[Path] = []
    if workspace_subdir:
        candidates.append(workspace_root / workspace_subdir / "patch.diff")
    for sidecar_key in ("report_path", "checkpoint_path"):
        sidecar_path = _resolve_job_sidecar_path(str(job.get(sidecar_key, "")), queue_json=queue_json)
        sidecar = _read_json_object_if_exists(sidecar_path)
        workspace_text = str(sidecar.get("workspace", "")).strip()
        if workspace_text:
            workspace_path = Path(workspace_text)
            candidates.append(workspace_path / "patch.diff")
            if not workspace_path.is_absolute():
                candidates.append(Path.cwd() / workspace_path / "patch.diff")
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_patch_path(
    *,
    workspace_root: Path,
    workspace_subdir: str,
    job: dict[str, Any],
    queue_json: Path,
) -> Path:
    candidates = _candidate_workspace_patch_paths(
        workspace_root=workspace_root,
        workspace_subdir=workspace_subdir,
        job=job,
        queue_json=queue_json,
    )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return candidates[0] if candidates else workspace_root / workspace_subdir / "patch.diff"


def verify_swe_patch_jobs(
    *,
    queue_json: Path,
    queue_manifest: Path,
    workspace_root: Path,
    allow_nonterminal: bool = False,
    include_terminal_abstentions: bool = True,
    semantic_failures_as_abstentions: bool = True,
    skip_semantic_artifact_check: bool = False,
    allow_missing_jobs: bool = False,
    missing_patches_as_abstentions: bool = False,
    completed_after: str = "",
) -> dict[str, Any]:
    queue_payload = _read_json(queue_json)
    manifest_payload = _read_json(queue_manifest)
    verified_at = datetime.now(UTC).isoformat()
    queue_epoch = {
        "queue_json": str(queue_json),
        "queue_json_sha256": _sha256_text(queue_json.read_text(encoding="utf-8")),
        "queue_manifest": str(queue_manifest),
        "queue_manifest_sha256": _sha256_text(queue_manifest.read_text(encoding="utf-8")),
        "workspace_root": str(workspace_root),
    }
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
    abstained: list[dict[str, str]] = []
    failed_jobs: list[dict[str, str]] = []
    skipped_nonterminal: list[dict[str, str]] = []
    skipped_pre_epoch: list[dict[str, str]] = []
    skipped_missing: list[dict[str, str]] = []
    failures: list[str] = []
    terminal_abstain_states = {"cancelled", "expired", "failed", "safe_stop"}
    nonterminal_states = {"queued", "in_progress"}
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
            if allow_missing_jobs:
                skipped_missing.append(
                    {
                        "task_id": task_id,
                        "instance_id": instance_id,
                        "state": "",
                        "outcome": "",
                        "reason": "missing_job_skipped",
                    }
                )
                continue
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
        finished_at = str(job.get("finished_at", "")).strip()
        report_path = _resolve_job_sidecar_path(str(job.get("report_path", "")), queue_json=queue_json)
        report_hash = _sha256_file(report_path) if report_path.exists() and report_path.is_file() else ""
        artifact_ready_outcomes = {"semantic_unverified", "artifact_ready", "official_passed"}
        if state == "completed" and outcome == "success":
            abstained.append(
                {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "job_id": str(job.get("job_id", "")).strip(),
                    "state": state,
                    "outcome": outcome,
                    "reason": "stale_artifact_success_requires_reconciliation",
                }
            )
            continue
        if state != "completed" or outcome not in artifact_ready_outcomes:
            if state in nonterminal_states and allow_nonterminal:
                skipped_nonterminal.append(
                    {
                        "task_id": task_id,
                        "instance_id": instance_id,
                        "job_id": str(job.get("job_id", "")).strip(),
                        "state": state,
                        "outcome": outcome,
                        "reason": "nonterminal_skipped",
                    }
                )
                continue
            if state in terminal_abstain_states and include_terminal_abstentions:
                abstained.append(
                    {
                        "task_id": task_id,
                        "instance_id": instance_id,
                        "job_id": str(job.get("job_id", "")).strip(),
                        "state": state,
                        "outcome": outcome,
                        "reason": "terminal_abstention",
                    }
                )
                continue
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
        if completed_after and (not finished_at or finished_at <= completed_after):
            skipped_pre_epoch.append(
                {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "job_id": str(job.get("job_id", "")).strip(),
                    "state": state,
                    "outcome": outcome,
                    "finished_at": finished_at,
                    "reason": "completed_before_epoch_skipped",
                }
            )
            continue
        patch_path = _resolve_patch_path(
            workspace_root=workspace_root,
            workspace_subdir=workspace_subdir,
            job=job,
            queue_json=queue_json,
        )
        if not patch_path.exists() or patch_path.stat().st_size <= 0:
            item = {
                "task_id": task_id,
                "instance_id": instance_id,
                "job_id": str(job.get("job_id", "")).strip(),
                "state": state,
                "outcome": outcome,
                "reason": "missing_patch",
            }
            if missing_patches_as_abstentions:
                abstained.append(item)
            else:
                failures.append(f"{task_id} missing patch.diff at {patch_path}")
                failed_jobs.append(item)
            continue
        patch_provenance = _patch_provenance(
            patch_path,
            task_id=task_id,
            instance_id=instance_id,
            job=job,
            report_hash=report_hash,
            verified_at=verified_at,
        )
        if not skip_semantic_artifact_check:
            semantic_result = _verify_semantic_artifact(
                task,
                workspace_root=workspace_root,
                workspace_path=patch_path.parent,
            )
            if semantic_result:
                semantic_reasons = [
                    str(reason).strip()
                    for reason in semantic_result.get("reasons", [])
                    if str(reason).strip() and str(reason).strip() != "verification passed"
                ]
                item = {
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "job_id": str(job.get("job_id", "")).strip(),
                    "state": state,
                    "outcome": outcome,
                    "reason": "semantic_artifact_failure",
                    "verification_reasons": semantic_reasons[:8],
                    **patch_provenance,
                }
                if semantic_failures_as_abstentions:
                    abstained.append(item)
                else:
                    failures.append(
                        f"{task_id} semantic artifact verification failed: "
                        + "; ".join(semantic_reasons[:3])
                    )
                    failed_jobs.append(item)
                continue
        verified.append(
            {
                **patch_provenance,
            }
        )
    return {
        "report_kind": "swe_bench_patch_job_verification",
        "created_at": verified_at,
        "verified_at": verified_at,
        "queue_epoch": queue_epoch,
        "queue_json_sha256": queue_epoch["queue_json_sha256"],
        "queue_manifest_sha256": queue_epoch["queue_manifest_sha256"],
        "queue_json": str(queue_json),
        "queue_manifest": str(queue_manifest),
        "workspace_root": str(workspace_root),
        "semantic_artifact_check": not skip_semantic_artifact_check,
        "task_count": len(tasks),
        "verified_patch_count": len(verified),
        "abstained_patch_count": len(abstained),
        "failed_patch_count": len(failed_jobs),
        "skipped_nonterminal_count": len(skipped_nonterminal),
        "skipped_pre_epoch_count": len(skipped_pre_epoch),
        "skipped_missing_count": len(skipped_missing),
        "success": not failures,
        "failures": failures,
        "retry_instance_ids": [
            item["instance_id"]
            for item in [*abstained, *failed_jobs]
            if item.get("instance_id")
        ],
        "successful_instance_ids": [
            item["instance_id"]
            for item in verified
            if item.get("instance_id")
        ],
        "abstained_instance_ids": [
            item["instance_id"]
            for item in abstained
            if item.get("instance_id")
        ],
        "failed_jobs": failed_jobs,
        "abstained_jobs": abstained,
        "skipped_nonterminal_jobs": skipped_nonterminal,
        "skipped_pre_epoch_jobs": skipped_pre_epoch,
        "skipped_missing_jobs": skipped_missing,
        "verified_patches": verified,
    }


def _verify_semantic_artifact(
    task: dict[str, Any],
    *,
    workspace_root: Path,
    workspace_path: Path | None = None,
) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    semantic_verifier = metadata.get("semantic_verifier", {})
    if not isinstance(semantic_verifier, dict) or not semantic_verifier:
        return {}
    workspace_subdir = str(task.get("workspace_subdir", "")).strip()
    if not workspace_subdir:
        return {"passed": False, "reasons": ["queue manifest task missing workspace_subdir"]}
    runtime_task = TaskSpec(
        task_id=str(task.get("task_id", "")).strip() or "swe_patch_task",
        prompt=str(task.get("prompt", "")).strip() or "verify patch artifact",
        workspace_subdir=workspace_subdir,
        setup_commands=[str(value) for value in task.get("setup_commands", []) if str(value).strip()]
        if isinstance(task.get("setup_commands", []), list)
        else [],
        success_command=str(task.get("success_command", "")).strip(),
        suggested_commands=[str(value) for value in task.get("suggested_commands", []) if str(value).strip()]
        if isinstance(task.get("suggested_commands", []), list)
        else [],
        expected_files=[str(value) for value in task.get("expected_files", ["patch.diff"]) if str(value).strip()]
        if isinstance(task.get("expected_files", ["patch.diff"]), list)
        else ["patch.diff"],
        max_steps=int(task.get("max_steps", 1) or 1),
        metadata=dict(metadata),
    )
    verification_workspace = workspace_path if workspace_path is not None else workspace_root / workspace_subdir
    verification = Verifier().verify(
        runtime_task,
        verification_workspace,
        CommandResult(command="semantic_artifact_check", exit_code=0, stdout="", stderr=""),
    )
    if verification.passed:
        return {}
    return {
        "passed": False,
        "reasons": list(verification.reasons),
        "failure_codes": list(verification.failure_codes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-nonterminal", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--fail-on-semantic-artifact-failure", action="store_true")
    parser.add_argument("--skip-semantic-artifact-check", action="store_true")
    parser.add_argument("--allow-missing-jobs", action="store_true")
    parser.add_argument("--missing-patches-as-abstentions", action="store_true")
    parser.add_argument("--completed-after", default="")
    args = parser.parse_args()

    result = verify_swe_patch_jobs(
        queue_json=Path(args.queue_json),
        queue_manifest=Path(args.queue_manifest),
        workspace_root=Path(args.workspace_root),
        allow_nonterminal=bool(args.allow_nonterminal),
        # completed-only means only completed successful patches are eligible
        # for scoring; terminal non-success jobs should still be excluded as
        # abstentions rather than crashing the rolling scorer.
        include_terminal_abstentions=True,
        semantic_failures_as_abstentions=not bool(args.fail_on_semantic_artifact_failure),
        skip_semantic_artifact_check=bool(args.skip_semantic_artifact_check),
        allow_missing_jobs=bool(args.allow_missing_jobs),
        missing_patches_as_abstentions=bool(args.missing_patches_as_abstentions),
        completed_after=str(args.completed_after).strip(),
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["success"]:
        raise SystemExit("SWE patch job verification failed: " + "; ".join(result["failures"]))
    print(
        f"verified_patch_count={result['verified_patch_count']} "
        f"abstained_patch_count={result['abstained_patch_count']} "
        f"task_count={result['task_count']} "
        f"output_json={output_path}"
    )


if __name__ == "__main__":
    main()
