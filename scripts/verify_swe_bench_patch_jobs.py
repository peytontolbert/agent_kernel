from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
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


def verify_swe_patch_jobs(
    *,
    queue_json: Path,
    queue_manifest: Path,
    workspace_root: Path,
    allow_nonterminal: bool = False,
    include_terminal_abstentions: bool = True,
    semantic_failures_as_abstentions: bool = True,
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
    abstained: list[dict[str, str]] = []
    failed_jobs: list[dict[str, str]] = []
    skipped_nonterminal: list[dict[str, str]] = []
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
        semantic_result = _verify_semantic_artifact(task, workspace_root=workspace_root)
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
        "abstained_patch_count": len(abstained),
        "failed_patch_count": len(failed_jobs),
        "skipped_nonterminal_count": len(skipped_nonterminal),
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
        "verified_patches": verified,
    }


def _verify_semantic_artifact(task: dict[str, Any], *, workspace_root: Path) -> dict[str, Any]:
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
    verification = Verifier().verify(
        runtime_task,
        workspace_root / workspace_subdir,
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
    args = parser.parse_args()

    result = verify_swe_patch_jobs(
        queue_json=Path(args.queue_json),
        queue_manifest=Path(args.queue_manifest),
        workspace_root=Path(args.workspace_root),
        allow_nonterminal=bool(args.allow_nonterminal),
        include_terminal_abstentions=not bool(args.completed_only),
        semantic_failures_as_abstentions=not bool(args.fail_on_semantic_artifact_failure),
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
