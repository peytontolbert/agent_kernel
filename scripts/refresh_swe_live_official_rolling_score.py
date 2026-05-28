from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_swe_bench_live_submission import write_live_predictions_json
from collect_swe_bench_predictions import collect_swe_predictions

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_swe_bench_patch_jobs import verify_swe_patch_jobs


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clear_dir(path: Path, *, require_under: Path) -> int:
    path = path.resolve()
    require_under = require_under.resolve()
    if not path.is_relative_to(require_under):
        raise SystemExit(f"refusing to clear {path}; not under {require_under}")
    removed = 0
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return removed
    for item in path.iterdir():
        if item.is_dir():
            import shutil

            shutil.rmtree(item)
        else:
            item.unlink()
        removed += 1
    return removed


def refresh_official_rolling_score(
    *,
    queue_json: Path,
    queue_manifest: Path,
    prediction_task_manifest: Path,
    workspace_root: Path,
    output_root: Path,
    swe_bench_live_root: Path,
    python: str,
    workers: int,
    launch_evaluator: bool,
    skip_semantic_artifact_check: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    verification_json = output_root / "patch_jobs_selection.json"
    predictions_jsonl = output_root / "predictions.jsonl"
    preds_json = output_root / "preds.json"
    evaluation_results_dir = output_root / "evaluation_results"
    log_path = output_root / "official_score_stdout.log"

    verification = verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
        allow_nonterminal=True,
        include_terminal_abstentions=True,
        semantic_failures_as_abstentions=True,
        skip_semantic_artifact_check=skip_semantic_artifact_check,
        allow_missing_jobs=True,
        missing_patches_as_abstentions=True,
    )
    _write_json(verification_json, verification)
    if not verification.get("success"):
        raise SystemExit("SWE patch job verification failed: " + "; ".join(verification.get("failures", [])))

    removed_count = _clear_dir(evaluation_results_dir, require_under=output_root)
    if int(verification.get("verified_patch_count") or 0) <= 0:
        predictions_jsonl.write_text("", encoding="utf-8")
        _write_json(preds_json, {})
        _write_json(
            evaluation_results_dir / "results.json",
            {
                "submitted": 0,
                "submitted_ids": [],
                "empty_patch": 0,
                "empty_patch_ids": [],
                "success_ids": [],
                "failure_ids": [],
                "error_ids": [],
                "success": 0,
                "failure": 0,
                "error": 0,
            },
        )
        collection: dict[str, Any] = {"prediction_count": 0}
    else:
        collection = collect_swe_predictions(
            _read_json_object(prediction_task_manifest),
            _read_json_object(queue_manifest),
            workspace_root=str(workspace_root),
            output_jsonl=str(predictions_jsonl),
            patch_job_verification=verification,
            include_abstained=False,
        )
        write_live_predictions_json(predictions_jsonl, preds_json)

    evaluator_pid: int | None = None
    if launch_evaluator and int(collection.get("prediction_count") or 0) > 0:
        cmd = [
            python,
            "-m",
            "evaluation.evaluation",
            "--dataset",
            "SWE-bench-Live/SWE-bench-Live",
            "--split",
            "verified",
            "--platform",
            "linux",
            "--patch_dir",
            str(preds_json.resolve()),
            "--output_dir",
            str(evaluation_results_dir.resolve()),
            "--workers",
            str(workers),
            "--overwrite",
            "1",
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(swe_bench_live_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        evaluator_pid = int(process.pid)

    return {
        "output_root": str(output_root),
        "verification_json": str(verification_json),
        "predictions_jsonl": str(predictions_jsonl),
        "preds_json": str(preds_json),
        "evaluation_results_dir": str(evaluation_results_dir),
        "selected_patch_count": int(verification.get("verified_patch_count") or 0),
        "prediction_count": int(collection.get("prediction_count") or 0),
        "skipped_nonterminal_count": int(verification.get("skipped_nonterminal_count") or 0),
        "cleared_result_entries": removed_count,
        "evaluator_pid": evaluator_pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--swe-bench-live-root", default="/data/agentkernel/other_repos/SWE-bench-Live")
    parser.add_argument("--python", default="/home/peyton/miniconda3/envs/ai/bin/python")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--launch-evaluator", action="store_true")
    parser.add_argument("--skip-semantic-artifact-check", action="store_true")
    args = parser.parse_args()
    result = refresh_official_rolling_score(
        queue_json=Path(args.queue_json),
        queue_manifest=Path(args.queue_manifest),
        prediction_task_manifest=Path(args.prediction_task_manifest),
        workspace_root=Path(args.workspace_root),
        output_root=Path(args.output_root),
        swe_bench_live_root=Path(args.swe_bench_live_root),
        python=args.python,
        workers=args.workers,
        launch_evaluator=bool(args.launch_evaluator),
        skip_semantic_artifact_check=bool(args.skip_semantic_artifact_check),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
