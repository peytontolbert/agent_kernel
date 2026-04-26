from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

from prepare_swe_bench_predictions import build_swe_predictions_from_manifest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def collect_swe_predictions(
    prediction_task_manifest: dict[str, Any],
    queue_manifest: dict[str, Any],
    *,
    workspace_root: str,
    output_jsonl: str,
) -> dict[str, Any]:
    prediction_manifest = prediction_task_manifest.get("prediction_manifest")
    if not isinstance(prediction_manifest, dict):
        raise ValueError("prediction task manifest missing prediction_manifest")
    queue_tasks = queue_manifest.get("tasks", [])
    if not isinstance(queue_tasks, list) or not queue_tasks:
        raise ValueError("queue manifest must contain non-empty tasks list")
    queue_by_instance: dict[str, dict[str, Any]] = {}
    for task in queue_tasks:
        if not isinstance(task, dict):
            continue
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        instance_id = str(metadata.get("swe_instance_id", "")).strip()
        if instance_id:
            queue_by_instance[instance_id] = task
    copied: list[dict[str, str]] = []
    workspace_base = Path(workspace_root)
    base_dir = Path(str(prediction_manifest.get("base_dir", "")))
    for prediction in prediction_manifest.get("predictions", []):
        if not isinstance(prediction, dict):
            raise ValueError("prediction_manifest predictions must be objects")
        instance_id = str(prediction.get("instance_id", "")).strip()
        queue_task = queue_by_instance.get(instance_id)
        if not queue_task:
            raise ValueError(f"missing queue task for instance_id={instance_id}")
        source = workspace_base / str(queue_task.get("workspace_subdir", "")).strip() / "patch.diff"
        if not source.exists() or source.stat().st_size <= 0:
            raise ValueError(f"missing generated patch for instance_id={instance_id}: {source}")
        patch_text = source.read_text(encoding="utf-8")
        if "diff --git " not in patch_text and "--- " not in patch_text:
            raise ValueError(f"generated patch does not look like a unified diff: {source}")
        target = Path(str(prediction.get("patch_path", "")).strip())
        if not target.is_absolute():
            target = base_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch_text, encoding="utf-8")
        copied.append({"instance_id": instance_id, "source": str(source), "target": str(target)})
    records = build_swe_predictions_from_manifest(prediction_manifest)
    _write_jsonl(Path(output_jsonl), records)
    return {
        "copied_patch_count": len(copied),
        "prediction_count": len(records),
        "copied_patches": copied,
        "output_jsonl": output_jsonl,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    result = collect_swe_predictions(
        _read_json(Path(args.prediction_task_manifest)),
        _read_json(Path(args.queue_manifest)),
        workspace_root=args.workspace_root,
        output_jsonl=args.output_jsonl,
    )
    print(
        f"copied_patch_count={result['copied_patch_count']} "
        f"prediction_count={result['prediction_count']} "
        f"output_jsonl={result['output_jsonl']}"
    )


if __name__ == "__main__":
    main()
