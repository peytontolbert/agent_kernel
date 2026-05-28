from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_by_instance(records: object) -> dict[str, dict[str, Any]]:
    by_instance: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        return by_instance
    for record in records:
        if not isinstance(record, dict):
            continue
        instance_id = str(record.get("instance_id", "")).strip()
        if instance_id:
            by_instance[instance_id] = record
    return by_instance


def _require_matching_patch_provenance(
    *,
    instance_id: str,
    task_id: str,
    source: Path,
    verification_record: dict[str, Any],
) -> None:
    if not verification_record:
        raise ValueError(f"missing verification provenance for instance_id={instance_id}")
    verified_task_id = str(verification_record.get("task_id", "")).strip()
    if verified_task_id and verified_task_id != task_id:
        raise ValueError(
            f"verification task_id mismatch for instance_id={instance_id}: "
            f"{verified_task_id} != {task_id}"
        )
    patch_path = str(verification_record.get("patch_path", "")).strip()
    if patch_path and Path(patch_path) != source:
        raise ValueError(
            f"verification patch_path mismatch for instance_id={instance_id}: "
            f"{patch_path} != {source}"
        )
    if not source.exists() or source.stat().st_size <= 0:
        raise ValueError(f"verified patch is missing or empty for instance_id={instance_id}: {source}")
    expected_size = verification_record.get("patch_size")
    if expected_size not in (None, "") and int(expected_size) != source.stat().st_size:
        raise ValueError(f"patch size changed after verification for instance_id={instance_id}")
    expected_mtime_ns = verification_record.get("patch_mtime_ns")
    if expected_mtime_ns not in (None, "") and int(expected_mtime_ns) != source.stat().st_mtime_ns:
        raise ValueError(f"patch mtime changed after verification for instance_id={instance_id}")
    expected_hash = str(verification_record.get("patch_sha256", "")).strip()
    if expected_hash and _sha256_file(source) != expected_hash:
        raise ValueError(f"patch sha256 changed after verification for instance_id={instance_id}")


def _verified_patch_source(
    *,
    workspace_base: Path,
    queue_task: dict[str, Any],
    verification_record: dict[str, Any],
    strict_verification_report: bool,
) -> Path:
    if strict_verification_report:
        patch_path = str(verification_record.get("patch_path", "")).strip()
        if patch_path:
            return Path(patch_path)
    return workspace_base / str(queue_task.get("workspace_subdir", "")).strip() / "patch.diff"


def collect_swe_predictions(
    prediction_task_manifest: dict[str, Any],
    queue_manifest: dict[str, Any],
    *,
    workspace_root: str,
    output_jsonl: str,
    instance_ids: list[str] | None = None,
    patch_job_verification: dict[str, Any] | None = None,
    include_abstained: bool = True,
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
    requested_ids = {str(value).strip() for value in (instance_ids or []) if str(value).strip()}
    abstained_ids: set[str] = set()
    inferred_abstained_ids: set[str] = set()
    verified_patch_records: dict[str, dict[str, Any]] = {}
    abstained_job_records: dict[str, dict[str, Any]] = {}
    strict_verification_report = False
    if patch_job_verification is not None:
        strict_verification_report = (
            str(patch_job_verification.get("report_kind", "")).strip()
            == "swe_bench_patch_job_verification"
        )
        verified_patch_records = _records_by_instance(patch_job_verification.get("verified_patches", []))
        abstained_job_records = _records_by_instance(patch_job_verification.get("abstained_jobs", []))
        if strict_verification_report and not verified_patch_records and patch_job_verification.get("successful_instance_ids"):
            raise ValueError("strict patch job verification missing verified_patches provenance records")
        successful_ids = {
            str(value).strip()
            for value in patch_job_verification.get("successful_instance_ids", [])
            if str(value).strip()
        }
        abstained_ids = {
            str(value).strip()
            for value in patch_job_verification.get("abstained_instance_ids", [])
            if str(value).strip()
        }
        selectable_ids = successful_ids | abstained_ids
        if not selectable_ids:
            raise ValueError("patch job verification has no successful or abstained instance ids")
        if requested_ids:
            missing = sorted(requested_ids - selectable_ids)
            if missing:
                raise ValueError("requested instance_ids are not verified successful or abstained: " + ",".join(missing))
        else:
            requested_ids = selectable_ids if include_abstained else successful_ids
    selected_predictions: list[dict[str, Any]] = []
    for prediction in prediction_manifest.get("predictions", []):
        if not isinstance(prediction, dict):
            raise ValueError("prediction_manifest predictions must be objects")
        instance_id = str(prediction.get("instance_id", "")).strip()
        if requested_ids and instance_id not in requested_ids:
            continue
        queue_task = queue_by_instance.get(instance_id)
        if not queue_task:
            raise ValueError(f"missing queue task for instance_id={instance_id}")
        task_id = str(queue_task.get("task_id", "")).strip()
        verification_record = verified_patch_records.get(instance_id, {})
        abstention_record = abstained_job_records.get(instance_id, {})
        source = _verified_patch_source(
            workspace_base=workspace_base,
            queue_task=queue_task,
            verification_record=abstention_record if instance_id in abstained_ids else verification_record,
            strict_verification_report=strict_verification_report,
        )
        if instance_id in abstained_ids:
            if strict_verification_report and not abstention_record:
                raise ValueError(f"missing abstention provenance for instance_id={instance_id}")
            if source.exists() and source.stat().st_size > 0:
                if strict_verification_report and not str(abstention_record.get("patch_sha256", "")).strip():
                    raise ValueError(
                        f"abstention for instance_id={instance_id} is stale or lacks patch provenance"
                    )
                _require_matching_patch_provenance(
                    instance_id=instance_id,
                    task_id=task_id,
                    source=source,
                    verification_record=abstention_record,
                )
            if not include_abstained:
                continue
            selected_prediction = dict(prediction)
            selected_prediction.pop("patch_path", None)
            selected_prediction["model_patch"] = ""
            selected_prediction["abstained"] = True
            selected_predictions.append(selected_prediction)
            continue
        if strict_verification_report:
            _require_matching_patch_provenance(
                instance_id=instance_id,
                task_id=task_id,
                source=source,
                verification_record=verification_record,
            )
        if not source.exists() or source.stat().st_size <= 0:
            inferred_abstained_ids.add(instance_id)
            if not include_abstained:
                continue
            selected_prediction = dict(prediction)
            selected_prediction.pop("patch_path", None)
            selected_prediction["model_patch"] = ""
            selected_prediction["abstained"] = True
            selected_predictions.append(selected_prediction)
            continue
        patch_text = source.read_text(encoding="utf-8")
        if "diff --git " not in patch_text and "--- " not in patch_text:
            raise ValueError(f"generated patch does not look like a unified diff: {source}")
        target = Path(str(prediction.get("patch_path", "")).strip())
        if not target.is_absolute():
            target = base_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch_text, encoding="utf-8")
        copied.append({"instance_id": instance_id, "source": str(source), "target": str(target)})
        selected_predictions.append(dict(prediction))
    if not selected_predictions:
        raise ValueError("no predictions selected for collection")
    selected_prediction_manifest = dict(prediction_manifest)
    selected_prediction_manifest["predictions"] = selected_predictions
    records = build_swe_predictions_from_manifest(selected_prediction_manifest)
    _write_jsonl(Path(output_jsonl), records)
    return {
        "copied_patch_count": len(copied),
        "abstained_prediction_count": len((abstained_ids | inferred_abstained_ids) & set(record["instance_id"] for record in records)),
        "prediction_count": len(records),
        "copied_patches": copied,
        "output_jsonl": output_jsonl,
        "selected_instance_ids": [record["instance_id"] for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--patch-job-verification-json", default="")
    parser.add_argument("--exclude-abstained", action="store_true")
    args = parser.parse_args()

    patch_job_verification = (
        _read_json(Path(args.patch_job_verification_json)) if str(args.patch_job_verification_json).strip() else None
    )
    result = collect_swe_predictions(
        _read_json(Path(args.prediction_task_manifest)),
        _read_json(Path(args.queue_manifest)),
        workspace_root=args.workspace_root,
        output_jsonl=args.output_jsonl,
        instance_ids=args.instance_ids,
        patch_job_verification=patch_job_verification,
        include_abstained=not bool(args.exclude_abstained),
    )
    print(
        f"copied_patch_count={result['copied_patch_count']} "
        f"prediction_count={result['prediction_count']} "
        f"output_jsonl={result['output_jsonl']}"
    )


if __name__ == "__main__":
    main()
