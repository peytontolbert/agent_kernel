from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any


REQUIRED_FIELDS = ("instance_id", "model_name_or_path", "model_patch")


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict | list):
        raise SystemExit(f"expected JSON object or array at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        records.append(payload)
    return records


def validate_swe_predictions(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if not records:
        failures.append("predictions must contain at least one record")
        return failures
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        for field in REQUIRED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str):
                failures.append(f"record {index} field {field} must be a string")
            elif field != "model_patch" and not value.strip():
                failures.append(f"record {index} field {field} must be a non-empty string")
        instance_id = str(record.get("instance_id", "")).strip()
        if instance_id:
            if instance_id in seen:
                failures.append(f"duplicate instance_id: {instance_id}")
            seen.add(instance_id)
        patch = str(record.get("model_patch", "")).strip()
        if patch and "diff --git " not in patch and "--- " not in patch:
            failures.append(f"record {index} model_patch does not look like a unified diff")
    return failures


def build_swe_predictions_from_manifest(manifest: dict[str, Any] | list[Any]) -> list[dict[str, str]]:
    if isinstance(manifest, dict) and isinstance(manifest.get("prediction_manifest"), dict):
        manifest = manifest["prediction_manifest"]
    items = manifest.get("predictions", manifest.get("instances", [])) if isinstance(manifest, dict) else manifest
    if not isinstance(items, list):
        raise ValueError("manifest must be a list or contain predictions/instances list")
    predictions: list[dict[str, str]] = []
    base_dir = Path(str(manifest.get("base_dir", ""))).expanduser() if isinstance(manifest, dict) else Path()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        instance_id = str(item.get("instance_id", "")).strip()
        model_name = str(item.get("model_name_or_path", item.get("model", "agentkernel"))).strip()
        patch = str(item.get("model_patch", "")).strip()
        patch_path = str(item.get("patch_path", "")).strip()
        if patch_path:
            path = Path(patch_path)
            if not path.is_absolute() and str(base_dir):
                path = base_dir / path
            patch = path.read_text(encoding="utf-8")
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }
        )
    failures = validate_swe_predictions(predictions)
    if failures:
        raise ValueError("invalid SWE predictions manifest: " + "; ".join(failures))
    return predictions


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _validate_mode(args: argparse.Namespace) -> None:
    path = Path(args.predictions_jsonl)
    failures = validate_swe_predictions(_read_jsonl(path))
    if failures:
        raise SystemExit("SWE predictions validation failed: " + "; ".join(failures))
    print(f"verified_swe_predictions={path}")


def _build_mode(args: argparse.Namespace) -> None:
    output_path = Path(args.output_jsonl)
    records = build_swe_predictions_from_manifest(_read_json(Path(args.manifest_json)))
    _write_jsonl(output_path, records)
    print(f"prediction_count={len(records)} output_jsonl={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--manifest-json", required=True)
    build_parser.add_argument("--output-jsonl", required=True)
    build_parser.set_defaults(func=_build_mode)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--predictions-jsonl", required=True)
    validate_parser.set_defaults(func=_validate_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
