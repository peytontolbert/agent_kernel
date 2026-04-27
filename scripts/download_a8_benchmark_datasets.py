from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
from datetime import datetime, timezone
from typing import Any


DEFAULT_SOURCE_MANIFEST = Path("config/a8_benchmark_dataset_sources.json")
DEFAULT_STATUS_JSON = Path("benchmarks/a8_dataset_sources/status.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - best-effort JSON conversion for dataset scalar types.
            pass
    return value


def _source_path(root: Path, source: dict[str, Any]) -> Path:
    raw = Path(str(source.get("local_path", "")).strip())
    return raw if raw.is_absolute() else root / raw


def _file_status(path: Path) -> dict[str, Any]:
    if path.exists():
        return {
            "exists": True,
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "kind": "directory" if path.is_dir() else "file",
        }
    return {"exists": False, "size_bytes": 0, "kind": ""}


def _dataset_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file() or path.suffix != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("instances", "data", "rows", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def _download_hf_dataset(source: dict[str, Any], target: Path, *, dry_run: bool) -> dict[str, Any]:
    dataset_id = str(source.get("dataset_id", "")).strip()
    split = str(source.get("split", "test")).strip() or "test"
    if dry_run:
        return {"action": "download_huggingface_dataset", "dataset_id": dataset_id, "split": split}
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split)
        rows = [dict(row) for row in dataset]
    except Exception as first_error:  # noqa: BLE001 - fallback supports envs with broken optional imports.
        rows = _download_hf_dataset_from_parquet(dataset_id, split, first_error=first_error)
    _write_json(target, rows)
    return {"action": "downloaded_huggingface_dataset", "dataset_id": dataset_id, "split": split, "rows": len(rows)}


def _download_hf_dataset_from_parquet(dataset_id: str, split: str, *, first_error: Exception) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"datasets load failed ({first_error}); parquet fallback dependencies unavailable") from exc
    info = HfApi().dataset_info(dataset_id)
    parquet_files = [
        getattr(item, "rfilename", "")
        for item in getattr(info, "siblings", [])
        if str(getattr(item, "rfilename", "")).endswith(".parquet")
    ]
    split_files = [name for name in parquet_files if f"/{split}-" in name or f"{split}-" in Path(name).name]
    selected = split_files or parquet_files
    if not selected:
        raise RuntimeError(f"datasets load failed ({first_error}); no parquet files found for {dataset_id}")
    rows: list[dict[str, Any]] = []
    for filename in sorted(selected):
        local_file = hf_hub_download(repo_id=dataset_id, filename=filename, repo_type="dataset")
        frame = pd.read_parquet(local_file)
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _download_hf_metadata(source: dict[str, Any], target: Path, *, dry_run: bool) -> dict[str, Any]:
    dataset_id = str(source.get("dataset_id", "")).strip()
    if dry_run:
        return {"action": "download_huggingface_metadata", "dataset_id": dataset_id}
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub package is required for Hugging Face metadata downloads") from exc
    info = HfApi().dataset_info(dataset_id)
    siblings = [getattr(item, "rfilename", "") for item in getattr(info, "siblings", [])]
    payload = {
        "dataset_id": dataset_id,
        "downloads": getattr(info, "downloads", None),
        "gated": getattr(info, "gated", None),
        "last_modified": str(getattr(info, "last_modified", "") or ""),
        "likes": getattr(info, "likes", None),
        "private": getattr(info, "private", None),
        "sha": getattr(info, "sha", ""),
        "siblings": [item for item in siblings if item],
        "tags": list(getattr(info, "tags", []) or []),
    }
    _write_json(target, payload)
    return {"action": "downloaded_huggingface_metadata", "dataset_id": dataset_id}


def _clone_git_repo(source: dict[str, Any], target: Path, *, dry_run: bool) -> dict[str, Any]:
    repo_url = str(source.get("repo_url", "")).strip()
    if target.exists():
        return {"action": "already_present", "repo_url": repo_url}
    if dry_run:
        return {"action": "git_clone", "repo_url": repo_url}
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
    return {"action": "cloned_git_repo", "repo_url": repo_url}


def _status_for_source(source: dict[str, Any], root: Path) -> dict[str, Any]:
    target = _source_path(root, source)
    file_status = _file_status(target)
    rows = _dataset_rows(target)
    status = "available" if file_status["exists"] else "missing"
    if source.get("kind") in {"pending_source", "summary_only"} and not file_status["exists"]:
        status = "not_ready"
    if source.get("requires_credentials") and not file_status["exists"]:
        status = "needs_credentials"
    return {
        "benchmark": source.get("benchmark", ""),
        "label": source.get("label", ""),
        "kind": source.get("kind", ""),
        "required_for_a8": bool(source.get("required_for_a8", False)),
        "requires_credentials": bool(source.get("requires_credentials", False)),
        "large": bool(source.get("large", False)),
        "dataset_id": source.get("dataset_id", ""),
        "repo_url": source.get("repo_url", ""),
        "local_path": str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
        "status": status,
        "rows": rows,
        "notes": source.get("notes", ""),
        **file_status,
    }


def download_a8_benchmark_datasets(
    *,
    root: Path,
    source_manifest: Path,
    output_status: Path,
    include_large: bool = False,
    include_git: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = _read_json_object(source_manifest)
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("source manifest must contain sources list")
    records: list[dict[str, Any]] = []
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            continue
        source = dict(raw_source)
        target = _source_path(root, source)
        action: dict[str, Any] = {"action": "status_only"}
        error = ""
        try:
            if source.get("large") and source.get("kind") != "huggingface_metadata" and not include_large:
                action = {"action": "skipped_large_source"}
            elif source.get("kind") == "huggingface_dataset" and (force or not target.exists()):
                action = _download_hf_dataset(source, target, dry_run=dry_run)
            elif source.get("kind") == "huggingface_metadata" and (force or not target.exists()):
                action = _download_hf_metadata(source, target, dry_run=dry_run)
            elif source.get("kind") == "git_repo":
                if include_git:
                    action = _clone_git_repo(source, target, dry_run=dry_run)
                elif not target.exists():
                    action = {"action": "skipped_git_source"}
            elif source.get("kind") in {"pending_source", "summary_only"}:
                action = {"action": "not_downloadable"}
        except Exception as exc:  # noqa: BLE001 - persisted into status report for UI.
            error = str(exc)
            action = {"action": "error"}
        status = _status_for_source(source, root)
        if error:
            status["status"] = "error"
            status["error"] = error
        status["last_action"] = action
        records.append(status)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest.relative_to(root)) if source_manifest.is_relative_to(root) else str(source_manifest),
        "sources": records,
    }
    if not dry_run:
        _write_json(output_status, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-status", default=str(DEFAULT_STATUS_JSON))
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--include-git", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source_manifest = Path(args.source_manifest)
    if not source_manifest.is_absolute():
        source_manifest = root / source_manifest
    output_status = Path(args.output_status)
    if not output_status.is_absolute():
        output_status = root / output_status
    result = download_a8_benchmark_datasets(
        root=root,
        source_manifest=source_manifest,
        output_status=output_status,
        include_large=bool(args.include_large),
        include_git=bool(args.include_git),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    counts: dict[str, int] = {}
    for source in result["sources"]:
        counts[str(source["status"])] = counts.get(str(source["status"]), 0) + 1
    print(json.dumps({"status_counts": counts, "output_status": str(output_status)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
