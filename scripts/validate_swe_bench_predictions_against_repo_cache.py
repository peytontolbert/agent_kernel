from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
from typing import Any


def _read_json_or_jsonl(path: Path) -> dict[str, Any] | list[Any]:
    if path.suffix == ".arrow":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise SystemExit("reading .arrow datasets requires the datasets package") from exc
        return [dict(row) for row in Dataset.from_file(str(path))]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict | list):
        raise SystemExit(f"expected JSON object, array, JSONL, or Arrow dataset at {path}")
    return payload


def _dataset_items(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("instances", payload.get("data", payload.get("rows", payload.get("tasks", []))))
    if not isinstance(items, list):
        raise ValueError("dataset must be a list or contain instances/data/rows/tasks list")
    return [item for item in items if isinstance(item, dict)]


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _repo(item: dict[str, Any]) -> str:
    direct = _text(item.get("repo") or item.get("repository"))
    if direct:
        return direct
    base = item.get("base") if isinstance(item.get("base"), dict) else {}
    repo = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    return _text(repo.get("full_name"))


def _base_commit(item: dict[str, Any]) -> str:
    direct = _text(item.get("base_commit") or item.get("commit") or item.get("base_sha"))
    if direct:
        return direct
    base = item.get("base") if isinstance(item.get("base"), dict) else {}
    return _text(base.get("sha"))


def _repo_cache_path(repo_cache_root: str, repo: str) -> Path | None:
    root = Path(repo_cache_root)
    candidates = [root / repo, root / repo.replace("/", "__"), root / repo.split("/")[-1]]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _read_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("prediction records must be objects")
        instance_id = _text(record.get("instance_id"))
        raw_patch = record.get("model_patch")
        patch = raw_patch if isinstance(raw_patch, str) else ""
        if not instance_id:
            raise ValueError("prediction missing instance_id")
        predictions[instance_id] = patch
    return predictions


def validate_predictions_against_repo_cache(
    dataset: dict[str, Any] | list[Any],
    *,
    predictions_jsonl: str,
    repo_cache_root: str,
    limit: int = 0,
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected_ids = {value.strip() for value in (instance_ids or []) if value.strip()}
    predictions = _read_predictions(Path(predictions_jsonl))
    results: list[dict[str, Any]] = []
    selected_count = 0
    with tempfile.TemporaryDirectory(prefix="swe_patch_validate_") as tmp:
        tmp_root = Path(tmp)
        for item in _dataset_items(dataset):
            instance_id = _text(item.get("instance_id") or item.get("id"))
            if not instance_id or instance_id not in predictions:
                continue
            if selected_ids and instance_id not in selected_ids:
                continue
            selected_count += 1
            repo = _repo(item)
            commit = _base_commit(item)
            repo_path = _repo_cache_path(repo_cache_root, repo)
            patch_text = predictions[instance_id]
            result: dict[str, Any] = {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": commit,
                "apply_check_passed": False,
                "reason": "",
            }
            if repo_path is None:
                result["reason"] = "missing_repo_cache"
                results.append(result)
                continue
            if not patch_text:
                result["reason"] = "empty_patch"
                results.append(result)
                continue
            worktree = tmp_root / instance_id.replace("/", "_")
            clone = subprocess.run(
                ["git", "clone", "--shared", "--no-checkout", str(repo_path), str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if clone.returncode != 0:
                result["reason"] = "clone_failed"
                result["stderr"] = clone.stderr.strip()
                results.append(result)
                continue
            checkout = subprocess.run(
                ["git", "-C", str(worktree), "checkout", "--detach", commit],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if checkout.returncode != 0:
                result["reason"] = "checkout_failed"
                result["stderr"] = checkout.stderr.strip()
                results.append(result)
                continue
            patch_path = worktree / "candidate.patch"
            patch_path.write_text(patch_text, encoding="utf-8")
            apply = subprocess.run(
                ["git", "-C", str(worktree), "apply", "--check", str(patch_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            result["apply_check_passed"] = apply.returncode == 0
            if apply.returncode != 0:
                result["reason"] = "apply_check_failed"
                result["stderr"] = apply.stderr.strip()
            results.append(result)
            shutil.rmtree(worktree, ignore_errors=True)
            if limit > 0 and selected_count >= limit:
                break
    return {
        "prediction_count": len(predictions),
        "selected_instance_count": len(results),
        "apply_check_passed_count": sum(1 for result in results if result["apply_check_passed"]),
        "all_apply_check_passed": bool(results) and all(result["apply_check_passed"] for result in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--repo-cache-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--instance-ids", nargs="*")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = validate_predictions_against_repo_cache(
        _read_json_or_jsonl(Path(args.dataset_json)),
        predictions_jsonl=args.predictions_jsonl,
        repo_cache_root=args.repo_cache_root,
        limit=int(args.limit),
        instance_ids=args.instance_ids,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_apply_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
