from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import subprocess
from typing import Any


def _read_json_or_jsonl(path: Path) -> dict[str, Any] | list[Any]:
    if path.suffix == ".arrow":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise SystemExit("reading .arrow datasets requires the datasets package") from exc
        return [dict(row) for row in Dataset.from_file(str(path))]
    if path.suffix == ".jsonl":
        records: list[Any] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict | list):
        raise SystemExit(f"expected JSON object, array, or JSONL at {path}")
    return payload


def _dataset_items(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("instances", payload.get("data", payload.get("rows", [])))
    if not isinstance(items, list):
        raise ValueError("dataset must be a list or contain instances/data/rows list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"dataset item {index} must be an object")
        records.append(item)
    return records


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _instance_id(item: dict[str, Any]) -> str:
    return _text(item.get("instance_id") or item.get("id") or item.get("pull_number") or item.get("number"))


def _safe_filename(instance_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in instance_id)
    return safe.strip("._") or "instance"


def _problem_statement(item: dict[str, Any]) -> str:
    return _text(
        item.get("problem_statement")
        or item.get("issue_description")
        or item.get("body")
        or item.get("title")
    )


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


def _paths_from_unified_diff(diff_text: Any) -> list[str]:
    if not isinstance(diff_text, str) or not diff_text.strip():
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for raw_line in diff_text.splitlines():
        line = raw_line.strip()
        candidate = ""
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[2]
        elif line.startswith("+++ ") or line.startswith("--- "):
            parts = line.split()
            if len(parts) >= 2:
                candidate = parts[1]
        if not candidate or candidate == "/dev/null":
            continue
        if candidate.startswith("a/") or candidate.startswith("b/"):
            candidate = candidate[2:]
        if candidate and candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)
    return paths


def _candidate_files(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for field in ("patch", "test_patch"):
        for path in _paths_from_unified_diff(item.get(field)):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    for field in ("FAIL_TO_PASS", "fail_to_pass", "PASS_TO_PASS", "pass_to_pass"):
        values = item.get(field, [])
        if isinstance(values, str):
            try:
                decoded = json.loads(values)
            except json.JSONDecodeError:
                decoded = [values]
            values = decoded
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            path = value.split("::", 1)[0].strip()
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _repo_cache_path(repo_cache_root: str, repo: str) -> Path | None:
    root = Path(repo_cache_root)
    if not str(root).strip():
        return None
    candidates = [
        root / repo,
        root / repo.replace("/", "__"),
        root / repo.split("/")[-1],
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _source_context(
    *,
    repo_cache_root: str,
    repo: str,
    base_commit: str,
    candidate_files: list[str],
    max_bytes_per_file: int,
) -> list[dict[str, Any]]:
    repo_root = _repo_cache_path(repo_cache_root, repo)
    if repo_root is None:
        return []
    contexts: list[dict[str, Any]] = []
    byte_limit = max(0, int(max_bytes_per_file))
    for relative in candidate_files:
        raw: bytes | None = None
        if base_commit:
            try:
                raw = subprocess.check_output(
                    ["git", "-C", str(repo_root), "show", f"{base_commit}:{relative}"],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                raw = None
        path = repo_root / relative
        try:
            resolved = path.resolve()
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            continue
        if raw is None:
            if not resolved.exists() or not resolved.is_file():
                continue
            try:
                raw = resolved.read_bytes()
            except OSError:
                continue
        truncated = byte_limit > 0 and len(raw) > byte_limit
        if byte_limit > 0:
            raw = raw[:byte_limit]
        contexts.append(
            {
                "path": relative,
                "content": raw.decode("utf-8", errors="replace"),
                "truncated": truncated,
            }
        )
    return contexts


def build_swe_prediction_task_manifest(
    dataset: dict[str, Any] | list[Any],
    *,
    output_patch_dir: str,
    model_name_or_path: str,
    limit: int = 0,
    instance_ids: list[str] | None = None,
    repo_cache_root: str = "",
    max_source_context_bytes: int = 12000,
) -> dict[str, Any]:
    selected_ids = {value.strip() for value in (instance_ids or []) if value.strip()}
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _dataset_items(dataset):
        instance_id = _instance_id(item)
        if not instance_id:
            continue
        if selected_ids and instance_id not in selected_ids:
            continue
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id in dataset: {instance_id}")
        seen.add(instance_id)
        problem_statement = _problem_statement(item)
        if not problem_statement:
            raise ValueError(f"missing problem statement for instance_id={instance_id}")
        patch_path = str(Path(output_patch_dir) / f"{_safe_filename(instance_id)}.diff")
        repo = _repo(item)
        base_commit = _base_commit(item)
        candidate_files = _candidate_files(item)
        task = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "repo": repo,
            "base_commit": base_commit,
            "patch_path": patch_path,
            "repo_cache_root": repo_cache_root,
            "problem_statement": problem_statement,
            "hints_text": _text(item.get("hints_text")),
            "candidate_files": candidate_files,
            "source_context": _source_context(
                repo_cache_root=repo_cache_root,
                repo=repo,
                base_commit=base_commit,
                candidate_files=candidate_files,
                max_bytes_per_file=max_source_context_bytes,
            ),
            "fail_to_pass": item.get("FAIL_TO_PASS", item.get("fail_to_pass", [])),
            "pass_to_pass": item.get("PASS_TO_PASS", item.get("pass_to_pass", [])),
            "task_prompt": (
                "Produce a minimal unified diff that resolves the SWE-bench instance. "
                "Return only the diff; do not include prose or markdown fences."
            ),
        }
        tasks.append(task)
        if limit > 0 and len(tasks) >= limit:
            break
    if selected_ids:
        found = {task["instance_id"] for task in tasks}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError("requested instance_ids not found: " + ",".join(missing))
    if not tasks:
        raise ValueError("no SWE prediction tasks selected")
    return {
        "spec_version": "asi_v1",
        "report_kind": "swe_bench_prediction_task_manifest",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name_or_path": model_name_or_path,
        "output_patch_dir": output_patch_dir,
        "task_count": len(tasks),
        "tasks": tasks,
        "prediction_manifest": {
            "base_dir": output_patch_dir,
            "predictions": [
                {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": model_name_or_path,
                    "patch_path": Path(task["patch_path"]).name,
                }
                for task in tasks
            ],
        },
        "open_limits": [
            "This manifest does not contain predictions; each patch_path must be filled with a real unified diff.",
            "Run scripts/prepare_swe_bench_predictions.py build after patches exist to create benchmark JSONL.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--output-patch-dir", required=True)
    parser.add_argument("--model-name-or-path", default="agentkernel")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--instance-ids", nargs="*")
    parser.add_argument("--repo-cache-root", default="")
    parser.add_argument("--max-source-context-bytes", type=int, default=12000)
    args = parser.parse_args()

    manifest = build_swe_prediction_task_manifest(
        _read_json_or_jsonl(Path(args.dataset_json)),
        output_patch_dir=args.output_patch_dir,
        model_name_or_path=args.model_name_or_path,
        limit=int(args.limit),
        instance_ids=args.instance_ids,
        repo_cache_root=args.repo_cache_root,
        max_source_context_bytes=int(args.max_source_context_bytes),
    )
    output_path = Path(args.output_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"task_count={manifest['task_count']} output_manifest_json={output_path}")


if __name__ == "__main__":
    main()
