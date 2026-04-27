from __future__ import annotations

from pathlib import Path
import argparse
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


def _safe_repo_relpath(repo: str) -> Path:
    parts = [part for part in repo.split("/") if part]
    if len(parts) >= 2:
        return Path(parts[-2]) / parts[-1]
    return Path(repo.replace("/", "__"))


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    if dry_run:
        print("dry_run " + json.dumps({"cmd": cmd, "cwd": str(cwd) if cwd else ""}, sort_keys=True))
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_repo_cache(
    dataset: dict[str, Any] | list[Any],
    *,
    repo_cache_root: str,
    fetch: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    instance_ids: list[str] | None = None,
    repos_filter: list[str] | None = None,
) -> dict[str, Any]:
    selected_ids = {value.strip() for value in (instance_ids or []) if value.strip()}
    selected_repos = {value.strip() for value in (repos_filter or []) if value.strip()}
    repos: dict[str, set[str]] = {}
    selected_count = 0
    for item in _dataset_items(dataset):
        instance_id = _text(item.get("instance_id") or item.get("id"))
        if selected_ids and instance_id not in selected_ids:
            continue
        repo = _repo(item)
        if not repo:
            continue
        if selected_repos and repo not in selected_repos:
            continue
        selected_count += 1
        repos.setdefault(repo, set())
        commit = _base_commit(item)
        if commit:
            repos[repo].add(commit)
        if limit > 0 and selected_count >= limit:
            break
    if not repos:
        raise ValueError("no repositories found in dataset")
    root = Path(repo_cache_root)
    root.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for repo, commits in sorted(repos.items()):
        target = root / _safe_repo_relpath(repo)
        if not target.exists():
            _run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(target)], dry_run=dry_run)
        elif fetch:
            _run(["git", "fetch", "--all", "--tags", "--prune"], cwd=target, dry_run=dry_run)
        missing_commits: list[str] = []
        for commit in sorted(commits):
            if dry_run:
                continue
            result = subprocess.run(
                ["git", "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                missing_commits.append(commit)
        prepared.append(
            {
                "repo": repo,
                "path": str(target),
                "commit_count": len(commits),
                "missing_commits": missing_commits,
            }
        )
    return {
        "repo_cache_root": str(root),
        "selected_instance_count": selected_count,
        "repo_count": len(prepared),
        "repositories": prepared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--repo-cache-root", required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--instance-ids", nargs="*")
    parser.add_argument("--repos", nargs="*")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = prepare_repo_cache(
        _read_json_or_jsonl(Path(args.dataset_json)),
        repo_cache_root=args.repo_cache_root,
        fetch=bool(args.fetch),
        dry_run=bool(args.dry_run),
        limit=int(args.limit),
        instance_ids=args.instance_ids,
        repos_filter=args.repos,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
