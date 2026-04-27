from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import re
import subprocess
from typing import Any


DEFAULT_SOURCE_CONTEXT_BYTES = 30000


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


def _iter_test_names(item: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for field in fields:
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
            name = value.rsplit("::", 1)[-1].strip()
            if name:
                names.append(name)
    return names


def _source_focus_terms(item: dict[str, Any], problem_statement: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip("`'\"()[]{}.,:;")
        if len(cleaned) < 3 or cleaned in seen:
            return
        if "/" in cleaned or "\\" in cleaned:
            return
        seen.add(cleaned)
        terms.append(cleaned)

    for match in re.finditer(r"`([^`\n]{3,80})`", problem_statement):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(1)):
            add(token)
    for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", problem_statement):
        add(token)
    for token in re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\b", problem_statement):
        add(token)
    for token in re.findall(r"\b([a-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)\b", problem_statement):
        add(token)
    for name in _iter_test_names(item, ("FAIL_TO_PASS", "fail_to_pass")):
        add(name)
        if name.startswith("test_"):
            add(name.removeprefix("test_"))
    return terms


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


def _line_start_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _line_number_for_offset(offsets: list[int], offset: int) -> int:
    lo = 0
    hi = len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def _focused_source_windows(
    *,
    relative: str,
    text: str,
    prefix: str,
    focus_terms: list[str],
    max_bytes_per_file: int,
) -> list[dict[str, Any]]:
    if not focus_terms:
        return []
    contexts: list[dict[str, Any]] = []
    seen_offsets: set[int] = set()
    line_offsets = _line_start_offsets(text)
    line_count = len(text.splitlines()) or 1
    max_snippet_bytes = max(
        1200,
        min(
            max_bytes_per_file if max_bytes_per_file > 0 else DEFAULT_SOURCE_CONTEXT_BYTES,
            DEFAULT_SOURCE_CONTEXT_BYTES,
        ),
    )
    prefix_len = len(prefix)
    for term in focus_terms:
        patterns = [
            rf"(?m)^(\s*def\s+{re.escape(term)}\b)",
            rf"(?m)^(\s*class\s+{re.escape(term)}\b)",
            rf"\b{re.escape(term)}\b",
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                break
        if not match or match.start() < prefix_len:
            continue
        if any(abs(match.start() - prior) < max_snippet_bytes for prior in seen_offsets):
            continue
        seen_offsets.add(match.start())
        center_line = _line_number_for_offset(line_offsets, match.start())
        line_start = max(1, center_line - 60)
        line_end = min(line_count, center_line + 100)
        lines = text.splitlines(keepends=True)
        while line_start < center_line:
            snippet = "".join(lines[line_start - 1 : line_end])
            if len(snippet.encode("utf-8")) <= max_snippet_bytes:
                break
            line_start += 10
            if line_end > center_line:
                line_end -= 10
        snippet = "".join(lines[line_start - 1 : line_end])
        contexts.append(
            {
                "path": relative,
                "content": snippet,
                "truncated": True,
                "context_kind": "focused_window",
                "focus_term": term,
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return contexts


def _source_context(
    *,
    repo_cache_root: str,
    repo: str,
    base_commit: str,
    candidate_files: list[str],
    max_bytes_per_file: int,
    focus_terms: list[str] | None = None,
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
        full_raw = raw
        truncated = byte_limit > 0 and len(raw) > byte_limit
        if byte_limit > 0:
            raw = raw[:byte_limit]
        prefix = raw.decode("utf-8", errors="replace")
        contexts.append(
            {
                "path": relative,
                "content": prefix,
                "truncated": truncated,
            }
        )
        if truncated:
            contexts.extend(
                _focused_source_windows(
                    relative=relative,
                    text=full_raw.decode("utf-8", errors="replace"),
                    prefix=prefix,
                    focus_terms=focus_terms or [],
                    max_bytes_per_file=max_bytes_per_file,
                )
            )
    return contexts


def build_swe_prediction_task_manifest(
    dataset: dict[str, Any] | list[Any],
    *,
    output_patch_dir: str,
    model_name_or_path: str,
    limit: int = 0,
    instance_ids: list[str] | None = None,
    repos_filter: list[str] | None = None,
    repo_cache_root: str = "",
    max_source_context_bytes: int = DEFAULT_SOURCE_CONTEXT_BYTES,
) -> dict[str, Any]:
    selected_ids = {value.strip() for value in (instance_ids or []) if value.strip()}
    selected_repos = {value.strip() for value in (repos_filter or []) if value.strip()}
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
        problem_statement = _problem_statement(item)
        if not problem_statement:
            raise ValueError(f"missing problem statement for instance_id={instance_id}")
        patch_path = str(Path(output_patch_dir) / f"{_safe_filename(instance_id)}.diff")
        repo = _repo(item)
        if selected_repos and repo not in selected_repos:
            continue
        seen.add(instance_id)
        base_commit = _base_commit(item)
        candidate_files = _candidate_files(item)
        focus_terms = _source_focus_terms(item, problem_statement)
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
                focus_terms=focus_terms,
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
    parser.add_argument("--repos", nargs="*")
    parser.add_argument("--repo-cache-root", default="")
    parser.add_argument("--max-source-context-bytes", type=int, default=DEFAULT_SOURCE_CONTEXT_BYTES)
    args = parser.parse_args()

    manifest = build_swe_prediction_task_manifest(
        _read_json_or_jsonl(Path(args.dataset_json)),
        output_patch_dir=args.output_patch_dir,
        model_name_or_path=args.model_name_or_path,
        limit=int(args.limit),
        instance_ids=args.instance_ids,
        repos_filter=args.repos,
        repo_cache_root=args.repo_cache_root,
        max_source_context_bytes=int(args.max_source_context_bytes),
    )
    output_path = Path(args.output_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"task_count={manifest['task_count']} output_manifest_json={output_path}")


if __name__ == "__main__":
    main()
