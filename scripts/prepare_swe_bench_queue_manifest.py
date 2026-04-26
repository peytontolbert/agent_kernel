from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import re
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _safe_task_id(instance_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in instance_id)
    return f"swe_patch_{safe.strip('_') or 'instance'}"


def _candidate_file_success_command(candidate_files: list[str]) -> str:
    base = "test -s patch.diff && grep -Eq '^(diff --git |--- )' patch.diff"
    base = (
        f"{base} && ! grep -Eiq "
        "'placeholder|satisfy the verifier|dummy|TODO|Some comment|New line added' patch.diff"
    )
    if not candidate_files:
        return base
    pattern = "|".join(re.escape(path) for path in candidate_files)
    pattern = pattern.replace("'", "'\"'\"'")
    return f"{base} && grep -Eq '{pattern}' patch.diff"


def _safe_source_context_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized or normalized == "." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe source context path: {path!r}")
    return f"source_context/{normalized}"


def _safe_source_lines_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized or normalized == "." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe source context path: {path!r}")
    return f"source_lines/{normalized}.lines"


def _line_numbered_content(content: str) -> str:
    lines = content.splitlines()
    if content.endswith("\n"):
        lines.append("")
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{index:>{width}}: {line}" for index, line in enumerate(lines, start=1)) + "\n"


def _source_context_entries(source_context: list[object]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for context in source_context:
        if not isinstance(context, dict):
            continue
        path = str(context.get("path", "")).strip()
        content = str(context.get("content", ""))
        if not path or not content:
            continue
        workspace_path = _safe_source_context_path(path)
        line_path = _safe_source_lines_path(path)
        if workspace_path in seen:
            continue
        seen.add(workspace_path)
        entries.append(
            {
                "path": path,
                "workspace_path": workspace_path,
                "line_path": line_path,
                "content": content,
                "line_content": _line_numbered_content(content),
            }
        )
    return entries


def build_swe_queue_manifest(
    prediction_task_manifest: dict[str, Any],
    *,
    workspace_prefix: str = "swe_bench_predictions",
) -> dict[str, Any]:
    tasks = prediction_task_manifest.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prediction task manifest must contain non-empty tasks list")
    queue_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be an object")
        instance_id = str(task.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(f"task {index} missing instance_id")
        problem = str(task.get("problem_statement", "")).strip()
        if not problem:
            raise ValueError(f"task {instance_id} missing problem_statement")
        repo = str(task.get("repo", "")).strip()
        base_commit = str(task.get("base_commit", "")).strip()
        repo_cache_root = str(task.get("repo_cache_root", "")).strip()
        hints = str(task.get("hints_text", "")).strip()
        candidate_files = task.get("candidate_files", [])
        if not isinstance(candidate_files, list):
            candidate_files = []
        candidate_files = [
            str(path).strip()
            for path in candidate_files
            if str(path).strip()
        ]
        source_context = task.get("source_context", [])
        if not isinstance(source_context, list):
            source_context = []
        source_entries = _source_context_entries(source_context)
        fail_to_pass = task.get("fail_to_pass", [])
        pass_to_pass = task.get("pass_to_pass", [])
        task_id = _safe_task_id(instance_id)
        prompt_parts = [
            f"SWE-bench instance: {instance_id}",
            f"Repository: {repo or 'unknown'}",
            f"Base commit: {base_commit or 'unknown'}",
            (
                "Execution contract: create patch.diff directly in the current workspace. "
                "Do not cd to /testbed, clone repositories, fetch network resources, or write any file other than patch.diff."
            ),
            (
                "Patch validity contract: patch.diff must be a real unified diff that applies cleanly to the listed base commit. "
                "Use exact paths and context lines from the source excerpts; do not invent files, placeholder functions, test stubs, or fake comments."
            ),
            (
                "Command contract: do not run git, ls, find, or read /testbed paths. "
                "Use only the provided candidate source files already present in this workspace, "
                "for example sed -n '1,120p' astropy/<path> or sed -n '1,120p' source_context/astropy/<path>, "
                "then write patch.diff. A diff against any path not listed as a likely relevant file is invalid."
            ),
            "Problem statement:",
            problem,
        ]
        if hints:
            prompt_parts.extend(["Hints:", hints])
        if candidate_files:
            prompt_parts.extend(
                [
                    "Likely relevant files from dataset metadata:",
                    "\n".join(f"- {path}" for path in candidate_files),
                ]
            )
        if source_entries:
            prompt_parts.extend(
                [
                    "Workspace source-context files:",
                    "\n".join(
                        f"- {entry['path']} is available at {entry['path']}, {entry['workspace_path']}, and line-numbered {entry['line_path']}"
                        for entry in source_entries
                    ),
                    (
                        "Inspect these files with commands such as sed -n '1,120p' astropy/path.py or "
                        "sed -n '1,120p' source_lines/astropy/path.py.lines. Use the line-numbered files to choose exact hunk anchors. "
                        "Do not use git show or /testbed paths."
                    ),
                ]
            )
        source_parts: list[str] = []
        for entry in source_entries:
            path = entry["path"]
            content = entry["content"]
            context = next(
                (
                    item
                    for item in source_context
                    if isinstance(item, dict) and str(item.get("path", "")).strip() == path
                ),
                {},
            )
            truncated = " (truncated)" if bool(context.get("truncated", False)) else ""
            source_parts.append(f"### {path}{truncated}\n{content}")
        if source_parts:
            prompt_parts.extend(["Source excerpts:", "\n\n".join(source_parts)])
        if fail_to_pass:
            prompt_parts.extend(["Fail-to-pass tests:", json.dumps(fail_to_pass, sort_keys=True)])
        if pass_to_pass:
            prompt_parts.extend(["Pass-to-pass tests:", json.dumps(pass_to_pass, sort_keys=True)])
        prompt_parts.append(
            "Write only a minimal applyable unified diff to patch.diff. If executing a command, use printf/cat redirection to create patch.diff in the current directory. Do not write prose, markdown fences, placeholders, fake imports, '# This is a test file', or explanations."
        )
        queue_tasks.append(
            {
                "task_id": task_id,
                "prompt": "\n\n".join(prompt_parts),
                "workspace_subdir": f"{workspace_prefix.strip('/') or 'swe_bench_predictions'}/{task_id}",
                "setup_commands": [],
                "success_command": _candidate_file_success_command(candidate_files),
                "expected_files": ["patch.diff"],
                "max_steps": 12,
                "metadata": {
                    "benchmark_family": "repository",
                    "capability": "swe_bench_patch_generation",
                    "task_origin": "external_manifest",
                    "swe_bench_prediction_task": True,
                    "swe_instance_id": instance_id,
                    "swe_repo": repo,
                    "swe_base_commit": base_commit,
                    "swe_candidate_files": candidate_files,
                    "swe_patch_workspace_relpath": "patch.diff",
                    "swe_patch_output_path": str(task.get("patch_path", "")).strip(),
                    "swe_repo_cache_root": repo_cache_root,
                    "setup_file_contents": {
                        path: entry["content"]
                        for entry in source_entries
                        for path in (entry["path"], entry["workspace_path"])
                    }
                    | {
                        entry["line_path"]: entry["line_content"]
                        for entry in source_entries
                    },
                    "semantic_verifier": {
                        "kind": "swe_patch_apply_check",
                        "repo": repo,
                        "base_commit": base_commit,
                        "repo_cache_root": repo_cache_root,
                        "patch_path": "patch.diff",
                        "expected_changed_paths": candidate_files,
                    }
                    if repo_cache_root
                    else {},
                    "workflow_guard": {
                        "managed_paths": [
                            "patch.diff",
                            *[entry["workspace_path"] for entry in source_entries],
                        ],
                    },
                },
            }
        )
    return {
        "manifest_kind": "swe_bench_patch_generation_queue_manifest",
        "manifest_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_task_count": len(tasks),
        "tasks": queue_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--workspace-prefix", default="swe_bench_predictions")
    args = parser.parse_args()

    manifest = build_swe_queue_manifest(
        _read_json(Path(args.prediction_task_manifest)),
        workspace_prefix=args.workspace_prefix,
    )
    output_path = Path(args.output_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"task_count={len(manifest['tasks'])} output_manifest_json={output_path}")


if __name__ == "__main__":
    main()
