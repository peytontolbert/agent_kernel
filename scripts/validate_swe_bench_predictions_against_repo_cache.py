from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
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


def _patch_changed_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        path = ""
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
        elif line.startswith("+++ "):
            path = line[4:].strip()
        if path.startswith("b/"):
            path = path[2:]
        if path and path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def _test_ids(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fail_to_pass", "pass_to_pass"):
        raw = item.get(key, [])
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = [stripped]
            raw = decoded
        if not isinstance(raw, list):
            continue
        for value in raw:
            normalized = str(value).strip()
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def _sparse_checkout_paths(paths: list[str]) -> list[str]:
    sparse_paths: list[str] = []
    for path in paths:
        normalized = str(path).split("::", 1)[0].strip().strip("/")
        if not normalized or normalized.startswith("../") or "/../" in normalized:
            continue
        if normalized not in sparse_paths:
            sparse_paths.append(normalized)
    return sparse_paths


def _sparse_checkout_patterns(paths: list[str], *, run_tests: bool) -> list[str]:
    sparse_paths = _sparse_checkout_paths(paths)
    if not run_tests:
        return sparse_paths
    for path in list(sparse_paths):
        current = Path(path).parent
        while str(current) not in ("", "."):
            init_path = str(current / "__init__.py")
            if init_path not in sparse_paths:
                sparse_paths.append(init_path)
            current = current.parent
        top_package = Path(path).parts[0] if Path(path).parts else ""
        for bootstrap_path in (f"{top_package}/version.py", f"{top_package}/conftest.py"):
            if top_package and bootstrap_path not in sparse_paths:
                sparse_paths.append(bootstrap_path)
        if "/tests/" in path:
            common_path = str(Path(path).parent / "common.py")
            if common_path not in sparse_paths:
                sparse_paths.append(common_path)
    for path in ("pyproject.toml", "setup.cfg", "setup.py", "conftest.py"):
        if path not in sparse_paths:
            sparse_paths.append(path)
    return sparse_paths


def _test_failure_reason(exit_code: int, stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}"
    if exit_code == 4 or "ERROR collecting" in combined or "found no collectors" in combined:
        return "test_environment_failed"
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        return "test_environment_failed"
    return "tests_failed"


def _checkout_stderr_indicates_incomplete_worktree(stderr: str, required_paths: list[str] | None = None) -> bool:
    normalized = str(stderr)
    if "unable to checkout working tree" in normalized:
        return True
    markers = ("unable to read sha1 file", "invalid object")
    if not any(marker in normalized for marker in markers):
        return False
    if not required_paths:
        return True
    required = set(_sparse_checkout_paths(required_paths))
    if not required:
        return True
    referenced_paths = set(re.findall(r"for '([^']+)'", normalized))
    referenced_paths.update(re.findall(r"of ([^\n]+)", normalized))
    return any(path.strip().strip("/") in required for path in referenced_paths)


def validate_predictions_against_repo_cache(
    dataset: dict[str, Any] | list[Any],
    *,
    predictions_jsonl: str,
    repo_cache_root: str,
    limit: int = 0,
    instance_ids: list[str] | None = None,
    run_tests: bool = False,
    test_timeout_seconds: int = 300,
    sparse_checkout: bool = True,
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
                "tests_requested": False,
                "tests_passed": False,
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
            if sparse_checkout:
                sparse_inputs = [*_patch_changed_paths(patch_text)]
                if run_tests:
                    sparse_inputs.extend(_test_ids(item))
                sparse_patterns = _sparse_checkout_patterns(sparse_inputs, run_tests=run_tests)
                if sparse_patterns:
                    sparse_init = subprocess.run(
                        ["git", "-C", str(worktree), "sparse-checkout", "init", "--no-cone"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    sparse_set = subprocess.run(
                        ["git", "-C", str(worktree), "sparse-checkout", "set", "--", *sparse_patterns],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if sparse_init.returncode != 0 or sparse_set.returncode != 0:
                        result["reason"] = "sparse_checkout_failed"
                        result["stderr"] = "\n".join(
                            part.strip()
                            for part in (sparse_init.stderr, sparse_set.stderr)
                            if part.strip()
                        )
                        results.append(result)
                        continue
                    result["sparse_checkout_paths"] = sparse_patterns
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
            if _checkout_stderr_indicates_incomplete_worktree(checkout.stderr, sparse_patterns if sparse_checkout else None):
                result["reason"] = "checkout_incomplete"
                result["stderr"] = checkout.stderr.strip()[-4000:]
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
            elif run_tests:
                tests = _test_ids(item)
                result["tests_requested"] = True
                result["test_count"] = len(tests)
                if not tests:
                    result["reason"] = "no_declared_tests"
                else:
                    apply_patch = subprocess.run(
                        ["git", "-C", str(worktree), "apply", str(patch_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if apply_patch.returncode != 0:
                        result["reason"] = "apply_failed_after_check"
                        result["stderr"] = apply_patch.stderr.strip()
                    else:
                        try:
                            test_run = subprocess.run(
                                [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *tests],
                                cwd=worktree,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=max(1, int(test_timeout_seconds)),
                            )
                            result["tests_passed"] = test_run.returncode == 0
                            result["test_exit_code"] = test_run.returncode
                            result["test_stdout_tail"] = test_run.stdout[-4000:]
                            result["test_stderr_tail"] = test_run.stderr[-4000:]
                            if test_run.returncode != 0:
                                result["reason"] = _test_failure_reason(
                                    test_run.returncode,
                                    test_run.stdout,
                                    test_run.stderr,
                                )
                        except subprocess.TimeoutExpired as exc:
                            result["tests_passed"] = False
                            result["test_exit_code"] = 124
                            result["test_stdout_tail"] = str(exc.stdout or "")[-4000:]
                            result["test_stderr_tail"] = str(exc.stderr or "")[-4000:]
                            result["reason"] = "tests_timed_out"
            results.append(result)
            shutil.rmtree(worktree, ignore_errors=True)
            if limit > 0 and selected_count >= limit:
                break
    return {
        "prediction_count": len(predictions),
        "selected_instance_count": len(results),
        "apply_check_passed_count": sum(1 for result in results if result["apply_check_passed"]),
        "tests_requested": run_tests,
        "tests_passed_count": sum(1 for result in results if result.get("tests_passed") is True),
        "all_tests_passed": (not run_tests) or (bool(results) and all(result.get("tests_passed") is True for result in results)),
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
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--test-timeout-seconds", type=int, default=300)
    parser.add_argument("--no-sparse-checkout", action="store_true")
    args = parser.parse_args()

    result = validate_predictions_against_repo_cache(
        _read_json_or_jsonl(Path(args.dataset_json)),
        predictions_jsonl=args.predictions_jsonl,
        repo_cache_root=args.repo_cache_root,
        limit=int(args.limit),
        instance_ids=args.instance_ids,
        run_tests=bool(args.run_tests),
        test_timeout_seconds=int(args.test_timeout_seconds),
        sparse_checkout=not bool(args.no_sparse_checkout),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_apply_check_passed"] or not result["all_tests_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
