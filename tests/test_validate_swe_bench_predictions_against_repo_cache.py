from pathlib import Path
import importlib.util
import json
import subprocess


def _load_validator_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "validate_swe_bench_predictions_against_repo_cache.py"
    spec = importlib.util.spec_from_file_location("validate_swe_bench_predictions_against_repo_cache", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    return tmp_path / "repos", commit


def _make_repo_with_pytest(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    test_source = repo_root / "tests" / "test_module.py"
    source.parent.mkdir(parents=True)
    test_source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    test_source.write_text(
        "from pkg.module import value\n\n"
        "def test_value():\n"
        "    assert value() == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    return tmp_path / "repos", commit


def test_validate_predictions_against_repo_cache_accepts_applyable_patch(tmp_path):
    module = _load_validator_module()
    repo_cache_root, commit = _make_repo(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    repo_root = repo_cache_root / "owner" / "repo"
    (repo_root / "pkg" / "module.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    subprocess.run(["git", "checkout", "--", "pkg/module.py"], cwd=repo_root, check=True)
    predictions_path.write_text(
        json.dumps({"instance_id": "owner__repo-1", "model_name_or_path": "m", "model_patch": patch}) + "\n",
        encoding="utf-8",
    )

    result = module.validate_predictions_against_repo_cache(
        [{"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": commit}],
        predictions_jsonl=str(predictions_path),
        repo_cache_root=str(repo_cache_root),
    )

    assert result["all_apply_check_passed"] is True
    assert result["apply_check_passed_count"] == 1


def test_validate_predictions_against_repo_cache_rejects_bad_patch(tmp_path):
    module = _load_validator_module()
    repo_cache_root, commit = _make_repo(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "owner__repo-1",
                "model_name_or_path": "m",
                "model_patch": "diff --git a/pkg/missing.py b/pkg/missing.py\n--- a/pkg/missing.py\n+++ b/pkg/missing.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.validate_predictions_against_repo_cache(
        [{"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": commit}],
        predictions_jsonl=str(predictions_path),
        repo_cache_root=str(repo_cache_root),
    )

    assert result["all_apply_check_passed"] is False
    assert result["results"][0]["reason"] == "apply_check_failed"


def test_validate_predictions_against_repo_cache_accepts_empty_patch_noop(tmp_path):
    module = _load_validator_module()
    repo_cache_root, commit = _make_repo(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": "owner__repo-1", "model_name_or_path": "m", "model_patch": ""}) + "\n",
        encoding="utf-8",
    )

    result = module.validate_predictions_against_repo_cache(
        [{"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": commit}],
        predictions_jsonl=str(predictions_path),
        repo_cache_root=str(repo_cache_root),
    )

    assert result["all_apply_check_passed"] is True
    assert result["apply_check_passed_count"] == 1
    assert result["results"][0]["reason"] == "empty_patch_noop"


def test_validate_predictions_against_repo_cache_can_run_declared_pytest_tests(tmp_path):
    module = _load_validator_module()
    repo_cache_root, commit = _make_repo_with_pytest(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    repo_root = repo_cache_root / "owner" / "repo"
    (repo_root / "pkg" / "module.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    subprocess.run(["git", "checkout", "--", "pkg/module.py"], cwd=repo_root, check=True)
    predictions_path.write_text(
        json.dumps({"instance_id": "owner__repo-1", "model_name_or_path": "m", "model_patch": patch}) + "\n",
        encoding="utf-8",
    )

    result = module.validate_predictions_against_repo_cache(
        [
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": commit,
                "fail_to_pass": ["tests/test_module.py::test_value"],
            }
        ],
        predictions_jsonl=str(predictions_path),
        repo_cache_root=str(repo_cache_root),
        run_tests=True,
    )

    assert result["all_apply_check_passed"] is True
    assert result["tests_requested"] is True
    assert result["all_tests_passed"] is True
    assert result["tests_passed_count"] == 1
    assert result["results"][0]["tests_passed"] is True


def test_validate_predictions_against_repo_cache_reports_declared_pytest_failure(tmp_path):
    module = _load_validator_module()
    repo_cache_root, commit = _make_repo_with_pytest(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    repo_root = repo_cache_root / "owner" / "repo"
    (repo_root / "pkg" / "module.py").write_text("def value():\n    return 3\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
    subprocess.run(["git", "checkout", "--", "pkg/module.py"], cwd=repo_root, check=True)
    predictions_path.write_text(
        json.dumps({"instance_id": "owner__repo-1", "model_name_or_path": "m", "model_patch": patch}) + "\n",
        encoding="utf-8",
    )

    result = module.validate_predictions_against_repo_cache(
        [
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": commit,
                "fail_to_pass": ["tests/test_module.py::test_value"],
            }
        ],
        predictions_jsonl=str(predictions_path),
        repo_cache_root=str(repo_cache_root),
        run_tests=True,
    )

    assert result["all_apply_check_passed"] is True
    assert result["all_tests_passed"] is False
    assert result["tests_passed_count"] == 0
    assert result["results"][0]["reason"] == "tests_failed"


def test_checkout_stderr_indicates_incomplete_worktree():
    module = _load_validator_module()

    assert module._checkout_stderr_indicates_incomplete_worktree(
        "error: unable to read sha1 file of astropy/__init__.py"
    )
    assert module._checkout_stderr_indicates_incomplete_worktree("error: invalid object 100644 abc for 'x.py'")
    assert not module._checkout_stderr_indicates_incomplete_worktree("HEAD is now at abc base")
    assert module._checkout_stderr_indicates_incomplete_worktree(
        "error: invalid object 100644 abc for 'pkg/module.py'",
        ["pkg/module.py"],
    )
    assert not module._checkout_stderr_indicates_incomplete_worktree(
        "error: invalid object 100644 abc for '.circleci/config.yml'",
        ["pkg/module.py"],
    )


def test_sparse_checkout_paths_use_exact_files():
    module = _load_validator_module()

    assert module._sparse_checkout_paths(
        [
            "astropy/io/ascii/rst.py",
            "astropy/io/ascii/tests/test_rst.py::test_rst_with_header_rows",
            "../bad",
        ]
    ) == ["astropy/io/ascii/rst.py", "astropy/io/ascii/tests/test_rst.py"]
    assert module._sparse_checkout_paths(["setup.py"]) == ["setup.py"]


def test_sparse_checkout_patterns_add_root_test_files_only_when_running_tests():
    module = _load_validator_module()

    assert module._sparse_checkout_patterns(["pkg/module.py"], run_tests=False) == ["pkg/module.py"]
    assert module._sparse_checkout_patterns(["pkg/module.py"], run_tests=True) == [
        "pkg/module.py",
        "pkg/__init__.py",
        "pkg/version.py",
        "pkg/conftest.py",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "conftest.py",
    ]
    assert module._sparse_checkout_patterns(["pkg/tests/test_module.py::test_value"], run_tests=True) == [
        "pkg/tests/test_module.py",
        "pkg/tests/__init__.py",
        "pkg/__init__.py",
        "pkg/version.py",
        "pkg/conftest.py",
        "pkg/tests/common.py",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "conftest.py",
    ]
