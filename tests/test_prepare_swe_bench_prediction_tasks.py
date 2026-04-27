from pathlib import Path
import importlib.util
import json
import subprocess
import sys


def _load_tasks_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_prediction_tasks.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_prediction_tasks", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset() -> list[dict[str, object]]:
    return [
        {
            "instance_id": "django__django-1",
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Fix timezone parsing regression.",
            "hints_text": "Look at parser edge cases.",
            "patch": "diff --git a/django/utils/dateparse.py b/django/utils/dateparse.py\n--- a/django/utils/dateparse.py\n+++ b/django/utils/dateparse.py\n",
            "test_patch": "diff --git a/tests/test_time.py b/tests/test_time.py\n--- a/tests/test_time.py\n+++ b/tests/test_time.py\n",
            "FAIL_TO_PASS": ["tests/test_time.py::test_parse"],
            "PASS_TO_PASS": ["tests/test_time.py::test_existing"],
        },
        {
            "instance_id": "sympy__sympy-2",
            "repo": "sympy/sympy",
            "base_commit": "def456",
            "problem_statement": "Fix simplification crash.",
        },
    ]


def test_build_swe_prediction_task_manifest_selects_tasks_and_patch_paths(tmp_path):
    module = _load_tasks_module()

    manifest = module.build_swe_prediction_task_manifest(
        _dataset(),
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel-vllm",
        limit=1,
    )

    assert manifest["report_kind"] == "swe_bench_prediction_task_manifest"
    assert manifest["task_count"] == 1
    task = manifest["tasks"][0]
    assert task["instance_id"] == "django__django-1"
    assert task["repo"] == "django/django"
    assert task["base_commit"] == "abc123"
    assert task["repo_cache_root"] == ""
    assert task["candidate_files"] == ["django/utils/dateparse.py", "tests/test_time.py"]
    assert task["patch_path"].endswith("django__django-1.diff")
    assert manifest["prediction_manifest"]["predictions"][0]["patch_path"] == "django__django-1.diff"


def test_paths_from_unified_diff_supports_git_and_file_headers():
    module = _load_tasks_module()

    paths = module._paths_from_unified_diff(
        "diff --git a/pkg/source.py b/pkg/source.py\n"
        "--- a/pkg/source.py\n"
        "+++ b/pkg/source.py\n"
        "diff --git a/tests/test_source.py b/tests/test_source.py\n"
        "--- /dev/null\n"
        "+++ b/tests/test_source.py\n"
    )

    assert paths == ["pkg/source.py", "tests/test_source.py"]


def test_build_swe_prediction_task_manifest_includes_local_source_context(tmp_path):
    module = _load_tasks_module()
    repo_root = tmp_path / "repos" / "django" / "django"
    source = repo_root / "django" / "utils" / "dateparse.py"
    source.parent.mkdir(parents=True)
    source.write_text("def parse_datetime(value):\n    return value\n", encoding="utf-8")

    manifest = module.build_swe_prediction_task_manifest(
        _dataset(),
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        limit=1,
        repo_cache_root=str(tmp_path / "repos"),
    )

    context = manifest["tasks"][0]["source_context"]
    assert context == [
        {
            "path": "django/utils/dateparse.py",
            "content": "def parse_datetime(value):\n    return value\n",
            "truncated": False,
        }
    ]


def test_build_swe_prediction_task_manifest_reads_source_at_base_commit(tmp_path):
    module = _load_tasks_module()
    repo_root = tmp_path / "repos" / "django" / "django"
    source = repo_root / "django" / "utils" / "dateparse.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text("def parse_datetime(value):\n    return 'base'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source.write_text("def parse_datetime(value):\n    return 'worktree'\n", encoding="utf-8")
    dataset = _dataset()
    dataset[0]["base_commit"] = base_commit

    manifest = module.build_swe_prediction_task_manifest(
        dataset,
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        limit=1,
        repo_cache_root=str(tmp_path / "repos"),
    )

    context = manifest["tasks"][0]["source_context"]
    assert context[0]["content"] == "def parse_datetime(value):\n    return 'base'\n"


def test_build_swe_prediction_task_manifest_adds_focused_windows_for_large_files(tmp_path):
    module = _load_tasks_module()
    repo_root = tmp_path / "repos" / "sympy" / "sympy"
    source = repo_root / "sympy" / "utilities" / "iterables.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# prefix\n"
        + "\n".join(f"# filler {index}" for index in range(200))
        + "\n\ndef uniq(seq):\n    return list(seq)\n",
        encoding="utf-8",
    )
    dataset = [
        {
            "instance_id": "sympy__sympy-18835",
            "repo": "sympy/sympy",
            "base_commit": "",
            "problem_statement": "uniq should reject mutated input.",
            "patch": (
                "diff --git a/sympy/utilities/iterables.py b/sympy/utilities/iterables.py\n"
                "--- a/sympy/utilities/iterables.py\n"
                "+++ b/sympy/utilities/iterables.py\n"
            ),
            "FAIL_TO_PASS": ["sympy/utilities/tests/test_iterables.py::test_uniq"],
        }
    ]

    manifest = module.build_swe_prediction_task_manifest(
        dataset,
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        repo_cache_root=str(tmp_path / "repos"),
        max_source_context_bytes=128,
    )

    context = manifest["tasks"][0]["source_context"]
    assert context[0]["truncated"] is True
    focused = [entry for entry in context if entry.get("context_kind") == "focused_window"]
    assert focused
    assert focused[0]["focus_term"] == "uniq"
    assert "def uniq(seq):" in focused[0]["content"]
    assert focused[0]["line_start"] <= focused[0]["line_end"]


def test_build_swe_prediction_task_manifest_default_context_covers_late_edit_sites(tmp_path):
    module = _load_tasks_module()
    repo_root = tmp_path / "repos" / "scikit-learn" / "scikit-learn"
    source = repo_root / "sklearn" / "ensemble" / "_iforest.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# prefix\n"
        + "\n".join(f"# filler {index:04d} {'x' * 70}" for index in range(220))
        + "\n        self.offset_ = np.percentile(self.score_samples(X), 100.0 * self.contamination)\n",
        encoding="utf-8",
    )
    dataset = [
        {
            "instance_id": "scikit-learn__scikit-learn-25931",
            "repo": "scikit-learn/scikit-learn",
            "base_commit": "",
            "problem_statement": (
                "X does not have valid feature names, but IsolationForest was fitted with feature names "
                "when contamination is not auto."
            ),
            "patch": (
                "diff --git a/sklearn/ensemble/_iforest.py b/sklearn/ensemble/_iforest.py\n"
                "--- a/sklearn/ensemble/_iforest.py\n"
                "+++ b/sklearn/ensemble/_iforest.py\n"
            ),
        }
    ]

    manifest = module.build_swe_prediction_task_manifest(
        dataset,
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        repo_cache_root=str(tmp_path / "repos"),
    )

    context = manifest["tasks"][0]["source_context"]
    assert "self.offset_ = np.percentile(self.score_samples(X)" in context[0]["content"]


def test_build_swe_prediction_task_manifest_filters_instance_ids(tmp_path):
    module = _load_tasks_module()

    manifest = module.build_swe_prediction_task_manifest(
        _dataset(),
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        instance_ids=["sympy__sympy-2"],
    )

    assert manifest["task_count"] == 1
    assert manifest["tasks"][0]["instance_id"] == "sympy__sympy-2"


def test_build_swe_prediction_task_manifest_filters_repositories(tmp_path):
    module = _load_tasks_module()
    dataset = [
        *_dataset(),
        {
            "instance_id": "sphinx-doc__sphinx-10323",
            "repo": "sphinx-doc/sphinx",
            "base_commit": "fedcba",
            "problem_statement": "Fix Sphinx rendering.",
        },
    ]

    manifest = module.build_swe_prediction_task_manifest(
        dataset,
        output_patch_dir=str(tmp_path / "patches"),
        model_name_or_path="agentkernel",
        repos_filter=["sphinx-doc/sphinx"],
    )

    assert manifest["task_count"] == 1
    assert manifest["tasks"][0]["instance_id"] == "sphinx-doc__sphinx-10323"
    assert manifest["tasks"][0]["repo"] == "sphinx-doc/sphinx"


def test_build_swe_prediction_task_manifest_rejects_missing_requested_id(tmp_path):
    module = _load_tasks_module()

    try:
        module.build_swe_prediction_task_manifest(
            _dataset(),
            output_patch_dir=str(tmp_path / "patches"),
            model_name_or_path="agentkernel",
            instance_ids=["missing__repo-3"],
        )
    except ValueError as exc:
        assert "requested instance_ids not found: missing__repo-3" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_prepare_swe_bench_prediction_tasks_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    module = _load_tasks_module()
    dataset_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    dataset_path.write_text("".join(json.dumps(item) + "\n" for item in _dataset()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_prediction_tasks.py",
            "--dataset-json",
            str(dataset_path),
            "--output-manifest-json",
            str(manifest_path),
            "--output-patch-dir",
            str(tmp_path / "patches"),
            "--model-name-or-path",
            "agentkernel",
            "--limit",
            "2",
        ],
    )

    module.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_count"] == 2
    assert f"output_manifest_json={manifest_path}" in capsys.readouterr().out


def test_read_arrow_dataset_when_datasets_package_available(tmp_path):
    module = _load_tasks_module()
    try:
        from datasets import Dataset
    except ImportError:
        return
    dataset_dir = tmp_path / "dataset"
    Dataset.from_list(_dataset()).save_to_disk(str(dataset_dir))
    arrow_path = next(dataset_dir.glob("*.arrow"))

    records = module._read_json_or_jsonl(arrow_path)

    assert isinstance(records, list)
    assert records[0]["instance_id"] == "django__django-1"
