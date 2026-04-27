from pathlib import Path
import importlib.util
import json
import sys


def _load_cache_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_repo_cache.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_repo_cache", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset() -> list[dict[str, object]]:
    return [
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
        },
        {
            "instance_id": "astropy__astropy-14182",
            "repo": "astropy/astropy",
            "base_commit": "def456",
        },
        {
            "instance_id": "sphinx-doc__sphinx-10323",
            "repo": "sphinx-doc/sphinx",
            "base_commit": "fedcba",
        },
    ]


def test_prepare_repo_cache_dry_run_reports_clone_target(tmp_path, capsys):
    module = _load_cache_module()

    result = module.prepare_repo_cache(
        _dataset(),
        repo_cache_root=str(tmp_path / "repos"),
        dry_run=True,
    )

    assert result["repo_count"] == 2
    assert result["selected_instance_count"] == 3
    repo = result["repositories"][0]
    assert repo["repo"] == "astropy/astropy"
    assert repo["path"].endswith("astropy/astropy")
    assert repo["commit_count"] == 2
    assert "git\", \"clone" in capsys.readouterr().out


def test_prepare_repo_cache_filters_limit_and_instance_ids(tmp_path):
    module = _load_cache_module()

    result = module.prepare_repo_cache(
        _dataset(),
        repo_cache_root=str(tmp_path / "repos"),
        dry_run=True,
        limit=1,
        instance_ids=["astropy__astropy-14182"],
    )

    assert result["selected_instance_count"] == 1
    assert result["repositories"][0]["commit_count"] == 1


def test_prepare_repo_cache_filters_repositories(tmp_path):
    module = _load_cache_module()

    result = module.prepare_repo_cache(
        _dataset(),
        repo_cache_root=str(tmp_path / "repos"),
        dry_run=True,
        repos_filter=["sphinx-doc/sphinx"],
    )

    assert result["selected_instance_count"] == 1
    assert result["repo_count"] == 1
    assert result["repositories"][0]["repo"] == "sphinx-doc/sphinx"


def test_prepare_swe_bench_repo_cache_cli_writes_summary(tmp_path, monkeypatch):
    module = _load_cache_module()
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "summary.json"
    dataset_path.write_text("".join(json.dumps(item) + "\n" for item in _dataset()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_repo_cache.py",
            "--dataset-json",
            str(dataset_path),
            "--repo-cache-root",
            str(tmp_path / "repos"),
            "--dry-run",
            "--limit",
            "1",
            "--output-json",
            str(output_path),
        ],
    )

    module.main()

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["repo_count"] == 1
    assert summary["selected_instance_count"] == 1
