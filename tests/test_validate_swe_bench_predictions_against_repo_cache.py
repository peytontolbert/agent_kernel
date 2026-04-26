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
