from pathlib import Path
import importlib.util
import json
import subprocess


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_swe_bench_patch_jobs.py"
    spec = importlib.util.spec_from_file_location("verify_swe_bench_patch_jobs", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(tmp_path: Path, *, state: str = "completed", outcome: str = "success", write_patch: bool = True):
    queue_manifest = tmp_path / "queue_manifest.json"
    queue_json = tmp_path / "queue.json"
    workspace_root = tmp_path / "workspace"
    workspace = workspace_root / "probe" / "swe_patch_instance"
    workspace.mkdir(parents=True)
    if write_patch:
        (workspace / "patch.diff").write_text(
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            encoding="utf-8",
        )
    queue_manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "swe_patch_instance",
                        "workspace_subdir": "probe/swe_patch_instance",
                        "metadata": {"swe_instance_id": "repo__pkg-1"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "job-1",
                        "task_id": "swe_patch_instance",
                        "state": state,
                        "outcome": outcome,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return queue_json, queue_manifest, workspace_root


def test_verify_swe_patch_jobs_accepts_successful_jobs_with_patches(tmp_path):
    module = _load_module()
    queue_json, queue_manifest, workspace_root = _write_fixture(tmp_path)

    result = module.verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
    )

    assert result["success"] is True
    assert result["verified_patch_count"] == 1
    assert result["successful_instance_ids"] == ["repo__pkg-1"]
    assert result["retry_instance_ids"] == []
    assert result["failures"] == []


def test_verify_swe_patch_jobs_accepts_safe_stop_as_abstention(tmp_path):
    module = _load_module()
    queue_json, queue_manifest, workspace_root = _write_fixture(
        tmp_path,
        state="safe_stop",
        outcome="safe_stop",
        write_patch=False,
    )

    result = module.verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
    )

    assert result["success"] is True
    assert result["failures"] == []
    assert result["abstained_patch_count"] == 1
    assert result["abstained_instance_ids"] == ["repo__pkg-1"]
    assert result["retry_instance_ids"] == ["repo__pkg-1"]
    assert result["failed_jobs"] == []
    assert result["abstained_jobs"] == [
        {
            "task_id": "swe_patch_instance",
            "instance_id": "repo__pkg-1",
            "job_id": "job-1",
            "state": "safe_stop",
            "outcome": "safe_stop",
            "reason": "terminal_abstention",
        }
    ]


def test_verify_swe_patch_jobs_marks_semantic_artifact_failure_retryable(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repos" / "owner" / "repo"
    source = repo_root / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    source.write_text(
        "class Reader:\n"
        "    def __init__(self, rows=None):\n"
        "        self.rows = rows\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    queue_manifest = tmp_path / "queue_manifest.json"
    queue_json = tmp_path / "queue.json"
    workspace_root = tmp_path / "workspace"
    workspace = workspace_root / "probe" / "swe_patch_instance"
    workspace.mkdir(parents=True)
    (workspace / "patch.diff").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1,3 +1,3 @@\n"
        " class Reader:\n"
        "     def __init__(self, rows=None):\n"
        "-        self.rows = rows\n"
        "+        yield from rows\n",
        encoding="utf-8",
    )
    queue_manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "swe_patch_instance",
                        "workspace_subdir": "probe/swe_patch_instance",
                        "expected_files": ["patch.diff"],
                        "metadata": {
                            "swe_instance_id": "repo__pkg-1",
                            "semantic_verifier": {
                                "kind": "swe_patch_apply_check",
                                "repo": "owner/repo",
                                "base_commit": commit,
                                "repo_cache_root": str(tmp_path / "repos"),
                                "patch_path": "patch.diff",
                                "expected_changed_paths": ["pkg/module.py"],
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "job-1",
                        "task_id": "swe_patch_instance",
                        "state": "completed",
                        "outcome": "success",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = module.verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
    )

    assert result["success"] is True
    assert result["verified_patch_count"] == 0
    assert result["abstained_patch_count"] == 1
    assert result["retry_instance_ids"] == ["repo__pkg-1"]
    assert result["abstained_jobs"][0]["reason"] == "semantic_artifact_failure"
    assert "SWE patch leaves invalid __init__ generators" in result["abstained_jobs"][0]["verification_reasons"][0]


def test_verify_swe_patch_jobs_rejects_nonterminal_unresolved_job(tmp_path):
    module = _load_module()
    queue_json, queue_manifest, workspace_root = _write_fixture(
        tmp_path,
        state="in_progress",
        outcome="",
        write_patch=False,
    )

    result = module.verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
    )

    assert result["success"] is False
    assert result["failures"] == ["swe_patch_instance state=in_progress outcome=-"]
    assert result["failed_jobs"][0]["reason"] == "job_not_successful"


def test_verify_swe_patch_jobs_can_skip_nonterminal_for_rolling_score(tmp_path):
    module = _load_module()
    queue_json, queue_manifest, workspace_root = _write_fixture(
        tmp_path,
        state="in_progress",
        outcome="",
        write_patch=False,
    )

    result = module.verify_swe_patch_jobs(
        queue_json=queue_json,
        queue_manifest=queue_manifest,
        workspace_root=workspace_root,
        allow_nonterminal=True,
    )

    assert result["success"] is True
    assert result["verified_patch_count"] == 0
    assert result["skipped_nonterminal_count"] == 1
    assert result["skipped_nonterminal_jobs"][0]["reason"] == "nonterminal_skipped"
    assert result["retry_instance_ids"] == []
