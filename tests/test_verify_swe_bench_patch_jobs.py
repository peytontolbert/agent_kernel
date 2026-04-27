from pathlib import Path
import importlib.util
import json


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


def test_verify_swe_patch_jobs_rejects_safe_stop_and_missing_patch(tmp_path):
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

    assert result["success"] is False
    assert result["failures"] == ["swe_patch_instance state=safe_stop outcome=safe_stop"]
    assert result["retry_instance_ids"] == ["repo__pkg-1"]
    assert result["failed_jobs"] == [
        {
            "task_id": "swe_patch_instance",
            "instance_id": "repo__pkg-1",
            "job_id": "job-1",
            "state": "safe_stop",
            "outcome": "safe_stop",
            "reason": "job_not_successful",
        }
    ]
