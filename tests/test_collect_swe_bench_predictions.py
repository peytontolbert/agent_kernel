from pathlib import Path
import importlib.util
import json
import sys


def _load_collector_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "collect_swe_bench_predictions.py"
    spec = importlib.util.spec_from_file_location("collect_swe_bench_predictions", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch() -> str:
    return (
        "diff --git a/pkg/file.py b/pkg/file.py\n"
        "--- a/pkg/file.py\n"
        "+++ b/pkg/file.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def _manifests(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    workspace_root = tmp_path / "workspace"
    workspace = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1"
    workspace.mkdir(parents=True)
    (workspace / "patch.diff").write_text(_patch(), encoding="utf-8")
    prediction_manifest = {
        "prediction_manifest": {
            "base_dir": str(tmp_path / "patches"),
            "predictions": [
                {
                    "instance_id": "django__django-1",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "django__django-1.diff",
                }
            ],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_django__django-1",
                "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-1",
                "metadata": {"swe_instance_id": "django__django-1"},
            }
        ]
    }
    return prediction_manifest, queue_manifest, workspace_root


def test_collect_swe_predictions_copies_workspace_patches_and_writes_jsonl(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
    )

    assert result["copied_patch_count"] == 1
    assert result["prediction_count"] == 1
    assert (tmp_path / "patches" / "django__django-1.diff").read_text(encoding="utf-8") == _patch()
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[0]["instance_id"] == "django__django-1"
    assert records[0]["model_patch"] == _patch()


def test_collect_swe_predictions_filters_to_verified_successes(tmp_path):
    module = _load_collector_module()
    workspace_root = tmp_path / "workspace"
    for instance_id in ("django__django-1", "django__django-2"):
        workspace = workspace_root / "swe_bench_predictions" / f"swe_patch_{instance_id}"
        workspace.mkdir(parents=True)
        (workspace / "patch.diff").write_text(_patch(), encoding="utf-8")
    prediction_manifest = {
        "prediction_manifest": {
            "base_dir": str(tmp_path / "patches"),
            "predictions": [
                {
                    "instance_id": "django__django-1",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "django__django-1.diff",
                },
                {
                    "instance_id": "django__django-2",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "django__django-2.diff",
                },
            ],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_django__django-1",
                "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-1",
                "metadata": {"swe_instance_id": "django__django-1"},
            },
            {
                "task_id": "swe_patch_django__django-2",
                "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-2",
                "metadata": {"swe_instance_id": "django__django-2"},
            },
        ]
    }
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
        patch_job_verification={"successful_instance_ids": ["django__django-2"]},
    )

    assert result["selected_instance_ids"] == ["django__django-2"]
    assert result["prediction_count"] == 1
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["instance_id"] for record in records] == ["django__django-2"]
    assert not (tmp_path / "patches" / "django__django-1.diff").exists()
    assert (tmp_path / "patches" / "django__django-2.diff").exists()


def test_collect_swe_predictions_requires_fresh_verified_patch_provenance(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    patch_path = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1" / "patch.diff"
    stat = patch_path.stat()
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
        patch_job_verification={
            "report_kind": "swe_bench_patch_job_verification",
            "successful_instance_ids": ["django__django-1"],
            "verified_patches": [
                {
                    "task_id": "swe_patch_django__django-1",
                    "instance_id": "django__django-1",
                    "job_id": "job-1",
                    "patch_path": str(patch_path),
                    "patch_sha256": module._sha256_file(patch_path),
                    "patch_size": stat.st_size,
                    "patch_mtime_ns": stat.st_mtime_ns,
                    "verified_at": "2026-05-17T00:00:00+00:00",
                }
            ],
        },
    )

    assert result["copied_patch_count"] == 1
    assert result["selected_instance_ids"] == ["django__django-1"]


def test_collect_swe_predictions_uses_verified_patch_path_for_mixed_workspace_roots(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    stale_patch = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1" / "patch.diff"
    stale_patch.unlink()
    actual_workspace = tmp_path / "resumed_workspace" / "swe_patch_django__django-1"
    actual_workspace.mkdir(parents=True)
    actual_patch = actual_workspace / "patch.diff"
    actual_patch.write_text(_patch(), encoding="utf-8")
    stat = actual_patch.stat()

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(tmp_path / "predictions.jsonl"),
        patch_job_verification={
            "report_kind": "swe_bench_patch_job_verification",
            "successful_instance_ids": ["django__django-1"],
            "verified_patches": [
                {
                    "task_id": "swe_patch_django__django-1",
                    "instance_id": "django__django-1",
                    "job_id": "job-1",
                    "patch_path": str(actual_patch),
                    "patch_sha256": module._sha256_file(actual_patch),
                    "patch_size": stat.st_size,
                    "patch_mtime_ns": stat.st_mtime_ns,
                    "verified_at": "2026-05-17T00:00:00+00:00",
                }
            ],
        },
    )

    assert result["copied_patch_count"] == 1
    assert (tmp_path / "patches" / "django__django-1.diff").read_text(encoding="utf-8") == _patch()


def test_collect_swe_predictions_rejects_stale_verified_patch_hash(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    patch_path = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1" / "patch.diff"
    stat = patch_path.stat()

    try:
        module.collect_swe_predictions(
            prediction_manifest,
            queue_manifest,
            workspace_root=str(workspace_root),
            output_jsonl=str(tmp_path / "predictions.jsonl"),
            patch_job_verification={
                "report_kind": "swe_bench_patch_job_verification",
                "successful_instance_ids": ["django__django-1"],
                "verified_patches": [
                    {
                        "task_id": "swe_patch_django__django-1",
                        "instance_id": "django__django-1",
                        "job_id": "job-1",
                        "patch_path": str(patch_path),
                        "patch_sha256": "0" * 64,
                        "patch_size": stat.st_size,
                        "patch_mtime_ns": stat.st_mtime_ns,
                        "verified_at": "2026-05-17T00:00:00+00:00",
                    }
                ],
            },
        )
    except ValueError as exc:
        assert "patch sha256 changed after verification" in str(exc)
    else:
        raise AssertionError("stale verified patch hash should be rejected")


def test_collect_swe_predictions_rejects_stale_abstention_when_patch_now_exists(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)

    try:
        module.collect_swe_predictions(
            prediction_manifest,
            queue_manifest,
            workspace_root=str(workspace_root),
            output_jsonl=str(tmp_path / "predictions.jsonl"),
            patch_job_verification={
                "report_kind": "swe_bench_patch_job_verification",
                "abstained_instance_ids": ["django__django-1"],
                "abstained_jobs": [
                    {
                        "task_id": "swe_patch_django__django-1",
                        "instance_id": "django__django-1",
                        "job_id": "job-older",
                        "reason": "terminal_abstention",
                    }
                ],
            },
        )
    except ValueError as exc:
        assert "stale or lacks patch provenance" in str(exc)
    else:
        raise AssertionError("stale abstention over an existing patch should be rejected")


def test_collect_swe_predictions_writes_noop_for_abstained_jobs(tmp_path):
    module = _load_collector_module()
    workspace_root = tmp_path / "workspace"
    workspace = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1"
    workspace.mkdir(parents=True)
    prediction_manifest = {
        "prediction_manifest": {
            "base_dir": str(tmp_path / "patches"),
            "predictions": [
                {
                    "instance_id": "django__django-1",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "django__django-1.diff",
                }
            ],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_django__django-1",
                "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-1",
                "metadata": {"swe_instance_id": "django__django-1"},
            }
        ]
    }
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
        patch_job_verification={"abstained_instance_ids": ["django__django-1"]},
    )

    assert result["copied_patch_count"] == 0
    assert result["abstained_prediction_count"] == 1
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "instance_id": "django__django-1",
            "model_name_or_path": "agentkernel",
            "model_patch": "",
        }
    ]
    assert not (tmp_path / "patches" / "django__django-1.diff").exists()


def test_collect_swe_predictions_can_exclude_abstained_jobs_from_rolling_subset(tmp_path):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    workspace = workspace_root / "swe_bench_predictions" / "swe_patch_django__django-2"
    workspace.mkdir(parents=True)
    prediction_manifest["prediction_manifest"]["predictions"].append(
        {
            "instance_id": "django__django-2",
            "model_name_or_path": "agentkernel",
            "patch_path": "django__django-2.diff",
        }
    )
    queue_manifest["tasks"].append(
        {
            "task_id": "swe_patch_django__django-2",
            "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-2",
            "metadata": {"swe_instance_id": "django__django-2"},
        }
    )
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
        patch_job_verification={
            "successful_instance_ids": ["django__django-1"],
            "abstained_instance_ids": ["django__django-2"],
        },
        include_abstained=False,
    )

    assert result["copied_patch_count"] == 1
    assert result["abstained_prediction_count"] == 0
    assert result["selected_instance_ids"] == ["django__django-1"]
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["instance_id"] for record in records] == ["django__django-1"]


def test_collect_swe_predictions_infers_noop_when_patch_missing(tmp_path):
    module = _load_collector_module()
    workspace_root = tmp_path / "workspace"
    (workspace_root / "swe_bench_predictions" / "swe_patch_django__django-1").mkdir(parents=True)
    prediction_manifest = {
        "prediction_manifest": {
            "base_dir": str(tmp_path / "patches"),
            "predictions": [
                {
                    "instance_id": "django__django-1",
                    "model_name_or_path": "agentkernel",
                    "patch_path": "django__django-1.diff",
                }
            ],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_django__django-1",
                "workspace_subdir": "swe_bench_predictions/swe_patch_django__django-1",
                "metadata": {"swe_instance_id": "django__django-1"},
            }
        ]
    }
    output_jsonl = tmp_path / "predictions.jsonl"

    result = module.collect_swe_predictions(
        prediction_manifest,
        queue_manifest,
        workspace_root=str(workspace_root),
        output_jsonl=str(output_jsonl),
    )

    assert result["copied_patch_count"] == 0
    assert result["abstained_prediction_count"] == 1
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[0]["model_patch"] == ""


def test_collect_swe_predictions_cli_writes_jsonl(tmp_path, monkeypatch, capsys):
    module = _load_collector_module()
    prediction_manifest, queue_manifest, workspace_root = _manifests(tmp_path)
    prediction_path = tmp_path / "prediction_tasks.json"
    queue_path = tmp_path / "queue_manifest.json"
    output_jsonl = tmp_path / "predictions.jsonl"
    prediction_path.write_text(json.dumps(prediction_manifest), encoding="utf-8")
    queue_path.write_text(json.dumps(queue_manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_swe_bench_predictions.py",
            "--prediction-task-manifest",
            str(prediction_path),
            "--queue-manifest",
            str(queue_path),
            "--workspace-root",
            str(workspace_root),
            "--output-jsonl",
            str(output_jsonl),
        ],
    )

    module.main()

    assert output_jsonl.exists()
    assert f"output_jsonl={output_jsonl}" in capsys.readouterr().out
