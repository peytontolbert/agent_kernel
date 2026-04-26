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
