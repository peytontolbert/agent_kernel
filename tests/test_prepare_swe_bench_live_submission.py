from pathlib import Path
import importlib.util
import json


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_live_submission.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_live_submission", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_live_predictions_converts_jsonl_records_to_instance_map():
    module = _load_module()

    predictions = module.build_live_predictions(
        [
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            }
        ]
    )

    assert sorted(predictions) == ["repo__pkg-1"]
    assert predictions["repo__pkg-1"]["model_name_or_path"] == "agentkernel"
    assert "diff --git" in predictions["repo__pkg-1"]["model_patch"]


def test_package_live_submission_writes_leaderboard_artifacts(tmp_path):
    module = _load_module()
    predictions_jsonl = tmp_path / "predictions.jsonl"
    predictions_jsonl.write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"resolved": ["repo__pkg-1"], "total": 1}), encoding="utf-8")
    output_dir = tmp_path / "submission"

    manifest = module.package_live_submission(
        predictions_jsonl=predictions_jsonl,
        results_json=results_json,
        output_dir=output_dir,
        model_name="agentkernel-model",
        system_name="Agent Kernel",
        subset="verified",
        run_command="python -m evaluation.evaluation ...",
    )

    preds = json.loads((output_dir / "preds.json").read_text(encoding="utf-8"))
    copied_results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert manifest["prediction_count"] == 1
    assert preds["repo__pkg-1"]["model_name_or_path"] == "agentkernel"
    assert copied_results["total"] == 1
    assert "Agent Kernel" in readme
    assert "python -m evaluation.evaluation" in readme
