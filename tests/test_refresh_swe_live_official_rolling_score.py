from pathlib import Path
import importlib.util
import json
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "refresh_swe_live_official_rolling_score.py"
    spec = importlib.util.spec_from_file_location("refresh_swe_live_official_rolling_score", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_official_rolling_score_writes_zero_packet_when_no_verified_patches(tmp_path, monkeypatch):
    module = _load_module()

    def fake_verify_swe_patch_jobs(**_kwargs):
        return {
            "success": True,
            "verified_patch_count": 0,
            "skipped_nonterminal_count": 0,
            "failures": [],
        }

    def fail_collect(*_args, **_kwargs):
        raise AssertionError("collect should not run when no verified patches exist")

    monkeypatch.setattr(module, "verify_swe_patch_jobs", fake_verify_swe_patch_jobs)
    monkeypatch.setattr(module, "collect_swe_predictions", fail_collect)
    output_root = tmp_path / "score"

    result = module.refresh_official_rolling_score(
        queue_json=tmp_path / "queue.json",
        queue_manifest=tmp_path / "queue_manifest.json",
        prediction_task_manifest=tmp_path / "prediction_tasks.json",
        workspace_root=tmp_path / "workspace",
        output_root=output_root,
        swe_bench_live_root=tmp_path / "SWE-bench-Live",
        python=sys.executable,
        workers=1,
        launch_evaluator=True,
        skip_semantic_artifact_check=False,
    )

    assert result["selected_patch_count"] == 0
    assert result["prediction_count"] == 0
    assert result["evaluator_pid"] is None
    assert (output_root / "predictions.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((output_root / "preds.json").read_text(encoding="utf-8")) == {}
    assert json.loads((output_root / "evaluation_results" / "results.json").read_text(encoding="utf-8"))[
        "submitted"
    ] == 0
