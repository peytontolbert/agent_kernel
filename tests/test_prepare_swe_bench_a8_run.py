from pathlib import Path
import importlib.util
import json
import sys


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_a8_run.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_a8_run", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_swe_bench_command_uses_harness_module_and_safe_argv():
    module = _load_prepare_module()

    command = module.build_swe_bench_command(
        python_bin="/env/bin/python",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        predictions_path="predictions.jsonl",
        run_id="run-1",
        max_workers=2,
        timeout=900,
        cache_level="base",
        namespace="swebench",
        report_dir="evaluation_results",
        instance_ids=["repo__pkg-1", "repo__pkg-2"],
    )

    assert command[:3] == ["/env/bin/python", "-m", "swebench.harness.run_evaluation"]
    assert "--dataset_name" in command
    assert "princeton-nlp/SWE-bench_Verified" in command
    assert "--predictions_path" in command
    assert "predictions.jsonl" in command
    assert "--instance_ids" in command
    assert "repo__pkg-1" in command
    assert "repo__pkg-2" in command


def test_summarize_swe_bench_results_from_counts():
    module = _load_prepare_module()

    summary = module.summarize_swe_bench_results(
        {"resolved_count": 80, "total_instances": 100},
        source_path="results.json",
    )

    assert summary["report_kind"] == "official_swe_bench_summary"
    assert summary["source_path"] == "results.json"
    assert summary["resolved_count"] == 80
    assert summary["task_count"] == 100
    assert summary["resolve_rate"] == 0.8


def test_summarize_swe_bench_results_from_id_lists():
    module = _load_prepare_module()

    summary = module.summarize_swe_bench_results(
        {
            "resolved_ids": ["a", "b"],
            "unresolved_ids": ["c"],
            "error_ids": ["d"],
        }
    )

    assert summary["resolved_count"] == 2
    assert summary["task_count"] == 4
    assert summary["resolve_rate"] == 0.5


def test_prepare_swe_bench_command_cli_writes_command_and_spec(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    command_path = tmp_path / "command.json"
    spec_path = tmp_path / "spec.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "command",
            "--benchmark",
            "swe_bench_verified",
            "--predictions-path",
            "predictions.jsonl",
            "--run-id",
            "verified-run",
            "--output-command-json",
            str(command_path),
            "--output-spec-json",
            str(spec_path),
            "--summary-json",
            "summary.json",
            "--output-packet-json",
            "packet.json",
        ],
    )

    module.main()

    command = json.loads(command_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert command[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
    assert "princeton-nlp/SWE-bench_Verified" in command
    assert spec["report_kind"] == "a8_swe_bench_runner_spec"
    assert spec["benchmark"] == "swe_bench_verified"
    assert f"command_json={command_path}" in capsys.readouterr().out


def test_prepare_swe_bench_summarize_cli_writes_summary(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    results_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(json.dumps({"resolved": ["a", "b", "c"], "total": 5}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "summarize",
            "--results-json",
            str(results_path),
            "--output-summary-json",
            str(summary_path),
        ],
    )

    module.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["resolved_count"] == 3
    assert summary["task_count"] == 5
    assert summary["resolve_rate"] == 0.6
    assert f"summary_json={summary_path}" in capsys.readouterr().out
