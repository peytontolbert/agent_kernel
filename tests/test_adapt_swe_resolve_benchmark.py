from pathlib import Path
import importlib.util
import json
import sys


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "adapt_swe_resolve_benchmark.py"
    spec = importlib.util.spec_from_file_location("adapt_swe_resolve_benchmark", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_swe_resolve_benchmark_result_from_counts():
    module = _load_adapter_module()

    packet = module.build_swe_resolve_benchmark_result(
        {"report_kind": "official_swe_bench_verified_run", "resolved": 80, "total": 100},
        benchmark="swe_bench_verified",
        source_path="run.json",
        conservative_comparison_report=True,
    )

    assert packet["report_kind"] == "a8_benchmark_result"
    assert packet["benchmark"] == "swe_bench_verified"
    assert packet["metrics"]["resolve_rate"] == 0.8
    assert packet["metrics"]["resolved_count"] == 80
    assert packet["metrics"]["task_count"] == 100
    assert packet["metrics"]["conservative_comparison_report"] is True
    assert packet["source"]["source_path"] == "run.json"


def test_build_swe_resolve_benchmark_result_from_nested_metrics_and_lower_bound():
    module = _load_adapter_module()

    packet = module.build_swe_resolve_benchmark_result(
        {
            "metrics": {
                "resolved_count": "36",
                "task_count": "60",
                "resolve_rate_lower_bound": 0.52,
            }
        },
        benchmark="swe_rebench",
    )

    assert packet["benchmark"] == "swe_rebench"
    assert packet["metrics"]["resolve_rate"] == 0.6
    assert packet["metrics"]["resolve_rate_lower_bound"] == 0.52


def test_build_swe_resolve_benchmark_result_rejects_impossible_counts():
    module = _load_adapter_module()

    try:
        module.build_swe_resolve_benchmark_result(
            {"resolved": 11, "total": 10},
            benchmark="swe_bench_verified",
        )
    except ValueError as exc:
        assert "resolved count must be between zero and total" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_adapt_swe_resolve_benchmark_cli_writes_packet(tmp_path, monkeypatch, capsys):
    module = _load_adapter_module()
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "packet.json"
    summary_path.write_text(json.dumps({"resolved_count": 48, "task_count": 60}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapt_swe_resolve_benchmark.py",
            "--benchmark",
            "swe_rebench",
            "--summary-json",
            str(summary_path),
            "--output-json",
            str(output_path),
            "--conservative-comparison-report",
        ],
    )

    module.main()

    packet = json.loads(output_path.read_text(encoding="utf-8"))
    assert packet["benchmark"] == "swe_rebench"
    assert packet["metrics"]["resolve_rate"] == 0.8
    assert f"output_json={output_path}" in capsys.readouterr().out
