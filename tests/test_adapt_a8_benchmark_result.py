from pathlib import Path
import importlib.util
import json
import sys


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "adapt_a8_benchmark_result.py"
    spec = importlib.util.spec_from_file_location("adapt_a8_benchmark_result", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_codeforces_result_from_rating_alias():
    module = _load_adapter_module()

    packet = module.build_a8_benchmark_result(
        {"metrics": {"codeforces_rating": "3012"}},
        benchmark="codeforces",
        source_path="cf.json",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "codeforces"
    assert packet["metrics"]["rating_equivalent"] == 3012
    assert packet["metrics"]["conservative_comparison_report"] is True
    assert packet["source"]["source_path"] == "cf.json"


def test_build_mle_bench_result_from_gold_rate():
    module = _load_adapter_module()

    packet = module.build_a8_benchmark_result(
        {"gold_rate": 0.23},
        benchmark="mle_bench",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "mle_bench"
    assert packet["metrics"]["gold_medal_rate"] == 0.23


def test_build_re_bench_result_from_human_expert_win_rate():
    module = _load_adapter_module()

    packet = module.build_a8_benchmark_result(
        {"metrics": {"human_expert_win_rate": "0.55"}},
        benchmark="re_bench",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "re_bench"
    assert packet["metrics"]["human_expert_win_rate"] == 0.55


def test_build_sustained_window_result_from_summary():
    module = _load_adapter_module()

    packet = module.build_a8_benchmark_result(
        {
            "window_count": 3,
            "task_count": 100,
            "baseline_win_rate": 0.84,
            "baseline_win_rate_lower_bound": 0.63,
            "unfamiliar_slices": 5,
            "long_horizon_slices": 3,
            "baseline_comparison_slices": 5,
            "regression_rate": 0.01,
        },
        benchmark="sustained_coding_window",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "sustained_coding_window"
    assert packet["metrics"]["window_count"] == 3
    assert packet["metrics"]["task_count"] == 100
    assert packet["metrics"]["strong_human_baseline_win_rate"] == 0.84
    assert packet["metrics"]["strong_human_baseline_win_rate_lower_bound"] == 0.63
    assert packet["metrics"]["unfamiliar_domain_slice_count"] == 5
    assert packet["metrics"]["long_horizon_transfer_slice_count"] == 3
    assert packet["metrics"]["strong_baseline_comparison_slice_count"] == 5
    assert packet["metrics"]["regression_rate"] == 0.01


def test_build_recursive_compounding_result_requires_verified_true():
    module = _load_adapter_module()

    try:
        module.build_a8_benchmark_result(
            {"retained_gain_runs": 5, "window_count": 3},
            benchmark="recursive_compounding",
            conservative_comparison_report=True,
        )
    except ValueError as exc:
        assert "verified_recursive_compounding true" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_recursive_compounding_result_from_verified_summary():
    module = _load_adapter_module()

    packet = module.build_a8_benchmark_result(
        {
            "retained_runs": 5,
            "windows": 3,
            "verified_recursive_compounding": True,
        },
        benchmark="recursive_compounding",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "recursive_compounding"
    assert packet["metrics"]["retained_gain_runs"] == 5
    assert packet["metrics"]["window_count"] == 3
    assert packet["metrics"]["verified_recursive_compounding"] is True


def test_adapt_a8_benchmark_result_cli_writes_packet(tmp_path, monkeypatch, capsys):
    module = _load_adapter_module()
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "packet.json"
    summary_path.write_text(json.dumps({"rating": 3000}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapt_a8_benchmark_result.py",
            "--benchmark",
            "codeforces",
            "--summary-json",
            str(summary_path),
            "--output-json",
            str(output_path),
            "--conservative-comparison-report",
        ],
    )

    module.main()

    packet = json.loads(output_path.read_text(encoding="utf-8"))
    assert packet["benchmark"] == "codeforces"
    assert packet["metrics"]["rating_equivalent"] == 3000
    assert f"output_json={output_path}" in capsys.readouterr().out
