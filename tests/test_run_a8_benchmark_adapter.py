from pathlib import Path
import importlib.util
import json
import sys


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "run_a8_benchmark_adapter.py"
    spec = importlib.util.spec_from_file_location("run_a8_benchmark_adapter", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_result_for_benchmark_routes_swe_family():
    module = _load_runner_module()

    packet = module.build_result_for_benchmark(
        {"resolved": 48, "total": 60},
        benchmark="swe_rebench",
        source_path="summary.json",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "swe_rebench"
    assert packet["metrics"]["resolve_rate"] == 0.8
    assert packet["metrics"]["task_count"] == 60


def test_build_result_for_benchmark_routes_generic_family():
    module = _load_runner_module()

    packet = module.build_result_for_benchmark(
        {"gold_medal_rate": 0.24},
        benchmark="mle_bench",
        source_path="summary.json",
        conservative_comparison_report=True,
    )

    assert packet["benchmark"] == "mle_bench"
    assert packet["metrics"]["gold_medal_rate"] == 0.24


def test_run_a8_benchmark_adapter_cli_runs_command_and_writes_verified_packet(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_runner_module()
    summary_path = tmp_path / "summary.json"
    command_path = tmp_path / "command.json"
    output_path = tmp_path / "packet.json"
    log_path = tmp_path / "runner_log.json"
    command_path.write_text(
        json.dumps(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(r'"
                    + str(summary_path)
                    + "').write_text('{\"rating\": 3001}', encoding='utf-8')"
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_a8_benchmark_adapter.py",
            "--benchmark",
            "codeforces",
            "--runner-command-json",
            str(command_path),
            "--runner-log-json",
            str(log_path),
            "--summary-json",
            str(summary_path),
            "--output-json",
            str(output_path),
            "--conservative-comparison-report",
        ],
    )

    module.main()

    packet = json.loads(output_path.read_text(encoding="utf-8"))
    runner_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert packet["benchmark"] == "codeforces"
    assert packet["metrics"]["rating_equivalent"] == 3001
    assert packet["source"]["runner"]["returncode"] == 0
    assert runner_log["returncode"] == 0
    assert f"output_json={output_path}" in capsys.readouterr().out


def test_run_a8_benchmark_adapter_cli_rejects_failing_runner(tmp_path, monkeypatch):
    module = _load_runner_module()
    summary_path = tmp_path / "summary.json"
    command_path = tmp_path / "command.json"
    output_path = tmp_path / "packet.json"
    command_path.write_text(json.dumps([sys.executable, "-c", "raise SystemExit(7)"]), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_a8_benchmark_adapter.py",
            "--benchmark",
            "codeforces",
            "--runner-command-json",
            str(command_path),
            "--summary-json",
            str(summary_path),
            "--output-json",
            str(output_path),
            "--conservative-comparison-report",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "benchmark runner failed returncode=7" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
