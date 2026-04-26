from pathlib import Path
import importlib.util
import json
import sys


def _load_validator_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "validate_a8_benchmark_specs.py"
    spec = importlib.util.spec_from_file_location("validate_a8_benchmark_specs", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_a8_benchmark_spec_accepts_repo_templates():
    module = _load_validator_module()
    repo_root = Path(__file__).resolve().parents[1]
    spec_paths = sorted((repo_root / "config" / "a8_benchmark_run_specs").glob("*.json"))

    assert spec_paths
    for spec_path in spec_paths:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        assert module.validate_a8_benchmark_spec(payload, spec_path=str(spec_path)) == []


def test_validate_a8_benchmark_spec_rejects_ready_swe_without_predictions():
    module = _load_validator_module()
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (repo_root / "config" / "a8_benchmark_run_specs" / "swe_bench_verified_local.json").read_text(
            encoding="utf-8"
        )
    )
    payload["ready_to_run"] = True

    failures = module.validate_a8_benchmark_spec(payload)

    assert any("runner.predictions_path does not exist" in failure for failure in failures)


def test_validate_a8_benchmark_spec_checks_ready_swe_prediction_format(tmp_path):
    module = _load_validator_module()
    repo_root = Path(__file__).resolve().parents[1]
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = json.loads(
        (repo_root / "config" / "a8_benchmark_run_specs" / "swe_bench_verified_local.json").read_text(
            encoding="utf-8"
        )
    )
    payload["ready_to_run"] = True
    payload["runner"]["predictions_path"] = str(predictions_path)

    failures = module.validate_a8_benchmark_spec(payload)

    assert any("runner.predictions_path invalid" in failure for failure in failures)


def test_validate_a8_benchmark_spec_rejects_non_conservative_adapter():
    module = _load_validator_module()
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "codeforces",
        "ready_to_run": False,
        "runner": {"kind": "summary_only", "summary_source": "summary.json"},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": "summary.json",
            "output_packet_json": "packet.json",
            "conservative_comparison_report": False,
        },
        "open_limits": ["Should fail."],
    }

    failures = module.validate_a8_benchmark_spec(payload)

    assert "adapter.conservative_comparison_report must be true for A8 evidence" in failures


def test_validate_a8_benchmark_specs_cli_accepts_repo_templates(monkeypatch, capsys):
    module = _load_validator_module()
    repo_root = Path(__file__).resolve().parents[1]
    spec_paths = sorted((repo_root / "config" / "a8_benchmark_run_specs").glob("*.json"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_a8_benchmark_specs.py", *[str(path) for path in spec_paths]],
    )

    module.main()

    output = capsys.readouterr().out
    assert "verified_a8_benchmark_spec=" in output
    assert str(repo_root / "config" / "a8_benchmark_run_specs" / "swe_bench_verified_local.json") in output
