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


def test_validate_a8_benchmark_spec_rejects_placeholder_swe_rebench_dataset(tmp_path):
    module = _load_validator_module()
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "swe_rebench",
        "ready_to_run": False,
        "runner": {
            "kind": "swebench_harness",
            "harness_root": str(tmp_path),
            "dataset_name": "SWE-ReBench",
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "run_id": "fixture",
        },
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": str(tmp_path / "summary.json"),
            "output_packet_json": str(tmp_path / "packet.json"),
            "conservative_comparison_report": True,
        },
        "open_limits": ["Fixture placeholder spec."],
    }

    failures = module.validate_a8_benchmark_spec(payload)

    assert "runner.dataset_name must be the confirmed official SWE-ReBench dataset identifier" in failures


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


def test_validate_a8_benchmark_spec_requires_codeforces_account_gate():
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
            "conservative_comparison_report": True,
        },
        "open_limits": ["Should fail."],
    }

    failures = module.validate_a8_benchmark_spec(payload)

    assert "codeforces specs must declare an account prerequisite" in failures


def test_validate_a8_benchmark_spec_rejects_ready_codeforces_without_account(tmp_path, monkeypatch):
    module = _load_validator_module()
    monkeypatch.delenv("CODEFORCES_HANDLE", raising=False)
    payload = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "codeforces",
        "ready_to_run": True,
        "prerequisites": [
            {
                "blocking": True,
                "kind": "account",
                "name": "codeforces_account",
                "proof_path": str(tmp_path / "missing_account.json"),
                "required_env": ["CODEFORCES_HANDLE"],
                "satisfied_by": "env_or_proof",
            }
        ],
        "runner": {"kind": "summary_only", "summary_source": str(tmp_path / "summary.json")},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": str(tmp_path / "summary.json"),
            "output_packet_json": str(tmp_path / "packet.json"),
            "conservative_comparison_report": True,
        },
        "open_limits": ["Should fail."],
    }

    failures = module.validate_a8_benchmark_spec(payload)

    assert "codeforces ready_to_run requires CODEFORCES_HANDLE or account proof_path" in failures


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
