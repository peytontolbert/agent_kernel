from pathlib import Path
import importlib.util
import json
import sys


def _load_materializer_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "materialize_a8_benchmark_from_spec.py"
    spec = importlib.util.spec_from_file_location("materialize_a8_benchmark_from_spec", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary_only_spec(tmp_path: Path) -> dict[str, object]:
    summary_path = tmp_path / "codeforces_summary.json"
    packet_path = tmp_path / "codeforces_packet.json"
    summary_path.write_text(json.dumps({"rating": 3002}), encoding="utf-8")
    return {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "codeforces",
        "ready_to_run": False,
        "prerequisites": [
            {
                "blocking": True,
                "kind": "account",
                "name": "codeforces_account",
                "proof_path": str(tmp_path / "codeforces_account.json"),
                "required_env": ["CODEFORCES_HANDLE"],
                "satisfied_by": "env_or_proof",
            }
        ],
        "runner": {"kind": "summary_only", "summary_source": str(summary_path)},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": str(summary_path),
            "output_packet_json": str(packet_path),
            "conservative_comparison_report": True,
        },
        "open_limits": ["Fixture spec."],
    }


def test_materialize_a8_benchmark_from_summary_spec(tmp_path):
    module = _load_materializer_module()
    spec = _summary_only_spec(tmp_path)

    packet = module.materialize_a8_benchmark_from_spec(spec, spec_path="spec.json", allow_not_ready=True)

    assert packet["benchmark"] == "codeforces"
    assert packet["metrics"]["rating_equivalent"] == 3002
    assert packet["source"]["benchmark_run_spec_path"] == "spec.json"
    assert Path(packet["source"]["output_packet_json"]).exists()


def test_materialize_a8_benchmark_from_spec_rejects_not_ready_by_default(tmp_path):
    module = _load_materializer_module()

    try:
        module.materialize_a8_benchmark_from_spec(_summary_only_spec(tmp_path))
    except ValueError as exc:
        assert "spec is not ready_to_run" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_materialize_a8_benchmark_from_spec_uses_adapter_spec_json(tmp_path):
    module = _load_materializer_module()
    spec = _summary_only_spec(tmp_path)
    summary_path = Path(spec["adapter"]["summary_json"])
    summary_path.write_text(json.dumps({"custom": {"rating": "3007"}}), encoding="utf-8")
    adapter_spec_path = tmp_path / "codeforces_adapter_spec.json"
    adapter_spec_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "report_kind": "a8_benchmark_adapter_spec",
                "benchmark": "codeforces",
                "metrics": {
                    "rating_equivalent": {
                        "type": "integer",
                        "aliases": ["custom.rating"],
                        "required": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    spec["adapter"]["adapter_spec_json"] = str(adapter_spec_path)

    packet = module.materialize_a8_benchmark_from_spec(spec, spec_path="spec.json", allow_not_ready=True)

    assert packet["metrics"]["rating_equivalent"] == 3007
    assert packet["source"]["adapter_spec_path"] == str(adapter_spec_path)


def test_materialize_a8_benchmark_from_swe_results_json(tmp_path):
    module = _load_materializer_module()
    results_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.json"
    packet_path = tmp_path / "packet.json"
    results_path.write_text(
        json.dumps({"resolved_ids": ["a", "b", "c"], "unresolved_ids": ["d"]}),
        encoding="utf-8",
    )
    spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "swe_bench_verified",
        "ready_to_run": False,
        "runner": {
            "kind": "swebench_harness",
            "harness_root": str(tmp_path),
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "predictions_path": str(tmp_path / "missing_predictions.jsonl"),
            "run_id": "fixture",
            "results_json": str(results_path),
        },
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": str(summary_path),
            "output_packet_json": str(packet_path),
            "conservative_comparison_report": True,
        },
        "open_limits": ["Fixture spec."],
    }

    packet = module.materialize_a8_benchmark_from_spec(spec, allow_not_ready=True)

    assert packet["benchmark"] == "swe_bench_verified"
    assert packet["metrics"]["resolved_count"] == 3
    assert packet["metrics"]["task_count"] == 4
    assert packet["metrics"]["resolve_rate"] == 0.75
    assert json.loads(summary_path.read_text(encoding="utf-8"))["resolved_count"] == 3


def test_materialize_a8_benchmark_from_spec_cli_writes_packet(tmp_path, monkeypatch, capsys):
    module = _load_materializer_module()
    spec = _summary_only_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_a8_benchmark_from_spec.py",
            "--spec-json",
            str(spec_path),
            "--allow-not-ready",
        ],
    )

    module.main()

    assert "benchmark=codeforces status=verified" in capsys.readouterr().out
