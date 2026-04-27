from pathlib import Path
import importlib.util
import json
import sys


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_autonomous_benchmark_harness.py"
    spec = importlib.util.spec_from_file_location("prepare_autonomous_benchmark_harness", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_autonomous_benchmark_harness.py"
    spec = importlib.util.spec_from_file_location("run_autonomous_benchmark_harness", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_autonomous_harness_from_all_repo_a8_specs():
    module = _load_prepare_module()
    runner_module = _load_runner_module()
    repo_root = Path(__file__).resolve().parents[1]
    spec_paths = sorted((repo_root / "config" / "a8_benchmark_run_specs").glob("*.json"))

    assert spec_paths
    for spec_path in spec_paths:
        harness = module.build_autonomous_harness_from_run_spec(
            json.loads(spec_path.read_text(encoding="utf-8")),
            spec_path=str(spec_path),
            python_bin=sys.executable,
        )
        assert harness["report_kind"] == "autonomous_benchmark_harness_spec"
        assert harness["benchmark"]
        assert harness["phases"]
        assert runner_module.validate_harness_spec(harness) == []


def test_summary_only_harness_adapts_summary_packet():
    module = _load_prepare_module()
    spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "codeforces",
        "ready_to_run": False,
        "runner": {"kind": "summary_only", "summary_source": "benchmarks/codeforces/summary.json"},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": "benchmarks/codeforces/summary.json",
            "output_packet_json": "benchmarks/codeforces/a8_benchmark_result.json",
            "conservative_comparison_report": True,
        },
        "open_limits": ["template"],
    }

    harness = module.build_autonomous_harness_from_run_spec(spec, spec_path="codeforces.json")

    assert harness["phases"][0]["name"] == "adapt_summary_packet"
    assert "--benchmark" in harness["phases"][0]["argv"]
    assert "codeforces" in harness["phases"][0]["argv"]
    assert harness["phases"][0]["preflight_argv"] == harness["phases"][0]["argv"] + ["--validate-only"]
    assert harness["artifacts"]["summary_source"] == "benchmarks/codeforces/summary.json"
    assert harness["phases"][0]["required_inputs"] == ["benchmarks/codeforces/summary.json"]
    assert harness["prerequisites"][0]["name"] == "codeforces_account"


def test_swebench_harness_wraps_existing_predictions_and_official_runner():
    module = _load_prepare_module()
    spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "swe_bench_verified",
        "ready_to_run": False,
        "runner": {
            "kind": "swebench_harness",
            "harness_root": "/tmp/swe-bench",
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "predictions_path": "benchmarks/predictions/swe_bench_verified_predictions.jsonl",
            "run_id": "verified-run",
            "max_workers": 1,
            "timeout": 1800,
            "cache_level": "env",
            "namespace": "swebench",
            "report_dir": "benchmarks/swe_bench_verified/evaluation_results",
            "results_json": "benchmarks/swe_bench_verified/evaluation_results/results.json",
        },
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": "benchmarks/swe_bench_verified/summary.json",
            "output_packet_json": "benchmarks/swe_bench_verified/a8_benchmark_result.json",
            "conservative_comparison_report": True,
        },
        "open_limits": ["template"],
    }

    harness = module.build_autonomous_harness_from_run_spec(spec, spec_path="swe.json")

    phase_names = [phase["name"] for phase in harness["phases"]]
    assert phase_names == [
        "validate_predictions",
        "official_harness",
        "materialize_results",
        "summarize_results",
        "adapt_a8_packet",
    ]
    assert harness["phases"][0]["required_inputs"] == ["benchmarks/predictions/swe_bench_verified_predictions.jsonl"]
    assert harness["artifacts"]["predictions_jsonl"] == "benchmarks/predictions/swe_bench_verified_predictions.jsonl"
    assert any("predictions already exist" in limit for limit in harness["open_limits"])


def test_swebench_autonomous_queue_generates_prediction_and_official_phases():
    module = _load_prepare_module()
    spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "swe_bench_verified",
        "ready_to_run": False,
        "runner": {
            "kind": "swebench_autonomous_queue",
            "harness_root": "/tmp/swe-bench",
            "dataset_json": "benchmarks/swe_bench_verified/dataset.json",
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "repo_cache_root": "benchmarks/repo_cache",
            "artifact_root": "benchmarks/swe_bench_verified/autonomous/full",
            "provider": "vllm",
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "run_id": "verified-full",
            "limit": 2,
            "drain_limit": 2,
        },
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": "benchmarks/swe_bench_verified/summary.json",
            "output_packet_json": "benchmarks/swe_bench_verified/a8_benchmark_result.json",
            "adapter_spec_json": "config/a8_benchmark_adapter_specs/swe_bench_verified_adapter_spec.json",
            "conservative_comparison_report": True,
        },
        "open_limits": ["template"],
    }

    harness = module.build_autonomous_harness_from_run_spec(spec, spec_path="swe.json")

    phase_names = [phase["name"] for phase in harness["phases"]]
    assert phase_names == [
        "validate_adapter_spec",
        "prepare_prediction_tasks",
        "prepare_queue_manifest",
        "enqueue_patch_jobs",
        "drain_patch_jobs",
        "verify_patch_jobs",
        "collect_predictions",
        "repo_cache_apply_check",
        "build_run_spec",
        "official_harness",
        "materialize_results",
        "summarize_results",
        "adapt_a8_packet",
    ]
    drain_phase = next(phase for phase in harness["phases"] if phase["name"] == "drain_patch_jobs")
    assert "--provider" in drain_phase["argv"]
    assert "vllm" in drain_phase["argv"]
    assert "--model" in drain_phase["argv"]
    assert "Qwen/Qwen3.5-9B" in drain_phase["argv"]
    adapt_phase = harness["phases"][-1]
    assert "--adapter-spec-json" in adapt_phase["argv"]
    assert "config/a8_benchmark_adapter_specs/swe_bench_verified_adapter_spec.json" in adapt_phase["required_inputs"]
    assert harness["autonomy_contract"]["operator_role"] == "launch_and_monitor_only"


def test_adapter_spec_from_run_spec_builds_default_metric_adapter():
    module = _load_prepare_module()

    adapter_spec = module.adapter_spec_from_run_spec(
        {
            "spec_version": "asi_v1",
            "report_kind": "a8_benchmark_run_spec",
            "benchmark": "mle_bench",
            "adapter": {
                "script": "scripts/run_a8_benchmark_adapter.py",
                "summary_json": "summary.json",
                "output_packet_json": "packet.json",
                "conservative_comparison_report": True,
            },
        }
    )

    assert adapter_spec["report_kind"] == "a8_benchmark_adapter_spec"
    assert adapter_spec["benchmark"] == "mle_bench"
    assert adapter_spec["metrics"]["gold_medal_rate"]["type"] == "rate"


def test_adapter_command_includes_generated_adapter_spec_json():
    module = _load_prepare_module()
    spec = {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_run_spec",
        "benchmark": "mle_bench",
        "ready_to_run": False,
        "runner": {"kind": "summary_only", "summary_source": "benchmarks/mle_bench/summary.json"},
        "adapter": {
            "script": "scripts/run_a8_benchmark_adapter.py",
            "summary_json": "benchmarks/mle_bench/summary.json",
            "output_packet_json": "benchmarks/mle_bench/a8_benchmark_result.json",
            "adapter_spec_json": "config/a8_benchmark_adapter_specs/mle_bench_adapter_spec.json",
            "conservative_comparison_report": True,
        },
        "open_limits": ["template"],
    }

    harness = module.build_autonomous_harness_from_run_spec(spec, spec_path="mle.json")

    assert [phase["name"] for phase in harness["phases"]] == ["validate_adapter_spec", "adapt_summary_packet"]
    assert harness["phases"][0]["required_inputs"] == [
        "config/a8_benchmark_adapter_specs/mle_bench_adapter_spec.json"
    ]
    argv = harness["phases"][1]["argv"]
    assert "--adapter-spec-json" in argv
    assert "config/a8_benchmark_adapter_specs/mle_bench_adapter_spec.json" in argv
    assert harness["artifacts"]["adapter_spec_json"] == "config/a8_benchmark_adapter_specs/mle_bench_adapter_spec.json"
    assert harness["phases"][1]["required_inputs"] == [
        "benchmarks/mle_bench/summary.json",
        "config/a8_benchmark_adapter_specs/mle_bench_adapter_spec.json",
    ]


def test_prepare_autonomous_harness_cli_writes_harness(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    run_spec = tmp_path / "codeforces_spec.json"
    harness_path = tmp_path / "codeforces_harness.json"
    run_spec.write_text(
        json.dumps(
            {
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
                "open_limits": ["template"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_autonomous_benchmark_harness.py",
            "--run-spec-json",
            str(run_spec),
            "--output-harness-json",
            str(harness_path),
        ],
    )

    module.main()

    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    assert harness["report_kind"] == "autonomous_benchmark_harness_spec"
    assert f"harness_json={harness_path}" in capsys.readouterr().out


def test_prepare_autonomous_harness_cli_writes_all_harnesses_from_run_spec_dir(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    run_spec_dir = tmp_path / "run_specs"
    output_dir = tmp_path / "harnesses"
    adapter_spec_dir = tmp_path / "adapter_specs"
    run_spec_dir.mkdir()
    for name, benchmark in (("codeforces", "codeforces"), ("mle", "mle_bench")):
        (run_spec_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "spec_version": "asi_v1",
                    "report_kind": "a8_benchmark_run_spec",
                    "benchmark": benchmark,
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
                    ]
                    if benchmark == "codeforces"
                    else [],
                    "runner": {"kind": "summary_only", "summary_source": f"{benchmark}/summary.json"},
                    "adapter": {
                        "script": "scripts/run_a8_benchmark_adapter.py",
                        "summary_json": f"{benchmark}/summary.json",
                        "output_packet_json": f"{benchmark}/packet.json",
                        "conservative_comparison_report": True,
                    },
                    "open_limits": ["template"],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_autonomous_benchmark_harness.py",
            "--run-spec-dir",
            str(run_spec_dir),
            "--output-dir",
            str(output_dir),
            "--adapter-spec-dir",
            str(adapter_spec_dir),
        ],
    )

    module.main()

    assert (output_dir / "codeforces_harness.json").exists()
    assert (output_dir / "mle_harness.json").exists()
    assert (adapter_spec_dir / "codeforces_adapter_spec.json").exists()
    assert (adapter_spec_dir / "mle_adapter_spec.json").exists()
    mle_harness = json.loads((output_dir / "mle_harness.json").read_text(encoding="utf-8"))
    assert "--adapter-spec-json" in mle_harness["phases"][0]["argv"]
    assert "generated_harness_count=2" in capsys.readouterr().out
