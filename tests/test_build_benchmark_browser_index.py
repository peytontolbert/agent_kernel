from pathlib import Path
import importlib.util
import json
import sys


def _load_indexer_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "build_benchmark_browser_index.py"
    spec = importlib.util.spec_from_file_location("build_benchmark_browser_index", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_benchmark_browser_index_normalizes_swe_artifacts(tmp_path):
    module = _load_indexer_module()
    dataset_path = tmp_path / "benchmarks/swe_bench_lite_probe/swe_bench_lite_test_dataset.json"
    _write_json(
        dataset_path,
        [
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "version": "3.2",
                "created_at": "2024-01-02T00:00:00Z",
                "base_commit": "abc123",
                "problem_statement": "Fix response handling",
                "hints_text": "Look at response.py",
                "FAIL_TO_PASS": json.dumps(["test_response"]),
                "PASS_TO_PASS": json.dumps(["test_existing"]),
                "patch": "diff --git a/a b/a\n",
                "test_patch": "diff --git a/t b/t\n",
            }
        ],
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_lite_probe/evaluation_results_probe/results.json",
        {
            "total_instances": 1,
            "completed_ids": ["django__django-1"],
            "resolved_ids": ["django__django-1"],
            "unresolved_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        },
    )
    (tmp_path / "benchmarks/swe_bench_lite_probe/predictions.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "django__django-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "diff --git a/a b/a\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "docs/evidence/a8_coding_superhuman_target_packet_20260426.json",
        {
            "target": {
                "thresholds": {
                    "codeforces_rating_equivalent": 3000,
                    "mle_bench_gold_medal_rate": 0.2,
                    "swe_bench_verified_resolve_rate": 0.8,
                    "swe_rebench_resolve_rate": 0.6,
                    "re_bench_human_expert_win_rate": 0.5,
                    "superhuman_coding_task_count": 100,
                    "superhuman_coding_window_count": 3,
                    "recursive_compounding_retained_gain_runs": 5,
                    "recursive_compounding_window_count": 3,
                }
            }
        },
    )

    index = module.build_benchmark_browser_index(tmp_path)

    assert index["datasets"][0]["task_count"] == 1
    assert index["datasets"][0]["repo_counts"] == {"django/django": 1}
    assert index["datasets"][0]["instances"][0]["fail_to_pass"] == ["test_response"]
    assert index["results"][0]["resolved"] == 1
    assert index["results"][0]["resolve_rate"] == 1.0
    assert index["predictions"][0]["prediction_count"] == 1
    assert index["targets"]["thresholds"]["swe_bench_verified_resolve_rate"] == 0.8
    swe_gate = next(
        gate for gate in index["a8_progress"]["benchmark_gates"] if gate["benchmark"] == "swe_bench_verified"
    )
    assert swe_gate["threshold"] == 0.8
    assert swe_gate["status"] == "no_evidence"
    assert index["a8_progress"]["gate_count"] == 7


def test_a8_progress_tracks_non_swe_summary_metrics(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "benchmarks/codeforces/summary.json",
        {
            "report_kind": "official_codeforces_summary",
            "benchmark": "codeforces",
            "created_at": "2026-04-27T00:00:00Z",
            "rating_equivalent": 3100,
        },
    )
    _write_json(
        tmp_path / "docs/evidence/a8_coding_superhuman_target_packet_20260426.json",
        {
            "target": {
                "thresholds": {
                    "codeforces_rating_equivalent": 3000,
                    "mle_bench_gold_medal_rate": 0.2,
                    "swe_bench_verified_resolve_rate": 0.8,
                    "swe_rebench_resolve_rate": 0.6,
                    "re_bench_human_expert_win_rate": 0.5,
                    "superhuman_coding_task_count": 100,
                    "recursive_compounding_retained_gain_runs": 5,
                }
            }
        },
    )

    index = module.build_benchmark_browser_index(tmp_path)

    codeforces = next(gate for gate in index["a8_progress"]["benchmark_gates"] if gate["benchmark"] == "codeforces")
    assert codeforces["current_value"] == 3100
    assert codeforces["status"] == "met"
    assert codeforces["evidence_path"] == "benchmarks/codeforces/summary.json"


def test_a8_progress_treats_swe_verified_slice_as_partial_until_full_count_met(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/swe_bench_verified_test_dataset.json",
        [{"instance_id": "django__django-1"}, {"instance_id": "django__django-2"}],
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/selected_probe/summary.json",
        {
            "report_kind": "official_swe_bench_summary",
            "created_at": "2026-04-27T00:00:00Z",
            "task_count": 1,
            "resolved_count": 1,
            "resolve_rate": 1.0,
        },
    )
    _write_json(
        tmp_path / "docs/evidence/a8_coding_superhuman_target_packet_20260426.json",
        {
            "target": {
                "thresholds": {
                    "codeforces_rating_equivalent": 3000,
                    "mle_bench_gold_medal_rate": 0.2,
                    "swe_bench_verified_resolve_rate": 0.8,
                    "swe_rebench_resolve_rate": 0.6,
                    "re_bench_human_expert_win_rate": 0.5,
                    "superhuman_coding_task_count": 100,
                    "recursive_compounding_retained_gain_runs": 5,
                }
            }
        },
    )

    index = module.build_benchmark_browser_index(tmp_path)

    swe_gate = next(
        gate for gate in index["a8_progress"]["benchmark_gates"] if gate["benchmark"] == "swe_bench_verified"
    )
    assert swe_gate["current_value"] == 1.0
    assert swe_gate["required_count"] == 2
    assert swe_gate["status"] == "partial"


def test_indexer_cli_writes_output(tmp_path, monkeypatch, capsys):
    module = _load_indexer_module()
    _write_json(tmp_path / "benchmarks/swe_bench_verified/swe_bench_verified_test_dataset.json", [])
    output = tmp_path / "web/benchmark_browser/benchmark_index.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_benchmark_browser_index.py",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    module.main()

    assert output.exists()
    assert "datasets=1" in capsys.readouterr().out
