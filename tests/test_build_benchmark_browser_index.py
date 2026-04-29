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


def test_browser_index_normalizes_swe_bench_live_results_shape(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/rolling_score/evaluation_results/results.json",
        {
            "submitted": 3,
            "submitted_ids": ["a", "b", "c"],
            "success_ids": ["a"],
            "failure_ids": ["b"],
            "error_ids": ["c"],
        },
    )

    index = module.build_benchmark_browser_index(tmp_path)

    result = index["results"][0]
    assert result["total"] == 3
    assert result["resolved"] == 1
    assert result["unresolved"] == 1
    assert result["errors"] == 1
    assert result["resolve_rate"] == 1 / 3


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


def test_browser_index_surfaces_active_harness_run_separately_from_completed_slice(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/swe_bench_verified_test_dataset.json",
        [{"instance_id": "django__django-1"}, {"instance_id": "django__django-2"}],
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/selected_probe/evaluation_results_slice/summary.json",
        {
            "report_kind": "official_swe_bench_summary",
            "task_count": 1,
            "resolved_count": 1,
            "resolve_rate": 1.0,
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/autonomous_harness_runs/full/harness_full_log.json",
        {
            "report_kind": "autonomous_benchmark_harness_run_log",
            "benchmark": "swe_bench_verified",
            "success": False,
            "phase_results": [{"name": "enqueue_patch_jobs", "returncode": 0, "elapsed_seconds": 4.5}],
            "active_phase": {
                "name": "drain_patch_jobs",
                "pid": 123,
                "elapsed_seconds": 42.0,
                "heartbeat_at": "2026-04-27T22:00:00+00:00",
                "started_at": "2026-04-27T21:59:00+00:00",
            },
        },
    )
    _write_json(
        tmp_path / "config/autonomous_benchmark_harnesses/swe_harness.json",
        {
            "report_kind": "autonomous_benchmark_harness_spec",
            "benchmark": "swe_bench_verified",
            "artifacts": {
                "queue_root": "benchmarks/swe_bench_verified/autonomous_harness_runs/full/queue"
            },
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_verified/autonomous_harness_runs/full/queue/queue.json",
        {
            "jobs": [
                {
                    "task_id": "swe_patch_django__django-1",
                    "state": "completed",
                    "outcome": "success",
                    "history": [
                        {
                            "recorded_at": "2026-04-27T22:00:10+00:00",
                            "event": "completed",
                        }
                    ],
                },
                {
                    "task_id": "swe_patch_django__django-2",
                    "state": "queued",
                    "outcome": "",
                    "history": [
                        {
                            "recorded_at": "2026-04-27T22:00:00+00:00",
                            "event": "queued",
                        }
                    ],
                },
            ]
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
    live = module.build_benchmark_live_status(tmp_path)

    swe_gate = next(
        gate for gate in index["a8_progress"]["benchmark_gates"] if gate["benchmark"] == "swe_bench_verified"
    )
    assert swe_gate["current_denominator"] == 1
    assert swe_gate["active_run"]["active_phase"]["name"] == "drain_patch_jobs"
    assert live["active_runs_by_benchmark"]["swe_bench_verified"]["active_phase"]["pid"] == 123
    assert live["queue_snapshots_by_benchmark"]["swe_bench_verified"]["completed_jobs"] == 1
    assert any(event["kind"] == "queue_summary" for event in live["semantic_events"])
    assert index["harness_runs"][0]["completed_phase_count"] == 1


def test_browser_index_tracks_swe_bench_live_as_standalone_leaderboard(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/swe_bench_live_verified_dataset.json",
        [{"instance_id": "django__django-1"}, {"instance_id": "django__django-2"}],
    )
    _write_json(
        tmp_path / "config/standalone_benchmark_run_specs/swe_bench_live_verified_template.json",
        {
            "spec_version": "asi_v1",
            "report_kind": "standalone_benchmark_run_spec",
            "benchmark": "swe_bench_live",
            "benchmark_role": "standalone_leaderboard",
            "ready_to_run": False,
            "runner": {
                "kind": "swebench_live_autonomous_queue",
                "dataset_name": "SWE-bench-Live/SWE-bench-Live",
                "predictions_path": "benchmarks/predictions/swe_bench_live_verified_predictions.jsonl",
                "results_json": "benchmarks/swe_bench_live/evaluation_results_verified/results.json",
            },
            "adapter": {
                "summary_json": "benchmarks/swe_bench_live/summary_verified.json",
                "output_packet_json": "benchmarks/swe_bench_live/leaderboard_result_verified.json",
            },
            "open_limits": ["standalone leaderboard"],
        },
    )
    _write_json(
        tmp_path / "config/autonomous_benchmark_harnesses/swe_bench_live_harness.json",
        {
            "report_kind": "autonomous_benchmark_harness_spec",
            "benchmark": "swe_bench_live",
            "artifacts": {
                "queue_root": "benchmarks/swe_bench_live/autonomous_harness_runs/verified/queue"
            },
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/autonomous_harness_runs/verified/queue/queue.json",
        {
            "jobs": [
                {"task_id": "swe_patch_django__django-1", "state": "completed", "outcome": "success"},
                {"task_id": "swe_patch_django__django-2", "state": "queued", "outcome": ""},
            ]
        },
    )

    index = module.build_benchmark_browser_index(tmp_path)
    live = module.build_benchmark_live_status(tmp_path)

    assert index["datasets"][0]["name"] == "SWE-bench Live Verified"
    live_gate = index["standalone_leaderboards"]["gates"][0]
    assert live_gate["benchmark"] == "swe_bench_live"
    assert live_gate["support_gate"] is False
    assert live_gate["run_spec_path"] == "config/standalone_benchmark_run_specs/swe_bench_live_verified_template.json"
    assert live["queue_snapshots_by_benchmark"]["swe_bench_live"]["completed_jobs"] == 1
    assert live["official_scores_by_benchmark"]["swe_bench_live"]["status"] == "pending"


def test_live_status_surfaces_official_score_from_summary(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "config/standalone_benchmark_run_specs/swe_bench_live_verified_template.json",
        {
            "spec_version": "asi_v1",
            "report_kind": "standalone_benchmark_run_spec",
            "benchmark": "swe_bench_live",
            "benchmark_role": "standalone_leaderboard",
            "ready_to_run": True,
            "runner": {
                "kind": "swebench_live_autonomous_queue",
                "results_json": "benchmarks/swe_bench_live/evaluation_results_verified/results.json",
            },
            "adapter": {
                "summary_json": "benchmarks/swe_bench_live/summary_verified.json",
            },
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/summary_verified.json",
        {
            "report_kind": "official_swe_bench_summary",
            "benchmark": "swe_bench_live",
            "task_count": 10,
            "resolved_count": 8,
            "resolve_rate": 0.8,
        },
    )

    live = module.build_benchmark_live_status(tmp_path)

    score = live["official_scores_by_benchmark"]["swe_bench_live"]
    assert score["status"] == "available"
    assert score["resolved_count"] == 8
    assert score["task_count"] == 10
    assert score["resolve_rate"] == 0.8
    assert score["score_source"] == "summary_json"


def test_live_status_surfaces_rolling_subset_score(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "config/autonomous_benchmark_harnesses/swe_bench_live_rolling_score.json",
        {
            "spec_version": "asi_v1",
            "report_kind": "autonomous_benchmark_harness_spec",
            "benchmark": "swe_bench_live",
            "run_config": {
                "run_id": "rolling-live",
                "score_kind": "first_pass_completed_subset",
            },
            "artifacts": {
                "summary_json": "benchmarks/swe_bench_live/rolling_score/summary.json",
                "results_json": "benchmarks/swe_bench_live/rolling_score/results.json",
                "predictions_patch_json": "benchmarks/swe_bench_live/rolling_score/preds.json",
            },
            "autonomy_contract": {
                "operator_role": "launch_and_monitor_only",
                "selection_mode": "completed_patch_jobs_subset",
                "kernel_owned_phases": ["score"],
                "prohibited_manual_interventions": ["manual patching"],
                "countable_evidence": ["rolling subset score"],
            },
            "phases": [
                {
                    "name": "score",
                    "kind": "command",
                    "argv": ["python", "--version"],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/rolling_score/summary.json",
        {
            "report_kind": "official_swe_bench_summary",
            "benchmark": "swe_bench_live",
            "task_count": 4,
            "resolved_count": 3,
            "resolve_rate": 0.75,
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/rolling_score/preds.json",
        {"a": {"model_patch": "diff"}, "b": {"model_patch": "diff"}},
    )

    live = module.build_benchmark_live_status(tmp_path)

    score = live["rolling_scores"]["swe_bench_live:first_pass_completed_subset"]
    assert score["status"] == "available"
    assert score["resolved_count"] == 3
    assert score["task_count"] == 4
    assert score["prediction_count"] == 2
    assert score["final_leaderboard_score"] is False


def test_live_status_surfaces_partial_rolling_score_from_reports(tmp_path):
    module = _load_indexer_module()
    _write_json(
        tmp_path / "config/autonomous_benchmark_harnesses/swe_bench_live_rolling_score.json",
        {
            "spec_version": "asi_v1",
            "report_kind": "autonomous_benchmark_harness_spec",
            "benchmark": "swe_bench_live",
            "run_config": {
                "run_id": "rolling-live",
                "score_kind": "first_pass_completed_subset",
            },
            "artifacts": {
                "summary_json": "benchmarks/swe_bench_live/rolling_score/summary.json",
                "results_json": "benchmarks/swe_bench_live/rolling_score/evaluation_results/results.json",
                "predictions_patch_json": "benchmarks/swe_bench_live/rolling_score/preds.json",
            },
            "autonomy_contract": {
                "operator_role": "launch_and_monitor_only",
                "selection_mode": "completed_patch_jobs_subset",
                "kernel_owned_phases": ["score"],
                "prohibited_manual_interventions": ["manual patching"],
                "countable_evidence": ["rolling subset score"],
            },
            "phases": [
                {
                    "name": "score",
                    "kind": "command",
                    "argv": ["python", "--version"],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/rolling_score/evaluation_results/a/report.json",
        {"instance_id": "a", "resolved": True},
    )
    _write_json(
        tmp_path / "benchmarks/swe_bench_live/rolling_score/evaluation_results/b/report.json",
        {"instance_id": "b", "resolved": False},
    )

    live = module.build_benchmark_live_status(tmp_path)

    score = live["rolling_scores"]["swe_bench_live:first_pass_completed_subset"]
    assert score["status"] == "partial"
    assert score["resolved_count"] == 1
    assert score["task_count"] == 2
    assert score["resolve_rate"] == 0.5
    assert score["score_source"] == "partial_report_json"
    assert score["passed_instance_ids"] == ["a"]
    assert score["failed_instance_ids"] == ["b"]


def test_live_status_classifies_artifact_contract_failures_generically(tmp_path):
    module = _load_indexer_module()
    queue_root = tmp_path / "benchmarks/generic_benchmark/autonomous_harness_runs/full/queue"
    _write_json(
        tmp_path / "config/autonomous_benchmark_harnesses/generic_harness.json",
        {
            "report_kind": "autonomous_benchmark_harness_spec",
            "benchmark": "generic_benchmark",
            "artifacts": {
                "queue_root": "benchmarks/generic_benchmark/autonomous_harness_runs/full/queue"
            },
        },
    )
    _write_json(
        queue_root / "queue.json",
        {
            "jobs": [
                {
                    "job_id": "job:artifact_case:20260428T040000Z:abc123",
                    "task_id": "artifact_case",
                    "state": "safe_stop",
                    "outcome": "safe_stop",
                    "finished_at": "2026-04-28T04:00:00+00:00",
                    "history": [
                        {
                            "recorded_at": "2026-04-28T04:00:00+00:00",
                            "event": "safe_stop",
                        }
                    ],
                }
            ]
        },
    )
    _write_json(
        queue_root / "reports/job_report_job_artifact_case_20260428T040000Z_abc123.json",
        {
            "outcome": "safe_stop",
            "last_decision_source": "artifact_materialization_guard",
            "termination_reason": "policy_terminated",
            "outcome_reasons": ["policy_terminated"],
            "task_metadata": {
                "artifact_repair_contract": {
                    "artifact_path": "patch.diff",
                    "builder_commands": ["patch_builder"],
                }
            },
            "policy_trace": [
                {
                    "decision_source": "artifact_materialization_guard",
                    "verification_reasons": ["missing expected file: patch.diff"],
                    "proposal_metadata": {
                        "retry_rejected_reason": "invalid_python_replacement",
                    },
                }
            ],
        },
    )

    live = module.build_benchmark_live_status(tmp_path)

    snapshot = live["queue_snapshots_by_benchmark"]["generic_benchmark"]
    assert snapshot["artifact_failure_mode_counts"] == {
        "artifact_materialization_guard_terminal": 1
    }
    assert snapshot["recent_artifact_failures"][0]["mode"] == "artifact_materialization_guard_terminal"
    assert "retry_rejected_reason:invalid_python_replacement" in snapshot["recent_artifact_failures"][0]["evidence"]
    assert any(event["kind"] == "artifact_contract_failure" for event in live["semantic_events"])


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
