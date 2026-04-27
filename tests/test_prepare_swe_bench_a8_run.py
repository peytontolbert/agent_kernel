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


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_autonomous_benchmark_harness.py"
    spec = importlib.util.spec_from_file_location("run_autonomous_benchmark_harness", script_path)
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


def test_build_a8_swe_benchmark_run_spec_auto_ready_from_predictions_file(tmp_path):
    module = _load_prepare_module()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = module.build_a8_swe_benchmark_run_spec(
        benchmark="swe_bench_verified",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        predictions_path=str(predictions_path),
        run_id="run-1",
        harness_root=str(tmp_path),
        max_workers=1,
        timeout=600,
        cache_level="base",
        namespace="swebench",
        report_dir=str(tmp_path / "results"),
        results_json=str(tmp_path / "results" / "results.json"),
        summary_json=str(tmp_path / "summary.json"),
        output_packet_json=str(tmp_path / "packet.json"),
        instance_ids=["repo__pkg-1"],
    )

    assert spec["report_kind"] == "a8_benchmark_run_spec"
    assert spec["ready_to_run"] is True
    assert spec["runner"]["kind"] == "swebench_harness"
    assert spec["runner"]["predictions_path"] == str(predictions_path)
    assert spec["runner"]["instance_ids"] == ["repo__pkg-1"]
    assert spec["adapter"]["conservative_comparison_report"] is True
    assert any("official SWE harness" in limit for limit in spec["open_limits"])


def test_prepare_swe_bench_spec_cli_writes_a8_run_spec(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    predictions_path = tmp_path / "predictions.jsonl"
    spec_path = tmp_path / "spec.json"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "model_name_or_path": "agentkernel",
                "model_patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "spec",
            "--benchmark",
            "swe_bench_verified",
            "--predictions-path",
            str(predictions_path),
            "--run-id",
            "verified-run",
            "--report-dir",
            str(tmp_path / "results"),
            "--results-json",
            str(tmp_path / "results" / "results.json"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--output-packet-json",
            str(tmp_path / "packet.json"),
            "--swe-bench-root",
            str(tmp_path),
            "--instance-ids",
            "repo__pkg-1",
            "--output-spec-json",
            str(spec_path),
        ],
    )

    module.main()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["report_kind"] == "a8_benchmark_run_spec"
    assert spec["ready_to_run"] is True
    assert spec["runner"]["predictions_path"] == str(predictions_path)
    assert spec["runner"]["instance_ids"] == ["repo__pkg-1"]
    assert f"spec_json={spec_path}" in capsys.readouterr().out


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


def test_build_swe_autonomous_harness_spec_defines_end_to_end_phases(tmp_path):
    module = _load_prepare_module()

    spec = module.build_swe_autonomous_harness_spec(
        benchmark="swe_bench_verified",
        dataset_json="dataset.json",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        repo_cache_root="repo_cache",
        prediction_task_manifest="tasks.json",
        patch_dir="patches",
        queue_manifest="queue_manifest.json",
        queue_root=str(tmp_path / "queue"),
        workspace_root="workspace",
        workspace_prefix="swe_harness_probe",
        predictions_jsonl="predictions.jsonl",
        apply_check_json="apply_check.json",
        run_spec_json="run_spec.json",
        run_id="autonomous-run",
        harness_root=str(tmp_path),
        max_workers=1,
        timeout=1800,
        cache_level="env",
        namespace="swebench",
        report_dir="results",
        results_json="results/results.json",
        summary_json="results/summary.json",
        output_packet_json="results/a8_benchmark_result.json",
        python_bin=sys.executable,
        model_name_or_path="agentkernel",
        provider="vllm",
        drain_limit=0,
        max_source_context_bytes=30000,
        limit=2,
    )

    assert spec["report_kind"] == "autonomous_benchmark_harness_spec"
    assert spec["autonomy_contract"]["operator_role"] == "launch_and_monitor_only"
    assert spec["autonomy_contract"]["selection_mode"] == "dataset_limit"
    assert spec["inputs"]["split"] == "test"
    assert spec["run_config"]["run_id"] == "autonomous-run"
    assert spec["run_config"]["queue_max_queued_per_budget_group"] == 2
    assert "manual prediction JSONL editing" in spec["autonomy_contract"]["prohibited_manual_interventions"]
    assert "summarize_results" in spec["autonomy_contract"]["kernel_owned_phases"]
    assert "adapt_a8_packet" in spec["autonomy_contract"]["kernel_owned_phases"]
    phase_names = [phase["name"] for phase in spec["phases"]]
    assert phase_names == [
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
    assert spec["phases"][0]["required_inputs"] == ["dataset.json", "repo_cache"]
    assert spec["phases"][1]["required_inputs"] == ["tasks.json"]
    assert spec["phases"][4]["required_inputs"][1] == "queue_manifest.json"
    assert spec["phases"][4]["expected_outputs"][0].endswith("patch_jobs_verification.json")
    assert spec["phases"][5]["required_inputs"] == ["tasks.json", "queue_manifest.json"]
    assert spec["phases"][6]["required_inputs"] == ["dataset.json", "predictions.jsonl", "repo_cache"]
    assert spec["phases"][8]["required_inputs"] == ["predictions.jsonl"]
    assert spec["phases"][10]["required_inputs"] == ["results/results.json"]
    assert spec["phases"][11]["preflight_argv"] == spec["phases"][11]["argv"] + ["--validate-only"]
    assert spec["phases"][11]["required_inputs"] == ["results/summary.json"]
    assert spec["phases"][2]["env"]["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"].endswith("queue.json")
    enqueue_argv = spec["phases"][2]["argv"]
    assert enqueue_argv[enqueue_argv.index("--max-queued-per-budget-group") + 1] == "2"
    assert spec["phases"][3]["env"]["AGENT_KERNEL_WORKSPACE_ROOT"] == "workspace"
    assert spec["artifacts"]["patch_jobs_verification_json"].endswith("patch_jobs_verification.json")
    assert any("not full benchmark evidence" in limit for limit in spec["open_limits"])


def test_swe_autonomous_harness_queue_budget_scales_with_selected_slice(tmp_path):
    module = _load_prepare_module()

    spec = module.build_swe_autonomous_harness_spec(
        benchmark="swe_bench_verified",
        dataset_json="dataset.json",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        repo_cache_root="repo_cache",
        prediction_task_manifest="tasks.json",
        patch_dir="patches",
        queue_manifest="queue_manifest.json",
        queue_root=str(tmp_path / "queue"),
        workspace_root="workspace",
        workspace_prefix="swe_harness_probe",
        predictions_jsonl="predictions.jsonl",
        apply_check_json="apply_check.json",
        run_spec_json="run_spec.json",
        run_id="autonomous-run",
        harness_root=str(tmp_path),
        max_workers=1,
        timeout=1800,
        cache_level="env",
        namespace="swebench",
        report_dir="results",
        results_json="results/results.json",
        summary_json="results/summary.json",
        output_packet_json="results/a8_benchmark_result.json",
        python_bin=sys.executable,
        model_name_or_path="agentkernel",
        provider="vllm",
        drain_limit=10,
        max_source_context_bytes=30000,
        instance_ids=[f"repo__pkg-{index}" for index in range(10)],
    )

    enqueue_argv = spec["phases"][2]["argv"]
    assert enqueue_argv[enqueue_argv.index("--max-queued-per-budget-group") + 1] == "10"
    assert spec["run_config"]["queue_max_queued_per_budget_group"] == 10


def test_prepare_swe_bench_harness_cli_writes_autonomous_harness(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    harness_path = tmp_path / "harness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "harness",
            "--benchmark",
            "swe_bench_verified",
            "--dataset-json",
            "dataset.json",
            "--repo-cache-root",
            "repo_cache",
            "--prediction-task-manifest",
            "tasks.json",
            "--patch-dir",
            "patches",
            "--queue-manifest",
            "queue_manifest.json",
            "--queue-root",
            str(tmp_path / "queue"),
            "--workspace-prefix",
            "swe_harness_probe",
            "--predictions-jsonl",
            "predictions.jsonl",
            "--apply-check-json",
            "apply_check.json",
            "--run-spec-json",
            "run_spec.json",
            "--run-id",
            "autonomous-run",
            "--report-dir",
            "results",
            "--results-json",
            "results/results.json",
            "--summary-json",
            "results/summary.json",
            "--output-packet-json",
            "results/a8_benchmark_result.json",
            "--limit",
            "2",
            "--output-harness-json",
            str(harness_path),
        ],
    )

    module.main()

    spec = json.loads(harness_path.read_text(encoding="utf-8"))
    assert spec["report_kind"] == "autonomous_benchmark_harness_spec"
    assert spec["autonomy_contract"]["selection_mode"] == "dataset_limit"
    assert f"harness_json={harness_path}" in capsys.readouterr().out


def test_build_swe_retry_harness_spec_from_patch_job_verification(tmp_path):
    module = _load_prepare_module()
    runner_module = _load_runner_module()
    source = module.build_swe_autonomous_harness_spec(
        benchmark="swe_bench_verified",
        dataset_json="dataset.json",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        repo_cache_root="repo_cache",
        prediction_task_manifest="tasks.json",
        patch_dir="patches",
        queue_manifest="queue_manifest.json",
        queue_root=str(tmp_path / "queue"),
        workspace_root="workspace",
        workspace_prefix="swe_probe",
        predictions_jsonl="predictions.jsonl",
        apply_check_json="apply_check.json",
        run_spec_json="run_spec.json",
        run_id="autonomous-run",
        harness_root=str(tmp_path / "swe-bench"),
        max_workers=1,
        timeout=1800,
        cache_level="env",
        namespace="swebench",
        report_dir="results",
        results_json="results/results.json",
        summary_json="results/summary.json",
        output_packet_json="results/a8_benchmark_result.json",
        python_bin=sys.executable,
        model_name_or_path="agentkernel",
        provider="vllm",
        drain_limit=2,
        max_source_context_bytes=30000,
        instance_ids=["repo__pkg-1", "repo__pkg-2"],
    )
    retry = module.build_swe_retry_harness_spec(
        source_harness=source,
        patch_job_verification={"retry_instance_ids": ["repo__pkg-2"]},
        retry_label="retry1",
        artifact_dir=tmp_path / "retry",
        output_harness_json=str(tmp_path / "retry" / "harness.json"),
    )

    assert retry["autonomy_contract"]["selection_mode"] == "operator_selected_instance_ids"
    assert retry["inputs"]["instance_ids"] == ["repo__pkg-2"]
    assert retry["inputs"]["limit"] == 0
    assert retry["run_config"]["run_id"] == "autonomous-run_retry1"
    assert retry["artifacts"]["queue_root"].endswith("queue_retry1")
    assert retry["artifacts"]["workspace_prefix"] == "swe_probe_retry1"
    assert retry["source_retry"]["retry_instance_ids"] == ["repo__pkg-2"]
    assert runner_module.validate_harness_spec(retry) == []


def test_build_swe_success_continuation_harness_from_patch_job_verification(tmp_path):
    module = _load_prepare_module()
    runner_module = _load_runner_module()
    source = module.build_swe_autonomous_harness_spec(
        benchmark="swe_bench_verified",
        dataset_json="dataset.json",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        repo_cache_root="repo_cache",
        prediction_task_manifest="tasks.json",
        patch_dir="patches",
        queue_manifest="queue_manifest.json",
        queue_root=str(tmp_path / "queue"),
        workspace_root="workspace",
        workspace_prefix="swe_probe",
        predictions_jsonl="predictions.jsonl",
        apply_check_json="apply_check.json",
        run_spec_json="run_spec.json",
        run_id="autonomous-run",
        harness_root=str(tmp_path / "swe-bench"),
        max_workers=1,
        timeout=1800,
        cache_level="env",
        namespace="swebench",
        report_dir="results",
        results_json="results/results.json",
        summary_json="results/summary.json",
        output_packet_json="results/a8_benchmark_result.json",
        python_bin=sys.executable,
        model_name_or_path="agentkernel",
        provider="vllm",
        drain_limit=2,
        max_source_context_bytes=30000,
        instance_ids=["repo__pkg-1", "repo__pkg-2"],
    )
    source["source_harness_json"] = "source_harness.json"
    success = module.build_swe_success_continuation_harness_spec(
        source_harness=source,
        patch_job_verification={
            "patch_job_verification_json": "patch_jobs_verification.json",
            "successful_instance_ids": ["repo__pkg-1"],
            "retry_instance_ids": ["repo__pkg-2"],
        },
        success_label="success1",
        artifact_dir=tmp_path / "success",
        output_harness_json=str(tmp_path / "success" / "harness.json"),
    )

    assert success["autonomy_contract"]["selection_mode"] == "verified_patch_success_continuation"
    assert success["inputs"]["instance_ids"] == ["repo__pkg-1"]
    assert success["run_config"]["run_id"] == "autonomous-run_success1"
    assert success["source_success_continuation"]["successful_instance_ids"] == ["repo__pkg-1"]
    assert [phase["name"] for phase in success["phases"]][:3] == [
        "collect_predictions",
        "repo_cache_apply_check",
        "build_run_spec",
    ]
    assert "--patch-job-verification-json" in success["phases"][0]["argv"]
    assert runner_module.validate_harness_spec(success) == []


def test_prepare_swe_bench_retry_harness_cli_writes_retry_harness(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    source_path = tmp_path / "source_harness.json"
    verification_path = tmp_path / "patch_jobs_verification.json"
    output_path = tmp_path / "retry" / "harness.json"
    source = module.build_swe_autonomous_harness_spec(
        benchmark="swe_bench_verified",
        dataset_json="dataset.json",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        repo_cache_root="repo_cache",
        prediction_task_manifest="tasks.json",
        patch_dir="patches",
        queue_manifest="queue_manifest.json",
        queue_root=str(tmp_path / "queue"),
        workspace_root="workspace",
        workspace_prefix="swe_probe",
        predictions_jsonl="predictions.jsonl",
        apply_check_json="apply_check.json",
        run_spec_json="run_spec.json",
        run_id="autonomous-run",
        harness_root=str(tmp_path / "swe-bench"),
        max_workers=1,
        timeout=1800,
        cache_level="env",
        namespace="swebench",
        report_dir="results",
        results_json="results/results.json",
        summary_json="results/summary.json",
        output_packet_json="results/a8_benchmark_result.json",
        python_bin=sys.executable,
        model_name_or_path="agentkernel",
        provider="vllm",
        drain_limit=2,
        max_source_context_bytes=30000,
        instance_ids=["repo__pkg-1", "repo__pkg-2"],
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    verification_path.write_text(json.dumps({"retry_instance_ids": ["repo__pkg-2"]}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "retry-harness",
            "--source-harness-json",
            str(source_path),
            "--patch-job-verification-json",
            str(verification_path),
            "--retry-label",
            "retry1",
            "--artifact-dir",
            str(tmp_path / "retry"),
            "--output-harness-json",
            str(output_path),
        ],
    )

    module.main()

    retry = json.loads(output_path.read_text(encoding="utf-8"))
    assert retry["inputs"]["instance_ids"] == ["repo__pkg-2"]
    assert "retry_count=1" in capsys.readouterr().out


def test_materialize_swe_bench_results_copies_namespace_report(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    report = tmp_path / "swebench.run-1.json"
    output = tmp_path / "results" / "results.json"
    report.write_text(json.dumps({"resolved_instances": 1, "total_instances": 1}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "materialize-results",
            "--run-id",
            "run-1",
            "--namespace",
            "swebench",
            "--report-dir",
            str(tmp_path / "missing-report-dir"),
            "--search-root",
            str(tmp_path),
            "--output-results-json",
            str(output),
        ],
    )

    module.main()

    assert json.loads(output.read_text(encoding="utf-8"))["resolved_instances"] == 1
    assert f"results_json={output}" in capsys.readouterr().out


def test_materialize_swe_bench_results_finds_model_prefixed_report(tmp_path, monkeypatch, capsys):
    module = _load_prepare_module()
    report = tmp_path / "agentkernel.run-1.json"
    output = tmp_path / "results" / "results.json"
    report.write_text(json.dumps({"resolved_instances": 1, "total_instances": 1}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_a8_run.py",
            "materialize-results",
            "--run-id",
            "run-1",
            "--namespace",
            "swebench",
            "--report-dir",
            str(tmp_path / "missing-report-dir"),
            "--search-root",
            str(tmp_path),
            "--output-results-json",
            str(output),
        ],
    )

    module.main()

    assert json.loads(output.read_text(encoding="utf-8"))["resolved_instances"] == 1
    assert f"results_json={output}" in capsys.readouterr().out
