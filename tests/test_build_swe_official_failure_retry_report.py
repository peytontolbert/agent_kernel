import json

from scripts.build_swe_official_failure_retry_report import build_official_failure_retry_report


def test_build_official_failure_retry_report_extracts_unresolved_instances(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps({"instance_id": "repo__pkg-1", "resolved": False}),
        encoding="utf-8",
    )
    (failed_dir / "status.json").write_text(
        json.dumps({"pkg.module:file.py:12": "fail", "tests/test_pkg.py::test_expected": "fail", "tests/test_pkg.py::test_other": "pass"}),
        encoding="utf-8",
    )
    (failed_dir / "post_patch_log.txt").write_text("FAILED tests/test_pkg.py::test_expected\n", encoding="utf-8")
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 2,
                "submitted_ids": ["repo__pkg-1", "repo__pkg-2"],
                "success_ids": ["repo__pkg-2"],
                "failure_ids": ["repo__pkg-1"],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
    )

    assert report["retry_instance_ids"] == ["repo__pkg-1"]
    assert report["successful_instance_ids"] == ["repo__pkg-2"]
    assert report["failed_jobs"][0]["failed_tests"] == ["tests/test_pkg.py::test_expected"]
    assert "FAILED tests/test_pkg.py::test_expected" in report["failed_jobs"][0]["post_patch_log_tail"]


def test_build_official_failure_retry_report_retries_errors_and_unresolved_without_failure_id(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 3,
                "submitted_ids": ["repo__pkg-1", "repo__pkg-2", "repo__pkg-3"],
                "success_ids": ["repo__pkg-3"],
                "failure_ids": [],
                "error_ids": ["repo__pkg-2"],
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
    )

    assert report["retry_instance_ids"] == ["repo__pkg-2", "repo__pkg-1"]
    assert report["failed_patch_count"] == 2


def test_build_official_failure_retry_report_supports_partial_reports(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    success_dir = results_dir / "repo__pkg-2"
    failed_dir.mkdir(parents=True)
    success_dir.mkdir()
    (failed_dir / "report.json").write_text(
        json.dumps({"instance_id": "repo__pkg-1", "resolved": False}),
        encoding="utf-8",
    )
    (success_dir / "report.json").write_text(
        json.dumps({"instance_id": "repo__pkg-2", "resolved": True}),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps({"repo__pkg-1": {"model_patch": "diff"}, "repo__pkg-2": {"model_patch": "diff"}, "repo__pkg-3": {"model_patch": "diff"}}),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
    )

    assert report["report_mode"] == "partial_reports"
    assert report["task_count"] == 3
    assert report["scored_instance_count"] == 2
    assert report["retry_instance_ids"] == ["repo__pkg-1"]
    assert report["successful_instance_ids"] == ["repo__pkg-2"]


def test_build_official_failure_retry_report_separates_fail_to_pass_and_regressions(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "resolved": False,
                "FAIL_TO_PASS": {"failure": ["tests/test_target.py::test_new"]},
                "PASS_TO_PASS": {
                    "failure": [
                        "tests/test_existing.py::test_one",
                        "tests/test_existing.py::test_two",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps({"repo__pkg-1": {"model_patch": "diff --git a/pkg.py b/pkg.py\n+bad patch"}}),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["failed_tests"] == ["tests/test_target.py::test_new"]
    assert failed["fail_to_pass_failures"] == ["tests/test_target.py::test_new"]
    assert failed["pass_to_pass_failures"] == [
        "tests/test_existing.py::test_one",
        "tests/test_existing.py::test_two",
    ]
    assert failed["pass_to_pass_failure_count"] == 2
    assert failed["official_failure_mode"] == "official_pass_to_pass_regression"
    assert "preserve existing" in failed["official_repair_directive"].lower()
    assert "bad patch" in failed["prior_model_patch_tail"]


def test_build_official_failure_retry_report_classifies_shallow_constant_patch(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "resolved": False,
                "FAIL_TO_PASS": {"failure": ["tests/test_target.py::test_new"]},
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps(
            {
                "repo__pkg-1": {
                    "model_patch": "--- a/pkg.py\n+++ b/pkg.py\n@@ -1 +1 @@\n-value = compute()\n+value = 67.0\n"
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["reason"] == "official_shallow_constant_patch_failed"
    assert failed["official_failure_mode"] == "official_shallow_constant_patch_failed"
    assert "magic constants" in failed["official_repair_directive"]


def test_build_official_failure_retry_report_classifies_self_literal_arithmetic_patch(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "resolved": False,
                "FAIL_TO_PASS": {"failure": ["tests/test_target.py::test_new"]},
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps(
            {
                "repo__pkg-1": {
                    "model_patch": "--- a/pkg.py\n+++ b/pkg.py\n@@ -1 +1 @@\n-value = compute()\n+value = value - 1\n"
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["official_failure_mode"] == "official_shallow_constant_patch_failed"


def test_build_official_failure_retry_report_classifies_date_sensitive_p2p_drift(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "resolved": False,
                "FAIL_TO_PASS": {"success": ["tests/test_target.py::test_new"], "failure": []},
                "PASS_TO_PASS": {"failure": ["tests/test_existing.py::test_one"]},
            }
        ),
        encoding="utf-8",
    )
    (failed_dir / "post_patch_log.txt").write_text(
        "E2533 Runtime 'nodejs18.x' was deprecated on '2025-07-31'. "
        "Creation was disabled on '2025-09-01' and update on '2025-10-01'.",
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps({"repo__pkg-1": {"model_patch": "diff --git a/pkg.py b/pkg.py\n+real fix"}}),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["fail_to_pass_failures"] == []
    assert failed["official_failure_mode"] == "official_environment_pass_to_pass_drift"
    assert "environment drift" in failed["official_repair_directive"].lower()


def test_build_official_failure_retry_report_carries_cumulative_rejected_patch_tails(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    failed_dir = results_dir / "repo__pkg-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "report.json").write_text(
        json.dumps(
            {
                "instance_id": "repo__pkg-1",
                "resolved": False,
                "FAIL_TO_PASS": {"failure": ["tests/test_target.py::test_new"]},
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(
        json.dumps({"repo__pkg-1": {"model_patch": "diff --git a/pkg.py b/pkg.py\n+current"}}),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-1",
                        "runtime_overrides": {
                            "task_payload": {
                                "metadata": {
                                    "swe_official_feedback": {
                                        "prior_model_patch_tail": "diff --git a/pkg.py b/pkg.py\n+previous",
                                        "rejected_patch_tails": ["diff --git a/pkg.py b/pkg.py\n+older"],
                                    }
                                }
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
        queue_json=queue_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["rejected_patch_tails"] == [
        "diff --git a/pkg.py b/pkg.py\n+older",
        "diff --git a/pkg.py b/pkg.py\n+previous",
        "diff --git a/pkg.py b/pkg.py\n+current",
    ]


def test_build_official_failure_retry_report_merges_terminal_queue_safe_stops(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 1,
                "submitted_ids": ["repo__pkg-1"],
                "success_ids": [],
                "failure_ids": ["repo__pkg-1"],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "safe.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "artifact_contract_failure": {
                    "mode": "artifact_escaped_newline_replacement",
                    "repairable": True,
                }
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-2",
                        "state": "safe_stop",
                        "outcome": "safe_stop",
                        "report_path": str(report_path),
                        "outcome_reasons": ["policy_terminated"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        queue_json=queue_json,
    )

    assert report["retry_instance_ids"] == ["repo__pkg-1", "repo__pkg-2"]
    safe_stop = report["failed_jobs"][1]
    assert safe_stop["instance_id"] == "repo__pkg-2"
    assert safe_stop["state"] == "safe_stop"
    assert safe_stop["reason"] == "artifact_escaped_newline_replacement"
    assert safe_stop["artifact_contract_failure"]["repairable"] is True


def test_build_official_failure_retry_report_adds_terminal_artifact_repair_directive(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 0,
                "submitted_ids": [],
                "success_ids": [],
                "failure_ids": [],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "safe.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "artifact_contract_failure": {
                    "mode": "artifact_invalid_python_replacement",
                    "repairable": True,
                }
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-2",
                        "state": "safe_stop",
                        "outcome": "safe_stop",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        queue_json=queue_json,
    )

    safe_stop = report["failed_jobs"][0]
    assert safe_stop["reason"] == "artifact_invalid_python_replacement"
    assert "complete syntactic statement" in safe_stop["artifact_repair_directive"]


def test_build_official_failure_retry_report_directs_escaped_newline_repairs(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 0,
                "submitted_ids": [],
                "success_ids": [],
                "failure_ids": [],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "safe.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "artifact_contract_failure": {
                    "mode": "artifact_escaped_newline_replacement",
                    "repairable": True,
                }
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-2",
                        "state": "safe_stop",
                        "outcome": "safe_stop",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        queue_json=queue_json,
    )

    safe_stop = report["failed_jobs"][0]
    assert safe_stop["reason"] == "artifact_escaped_newline_replacement"
    assert "separate --with argument" in safe_stop["artifact_repair_directive"]


def test_build_official_failure_retry_report_enriches_empty_official_with_terminal_queue_failure(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 1,
                "submitted_ids": ["repo__pkg-1"],
                "empty_patch_ids": ["repo__pkg-1"],
                "success_ids": [],
                "failure_ids": [],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(json.dumps({"repo__pkg-1": {"model_patch": ""}}), encoding="utf-8")
    report_path = tmp_path / "reports" / "safe.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "outcome": "safe_stop",
                "last_decision_source": "artifact_materialization_guard",
                "task_metadata": {"semantic_verifier": {"kind": "swe_patch_apply_check", "patch_path": "patch.diff"}},
                "policy_trace": [
                    {
                        "decision_source": "artifact_anchor_candidate_suggestion_direct",
                        "verification_reasons": [
                            "SWE patch introduces None container misuse in pkg/module.py: patterns:subscript",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-1",
                        "state": "safe_stop",
                        "outcome": "safe_stop",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
        queue_json=queue_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["reason"] == "artifact_none_container_misuse"
    assert failed["artifact_contract_failure"]["mode"] == "artifact_none_container_misuse"
    assert failed["local_terminal_failure"]["state"] == "safe_stop"


def test_build_official_failure_retry_report_enriches_empty_official_with_terminal_directive(tmp_path):
    results_dir = tmp_path / "evaluation_results"
    results_dir.mkdir()
    results_json = results_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "submitted": 1,
                "submitted_ids": ["repo__pkg-1"],
                "empty_patch_ids": ["repo__pkg-1"],
                "success_ids": [],
                "failure_ids": [],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )
    predictions_json = tmp_path / "preds.json"
    predictions_json.write_text(json.dumps({"repo__pkg-1": {"model_patch": ""}}), encoding="utf-8")
    report_path = tmp_path / "reports" / "safe.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "artifact_contract_failure": {
                    "mode": "artifact_missing_after_response",
                    "repairable": True,
                }
            }
        ),
        encoding="utf-8",
    )
    queue_json = tmp_path / "queue.json"
    queue_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "task_id": "swe_patch_repo__pkg-1",
                        "state": "safe_stop",
                        "outcome": "safe_stop",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=results_dir,
        output_json=tmp_path / "retry.json",
        predictions_json=predictions_json,
        queue_json=queue_json,
    )

    failed = report["failed_jobs"][0]
    assert failed["artifact_contract_failure"]["mode"] == "artifact_missing_after_response"
    assert "create patch.diff directly" in failed["artifact_repair_directive"]
