from pathlib import Path
import json
import subprocess
import sys


def test_report_unattended_run_metrics_summarizes_hidden_side_effect_risk(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "report_unattended_run_metrics.py"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    reports_dir.joinpath("repo_chore_success.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "task_id": "repo_patch_review_task",
                "benchmark_family": "repo_chore",
                "outcome": "success",
                "success": True,
                "summary": {"unexpected_change_files": 0},
                "side_effects": {"hidden_side_effect_risk": False},
                "acceptance_packet": {
                    "required_merged_branches": ["worker/api-status"],
                    "tests": [{"command": "tests/test_api.sh"}],
                    "selected_edits": [{"path": "src/api_status.txt"}],
                    "candidate_edit_sets": [{"path": "src/api_status.txt"}],
                },
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("repo_chore_hidden_risk.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "task_id": "repo_cleanup_review_task",
                "benchmark_family": "repo_chore",
                "outcome": "success",
                "success": True,
                "summary": {"unexpected_change_files": 2},
                "side_effects": {"hidden_side_effect_risk": True},
                "acceptance_packet": {
                    "synthetic_worker": True,
                    "selected_edits": [{"path": "src/api_status.txt"}],
                    "candidate_edit_sets": [{"path": "src/api_status.txt"}],
                },
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("tooling_safe_stop.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "task_id": "api_contract_task",
                "benchmark_family": "tooling",
                "outcome": "safe_stop",
                "success": False,
                "summary": {"unexpected_change_files": 1},
                "side_effects": {"hidden_side_effect_risk": True},
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("external_lab_success.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "task_id": "external_lab_task",
                "benchmark_family": "external_lab",
                "task_metadata": {"task_origin": "external_manifest"},
                "outcome": "success",
                "success": True,
                "summary": {"unexpected_change_files": 0},
                "side_effects": {"hidden_side_effect_risk": False},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--reports-dir",
            str(reports_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "reports_total=4" in completed.stdout
    assert "hidden_side_effect_risk_count=2" in completed.stdout
    assert "success_hidden_side_effect_risk_count=1" in completed.stdout
    assert "false_pass_risk_count=1" in completed.stdout
    assert "unexpected_change_files_total=3" in completed.stdout
    assert "unexpected_change_report_count=2" in completed.stdout
    assert "acceptance_packet_count=2" in completed.stdout
    assert "synthetic_worker_count=1" in completed.stdout
    assert "selected_edit_total=2" in completed.stdout
    assert "candidate_edit_set_total=2" in completed.stdout
    assert "required_merged_branch_total=1" in completed.stdout
    assert "test_command_total=1" in completed.stdout
    assert "outcome_count outcome=safe_stop count=1" in completed.stdout
    assert "outcome_count outcome=success count=3" in completed.stdout
    assert "external_report_count=1" in completed.stdout
    assert "external_distinct_families=1" in completed.stdout
    assert "benchmark_family=external_lab total=1 external_total=1 hidden_side_effect_risk=0 success_hidden_side_effect_risk=0" in completed.stdout
    assert "benchmark_family=repo_chore total=2 external_total=0 hidden_side_effect_risk=1 success_hidden_side_effect_risk=1" in completed.stdout
    assert "benchmark_family=tooling total=1 external_total=0 hidden_side_effect_risk=1 success_hidden_side_effect_risk=0" in completed.stdout
    assert "trust_overall status=bootstrap" in completed.stdout
    assert "false_pass_risk_rate=0.500" in completed.stdout
    assert "clean_success_streak=0" in completed.stdout
    assert "trust_coverage reports=4 families=external_lab,repo_chore,tooling external_reports=1 external_families=external_lab" in completed.stdout
    assert "trust_required_coverage " in completed.stdout
    assert "required=integration,project,repo_chore,repo_sandbox,repository" in completed.stdout
    assert "with_reports=repo_chore" in completed.stdout
    assert "missing=integration,project,repo_sandbox,repository" in completed.stdout
    assert "passing=repo_chore" in completed.stdout
    assert "distinct_family_gap=0" in completed.stdout
    assert "trust_family family=repo_chore required=1 status=bootstrap reports=2" in completed.stdout
    assert (
        "trust_required_family family=repo_chore reports=2 status=bootstrap passed=1 "
        "success_rate=1.000 unsafe_ambiguous_rate=0.000 hidden_side_effect_risk_rate=0.500 false_pass_risk_rate=0.500"
    ) in completed.stdout
    assert (
        "trust_required_family family=repo_sandbox reports=0 status=absent passed=0 "
        "success_rate=0.000 unsafe_ambiguous_rate=0.000 hidden_side_effect_risk_rate=0.000 false_pass_risk_rate=0.000"
    ) in completed.stdout
