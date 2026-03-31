from pathlib import Path
import json
import subprocess
import sys


def test_report_failure_recovery_filters_failures_and_emits_recovery_fields(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "report_failure_recovery.py"
    trajectories_root = tmp_path / "trajectories"
    trajectories_root.mkdir()
    trajectories_root.joinpath("project_recovery_fail.json").write_text(
        json.dumps(
            {
                "task_id": "project_recovery_fail",
                "success": False,
                "termination_reason": "repeated_failed_action",
                "task_metadata": {
                    "benchmark_family": "project",
                    "curriculum_kind": "failure_recovery",
                    "parent_task": "project_seed_task",
                    "failed_command": "false",
                    "reference_task_ids": ["project_seed_task", "project_adjacent"],
                    "reference_commands": ["printf 'done\\n' > status.txt"],
                    "failure_types": ["missing_expected_file"],
                },
                "task_contract": {
                    "suggested_commands": [
                        "printf 'file recovery complete\\n' > recovery.txt",
                        "cat recovery.txt",
                    ]
                },
                "summary": {
                    "executed_commands": ["printf 'oops\\n' > wrong.txt"],
                },
                "steps": [
                    {
                        "selected_skill_id": "skill:hello_task:primary",
                        "selected_retrieval_span_id": "span:project",
                        "action": "code_execute",
                        "content": "printf 'oops\\n' > wrong.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trajectories_root.joinpath("workflow_recovery_success.json").write_text(
        json.dumps(
            {
                "task_id": "workflow_recovery_success",
                "success": True,
                "termination_reason": "success",
                "task_metadata": {
                    "benchmark_family": "workflow",
                    "curriculum_kind": "failure_recovery",
                },
                "task_contract": {"suggested_commands": ["printf 'ok\\n' > recovery.txt"]},
                "summary": {"executed_commands": ["printf 'ok\\n' > recovery.txt"]},
                "steps": [{"action": "code_execute", "content": "printf 'ok\\n' > recovery.txt"}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--trajectories-root",
            str(trajectories_root),
            "--benchmark-family",
            "project",
            "--failures-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "failure_recovery total=1 passed=0 failed=1 pass_rate=0.00 families=1" in completed.stdout
    assert "failure_recovery_family benchmark_family=project total=1 passed=0 failed=1 pass_rate=0.00" in completed.stdout
    assert "task_id=project_recovery_fail" in completed.stdout
    assert "family=project" in completed.stdout
    assert 'source_task=project_seed_task' in completed.stdout
    assert 'failed_command="false"' in completed.stdout
    assert 'selected_skill_id=skill:hello_task:primary' in completed.stdout
    assert 'reference_task_ids=["project_seed_task", "project_adjacent"]' in completed.stdout
    assert 'suggested_commands=["printf \'file recovery complete\\\\n\' > recovery.txt", "cat recovery.txt"]' in completed.stdout
    assert "task_id=workflow_recovery_success" not in completed.stdout


def test_report_failure_recovery_recurses_into_generated_failure_directories(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "report_failure_recovery.py"
    episodes_root = tmp_path / "episodes"
    generated_root = episodes_root / "generated_failure"
    generated_root.mkdir(parents=True)
    generated_root.joinpath("nested_recovery_fail.json").write_text(
        json.dumps(
            {
                "task_id": "nested_recovery_fail",
                "success": False,
                "termination_reason": "repeated_failed_action",
                "task_metadata": {
                    "benchmark_family": "repository",
                    "curriculum_kind": "failure_recovery",
                    "parent_task": "repo_seed_task",
                    "failed_command": "false",
                },
                "task_contract": {"suggested_commands": ["printf 'fixed\\n' > status.txt"]},
                "summary": {"executed_commands": ["false"]},
                "steps": [{"action": "code_execute", "content": "false"}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--trajectories-root",
            str(episodes_root),
            "--failures-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "failure_recovery total=1 passed=0 failed=1 pass_rate=0.00 families=1" in completed.stdout
    assert "task_id=nested_recovery_fail" in completed.stdout
