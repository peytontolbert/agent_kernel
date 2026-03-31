from pathlib import Path
import json
import subprocess
import sys


def test_report_tolbert_first_steps_filters_failures_and_families(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "report_tolbert_first_steps.py"
    trajectories_root = tmp_path / "trajectories"
    trajectories_root.mkdir()
    trajectories_root.joinpath("project_fail.json").write_text(
        json.dumps(
            {
                "task_id": "project_fail",
                "success": False,
                "termination_reason": "repeated_failed_action",
                "task_metadata": {"benchmark_family": "project"},
                "steps": [
                    {
                        "path_confidence": 0.41,
                        "trust_retrieval": True,
                        "selected_skill_id": None,
                        "selected_retrieval_span_id": "span:project",
                        "retrieval_ranked_skill": False,
                        "action": "code_execute",
                        "content": "printf 'bad\\n' > wrong.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trajectories_root.joinpath("workflow_success.json").write_text(
        json.dumps(
            {
                "task_id": "workflow_success",
                "success": True,
                "termination_reason": "success",
                "task_metadata": {"benchmark_family": "workflow"},
                "steps": [
                    {
                        "path_confidence": 0.95,
                        "trust_retrieval": True,
                        "selected_skill_id": "skill:workflow:primary",
                        "selected_retrieval_span_id": "span:workflow",
                        "retrieval_ranked_skill": True,
                        "action": "code_execute",
                        "content": "printf 'ok\\n' > audit.txt",
                    }
                ],
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
    assert "task_id=project_fail" in completed.stdout
    assert "family=project" in completed.stdout
    assert "task_id=workflow_success" not in completed.stdout
