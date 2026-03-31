from pathlib import Path
import subprocess


def test_verify_impl_script_has_valid_bash_syntax():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "verify_impl.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_verify_impl_exports_all_asi_artifact_paths():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "verify_impl.sh").read_text(encoding="utf-8")

    assert "AGENT_KERNEL_OPERATOR_CLASSES_PATH" in script
    assert "AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH" in script
    assert "AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH" in script
    assert "AGENT_KERNEL_VERIFIER_CONTRACTS_PATH" in script
    assert "AGENT_KERNEL_PROMPT_PROPOSALS_PATH" in script
    assert "AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH" in script
    assert "AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH" in script
    assert "AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR" in script
    assert "--compare-tolbert-features" in script
    assert "assert_verify_metrics.py" in script
    assert "AGENT_KERNEL_VERIFY_ARCHIVE_ROOT" in script
    assert "archive_stage" in script
    assert "manifest.json" in script
    assert "report_tolbert_first_steps.py" in script
    assert "report_failure_recovery.py" in script
    assert "tolbert_first_steps.txt" in script
    assert "failure_recovery.txt" in script
    assert "run_eval_compare_tolbert" in script
    assert "run_eval_compare_tolbert_features" in script
    assert "--use-skills 1" in script
    assert "extract_operators.py" in script
