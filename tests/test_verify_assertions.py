from pathlib import Path
import subprocess
import sys


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_assert_verify_metrics_accepts_healthy_logs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "assert_verify_metrics.py"
    full_eval = _write(
        tmp_path / "full.txt",
        """
passed=87 total=88 pass_rate=0.99
generated_passed=99 generated_total=176 generated_pass_rate=0.56
generated_kind=adjacent_success passed=87 total=87 pass_rate=1.00
generated_kind=failure_recovery passed=12 total=89 pass_rate=0.13
""",
    )
    tolbert_compare = _write(
        tmp_path / "tolbert_compare.txt",
        """
tolbert_compare pass_rate_delta=0.03 average_steps_delta=-0.10
tolbert_compare benchmark_family=workflow pass_rate_delta=0.00
tolbert_compare benchmark_family=project pass_rate_delta=0.25
tolbert_compare benchmark_family=repository pass_rate_delta=0.25
tolbert_compare benchmark_family=tooling pass_rate_delta=0.10
""",
    )
    tolbert_features = _write(
        tmp_path / "tolbert_features.txt",
        """
tolbert_mode mode=path_only passed=30 total=40 pass_rate=0.75 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=retrieval_only passed=31 total=40 pass_rate=0.78 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=deterministic_command passed=30 total=40 pass_rate=0.75 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=skill_ranking passed=31 total=40 pass_rate=0.78 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=full passed=33 total=40 pass_rate=0.82 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
""",
    )
    skill_compare = _write(
        tmp_path / "skill_compare.txt",
        """
skill_compare pass_rate_delta=0.30 average_steps_delta=-0.30
""",
    )
    baseline_eval = _write(
        tmp_path / "baseline.txt",
        """
passed=372 total=440 pass_rate=0.85
""",
    )
    tolbert_first_steps = _write(
        tmp_path / "tolbert_first_steps.txt",
        """
family=workflow task_id=wf1 success=false termination_reason=failed path_confidence=0.72 trust_retrieval=true selected_skill_id=skill:one selected_retrieval_span_id=span:1 retrieval_ranked_skill=true action=execute content="run check"
""",
    )
    failure_recovery = _write(
        tmp_path / "failure_recovery.txt",
        """
failure_recovery total=1 passed=1 failed=0 pass_rate=1.00 families=1
failure_recovery_family benchmark_family=workflow total=1 passed=1 failed=0 pass_rate=1.00
family=workflow task_id=fr1 success=false termination_reason=failed source_task=wf0 failed_command="python app.py" selected_skill_id=skill:recover selected_retrieval_span_id=span:2 first_action=execute first_content="retry with fix" reference_task_ids=["wf0"] reference_commands=["python app.py"] suggested_commands=["pytest -q"] executed_commands=["pytest -q"]
""",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-eval",
            str(full_eval),
            "--tolbert-compare",
            str(tolbert_compare),
            "--tolbert-features",
            str(tolbert_features),
            "--skill-compare",
            str(skill_compare),
            "--baseline-eval",
            str(baseline_eval),
            "--tolbert-first-steps",
            str(tolbert_first_steps),
            "--failure-recovery",
            str(failure_recovery),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "empirical gates passed" in completed.stdout


def test_assert_verify_metrics_rejects_tolbert_regression(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "assert_verify_metrics.py"
    full_eval = _write(
        tmp_path / "full.txt",
        """
passed=87 total=88 pass_rate=0.99
generated_passed=99 generated_total=176 generated_pass_rate=0.56
generated_kind=adjacent_success passed=87 total=87 pass_rate=1.00
generated_kind=failure_recovery passed=12 total=89 pass_rate=0.13
""",
    )
    tolbert_compare = _write(
        tmp_path / "tolbert_compare.txt",
        """
tolbert_compare pass_rate_delta=-0.31 average_steps_delta=-0.40
tolbert_compare benchmark_family=workflow pass_rate_delta=-0.25
tolbert_compare benchmark_family=project pass_rate_delta=-1.00
tolbert_compare benchmark_family=repository pass_rate_delta=-0.75
tolbert_compare benchmark_family=tooling pass_rate_delta=-0.50
""",
    )
    tolbert_features = _write(
        tmp_path / "tolbert_features.txt",
        """
tolbert_mode mode=path_only passed=35 total=40 pass_rate=0.88 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=retrieval_only passed=36 total=40 pass_rate=0.90 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=deterministic_command passed=37 total=40 pass_rate=0.92 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=skill_ranking passed=36 total=40 pass_rate=0.90 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=full passed=30 total=40 pass_rate=0.75 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
""",
    )
    skill_compare = _write(
        tmp_path / "skill_compare.txt",
        """
skill_compare pass_rate_delta=0.30 average_steps_delta=-0.30
""",
    )
    baseline_eval = _write(
        tmp_path / "baseline.txt",
        """
passed=372 total=440 pass_rate=0.85
""",
    )
    tolbert_first_steps = _write(
        tmp_path / "tolbert_first_steps.txt",
        """
family=workflow task_id=wf1 success=false termination_reason=failed path_confidence=0.72 trust_retrieval=true selected_skill_id=skill:one selected_retrieval_span_id=span:1 retrieval_ranked_skill=true action=execute content="run check"
""",
    )
    failure_recovery = _write(
        tmp_path / "failure_recovery.txt",
        """
failure_recovery total=1 passed=1 failed=0 pass_rate=1.00 families=1
failure_recovery_family benchmark_family=workflow total=1 passed=1 failed=0 pass_rate=1.00
family=workflow task_id=fr1 success=false termination_reason=failed source_task=wf0 failed_command="python app.py" selected_skill_id=skill:recover selected_retrieval_span_id=span:2 first_action=execute first_content="retry with fix" reference_task_ids=["wf0"] reference_commands=["python app.py"] suggested_commands=["pytest -q"] executed_commands=["pytest -q"]
""",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-eval",
            str(full_eval),
            "--tolbert-compare",
            str(tolbert_compare),
            "--tolbert-features",
            str(tolbert_features),
            "--skill-compare",
            str(skill_compare),
            "--baseline-eval",
            str(baseline_eval),
            "--tolbert-first-steps",
            str(tolbert_first_steps),
            "--failure-recovery",
            str(failure_recovery),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "compare-tolbert regressed overall fixed-bank pass rate" in completed.stderr
    assert "full Tolbert mode underperformed isolated feature modes" in completed.stderr


def test_assert_verify_metrics_rejects_inert_tolbert_feature_lane(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "assert_verify_metrics.py"
    full_eval = _write(
        tmp_path / "full.txt",
        """
passed=88 total=88 pass_rate=1.00
generated_passed=84 generated_total=176 generated_pass_rate=0.48
generated_kind=adjacent_success passed=80 total=88 pass_rate=0.91
generated_kind=failure_recovery passed=4 total=88 pass_rate=0.05
""",
    )
    tolbert_compare = _write(
        tmp_path / "tolbert_compare.txt",
        """
tolbert_compare pass_rate_delta=0.01 average_steps_delta=-0.05
tolbert_compare benchmark_family=workflow pass_rate_delta=0.00
tolbert_compare benchmark_family=project pass_rate_delta=0.00
tolbert_compare benchmark_family=repository pass_rate_delta=0.00
tolbert_compare benchmark_family=tooling pass_rate_delta=0.00
""",
    )
    tolbert_features = _write(
        tmp_path / "tolbert_features.txt",
        """
tolbert_mode mode=path_only passed=13 total=40 pass_rate=0.33 average_steps=1.68 retrieval_influenced_steps=40 retrieval_ranked_skill_steps=0 retrieval_selected_steps=0
tolbert_mode mode=retrieval_only passed=13 total=40 pass_rate=0.33 average_steps=1.68 retrieval_influenced_steps=40 retrieval_ranked_skill_steps=0 retrieval_selected_steps=0
tolbert_mode mode=deterministic_command passed=13 total=40 pass_rate=0.33 average_steps=1.68 retrieval_influenced_steps=40 retrieval_ranked_skill_steps=0 retrieval_selected_steps=0
tolbert_mode mode=skill_ranking passed=13 total=40 pass_rate=0.33 average_steps=1.68 retrieval_influenced_steps=40 retrieval_ranked_skill_steps=0 retrieval_selected_steps=0
tolbert_mode mode=full passed=13 total=40 pass_rate=0.33 average_steps=1.68 retrieval_influenced_steps=40 retrieval_ranked_skill_steps=0 retrieval_selected_steps=0
""",
    )
    skill_compare = _write(
        tmp_path / "skill_compare.txt",
        """
skill_compare pass_rate_delta=0.50 average_steps_delta=-0.50
""",
    )
    baseline_eval = _write(
        tmp_path / "baseline.txt",
        """
passed=84 total=122 pass_rate=0.69
""",
    )
    tolbert_first_steps = _write(
        tmp_path / "tolbert_first_steps.txt",
        """
family=workflow task_id=wf1 success=false termination_reason=failed path_confidence=0.75 trust_retrieval=true selected_skill_id=skill:one selected_retrieval_span_id=span:1 retrieval_ranked_skill=true action=execute content="run check"
""",
    )
    failure_recovery = _write(
        tmp_path / "failure_recovery.txt",
        """
failure_recovery total=1 passed=1 failed=0 pass_rate=1.00 families=1
failure_recovery_family benchmark_family=workflow total=1 passed=1 failed=0 pass_rate=1.00
family=workflow task_id=fr1 success=false termination_reason=failed source_task=wf0 failed_command="python app.py" selected_skill_id=skill:recover selected_retrieval_span_id=span:2 first_action=execute first_content="retry with fix" reference_task_ids=["wf0"] reference_commands=["python app.py"] suggested_commands=["pytest -q"] executed_commands=["pytest -q"]
""",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-eval",
            str(full_eval),
            "--tolbert-compare",
            str(tolbert_compare),
            "--tolbert-features",
            str(tolbert_features),
            "--skill-compare",
            str(skill_compare),
            "--baseline-eval",
            str(baseline_eval),
            "--tolbert-first-steps",
            str(tolbert_first_steps),
            "--failure-recovery",
            str(failure_recovery),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "identical metrics for every Tolbert mode" in completed.stderr
    assert "no skill-ranking or retrieval-selection activity" in completed.stderr


def test_assert_verify_metrics_rejects_missing_diagnostic_signal(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "assert_verify_metrics.py"
    full_eval = _write(
        tmp_path / "full.txt",
        """
passed=87 total=88 pass_rate=0.99
generated_passed=99 generated_total=176 generated_pass_rate=0.56
generated_kind=adjacent_success passed=87 total=87 pass_rate=1.00
generated_kind=failure_recovery passed=12 total=89 pass_rate=0.13
""",
    )
    tolbert_compare = _write(
        tmp_path / "tolbert_compare.txt",
        """
tolbert_compare pass_rate_delta=0.03 average_steps_delta=-0.10
tolbert_compare benchmark_family=workflow pass_rate_delta=0.00
tolbert_compare benchmark_family=project pass_rate_delta=0.25
tolbert_compare benchmark_family=repository pass_rate_delta=0.25
tolbert_compare benchmark_family=tooling pass_rate_delta=0.10
""",
    )
    tolbert_features = _write(
        tmp_path / "tolbert_features.txt",
        """
tolbert_mode mode=path_only passed=30 total=40 pass_rate=0.75 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=retrieval_only passed=31 total=40 pass_rate=0.78 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=deterministic_command passed=30 total=40 pass_rate=0.75 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=skill_ranking passed=31 total=40 pass_rate=0.78 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
tolbert_mode mode=full passed=33 total=40 pass_rate=0.82 average_steps=1.00 retrieval_influenced_steps=10 retrieval_ranked_skill_steps=5 retrieval_selected_steps=5
""",
    )
    skill_compare = _write(
        tmp_path / "skill_compare.txt",
        """
skill_compare pass_rate_delta=0.30 average_steps_delta=-0.30
""",
    )
    baseline_eval = _write(
        tmp_path / "baseline.txt",
        """
passed=372 total=440 pass_rate=0.85
""",
    )
    tolbert_first_steps = _write(
        tmp_path / "tolbert_first_steps.txt",
        """
family=workflow task_id=wf1 success=false termination_reason=failed path_confidence=0.00 trust_retrieval=false selected_skill_id= selected_retrieval_span_id= retrieval_ranked_skill=false action= content=""
""",
    )
    failure_recovery = _write(
        tmp_path / "failure_recovery.txt",
        """
failure_recovery total=1 passed=0 failed=1 pass_rate=0.00 families=1
failure_recovery_family benchmark_family=workflow total=1 passed=0 failed=1 pass_rate=0.00
family=workflow task_id=fr1 success=false termination_reason=failed source_task=wf0 failed_command="python app.py" selected_skill_id= selected_retrieval_span_id= first_action= first_content="" reference_task_ids=[] reference_commands=[] suggested_commands=[] executed_commands=[]
""",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-eval",
            str(full_eval),
            "--tolbert-compare",
            str(tolbert_compare),
            "--tolbert-features",
            str(tolbert_features),
            "--skill-compare",
            str(skill_compare),
            "--baseline-eval",
            str(baseline_eval),
            "--tolbert-first-steps",
            str(tolbert_first_steps),
            "--failure-recovery",
            str(failure_recovery),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "failure-recovery diagnostic showed zero recovery pass rate" in completed.stderr
    assert "failure-recovery diagnostic showed zero pass rate in monitored families" in completed.stderr
    assert "Tolbert first-step diagnostic showed zero path confidence" in completed.stderr
    assert "failure-recovery diagnostic showed no first-step guidance" in completed.stderr
