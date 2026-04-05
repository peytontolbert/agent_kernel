from pathlib import Path
from dataclasses import asdict, replace
import importlib.util
import itertools
import json
from io import StringIO
from subprocess import CompletedProcess
import subprocess
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementExperiment, ImprovementPlanner, ImprovementVariant
from evals.harness import scoped_improvement_cycle_config
from evals.metrics import EvalMetrics


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_human_guided_improvement_cycle_cli_help_executes():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_human_guided_improvement_cycle.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--subsystem" in completed.stdout


def test_run_supervised_improvement_cycle_cli_help_executes():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_supervised_improvement_cycle.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--subsystem" in completed.stdout


def test_run_human_guided_improvement_cycle_enables_shared_repo_operator_policy_for_bounded_project_runs():
    module = _load_script("run_human_guided_improvement_cycle.py")

    reason = module._observation_operator_policy_override_reason(
        eval_kwargs={
            "task_limit": 3,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
        },
        max_observation_seconds=60.0,
    )

    assert reason == "shared_repo_long_horizon_project_guard"
    assert (
        module._observation_operator_policy_override_reason(
            eval_kwargs={
                "task_limit": 5,
                "priority_benchmark_families": ["project"],
                "prefer_low_cost_tasks": True,
            },
            max_observation_seconds=60.0,
        )
        == "shared_repo_long_horizon_project_guard"
    )
    assert (
        module._observation_operator_policy_override_reason(
            eval_kwargs={
                "task_limit": 5,
                "priority_benchmark_families": ["project"],
                "prefer_low_cost_tasks": True,
            },
            max_observation_seconds=80.0,
        )
        == "shared_repo_long_horizon_project_guard"
    )
    assert (
        module._observation_operator_policy_override_reason(
            eval_kwargs={
                "task_limit": 3,
                "priority_benchmark_families": ["repo_sandbox"],
                "prefer_low_cost_tasks": True,
            },
            max_observation_seconds=60.0,
        )
        == "repo_sandbox_git_workflow_guard"
    )
    assert (
        module._observation_operator_policy_override_reason(
            eval_kwargs={
                "task_limit": 6,
                "priority_benchmark_families": ["repository"],
                "prefer_low_cost_tasks": True,
            },
            max_observation_seconds=60.0,
        )
        == ""
    )


def test_run_human_guided_improvement_cycle_expands_budget_for_long_horizon_project_integrators():
    module = _load_script("run_human_guided_improvement_cycle.py")

    budget_seconds, source = module._resolve_long_horizon_project_observation_budget_seconds(
        eval_kwargs={
            "task_limit": 5,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
        },
        max_observation_seconds=60.0,
    )

    assert budget_seconds == 80.0
    assert source == "shared_repo_integrator_budget_guard"
    unchanged_budget_seconds, unchanged_source = (
        module._resolve_long_horizon_project_observation_budget_seconds(
            eval_kwargs={
                "task_limit": 4,
                "priority_benchmark_families": ["project"],
                "prefer_low_cost_tasks": True,
            },
            max_observation_seconds=60.0,
        )
    )
    assert unchanged_budget_seconds == 60.0
    assert unchanged_source == ""


def test_run_human_guided_improvement_cycle_expands_budget_for_long_horizon_generated_success_followup():
    module = _load_script("run_human_guided_improvement_cycle.py")

    budget_seconds, source = module._resolve_long_horizon_generated_success_followup_budget_seconds(
        followup_kind="generated_success",
        eval_kwargs={
            "task_limit": 1,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
            "generated_success_seed_documents_path": "/tmp/generated_success_wave_seeds.json",
        },
        budget_seconds=40.0,
    )

    assert budget_seconds == 80.0
    assert source == "shared_repo_integrator_followup_budget_guard"
    unchanged_budget_seconds, unchanged_source = module._resolve_long_horizon_generated_success_followup_budget_seconds(
        followup_kind="generated_failure",
        eval_kwargs={
            "task_limit": 1,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
            "generated_success_seed_documents_path": "/tmp/generated_success_wave_seeds.json",
        },
        budget_seconds=40.0,
    )
    assert unchanged_budget_seconds == 40.0
    assert unchanged_source == ""


def test_generated_metrics_from_partial_summary_salvages_verified_single_generated_success():
    module = _load_script("run_human_guided_improvement_cycle.py")

    metrics = module._generated_metrics_from_partial_summary(
        followup_kind="generated_success",
        partial_summary={
            "phase": "generated_success",
            "completed_generated_tasks": 0,
            "generated_passed": 0,
            "current_task_total": 1,
            "current_task_completed_steps": 9,
            "current_task_step_stage": "step_complete",
            "current_task_verification_passed": True,
            "current_task_benchmark_family": "repository",
        },
    )

    assert metrics is not None
    assert metrics.generated_total == 1
    assert metrics.generated_passed == 1
    assert metrics.generated_by_kind == {"adjacent_success": 1}
    assert metrics.generated_passed_by_kind == {"adjacent_success": 1}
    assert metrics.generated_by_benchmark_family == {"repository": 1}
    assert metrics.generated_passed_by_benchmark_family == {"repository": 1}


def test_generated_metrics_from_partial_summary_adds_verified_current_generated_success_to_existing_counts():
    module = _load_script("run_human_guided_improvement_cycle.py")

    metrics = module._generated_metrics_from_partial_summary(
        followup_kind="generated_success",
        partial_summary={
            "phase": "generated_success",
            "generated_tasks_scheduled": 2,
            "completed_generated_tasks": 1,
            "generated_passed": 1,
            "generated_by_kind": {"adjacent_success": 1},
            "generated_passed_by_kind": {"adjacent_success": 1},
            "generated_by_benchmark_family": {"repository": 1},
            "generated_passed_by_benchmark_family": {"repository": 1},
            "last_completed_generated_task_id": "repo_bridge_adjacent",
            "last_completed_generated_benchmark_family": "repository",
            "current_task_id": "workflow_bridge_adjacent",
            "current_task_total": 2,
            "current_task_completed_steps": 7,
            "current_task_step_stage": "step_complete",
            "current_task_verification_passed": True,
            "current_task_benchmark_family": "workflow",
        },
    )

    assert metrics is not None
    assert metrics.generated_total == 2
    assert metrics.generated_passed == 2
    assert metrics.generated_by_kind == {"adjacent_success": 2}
    assert metrics.generated_passed_by_kind == {"adjacent_success": 2}
    assert metrics.generated_by_benchmark_family == {"repository": 1, "workflow": 1}
    assert metrics.generated_passed_by_benchmark_family == {"repository": 1, "workflow": 1}


def test_merge_partial_summary_preserves_verified_current_generated_task_when_later_snapshot_clears_it():
    module = _load_script("run_human_guided_improvement_cycle.py")

    merged = module._merge_partial_summary(
        {
            "phase": "generated_success",
            "generated_tasks_scheduled": 1,
            "completed_generated_tasks": 0,
            "current_task_id": "",
            "current_task_progress_timeline": [],
        },
        {
            "phase": "generated_success",
            "generated_tasks_scheduled": 1,
            "completed_generated_tasks": 0,
            "current_task_id": "repo_docs_adjacent",
            "current_task_total": 1,
            "current_task_completed_steps": 5,
            "current_task_step_stage": "step_complete",
            "current_task_verification_passed": True,
            "current_task_benchmark_family": "repository",
            "current_task_progress_timeline": [{"event": "step_complete", "completed_steps": 5}],
        },
    )

    assert merged["current_task_id"] == "repo_docs_adjacent"
    assert merged["current_task_verification_passed"] is True
    assert merged["current_task_step_stage"] == "step_complete"
    assert merged["current_task_progress_timeline"] == [{"event": "step_complete", "completed_steps": 5}]

    metrics = module._generated_metrics_from_partial_summary(
        followup_kind="generated_success",
        partial_summary=merged,
    )

    assert metrics is not None
    assert metrics.generated_total == 1
    assert metrics.generated_passed == 1
    assert metrics.generated_by_benchmark_family == {"repository": 1}


def test_run_human_guided_improvement_cycle_backfills_task_outcomes_from_partial_summary():
    module = _load_script("run_human_guided_improvement_cycle.py")
    metrics = EvalMetrics(
        total=5,
        passed=5,
        trusted_retrieval_steps=1,
        retrieval_influenced_steps=1,
    )

    enriched = module._enrich_metrics_task_outcomes_from_partial_summaries(
        metrics,
        {
            "completed_task_summaries": {
                "repo_sync_matrix_task": {
                    "termination_reason": "success",
                    "benchmark_family": "repository",
                    "retrieval_selected_steps": 1,
                    "retrieval_influenced_steps": 0,
                    "trusted_retrieval_steps": 0,
                    "failure_signals": ["inference_failure"],
                },
                "git_generated_conflict_resolution_task__worker__worker_status-refresh": {
                    "termination_reason": "success",
                    "benchmark_family": "project",
                    "retrieval_selected_steps": 1,
                    "retrieval_influenced_steps": 1,
                    "trusted_retrieval_steps": 1,
                    "last_selected_retrieval_span_id": "learning:success_skill:worker_status_refresh",
                    "failure_signals": [],
                },
            }
        },
    )

    assert metrics.task_outcomes == {}
    assert enriched.task_outcomes["repo_sync_matrix_task"]["retrieval_selected_steps"] == 1
    assert enriched.task_outcomes["repo_sync_matrix_task"]["retrieval_influenced_steps"] == 0
    assert enriched.task_outcomes["repo_sync_matrix_task"]["benchmark_family"] == "repository"
    assert enriched.task_outcomes["repo_sync_matrix_task"]["failure_signals"] == ["inference_failure"]
    assert (
        enriched.task_outcomes["git_generated_conflict_resolution_task__worker__worker_status-refresh"][
            "trusted_retrieval_steps"
        ]
        == 1
    )
    assert (
        enriched.task_outcomes["git_generated_conflict_resolution_task__worker__worker_status-refresh"][
            "last_selected_retrieval_span_id"
        ]
        == "learning:success_skill:worker_status_refresh"
    )


def test_run_human_guided_improvement_cycle_disables_tolbert_for_long_horizon_project_integrators():
    module = _load_script("run_human_guided_improvement_cycle.py")

    reason = module._observation_tolbert_context_disabled_reason(
        subsystem="tooling",
        eval_kwargs={
            "task_limit": 5,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
        },
        max_observation_seconds=80.0,
    )

    assert reason == "shared_repo_integrator_context_guard"


def test_run_parallel_supervised_cycles_cli_help_executes():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_parallel_supervised_cycles.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--workers" in completed.stdout


def test_run_human_guided_improvement_cycle_records_protocol(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_human_guided_improvement_cycle.py", "--subsystem", "policy", "--notes", "human chose verifier alignment"],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["state"] for record in records[:3]] == ["observe", "select", "generate"]
    assert records[0]["metrics_summary"]["protocol"] == "human_guided"
    assert records[0]["metrics_summary"]["protocol_strategy"] == "careful_guided"
    assert records[1]["metrics_summary"]["guidance_notes"] == "human chose verifier alignment"


def test_run_human_guided_improvement_cycle_can_auto_select_subsystem_and_variant(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda **kwargs: EvalMetrics(total=10, passed=8, average_steps=1.0),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=3,
                expected_gain=0.03,
                estimated_cost=4,
                score=0.10,
                evidence={},
            ),
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.025,
                estimated_cost=2,
                score=0.09,
                evidence={},
            ),
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem=experiment.subsystem,
                variant_id="expensive",
                description="higher-cost variant",
                expected_gain=0.02,
                estimated_cost=3,
                score=0.02,
                controls={"focus": "expensive"},
            ),
            ImprovementVariant(
                subsystem=experiment.subsystem,
                variant_id="careful",
                description="lower-cost careful variant",
                expected_gain=0.018,
                estimated_cost=1,
                score=0.018,
                controls={"focus": "careful"},
            ),
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("retain", "finalized"))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_human_guided_improvement_cycle.py", "--protocol-match-id", "match:test:1"],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records[0]["subsystem"] == "policy"
    assert records[0]["metrics_summary"]["protocol_match_id"] == "match:test:1"
    assert records[1]["metrics_summary"]["selected_variant"]["variant_id"] == "careful"


def test_run_supervised_cycle_records_hard_observation_timeout_and_exits(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 102.1])

    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module,
        "_run_observation_eval",
        lambda **kwargs: {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": "observation child exceeded max runtime of 2.0 seconds",
            "returncode": -9,
            "error": "",
            "last_progress_line": "[eval:timeout-test] phase=generated_success_schedule family=workflow",
            "last_progress_phase": "generated_success_schedule",
            "last_progress_task_id": "",
            "last_progress_benchmark_family": "workflow",
            "partial_summary": {
                "completed_primary_tasks": 1,
                "current_task_completed_steps": 0,
                "current_task_step_index": 1,
                "current_task_step_stage": "decision_pending",
                "retrieval_selected_steps": 1,
                "retrieval_influenced_steps": 1,
                "trusted_retrieval_steps": 1,
                "selected_retrieval_span_ids": ["learning:success_skill:workflow_ready"],
                "last_selected_retrieval_span_id": "learning:success_skill:workflow_ready",
                "retrieval_influenced_task_ids": ["workflow_ready"],
                "observed_benchmark_families": ["workflow"],
                "last_completed_task_id": "workflow_ready",
                "scheduled_task_summaries": {
                    "workflow_ready": {"benchmark_family": "workflow"},
                },
            },
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--max-observation-seconds",
            "2",
            "--notes",
            "timing check",
        ],
    )

    try:
        module.main()
        raise AssertionError("expected SystemExit for hard observation timeout")
    except SystemExit as exc:
        assert "observation child exceeded max runtime of 2.0 seconds" in str(exc)

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["state"] for record in records] == ["observe"]
    assert records[0]["subsystem"] == "policy"
    observe = records[0]["metrics_summary"]
    assert observe["observation_elapsed_seconds"] == 2.1
    assert observe["observation_budget_seconds"] == 2.0
    assert observe["observation_budget_exceeded"] is True
    assert observe["observation_timed_out"] is True
    assert observe["observation_mode"] == "child_process"
    assert observe["observation_returncode"] == -9
    assert observe["observation_warning"] == "observation child exceeded max runtime of 2.0 seconds"
    assert observe["observation_last_progress_line"] == "[eval:timeout-test] phase=generated_success_schedule family=workflow"
    assert observe["observation_last_progress_phase"] == "generated_success_schedule"
    assert observe["observation_last_progress_benchmark_family"] == "workflow"
    assert observe["observation_initial_last_progress_line"] == "[eval:timeout-test] phase=generated_success_schedule family=workflow"
    assert observe["observation_initial_last_progress_benchmark_family"] == "workflow"
    assert observe["observation_partial_tasks_completed"] == 1
    assert observe["observation_partial_observed_benchmark_families"] == ["workflow"]
    assert observe["observation_partial_summary"]["last_completed_task_id"] == "workflow_ready"
    assert observe["observation_partial_current_task_completed_steps"] == 0
    assert observe["observation_partial_current_task_step_index"] == 1
    assert observe["observation_partial_current_task_step_stage"] == "decision_pending"
    assert observe["observation_partial_current_task_benchmark_family"] == ""
    assert observe["observation_partial_retrieval_selected_steps"] == 1
    assert observe["observation_partial_retrieval_influenced_steps"] == 1
    assert observe["observation_partial_trusted_retrieval_steps"] == 1
    assert observe["observation_partial_selected_retrieval_span_ids"] == [
        "learning:success_skill:workflow_ready"
    ]
    assert observe["observation_partial_last_selected_retrieval_span_id"] == "learning:success_skill:workflow_ready"
    assert observe["observation_partial_retrieval_influenced_task_ids"] == ["workflow_ready"]


def test_run_supervised_cycle_observation_summary_surfaces_retrieval_activation(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    monkeypatch.setattr(
        module,
        "_run_observation_eval",
        lambda **kwargs: {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=1,
                passed=1,
                average_steps=1.0,
                retrieval_selected_steps=1,
                retrieval_influenced_steps=1,
                trusted_retrieval_steps=1,
                task_outcomes={
                    "workflow_guarded_task": {
                        "retrieval_influenced_steps": 1,
                    }
                },
                task_trajectories={
                    "workflow_guarded_task": {
                        "steps": [
                            {
                                "selected_retrieval_span_id": "learning:success_skill:workflow_guarded_project_task",
                            }
                        ]
                    }
                },
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="policy variant",
                expected_gain=0.01,
                estimated_cost=2,
                score=0.05,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["retrieval_selected_steps"] == 1
    assert observe["retrieval_influenced_steps"] == 1
    assert observe["trusted_retrieval_steps"] == 1
    assert observe["selected_retrieval_span_ids"] == ["learning:success_skill:workflow_guarded_project_task"]
    assert observe["last_selected_retrieval_span_id"] == "learning:success_skill:workflow_guarded_project_task"
    assert observe["retrieval_influenced_task_ids"] == ["workflow_guarded_task"]


def test_run_supervised_cycle_observation_respects_curriculum_flags(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observed = {}

    def fake_run_observation_eval(**kwargs):
        observed["eval_kwargs"] = dict(kwargs["eval_kwargs"])
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=4, passed=3, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="long horizon chained curriculum",
                priority=3,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="long_horizon_chain_bridge",
                description="extend generated success through bounded long-horizon family bridges",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "long_horizon_chain_bridge"},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
        ],
    )

    module.main()

    assert observed["eval_kwargs"]["include_generated"] is False
    assert observed["eval_kwargs"]["include_failure_generated"] is False


def test_run_supervised_cycle_stages_generated_curriculum_into_followup_observations(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.6, 100.9, 101.1])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
            }
        )
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=4,
                passed=3,
                average_steps=1.0,
                generated_total=1 if len(observation_calls) > 1 else 0,
                generated_passed=1 if len(observation_calls) > 1 else 0,
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-test] phase=generated_success total=1 family=bounded",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "hello_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "generated_success" if len(observation_calls) > 1 else "primary",
                "completed_primary_tasks": 4,
                "observed_benchmark_families": ["bounded", "workflow"],
                "last_completed_task_id": "hello_task",
                "last_completed_benchmark_family": "bounded",
                "generated_tasks_scheduled": 1 if len(observation_calls) > 1 else 0,
                "completed_generated_tasks": 1 if len(observation_calls) > 1 else 0,
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "2",
            "--include-curriculum",
            "--include-failure-curriculum",
            "--generated-curriculum-budget-seconds",
            "0.5",
            "--failure-curriculum-budget-seconds",
            "0.5",
        ],
    )

    module.main()

    assert len(observation_calls) == 3
    assert observation_calls[0]["eval_kwargs"]["include_generated"] is False
    assert observation_calls[0]["eval_kwargs"]["include_failure_generated"] is False
    assert observation_calls[1]["eval_kwargs"]["include_generated"] is True
    assert observation_calls[1]["eval_kwargs"]["include_failure_generated"] is False
    assert observation_calls[2]["eval_kwargs"]["include_generated"] is False
    assert observation_calls[2]["eval_kwargs"]["include_failure_generated"] is True
    assert observation_calls[1]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[2]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[1]["max_observation_seconds"] == 0.5
    assert observation_calls[2]["max_observation_seconds"] == 0.5
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_generated_curriculum"] is False
    assert observe["observation_generated_curriculum_budget_seconds"] == 0.5
    assert observe["observation_failure_curriculum_budget_seconds"] == 0.5
    assert observe["observation_elapsed_seconds"] == 1.1
    assert observe["observation_last_progress_phase"] == "generated_success"
    assert observe["observation_last_progress_line"] == "[eval:curriculum-test] phase=generated_success total=1 family=bounded"
    assert observe["observation_last_progress_benchmark_family"] == "bounded"
    assert observe["generated_total"] == 2
    assert observe["generated_passed"] == 2
    assert observe["generated_pass_rate"] == 1.0
    assert observe["generated_by_kind"] == {"adjacent_success": 1, "failure_recovery": 1}
    assert observe["generated_passed_by_kind"] == {"adjacent_success": 1, "failure_recovery": 1}
    assert len(observe["observation_curriculum_followups"]) == 2
    assert observe["observation_curriculum_followups"][0]["kind"] == "generated_success"
    assert observe["observation_curriculum_followups"][0]["merged_generated_metrics"] is True
    assert observe["observation_curriculum_followups"][0]["last_progress_benchmark_family"] == "bounded"
    assert observe["observation_curriculum_followups"][1]["kind"] == "generated_failure"
    assert observe["observation_curriculum_followups"][1]["merged_generated_metrics"] is True
    assert observe["observation_partial_tasks_completed"] == 4
    assert observe["observation_partial_last_completed_task_id"] == "hello_task"
    assert observe["observation_partial_observed_benchmark_families"] == ["bounded", "workflow"]
    assert observe["observation_partial_summary"]["generated_tasks_scheduled"] == 0
    assert observe["observation_partial_summary"]["phase"] == "primary"


def test_run_supervised_cycle_applies_bounded_default_priority_families(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 1,
            "include_generated": False,
            "include_failure_generated": False,
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(dict(kwargs["eval_kwargs"]))
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=1, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:priority-test] task 1/1 archive_command_seed_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "archive_command_seed_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 1,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "archive_command_seed_task",
                "last_completed_benchmark_family": "bounded",
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "12",
        ],
    )

    module.main()

    assert observation_calls[0]["priority_benchmark_families"] == ["bounded", "episode_memory", "tool_memory"]
    assert observation_calls[0]["prefer_low_cost_tasks"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_priority_source"] == "bounded_default"
    assert observe["observation_priority_benchmark_families"] == ["bounded", "episode_memory", "tool_memory"]
    assert observe["observation_task_selection_mode"] == "low_cost"


def test_run_supervised_cycle_preserves_explicit_priority_families(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 1,
            "include_generated": False,
            "include_failure_generated": False,
            "priority_benchmark_families": ["benchmark_candidate"],
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(dict(kwargs["eval_kwargs"]))
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=1, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:priority-test] task 1/1 api_contract_retrieval_task_benchmark_candidate family=benchmark_candidate",
            "last_progress_phase": "",
            "last_progress_task_id": "api_contract_retrieval_task_benchmark_candidate",
            "last_progress_benchmark_family": "benchmark_candidate",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 1,
                "observed_benchmark_families": ["benchmark_candidate"],
                "last_completed_task_id": "api_contract_retrieval_task_benchmark_candidate",
                "last_completed_benchmark_family": "benchmark_candidate",
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "12",
        ],
    )

    module.main()

    assert observation_calls[0]["priority_benchmark_families"] == ["benchmark_candidate"]
    assert observation_calls[0]["prefer_low_cost_tasks"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_priority_source"] == "explicit"
    assert observe["observation_priority_benchmark_families"] == ["benchmark_candidate"]
    assert observe["observation_task_selection_mode"] == "low_cost"


def test_run_supervised_cycle_balances_primary_task_limit_for_wide_generated_success_fanout(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.4, 100.9])

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 6,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "unattended_allow_git_commands": kwargs["config"].unattended_allow_git_commands,
                "unattended_allow_generated_path_mutations": kwargs["config"].unattended_allow_generated_path_mutations,
            }
        )
        seed_output_path = str(kwargs["eval_kwargs"].get("generated_success_seed_output_path", "")).strip()
        if seed_output_path:
            Path(seed_output_path).parent.mkdir(parents=True, exist_ok=True)
        if len(observation_calls) == 1:
            if seed_output_path:
                Path(seed_output_path).write_text(json.dumps({"episodes": []}), encoding="utf-8")
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=4,
                    passed=4,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 2, "project": 2},
                    passed_by_benchmark_family={"repository": 2, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:fanout-balance] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {
                    "phase": "complete",
                    "completed_primary_tasks": 4,
                    "observed_benchmark_families": ["repository", "project"],
                    "last_completed_task_id": "deployment_manifest_task",
                    "last_completed_benchmark_family": "project",
                },
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=4,
                generated_passed=4,
                generated_by_kind={"adjacent_success": 4},
                generated_passed_by_kind={"adjacent_success": 4},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:fanout-balance] phase=generated_success total=4 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "6",
            "--max-observation-seconds",
            "60",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "48",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 4
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 12.0
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 4
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_requested_primary_task_limit"] == 6
    assert observe["observation_primary_task_limit"] == 4
    assert observe["observation_primary_task_limit_source"] == "generated_success_family_balance"
    assert observe["observation_current_task_decision_budget_seconds"] == 12.0
    assert observe["generated_total"] == 4
    assert observe["generated_passed"] == 4


def test_run_supervised_cycle_preserves_five_task_tooling_primary_when_tolbert_budget_is_strong(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=5,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 3, "project": 2},
                    passed_by_benchmark_family={"repository": 3, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:fanout-preserve] task 5/5 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {
                    "phase": "complete",
                    "completed_primary_tasks": 5,
                    "observed_benchmark_families": ["repository", "project"],
                    "last_completed_task_id": "deployment_manifest_task",
                    "last_completed_benchmark_family": "project",
                },
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=5,
                generated_passed=5,
                generated_by_kind={"adjacent_success": 5},
                generated_passed_by_kind={"adjacent_success": 5},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:fanout-preserve] phase=generated_success total=5 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 5
    assert observation_calls[0]["use_tolbert_context"] is True
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 10.0
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 5
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_requested_primary_task_limit"] == 5
    assert observe["observation_primary_task_limit"] == 5
    assert observe["observation_primary_task_limit_source"] == ""
    assert observe["observation_tolbert_context_disabled_reason"] == ""
    assert observe["observation_current_task_decision_budget_seconds"] == 10.0
    assert observe["generated_total"] == 5
    assert observe["generated_passed"] == 5


def test_run_supervised_cycle_reduces_five_task_tooling_primary_when_generated_budget_cannot_support_full_fanout(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=4,
                    passed=4,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 2, "project": 2},
                    passed_by_benchmark_family={"repository": 2, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:fanout-guard] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {
                    "phase": "complete",
                    "completed_primary_tasks": 4,
                    "observed_benchmark_families": ["repository", "project"],
                    "last_completed_task_id": "deployment_manifest_task",
                    "last_completed_benchmark_family": "project",
                },
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=2,
                generated_passed=2,
                generated_by_kind={"adjacent_success": 2},
                generated_passed_by_kind={"adjacent_success": 2},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:fanout-guard] phase=generated_success total=2 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "20",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 4
    assert observation_calls[0]["use_tolbert_context"] is False
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 12.0
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 2
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_requested_primary_task_limit"] == 5
    assert observe["observation_primary_task_limit"] == 4
    assert observe["observation_primary_task_limit_source"] == "tooling_repository_project_fanout_guard"
    assert observe["observation_tolbert_context_disabled_reason"] == "tooling_repository_project_bounded_startup_guard"
    assert observe["generated_total"] == 2
    assert observe["generated_passed"] == 2


def test_run_supervised_cycle_expands_tooling_priority_families_for_high_task_limit(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 10,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )
    monkeypatch.setattr(
        module,
        "_task_bank_low_cost_family_counts",
        lambda: {
            "repository": 3,
            "project": 3,
            "workflow": 2,
            "repo_chore": 2,
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=10, passed=10, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:fanout-expand] task 10/10 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {"phase": "complete", "completed_primary_tasks": 10},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=10,
                generated_passed=10,
                generated_by_kind={"adjacent_success": 10},
                generated_passed_by_kind={"adjacent_success": 10},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:fanout-expand] phase=generated_success total=10 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "10",
            "--max-observation-seconds",
            "120",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "80",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 10
    assert observation_calls[0]["eval_kwargs"]["priority_benchmark_families"] == [
        "repository",
        "project",
        "workflow",
        "repo_chore",
    ]
    assert observation_calls[0]["eval_kwargs"].get("include_discovered_tasks", False) is False
    assert observation_calls[0]["use_tolbert_context"] is True
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 8.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_priority_expansion_source"] == "tooling_coding_family_scale"
    assert observe["observation_priority_benchmark_families"] == [
        "repository",
        "project",
        "workflow",
        "repo_chore",
    ]
    assert observe["observation_primary_task_limit"] == 10
    assert observe["generated_total"] == 10
    assert observe["generated_passed"] == 10


def test_run_supervised_cycle_enables_discovered_tasks_for_very_high_tooling_task_limit(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 50,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )
    monkeypatch.setattr(
        module,
        "_task_bank_low_cost_family_counts",
        lambda: {
            "repository": 3,
            "project": 3,
            "workflow": 2,
            "repo_chore": 2,
            "integration": 3,
            "repo_sandbox": 7,
            "tooling": 2,
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=50, passed=50, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:fanout-discovered] task 50/50 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {"phase": "complete", "completed_primary_tasks": 50},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=50,
                generated_passed=50,
                generated_by_kind={"adjacent_success": 50},
                generated_passed_by_kind={"adjacent_success": 50},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:fanout-discovered] phase=generated_success total=50 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "50",
            "--max-observation-seconds",
            "420",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "400",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 50
    assert observation_calls[0]["eval_kwargs"]["priority_benchmark_families"] == [
        "repository",
        "project",
        "workflow",
        "repo_chore",
        "integration",
        "repo_sandbox",
        "tooling",
    ]
    assert observation_calls[0]["eval_kwargs"]["include_discovered_tasks"] is True
    assert observation_calls[0]["use_tolbert_context"] is True
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 8.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_priority_expansion_source"] == "tooling_coding_family_scale+discovered"
    assert observe["observation_priority_benchmark_families"] == [
        "repository",
        "project",
        "workflow",
        "repo_chore",
        "integration",
        "repo_sandbox",
        "tooling",
    ]
    assert observe["observation_primary_task_limit"] == 50
    assert observe["generated_total"] == 50
    assert observe["generated_passed"] == 50


def test_run_supervised_cycle_retries_generated_success_without_tolbert_context_after_startup_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.4, 100.9, 101.5])

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 4,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=4,
                    passed=4,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 2, "project": 2},
                    passed_by_benchmark_family={"repository": 2, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:followup-tolbert] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 1,
                "error": "RuntimeError: TOLBERT service failed to become ready after 15.000 seconds.",
                "last_progress_line": "RuntimeError: TOLBERT service failed to become ready after 15.000 seconds.",
                "last_progress_phase": "",
                "last_progress_task_id": "",
                "last_progress_benchmark_family": "",
                "partial_summary": {"phase": "generated_success_schedule", "generated_tasks_scheduled": 4},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=4,
                generated_passed=4,
                generated_by_kind={"adjacent_success": 4},
                generated_passed_by_kind={"adjacent_success": 4},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:followup-tolbert] phase=generated_success total=4 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {"phase": "complete", "generated_tasks_scheduled": 4, "completed_generated_tasks": 4},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "4",
            "--max-observation-seconds",
            "60",
            "--max-current-task-decision-seconds",
            "12",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
        ],
    )

    module.main()

    assert len(observation_calls) == 3
    assert observation_calls[0]["use_tolbert_context"] is False
    assert observation_calls[1]["use_tolbert_context"] is True
    assert observation_calls[2]["use_tolbert_context"] is False
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["retried_without_tolbert_context"] is True
    assert "retrying curriculum generated_success without tolbert context after startup failure with fresh followup budget 40.0s" in followup["tolbert_retry_warning"]
    assert followup["merged_generated_metrics"] is True
    assert followup["generated_total"] == 4
    assert followup["generated_passed"] == 4
    assert observe["generated_total"] == 4
    assert observe["generated_passed"] == 4


def test_run_supervised_cycle_caps_transition_model_primary_tail_for_bounded_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": False,
            "include_failure_generated": False,
            "include_discovered_tasks": True,
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=4, passed=4, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:transition-tail-guard] task 4/4 nested_file_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "nested_file_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "complete",
                "completed_primary_tasks": 4,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "nested_file_task",
                "last_completed_benchmark_family": "bounded",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="transition_model",
                reason="transition gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="transition_model",
                variant_id="repeat_avoidance",
                description="penalize repeated stalled commands more aggressively",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "repeat_avoidance"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "transition_model_candidate.json"),
            "action": "generate_transition_model_update",
            "artifact_kind": "transition_model_policy_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            transition_model_proposals_path=tmp_path / "transition" / "transition_model_proposals.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "transition_model",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
        ],
    )

    module.main()

    assert len(observation_calls) == 1
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 4
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 12.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_requested_primary_task_limit"] == 5
    assert observe["observation_primary_task_limit"] == 4
    assert observe["observation_primary_task_limit_source"] == "transition_model_discovered_tail_guard"
    assert observe["passed"] == 4


def test_run_supervised_cycle_caps_long_horizon_project_primary_task_limit_for_ollama(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": False,
            "include_failure_generated": False,
            "prefer_low_cost_tasks": True,
            "priority_benchmark_families": ["project"],
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(kwargs)
        return {
            "metrics": EvalMetrics(total=2, passed=2, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "error": "",
            "mode": "in_process",
            "returncode": 0,
            "last_progress_line": "[eval:ollama-long-horizon] task 2/2 deployment_manifest_task family=project",
            "last_progress_phase": "",
            "last_progress_task_id": "deployment_manifest_task",
            "last_progress_benchmark_family": "project",
            "partial_summary": {
                "phase": "complete",
                "completed_primary_tasks": 2,
                "observed_benchmark_families": ["project"],
                "last_completed_task_id": "deployment_manifest_task",
                "last_completed_benchmark_family": "project",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote reusable long-horizon procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "extract_tools",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--provider",
            "ollama",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "80",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 1
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 2
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 12.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_requested_primary_task_limit"] == 5
    assert observe["observation_primary_task_limit"] == 2
    assert observe["observation_primary_task_limit_source"] == "tooling_long_horizon_project_ollama_guard"
    assert observe["passed"] == 2


def test_run_supervised_cycle_disables_tolbert_context_for_bounded_transition_model_run(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": False,
            "include_failure_generated": False,
            "include_discovered_tasks": True,
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
            }
        )
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=4, passed=4, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:transition-no-tolbert] task 4/4 nested_file_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "nested_file_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "complete",
                "completed_primary_tasks": 4,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "nested_file_task",
                "last_completed_benchmark_family": "bounded",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="transition_model",
                reason="transition gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="transition_model",
                variant_id="repeat_avoidance",
                description="penalize repeated stalled commands more aggressively",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "repeat_avoidance"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "transition_model_candidate.json"),
            "action": "generate_transition_model_update",
            "artifact_kind": "transition_model_policy_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            transition_model_proposals_path=tmp_path / "transition" / "transition_model_proposals.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "transition_model",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
        ],
    )

    module.main()

    assert len(observation_calls) == 1
    assert observation_calls[0]["use_tolbert_context"] is False
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 4
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_tolbert_context_disabled_reason"] == "transition_model_bounded_startup_guard"
    assert observe["observation_retried_without_tolbert_context"] is False
    assert observe["passed"] == 4


def test_run_supervised_cycle_disables_tolbert_context_for_bounded_tooling_coding_run(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 6,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": list(getattr(args, "priority_benchmark_family", [])),
        },
    )

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=4,
                    passed=4,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 2, "project": 2},
                    passed_by_benchmark_family={"repository": 2, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:tooling-no-tolbert] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {
                    "phase": "complete",
                    "completed_primary_tasks": 4,
                    "observed_benchmark_families": ["repository", "project"],
                    "last_completed_task_id": "deployment_manifest_task",
                    "last_completed_benchmark_family": "project",
                },
                "current_task_decision_budget_exceeded": False,
                "current_task_timeout_budget_seconds": 0.0,
                "current_task_timeout_budget_source": "none",
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=4,
                generated_passed=4,
                generated_by_kind={"adjacent_success": 4},
                generated_passed_by_kind={"adjacent_success": 4},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:tooling-no-tolbert] phase=generated_success total=4 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "deployment_manifest_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {"phase": "complete", "generated_tasks_scheduled": 4, "completed_generated_tasks": 4},
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "6",
            "--max-observation-seconds",
            "60",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "48",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["use_tolbert_context"] is False
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 4
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 12.0
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 4
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_primary_task_limit"] == 4
    assert observe["observation_tolbert_context_disabled_reason"] == "tooling_repository_project_bounded_startup_guard"
    assert observe["observation_retried_without_tolbert_context"] is False
    assert observe["generated_total"] == 4
    assert observe["generated_passed"] == 4


def test_run_supervised_cycle_smoke_profile_applies_one_task_budget_defaults(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=1, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:smoke-profile] task 1/1 archive_command_seed_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "archive_command_seed_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 1,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "archive_command_seed_task",
                "last_completed_benchmark_family": "bounded",
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--observation-profile",
            "smoke",
        ],
    )

    module.main()

    assert observation_calls[0]["max_observation_seconds"] == 24.0
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 8.0
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 1
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_profile"] == "smoke"
    assert observe["observation_profile_defaults_applied"] == {
        "max_observation_seconds": 24.0,
        "max_current_task_decision_seconds": 8.0,
        "task_limit": 1,
    }
    assert observe["observation_current_task_decision_budget_seconds"] == 8.0
    assert observe["observation_current_task_decision_budget_source"] == "smoke_default"


def test_run_supervised_cycle_applies_one_task_default_decision_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(float(kwargs["current_task_decision_budget_seconds"]))
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=1, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:decision-budget] task 1/1 math_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "math_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 1,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "math_task",
                "last_completed_benchmark_family": "bounded",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "1",
            "--max-observation-seconds",
            "12",
        ],
    )

    module.main()

    assert observation_calls == [12.0]
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_current_task_decision_budget_seconds"] == 12.0
    assert observe["observation_current_task_decision_budget_source"] == "one_task_default"


def test_run_supervised_cycle_applies_multi_task_default_decision_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(float(kwargs["current_task_decision_budget_seconds"]))
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=1, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:decision-budget] task 1/5 archive_command_seed_task family=bounded",
            "last_progress_phase": "",
            "last_progress_task_id": "archive_command_seed_task",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 0,
                "observed_benchmark_families": ["bounded"],
                "last_completed_task_id": "archive_command_seed_task",
                "last_completed_benchmark_family": "bounded",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "15",
        ],
    )

    module.main()

    assert observation_calls == [2.5]
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_current_task_decision_budget_seconds"] == 2.5
    assert observe["observation_current_task_decision_budget_source"] == "multi_task_default"


def test_run_supervised_cycle_retries_without_tolbert_context_after_startup_failure(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 124.0, 130.0])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 1,
                "error": "RuntimeError: TOLBERT service failed to become ready after 15.000 seconds.",
                "last_progress_line": "",
                "last_progress_phase": "",
                "last_progress_task_id": "",
                "last_progress_benchmark_family": "",
                "partial_summary": {},
                "current_task_decision_budget_exceeded": False,
                "current_task_timeout_budget_seconds": 0.0,
                "current_task_timeout_budget_source": "none",
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=5, passed=4, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:tolbert-retry] task 4/5 repo_sync_matrix_task family=repository",
            "last_progress_phase": "",
            "last_progress_task_id": "repo_sync_matrix_task",
            "last_progress_benchmark_family": "repository",
            "partial_summary": {
                "phase": "primary",
                "completed_primary_tasks": 4,
                "observed_benchmark_families": ["repository", "project"],
                "last_completed_task_id": "repo_sync_matrix_task",
                "last_completed_benchmark_family": "repository",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["use_tolbert_context"] is True
    assert observation_calls[1]["use_tolbert_context"] is False
    assert observation_calls[1]["max_observation_seconds"] == 60.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_tolbert_context"] is True
    assert "retrying observation without tolbert context after startup failure with fresh observation budget 60.0s" in observe["observation_tolbert_retry_warning"]
    assert observe["observation_timed_out"] is False
    assert observe["passed"] == 4


def test_run_supervised_cycle_retries_with_reduced_primary_task_limit_after_timeout(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 161.0, 170.0])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            seed_output_path = str(kwargs["eval_kwargs"].get("generated_success_seed_output_path", "")).strip()
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "repo_adjacent_seed_a"}, {"task_id": "repo_adjacent_seed_b"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": True,
                "timeout_reason": "observation child exceeded max runtime of 60.0 seconds",
                "returncode": -9,
                "error": "",
                "last_progress_line": "[eval:runtime-retry] task 2/5 schema_alignment_task family=repository",
                "last_progress_phase": "",
                "last_progress_task_id": "schema_alignment_task",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {
                    "phase": "primary",
                    "completed_primary_tasks": 2,
                    "observed_benchmark_families": ["repository", "project"],
                    "last_completed_task_id": "schema_alignment_task",
                    "last_completed_benchmark_family": "repository",
                    "scheduled_task_order": [
                        "service_release_task",
                        "schema_alignment_task",
                        "report_rollup_task",
                        "repo_sync_matrix_task",
                        "deployment_manifest_task",
                    ],
                },
                "current_task_decision_budget_exceeded": False,
                "current_task_timeout_budget_seconds": 0.0,
                "current_task_timeout_budget_source": "none",
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=2, passed=2, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:runtime-retry] task 2/2 schema_alignment_task family=repository",
            "last_progress_phase": "",
            "last_progress_task_id": "schema_alignment_task",
            "last_progress_benchmark_family": "repository",
            "partial_summary": {
                "phase": "complete",
                "completed_primary_tasks": 2,
                "observed_benchmark_families": ["repository"],
                "last_completed_task_id": "schema_alignment_task",
                "last_completed_benchmark_family": "repository",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "none",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="promote local shell procedures",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "procedure_promotion"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["task_limit"] == 5
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 2
    assert observation_calls[1]["max_observation_seconds"] == 60.0
    assert observation_calls[1]["current_task_decision_budget_seconds"] == 12.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_with_reduced_primary_task_limit"] is True
    assert observe["observation_primary_task_limit_retry_applied"] == 2
    assert "retrying observation with reduced primary task_limit 2/5" in observe[
        "observation_primary_task_limit_retry_warning"
    ]
    assert observe["observation_timed_out"] is False
    assert observe["passed"] == 2


def test_run_supervised_cycle_skips_generated_followups_without_separate_budgets(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(dict(kwargs["eval_kwargs"]))
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=4, passed=3, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "2",
            "--include-curriculum",
            "--include-failure-curriculum",
        ],
    )

    module.main()

    assert len(observation_calls) == 1
    assert observation_calls[0]["include_generated"] is False
    assert observation_calls[0]["include_failure_generated"] is False
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_staged_curriculum"] is True
    assert observe["observation_generated_curriculum_budget_seconds"] == 0.0
    assert observe["observation_failure_curriculum_budget_seconds"] == 0.0
    assert observe["observation_curriculum_followups"][0]["skipped_reason"] == "no separate curriculum budget configured"
    assert observe["observation_curriculum_followups"][1]["skipped_reason"] == "no separate curriculum budget configured"


def test_run_supervised_cycle_auto_allocates_staged_curriculum_budget_for_explicit_priority_families(
    tmp_path, monkeypatch
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 101.0, 101.3, 101.6])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
            }
        )
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(total=2, passed=1, average_steps=1.0),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "elapsed_seconds": 0.3,
            "last_progress_line": "",
            "last_progress_phase": "",
            "last_progress_task_id": "",
            "last_progress_benchmark_family": "",
            "partial_summary": {"observed_benchmark_families": ["repository"]},
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "",
            "error": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 2,
            "include_generated": True,
            "include_failure_generated": True,
            "priority_benchmark_families": ["repository", "project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "2",
            "--include-curriculum",
            "--include-failure-curriculum",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 3
    assert observation_calls[1]["eval_kwargs"]["include_generated"] is True
    assert observation_calls[2]["eval_kwargs"]["include_failure_generated"] is True
    assert observation_calls[0]["eval_kwargs"]["generated_success_seed_output_path"]
    assert observation_calls[1]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[2]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[1]["eval_kwargs"]["allow_generated_success_seed_fallback"] is True
    assert (
        observation_calls[1]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[0]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    assert observation_calls[1]["eval_kwargs"]["generated_success_seed_workspace_root"]
    assert observation_calls[1]["eval_kwargs"]["max_generated_success_schedule_tasks"] == 1
    assert observation_calls[1]["max_observation_seconds"] == 0.5
    assert observation_calls[2]["max_observation_seconds"] == 0.5
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_generated_curriculum_budget_seconds"] == 0.5
    assert observe["observation_failure_curriculum_budget_seconds"] == 0.5
    assert observe["observation_curriculum_followups"][0]["skipped_reason"] == ""
    assert observe["observation_curriculum_followups"][1]["skipped_reason"] == ""


def test_run_supervised_cycle_caps_generated_failure_followup_to_fit_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.6, 101.0])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=0, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-cap] task 5/5 fail_seed family=bounded",
                "last_progress_phase": "primary",
                "last_progress_task_id": "fail_seed",
                "last_progress_benchmark_family": "bounded",
                "partial_summary": {
                    "phase": "primary",
                    "completed_primary_tasks": 5,
                    "observed_benchmark_families": ["bounded"],
                    "last_completed_task_id": "fail_seed",
                    "last_completed_benchmark_family": "bounded",
                },
                "current_task_decision_budget_exceeded": False,
                "current_task_timeout_budget_seconds": 0.0,
                "current_task_timeout_budget_source": "",
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=0.0,
                generated_total=1,
                generated_passed=0,
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-cap] phase=generated_failure total=1 family=bounded",
            "last_progress_phase": "generated_failure",
            "last_progress_task_id": "recovery_seed",
            "last_progress_benchmark_family": "bounded",
            "partial_summary": {
                "phase": "generated_failure",
                "completed_primary_tasks": 0,
                "observed_benchmark_families": ["bounded"],
                "generated_tasks_scheduled": 1,
                "completed_generated_tasks": 1,
                "last_completed_task_id": "recovery_seed",
                "last_completed_benchmark_family": "bounded",
            },
            "current_task_decision_budget_exceeded": False,
            "current_task_timeout_budget_seconds": 0.0,
            "current_task_timeout_budget_source": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "append_cycle_record",
        lambda self, path, record: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text(
            json.dumps(record.__dict__) + "\n",
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": False,
            "include_failure_generated": True,
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-failure-curriculum",
            "--failure-curriculum-budget-seconds",
            "18.75",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["current_task_decision_budget_seconds"] == 10.0
    assert observation_calls[1]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 1
    assert observation_calls[1]["current_task_decision_budget_seconds"] == 9.375
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][1]
    assert followup["kind"] == "generated_failure"
    assert followup["applied_task_limit"] == 1
    assert followup["applied_current_task_decision_budget_seconds"] == 9.375
    assert followup["merged_generated_metrics"] is True


def test_run_supervised_cycle_allows_two_generated_success_followups_when_budget_supports_it(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.6, 101.0])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=2,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 3, "project": 2},
                    passed_by_benchmark_family={"repository": 1, "project": 1},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "",
                "last_progress_phase": "",
                "last_progress_task_id": "",
                "last_progress_benchmark_family": "",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=2,
                generated_passed=2,
                generated_by_kind={"adjacent_success": 2},
                generated_passed_by_kind={"adjacent_success": 2},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-budget] phase=generated_success total=2 family=repository",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "repo_sync_matrix_task_repository_adjacent",
            "last_progress_benchmark_family": "repository",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "append_cycle_record",
        lambda self, path, record: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_text(
            json.dumps(record.__dict__) + "\n",
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["repository", "project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "20",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[1]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 2
    assert observation_calls[1]["eval_kwargs"]["max_generated_success_schedule_tasks"] == 2
    assert observation_calls[1]["current_task_decision_budget_seconds"] == 10.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["applied_task_limit"] == 2
    assert followup["applied_current_task_decision_budget_seconds"] == 10.0
    assert followup["merged_generated_metrics"] is True


def test_run_supervised_cycle_runs_second_generated_success_wave_after_repository_bridge(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = itertools.count(start=100.0, step=0.6)

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(kwargs["eval_kwargs"].get("generated_success_seed_output_path", "")).strip()
        if seed_output_path:
            Path(seed_output_path).parent.mkdir(parents=True, exist_ok=True)
        if len(observation_calls) == 1:
            if seed_output_path:
                Path(seed_output_path).write_text(json.dumps({"episodes": []}), encoding="utf-8")
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=5, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave2] task 5/5 git_generated_conflict_resolution_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "git_generated_conflict_resolution_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "repository_seed_workflow_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=5,
                    generated_passed=5,
                    generated_by_kind={"adjacent_success": 5},
                    generated_passed_by_kind={"adjacent_success": 5},
                    generated_by_benchmark_family={"repository": 5},
                    generated_passed_by_benchmark_family={"repository": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave2] phase=generated_success total=5 family=repository",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {},
            }
        if len(observation_calls) == 3:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "workflow_seed_tooling_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=2,
                    generated_passed=2,
                    generated_by_kind={"adjacent_success": 2},
                    generated_passed_by_kind={"adjacent_success": 2},
                    generated_by_benchmark_family={"workflow": 2},
                    generated_passed_by_benchmark_family={"workflow": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave2] phase=generated_success total=2 family=workflow",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
                "last_progress_benchmark_family": "workflow",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
                generated_by_benchmark_family={"tooling": 1},
                generated_passed_by_benchmark_family={"tooling": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-wave3] phase=generated_success total=1 family=tooling",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
            "last_progress_benchmark_family": "tooling",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 4
    assert observation_calls[1]["eval_kwargs"]["generated_success_seed_output_path"]
    assert (
        observation_calls[2]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[1]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    assert observation_calls[2]["eval_kwargs"]["allow_generated_success_seed_fallback"] is True
    assert observation_calls[2]["eval_kwargs"]["generated_success_seed_output_path"]
    assert (
        observation_calls[3]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[2]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    assert observation_calls[3]["eval_kwargs"]["allow_generated_success_seed_fallback"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["generated_total"] == 8
    assert observe["generated_passed"] == 8
    assert observe["generated_by_benchmark_family"] == {"repository": 5, "workflow": 2, "tooling": 1}
    assert observe["generated_passed_by_benchmark_family"] == {"repository": 5, "workflow": 2, "tooling": 1}
    assert observe["observation_curriculum_followups"][1]["kind"] == "generated_success_wave2"
    assert observe["observation_curriculum_followups"][1]["merged_generated_metrics"] is True
    assert observe["observation_curriculum_followups"][1]["last_progress_benchmark_family"] == "workflow"
    assert observe["observation_curriculum_followups"][2]["kind"] == "generated_success_wave3"
    assert observe["observation_curriculum_followups"][2]["merged_generated_metrics"] is True
    assert observe["observation_curriculum_followups"][2]["last_progress_benchmark_family"] == "tooling"


def test_run_supervised_cycle_launches_wave3_from_timed_out_wave2_when_seed_bundle_exists(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = itertools.count(start=100.0, step=0.6)

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(kwargs["eval_kwargs"].get("generated_success_seed_output_path", "")).strip()
        if seed_output_path:
            Path(seed_output_path).parent.mkdir(parents=True, exist_ok=True)
        if len(observation_calls) == 1:
            if seed_output_path:
                Path(seed_output_path).write_text(json.dumps({"episodes": []}), encoding="utf-8")
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=5, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave3-timeout] task 5/5 git_generated_conflict_resolution_task family=project",
                "last_progress_phase": "primary",
                "last_progress_task_id": "git_generated_conflict_resolution_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "repository_seed_workflow_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=5,
                    generated_passed=5,
                    generated_by_kind={"adjacent_success": 5},
                    generated_passed_by_kind={"adjacent_success": 5},
                    generated_by_benchmark_family={"repository": 5},
                    generated_passed_by_benchmark_family={"repository": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave3-timeout] phase=generated_success total=5 family=repository",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {},
            }
        if len(observation_calls) == 3:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "workflow_seed_tooling_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=1,
                    generated_passed=1,
                    generated_by_kind={"adjacent_success": 1},
                    generated_passed_by_kind={"adjacent_success": 1},
                    generated_by_benchmark_family={"workflow": 1},
                    generated_passed_by_benchmark_family={"workflow": 1},
                ),
                "timed_out": True,
                "timeout_reason": "observation child exceeded max runtime of 20.0 seconds",
                "returncode": -9,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave3-timeout] phase=generated_success total=1 family=workflow",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
                "last_progress_benchmark_family": "workflow",
                "partial_summary": {
                    "phase": "generated_success",
                    "completed_generated_tasks": 1,
                    "generated_total": 1,
                    "generated_passed": 1,
                    "generated_by_kind": {"adjacent_success": 1},
                    "generated_passed_by_kind": {"adjacent_success": 1},
                    "generated_by_benchmark_family": {"workflow": 1},
                    "generated_passed_by_benchmark_family": {"workflow": 1},
                    "last_completed_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
                    "last_completed_benchmark_family": "workflow",
                    "observed_benchmark_families": ["workflow"],
                },
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
                generated_by_benchmark_family={"tooling": 1},
                generated_passed_by_benchmark_family={"tooling": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-wave3-timeout] phase=generated_success total=1 family=tooling",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
            "last_progress_benchmark_family": "tooling",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="tooling bridge",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.4,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="bridge workflow completions into tooling followups",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.3,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 4
    assert observation_calls[2]["max_observation_seconds"] == 80.0
    assert observation_calls[2]["use_tolbert_context"] is False
    assert observation_calls[2]["unattended_allow_git_commands"] is True
    assert observation_calls[2]["unattended_allow_generated_path_mutations"] is True
    assert (
        observation_calls[3]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[2]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["generated_total"] == 7
    assert observe["generated_passed"] == 7
    assert observe["generated_by_benchmark_family"] == {"repository": 5, "workflow": 1, "tooling": 1}
    assert observe["observation_curriculum_followups"][1]["kind"] == "generated_success_wave2"
    assert observe["observation_curriculum_followups"][1]["timed_out"] is True
    assert observe["observation_curriculum_followups"][1]["merged_generated_metrics"] is True
    assert observe["observation_curriculum_followups"][1]["applied_max_observation_seconds"] == 80.0
    assert (
        observe["observation_curriculum_followups"][1]["max_observation_seconds_source"]
        == "shared_repo_integrator_followup_budget_guard"
    )
    assert (
        observe["observation_curriculum_followups"][1]["tolbert_context_disabled_reason"]
        == "shared_repo_integrator_context_guard"
    )
    assert (
        observe["observation_curriculum_followups"][1]["operator_policy_override_reason"]
        == "shared_repo_long_horizon_project_guard"
    )
    assert observe["observation_curriculum_followups"][2]["kind"] == "generated_success_wave3"
    assert observe["observation_curriculum_followups"][2]["ran"] is True
    assert observe["observation_curriculum_followups"][2]["generated_total"] == 1


def test_run_supervised_cycle_runs_bounded_wave4_after_tooling_bridge(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = itertools.count(start=100.0, step=0.6)

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(kwargs["eval_kwargs"].get("generated_success_seed_output_path", "")).strip()
        if seed_output_path:
            Path(seed_output_path).parent.mkdir(parents=True, exist_ok=True)
        if len(observation_calls) == 1:
            if seed_output_path:
                Path(seed_output_path).write_text(json.dumps({"episodes": []}), encoding="utf-8")
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=5, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave4] task 5/5 git_generated_conflict_resolution_task family=project",
                "last_progress_phase": "primary",
                "last_progress_task_id": "git_generated_conflict_resolution_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "repository_seed_workflow_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=5,
                    generated_passed=5,
                    generated_by_kind={"adjacent_success": 5},
                    generated_passed_by_kind={"adjacent_success": 5},
                    generated_by_benchmark_family={"repository": 5},
                    generated_passed_by_benchmark_family={"repository": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave4] phase=generated_success total=5 family=repository",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {},
            }
        if len(observation_calls) == 3:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "workflow_seed_tooling_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=2,
                    generated_passed=2,
                    generated_by_kind={"adjacent_success": 2},
                    generated_passed_by_kind={"adjacent_success": 2},
                    generated_by_benchmark_family={"workflow": 2},
                    generated_passed_by_benchmark_family={"workflow": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave4] phase=generated_success total=2 family=workflow",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
                "last_progress_benchmark_family": "workflow",
                "partial_summary": {},
            }
        if len(observation_calls) == 4:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "tooling_seed_integration_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=1,
                    generated_passed=1,
                    generated_by_kind={"adjacent_success": 1},
                    generated_passed_by_kind={"adjacent_success": 1},
                    generated_by_benchmark_family={"tooling": 1},
                    generated_passed_by_benchmark_family={"tooling": 1},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave4] phase=generated_success total=1 family=tooling",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
                "last_progress_benchmark_family": "tooling",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
                generated_by_benchmark_family={"integration": 1},
                generated_passed_by_benchmark_family={"integration": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-wave4] phase=generated_success total=1 family=integration",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
            "last_progress_benchmark_family": "integration",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="integration bridge",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.4,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="bridge tooling completions into integration followups",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.3,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 5
    assert (
        observation_calls[4]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[3]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["generated_total"] == 9
    assert observe["generated_passed"] == 9
    assert observe["generated_by_benchmark_family"] == {
        "repository": 5,
        "workflow": 2,
        "tooling": 1,
        "integration": 1,
    }
    assert observe["observation_curriculum_followups"][3]["kind"] == "generated_success_wave4"
    assert observe["observation_curriculum_followups"][3]["ran"] is True
    assert observe["observation_curriculum_followups"][3]["last_progress_benchmark_family"] == "integration"


def test_run_supervised_cycle_runs_bounded_wave5_after_integration_bridge(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])

    def fake_run_observation_eval(**kwargs):
        eval_kwargs = dict(kwargs["eval_kwargs"])
        observation_calls.append(
            {
                "eval_kwargs": eval_kwargs,
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(eval_kwargs.get("generated_success_seed_output_path", "")).strip()
        if len(observation_calls) == 1:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "primary_seed_repository_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=5,
                    average_steps=1.0,
                    total_by_benchmark_family={"project": 5},
                    passed_by_benchmark_family={"project": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave5] task 5/5 git_generated_conflict_resolution_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "git_generated_conflict_resolution_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "repository_seed_workflow_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=5,
                    generated_passed=5,
                    generated_by_kind={"adjacent_success": 5},
                    generated_passed_by_kind={"adjacent_success": 5},
                    generated_by_benchmark_family={"repository": 5},
                    generated_passed_by_benchmark_family={"repository": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave5] phase=generated_success total=5 family=repository",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {},
            }
        if len(observation_calls) == 3:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "workflow_seed_tooling_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=2,
                    generated_passed=2,
                    generated_by_kind={"adjacent_success": 2},
                    generated_passed_by_kind={"adjacent_success": 2},
                    generated_by_benchmark_family={"workflow": 2},
                    generated_passed_by_benchmark_family={"workflow": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave5] phase=generated_success total=2 family=workflow",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent",
                "last_progress_benchmark_family": "workflow",
                "partial_summary": {},
            }
        if len(observation_calls) == 4:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "tooling_seed_integration_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=1,
                    generated_passed=1,
                    generated_by_kind={"adjacent_success": 1},
                    generated_passed_by_kind={"adjacent_success": 1},
                    generated_by_benchmark_family={"tooling": 1},
                    generated_passed_by_benchmark_family={"tooling": 1},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave5] phase=generated_success total=1 family=tooling",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent",
                "last_progress_benchmark_family": "tooling",
                "partial_summary": {},
            }
        if len(observation_calls) == 5:
            if seed_output_path:
                Path(seed_output_path).write_text(
                    json.dumps({"episodes": [{"task_id": "integration_seed_repo_chore_adjacent"}]}),
                    encoding="utf-8",
                )
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=0,
                    passed=0,
                    average_steps=1.0,
                    generated_total=1,
                    generated_passed=1,
                    generated_by_kind={"adjacent_success": 1},
                    generated_passed_by_kind={"adjacent_success": 1},
                    generated_by_benchmark_family={"integration": 1},
                    generated_passed_by_benchmark_family={"integration": 1},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-wave5] phase=generated_success total=1 family=integration",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent",
                "last_progress_benchmark_family": "integration",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
                generated_by_benchmark_family={"repo_chore": 1},
                generated_passed_by_benchmark_family={"repo_chore": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-wave5] phase=generated_success total=1 family=repo_chore",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "git_generated_conflict_resolution_task_repository_adjacent_workflow_adjacent_tooling_adjacent_integration_adjacent_repo_chore_adjacent",
            "last_progress_benchmark_family": "repo_chore",
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="repo chore bridge",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.4,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="bridge integration completions into repo chore followups",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.3,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 6
    assert (
        observation_calls[5]["eval_kwargs"]["generated_success_seed_documents_path"]
        == observation_calls[4]["eval_kwargs"]["generated_success_seed_output_path"]
    )
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["generated_total"] == 10
    assert observe["generated_passed"] == 10
    assert observe["generated_by_benchmark_family"] == {
        "repository": 5,
        "workflow": 2,
        "tooling": 1,
        "integration": 1,
        "repo_chore": 1,
    }
    assert observe["observation_curriculum_followups"][4]["kind"] == "generated_success_wave5"
    assert observe["observation_curriculum_followups"][4]["ran"] is True
    assert observe["observation_curriculum_followups"][4]["last_progress_benchmark_family"] == "repo_chore"


def test_late_wave_generated_success_controls_keep_two_task_fanout(tmp_path):
    module = _load_script("run_human_guided_improvement_cycle.py")
    seed_bundle_path = tmp_path / "generated_success_wave3_seeds.json"
    seed_bundle_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {"task_id": "a"},
                    {"task_id": "b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved_eval_kwargs, decision_budget_seconds = module._resolve_staged_followup_execution_controls(
        followup_kind="generated_success",
        eval_kwargs={
            "task_limit": 2,
            "generated_success_seed_documents_path": str(seed_bundle_path),
        },
        budget_seconds=14.0,
        current_task_decision_budget_seconds=12.0,
    )

    assert resolved_eval_kwargs["task_limit"] == 2
    assert resolved_eval_kwargs["max_generated_success_schedule_tasks"] == 2
    assert decision_budget_seconds == 7.0


def test_run_supervised_cycle_can_continue_past_wave5(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0 + i for i in range(12)])

    def fake_run_observation_eval(**kwargs):
        eval_kwargs = dict(kwargs["eval_kwargs"])
        observation_calls.append(
            {
                "eval_kwargs": eval_kwargs,
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(eval_kwargs.get("generated_success_seed_output_path", "")).strip()
        family_plan = [
            ("project", 5),
            ("repository", 2),
            ("workflow", 2),
            ("tooling", 1),
            ("integration", 1),
            ("repo_chore", 1),
            ("repo_chore", 1),
        ]
        family, total = family_plan[len(observation_calls) - 1]
        if seed_output_path and len(observation_calls) < 7:
            Path(seed_output_path).write_text(
                json.dumps({"episodes": [{"task_id": f"{family}_next_seed"}]}),
                encoding="utf-8",
            )
        if family == "project":
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=5,
                    average_steps=1.0,
                    total_by_benchmark_family={"project": 5},
                    passed_by_benchmark_family={"project": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:wave6] task 5/5 primary family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "primary_project_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=total,
                generated_passed=total,
                generated_by_kind={"adjacent_success": total},
                generated_passed_by_kind={"adjacent_success": total},
                generated_by_benchmark_family={family: total},
                generated_passed_by_benchmark_family={family: total},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": f"[eval:wave6] phase=generated_success total={total} family={family}",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": f"{family}_task",
            "last_progress_benchmark_family": family,
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="extend wave chain",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.4,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="extend generated_success chain",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.3,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_curriculum_followups"][5]["kind"] == "generated_success_wave6"
    assert observe["observation_curriculum_followups"][5]["ran"] is True
    assert observe["observation_curriculum_followups"][5]["last_progress_benchmark_family"] == "repo_chore"


def test_run_supervised_cycle_can_continue_past_wave8(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0 + (i * 0.5) for i in range(20)])

    def fake_run_observation_eval(**kwargs):
        eval_kwargs = dict(kwargs["eval_kwargs"])
        observation_calls.append(
            {
                "eval_kwargs": eval_kwargs,
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        seed_output_path = str(eval_kwargs.get("generated_success_seed_output_path", "")).strip()
        family_plan = [
            ("project", 5),
            ("repository", 2),
            ("workflow", 2),
            ("tooling", 1),
            ("integration", 1),
            ("repo_chore", 1),
            ("validation", 1),
            ("governance", 1),
            ("oversight", 1),
            ("assurance", 1),
            ("adjudication", 1),
        ]
        plan_index = len(observation_calls) - 1
        if plan_index >= len(family_plan):
            family, total = ("adjudication", 0)
        else:
            family, total = family_plan[plan_index]
        if seed_output_path and len(observation_calls) < len(family_plan):
            Path(seed_output_path).write_text(
                json.dumps({"episodes": [{"task_id": f"{family}_next_seed"}]}),
                encoding="utf-8",
            )
        if family == "project":
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=5,
                    average_steps=1.0,
                    total_by_benchmark_family={"project": 5},
                    passed_by_benchmark_family={"project": 5},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:wave9] task 5/5 primary family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "primary_project_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=total,
                generated_passed=total,
                generated_by_kind={"adjacent_success": total},
                generated_passed_by_kind={"adjacent_success": total},
                generated_by_benchmark_family={family: total},
                generated_passed_by_benchmark_family={family: total},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": f"[eval:wave9] phase=generated_success total={total} family={family}",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": f"{family}_task",
            "last_progress_benchmark_family": family,
            "partial_summary": {},
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="tooling",
                reason="extend wave chain",
                priority=5,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.4,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="tooling",
                variant_id="procedure_promotion",
                description="extend generated_success chain",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.3,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "tool_candidate.json"),
            "action": "generate_tool_update",
            "artifact_kind": "tool_candidate_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    kinds = [entry["kind"] for entry in observe["observation_curriculum_followups"]]
    assert "generated_success_wave9" in kinds
    wave9_summary = next(entry for entry in observe["observation_curriculum_followups"] if entry["kind"] == "generated_success_wave9")
    assert wave9_summary["ran"] is True
    assert wave9_summary["last_progress_benchmark_family"] == "assurance"


def test_run_supervised_cycle_auto_expands_generated_success_budget_from_primary_wins(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.6, 101.0, 101.4])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(
                    total=5,
                    passed=5,
                    average_steps=1.0,
                    total_by_benchmark_family={"repository": 3, "project": 2},
                    passed_by_benchmark_family={"repository": 3, "project": 2},
                ),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "",
                "last_progress_phase": "",
                "last_progress_task_id": "",
                "last_progress_benchmark_family": "",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=5,
                generated_passed=5,
                generated_by_kind={"adjacent_success": 5},
                generated_passed_by_kind={"adjacent_success": 5},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
                "last_progress_line": "[eval:curriculum-autobudget] phase=generated_success total=5 family=project",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "deployment_manifest_task_project_adjacent",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["repository", "project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[1]["eval_kwargs"]["include_primary_tasks"] is False
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 5
    assert observation_calls[1]["eval_kwargs"]["max_generated_success_schedule_tasks"] == 5
    assert observation_calls[1]["max_observation_seconds"] == 40.0
    assert observation_calls[1]["current_task_decision_budget_seconds"] == 8.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert observe["observation_generated_curriculum_budget_seconds"] == 40.0
    assert followup["kind"] == "generated_success"
    assert followup["applied_task_limit"] == 5
    assert followup["applied_current_task_decision_budget_seconds"] == 8.0
    assert followup["merged_generated_metrics"] is True


def test_run_supervised_cycle_preserves_partial_context_for_timed_out_curriculum_followup(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    observation_calls = []
    monotonic_values = iter([100.0, 100.5, 100.9])

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=4, passed=3, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-timeout] task 4/4 hello_task",
                "last_progress_phase": "",
                "last_progress_task_id": "hello_task",
                "partial_summary": {
                    "phase": "primary",
                    "completed_primary_tasks": 4,
                    "observed_benchmark_families": ["bounded", "workflow"],
                    "last_completed_task_id": "hello_task",
                    "last_completed_benchmark_family": "bounded",
                },
            }
        return {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": "observation child exceeded max runtime of 0.5 seconds",
            "returncode": -9,
            "error": "",
            "last_progress_line": "[eval:curriculum-timeout] phase=generated_success_schedule",
            "last_progress_phase": "generated_success_schedule",
            "last_progress_task_id": "",
            "partial_summary": {
                "phase": "generated_success_schedule",
                "completed_primary_tasks": 4,
                "observed_benchmark_families": ["repository"],
                "last_completed_task_id": "repo_seed",
                "last_completed_benchmark_family": "repository",
                "generated_tasks_scheduled": 3,
                "completed_generated_tasks": 0,
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--max-observation-seconds",
            "2",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "0.5",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[0]["eval_kwargs"]["include_generated"] is False
    assert observation_calls[1]["eval_kwargs"]["include_generated"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert "supplemental curriculum follow-up warning" in observe["observation_warning"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["timed_out"] is True
    assert followup["last_progress_phase"] == "generated_success_schedule"
    assert followup["partial_phase"] == "generated_success_schedule"
    assert followup["partial_generated_tasks_scheduled"] == 3
    assert followup["partial_generated_tasks_completed"] == 0
    assert followup["partial_last_completed_task_id"] == "repo_seed"
    assert followup["partial_last_completed_benchmark_family"] == "repository"
    assert followup["partial_observed_benchmark_families"] == ["repository"]
    assert followup["partial_summary"]["generated_tasks_scheduled"] == 3


def test_run_supervised_cycle_retries_generated_success_followup_with_reduced_task_limit(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 100.8, 101.6, 102.4, 103.2, 104.0])
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=4, passed=4, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-retry] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": True,
                "timeout_reason": "observation child exceeded max runtime of 20.0 seconds",
                "returncode": -9,
                "error": "",
                "last_progress_line": "[eval:curriculum-retry] phase=generated_success task 1/2 service_release_task_repository_adjacent family=repository",
                "last_progress_phase": "generated_success",
                "last_progress_task_id": "service_release_task_repository_adjacent",
                "last_progress_benchmark_family": "repository",
                "partial_summary": {
                    "phase": "generated_success",
                    "generated_tasks_scheduled": 2,
                    "completed_generated_tasks": 0,
                    "current_task_id": "service_release_task_repository_adjacent",
                    "current_task_benchmark_family": "repository",
                },
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:curriculum-retry] phase=generated_success total=1 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "report_rollup_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {
                "phase": "generated_success",
                "generated_tasks_scheduled": 1,
                "completed_generated_tasks": 1,
                "last_completed_task_id": "report_rollup_task_project_adjacent",
                "last_completed_benchmark_family": "project",
                "observed_benchmark_families": ["project"],
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "append_cycle_record",
        lambda self, path, record: path.parent.mkdir(parents=True, exist_ok=True)
        or path.open("a", encoding="utf-8").write(json.dumps(asdict(record)) + "\n"),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["repository", "project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "20",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 3
    assert observation_calls[1]["eval_kwargs"]["task_limit"] == 2
    assert observation_calls[2]["eval_kwargs"]["task_limit"] == 1
    assert observation_calls[2]["eval_kwargs"]["max_generated_success_schedule_tasks"] == 1
    assert observation_calls[2]["max_observation_seconds"] == 80.0
    assert observation_calls[2]["current_task_decision_budget_seconds"] == 12.0
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["retried_with_reduced_task_limit"] is True
    assert followup["reduced_task_limit_retry_applied"] == 1
    assert "retrying curriculum generated_success with reduced task_limit 1/2" in followup[
        "reduced_task_limit_retry_warning"
    ]
    assert followup["timed_out"] is False
    assert followup["generated_total"] == 1
    assert followup["generated_passed"] == 1
    assert followup["merged_generated_metrics"] is True


def test_run_supervised_cycle_applies_long_horizon_runtime_controls_to_generated_success_followup(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 100.8, 101.6, 102.4])
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "unattended_allow_git_commands": kwargs["config"].unattended_allow_git_commands,
                "unattended_allow_generated_path_mutations": kwargs["config"].unattended_allow_generated_path_mutations,
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=5, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:long-horizon-followup] task 5/5 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:long-horizon-followup] phase=generated_success total=1 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "report_rollup_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {
                "phase": "generated_success",
                "generated_tasks_scheduled": 1,
                "completed_generated_tasks": 1,
                "last_completed_task_id": "report_rollup_task_project_adjacent",
                "last_completed_benchmark_family": "project",
                "observed_benchmark_families": ["project"],
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "append_cycle_record",
        lambda self, path, record: path.parent.mkdir(parents=True, exist_ok=True)
        or path.open("a", encoding="utf-8").write(json.dumps(asdict(record)) + "\n"),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    assert observation_calls[1]["max_observation_seconds"] == 80.0
    assert observation_calls[1]["use_tolbert_context"] is False
    assert observation_calls[1]["unattended_allow_git_commands"] is True
    assert observation_calls[1]["unattended_allow_generated_path_mutations"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["applied_max_observation_seconds"] == 80.0
    assert followup["max_observation_seconds_source"] == "shared_repo_integrator_followup_budget_guard"
    assert followup["tolbert_context_disabled_reason"] == "shared_repo_integrator_context_guard"
    assert followup["operator_policy_override_reason"] == "shared_repo_long_horizon_project_guard"


def test_run_supervised_cycle_preserves_followup_runtime_controls_across_tolbert_retry(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 100.8, 101.6, 102.4, 103.2])
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
                "use_tolbert_context": kwargs["config"].use_tolbert_context,
                "unattended_allow_git_commands": kwargs["config"].unattended_allow_git_commands,
                "unattended_allow_generated_path_mutations": kwargs["config"].unattended_allow_generated_path_mutations,
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=5, passed=5, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:followup-retry-controls] task 5/5 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        if len(observation_calls) == 2:
            return {
                "mode": "child_process",
                "metrics": None,
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 1,
                "error": "tolbert service failed to become ready",
                "last_progress_line": "[eval:followup-retry-controls] phase=context_compile",
                "last_progress_phase": "context_compile",
                "last_progress_task_id": "report_rollup_task_project_adjacent",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=0,
                passed=0,
                average_steps=1.0,
                generated_total=1,
                generated_passed=1,
                generated_by_kind={"adjacent_success": 1},
                generated_passed_by_kind={"adjacent_success": 1},
                generated_by_benchmark_family={"project": 1},
                generated_passed_by_benchmark_family={"project": 1},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
            "last_progress_line": "[eval:followup-retry-controls] phase=generated_success total=1 family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "report_rollup_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {
                "phase": "generated_success",
                "generated_tasks_scheduled": 1,
                "completed_generated_tasks": 1,
                "last_completed_task_id": "report_rollup_task_project_adjacent",
                "last_completed_benchmark_family": "project",
                "observed_benchmark_families": ["project"],
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "append_cycle_record",
        lambda self, path, record: path.parent.mkdir(parents=True, exist_ok=True)
        or path.open("a", encoding="utf-8").write(json.dumps(asdict(record)) + "\n"),
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "policy_candidate.json"),
            "action": "generate_policy_update",
            "artifact_kind": "prompt_proposal_set",
        },
    )
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["project"],
            "prefer_low_cost_tasks": True,
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "40",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 3
    assert observation_calls[1]["max_observation_seconds"] == 80.0
    assert observation_calls[1]["use_tolbert_context"] is True
    assert observation_calls[1]["unattended_allow_git_commands"] is True
    assert observation_calls[1]["unattended_allow_generated_path_mutations"] is True
    assert observation_calls[2]["max_observation_seconds"] == 80.0
    assert observation_calls[2]["use_tolbert_context"] is False
    assert observation_calls[2]["unattended_allow_git_commands"] is True
    assert observation_calls[2]["unattended_allow_generated_path_mutations"] is True
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["retried_without_tolbert_context"] is True
    assert followup["applied_max_observation_seconds"] == 80.0
    assert followup["max_observation_seconds_source"] == "shared_repo_integrator_followup_budget_guard"
    assert followup["operator_policy_override_reason"] == "shared_repo_long_horizon_project_guard"
    assert followup["generated_total"] == 1
    assert followup["generated_passed"] == 1
    assert followup["merged_generated_metrics"] is True


def test_run_supervised_cycle_salvages_partial_generated_success_metrics_on_timeout(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 100.8, 101.6, 102.4, 103.2])
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(
            {
                "eval_kwargs": dict(kwargs["eval_kwargs"]),
                "max_observation_seconds": float(kwargs["max_observation_seconds"]),
                "current_task_decision_budget_seconds": float(kwargs["current_task_decision_budget_seconds"]),
            }
        )
        if len(observation_calls) == 1:
            return {
                "mode": "child_process",
                "metrics": EvalMetrics(total=4, passed=4, average_steps=1.0),
                "timed_out": False,
                "timeout_reason": "",
                "returncode": 0,
                "error": "",
                "last_progress_line": "[eval:curriculum-salvage] task 4/4 deployment_manifest_task family=project",
                "last_progress_phase": "",
                "last_progress_task_id": "deployment_manifest_task",
                "last_progress_benchmark_family": "project",
                "partial_summary": {},
            }
        return {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": "observation child exceeded max runtime of 20.0 seconds",
            "returncode": -9,
            "error": "",
            "last_progress_line": "[eval:curriculum-salvage] phase=generated_success task 2/2 report_rollup_task_project_adjacent family=project",
            "last_progress_phase": "generated_success",
            "last_progress_task_id": "report_rollup_task_project_adjacent",
            "last_progress_benchmark_family": "project",
            "partial_summary": {
                "phase": "generated_success",
                "generated_tasks_scheduled": 2,
                "completed_generated_tasks": 1,
                "generated_passed": 1,
                "generated_by_kind": {"adjacent_success": 1},
                "generated_passed_by_kind": {"adjacent_success": 1},
                "generated_by_benchmark_family": {"project": 1},
                "generated_passed_by_benchmark_family": {"project": 1},
                "last_completed_generated_task_id": "deployment_manifest_task_project_adjacent",
                "last_completed_generated_benchmark_family": "project",
                "current_task_id": "report_rollup_task_project_adjacent",
                "current_task_benchmark_family": "project",
            },
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_observation_eval_kwargs",
        lambda config, args: {
            "task_limit": 5,
            "include_generated": True,
            "include_failure_generated": False,
            "priority_benchmark_families": ["repository", "project"],
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--task-limit",
            "5",
            "--max-observation-seconds",
            "60",
            "--include-curriculum",
            "--generated-curriculum-budget-seconds",
            "20",
            "--priority-benchmark-family",
            "repository",
            "--priority-benchmark-family",
            "project",
        ],
    )

    module.main()

    assert len(observation_calls) == 2
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    followup = observe["observation_curriculum_followups"][0]
    assert followup["kind"] == "generated_success"
    assert followup["timed_out"] is True
    assert followup["salvaged_partial_generated_metrics"] is True
    assert followup["generated_total"] == 1
    assert followup["generated_passed"] == 1
    assert followup["merged_generated_metrics"] is True
    assert followup["partial_last_completed_generated_task_id"] == "deployment_manifest_task_project_adjacent"
    assert followup["partial_last_completed_generated_benchmark_family"] == "project"
    assert observe["generated_total"] == 1
    assert observe["generated_passed"] == 1
    assert observe["generated_by_kind"] == {"adjacent_success": 1}
    assert observe["generated_by_benchmark_family"] == {"project": 1}
    assert observe["generated_passed_by_benchmark_family"] == {"project": 1}

def test_run_supervised_cycle_does_not_retry_when_no_budget_remains(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    monotonic_values = iter([100.0, 102.1])
    observation_calls = []

    def fake_run_observation_eval(**kwargs):
        observation_calls.append(float(kwargs["max_observation_seconds"]))
        return {
            "mode": "child_process",
            "metrics": None,
            "timed_out": True,
            "timeout_reason": "observation child exceeded max runtime of 2.0 seconds",
            "returncode": -9,
            "error": "",
        }

    monkeypatch.setattr(module, "_run_observation_eval", fake_run_observation_eval)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--max-observation-seconds",
            "2",
            "--include-curriculum",
            "--include-failure-curriculum",
        ],
    )

    try:
        module.main()
        raise AssertionError("expected SystemExit when no observation budget remains")
    except SystemExit as exc:
        assert "observation child exceeded max runtime of 2.0 seconds" in str(exc)

    assert observation_calls == [2.0]
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["observation_retried_without_generated_curriculum"] is False


def test_observation_child_entry_writes_metrics_payload(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    payload_path = tmp_path / "payload.json"
    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "episodes",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
    )
    payload_path.write_text(
        json.dumps(
            {
                "config": module._kernel_config_snapshot(config),
                "eval_kwargs": {"include_discovered_tasks": True},
                "progress_label": "child-observe",
                "result_path": str(result_path),
                "progress_path": str(progress_path),
            }
        ),
        encoding="utf-8",
    )

    seen = {}

    def fake_run_eval(**kwargs):
        seen["progress_snapshot_path"] = kwargs.get("progress_snapshot_path")
        seen["progress_label"] = kwargs.get("progress_label")
        progress_path.write_text(
            json.dumps({"completed_primary_tasks": 1, "observed_benchmark_families": ["workflow"]}),
            encoding="utf-8",
        )
        return EvalMetrics(total=2, passed=1, average_steps=1.5, generated_total=1, generated_passed=1)

    monkeypatch.setattr(
        module,
        "run_eval",
        fake_run_eval,
    )

    module._observation_child_entry(payload_path)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["metrics"]["total"] == 2
    assert payload["metrics"]["passed"] == 1
    assert payload["metrics"]["generated_total"] == 1
    assert str(seen["progress_snapshot_path"]).endswith("progress.json")
    assert seen["progress_label"] == "child-observe"


def test_run_observation_eval_terminates_timed_out_child(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def wait(self, timeout=None):
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write("[eval:timeout-test] phase=generated_success_schedule\n")
                kwargs_capture["stderr"].write(
                    "[eval:timeout-test] task 1/3 api_contract_retrieval_task_benchmark_candidate\n"
                )
                kwargs_capture["stderr"].flush()
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="timeout-test",
        max_observation_seconds=1.5,
    )

    assert result["timed_out"] is True
    assert result["returncode"] == -9
    assert "1.5 seconds" in result["timeout_reason"]
    assert "process" in terminated
    assert result["last_progress_line"] == "[eval:timeout-test] task 1/3 api_contract_retrieval_task_benchmark_candidate"
    assert result["last_progress_phase"] == ""
    assert result["last_progress_task_id"] == "api_contract_retrieval_task_benchmark_candidate"


def test_run_observation_eval_returns_child_error_payload_when_exit_nonzero_and_no_metrics(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    progress_payload = json.dumps(
        {
            "current_task_id": "math_task",
            "current_task_step_stage": "context_compile",
            "current_task_elapsed_seconds": 0.25,
        },
        sort_keys=True,
    )

    class FakeProcess:
        def __init__(self):
            self.returncode = 2

        def wait(self, timeout=None):
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write("[eval:error-test] task 1/1 math_task family=bounded\n")
                kwargs_capture["stderr"].flush()
            return self.returncode

    kwargs_capture = {}

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "result.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "child failed",
                    }
                ),
                encoding="utf-8",
            )
            (self.path / "payload.json").write_text("{}", encoding="utf-8")
            (self.path / "progress.json").write_text(progress_payload, encoding="utf-8")
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        assert args[0][0].endswith("python")
        assert str(args[0][1]).endswith("run_human_guided_improvement_cycle.py")
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="error-test",
        max_observation_seconds=5.0,
    )

    assert result["timed_out"] is False
    assert result["returncode"] == 2
    assert result["error"] == "child failed"
    assert result["metrics"] is None
    assert result["partial_summary"]["current_task_id"] == "math_task"
    assert result["current_task_timeout_budget_seconds"] == 0.0
    assert result["current_task_timeout_budget_source"] == "none"
    assert result["last_progress_task_id"] == "math_task"


def test_run_observation_eval_terminates_child_when_decision_budget_exceeded(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def wait(self, timeout=None):
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write("[eval:decision-test] task 1/1 math_task family=bounded\n")
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "decision_pending",
                        "current_task_elapsed_seconds": 2.2,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=5.0,
        current_task_decision_budget_seconds=2.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 2.0
    assert "current task stage decision_pending exceeded max runtime of 2.0 seconds" in result["timeout_reason"]
    assert result["partial_summary"]["current_task_id"] == "math_task"
    assert "process" in terminated


def test_run_observation_eval_terminates_child_when_unclassified_prestep_budget_exceeded(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def wait(self, timeout=None):
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write("[eval:decision-test] task 1/1 math_task family=bounded\n")
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "",
                        "current_task_step_index": 1,
                        "current_task_completed_steps": 0,
                        "current_task_step_action": "",
                        "current_task_elapsed_seconds": 2.2,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=5.0,
        current_task_decision_budget_seconds=2.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 2.0
    assert "current task stage prestep_unclassified exceeded max runtime of 2.0 seconds" in result["timeout_reason"]
    assert result["partial_summary"]["current_task_id"] == "math_task"
    assert "process" in terminated


def test_run_observation_eval_terminates_child_when_context_compile_subphase_budget_exceeded(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.4
            else:
                step_elapsed_seconds = 3.6
            task_elapsed_seconds = 0.4 if self.calls == 1 else 3.6
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "tolbert_query",
                        "current_task_elapsed_seconds": task_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 6.0
    assert "context_compile subphase tolbert_query exceeded max runtime of 3.0 seconds" in result["timeout_reason"]
    assert result["current_task_timeout_budget_seconds"] == 3.0
    assert result["current_task_timeout_budget_source"] == "prestep_subphase:tolbert_query"
    assert result["partial_summary"]["current_task_id"] == "math_task"
    assert "process" in terminated


def test_run_observation_eval_prefers_progress_step_budget_over_static_subphase_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 5.4
            task_elapsed_seconds = step_elapsed_seconds
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "tolbert_query",
                        "current_task_elapsed_seconds": task_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 4.8,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert "context_compile subphase tolbert_query exceeded max runtime of 4.8 seconds" in result["timeout_reason"]


def test_run_observation_eval_uses_guidance_build_context_subphase_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 1.2
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "guidance_build",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 6.0
    assert "context_compile subphase guidance_build exceeded max runtime of 1.0 seconds" in result["timeout_reason"]


def test_run_observation_eval_uses_tool_query_context_subphase_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 2.0
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "tool_query",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 6.0
    assert "context_compile subphase tool_query exceeded max runtime of 1.5 seconds" in result["timeout_reason"]


def test_resolve_context_compile_subphase_budget_uses_static_budget_without_decision_budget():
    module = _load_script("run_human_guided_improvement_cycle.py")

    budget = module._resolve_current_task_prestep_subphase_budget_seconds(
        partial_summary={
            "current_task_step_stage": "context_compile",
            "current_task_step_subphase": "tool_query",
            "current_task_step_budget_seconds": 0.0,
        },
        decision_budget_seconds=0.0,
    )

    assert budget == 1.5


def test_resolve_unknown_context_compile_subphase_budget_falls_back_to_unknown_default():
    module = _load_script("run_human_guided_improvement_cycle.py")

    budget = module._resolve_current_task_prestep_subphase_budget_seconds(
        partial_summary={
            "current_task_step_stage": "context_compile",
            "current_task_step_subphase": "new_vendor_hook",
            "current_task_step_budget_seconds": 0.0,
        },
        decision_budget_seconds=0.0,
    )

    assert budget == 1.0


def test_run_observation_eval_uses_unknown_context_compile_subphase_default_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 1.4
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:unknown-subphase] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "context_compile",
                        "current_task_step_subphase": "new_vendor_hook",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="unknown-subphase",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=0.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 0.0
    assert "context_compile subphase new_vendor_hook exceeded max runtime of 1.0 seconds" in result["timeout_reason"]
    assert result["current_task_timeout_budget_seconds"] == 1.0
    assert result["current_task_timeout_budget_source"] == "prestep_subphase:new_vendor_hook"
    assert "process" in terminated


def test_run_observation_eval_uses_max_observation_budget_for_context_compile_when_no_decision_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = 2

        def wait(self, timeout=None):
            return self.returncode

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "result.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "child failed",
                    },
                ),
                encoding="utf-8",
            )
            (self.path / "payload.json").write_text("{}", encoding="utf-8")
            (self.path / "progress.json").write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "decision_pending",
                        "current_task_step_budget_seconds": 0.0,
                    },
                ),
                encoding="utf-8",
            )
            (self.path / "stdout.log").write_text("", encoding="utf-8")
            (self.path / "stderr.log").write_text(
                "[eval:context-budget-test] task 1/1 math_task family=bounded\n",
                encoding="utf-8",
            )
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        cmd = list(args[0])
        payload_path = Path(cmd[cmd.index("--_observation-child-payload") + 1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        captured["tolbert_context_compile_budget_seconds"] = payload["config"]["tolbert_context_compile_budget_seconds"]
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="context-budget-test",
        max_observation_seconds=4.0,
        current_task_decision_budget_seconds=0.0,
    )

    assert result["returncode"] == 2
    assert captured["tolbert_context_compile_budget_seconds"] == 4.0


def test_resolve_current_task_prestep_budget_stage_prefers_context_provider_named_stage():
    module = _load_script("run_human_guided_improvement_cycle.py")

    stage = module._current_task_prestep_budget_stage(
        {
            "current_task_id": "math_task",
            "current_task_step_stage": "skill_query",
            "current_task_completed_steps": 0,
            "current_task_step_index": 1,
            "current_task_step_action": "",
        }
    )

    assert stage == "skill_query"


def test_resolve_current_task_prestep_budget_stage_falls_back_to_budgeted_subphase_when_stage_empty():
    module = _load_script("run_human_guided_improvement_cycle.py")

    stage = module._current_task_prestep_budget_stage(
        {
            "current_task_id": "math_task",
            "current_task_step_stage": "",
            "current_task_step_subphase": "tool_query",
            "current_task_completed_steps": 0,
            "current_task_step_index": 1,
            "current_task_step_action": "",
        }
    )

    assert stage == "tool_query"


def test_resolve_current_task_prestep_budget_stage_ignores_stale_context_subphase_after_stage_advance():
    module = _load_script("run_human_guided_improvement_cycle.py")

    stage = module._current_task_prestep_budget_stage(
        {
            "current_task_id": "math_task",
            "current_task_step_stage": "llm_request",
            "current_task_step_subphase": "complete",
            "current_task_completed_steps": 0,
            "current_task_step_index": 1,
            "current_task_step_action": "",
        }
    )

    assert stage == ""


def test_resolve_context_compile_subphase_budget_ignores_stale_subphase_after_stage_advance():
    module = _load_script("run_human_guided_improvement_cycle.py")

    budget = module._resolve_current_task_prestep_subphase_budget_seconds(
        partial_summary={
            "current_task_step_stage": "llm_request",
            "current_task_step_subphase": "complete",
            "current_task_step_budget_seconds": 0.0,
        },
        decision_budget_seconds=6.0,
    )

    assert budget == 0.0


def test_run_observation_eval_uses_skill_query_stage_budget_even_without_subphase(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 2.0
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "skill_query",
                        "current_task_step_subphase": "",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 6.0
    assert "current task stage skill_query exceeded max runtime of 1.5 seconds" in result["timeout_reason"]
    assert "process" in terminated


def test_run_observation_eval_prefers_stage_named_subphase_budget(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    terminated = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                step_elapsed_seconds = 0.2
            else:
                step_elapsed_seconds = 2.0
            if kwargs_capture.get("stderr") is not None:
                kwargs_capture["stderr"].write(
                    "[eval:decision-test] task 1/1 math_task family=bounded\n"
                )
                kwargs_capture["stderr"].flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "current_task_id": "math_task",
                        "current_task_step_stage": "tool_query",
                        "current_task_step_subphase": "",
                        "current_task_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_elapsed_seconds": step_elapsed_seconds,
                        "current_task_step_budget_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

    kwargs_capture = {}
    progress_path = tmp_path / "progress.json"

    class FakeTemporaryDirectory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(*args, **kwargs):
        kwargs_capture.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: terminated.setdefault("process", process))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix="": FakeTemporaryDirectory(tmp_path))

    result = module._run_observation_eval(
        config=config,
        eval_kwargs={},
        progress_label="decision-test",
        max_observation_seconds=10.0,
        current_task_decision_budget_seconds=6.0,
    )

    assert result["timed_out"] is True
    assert result["current_task_decision_budget_exceeded"] is True
    assert result["current_task_decision_budget_seconds"] == 6.0
    assert "current task stage tool_query exceeded max runtime of 1.5 seconds" in result["timeout_reason"]


def test_run_supervised_cycle_rejects_scoped_run_without_generate_only(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
            improvement_reports_dir=tmp_path / "improvement" / "reports",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--scope-id",
            "runner-a",
        ],
    )

    try:
        module.main()
        raise AssertionError("expected SystemExit for scoped non-generate-only run")
    except SystemExit as exc:
        assert "--scope-id currently requires --generate-only" in str(exc)


def test_run_supervised_cycle_uses_scoped_config_for_parallel_generate_only_runs(tmp_path, monkeypatch):
    module = _load_script("run_human_guided_improvement_cycle.py")
    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
        tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
        curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
        operator_classes_path=tmp_path / "operators" / "operator_classes.json",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
    )
    captured = {}

    monkeypatch.setattr(module, "KernelConfig", lambda: base_config)

    def fake_scoped_improvement_cycle_config(config, scope, **overrides):
        assert scope == "runner_a"
        scoped = replace(
            config,
            workspace_root=tmp_path / "workspace" / scope,
            trajectories_root=tmp_path / "episodes",
            prompt_proposals_path=tmp_path / "prompts" / f"prompt_proposals_{scope}.json",
            improvement_cycles_path=tmp_path / "improvement" / f"cycles_{scope}.jsonl",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates" / scope,
            improvement_reports_dir=tmp_path / "improvement" / "reports" / scope,
            run_reports_dir=tmp_path / "reports" / scope,
        )
        for key, value in overrides.items():
            setattr(scoped, key, value)
        return scoped

    monkeypatch.setattr(module, "scoped_improvement_cycle_config", fake_scoped_improvement_cycle_config)

    def fake_run_eval(**kwargs):
        captured["workspace_root"] = str(kwargs["config"].workspace_root)
        captured["trajectories_root"] = str(kwargs["config"].trajectories_root)
        return EvalMetrics(total=3, passed=2, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy gap",
                priority=3,
                expected_gain=0.01,
                estimated_cost=2,
                score=0.1,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="policy",
                variant_id="verifier_alignment",
                description="bias decisions toward verifier-compatible artifact checks",
                expected_gain=0.012,
                estimated_cost=2,
                score=0.01,
                controls={"focus": "verifier_alignment"},
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "policy",
            "--generate-only",
            "--scope-id",
            "runner a",
        ],
    )

    module.main()

    assert captured["workspace_root"].endswith("workspace/runner_a")
    assert captured["trajectories_root"].endswith("episodes")
    records = [
        json.loads(line)
        for line in (tmp_path / "improvement" / "cycles_runner_a.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["metrics_summary"]["scoped_run"] is True
    assert records[0]["metrics_summary"]["scope_id"] == "runner_a"


def test_run_supervised_cycle_scoped_generate_only_falls_back_when_requested_subsystem_is_unavailable(
    tmp_path, monkeypatch
):
    module = _load_script("run_human_guided_improvement_cycle.py")
    cycles_path = tmp_path / "improvement" / "cycles_runner_a.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"

    monkeypatch.setattr(
        module,
        "_run_observation_eval",
        lambda **kwargs: {
            "mode": "child_process",
            "metrics": EvalMetrics(
                total=5,
                passed=5,
                average_steps=1.0,
                total_by_benchmark_family={"project": 5},
                passed_by_benchmark_family={"project": 5},
            ),
            "timed_out": False,
            "timeout_reason": "",
            "returncode": 0,
            "error": "",
        },
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_experiments",
        lambda self, metrics: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval gap",
                priority=4,
                expected_gain=0.02,
                estimated_cost=2,
                score=0.2,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: [
            ImprovementVariant(
                subsystem="retrieval",
                variant_id="confidence_gating",
                description="retrieval variant",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.1,
                controls={},
            )
        ],
    )
    monkeypatch.setattr(module, "finalize_cycle", lambda **kwargs: ("generate_only", "preview"))
    monkeypatch.setattr(
        module.autonomous_cycle,
        "_generate_candidate_artifact",
        lambda **kwargs: {
            "artifact": str(tmp_path / "candidates" / "retrieval_candidate.json"),
            "action": "generate_retrieval_update",
            "artifact_kind": "retrieval_policy_set",
        },
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
            improvement_reports_dir=reports_dir,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_human_guided_improvement_cycle.py",
            "--subsystem",
            "tooling",
            "--generate-only",
            "--scope-id",
            "runner_a",
        ],
    )

    module.main()

    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observe = records[0]["metrics_summary"]
    assert observe["requested_subsystem"] == "tooling"
    assert observe["requested_subsystem_available"] is False
    assert observe["requested_subsystem_fallback_allowed"] is True
    assert observe["selected_experiment_subsystem"] == "retrieval"
    assert "falling back to ranked subsystem retrieval" in observe["requested_subsystem_fallback_warning"]


def test_scoped_improvement_cycle_config_scopes_tolbert_runtime_surfaces(tmp_path):
    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "learning_artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "trust" / "ledger.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff_gate_report.json",
    )
    base_config.ensure_directories()
    base_config.tolbert_model_artifact_path.write_text(
        json.dumps({"artifact_kind": "tolbert_model_bundle", "runtime_policy": {"primary_benchmark_families": ["workflow"]}}),
        encoding="utf-8",
    )
    (base_config.tolbert_supervised_datasets_dir / "train.jsonl").write_text('{"task_id":"base"}\n', encoding="utf-8")
    base_config.tolbert_liftoff_report_path.write_text(
        json.dumps({"report_kind": "tolbert_liftoff_loop_report", "status": "completed"}),
        encoding="utf-8",
    )

    scoped = scoped_improvement_cycle_config(base_config, "tolbert_preview")

    assert scoped.tolbert_model_artifact_path != base_config.tolbert_model_artifact_path
    assert scoped.tolbert_supervised_datasets_dir == base_config.tolbert_supervised_datasets_dir
    assert scoped.tolbert_liftoff_report_path != base_config.tolbert_liftoff_report_path
    assert "tolbert_preview" in str(scoped.tolbert_model_artifact_path)
    assert "tolbert_preview" in str(scoped.tolbert_liftoff_report_path)
    assert json.loads(scoped.tolbert_model_artifact_path.read_text(encoding="utf-8"))["artifact_kind"] == "tolbert_model_bundle"
    assert (scoped.tolbert_supervised_datasets_dir / "train.jsonl").read_text(encoding="utf-8") == '{"task_id":"base"}\n'
    assert json.loads(scoped.tolbert_liftoff_report_path.read_text(encoding="utf-8"))["report_kind"] == "tolbert_liftoff_loop_report"


def test_scoped_improvement_cycle_config_can_skip_tolbert_seed_copy(tmp_path):
    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "learning_artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "trust" / "ledger.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff_gate_report.json",
    )
    base_config.ensure_directories()
    base_config.tolbert_model_artifact_path.write_text(
        json.dumps({"artifact_kind": "tolbert_model_bundle"}),
        encoding="utf-8",
    )
    (base_config.tolbert_supervised_datasets_dir / "train.jsonl").write_text('{"task_id":"base"}\n', encoding="utf-8")

    scoped = scoped_improvement_cycle_config(base_config, "tolbert_unseeded", seed_from_base=False)

    assert scoped.tolbert_model_artifact_path.parent.exists()
    assert scoped.tolbert_supervised_datasets_dir.exists()
    assert scoped.tolbert_liftoff_report_path.parent.exists()
    assert not scoped.tolbert_model_artifact_path.exists()
    assert (scoped.tolbert_supervised_datasets_dir / "train.jsonl").exists()


def test_scoped_improvement_cycle_config_does_not_seed_base_cycles_by_default(tmp_path):
    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "learning_artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "trust" / "ledger.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff_gate_report.json",
    )
    base_config.ensure_directories()
    base_config.improvement_cycles_path.write_text(
        json.dumps({"cycle_id": "cycle:test:base", "metrics_summary": {"scope_id": ""}}) + "\n",
        encoding="utf-8",
    )
    base_config.unattended_trust_ledger_path.write_text(json.dumps({"status": "trusted"}), encoding="utf-8")

    scoped = scoped_improvement_cycle_config(base_config, "runner_a")

    assert scoped.improvement_cycles_path != base_config.improvement_cycles_path
    assert not scoped.improvement_cycles_path.exists()
    assert json.loads(scoped.unattended_trust_ledger_path.read_text(encoding="utf-8"))["status"] == "trusted"


def test_run_parallel_supervised_cycles_writes_batch_report(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff.json",
    )

    seen_scope_ids: list[str] = []

    class _FakePipe:
        def __init__(self, lines):
            self._lines = [line if line.endswith("\n") else f"{line}\n" for line in lines]

        def __iter__(self):
            return iter(self._lines)

        def close(self):
            return None

    class _FakePopen:
        def __init__(self, cmd, *, cwd, env, stdout, stderr, text, bufsize):
            del cwd, env, stdout, stderr, text, bufsize
            scope_id = cmd[cmd.index("--scope-id") + 1]
            progress_label = cmd[cmd.index("--progress-label") + 1]
            seen_scope_ids.append(scope_id)
            scoped = module.scoped_improvement_cycle_config(config, scope_id)
            candidate_path = scoped.candidate_artifacts_root / "policy" / f"{scope_id}.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps({"artifact_kind": "prompt_proposal_set"}), encoding="utf-8")
            records = [
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="policy",
                    state="observe",
                    action="run_eval",
                    artifact_path=str(candidate_path),
                    artifact_kind="prompt_proposal_set",
                    reason="observe",
                    metrics_summary={
                        "total": 3,
                        "passed": 2,
                        "pass_rate": 2 / 3,
                        "generated_total": 1,
                        "generated_passed": 1,
                        "generated_pass_rate": 1.0,
                        "observation_returncode": 0,
                        "observation_elapsed_seconds": 3.0,
                        "observation_timed_out": False,
                        "observation_warning": "",
                        "progress_label": progress_label,
                    },
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="policy",
                    state="select",
                    action="select_experiment",
                    artifact_path=str(candidate_path),
                    artifact_kind="prompt_proposal_set",
                    reason="select",
                    metrics_summary={"selected_variant_id": "verifier_alignment"},
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="policy",
                    state="generate",
                    action="generate_candidate",
                    artifact_path=str(candidate_path),
                    artifact_kind="prompt_proposal_set",
                    reason="generate",
                    metrics_summary={},
                    candidate_artifact_path=str(candidate_path),
                ),
            ]
            scoped.improvement_cycles_path.parent.mkdir(parents=True, exist_ok=True)
            scoped.improvement_cycles_path.write_text(
                "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
                encoding="utf-8",
            )
            self.stdout = _FakePipe([f"[supervised:{progress_label}] phase=observe start", "generated candidate"])
            self.stderr = _FakePipe([f"[eval:{progress_label}] task 1/1 hello_task"])
            self._returncode = 0

        def wait(self):
            return self._returncode

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_supervised_cycles.py",
            "--workers",
            "2",
            "--task-limit",
            "3",
            "--no-auto-diversify-subsystems",
            "--scope-prefix",
            "batch_scope",
            "--notes",
            "batch probe",
        ],
    )
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "parallel_supervised_preview_report"
    assert payload["worker_count"] == 2
    assert payload["summary"]["completed_runs"] == 2
    assert payload["summary"]["generated_runs"] == 2
    assert payload["summary"]["generated_success_runs"] == 2
    assert payload["summary"]["generated_failure_runs"] == 0
    assert payload["summary"]["failed_runs"] == 0
    assert payload["summary"]["healthy_runs"] == 2
    assert payload["summary"]["warning_runs"] == 0
    assert payload["summary"]["budget_exceeded_runs"] == 0
    assert sorted(seen_scope_ids) == ["batch_scope_1", "batch_scope_2"]
    assert payload["runs"][0]["generated_candidate"] is True
    assert payload["runs"][0]["status"] == "healthy"
    assert payload["runs"][0]["run_succeeded"] is True
    assert payload["runs"][0]["healthy_run"] is True
    assert payload["runs"][0]["primary_passed"] == 2
    assert payload["runs"][0]["primary_total"] == 3
    assert payload["runs"][0]["generated_success_passed"] == 1
    assert payload["runs"][0]["generated_success_total"] == 1
    assert payload["runs"][0]["generated_failure_passed"] == 0
    assert payload["runs"][0]["selected_variant_id"] == "verifier_alignment"
    assert payload["runs"][0]["command"][-2:] == ["--notes", "batch probe [batch_scope_1]"]
    assert payload["runs"][1]["scope_id"] == "batch_scope_2"
    assert payload["summary"]["requested_subsystems"] == []
    history_path = reports_dir / "parallel_supervised_preview_history.jsonl"
    history_records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert history_records[-1]["report_kind"] == "parallel_supervised_preview_history"
    assert history_records[-1]["summary"]["generated_runs"] == 2
    assert "[parallel:batch_scope_1]" in err_stream.getvalue()
    assert "[parallel:batch_scope_2]" in err_stream.getvalue()


def test_run_parallel_supervised_cycles_supports_per_worker_subsystems_and_timeout_summary(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff.json",
    )

    seen_subsystems: list[str] = []

    class _FakePipe:
        def __init__(self, lines):
            self._lines = [line if line.endswith("\n") else f"{line}\n" for line in lines]

        def __iter__(self):
            return iter(self._lines)

        def close(self):
            return None

    class _FakePopen:
        def __init__(self, cmd, *, cwd, env, stdout, stderr, text, bufsize):
            del cwd, env, stdout, stderr, text, bufsize
            scope_id = cmd[cmd.index("--scope-id") + 1]
            subsystem = cmd[cmd.index("--subsystem") + 1]
            seen_subsystems.append(subsystem)
            scoped = module.scoped_improvement_cycle_config(config, scope_id)
            candidate_path = scoped.candidate_artifacts_root / subsystem / f"{scope_id}.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps({"artifact_kind": "candidate"}), encoding="utf-8")
            if subsystem == "policy":
                records = [
                    ImprovementCycleRecord(
                        cycle_id=f"cycle:{scope_id}",
                        subsystem=subsystem,
                        state="observe",
                        action="run_eval",
                        artifact_path=str(candidate_path),
                        artifact_kind="prompt_proposal_set",
                        reason="observe",
                        metrics_summary={
                            "total": 5,
                            "passed": 0,
                            "pass_rate": 0.0,
                            "generated_total": 0,
                            "generated_passed": 0,
                            "generated_pass_rate": 0.0,
                            "observation_returncode": 1,
                            "observation_elapsed_seconds": 5.0,
                            "observation_timed_out": True,
                            "observation_warning": "observation child exceeded max runtime of 5.0 seconds",
                        },
                    )
                ]
                self._returncode = 1
            else:
                records = [
                    ImprovementCycleRecord(
                        cycle_id=f"cycle:{scope_id}",
                        subsystem=subsystem,
                        state="observe",
                        action="run_eval",
                        artifact_path=str(candidate_path),
                        artifact_kind="benchmark_candidate_set",
                        reason="observe",
                        metrics_summary={
                            "total": 4,
                            "passed": 2,
                            "pass_rate": 0.5,
                            "generated_total": 1,
                            "generated_passed": 1,
                            "generated_pass_rate": 1.0,
                            "observation_returncode": 0,
                            "observation_elapsed_seconds": 2.0,
                            "observation_timed_out": False,
                        },
                    ),
                    ImprovementCycleRecord(
                        cycle_id=f"cycle:{scope_id}",
                        subsystem=subsystem,
                        state="select",
                        action="select_experiment",
                        artifact_path=str(candidate_path),
                        artifact_kind="benchmark_candidate_set",
                        reason="select",
                        metrics_summary={"selected_variant": {"variant_id": "failure_cluster_growth"}},
                    ),
                    ImprovementCycleRecord(
                        cycle_id=f"cycle:{scope_id}",
                        subsystem=subsystem,
                        state="generate",
                        action="generate_candidate",
                        artifact_path=str(candidate_path),
                        artifact_kind="benchmark_candidate_set",
                        reason="generate",
                        metrics_summary={},
                        candidate_artifact_path=str(candidate_path),
                    ),
                ]
                self._returncode = 0
            scoped.improvement_cycles_path.parent.mkdir(parents=True, exist_ok=True)
            scoped.improvement_cycles_path.write_text(
                "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
                encoding="utf-8",
            )
            self.stdout = _FakePipe([f"worker={scope_id}"])
            self.stderr = _FakePipe([f"subsystem={subsystem}"])

        def wait(self):
            return self._returncode

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_supervised_cycles.py",
            "--workers",
            "2",
            "--scope-prefix",
            "lane_scope",
            "--subsystem",
            "policy",
            "--subsystem",
            "benchmark",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert seen_subsystems == ["policy", "benchmark"]
    assert payload["summary"]["timed_out_runs"] == 1
    assert payload["summary"]["generated_runs"] == 1
    assert payload["summary"]["generated_success_runs"] == 1
    assert payload["summary"]["generated_failure_runs"] == 0
    assert payload["summary"]["failed_runs"] == 1
    assert payload["summary"]["healthy_runs"] == 1
    assert payload["summary"]["warning_runs"] == 0
    assert payload["summary"]["budget_exceeded_runs"] == 0
    assert payload["summary"]["selected_subsystems"] == ["benchmark"]
    assert payload["runs"][0]["scope_id"] == "lane_scope_1"
    assert payload["runs"][0]["observation_timed_out"] is True
    assert payload["runs"][0]["status"] == "failed"
    assert payload["runs"][0]["run_succeeded"] is False
    assert payload["runs"][0]["healthy_run"] is False
    assert payload["runs"][0]["generated_candidate"] is False
    assert payload["runs"][1]["selected_variant_id"] == "failure_cluster_growth"
    assert payload["runs"][1]["status"] == "healthy"
    assert payload["runs"][1]["primary_passed"] == 2
    assert payload["runs"][1]["generated_success_passed"] == 1
    assert payload["summary"]["requested_subsystems"] == ["benchmark", "policy"]


def test_run_parallel_supervised_cycles_treats_budget_exceeded_success_as_warning(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff.json",
    )

    class _FakePipe:
        def __init__(self, lines):
            self._lines = [line if line.endswith("\n") else f"{line}\n" for line in lines]

        def __iter__(self):
            return iter(self._lines)

        def close(self):
            return None

    class _FakePopen:
        def __init__(self, cmd, *, cwd, env, stdout, stderr, text, bufsize):
            del cwd, env, stdout, stderr, text, bufsize
            scope_id = cmd[cmd.index("--scope-id") + 1]
            scoped = module.scoped_improvement_cycle_config(config, scope_id)
            candidate_path = scoped.candidate_artifacts_root / "retrieval" / f"{scope_id}.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps({"artifact_kind": "retrieval_policy_set"}), encoding="utf-8")
            records = [
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="retrieval",
                    state="observe",
                    action="run_eval",
                    artifact_path=str(candidate_path),
                    artifact_kind="retrieval_policy_set",
                    reason="observe",
                    metrics_summary={
                        "total": 5,
                        "passed": 5,
                        "pass_rate": 1.0,
                        "generated_total": 5,
                        "generated_passed": 5,
                        "generated_pass_rate": 1.0,
                        "observation_returncode": 0,
                        "observation_elapsed_seconds": 105.268,
                        "observation_timed_out": False,
                        "observation_budget_exceeded": True,
                        "observation_warning": "observation exceeded budget 70.0s with elapsed 70.7s",
                    },
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="retrieval",
                    state="select",
                    action="select_experiment",
                    artifact_path=str(candidate_path),
                    artifact_kind="retrieval_policy_set",
                    reason="select",
                    metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem="retrieval",
                    state="generate",
                    action="generate_candidate",
                    artifact_path=str(candidate_path),
                    artifact_kind="retrieval_policy_set",
                    reason="generate",
                    metrics_summary={},
                    candidate_artifact_path=str(candidate_path),
                ),
            ]
            scoped.improvement_cycles_path.parent.mkdir(parents=True, exist_ok=True)
            scoped.improvement_cycles_path.write_text(
                "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
                encoding="utf-8",
            )
            self.stdout = _FakePipe(["generated candidate"])
            self.stderr = _FakePipe(["warning run"])
            self._returncode = 0

        def wait(self):
            return self._returncode

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_supervised_cycles.py",
            "--workers",
            "1",
            "--scope-prefix",
            "warning_scope",
            "--subsystem",
            "retrieval",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["completed_runs"] == 1
    assert payload["summary"]["failed_runs"] == 0
    assert payload["summary"]["healthy_runs"] == 0
    assert payload["summary"]["warning_runs"] == 1
    assert payload["summary"]["timed_out_runs"] == 0
    assert payload["summary"]["budget_exceeded_runs"] == 1
    assert payload["runs"][0]["status"] == "completed_with_warnings"
    assert payload["runs"][0]["run_succeeded"] is True
    assert payload["runs"][0]["healthy_run"] is False
    assert payload["runs"][0]["observation_timed_out"] is False
    assert payload["runs"][0]["observation_budget_exceeded"] is True


def test_parallel_supervised_subsystem_diversification_uses_recent_batch_history(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )
    config.ensure_directories()
    history_path = config.improvement_reports_dir / "parallel_supervised_preview_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "report_kind": "parallel_supervised_preview_history",
                "runs": [
                    {
                        "requested_subsystem": "policy",
                        "selected_subsystem": "",
                        "generated_candidate": False,
                        "observation_timed_out": True,
                        "returncode": 1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_latest_observe_metrics", lambda config: EvalMetrics(total=10, passed=7, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates: [
            ImprovementExperiment(
                subsystem="policy",
                reason="policy",
                priority=3,
                expected_gain=0.05,
                estimated_cost=2,
                score=0.92,
                evidence={},
            ),
            ImprovementExperiment(
                subsystem="benchmark",
                reason="benchmark",
                priority=3,
                expected_gain=0.04,
                estimated_cost=2,
                score=0.9,
                evidence={},
            ),
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval",
                priority=3,
                expected_gain=0.03,
                estimated_cost=2,
                score=0.88,
                evidence={},
            ),
        ],
    )

    diversified = module._planner_diversified_subsystems(config, workers=2)

    assert diversified == ["benchmark", "retrieval"]


def test_parallel_supervised_variant_diversification_assigns_distinct_ranked_variants(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
    )
    config.ensure_directories()

    monkeypatch.setattr(module, "_latest_observe_metrics", lambda config: EvalMetrics(total=10, passed=7, average_steps=1.0))
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "select_portfolio_campaign",
        lambda self, metrics, max_candidates: [
            ImprovementExperiment(
                subsystem="retrieval",
                reason="retrieval",
                priority=3,
                expected_gain=0.05,
                estimated_cost=2,
                score=0.92,
                evidence={},
            ),
            ImprovementExperiment(
                subsystem="policy",
                reason="policy",
                priority=3,
                expected_gain=0.04,
                estimated_cost=2,
                score=0.9,
                evidence={},
            ),
        ],
    )
    monkeypatch.setattr(
        module.ImprovementPlanner,
        "rank_variants",
        lambda self, experiment, metrics: (
            [
                ImprovementVariant(
                    subsystem="retrieval",
                    variant_id="confidence_gating",
                    description="retrieval variant one",
                    expected_gain=0.02,
                    estimated_cost=2,
                    score=0.05,
                    controls={},
                ),
                ImprovementVariant(
                    subsystem="retrieval",
                    variant_id="breadth_rebalance",
                    description="retrieval variant two",
                    expected_gain=0.019,
                    estimated_cost=2,
                    score=0.049,
                    controls={},
                ),
            ]
            if experiment.subsystem == "retrieval"
            else [
                ImprovementVariant(
                    subsystem="policy",
                    variant_id="verifier_alignment",
                    description="policy variant",
                    expected_gain=0.018,
                    estimated_cost=1,
                    score=0.03,
                    controls={},
                )
            ]
        ),
    )

    variant_ids = module._planner_variant_ids_for_subsystems(
        config,
        requested_subsystems=["retrieval", "policy", "retrieval"],
    )

    assert variant_ids == ["confidence_gating", "verifier_alignment", "breadth_rebalance"]


def test_run_parallel_supervised_cycles_can_auto_diversify_variants_per_worker(tmp_path, monkeypatch):
    module = _load_script("run_parallel_supervised_cycles.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "artifacts.json",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff.json",
    )

    seen_worker_requests: list[tuple[str, str, str]] = []

    class _FakePipe:
        def __init__(self, lines):
            self._lines = [line if line.endswith("\n") else f"{line}\n" for line in lines]

        def __iter__(self):
            return iter(self._lines)

        def close(self):
            return None

    class _FakePopen:
        def __init__(self, cmd, *, cwd, env, stdout, stderr, text, bufsize):
            del cwd, env, stdout, stderr, text, bufsize
            scope_id = cmd[cmd.index("--scope-id") + 1]
            subsystem = cmd[cmd.index("--subsystem") + 1]
            variant_id = cmd[cmd.index("--variant-id") + 1]
            seen_worker_requests.append((scope_id, subsystem, variant_id))
            scoped = module.scoped_improvement_cycle_config(config, scope_id)
            candidate_path = scoped.candidate_artifacts_root / subsystem / f"{scope_id}.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps({"artifact_kind": "candidate"}), encoding="utf-8")
            records = [
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem=subsystem,
                    state="observe",
                    action="run_eval",
                    artifact_path=str(candidate_path),
                    artifact_kind="candidate",
                    reason="observe",
                    metrics_summary={"observation_elapsed_seconds": 2.0, "observation_timed_out": False},
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem=subsystem,
                    state="select",
                    action="select_experiment",
                    artifact_path=str(candidate_path),
                    artifact_kind="candidate",
                    reason="select",
                    metrics_summary={"selected_variant_id": variant_id},
                ),
                ImprovementCycleRecord(
                    cycle_id=f"cycle:{scope_id}",
                    subsystem=subsystem,
                    state="generate",
                    action="generate_candidate",
                    artifact_path=str(candidate_path),
                    artifact_kind="candidate",
                    reason="generate",
                    metrics_summary={},
                    candidate_artifact_path=str(candidate_path),
                ),
            ]
            scoped.improvement_cycles_path.parent.mkdir(parents=True, exist_ok=True)
            scoped.improvement_cycles_path.write_text(
                "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
                encoding="utf-8",
            )
            self.stdout = _FakePipe([f"worker={scope_id}"])
            self.stderr = _FakePipe([f"subsystem={subsystem} variant={variant_id}"])
            self._returncode = 0

        def wait(self):
            return self._returncode

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        module,
        "_planner_variant_ids_for_subsystems",
        lambda config, requested_subsystems: ["confidence_gating", "verifier_alignment", "breadth_rebalance"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_supervised_cycles.py",
            "--workers",
            "3",
            "--scope-prefix",
            "lane_scope",
            "--subsystem",
            "retrieval",
            "--subsystem",
            "policy",
            "--subsystem",
            "retrieval",
            "--auto-diversify-variants",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert seen_worker_requests == [
        ("lane_scope_1", "retrieval", "confidence_gating"),
        ("lane_scope_2", "policy", "verifier_alignment"),
        ("lane_scope_3", "retrieval", "breadth_rebalance"),
    ]
    assert payload["summary"]["requested_variant_ids"] == [
        "breadth_rebalance",
        "confidence_gating",
        "verifier_alignment",
    ]
    assert payload["runs"][0]["requested_variant_id"] == "confidence_gating"
    assert payload["runs"][2]["requested_variant_id"] == "breadth_rebalance"


def test_compare_improvement_protocols_writes_summary_report(tmp_path, monkeypatch):
    module = _load_script("compare_improvement_protocols.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:auto",
            state="observe",
            subsystem="policy",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="auto",
            metrics_summary={"protocol": "autonomous", "protocol_match_id": "match:test:1"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:auto",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="retained",
            metrics_summary={
                "baseline_pass_rate": 0.7,
                "candidate_pass_rate": 0.8,
                "baseline_average_steps": 1.5,
                "candidate_average_steps": 1.2,
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:guided",
            state="observe",
            subsystem="policy",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="guided",
            metrics_summary={"protocol": "human_guided", "protocol_match_id": "match:test:1"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:guided",
            state="reject",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="rejected",
            metrics_summary={
                "baseline_pass_rate": 0.8,
                "candidate_pass_rate": 0.75,
                "baseline_average_steps": 1.2,
                "candidate_average_steps": 1.4,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=tmp_path / "episodes",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["compare_improvement_protocols.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert payload["report_kind"] == "improvement_protocol_comparison"
    assert payload["protocol_summary"]["autonomous"]["retained_cycles"] == 1
    assert payload["protocol_summary"]["human_guided"]["rejected_cycles"] == 1
    assert payload["winner_summary"]["winner"] == "autonomous"
    assert payload["winner_summary"]["autonomous_beats_human_guided"] is True
    assert payload["head_to_head_summary"]["matched_pairs"] == 1
    assert payload["matched_results"][0]["winner"] == "autonomous"
    assert records[-1]["artifact_kind"] == "improvement_protocol_comparison"


def test_run_protocol_head_to_head_writes_isolated_match_report(tmp_path, monkeypatch):
    module = _load_script("run_protocol_head_to_head.py")
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    reports_dir = tmp_path / "improvement" / "reports"
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    (episodes_root / "seed_task.json").write_text(
        json.dumps({"task_id": "seed_task", "success": True, "summary": {}, "task_metadata": {"benchmark_family": "workflow"}}),
        encoding="utf-8",
    )
    (episodes_root / "generated_success").mkdir(parents=True, exist_ok=True)
    (episodes_root / "generated_success" / "followup_task.json").write_text(
        json.dumps({"task_id": "followup_task", "success": False, "summary": {"failure_types": ["timeout"]}, "task_metadata": {"benchmark_family": "generated_task"}}),
        encoding="utf-8",
    )
    (episodes_root / "improvement").mkdir(parents=True, exist_ok=True)
    (episodes_root / "improvement" / "stale.txt").write_text("stale", encoding="utf-8")
    bundle_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_retrieval_asset_bundle"}), encoding="utf-8")
    world_model_path = tmp_path / "world_model" / "world_model_proposals.json"
    world_model_path.parent.mkdir(parents=True, exist_ok=True)
    world_model_path.write_text(json.dumps({"artifact_kind": "world_model_policy_set"}), encoding="utf-8")
    trust_path = tmp_path / "trust" / "trust_proposals.json"
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.write_text(json.dumps({"artifact_kind": "trust_policy_set"}), encoding="utf-8")
    recovery_path = tmp_path / "recovery" / "recovery_proposals.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(json.dumps({"artifact_kind": "recovery_policy_set"}), encoding="utf-8")
    delegation_path = tmp_path / "delegation" / "delegation_proposals.json"
    delegation_path.parent.mkdir(parents=True, exist_ok=True)
    delegation_path.write_text(json.dumps({"artifact_kind": "delegated_runtime_policy_set"}), encoding="utf-8")
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(json.dumps({"artifact_kind": "operator_policy_set"}), encoding="utf-8")
    transition_model_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    transition_model_path.parent.mkdir(parents=True, exist_ok=True)
    transition_model_path.write_text(json.dumps({"artifact_kind": "transition_model_policy_set"}), encoding="utf-8")
    queue_path = tmp_path / "jobs" / "queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    runtime_state_path = tmp_path / "jobs" / "runtime_state.json"
    runtime_state_path.write_text(json.dumps({"active": []}), encoding="utf-8")
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "resume.json").write_text(json.dumps({"checkpoint": True}), encoding="utf-8")
    snapshot_root = tmp_path / "recovery" / "workspaces"
    (snapshot_root / "hello_task").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "hello_task" / "state.txt").write_text("snapshot", encoding="utf-8")
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(json.dumps({"modules": [{"module_id": "github", "enabled": True}]}), encoding="utf-8")
    trust_ledger_path = tmp_path / "reports" / "trust.json"
    trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    trust_ledger_path.write_text(json.dumps({"ledger_kind": "unattended_trust_ledger"}), encoding="utf-8")

    def fake_run(cmd, cwd, capture_output, text, check, env):
        del cwd, capture_output, text, check
        assert "AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH" in env
        assert Path(env["AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH"]).exists()
        assert env["AGENT_KERNEL_USE_PROMPT_PROPOSALS"] == "0"
        assert env["AGENT_KERNEL_USE_CURRICULUM_PROPOSALS"] == "0"
        assert env["AGENT_KERNEL_USE_RETRIEVAL_PROPOSALS"] == "0"
        assert "AGENT_KERNEL_RUN_CHECKPOINTS_DIR" in env
        assert Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]).exists()
        assert json.loads((Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]) / "resume.json").read_text(encoding="utf-8"))["checkpoint"] is True
        assert json.loads((Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]) / "seed_task.json").read_text(encoding="utf-8"))["task_id"] == "seed_task"
        assert json.loads((Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]) / "generated_success" / "followup_task.json").read_text(encoding="utf-8"))["task_id"] == "followup_task"
        assert not (Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]) / "improvement" / "stale.txt").exists()
        assert json.loads(Path(env["AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "world_model_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_TRUST_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "trust_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_RECOVERY_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "recovery_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_DELEGATION_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "delegated_runtime_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "operator_policy_set"
        assert json.loads(Path(env["AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH"]).read_text(encoding="utf-8"))["artifact_kind"] == "transition_model_policy_set"
        assert "AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH" in env
        assert Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]).parent.exists()
        assert "AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH" in env
        assert Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]).parent.exists()
        assert "AGENT_KERNEL_CAPABILITY_MODULES_PATH" in env
        assert json.loads(Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]).read_text(encoding="utf-8"))["modules"][0]["module_id"] == "github"
        assert "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH" in env
        assert json.loads(Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]).read_text(encoding="utf-8"))["ledger_kind"] == "unattended_trust_ledger"
        assert (Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]) / "hello_task" / "state.txt").read_text(encoding="utf-8") == "snapshot"
        assert "AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD" in env
        protocol = "autonomous" if "run_improvement_cycle.py" in cmd[1] else "human_guided"
        isolated_cycles_path = Path(env["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"])
        isolated_cycles_path.parent.mkdir(parents=True, exist_ok=True)
        planner = ImprovementPlanner(
            memory_root=Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]),
            cycles_path=isolated_cycles_path,
        )
        cycle_id = f"cycle:policy:{protocol}"
        planner.append_cycle_record(
            isolated_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem="policy",
                action="run_eval",
                artifact_path="",
                artifact_kind="eval_metrics",
                reason=protocol,
                metrics_summary={"protocol": protocol, "protocol_match_id": "match:test"},
            ),
        )
        planner.append_cycle_record(
            isolated_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="policy",
                action="choose_target",
                artifact_path="",
                artifact_kind="improvement_target",
                reason=protocol,
                metrics_summary={
                    "selected_variant": {"variant_id": "verifier_alignment" if protocol == "autonomous" else "careful"},
                },
            ),
        )
        planner.append_cycle_record(
            isolated_cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="retain" if protocol == "autonomous" else "reject",
                subsystem="policy",
                action="finalize_cycle",
                artifact_path="policy.json",
                artifact_kind="prompt_proposal_set",
                reason=protocol,
                metrics_summary={
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": 0.8 if protocol == "autonomous" else 0.68,
                    "baseline_average_steps": 1.5,
                    "candidate_average_steps": 1.2 if protocol == "autonomous" else 1.6,
                },
            ),
        )
        return CompletedProcess(args=cmd, returncode=0, stdout=f"{protocol}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            improvement_cycles_path=cycles_path,
            improvement_reports_dir=reports_dir,
            trajectories_root=episodes_root,
            prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
            retrieval_proposals_path=tmp_path / "retrieval" / "retrieval_proposals.json",
            retrieval_asset_bundle_path=bundle_path,
            benchmark_candidates_path=tmp_path / "benchmarks" / "benchmark_candidates.json",
            verifier_contracts_path=tmp_path / "verifiers" / "verifier_contracts.json",
            skills_path=tmp_path / "skills" / "command_skills.json",
            operator_classes_path=tmp_path / "operators" / "operator_classes.json",
            tool_candidates_path=tmp_path / "tools" / "tool_candidates.json",
            world_model_proposals_path=world_model_path,
            trust_proposals_path=trust_path,
            recovery_proposals_path=recovery_path,
            delegation_proposals_path=delegation_path,
            operator_policy_proposals_path=operator_policy_path,
            transition_model_proposals_path=transition_model_path,
            curriculum_proposals_path=tmp_path / "curriculum" / "curriculum_proposals.json",
            capability_modules_path=modules_path,
            delegated_job_queue_path=queue_path,
            delegated_job_runtime_state_path=runtime_state_path,
            run_checkpoints_dir=checkpoints_dir,
            unattended_workspace_snapshot_root=snapshot_root,
            unattended_trust_ledger_path=trust_ledger_path,
            use_prompt_proposals=False,
            use_curriculum_proposals=False,
            use_retrieval_proposals=False,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_protocol_head_to_head.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert payload["report_kind"] == "protocol_head_to_head_report"
    assert payload["summary"]["autonomous_wins"] == 1
    assert payload["summary"]["autonomous_beats_human_guided"] is True
    assert payload["matches"][0]["winner"] == "autonomous"
    assert records[-1]["artifact_kind"] == "protocol_head_to_head_report"
