import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.trust import (
    build_unattended_trust_ledger,
    evaluate_unattended_trust,
    write_unattended_trust_ledger,
)


def _write_report(
    reports_dir: Path,
    name: str,
    *,
    generated_at: str,
    benchmark_family: str,
    outcome: str,
    success: bool,
    hidden_side_effect_risk: bool,
    unexpected_change_files: int = 0,
    rollback_performed: bool = False,
    task_origin: str = "",
    task_metadata: dict[str, object] | None = None,
    trust_scope: str = "",
    command_steps: int = 1,
    supervision: dict[str, object] | None = None,
) -> None:
    report_task_metadata = dict(task_metadata or {})
    if task_origin:
        report_task_metadata["task_origin"] = task_origin
    if supervision is None:
        supervision = {
            "mode": "unattended",
            "independent_execution": True,
        }
    reports_dir.joinpath(name).write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "generated_at": generated_at,
                "task_id": name.removesuffix(".json"),
                "benchmark_family": benchmark_family,
                "outcome": outcome,
                "success": success,
                "task_metadata": report_task_metadata,
                "trust_scope": trust_scope,
                "supervision": dict(supervision),
                "summary": {
                    "unexpected_change_files": unexpected_change_files,
                    "command_steps": command_steps,
                },
                "commands": [{} for _ in range(max(0, int(command_steps)))],
                "side_effects": {"hidden_side_effect_risk": hidden_side_effect_risk},
                "recovery": {"rollback_performed": rollback_performed},
            }
        ),
        encoding="utf-8",
    )


def _write_campaign_report(
    reports_dir: Path,
    name: str,
    *,
    generated_at: str,
    required_families_with_reports: list[str],
    sampled_families_from_progress: list[str],
    family_observed_decisions: dict[str, int] | None = None,
    required_family_clean_task_root_counts: dict[str, int] | None = None,
) -> None:
    normalized_decisions = {
        str(family).strip(): max(0, int(count))
        for family, count in dict(family_observed_decisions or {}).items()
        if str(family).strip()
    }
    priority_families = ["project", "repository", "integration", "repo_chore"]
    family_summaries = {
        family: {
            "observed_decisions": normalized_decisions.get(family, 0),
            "retained_positive_delta_decisions": 0,
        }
        for family in priority_families
    }
    priority_families_with_signal = [
        family for family in priority_families if int(family_summaries[family]["observed_decisions"]) > 0
    ]
    priority_families_without_signal = [
        family for family in priority_families if int(family_summaries[family]["observed_decisions"]) <= 0
    ]
    reports_dir.joinpath(name).write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "generated_at": generated_at,
                "trust_breadth_summary": {
                    "required_families": [
                        "integration",
                        "project",
                        "repo_chore",
                        "repo_sandbox",
                        "repository",
                    ],
                    "required_families_with_reports": required_families_with_reports,
                    "required_family_clean_task_root_counts": dict(required_family_clean_task_root_counts or {}),
                },
                "priority_family_yield_summary": {
                    "priority_families": priority_families,
                    "family_summaries": family_summaries,
                    "priority_families_with_signal": priority_families_with_signal,
                    "priority_families_with_retained_gain": [],
                    "priority_families_without_signal": priority_families_without_signal,
                    "priority_families_with_signal_but_no_retained_gain": [],
                },
                "decision_stream_summary": {
                    "runtime_managed": {
                        "total_decisions": 0,
                        "retained_cycles": 0,
                        "rejected_cycles": 0,
                    }
                },
                "runs": [
                    {
                        "partial_progress": {
                            "sampled_families_from_progress": sampled_families_from_progress,
                        },
                        "priority_family_rerouting": {
                            "applied": True,
                            "reason": "unsampled_priority_families",
                            "priority_benchmark_families": ["integration", "project", "repository"],
                            "unsampled_priority_families": ["integration"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_unattended_trust_ledger_summarizes_recent_reports(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repo_chore_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    _write_report(
        reports_dir,
        "repo_sandbox_safe_stop.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repo_sandbox",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        rollback_performed=True,
    )
    _write_report(
        reports_dir,
        "repo_chore_hidden.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=True,
    )
    _write_report(
        reports_dir,
        "external_lab_success.json",
        generated_at="2026-03-25T00:00:04+00:00",
        benchmark_family="external_lab",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_origin="external_manifest",
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=2,
        unattended_trust_breadth_min_reports=2,
        unattended_trust_required_benchmark_families=("repo_chore", "repo_sandbox"),
        unattended_trust_min_distinct_families=2,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["reports_considered"] == 4
    assert ledger["overall_summary"]["distinct_benchmark_families"] == 3
    assert ledger["gated_summary"]["success_count"] == 2
    assert ledger["gated_summary"]["rollback_performed_count"] == 1
    assert ledger["family_summaries"]["repo_chore"]["hidden_side_effect_risk_count"] == 1
    assert ledger["family_summaries"]["repo_chore"]["false_pass_risk_count"] == 1
    assert ledger["external_summary"]["total"] == 1
    assert ledger["external_summary"]["distinct_benchmark_families"] == 1
    assert ledger["external_family_summaries"]["external_lab"]["success_count"] == 1
    assert ledger["coverage_summary"]["required_families"] == ["repo_chore", "repo_sandbox"]
    assert ledger["coverage_summary"]["required_families_with_reports"] == ["repo_chore", "repo_sandbox"]
    assert ledger["coverage_summary"]["required_families_with_gated_reports"] == ["repo_chore", "repo_sandbox"]
    assert ledger["coverage_summary"]["missing_required_families"] == []
    assert ledger["coverage_summary"]["missing_required_gated_families"] == []
    assert ledger["coverage_summary"]["distinct_family_gap"] == 0


def test_build_unattended_trust_ledger_recursively_counts_nested_reports_but_skips_generated_failure_seed(
    tmp_path: Path,
):
    reports_dir = tmp_path / "reports"
    nested_reports_dir = reports_dir / "cycle_retrieval_preview"
    generated_failure_seed_dir = nested_reports_dir / "generated_failure_seed"
    generated_failure_seed_dir.mkdir(parents=True)
    _write_report(
        nested_reports_dir,
        "project_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="project",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        trust_scope="gated",
    )
    _write_report(
        nested_reports_dir,
        "repository_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        trust_scope="gated",
    )
    _write_report(
        generated_failure_seed_dir,
        "repository_seed_failure.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repository",
        outcome="unsafe_ambiguous",
        success=False,
        hidden_side_effect_risk=False,
        trust_scope="gated",
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=2,
        unattended_trust_breadth_min_reports=2,
        unattended_trust_required_benchmark_families=("project", "repository"),
        unattended_trust_min_distinct_families=2,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["reports_considered"] == 2
    assert ledger["overall_summary"]["total"] == 2
    assert ledger["family_summaries"]["project"]["success_count"] == 1
    assert ledger["family_summaries"]["repository"]["success_count"] == 1
    assert ledger["coverage_summary"]["required_families_with_reports"] == ["project", "repository"]
    assert ledger["coverage_summary"]["missing_required_families"] == []


def test_build_unattended_trust_ledger_distinguishes_open_world_and_replay_task_yield(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "semantic_repo_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_origin="semantic_hub",
    )
    _write_report(
        reports_dir,
        "external_project_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="project",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_origin="external_manifest",
    )
    _write_report(
        reports_dir,
        "episode_replay_safe_stop.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repository",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        task_origin="episode_replay",
    )
    config = KernelConfig(run_reports_dir=reports_dir, unattended_trust_recent_report_limit=10)

    ledger = build_unattended_trust_ledger(config)

    bucket_summary = ledger["overall_summary"]["task_yield_bucket_summary"]
    assert bucket_summary["semantic_hub"]["reports"] == 1
    assert bucket_summary["semantic_hub"]["clean_success_count"] == 1
    assert bucket_summary["semantic_hub"]["benchmark_families"] == ["repository"]
    assert bucket_summary["external_manifest"]["reports"] == 1
    assert bucket_summary["external_manifest"]["clean_success_count"] == 1
    assert bucket_summary["replay_derived"]["reports"] == 1
    assert bucket_summary["replay_derived"]["success_count"] == 0
    assert ledger["coverage_summary"]["open_world_task_yield_summary"]["semantic_hub"]["reports"] == 1
    assert ledger["coverage_summary"]["open_world_task_yield_summary"]["external_manifest"]["reports"] == 1
    assert ledger["coverage_summary"]["replay_derived_task_yield_summary"]["reports"] == 1


def test_build_unattended_trust_ledger_surfaces_repo_semantic_clusters(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "project_validation_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="project",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"repo_semantics": ["project", "validation"]},
    )
    _write_report(
        reports_dir,
        "repository_integration_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"repo_semantics": ["repository", "integration", "shared_repo"]},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("project", "repository"),
        unattended_trust_min_distinct_families=2,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["overall_summary"]["distinct_repo_semantic_clusters"] == 5
    assert ledger["overall_summary"]["repo_semantic_clusters"] == [
        "integration",
        "project",
        "repository",
        "shared_repo",
        "validation",
    ]
    assert ledger["coverage_summary"]["distinct_repo_semantic_clusters"] == 5
    assert ledger["coverage_summary"]["observed_repo_semantic_clusters"] == [
        "integration",
        "project",
        "repository",
        "shared_repo",
        "validation",
    ]


def test_build_unattended_trust_ledger_uses_campaign_runtime_managed_signals(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_campaign_report(
        reports_dir,
        "campaign_report.json",
        generated_at="2026-04-05T00:00:01+00:00",
        required_families_with_reports=["project", "repo_sandbox", "repository"],
        sampled_families_from_progress=["project", "repository", "integration"],
        family_observed_decisions={"project": 2, "repository": 1},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=2,
        unattended_trust_breadth_min_reports=2,
        unattended_trust_required_benchmark_families=("project", "repository", "integration", "repo_chore", "repo_sandbox"),
        unattended_trust_min_distinct_families=2,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["runtime_managed_breadth_signal_families"] == [
        "project",
        "repository",
    ]
    assert ledger["campaign_summary"]["required_families_with_sampled_progress"] == [
        "integration",
        "project",
        "repository",
    ]
    assert ledger["campaign_summary"]["required_family_sampled_progress_counts"] == {
        "integration": 1,
        "project": 1,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 1,
    }
    assert ledger["coverage_summary"]["required_family_runtime_managed_signal_counts"] == {
        "integration": 0,
        "project": 2,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 1,
    }
    assert ledger["coverage_summary"]["required_family_sampled_progress_counts"] == {
        "integration": 1,
        "project": 1,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 1,
    }
    assert ledger["coverage_summary"]["required_families_with_sampled_progress"] == [
        "integration",
        "project",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_with_runtime_managed_signals"] == [
        "project",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_family_runtime_managed_decision_yield_counts"] == {
        "integration": 0,
        "project": 0,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_families_missing_runtime_managed_decision_yield"] == [
        "integration",
        "project",
        "repo_chore",
        "repo_sandbox",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_missing_runtime_managed_signal"] == [
        "integration",
        "repo_chore",
        "repo_sandbox",
    ]
    assert ledger["coverage_summary"]["required_families_with_reports"] == []
    assert ledger["coverage_summary"]["required_families_with_gated_reports"] == []
    assert ledger["coverage_summary"]["missing_required_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repo_sandbox",
        "repository",
    ]
    assert ledger["coverage_summary"]["missing_required_gated_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repo_sandbox",
        "repository",
    ]


def test_build_unattended_trust_ledger_does_not_promote_sampled_families_without_runtime_managed_decisions(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_campaign_report(
        reports_dir,
        "campaign_report.json",
        generated_at="2026-04-05T00:00:01+00:00",
        required_families_with_reports=["project", "repository", "integration", "repo_chore"],
        sampled_families_from_progress=["project", "repository", "integration", "repo_chore"],
        family_observed_decisions={"project": 1},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=2,
        unattended_trust_breadth_min_reports=2,
        unattended_trust_required_benchmark_families=("project", "repository", "integration", "repo_chore"),
        unattended_trust_min_distinct_families=2,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["required_families_with_reports"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["campaign_summary"]["sampled_families_from_progress"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["campaign_summary"]["required_families_with_sampled_progress"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["campaign_summary"]["runtime_managed_breadth_signal_families"] == ["project"]
    assert ledger["coverage_summary"]["required_family_runtime_managed_signal_counts"] == {
        "integration": 0,
        "project": 1,
        "repo_chore": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_family_runtime_managed_decision_yield_counts"] == {
        "integration": 0,
        "project": 0,
        "repo_chore": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_families_with_runtime_managed_signals"] == ["project"]
    assert ledger["coverage_summary"]["required_family_sampled_progress_counts"] == {
        "integration": 1,
        "project": 1,
        "repo_chore": 1,
        "repository": 1,
    }
    assert ledger["coverage_summary"]["required_families_with_sampled_progress"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts"] == {
        "integration": 1,
        "project": 1,
        "repo_chore": 1,
        "repository": 1,
    }
    assert ledger["coverage_summary"]["required_families_missing_runtime_managed_signal"] == [
        "integration",
        "repo_chore",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_family_report_counts"] == {
        "integration": 0,
        "project": 0,
        "repo_chore": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_families_with_reports"] == []
    assert ledger["coverage_summary"]["missing_required_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_missing_runtime_managed_decision_yield"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]


def test_build_unattended_trust_ledger_does_not_grant_report_credit_from_sampled_progress(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_campaign_report(
        reports_dir,
        "campaign_report.json",
        generated_at="2026-04-05T00:00:01+00:00",
        required_families_with_reports=["project", "repository"],
        sampled_families_from_progress=["project", "repository", "integration", "workflow"],
        family_observed_decisions={},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("project", "repository", "integration", "repo_chore"),
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["required_families_with_sampled_progress"] == [
        "integration",
        "project",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_with_reports"] == []
    assert ledger["coverage_summary"]["missing_required_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]
    assert ledger["coverage_summary"]["required_families_with_gated_reports"] == []
    assert ledger["coverage_summary"]["missing_required_gated_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repository",
    ]


def test_build_unattended_trust_ledger_merges_campaign_clean_task_root_counts(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_campaign_report(
        reports_dir,
        "campaign_report.json",
        generated_at="2026-04-05T00:00:01+00:00",
        required_families_with_reports=["project", "repository"],
        sampled_families_from_progress=["project", "repository"],
        family_observed_decisions={"project": 1},
        required_family_clean_task_root_counts={"project": 2, "repository": 1},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("project", "repository", "integration", "repo_chore"),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["required_family_clean_task_root_counts"]["project"] == 2
    assert ledger["coverage_summary"]["required_family_clean_task_root_counts"]["project"] == 2
    assert ledger["coverage_summary"]["required_family_clean_task_root_counts"]["repository"] == 1



def test_build_unattended_trust_ledger_tracks_family_decision_yield(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    reports_dir.joinpath("campaign_report.json").write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "generated_at": "2026-04-05T00:00:01+00:00",
                "trust_breadth_summary": {
                    "required_families": ["project", "repository", "integration", "repo_chore"],
                    "required_families_with_reports": ["project", "repository", "integration", "repo_chore"],
                },
                "priority_family_yield_summary": {
                    "priority_families": ["project", "repository", "integration", "repo_chore"],
                    "family_summaries": {
                        "project": {"observed_decisions": 2, "retained_decisions": 1, "retained_positive_delta_decisions": 1},
                        "repository": {"observed_decisions": 2, "retained_decisions": 1, "retained_positive_delta_decisions": 0},
                        "integration": {"observed_decisions": 1, "retained_decisions": 0, "retained_positive_delta_decisions": 0},
                        "repo_chore": {"observed_decisions": 0, "retained_decisions": 0, "retained_positive_delta_decisions": 0},
                    },
                },
                "decision_stream_summary": {"runtime_managed": {"total_decisions": 3, "retained_cycles": 1, "rejected_cycles": 2}},
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("project", "repository", "integration", "repo_chore"),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["runtime_managed_decision_yield_families"] == ["project"]
    assert ledger["campaign_summary"]["required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield"] == []
    assert ledger["coverage_summary"]["required_family_runtime_managed_decision_yield_counts"] == {
        "integration": 0,
        "project": 1,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_family_counted_evidence_summary"]["project"] == {
        "sampled_progress_count": 0,
        "verified_signal_count": 2,
        "retained_decision_count": 1,
        "decision_yield_count": 1,
        "clean_task_root_count": 0,
        "highest_confirmed_stage": "yielded",
        "missing_decision_yield_after_sampling": False,
    }
    assert ledger["coverage_summary"]["required_family_counted_evidence_summary"]["repository"] == {
        "sampled_progress_count": 0,
        "verified_signal_count": 2,
        "retained_decision_count": 1,
        "decision_yield_count": 0,
        "clean_task_root_count": 0,
        "highest_confirmed_stage": "retained",
        "missing_decision_yield_after_sampling": False,
    }
    assert ledger["coverage_summary"]["required_families_with_runtime_managed_decision_yield"] == ["project"]
    assert ledger["coverage_summary"]["required_families_with_sampled_progress_but_missing_runtime_managed_decision_yield"] == []
    assert ledger["coverage_summary"]["required_family_sampled_progress_but_missing_runtime_managed_decision_yield_counts"] == {
        "integration": 0,
        "project": 0,
        "repo_chore": 0,
        "repo_sandbox": 0,
        "repository": 0,
    }
    assert ledger["coverage_summary"]["required_families_missing_runtime_managed_decision_yield"] == [
        "integration",
        "repo_chore",
        "repo_sandbox",
        "repository",
    ]


def test_build_unattended_trust_ledger_accumulates_recent_campaign_family_signal(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_campaign_report(
        reports_dir,
        "campaign_report_older.json",
        generated_at="2026-04-05T00:00:01+00:00",
        required_families_with_reports=["integration"],
        sampled_families_from_progress=["integration"],
        family_observed_decisions={"integration": 1},
    )
    _write_campaign_report(
        reports_dir,
        "campaign_report_newer.json",
        generated_at="2026-04-05T00:00:02+00:00",
        required_families_with_reports=["repo_chore"],
        sampled_families_from_progress=["repo_chore"],
        family_observed_decisions={"repo_chore": 1},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("integration", "repo_chore"),
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["campaign_summary"]["reports_considered"] == 2
    assert ledger["campaign_summary"]["required_families_with_reports"] == ["integration", "repo_chore"]
    assert ledger["campaign_summary"]["runtime_managed_breadth_signal_families"] == [
        "integration",
        "repo_chore",
    ]
    assert ledger["coverage_summary"]["required_family_runtime_managed_signal_counts"] == {
        "integration": 1,
        "repo_chore": 1,
    }
    assert ledger["coverage_summary"]["required_families_with_runtime_managed_signals"] == [
        "integration",
        "repo_chore",
    ]
    assert ledger["coverage_summary"]["required_families_with_reports"] == []
    assert ledger["coverage_summary"]["missing_required_families"] == ["integration", "repo_chore"]


def test_build_unattended_trust_ledger_tracks_light_supervision_independence(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repo_sandbox_light_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repo_sandbox",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        supervision={
            "mode": "light_supervision",
            "operator_turns": 1,
            "independent_execution": True,
            "light_supervision_candidate": True,
            "light_supervision_success": True,
            "light_supervision_clean_success": True,
        },
    )
    _write_report(
        reports_dir,
        "repo_sandbox_guided_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repo_sandbox",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        supervision={
            "mode": "guided",
            "operator_turns": 3,
            "independent_execution": False,
            "light_supervision_candidate": False,
            "light_supervision_success": False,
            "light_supervision_clean_success": False,
        },
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repo_sandbox",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["overall_summary"]["supervision_modes"] == {
        "guided": 1,
        "light_supervision": 1,
    }
    assert ledger["overall_summary"]["independent_execution_count"] == 1
    assert ledger["overall_summary"]["light_supervision_candidate_count"] == 1
    assert ledger["overall_summary"]["light_supervision_success_count"] == 1
    assert ledger["overall_summary"]["light_supervision_clean_success_count"] == 1
    assert ledger["coverage_summary"]["required_family_light_supervision_report_counts"]["repo_sandbox"] == 1
    assert ledger["coverage_summary"]["required_family_light_supervision_success_counts"]["repo_sandbox"] == 1
    assert ledger["coverage_summary"]["required_family_light_supervision_clean_success_counts"]["repo_sandbox"] == 1


def test_build_unattended_trust_ledger_tracks_contract_clean_failure_recovery_counts(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repository_failure_recovery_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={
            "curriculum_kind": "failure_recovery",
            "source_task": "repository_repair_seed",
        },
        supervision={
            "mode": "unattended",
            "operator_turns": 0,
            "independent_execution": True,
            "contract_clean_failure_recovery_origin": True,
            "contract_clean_failure_recovery_step_floor": 12,
            "contract_clean_failure_recovery_candidate": True,
            "contract_clean_failure_recovery_success": True,
            "contract_clean_failure_recovery_clean_success": True,
        },
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["overall_summary"]["contract_clean_failure_recovery_candidate_count"] == 1
    assert ledger["overall_summary"]["contract_clean_failure_recovery_success_count"] == 1
    assert ledger["overall_summary"]["contract_clean_failure_recovery_clean_success_count"] == 1
    assert ledger["overall_summary"]["failure_recovery_summary"]["reports"] == 1
    assert ledger["overall_summary"]["failure_recovery_summary"]["success_count"] == 1
    assert ledger["overall_summary"]["failure_recovery_summary"]["clean_success_count"] == 1
    assert ledger["overall_summary"]["failure_recovery_summary"]["task_roots"] == ["repository_repair_seed"]
    assert ledger["overall_summary"]["success_rate_confidence_interval"]["upper"] >= 0.9
    assert (
        ledger["coverage_summary"]["required_family_contract_clean_failure_recovery_report_counts"]["repository"] == 1
    )
    assert (
        ledger["coverage_summary"]["required_family_contract_clean_failure_recovery_success_counts"]["repository"] == 1
    )
    assert (
        ledger["coverage_summary"]["required_family_contract_clean_failure_recovery_clean_success_counts"][
            "repository"
        ]
        == 1
    )
    assert ledger["coverage_summary"]["required_family_failure_recovery_report_counts"]["repository"] == 1
    assert ledger["coverage_summary"]["required_family_failure_recovery_success_counts"]["repository"] == 1
    assert ledger["coverage_summary"]["required_family_failure_recovery_clean_success_counts"]["repository"] == 1


def test_build_unattended_trust_ledger_tracks_failure_recovery_viability_and_confidence(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repository_failure_recovery_success.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"curriculum_kind": "failure_recovery", "source_task": "repo_seed_a"},
    )
    _write_report(
        reports_dir,
        "repository_failure_recovery_safe_stop.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        task_metadata={"curriculum_kind": "failure_recovery", "source_task": "repo_seed_b"},
    )
    _write_report(
        reports_dir,
        "project_clean_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="project",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    failure_recovery = ledger["overall_summary"]["failure_recovery_summary"]
    assert failure_recovery["reports"] == 2
    assert failure_recovery["success_count"] == 1
    assert failure_recovery["clean_success_count"] == 1
    assert failure_recovery["success_rate"] == 0.5
    assert 0.0 < failure_recovery["success_rate_confidence_interval"]["lower"] < 0.5
    assert 0.5 < failure_recovery["success_rate_confidence_interval"]["upper"] <= 1.0
    assert ledger["overall_summary"]["success_rate_confidence_interval"]["lower"] > 0.2


def test_build_unattended_trust_ledger_tracks_false_pass_risk_and_clean_success_streak(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "clean_newest.json",
        generated_at="2026-03-25T00:00:04+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        unexpected_change_files=0,
    )
    _write_report(
        reports_dir,
        "clean_older.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        unexpected_change_files=0,
    )
    _write_report(
        reports_dir,
        "risky_oldest.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        unexpected_change_files=2,
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["overall_summary"]["false_pass_risk_count"] == 1
    assert ledger["overall_summary"]["false_pass_risk_rate"] == 1 / 3
    assert ledger["overall_summary"]["clean_success_streak"] == 2


def test_evaluate_unattended_trust_restricts_high_risk_repo_family(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for index in range(4):
        _write_report(
            reports_dir,
            f"repo_chore_{index}.json",
            generated_at=f"2026-03-25T00:00:0{index}+00:00",
            benchmark_family="repo_chore",
            outcome="unsafe_ambiguous",
            success=False,
            hidden_side_effect_risk=True,
            rollback_performed=True,
        )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_bootstrap_min_reports=3,
        unattended_trust_breadth_min_reports=3,
    )

    evaluation = evaluate_unattended_trust(config, benchmark_family="repo_chore")

    assert evaluation["passed"] is False
    assert evaluation["status"] == "restricted"
    assert any("unsafe_ambiguous_rate" in failure for failure in evaluation["failing_thresholds"])


def test_write_unattended_trust_ledger_persists_json(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    ledger_path = tmp_path / "trust" / "ledger.json"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repo_chore_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    config = KernelConfig(run_reports_dir=reports_dir, unattended_trust_ledger_path=ledger_path)

    target = write_unattended_trust_ledger(config)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert target == ledger_path
    assert payload["ledger_kind"] == "unattended_trust_ledger"
    assert payload["reports_considered"] == 1


def test_evaluate_unattended_trust_applies_retained_trust_controls(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for index in range(3):
        _write_report(
            reports_dir,
            f"repo_chore_success_{index}.json",
            generated_at=f"2026-03-25T00:00:0{index}+00:00",
            benchmark_family="repo_chore",
            outcome="success",
            success=True,
            hidden_side_effect_risk=False,
        )
    trust_path = tmp_path / "trust" / "trust_proposals.json"
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.write_text(
        json.dumps(
            {
                "artifact_kind": "trust_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "required_benchmark_families": ["repo_chore"],
                    "bootstrap_min_reports": 2,
                    "breadth_min_reports": 2,
                    "min_distinct_families": 1,
                    "min_success_rate": 0.6,
                    "max_unsafe_ambiguous_rate": 0.1,
                    "max_hidden_side_effect_rate": 0.1,
                    "max_success_hidden_side_effect_rate": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        trust_proposals_path=trust_path,
        unattended_trust_bootstrap_min_reports=5,
        unattended_trust_breadth_min_reports=10,
        unattended_trust_min_distinct_families=2,
    )

    evaluation = evaluate_unattended_trust(config, benchmark_family="repo_chore")

    assert evaluation["passed"] is True
    assert evaluation["status"] == "trusted"
    assert "repo_chore" in evaluation["family_posture"]["required"]


def test_build_unattended_trust_ledger_excludes_coverage_only_retrieval_reports_from_gated_summary(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repository_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    _write_report(
        reports_dir,
        "repository_retrieval_probe.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repository",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        trust_scope="coverage_only",
        task_metadata={
            "requires_retrieval": True,
            "source_task": "repo_sync_matrix_task",
        },
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_recent_report_limit=10,
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)
    evaluation = evaluate_unattended_trust(config, benchmark_family="repository")

    assert ledger["reports_considered"] == 2
    assert ledger["overall_summary"]["total"] == 2
    assert ledger["family_summaries"]["repository"]["total"] == 2
    assert ledger["gated_summary"]["total"] == 1
    assert ledger["gated_summary"]["success_count"] == 1
    assert ledger["gated_family_summaries"]["repository"]["total"] == 1
    assert ledger["coverage_summary"]["required_family_report_counts"]["repository"] == 2
    assert ledger["coverage_summary"]["required_family_gated_report_counts"]["repository"] == 1
    assert ledger["coverage_summary"]["required_family_coverage_only_report_counts"]["repository"] == 1
    assert ledger["coverage_summary"]["required_families_with_gated_reports"] == ["repository"]
    assert ledger["coverage_summary"]["missing_required_gated_families"] == []
    assert ledger["family_assessments"]["repository"]["status"] == "bootstrap"
    assert evaluation["status"] == "bootstrap"
    assert evaluation["family_summary"]["total"] == 1
    assert evaluation["family_summary"]["distinct_clean_success_task_roots"] == 1


def test_build_unattended_trust_ledger_tracks_missing_required_gated_family_when_only_coverage_only_probe_exists(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "integration_probe.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="integration",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        trust_scope="coverage_only",
        task_metadata={
            "requires_retrieval": True,
            "source_task": "incident_matrix_task",
        },
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_required_benchmark_families=("integration",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["coverage_summary"]["required_family_report_counts"]["integration"] == 1
    assert ledger["coverage_summary"]["required_family_gated_report_counts"]["integration"] == 0
    assert ledger["coverage_summary"]["required_family_coverage_only_report_counts"]["integration"] == 1
    assert ledger["coverage_summary"]["required_families_with_reports"] == ["integration"]
    assert ledger["coverage_summary"]["required_families_with_gated_reports"] == []
    assert ledger["coverage_summary"]["missing_required_gated_families"] == ["integration"]


def test_build_unattended_trust_ledger_excludes_nonexecuting_safe_stop_reports_from_gated_summary(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "integration_setup_only.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="integration",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        trust_scope="coverage_only",
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        use_trust_proposals=False,
        unattended_trust_required_benchmark_families=("integration",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["overall_summary"]["total"] == 1
    assert ledger["gated_summary"]["total"] == 0
    assert ledger["coverage_summary"]["required_family_report_counts"]["integration"] == 1
    assert ledger["coverage_summary"]["required_family_gated_report_counts"]["integration"] == 0
    assert ledger["coverage_summary"]["required_family_coverage_only_report_counts"]["integration"] == 1
    assert ledger["coverage_summary"]["missing_required_gated_families"] == ["integration"]


def test_build_unattended_trust_ledger_counts_only_executable_gated_reports_for_assessment(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "project_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="project",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        command_steps=1,
    )
    _write_report(
        reports_dir,
        "project_policy_terminated.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="project",
        outcome="safe_stop",
        success=False,
        hidden_side_effect_risk=False,
        command_steps=0,
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("project",),
        unattended_trust_bootstrap_min_reports=2,
        unattended_trust_breadth_min_reports=2,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)
    evaluation = evaluate_unattended_trust(config, benchmark_family="project")

    assert ledger["gated_family_summaries"]["project"]["total"] == 2
    assert ledger["counted_gated_family_summaries"]["project"]["total"] == 1
    assert ledger["counted_gated_family_summaries"]["project"]["success_count"] == 1
    assert ledger["coverage_summary"]["required_family_gated_report_counts"]["project"] == 2
    assert ledger["coverage_summary"]["required_family_counted_gated_report_counts"]["project"] == 1
    assert ledger["coverage_summary"]["required_families_with_counted_gated_reports"] == ["project"]
    assert evaluation["status"] == "bootstrap"
    assert evaluation["family_summary"]["total"] == 1
    assert evaluation["family_attempt_summary"]["total"] == 2
    assert evaluation["gated_summary"]["total"] == 1
    assert evaluation["gated_attempt_summary"]["total"] == 2


def test_build_unattended_trust_ledger_excludes_legacy_unsupervised_reports_from_counted_summary(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repo_sandbox_legacy_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repo_sandbox",
        outcome="success",
        success=True,
        hidden_side_effect_risk=True,
        supervision={},
    )
    _write_report(
        reports_dir,
        "repo_sandbox_modern_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repo_sandbox",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repo_sandbox",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["gated_family_summaries"]["repo_sandbox"]["total"] == 2
    assert ledger["gated_family_summaries"]["repo_sandbox"]["hidden_side_effect_risk_count"] == 1
    assert ledger["counted_gated_family_summaries"]["repo_sandbox"]["total"] == 1
    assert ledger["counted_gated_family_summaries"]["repo_sandbox"]["hidden_side_effect_risk_count"] == 0


def test_build_unattended_trust_ledger_tracks_distinct_clean_success_task_roots(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repository_repo_sync_a.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"source_task": "repo_sync_matrix_task"},
    )
    _write_report(
        reports_dir,
        "repository_repo_sync_b.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"source_task": "repo_sync_matrix_task"},
    )
    _write_report(
        reports_dir,
        "repository_service_release.json",
        generated_at="2026-03-25T00:00:03+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"source_task": "service_release_task"},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    ledger = build_unattended_trust_ledger(config)

    assert ledger["family_summaries"]["repository"]["distinct_clean_success_task_roots"] == 2
    assert ledger["family_summaries"]["repository"]["clean_success_task_roots"] == [
        "repo_sync_matrix_task",
        "service_release_task",
    ]
    assert ledger["coverage_summary"]["required_family_clean_task_root_counts"]["repository"] == 2


def test_evaluate_unattended_trust_bootstraps_required_family_with_narrow_clean_task_roots(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for index in range(3):
        _write_report(
            reports_dir,
            f"repository_repo_sync_{index}.json",
            generated_at=f"2026-03-25T00:00:0{index}+00:00",
            benchmark_family="repository",
            outcome="success",
            success=True,
            hidden_side_effect_risk=False,
            task_metadata={"source_task": "repo_sync_matrix_task"},
        )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    evaluation = evaluate_unattended_trust(config, benchmark_family="repository")

    assert evaluation["passed"] is True
    assert evaluation["status"] == "bootstrap"
    assert "trust breadth is in bootstrap mode" in evaluation["detail"]
    assert evaluation["family_summary"]["distinct_clean_success_task_roots"] == 1


def test_evaluate_unattended_trust_trusts_required_family_after_clean_task_root_breadth(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repository_repo_sync.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"source_task": "repo_sync_matrix_task"},
    )
    _write_report(
        reports_dir,
        "repository_service_release.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="repository",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
        task_metadata={"source_task": "service_release_task"},
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_min_distinct_families=1,
    )

    evaluation = evaluate_unattended_trust(config, benchmark_family="repository")

    assert evaluation["passed"] is True
    assert evaluation["status"] == "trusted"
    assert evaluation["family_summary"]["distinct_clean_success_task_roots"] == 2


def test_evaluate_unattended_trust_exposes_family_posture_for_non_required_family(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_report(
        reports_dir,
        "repo_chore_success.json",
        generated_at="2026-03-25T00:00:01+00:00",
        benchmark_family="repo_chore",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    _write_report(
        reports_dir,
        "tooling_success.json",
        generated_at="2026-03-25T00:00:02+00:00",
        benchmark_family="tooling",
        outcome="success",
        success=True,
        hidden_side_effect_risk=False,
    )
    config = KernelConfig(
        run_reports_dir=reports_dir,
        unattended_trust_required_benchmark_families=("repo_chore",),
    )

    evaluation = evaluate_unattended_trust(config, benchmark_family="tooling")

    assert evaluation["required"] is False
    assert evaluation["status"] == "not_required"
    assert evaluation["family_posture"]["required"]["repo_chore"]["required"] is True
    assert evaluation["family_posture"]["observed"]["tooling"]["required"] is False
