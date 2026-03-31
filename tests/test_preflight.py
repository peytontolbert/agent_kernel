import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.preflight import (
    PreflightCheck,
    PreflightReport,
    WorkspaceFileSnapshot,
    build_unattended_task_report,
    capture_workspace_snapshot,
    classify_run_outcome,
    run_unattended_preflight,
    summarize_workspace_side_effects,
    write_unattended_task_report,
)
from agent_kernel.schemas import EpisodeRecord, StepRecord, TaskSpec


def test_run_unattended_preflight_passes_for_mock_with_verifier_contract(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    assert report.passed is True
    assert {check.name for check in report.checks} == {
        "provider_health",
        "tolbert_assets",
        "paper_research_assets",
        "verifier_inputs",
        "workspace_readiness",
        "execution_containment",
        "operator_policy",
        "trust_posture",
    }


def test_run_unattended_preflight_marks_missing_execution_containment_optional(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        sandbox_command_containment_mode="auto",
    )
    monkeypatch.setattr(
        "agent_kernel.preflight.sandbox_containment_status",
        lambda config, cwd=None: {
            "mode": "auto",
            "available": False,
            "detail": "bubblewrap unavailable for this runtime: user namespaces disabled",
        },
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    containment_check = next(check for check in report.checks if check.name == "execution_containment")
    assert report.passed is True
    assert containment_check.passed is False
    assert containment_check.severity == "optional"


def test_run_unattended_preflight_fails_when_required_execution_containment_missing(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        sandbox_command_containment_mode="required",
    )
    monkeypatch.setattr(
        "agent_kernel.preflight.sandbox_containment_status",
        lambda config, cwd=None: {
            "mode": "required",
            "available": False,
            "detail": "OS-level command containment tool not found: bwrap",
        },
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    containment_check = next(check for check in report.checks if check.name == "execution_containment")
    assert report.passed is False
    assert containment_check.passed is False
    assert containment_check.severity == "required"


def test_run_unattended_preflight_fails_without_verifier_inputs(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="underconstrained",
        prompt="do something",
        workspace_subdir="underconstrained",
        success_command="test -f ok.txt",
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    verifier_check = next(check for check in report.checks if check.name == "verifier_inputs")
    assert report.passed is False
    assert verifier_check.passed is False
    assert "success_command" in verifier_check.detail


def test_run_unattended_preflight_allows_external_manifest_family(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="external_task",
        prompt="create external.txt",
        workspace_subdir="external_task",
        expected_files=["external.txt"],
        metadata={
            "benchmark_family": "external_lab",
            "task_origin": "external_manifest",
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert policy_check.passed is True
    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=None,
        preflight=report,
    )
    assert payload["operator_policy"]["family_allowed"] is True
    assert payload["operator_policy"]["family_allowed_by_manifest"] is True


def test_run_unattended_preflight_accepts_retained_retrieval_bundle_runtime_paths(tmp_path):
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    runtime_dir = tmp_path / "tolbert"
    config_path = runtime_dir / "config.json"
    checkpoint_path = runtime_dir / "checkpoint.pt"
    nodes_path = runtime_dir / "nodes.jsonl"
    label_map_path = runtime_dir / "label_map.json"
    source_spans_path = runtime_dir / "source_spans.jsonl"
    cache_path = runtime_dir / "cache.pt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (config_path, nodes_path, label_map_path, source_spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "nodes_path": str(nodes_path),
                    "label_map_path": str(label_map_path),
                    "source_spans_paths": [str(source_spans_path)],
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_config_path="missing/config.json",
        tolbert_checkpoint_path="missing/checkpoint.pt",
        tolbert_nodes_path="missing/nodes.jsonl",
        tolbert_label_map_path="missing/label_map.json",
        tolbert_source_spans_paths=("missing/source_spans.jsonl",),
        tolbert_cache_paths=("missing/cache.pt",),
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    tolbert_check = next(check for check in report.checks if check.name == "tolbert_assets")
    assert tolbert_check.passed is True


def test_run_unattended_preflight_marks_missing_paper_research_assets_optional(monkeypatch, tmp_path):
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    runtime_dir = tmp_path / "tolbert"
    config_path = runtime_dir / "config.json"
    checkpoint_path = runtime_dir / "checkpoint.pt"
    nodes_path = runtime_dir / "nodes.jsonl"
    label_map_path = runtime_dir / "label_map.json"
    source_spans_path = runtime_dir / "source_spans.jsonl"
    cache_path = runtime_dir / "cache.pt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (config_path, nodes_path, label_map_path, source_spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "nodes_path": str(nodes_path),
                    "label_map_path": str(label_map_path),
                    "source_spans_paths": [str(source_spans_path)],
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_kernel.preflight._paper_research_runtime_paths", lambda config, repo_root: None)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        paper_research_query_mode="auto",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_config_path="missing/config.json",
        tolbert_checkpoint_path="missing/checkpoint.pt",
        tolbert_nodes_path="missing/nodes.jsonl",
        tolbert_label_map_path="missing/label_map.json",
        tolbert_source_spans_paths=("missing/source_spans.jsonl",),
        tolbert_cache_paths=("missing/cache.pt",),
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    paper_check = next(check for check in report.checks if check.name == "paper_research_assets")
    assert report.passed is True
    assert paper_check.passed is False
    assert paper_check.severity == "optional"


def test_run_unattended_preflight_fails_when_required_paper_research_assets_missing(monkeypatch, tmp_path):
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    runtime_dir = tmp_path / "tolbert"
    config_path = runtime_dir / "config.json"
    checkpoint_path = runtime_dir / "checkpoint.pt"
    nodes_path = runtime_dir / "nodes.jsonl"
    label_map_path = runtime_dir / "label_map.json"
    source_spans_path = runtime_dir / "source_spans.jsonl"
    cache_path = runtime_dir / "cache.pt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (config_path, nodes_path, label_map_path, source_spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "nodes_path": str(nodes_path),
                    "label_map_path": str(label_map_path),
                    "source_spans_paths": [str(source_spans_path)],
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_kernel.preflight._paper_research_runtime_paths", lambda config, repo_root: None)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        paper_research_query_mode="always",
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_config_path="missing/config.json",
        tolbert_checkpoint_path="missing/checkpoint.pt",
        tolbert_nodes_path="missing/nodes.jsonl",
        tolbert_label_map_path="missing/label_map.json",
        tolbert_source_spans_paths=("missing/source_spans.jsonl",),
        tolbert_cache_paths=("missing/cache.pt",),
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    paper_check = next(check for check in report.checks if check.name == "paper_research_assets")
    assert report.passed is False
    assert paper_check.passed is False
    assert paper_check.severity == "required"


def test_run_unattended_preflight_fails_for_workspace_escape(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="escape",
        prompt="do something",
        workspace_subdir="../escape",
        expected_files=["ok.txt"],
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    workspace_check = next(check for check in report.checks if check.name == "workspace_readiness")
    assert report.passed is False
    assert workspace_check.passed is False
    assert "must not escape workspace root" in workspace_check.detail


def test_unattended_task_report_includes_capability_usage():
    task = TaskSpec(
        task_id="http_fetch_task",
        prompt="fetch status",
        workspace_subdir="http_fetch_task",
        expected_files=["reports/status.txt"],
        metadata={
            "benchmark_family": "tooling",
            "workflow_guard": {
                "required_capabilities": ["github_read"],
                "requires_http": True,
            },
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace="workspace/http_fetch_task",
        success=True,
        steps=[
            StepRecord(
                index=0,
                thought="fetch",
                action="code_execute",
                content="http_request GET https://api.github.com/repos/openai/openai-python > reports/status.txt",
                selected_skill_id=None,
                command_result={
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "capabilities_used": ["workspace_exec", "workspace_fs", "http_request", "github_read"],
                },
                verification={"passed": True, "reasons": []},
            )
        ],
        task_metadata=dict(task.metadata),
        task_contract={},
    )
    config = KernelConfig(provider="mock", use_tolbert_context=False)

    report = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
    )

    assert report["capability_usage"]["required_capabilities"] == ["github_read", "http_request"]
    assert report["capability_usage"]["used_capabilities"] == [
        "github_read",
        "http_request",
        "workspace_exec",
        "workspace_fs",
    ]
    assert report["commands"][0]["capabilities_used"] == [
        "workspace_exec",
        "workspace_fs",
        "http_request",
        "github_read",
    ]


def test_unattended_task_report_includes_edit_synthesis():
    task = TaskSpec(
        task_id="synthetic_worker_task",
        prompt="apply synthesized worker edit",
        workspace_subdir="synthetic_worker_task",
        expected_files=["src/api_status.txt"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "synthetic_worker": True,
            "source_integrator_task_id": "heuristic_integrator",
            "synthetic_edit_plan": [
                {
                    "path": "src/api_status.txt",
                    "baseline_content": "API_STATUS=pending\n",
                    "target_content": "API_STATUS=ready\n",
                    "edit_kind": "token_replace",
                    "intent_source": "assigned_tests",
                    "edit_score": 27,
                    "replacements": [
                        {
                            "line_number": 1,
                            "before_fragment": "pending",
                            "after_fragment": "ready",
                            "before_line": "API_STATUS=pending",
                            "after_line": "API_STATUS=ready",
                        }
                    ],
                }
            ],
            "synthetic_edit_candidates": [
                {
                    "path": "src/api_status.txt",
                    "selected_kind": "token_replace",
                    "selected_score": 27,
                    "selected": {
                        "path": "src/api_status.txt",
                        "edit_kind": "token_replace",
                        "edit_score": 27,
                    },
                    "candidates": [
                        {"path": "src/api_status.txt", "edit_kind": "token_replace", "edit_score": 27},
                        {"path": "src/api_status.txt", "edit_kind": "line_replace", "edit_score": 64},
                        {"path": "src/api_status.txt", "edit_kind": "rewrite", "edit_score": 137},
                    ],
                }
            ],
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace="workspace/synthetic_worker_task",
        success=True,
        steps=[],
        task_metadata=dict(task.metadata),
        task_contract={
            "prompt": task.prompt,
            "workspace_subdir": task.workspace_subdir,
            "setup_commands": [],
            "success_command": "",
            "suggested_commands": [],
            "expected_files": ["src/api_status.txt"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 5,
            "metadata": dict(task.metadata),
            "synthetic_edit_plan": list(task.metadata["synthetic_edit_plan"]),
            "synthetic_edit_candidates": list(task.metadata["synthetic_edit_candidates"]),
        },
    )
    config = KernelConfig(provider="mock", use_tolbert_context=False)

    report = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
    )

    assert report["task_contract"]["synthetic_edit_plan"] == task.metadata["synthetic_edit_plan"]
    assert report["task_contract"]["synthetic_edit_candidates"] == task.metadata["synthetic_edit_candidates"]
    assert report["edit_synthesis"] == {
        "synthetic_worker": True,
        "source_integrator_task_id": "heuristic_integrator",
        "selected_edits": task.metadata["synthetic_edit_plan"],
        "candidate_sets": task.metadata["synthetic_edit_candidates"],
    }
    assert report["acceptance_packet"]["selected_edits"] == task.metadata["synthetic_edit_plan"]
    assert report["acceptance_packet"]["candidate_edit_sets"] == task.metadata["synthetic_edit_candidates"]
    assert report["acceptance_packet"]["verifier_result"]["passed"] is True


def test_unattended_task_report_includes_acceptance_packet_for_integrator():
    task = TaskSpec(
        task_id="git_parallel_merge_acceptance_task",
        prompt="accept worker branches",
        workspace_subdir="git_parallel_merge_acceptance_task",
        expected_files=["reports/merge_report.txt", "reports/test_report.txt"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "workflow_guard": {
                "shared_repo_id": "repo-a",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                "diff_base_ref": "origin/main",
                "test_commands": [
                    {"command": "tests/test_api.sh", "label": "api"},
                    {"command": "tests/test_docs.sh", "label": "docs"},
                ],
                "report_rules": [
                    {"path": "reports/merge_report.txt"},
                    {"path": "reports/test_report.txt"},
                ],
            },
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace="workspace/git_parallel_merge_acceptance_task",
        success=True,
        steps=[],
        task_metadata=dict(task.metadata),
        task_contract={
            "prompt": task.prompt,
            "workspace_subdir": task.workspace_subdir,
            "setup_commands": [],
            "success_command": "",
            "suggested_commands": [],
            "expected_files": list(task.expected_files),
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": {},
            "max_steps": 5,
            "metadata": dict(task.metadata),
        },
    )
    report = build_unattended_task_report(
        task=task,
        config=KernelConfig(provider="mock", use_tolbert_context=False),
        episode=episode,
        preflight=None,
    )

    assert report["acceptance_packet"]["target_branch"] == "main"
    assert report["acceptance_packet"]["required_merged_branches"] == [
        "worker/api-status",
        "worker/docs-status",
    ]
    assert report["acceptance_packet"]["tests"] == [
        {"command": "tests/test_api.sh", "label": "api"},
        {"command": "tests/test_docs.sh", "label": "docs"},
    ]


def test_preflight_fails_when_required_module_policy_scope_is_missing(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_repo_write"],
                        "settings": {
                            "repo_scopes": ["openai/other"],
                            "access_tier": "write",
                            "write_tier": "branch_only",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        capability_modules_path=modules_path,
    )
    task = TaskSpec(
        task_id="github_acceptance",
        prompt="open a PR branch",
        workspace_subdir="github_acceptance",
        expected_files=["reports/pr.txt"],
        metadata={
            "benchmark_family": "tooling",
            "workflow_guard": {
                "required_capabilities": ["github_repo_write"],
                "required_capability_policies": {
                    "github_repo_write": {
                        "repo_scopes": ["openai/agentkernel"],
                        "access_tier": "write",
                        "allow_write": True,
                        "write_tier": "branch_only",
                    }
                },
            },
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert report.passed is False
    assert "unsatisfied capability policy scope" in policy_check.detail


def test_preflight_fails_when_enabled_capability_module_is_invalid(tmp_path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "adapter_kind": "github",
                        "enabled": True,
                        "capabilities": ["github_repo_write"],
                        "settings": {
                            "read_only": True,
                            "write_tier": "branch_only",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        capability_modules_path=modules_path,
    )
    task = TaskSpec(
        task_id="github_invalid_module",
        prompt="inspect capability module",
        workspace_subdir="github_invalid_module",
        expected_files=["reports/module.txt"],
        metadata={
            "benchmark_family": "tooling",
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert report.passed is False
    assert "enabled capability modules invalid" in policy_check.detail
    assert "github_repo_write cannot be read_only" in policy_check.detail


def test_classify_run_outcome_marks_preflight_failure_as_safe_stop():
    preflight = PreflightReport(
        passed=False,
        checked_at="2026-03-24T00:00:00+00:00",
        checks=[PreflightCheck(name="provider_health", passed=False, detail="probe failed")],
    )

    outcome, reasons = classify_run_outcome(episode=None, preflight=preflight)

    assert outcome == "safe_stop"
    assert reasons == ["preflight_failed:provider_health"]


def test_run_unattended_preflight_fails_for_disallowed_benchmark_family(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allowed_benchmark_families=("micro",),
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repo_chore"},
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert report.passed is False
    assert policy_check.passed is False
    assert "not allowed for unattended execution" in policy_check.detail


def test_run_unattended_preflight_uses_retained_operator_policy(tmp_path):
    operator_policy_path = tmp_path / "operator_policy" / "operator_policy_proposals.json"
    operator_policy_path.parent.mkdir(parents=True, exist_ok=True)
    operator_policy_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "unattended_allowed_benchmark_families": ["micro", "repo_chore"],
                    "unattended_allow_git_commands": True,
                    "unattended_allow_http_requests": False,
                    "unattended_http_allowed_hosts": [],
                    "unattended_http_timeout_seconds": 10,
                    "unattended_http_max_body_bytes": 65536,
                    "unattended_allow_generated_path_mutations": False,
                    "unattended_generated_path_prefixes": ["build", "dist"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allowed_benchmark_families=("micro",),
        unattended_allow_git_commands=False,
        operator_policy_proposals_path=operator_policy_path,
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repo_chore", "workflow_guard": {"requires_git": True}},
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert policy_check.passed is True


def test_run_unattended_preflight_bootstraps_trust_gate_for_repo_family(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        run_reports_dir=tmp_path / "reports",
        unattended_trust_enforce=True,
        unattended_trust_bootstrap_min_reports=3,
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repo_chore"},
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    trust_check = next(check for check in report.checks if check.name == "trust_posture")
    assert report.passed is True
    assert trust_check.passed is True
    assert "bootstrap mode" in trust_check.detail


def test_run_unattended_preflight_fails_when_recent_trust_is_restricted(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    for index in range(5):
        reports_dir.joinpath(f"repo_chore_{index}.json").write_text(
            json.dumps(
                {
                    "report_kind": "unattended_task_report",
                    "generated_at": f"2026-03-25T00:00:0{index}+00:00",
                    "task_id": f"repo_task_{index}",
                    "benchmark_family": "repo_chore",
                    "outcome": "unsafe_ambiguous",
                    "success": False,
                    "side_effects": {"hidden_side_effect_risk": True},
                    "recovery": {"rollback_performed": True},
                }
            ),
            encoding="utf-8",
        )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        run_reports_dir=reports_dir,
        unattended_trust_enforce=True,
        unattended_trust_bootstrap_min_reports=3,
        unattended_trust_breadth_min_reports=3,
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repo_chore"},
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    trust_check = next(check for check in report.checks if check.name == "trust_posture")
    assert report.passed is False
    assert trust_check.passed is False
    assert "restricted unattended execution" in trust_check.detail


def test_classify_run_outcome_marks_nonzero_failures_as_safe_stop(tmp_path):
    episode = EpisodeRecord(
        task_id="fail",
        prompt="fail safely",
        workspace=str(tmp_path / "workspace" / "fail"),
        success=False,
        steps=[
            StepRecord(
                index=1,
                thought="run false",
                action="code_execute",
                content="false",
                selected_skill_id=None,
                command_result={"exit_code": 1, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["exit code was 1"]},
            )
        ],
        termination_reason="repeated_failed_action",
    )

    outcome, reasons = classify_run_outcome(episode=episode, preflight=None)

    assert outcome == "safe_stop"
    assert reasons == ["repeated_failed_action"]


def test_classify_run_outcome_marks_zero_exit_without_success_as_unsafe_ambiguous(tmp_path):
    episode = EpisodeRecord(
        task_id="ambiguous",
        prompt="write something incomplete",
        workspace=str(tmp_path / "workspace" / "ambiguous"),
        success=False,
        steps=[
            StepRecord(
                index=1,
                thought="write partial file",
                action="code_execute",
                content="printf 'draft\\n' > draft.txt",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["missing expected file: final.txt"]},
            )
        ],
        termination_reason="max_steps_reached",
    )

    outcome, reasons = classify_run_outcome(episode=episode, preflight=None)

    assert outcome == "unsafe_ambiguous"
    assert reasons == ["max_steps_reached"]


def test_write_unattended_task_report_persists_outcome_and_workspace_listing(tmp_path):
    workspace = tmp_path / "workspace" / "hello_task"
    workspace.mkdir(parents=True)
    (workspace / "hello.txt").write_text("hello\n", encoding="utf-8")
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        run_reports_dir=tmp_path / "reports",
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )
    episode = EpisodeRecord(
        task_id="hello_task",
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        steps=[
            StepRecord(
                index=1,
                thought="write file",
                action="code_execute",
                content="printf 'hello\\n' > hello.txt",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": True, "reasons": ["verification passed"]},
            )
        ],
        termination_reason="success",
    )
    preflight = PreflightReport(
        passed=True,
        checked_at="2026-03-24T00:00:00+00:00",
        checks=[PreflightCheck(name="provider_health", passed=True, detail="ok")],
    )

    report_path = write_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=preflight,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["report_kind"] == "unattended_task_report"
    assert payload["outcome"] == "success"
    assert payload["benchmark_family"] == "bounded"
    assert payload["operator_policy"]["family_allowed"] is True
    assert payload["summary"]["workspace_files"] == 1
    assert payload["summary"]["created_files"] == 1
    assert payload["summary"]["modified_files"] == 0
    assert payload["summary"]["deleted_files"] == 0
    assert payload["summary"]["accounted_change_files"] == 1
    assert payload["summary"]["unexpected_change_files"] == 0
    assert payload["summary"]["hidden_side_effect_risk"] is False
    assert payload["workspace_files"] == ["hello.txt"]
    assert payload["commands"][0]["verification_passed"] is True
    assert payload["side_effects"]["created_files"] == ["hello.txt"]
    assert payload["side_effects"]["accounted_change_files"] == ["hello.txt"]
    assert payload["side_effects"]["all_changes_accounted_for"] is True
    assert payload["side_effects"]["unexpected_change_files"] == []
    assert payload["side_effects"]["changed_file_details"][0]["contract_status"] == "managed_create"
    assert payload["side_effects"]["changed_file_details"][0]["accounted_for"] is True
    assert payload["uncertainties"] == []


def test_build_unattended_task_report_uses_preflight_failure_without_episode(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )
    preflight = PreflightReport(
        passed=False,
        checked_at="2026-03-24T00:00:00+00:00",
        checks=[PreflightCheck(name="provider_health", passed=False, detail="probe failed")],
    )

    payload = build_unattended_task_report(task=task, config=config, episode=None, preflight=preflight)

    assert payload["outcome"] == "safe_stop"
    assert payload["termination_reason"] == "preflight_failed"
    assert payload["commands"] == []


def test_capture_workspace_snapshot_and_side_effect_summary_detect_changes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "delete.txt").write_text("stale\n", encoding="utf-8")
    before = capture_workspace_snapshot(workspace)

    (workspace / "keep.txt").write_text("beta\n", encoding="utf-8")
    (workspace / "delete.txt").unlink()
    (workspace / "create.txt").write_text("new\n", encoding="utf-8")
    after = capture_workspace_snapshot(workspace)
    side_effects = summarize_workspace_side_effects(
        before_snapshot=before,
        after_snapshot=after,
        clean_workspace=False,
    )

    assert side_effects["created_files"] == ["create.txt"]
    assert side_effects["modified_files"] == ["keep.txt"]
    assert side_effects["deleted_files"] == ["delete.txt"]
    assert side_effects["changed_files_total"] == 3
    assert side_effects["workspace_reset_detected"] is False


def test_build_unattended_task_report_records_side_effect_deletions_and_uncertainty(tmp_path):
    workspace = tmp_path / "workspace" / "ambiguous"
    workspace.mkdir(parents=True)
    (workspace / "keep.txt").write_text("after\n", encoding="utf-8")
    before_snapshot = {
        "keep.txt": capture_workspace_snapshot(workspace)["keep.txt"],
        "old.txt": WorkspaceFileSnapshot(
            relative_path="old.txt",
            size_bytes=5,
            sha256="old",
        ),
    }
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="ambiguous",
        prompt="partial change",
        workspace_subdir="ambiguous",
        expected_files=["final.txt"],
    )
    episode = EpisodeRecord(
        task_id="ambiguous",
        prompt=task.prompt,
        workspace=str(workspace),
        success=False,
        steps=[
            StepRecord(
                index=1,
                thought="overwrite keep",
                action="code_execute",
                content="printf 'after\\n' > keep.txt && rm -f old.txt",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["missing expected file: final.txt"]},
            )
        ],
        termination_reason="max_steps_reached",
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot=before_snapshot,
    )

    assert payload["outcome"] == "unsafe_ambiguous"
    assert payload["side_effects"]["deleted_files"] == ["old.txt"]
    assert payload["side_effects"]["accounted_change_files"] == []
    assert payload["side_effects"]["all_changes_accounted_for"] is False
    assert payload["side_effects"]["unexpected_deleted_files"] == ["old.txt"]
    assert payload["side_effects"]["hidden_side_effect_risk"] is True
    assert payload["side_effects"]["changed_file_details"][0]["contract_status"] == "unexpected_delete"
    assert payload["side_effects"]["changed_file_details"][0]["accounted_for"] is False
    assert "inspect side effects manually" in payload["uncertainties"][0]
    assert any("workspace deletions were observed" in note for note in payload["uncertainties"])
    assert any("unexpected workspace changes" in note for note in payload["uncertainties"])
