import json
from pathlib import Path

import agent_kernel.preflight as preflight
import agent_kernel.syntax_motor as syntax_motor
import pytest
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
        "syntax_preflight",
        "execution_containment",
        "operator_policy",
        "trust_posture",
    }


def test_run_unattended_preflight_rejects_tolbert_provider_alias(tmp_path):
    config = KernelConfig(
        provider="tolbert",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    with pytest.raises(ValueError, match="unsupported provider"):
        run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])


def test_run_unattended_preflight_fails_when_tolbert_runtime_modules_are_missing(monkeypatch, tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_python_bin="/bin/sh",
        tolbert_config_path=str(tmp_path / "tolbert" / "config.json"),
        tolbert_checkpoint_path=str(tmp_path / "tolbert" / "checkpoint.pt"),
        tolbert_nodes_path=str(tmp_path / "tolbert" / "nodes.jsonl"),
        tolbert_label_map_path=str(tmp_path / "tolbert" / "label_map.json"),
        tolbert_source_spans_paths=(str(tmp_path / "tolbert" / "source_spans.jsonl"),),
        tolbert_cache_paths=(str(tmp_path / "tolbert" / "cache.pt"),),
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    for path in (
        Path(config.tolbert_config_path),
        Path(config.tolbert_checkpoint_path),
        Path(config.tolbert_nodes_path),
        Path(config.tolbert_label_map_path),
        Path(config.tolbert_source_spans_paths[0]),
        Path(config.tolbert_cache_paths[0]),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    task = TaskSpec(
        task_id="hello_task",
        prompt="create hello.txt",
        workspace_subdir="hello_task",
        expected_files=["hello.txt"],
    )

    monkeypatch.setattr(preflight, "_tolbert_runtime_import_check", lambda repo_root: PreflightCheck("tolbert_assets", False, "missing tolbert runtime modules: tolbert_brain.brain"))
    monkeypatch.setattr(preflight, "_paper_research_assets_check", lambda config, repo_root: PreflightCheck("paper_research_assets", True, "ok", severity="optional"))

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    tolbert_check = next(check for check in report.checks if check.name == "tolbert_assets")
    assert tolbert_check.passed is False
    assert "runtime modules" in tolbert_check.detail


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


def test_run_unattended_preflight_includes_python_syntax_preflight(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "syntax_worker"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "service.py").write_text(
        "import os\n\n\ndef apply_status(value):\n    return normalize(value)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="syntax_worker",
        prompt="update service.py",
        workspace_subdir="syntax_worker",
        expected_files=["service.py"],
        metadata={
            "benchmark_family": "repository",
            "synthetic_edit_plan": [
                {
                    "path": "service.py",
                    "edit_kind": "line_replace",
                    "replacements": [
                        {
                            "line_number": 4,
                            "before_line": "def apply_status(value):",
                            "after_line": "def apply_status(value, *, strict=False):",
                        }
                    ],
                }
            ],
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    syntax_check = next(check for check in report.checks if check.name == "syntax_preflight")
    assert syntax_check.passed is True
    assert syntax_check.severity == "required"
    assert "targeted_symbols=1" in syntax_check.detail


def test_unattended_task_report_includes_syntax_motor_summary_for_python_edits(tmp_path):
    workspace = tmp_path / "workspace" / "syntax_task"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "service.py").write_text(
        "import os\n\n\ndef apply_status(value):\n    return normalize(value)\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="syntax_task",
        prompt="update service.py",
        workspace_subdir="syntax_task",
        expected_files=["service.py"],
        expected_file_contents={
            "service.py": "import os\n\n\ndef apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n",
        },
        metadata={
            "benchmark_family": "repository",
            "synthetic_edit_plan": [
                {
                    "path": "service.py",
                    "edit_kind": "line_replace",
                    "baseline_content": "import os\n\n\ndef apply_status(value):\n    return normalize(value)\n",
                    "target_content": "import os\n\n\ndef apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n",
                    "replacements": [
                        {
                            "line_number": 4,
                            "before_line": "def apply_status(value):",
                            "after_line": "def apply_status(value, *, strict=False):",
                        }
                    ],
                }
            ],
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        steps=[],
        task_metadata=dict(task.metadata),
        task_contract={
            "prompt": task.prompt,
            "workspace_subdir": task.workspace_subdir,
            "setup_commands": [],
            "success_command": "",
            "suggested_commands": [],
            "expected_files": ["service.py"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": dict(task.expected_file_contents),
            "max_steps": 5,
            "metadata": dict(task.metadata),
            "synthetic_edit_plan": list(task.metadata["synthetic_edit_plan"]),
            "synthetic_edit_candidates": [],
        },
    )
    config = KernelConfig(provider="mock", use_tolbert_context=False)

    report = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
    )

    syntax_motor = report["syntax_motor"]
    assert syntax_motor["parser"] in {"python_ast", "repository_library_python_ast"}
    assert syntax_motor["syntax_preflight_passed"] is True
    assert syntax_motor["targeted_symbol_count"] == 1
    file_summary = syntax_motor["files"][0]
    assert file_summary["path"] == "service.py"
    assert file_summary["edited_region"] == {"start_line": 4, "end_line": 4}
    assert file_summary["baseline_target_symbol"]["name"] == "apply_status"
    assert file_summary["baseline_target_symbol"]["signature"] == "apply_status(value)"
    assert file_summary["target_target_symbol"]["signature"] == "apply_status(value, *, strict)"
    assert file_summary["edited_symbol_fqn"].endswith("apply_status")
    assert file_summary["call_targets_before"] == ["normalize"]
    assert file_summary["call_targets_after"] == ["persist", "normalize"]
    assert file_summary["signature_changed"] is True
    assert file_summary["signature_change_risk"] is True
    assert file_summary["imports_before"] == ["os"]
    assert file_summary["imports_added"] == []
    assert file_summary["imports_removed"] == []
    assert file_summary["module_symbols_before"][0]["name"] == "apply_status"
    assert file_summary["module_symbols_after"][0]["name"] == "apply_status"


def test_unattended_task_report_tracks_import_delta_in_syntax_motor_summary(tmp_path):
    workspace = tmp_path / "workspace" / "syntax_imports"
    workspace.mkdir(parents=True, exist_ok=True)
    baseline = "import os\n\n\ndef load_env(value):\n    return value.strip()\n"
    target = "import os\nimport json\n\n\ndef load_env(value):\n    return json.dumps(value.strip())\n"
    (workspace / "service.py").write_text(baseline, encoding="utf-8")
    task = TaskSpec(
        task_id="syntax_imports",
        prompt="update service imports",
        workspace_subdir="syntax_imports",
        expected_files=["service.py"],
        expected_file_contents={"service.py": target},
        metadata={
            "benchmark_family": "repository",
            "synthetic_edit_plan": [
                {
                    "path": "service.py",
                    "edit_kind": "rewrite",
                    "baseline_content": baseline,
                    "target_content": target,
                }
            ],
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        steps=[],
        task_metadata=dict(task.metadata),
        task_contract={
            "prompt": task.prompt,
            "workspace_subdir": task.workspace_subdir,
            "setup_commands": [],
            "success_command": "",
            "suggested_commands": [],
            "expected_files": ["service.py"],
            "expected_output_substrings": [],
            "forbidden_files": [],
            "forbidden_output_substrings": [],
            "expected_file_contents": dict(task.expected_file_contents),
            "max_steps": 5,
            "metadata": dict(task.metadata),
            "synthetic_edit_plan": list(task.metadata["synthetic_edit_plan"]),
            "synthetic_edit_candidates": [],
        },
    )
    config = KernelConfig(provider="mock", use_tolbert_context=False)

    report = build_unattended_task_report(task=task, config=config, episode=episode, preflight=None)

    file_summary = report["syntax_motor"]["files"][0]
    assert file_summary["imports_before"] == ["os"]
    assert file_summary["imports_after"] == ["os", "json"]
    assert file_summary["imports_added"] == ["json"]
    assert file_summary["imports_removed"] == []
    assert file_summary["module_symbols_after"][0]["name"] == "load_env"


def test_syntax_motor_prefers_repository_library_style_visitor(monkeypatch):
    class DonorSymbol:
        def __init__(self, name: str, qualname: str, signature: str, line: int, end_line: int) -> None:
            self.kind = "function"
            self.name = name
            self.qualname = qualname
            self.signature = signature
            self.returns = "str"
            self.line = line
            self.end_line = end_line

    class DonorVisitor:
        def __init__(self, module: str, path: str) -> None:
            del module, path
            self.imports = {"json": "json"}
            self.import_modules = ["json"]
            self.star_imports = ["pkg.shared"]
            self.exports = ["load_env"]
            self.symbols = []

        def visit(self, tree) -> None:
            del tree
            self.symbols = [DonorSymbol("load_env", "load_env", "(value:str)", 4, 5)]

    monkeypatch.setattr(
        syntax_motor,
        "_repository_library_module_visitor_class",
        lambda: (DonorVisitor, "repository_library_python_ast"),
    )

    parsed = syntax_motor._parse_python_source(
        "import json\n\n\ndef load_env(value):\n    return value\n",
        label="service.py",
    )

    assert parsed["parser"] == "repository_library_python_ast"
    assert parsed["imports"] == ["json"]
    assert parsed["star_imports"] == ["pkg.shared"]
    assert parsed["exports"] == ["load_env"]
    assert parsed["provider"] == "repository_library"
    assert parsed["symbols"][0]["name"] == "load_env"
    assert parsed["symbols"][0]["signature"] == "load_env(value:str)"
    assert parsed["symbols"][0]["returns"] == "str"


def test_syntax_motor_step_summary_tracks_edited_symbol_fqn_and_calls():
    summary = syntax_motor.summarize_python_edit_step(
        {
            "path": "service.py",
            "edit_kind": "line_replace",
            "baseline_content": "def apply_status(value):\n    return normalize(value)\n",
            "target_content": "def apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n",
            "replacements": [
                {
                    "line_number": 1,
                    "before_line": "def apply_status(value):",
                    "after_line": "def apply_status(value, *, strict=False):",
                }
            ],
        }
    )

    assert summary is not None
    assert summary["edited_symbol_fqn"].endswith("apply_status")
    assert summary["call_targets_before"] == ["normalize"]
    assert summary["call_targets_after"] == ["persist", "normalize"]
    assert summary["syntax_ok"] is True


def test_syntax_motor_step_summary_includes_repository_context(monkeypatch, tmp_path):
    class FakeCodeGraph:
        def owners_of(self, symbol: str) -> list[str]:
            assert symbol == "apply_status"
            return ["service.py"]

        def who_calls(self, fqn: str) -> list[str]:
            assert fqn.endswith("apply_status")
            return ["client.sync_status"]

        def calls_of(self, fqn: str) -> list[str]:
            assert fqn.endswith("apply_status")
            return ["persist", "normalize"]

    monkeypatch.setattr(
        syntax_motor,
        "_repository_library_code_graph_for_root",
        lambda workspace: FakeCodeGraph(),
    )

    summary = syntax_motor.summarize_python_edit_step(
        {
            "path": "service.py",
            "edit_kind": "line_replace",
            "baseline_content": "def apply_status(value):\n    return normalize(value)\n",
            "target_content": "def apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n",
            "replacements": [
                {
                    "line_number": 1,
                    "before_line": "def apply_status(value):",
                    "after_line": "def apply_status(value, *, strict=False):",
                }
            ],
        },
        workspace=tmp_path,
    )

    assert summary is not None
    repository_context = summary["repository_context"]
    assert repository_context["provider"] == "repository_library_code_graph"
    assert repository_context["owner_files"] == ["service.py"]
    assert repository_context["owner_matches_path"] is True
    assert repository_context["repo_callers"] == ["client.sync_status"]
    assert repository_context["repo_callees"] == ["persist", "normalize"]


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


def test_run_unattended_preflight_surfaces_repository_task_root_breadth_detail(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    reports_dir.joinpath("repository_clean.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "generated_at": "2026-04-03T00:00:00+00:00",
                "task_id": "repository_adjacent_task",
                "benchmark_family": "repository",
                "outcome": "success",
                "success": True,
                "trust_scope": "gated",
                "task_metadata": {"source_task": "repo_sync_matrix_task"},
                "summary": {"unexpected_change_files": 0, "command_steps": 1},
                "commands": [{}],
                "side_effects": {"hidden_side_effect_risk": False},
                "recovery": {"rollback_performed": False},
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
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_min_distinct_families=1,
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repository"},
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    trust_check = next(check for check in report.checks if check.name == "trust_posture")
    assert report.passed is True
    assert trust_check.passed is True
    assert "repository clean task-root breadth: observed=1/2 remaining=1" in trust_check.detail
    assert "roots=[repo_sync_matrix_task]" in trust_check.detail


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
                    "supervision": {
                        "mode": "unattended",
                        "operator_turns": 0,
                        "independent_execution": True,
                    },
                    "summary": {"command_steps": 1},
                    "commands": [{}],
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


def test_write_unattended_task_report_skips_storage_governance(tmp_path, monkeypatch):
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

    seen: dict[str, object] = {}

    def capture(path, payload, **kwargs):
        seen["path"] = path
        seen["payload"] = payload
        seen["kwargs"] = kwargs
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(preflight, "atomic_write_json", capture)

    report_path = write_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "unattended_task_report"
    assert seen["path"] == report_path
    assert seen["kwargs"] == {"config": config, "govern_storage": False}


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


def test_build_unattended_task_report_surfaces_light_supervision_summary(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "repo_sandbox_task"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "reports").mkdir()
    (workspace / "reports" / "status.txt").write_text("ready\n", encoding="utf-8")
    task = TaskSpec(
        task_id="repo_sandbox_task",
        prompt="materialize repo sandbox report",
        workspace_subdir="repo_sandbox_task",
        expected_files=["reports/status.txt"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "supervision_mode": "light_supervision",
            "operator_turns": 1,
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        termination_reason="success",
        steps=[
            StepRecord(
                index=0,
                thought="write report",
                action="code_execute",
                content="printf 'ready\\n' > reports/status.txt",
                selected_skill_id=None,
                command_result={"command": "printf 'ready\\n' > reports/status.txt", "exit_code": 0},
                verification={"passed": True, "reasons": ["ok"]},
            )
        ],
        task_metadata=dict(task.metadata),
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot={},
    )

    assert payload["supervision"] == {
        "mode": "light_supervision",
        "operator_turns": 1,
        "verifier_defined": True,
        "independent_execution": True,
        "preflight_passed": True,
        "command_steps": 1,
        "changed_files": 1,
        "hidden_side_effect_risk": False,
        "light_supervision_candidate": True,
        "light_supervision_success": True,
        "light_supervision_clean_success": True,
        "contract_clean_failure_recovery_origin": False,
        "contract_clean_failure_recovery_step_floor": 0,
        "contract_clean_failure_recovery_candidate": False,
        "contract_clean_failure_recovery_success": False,
        "contract_clean_failure_recovery_clean_success": False,
    }


def test_build_unattended_task_report_infers_guided_mode_from_operator_turns(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    task = TaskSpec(
        task_id="guided_task",
        prompt="refine guided patch",
        workspace_subdir="guided_task",
        expected_files=["patch.txt"],
        metadata={"operator_turns": 3},
    )

    payload = build_unattended_task_report(task=task, config=config, episode=None, preflight=None)

    assert payload["supervision"]["mode"] == "guided"
    assert payload["supervision"]["operator_turns"] == 3
    assert payload["supervision"]["light_supervision_candidate"] is False
    assert payload["supervision"]["independent_execution"] is False


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


def test_summarize_workspace_side_effects_ignores_git_internal_changes_for_git_tasks():
    task = TaskSpec(
        task_id="git_task",
        prompt="update repo state",
        workspace_subdir="repo",
        expected_files=["src/status.txt"],
        metadata={"requires_git": True},
    )
    side_effects = summarize_workspace_side_effects(
        before_snapshot={},
        after_snapshot={
            ".git/HEAD": WorkspaceFileSnapshot(relative_path=".git/HEAD", size_bytes=20, sha256="git-head"),
            "src/status.txt": WorkspaceFileSnapshot(
                relative_path="src/status.txt",
                size_bytes=6,
                sha256="status",
            ),
        },
        clean_workspace=True,
        task=task,
    )

    assert side_effects["created_files"] == [".git/HEAD", "src/status.txt"]
    assert side_effects["unexpected_change_files"] == []
    assert side_effects["hidden_side_effect_risk"] is False


def test_summarize_workspace_side_effects_keeps_non_git_unexpected_changes_for_git_tasks():
    task = TaskSpec(
        task_id="git_task",
        prompt="update repo state",
        workspace_subdir="repo",
        expected_files=["src/status.txt"],
        metadata={"requires_git": True},
    )
    side_effects = summarize_workspace_side_effects(
        before_snapshot={},
        after_snapshot={
            ".git/HEAD": WorkspaceFileSnapshot(relative_path=".git/HEAD", size_bytes=20, sha256="git-head"),
            "notes.txt": WorkspaceFileSnapshot(relative_path="notes.txt", size_bytes=5, sha256="notes"),
            "src/status.txt": WorkspaceFileSnapshot(
                relative_path="src/status.txt",
                size_bytes=6,
                sha256="status",
            ),
        },
        clean_workspace=True,
        task=task,
    )

    assert side_effects["unexpected_change_files"] == ["notes.txt"]
    assert side_effects["hidden_side_effect_risk"] is True


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


def test_build_unattended_task_report_serializes_command_route_metadata(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "shadow_report_task"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "deploy.txt").write_text("ready\n", encoding="utf-8")
    task = TaskSpec(
        task_id="shadow_report_task",
        prompt="write deploy.txt",
        workspace_subdir="shadow_report_task",
        expected_files=["deploy.txt"],
        expected_file_contents={"deploy.txt": "ready\n"},
        metadata={"benchmark_family": "project"},
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        termination_reason="success",
        steps=[
            StepRecord(
                index=0,
                thought="fallback",
                action="code_execute",
                content="printf 'ready\\n' > deploy.txt",
                selected_skill_id=None,
                command_result={"command": "printf 'ready\\n' > deploy.txt", "exit_code": 0},
                verification={"passed": True, "reasons": ["ok"]},
                decision_source="deterministic_fallback",
                tolbert_route_mode="shadow",
            )
        ],
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot={},
    )

    assert payload["commands"][0]["decision_source"] == "deterministic_fallback"
    assert payload["commands"][0]["tolbert_route_mode"] == "shadow"


def test_build_unattended_task_report_marks_nonacting_retrieval_companion_as_coverage_only(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "repo_sync_matrix_retrieval_task"
    workspace.mkdir(parents=True, exist_ok=True)
    task = TaskSpec(
        task_id="repo_sync_matrix_retrieval_task",
        prompt="inspect repository context",
        workspace_subdir="repo_sync_matrix_retrieval_task",
        metadata={
            "benchmark_family": "repository",
            "requires_retrieval": True,
            "source_task": "repo_sync_matrix_task",
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=False,
        termination_reason="policy_terminated",
        steps=[],
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot={},
    )

    assert payload["outcome"] == "safe_stop"
    assert payload["trust_scope"] == "coverage_only"


def test_build_unattended_task_report_marks_nonexecuting_safe_stop_with_accounted_setup_changes_as_coverage_only(
    tmp_path,
):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "integration_failover_drill_task"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "docs").mkdir()
    (workspace / "docs" / "runbook.md").write_text("seed\n", encoding="utf-8")
    task = TaskSpec(
        task_id="integration_failover_drill_task",
        prompt="prepare the integration drill workspace",
        workspace_subdir="integration_failover_drill_task",
        expected_files=["docs/runbook.md"],
        metadata={"benchmark_family": "integration"},
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=False,
        termination_reason="policy_terminated",
        steps=[],
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot={},
    )

    assert payload["outcome"] == "safe_stop"
    assert payload["side_effects"]["changed_files_total"] == 1
    assert payload["side_effects"]["unexpected_change_files"] == []
    assert payload["trust_scope"] == "coverage_only"


def test_build_unattended_task_report_surfaces_repository_trust_breadth_summary(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    reports_dir.joinpath("repository_clean.json").write_text(
        json.dumps(
            {
                "report_kind": "unattended_task_report",
                "generated_at": "2026-04-03T00:00:00+00:00",
                "task_id": "repository_adjacent_task",
                "benchmark_family": "repository",
                "outcome": "success",
                "success": True,
                "trust_scope": "gated",
                "task_metadata": {"source_task": "repo_sync_matrix_task"},
                "summary": {"unexpected_change_files": 0, "command_steps": 1},
                "commands": [{}],
                "side_effects": {"hidden_side_effect_risk": False},
                "recovery": {"rollback_performed": False},
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
        unattended_trust_bootstrap_min_reports=1,
        unattended_trust_breadth_min_reports=1,
        unattended_trust_required_benchmark_families=("repository",),
        unattended_trust_min_distinct_families=1,
    )
    task = TaskSpec(
        task_id="repo_patch_review_task",
        prompt="prepare repo review packet",
        workspace_subdir="repo_patch_review_task",
        expected_files=["reports/diff_summary.txt"],
        metadata={"benchmark_family": "repository"},
    )

    payload = build_unattended_task_report(task=task, config=config, episode=None, preflight=None)

    assert payload["trust"]["status"] == "bootstrap"
    assert payload["trust_breadth_summary"] == {
        "benchmark_family": "repository",
        "required": True,
        "status": "bootstrap",
        "reports": 1,
        "bootstrap_min_reports": 1,
        "breadth_min_reports": 1,
        "min_distinct_task_roots": 2,
        "distinct_clean_success_task_roots": 1,
        "clean_success_task_roots": ["repo_sync_matrix_task"],
        "remaining_distinct_task_roots": 1,
        "required_family_clean_task_root_count": 1,
        "light_supervision_reports": 0,
        "light_supervision_successes": 0,
        "light_supervision_clean_successes": 0,
        "contract_clean_failure_recovery_reports": 0,
        "contract_clean_failure_recovery_successes": 0,
        "contract_clean_failure_recovery_clean_successes": 0,
    }


def test_build_unattended_task_report_surfaces_contract_clean_failure_recovery_summary(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    workspace = config.workspace_root / "repository_failure_recovery_task"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "reports").mkdir()
    (workspace / "reports" / "status.txt").write_text("recovered\n", encoding="utf-8")
    task = TaskSpec(
        task_id="repository_failure_recovery_task",
        prompt="repair repository report",
        workspace_subdir="repository_failure_recovery_task",
        expected_files=["reports/status.txt"],
        metadata={
            "benchmark_family": "repository",
            "curriculum_kind": "failure_recovery",
            "contract_clean_failure_recovery_origin": True,
            "contract_clean_failure_recovery_step_floor": 12,
        },
    )
    episode = EpisodeRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        workspace=str(workspace),
        success=True,
        termination_reason="success",
        steps=[
            StepRecord(
                index=0,
                thought="repair report",
                action="code_execute",
                content="printf 'recovered\\n' > reports/status.txt",
                selected_skill_id=None,
                command_result={"command": "printf 'recovered\\n' > reports/status.txt", "exit_code": 0},
                verification={"passed": True, "reasons": ["ok"]},
            )
        ],
        task_metadata=dict(task.metadata),
    )

    payload = build_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=None,
        before_workspace_snapshot={},
    )

    assert payload["supervision"]["contract_clean_failure_recovery_origin"] is True
    assert payload["supervision"]["contract_clean_failure_recovery_step_floor"] == 12
    assert payload["supervision"]["contract_clean_failure_recovery_candidate"] is True
    assert payload["supervision"]["contract_clean_failure_recovery_success"] is True
    assert payload["supervision"]["contract_clean_failure_recovery_clean_success"] is True
