import json
from pathlib import Path

from agent_kernel.extensions.capabilities import (
    capability_enabled,
    capability_policy_satisfies,
    capability_registry_snapshot,
    capability_usage_summary,
    command_capabilities,
    declared_task_capabilities,
    enabled_corpus_sources,
    effective_http_allowed_hosts,
    load_capability_modules,
)
from agent_kernel.extensions.improvement.capability_improvement import build_capability_module_artifact, capability_surface_summary
from agent_kernel.config import KernelConfig
from agent_kernel.ops.preflight import run_unattended_preflight
from agent_kernel.schemas import StepRecord, TaskSpec


def test_capability_registry_exposes_builtin_and_module_capabilities(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read", "github_repo_write"],
                        "settings": {
                            "allowed_orgs": ["openai"],
                            "repo_scopes": ["openai/agentkernel"],
                        },
                    },
                    {
                        "module_id": "youtube",
                        "enabled": False,
                        "capabilities": ["youtube_read"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_allow_http_requests=True,
        unattended_http_allowed_hosts=("api.github.com",),
    )

    registry = capability_registry_snapshot(config)

    assert "workspace_fs" in registry["enabled_builtin_capabilities"]
    assert "http_request" in registry["enabled_builtin_capabilities"]
    assert "github_read" in registry["enabled_external_capabilities"]
    assert "github_repo_write" in registry["enabled_capabilities"]
    assert registry["builtin"]["http_request"]["scope"]["allowed_hosts"] == [
        "api.github.com",
        "uploads.github.com",
    ]
    assert capability_enabled(config, "github_read") is True
    assert capability_enabled(config, "youtube_read") is False


def test_preflight_fails_when_required_module_capability_missing(tmp_path: Path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        capability_modules_path=tmp_path / "config" / "capabilities.json",
    )
    task = TaskSpec(
        task_id="github_issue_triage",
        prompt="triage issue",
        workspace_subdir="github_issue_triage",
        expected_files=["reports/issue.txt"],
        metadata={
            "benchmark_family": "tooling",
            "workflow_guard": {
                "required_capabilities": ["github_read"],
            },
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert report.passed is False
    assert "missing capabilities" in policy_check.detail


def test_preflight_accepts_enabled_module_capability(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
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
        task_id="github_issue_triage",
        prompt="triage issue",
        workspace_subdir="github_issue_triage",
        expected_files=["reports/issue.txt"],
        metadata={
            "benchmark_family": "tooling",
            "workflow_guard": {
                "required_capabilities": ["github_read"],
            },
        },
    )

    report = run_unattended_preflight(config, task, repo_root=Path(__file__).resolve().parents[1])

    policy_check = next(check for check in report.checks if check.name == "operator_policy")
    assert policy_check.passed is True


def test_load_capability_modules_only_activates_retained_wrapped_artifacts(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "artifact_kind": "capability_module_set",
                "spec_version": "asi_v1",
                "lifecycle_state": "proposed",
                "retention_gate": {"min_pass_rate_delta_abs": 0.01},
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_capability_modules(modules_path) == []

    modules_path.write_text(
        json.dumps(
            {
                "artifact_kind": "capability_module_set",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_gate": {"min_pass_rate_delta_abs": 0.01},
                "retention_decision": {"state": "retain"},
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    modules = load_capability_modules(modules_path)

    assert [module["module_id"] for module in modules] == ["github"]


def test_declared_task_capabilities_includes_builtin_policy_requirements():
    task = TaskSpec(
        task_id="repo_fetch",
        prompt="fetch release notes",
        workspace_subdir="repo_fetch",
        expected_files=["reports/release.txt"],
        metadata={
            "workflow_guard": {
                "required_capabilities": ["github_read"],
                "requires_http": True,
                "requires_git": True,
                "touches_generated_paths": True,
            }
        },
    )

    assert declared_task_capabilities(task) == [
        "generated_path_mutation",
        "git_commands",
        "github_read",
        "http_request",
    ]


def test_command_capabilities_detects_http_and_workspace_redirection():
    capabilities = command_capabilities("http_request GET https://example.com/data > reports/data.txt")

    assert capabilities == ["http_request", "workspace_exec", "workspace_fs"]


def test_effective_http_allowed_hosts_include_enabled_module_scope(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                        "settings": {"http_allowed_hosts": ["api.github.com", "uploads.github.com"]},
                    },
                    {
                        "module_id": "youtube",
                        "enabled": False,
                        "capabilities": ["youtube_read"],
                        "settings": {"http_allowed_hosts": ["www.googleapis.com"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_http_allowed_hosts=("example.com",),
    )

    assert effective_http_allowed_hosts(config) == [
        "api.github.com",
        "example.com",
        "uploads.github.com",
    ]


def test_enabled_corpus_sources_derive_github_repo_endpoints(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                        "settings": {
                            "repo_scopes": ["openai/agentkernel"],
                            "account_scopes": ["openai"],
                            "http_allowed_hosts": ["api.github.com"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_http_allowed_hosts=("api.github.com",),
    )

    sources = enabled_corpus_sources(config)

    assert any(source["url"] == "https://api.github.com/repos/openai/agentkernel" for source in sources)
    assert any(source["url"] == "https://api.github.com/repos/openai/agentkernel/readme" for source in sources)
    assert any(source["url"] == "https://api.github.com/repos/openai/agentkernel/releases/latest" for source in sources)
    assert any(source["url"] == "https://api.github.com/users/openai" for source in sources)


def test_enabled_corpus_sources_derive_gitlab_project_endpoints(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "gitlab",
                        "enabled": True,
                        "capabilities": ["gitlab_read"],
                        "settings": {
                            "repo_scopes": ["openai/agentkernel"],
                            "account_scopes": ["openai"],
                            "http_allowed_hosts": ["gitlab.com"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_http_allowed_hosts=("gitlab.com",),
    )

    sources = enabled_corpus_sources(config)

    assert any(source["url"] == "https://gitlab.com/api/v4/projects/openai%2Fagentkernel" for source in sources)
    assert any(
        source["url"]
        == "https://gitlab.com/api/v4/projects/openai%2Fagentkernel/repository/files/README.md/raw?ref=HEAD"
        for source in sources
    )
    assert any(source["url"] == "https://gitlab.com/api/v4/projects/openai%2Fagentkernel/releases" for source in sources)
    assert any(source["url"] == "https://gitlab.com/api/v4/users?username=openai" for source in sources)


def test_enabled_corpus_sources_derive_slack_channel_endpoints(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "slack",
                        "enabled": True,
                        "capabilities": ["slack_read"],
                        "settings": {
                            "channel_scopes": ["C12345"],
                            "account_scopes": ["U999"],
                            "http_allowed_hosts": ["slack.com"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_http_allowed_hosts=("slack.com",),
    )

    sources = enabled_corpus_sources(config)

    assert any(
        source["url"] == "https://slack.com/api/conversations.history?channel=C12345&limit=200" for source in sources
    )
    assert any(source["url"] == "https://slack.com/api/pins.list?channel=C12345" for source in sources)
    assert any(source["url"] == "https://slack.com/api/users.info?user=U999" for source in sources)


def test_capability_registry_uses_retained_operator_policy_snapshot(tmp_path: Path):
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
                    "unattended_allow_http_requests": True,
                    "unattended_http_allowed_hosts": ["example.com"],
                    "unattended_http_timeout_seconds": 15,
                    "unattended_http_max_body_bytes": 131072,
                    "unattended_allow_generated_path_mutations": True,
                    "unattended_generated_path_prefixes": ["build", "dist"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        unattended_allow_git_commands=False,
        unattended_allow_http_requests=False,
        unattended_allow_generated_path_mutations=False,
        operator_policy_proposals_path=operator_policy_path,
    )

    registry = capability_registry_snapshot(config)

    assert registry["builtin"]["git_commands"]["enabled"] is True
    assert registry["builtin"]["generated_path_mutation"]["enabled"] is True
    assert registry["builtin"]["http_request"]["enabled"] is True
    assert registry["builtin"]["http_request"]["scope"]["allowed_hosts"] == ["example.com"]
    assert registry["builtin"]["http_request"]["scope"]["timeout_seconds"] == 15
    assert registry["builtin"]["http_request"]["scope"]["max_body_bytes"] == 131072


def test_capability_registry_exposes_module_policy_scopes(tmp_path: Path):
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
                            "http_allowed_hosts": ["api.github.com"],
                            "repo_scopes": ["openai/agentkernel"],
                            "access_tier": "write",
                            "write_tier": "branch_only",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(capability_modules_path=modules_path)

    registry = capability_registry_snapshot(config)
    policy = registry["capability_policies"]["github_repo_write"][0]

    assert policy["module_id"] == "github"
    assert policy["repo_scopes"] == ["openai/agentkernel"]
    assert policy["access_tier"] == "write"
    assert policy["write_tier"] == "branch_only"


def test_capability_registry_applies_github_adapter_defaults(tmp_path: Path):
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
                        "settings": {
                            "repo_scopes": ["openai/agentkernel"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(capability_modules_path=modules_path)

    registry = capability_registry_snapshot(config)

    assert registry["modules"][0]["adapter_kind"] == "github"
    assert registry["modules"][0]["valid"] is True
    assert registry["modules"][0]["capabilities"] == ["github_read"]
    assert capability_enabled(config, "github_read") is True
    policy = registry["capability_policies"]["github_read"][0]
    assert policy["adapter_kind"] == "github"
    assert policy["http_allowed_hosts"] == ["api.github.com", "uploads.github.com"]
    assert policy["repo_scopes"] == ["openai/agentkernel"]


def test_invalid_enabled_module_does_not_expand_capability_surface(tmp_path: Path):
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
    config = KernelConfig(capability_modules_path=modules_path)

    registry = capability_registry_snapshot(config)

    assert registry["modules"][0]["valid"] is False
    assert "github_repo_write" not in registry["enabled_external_capabilities"]
    assert capability_enabled(config, "github_repo_write") is False
    assert registry["invalid_enabled_modules"][0]["module_id"] == "github"


def test_capability_policy_satisfies_repo_scope_and_access_tier(tmp_path: Path):
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
                            "http_allowed_hosts": ["api.github.com"],
                            "repo_scopes": ["openai/agentkernel"],
                            "access_tier": "write",
                            "write_tier": "branch_only",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(capability_modules_path=modules_path)

    assert capability_policy_satisfies(
        config,
        "github_repo_write",
        {
            "repo_scopes": ["openai/agentkernel"],
            "http_allowed_hosts": ["api.github.com"],
            "access_tier": "read",
            "allow_write": True,
            "write_tier": "branch_only",
        },
    )
    assert capability_policy_satisfies(
        config,
        "github_repo_write",
        {
            "repo_scopes": ["other/repo"],
        },
    ) is False


def test_capability_usage_summary_combines_declared_and_used_capabilities():
    task = TaskSpec(
        task_id="repo_fetch",
        prompt="fetch release notes",
        workspace_subdir="repo_fetch",
        expected_files=["reports/release.txt"],
        metadata={
            "workflow_guard": {
                "required_capabilities": ["github_read"],
                "requires_http": True,
            }
        },
    )
    step = StepRecord(
        index=0,
        thought="fetch",
        action="code_execute",
        content="http_request GET https://api.github.com/repos/openai/openai-python > reports/release.txt",
        selected_skill_id=None,
        command_result={
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "capabilities_used": ["workspace_exec", "workspace_fs", "http_request", "github_read"],
        },
        verification={"passed": True, "reasons": []},
    )

    usage = capability_usage_summary(task, [step])

    assert usage["required_capabilities"] == ["github_read", "http_request"]
    assert usage["used_capabilities"] == ["github_read", "http_request", "workspace_exec", "workspace_fs"]
    assert usage["external_capabilities_used"] == ["github_read"]


def test_build_capability_module_artifact_bootstraps_improvement_surfaces(tmp_path: Path):
    config = KernelConfig(
        capability_modules_path=tmp_path / "config" / "capabilities.json",
    )

    artifact = build_capability_module_artifact(
        config,
        metrics=type("Metrics", (), {"generated_total": 0, "generated_pass_rate": 0.0, "pass_rate": 0.0, "low_confidence_episodes": 0})(),
        failure_counts={},
    )

    assert artifact["artifact_kind"] == "capability_module_set"
    assert artifact["modules"][0]["module_id"] == "kernel_self_extension"
    assert artifact["modules"][0]["settings"]["improvement_subsystems"][0]["base_subsystem"] == "policy"
    assert capability_surface_summary(artifact)["improvement_surface_count"] == 1
