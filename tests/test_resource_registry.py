import json

from agent_kernel.config import KernelConfig
from agent_kernel.llm import MockLLMClient
from agent_kernel.policy import LLMDecisionPolicy
from agent_kernel.resource_registry import runtime_resource_registry
from agent_kernel.resource_types import PROMPT_RESOURCE_SYSTEM, subsystem_resource_id
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.universe_model import UniverseModel


def _write_prompt_templates(repo_root):
    prompts_dir = repo_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "system.md").write_text("custom system prompt", encoding="utf-8")
    (prompts_dir / "decision.md").write_text("custom decision prompt", encoding="utf-8")
    (prompts_dir / "reflection.md").write_text("custom reflection prompt", encoding="utf-8")


def test_runtime_resource_registry_resolves_prompt_and_policy_resources(tmp_path):
    repo_root = tmp_path / "repo"
    _write_prompt_templates(repo_root)
    prompt_policy_path = tmp_path / "trajectories" / "prompts" / "prompt_proposals.json"
    prompt_policy_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_policy_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "proposals": [{"area": "decision", "priority": 3, "suggestion": "be careful"}],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(provider="mock", prompt_proposals_path=prompt_policy_path)

    registry = runtime_resource_registry(config, repo_root=repo_root)

    prompt_version = registry.resolve_active_version(PROMPT_RESOURCE_SYSTEM)
    policy_version = registry.resolve_active_version(subsystem_resource_id("policy"))

    assert prompt_version is not None
    assert prompt_version.payload == "custom system prompt"
    assert prompt_version.version_id.startswith("sha256:")
    assert policy_version is not None
    assert policy_version.lifecycle_state == "retained"
    assert policy_version.payload["artifact_kind"] == "prompt_proposal_set"
    assert policy_version.payload["proposals"][0]["area"] == "decision"


def test_llm_policy_loads_base_prompts_via_resource_registry(tmp_path):
    repo_root = tmp_path / "repo"
    _write_prompt_templates(repo_root)

    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(provider="mock"),
        repo_root=repo_root,
    )

    assert policy.system_prompt == "custom system prompt"
    assert policy.decision_prompt == "custom decision prompt"


def test_universe_model_loads_split_artifacts_via_resource_registry(tmp_path):
    constitution_path = tmp_path / "trajectories" / "universe" / "universe_constitution.json"
    envelope_path = tmp_path / "trajectories" / "universe" / "operating_envelope.json"
    contract_path = tmp_path / "trajectories" / "universe" / "universe_contract.json"
    constitution_path.parent.mkdir(parents=True, exist_ok=True)
    constitution_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "retained",
                "invariants": ["custom invariant"],
                "preferred_command_prefixes": ["python -m pytest"],
            }
        ),
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "operating_envelope",
                "lifecycle_state": "retained",
                "environment_assumptions": {
                    "network_access_mode": "allowlist_only",
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.example.com"],
                "toolchain_requirements": ["git", "pytest", "ruff"],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        universe_constitution_path=constitution_path,
        operating_envelope_path=envelope_path,
        universe_contract_path=contract_path,
    )

    model = UniverseModel(config)
    summary = model.summarize(TaskBank().get("hello_task"))

    assert "custom invariant" in summary["constitution"]["invariants"]
    assert "python -m pytest" in summary["constitution"]["preferred_command_prefixes"]
    assert summary["operating_envelope"]["allowed_http_hosts"] == ["api.example.com"]
    assert summary["operating_envelope"]["toolchain_requirements"] == ["git", "pytest", "ruff"]
    assert summary["environment_assumptions"]["network_access_mode"] == "allowlist_only"
