from pathlib import Path

from evals.metrics import EvalMetrics

from agent_kernel.config import KernelConfig
from agent_kernel.capabilities import adapter_catalog_snapshot
from agent_kernel.loop import _FRONTIER_STEP_FLOOR_FAMILIES
from agent_kernel.subsystems import baseline_candidate_flags, subsystem_spec
from agent_kernel.task_bank import _synthetic_lineage_seed_skipped
from agent_kernel.tolbert_model_improvement import (
    _DEFAULT_TOLBERT_BUILD_POLICY,
    _TOLBERT_SHARED_STORE_GROUPS,
    _tolbert_family_proposal_gate,
)
from agent_kernel.trust_improvement import trust_behavior_controls
from agent_kernel.unattended_controller import _FOCUSES, _STATE_FEATURE_ORDER, _policy_features


def test_subsystem_specs_and_flags_are_dataset_backed():
    spec = subsystem_spec("retrieval")
    assert spec.artifact_kind == "retrieval_policy_set"
    assert spec.proposal_toggle_attr == "use_retrieval_proposals"

    baseline, candidate = baseline_candidate_flags("operators")
    assert baseline["include_discovered_tasks"] is True
    assert baseline["include_skill_transfer"] is True
    assert candidate["include_operator_memory"] is True
    assert candidate["include_failure_generated"] is True


def test_adapter_catalog_snapshot_exposes_dataset_backed_defaults():
    snapshot = adapter_catalog_snapshot()
    assert snapshot["github"]["label"] == "GitHub"
    assert snapshot["github"]["default_capabilities"] == ["github_read"]
    assert snapshot["github"]["default_settings"]["http_allowed_hosts"] == [
        "api.github.com",
        "uploads.github.com",
    ]


def test_synthetic_lineage_skip_uses_dataset_backed_filters(tmp_path: Path):
    assert _synthetic_lineage_seed_skipped(
        {
            "task_id": "hello_task_episode_replay",
            "task_metadata": {"benchmark_family": "workflow"},
        }
    )
    assert _synthetic_lineage_seed_skipped(
        {
            "task_id": "hello_task",
            "task_metadata": {"benchmark_family": "workflow", "memory_source": "operator"},
        }
    )
    assert not _synthetic_lineage_seed_skipped(
        {
            "task_id": "hello_task",
            "task_metadata": {"benchmark_family": "workflow"},
            "workspace": str(tmp_path / "workspace" / "hello_task"),
        }
    )


def test_unattended_controller_feature_catalogs_are_dataset_backed():
    assert _FOCUSES == ("balanced", "recovery_alignment", "discovered_task_adaptation")
    assert _STATE_FEATURE_ORDER[-4:] == (
        "allow_kernel_autobuild",
        "liftoff_shadow",
        "liftoff_reject",
        "liftoff_retain",
    )

    features = _policy_features(
        {
            "focus": "recovery_alignment",
            "adaptive_search": True,
            "cycles": 2,
            "campaign_width": 2,
            "variant_width": 1,
            "task_limit": 64,
            "priority_benchmark_families": ["integration"],
        }
    )

    assert features["focus_recovery_alignment"] == 1.0
    assert features["priority_broad_required"] == 1.0


def test_tolbert_family_gates_are_dataset_backed():
    assert _DEFAULT_TOLBERT_BUILD_POLICY["min_total_examples"] == 512
    assert _DEFAULT_TOLBERT_BUILD_POLICY["require_synthetic_dataset"] is True
    assert _TOLBERT_SHARED_STORE_GROUPS == (
        "assets",
        "dataset",
        "universal_dataset",
        "training",
        "retrieval_cache",
        "hybrid_runtime",
        "universal_runtime",
    )
    assert _tolbert_family_proposal_gate("project") == {
        "require_novel_command_signal": True,
        "min_proposal_selected_steps_delta": 1,
        "min_novel_valid_command_steps": 1,
        "min_novel_valid_command_rate_delta": 0.1,
    }
    assert _tolbert_family_proposal_gate("repo_sandbox") == {
        "require_novel_command_signal": True,
        "min_proposal_selected_steps_delta": 0,
        "min_novel_valid_command_steps": 1,
        "min_novel_valid_command_rate_delta": 0.0,
    }


def test_trust_defaults_are_dataset_backed():
    controls = trust_behavior_controls(
        EvalMetrics(total=10, passed=8),
        {
            "overall_summary": {
                "distinct_benchmark_families": 3,
                "total": 12,
            }
        },
    )

    assert controls["required_benchmark_families"] == [
        "integration",
        "project",
        "repo_chore",
        "repo_sandbox",
        "repository",
    ]


def test_runtime_defaults_are_dataset_backed():
    config = KernelConfig()

    assert "repo_chore" in config.unattended_allowed_benchmark_families
    assert "verifier_candidate" in config.unattended_allowed_benchmark_families
    assert config.unattended_generated_path_prefixes[:2] == ("dist", "build")
    assert config.unattended_trust_required_benchmark_families == (
        "repo_chore",
        "repo_sandbox",
        "project",
        "repository",
        "integration",
    )
    assert _FRONTIER_STEP_FLOOR_FAMILIES == {
        "integration",
        "project",
        "repo_sandbox",
        "repository",
        "tooling",
    }
