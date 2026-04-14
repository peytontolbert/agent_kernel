from agent_kernel.extensions.improvement.improvement_catalog import (
    catalog_mapping,
    catalog_nested_string_sets,
    catalog_string_list,
    catalog_string_set,
)
from agent_kernel.extensions.improvement.improvement_common import retention_gate_preset
from agent_kernel.tasking.task_budget import uplifted_task_max_steps


def test_improvement_catalog_exposes_expected_schema_metadata():
    assert "workflow_alignment" in catalog_string_set("improvement", "world_model_proposal_areas")
    assert catalog_nested_string_sets("universe", "environment_assumption_enum_fields")["network_access_mode"] == {
        "allowlist_only",
        "blocked",
        "open",
    }
    assert catalog_string_list("operator_policy", "default_generated_prefixes") == [
        "build",
        "dist",
        "generated",
        "reports",
        "tmp",
    ]


def test_improvement_catalog_exposes_artifact_contract_registry_and_gate_presets():
    contracts = catalog_mapping("improvement", "artifact_contracts")
    profiles = catalog_mapping("improvement", "artifact_validation_profiles")
    assert contracts["policy"]["artifact_kind"] == "prompt_proposal_set"
    assert contracts["universe"]["artifact_kind"] == [
        "universe_contract",
        "universe_constitution",
        "operating_envelope",
    ]
    assert contracts["tooling"]["lifecycle_states"] == [
        "candidate",
        "replay_verified",
        "retained",
        "rejected",
    ]
    assert retention_gate_preset("operators") == {
        "min_transfer_pass_rate_delta_abs": 0.05,
        "require_cross_task_support": True,
        "min_support": 2,
    }
    assert profiles["operator_policy"]["control_schema"] == "unattended_operator_controls_v1"
    assert "benchmark" in profiles
    assert "tooling" in profiles
    assert "signatures" in {
        section["field"] for section in profiles["transition_model"]["sections"] if isinstance(section, dict)
    }


def test_uplifted_task_max_steps_uses_dataset_backed_budget_metadata():
    assert uplifted_task_max_steps(1, metadata={"benchmark_family": "integration"}) == 16
    assert uplifted_task_max_steps(
        1,
        metadata={
            "benchmark_family": "episode_memory",
            "origin_benchmark_family": "repository",
        },
    ) == 14
    assert uplifted_task_max_steps(1, metadata={"budget_step_floor": 9}) == 9
