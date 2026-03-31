import json
from pathlib import Path

from agent_kernel.cycle_runner import prior_retained_guard_reason
from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import tolbert_kernel_autobuild_ready
from agent_kernel.improvement import (
    ImprovementCycleRecord,
    ImprovementExperiment,
    ImprovementPlanner,
    ImprovementVariant,
    apply_artifact_retention_decision,
    assess_artifact_compatibility,
    effective_artifact_payload_for_retention,
    evaluate_artifact_retention,
    materialize_replay_verified_tool_payload,
    persist_replay_verified_tool_artifact,
    snapshot_artifact_state,
    staged_candidate_artifact_path,
    stamp_artifact_experiment_variant,
    stamp_artifact_generation_context,
)
from agent_kernel.benchmark_synthesis import synthesize_benchmark_candidates
from agent_kernel.curriculum_improvement import build_curriculum_proposal_artifact
from agent_kernel.delegation_improvement import build_delegation_proposal_artifact
from agent_kernel.operator_policy_improvement import build_operator_policy_proposal_artifact
from agent_kernel.policy_improvement import build_policy_proposal_artifact
from agent_kernel.prompt_improvement import (
    build_prompt_proposal_artifact,
    propose_prompt_adjustments,
    retained_improvement_planner_controls,
    retained_planner_controls,
    retained_role_directives,
)
from agent_kernel.recovery_improvement import build_recovery_proposal_artifact
from agent_kernel.retrieval_improvement import build_retrieval_proposal_artifact, retained_retrieval_overrides
from agent_kernel.state_estimation_improvement import build_state_estimation_proposal_artifact
from agent_kernel.subsystems import (
    active_artifact_path_for_subsystem,
    comparison_config_for_subsystem_artifact,
    generate_candidate_artifact,
)
from agent_kernel.transition_model_improvement import build_transition_model_proposal_artifact
from agent_kernel.trust_improvement import build_trust_proposal_artifact
from agent_kernel.universe_improvement import build_universe_contract_artifact
from agent_kernel.verifier_improvement import synthesize_verifier_contracts
from agent_kernel.world_model_improvement import build_world_model_proposal_artifact
from evals.metrics import EvalMetrics


SKILL_RETENTION_GATE = {"require_non_regression": True}
OPERATOR_RETENTION_GATE = {
    "min_transfer_pass_rate_delta_abs": 0.05,
    "require_cross_task_support": True,
    "min_support": 2,
}


def _skill_record(
    skill_id: str,
    *,
    source_task_id: str = "hello_task",
    command: str = "printf 'hello agent kernel\\n' > hello.txt",
    expected_files: list[str] | None = None,
    benchmark_family: str = "micro",
    quality: float = 0.9,
) -> dict[str, object]:
    return {
        "skill_id": skill_id,
        "source_task_id": source_task_id,
        "benchmark_family": benchmark_family,
        "quality": quality,
        "procedure": {"commands": [command]},
        "task_contract": {"expected_files": list(expected_files or ["hello.txt"])},
        "verifier": {"termination_reason": "success"},
    }


def _tool_candidate_record(
    tool_id: str = "tool:hello_task:primary",
    *,
    lifecycle_state: str = "candidate",
    promotion_stage: str = "candidate_procedure",
) -> dict[str, object]:
    return {
        "spec_version": "asi_v1",
        "tool_id": tool_id,
        "kind": "local_shell_procedure",
        "lifecycle_state": lifecycle_state,
        "promotion_stage": promotion_stage,
        "source_task_id": "hello_task",
        "benchmark_family": "micro",
        "quality": 0.9,
        "script_name": "hello_task_tool.sh",
        "script_body": "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'hello agent kernel\\n' > hello.txt\n",
        "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
        "task_contract": {"expected_files": ["hello.txt"]},
        "verifier": {"termination_reason": "success"},
    }


def _retained_universe_artifact(
    *,
    network_access_mode: str = "allowlist_only",
    git_write_mode: str = "operator_gated",
    workspace_write_scope: str = "task_only",
    scope_escape_penalty: int = 11,
    network_fetch_penalty: int = 6,
) -> dict[str, object]:
    return {
        "spec_version": "asi_v1",
        "artifact_kind": "universe_contract",
        "lifecycle_state": "retained",
        "control_schema": "universe_contract_v1",
        "retention_gate": {"require_non_regression": True},
        "governance": {
            "require_verification": True,
            "require_bounded_steps": True,
            "prefer_reversible_actions": True,
            "respect_task_forbidden_artifacts": True,
            "respect_preserved_artifacts": True,
        },
        "invariants": ["verify before accepting terminal success"],
        "forbidden_command_patterns": ["git reset --hard"],
        "preferred_command_prefixes": ["pytest -q"],
        "action_risk_controls": {
            "destructive_mutation_penalty": 12,
            "git_mutation_penalty": 8,
            "inline_destructive_interpreter_penalty": 8,
            "network_fetch_penalty": network_fetch_penalty,
            "privileged_command_penalty": 10,
            "read_only_discovery_bonus": 3,
            "remote_execution_penalty": 10,
            "reversible_file_operation_bonus": 2,
            "scope_escape_penalty": scope_escape_penalty,
            "unbounded_execution_penalty": 7,
            "verification_bonus": 4,
        },
        "environment_assumptions": {
            "network_access_mode": network_access_mode,
            "git_write_mode": git_write_mode,
            "workspace_write_scope": workspace_write_scope,
            "require_path_scoped_mutations": True,
            "require_rollback_on_mutation": True,
        },
    }


def test_improvement_planner_prefers_retrieval_when_low_confidence_is_high(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(
        total=10,
        passed=10,
        low_confidence_episodes=4,
        trusted_retrieval_steps=2,
    )

    target = planner.choose_target(metrics)

    assert target.subsystem == "retrieval"


def test_improvement_planner_uses_tool_memory_failure_signal_in_scoring(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.failure_counts = lambda: {"command_failure": 1, "missing_expected_file": 0}
    planner.transition_failure_counts = lambda: {}
    planner.transition_summary = lambda: {}
    metrics = EvalMetrics(
        total=4,
        passed=3,
        total_by_benchmark_family={"bounded": 3, "benchmark_candidate": 1},
        passed_by_benchmark_family={"bounded": 2, "benchmark_candidate": 1},
        total_by_memory_source={"tool": 1, "episode": 3, "verifier": 1, "skill_transfer": 1},
        passed_by_memory_source={"tool": 0, "episode": 3, "verifier": 1, "skill_transfer": 1},
        generated_total=1,
        generated_passed=1,
        trusted_retrieval_steps=4,
        task_outcomes={
            "tool_task": {
                "task_id": "tool_task",
                "memory_source": "tool",
                "success": False,
                "completion_ratio": 0.0,
                "proposal_selected_steps": 0,
                "novel_valid_command_steps": 0,
                "no_state_progress_steps": 0,
                "state_regression_steps": 0,
                "failure_signals": ["command_failure"],
            },
            "episode_task": {
                "task_id": "episode_task",
                "memory_source": "episode",
                "success": True,
                "completion_ratio": 1.0,
                "proposal_selected_steps": 0,
                "novel_valid_command_steps": 0,
                "no_state_progress_steps": 0,
                "state_regression_steps": 0,
                "failure_signals": [],
            },
        },
    )

    experiments = planner.rank_experiments(metrics)

    tooling = next(candidate for candidate in experiments if candidate.subsystem == "tooling")
    assert tooling.evidence["memory_source_pressure"]["bonus"] > 0.0
    assert "tool" in tooling.evidence["memory_source_pressure"]["relevant_sources"]
    assert planner.choose_target(metrics).subsystem == "tooling"


def test_curriculum_expected_gain_does_not_treat_missing_generated_lane_as_full_failure(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.failure_counts = lambda: {}
    planner.transition_failure_counts = lambda: {"no_state_progress": 1, "state_regression": 0}
    planner.transition_summary = lambda: {}
    metrics = EvalMetrics(
        total=5,
        passed=5,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
        passed_by_memory_source={"verifier": 1, "skill_transfer": 1},
        generated_total=0,
        generated_passed=0,
    )

    experiments = planner.rank_experiments(metrics)

    curriculum = next(candidate for candidate in experiments if candidate.subsystem == "curriculum")
    assert curriculum.expected_gain == 0.05
    assert curriculum.evidence["generated_pass_rate_gap"] == 0.0


def test_assess_artifact_compatibility_accepts_valid_tolbert_model_artifact(tmp_path: Path):
    runtime_paths = {}
    for name in ("config.json", "checkpoint.pt", "nodes.jsonl", "label_map.json", "source_spans.jsonl", "cache.pt"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        runtime_paths[name] = str(path)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": "balanced",
        "model_surfaces": {
            "retrieval_surface": True,
            "policy_head": True,
            "value_head": True,
            "transition_head": True,
            "latent_state": True,
        },
        "runtime_policy": {
            "shadow_benchmark_families": ["workflow"],
            "primary_benchmark_families": [],
            "min_path_confidence": 0.75,
            "require_trusted_retrieval": True,
            "fallback_to_vllm_on_low_confidence": True,
            "allow_direct_command_primary": True,
            "allow_skill_primary": True,
            "primary_min_command_score": 2,
            "use_value_head": True,
            "use_transition_head": True,
            "use_policy_head": True,
            "use_latent_state": True,
        },
        "decoder_policy": {
            "allow_retrieval_guidance": True,
            "allow_skill_commands": True,
            "allow_task_suggestions": True,
            "allow_stop_decision": True,
            "min_stop_completion_ratio": 0.95,
            "max_task_suggestions": 3,
        },
        "rollout_policy": {
            "predicted_progress_gain_weight": 3.0,
            "predicted_conflict_penalty_weight": 4.0,
            "predicted_preserved_bonus_weight": 1.0,
            "predicted_workflow_bonus_weight": 1.5,
            "latent_progress_bonus_weight": 1.0,
            "latent_risk_penalty_weight": 2.0,
            "recover_from_stall_bonus_weight": 1.5,
            "stop_completion_weight": 8.0,
            "stop_missing_expected_penalty_weight": 6.0,
            "stop_forbidden_penalty_weight": 6.0,
            "stop_preserved_penalty_weight": 4.0,
            "stable_stop_bonus_weight": 1.5,
        },
        "liftoff_gate": {
            "min_pass_rate_delta": 0.0,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "build_policy": {
            "allow_kernel_autobuild": False,
            "allow_kernel_rebuild": False,
            "require_synthetic_dataset": True,
            "min_total_examples": 512,
            "min_synthetic_examples": 64,
            "ready_total_examples": 4,
            "ready_synthetic_examples": 0,
        },
        "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
        "dataset_manifest": {"total_examples": 4},
        "runtime_paths": {
            "config_path": runtime_paths["config.json"],
            "checkpoint_path": runtime_paths["checkpoint.pt"],
            "nodes_path": runtime_paths["nodes.jsonl"],
            "label_map_path": runtime_paths["label_map.json"],
            "source_spans_paths": [runtime_paths["source_spans.jsonl"]],
            "cache_paths": [runtime_paths["cache.pt"]],
        },
        "proposals": [{"area": "balanced", "priority": 4, "reason": "test"}],
    }

    compatibility = assess_artifact_compatibility(subsystem="tolbert_model", payload=payload)

    assert compatibility["compatible"] is True


def test_evaluate_artifact_retention_accepts_tolbert_model_candidate(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_path = tmp_path / "cache.pt"
    for path in (checkpoint_path, cache_path, tmp_path / "config.json", tmp_path / "nodes.jsonl", tmp_path / "label_map.json", tmp_path / "source_spans.jsonl"):
        path.write_text("{}", encoding="utf-8")
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": "balanced",
        "model_surfaces": {
            "retrieval_surface": True,
            "policy_head": True,
            "value_head": True,
            "transition_head": True,
            "latent_state": True,
        },
        "runtime_policy": {
            "shadow_benchmark_families": ["workflow"],
            "primary_benchmark_families": [],
            "min_path_confidence": 0.75,
            "require_trusted_retrieval": True,
            "fallback_to_vllm_on_low_confidence": True,
            "allow_direct_command_primary": True,
            "allow_skill_primary": True,
            "primary_min_command_score": 2,
            "use_value_head": True,
            "use_transition_head": True,
            "use_policy_head": True,
            "use_latent_state": True,
        },
        "decoder_policy": {
            "allow_retrieval_guidance": True,
            "allow_skill_commands": True,
            "allow_task_suggestions": True,
            "allow_stop_decision": True,
            "min_stop_completion_ratio": 0.95,
            "max_task_suggestions": 3,
        },
        "rollout_policy": {
            "predicted_progress_gain_weight": 3.0,
            "predicted_conflict_penalty_weight": 4.0,
            "predicted_preserved_bonus_weight": 1.0,
            "predicted_workflow_bonus_weight": 1.5,
            "latent_progress_bonus_weight": 1.0,
            "latent_risk_penalty_weight": 2.0,
            "recover_from_stall_bonus_weight": 1.5,
            "stop_completion_weight": 8.0,
            "stop_missing_expected_penalty_weight": 6.0,
            "stop_forbidden_penalty_weight": 6.0,
            "stop_preserved_penalty_weight": 4.0,
            "stable_stop_bonus_weight": 1.5,
        },
        "liftoff_gate": {
            "min_pass_rate_delta": 0.0,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "build_policy": {
            "allow_kernel_autobuild": False,
            "allow_kernel_rebuild": False,
            "require_synthetic_dataset": True,
            "min_total_examples": 512,
            "min_synthetic_examples": 64,
            "ready_total_examples": 8,
            "ready_synthetic_examples": 0,
        },
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.0,
            "max_step_regression": 0.0,
            "max_low_confidence_episode_regression": 0,
            "min_first_step_confidence_delta": 0.0,
            "min_trusted_retrieval_delta": 0,
            "require_novel_command_signal": True,
            "min_proposal_selected_steps_delta": 0,
            "min_novel_valid_command_steps": 1,
            "min_novel_valid_command_rate_delta": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
        "dataset_manifest": {"total_examples": 8},
        "runtime_paths": {
            "config_path": str(tmp_path / "config.json"),
            "checkpoint_path": str(checkpoint_path),
            "nodes_path": str(tmp_path / "nodes.jsonl"),
            "label_map_path": str(tmp_path / "label_map.json"),
            "source_spans_paths": [str(tmp_path / "source_spans.jsonl")],
            "cache_paths": [str(cache_path)],
        },
        "proposals": [{"area": "balanced", "priority": 4, "reason": "test"}],
    }

    state, reason = evaluate_artifact_retention(
        "tolbert_model",
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            trusted_retrieval_steps=2,
            low_confidence_episodes=3,
            average_first_step_path_confidence=0.4,
            proposal_selected_steps=1,
            novel_command_steps=1,
            novel_valid_command_steps=0,
            generated_total=2,
            generated_passed=1,
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.4,
            trusted_retrieval_steps=4,
            low_confidence_episodes=2,
            average_first_step_path_confidence=0.7,
            proposal_selected_steps=2,
            novel_command_steps=2,
            novel_valid_command_steps=1,
            generated_total=2,
            generated_passed=1,
        ),
        payload=payload,
    )

    assert state == "retain"
    assert "Tolbert model candidate improved" in reason


def test_evaluate_artifact_retention_rejects_tolbert_model_without_verifier_valid_novel_commands(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_path = tmp_path / "cache.pt"
    for path in (
        checkpoint_path,
        cache_path,
        tmp_path / "config.json",
        tmp_path / "nodes.jsonl",
        tmp_path / "label_map.json",
        tmp_path / "source_spans.jsonl",
    ):
        path.write_text("{}", encoding="utf-8")
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": "balanced",
        "model_surfaces": {
            "retrieval_surface": True,
            "policy_head": True,
            "value_head": True,
            "transition_head": True,
            "latent_state": True,
        },
        "runtime_policy": {
            "shadow_benchmark_families": ["workflow"],
            "primary_benchmark_families": [],
            "min_path_confidence": 0.75,
            "require_trusted_retrieval": True,
            "fallback_to_vllm_on_low_confidence": True,
            "allow_direct_command_primary": True,
            "allow_skill_primary": True,
            "primary_min_command_score": 2,
            "use_value_head": True,
            "use_transition_head": True,
            "use_policy_head": True,
            "use_latent_state": True,
        },
        "decoder_policy": {
            "allow_retrieval_guidance": True,
            "allow_skill_commands": True,
            "allow_task_suggestions": True,
            "allow_stop_decision": True,
            "min_stop_completion_ratio": 0.95,
            "max_task_suggestions": 3,
        },
        "rollout_policy": {
            "predicted_progress_gain_weight": 3.0,
            "predicted_conflict_penalty_weight": 4.0,
            "predicted_preserved_bonus_weight": 1.0,
            "predicted_workflow_bonus_weight": 1.5,
            "latent_progress_bonus_weight": 1.0,
            "latent_risk_penalty_weight": 2.0,
            "recover_from_stall_bonus_weight": 1.5,
            "stop_completion_weight": 8.0,
            "stop_missing_expected_penalty_weight": 6.0,
            "stop_forbidden_penalty_weight": 6.0,
            "stop_preserved_penalty_weight": 4.0,
            "stable_stop_bonus_weight": 1.5,
        },
        "liftoff_gate": {
            "min_pass_rate_delta": 0.0,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.0,
            "max_step_regression": 0.0,
            "max_low_confidence_episode_regression": 0,
            "min_first_step_confidence_delta": 0.0,
            "min_trusted_retrieval_delta": 0,
            "require_novel_command_signal": True,
            "min_proposal_selected_steps_delta": 0,
            "min_novel_valid_command_steps": 1,
            "min_novel_valid_command_rate_delta": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "build_policy": {
            "allow_kernel_autobuild": False,
            "allow_kernel_rebuild": False,
            "require_synthetic_dataset": True,
            "min_total_examples": 512,
            "min_synthetic_examples": 64,
            "ready_total_examples": 8,
            "ready_synthetic_examples": 0,
        },
        "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
        "dataset_manifest": {"total_examples": 8},
        "runtime_paths": {
            "config_path": str(tmp_path / "config.json"),
            "checkpoint_path": str(checkpoint_path),
            "nodes_path": str(tmp_path / "nodes.jsonl"),
            "label_map_path": str(tmp_path / "label_map.json"),
            "source_spans_paths": [str(tmp_path / "source_spans.jsonl")],
            "cache_paths": [str(cache_path)],
        },
        "proposals": [{"area": "balanced", "priority": 4, "reason": "test"}],
    }

    state, reason = evaluate_artifact_retention(
        "tolbert_model",
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.5,
            trusted_retrieval_steps=2,
            low_confidence_episodes=3,
            average_first_step_path_confidence=0.4,
            proposal_selected_steps=1,
            novel_command_steps=1,
            novel_valid_command_steps=1,
            generated_total=2,
            generated_passed=1,
        ),
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.4,
            trusted_retrieval_steps=4,
            low_confidence_episodes=2,
            average_first_step_path_confidence=0.7,
            proposal_selected_steps=2,
            novel_command_steps=2,
            novel_valid_command_steps=0,
            generated_total=2,
            generated_passed=1,
        ),
        payload=payload,
    )

    assert state == "reject"
    assert "verifier-valid novel commands" in reason


def test_evaluate_artifact_retention_rejects_tolbert_model_when_hard_family_lacks_novel_command_signal(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_path = tmp_path / "cache.pt"
    for path in (
        checkpoint_path,
        cache_path,
        tmp_path / "config.json",
        tmp_path / "nodes.jsonl",
        tmp_path / "label_map.json",
        tmp_path / "source_spans.jsonl",
    ):
        path.write_text("{}", encoding="utf-8")
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": "balanced",
        "model_surfaces": {
            "retrieval_surface": True,
            "policy_head": True,
            "value_head": True,
            "transition_head": True,
            "latent_state": True,
        },
        "runtime_policy": {
            "shadow_benchmark_families": ["repository"],
            "primary_benchmark_families": [],
            "min_path_confidence": 0.75,
            "require_trusted_retrieval": True,
            "fallback_to_vllm_on_low_confidence": True,
            "allow_direct_command_primary": True,
            "allow_skill_primary": True,
            "primary_min_command_score": 2,
            "use_value_head": True,
            "use_transition_head": True,
            "use_policy_head": True,
            "use_latent_state": True,
        },
        "decoder_policy": {
            "allow_retrieval_guidance": True,
            "allow_skill_commands": True,
            "allow_task_suggestions": True,
            "allow_stop_decision": True,
            "min_stop_completion_ratio": 0.95,
            "max_task_suggestions": 3,
        },
        "rollout_policy": {
            "predicted_progress_gain_weight": 3.0,
            "predicted_conflict_penalty_weight": 4.0,
            "predicted_preserved_bonus_weight": 1.0,
            "predicted_workflow_bonus_weight": 1.5,
            "latent_progress_bonus_weight": 1.0,
            "latent_risk_penalty_weight": 2.0,
            "recover_from_stall_bonus_weight": 1.5,
            "stop_completion_weight": 8.0,
            "stop_missing_expected_penalty_weight": 6.0,
            "stop_forbidden_penalty_weight": 6.0,
            "stop_preserved_penalty_weight": 4.0,
            "stable_stop_bonus_weight": 1.5,
        },
        "liftoff_gate": {
            "min_pass_rate_delta": 0.0,
            "max_step_regression": 0.0,
            "max_regressed_families": 0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
        },
        "build_policy": {
            "allow_kernel_autobuild": False,
            "allow_kernel_rebuild": False,
            "require_synthetic_dataset": True,
            "min_total_examples": 512,
            "min_synthetic_examples": 64,
            "ready_total_examples": 8,
            "ready_synthetic_examples": 0,
        },
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.0,
            "max_step_regression": 0.0,
            "max_low_confidence_episode_regression": 0,
            "min_first_step_confidence_delta": 0.0,
            "min_trusted_retrieval_delta": 0,
            "require_novel_command_signal": True,
            "min_proposal_selected_steps_delta": 0,
            "min_novel_valid_command_steps": 1,
            "min_novel_valid_command_rate_delta": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
            "proposal_gate_by_benchmark_family": {
                "repository": {
                    "require_novel_command_signal": True,
                    "min_proposal_selected_steps_delta": 1,
                    "min_novel_valid_command_steps": 1,
                    "min_novel_valid_command_rate_delta": 0.1,
                },
                "micro": {
                    "require_novel_command_signal": False,
                    "min_proposal_selected_steps_delta": 0,
                    "min_novel_valid_command_steps": 0,
                    "min_novel_valid_command_rate_delta": 0.0,
                },
            },
            "required_confirmation_runs": 2,
        },
        "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
        "dataset_manifest": {"total_examples": 8},
        "runtime_paths": {
            "config_path": str(tmp_path / "config.json"),
            "checkpoint_path": str(checkpoint_path),
            "nodes_path": str(tmp_path / "nodes.jsonl"),
            "label_map_path": str(tmp_path / "label_map.json"),
            "source_spans_paths": [str(tmp_path / "source_spans.jsonl")],
            "cache_paths": [str(cache_path)],
        },
        "proposals": [{"area": "balanced", "priority": 4, "reason": "test"}],
    }
    baseline_metrics = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.5,
        trusted_retrieval_steps=2,
        low_confidence_episodes=3,
        average_first_step_path_confidence=0.4,
        proposal_selected_steps=0,
        novel_command_steps=0,
        novel_valid_command_steps=0,
        generated_total=2,
        generated_passed=1,
        task_trajectories={
            "repo-task": {"benchmark_family": "repository", "success": True, "steps": []},
            "micro-task": {"benchmark_family": "micro", "success": True, "steps": []},
        },
    )
    candidate_metrics = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.4,
        trusted_retrieval_steps=4,
        low_confidence_episodes=2,
        average_first_step_path_confidence=0.7,
        proposal_selected_steps=1,
        novel_command_steps=1,
        novel_valid_command_steps=1,
        generated_total=2,
        generated_passed=1,
        task_trajectories={
            "repo-task": {"benchmark_family": "repository", "success": True, "steps": []},
            "micro-task": {
                "benchmark_family": "micro",
                "success": True,
                "steps": [
                    {
                        "proposal_source": "action_generation",
                        "proposal_novel": True,
                        "verification_passed": True,
                    }
                ],
            },
        },
    )

    state, reason = evaluate_artifact_retention(
        "tolbert_model",
        baseline_metrics,
        candidate_metrics,
        payload=payload,
    )

    assert state == "reject"
    assert "repository tasks" in reason


def test_prior_retained_guard_reason_rejects_hard_family_novel_command_regression() -> None:
    reason = prior_retained_guard_reason(
        subsystem="tolbert_model",
        gate={
            "max_step_regression": 0.0,
            "require_generated_lane_non_regression": True,
            "require_failure_recovery_non_regression": True,
            "proposal_gate_by_benchmark_family": {
                "repository": {
                    "require_novel_command_signal": True,
                    "min_proposal_selected_steps_delta": 1,
                    "min_novel_valid_command_steps": 1,
                    "min_novel_valid_command_rate_delta": 0.1,
                }
            },
        },
        comparison={
            "available": True,
            "baseline_metrics": {
                "pass_rate": 0.8,
                "average_steps": 1.5,
                "generated_pass_rate": 0.5,
                "proposal_selected_steps": 0,
                "novel_valid_command_steps": 0,
                "novel_valid_command_rate": 0.0,
            },
            "current_metrics": {
                "pass_rate": 0.8,
                "average_steps": 1.4,
                "generated_pass_rate": 0.5,
                "proposal_selected_steps": 1,
                "novel_valid_command_steps": 1,
                "novel_valid_command_rate": 1.0,
            },
            "evidence": {
                "proposal_metrics_by_benchmark_family": {
                    "repository": {
                        "baseline_task_count": 1,
                        "candidate_task_count": 1,
                        "baseline_proposal_selected_steps": 0,
                        "candidate_proposal_selected_steps": 0,
                        "proposal_selected_steps_delta": 0,
                        "baseline_novel_valid_command_steps": 0,
                        "candidate_novel_valid_command_steps": 0,
                        "baseline_novel_valid_command_rate": 0.0,
                        "candidate_novel_valid_command_rate": 0.0,
                        "novel_valid_command_rate_delta": 0.0,
                    }
                }
            },
        },
    )

    assert reason is not None
    assert "repository tasks" in reason


def test_comparison_config_for_tolbert_model_artifact_applies_runtime_paths(tmp_path: Path):
    artifact_path = tmp_path / "tolbert_model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_paths": {
                    "config_path": str(tmp_path / "candidate_config.json"),
                    "checkpoint_path": str(tmp_path / "candidate_checkpoint.pt"),
                    "nodes_path": str(tmp_path / "candidate_nodes.jsonl"),
                    "label_map_path": str(tmp_path / "candidate_label_map.json"),
                    "source_spans_paths": [str(tmp_path / "candidate_spans.jsonl")],
                    "cache_paths": [str(tmp_path / "candidate_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(tolbert_model_artifact_path=tmp_path / "retained_tolbert_model.json")

    candidate = comparison_config_for_subsystem_artifact(config, "tolbert_model", artifact_path)

    assert candidate.tolbert_checkpoint_path == str(tmp_path / "candidate_checkpoint.pt")
    assert candidate.tolbert_cache_paths == (str(tmp_path / "candidate_cache.pt"),)


def test_comparison_config_for_tolbert_model_artifact_materializes_delta_checkpoint(
    monkeypatch,
    tmp_path: Path,
):
    artifact_path = tmp_path / "tolbert_model.json"
    parent_checkpoint = tmp_path / "parent.pt"
    delta_checkpoint = tmp_path / "delta.pt"
    materialized_checkpoint = tmp_path / ".materialized_checkpoints" / "candidate.pt"
    parent_checkpoint.write_bytes(b"parent")
    delta_checkpoint.write_bytes(b"delta")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_paths": {
                    "config_path": str(tmp_path / "candidate_config.json"),
                    "checkpoint_path": "",
                    "parent_checkpoint_path": str(parent_checkpoint),
                    "checkpoint_delta_path": str(delta_checkpoint),
                    "cache_paths": [str(tmp_path / "candidate_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent_kernel.subsystems.resolve_tolbert_runtime_checkpoint_path",
        lambda runtime_paths, *, artifact_path=None: str(materialized_checkpoint),
    )
    config = KernelConfig(tolbert_model_artifact_path=tmp_path / "retained_tolbert_model.json")

    candidate = comparison_config_for_subsystem_artifact(config, "tolbert_model", artifact_path)

    assert candidate.tolbert_checkpoint_path == str(materialized_checkpoint)


def test_generate_candidate_artifact_supports_tolbert_model(monkeypatch, tmp_path: Path):
    from agent_kernel import subsystems as subsystems_module

    config = KernelConfig(
        candidate_artifacts_root=tmp_path / "candidates",
        tolbert_model_artifact_path=tmp_path / "tolbert_model" / "artifact.json",
    )
    candidate_artifact_path = tmp_path / "candidates" / "tolbert_model_candidate.json"
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        subsystems_module,
        "build_tolbert_model_candidate_artifact",
        lambda **kwargs: {
            "spec_version": "asi_v1",
            "artifact_kind": "tolbert_model_bundle",
            "lifecycle_state": "candidate",
            "generation_focus": "balanced",
            "model_surfaces": {
                "retrieval_surface": True,
                "policy_head": True,
                "value_head": True,
                "transition_head": True,
                "latent_state": True,
            },
            "runtime_policy": {
                "shadow_benchmark_families": ["workflow"],
                "primary_benchmark_families": [],
                "min_path_confidence": 0.75,
                "require_trusted_retrieval": True,
                "fallback_to_vllm_on_low_confidence": True,
                "allow_direct_command_primary": True,
                "allow_skill_primary": True,
                "primary_min_command_score": 2,
                "use_value_head": True,
                "use_transition_head": True,
                "use_policy_head": True,
                "use_latent_state": True,
            },
            "liftoff_gate": {
                "min_pass_rate_delta": 0.0,
                "max_step_regression": 0.0,
                "max_regressed_families": 0,
                "require_generated_lane_non_regression": True,
                "require_failure_recovery_non_regression": True,
            },
            "build_policy": {
                "allow_kernel_autobuild": False,
                "allow_kernel_rebuild": False,
                "require_synthetic_dataset": True,
                "min_total_examples": 512,
                "min_synthetic_examples": 64,
                "ready_total_examples": 4,
                "ready_synthetic_examples": 0,
            },
            "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8},
            "dataset_manifest": {"total_examples": 4},
            "runtime_paths": {"config_path": "config.json", "checkpoint_path": "checkpoint.pt", "nodes_path": "nodes.jsonl", "label_map_path": "label_map.json", "source_spans_paths": ["spans.jsonl"], "cache_paths": ["cache.pt"]},
            "proposals": [{"area": "balanced", "priority": 5, "reason": "test"}],
        },
    )

    artifact, action, artifact_kind = generate_candidate_artifact(
        config=config,
        planner=ImprovementPlanner(memory_root=tmp_path / "episodes"),
        subsystem="tolbert_model",
        metrics=EvalMetrics(total=10, passed=8),
        generation_kwargs={"focus": "balanced"},
        candidate_artifact_path=candidate_artifact_path,
    )

    assert artifact == str(candidate_artifact_path)
    assert action == "propose_tolbert_model_update"
    assert artifact_kind == "tolbert_model_bundle"


def test_tolbert_kernel_autobuild_ready_requires_synthetic_dataset_threshold():
    ready, detail = tolbert_kernel_autobuild_ready(
        {
            "artifact_kind": "tolbert_model_bundle",
            "lifecycle_state": "retained",
            "build_policy": {
                "allow_kernel_autobuild": True,
                "allow_kernel_rebuild": False,
                "require_synthetic_dataset": True,
                "min_total_examples": 10,
                "min_synthetic_examples": 3,
            },
            "dataset_manifest": {
                "total_examples": 12,
                "synthetic_trajectory_examples": 2,
            },
        }
    )

    assert ready is False
    assert "synthetic dataset threshold not met" in detail


def test_tolbert_kernel_autobuild_ready_requires_head_target_thresholds():
    ready, detail = tolbert_kernel_autobuild_ready(
        {
            "artifact_kind": "tolbert_model_bundle",
            "lifecycle_state": "retained",
            "build_policy": {
                "allow_kernel_autobuild": True,
                "allow_kernel_rebuild": False,
                "require_synthetic_dataset": False,
                "require_head_targets": True,
                "min_total_examples": 10,
                "min_synthetic_examples": 0,
                "min_policy_examples": 8,
                "min_transition_examples": 8,
                "min_value_examples": 8,
                "min_stop_examples": 4,
            },
            "dataset_manifest": {
                "total_examples": 12,
                "synthetic_trajectory_examples": 0,
                "policy_examples": 8,
                "transition_examples": 7,
                "value_examples": 8,
                "stop_examples": 4,
            },
        }
    )

    assert ready is False
    assert "head target threshold not met" in detail
    assert "transition_examples=7" in detail


def test_improvement_planner_ranks_multiple_candidates(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": False,
                "summary": {"failure_types": ["missing_expected_file"]},
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=episodes)
    metrics = EvalMetrics(
        total=10,
        passed=7,
        low_confidence_episodes=3,
        trusted_retrieval_steps=1,
        total_by_memory_source={},
        skill_selected_steps=6,
        retrieval_ranked_skill_steps=1,
        generated_total=10,
        generated_passed=5,
    )

    ranked = planner.rank_targets(metrics)

    ranked_subsystems = {candidate.subsystem for candidate in ranked}
    assert {"benchmark", "retrieval", "verifier", "curriculum", "operators"}.issubset(ranked_subsystems)
    assert ranked[-1].subsystem == "policy"


def test_improvement_planner_prefers_verifier_when_verifier_lane_absent(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
    )

    target = planner.choose_target(metrics)

    assert target.subsystem == "verifier"


def test_improvement_planner_surfaces_capabilities_when_registry_lacks_improvement_surfaces(tmp_path: Path):
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
                        "settings": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    capabilities = next(candidate for candidate in experiments if candidate.subsystem == "capabilities")
    assert capabilities.evidence["enabled_module_count"] == 1
    assert capabilities.evidence["improvement_surface_count"] == 0


def test_improvement_planner_surfaces_world_model_for_repo_workflow_failures(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / "repo_task.json").write_text(
        json.dumps(
            {
                "task_id": "repo_task",
                "success": False,
                "summary": {"failure_types": ["missing_expected_file", "command_failure"]},
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=episodes)
    metrics = EvalMetrics(
        total=10,
        passed=7,
        trusted_retrieval_steps=5,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
        total_by_benchmark_family={"repo_sandbox": 4},
    )

    experiments = planner.rank_experiments(metrics)

    world_model = next(candidate for candidate in experiments if candidate.subsystem == "world_model")
    assert world_model.evidence["repo_world_model_total"] == 4


def test_improvement_planner_surfaces_transition_failures_for_recursive_optimization(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / "repo_task.json").write_text(
        json.dumps(
            {
                "task_id": "repo_task",
                "success": False,
                "summary": {
                    "failure_types": [],
                    "transition_failures": ["no_state_progress", "state_regression"],
                    "final_completion_ratio": 0.2,
                    "net_state_progress_delta": 0.0,
                    "state_regression_steps": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=episodes)
    metrics = EvalMetrics(
        total=10,
        passed=6,
        trusted_retrieval_steps=5,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
        total_by_benchmark_family={"repo_sandbox": 4},
        generated_total=10,
        generated_passed=4,
    )

    assert planner.transition_failure_counts()["no_state_progress"] == 1
    experiments = planner.rank_experiments(metrics)

    world_model = next(candidate for candidate in experiments if candidate.subsystem == "world_model")
    curriculum = next(candidate for candidate in experiments if candidate.subsystem == "curriculum")
    state_estimation = next(candidate for candidate in experiments if candidate.subsystem == "state_estimation")
    transition_model = next(candidate for candidate in experiments if candidate.subsystem == "transition_model")
    assert world_model.evidence["transition_failure_counts"]["state_regression"] == 1
    assert curriculum.evidence["transition_failure_counts"]["no_state_progress"] == 1
    assert state_estimation.evidence["transition_failure_counts"]["state_regression"] == 1
    assert transition_model.evidence["transition_failure_counts"]["state_regression"] == 1


def test_build_world_model_proposal_artifact_emits_retained_runtime_controls():
    artifact = build_world_model_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=2),
        {"missing_expected_file": 2, "command_failure": 1},
        focus="workflow_alignment",
    )

    assert artifact["artifact_kind"] == "world_model_policy_set"
    assert artifact["generation_focus"] == "workflow_alignment"
    assert artifact["controls"]["workflow_changed_path_score_weight"] >= 4
    assert artifact["planning_controls"]["append_preservation_subgoal"] is True


def test_build_world_model_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_world_model_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=0),
        {},
        current_payload={
            "artifact_kind": "world_model_policy_set",
            "lifecycle_state": "retained",
            "controls": {"expected_artifact_score_weight": 7},
            "planning_controls": {"max_preserved_artifacts": 6},
        },
    )

    assert artifact["controls"]["expected_artifact_score_weight"] == 7
    assert artifact["planning_controls"]["max_preserved_artifacts"] == 6


def test_improvement_planner_surfaces_universe_for_governance_failures(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.failure_counts = lambda: {"command_failure": 2, "missing_expected_file": 0}
    planner.transition_failure_counts = lambda: {"no_state_progress": 1, "state_regression": 1}
    planner.transition_summary = lambda: {"state_regression_steps": 1}
    planner.environment_violation_summary = lambda: {
        "violation_counts": {"network_access_conflict": 2, "path_scope_conflict": 1},
        "alignment_failure_counts": {"network_access_aligned": 2},
        "observed_environment_modes": {"network_access_mode": {"allowlist_only": 2}},
        "violation_total": 3,
        "alignment_failure_total": 2,
    }
    metrics = EvalMetrics(
        total=10,
        passed=7,
        low_confidence_episodes=2,
        trusted_retrieval_steps=8,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    constitution = next(candidate for candidate in experiments if candidate.subsystem == "universe_constitution")
    envelope = next(candidate for candidate in experiments if candidate.subsystem == "operating_envelope")
    assert constitution.evidence["failure_counts"]["command_failure"] == 2
    assert constitution.evidence["transition_failure_counts"]["state_regression"] == 1
    assert constitution.evidence["low_confidence_episodes"] == 2
    assert envelope.evidence["environment_violation_summary"]["violation_counts"]["network_access_conflict"] == 2


def test_build_universe_contract_artifact_emits_retained_runtime_controls():
    artifact = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=7, low_confidence_episodes=2),
        {"command_failure": 2, "state_regression": 1},
        environment_violation_summary={
            "violation_counts": {"path_scope_conflict": 2},
            "alignment_failure_counts": {"network_access_aligned": 2},
            "observed_environment_modes": {"network_access_mode": {"allowlist_only": 2}},
        },
        focus="governance",
    )

    assert artifact["artifact_kind"] == "universe_contract"
    assert artifact["generation_focus"] == "governance"
    assert artifact["control_schema"] == "universe_contract_v1"
    assert artifact["governance"]["require_verification"] is True
    assert artifact["invariants"]
    assert artifact["forbidden_command_patterns"]
    assert artifact["preferred_command_prefixes"]
    assert artifact["action_risk_controls"]["destructive_mutation_penalty"] >= 14
    assert artifact["action_risk_controls"]["scope_escape_penalty"] >= 11
    assert artifact["action_risk_controls"]["verification_bonus"] >= 4
    assert artifact["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert artifact["environment_assumptions"]["require_path_scoped_mutations"] is True


def test_build_universe_contract_artifact_calibrates_environment_assumptions_from_observed_modes():
    artifact = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=8),
        {},
        environment_violation_summary={
            "violation_counts": {},
            "alignment_failure_counts": {
                "network_access_aligned": 3,
                "git_write_aligned": 2,
            },
            "observed_environment_modes": {
                "network_access_mode": {"allowlist_only": 3},
                "git_write_mode": {"task_scoped": 2},
                "workspace_write_scope": {"generated_only": 1},
            },
        },
        current_payload={
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "environment_assumptions": {
                "network_access_mode": "blocked",
                "git_write_mode": "operator_gated",
                "workspace_write_scope": "task_only",
                "require_path_scoped_mutations": True,
                "require_rollback_on_mutation": True,
            },
        },
    )

    assert artifact["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert artifact["environment_assumptions"]["git_write_mode"] == "task_scoped"


def test_improvement_planner_summarizes_retained_universe_cycle_feedback(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    artifact_one = tmp_path / "artifacts" / "universe_one.json"
    artifact_two = tmp_path / "artifacts" / "universe_two.json"
    artifact_one.parent.mkdir(parents=True, exist_ok=True)
    artifact_one.write_text(
        json.dumps(
            _retained_universe_artifact(
                network_access_mode="allowlist_only",
                git_write_mode="operator_gated",
                scope_escape_penalty=11,
                network_fetch_penalty=6,
            )
        ),
        encoding="utf-8",
    )
    artifact_two.write_text(
        json.dumps(
            _retained_universe_artifact(
                network_access_mode="allowlist_only",
                git_write_mode="operator_gated",
                scope_escape_penalty=13,
                network_fetch_penalty=7,
            )
        ),
        encoding="utf-8",
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:universe:1",
            state="retain",
            subsystem="universe",
            action="finalize_cycle",
            artifact_path=str(artifact_one),
            artifact_kind="universe_contract",
            reason="first retained universe calibration",
            metrics_summary={
                "family_pass_rate_delta": {"workflow": 0.1, "tooling": 0.05},
            },
            active_artifact_path=str(artifact_one),
            selected_variant_id="environment_envelope",
            baseline_pass_rate=0.70,
            candidate_pass_rate=0.80,
            baseline_average_steps=1.6,
            candidate_average_steps=1.4,
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:universe:2",
            state="retain",
            subsystem="universe",
            action="finalize_cycle",
            artifact_path=str(artifact_two),
            artifact_kind="universe_contract",
            reason="second retained universe calibration",
            metrics_summary={
                "family_pass_rate_delta": {"workflow": 0.1, "tooling": 0.05},
            },
            active_artifact_path=str(artifact_two),
            selected_variant_id="environment_envelope",
            baseline_pass_rate=0.80,
            candidate_pass_rate=0.90,
            baseline_average_steps=1.5,
            candidate_average_steps=1.2,
        ),
    )

    summary = planner.universe_cycle_feedback_summary()

    assert summary["retained_cycle_count"] == 2
    assert summary["selected_variant_counts"] == {"environment_envelope": 2}
    assert summary["selected_variant_weights"]["environment_envelope"] > 0.0
    assert summary["successful_environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert summary["successful_environment_assumptions"]["git_write_mode"] == "operator_gated"
    assert summary["successful_environment_assumption_weights"]["network_access_mode"]["allowlist_only"] > 0.0
    assert summary["successful_action_risk_control_floor"]["scope_escape_penalty"] == 13
    assert summary["successful_action_risk_control_floor"]["network_fetch_penalty"] == 7
    assert summary["successful_action_risk_control_weighted_mean"]["scope_escape_penalty"] >= 12.0
    assert summary["average_retained_pass_rate_delta"] == 0.1
    assert summary["average_retained_step_delta"] == -0.25
    assert summary["broad_support_cycle_count"] == 2


def test_build_universe_contract_artifact_reuses_successful_cycle_feedback_priors():
    artifact = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=8),
        {},
        cycle_feedback_summary={
            "retained_cycle_count": 3,
            "selected_variant_counts": {"environment_envelope": 2, "governance": 1},
            "selected_variant_weights": {"environment_envelope": 8.0, "governance": 1.5},
            "successful_environment_assumptions": {
                "network_access_mode": "allowlist_only",
                "git_write_mode": "operator_gated",
                "workspace_write_scope": "task_only",
            },
            "successful_action_risk_control_floor": {
                "scope_escape_penalty": 14,
                "network_fetch_penalty": 7,
            },
            "successful_action_risk_control_weighted_mean": {
                "scope_escape_penalty": 13.5,
                "network_fetch_penalty": 6.6,
            },
        },
        current_payload=_retained_universe_artifact(
            network_access_mode="blocked",
            git_write_mode="blocked",
            workspace_write_scope="generated_only",
            scope_escape_penalty=9,
            network_fetch_penalty=4,
        ),
    )

    assert artifact["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert artifact["environment_assumptions"]["git_write_mode"] == "operator_gated"
    assert artifact["environment_assumptions"]["workspace_write_scope"] == "task_only"
    assert artifact["action_risk_controls"]["scope_escape_penalty"] == 14
    assert artifact["action_risk_controls"]["network_fetch_penalty"] == 7
    assert any(
        proposal["area"] == "environment_envelope" and "dominant variant=environment_envelope" in proposal["suggestion"]
        for proposal in artifact["proposals"]
    )


def test_assess_artifact_compatibility_accepts_split_operating_envelope_artifact():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "operating_envelope",
        "lifecycle_state": "retained",
        "control_schema": "operating_envelope_v1",
        "retention_gate": {"require_universe_improvement": True},
        "action_risk_controls": {
            "destructive_mutation_penalty": 12,
            "git_mutation_penalty": 8,
            "inline_destructive_interpreter_penalty": 8,
            "network_fetch_penalty": 6,
            "privileged_command_penalty": 10,
            "read_only_discovery_bonus": 3,
            "remote_execution_penalty": 10,
            "reversible_file_operation_bonus": 2,
            "scope_escape_penalty": 11,
            "unbounded_execution_penalty": 7,
            "verification_bonus": 4,
        },
        "environment_assumptions": {
            "network_access_mode": "allowlist_only",
            "git_write_mode": "operator_gated",
            "workspace_write_scope": "task_only",
            "require_path_scoped_mutations": True,
            "require_rollback_on_mutation": True,
        },
        "proposals": [{"area": "environment_envelope", "priority": 5, "reason": "test", "suggestion": "test"}],
    }

    report = assess_artifact_compatibility(subsystem="universe", payload=payload)

    assert report["compatible"] is True


def test_generate_candidate_artifact_supports_split_universe_subsystems(tmp_path: Path):
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        universe_constitution_path=tmp_path / "universe" / "universe_constitution.json",
        operating_envelope_path=tmp_path / "universe" / "operating_envelope.json",
    )
    metrics = EvalMetrics(total=10, passed=7, low_confidence_episodes=2)

    class PlannerStub:
        @staticmethod
        def failure_counts():
            return {"command_failure": 2, "state_regression": 1}

        @staticmethod
        def environment_violation_summary():
            return {
                "violation_counts": {"network_access_conflict": 2},
                "alignment_failure_counts": {"network_access_aligned": 2},
                "observed_environment_modes": {"network_access_mode": {"allowlist_only": 2}},
            }

        @staticmethod
        def universe_cycle_feedback_summary():
            return {
                "retained_cycle_count": 2,
                "selected_variant_weights": {"environment_envelope": 4.0},
                "successful_environment_assumptions": {"network_access_mode": "allowlist_only"},
                "successful_environment_assumption_weights": {"network_access_mode": {"allowlist_only": 4.0}},
            }

    constitution_candidate = tmp_path / "candidates" / "constitution.json"
    envelope_candidate = tmp_path / "candidates" / "envelope.json"
    constitution_candidate.parent.mkdir(parents=True, exist_ok=True)

    constitution_artifact, constitution_action, constitution_kind = generate_candidate_artifact(
        config=config,
        planner=PlannerStub(),
        subsystem="universe_constitution",
        metrics=metrics,
        generation_kwargs={"focus": "governance"},
        candidate_artifact_path=constitution_candidate,
    )
    envelope_artifact, envelope_action, envelope_kind = generate_candidate_artifact(
        config=config,
        planner=PlannerStub(),
        subsystem="operating_envelope",
        metrics=metrics,
        generation_kwargs={"focus": "environment_envelope"},
        candidate_artifact_path=envelope_candidate,
    )

    constitution_payload = json.loads(constitution_candidate.read_text(encoding="utf-8"))
    envelope_payload = json.loads(envelope_candidate.read_text(encoding="utf-8"))

    assert constitution_artifact == str(constitution_candidate)
    assert constitution_action == "propose_universe_constitution_update"
    assert constitution_kind == "universe_constitution"
    assert constitution_payload["artifact_kind"] == "universe_constitution"
    assert constitution_payload["control_schema"] == "universe_constitution_v1"
    assert "governance" in constitution_payload

    assert envelope_artifact == str(envelope_candidate)
    assert envelope_action == "propose_operating_envelope_update"
    assert envelope_kind == "operating_envelope"
    assert envelope_payload["artifact_kind"] == "operating_envelope"
    assert envelope_payload["control_schema"] == "operating_envelope_v1"
    assert envelope_payload["environment_assumptions"]["network_access_mode"] == "allowlist_only"


def test_generate_candidate_artifact_uses_split_universe_bundle_for_legacy_universe_lane(tmp_path: Path):
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        universe_contract_path=tmp_path / "universe" / "universe_contract.json",
        universe_constitution_path=tmp_path / "universe" / "universe_constitution.json",
        operating_envelope_path=tmp_path / "universe" / "operating_envelope.json",
    )
    config.universe_contract_path.parent.mkdir(parents=True, exist_ok=True)
    config.universe_contract_path.write_text(
        json.dumps(
            _retained_universe_artifact(network_access_mode="blocked", git_write_mode="blocked")
            | {"invariants": ["stale combined invariant"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    config.universe_constitution_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "retained",
                "control_schema": "universe_constitution_v1",
                "retention_gate": {"require_non_regression": True},
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "invariants": ["split constitution invariant"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest -q"],
                "proposals": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config.operating_envelope_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "operating_envelope",
                "lifecycle_state": "retained",
                "control_schema": "operating_envelope_v1",
                "retention_gate": {"require_non_regression": True},
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 8,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 6,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 10,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 11,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.github.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["python", "pytest"],
                "learned_calibration_priors": {},
                "proposals": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = EvalMetrics(total=10, passed=7, low_confidence_episodes=2)

    class PlannerStub:
        @staticmethod
        def failure_counts():
            return {"command_failure": 2, "state_regression": 1}

        @staticmethod
        def environment_violation_summary():
            return {}

        @staticmethod
        def universe_cycle_feedback_summary():
            return {}

    candidate_path = tmp_path / "candidates" / "universe.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    artifact, action, kind = generate_candidate_artifact(
        config=config,
        planner=PlannerStub(),
        subsystem="universe",
        metrics=metrics,
        generation_kwargs={"focus": "environment_envelope"},
        candidate_artifact_path=candidate_path,
    )

    payload = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert artifact == str(candidate_path)
    assert action == "propose_universe_update"
    assert kind == "universe_contract"
    assert "split constitution invariant" in payload["invariants"]
    assert "stale combined invariant" not in payload["invariants"]
    assert payload["environment_assumptions"]["network_access_mode"] == "allowlist_only"


def test_build_state_estimation_proposal_artifact_emits_retained_runtime_controls():
    artifact = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=7, low_confidence_episodes=2, trusted_retrieval_steps=3),
        {
            "state_regression_steps": 2,
            "state_progress_gain_steps": 0,
            "average_net_state_progress_delta": -0.05,
        },
        focus="risk_sensitivity",
    )

    assert artifact["artifact_kind"] == "state_estimation_policy_set"
    assert artifact["generation_focus"] == "risk_sensitivity"
    assert artifact["control_schema"] == "state_estimation_controls_v1"
    assert artifact["controls"]["regression_path_budget"] >= 8
    assert artifact["latent_controls"]["active_path_budget"] >= 8
    assert artifact["policy_controls"]["regressive_path_match_bonus"] >= 3


def test_build_state_estimation_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {"state_regression_steps": 0, "state_progress_gain_steps": 1},
        current_payload={
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "controls": {"regression_path_budget": 9},
            "latent_controls": {"active_path_budget": 10},
            "policy_controls": {"blocked_command_bonus": 3},
        },
    )

    assert artifact["controls"]["regression_path_budget"] == 9
    assert artifact["latent_controls"]["active_path_budget"] == 10
    assert artifact["policy_controls"]["blocked_command_bonus"] == 3


def test_build_transition_model_proposal_artifact_emits_signatures(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("cleanup_task.json").write_text(
        json.dumps(
            {
                "task_id": "cleanup_task",
                "success": False,
                "summary": {"transition_failures": ["no_state_progress"]},
                "fragments": [
                    {"kind": "command", "step_index": 1, "command": "false", "passed": False},
                    {
                        "kind": "failure",
                        "step_index": 1,
                        "reason": "repeated stalled command",
                        "failure_types": [],
                        "failure_signals": ["no_state_progress"],
                    },
                    {
                        "kind": "state_transition",
                        "step_index": 1,
                        "progress_delta": 0.0,
                        "regressions": [],
                        "cleared_forbidden_artifacts": [],
                        "newly_materialized_expected_artifacts": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_transition_model_proposal_artifact(episodes, focus="repeat_avoidance")

    assert artifact["artifact_kind"] == "transition_model_policy_set"
    assert artifact["generation_focus"] == "repeat_avoidance"
    assert artifact["controls"]["repeat_command_penalty"] >= 6
    assert artifact["signatures"][0]["signal"] == "no_state_progress"
    assert artifact["signatures"][0]["command"] == "false"


def test_build_transition_model_proposal_artifact_merges_same_pattern_signatures(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("config_task_a.json").write_text(
        json.dumps(
            {
                "task_id": "config_task_a",
                "success": False,
                "summary": {"transition_failures": ["no_state_progress"]},
                "fragments": [
                    {"kind": "command", "step_index": 1, "command": "echo 'ENV=base' > config/base.env", "passed": False},
                    {
                        "kind": "failure",
                        "step_index": 1,
                        "reason": "stalled config write",
                        "failure_types": [],
                        "failure_signals": ["no_state_progress"],
                    },
                    {
                        "kind": "state_transition",
                        "step_index": 1,
                        "progress_delta": 0.0,
                        "regressions": [],
                        "cleared_forbidden_artifacts": [],
                        "newly_materialized_expected_artifacts": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    episodes.joinpath("config_task_b.json").write_text(
        json.dumps(
            {
                "task_id": "config_task_b",
                "success": False,
                "summary": {"transition_failures": ["no_state_progress"]},
                "fragments": [
                    {"kind": "command", "step_index": 1, "command": "echo 'ENV=prod' > config/prod.env", "passed": False},
                    {
                        "kind": "failure",
                        "step_index": 1,
                        "reason": "stalled config write",
                        "failure_types": [],
                        "failure_signals": ["no_state_progress"],
                    },
                    {
                        "kind": "state_transition",
                        "step_index": 1,
                        "progress_delta": 0.0,
                        "regressions": [],
                        "cleared_forbidden_artifacts": [],
                        "newly_materialized_expected_artifacts": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_transition_model_proposal_artifact(episodes, focus="repeat_avoidance")

    assert artifact["signatures"][0]["signal"] == "no_state_progress"
    assert artifact["signatures"][0]["command_pattern"] == "echo <str> > <path>"
    assert artifact["signatures"][0]["support"] == 2


def test_assess_artifact_compatibility_accepts_valid_transition_model_artifact(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("cleanup_task.json").write_text(
        json.dumps(
            {
                "task_id": "cleanup_task",
                "success": False,
                "summary": {"transition_failures": ["state_regression"], "executed_commands": ["printf 'x\\n' > keep.txt"]},
            }
        ),
        encoding="utf-8",
    )

    payload = build_transition_model_proposal_artifact(episodes)

    report = assess_artifact_compatibility(subsystem="transition_model", payload=payload)

    assert report["compatible"] is True


def test_improvement_planner_surfaces_trust_for_restricted_unattended_ledger(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = reports_dir / "unattended_trust_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "reports_considered": 4,
                "overall_summary": {"distinct_benchmark_families": 1},
                "gated_summary": {
                    "success_rate": 0.5,
                    "unsafe_ambiguous_rate": 0.25,
                    "hidden_side_effect_risk_rate": 0.25,
                    "success_hidden_side_effect_risk_rate": 0.0,
                },
                "overall_assessment": {"status": "restricted", "passed": False},
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", trust_ledger_path=ledger_path)
    metrics = EvalMetrics(
        total=10,
        passed=8,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    trust = next(candidate for candidate in experiments if candidate.subsystem == "trust")
    assert trust.evidence["overall_status"] == "restricted"
    assert trust.evidence["unsafe_ambiguous_rate"] == 0.25


def test_build_trust_proposal_artifact_emits_retained_runtime_controls():
    artifact = build_trust_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {
            "overall_summary": {"distinct_benchmark_families": 1},
            "gated_summary": {"unsafe_ambiguous_rate": 0.2, "hidden_side_effect_risk_rate": 0.1},
            "overall_assessment": {"status": "restricted", "passed": False},
        },
        focus="safety",
    )

    assert artifact["artifact_kind"] == "trust_policy_set"
    assert artifact["generation_focus"] == "safety"
    assert artifact["controls"]["min_success_rate"] >= 0.8
    assert artifact["controls"]["max_hidden_side_effect_rate"] <= 0.05


def test_build_trust_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_trust_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {},
        current_payload={
            "artifact_kind": "trust_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "recent_report_limit": 80,
                "required_benchmark_families": ["micro", "repo_chore"],
                "bootstrap_min_reports": 7,
                "breadth_min_reports": 14,
                "min_distinct_families": 3,
                "min_success_rate": 0.82,
                "max_unsafe_ambiguous_rate": 0.04,
                "max_hidden_side_effect_rate": 0.03,
                "max_success_hidden_side_effect_rate": 0.01,
            },
        },
    )

    assert artifact["controls"]["recent_report_limit"] == 80
    assert artifact["controls"]["min_success_rate"] == 0.82


def test_improvement_planner_surfaces_recovery_for_unattended_rollback_risk(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = reports_dir / "unattended_trust_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "overall_summary": {
                    "rollback_performed_rate": 0.2,
                    "hidden_side_effect_risk_rate": 0.1,
                },
                "gated_summary": {
                    "rollback_performed_rate": 0.25,
                    "hidden_side_effect_risk_rate": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", trust_ledger_path=ledger_path)
    metrics = EvalMetrics(
        total=10,
        passed=9,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    recovery = next(candidate for candidate in experiments if candidate.subsystem == "recovery")
    assert recovery.evidence["rollback_performed_rate"] == 0.25
    assert recovery.evidence["hidden_side_effect_risk_rate"] == 0.05


def test_build_recovery_proposal_artifact_emits_retained_runtime_controls():
    artifact = build_recovery_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {
            "overall_summary": {
                "rollback_performed_rate": 0.2,
                "hidden_side_effect_risk_rate": 0.1,
            }
        },
        focus="rollback_safety",
    )

    assert artifact["artifact_kind"] == "recovery_policy_set"
    assert artifact["generation_focus"] == "rollback_safety"
    assert artifact["control_schema"] == "workspace_recovery_controls_v1"
    assert artifact["controls"]["rollback_on_safe_stop"] is True
    assert artifact["controls"]["verify_post_rollback_file_count"] is True


def test_build_recovery_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_recovery_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {},
        current_payload={
            "artifact_kind": "recovery_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "snapshot_before_execution": True,
                "rollback_on_runner_exception": True,
                "rollback_on_failed_outcome": True,
                "rollback_on_safe_stop": True,
                "verify_post_rollback_file_count": True,
                "max_post_rollback_file_count": 2,
            },
        },
    )

    assert artifact["controls"]["rollback_on_safe_stop"] is True
    assert artifact["controls"]["max_post_rollback_file_count"] == 2


def test_improvement_planner_surfaces_delegation_for_throttled_runtime(tmp_path: Path):
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        runtime_config=KernelConfig(
            delegated_job_max_concurrency=1,
            delegated_job_max_subprocesses_per_job=1,
            max_steps=5,
        ),
    )
    metrics = EvalMetrics(
        total=10,
        passed=9,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    delegation = next(candidate for candidate in experiments if candidate.subsystem == "delegation")
    assert delegation.evidence["delegated_job_max_concurrency"] == 1
    assert delegation.evidence["max_steps"] == 5


def test_build_delegation_proposal_artifact_emits_retained_runtime_controls():
    artifact = build_delegation_proposal_artifact(
        KernelConfig(
            delegated_job_max_concurrency=1,
            delegated_job_max_subprocesses_per_job=1,
            max_steps=5,
        ),
        focus="worker_depth",
    )

    assert artifact["artifact_kind"] == "delegated_runtime_policy_set"
    assert artifact["generation_focus"] == "worker_depth"
    assert artifact["control_schema"] == "delegated_resource_controls_v1"
    assert artifact["controls"]["delegated_job_max_subprocesses_per_job"] >= 2
    assert artifact["controls"]["max_steps"] >= 7


def test_build_delegation_proposal_artifact_uses_retained_runtime_snapshot(tmp_path: Path):
    delegation_path = tmp_path / "delegation" / "delegation_proposals.json"
    delegation_path.parent.mkdir(parents=True, exist_ok=True)
    delegation_path.write_text(
        json.dumps(
            {
                "artifact_kind": "delegated_runtime_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "delegated_job_max_concurrency": 3,
                    "delegated_job_max_active_per_budget_group": 2,
                    "delegated_job_max_queued_per_budget_group": 4,
                    "delegated_job_max_artifact_bytes": 8388608,
                    "delegated_job_max_subprocesses_per_job": 2,
                    "command_timeout_seconds": 30,
                    "llm_timeout_seconds": 30,
                    "max_steps": 9,
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_delegation_proposal_artifact(
        KernelConfig(
            delegated_job_max_concurrency=1,
            delegated_job_max_subprocesses_per_job=1,
            max_steps=5,
            delegation_proposals_path=delegation_path,
        ),
        focus="worker_depth",
    )

    assert artifact["controls"]["delegated_job_max_subprocesses_per_job"] == 3
    assert artifact["controls"]["max_steps"] == 11


def test_assess_artifact_compatibility_accepts_valid_world_model_artifact():
    payload = build_world_model_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=2),
        {"missing_expected_file": 2, "command_failure": 1},
        focus="workflow_alignment",
    )

    report = assess_artifact_compatibility(subsystem="world_model", payload=payload)

    assert report["compatible"] is True


def test_assess_artifact_compatibility_accepts_valid_universe_artifact():
    payload = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=1),
        {"command_failure": 1, "state_regression": 1},
        focus="verification",
    )

    report = assess_artifact_compatibility(subsystem="universe", payload=payload)

    assert report["compatible"] is True


def test_assess_artifact_compatibility_accepts_valid_state_estimation_artifact():
    payload = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=7, low_confidence_episodes=1),
        {"state_regression_steps": 1, "state_progress_gain_steps": 0},
        focus="recovery_bias",
    )

    report = assess_artifact_compatibility(subsystem="state_estimation", payload=payload)

    assert report["compatible"] is True


def test_assess_artifact_compatibility_accepts_valid_trust_artifact():
    payload = build_trust_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {
            "overall_summary": {"distinct_benchmark_families": 1},
            "gated_summary": {"unsafe_ambiguous_rate": 0.2},
            "overall_assessment": {"status": "restricted", "passed": False},
        },
    )

    report = assess_artifact_compatibility(subsystem="trust", payload=payload)

    assert report["compatible"] is True


def test_assess_artifact_compatibility_accepts_valid_recovery_artifact():
    payload = build_recovery_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {},
    )

    report = assess_artifact_compatibility(subsystem="recovery", payload=payload)

    assert report["compatible"] is True


def test_assess_artifact_compatibility_accepts_valid_delegation_artifact():
    payload = build_delegation_proposal_artifact(KernelConfig())

    report = assess_artifact_compatibility(subsystem="delegation", payload=payload)

    assert report["compatible"] is True


def test_build_operator_policy_proposal_artifact_uses_retained_runtime_snapshot(tmp_path: Path):
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
                    "unattended_http_allowed_hosts": ["example.com"],
                    "unattended_http_timeout_seconds": 20,
                    "unattended_http_max_body_bytes": 131072,
                    "unattended_allow_generated_path_mutations": False,
                    "unattended_generated_path_prefixes": ["build"],
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_operator_policy_proposal_artifact(
        KernelConfig(
            unattended_allowed_benchmark_families=("micro",),
            unattended_allow_git_commands=False,
            unattended_allow_http_requests=False,
            unattended_http_allowed_hosts=(),
            unattended_allow_generated_path_mutations=False,
            operator_policy_proposals_path=operator_policy_path,
        ),
        focus="git_http_scope",
    )

    assert artifact["controls"]["unattended_allow_git_commands"] is True
    assert artifact["controls"]["unattended_http_allowed_hosts"] == ["api.github.com", "example.com"]
    assert artifact["controls"]["unattended_http_timeout_seconds"] == 20


def test_build_prompt_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_prompt_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=0),
        {},
        current_payload={
            "artifact_kind": "prompt_proposal_set",
            "lifecycle_state": "retained",
            "controls": {"verifier_alignment_bias": 5},
            "planner_controls": {"max_initial_subgoals": 7},
            "improvement_planner_controls": {"portfolio_exploration_bonus": 0.02},
            "role_directives": {"planner": "Preserve retained directive."},
        },
    )

    assert artifact["controls"]["verifier_alignment_bias"] == 5
    assert artifact["planner_controls"]["max_initial_subgoals"] == 7
    assert artifact["improvement_planner_controls"]["portfolio_exploration_bonus"] == 0.02
    assert artifact["role_directives"]["planner"] == "Preserve retained directive."


def test_build_curriculum_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_curriculum_proposal_artifact(
        EvalMetrics(total=10, passed=8, generated_total=0),
        current_payload={
            "artifact_kind": "curriculum_proposal_set",
            "lifecycle_state": "retained",
            "controls": {
                "success_reference_limit": 5,
                "max_generated_adjacent_tasks": 6,
                "max_generated_failure_recovery_tasks": 7,
            },
        },
    )

    assert artifact["controls"]["success_reference_limit"] == 5
    assert artifact["controls"]["max_generated_adjacent_tasks"] == 6
    assert artifact["controls"]["max_generated_failure_recovery_tasks"] == 7


def test_build_retrieval_proposal_artifact_uses_retained_runtime_snapshot():
    artifact = build_retrieval_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=0, trusted_retrieval_steps=10),
        current_payload={
            "artifact_kind": "retrieval_policy_set",
            "lifecycle_state": "retained",
            "overrides": {"tolbert_context_char_budget": 3000},
            "asset_controls": {"max_episode_step_spans_per_task": 5},
        },
    )

    assert artifact["overrides"]["tolbert_context_char_budget"] == 3000
    assert artifact["asset_controls"]["max_episode_step_spans_per_task"] == 5


def test_assess_artifact_compatibility_rejects_invalid_world_model_controls():
    payload = build_world_model_proposal_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=2),
        {"missing_expected_file": 2},
        focus="workflow_alignment",
    )
    payload["controls"]["unsupported_control"] = 1
    payload["planning_controls"]["append_preservation_subgoal"] = "yes"

    report = assess_artifact_compatibility(subsystem="world_model", payload=payload)

    assert report["compatible"] is False
    assert "world_model control is unsupported: unsupported_control" in report["violations"]
    assert "world_model planning control append_preservation_subgoal must be boolean" in report["violations"]


def test_assess_artifact_compatibility_rejects_invalid_universe_controls():
    payload = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=1),
        {"command_failure": 1},
    )
    payload["governance"]["require_verification"] = "yes"
    payload["governance"]["unsupported_control"] = True
    payload["action_risk_controls"]["verification_bonus"] = 0
    payload["action_risk_controls"]["unsupported_control"] = 1
    payload["environment_assumptions"]["network_access_mode"] = "wide_open"
    payload["environment_assumptions"]["require_path_scoped_mutations"] = "yes"
    payload["invariants"] = []
    payload["forbidden_command_patterns"] = []
    payload["preferred_command_prefixes"] = []

    report = assess_artifact_compatibility(subsystem="universe", payload=payload)

    assert report["compatible"] is False
    assert "universe governance control require_verification must be boolean" in report["violations"]
    assert "universe governance control is unsupported: unsupported_control" in report["violations"]
    assert "universe action risk control verification_bonus must be a positive integer" in report["violations"]
    assert "universe action risk control is unsupported: unsupported_control" in report["violations"]
    assert "universe environment assumption network_access_mode must be one of ['allowlist_only', 'blocked', 'open']" in report["violations"]
    assert "universe environment assumption require_path_scoped_mutations must be boolean" in report["violations"]
    assert "universe artifact must contain a non-empty invariants list" in report["violations"]
    assert "universe artifact must contain a non-empty forbidden_command_patterns list" in report["violations"]
    assert "universe artifact must contain a non-empty preferred_command_prefixes list" in report["violations"]


def test_assess_artifact_compatibility_rejects_invalid_state_estimation_controls():
    payload = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=8),
        {"state_regression_steps": 1, "state_progress_gain_steps": 0},
    )
    payload["controls"]["unsupported_control"] = 1
    payload["latent_controls"]["active_path_budget"] = 0
    payload["policy_controls"]["blocked_command_bonus"] = -1

    report = assess_artifact_compatibility(subsystem="state_estimation", payload=payload)

    assert report["compatible"] is False
    assert "state_estimation control is unsupported: unsupported_control" in report["violations"]
    assert "state_estimation latent control active_path_budget must be a positive integer" in report["violations"]
    assert "state_estimation policy control blocked_command_bonus must be a non-negative integer" in report["violations"]


def test_assess_artifact_compatibility_rejects_invalid_trust_controls():
    payload = build_trust_proposal_artifact(EvalMetrics(total=10, passed=8), {})
    payload["controls"]["min_success_rate"] = 2.0
    payload["controls"]["required_benchmark_families"] = []

    report = assess_artifact_compatibility(subsystem="trust", payload=payload)

    assert report["compatible"] is False
    assert "trust control min_success_rate must stay within [0.0, 1.0]" in report["violations"]
    assert "trust control required_benchmark_families must be a non-empty list" in report["violations"]


def test_assess_artifact_compatibility_rejects_invalid_recovery_controls():
    payload = build_recovery_proposal_artifact(EvalMetrics(total=10, passed=8), {})
    payload["controls"]["snapshot_before_execution"] = "yes"
    payload["controls"]["max_post_rollback_file_count"] = -1
    payload["controls"]["unsupported_control"] = True

    report = assess_artifact_compatibility(subsystem="recovery", payload=payload)

    assert report["compatible"] is False
    assert "recovery control snapshot_before_execution must be boolean" in report["violations"]
    assert "recovery control max_post_rollback_file_count must be a non-negative integer" in report["violations"]
    assert "recovery control is unsupported: unsupported_control" in report["violations"]


def test_assess_artifact_compatibility_rejects_invalid_delegation_controls():
    payload = build_delegation_proposal_artifact(KernelConfig())
    payload["controls"]["max_steps"] = -1
    payload["controls"]["unsupported_control"] = 1

    report = assess_artifact_compatibility(subsystem="delegation", payload=payload)

    assert report["compatible"] is False
    assert "delegation control max_steps must be a non-negative integer" in report["violations"]
    assert "delegation control is unsupported: unsupported_control" in report["violations"]


def test_improvement_planner_ranks_module_defined_external_subsystem(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "reason": "github module policy remains under-optimized",
                                    "priority": 4,
                                    "expected_gain": 0.03,
                                    "estimated_cost": 2,
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)
    external = next(candidate for candidate in experiments if candidate.subsystem == "github_policy")

    assert external.reason == "github module policy remains under-optimized"
    assert external.evidence["external_subsystem"] is True
    assert external.evidence["base_subsystem"] == "policy"


def test_improvement_planner_ignores_unresolvable_module_defined_external_subsystem(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_unknown",
                                    "base_subsystem": "unknown_surface",
                                    "reason": "should not be surfaced",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    assert all(candidate.subsystem != "github_unknown" for candidate in experiments)


def test_improvement_planner_ignores_non_retained_wrapped_capability_artifact(tmp_path: Path):
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
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "reason": "should remain inactive until retained",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    experiments = planner.rank_experiments(metrics)

    assert all(candidate.subsystem != "github_policy" for candidate in experiments)


def test_module_defined_external_subsystem_inherits_base_variants_and_artifact_path(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=10,
        trusted_retrieval_steps=10,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )
    experiment = ImprovementExperiment("github_policy", "module-defined", 4, 0.03, 2, 0.0, {})

    variants = planner.rank_variants(experiment, metrics)
    artifact_path = active_artifact_path_for_subsystem(
        KernelConfig(
            prompt_proposals_path=prompt_path,
            capability_modules_path=modules_path,
        ),
        "github_policy",
    )

    assert {variant.variant_id for variant in variants} >= {"retrieval_caution", "verifier_alignment"}
    assert artifact_path == prompt_path


def test_evaluate_artifact_retention_supports_module_defined_policy_subsystem(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "prompt_proposal_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.01},
        "proposals": [{"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}],
    }
    baseline = EvalMetrics(total=10, passed=7, average_steps=1.5)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.2)

    state, reason = evaluate_artifact_retention(
        "github_policy",
        baseline,
        candidate,
        payload=payload,
        capability_modules_path=modules_path,
    )

    assert state == "retain"
    assert "policy candidate" in reason


def test_recommend_campaign_budget_widens_for_close_scores(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.039, 2, 0.095, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]

    budget = planner.recommend_campaign_budget(metrics, max_width=3)

    assert budget.scope == "campaign"
    assert budget.width == 2
    assert budget.selected_ids == ["retrieval", "policy"]
    assert budget.strategy == "adaptive_history"


def test_recommend_campaign_budget_does_not_widen_on_stalled_lead_without_score_support(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 4, 0.02, 2, 0.062, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stalled",
            state="reject",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="reconciled stale cycle",
            metrics_summary={"incomplete_cycle": True},
        ),
    )

    budget = planner.recommend_campaign_budget(metrics, max_width=3)

    assert budget.width == 1
    assert budget.selected_ids == ["retrieval"]


def test_recommend_variant_budget_widens_for_close_siblings(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    experiment = ImprovementExperiment("policy", "p", 3, 0.02, 2, 0.05, {})
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_variants = lambda current_experiment, current_metrics: [
        ImprovementVariant("policy", "verifier_alignment", "a", 0.02, 2, 0.0200, {}),
        ImprovementVariant("policy", "retrieval_caution", "b", 0.019, 2, 0.0189, {}),
        ImprovementVariant("policy", "fallback", "c", 0.01, 2, 0.0060, {}),
    ]

    budget = planner.recommend_variant_budget(experiment, metrics, max_width=3)

    assert budget.scope == "variant"
    assert budget.width == 2
    assert budget.selected_ids == ["verifier_alignment", "retrieval_caution"]


def test_recommend_variant_budget_does_not_widen_on_stalled_lead_without_score_support(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    experiment = ImprovementExperiment("policy", "p", 3, 0.02, 2, 0.05, {})
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_variants = lambda current_experiment, current_metrics: [
        ImprovementVariant("policy", "stalled_lead", "a", 0.02, 2, 0.0200, {}),
        ImprovementVariant("policy", "fresh_sibling", "b", 0.015, 2, 0.0132, {}),
        ImprovementVariant("policy", "also_stalled", "c", 0.015, 2, 0.0131, {}),
    ]
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:stalled",
            state="select",
            subsystem="policy",
            action="choose_target",
            artifact_path="policy.json",
            artifact_kind="improvement_target",
            reason="policy gap",
            metrics_summary={"selected_variant": {"variant_id": "stalled_lead"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:stalled",
            state="generate",
            subsystem="policy",
            action="propose_prompt_update",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="policy gap",
            metrics_summary={"protocol": "autonomous", "selected_variant": {"variant_id": "stalled_lead"}},
            candidate_artifact_path="candidates/policy.json",
            active_artifact_path="policy.json",
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:also_stalled",
            state="reject",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="reconciled stale cycle",
            metrics_summary={"incomplete_cycle": True, "selected_variant_id": "also_stalled"},
        ),
    )

    budget = planner.recommend_variant_budget(experiment, metrics, max_width=3)

    assert budget.scope == "variant"
    assert budget.width == 1
    assert budget.selected_ids == ["stalled_lead"]


def test_select_portfolio_campaign_prefers_underexplored_near_tie(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.039, 2, 0.095, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]
    for cycle_id, state, subsystem, baseline, candidate in (
        ("cycle:retrieval:1", "select", "retrieval", None, None),
        ("cycle:retrieval:1", "reject", "retrieval", 0.80, 0.78),
        ("cycle:retrieval:2", "select", "retrieval", None, None),
        ("cycle:retrieval:2", "reject", "retrieval", 0.78, 0.75),
    ):
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state=state,
                subsystem=subsystem,
                action="choose_target" if state == "select" else "finalize_cycle",
                artifact_path="",
                artifact_kind="improvement_target" if state == "select" else "retrieval_policy_set",
                reason="history",
                metrics_summary={}
                if baseline is None
                else {
                    "baseline_pass_rate": baseline,
                    "candidate_pass_rate": candidate,
                    "baseline_average_steps": 1.0,
                    "candidate_average_steps": 1.0,
                },
            ),
        )

    campaign = planner.select_portfolio_campaign(EvalMetrics(total=10, passed=8), max_candidates=1)

    assert [candidate.subsystem for candidate in campaign] == ["policy"]
    assert campaign[0].evidence["portfolio"]["recent_activity"]["selected_cycles"] == 0


def test_select_portfolio_campaign_does_not_relax_scored_eligibility_under_stalled_lead_pressure(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 4, 0.02, 2, 0.04, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stalled",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:stalled",
            state="generate",
            subsystem="retrieval",
            action="propose_retrieval_update",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous"},
            candidate_artifact_path="candidates/retrieval.json",
            active_artifact_path="retrieval.json",
        ),
    )

    campaign = planner.select_portfolio_campaign(EvalMetrics(total=10, passed=8), max_candidates=3)

    assert [candidate.subsystem for candidate in campaign] == ["policy", "retrieval"]
    assert "skills" not in [candidate.subsystem for candidate in campaign]
    assert campaign[0].evidence["portfolio"]["campaign_breadth_pressure"] > 0.0
    assert any("campaign_breadth_pressure" in reason for reason in campaign[0].evidence["portfolio"]["reasons"])


def test_select_portfolio_campaign_keeps_clear_leader_despite_recent_saturation(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.01, 2, 0.03, {}),
    ]
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="history",
            metrics_summary={},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="history",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.90,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 1.0,
            },
        ),
    )

    campaign = planner.select_portfolio_campaign(EvalMetrics(total=10, passed=8), max_candidates=1)

    assert [candidate.subsystem for candidate in campaign] == ["retrieval"]


def test_rank_experiments_cools_off_repeated_no_yield_bootstrap_subsystem(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    for cycle_id, candidate_pass_rate in (
        ("cycle:verifier:1", 0.75),
        ("cycle:verifier:2", 0.74),
    ):
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem="verifier",
                action="finalize_cycle",
                artifact_path="verifier.json",
                artifact_kind="verifier_candidate_set",
                reason="rejected",
                metrics_summary={
                    "baseline_pass_rate": 0.80,
                    "candidate_pass_rate": candidate_pass_rate,
                    "baseline_average_steps": 1.0,
                    "candidate_average_steps": 1.0,
                },
            ),
        )

    experiments = planner.rank_experiments(EvalMetrics(total=10, passed=8, total_by_memory_source={}))

    verifier = next(candidate for candidate in experiments if candidate.subsystem == "verifier")
    assert verifier.evidence["selection_penalties"] == ["bootstrap_no_yield_penalty=0.0400"]
    assert verifier.score == 0.0


def test_rank_experiments_penalizes_repeated_no_yield_benchmark_subsystem(tmp_path: Path, monkeypatch):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)

    # Two recent benchmark selections with no retain/reject decisions should incur a stronger penalty
    # than other subsystems, so we don't get stuck generating only benchmark candidates.
    for cycle_id in ("cycle:benchmark:1", "cycle:benchmark:2", "cycle:benchmark:3"):
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="benchmark",
                action="choose_target",
                artifact_path="",
                artifact_kind="improvement_target",
                reason="history",
                metrics_summary={},
            ),
        )

    monkeypatch.setattr(planner, "failure_counts", lambda: {"missing_expected_file": 10})
    monkeypatch.setattr(planner, "transition_failure_counts", lambda: {})
    monkeypatch.setattr(planner, "transition_summary", lambda: {})
    monkeypatch.setattr(planner, "environment_violation_summary", lambda: {})
    monkeypatch.setattr(planner, "universe_cycle_feedback_summary", lambda: {})
    monkeypatch.setattr(planner, "trust_ledger_summary", lambda: {})
    monkeypatch.setattr(planner, "delegation_policy_summary", lambda: {})
    monkeypatch.setattr(planner, "operator_policy_summary", lambda: {})
    monkeypatch.setattr(planner, "capability_surface_summary", lambda: {})

    metrics = EvalMetrics(
        total=20,
        passed=12,
        low_confidence_episodes=1,  # adds retrieval as a competing candidate
        trusted_retrieval_steps=20,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
        skill_selected_steps=2,
        retrieval_ranked_skill_steps=1,
    )

    experiments = planner.rank_experiments(metrics)

    assert experiments[0].subsystem == "retrieval"
    benchmark = next(candidate for candidate in experiments if candidate.subsystem == "benchmark")
    assert "benchmark_recent_no_yield_penalty=" in " ".join(benchmark.evidence.get("selection_penalties", []))


def test_improvement_planner_ranks_experiments_with_scores(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(
        total=10,
        passed=7,
        low_confidence_episodes=3,
        trusted_retrieval_steps=1,
        total_by_memory_source={},
        skill_selected_steps=6,
        retrieval_ranked_skill_steps=1,
        generated_total=10,
        generated_passed=5,
    )

    experiments = planner.rank_experiments(metrics)

    assert experiments[0].score >= experiments[-1].score
    assert experiments[0].expected_gain > 0.0
    assert experiments[0].estimated_cost >= 1
    assert isinstance(experiments[0].evidence, dict)


def test_improvement_planner_generates_ranked_variants(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(total=10, passed=7, generated_total=10, generated_passed=5)
    experiment = planner.rank_experiments(metrics)[0]

    variants = planner.rank_variants(experiment, metrics)

    assert variants
    assert variants[0].score >= variants[-1].score
    assert variants[0].variant_id


def test_improvement_planner_uses_retained_history_to_score_experiments(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills.json",
            artifact_kind="skill_set",
            reason="retained gain",
            metrics_summary={
                "baseline_pass_rate": 0.70,
                "candidate_pass_rate": 0.85,
                "baseline_average_steps": 1.5,
                "candidate_average_steps": 1.1,
            },
        ),
    )

    metrics = EvalMetrics(
        total=10,
        passed=7,
        trusted_retrieval_steps=10,
        total_by_memory_source={},
        skill_selected_steps=6,
        retrieval_ranked_skill_steps=1,
        generated_total=10,
        generated_passed=5,
    )

    experiments = planner.rank_experiments(metrics)
    skills_experiment = next(experiment for experiment in experiments if experiment.subsystem == "skills")

    assert skills_experiment.evidence["history"]["retained_cycles"] == 1
    assert round(skills_experiment.evidence["history"]["average_retained_pass_rate_delta"], 3) == 0.15
    assert skills_experiment.score > 0.04


def test_improvement_planner_history_scoring_prefers_retained_outcome_over_rejected_peer(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:1",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="retained policy gain",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.90,
                "baseline_average_steps": 1.2,
                "candidate_average_steps": 1.0,
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="reject",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retrieval regression",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.75,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 1.1,
            },
        ),
    )

    metrics = EvalMetrics(total=10, passed=8)
    policy = planner._score_experiment(ImprovementExperiment("policy", "p", 4, 0.02, 2, 0.0, {}), metrics)
    retrieval = planner._score_experiment(ImprovementExperiment("retrieval", "r", 4, 0.02, 2, 0.0, {}), metrics)

    assert policy.evidence["history"]["last_decision_state"] == "retain"
    assert retrieval.evidence["history"]["last_decision_state"] == "reject"
    assert policy.score > retrieval.score


def test_improvement_planner_penalizes_repeated_stalled_selections(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:2",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap again",
            metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:2",
            state="evaluate",
            subsystem="retrieval",
            action="compare_candidate_to_baseline",
            artifact_path="retrieval.json",
            artifact_kind="retention_evaluation",
            reason="preview ended before decision",
            metrics_summary={
                "phase_gate_passed": False,
                "phase_gate_failure_count": 1,
            },
        ),
    )

    metrics = EvalMetrics(total=10, passed=8)
    retrieval = planner._score_experiment(ImprovementExperiment("retrieval", "r", 5, 0.02, 2, 0.0, {}), metrics)
    policy = planner._score_experiment(ImprovementExperiment("policy", "p", 5, 0.02, 2, 0.0, {}), metrics)

    assert retrieval.evidence["recent_history"]["selected_cycles"] == 2
    assert retrieval.evidence["recent_history"]["no_yield_cycles"] == 2
    assert retrieval.evidence["recent_history"]["recent_incomplete_cycles"] == 1
    assert "recent_stalled_selection_penalty=0.0450" in retrieval.evidence["selection_penalties"]
    assert policy.score > retrieval.score


def test_recent_subsystem_activity_summary_tracks_observation_timeout_provenance(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    for cycle_id in ("cycle:retrieval:1", "cycle:retrieval:2"):
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="retrieval.json",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem="retrieval",
                action="run_eval",
                artifact_path="retrieval.json",
                artifact_kind="eval_metrics",
                reason="observation timed out",
                metrics_summary={
                    "observation_timed_out": True,
                    "observation_current_task_timeout_budget_source": "prestep_subphase:tolbert_query",
                },
            ),
        )

    summary = planner.recent_subsystem_activity_summary(subsystem="retrieval")

    assert summary["selected_cycles"] == 2
    assert summary["no_yield_cycles"] == 2
    assert summary["recent_observation_timeout_cycles"] == 2
    assert summary["recent_budgeted_observation_timeout_cycles"] == 2
    assert summary["last_observation_timeout_budget_source"] == "prestep_subphase:tolbert_query"
    assert summary["repeated_observation_timeout_budget_source_count"] == 2


def test_improvement_planner_penalizes_repeated_observation_timeouts(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    for cycle_id in ("cycle:retrieval:1", "cycle:retrieval:2"):
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="retrieval.json",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="observe",
                subsystem="retrieval",
                action="run_eval",
                artifact_path="retrieval.json",
                artifact_kind="eval_metrics",
                reason="observation timed out",
                metrics_summary={
                    "observation_timed_out": True,
                    "observation_current_task_timeout_budget_source": "prestep_subphase:tolbert_query",
                },
            ),
        )

    metrics = EvalMetrics(total=10, passed=8)
    retrieval = planner._score_experiment(ImprovementExperiment("retrieval", "r", 5, 0.02, 2, 0.0, {}), metrics)
    policy = planner._score_experiment(ImprovementExperiment("policy", "p", 5, 0.02, 2, 0.0, {}), metrics)

    assert "recent_observation_timeout_penalty=0.0380" in retrieval.evidence["selection_penalties"]
    assert policy.score > retrieval.score


def test_improvement_planner_penalizes_recent_promotion_failures(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    for index, candidate_pass_rate in ((1, 0.55), (2, 0.5)):
        cycle_id = f"cycle:retrieval:{index}"
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="retrieval.json",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem="retrieval",
                action="finalize_cycle",
                artifact_path="retrieval.json",
                artifact_kind="retrieval_policy_set",
                reason="retrieval candidate regressed against baseline",
                metrics_summary={
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": candidate_pass_rate,
                    "phase_gate_passed": False,
                    "phase_gate_failure_count": 1,
                },
            ),
        )

    metrics = EvalMetrics(total=10, passed=8)
    retrieval = planner._score_experiment(ImprovementExperiment("retrieval", "r", 5, 0.02, 2, 0.0, {}), metrics)
    policy = planner._score_experiment(ImprovementExperiment("policy", "p", 5, 0.02, 2, 0.0, {}), metrics)

    assert retrieval.evidence["recent_history"]["rejected_cycles"] == 2
    assert round(retrieval.evidence["recent_history"]["average_rejected_pass_rate_delta"], 3) == -0.175
    assert "recent_promotion_failure_penalty=0.1000" in retrieval.evidence["selection_penalties"]
    assert policy.score > retrieval.score


def test_improvement_planner_caps_cold_start_low_confidence_retrieval_score(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    metrics = EvalMetrics(total=10, passed=4, low_confidence_episodes=6, trusted_retrieval_steps=1)

    retrieval = planner._score_experiment(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=5,
            expected_gain=0.6,
            estimated_cost=3,
            score=0.0,
            evidence={
                "total": metrics.total,
                "low_confidence_episodes": metrics.low_confidence_episodes,
                "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
            },
        ),
        metrics,
    )

    assert "cold_start_low_confidence_penalty=0.8650" in retrieval.evidence["selection_penalties"]
    assert retrieval.score == 0.12


def test_improvement_planner_removes_cold_start_low_confidence_penalty_after_history_exists(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retrieval improved",
            metrics_summary={
                "baseline_pass_rate": 0.6,
                "candidate_pass_rate": 0.7,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 1.0,
            },
        ),
    )
    metrics = EvalMetrics(total=10, passed=4, low_confidence_episodes=6, trusted_retrieval_steps=1)

    retrieval = planner._score_experiment(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=5,
            expected_gain=0.6,
            estimated_cost=3,
            score=0.0,
            evidence={
                "total": metrics.total,
                "low_confidence_episodes": metrics.low_confidence_episodes,
                "trusted_retrieval_steps": metrics.trusted_retrieval_steps,
            },
        ),
        metrics,
    )

    assert "selection_penalties" not in retrieval.evidence or not any(
        "cold_start_low_confidence_penalty" in reason
        for reason in retrieval.evidence.get("selection_penalties", [])
    )
    assert retrieval.score > 0.12


def test_improvement_planner_uses_variant_history_to_flip_variant_ranking(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:verifier:1",
            state="select",
            subsystem="verifier",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="verifier gap",
            metrics_summary={
                "selected_variant": {
                    "variant_id": "false_failure_guard",
                    "description": "tighten false-failure controls from failed traces",
                }
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:verifier:1",
            state="retain",
            subsystem="verifier",
            action="finalize_cycle",
            artifact_path="verifiers_v1.json",
            artifact_kind="verifier_candidate_set",
            reason="retained verifier gain",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.90,
                "baseline_average_steps": 1.2,
                "candidate_average_steps": 1.0,
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:verifier:2",
            state="select",
            subsystem="verifier",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="verifier gap again",
            metrics_summary={
                "selected_variant": {
                    "variant_id": "strict_contract_growth",
                    "description": "increase verifier strictness with successful traces",
                }
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:verifier:2",
            state="reject",
            subsystem="verifier",
            action="finalize_cycle",
            artifact_path="verifiers_v2.json",
            artifact_kind="verifier_candidate_set",
            reason="false-failure regression",
            metrics_summary={
                "baseline_pass_rate": 0.90,
                "candidate_pass_rate": 0.86,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 1.1,
            },
        ),
    )

    metrics = EvalMetrics(total=10, passed=10, trusted_retrieval_steps=10)
    experiment = next(
        candidate for candidate in planner.rank_experiments(metrics) if candidate.subsystem == "verifier"
    )

    variants = planner.rank_variants(experiment, metrics)

    assert variants[0].variant_id == "false_failure_guard"
    assert variants[0].controls["history"]["variant"]["retained_cycles"] == 1
    assert variants[1].controls["history"]["variant"]["rejected_cycles"] == 1


def test_improvement_planner_variant_learning_prefers_unseen_sibling_over_repeated_reject(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:1",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy_v1.json",
            artifact_kind="prompt_proposal_set",
            reason="retained subsystem gain",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.88,
                "baseline_average_steps": 1.1,
                "candidate_average_steps": 1.0,
                "selected_variant": {"variant_id": "stable_guard"},
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:2",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy_v2.json",
            artifact_kind="prompt_proposal_set",
            reason="retained subsystem gain again",
            metrics_summary={
                "baseline_pass_rate": 0.88,
                "candidate_pass_rate": 0.91,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 0.9,
                "selected_variant": {"variant_id": "stable_guard"},
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:3",
            state="reject",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy_v3.json",
            artifact_kind="prompt_proposal_set",
            reason="rejected sibling",
            metrics_summary={
                "baseline_pass_rate": 0.91,
                "candidate_pass_rate": 0.84,
                "baseline_average_steps": 0.9,
                "candidate_average_steps": 1.1,
                "selected_variant": {"variant_id": "risky_retry"},
            },
        ),
    )
    experiment = ImprovementExperiment("policy", "p", 4, 0.02, 2, 0.05, {})
    metrics = EvalMetrics(total=10, passed=8)
    planner._variants_for_experiment = lambda current_experiment, current_metrics, planner_controls=None: [  # type: ignore[method-assign]
        ImprovementVariant("policy", "risky_retry", "rejected sibling", 0.02, 2, 0.01, {}),
        ImprovementVariant("policy", "fresh_probe", "untested sibling", 0.02, 2, 0.01, {}),
    ]

    variants = planner.rank_variants(experiment, metrics)

    assert variants[0].variant_id == "fresh_probe"
    assert variants[0].controls["variant_exploration_bonus"] > 0.0
    assert variants[1].controls["history"]["variant"]["rejected_cycles"] == 1


def test_stamp_artifact_experiment_variant_updates_payload(tmp_path: Path):
    artifact_path = tmp_path / "prompts.json"
    artifact_path.write_text(
        json.dumps({"artifact_kind": "prompt_proposal_set", "proposals": []}),
        encoding="utf-8",
    )
    variant = ImprovementVariant(
        subsystem="policy",
        variant_id="retrieval_caution",
        description="make low-confidence retrieval less binding",
        expected_gain=0.015,
        estimated_cost=2,
        score=0.0075,
        controls={"focus": "retrieval_caution"},
    )

    stamp_artifact_experiment_variant(artifact_path, variant)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["experiment_variant"]["variant_id"] == "retrieval_caution"


def test_benchmark_synthesis_emits_candidates(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro"},"summary":{"failure_types":["command_failure"],"executed_commands":["false","printf \\"x\\\\n\\" > hello.txt"]}}',
        encoding="utf-8",
    )
    output = tmp_path / "benchmarks.json"

    synthesize_benchmark_candidates(episodes, output)

    payload = output.read_text(encoding="utf-8")
    assert "failure_cluster" in payload
    assert "benchmark_candidate_set" in payload
    assert "retention_gate" in payload
    assert "recovery_path" in payload


def test_benchmark_synthesis_emits_transition_failure_candidates(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro"},"summary":{"failure_types":[],"transition_failures":["no_state_progress","state_regression"],"executed_commands":["false","printf \\"x\\\\n\\" > hello.txt"]}}',
        encoding="utf-8",
    )
    output = tmp_path / "benchmarks.json"

    synthesize_benchmark_candidates(episodes, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    kinds = {proposal["kind"] for proposal in payload["proposals"]}
    assert "transition_failure" in kinds


def test_benchmark_synthesis_applies_focus(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro"},"summary":{"failure_types":["policy_terminated"],"executed_commands":["false","printf \\"x\\\\n\\" > hello.txt"]}}',
        encoding="utf-8",
    )
    output = tmp_path / "benchmarks.json"

    synthesize_benchmark_candidates(episodes, output, focus="confidence")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generation_focus"] == "confidence"
    assert payload["proposals"][0]["kind"] == "failure_cluster"
    assert "low-confidence retrieval" in payload["proposals"][0]["prompt"]
    assert payload["retention_gate"]["max_regressed_families"] == 0


def test_verifier_improvement_emits_contracts(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":true,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    episodes.joinpath("hello_task_failed.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"summary":{"failure_types":["missing_expected_file","command_failure"]},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    output = tmp_path / "verifiers.json"

    synthesize_verifier_contracts(episodes, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "verifier_candidate_set"
    assert payload["proposals"][0]["proposal_id"].endswith(":strict")
    assert payload["proposals"][0]["evidence"]["failed_trace_count"] == 1
    assert payload["proposals"][0]["evidence"]["strict_contract"] is True


def test_verifier_improvement_records_transition_failures(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":true,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    episodes.joinpath("hello_task_failed.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"summary":{"failure_types":[],"transition_failures":["no_state_progress","state_regression"]},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    output = tmp_path / "verifiers.json"

    synthesize_verifier_contracts(episodes, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["proposals"][0]["evidence"]["transition_failures"] == ["no_state_progress", "state_regression"]


def test_verifier_improvement_records_generation_strategy(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episodes.joinpath("hello_task.json").write_text(
        '{"task_id":"hello_task","success":true,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    episodes.joinpath("hello_task_failed.json").write_text(
        '{"task_id":"hello_task","success":false,"task_metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"},"summary":{"failure_types":["missing_expected_file"]},"task_contract":{"prompt":"Create hello.txt containing hello agent kernel.","workspace_subdir":"hello_task","setup_commands":[],"success_command":"test -f hello.txt && grep -q \\"hello agent kernel\\" hello.txt","suggested_commands":["printf \\"hello agent kernel\\\\n\\" > hello.txt"],"expected_files":["hello.txt"],"expected_output_substrings":[],"forbidden_files":[],"forbidden_output_substrings":[],"expected_file_contents":{"hello.txt":"hello agent kernel\\n"},"max_steps":5,"metadata":{"benchmark_family":"micro","capability":"file_write","difficulty":"seed"}}}',
        encoding="utf-8",
    )
    output = tmp_path / "verifiers.json"

    synthesize_verifier_contracts(episodes, output, strategy="false_failure_guard")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generation_strategy"] == "false_failure_guard"


def test_prompt_improvement_emits_proposals():
    metrics = EvalMetrics(
        total=10,
        passed=9,
        low_confidence_episodes=2,
        trusted_retrieval_steps=3,
        generated_total=5,
        generated_passed=3,
    )

    proposals = propose_prompt_adjustments(metrics, {"command_failure": 2, "missing_expected_file": 1})

    assert any(proposal["area"] == "decision" for proposal in proposals)
    assert any(proposal["area"] == "system" for proposal in proposals)


def test_prompt_improvement_applies_variant_focus():
    metrics = EvalMetrics(
        total=10,
        passed=9,
        low_confidence_episodes=2,
        trusted_retrieval_steps=3,
        generated_total=5,
        generated_passed=3,
    )

    artifact = build_prompt_proposal_artifact(metrics, {"command_failure": 2}, focus="verifier_alignment")

    assert artifact["generation_focus"] == "verifier_alignment"
    assert artifact["proposals"]


def test_policy_improvement_emits_architecture_aligned_overrides():
    metrics = EvalMetrics(total=10, passed=8, low_confidence_episodes=3)

    artifact = build_policy_proposal_artifact(
        metrics,
        {"missing_expected_file": 1, "command_failure": 1},
        focus="verifier_alignment",
    )

    assert artifact["artifact_kind"] == "prompt_proposal_set"
    assert artifact["policy_schema"] == "agentic_policy_controls_v1"
    assert artifact["tolbert_runtime_policy_overrides"]["min_path_confidence"] >= 0.8
    assert artifact["tolbert_runtime_policy_overrides"]["primary_min_command_score"] >= 3
    assert artifact["tolbert_decoder_policy_overrides"]["min_stop_completion_ratio"] >= 0.98
    assert artifact["tolbert_rollout_policy_overrides"]["stop_missing_expected_penalty_weight"] >= 7.5
    assert artifact["tolbert_hybrid_scoring_policy_overrides"]["world_progress_weight"] >= 0.28
    assert all(proposal["area"] in {"system", "decision"} for proposal in artifact["proposals"])


def test_prompt_improvement_emits_variant_expansions_for_retrieval_caution():
    metrics = EvalMetrics(
        total=10,
        passed=8,
        low_confidence_episodes=3,
        trusted_retrieval_steps=2,
    )

    artifact = build_prompt_proposal_artifact(metrics, {"command_failure": 1}, focus="retrieval_caution")

    expansions = artifact["improvement_planner_controls"]["variant_expansions"]["retrieval"]
    assert {entry["variant_id"] for entry in expansions} == {"routing_depth", "direct_command_safety"}


def test_prompt_improvement_learns_allocation_confidence_controls_for_retrieval_caution():
    metrics = EvalMetrics(
        total=12,
        passed=8,
        low_confidence_episodes=3,
        trusted_retrieval_steps=2,
        generated_total=6,
        generated_passed=2,
    )

    artifact = build_prompt_proposal_artifact(metrics, {"command_failure": 1}, focus="retrieval_caution")

    allocation_confidence = artifact["improvement_planner_controls"]["priority_family_allocation_confidence"]
    assert allocation_confidence["minimum_runs"] >= 5
    assert allocation_confidence["target_priority_tasks"] >= 20
    assert allocation_confidence["target_family_tasks"] >= 4
    assert allocation_confidence["history_window_runs"] >= 4
    assert allocation_confidence["history_weight"] >= 0.65
    assert allocation_confidence["bonus_history_weight"] >= 0.9
    assert allocation_confidence["normalization_history_weight"] <= 0.15


def test_prompt_improvement_emits_planner_controls_and_role_directives():
    metrics = EvalMetrics(
        total=10,
        passed=8,
        low_confidence_episodes=2,
        trusted_retrieval_steps=3,
    )

    artifact = build_prompt_proposal_artifact(metrics, {"missing_expected_file": 1}, focus="verifier_alignment")

    assert artifact["planner_controls"]["prepend_verifier_contract_check"] is True
    assert artifact["planner_controls"]["append_validation_subgoal"] is True
    assert artifact["improvement_planner_controls"]["subsystem_expected_gain_multiplier"]["policy"] >= 1.35
    assert "planner" in artifact["role_directives"]
    assert retained_planner_controls({**artifact, "lifecycle_state": "retained"})["prepend_verifier_contract_check"] is True
    assert (
        retained_improvement_planner_controls({**artifact, "lifecycle_state": "retained"})["subsystem_score_bias"]["policy"]
        >= 0.012
    )
    assert retained_role_directives({**artifact, "lifecycle_state": "retained"})["planner"]


def test_improvement_planner_applies_retained_improvement_planner_controls(tmp_path: Path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "subsystem_expected_gain_multiplier": {"policy": 3.0, "retrieval": 0.5},
                    "subsystem_cost_multiplier": {"policy": 1.0, "retrieval": 1.0},
                    "subsystem_score_bias": {"policy": 0.02},
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
    )
    metrics = EvalMetrics(
        total=100,
        passed=80,
        low_confidence_episodes=2,
        trusted_retrieval_steps=60,
    )

    experiments = planner.rank_experiments(metrics)

    assert experiments[0].subsystem == "policy"
    assert experiments[0].evidence["improvement_planner_mutation"]["expected_gain_multiplier"] == 3.0
    retrieval = next(candidate for candidate in experiments if candidate.subsystem == "retrieval")
    assert retrieval.evidence["improvement_planner_mutation"]["expected_gain_multiplier"] == 0.5


def test_retained_improvement_planner_controls_normalize_priority_family_allocation_confidence():
    controls = retained_improvement_planner_controls(
        {
            "artifact_kind": "prompt_proposal_set",
            "lifecycle_state": "retained",
            "improvement_planner_controls": {
                "priority_family_allocation_confidence": {
                    "minimum_runs": 5,
                    "target_priority_tasks": 18,
                    "target_family_tasks": 4,
                    "history_window_runs": 2,
                    "history_weight": 0.4,
                    "bonus_history_weight": 0.9,
                }
            },
        }
    )

    assert controls["priority_family_allocation_confidence"]["minimum_runs"] == 5
    assert controls["priority_family_allocation_confidence"]["target_priority_tasks"] == 18
    assert controls["priority_family_allocation_confidence"]["target_family_tasks"] == 4
    assert controls["priority_family_allocation_confidence"]["history_window_runs"] == 2
    assert controls["priority_family_allocation_confidence"]["history_weight"] == 0.4
    assert controls["priority_family_allocation_confidence"]["bonus_history_weight"] == 0.9
    assert controls["priority_family_allocation_confidence"]["normalization_history_weight"] == 0.25


def test_improvement_planner_ignores_retained_controls_when_prompt_proposals_disabled(tmp_path: Path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "subsystem_expected_gain_multiplier": {"policy": 3.0},
                    "subsystem_score_bias": {"policy": 0.02},
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
        use_prompt_proposals=False,
    )
    metrics = EvalMetrics(total=100, passed=80, low_confidence_episodes=2, trusted_retrieval_steps=60)

    experiments = planner.rank_experiments(metrics)

    assert all("improvement_planner_mutation" not in experiment.evidence for experiment in experiments)


def test_recommend_campaign_budget_respects_retained_improvement_planner_search_controls(tmp_path: Path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "search_guardrails": {
                        "campaign": {
                            "close_score_relative_threshold": 0.8,
                            "close_score_margin_threshold": 0.03,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
    )
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.032, 2, 0.08, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]

    budget = planner.recommend_campaign_budget(metrics, max_width=3)

    assert budget.width == 2
    assert budget.selected_ids == ["retrieval", "policy"]


def test_recommend_campaign_budget_supports_legacy_flat_search_guardrails(tmp_path: Path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "campaign_close_score_relative_threshold": 0.8,
                    "campaign_close_score_margin_threshold": 0.03,
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
    )
    metrics = EvalMetrics(total=10, passed=8, average_steps=1.0)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.032, 2, 0.08, {}),
        ImprovementExperiment("skills", "s", 4, 0.01, 2, 0.03, {}),
    ]

    budget = planner.recommend_campaign_budget(metrics, max_width=3)

    assert budget.width == 2
    assert budget.selected_ids == ["retrieval", "policy"]


def test_improvement_planner_rank_variants_includes_retained_variant_expansion(tmp_path: Path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "variant_expansions": {
                        "retrieval": [
                            {
                                "variant_id": "routing_depth",
                                "description": "widen branch routing depth under persistent uncertainty",
                                "expected_gain": 0.024,
                                "estimated_cost": 3,
                                "controls": {"focus": "routing"},
                            }
                        ]
                    },
                    "variant_score_bias": {"retrieval": {"routing_depth": 0.01}},
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
    )
    experiment = ImprovementExperiment("retrieval", "r", 5, 0.04, 2, 0.10, {})
    metrics = EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2)

    variants = planner.rank_variants(experiment, metrics)

    assert any(variant.variant_id == "routing_depth" for variant in variants)
    assert variants[0].variant_id == "routing_depth"
    assert variants[0].controls["focus"] == "routing"


def test_external_subsystem_inherits_base_planner_controls_and_variant_expansions(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "reason": "github policy remains under-optimized",
                                    "priority": 4,
                                    "expected_gain": 0.03,
                                    "estimated_cost": 2,
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "improvement_planner_controls": {
                    "subsystem_expected_gain_multiplier": {"policy": 2.0},
                    "subsystem_score_bias": {"policy": 0.01},
                    "variant_expansions": {
                        "policy": [
                            {
                                "variant_id": "expected_artifact_guardrail",
                                "description": "bias planning toward expected artifacts",
                                "expected_gain": 0.03,
                                "estimated_cost": 2,
                                "controls": {"focus": "verifier_alignment"},
                            }
                        ]
                    },
                    "variant_score_bias": {"policy": {"expected_artifact_guardrail": 0.01}},
                },
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        prompt_proposals_path=prompt_path,
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(
        total=10,
        passed=8,
        low_confidence_episodes=1,
        trusted_retrieval_steps=9,
        total_by_memory_source={"verifier": 1, "skill_transfer": 1},
    )

    external = next(
        candidate for candidate in planner.rank_experiments(metrics) if candidate.subsystem == "github_policy"
    )
    variants = planner.rank_variants(external, metrics)

    assert external.evidence["base_subsystem"] == "policy"
    assert external.evidence["improvement_planner_mutation"]["expected_gain_multiplier"] == 2.0
    assert any(variant.variant_id == "expected_artifact_guardrail" for variant in variants)
    assert variants[0].controls["base_subsystem"] == "policy"


def test_module_defined_external_subsystem_ignores_invalid_runtime_field_overrides(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "artifact_path_attr": "missing_artifact_path_attr",
                                    "proposal_toggle_attr": "missing_toggle_attr",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"

    artifact_path = active_artifact_path_for_subsystem(
        KernelConfig(
            prompt_proposals_path=prompt_path,
            capability_modules_path=modules_path,
        ),
        "github_policy",
    )

    assert artifact_path == prompt_path


def test_module_defined_external_subsystem_uses_base_generation_contract(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                    "generator_kind": "unsupported_generator",
                                    "artifact_kind": "unsupported_artifact_kind",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
        capability_modules_path=modules_path,
    )
    candidate_path = tmp_path / "candidates" / "github_policy.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = EvalMetrics(total=10, passed=8, low_confidence_episodes=2, trusted_retrieval_steps=2)

    class PlannerStub:
        @staticmethod
        def failure_counts():
            return {"command_failure": 1}

    artifact_path, action, artifact_kind = generate_candidate_artifact(
        config=config,
        planner=PlannerStub(),
        subsystem="github_policy",
        metrics=metrics,
        generation_kwargs={},
        candidate_artifact_path=candidate_path,
    )

    payload = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert artifact_path == str(candidate_path)
    assert action == "propose_prompt_update"
    assert artifact_kind == "prompt_proposal_set"
    assert payload["artifact_kind"] == "prompt_proposal_set"


def test_assess_artifact_compatibility_rejects_invalid_capability_surface_runtime_fields(tmp_path: Path):
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "capability_module_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.01},
        "modules": [
            {
                "module_id": "github",
                "enabled": True,
                "capabilities": ["github_read"],
                "settings": {
                    "improvement_subsystems": [
                        {
                            "subsystem_id": "github_policy",
                            "base_subsystem": "policy",
                            "artifact_path_attr": "missing_artifact_path_attr",
                            "proposal_toggle_attr": "missing_toggle_attr",
                        }
                    ]
                },
            }
        ],
    }

    report = assess_artifact_compatibility(subsystem="capabilities", payload=payload)

    assert report["compatible"] is False
    assert "valid config artifact_path_attr" in " ".join(report["violations"])
    assert "valid config proposal_toggle_attr" in " ".join(report["violations"])


def test_assess_artifact_compatibility_rejects_invalid_capability_surface_generation_contract(tmp_path: Path):
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "capability_module_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.01},
        "modules": [
            {
                "module_id": "github",
                "enabled": True,
                "capabilities": ["github_read"],
                "settings": {
                    "improvement_subsystems": [
                        {
                            "subsystem_id": "github_policy",
                            "base_subsystem": "policy",
                            "generator_kind": "unsupported_generator",
                            "artifact_kind": "unsupported_artifact_kind",
                        }
                    ]
                },
            }
        ],
    }

    report = assess_artifact_compatibility(subsystem="capabilities", payload=payload)

    assert report["compatible"] is False
    assert "base_subsystem generator_kind" in " ".join(report["violations"])
    assert "base_subsystem artifact_kind" in " ".join(report["violations"])


def test_assess_artifact_compatibility_rejects_capability_artifact_missing_retention_gate():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "capability_module_set",
        "lifecycle_state": "proposed",
        "modules": [
            {
                "module_id": "github",
                "enabled": True,
                "capabilities": ["github_read"],
            }
        ],
    }

    report = assess_artifact_compatibility(subsystem="capabilities", payload=payload)

    assert report["compatible"] is False
    assert any("retention_gate" in violation for violation in report["violations"])


def test_evaluate_artifact_retention_retains_world_model_gain_without_regression():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.2, generated_total=4, generated_passed=3)

    state, reason = evaluate_artifact_retention("world_model", baseline, candidate)

    assert state == "retain"
    assert "world-model candidate" in reason


def test_evaluate_artifact_retention_rejects_world_model_low_confidence_regression():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.5,
        generated_total=4,
        generated_passed=3,
        low_confidence_episodes=1,
        first_step_successes=6,
    )
    candidate = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.2,
        generated_total=4,
        generated_passed=3,
        low_confidence_episodes=2,
        first_step_successes=6,
    )

    state, reason = evaluate_artifact_retention("world_model", baseline, candidate)

    assert state == "reject"
    assert "low-confidence episodes" in reason


def test_evaluate_artifact_retention_rejects_world_model_first_step_regression():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.5,
        generated_total=4,
        generated_passed=3,
        low_confidence_episodes=1,
        first_step_successes=6,
    )
    candidate = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.2,
        generated_total=4,
        generated_passed=3,
        low_confidence_episodes=1,
        first_step_successes=5,
    )

    state, reason = evaluate_artifact_retention("world_model", baseline, candidate)

    assert state == "reject"
    assert "first-step success" in reason


def test_evaluate_artifact_retention_retains_state_estimation_improvement_without_regression():
    baseline = EvalMetrics(
        total=10,
        passed=7,
        average_steps=2.0,
        termination_reasons={"no_state_progress": 1},
        task_trajectories={
            "cleanup_task": {
                "success": False,
                "termination_reason": "no_state_progress",
                "steps": [
                    {
                        "state_regression_count": 1,
                        "failure_signals": ["state_regression", "no_state_progress"],
                        "exit_code": 1,
                    }
                ],
            }
        },
    )
    candidate = EvalMetrics(
        total=10,
        passed=7,
        average_steps=1.8,
        termination_reasons={"no_state_progress": 0},
        task_trajectories={
            "cleanup_task": {
                "success": True,
                "termination_reason": "",
                "steps": [
                    {
                        "state_regression_count": 0,
                        "failure_signals": [],
                        "exit_code": 0,
                    }
                ],
            }
        },
    )
    payload = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=7, low_confidence_episodes=1),
        {"state_regression_steps": 1, "state_progress_gain_steps": 0},
        focus="recovery_bias",
        current_payload={
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "no_progress_progress_epsilon": 0.0,
                "min_state_change_score_for_progress": 1,
                "regression_path_budget": 6,
                "regression_severity_weight": 1.0,
                "progress_recovery_credit": 1.0,
            },
            "latent_controls": {
                "advancing_completion_ratio": 0.8,
                "advancing_progress_delta": 0.2,
                "improving_progress_delta": 0.0,
                "regressing_progress_delta": -0.05,
                "regressive_regression_count": 1,
                "blocked_forbidden_count": 1,
                "active_path_budget": 6,
            },
            "policy_controls": {
                "regressive_path_match_bonus": 2,
                "regressive_cleanup_bonus": 1,
                "blocked_command_bonus": 1,
                "advancing_path_match_bonus": 1,
                "trusted_retrieval_path_bonus": 1,
            },
        },
    )
    payload["generation_context"] = {
        "active_artifact_payload": {
            "artifact_kind": "state_estimation_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "no_progress_progress_epsilon": 0.0,
                "min_state_change_score_for_progress": 1,
                "regression_path_budget": 6,
                "regression_severity_weight": 1.0,
                "progress_recovery_credit": 1.0,
            },
            "latent_controls": {
                "advancing_completion_ratio": 0.8,
                "advancing_progress_delta": 0.2,
                "improving_progress_delta": 0.0,
                "regressing_progress_delta": -0.05,
                "regressive_regression_count": 1,
                "blocked_forbidden_count": 1,
                "active_path_budget": 6,
            },
            "policy_controls": {
                "regressive_path_match_bonus": 2,
                "regressive_cleanup_bonus": 1,
                "blocked_command_bonus": 1,
                "advancing_path_match_bonus": 1,
                "trusted_retrieval_path_bonus": 1,
            },
        }
    }

    state, reason = evaluate_artifact_retention("state_estimation", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "state-estimation candidate" in reason


def test_evaluate_artifact_retention_rejects_state_estimation_regression_growth():
    baseline = EvalMetrics(
        total=10,
        passed=7,
        average_steps=1.8,
        termination_reasons={"no_state_progress": 0},
        task_trajectories={
            "cleanup_task": {
                "success": True,
                "termination_reason": "",
                "steps": [{"state_regression_count": 0, "failure_signals": [], "exit_code": 0}],
            }
        },
    )
    candidate = EvalMetrics(
        total=10,
        passed=7,
        average_steps=1.8,
        termination_reasons={"no_state_progress": 1},
        task_trajectories={
            "cleanup_task": {
                "success": False,
                "termination_reason": "no_state_progress",
                "steps": [
                    {
                        "state_regression_count": 1,
                        "failure_signals": ["state_regression", "no_state_progress"],
                        "exit_code": 1,
                    }
                ],
            }
        },
    )
    payload = build_state_estimation_proposal_artifact(
        EvalMetrics(total=10, passed=7),
        {"state_regression_steps": 1, "state_progress_gain_steps": 0},
    )

    state, reason = evaluate_artifact_retention("state_estimation", baseline, candidate, payload=payload)

    assert state == "reject"
    assert "no_state_progress terminations" in reason or "state_regression traces" in reason


def test_evaluate_artifact_retention_retains_universe_improvement_without_regression():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_universe_contract_artifact(
        EvalMetrics(total=10, passed=8, low_confidence_episodes=1),
        {"command_failure": 1, "state_regression": 1},
        focus="governance",
    )
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "universe_contract",
            "lifecycle_state": "retained",
            "control_schema": "universe_contract_v1",
            "governance": {
                "require_verification": False,
                "require_bounded_steps": True,
                "prefer_reversible_actions": False,
                "respect_task_forbidden_artifacts": True,
                "respect_preserved_artifacts": False,
            },
            "action_risk_controls": {
                "destructive_mutation_penalty": 10,
                "git_mutation_penalty": 4,
                "inline_destructive_interpreter_penalty": 6,
                "network_fetch_penalty": 3,
                "privileged_command_penalty": 8,
                "read_only_discovery_bonus": 2,
                "remote_execution_penalty": 6,
                "reversible_file_operation_bonus": 1,
                "scope_escape_penalty": 7,
                "unbounded_execution_penalty": 5,
                "verification_bonus": 3,
            },
            "environment_assumptions": {
                "git_write_mode": "task_scoped",
                "network_access_mode": "open",
                "workspace_write_scope": "generated_only",
                "require_path_scoped_mutations": False,
                "require_rollback_on_mutation": False,
            },
            "invariants": ["preserve verifier contract alignment"],
            "forbidden_command_patterns": ["rm -rf /"],
            "preferred_command_prefixes": ["rg "],
        }
    }

    state, reason = evaluate_artifact_retention("universe", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "universe candidate" in reason


def test_evaluate_artifact_retention_rejects_universe_constitution_without_history_support():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.5,
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 4, "tooling": 4},
    )
    candidate = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.4,
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "tooling": 4},
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "universe_constitution",
        "lifecycle_state": "retained",
        "control_schema": "universe_constitution_v1",
        "retention_gate": {
            "require_universe_improvement": True,
            "min_prior_retained_universe_cycles": 2,
            "min_cross_family_support": 2,
        },
        "retention_context": {
            "prior_retained_universe_cycle_count": 1,
            "constitution_cooldown_cycles_remaining": 0,
        },
        "governance": {
            "require_verification": True,
            "require_bounded_steps": True,
            "prefer_reversible_actions": True,
            "respect_task_forbidden_artifacts": True,
            "respect_preserved_artifacts": True,
        },
        "invariants": ["preserve verifier contract alignment"],
        "forbidden_command_patterns": ["rm -rf /", "git reset --hard", "git checkout --"],
        "preferred_command_prefixes": ["pytest", "rg "],
        "proposals": [{"area": "governance", "priority": 5, "reason": "test", "suggestion": "test"}],
        "generation_context": {
            "active_artifact_payload": {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "retained",
                "control_schema": "universe_constitution_v1",
                "governance": {
                    "require_verification": False,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": False,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": False,
                },
                "invariants": ["preserve verifier contract alignment"],
                "forbidden_command_patterns": ["rm -rf /"],
                "preferred_command_prefixes": ["rg "],
            }
        },
    }

    state, reason = evaluate_artifact_retention("universe", baseline, candidate, payload=payload)

    assert state == "reject"
    assert "multiple prior retained universe wins" in reason


def test_evaluate_artifact_retention_retains_trust_control_improvement_without_regression():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_trust_proposal_artifact(EvalMetrics(total=10, passed=8), {})
    payload["controls"].update(
        {
            "required_benchmark_families": ["repo_chore", "repo_sandbox", "project"],
            "min_success_rate": 0.8,
            "max_unsafe_ambiguous_rate": 0.05,
            "max_hidden_side_effect_rate": 0.05,
            "max_success_hidden_side_effect_rate": 0.01,
            "min_distinct_families": 3,
            "breadth_min_reports": 12,
        }
    )
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "trust_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "required_benchmark_families": ["repo_chore", "repo_sandbox"],
                "min_success_rate": 0.7,
                "max_unsafe_ambiguous_rate": 0.1,
                "max_hidden_side_effect_rate": 0.1,
                "max_success_hidden_side_effect_rate": 0.02,
                "min_distinct_families": 2,
                "breadth_min_reports": 10,
            },
        }
    }

    state, reason = evaluate_artifact_retention("trust", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "trust candidate" in reason


def test_evaluate_artifact_retention_retains_recovery_control_improvement_without_regression():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_recovery_proposal_artifact(EvalMetrics(total=10, passed=8), {})
    payload["controls"].update(
        {
            "snapshot_before_execution": True,
            "rollback_on_runner_exception": True,
            "rollback_on_failed_outcome": True,
            "rollback_on_safe_stop": True,
            "verify_post_rollback_file_count": True,
            "max_post_rollback_file_count": 0,
        }
    )
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "recovery_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "snapshot_before_execution": True,
                "rollback_on_runner_exception": True,
                "rollback_on_failed_outcome": True,
                "rollback_on_safe_stop": False,
                "verify_post_rollback_file_count": False,
                "max_post_rollback_file_count": 1,
            },
        }
    }

    state, reason = evaluate_artifact_retention("recovery", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "recovery candidate" in reason


def test_evaluate_artifact_retention_rejects_recovery_without_control_improvement():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_recovery_proposal_artifact(EvalMetrics(total=10, passed=8), {})
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "recovery_policy_set",
            "lifecycle_state": "retained",
            "controls": dict(payload["controls"]),
        }
    }

    state, reason = evaluate_artifact_retention("recovery", baseline, candidate, payload=payload)

    assert state == "reject"
    assert "did not strengthen unattended rollback or snapshot policy" in reason


def test_evaluate_artifact_retention_retains_delegation_control_improvement_without_regression():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_delegation_proposal_artifact(KernelConfig(), focus="worker_depth")
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "delegated_runtime_policy_set",
            "lifecycle_state": "retained",
            "controls": {
                "delegated_job_max_concurrency": 1,
                "delegated_job_max_active_per_budget_group": 0,
                "delegated_job_max_queued_per_budget_group": 0,
                "delegated_job_max_artifact_bytes": 1024,
                "delegated_job_max_subprocesses_per_job": 1,
                "command_timeout_seconds": 20,
                "llm_timeout_seconds": 20,
                "max_steps": 5,
            },
        }
    }

    state, reason = evaluate_artifact_retention("delegation", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "delegation candidate" in reason


def test_evaluate_artifact_retention_rejects_delegation_without_control_improvement():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)
    payload = build_delegation_proposal_artifact(KernelConfig())
    payload["generation_context"] = {
        "active_artifact_payload": {
            "spec_version": "asi_v1",
            "artifact_kind": "delegated_runtime_policy_set",
            "lifecycle_state": "retained",
            "controls": dict(payload["controls"]),
        }
    }

    state, reason = evaluate_artifact_retention("delegation", baseline, candidate, payload=payload)

    assert state == "reject"
    assert "did not expand delegated runtime capacity" in reason


def test_evaluate_artifact_retention_ignores_non_retained_wrapped_generation_context_active_artifact(tmp_path: Path):
    active_path = tmp_path / "delegation" / "delegation_proposals.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_delegation_proposal_artifact(KernelConfig())
    active_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "delegated_runtime_policy_set",
                "lifecycle_state": "proposed",
                "retention_gate": {"min_pass_rate_delta_abs": 0.0},
                "controls": dict(payload["controls"]),
            }
        ),
        encoding="utf-8",
    )
    payload["generation_context"] = {
        "active_artifact_path": str(active_path),
    }
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=4, generated_passed=3)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.4, generated_total=4, generated_passed=3)

    state, reason = evaluate_artifact_retention("delegation", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "delegation candidate" in reason


def test_select_portfolio_campaign_avoids_duplicate_shared_surface(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    metrics = EvalMetrics(total=10, passed=8)
    planner.rank_experiments = lambda current_metrics: [
        ImprovementExperiment("github_policy", "g", 5, 0.04, 2, 0.10, {}),
        ImprovementExperiment("policy", "p", 5, 0.039, 2, 0.099, {}),
        ImprovementExperiment("retrieval", "r", 4, 0.03, 2, 0.08, {}),
    ]

    campaign = planner.select_portfolio_campaign(metrics, max_candidates=2)
    budget = planner.recommend_campaign_budget(metrics, max_width=3)

    assert len([candidate for candidate in campaign if candidate.subsystem in {"policy", "github_policy"}]) == 1
    assert len(campaign) == 2
    assert len([subsystem for subsystem in budget.selected_ids if subsystem in {"policy", "github_policy"}]) == 1


def test_retrieval_improvement_emits_policy_artifact():
    metrics = EvalMetrics(total=10, passed=8, low_confidence_episodes=3, trusted_retrieval_steps=2)

    artifact = build_retrieval_proposal_artifact(metrics, focus="confidence")

    assert artifact["artifact_kind"] == "retrieval_policy_set"
    assert artifact["generation_focus"] == "confidence"
    assert artifact["proposals"]
    assert artifact["overrides"]["tolbert_confidence_threshold"] >= 0.15
    assert artifact["retention_gate"]["require_failure_recovery_non_regression"] is True


def test_retrieval_improvement_emits_broader_control_proposals():
    metrics = EvalMetrics(total=10, passed=8, low_confidence_episodes=3, trusted_retrieval_steps=2)

    artifact = build_retrieval_proposal_artifact(metrics)

    proposal_ids = {proposal["proposal_id"] for proposal in artifact["proposals"]}
    assert "retrieval:routing_depth" in proposal_ids
    assert "retrieval:direct_command_safety" in proposal_ids


def test_retrieval_improvement_returns_only_retained_overrides():
    assert retained_retrieval_overrides({"artifact_kind": "retrieval_policy_set", "lifecycle_state": "proposed", "overrides": {"tolbert_branch_results": 1}}) == {}
    assert retained_retrieval_overrides({"artifact_kind": "retrieval_policy_set", "lifecycle_state": "retained", "overrides": {"tolbert_branch_results": 1}}) == {"tolbert_branch_results": 1}


def test_improvement_cycle_record_is_appended(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    planner.append_cycle_record(
        output,
        planner_record := ImprovementCycleRecord(
            cycle_id="cycle:test:1",
            state="generate",
            subsystem="benchmark",
            action="synthesize_benchmarks",
            artifact_path="benchmarks.json",
            artifact_kind="benchmark_candidate_set",
            reason="test",
            metrics_summary={"total": 1, "passed": 1},
        ),
    )
    payload = output.read_text(encoding="utf-8")
    assert planner_record.cycle_id in payload
    assert "benchmark_candidate_set" in payload


def test_improvement_cycle_record_promotes_lineage_and_evidence_fields(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:lineage",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retained",
            metrics_summary={
                "selected_variant": {"variant_id": "routing_depth"},
                "prior_retained_cycle_id": "cycle:retrieval:baseline",
                "baseline_pass_rate": 0.72,
                "candidate_pass_rate": 0.81,
                "baseline_average_steps": 3.0,
                "candidate_average_steps": 2.0,
                "phase_gate_passed": True,
            },
            candidate_artifact_path="candidates/retrieval.json",
            active_artifact_path="retrieval.json",
        ),
    )

    record = planner.load_cycle_records(output)[0]

    assert record["selected_variant_id"] == "routing_depth"
    assert record["prior_retained_cycle_id"] == "cycle:retrieval:baseline"
    assert record["baseline_pass_rate"] == 0.72
    assert record["candidate_pass_rate"] == 0.81
    assert record["baseline_average_steps"] == 3.0
    assert record["candidate_average_steps"] == 2.0
    assert record["phase_gate_passed"] is True


def test_improvement_cycle_record_rejects_subsystem_artifact_mismatch(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"

    try:
        planner.append_cycle_record(
            output,
            ImprovementCycleRecord(
                cycle_id="cycle:test:mismatch",
                state="generate",
                subsystem="retrieval",
                action="synthesize_benchmarks",
                artifact_path="benchmarks.json",
                artifact_kind="benchmark_candidate_set",
                reason="test",
                metrics_summary={"total": 1},
            ),
        )
    except ValueError as exc:
        assert "does not match subsystem" in str(exc)
    else:
        raise AssertionError("expected cycle record validation failure")


def test_incomplete_cycle_summaries_identify_autonomous_generated_cycles_without_decision(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:pending",
            state="observe",
            subsystem="retrieval",
            action="run_eval",
            artifact_path="",
            artifact_kind="eval_metrics",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:pending",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous", "selected_variant_id": "routing_depth"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:pending",
            state="generate",
            subsystem="retrieval",
            action="propose_retrieval_update",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous", "selected_variant": {"variant_id": "routing_depth"}},
            candidate_artifact_path="candidates/retrieval.json",
            active_artifact_path="retrieval.json",
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:done",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="policy.json",
            artifact_kind="prompt_proposal_set",
            reason="retained",
            metrics_summary={},
        ),
    )

    summaries = planner.incomplete_cycle_summaries(cycles_path, protocol="autonomous")

    assert len(summaries) == 1
    assert summaries[0]["cycle_id"] == "cycle:retrieval:pending"
    assert summaries[0]["subsystem"] == "retrieval"
    assert summaries[0]["candidate_artifact_path"] == "candidates/retrieval.json"
    assert summaries[0]["selected_variant_id"] == "routing_depth"


def test_cycle_audit_summary_surfaces_lineage_and_decision_evidence(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:audit",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"protocol": "autonomous", "selected_variant_id": "routing_depth"},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:audit",
            state="retain",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="retained",
            metrics_summary={
                "baseline_pass_rate": 0.72,
                "candidate_pass_rate": 0.81,
                "baseline_average_steps": 3.0,
                "candidate_average_steps": 2.0,
                "phase_gate_passed": True,
                "prior_retained_cycle_id": "cycle:retrieval:baseline",
            },
            candidate_artifact_path="candidates/retrieval.json",
            active_artifact_path="retrieval.json",
            artifact_lifecycle_state="retained",
            artifact_sha256="sha-new",
            previous_artifact_sha256="sha-old",
            rollback_artifact_path="snapshots/retrieval-baseline.json",
            artifact_snapshot_path="snapshots/retrieval-candidate.json",
        ),
    )

    summary = planner.cycle_audit_summary(cycles_path, cycle_id="cycle:retrieval:audit")

    assert summary is not None
    assert summary["selected_variant_id"] == "routing_depth"
    assert summary["prior_retained_cycle_id"] == "cycle:retrieval:baseline"
    assert summary["final_state"] == "retain"
    assert summary["candidate_artifact_path"] == "candidates/retrieval.json"
    assert summary["active_artifact_path"] == "retrieval.json"
    assert summary["baseline_pass_rate"] == 0.72
    assert summary["candidate_pass_rate"] == 0.81
    assert summary["phase_gate_passed"] is True
    assert summary["artifact_sha256"] == "sha-new"


def test_curriculum_improvement_emits_artifact():
    metrics = EvalMetrics(
        total=10,
        passed=9,
        generated_total=6,
        generated_passed=2,
        generated_by_kind={"failure_recovery": 3, "adjacent_success": 3},
        generated_passed_by_kind={"failure_recovery": 1, "adjacent_success": 1},
        generated_by_benchmark_family={"workflow": 2, "tooling": 4},
        generated_passed_by_benchmark_family={"workflow": 1, "tooling": 1},
    )

    artifact = build_curriculum_proposal_artifact(metrics)

    assert artifact["artifact_kind"] == "curriculum_proposal_set"
    assert artifact["retention_gate"]["require_failure_recovery_improvement"] is True
    assert artifact["proposals"]


def test_curriculum_improvement_applies_variant_focus():
    metrics = EvalMetrics(
        total=10,
        passed=9,
        generated_total=6,
        generated_passed=2,
        generated_by_kind={"failure_recovery": 3, "adjacent_success": 3},
        generated_passed_by_kind={"failure_recovery": 1, "adjacent_success": 1},
        generated_by_benchmark_family={"workflow": 2, "tooling": 4},
        generated_passed_by_benchmark_family={"workflow": 1, "tooling": 1},
    )

    artifact = build_curriculum_proposal_artifact(metrics, focus="benchmark_family", family="workflow")

    assert artifact["generation_focus"] == "benchmark_family"
    assert artifact["proposals"]
    assert all(proposal["area"] in {"benchmark_family", "failure_recovery"} for proposal in artifact["proposals"])
    assert any("workflow" in proposal["reason"] for proposal in artifact["proposals"] if proposal["area"] == "benchmark_family")


def test_evaluate_artifact_retention_rejects_unchanged_policy_candidate():
    baseline = EvalMetrics(total=10, passed=9, average_steps=1.5)
    candidate = EvalMetrics(total=10, passed=9, average_steps=1.5)

    state, _ = evaluate_artifact_retention("policy", baseline, candidate)

    assert state == "reject"


def test_evaluate_artifact_retention_retains_curriculum_improvement():
    baseline = EvalMetrics(
        total=10,
        passed=9,
        generated_total=10,
        generated_passed=4,
        generated_by_kind={"failure_recovery": 4},
        generated_passed_by_kind={"failure_recovery": 1},
    )
    candidate = EvalMetrics(
        total=10,
        passed=9,
        generated_total=10,
        generated_passed=6,
        generated_by_kind={"failure_recovery": 4},
        generated_passed_by_kind={"failure_recovery": 2},
    )

    state, _ = evaluate_artifact_retention("curriculum", baseline, candidate)

    assert state == "retain"


def test_evaluate_artifact_retention_rejects_policy_regression_on_generated_lane():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.0, generated_total=10, generated_passed=7)
    candidate = EvalMetrics(total=10, passed=9, average_steps=1.0, generated_total=10, generated_passed=6)

    state, _ = evaluate_artifact_retention("policy", baseline, candidate)

    assert state == "reject"


def test_evaluate_artifact_retention_rejects_policy_failure_recovery_regression():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.0,
        generated_total=10,
        generated_passed=7,
        generated_by_kind={"failure_recovery": 4},
        generated_passed_by_kind={"failure_recovery": 3},
        total_by_origin_benchmark_family={"workflow": 5, "project": 5},
        passed_by_origin_benchmark_family={"workflow": 4, "project": 4},
    )
    candidate = EvalMetrics(
        total=10,
        passed=9,
        average_steps=1.0,
        generated_total=10,
        generated_passed=8,
        generated_by_kind={"failure_recovery": 4},
        generated_passed_by_kind={"failure_recovery": 2},
        total_by_origin_benchmark_family={"workflow": 5, "project": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "project": 4},
    )

    state, reason = evaluate_artifact_retention("policy", baseline, candidate)

    assert state == "reject"
    assert "failure-recovery" in reason


def test_evaluate_artifact_retention_retains_skill_gain():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.2)

    state, _ = evaluate_artifact_retention("skills", baseline, candidate)

    assert state == "retain"


def test_evaluate_artifact_retention_uses_verifier_gate_and_evidence():
    baseline = EvalMetrics(total=10, passed=9, average_steps=1.0)
    candidate = EvalMetrics(
        total=11,
        passed=10,
        average_steps=1.0,
        total_by_benchmark_family={"verifier_candidate": 1},
        passed_by_benchmark_family={"verifier_candidate": 1},
    )

    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "verifier_candidate_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "min_discrimination_gain": 0.02,
            "max_false_failure_rate": 0.01,
            "require_contract_strictness": True,
        },
        "proposals": [
            {
                "proposal_id": "verifier:math_task:strict",
                "source_task_id": "math_task",
                "benchmark_family": "micro",
                "contract": {
                    "expected_files": ["result.txt"],
                    "forbidden_files": [],
                    "expected_file_contents": {"result.txt": "42\n"},
                    "forbidden_output_substrings": ["3"],
                },
                "evidence": {
                    "strict_contract": True,
                    "discrimination_gain_estimate": 0.03,
                }
            }
        ],
    }

    state, _ = evaluate_artifact_retention("verifier", baseline, candidate, payload=payload)

    assert state == "retain"


def test_evaluate_artifact_retention_rejects_family_regression_for_retrieval():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.0,
        trusted_retrieval_steps=4,
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "tooling": 3},
    )
    candidate = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.0,
        trusted_retrieval_steps=5,
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 4, "tooling": 4},
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "overrides": {"tolbert_top_branches": 4},
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.02,
            "require_family_discrimination": True,
            "max_false_failure_rate": 0.02,
            "required_confirmation_runs": 2,
            "max_regressed_families": 0,
        },
        "proposals": [
            {
                "proposal_id": "retrieval:routing_depth",
                "area": "routing",
                "reason": "test retrieval routing depth",
                "overrides": {"tolbert_top_branches": 4},
            }
        ],
    }

    state, _ = evaluate_artifact_retention("retrieval", baseline, candidate, payload=payload)

    assert state == "reject"


def test_evaluate_artifact_retention_rejects_retrieval_failure_recovery_regression():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.0,
        trusted_retrieval_steps=4,
        generated_total=8,
        generated_passed=5,
        generated_by_kind={"failure_recovery": 4, "adjacent_success": 4},
        generated_passed_by_kind={"failure_recovery": 3, "adjacent_success": 2},
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 4, "tooling": 4},
    )
    candidate = EvalMetrics(
        total=10,
        passed=9,
        average_steps=1.0,
        trusted_retrieval_steps=5,
        generated_total=8,
        generated_passed=6,
        generated_by_kind={"failure_recovery": 4, "adjacent_success": 4},
        generated_passed_by_kind={"failure_recovery": 2, "adjacent_success": 4},
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "tooling": 4},
    )

    state, reason = evaluate_artifact_retention("retrieval", baseline, candidate)

    assert state == "reject"
    assert "failure-recovery" in reason


def test_evaluate_artifact_retention_retains_capability_surface_growth_without_runtime_regression(tmp_path: Path):
    active_path = tmp_path / "config" / "capabilities.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "capabilities": ["github_read"],
                        "settings": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "capability_module_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.0},
        "modules": [
            {
                "module_id": "github",
                "enabled": True,
                "capabilities": ["github_read"],
                "settings": {
                    "improvement_subsystems": [
                        {
                            "subsystem_id": "github_policy",
                            "base_subsystem": "policy",
                            "reason": "github policy remains under-optimized",
                            "priority": 4,
                            "expected_gain": 0.02,
                            "estimated_cost": 2,
                        }
                    ]
                },
            }
        ],
        "generation_context": {"active_artifact_path": str(active_path)},
    }
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=2, generated_passed=1)
    candidate = EvalMetrics(total=10, passed=8, average_steps=1.5, generated_total=2, generated_passed=1)

    state, reason = evaluate_artifact_retention("capabilities", baseline, candidate, payload=payload)

    assert state == "retain"
    assert "capability registry candidate preserved runtime quality" in reason


def test_evaluate_artifact_retention_rejects_curriculum_generated_family_regression():
    baseline = EvalMetrics(
        total=10,
        passed=9,
        generated_total=8,
        generated_passed=5,
        generated_by_kind={"failure_recovery": 4, "adjacent_success": 4},
        generated_passed_by_kind={"failure_recovery": 3, "adjacent_success": 2},
        generated_by_benchmark_family={"workflow": 4, "tooling": 4},
        generated_passed_by_benchmark_family={"workflow": 3, "tooling": 2},
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "tooling": 4},
    )
    candidate = EvalMetrics(
        total=10,
        passed=9,
        generated_total=8,
        generated_passed=6,
        generated_by_kind={"failure_recovery": 4, "adjacent_success": 4},
        generated_passed_by_kind={"failure_recovery": 4, "adjacent_success": 2},
        generated_by_benchmark_family={"workflow": 4, "tooling": 4},
        generated_passed_by_benchmark_family={"workflow": 4, "tooling": 1},
        total_by_origin_benchmark_family={"workflow": 5, "tooling": 5},
        passed_by_origin_benchmark_family={"workflow": 5, "tooling": 4},
    )

    state, reason = evaluate_artifact_retention("curriculum", baseline, candidate)

    assert state == "reject"
    assert "generated benchmark families" in reason


def test_evaluate_artifact_retention_uses_operator_transfer_gate():
    baseline = EvalMetrics(total=1, passed=0, total_by_memory_source={"skill_transfer": 1}, passed_by_memory_source={})
    candidate = EvalMetrics(
        total=1,
        passed=1,
        total_by_memory_source={"operator": 1},
        passed_by_memory_source={"operator": 1},
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "operator_class_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(OPERATOR_RETENTION_GATE),
        "operators": [
            {
                "operator_id": "operator:file_write:bounded",
                "support": 2,
                "benchmark_families": ["micro"],
                "steps": ["printf 'hello agent kernel\\n' > hello.txt"],
                "task_contract": {"expected_files": ["hello.txt"]},
                "source_task_ids": ["hello_task", "math_task"],
                "template_procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "template_contract": {"expected_files": ["hello.txt"]},
            }
        ],
    }

    state, _ = evaluate_artifact_retention("operators", baseline, candidate, payload=payload)

    assert state == "retain"


def test_apply_artifact_retention_decision_updates_tool_artifact(tmp_path: Path):
    artifact_path = tmp_path / "tools.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "spec_version": "asi_v1",
                        "tool_id": "tool:service_mesh_task:primary",
                        "kind": "local_shell_procedure",
                        "lifecycle_state": "candidate",
                        "promotion_stage": "candidate_procedure",
                        "source_task_id": "service_mesh_task",
                        "benchmark_family": "workflow",
                        "quality": 0.9,
                        "script_name": "service_mesh_task_tool.sh",
                        "script_body": "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'ready\\n' > status.txt\n",
                        "procedure": {"commands": ["printf 'ready\\n' > status.txt"]},
                        "task_contract": {"expected_files": ["status.txt"]},
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=artifact_path,
        subsystem="tooling",
        cycle_id="cycle:tooling:1",
        decision_state="retain",
        decision_reason="verified replay and retained gain",
        baseline_metrics=EvalMetrics(total=10, passed=8, average_steps=1.5),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.2),
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "retained"
    assert payload["candidates"][0]["promotion_stage"] == "promoted_tool"
    assert payload["retention_decision"]["cycle_id"] == "cycle:tooling:1"
    assert summary["artifact_lifecycle_state"] == "retained"
    assert summary["artifact_sha256"]
    assert Path(str(summary["artifact_snapshot_path"])).exists()


def test_apply_artifact_retention_decision_marks_rejected_skill_artifact(tmp_path: Path):
    artifact_path = tmp_path / "skills.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "promoted",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:primary")],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=artifact_path,
        subsystem="skills",
        cycle_id="cycle:skills:1",
        decision_state="reject",
        decision_reason="no verified gain",
        baseline_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "rejected"
    assert payload["compatibility"]["compatible"] is True
    assert summary["previous_artifact_sha256"]
    assert Path(str(summary["artifact_snapshot_path"])).exists()


def test_evaluate_artifact_retention_rejects_incompatible_artifact_before_metric_gate():
    baseline = EvalMetrics(total=10, passed=8, average_steps=1.2)
    candidate = EvalMetrics(total=10, passed=10, average_steps=1.0)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(SKILL_RETENTION_GATE),
        "skills": [_skill_record("skill:hello_task:bad", command="printf 'oops\\n' > unrelated.txt")],
    }

    state, reason = evaluate_artifact_retention("skills", baseline, candidate, payload=payload)

    assert state == "reject"
    assert "not obviously aligned" in reason


def test_materialize_replay_verified_tool_payload_marks_candidates_replay_verified():
    payload = {
        "artifact_kind": "tool_candidate_set",
        "candidates": [
            {
                "tool_id": "tool:service_mesh_task:primary",
                "promotion_stage": "candidate_procedure",
            }
        ],
    }

    replay_verified = materialize_replay_verified_tool_payload(payload)

    assert replay_verified["candidates"][0]["promotion_stage"] == "replay_verified"
    assert replay_verified["candidates"][0]["lifecycle_state"] == "replay_verified"


def test_effective_artifact_payload_for_retention_marks_tool_candidates_replay_verified():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "candidate",
        "candidates": [
            {
                "tool_id": "tool:service_mesh_task:primary",
                "promotion_stage": "candidate_procedure",
            }
        ],
    }

    effective = effective_artifact_payload_for_retention("tooling", payload)

    assert payload["candidates"][0]["promotion_stage"] == "candidate_procedure"
    assert effective["lifecycle_state"] == "replay_verified"
    assert effective["candidates"][0]["promotion_stage"] == "replay_verified"
    assert effective["candidates"][0]["lifecycle_state"] == "replay_verified"


def test_effective_artifact_payload_for_retention_supports_module_defined_tooling_subsystem(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_tooling",
                                    "base_subsystem": "tooling",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "candidate",
        "candidates": [
            {
                "tool_id": "tool:service_mesh_task:primary",
                "kind": "local_shell_procedure",
                "promotion_stage": "candidate_procedure",
            }
        ],
    }

    effective = effective_artifact_payload_for_retention(
        "github_tooling",
        payload,
        capability_modules_path=modules_path,
    )

    assert effective["lifecycle_state"] == "replay_verified"
    assert effective["candidates"][0]["promotion_stage"] == "replay_verified"


def test_persist_replay_verified_tool_artifact_updates_file(tmp_path: Path):
    artifact_path = tmp_path / "tools.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "spec_version": "asi_v1",
                        "tool_id": "tool:x:primary",
                        "kind": "local_shell_procedure",
                        "lifecycle_state": "candidate",
                        "promotion_stage": "candidate_procedure",
                        "source_task_id": "x",
                        "benchmark_family": "workflow",
                        "quality": 0.8,
                        "script_name": "x_tool.sh",
                        "script_body": "#!/usr/bin/env bash\nset -euo pipefail\ntrue\n",
                        "procedure": {"commands": ["true"]},
                        "task_contract": {"expected_files": []},
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = persist_replay_verified_tool_artifact(artifact_path, cycle_id="cycle:tooling:1")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["lifecycle_state"] == "replay_verified"
    assert payload["candidates"][0]["promotion_stage"] == "replay_verified"
    assert summary["artifact_lifecycle_state"] == "replay_verified"
    assert Path(str(summary["rollback_artifact_path"])).exists()
    assert Path(str(summary["rollback_artifact_path"])) != artifact_path
    assert Path(str(summary["artifact_snapshot_path"])).exists()


def test_skill_retention_rejects_generated_lane_regression():
    baseline = EvalMetrics(
        total=10,
        passed=8,
        average_steps=1.0,
        generated_total=4,
        generated_passed=3,
    )
    candidate = EvalMetrics(
        total=10,
        passed=9,
        average_steps=1.0,
        generated_total=4,
        generated_passed=2,
    )

    state, reason = evaluate_artifact_retention("skills", baseline, candidate)

    assert state == "reject"
    assert reason == "candidate regressed the generated-task lane"


def test_staged_candidate_artifact_path_uses_candidate_root(tmp_path: Path):
    active_artifact_path = tmp_path / "skills" / "command_skills.json"
    candidate_path = staged_candidate_artifact_path(
        active_artifact_path,
        candidates_root=tmp_path / "improvement" / "candidates",
        subsystem="skills",
        cycle_id="cycle:skills:1",
    )

    assert candidate_path != active_artifact_path
    assert candidate_path.parent.name == "cycle_skills_1"
    assert candidate_path.name == "command_skills.json"


def test_apply_artifact_retention_decision_promotes_staged_candidate_on_retain(tmp_path: Path):
    live_artifact_path = tmp_path / "skills" / "command_skills.json"
    live_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    live_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:current")],
            }
        ),
        encoding="utf-8",
    )
    candidate_artifact_path = tmp_path / "improvement" / "candidates" / "skills.json"
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:candidate")],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=candidate_artifact_path,
        active_artifact_path=live_artifact_path,
        subsystem="skills",
        cycle_id="cycle:skills:stage_retain",
        decision_state="retain",
        decision_reason="verified staged gain",
        baseline_metrics=EvalMetrics(total=10, passed=8, average_steps=1.2),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    live_payload = json.loads(live_artifact_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_artifact_path.read_text(encoding="utf-8"))

    assert live_payload["skills"][0]["skill_id"] == "skill:hello_task:candidate"
    assert live_payload["lifecycle_state"] == "retained"
    assert candidate_payload["lifecycle_state"] == "retained"
    assert summary["active_artifact_path"] == str(live_artifact_path)
    assert summary["candidate_artifact_path"] == str(candidate_artifact_path)
    assert Path(str(summary["rollback_artifact_path"])).exists()


def test_apply_artifact_retention_decision_materializes_split_universe_artifacts_from_legacy_retain(tmp_path: Path):
    config = KernelConfig(
        universe_contract_path=tmp_path / "universe" / "universe_contract.json",
        universe_constitution_path=tmp_path / "universe" / "universe_constitution.json",
        operating_envelope_path=tmp_path / "universe" / "operating_envelope.json",
    )
    config.ensure_directories()
    payload = _retained_universe_artifact(
        network_access_mode="allowlist_only",
        git_write_mode="operator_gated",
        workspace_write_scope="task_only",
    )
    payload["lifecycle_state"] = "candidate"
    config.universe_contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = apply_artifact_retention_decision(
        artifact_path=config.universe_contract_path,
        subsystem="universe",
        cycle_id="cycle:universe:split_sync",
        decision_state="retain",
        decision_reason="verified universe gain",
        baseline_metrics=EvalMetrics(total=10, passed=8, average_steps=1.5),
        candidate_metrics=EvalMetrics(total=10, passed=8, average_steps=1.4),
        runtime_config=config,
    )

    constitution_payload = json.loads(config.universe_constitution_path.read_text(encoding="utf-8"))
    envelope_payload = json.loads(config.operating_envelope_path.read_text(encoding="utf-8"))

    assert constitution_payload["artifact_kind"] == "universe_constitution"
    assert constitution_payload["lifecycle_state"] == "retained"
    assert constitution_payload["control_schema"] == "universe_constitution_v1"
    assert envelope_payload["artifact_kind"] == "operating_envelope"
    assert envelope_payload["lifecycle_state"] == "retained"
    assert envelope_payload["control_schema"] == "operating_envelope_v1"
    assert envelope_payload["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert summary["synchronized_artifact_paths"]["universe_constitution"] == str(config.universe_constitution_path)
    assert summary["synchronized_artifact_paths"]["operating_envelope"] == str(config.operating_envelope_path)


def test_apply_artifact_retention_decision_refreshes_legacy_universe_contract_from_split_retain(tmp_path: Path):
    config = KernelConfig(
        universe_contract_path=tmp_path / "universe" / "universe_contract.json",
        universe_constitution_path=tmp_path / "universe" / "universe_constitution.json",
        operating_envelope_path=tmp_path / "universe" / "operating_envelope.json",
    )
    config.ensure_directories()
    config.universe_constitution_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "retained",
                "control_schema": "universe_constitution_v1",
                "retention_gate": {"require_non_regression": True},
                "governance": {
                    "require_verification": False,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": False,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": False,
                },
                "invariants": ["preserve verifier contract alignment"],
                "forbidden_command_patterns": ["rm -rf /"],
                "preferred_command_prefixes": ["rg "],
                "proposals": [{"area": "governance", "priority": 5, "reason": "baseline", "suggestion": "baseline"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config.operating_envelope_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "operating_envelope",
                "lifecycle_state": "retained",
                "control_schema": "operating_envelope_v1",
                "retention_gate": {"require_non_regression": True},
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 8,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 6,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 10,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 11,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "network_access_mode": "allowlist_only",
                    "git_write_mode": "operator_gated",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.example.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["git", "python", "pytest", "rg"],
                "learned_calibration_priors": {"selected_variant_weights": {"environment_envelope": 2.0}},
                "proposals": [{"area": "environment_envelope", "priority": 4, "reason": "baseline", "suggestion": "baseline"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_artifact_path = tmp_path / "candidates" / "universe_constitution.json"
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "candidate",
                "control_schema": "universe_constitution_v1",
                "retention_gate": {"require_non_regression": True},
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "invariants": ["verify before accepting terminal success"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest -q"],
                "proposals": [{"area": "verification", "priority": 6, "reason": "candidate", "suggestion": "candidate"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=candidate_artifact_path,
        active_artifact_path=config.universe_constitution_path,
        subsystem="universe_constitution",
        cycle_id="cycle:universe_constitution:retain",
        decision_state="retain",
        decision_reason="verified constitutional gain",
        baseline_metrics=EvalMetrics(total=10, passed=8, average_steps=1.5),
        candidate_metrics=EvalMetrics(total=10, passed=8, average_steps=1.4),
        runtime_config=config,
    )

    combined_payload = json.loads(config.universe_contract_path.read_text(encoding="utf-8"))

    assert combined_payload["artifact_kind"] == "universe_contract"
    assert combined_payload["lifecycle_state"] == "retained"
    assert combined_payload["governance"]["require_verification"] is True
    assert combined_payload["preferred_command_prefixes"] == ["pytest -q"]
    assert combined_payload["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert combined_payload["allowed_http_hosts"] == ["api.example.com"]
    assert summary["synchronized_artifact_paths"]["universe"] == str(config.universe_contract_path)


def test_apply_artifact_retention_decision_rejects_staged_candidate_without_mutating_live_artifact(tmp_path: Path):
    live_artifact_path = tmp_path / "skills" / "command_skills.json"
    live_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    live_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:current")],
            }
        ),
        encoding="utf-8",
    )
    candidate_artifact_path = tmp_path / "improvement" / "candidates" / "skills.json"
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:candidate")],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=candidate_artifact_path,
        active_artifact_path=live_artifact_path,
        subsystem="skills",
        cycle_id="cycle:skills:stage_reject",
        decision_state="reject",
        decision_reason="no verified gain",
        baseline_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    live_payload = json.loads(live_artifact_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_artifact_path.read_text(encoding="utf-8"))

    assert live_payload["skills"][0]["skill_id"] == "skill:hello_task:current"
    assert live_payload["lifecycle_state"] == "retained"
    assert candidate_payload["lifecycle_state"] == "rejected"
    assert summary["artifact_lifecycle_state"] == "rejected"
    assert summary["artifact_sha256"] == summary["previous_artifact_sha256"]


def test_improvement_planner_exposes_cycle_and_artifact_history_queries(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "tools.json"
    pre_snapshot = tmp_path / ".artifact_history" / "tools.cycle_tooling_1.pre.json"
    post_snapshot = tmp_path / ".artifact_history" / "tools.cycle_tooling_1.post.json"
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:tooling:1",
            state="evaluate",
            subsystem="tooling",
            action="compare_candidate_to_baseline",
            artifact_path=str(artifact_path),
            artifact_kind="tool_candidate_set",
            reason="measured candidate lane",
            metrics_summary={"baseline_pass_rate": 0.8, "candidate_pass_rate": 0.9},
            artifact_lifecycle_state="replay_verified",
            artifact_sha256="sha-new",
            previous_artifact_sha256="sha-old",
            rollback_artifact_path=str(pre_snapshot),
            artifact_snapshot_path=str(pre_snapshot),
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:tooling:1",
            state="retain",
            subsystem="tooling",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="tool_candidate_set",
            reason="candidate improved pass rate",
            metrics_summary={"baseline_pass_rate": 0.8, "candidate_pass_rate": 0.9},
            artifact_lifecycle_state="retained",
            artifact_sha256="sha-final",
            previous_artifact_sha256="sha-new",
            rollback_artifact_path=str(post_snapshot),
            artifact_snapshot_path=str(post_snapshot),
        ),
    )

    history = planner.cycle_history(output, cycle_id="cycle:tooling:1")
    artifact_history = planner.artifact_history(output, artifact_path)
    latest_record = planner.latest_artifact_record(output, artifact_path)
    latest_decision = planner.latest_artifact_decision(output, artifact_path)
    rollback = planner.artifact_rollback_metadata(output, artifact_path)

    assert [record["state"] for record in history] == ["evaluate", "retain"]
    assert len(artifact_history) == 2
    assert latest_record["state"] == "retain"
    assert latest_decision["state"] == "retain"
    assert rollback["artifact_sha256"] == "sha-final"
    assert rollback["previous_artifact_sha256"] == "sha-new"
    assert rollback["artifact_snapshot_path"] == str(post_snapshot)


def test_apply_artifact_retention_decision_creates_real_snapshot(tmp_path: Path):
    artifact_path = tmp_path / "skills.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "promoted",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:primary")],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_artifact_retention_decision(
        artifact_path=artifact_path,
        subsystem="skills",
        cycle_id="cycle:skills:real_snapshot",
        decision_state="reject",
        decision_reason="no verified gain",
        baseline_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    snapshot_path = Path(str(summary["rollback_artifact_path"]))
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot_path.exists()
    assert snapshot_path != artifact_path
    assert snapshot_payload["lifecycle_state"] == "promoted"


def test_apply_artifact_retention_decision_restores_prior_active_artifact_on_reject(tmp_path: Path):
    artifact_path = tmp_path / "skills.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:primary")],
            }
        ),
        encoding="utf-8",
    )
    prior_active = snapshot_artifact_state(
        artifact_path,
        cycle_id="cycle:skills:restore",
        stage="pre_generate_active",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "promoted",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:bad", command="printf 'oops\\n' > unrelated.txt")],
            }
        ),
        encoding="utf-8",
    )
    stamp_artifact_generation_context(
        artifact_path,
        cycle_id="cycle:skills:restore",
        prior_active_artifact_path=prior_active,
    )

    summary = apply_artifact_retention_decision(
        artifact_path=artifact_path,
        subsystem="skills",
        cycle_id="cycle:skills:restore",
        decision_state="reject",
        decision_reason="compatibility failed",
        baseline_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    restored_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    rejected_snapshot = json.loads(Path(str(summary["artifact_snapshot_path"])).read_text(encoding="utf-8"))

    assert restored_payload["skills"][0]["skill_id"] == "skill:hello_task:primary"
    assert rejected_snapshot["retention_decision"]["state"] == "reject"
    assert summary["artifact_lifecycle_state"] == "rejected_restored"


def test_apply_artifact_retention_decision_compacts_rejected_tolbert_candidate_and_gcs_store(
    tmp_path: Path,
    monkeypatch,
):
    live_artifact_path = tmp_path / "tolbert_model" / "tolbert_model_artifact.json"
    candidate_artifact_path = tmp_path / "candidates" / "tolbert_model" / "cycle_tolbert_1" / "tolbert_model_artifact.json"
    live_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    keep_store = live_artifact_path.parent / "store" / "training" / "keep_digest"
    drop_store = live_artifact_path.parent / "store" / "training" / "drop_digest"
    keep_store.mkdir(parents=True, exist_ok=True)
    drop_store.mkdir(parents=True, exist_ok=True)
    (keep_store / "checkpoint.pt").write_bytes(b"keep")
    (drop_store / "checkpoint.pt").write_bytes(b"drop")

    live_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "retained",
                "shared_store": {"entries": {"training": {"path": str(keep_store)}}},
            }
        ),
        encoding="utf-8",
    )

    candidate_output_dir = candidate_artifact_path.parent / "tolbert_model_artifact"
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    (candidate_output_dir / "marker.txt").write_text("candidate", encoding="utf-8")
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "candidate",
                "generation_focus": "balanced",
                "retention_gate": {"require_non_regression": True},
                "training_controls": {"base_model_name": "bert-base-uncased", "num_epochs": 2},
                "dataset_manifest": {
                    "artifact_kind": "tolbert_supervised_dataset",
                    "manifest_path": str(drop_store / "dataset_manifest.json"),
                    "supervised_examples_path": str(drop_store / "supervised_examples.jsonl"),
                    "total_examples": 4,
                },
                "runtime_policy": {"primary_benchmark_families": ["workflow"]},
                "shared_store": {"entries": {"training": {"path": str(drop_store)}}},
                "runtime_paths": {"checkpoint_path": str(drop_store / "checkpoint.pt")},
                "output_dir": str(candidate_output_dir),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent_kernel.improvement.assess_artifact_compatibility",
        lambda **kwargs: {"compatible": True, "checked_rules": [], "violations": []},
    )

    summary = apply_artifact_retention_decision(
        artifact_path=candidate_artifact_path,
        active_artifact_path=live_artifact_path,
        subsystem="tolbert_model",
        cycle_id="cycle:tolbert_model:reject",
        decision_state="reject",
        decision_reason="no verified gain",
        baseline_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    candidate_payload = json.loads(candidate_artifact_path.read_text(encoding="utf-8"))
    assert candidate_payload["lifecycle_state"] == "rejected"
    assert candidate_payload["materialization_mode"] == "rejected_manifest_only"
    assert "shared_store" not in candidate_payload
    assert "runtime_paths" not in candidate_payload
    assert "output_dir" not in candidate_payload
    assert candidate_payload["lineage"]["rejected_candidate_policy"] == "metrics_and_mutation_record_only"
    assert summary["artifact_lifecycle_state"] == "rejected"
    assert summary["rejected_storage_gc"]["removed_output_dir"] == str(candidate_output_dir)
    assert str(drop_store) in summary["rejected_storage_gc"]["removed_shared_store"]
    assert keep_store.exists()
    assert not drop_store.exists()
    assert not candidate_output_dir.exists()


def test_apply_artifact_retention_decision_promotes_tolbert_delta_to_canonical_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    live_artifact_path = tmp_path / "tolbert_model" / "tolbert_model_artifact.json"
    candidate_artifact_path = tmp_path / "candidates" / "tolbert_model" / "cycle_tolbert_2" / "tolbert_model_artifact.json"
    live_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = tmp_path / "parent.pt"
    delta_checkpoint = tmp_path / "delta.pt"
    parent_checkpoint.write_bytes(b"parent")
    delta_checkpoint.write_bytes(b"delta")
    live_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {"checkpoint_path": str(parent_checkpoint)},
            }
        ),
        encoding="utf-8",
    )
    candidate_artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "candidate",
                "runtime_paths": {
                    "checkpoint_path": "",
                    "parent_checkpoint_path": str(parent_checkpoint),
                    "checkpoint_delta_path": str(delta_checkpoint),
                },
                "parameter_delta": {
                    "delta_checkpoint_path": str(delta_checkpoint),
                    "parent_checkpoint_path": str(parent_checkpoint),
                    "stats": {"changed_key_count": 1},
                },
                "lineage": {"mode": "canonical_parent_mutation"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent_kernel.improvement.assess_artifact_compatibility",
        lambda **kwargs: {"compatible": True, "checked_rules": [], "violations": []},
    )

    def _materialize(*, parent_checkpoint_path, delta_checkpoint_path, output_checkpoint_path):
        output_checkpoint_path.write_bytes(b"materialized")
        return output_checkpoint_path

    monkeypatch.setattr("agent_kernel.improvement.materialize_tolbert_checkpoint_from_delta", _materialize)

    summary = apply_artifact_retention_decision(
        artifact_path=candidate_artifact_path,
        active_artifact_path=live_artifact_path,
        subsystem="tolbert_model",
        cycle_id="cycle:tolbert_model:retain",
        decision_state="retain",
        decision_reason="verified gain",
        baseline_metrics=EvalMetrics(total=10, passed=8, average_steps=1.0),
        candidate_metrics=EvalMetrics(total=10, passed=9, average_steps=1.0),
    )

    live_payload = json.loads(live_artifact_path.read_text(encoding="utf-8"))
    assert live_payload["runtime_paths"]["checkpoint_path"].endswith("tolbert_cycle_tolbert_model_retain.pt")
    assert "checkpoint_delta_path" not in live_payload["runtime_paths"]
    assert live_payload["parameter_delta"]["promotion_applied"] is True
    assert Path(live_payload["runtime_paths"]["checkpoint_path"]).exists()
    assert summary["artifact_lifecycle_state"] == "retained"


def test_prior_retained_artifact_record_respects_cycle_order(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills_v1.json",
            artifact_kind="skill_set",
            reason="first",
            metrics_summary={},
            artifact_snapshot_path="snapshot_v1.json",
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:2",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills_v2.json",
            artifact_kind="skill_set",
            reason="second",
            metrics_summary={},
            artifact_snapshot_path="snapshot_v2.json",
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:3",
            state="select",
            subsystem="skills",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="third",
            metrics_summary={},
        ),
    )

    prior = planner.prior_retained_artifact_record(output, "skills", before_cycle_id="cycle:skills:3")

    assert prior is not None
    assert prior["cycle_id"] == "cycle:skills:2"


def test_prior_retained_and_history_merge_shared_surface_subsystems(tmp_path: Path):
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(
        memory_root=tmp_path / "episodes",
        capability_modules_path=modules_path,
    )
    output = tmp_path / "cycles.jsonl"
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:policy:1",
            state="retain",
            subsystem="policy",
            action="finalize_cycle",
            artifact_path="prompt_v1.json",
            artifact_kind="prompt_proposal_set",
            reason="first",
            metrics_summary={"baseline_pass_rate": 0.8, "candidate_pass_rate": 0.9},
            artifact_snapshot_path="snapshot_v1.json",
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:github_policy:2",
            state="select",
            subsystem="github_policy",
            action="choose_target",
            artifact_path="",
            artifact_kind="improvement_target",
            reason="second",
            metrics_summary={},
        ),
    )

    prior = planner.prior_retained_artifact_record(
        output,
        "github_policy",
        before_cycle_id="cycle:github_policy:2",
    )
    history = planner.subsystem_history_summary(subsystem="github_policy", output_path=output)

    assert prior is not None
    assert prior["cycle_id"] == "cycle:policy:1"
    assert history["retained_cycles"] == 1


def test_improvement_planner_can_restore_artifact_from_snapshot(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "skills.json"
    snapshot_path = tmp_path / ".artifact_history" / "skills.cycle_skills_1.pre.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "promoted",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:primary")],
            }
        ),
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "rejected",
                "retention_gate": dict(SKILL_RETENTION_GATE),
                "skills": [_skill_record("skill:hello_task:primary")],
            }
        ),
        encoding="utf-8",
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="reject",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="skill_set",
            reason="no gain",
            metrics_summary={"baseline_pass_rate": 0.8, "candidate_pass_rate": 0.8},
            rollback_artifact_path=str(snapshot_path),
        ),
    )

    restored = planner.rollback_artifact(output, artifact_path)
    restored_payload = json.loads(restored.read_text(encoding="utf-8"))

    assert restored == artifact_path
    assert restored_payload["lifecycle_state"] == "promoted"


def test_improvement_planner_summarizes_retained_gain_over_time(tmp_path: Path):
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    output = tmp_path / "cycles.jsonl"
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills.json",
            artifact_kind="skill_set",
            reason="improved",
            metrics_summary={
                "baseline_pass_rate": 0.70,
                "candidate_pass_rate": 0.80,
                "baseline_average_steps": 1.5,
                "candidate_average_steps": 1.2,
            },
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:tooling:1",
            state="reject",
            subsystem="tooling",
            action="finalize_cycle",
            artifact_path="tools.json",
            artifact_kind="tool_candidate_set",
            reason="no gain",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.75,
                "baseline_average_steps": 1.0,
                "candidate_average_steps": 1.1,
            },
        ),
    )
    planner.append_cycle_record(
        output,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:2",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path="skills_v2.json",
            artifact_kind="skill_set",
            reason="improved again",
            metrics_summary={
                "baseline_pass_rate": 0.80,
                "candidate_pass_rate": 0.85,
                "baseline_average_steps": 1.3,
                "candidate_average_steps": 1.1,
            },
        ),
    )

    summary = planner.retained_gain_summary(output)
    skills_summary = planner.retained_gain_summary(output, subsystem="skills")

    assert summary.total_decisions == 3
    assert summary.retained_cycles == 2
    assert summary.rejected_cycles == 1
    assert summary.retained_by_subsystem["skills"] == 2
    assert summary.rejected_by_subsystem["tooling"] == 1
    assert round(summary.average_retained_pass_rate_delta, 3) == 0.075
    assert round(summary.average_retained_step_delta, 3) == -0.25
    assert round(summary.average_rejected_pass_rate_delta, 3) == -0.05
    assert round(summary.average_rejected_step_delta, 3) == 0.1
    assert skills_summary.retained_cycles == 2
    assert skills_summary.rejected_cycles == 0


def test_semantic_compatibility_rejects_verifier_contract_that_is_not_stricter():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "verifier_candidate_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_discrimination_gain": 0.02},
        "proposals": [
            {
                "proposal_id": "verifier:hello_task:weak",
                "source_task_id": "hello_task",
                "benchmark_family": "micro",
                "contract": {
                    "expected_files": ["hello.txt"],
                    "forbidden_files": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\n"},
                    "forbidden_output_substrings": [],
                },
            }
        ],
    }

    compatibility = assess_artifact_compatibility(subsystem="verifier", payload=payload)

    assert compatibility["compatible"] is False
    assert any("does not strengthen" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_rejects_prompt_proposal_missing_required_fields():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "prompt_proposal_set",
        "lifecycle_state": "proposed",
        "retention_gate": {"min_pass_rate_delta_abs": 0.01},
        "proposals": [{"area": "decision", "priority": 5, "reason": "test"}],
    }

    compatibility = assess_artifact_compatibility(subsystem="policy", payload=payload)

    assert compatibility["compatible"] is False
    assert any("suggestion" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_accepts_operator_asi_alias_fields():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "operator_class_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(OPERATOR_RETENTION_GATE),
        "operators": [
            {
                "operator_id": "operator:file_write:bounded",
                "support": 2,
                "benchmark_families": ["micro"],
                "steps": ["printf 'hello agent kernel\\n' > hello.txt"],
                "task_contract": {"expected_files": ["hello.txt"]},
                "source_task_ids": ["hello_task", "math_task"],
            }
        ],
    }

    compatibility = assess_artifact_compatibility(subsystem="operators", payload=payload)

    assert compatibility["compatible"] is True


def test_assess_artifact_compatibility_rejects_tool_candidate_missing_lifecycle_state():
    candidate = _tool_candidate_record()
    candidate.pop("lifecycle_state")
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "candidate",
        "candidates": [candidate],
    }

    compatibility = assess_artifact_compatibility(subsystem="tooling", payload=payload)

    assert compatibility["compatible"] is False
    assert any("lifecycle_state" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_accepts_replay_verified_tool_candidate_state():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "replay_verified",
        "candidates": [
            _tool_candidate_record(
                tool_id="tool:hello_task:replay_verified",
                lifecycle_state="replay_verified",
                promotion_stage="replay_verified",
            )
        ],
    }

    compatibility = assess_artifact_compatibility(subsystem="tooling", payload=payload)

    assert compatibility["compatible"] is True


def test_assess_artifact_compatibility_rejects_tool_artifact_top_level_state_mismatch():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "retained",
        "candidates": [_tool_candidate_record()],
    }

    compatibility = assess_artifact_compatibility(subsystem="tooling", payload=payload)

    assert compatibility["compatible"] is False
    assert any("promotion_stage promoted_tool" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_rejects_prompt_artifact_candidate_lifecycle_state():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "prompt_proposal_set",
        "lifecycle_state": "candidate",
        "retention_gate": {"min_pass_rate_delta_abs": 0.01},
        "proposals": [{"area": "decision", "priority": 5, "reason": "test", "suggestion": "test"}],
    }

    compatibility = assess_artifact_compatibility(subsystem="policy", payload=payload)

    assert compatibility["compatible"] is False
    assert any("artifact lifecycle_state" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_rejects_skill_missing_required_fields():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(SKILL_RETENTION_GATE),
        "skills": [{"skill_id": "skill:hello_task:partial"}],
    }

    compatibility = assess_artifact_compatibility(subsystem="skills", payload=payload)

    assert compatibility["compatible"] is False
    assert any("benchmark_family" in violation for violation in compatibility["violations"])


def test_assess_artifact_compatibility_rejects_skill_missing_retention_gate():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "skills": [_skill_record("skill:hello_task:primary")],
    }

    compatibility = assess_artifact_compatibility(subsystem="skills", payload=payload)

    assert compatibility["compatible"] is False
    assert any("retention_gate" in violation for violation in compatibility["violations"])


def test_semantic_compatibility_rejects_misaligned_skill_procedure():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(SKILL_RETENTION_GATE),
        "skills": [_skill_record("skill:hello_task:bad", command="printf 'oops\\n' > unrelated.txt")],
    }

    compatibility = assess_artifact_compatibility(subsystem="skills", payload=payload)

    assert compatibility["compatible"] is False
    assert any("not obviously aligned" in violation for violation in compatibility["violations"])


def test_semantic_compatibility_accepts_external_manifest_source_task(
    tmp_path: Path,
    monkeypatch,
):
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "external_source_task",
                        "prompt": "Create external.txt containing external ready.",
                        "workspace_subdir": "external_source_task",
                        "expected_files": ["external.txt"],
                        "expected_file_contents": {"external.txt": "external ready\n"},
                        "metadata": {"benchmark_family": "external_lab", "capability": "external_flow"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_KERNEL_EXTERNAL_TASK_MANIFESTS_PATHS", str(manifest_path))
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "retention_gate": dict(SKILL_RETENTION_GATE),
        "skills": [
            _skill_record(
                "skill:external_source_task:ok",
                source_task_id="external_source_task",
                command="printf 'external ready\\n' > external.txt",
                expected_files=["external.txt"],
                benchmark_family="external_lab",
            )
        ],
    }

    compatibility = assess_artifact_compatibility(subsystem="skills", payload=payload)

    assert compatibility["compatible"] is True


def test_assess_artifact_compatibility_rejects_missing_top_level_contract_fields():
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "retrieval_policy_set",
        "proposals": [
            {
                "proposal_id": "retrieval:routing_depth",
                "area": "routing",
                "reason": "test",
                "overrides": {"tolbert_top_branches": 4},
            }
        ],
        "overrides": {"tolbert_top_branches": 4},
    }

    compatibility = assess_artifact_compatibility(subsystem="retrieval", payload=payload)

    assert compatibility["compatible"] is False
    assert any("lifecycle_state" in violation for violation in compatibility["violations"])
    assert any("retention_gate" in violation for violation in compatibility["violations"])


def test_recent_subsystem_activity_summary_tracks_no_yield_and_regression_cycles(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="evaluate",
            subsystem="retrieval",
            action="compare_candidate_to_baseline",
            artifact_path="retrieval.json",
            artifact_kind="retention_evaluation",
            reason="preview",
            metrics_summary={
                "phase_gate_passed": False,
                "phase_gate_failure_count": 1,
                "regressed_task_count": 2,
            },
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:2",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
        ),
    )

    summary = planner.recent_subsystem_activity_summary(subsystem="retrieval", output_path=cycles_path)
    adjusted, reasons = planner._portfolio_adjusted_experiment_score(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.05,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        recent_activity=summary,
    )

    assert summary["selected_cycles"] == 2
    assert summary["no_yield_cycles"] == 2
    assert summary["recent_incomplete_cycles"] == 1
    assert summary["recent_regression_cycles"] == 1
    assert summary["recent_phase_gate_failure_cycles"] == 1
    assert adjusted < 0.1
    assert any("no_yield_penalty" in reason for reason in reasons)
    assert any("recent_incomplete_cycle_penalty" in reason for reason in reasons)
    assert any("recent_phase_gate_penalty" in reason for reason in reasons)


def test_portfolio_scoring_penalizes_reconciled_fail_closed_cycles(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:reconciled",
            state="reject",
            subsystem="retrieval",
            action="finalize_cycle",
            artifact_path="retrieval.json",
            artifact_kind="retrieval_policy_set",
            reason="reconciled stale cycle",
            metrics_summary={
                "incomplete_cycle": True,
                "baseline_pass_rate": 0.8,
                "candidate_pass_rate": 0.8,
            },
        ),
    )

    summary = planner.recent_subsystem_activity_summary(subsystem="retrieval", output_path=cycles_path)
    adjusted, reasons = planner._portfolio_adjusted_experiment_score(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.05,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        recent_activity=summary,
    )

    assert summary["recent_reconciled_failure_cycles"] == 1
    assert adjusted < 0.1
    assert any("recent_reconciled_failure_penalty" in reason for reason in reasons)


def test_portfolio_scoring_penalizes_repeated_phase_gate_failure_reason(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    repeated_reason = "retrieval candidate showed no retrieval selection or skill ranking during autonomous evaluation"
    for index in (1, 2):
        cycle_id = f"cycle:retrieval:{index}"
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="retrieval.json",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem="retrieval",
                action="finalize_cycle",
                artifact_path="retrieval.json",
                artifact_kind="retrieval_policy_set",
                reason="retrieval candidate did not satisfy the retained retrieval gate",
                metrics_summary={
                    "phase_gate_passed": False,
                    "phase_gate_failure_count": 1,
                    "phase_gate_failures": [repeated_reason],
                    "baseline_pass_rate": 0.6,
                    "candidate_pass_rate": 0.6,
                },
            ),
        )

    summary = planner.recent_subsystem_activity_summary(subsystem="retrieval", output_path=cycles_path)
    adjusted, reasons = planner._portfolio_adjusted_experiment_score(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.05,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        recent_activity=summary,
    )

    assert summary["last_phase_gate_failure_reason"] == repeated_reason
    assert summary["repeated_phase_gate_reason_count"] == 2
    assert adjusted < 0.1
    assert any("repeated_phase_gate_reason_penalty" in reason for reason in reasons)


def test_portfolio_scoring_penalizes_recent_promotion_failures(tmp_path: Path):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    for index, candidate_pass_rate in ((1, 0.55), (2, 0.5)):
        cycle_id = f"cycle:retrieval:{index}"
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="select",
                subsystem="retrieval",
                action="choose_target",
                artifact_path="retrieval.json",
                artifact_kind="improvement_target",
                reason="retrieval gap",
                metrics_summary={"selected_variant": {"variant_id": "confidence_gating"}},
            ),
        )
        planner.append_cycle_record(
            cycles_path,
            ImprovementCycleRecord(
                cycle_id=cycle_id,
                state="reject",
                subsystem="retrieval",
                action="finalize_cycle",
                artifact_path="retrieval.json",
                artifact_kind="retrieval_policy_set",
                reason="retrieval candidate regressed against baseline",
                metrics_summary={
                    "baseline_pass_rate": 0.7,
                    "candidate_pass_rate": candidate_pass_rate,
                    "phase_gate_passed": False,
                    "phase_gate_failure_count": 1,
                },
            ),
        )

    summary = planner.recent_subsystem_activity_summary(subsystem="retrieval", output_path=cycles_path)
    adjusted, reasons = planner._portfolio_adjusted_experiment_score(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.05,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        recent_activity=summary,
    )

    assert adjusted < 0.1
    assert any("recent_promotion_failure_penalty=0.1000" in reason for reason in reasons)


def test_rank_variants_penalizes_recent_no_yield_variant(tmp_path: Path, monkeypatch):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"selected_variant": {"variant_id": "sticky_variant"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="evaluate",
            subsystem="retrieval",
            action="compare_candidate_to_baseline",
            artifact_path="retrieval.json",
            artifact_kind="retention_evaluation",
            reason="preview",
            metrics_summary={
                "phase_gate_passed": False,
                "phase_gate_failure_count": 1,
            },
        ),
    )

    monkeypatch.setattr(
        planner,
        "_variants_for_experiment",
        lambda experiment, metrics, planner_controls=None: [
            ImprovementVariant(
                subsystem="retrieval",
                variant_id="sticky_variant",
                description="historically over-selected retrieval knob",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.01,
                controls={},
            ),
            ImprovementVariant(
                subsystem="retrieval",
                variant_id="fresh_variant",
                description="fresh retrieval knob",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.01,
                controls={},
            ),
        ],
    )

    ranked = planner.rank_variants(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.02,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        EvalMetrics(total=10, passed=8),
    )

    assert ranked[0].variant_id == "fresh_variant"
    assert ranked[1].controls["history"]["recent_variant"]["no_yield_cycles"] == 1


def test_rank_variants_clamps_stalled_variant_scores_non_negative(tmp_path: Path, monkeypatch):
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes", cycles_path=cycles_path)
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="select",
            subsystem="retrieval",
            action="choose_target",
            artifact_path="retrieval.json",
            artifact_kind="improvement_target",
            reason="retrieval gap",
            metrics_summary={"selected_variant": {"variant_id": "stalled_variant"}},
        ),
    )
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:retrieval:1",
            state="evaluate",
            subsystem="retrieval",
            action="compare_candidate_to_baseline",
            artifact_path="retrieval.json",
            artifact_kind="retention_evaluation",
            reason="preview",
            metrics_summary={
                "phase_gate_passed": False,
                "phase_gate_failure_count": 1,
            },
        ),
    )

    monkeypatch.setattr(
        planner,
        "_variants_for_experiment",
        lambda experiment, metrics, planner_controls=None: [
            ImprovementVariant(
                subsystem="retrieval",
                variant_id="stalled_variant",
                description="variant with enough bad history to go negative without clamping",
                expected_gain=0.0,
                estimated_cost=2,
                score=0.0,
                controls={},
            ),
            ImprovementVariant(
                subsystem="retrieval",
                variant_id="fresh_variant",
                description="fresh retrieval knob",
                expected_gain=0.02,
                estimated_cost=2,
                score=0.01,
                controls={},
            ),
        ],
    )

    ranked = planner.rank_variants(
        ImprovementExperiment(
            subsystem="retrieval",
            reason="retrieval gap",
            priority=3,
            expected_gain=0.02,
            estimated_cost=2,
            score=0.1,
            evidence={},
        ),
        EvalMetrics(total=10, passed=8),
    )

    stalled = next(variant for variant in ranked if variant.variant_id == "stalled_variant")

    assert stalled.score == 0.0
