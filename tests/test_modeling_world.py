from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from agent_kernel.config import KernelConfig
from agent_kernel.memory import EpisodeMemory
from agent_kernel.schemas import EpisodeRecord, StepRecord
from agent_kernel.schemas import TaskSpec
from agent_kernel.world_model import WorldModel

try:
    import torch
except Exception:  # pragma: no cover - reduced environments
    torch = None  # type: ignore[assignment]

_TORCH_RUNTIME_READY = bool(
    torch is not None
    and hasattr(torch, "tensor")
    and hasattr(torch, "log")
    and hasattr(torch, "allclose")
)
_TORCH_CUDA_READY = bool(
    _TORCH_RUNTIME_READY
    and hasattr(torch, "cuda")
    and callable(getattr(torch.cuda, "is_available", None))
    and torch.cuda.is_available()
)

from agent_kernel.modeling.world import (
    CausalWorldProfile,
    MODELING_COUNTERFACTUAL_GROUPS,
    build_causal_state_signature,
    causal_belief_scan,
    causal_belief_scan_ref,
    condition_causal_world_prior,
    load_causal_world_profile,
    parse_modeling_counterfactual_groups,
    summarize_causal_machine_adoption,
)
from agent_kernel.modeling.world.kernels import (
    causal_belief_scan_cuda_metadata,
    cuda_belief_scan_status,
)
from agent_kernel.modeling.world.kernels.build import _build_lock

cuda_belief_scan_module = __import__(
    "agent_kernel.modeling.world.kernels.cuda_belief_scan",
    fromlist=["_load_extension"],
)


def test_parse_modeling_counterfactual_groups_filters_and_deduplicates():
    groups = parse_modeling_counterfactual_groups("state,retrieval,state,unknown,risk")

    assert groups == ["state", "retrieval", "risk"]
    assert "policy" in MODELING_COUNTERFACTUAL_GROUPS


def test_load_causal_world_profile_reads_profile_and_sidecar(tmp_path: Path):
    sidecar_path = tmp_path / "profile_spectral_eigenbases.npz"
    centroids = np.arange(24, dtype=np.float32).reshape(3, 8)
    log_probs = np.zeros((3, 3), dtype=np.float32)
    state_masses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.savez(
        sidecar_path,
        causal_machine_signature_centroids=centroids,
        causal_machine_log_probs=log_probs,
        causal_machine_state_masses=state_masses,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        """
        {
          "future_signature_profile": {
            "available": true,
            "horizons": [1, 2],
            "signature_dim": 8
          },
          "spectral_eigenbases": {}
        }
        """.strip(),
        encoding="utf-8",
    )

    profile = load_causal_world_profile(profile_path)

    assert profile.num_states == 3
    assert profile.horizons == (1, 2)
    assert profile.sketch_dim == 3
    assert profile.centroids_sketch.shape == (3, 6)
    assert profile.state_masses.tolist() == [1.0, 2.0, 3.0]


def test_summarize_causal_machine_adoption_points_to_owned_modules():
    summary = summarize_causal_machine_adoption()

    assert "profile_loaded_state_priors" in summary["recommended_capabilities"]
    assert "profile_conditioned_state_priors" in summary["recommended_capabilities"]
    assert "agent_kernel/modeling/world/kernels/" in summary["recommended_modules"]


def test_condition_causal_world_prior_uses_signature_alignment():
    sketch, token_count = build_causal_state_signature(
        ["repair missing report.txt", "report.txt status cleanup"],
        sketch_dim=4,
    )
    profile = CausalWorldProfile(
        profile_path=Path("/tmp/profile.json"),
        sidecar_path=Path("/tmp/profile_spectral_eigenbases.npz"),
        num_states=2,
        horizons=(1,),
        sketch_dim=4,
        log_probs=np.zeros((2,), dtype=np.float32),
        state_masses=np.array([1.0, 1.0], dtype=np.float32),
        centroids_sketch=np.stack([sketch, -sketch]).astype(np.float32),
    )

    prior = condition_causal_world_prior(
        profile,
        text_fragments=["repair missing report.txt", "report.txt status cleanup"],
    )

    assert token_count > 0
    assert prior.backend == "profile_conditioned"
    assert prior.matched_state_index == 0
    assert prior.matched_state_probability > 0.5
    assert np.isclose(np.exp(prior.log_prior).sum(), 1.0)


def test_condition_causal_world_prior_strengthens_long_horizon_bias():
    sketch, _ = build_causal_state_signature(
        ["repair missing report.txt", "report.txt status cleanup"],
        sketch_dim=4,
    )
    repeated = np.concatenate([sketch, sketch]).astype(np.float32)
    profile = CausalWorldProfile(
        profile_path=Path("/tmp/profile.json"),
        sidecar_path=Path("/tmp/profile_spectral_eigenbases.npz"),
        num_states=2,
        horizons=(1, 4),
        sketch_dim=4,
        log_probs=np.zeros((2,), dtype=np.float32),
        state_masses=np.array([1.0, 1.0], dtype=np.float32),
        centroids_sketch=np.stack([repeated, -repeated]).astype(np.float32),
    )

    base_prior = condition_causal_world_prior(
        profile,
        text_fragments=["repair missing report.txt", "report.txt status cleanup"],
    )
    long_horizon_prior = condition_causal_world_prior(
        profile,
        text_fragments=["repair missing report.txt", "report.txt status cleanup", "horizon:long_horizon"],
        horizon_hint="long_horizon",
    )

    assert long_horizon_prior.horizon_hint == "long_horizon"
    assert long_horizon_prior.applied_bias_strength > base_prior.applied_bias_strength
    assert long_horizon_prior.matched_state_probability > base_prior.matched_state_probability


def test_world_model_prioritized_long_horizon_hotspots_surface_pending_workflow_paths_early():
    model = WorldModel()
    task = TaskSpec(
        task_id="long_horizon_hotspot_task",
        prompt="Repair the release workflow and publish the missing report.",
        workspace_subdir="long_horizon_hotspot_task",
        metadata={
            "difficulty": "long_horizon",
            "semantic_verifier": {
                "expected_changed_paths": ["src/release_state.txt"],
                "generated_paths": ["generated/release.patch"],
                "report_rules": [{"path": "reports/release_review.txt"}],
            },
        },
    )
    summary = model.summarize(task)
    summary.update(
        {
            "updated_workflow_paths": [],
            "updated_generated_paths": [],
            "updated_report_paths": [],
            "missing_expected_artifacts": [],
            "unsatisfied_expected_contents": [],
            "present_forbidden_artifacts": [],
            "changed_preserved_artifacts": [],
            "missing_preserved_artifacts": [],
        }
    )

    hotspots = model.prioritized_long_horizon_hotspots(
        task,
        summary,
        latest_transition={"no_progress": True, "regressions": []},
        latent_state_summary={
            "active_paths": [],
            "learned_world_state": {
                "progress_signal": 0.12,
                "risk_signal": 0.41,
            },
        },
    )

    assert [entry["subgoal"] for entry in hotspots[:3]] == [
        "update workflow path src/release_state.txt",
        "regenerate generated artifact generated/release.patch",
        "write workflow report reports/release_review.txt",
    ]


def test_world_model_prioritized_long_horizon_hotspots_prioritize_regressed_preserved_paths():
    model = WorldModel()
    task = TaskSpec(
        task_id="long_horizon_preserved_hotspot_task",
        prompt="Repair the task without regressing preserved artifacts.",
        workspace_subdir="long_horizon_preserved_hotspot_task",
        expected_files=["status.txt"],
        metadata={"difficulty": "long_horizon"},
    )
    summary = model.summarize(task)
    summary.update(
        {
            "missing_expected_artifacts": ["status.txt"],
            "unsatisfied_expected_contents": [],
            "present_forbidden_artifacts": [],
            "changed_preserved_artifacts": ["docs/context.md"],
            "missing_preserved_artifacts": [],
            "workflow_preserved_paths": ["docs/context.md"],
        }
    )

    hotspots = model.prioritized_long_horizon_hotspots(
        task,
        summary,
        latest_transition={"no_progress": True, "regressions": ["docs/context.md"]},
        latent_state_summary={
            "active_paths": [],
            "learned_world_state": {
                "progress_signal": 0.1,
                "risk_signal": 0.43,
            },
        },
    )

    assert hotspots[0]["subgoal"] == "preserve required artifact docs/context.md"
    assert "state_regression" in hotspots[0]["signals"]


def test_world_model_uses_empirical_transition_priors_from_episode_traces(tmp_path: Path):
    config = KernelConfig(provider="mock", trajectories_root=tmp_path / "episodes")
    memory = EpisodeMemory(config.trajectories_root, config=config)
    memory.save(
        EpisodeRecord(
            task_id="release_repair_trace",
            prompt="Repair the release report.",
            workspace=str(tmp_path / "workspace" / "release_repair_trace"),
            success=True,
            task_metadata={"benchmark_family": "repository"},
            task_contract={
                "prompt": "Repair the release report.",
                "workspace_subdir": "release_repair_trace",
                "setup_commands": [],
                "success_command": "pytest -q tests/test_release.py",
                "suggested_commands": ["printf 'ready\\n' > reports/release_review.txt"],
                "expected_files": ["reports/release_review.txt"],
                "expected_output_substrings": [],
                "forbidden_files": [],
                "forbidden_output_substrings": [],
                "expected_file_contents": {"reports/release_review.txt": "ready\n"},
                "max_steps": 3,
                "metadata": {"benchmark_family": "repository"},
            },
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="repair the report",
                    action="code_execute",
                    content="printf 'ready\\n' > reports/release_review.txt",
                    selected_skill_id=None,
                    command_result={
                        "command": "printf 'ready\\n' > reports/release_review.txt",
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "timed_out": False,
                    },
                    verification={"passed": True, "reasons": ["verification passed"]},
                    state_progress_delta=0.6,
                    state_transition={
                        "progress_delta": 0.6,
                        "changed_paths": ["reports/release_review.txt"],
                        "newly_materialized_expected_artifacts": ["reports/release_review.txt"],
                        "newly_satisfied_expected_contents": ["reports/release_review.txt"],
                        "verifier_delta": 2,
                    },
                )
            ],
            world_model_summary={"completion_ratio": 1.0, "horizon": "bounded"},
        )
    )
    model = WorldModel(config=config)
    summary = {
        "benchmark_family": "repository",
        "horizon": "bounded",
        "expected_artifacts": ["reports/release_review.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["reports/release_review.txt"],
        "present_forbidden_artifacts": [],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": ["reports/release_review.txt"],
    }

    favored = "printf 'ready\\n' > reports/release_review.txt"
    unrelated = "printf 'todo\\n' > notes/todo.txt"

    favored_score = model.score_command(summary, favored)
    unrelated_score = model.score_command(summary, unrelated)
    favored_effect = model.simulate_command_effect(summary, favored)

    assert favored_score > unrelated_score
    assert favored_effect["empirical_prior"]["sample_count"] >= 1
    assert favored_effect["predicted_verifier_delta"] >= 1.0
    assert "reports/release_review.txt" in favored_effect["predicted_changed_paths"]


def test_world_model_uses_state_conditioned_model_with_bootstrap_prior_fallback(tmp_path: Path):
    config = KernelConfig(provider="mock", trajectories_root=tmp_path / "episodes")
    memory = EpisodeMemory(config.trajectories_root, config=config)
    memory.save(
        EpisodeRecord(
            task_id="release_trace",
            prompt="Repair the release report.",
            workspace=str(tmp_path / "workspace" / "release_trace"),
            success=True,
            task_metadata={
                "benchmark_family": "repository",
                "semantic_verifier": {
                    "report_rules": [{"path": "reports/release_review.txt"}],
                    "test_commands": [{"label": "release check", "argv": ["pytest", "-q", "tests/test_release.py"]}],
                },
            },
            task_contract={
                "prompt": "Repair the release report.",
                "workspace_subdir": "release_trace",
                "setup_commands": [],
                "success_command": "pytest -q tests/test_release.py",
                "suggested_commands": ["printf 'ready\\n' > reports/release_review.txt"],
                "expected_files": ["reports/release_review.txt"],
                "expected_output_substrings": [],
                "forbidden_files": [],
                "forbidden_output_substrings": [],
                "expected_file_contents": {"reports/release_review.txt": "ready\n"},
                "max_steps": 3,
                "metadata": {
                    "benchmark_family": "repository",
                    "semantic_verifier": {
                        "report_rules": [{"path": "reports/release_review.txt"}],
                        "test_commands": [{"label": "release check", "argv": ["pytest", "-q", "tests/test_release.py"]}],
                    },
                },
            },
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="repair release report",
                    action="code_execute",
                    content="printf 'ready\\n' > reports/release_review.txt",
                    selected_skill_id=None,
                    command_result={
                        "command": "printf 'ready\\n' > reports/release_review.txt",
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "timed_out": False,
                    },
                    verification={"passed": True, "reasons": ["verification passed"]},
                    state_progress_delta=0.6,
                    state_transition={
                        "progress_delta": 0.6,
                        "changed_paths": ["reports/release_review.txt"],
                        "newly_materialized_expected_artifacts": ["reports/release_review.txt"],
                        "newly_satisfied_expected_contents": ["reports/release_review.txt"],
                        "verifier_delta": 2,
                    },
                )
            ],
            world_model_summary={"completion_ratio": 1.0, "horizon": "bounded"},
        )
    )
    memory.save(
        EpisodeRecord(
            task_id="notes_trace",
            prompt="Write the todo note.",
            workspace=str(tmp_path / "workspace" / "notes_trace"),
            success=False,
            task_metadata={"benchmark_family": "repository"},
            task_contract={
                "prompt": "Write the todo note.",
                "workspace_subdir": "notes_trace",
                "setup_commands": [],
                "success_command": "test -f notes/todo.txt",
                "suggested_commands": ["printf 'todo\\n' > notes/todo.txt"],
                "expected_files": ["notes/todo.txt"],
                "expected_output_substrings": [],
                "forbidden_files": [],
                "forbidden_output_substrings": [],
                "expected_file_contents": {"notes/todo.txt": "todo\n"},
                "max_steps": 3,
                "metadata": {"benchmark_family": "repository"},
            },
            termination_reason="failure",
            steps=[
                StepRecord(
                    index=1,
                    thought="write todo note",
                    action="code_execute",
                    content="printf 'todo\\n' > notes/todo.txt",
                    selected_skill_id=None,
                    command_result={
                        "command": "printf 'todo\\n' > notes/todo.txt",
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "write failed",
                        "timed_out": False,
                    },
                    verification={"passed": False, "reasons": ["exit code was 1"]},
                    state_progress_delta=0.0,
                    state_transition={
                        "progress_delta": 0.0,
                        "changed_paths": ["notes/todo.txt"],
                        "regressions": ["notes/todo.txt"],
                        "verifier_delta": -1,
                    },
                )
            ],
            world_model_summary={"completion_ratio": 0.0, "horizon": "bounded"},
        )
    )
    model = WorldModel(config=config)
    summary = {
        "benchmark_family": "repository",
        "horizon": "bounded",
        "expected_artifacts": ["reports/release_review.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["reports/release_review.txt"],
        "present_forbidden_artifacts": [],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_required_tests": ["release check"],
        "workflow_required_merges": [],
    }

    effect = model.simulate_command_effect(summary, "printf 'ready\\n' > reports/release_review.txt")

    assert effect["empirical_prior"]["prediction_source"] == "learned_state_model"
    assert effect["empirical_prior"]["learned_state_model"]["sample_count"] >= 1
    assert effect["empirical_prior"]["bootstrap_prior"]["sample_count"] >= 1
    assert effect["predicted_verifier_delta"] >= 1.0


def test_world_model_uses_semantic_memory_prior_for_recalled_repair_paths(tmp_path: Path):
    config = KernelConfig(provider="mock", trajectories_root=tmp_path / "episodes")
    model = WorldModel(config=config)
    summary = {
        "benchmark_family": "repository",
        "horizon": "bounded",
        "expected_artifacts": ["reports/release_review.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["reports/release_review.txt"],
        "present_forbidden_artifacts": [],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": ["reports/release_review.txt"],
        "semantic_episodes": [
            {
                "task_id": "release_repair_memory",
                "benchmark_family": "repository",
                "memory_source": "episode",
                "success": True,
                "verifier_obligations": ["write workflow report reports/release_review.txt and mention ready"],
                "changed_paths": ["reports/release_review.txt"],
                "failure_signals": ["state_regression"],
                "edit_patches": [
                    {
                        "path": "reports/release_review.txt",
                        "status": "modified",
                        "patch_summary": "modified reports/release_review.txt (+1 -1)",
                        "patch_excerpt": "-BROKEN\n+READY",
                    }
                ],
                "recovery_trace": {
                    "failed_command": "printf 'broken\\n' > reports/release_review.txt",
                    "recovery_command": "printf 'ready\\n' > reports/release_review.txt",
                    "failure_signals": ["state_regression"],
                },
            }
        ],
    }

    favored = "printf 'ready\\n' > reports/release_review.txt"
    repeated_failure = "printf 'broken\\n' > reports/release_review.txt"

    favored_score = model.score_command(summary, favored)
    repeated_failure_score = model.score_command(summary, repeated_failure)
    favored_effect = model.simulate_command_effect(summary, favored)
    repeated_failure_effect = model.simulate_command_effect(summary, repeated_failure)

    assert favored_score > repeated_failure_score
    assert favored_effect["empirical_prior"]["prediction_source"] == "semantic_memory"
    assert favored_effect["predicted_progress_gain"] >= 1.0
    assert favored_effect["predicted_verifier_delta"] >= 1.0
    assert "reports/release_review.txt" in favored_effect["predicted_changed_paths"]
    assert repeated_failure_effect["empirical_prior"]["regression_rate"] >= 0.5


def test_world_model_uses_semantic_prototype_sequence_alignment_for_rollout(tmp_path: Path):
    del tmp_path
    model = WorldModel(config=KernelConfig(provider="mock"))
    summary = {
        "benchmark_family": "repository",
        "horizon": "bounded",
        "expected_artifacts": ["reports/release_review.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["reports/release_review.txt"],
        "present_forbidden_artifacts": [],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": ["reports/release_review.txt"],
        "semantic_prototypes": [
            {
                "benchmark_family": "repository",
                "changed_paths": ["reports/release_review.txt"],
                "verifier_obligations": ["write workflow report reports/release_review.txt"],
                "failure_signals": ["state_regression"],
                "episode_count": 2,
                "success_count": 2,
                "application_commands": ["mkdir -p reports", "printf 'ready\\n' > reports/release_review.txt"],
                "command_sequences": {
                    "mkdir -p reports || printf 'ready\\n' > reports/release_review.txt": 2
                },
                "failed_commands": {},
            }
        ],
    }

    favored = model.simulate_command_sequence_effect(
        summary,
        ["mkdir -p reports", "printf 'ready\\n' > reports/release_review.txt"],
    )
    partial = model.simulate_command_sequence_effect(summary, ["mkdir -p reports"])

    assert favored["sequence_alignment_bonus"] > 0.0
    assert favored["score"] > partial["score"]
    assert favored["predicted_progress_gain"] >= partial["predicted_progress_gain"]


def test_world_model_describe_transition_emits_preview_edit_patches():
    model = WorldModel()

    transition = model.describe_transition(
        {
            "completion_ratio": 0.0,
            "existing_expected_artifacts": [],
            "satisfied_expected_contents": [],
            "present_forbidden_artifacts": [],
            "changed_preserved_artifacts": [],
            "updated_workflow_paths": [],
            "updated_generated_paths": [],
            "updated_report_paths": [],
            "workspace_file_previews": {
                "reports/release_review.txt": {"content": "BROKEN\n"},
            },
        },
        {
            "completion_ratio": 0.5,
            "existing_expected_artifacts": ["reports/release_review.txt"],
            "satisfied_expected_contents": ["reports/release_review.txt"],
            "present_forbidden_artifacts": [],
            "changed_preserved_artifacts": [],
            "updated_workflow_paths": [],
            "updated_generated_paths": [],
            "updated_report_paths": ["reports/release_review.txt"],
            "workspace_file_previews": {
                "reports/release_review.txt": {"content": "READY\n"},
            },
        },
    )

    assert transition["edit_patches"][0]["path"] == "reports/release_review.txt"
    assert "-BROKEN" in transition["edit_patches"][0]["patch"]
    assert "+READY" in transition["edit_patches"][0]["patch"]


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_matches_manual_update():
    local_logits = torch.tensor([[[0.4, -0.2], [0.1, 0.0]]], dtype=torch.float32)
    transition_log_probs = torch.log(
        torch.tensor(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            dtype=torch.float32,
        )
    )
    transition_context = torch.tensor([[[0.0, 0.1], [0.2, -0.1]]], dtype=torch.float32)
    initial_log_belief = torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32))

    result = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.5,
        chunk_size=1,
    )

    prev = initial_log_belief[0]
    expected_steps = []
    for pos in range(local_logits.shape[1]):
        pred = torch.logsumexp(prev.unsqueeze(-1) + transition_log_probs, dim=0)
        obs = local_logits[0, pos] + 0.5 * (pred + transition_context[0, pos])
        q = obs - torch.logsumexp(obs, dim=0)
        expected_steps.append(q)
        prev = q
    expected_beliefs = torch.stack(expected_steps, dim=0).unsqueeze(0)

    assert torch.allclose(result.beliefs, expected_beliefs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.final_log_belief, expected_beliefs[:, -1, :], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_supports_diagonal_transition_fast_path():
    local_logits = torch.tensor([[[0.4, -0.2], [0.1, 0.0]]], dtype=torch.float32)
    transition_log_probs = torch.tensor([0.3, -0.1], dtype=torch.float32)
    transition_context = torch.tensor([[[0.0, 0.1], [0.2, -0.1]]], dtype=torch.float32)
    initial_log_belief = torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32))

    result = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.5,
        chunk_size=1,
    )

    prev = initial_log_belief[0]
    expected_steps = []
    for pos in range(local_logits.shape[1]):
        pred = prev + transition_log_probs
        obs = local_logits[0, pos] + 0.5 * (pred + transition_context[0, pos])
        q = obs - torch.logsumexp(obs, dim=0)
        expected_steps.append(q)
        prev = q
    expected_beliefs = torch.stack(expected_steps, dim=0).unsqueeze(0)

    assert torch.allclose(result.beliefs, expected_beliefs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.final_log_belief, expected_beliefs[:, -1, :], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_supports_masked_banded_transition_matrix():
    local_logits = torch.tensor([[[0.3, -0.1, 0.0], [0.2, 0.1, -0.3]]], dtype=torch.float32)
    transition_logits = torch.tensor(
        [
            [1.0, 0.5, float("-inf")],
            [0.4, 1.2, 0.2],
            [float("-inf"), 0.6, 0.9],
        ],
        dtype=torch.float32,
    )
    transition_log_probs = torch.log_softmax(transition_logits, dim=-1)
    transition_context = torch.tensor(
        [[[0.0, 0.1, -0.2], [0.1, -0.1, 0.0]]],
        dtype=torch.float32,
    )
    initial_log_belief = torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32))

    result = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.6,
        chunk_size=1,
    )

    assert torch.isfinite(result.beliefs).all()
    assert torch.isfinite(result.final_log_belief).all()
    assert torch.equal(torch.isneginf(transition_log_probs), torch.tensor(
        [[False, False, True], [False, False, False], [True, False, False]]
    ))


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_supports_structured_transition_factors_without_dense_child_matrix():
    local_logits = torch.tensor([[[0.3, -0.1, 0.0], [0.2, 0.1, -0.3]]], dtype=torch.float32)
    base_transition_logits = torch.tensor(
        [
            [0.8, 0.1, -0.4],
            [0.2, 0.7, 0.0],
            [-0.3, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    source_logits = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    dest_logits = torch.tensor(
        [
            [0.2, -0.1, 0.3],
            [0.4, 0.1, -0.2],
        ],
        dtype=torch.float32,
    )
    stay_logits = torch.tensor([0.05, -0.02, 0.08], dtype=torch.float32)
    dense_transition_logits = base_transition_logits + source_logits @ dest_logits + torch.diag(stay_logits)
    transition_log_probs = torch.log_softmax(dense_transition_logits, dim=-1)
    transition_context = torch.tensor(
        [[[0.0, 0.1, -0.2], [0.1, -0.1, 0.0]]],
        dtype=torch.float32,
    )
    initial_log_belief = torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32))

    dense_result = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.6,
        chunk_size=1,
    )
    structured_result = causal_belief_scan_ref(
        local_logits,
        base_transition_logits,
        transition_context,
        initial_log_belief,
        transition_structure={
            "base_transition_logits": base_transition_logits,
            "source_logits": source_logits,
            "dest_logits": dest_logits,
            "stay_logits": stay_logits,
            "family": "dense",
            "bandwidth": 0,
        },
        transition_gate=0.6,
        chunk_size=1,
    )

    assert torch.allclose(structured_result.beliefs, dense_result.beliefs, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(structured_result.final_log_belief, dense_result.final_log_belief, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_prefers_python_reference_when_requested():
    local_logits = torch.randn(2, 3, 4)
    transition_log_probs = torch.randn(4, 4)
    transition_context = torch.randn(2, 3, 4)
    initial_log_belief = torch.randn(2, 4)

    result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        prefer_python_ref=True,
    )

    assert result.backend == "python_ref"
    assert result.beliefs.shape == (2, 3, 4)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_is_chunk_invariant_on_longer_sequence():
    torch.manual_seed(0)
    local_logits = torch.randn(2, 64, 16)
    transition_log_probs = torch.randn(16, 16)
    transition_context = torch.randn(2, 64, 16)
    initial_log_belief = torch.randn(2, 16)

    ref_chunk_1 = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=1,
    )
    ref_chunk_8 = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=8,
    )
    ref_chunk_full = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=64,
    )

    assert torch.allclose(ref_chunk_1.beliefs, ref_chunk_8.beliefs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref_chunk_1.beliefs, ref_chunk_full.beliefs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref_chunk_1.final_log_belief, ref_chunk_8.final_log_belief, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref_chunk_1.final_log_belief, ref_chunk_full.final_log_belief, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_accepts_initial_belief_for_incremental_equivalence():
    torch.manual_seed(0)
    local_logits = torch.randn(2, 6, 5)
    transition_log_probs = torch.randn(5, 5)
    transition_context = torch.randn(2, 6, 5)
    initial_log_belief = torch.log_softmax(torch.randn(2, 5), dim=-1)

    full = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.6,
        chunk_size=3,
    )
    first = causal_belief_scan_ref(
        local_logits[:, :3, :],
        transition_log_probs,
        transition_context[:, :3, :],
        initial_log_belief,
        transition_gate=0.6,
        chunk_size=3,
    )
    second = causal_belief_scan_ref(
        local_logits[:, 3:, :],
        transition_log_probs,
        transition_context[:, 3:, :],
        first.final_log_belief,
        transition_gate=0.6,
        chunk_size=3,
    )

    stitched = torch.cat([first.beliefs, second.beliefs], dim=1)
    assert torch.allclose(stitched, full.beliefs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(second.final_log_belief, full.final_log_belief, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _TORCH_RUNTIME_READY, reason="full torch runtime is required")
def test_causal_belief_scan_ref_stability_smoke_large_rollout():
    torch.manual_seed(0)
    local_logits = torch.randn(2, 256, 32)
    transition_log_probs = torch.randn(32, 32)
    transition_context = torch.randn(2, 256, 32)
    initial_log_belief = torch.randn(2, 32)

    result = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.4,
        chunk_size=32,
    )

    assert torch.isfinite(result.beliefs).all()
    assert torch.isfinite(result.final_log_belief).all()


def test_world_kernel_build_lock_serializes_access(tmp_path: Path):
    lock_path = tmp_path / "world-kernel-build.lock"
    ready_path = tmp_path / "child_ready.txt"
    release_path = tmp_path / "release.txt"
    child_path = tmp_path / "child_acquired.txt"

    script = f"""
from pathlib import Path
import time
from agent_kernel.modeling.world.kernels.build import _build_lock

lock_path = Path(r"{lock_path}")
ready_path = Path(r"{ready_path}")
release_path = Path(r"{release_path}")
child_path = Path(r"{child_path}")
ready_path.write_text("ready", encoding="utf-8")
with _build_lock(lock_path):
    child_path.write_text("acquired", encoding="utf-8")
"""

    with _build_lock(lock_path):
        child = subprocess.Popen([sys.executable, "-c", script])
        deadline = time.time() + 5.0
        while time.time() < deadline and not ready_path.exists():
            time.sleep(0.05)
        assert ready_path.exists()
        time.sleep(0.3)
        assert not child_path.exists()
    child.wait(timeout=5.0)
    assert child.returncode == 0
    assert child_path.exists()


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_causal_belief_scan_cuda_missing_extension_fails_fast(monkeypatch: pytest.MonkeyPatch):
    local_logits = torch.randn(1, 3, 4, device="cuda")
    transition_log_probs = torch.randn(4, 4, device="cuda")
    transition_context = torch.randn(1, 3, 4, device="cuda")
    initial_log_belief = torch.randn(1, 4, device="cuda")

    monkeypatch.setattr(
        "agent_kernel.modeling.world.belief_scan.cuda_belief_scan_status",
        lambda: type("Status", (), {"available": False, "detail": "missing native extension"})(),
    )

    with pytest.raises(RuntimeError, match="missing native extension"):
        causal_belief_scan(
            local_logits,
            transition_log_probs,
            transition_context,
            initial_log_belief,
        )


def test_cuda_belief_scan_status_reports_expected_fields():
    status = cuda_belief_scan_status()

    assert status.source == "native"
    assert isinstance(status.available, bool)
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.compiled_extension, bool)


def test_cuda_belief_scan_status_surfaces_import_failure_detail(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cuda_belief_scan_module, "_LOADED_EXTENSION", None)
    monkeypatch.setattr(cuda_belief_scan_module, "_LOAD_ERROR_DETAIL", "synthetic import failure")
    monkeypatch.setattr(
        cuda_belief_scan_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("synthetic import failure")),
    )
    monkeypatch.setattr(cuda_belief_scan_module, "_autobuild_allowed", lambda: (False, ""))

    status = cuda_belief_scan_status()

    assert status.available is False
    assert "synthetic import failure" in status.detail


def test_cuda_belief_scan_metadata_has_expected_extension_name():
    metadata = causal_belief_scan_cuda_metadata()

    assert metadata["compiled_extension_name"] == "agent_kernel_causal_belief_scan_cuda"
    assert "agent_kernel/modeling/world/kernels/src/causal_belief_scan.cpp" in metadata["compiled_extension_sources"]


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_matches_python_reference():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    batch_size, seq_len, num_states = 2, 5, 8
    local_logits = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    transition_log_probs = torch.randn(num_states, num_states, device="cuda", requires_grad=True)
    transition_context = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    initial_log_belief = torch.randn(batch_size, num_states, device="cuda", requires_grad=True)

    cuda_result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=2,
    )
    beliefs_cuda, final_cuda = cuda_result.beliefs, cuda_result.final_log_belief
    ref = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=2,
    )

    assert torch.allclose(beliefs_cuda, ref.beliefs, atol=1e-4, rtol=1e-4)
    assert torch.allclose(final_cuda, ref.final_log_belief, atol=1e-4, rtol=1e-4)

    loss_cuda = beliefs_cuda.sum() + final_cuda.sum()
    loss_ref = ref.beliefs.sum() + ref.final_log_belief.sum()
    loss_cuda.backward()
    grad_cuda = {
        "local_logits": local_logits.grad.detach().clone(),
        "transition_log_probs": transition_log_probs.grad.detach().clone(),
        "transition_context": transition_context.grad.detach().clone(),
        "initial_log_belief": initial_log_belief.grad.detach().clone(),
    }
    for tensor in (local_logits, transition_log_probs, transition_context, initial_log_belief):
        tensor.grad.zero_()
    loss_ref.backward()
    grad_ref = {
        "local_logits": local_logits.grad.detach().clone(),
        "transition_log_probs": transition_log_probs.grad.detach().clone(),
        "transition_context": transition_context.grad.detach().clone(),
        "initial_log_belief": initial_log_belief.grad.detach().clone(),
    }

    for name in grad_cuda:
        assert torch.allclose(grad_cuda[name], grad_ref[name], atol=5e-3, rtol=1e-3), name


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_diagonal_transition_matches_python_reference():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    batch_size, seq_len, num_states = 2, 5, 8
    local_logits = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    transition_log_probs = torch.randn(num_states, device="cuda", requires_grad=True)
    transition_context = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    initial_log_belief = torch.randn(batch_size, num_states, device="cuda", requires_grad=True)

    cuda_result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=2,
    )
    ref = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=2,
    )

    assert torch.allclose(cuda_result.beliefs, ref.beliefs, atol=1e-4, rtol=1e-4)
    assert torch.allclose(cuda_result.final_log_belief, ref.final_log_belief, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_masked_banded_transition_matches_python_reference():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    batch_size, seq_len, num_states = 2, 6, 8
    local_logits = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    transition_logits = torch.randn(num_states, num_states, device="cuda")
    mask = (torch.arange(num_states, device="cuda")[:, None] - torch.arange(num_states, device="cuda")[None, :]).abs() <= 1
    transition_log_probs = torch.log_softmax(transition_logits.masked_fill(~mask, float("-inf")), dim=-1)
    transition_log_probs.requires_grad_(True)
    transition_context = torch.randn(batch_size, seq_len, num_states, device="cuda", requires_grad=True)
    initial_log_belief = torch.randn(batch_size, num_states, device="cuda", requires_grad=True)

    cuda_result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=3,
    )
    ref = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.7,
        chunk_size=3,
    )

    assert torch.allclose(cuda_result.beliefs, ref.beliefs, atol=1e-4, rtol=1e-4)
    assert torch.allclose(cuda_result.final_log_belief, ref.final_log_belief, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_transition_gate_receives_gradient():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    local_logits = torch.randn(2, 5, 8, device="cuda", requires_grad=True)
    transition_log_probs = torch.randn(8, device="cuda", requires_grad=True)
    transition_context = torch.randn(2, 5, 8, device="cuda", requires_grad=True)
    initial_log_belief = torch.randn(2, 8, device="cuda", requires_grad=True)
    transition_gate = torch.tensor(0.7, device="cuda", requires_grad=True)

    result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=transition_gate,
        chunk_size=2,
    )
    loss = result.beliefs.sum() + result.final_log_belief.sum()
    loss.backward()

    assert transition_gate.grad is not None
    assert torch.isfinite(transition_gate.grad).all()


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_is_chunk_invariant_for_longer_sequence():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    local_logits = torch.randn(2, 96, 16, device="cuda")
    transition_log_probs = torch.randn(16, 16, device="cuda")
    transition_context = torch.randn(2, 96, 16, device="cuda")
    initial_log_belief = torch.randn(2, 16, device="cuda")

    result_chunk_1 = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.5,
        chunk_size=1,
    )
    result_chunk_12 = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.5,
        chunk_size=12,
    )
    ref = causal_belief_scan_ref(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.5,
        chunk_size=96,
    )

    assert torch.allclose(result_chunk_1.beliefs, result_chunk_12.beliefs, atol=1e-4, rtol=1e-4)
    assert torch.allclose(result_chunk_1.final_log_belief, result_chunk_12.final_log_belief, atol=1e-4, rtol=1e-4)
    assert torch.allclose(result_chunk_12.beliefs, ref.beliefs, atol=1e-4, rtol=1e-4)
    assert torch.allclose(result_chunk_12.final_log_belief, ref.final_log_belief, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _TORCH_CUDA_READY, reason="CUDA is required")
def test_cuda_belief_scan_stability_smoke_large_rollout():
    status = cuda_belief_scan_status()
    if not status.available:
        pytest.skip(status.detail or "native causal-belief extension is unavailable")

    torch.manual_seed(0)
    local_logits = torch.randn(2, 192, 32, device="cuda")
    transition_log_probs = torch.randn(32, 32, device="cuda")
    transition_context = torch.randn(2, 192, 32, device="cuda")
    initial_log_belief = torch.randn(2, 32, device="cuda")

    result = causal_belief_scan(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate=0.4,
        chunk_size=24,
    )

    assert torch.isfinite(result.beliefs).all()
    assert torch.isfinite(result.final_log_belief).all()
