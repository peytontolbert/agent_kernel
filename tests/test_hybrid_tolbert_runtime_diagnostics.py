from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from agent_kernel.modeling.tolbert import (
    HybridTolbertSSMConfig,
    HybridTolbertSSMModel,
    generate_hybrid_decoder_text,
    save_hybrid_runtime_bundle,
    score_hybrid_candidates,
)
from agent_kernel.schemas import TaskSpec
from agent_kernel.state import AgentState


def test_hybrid_model_surfaces_recurrent_and_world_diagnostics() -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6, sequence_length=4, max_command_tokens=5)
    model = HybridTolbertSSMModel(config)

    output = model(
        command_token_ids=torch.randint(0, config.token_vocab_size, (2, config.sequence_length, config.max_command_tokens)),
        scalar_features=torch.randn(2, config.sequence_length, config.scalar_feature_dim),
        family_ids=torch.randint(0, config.family_vocab_size, (2,)),
        path_level_ids=torch.randint(0, config.path_vocab_size, (2, config.max_path_levels)),
        decoder_input_ids=torch.randint(0, config.decoder_vocab_size, (2, config.max_command_tokens)),
        prefer_python_ref=True,
        prefer_python_world_ref=True,
    )

    assert output.ssm_last_state is not None
    assert output.ssm_last_state.shape[0] == 2
    assert output.ssm_diagnostics["backend"] == "python_ref"
    assert output.ssm_diagnostics["last_state_norm_mean"] >= 0.0
    assert output.ssm_diagnostics["pooled_norm_mean"] >= 0.0
    assert output.world_diagnostics["backend"] == "python_ref"
    assert output.world_diagnostics["transition_family"] == "banded"
    assert output.world_diagnostics["transition_bandwidth"] == config.world_transition_bandwidth
    assert output.world_diagnostics["transition_gate"] > 0.0
    assert len(output.world_diagnostics["final_top_states"]) >= 1
    assert output.world_diagnostics["structure"]["family"] == "banded"


def test_hybrid_model_incremental_world_state_matches_prefix_recompute() -> None:
    torch.manual_seed(0)
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6, sequence_length=4, max_command_tokens=5, world_state_dim=4)
    model = HybridTolbertSSMModel(config)

    pooled_state = torch.randn(1, config.hidden_dim)
    decoder_input_ids = torch.tensor(
        [[config.decoder_bos_token_id, 4, 5, 6]],
        dtype=torch.long,
    )
    initial_log_belief = torch.log_softmax(torch.randn(1, config.world_state_dim), dim=-1)

    full_world_belief, full_world_diagnostics = model.decode_world_tokens(
        pooled_state=pooled_state,
        decoder_input_ids=decoder_input_ids,
        initial_log_belief=initial_log_belief,
        prefer_python_ref=True,
    )

    incremental_world_belief = initial_log_belief
    for position in range(decoder_input_ids.shape[1]):
        incremental_world_belief, step_diagnostics = model.advance_world_state(
            pooled_state=pooled_state,
            token_ids=decoder_input_ids[:, position],
            position=position,
            world_log_belief=incremental_world_belief,
            prefer_python_ref=True,
        )
        assert step_diagnostics["backend"] == "python_ref"
        assert step_diagnostics["position"] == position

    assert isinstance(full_world_belief, torch.Tensor)
    assert isinstance(incremental_world_belief, torch.Tensor)
    assert full_world_diagnostics["transition_family"] == "banded"
    assert torch.allclose(incremental_world_belief, full_world_belief, atol=1e-6, rtol=1e-6)


def test_hybrid_decoder_logits_shift_with_world_belief_feedback() -> None:
    torch.manual_seed(0)
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6, sequence_length=4, max_command_tokens=5, world_state_dim=4)
    model = HybridTolbertSSMModel(config)

    pooled_state = torch.randn(1, config.hidden_dim)
    decoder_input_ids = torch.tensor(
        [[config.decoder_bos_token_id, 4, 5, 0]],
        dtype=torch.long,
    )
    low_entropy_world = torch.log(torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float32))
    alternate_world = torch.log(torch.tensor([[0.01, 0.97, 0.01, 0.01]], dtype=torch.float32))

    logits_a, diag_a = model.decode_decoder_tokens(
        pooled_state=pooled_state,
        decoder_input_ids=decoder_input_ids,
        prefer_python_ref=True,
        initial_world_log_belief=low_entropy_world,
        prefer_python_world_ref=True,
    )
    logits_b, diag_b = model.decode_decoder_tokens(
        pooled_state=pooled_state,
        decoder_input_ids=decoder_input_ids,
        prefer_python_ref=True,
        initial_world_log_belief=alternate_world,
        prefer_python_world_ref=True,
    )

    assert diag_a["world_backend"] == "python_ref"
    assert diag_b["world_backend"] == "python_ref"
    assert not torch.allclose(logits_a, logits_b)


def test_score_hybrid_candidates_surfaces_runtime_diagnostics(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(
        hidden_dim=24,
        d_state=6,
        sequence_length=4,
        max_command_tokens=5,
        world_state_dim=2,
    )
    model = HybridTolbertSSMModel(config)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "future_signature_profile": {
                    "available": True,
                    "horizons": [1],
                    "signature_dim": 5,
                },
                "spectral_eigenbases": {
                    "sidecar_npz": str(tmp_path / "profile_spectral_eigenbases.npz"),
                },
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.savez(
        tmp_path / "profile_spectral_eigenbases.npz",
        causal_machine_signature_centroids=np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        causal_machine_log_probs=np.zeros((2,), dtype=np.float32),
        causal_machine_state_masses=np.array([1.0, 1.0], dtype=np.float32),
    )
    bundle_path = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "bundle",
        model=model,
        config=config,
        metadata={"causal_world_profile_path": str(profile_path)},
    )
    state = AgentState(
        task=TaskSpec(
            task_id="repair_task",
            prompt="repair missing report.txt",
            workspace_subdir="repair_task",
            suggested_commands=["printf 'ok\\n' > report.txt"],
            metadata={"benchmark_family": "repository"},
        )
    )
    state.world_model_summary = {"missing_expected_artifacts": ["report.txt"]}

    scored = score_hybrid_candidates(
        state=state,
        candidates=[{"action": "code_execute", "command": "printf 'ok\\n' > report.txt"}],
        bundle_manifest_path=bundle_path,
        device="cpu",
    )

    first = scored[0]
    assert first["hybrid_ssm_backend"] == "python_ref"
    assert first["hybrid_ssm_last_state_norm_mean"] >= 0.0
    assert first["hybrid_ssm_pooled_state_norm_mean"] >= 0.0
    assert first["hybrid_world_backend"] == "python_ref"
    assert first["hybrid_world_transition_family"] == "banded"
    assert first["hybrid_world_transition_bandwidth"] == 1
    assert first["hybrid_world_transition_gate"] > 0.0
    assert first["hybrid_world_prior_backend"] == "profile_conditioned"
    assert isinstance(first["hybrid_world_final_top_states"], list)


def test_generate_hybrid_decoder_text_incremental_cache_matches_full_recompute(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = HybridTolbertSSMConfig(hidden_dim=16, d_state=4, sequence_length=3, max_command_tokens=4, decoder_vocab_size=12)
    model = HybridTolbertSSMModel(config)
    decoder_vocab_path = tmp_path / "decoder_vocab.json"
    decoder_vocab_path.write_text(json.dumps({"printf": 4, "done": 5}, indent=2), encoding="utf-8")
    manifest_path = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "bundle",
        model=model,
        config=config,
        decoder_vocab_path=decoder_vocab_path,
    )
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow"},
        )
    )

    cached = generate_hybrid_decoder_text(
        state=state,
        bundle_manifest_path=manifest_path,
        max_new_tokens=3,
        use_incremental_cache=True,
        verify_cache_equivalence=True,
        cache_equivalence_steps=3,
    )
    full = generate_hybrid_decoder_text(
        state=state,
        bundle_manifest_path=manifest_path,
        max_new_tokens=3,
        use_incremental_cache=False,
        verify_cache_equivalence=False,
    )

    assert cached["generated_ids"] == full["generated_ids"]
    assert cached["generated_text"] == full["generated_text"]
    assert cached["cache_mode"] == "incremental_recurrent"
    assert cached["cache_equivalent"] is True
    assert cached["cache_fallback_used"] is False
    assert cached["cache_steps_reused"] >= 1
    assert cached["world_cache_mode"] == "incremental_belief"
    assert cached["world_cache_equivalent"] is True
    assert cached["world_cache_fallback_used"] is False
    assert cached["world_cache_steps_reused"] >= 1
    assert isinstance(cached["world_final_top_states"], list)
