from __future__ import annotations

from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.functional")

import agent_kernel.modeling.tolbert.runtime as tolbert_runtime_module
from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import retained_tolbert_hybrid_runtime
from agent_kernel.modeling.world.rollout import rollout_action_value
from agent_kernel.modeling.world import build_causal_state_signature
from agent_kernel.modeling.tolbert import (
    HybridTolbertSSMConfig,
    HybridTolbertSSMModel,
    generate_hybrid_decoder_text,
    load_hybrid_runtime_bundle,
    save_hybrid_runtime_bundle,
)
from agent_kernel.modeling.training.hybrid_dataset import materialize_hybrid_training_dataset
from agent_kernel.modeling.training.hybrid_trainer import train_hybrid_runtime_from_dataset, train_hybrid_runtime_from_trajectories
from agent_kernel.modeling.training.universal_trainer import train_hybrid_decoder_from_universal_dataset
from agent_kernel.modeling.adapter_training import InjectedLoRAState
from agent_kernel.policy import LLMDecisionPolicy, SkillLibrary
from agent_kernel.schemas import StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.task_bank import TaskBank
import numpy as np


def _write_hybrid_training_fixture(dataset_path: Path, config: HybridTolbertSSMConfig) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_vocab_path = dataset_path.parent / "decoder_vocab.json"
    decoder_vocab_path.write_text(json.dumps({"hello": 4, "world": 5}, indent=2), encoding="utf-8")
    example = {
        "family_id": 1,
        "path_level_ids": [1 for _ in range(config.max_path_levels)],
        "command_token_ids": [[1 for _ in range(config.max_command_tokens)] for _ in range(config.sequence_length)],
        "decoder_input_ids": [config.decoder_bos_token_id, 4, 5, 0],
        "decoder_target_ids": [4, 5, config.decoder_eos_token_id, 0],
        "scalar_features": [[0.1 for _ in range(config.scalar_feature_dim)] for _ in range(config.sequence_length)],
        "score_target": 0.5,
        "policy_target": 1.0,
        "value_target": 0.25,
        "stop_target": 0.0,
        "risk_target": 0.0,
        "transition_target": [1.0, 0.0],
        "world_target": [0.5 for _ in range(config.world_state_dim)],
    }
    dataset_path.write_text(json.dumps(example) + "\n", encoding="utf-8")
    dataset_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "decoder_vocab_path": str(decoder_vocab_path),
                "decoder_vocab_entry_count": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_universal_training_fixture(manifest_path: Path, config: HybridTolbertSSMConfig) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset_path = manifest_path.parent / "universal_train.jsonl"
    decoder_vocab_path = manifest_path.parent / "decoder_vocab.json"
    tokenizer_manifest_path = manifest_path.parent / "decoder_tokenizer_manifest.json"
    train_dataset_path.write_text(
        json.dumps({"prompt": "echo hello", "target": "hello world"}) + "\n",
        encoding="utf-8",
    )
    decoder_vocab_path.write_text(json.dumps({"hello": 4, "world": 5}, indent=2), encoding="utf-8")
    tokenizer_manifest_path.write_text(json.dumps({"tokenizer_kind": "regex_v1"}, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "train_dataset_path": str(train_dataset_path),
                "decoder_vocab_path": str(decoder_vocab_path),
                "decoder_tokenizer_manifest_path": str(tokenizer_manifest_path),
                "decoder_vocab_size": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_hybrid_tolbert_model_forward_shapes() -> None:
    config = HybridTolbertSSMConfig(hidden_dim=32, d_state=8, sequence_length=4, max_command_tokens=6)
    model = HybridTolbertSSMModel(config)

    output = model(
        command_token_ids=torch.randint(0, config.token_vocab_size, (3, config.sequence_length, config.max_command_tokens)),
        scalar_features=torch.randn(3, config.sequence_length, config.scalar_feature_dim),
        family_ids=torch.randint(0, config.family_vocab_size, (3,)),
        path_level_ids=torch.randint(0, config.path_vocab_size, (3, config.max_path_levels)),
        decoder_input_ids=torch.randint(0, config.decoder_vocab_size, (3, config.max_command_tokens)),
        prefer_python_ref=True,
    )

    assert output.score.shape == (3,)
    assert output.policy_logits.shape == (3,)
    assert output.value.shape == (3,)
    assert output.stop_logits.shape == (3,)
    assert output.risk_logits.shape == (3,)
    assert output.transition.shape == (3, 2)
    assert output.decoder_logits.shape == (3, config.max_command_tokens, config.decoder_vocab_size)
    assert output.pooled_state.shape == (3, config.hidden_dim)
    assert output.world_final_belief is not None
    assert output.world_final_belief.shape == (3, config.world_state_dim)
    assert output.world_backend == "python_ref"
    assert config.use_dense_world_transition is False


def test_hybrid_tolbert_decoder_is_causal() -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6, sequence_length=4, max_command_tokens=5, decoder_vocab_size=16)
    model = HybridTolbertSSMModel(config)
    common_kwargs = {
        "command_token_ids": torch.randint(0, config.token_vocab_size, (1, config.sequence_length, config.max_command_tokens)),
        "scalar_features": torch.randn(1, config.sequence_length, config.scalar_feature_dim),
        "family_ids": torch.randint(0, config.family_vocab_size, (1,)),
        "path_level_ids": torch.randint(0, config.path_vocab_size, (1, config.max_path_levels)),
        "prefer_python_ref": True,
        "prefer_python_world_ref": True,
    }
    input_a = torch.tensor([[config.decoder_bos_token_id, 4, 5, 6, 0]], dtype=torch.long)
    input_b = torch.tensor([[config.decoder_bos_token_id, 4, 9, 10, 0]], dtype=torch.long)

    output_a = model(decoder_input_ids=input_a, **common_kwargs)
    output_b = model(decoder_input_ids=input_b, **common_kwargs)

    assert torch.allclose(output_a.decoder_logits[:, 0, :], output_b.decoder_logits[:, 0, :], atol=1.0e-5)
    assert not torch.allclose(output_a.decoder_logits[:, 3, :], output_b.decoder_logits[:, 3, :], atol=1.0e-5)


def test_hybrid_tolbert_world_gate_receives_gradient() -> None:
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
    loss = output.score.sum() + output.policy_logits.sum() + output.value.sum() + output.transition.sum() + output.decoder_logits.sum()
    loss.backward()

    assert model.world_transition_gate.grad is not None
    assert model.score_head.weight.grad is not None
    assert model.decoder_head.weight.grad is not None


def test_hybrid_tolbert_world_scan_uses_structured_transition_inputs_when_parametrized(monkeypatch: pytest.MonkeyPatch) -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6, sequence_length=4, max_command_tokens=5)
    model = HybridTolbertSSMModel(config)
    InjectedLoRAState(
        model,
        rank=2,
        direct_parameter_filter=lambda parameter_path, parameter: parameter_path == "world_transition_logits",
    )
    captured: dict[str, object] = {}

    def _fake_scan(local_logits, transition_log_probs, transition_context, initial_log_belief, **kwargs):
        captured["transition_structure"] = kwargs.get("transition_structure")
        captured["transition_log_probs_shape"] = tuple(transition_log_probs.shape)
        beliefs = torch.log_softmax(local_logits, dim=-1)
        return type(
            "Result",
            (),
            {
                "beliefs": beliefs,
                "final_log_belief": beliefs[:, -1, :],
                "backend": "python_ref",
            },
        )()

    monkeypatch.setattr("agent_kernel.modeling.tolbert.hybrid_model.causal_belief_scan", _fake_scan)

    output = model(
        command_token_ids=torch.randint(0, config.token_vocab_size, (1, config.sequence_length, config.max_command_tokens)),
        scalar_features=torch.randn(1, config.sequence_length, config.scalar_feature_dim),
        family_ids=torch.randint(0, config.family_vocab_size, (1,)),
        path_level_ids=torch.randint(0, config.path_vocab_size, (1, config.max_path_levels)),
        decoder_input_ids=torch.randint(0, config.decoder_vocab_size, (1, config.max_command_tokens)),
        prefer_python_ref=True,
        prefer_python_world_ref=True,
    )

    transition_structure = captured["transition_structure"]
    assert isinstance(transition_structure, dict)
    assert transition_structure["kind"] == "source_dest_stay_v1"
    assert tuple(captured["transition_log_probs_shape"]) == (config.world_state_dim, config.world_state_dim)
    assert output.world_backend == "python_ref"


def test_hybrid_runtime_bundle_round_trip(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6)
    model = HybridTolbertSSMModel(config)
    decoder_vocab_path = tmp_path / "decoder_vocab.json"
    decoder_vocab_path.write_text(json.dumps({"hello": 4, "world": 5}, indent=2), encoding="utf-8")

    manifest_path = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "bundle",
        model=model,
        config=config,
        metadata={"example_count": 2},
        decoder_vocab_path=decoder_vocab_path,
    )
    loaded_model, loaded_config, manifest = load_hybrid_runtime_bundle(manifest_path)

    assert isinstance(loaded_model, HybridTolbertSSMModel)
    assert loaded_config.hidden_dim == config.hidden_dim
    assert manifest["artifact_kind"] == "tolbert_hybrid_runtime_bundle"
    assert Path(manifest["metadata"]["decoder_vocab_path"]).exists()


def test_hybrid_runtime_bundle_honors_explicit_paths(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6)
    model = HybridTolbertSSMModel(config)
    config_path = tmp_path / "runtime" / "tolbert_config.json"
    checkpoint_path = tmp_path / "runtime" / "weights" / "tolbert_checkpoint.pt"
    manifest_path = tmp_path / "runtime" / "manifests" / "bundle.json"

    saved_manifest = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "unused_default_dir",
        model=model,
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )

    assert saved_manifest == manifest_path
    assert config_path.exists()
    assert checkpoint_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_path"] == str(config_path)
    assert manifest["checkpoint_path"] == str(checkpoint_path)


def test_hybrid_runtime_training_emits_structured_adapter_delta_when_parent_checkpoint_exists(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(
        hidden_dim=16,
        d_state=4,
        sequence_length=2,
        max_command_tokens=4,
        scalar_feature_dim=4,
        world_state_dim=2,
        token_vocab_size=32,
        decoder_vocab_size=8,
        family_vocab_size=8,
        path_vocab_size=64,
    )
    dataset_path = tmp_path / "dataset" / "hybrid_training_dataset.jsonl"
    _write_hybrid_training_fixture(dataset_path, config)
    parent_output_dir = tmp_path / "parent_bundle"
    parent_manifest_path = save_hybrid_runtime_bundle(
        output_dir=parent_output_dir,
        model=HybridTolbertSSMModel(config),
        config=config,
    )
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    output_dir = tmp_path / "trained_bundle"

    manifest = train_hybrid_runtime_from_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
        epochs=1,
        batch_size=1,
        lr=1.0e-3,
        device="cpu",
        runtime_paths={
            "parent_checkpoint_path": str(parent_manifest["checkpoint_path"]),
            "checkpoint_delta_path": str(output_dir / "hybrid_checkpoint__delta.pt"),
            "bundle_manifest_path": str(output_dir / "hybrid_bundle_manifest.json"),
            "config_path": str(output_dir / "hybrid_config.json"),
            "checkpoint_path": str(output_dir / "hybrid_checkpoint.pt"),
        },
    )

    assert manifest["checkpoint_path"] == ""
    assert Path(str(manifest["checkpoint_delta_path"])).exists()
    assert manifest["checkpoint_mutation"]["mode"] == "parent_plus_structured_adapter_training"
    assert manifest["metadata"]["training_mode"] == "injected_lora_plus_structured_adapters"
    assert int(manifest["checkpoint_mutation"]["stats"]["dense_delta_key_count"]) == 0
    assert int(manifest["checkpoint_mutation"]["stats"]["structured_parameter_adapter_key_count"]) >= 1
    delta_payload = torch.load(Path(str(manifest["checkpoint_delta_path"])), map_location="cpu")
    assert delta_payload["state_dict_adapters"]["world_transition_logits"]["kind"] == "structured_transition_adapter"
    loaded_model, loaded_config, loaded_manifest = load_hybrid_runtime_bundle(output_dir / "hybrid_bundle_manifest.json")
    assert isinstance(loaded_model, HybridTolbertSSMModel)
    assert loaded_config.hidden_dim == config.hidden_dim
    assert Path(loaded_manifest["checkpoint_path"]).exists()


def test_world_initial_belief_payload_uses_profile_conditioned_prior(tmp_path: Path) -> None:
    sketch, token_count = build_causal_state_signature(
        ["repair missing report.txt", "status.txt"],
        sketch_dim=4,
    )
    sidecar_path = tmp_path / "profile_spectral_eigenbases.npz"
    np.savez(
        sidecar_path,
        causal_machine_signature_centroids=np.stack([sketch, -sketch]).astype(np.float32),
        causal_machine_log_probs=np.zeros((2,), dtype=np.float32),
        causal_machine_state_masses=np.array([1.0, 1.0], dtype=np.float32),
    )
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
                    "sidecar_npz": str(sidecar_path),
                },
            }
        ),
        encoding="utf-8",
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
    state.world_model_summary = {"missing_expected_artifacts": ["report.txt", "status.txt"]}

    payload = tolbert_runtime_module._world_initial_belief_payload(
        batch_size=2,
        state=state,
        config=HybridTolbertSSMConfig(world_state_dim=2),
        manifest={"metadata": {"causal_world_profile_path": str(profile_path)}},
        device="cpu",
    )

    assert isinstance(payload["log_belief"], torch.Tensor)
    assert payload["log_belief"].shape == (2, 2)
    assert payload["diagnostics"]["backend"] == "profile_conditioned"
    assert payload["diagnostics"]["matched_state_index"] == 0
    assert payload["diagnostics"]["token_count"] >= token_count


def test_universal_decoder_training_uses_injected_lora_plus_structured_adapters(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(
        hidden_dim=16,
        d_state=4,
        sequence_length=2,
        max_command_tokens=4,
        scalar_feature_dim=4,
        world_state_dim=2,
        token_vocab_size=32,
        decoder_vocab_size=8,
        family_vocab_size=8,
        path_vocab_size=64,
    )
    parent_output_dir = tmp_path / "parent_bundle"
    parent_manifest_path = save_hybrid_runtime_bundle(
        output_dir=parent_output_dir,
        model=HybridTolbertSSMModel(config),
        config=config,
    )
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest_path = tmp_path / "universal_dataset" / "manifest.json"
    _write_universal_training_fixture(dataset_manifest_path, config)

    manifest = train_hybrid_decoder_from_universal_dataset(
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "trained_universal_bundle",
        config=config,
        epochs=1,
        batch_size=1,
        lr=1.0e-3,
        device="cpu",
        parent_checkpoint_path=Path(str(parent_manifest["checkpoint_path"])),
    )

    assert manifest["checkpoint_path"] == ""
    assert Path(str(manifest["checkpoint_delta_path"])).exists()
    assert manifest["metadata"]["training_mode"] == "injected_lora_plus_structured_adapters"
    assert int(manifest["checkpoint_mutation"]["stats"]["dense_delta_key_count"]) == 0
    assert int(manifest["checkpoint_mutation"]["stats"]["structured_parameter_adapter_key_count"]) >= 1
    delta_payload = torch.load(Path(str(manifest["checkpoint_delta_path"])), map_location="cpu")
    assert delta_payload["state_dict_adapters"]["world_transition_logits"]["kind"] == "structured_transition_adapter"


def test_hybrid_runtime_bundle_uses_relative_paths_when_relocated(tmp_path: Path, monkeypatch) -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6)
    model = HybridTolbertSSMModel(config)

    manifest_path = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "bundle",
        model=model,
        config=config,
        metadata={"example_count": 2},
    )
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    moved_manifest = relocated_dir / manifest_path.name
    moved_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("hybrid_config.json", "hybrid_checkpoint.pt"):
        target = relocated_dir / name
        source = manifest_path.parent / name
        target.write_bytes(source.read_bytes())

    monkeypatch.chdir(tmp_path)
    loaded_model, loaded_config, manifest = load_hybrid_runtime_bundle(moved_manifest)

    assert isinstance(loaded_model, HybridTolbertSSMModel)
    assert loaded_config.hidden_dim == config.hidden_dim
    assert Path(manifest["config_path"]).resolve().parent == relocated_dir.resolve()


def test_hybrid_runtime_bundle_loads_legacy_checkpoint_without_score_head(tmp_path: Path) -> None:
    config = HybridTolbertSSMConfig(hidden_dim=24, d_state=6)
    model = HybridTolbertSSMModel(config)

    manifest_path = save_hybrid_runtime_bundle(
        output_dir=tmp_path / "bundle",
        model=model,
        config=config,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint_path"])
    payload = torch.load(checkpoint_path)
    payload["state_dict"].pop("score_head.weight", None)
    payload["state_dict"].pop("score_head.bias", None)
    torch.save(payload, checkpoint_path)

    loaded_model, _, _ = load_hybrid_runtime_bundle(manifest_path)

    assert isinstance(loaded_model, HybridTolbertSSMModel)


def test_materialize_hybrid_training_dataset_counts_step_examples(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
                "steps": [
                    {
                        "action": "code_execute",
                        "content": "printf 'hello\\n' > hello.txt",
                        "verification": {"passed": True},
                        "path_confidence": 0.8,
                        "trust_retrieval": True,
                        "state_progress_delta": 0.5,
                        "state_regression_count": 0,
                        "available_skill_count": 1,
                        "retrieval_candidate_count": 1,
                        "retrieval_evidence_count": 1,
                        "retrieval_influenced": True,
                        "retrieval_ranked_skill": False,
                        "state_transition": {"no_progress": False},
                        "proposal_metadata": {
                            "hybrid_total_score": 3.5,
                            "hybrid_world_progress_score": 0.81,
                            "hybrid_decoder_world_progress_score": 0.76,
                            "hybrid_decoder_world_risk_score": 0.12,
                            "hybrid_transition_progress": 0.4,
                            "hybrid_transition_regression": 0.1,
                            "hybrid_trusted_retrieval_alignment": 0.6,
                            "hybrid_graph_environment_alignment": 0.4,
                            "hybrid_transfer_novelty": 1.0,
                        },
                        "latent_state_summary": {
                            "progress_band": "advancing",
                            "risk_band": "stable",
                            "learned_world_state": {
                                "source": "tolbert_hybrid_runtime",
                                "model_family": "tolbert_ssm_v1",
                                "progress_signal": 0.82,
                                "risk_signal": 0.18,
                                "decoder_world_progress_score": 0.7,
                                "decoder_world_risk_score": 0.2,
                                "world_transition_family": "banded",
                            },
                        },
                    },
                    {
                        "action": "respond",
                        "content": "done",
                        "verification": {"passed": True},
                        "path_confidence": 0.9,
                        "trust_retrieval": True,
                        "state_progress_delta": 0.6,
                        "state_regression_count": 0,
                        "proposal_metadata": {
                            "hybrid_total_score": 5.0,
                            "hybrid_decoder_world_progress_score": 0.95,
                            "hybrid_decoder_world_risk_score": 0.04,
                            "hybrid_decoder_world_entropy_mean": 0.2,
                        },
                        "latent_state_summary": {
                            "progress_band": "advancing",
                            "risk_band": "stable",
                            "learned_world_state": {
                                "progress_signal": 0.9,
                                "risk_signal": 0.05,
                                "decoder_world_progress_score": 0.95,
                                "decoder_world_risk_score": 0.04,
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_hybrid_training_dataset(
        trajectories_root=episodes,
        output_path=tmp_path / "dataset" / "hybrid.jsonl",
        config=HybridTolbertSSMConfig(sequence_length=4),
    )

    assert manifest["artifact_kind"] == "tolbert_hybrid_training_dataset"
    assert manifest["example_count"] == 2
    lines = (tmp_path / "dataset" / "hybrid.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["value_target"] != second["value_target"] or first["stop_target"] != second["stop_target"]
    assert 0.0 <= first["score_target"] <= 1.0
    assert 0.0 <= first["risk_target"] <= 1.0
    assert len(first["transition_target"]) == 2
    assert len(first["decoder_input_ids"]) == HybridTolbertSSMConfig(sequence_length=4).max_command_tokens
    assert len(first["decoder_target_ids"]) == HybridTolbertSSMConfig(sequence_length=4).max_command_tokens
    assert len(first["world_target"]) == HybridTolbertSSMConfig(sequence_length=4).world_state_dim
    assert abs(sum(first["world_target"]) - 1.0) < 1.0e-6
    assert first["world_feedback"]["progress_signal"] >= 0.8
    assert first["world_feedback"]["model_family"] == "tolbert_ssm_v1"
    assert second["world_feedback"]["decoder_world_progress_score"] == 0.95
    assert first["world_feedback"]["trusted_retrieval_alignment"] == pytest.approx(0.6, rel=1e-6)
    assert first["world_feedback"]["graph_environment_alignment"] == pytest.approx(0.4, rel=1e-6)
    assert first["world_feedback"]["transfer_novelty"] == pytest.approx(1.0, rel=1e-6)
    assert first["scalar_features"][-1][6] == pytest.approx(0.6, rel=1e-6)
    assert first["scalar_features"][-1][7] == pytest.approx(0.4, rel=1e-6)
    assert first["scalar_features"][-1][8] == pytest.approx(1.0, rel=1e-6)
    assert first["scalar_features"][-1][-3] >= 0.0
    assert first["scalar_features"][-1][-2] >= 0.8
    assert manifest["shard_paths"] == []
    assert not (tmp_path / "dataset" / "hybrid_shards").exists()


def test_materialize_hybrid_training_dataset_tracks_long_horizon_weighting(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "project_task.json").write_text(
        json.dumps(
            {
                "task_id": "project_task",
                "success": True,
                "task_metadata": {"benchmark_family": "project", "capability": "handoff", "difficulty": "long_horizon"},
                "steps": [
                    {
                        "index": 1,
                        "action": "code_execute",
                        "content": "printf 'alpha\\n' > summary.txt",
                        "verification": {"passed": False},
                        "state_progress_delta": 0.2,
                        "state_regression_count": 0,
                        "state_transition": {"no_progress": False},
                    },
                    {
                        "index": 2,
                        "action": "code_execute",
                        "content": "printf 'beta\\n' >> summary.txt",
                        "verification": {"passed": True},
                        "state_progress_delta": 0.5,
                        "state_regression_count": 0,
                        "state_transition": {"no_progress": False},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_hybrid_training_dataset(
        trajectories_root=episodes,
        output_path=tmp_path / "dataset" / "hybrid.jsonl",
        config=HybridTolbertSSMConfig(sequence_length=4),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "hybrid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["difficulty_counts"]["long_horizon"] == 2
    assert manifest["long_horizon_example_count"] == 2
    assert manifest["long_horizon_weighted_example_share"] > 0.5
    assert all(row["task_difficulty"] == "long_horizon" for row in rows)
    assert all(float(row["example_weight"]) > 1.0 for row in rows)


def test_materialize_hybrid_training_dataset_tracks_transfer_alignment_examples(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "repository_task.json").write_text(
        json.dumps(
            {
                "task_id": "repository_task",
                "success": True,
                "task_metadata": {"benchmark_family": "repository", "capability": "repair", "difficulty": "long_horizon"},
                "steps": [
                    {
                        "index": 1,
                        "action": "code_execute",
                        "content": "git status --short",
                        "verification": {"passed": True},
                        "state_progress_delta": 0.15,
                        "state_regression_count": 0,
                        "state_transition": {"no_progress": False},
                        "proposal_metadata": {
                            "hybrid_trusted_retrieval_alignment": 0.8,
                            "hybrid_graph_environment_alignment": 0.6,
                            "hybrid_transfer_novelty": 1.0,
                        },
                    },
                    {
                        "index": 2,
                        "action": "code_execute",
                        "content": "git commit -am 'checkpoint'",
                        "verification": {"passed": False},
                        "state_progress_delta": 0.0,
                        "state_regression_count": 1,
                        "state_transition": {"no_progress": True},
                        "proposal_metadata": {
                            "hybrid_graph_environment_alignment": -0.7,
                            "hybrid_transfer_novelty": 1.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_hybrid_training_dataset(
        trajectories_root=episodes,
        output_path=tmp_path / "dataset" / "hybrid.jsonl",
        config=HybridTolbertSSMConfig(sequence_length=4),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "hybrid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["trusted_retrieval_aligned_example_count"] == 1
    assert manifest["transfer_novelty_example_count"] == 2
    assert manifest["environment_safe_example_count"] == 1
    assert rows[0]["world_feedback"]["trusted_retrieval_alignment"] == pytest.approx(0.8, rel=1e-6)
    assert rows[0]["risk_target"] < rows[1]["risk_target"]
    assert rows[0]["score_target"] > rows[1]["score_target"]


def test_train_hybrid_runtime_from_trajectories_writes_bundle(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
                "steps": [
                    {
                        "action": "code_execute",
                        "content": "printf 'hello\\n' > hello.txt",
                        "verification": {"passed": True},
                        "state_progress_delta": 0.5,
                        "state_regression_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = train_hybrid_runtime_from_trajectories(
        trajectories_root=episodes,
        output_dir=tmp_path / "out",
        config=HybridTolbertSSMConfig(hidden_dim=16, d_state=4, sequence_length=3, max_command_tokens=4),
        epochs=1,
        batch_size=1,
    )

    assert manifest["artifact_kind"] == "tolbert_hybrid_runtime_bundle"
    assert Path(manifest["checkpoint_path"]).exists()
    assert Path(manifest["metadata"]["decoder_vocab_path"]).exists()


def test_generate_hybrid_decoder_text_returns_bounded_text(tmp_path: Path) -> None:
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
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )

    generated = generate_hybrid_decoder_text(
        state=state,
        bundle_manifest_path=manifest_path,
        max_new_tokens=3,
    )

    assert generated["model_family"] == config.model_family
    assert isinstance(generated["generated_text"], str)
    assert len(generated["generated_ids"]) <= 3


def test_train_hybrid_runtime_from_trajectories_honors_runtime_paths(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
                "steps": [
                    {
                        "action": "code_execute",
                        "content": "printf 'hello\\n' > hello.txt",
                        "verification": {"passed": True},
                        "state_progress_delta": 0.5,
                        "state_regression_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runtime_dir = tmp_path / "runtime"
    manifest = train_hybrid_runtime_from_trajectories(
        trajectories_root=episodes,
        output_dir=tmp_path / "out",
        config=HybridTolbertSSMConfig(hidden_dim=16, d_state=4, sequence_length=3, max_command_tokens=4),
        epochs=1,
        batch_size=1,
        runtime_paths={
            "config_path": str(runtime_dir / "custom_config.json"),
            "checkpoint_path": str(runtime_dir / "checkpoints" / "custom_checkpoint.pt"),
            "bundle_manifest_path": str(runtime_dir / "manifests" / "custom_manifest.json"),
        },
    )

    assert manifest["config_path"].endswith("custom_config.json")
    assert manifest["checkpoint_path"].endswith("custom_checkpoint.pt")
    assert Path(manifest["config_path"]).exists()
    assert Path(manifest["checkpoint_path"]).exists()



def test_retained_tolbert_hybrid_runtime_normalizes_payload() -> None:
    normalized = retained_tolbert_hybrid_runtime(
        {
            "artifact_kind": "tolbert_model_bundle",
            "hybrid_runtime": {
                "model_family": "tolbert_ssm_v1",
                "shadow_enabled": True,
                "bundle_manifest_path": "bundle.json",
                "scoring_policy": {
                    "value_weight": 2.5,
                },
            },
        }
    )

    assert normalized["shadow_enabled"] is True
    assert normalized["primary_enabled"] is False
    assert normalized["bundle_manifest_path"] == "bundle.json"
    assert normalized["scoring_policy"]["value_weight"] == 2.5
    assert normalized["scoring_policy"]["risk_penalty_weight"] == 1.0


def test_score_hybrid_candidates_uses_retained_scoring_policy(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([-8.0, 8.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    monkeypatch.setattr(
        tolbert_runtime_module,
        "_world_initial_log_belief",
        lambda **kwargs: None,
    )
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )
    candidates = [
        {"action": "code_execute", "command": "printf 'hello\\n' > hello.txt"},
        {"action": "respond", "command": "done"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"respond_stop_weight": 4.0},
    )

    assert scored[0]["action"] == "respond"
    assert scored[0]["hybrid_scoring_policy"]["respond_stop_weight"] == 4.0


def test_score_hybrid_candidates_uses_decoder_world_feedback_in_scoring_policy(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "decoder_diagnostics": {
                        "world_final_belief": torch.log(
                            torch.tensor(
                                [
                                    [0.05, 0.95],
                                    [0.95, 0.05],
                                ],
                                dtype=torch.float32,
                            )
                        ),
                    },
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig(world_state_dim=2)
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )
    candidates = [
        {"action": "code_execute", "command": "printf 'hello\\n' > hello.txt"},
        {"action": "respond", "command": "done"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={
            "respond_stop_weight": 0.0,
            "respond_decoder_world_progress_weight": 3.0,
            "respond_decoder_world_risk_penalty_weight": 0.0,
            "decoder_world_progress_weight": 0.0,
            "decoder_world_risk_penalty_weight": 3.0,
        },
    )

    assert scored[0]["action"] == "respond"
    assert scored[0]["hybrid_decoder_world_progress_score"] > scored[1]["hybrid_decoder_world_progress_score"]


def test_score_hybrid_candidates_rewards_long_horizon_progress(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 8.0]),
                    "transition": torch.tensor([[0.9, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    monkeypatch.setattr(
        tolbert_runtime_module,
        "_world_initial_belief_payload",
        lambda **kwargs: {
            "log_belief": None,
            "diagnostics": {
                "backend": "profile_conditioned",
                "matched_state_index": 0,
                "matched_state_probability": 0.8,
                "signature_norm": 0.0,
                "token_count": 3,
                "horizons": [1, 4],
            },
        },
    )
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_task",
            prompt="complete a multi-step workflow safely",
            workspace_subdir="long_horizon_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "long_horizon"},
        )
    )
    state.world_model_summary = {"horizon": "long_horizon"}
    candidates = [
        {"action": "code_execute", "command": "printf 'hello\\n' > hello.txt"},
        {"action": "respond", "command": "done"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={
            "respond_stop_weight": 1.0,
            "long_horizon_progress_bonus_weight": 1.0,
            "long_horizon_risk_penalty_weight": 0.0,
            "long_horizon_horizon_scale_weight": 0.5,
        },
    )

    assert scored[0]["action"] == "code_execute"
    assert scored[0]["hybrid_task_horizon"] == "long_horizon"
    assert scored[0]["hybrid_long_horizon_scale"] > 1.0
    assert scored[0]["hybrid_long_horizon_progress_signal"] == pytest.approx(0.9, rel=1e-6)


def test_score_hybrid_candidates_prefers_ranked_planner_recovery_stage(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_tolbert_task",
            prompt="Recover the workflow state.",
            workspace_subdir="planner_recovery_tolbert_task",
            suggested_commands=[],
            metadata={"benchmark_family": "workflow", "difficulty": "long_horizon"},
        )
    )
    state.planner_recovery_artifact = {
        "kind": "planner_recovery_rewrite",
        "next_stage_objective": "write workflow report reports/status.txt",
        "ranked_objectives": [
            {
                "objective": "write workflow report reports/status.txt",
                "score": 96,
                "status": "pending",
            },
            {
                "objective": "run workflow test release status check",
                "score": 68,
                "status": "pending",
            },
        ],
    }
    candidates = [
        {"action": "code_execute", "command": "pytest tests/test_release_status.py"},
        {"action": "code_execute", "command": "printf 'ready\\n' > reports/status.txt"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"recovery_stage_alignment_weight": 2.0},
    )

    assert scored[0]["command"] == "printf 'ready\\n' > reports/status.txt"
    assert scored[0]["hybrid_recovery_stage_alignment"] > 0.0
    assert scored[0]["hybrid_recovery_stage_rank"] == 1


def test_score_hybrid_candidates_prefers_trusted_retrieval_aligned_repair(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_hybrid_task",
            prompt="Recover the missing workflow report.",
            workspace_subdir="trusted_retrieval_hybrid_task",
            suggested_commands=[],
            expected_files=["reports/status.txt"],
            metadata={"benchmark_family": "workflow", "difficulty": "long_horizon"},
        )
    )
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["reports/status.txt"],
    }
    state.active_subgoal = "materialize expected artifact reports/status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "reports/status.txt",
            "source_role": "critic",
        }
    }
    state.graph_summary = {
        "trusted_retrieval_command_counts": {
            "printf 'ready\\n' > reports/status.txt": 3,
        }
    }
    candidates = [
        {"action": "code_execute", "command": "printf 'ready\\n' > reports/status.txt"},
        {"action": "code_execute", "command": "printf 'pending\\n' > notes.txt"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"trusted_retrieval_alignment_weight": 2.0},
    )

    assert scored[0]["command"] == "printf 'ready\\n' > reports/status.txt"
    assert scored[0]["hybrid_trusted_retrieval_alignment"] > 0.0


def test_score_hybrid_candidates_prefers_trusted_retrieval_procedure_continuation(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_procedure_task",
            prompt="Finish the retained repair procedure.",
            workspace_subdir="trusted_retrieval_procedure_task",
            suggested_commands=[],
            expected_files=["reports/status.txt"],
            metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="completed the trusted write stage",
            action="code_execute",
            content="printf 'status ready\\n' > reports/status.txt",
            selected_skill_id=None,
            command_result={
                "command": "printf 'status ready\\n' > reports/status.txt",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["tests not run yet"]},
        )
    ]
    state.graph_summary = {
        "trusted_retrieval_procedures": [
            {
                "commands": [
                    "printf 'status ready\\n' > reports/status.txt",
                    "pytest -q tests/test_status.py",
                ],
                "count": 2,
            }
        ]
    }
    candidates = [
        {"action": "code_execute", "command": "printf 'notes\\n' > scratch.txt"},
        {"action": "code_execute", "command": "pytest -q tests/test_status.py"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"trusted_retrieval_procedure_alignment_weight": 2.0},
    )

    assert scored[0]["command"] == "pytest -q tests/test_status.py"
    assert scored[0]["hybrid_trusted_retrieval_procedure_alignment"] > 0.0


def test_score_hybrid_candidates_penalizes_environment_conflicting_mutation(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="environment_transfer_hybrid_task",
            prompt="Orient in an unfamiliar repo before mutating it.",
            workspace_subdir="environment_transfer_hybrid_task",
            suggested_commands=[],
            metadata={"benchmark_family": "repository", "difficulty": "long_horizon"},
        )
    )
    state.graph_summary = {
        "observed_environment_modes": {
            "git_write_mode": {"operator_gated": 4},
            "workspace_write_scope": {"task_only": 4},
        },
        "environment_alignment_failures": {
            "git_write_aligned": 2,
        },
    }
    state.universe_summary = {
        "environment_snapshot": {
            "git_write_mode": "allowed",
            "workspace_write_scope": "task_only",
        }
    }
    candidates = [
        {"action": "code_execute", "command": "git status --short"},
        {"action": "code_execute", "command": "git commit -am 'checkpoint'"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"graph_environment_alignment_weight": 2.0},
    )

    assert scored[0]["command"] == "git status --short"
    assert scored[0]["hybrid_graph_environment_alignment"] > scored[1]["hybrid_graph_environment_alignment"]
    assert scored[0]["hybrid_transfer_novelty"] == 1


def test_score_hybrid_candidates_prefers_campaign_contract_aligned_command(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="campaign_contract_hybrid_task",
            prompt="Stay on the release workflow and do not drift.",
            workspace_subdir="campaign_contract_hybrid_task",
            suggested_commands=[],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = "materialize expected artifact src/release_state.txt"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "materialize expected artifact src/release_state.txt",
        "last_status": "regressed",
        "objective_states": {
            "materialize expected artifact src/release_state.txt": "regressed",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"materialize expected artifact src/release_state.txt": 2},
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "regressed",
                "step_index": 1,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": True,
            }
        ],
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 1
    state.repeated_action_count = 2
    state.latest_state_transition = {"regressed": True, "regressions": ["src/release_state.txt"]}
    candidates = [
        {"action": "code_execute", "command": "printf 'scratch\\n' > notes/todo.txt"},
        {"action": "code_execute", "command": "python scripts/fix_release.py --path src/release_state.txt"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={
            "campaign_contract_alignment_weight": 2.0,
            "campaign_drift_penalty_weight": 1.5,
        },
    )

    assert scored[0]["command"] == "python scripts/fix_release.py --path src/release_state.txt"
    assert scored[0]["hybrid_campaign_contract_alignment"] > 0.0
    assert scored[1]["hybrid_campaign_drift_penalty"] > 0.0


def test_score_hybrid_candidates_prefers_long_horizon_software_work_stage(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_tolbert_task",
            prompt="Complete the staged software release work.",
            workspace_subdir="software_work_tolbert_task",
            suggested_commands=[],
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [{"path": "src/release_state.txt", "edit_kind": "line_replace"}],
            },
        )
    )
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "updated_report_paths": [],
    }
    state.software_work_stage_state = {
        "current_objective": "materialize expected artifact src/release_state.txt",
        "last_status": "stalled",
        "objective_states": {
            "materialize expected artifact src/release_state.txt": "stalled",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {
            "materialize expected artifact src/release_state.txt": 2,
        },
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "stalled",
                "step_index": 1,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }
    state.history = [
        StepRecord(
            index=1,
            thought="failed release fix",
            action="code_execute",
            content="python scripts/fix_release.py --path src/release_state.txt",
            selected_skill_id=None,
            command_result={"command": "python scripts/fix_release.py --path src/release_state.txt", "exit_code": 1, "stdout": "", "stderr": "failed", "timed_out": False},
            verification={"passed": False, "reasons": ["still missing"]},
        )
    ]
    candidates = [
        {"action": "code_execute", "command": "printf 'ok\\n' > reports/release_review.txt"},
        {"action": "code_execute", "command": "python scripts/fix_release.py --path src/release_state.txt"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"software_work_alignment_weight": 2.0, "software_work_transition_weight": 2.0},
    )

    assert scored[0]["command"] == "printf 'ok\\n' > reports/release_review.txt"
    assert scored[0]["hybrid_software_work_alignment"] > 0.0
    assert scored[0]["hybrid_software_work_rank"] == 1
    assert scored[0]["hybrid_software_work_stage_status"] == "pending"
    assert scored[1]["hybrid_software_work_stage_status"] == "stalled"
    assert scored[1]["hybrid_software_work_transition_alignment"] < 0.0


def test_score_hybrid_candidates_prefers_phase_handoff_when_migration_is_finished(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_phase_handoff_task",
            prompt="Advance the staged release workflow.",
            workspace_subdir="software_work_phase_handoff_task",
            suggested_commands=[],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "accept required branch release/main",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["release/main"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch release/main",
        "last_status": "advanced",
        "objective_states": {
            "accept required branch release/main": "advanced",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"accept required branch release/main": 1},
        "recent_outcomes": [
            {
                "objective": "accept required branch release/main",
                "status": "advanced",
                "step_index": 2,
                "command": "git merge origin/release/main",
                "progress_delta": 0.4,
                "regressed": False,
            }
        ],
    }
    candidates = [
        {"action": "code_execute", "command": "pytest tests/test_release.py -k smoke"},
        {"action": "code_execute", "command": "git merge origin/release/main"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"software_work_alignment_weight": 2.0, "software_work_phase_handoff_weight": 2.0},
    )

    assert scored[0]["command"] == "pytest tests/test_release.py -k smoke"
    assert scored[0]["hybrid_software_work_candidate_phase"] == "test"
    assert scored[0]["hybrid_software_work_suggested_phase"] == "test"
    assert scored[0]["hybrid_software_work_phase_alignment"] > 0.0
    assert scored[1]["hybrid_software_work_candidate_phase"] == "migration"
    assert scored[1]["hybrid_software_work_phase_alignment"] < 0.0


def test_score_hybrid_candidates_prefers_merge_acceptance_when_phase_gate_is_active(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, **kwargs):
            del kwargs
            return type(
                "Output",
                (),
                {
                    "score": torch.tensor([0.0, 0.0]),
                    "policy_logits": torch.tensor([0.0, 0.0]),
                    "value": torch.tensor([0.0, 0.0]),
                    "risk_logits": torch.tensor([0.0, 0.0]),
                    "stop_logits": torch.tensor([0.0, 0.0]),
                    "transition": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                    "decoder_logits": torch.zeros(2, 16, 4096),
                    "world_final_belief": None,
                    "ssm_backend": "python_ref",
                    "world_backend": "python_ref",
                    "ssm_diagnostics": {},
                    "world_diagnostics": {},
                },
            )()

    config = HybridTolbertSSMConfig()
    monkeypatch.setattr(
        tolbert_runtime_module,
        "load_hybrid_runtime_bundle",
        lambda bundle_manifest_path, device="cpu": (
            FakeModel(),
            config,
            {"model_family": "tolbert_ssm_v1", "metadata": {}},
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_phase_gate_tolbert_task",
            prompt="Accept the required release branch before testing.",
            workspace_subdir="software_work_phase_gate_tolbert_task",
            suggested_commands=[],
            metadata={"benchmark_family": "repo_sandbox", "difficulty": "long_horizon"},
        )
    )
    state.plan = [
        "accept required branch worker/api-release",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["worker/api-release"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch worker/api-release",
        "last_status": "stalled",
        "objective_states": {
            "accept required branch worker/api-release": "stalled",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"accept required branch worker/api-release": 1},
        "recent_outcomes": [
            {
                "objective": "accept required branch worker/api-release",
                "status": "stalled",
                "step_index": 1,
                "command": "git merge --no-ff worker/api-release",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }
    candidates = [
        {"action": "code_execute", "command": "tests/test_release.sh && printf 'release suite passed\\n' > reports/test_report.txt"},
        {"action": "code_execute", "command": "git merge --no-ff worker/api-release -m 'merge worker/api-release'"},
    ]

    scored = tolbert_runtime_module.score_hybrid_candidates(
        state=state,
        candidates=candidates,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
        scoring_policy={"software_work_phase_gate_weight": 2.0, "software_work_alignment_weight": 2.0},
    )

    assert scored[0]["command"].startswith("git merge --no-ff worker/api-release")
    assert scored[0]["hybrid_software_work_phase_gate_alignment"] > 0.0
    assert scored[0]["hybrid_software_work_gate_objective"] == "accept required branch worker/api-release"
    assert scored[1]["hybrid_software_work_phase_gate_alignment"] < 0.0


def test_rollout_action_value_rewards_long_horizon_progress() -> None:
    class FakeWorldModel:
        def simulate_command_effect(self, world_model_summary, content):
            del world_model_summary, content
            return {
                "predicted_progress_gain": 1.0,
                "predicted_conflicts": [],
                "predicted_preserved": ["keep.txt"],
                "predicted_workflow_paths": [],
            }

    rollout_policy = {
        "predicted_progress_gain_weight": 3.0,
        "predicted_conflict_penalty_weight": 4.0,
        "predicted_preserved_bonus_weight": 1.0,
        "predicted_workflow_bonus_weight": 1.5,
        "latent_progress_bonus_weight": 1.0,
        "latent_risk_penalty_weight": 2.0,
        "recover_from_stall_bonus_weight": 1.5,
        "long_horizon_progress_bonus_weight": 2.0,
        "long_horizon_preserved_bonus_weight": 1.5,
        "long_horizon_risk_penalty_weight": 3.0,
    }

    bounded = rollout_action_value(
        world_model_summary={"horizon": "bounded"},
        latent_state_summary={"progress_band": "flat", "risk_band": "stable"},
        latest_transition={},
        action="code_execute",
        content="printf 'hello\\n' > hello.txt",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )
    long_horizon = rollout_action_value(
        world_model_summary={"horizon": "long_horizon"},
        latent_state_summary={"progress_band": "flat", "risk_band": "stable"},
        latest_transition={},
        action="code_execute",
        content="printf 'hello\\n' > hello.txt",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )

    assert long_horizon > bounded


def test_infer_hybrid_world_signal_includes_decoder_world_feedback(monkeypatch) -> None:
    monkeypatch.setattr(
        tolbert_runtime_module,
        "score_hybrid_candidates",
        lambda **kwargs: [
            {
                "hybrid_world_progress_score": 0.2,
                "hybrid_world_risk_score": 0.1,
                "hybrid_decoder_world_progress_score": 0.8,
                "hybrid_decoder_world_risk_score": 0.4,
                "hybrid_decoder_world_entropy_mean": 0.6,
                "hybrid_transition_progress": 0.3,
                "hybrid_transition_regression": 0.05,
                "hybrid_model_family": "tolbert_ssm_v1",
                "reason": "respond candidate",
                "hybrid_world_prior_backend": "profile_conditioned",
                "hybrid_world_prior_top_state": 0,
                "hybrid_world_prior_top_probability": 0.9,
                "hybrid_world_profile_horizons": [1],
                "hybrid_world_signature_token_count": 4,
                "hybrid_world_transition_family": "banded",
                "hybrid_world_transition_bandwidth": 1,
                "hybrid_world_transition_gate": 0.5,
                "hybrid_world_final_entropy_mean": 0.4,
                "hybrid_ssm_last_state_norm_mean": 0.2,
                "hybrid_ssm_pooled_state_norm_mean": 0.3,
            }
        ],
    )
    state = AgentState(task=TaskBank().get("hello_task"))

    signal = tolbert_runtime_module.infer_hybrid_world_signal(
        state=state,
        bundle_manifest_path=Path("/tmp/fake_bundle.json"),
        device="cpu",
    )

    assert signal["decoder_world_progress_score"] == 0.8
    assert signal["decoder_world_risk_score"] == 0.4
    assert signal["progress_signal"] == 0.8
    assert signal["risk_signal"] == 0.4


def test_policy_surfaces_hybrid_shadow_decision(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_path = save_hybrid_runtime_bundle(
        output_dir=bundle_dir,
        model=HybridTolbertSSMModel(HybridTolbertSSMConfig(hidden_dim=16, d_state=4)),
        config=HybridTolbertSSMConfig(hidden_dim=16, d_state=4),
    )
    artifact_path = tmp_path / "tolbert_model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "model_family": "tolbert_ssm_v1",
                    "shadow_enabled": True,
                    "primary_enabled": False,
                    "bundle_manifest_path": str(bundle_path),
                    "preferred_device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        client=type("Client", (), {})(),
        skill_library=SkillLibrary([]),
        config=KernelConfig(
            use_tolbert_model_artifacts=True,
            tolbert_model_artifact_path=artifact_path,
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )

    shadow = policy._tolbert_shadow_decision(
        state,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": [{"command": "printf 'hello\\n' > hello.txt", "span_id": "span-1"}]},
        blocked_commands=[],
        route_mode="shadow",
    )

    assert shadow["mode"] == "shadow"
    assert shadow["reason"].startswith("hybrid_runtime:")
    assert shadow["model_family"] == "tolbert_ssm_v1"


def test_policy_surfaces_hybrid_shadow_error_reason(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_path = save_hybrid_runtime_bundle(
        output_dir=bundle_dir,
        model=HybridTolbertSSMModel(HybridTolbertSSMConfig(hidden_dim=16, d_state=4)),
        config=HybridTolbertSSMConfig(hidden_dim=16, d_state=4),
    )
    artifact_path = tmp_path / "tolbert_model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "model_family": "tolbert_ssm_v1",
                    "shadow_enabled": True,
                    "primary_enabled": False,
                    "bundle_manifest_path": str(bundle_path),
                    "preferred_device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        client=type("Client", (), {})(),
        skill_library=SkillLibrary([]),
        config=KernelConfig(
            use_tolbert_model_artifacts=True,
            tolbert_model_artifact_path=artifact_path,
        ),
    )
    monkeypatch.setattr("agent_kernel.policy.score_hybrid_candidates", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )

    shadow = policy._tolbert_shadow_decision(
        state,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": [{"command": "printf 'hello\\n' > hello.txt", "span_id": "span-1"}]},
        blocked_commands=[],
        route_mode="shadow",
    )

    assert shadow["reason"].startswith("hybrid_runtime_error:")


def test_policy_surfaces_missing_hybrid_bundle_manifest_reason(tmp_path: Path) -> None:
    artifact_path = tmp_path / "tolbert_model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "model_family": "tolbert_ssm_v1",
                    "shadow_enabled": True,
                    "primary_enabled": False,
                    "bundle_manifest_path": str(tmp_path / "missing" / "bundle.json"),
                    "preferred_device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        client=type("Client", (), {})(),
        skill_library=SkillLibrary([]),
        config=KernelConfig(
            use_tolbert_model_artifacts=True,
            tolbert_model_artifact_path=artifact_path,
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="hello_task",
            prompt="create file",
            workspace_subdir="hello_task",
            suggested_commands=["printf 'hello\\n' > hello.txt"],
            metadata={"benchmark_family": "workflow", "capability": "file_write", "difficulty": "seed"},
        )
    )

    shadow = policy._tolbert_shadow_decision(
        state,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": [{"command": "printf 'hello\\n' > hello.txt", "span_id": "span-1"}]},
        blocked_commands=[],
        route_mode="shadow",
    )

    assert shadow["reason"].startswith("hybrid_runtime_error:")
    assert "bundle manifest does not exist" in shadow["reason"]
