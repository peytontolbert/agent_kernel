from __future__ import annotations

from io import StringIO
from pathlib import Path
import importlib.util
import json
import sys

from evals.metrics import EvalMetrics

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.training_backends import (
    artifact_training_env,
    build_training_backend_launch,
    discover_training_backends,
    qwen_artifact_training_env,
    resolve_training_backend,
    tolbert_artifact_training_env,
)


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_backend_fixture(repo_root: Path) -> Path:
    backend_root = repo_root / "other_repos" / "StateSpace_CausalMachine"
    backend_root.mkdir(parents=True, exist_ok=True)
    (backend_root / "train_gpt.py").write_text("print('train')\n", encoding="utf-8")
    (backend_root / "agentkernel_backend.json").write_text(
        json.dumps(
            {
                "backend_id": "state_space_causal_machine",
                "label": "State Space Causal Machine",
                "backend_kind": "external_model_trainer",
                "trainer_entrypoint": "train_gpt.py",
                "trainer_cwd": ".",
                "default_launch": {
                    "launcher": "torchrun",
                    "launcher_args": ["--standalone", "--nproc_per_node=2"],
                    "env": {
                        "USE_CUDA_GRAPHS": "0",
                        "CAUSAL_MACHINE_PROFILE_JSON": "/tmp/profile.json",
                    },
                },
                "features": {
                    "quantized_export": True,
                    "efficient_training": True,
                    "state_space_manipulation": True,
                },
                "artifacts": {
                    "quantized_model_path": "final_model.int8.ptz",
                },
                "notes": ["test backend"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return backend_root


def test_discover_and_resolve_training_backend(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    backend_root = _write_backend_fixture(repo_root)

    backends = discover_training_backends(repo_root)

    assert len(backends) == 1
    backend = backends[0]
    assert backend["backend_id"] == "state_space_causal_machine"
    assert backend["trainer_exists"] is True
    assert backend["backend_root"] == str(backend_root.resolve())
    resolved = resolve_training_backend(repo_root, "state_space_causal_machine")
    assert resolved["manifest_path"].endswith("agentkernel_backend.json")


def test_build_training_backend_launch_uses_default_launcher_and_env_overrides(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_backend_fixture(repo_root)
    manifest = resolve_training_backend(repo_root, "state_space_causal_machine")

    launch = build_training_backend_launch(
        manifest,
        python_bin="/usr/bin/python3",
        env_overrides={"EXPORT_QUANT_BITS": "5"},
        extra_args=["--seed", "7"],
    )

    assert launch["command"][:4] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(repo_root / "other_repos" / "StateSpace_CausalMachine" / "train_gpt.py"),
    ]
    assert launch["command"][-2:] == ["--seed", "7"]
    assert launch["env"]["USE_CUDA_GRAPHS"] == "0"
    assert launch["env"]["EXPORT_QUANT_BITS"] == "5"
    assert launch["resolved_env"]["CAUSAL_MACHINE_PROFILE_JSON"] == "/tmp/profile.json"


def test_tolbert_artifact_training_env_exports_head_dataset_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training" / "training_inputs_manifest.json"
    payload = {
        "artifact_kind": "tolbert_model_bundle",
        "training_inputs": {
            "training_inputs_manifest_path": str(manifest_path),
            "dataset_manifest_path": str(tmp_path / "dataset" / "dataset_manifest.json"),
            "supervised_examples_path": str(tmp_path / "dataset" / "supervised_examples.jsonl"),
            "policy_examples_path": str(tmp_path / "dataset" / "policy_examples.jsonl"),
            "transition_examples_path": str(tmp_path / "dataset" / "transition_examples.jsonl"),
            "value_examples_path": str(tmp_path / "dataset" / "value_examples.jsonl"),
            "stop_examples_path": str(tmp_path / "dataset" / "stop_examples.jsonl"),
            "enabled_heads": ["policy", "transition", "value", "stop"],
        },
        "runtime_paths": {
            "config_path": str(tmp_path / "training" / "tolbert_train_config.json"),
            "checkpoint_path": str(tmp_path / "training" / "checkpoints" / "tolbert_epoch2.pt"),
            "parent_checkpoint_path": str(tmp_path / "training" / "checkpoints" / "tolbert_parent.pt"),
            "checkpoint_delta_path": str(tmp_path / "training" / "checkpoints" / "tolbert_epoch2__delta.pt"),
        },
    }

    env = tolbert_artifact_training_env(payload)

    assert env["AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH"].endswith("training_inputs_manifest.json")
    assert env["AGENTKERNEL_TOLBERT_POLICY_EXAMPLES_PATH"].endswith("policy_examples.jsonl")
    assert env["AGENTKERNEL_TOLBERT_TRANSITION_EXAMPLES_PATH"].endswith("transition_examples.jsonl")
    assert env["AGENTKERNEL_TOLBERT_VALUE_EXAMPLES_PATH"].endswith("value_examples.jsonl")
    assert env["AGENTKERNEL_TOLBERT_STOP_EXAMPLES_PATH"].endswith("stop_examples.jsonl")
    assert env["AGENTKERNEL_TOLBERT_ENABLED_HEADS"] == "policy,transition,value,stop"
    assert env["AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH"].endswith("tolbert_parent.pt")
    assert env["AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH"].endswith("tolbert_epoch2__delta.pt")


def test_qwen_artifact_training_env_exports_dataset_and_output_paths(tmp_path: Path) -> None:
    payload = {
        "artifact_kind": "qwen_adapter_bundle",
        "base_model_name": "Qwen/Qwen3.5-9B",
        "training_objective": "qlora_sft",
        "supported_benchmark_families": ["repository", "workflow"],
        "training_dataset_manifest": {
            "manifest_path": str(tmp_path / "dataset" / "manifest.json"),
            "train_dataset_path": str(tmp_path / "dataset" / "train.jsonl"),
            "eval_dataset_path": str(tmp_path / "dataset" / "eval.jsonl"),
        },
        "runtime_paths": {
            "adapter_output_dir": str(tmp_path / "adapter"),
            "merged_output_dir": str(tmp_path / "merged"),
            "adapter_manifest_path": str(tmp_path / "adapter" / "artifact.json"),
        },
    }

    env = qwen_artifact_training_env(payload)

    assert env["AGENTKERNEL_QWEN_BASE_MODEL_NAME"] == "Qwen/Qwen3.5-9B"
    assert env["AGENTKERNEL_QWEN_TRAINING_OBJECTIVE"] == "qlora_sft"
    assert env["AGENTKERNEL_QWEN_DATASET_MANIFEST_PATH"].endswith("manifest.json")
    assert env["AGENTKERNEL_QWEN_TRAIN_DATASET_PATH"].endswith("train.jsonl")
    assert env["AGENTKERNEL_QWEN_EVAL_DATASET_PATH"].endswith("eval.jsonl")
    assert env["AGENTKERNEL_QWEN_ADAPTER_OUTPUT_DIR"].endswith("adapter")
    assert env["AGENTKERNEL_QWEN_MERGED_OUTPUT_DIR"].endswith("merged")
    assert env["AGENTKERNEL_QWEN_SUPPORTED_BENCHMARK_FAMILIES"] == "repository,workflow"


def test_artifact_training_env_routes_by_artifact_kind(tmp_path: Path) -> None:
    tolbert_payload = {
        "artifact_kind": "tolbert_model_bundle",
        "training_inputs": {
            "training_inputs_manifest_path": str(tmp_path / "tolbert" / "training_inputs_manifest.json"),
        },
    }
    qwen_payload = {
        "artifact_kind": "qwen_adapter_bundle",
        "base_model_name": "Qwen/Qwen3.5-9B",
    }

    tolbert_env = artifact_training_env(tolbert_payload)
    qwen_env = artifact_training_env(qwen_payload)

    assert tolbert_env["AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH"].endswith("training_inputs_manifest.json")
    assert qwen_env["AGENTKERNEL_QWEN_BASE_MODEL_NAME"] == "Qwen/Qwen3.5-9B"


def test_run_training_backend_script_lists_discovered_backends(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_training_backend.py")
    repo_root = tmp_path / "repo"
    _write_backend_fixture(repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_backend.py",
            "--repo-root",
            str(repo_root),
            "--list",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["repo_root"] == str(repo_root.resolve())
    assert payload["backends"][0]["backend_id"] == "state_space_causal_machine"


def test_run_training_backend_script_dry_run_emits_launch_payload(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_training_backend.py")
    repo_root = tmp_path / "repo"
    _write_backend_fixture(repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_backend.py",
            "--repo-root",
            str(repo_root),
            "--backend",
            "state_space_causal_machine",
            "--env",
            "EXPORT_QUANT_BITS=5",
            "--arg",
            "--seed",
            "--arg",
            "7",
            "--dry-run",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["backend_id"] == "state_space_causal_machine"
    assert payload["dry_run"] is True
    assert payload["env"]["EXPORT_QUANT_BITS"] == "5"
    assert payload["command"][-2:] == ["--seed", "7"]


def test_run_training_backend_script_executes_launch_command(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_training_backend.py")
    repo_root = tmp_path / "repo"
    backend_root = _write_backend_fixture(repo_root)
    seen: dict[str, object] = {}

    def _fake_run(command, cwd=None, env=None, check=None):
        seen["command"] = list(command)
        seen["cwd"] = cwd
        seen["env"] = dict(env)
        seen["check"] = check
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_backend.py",
            "--repo-root",
            str(repo_root),
            "--backend",
            "state_space_causal_machine",
            "--dry-run",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert seen["check"] is True
    assert seen["cwd"] == str(backend_root.resolve())
    assert payload["returncode"] == 0


def test_run_training_backend_script_loads_tolbert_artifact_env(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_training_backend.py")
    repo_root = tmp_path / "repo"
    _write_backend_fixture(repo_root)
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "training_inputs": {
                    "training_inputs_manifest_path": str(tmp_path / "training" / "training_inputs_manifest.json"),
                    "policy_examples_path": str(tmp_path / "dataset" / "policy_examples.jsonl"),
                    "transition_examples_path": str(tmp_path / "dataset" / "transition_examples.jsonl"),
                    "value_examples_path": str(tmp_path / "dataset" / "value_examples.jsonl"),
                    "stop_examples_path": str(tmp_path / "dataset" / "stop_examples.jsonl"),
                    "enabled_heads": ["policy", "transition", "value", "stop"],
                },
                "runtime_paths": {
                    "config_path": str(tmp_path / "training" / "tolbert_train_config.json"),
                    "checkpoint_path": str(tmp_path / "training" / "checkpoints" / "tolbert_epoch2.pt"),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_backend.py",
            "--repo-root",
            str(repo_root),
            "--backend",
            "state_space_causal_machine",
            "--tolbert-artifact",
            str(artifact_path),
            "--dry-run",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["env"]["AGENTKERNEL_TOLBERT_POLICY_EXAMPLES_PATH"].endswith("policy_examples.jsonl")
    assert payload["env"]["AGENTKERNEL_TOLBERT_ENABLED_HEADS"] == "policy,transition,value,stop"


def test_run_training_backend_script_loads_generic_qwen_artifact_env(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_training_backend.py")
    repo_root = tmp_path / "repo"
    _write_backend_fixture(repo_root)
    artifact_path = tmp_path / "qwen_adapter" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen_adapter_bundle",
                "base_model_name": "Qwen/Qwen3.5-9B",
                "training_objective": "qlora_sft",
                "training_dataset_manifest": {
                    "manifest_path": str(tmp_path / "dataset" / "manifest.json"),
                    "train_dataset_path": str(tmp_path / "dataset" / "train.jsonl"),
                    "eval_dataset_path": str(tmp_path / "dataset" / "eval.jsonl"),
                },
                "runtime_paths": {
                    "adapter_output_dir": str(tmp_path / "adapter"),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_backend.py",
            "--repo-root",
            str(repo_root),
            "--backend",
            "state_space_causal_machine",
            "--artifact",
            str(artifact_path),
            "--dry-run",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(stream.getvalue())
    assert payload["env"]["AGENTKERNEL_QWEN_BASE_MODEL_NAME"] == "Qwen/Qwen3.5-9B"
    assert payload["env"]["AGENTKERNEL_QWEN_TRAINING_OBJECTIVE"] == "qlora_sft"
    assert payload["env"]["AGENTKERNEL_QWEN_ADAPTER_OUTPUT_DIR"].endswith("adapter")


def test_tolbert_model_candidate_artifact_exposes_external_training_backends(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel import tolbert_model_improvement as module

    repo_root = Path(__file__).resolve().parents[1]
    _write_backend_fixture(repo_root)
    output_dir = tmp_path / "candidate"
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.trajectories_root.mkdir(parents=True, exist_ok=True)
    assets_dir = tmp_path / "assets"
    training_dir = output_dir / "training"
    assets_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    config_path = assets_dir / "tolbert_config.json"
    config_path.write_text(json.dumps({"base_model_name": "bert-base-uncased"}), encoding="utf-8")
    nodes_path = assets_dir / "nodes.jsonl"
    nodes_path.write_text("", encoding="utf-8")
    label_map_path = assets_dir / "label_map.json"
    label_map_path.write_text("{}", encoding="utf-8")
    source_spans_path = assets_dir / "source_spans.jsonl"
    source_spans_path.write_text("", encoding="utf-8")
    model_spans_path = assets_dir / "model_spans.jsonl"
    model_spans_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "build_agentkernel_tolbert_assets",
        lambda **kwargs: type(
            "Assets",
            (),
            {
                "config_path": config_path,
                "nodes_path": nodes_path,
                "label_map_path": label_map_path,
                "source_spans_path": source_spans_path,
                "model_spans_path": model_spans_path,
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "build_tolbert_supervised_dataset_manifest",
        lambda **kwargs: {"total_examples": 1, "benchmark_families": ["repo_sandbox"]},
    )
    monkeypatch.setattr(
        module,
        "materialize_universal_decoder_dataset",
        lambda **kwargs: {
            "artifact_kind": "tolbert_universal_decoder_dataset",
            "manifest_path": str(output_dir / "universal_dataset" / "manifest.json"),
            "train_dataset_path": str(output_dir / "universal_dataset" / "train.jsonl"),
            "eval_dataset_path": str(output_dir / "universal_dataset" / "eval.jsonl"),
            "decoder_vocab_path": str(output_dir / "universal_dataset" / "decoder_vocab.json"),
            "decoder_tokenizer_manifest_path": str(output_dir / "universal_dataset" / "decoder_tokenizer_manifest.json"),
            "decoder_vocab_size": 320,
            "decoder_tokenizer_stats": {"tokenizer_kind": "regex_v1", "unique_token_count": 280},
            "train_examples": 1,
            "eval_examples": 1,
        },
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_finetune_pipeline",
        lambda **kwargs: (
            training_dir / "checkpoint.pt",
            output_dir / "retrieval_cache" / "cache.pt",
            [{"job_id": "train"}],
        ),
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_hybrid_runtime_pipeline",
        lambda **kwargs: (None, [{"job_id": "hybrid", "state": "safe_stop", "outcome": "safe_stop"}]),
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_universal_decoder_pipeline",
        lambda **kwargs: (None, [{"job_id": "universal", "state": "safe_stop", "outcome": "safe_stop"}]),
    )

    artifact = module.build_tolbert_model_candidate_artifact(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        metrics=EvalMetrics(total=10, passed=8),
    )

    assert artifact["artifact_kind"] == "tolbert_model_bundle"
    assert artifact["training_inputs"]["training_inputs_manifest_path"].endswith("training_inputs_manifest.json")
    assert any(
        backend["backend_id"] == "state_space_causal_machine"
        for backend in artifact["external_training_backends"]
    )
