from __future__ import annotations

from pathlib import Path
import json
import os
import threading
import time
from types import SimpleNamespace

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import retained_tolbert_hybrid_runtime
from evals.metrics import EvalMetrics


def test_retained_tolbert_hybrid_runtime_uses_defaults_without_payload() -> None:
    runtime = retained_tolbert_hybrid_runtime({})

    assert runtime["model_family"] == "tolbert_ssm_v1"
    assert runtime["shadow_enabled"] is False
    assert runtime["supports_encoder_surface"] is True
    assert runtime["supports_decoder_surface"] is True
    assert runtime["supports_policy_head"] is True


def test_tolbert_model_candidate_artifact_includes_hybrid_runtime(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        tolbert_model_artifact_path=tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json",
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
        lambda **kwargs: {
            "total_examples": 10,
            "synthetic_trajectory_examples": 2,
            "policy_examples": 4,
            "transition_examples": 4,
            "value_examples": 4,
            "stop_examples": 2,
            "benchmark_families": ["micro", "repository"],
            "action_generation_summary": {
                "example_count": 4,
                "positive_example_count": 2,
                "template_preferences": {
                    "repository": [
                        {
                            "template_kind": "structured_edit",
                            "support": 2,
                            "success_count": 1,
                            "pass_rate": 0.5,
                            "provenance": ["repository_audit_packet_task"],
                        }
                    ]
                },
            },
        },
    )
    monkeypatch.setattr(
        module,
        "materialize_universal_decoder_dataset",
        lambda **kwargs: {
            "artifact_kind": "tolbert_universal_decoder_dataset",
            "train_dataset_path": str(output_dir / "universal_dataset" / "train.jsonl"),
            "eval_dataset_path": str(output_dir / "universal_dataset" / "eval.jsonl"),
            "decoder_vocab_path": str(output_dir / "universal_dataset" / "decoder_vocab.json"),
            "decoder_tokenizer_manifest_path": str(output_dir / "universal_dataset" / "decoder_tokenizer_manifest.json"),
            "decoder_vocab_size": 768,
            "decoder_tokenizer_stats": {"tokenizer_kind": "regex_v1", "unique_token_count": 704},
            "total_examples": 20,
            "train_examples": 18,
            "eval_examples": 2,
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
        lambda **kwargs: (
            {
                "artifact_kind": "tolbert_hybrid_runtime_bundle",
                "model_family": "tolbert_ssm_v1",
                "config_path": str(output_dir / "hybrid_runtime" / "hybrid_config.json"),
                "checkpoint_path": str(output_dir / "hybrid_runtime" / "hybrid_checkpoint.pt"),
                "relative_config_path": "hybrid_config.json",
                "relative_checkpoint_path": "hybrid_checkpoint.pt",
            },
            [{"job_id": "hybrid"}],
        ),
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_universal_decoder_pipeline",
        lambda **kwargs: (
            {
                "artifact_kind": "tolbert_hybrid_runtime_bundle",
                "model_family": "tolbert_ssm_v1",
                "config_path": str(output_dir / "universal_runtime" / "hybrid_config.json"),
                "checkpoint_path": str(output_dir / "universal_runtime" / "hybrid_checkpoint.pt"),
                "relative_config_path": "hybrid_config.json",
                "relative_checkpoint_path": "hybrid_checkpoint.pt",
            },
            [{"job_id": "universal"}],
        ),
    )

    artifact = module.build_tolbert_model_candidate_artifact(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=output_dir,
        metrics=EvalMetrics(total=10, passed=8),
    )

    assert artifact["model_surfaces"]["encoder_surface"] is True
    assert artifact["model_surfaces"]["decoder_surface"] is True
    assert artifact["model_surfaces"]["universal_runtime"] is True
    assert artifact["action_generation_policy"]["enabled"] is True
    assert artifact["retention_gate"]["require_novel_command_signal"] is True
    assert artifact["retention_gate"]["allow_selection_signal_fallback"] is True
    assert artifact["retention_gate"]["require_primary_routing_signal"] is True
    assert artifact["retention_gate"]["min_novel_valid_command_steps"] == 1
    assert artifact["retention_gate"]["proposal_gate_by_benchmark_family"]["micro"]["require_novel_command_signal"] is False
    assert artifact["retention_gate"]["proposal_gate_by_benchmark_family"]["repository"]["require_novel_command_signal"] is True
    assert artifact["retention_gate"]["proposal_gate_by_benchmark_family"]["repository"]["allow_primary_routing_signal"] is True
    assert artifact["retention_gate"]["proposal_gate_by_benchmark_family"]["repository"]["min_proposal_selected_steps_delta"] == 1
    assert artifact["liftoff_gate"]["require_family_novel_command_evidence"] is True
    assert artifact["liftoff_gate"]["require_primary_routing_signal"] is True
    assert artifact["liftoff_gate"]["proposal_gate_by_benchmark_family"]["repository"]["min_proposal_selected_steps_delta"] == 1
    assert artifact["runtime_policy"]["allow_trusted_primary_without_min_confidence"] is True
    assert artifact["runtime_policy"]["trusted_primary_min_confidence"] <= artifact["runtime_policy"]["min_path_confidence"]
    assert artifact["hybrid_runtime"]["model_family"] == "tolbert_ssm_v1"
    assert artifact["hybrid_runtime"]["shadow_enabled"] is True
    assert artifact["hybrid_runtime"]["primary_enabled"] is True
    assert artifact["hybrid_runtime"]["supports_universal_runtime"] is True
    assert artifact["universal_dataset_manifest"]["artifact_kind"] == "tolbert_universal_decoder_dataset"
    assert artifact["universal_decoder_runtime"]["materialized"] is True
    assert artifact["universal_decoder_training_controls"]["decoder_vocab_size"] >= 256
    assert artifact["runtime_paths"]["universal_train_dataset_path"].endswith("train.jsonl")
    assert artifact["runtime_paths"]["universal_bundle_manifest_path"].endswith("hybrid_bundle_manifest.json")
    assert artifact["runtime_paths"]["hybrid_bundle_manifest_path"].endswith("hybrid_bundle_manifest.json")
    assert artifact["runtime_paths"]["hybrid_checkpoint_path"].endswith("hybrid_checkpoint.pt")
    assert artifact["job_records"][-1]["job_id"] == "universal"
    assert artifact["storage_compaction"]["retained_checkpoint_path"].endswith("checkpoint.pt")
    assert artifact["shared_store"]["mode"] == "content_addressed_shared_store"
    assert artifact["lineage"]["mode"] == "canonical_parent_mutation"
    assert artifact["lineage"]["candidate_artifact_strategy"] == "manifest_first_shared_store"
    assert isinstance(artifact["lineage"]["mutation_manifest"], dict)


def test_tolbert_model_candidate_artifact_runs_independent_generation_pipelines_in_parallel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        tolbert_model_artifact_path=tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json",
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
        lambda **kwargs: {
            "total_examples": 10,
            "synthetic_trajectory_examples": 2,
            "policy_examples": 4,
            "transition_examples": 4,
            "value_examples": 4,
            "stop_examples": 2,
            "benchmark_families": ["micro", "repository"],
        },
    )
    monkeypatch.setattr(
        module,
        "materialize_universal_decoder_dataset",
        lambda **kwargs: {
            "artifact_kind": "tolbert_universal_decoder_dataset",
            "train_dataset_path": str(output_dir / "universal_dataset" / "train.jsonl"),
            "eval_dataset_path": str(output_dir / "universal_dataset" / "eval.jsonl"),
            "decoder_vocab_path": str(output_dir / "universal_dataset" / "decoder_vocab.json"),
            "decoder_tokenizer_manifest_path": str(output_dir / "universal_dataset" / "decoder_tokenizer_manifest.json"),
            "decoder_vocab_size": 768,
            "decoder_tokenizer_stats": {"tokenizer_kind": "regex_v1", "unique_token_count": 704},
            "total_examples": 20,
            "train_examples": 18,
            "eval_examples": 2,
        },
    )

    active = 0
    max_active = 0
    lock = threading.Lock()

    def _record_parallelism() -> None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    monkeypatch.setattr(
        module,
        "run_tolbert_finetune_pipeline",
        lambda **kwargs: (
            _record_parallelism(),
            (
                training_dir / "checkpoint.pt",
                output_dir / "retrieval_cache" / "cache.pt",
                [{"job_id": "train"}],
            ),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_hybrid_runtime_pipeline",
        lambda **kwargs: (
            _record_parallelism(),
            (
                {
                    "artifact_kind": "tolbert_hybrid_runtime_bundle",
                    "model_family": "tolbert_ssm_v1",
                    "config_path": str(output_dir / "hybrid_runtime" / "hybrid_config.json"),
                    "checkpoint_path": str(output_dir / "hybrid_runtime" / "hybrid_checkpoint.pt"),
                    "relative_config_path": "hybrid_config.json",
                    "relative_checkpoint_path": "hybrid_checkpoint.pt",
                },
                [{"job_id": "hybrid"}],
            ),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_universal_decoder_pipeline",
        lambda **kwargs: (
            _record_parallelism(),
            (
                {
                    "artifact_kind": "tolbert_hybrid_runtime_bundle",
                    "model_family": "tolbert_ssm_v1",
                    "config_path": str(output_dir / "universal_runtime" / "hybrid_config.json"),
                    "checkpoint_path": str(output_dir / "universal_runtime" / "hybrid_checkpoint.pt"),
                    "relative_config_path": "hybrid_config.json",
                    "relative_checkpoint_path": "hybrid_checkpoint.pt",
                },
                [{"job_id": "universal"}],
            ),
        )[1],
    )

    artifact = module.build_tolbert_model_candidate_artifact(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=output_dir,
        metrics=EvalMetrics(total=10, passed=8),
    )

    assert max_active >= 2
    assert [record["job_id"] for record in artifact["job_records"]] == ["train", "hybrid", "universal"]


def test_tolbert_build_policy_requires_long_horizon_head_coverage_when_present() -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    insufficient = module._tolbert_build_policy(
        {
            "total_examples": 1024,
            "synthetic_trajectory_examples": 128,
            "policy_examples": 512,
            "transition_examples": 512,
            "value_examples": 512,
            "stop_examples": 256,
            "long_horizon_trajectory_examples": 6,
            "long_horizon_policy_examples": 2,
            "long_horizon_transition_examples": 2,
            "long_horizon_value_examples": 2,
            "long_horizon_stop_examples": 0,
        },
        current_payload=None,
    )
    sufficient = module._tolbert_build_policy(
        {
            "total_examples": 1024,
            "synthetic_trajectory_examples": 128,
            "policy_examples": 512,
            "transition_examples": 512,
            "value_examples": 512,
            "stop_examples": 256,
            "long_horizon_trajectory_examples": 6,
            "long_horizon_policy_examples": 6,
            "long_horizon_transition_examples": 6,
            "long_horizon_value_examples": 6,
            "long_horizon_stop_examples": 2,
        },
        current_payload=None,
    )

    assert insufficient["allow_kernel_autobuild"] is False
    assert insufficient["ready_long_horizon_policy_examples"] == 2
    assert insufficient["min_long_horizon_policy_examples"] == 4
    assert sufficient["allow_kernel_autobuild"] is True


def test_tolbert_generic_only_action_generation_surface_disables_primary_and_novel_gate() -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    dataset_manifest = {
        "policy_examples": 4,
        "transition_examples": 4,
        "value_examples": 4,
        "stop_examples": 4,
        "benchmark_families": ["project", "repository"],
        "action_generation_summary": {
            "example_count": 8,
            "positive_example_count": 0,
            "template_preferences": {
                "project": [
                    {
                        "template_kind": "generic_command",
                        "support": 4,
                        "success_count": 0,
                        "pass_rate": 0.0,
                        "provenance": ["deployment_manifest_task"],
                    }
                ]
            },
        },
    }

    policy = module._tolbert_action_generation_policy(dataset_manifest, current_payload=None)
    gate = module._tolbert_retention_gate(dataset_manifest, current_payload=None)
    runtime_policy = module._tolbert_runtime_policy(dataset_manifest, focus=None, current_payload=None)
    hybrid_runtime = module._tolbert_hybrid_runtime(
        output_dir=Path("/tmp/tolbert_generic_only"),
        dataset_manifest=dataset_manifest,
        manifest={"model_family": "tolbert_ssm_v1"},
        focus=None,
    )

    assert policy["template_preferences"] == {}
    assert gate["require_novel_command_signal"] is False
    assert gate["min_novel_valid_command_steps"] == 0
    assert gate["proposal_gate_by_benchmark_family"]["project"]["require_novel_command_signal"] is False
    assert runtime_policy["primary_benchmark_families"] == []
    assert hybrid_runtime["primary_enabled"] is False


def test_compact_tolbert_model_candidate_output_keeps_final_checkpoint(tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    checkpoints_dir = output_dir / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    epoch1 = checkpoints_dir / "tolbert_epoch1.pt"
    epoch2 = checkpoints_dir / "tolbert_epoch2.pt"
    epoch1.write_bytes(b"epoch1")
    epoch2.write_bytes(b"epoch2")
    for relative in (
        "jobs",
        "hybrid_jobs",
        "job_workspace",
        "hybrid_job_workspace",
        "job_reports",
        "hybrid_job_reports",
        "job_checkpoints",
        "hybrid_job_checkpoints",
        "universal_job_workspace",
        "universal_job_reports",
        "universal_job_checkpoints",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
        ((output_dir / relative) / "marker.txt").write_text("scratch", encoding="utf-8")

    compaction = module._compact_tolbert_model_candidate_output(
        output_dir,
        retained_checkpoint_path=epoch2,
    )

    assert not epoch1.exists()
    assert epoch2.exists()
    assert compaction["retained_checkpoint_path"] == str(epoch2)
    assert str(epoch1) in compaction["removed_paths"]
    assert not (output_dir / "jobs").exists()
    assert not (output_dir / "hybrid_jobs").exists()
    assert not (output_dir / "job_workspace").exists()
    assert not (output_dir / "hybrid_job_workspace").exists()
    assert not (output_dir / "universal_job_reports").exists()


def test_materialize_tolbert_model_shared_store_dedupes_normalized_candidate_paths(tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    config = KernelConfig(
        tolbert_model_artifact_path=tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json",
    )
    config.tolbert_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_candidate(root: Path) -> tuple[Path, dict[str, object]]:
        dataset_dir = root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        train_path = dataset_dir / "supervised_examples.jsonl"
        train_path.write_text('{"example": 1}\n', encoding="utf-8")
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest_payload = {
            "artifact_kind": "tolbert_supervised_dataset",
            "manifest_path": str(manifest_path),
            "supervised_examples_path": str(train_path),
            "total_examples": 1,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        payload = {
            "dataset_manifest": dict(manifest_payload),
            "runtime_paths": {"checkpoint_path": str(root / "training" / "checkpoints" / "tolbert_epoch1.pt")},
        }
        return dataset_dir, payload

    output_a = tmp_path / "candidate_a"
    _, payload_a = _make_candidate(output_a)
    shared_a = module._materialize_tolbert_model_shared_store(
        config=config,
        output_dir=output_a,
        payload=payload_a,
    )

    output_b = tmp_path / "candidate_b"
    _, payload_b = _make_candidate(output_b)
    shared_b = module._materialize_tolbert_model_shared_store(
        config=config,
        output_dir=output_b,
        payload=payload_b,
    )

    dataset_store_path_a = shared_a["entries"]["dataset"]["path"]
    dataset_store_path_b = shared_b["entries"]["dataset"]["path"]
    assert dataset_store_path_a == dataset_store_path_b
    assert payload_a["dataset_manifest"]["manifest_path"] == str(Path(dataset_store_path_a) / "dataset_manifest.json")
    assert payload_b["dataset_manifest"]["manifest_path"] == str(Path(dataset_store_path_b) / "dataset_manifest.json")
    stored_manifest = json.loads((Path(dataset_store_path_a) / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest["manifest_path"] == str(Path(dataset_store_path_a) / "dataset_manifest.json")
    assert stored_manifest["supervised_examples_path"] == str(Path(dataset_store_path_a) / "supervised_examples.jsonl")
    assert not (output_a / "dataset").exists()
    assert not (output_b / "dataset").exists()


def test_cleanup_tolbert_model_candidate_storage_prunes_old_dirs_and_unreferenced_store(tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    config = KernelConfig(
        candidate_artifacts_root=tmp_path / "candidates",
        tolbert_model_artifact_path=tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json",
        storage_keep_tolbert_candidate_dirs=0,
        storage_tolbert_candidate_budget_bytes=4096,
        storage_tolbert_shared_store_budget_bytes=1024,
    )
    candidates_root = config.candidate_artifacts_root / "tolbert_model"
    live_store = config.tolbert_model_artifact_path.parent / "store" / "training" / "live_keep"
    old_store = config.tolbert_model_artifact_path.parent / "store" / "training" / "old_drop"
    middle_store = config.tolbert_model_artifact_path.parent / "store" / "training" / "middle_drop"
    current_store = config.tolbert_model_artifact_path.parent / "store" / "training" / "current_keep"
    for path in (live_store, old_store, middle_store, current_store):
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoint.pt").write_bytes(b"x" * 64)

    config.tolbert_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    config.tolbert_model_artifact_path.write_text(
        json.dumps({"shared_store": {"entries": {"training": {"path": str(live_store)}}}}),
        encoding="utf-8",
    )

    cycle_old = candidates_root / "cycle_old"
    cycle_middle = candidates_root / "cycle_middle"
    cycle_current = candidates_root / "cycle_current"
    for index, (cycle_root, store_path) in enumerate(
        ((cycle_old, old_store), (cycle_middle, middle_store), (cycle_current, current_store)),
        start=1,
    ):
        cycle_root.mkdir(parents=True, exist_ok=True)
        (cycle_root / "tolbert_model_artifact.json").write_text(
            json.dumps({"shared_store": {"entries": {"training": {"path": str(store_path)}}}}),
            encoding="utf-8",
        )
        (cycle_root / "padding.bin").write_bytes(b"p" * 96)
        os.utime(cycle_root, (100 + index, 100 + index))

    cleanup = module.cleanup_tolbert_model_candidate_storage(
        config=config,
        preserve_paths=(cycle_current,),
    )

    assert not cycle_old.exists()
    assert not cycle_middle.exists()
    assert cycle_current.exists()
    assert live_store.exists()
    assert current_store.exists()
    assert not old_store.exists()
    assert not middle_store.exists()
    assert cleanup["candidate_budget_satisfied"] is True
    assert cleanup["shared_store_budget_satisfied"] is True
    assert str(cycle_old) in cleanup["removed_candidate_dirs"]
    assert str(cycle_middle) in cleanup["removed_candidate_dirs"]
    assert str(old_store) in cleanup["removed_shared_store"]
    assert str(middle_store) in cleanup["removed_shared_store"]


def test_tolbert_model_candidate_artifact_stores_parameter_delta_when_parent_checkpoint_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    trained_checkpoint = training_dir / "checkpoint.pt"
    trained_checkpoint.write_bytes(b"child")
    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"parent")
    delta_checkpoint = training_dir / "checkpoint__delta.pt"

    monkeypatch.setattr(
        module,
        "build_agentkernel_tolbert_assets",
        lambda **kwargs: SimpleNamespace(
            config_path=tmp_path / "config.json",
            nodes_path=tmp_path / "nodes.jsonl",
            label_map_path=tmp_path / "label_map.json",
            source_spans_path=tmp_path / "spans.jsonl",
            model_spans_path=tmp_path / "model_spans.jsonl",
        ),
    )
    for path in (tmp_path / "config.json", tmp_path / "nodes.jsonl", tmp_path / "label_map.json", tmp_path / "spans.jsonl", tmp_path / "model_spans.jsonl"):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "build_tolbert_supervised_dataset_manifest", lambda **kwargs: {"total_examples": 4})
    monkeypatch.setattr(
        module,
        "materialize_universal_decoder_dataset",
        lambda **kwargs: {"artifact_kind": "tolbert_universal_decoder_dataset", "train_dataset_path": str(output_dir / "universal_dataset" / "train.jsonl")},
    )
    monkeypatch.setattr(
        module,
        "run_tolbert_finetune_pipeline",
        lambda **kwargs: (trained_checkpoint, output_dir / "retrieval_cache" / "cache.pt", []),
    )
    monkeypatch.setattr(module, "run_tolbert_hybrid_runtime_pipeline", lambda **kwargs: (None, []))
    monkeypatch.setattr(module, "run_tolbert_universal_decoder_pipeline", lambda **kwargs: (None, []))
    monkeypatch.setattr(module, "_tolbert_hybrid_runtime", lambda **kwargs: {})
    monkeypatch.setattr(module, "_tolbert_hybrid_runtime_paths", lambda **kwargs: {})
    monkeypatch.setattr(module, "_tolbert_universal_decoder_runtime", lambda **kwargs: {})
    monkeypatch.setattr(module, "_tolbert_universal_decoder_runtime_paths", lambda **kwargs: {})

    def _create_delta(**kwargs):
        delta_checkpoint.write_bytes(b"delta")
        return {
            "artifact_kind": "tolbert_checkpoint_delta",
            "delta_checkpoint_path": str(delta_checkpoint),
            "parent_checkpoint_path": str(parent_checkpoint),
            "stats": {"changed_key_count": 1},
        }

    monkeypatch.setattr(
        module,
        "create_tolbert_checkpoint_delta",
        _create_delta,
    )

    artifact = module.build_tolbert_model_candidate_artifact(
        config=KernelConfig(trajectories_root=tmp_path / "episodes"),
        repo_root=tmp_path,
        output_dir=output_dir,
        metrics=EvalMetrics(total=10, passed=8),
        current_payload={"runtime_paths": {"checkpoint_path": str(parent_checkpoint)}},
    )

    assert artifact["runtime_paths"]["checkpoint_path"] == ""
    assert artifact["runtime_paths"]["checkpoint_delta_path"].endswith("checkpoint__delta.pt")
    assert artifact["runtime_paths"]["parent_checkpoint_path"] == str(parent_checkpoint)
    assert artifact["parameter_delta"]["stats"]["changed_key_count"] == 1


def test_build_runtime_bundle_checkpoint_delta_rewrites_manifest(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    bundle_dir = tmp_path / "hybrid_runtime"
    manifest_path = bundle_dir / "hybrid_bundle_manifest.json"
    checkpoint_path = bundle_dir / "hybrid_checkpoint.pt"
    parent_checkpoint = tmp_path / "parent_hybrid.pt"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"child")
    parent_checkpoint.write_bytes(b"parent")
    manifest = {
        "artifact_kind": "tolbert_hybrid_runtime_bundle",
        "model_family": "tolbert_ssm_v1",
        "checkpoint_path": str(checkpoint_path),
        "relative_checkpoint_path": checkpoint_path.name,
        "config_path": str(bundle_dir / "hybrid_config.json"),
        "relative_config_path": "hybrid_config.json",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    delta_path = bundle_dir / "hybrid_checkpoint__delta.pt"

    def _create_delta(**kwargs):
        delta_path.write_bytes(b"delta")
        return {
            "artifact_kind": "tolbert_checkpoint_delta",
            "delta_checkpoint_path": str(delta_path),
            "parent_checkpoint_path": str(parent_checkpoint),
            "stats": {"adapter_key_count": 1},
        }

    monkeypatch.setattr(module, "create_tolbert_checkpoint_delta", _create_delta)

    metadata = module._build_runtime_bundle_checkpoint_delta(
        current_payload={"hybrid_runtime": {"checkpoint_path": str(parent_checkpoint)}},
        current_runtime_key="hybrid_runtime",
        bundle_dir=bundle_dir,
        manifest=manifest,
    )

    rewritten = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata["stats"]["adapter_key_count"] == 1
    assert rewritten["checkpoint_path"] == ""
    assert rewritten["checkpoint_delta_path"] == "hybrid_checkpoint__delta.pt"
    assert rewritten["parent_checkpoint_path"] == str(parent_checkpoint)
    assert not checkpoint_path.exists()


def test_run_tolbert_finetune_pipeline_prefers_manifest_backed_cache(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    train_config_path = output_dir / "training" / "config.json"
    model_spans_path = output_dir / "dataset" / "model_spans.jsonl"
    train_config_path.parent.mkdir(parents=True, exist_ok=True)
    model_spans_path.parent.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text("{}", encoding="utf-8")
    model_spans_path.write_text("", encoding="utf-8")
    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime.json",
        tolbert_cache_shard_size=1234,
    )

    fake_queue = None

    class FakeQueue:
        def __init__(self, path: Path) -> None:
            nonlocal fake_queue
            self.path = path
            self.jobs: dict[str, SimpleNamespace] = {}
            fake_queue = self

        def enqueue(self, *, task_id: str, budget_group: str, runtime_overrides: dict[str, object]):
            job = SimpleNamespace(
                job_id=task_id,
                state="queued",
                outcome="pending",
                runtime_overrides=runtime_overrides,
                budget_group=budget_group,
            )
            self.jobs[job.job_id] = job
            return job

        def get(self, job_id: str):
            return self.jobs.get(job_id)

    def _fake_drain(queue, **kwargs):
        del kwargs
        for job in queue.jobs.values():
            job.state = "completed"
            job.outcome = "success"
        return list(queue.jobs.values())

    monkeypatch.setattr(module, "DelegatedJobQueue", FakeQueue)
    monkeypatch.setattr(module, "drain_delegated_jobs", _fake_drain)

    checkpoint_path, cache_artifact_path, job_records = module.run_tolbert_finetune_pipeline(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        train_config_path=train_config_path,
        model_spans_path=model_spans_path,
        num_epochs=2,
        training_inputs={},
    )

    assert checkpoint_path == output_dir / "training" / "checkpoints" / "tolbert_epoch2.pt"
    assert cache_artifact_path == output_dir / "retrieval_cache" / "tolbert_epoch2.json"
    assert fake_queue is not None
    train_success_command = fake_queue.jobs["tolbert_model_train"].runtime_overrides["task_payload"]["success_command"]
    train_command = fake_queue.jobs["tolbert_model_train"].runtime_overrides["worker_command"]
    cache_command = fake_queue.jobs["tolbert_model_cache"].runtime_overrides["worker_command"]
    train_timeout_seconds = fake_queue.jobs["tolbert_model_train"].runtime_overrides["worker_timeout_seconds"]
    assert "--shard-size" in cache_command
    assert "1234" in cache_command
    success_command = fake_queue.jobs["tolbert_model_cache"].runtime_overrides["task_payload"]["success_command"]
    assert train_success_command == f"test -f {module.sh_quote(checkpoint_path.resolve())}"
    assert train_command[-1] == str(config.tolbert_device)
    assert train_timeout_seconds == 300
    assert success_command == f"test -f {module.sh_quote(cache_artifact_path.resolve())}"
    assert [record["job_id"] for record in job_records] == ["tolbert_model_train", "tolbert_model_cache"]


def test_run_tolbert_finetune_pipeline_uses_parent_plus_delta_training(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    output_dir = tmp_path / "candidate"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    train_config_path = output_dir / "training" / "config.json"
    model_spans_path = output_dir / "dataset" / "model_spans.jsonl"
    parent_checkpoint_path = tmp_path / "live" / "tolbert.pt"
    artifact_path = tmp_path / "live" / "tolbert_model_artifact.json"
    train_config_path.parent.mkdir(parents=True, exist_ok=True)
    model_spans_path.parent.mkdir(parents=True, exist_ok=True)
    parent_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text("{}", encoding="utf-8")
    model_spans_path.write_text("", encoding="utf-8")
    parent_checkpoint_path.write_bytes(b"parent")
    artifact_path.write_text(json.dumps({"artifact_kind": "tolbert_model_bundle"}), encoding="utf-8")
    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime.json",
        tolbert_model_artifact_path=artifact_path,
    )

    fake_queue = None

    class FakeQueue:
        def __init__(self, path: Path) -> None:
            nonlocal fake_queue
            self.path = path
            self.jobs: dict[str, SimpleNamespace] = {}
            fake_queue = self

        def enqueue(self, *, task_id: str, budget_group: str, runtime_overrides: dict[str, object]):
            job = SimpleNamespace(
                job_id=task_id,
                state="queued",
                outcome="pending",
                runtime_overrides=runtime_overrides,
                budget_group=budget_group,
            )
            self.jobs[job.job_id] = job
            return job

        def get(self, job_id: str):
            return self.jobs.get(job_id)

    def _fake_drain(queue, **kwargs):
        del kwargs
        for job in queue.jobs.values():
            job.state = "completed"
            job.outcome = "success"
        return list(queue.jobs.values())

    monkeypatch.setattr(module, "DelegatedJobQueue", FakeQueue)
    monkeypatch.setattr(module, "drain_delegated_jobs", _fake_drain)

    checkpoint_path, cache_artifact_path, job_records = module.run_tolbert_finetune_pipeline(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        train_config_path=train_config_path,
        model_spans_path=model_spans_path,
        num_epochs=2,
        training_inputs={},
        current_payload={"runtime_paths": {"checkpoint_path": str(parent_checkpoint_path)}},
    )

    assert checkpoint_path == output_dir / "training" / "checkpoints" / "tolbert_epoch2.pt"
    expected_cache_artifact_path = (
        output_dir / "retrieval_cache" / "tolbert_epoch2.json"
        if int(config.tolbert_cache_shard_size) > 0
        else output_dir / "retrieval_cache" / "tolbert_epoch2.pt"
    )
    assert cache_artifact_path == expected_cache_artifact_path
    assert fake_queue is not None
    delta_checkpoint_path = checkpoint_path.with_name("tolbert_epoch2__delta.pt")
    train_success_command = fake_queue.jobs["tolbert_model_train"].runtime_overrides["task_payload"]["success_command"]
    train_env = fake_queue.jobs["tolbert_model_train"].runtime_overrides["worker_env"]
    train_timeout_seconds = fake_queue.jobs["tolbert_model_train"].runtime_overrides["worker_timeout_seconds"]
    cache_command = fake_queue.jobs["tolbert_model_cache"].runtime_overrides["worker_command"]
    assert train_success_command == f"test -f {module.sh_quote(delta_checkpoint_path.resolve())}"
    assert train_timeout_seconds == 300
    assert train_env["AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH"] == str(parent_checkpoint_path)
    assert train_env["AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH"] == str(delta_checkpoint_path)
    assert "--parent-checkpoint" in cache_command
    assert "--checkpoint-delta" in cache_command
    assert str(parent_checkpoint_path) in cache_command
    assert str(delta_checkpoint_path) in cache_command
    assert [record["job_id"] for record in job_records] == ["tolbert_model_train", "tolbert_model_cache"]


def test_tolbert_finetune_timeout_scales_with_epochs() -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    config = KernelConfig(command_timeout_seconds=20)

    assert module._tolbert_finetune_timeout_seconds(config=config, num_epochs=1) == 180
    assert module._tolbert_finetune_timeout_seconds(config=config, num_epochs=2) == 300
    assert module._tolbert_finetune_timeout_seconds(config=config, num_epochs=4) == 540


def test_tolbert_runtime_delegated_jobs_verify_absolute_manifest_paths(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime.json",
    )
    queued: list[SimpleNamespace] = []
    queue_paths: list[Path] = []

    class FakeQueue:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.jobs: dict[str, SimpleNamespace] = {}
            queue_paths.append(path)

        def enqueue(self, *, task_id: str, budget_group: str, runtime_overrides: dict[str, object]):
            job = SimpleNamespace(
                job_id=task_id,
                state="queued",
                outcome="pending",
                runtime_overrides=runtime_overrides,
                budget_group=budget_group,
            )
            self.jobs[job.job_id] = job
            queued.append(job)
            return job

        def get(self, job_id: str):
            return self.jobs.get(job_id)

    def _fake_drain(queue, **kwargs):
        del kwargs
        for job in queue.jobs.values():
            job.state = "completed"
            job.outcome = "success"
        return list(queue.jobs.values())

    monkeypatch.setattr(module, "DelegatedJobQueue", FakeQueue)
    monkeypatch.setattr(module, "drain_delegated_jobs", _fake_drain)
    monkeypatch.setattr(
        module,
        "_load_hybrid_runtime_manifest",
        lambda manifest_path: {
            "artifact_kind": "tolbert_hybrid_runtime_bundle",
            "manifest_path": str(manifest_path),
        },
    )

    hybrid_output_dir = tmp_path / "relative_hybrid_output"
    hybrid_manifest_path = (hybrid_output_dir / "hybrid_runtime" / "hybrid_bundle_manifest.json").resolve()
    hybrid_manifest, hybrid_records = module.run_tolbert_hybrid_runtime_pipeline(
        config=config,
        repo_root=repo_root,
        output_dir=hybrid_output_dir,
    )

    universal_output_dir = tmp_path / "relative_universal_output"
    universal_dataset_manifest = {"manifest_path": str((universal_output_dir / "dataset" / "manifest.json").resolve())}
    universal_manifest_path = (universal_output_dir / "universal_runtime" / "hybrid_bundle_manifest.json").resolve()
    universal_manifest, universal_records = module.run_tolbert_universal_decoder_pipeline(
        config=config,
        repo_root=repo_root,
        output_dir=universal_output_dir,
        universal_dataset_manifest=universal_dataset_manifest,
        training_controls={"epochs": 1, "batch_size": 8, "lr": 0.001},
    )

    hybrid_success_command = queued[0].runtime_overrides["task_payload"]["success_command"]
    universal_success_command = queued[1].runtime_overrides["task_payload"]["success_command"]

    assert hybrid_success_command == f"test -f {module.sh_quote(hybrid_manifest_path)}"
    assert universal_success_command == f"test -f {module.sh_quote(universal_manifest_path)}"
    assert queue_paths[0] == hybrid_output_dir / "hybrid_jobs" / "queue.json"
    assert queue_paths[1] == universal_output_dir / "jobs" / "universal_queue.json"
    assert hybrid_manifest["manifest_path"] == str(hybrid_output_dir / "hybrid_runtime" / "hybrid_bundle_manifest.json")
    assert universal_manifest["manifest_path"] == str(universal_output_dir / "universal_runtime" / "hybrid_bundle_manifest.json")
    assert [record["job_id"] for record in hybrid_records] == ["tolbert_hybrid_runtime_train"]
    assert [record["job_id"] for record in universal_records] == ["tolbert_universal_decoder_train"]


def test_tolbert_generation_device_plan_distributes_generic_cuda_across_three_gpus(monkeypatch, tmp_path: Path) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime.json",
        tolbert_device="cuda",
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="3\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._tolbert_generation_device_plan(config) == {
        "parallel": True,
        "finetune": "cuda:0",
        "hybrid": "cuda:1",
        "universal": "cuda:2",
    }


def test_tolbert_generation_device_plan_serializes_generic_cuda_when_gpu_count_is_limited(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from agent_kernel.extensions.improvement import tolbert_model_improvement as module

    config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        run_reports_dir=tmp_path / "reports",
        run_checkpoints_dir=tmp_path / "checkpoints",
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime.json",
        tolbert_device="cuda",
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="1\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._tolbert_generation_device_plan(config) == {
        "parallel": False,
        "finetune": "cuda:0",
        "hybrid": "cuda:0",
        "universal": "cuda:0",
    }
