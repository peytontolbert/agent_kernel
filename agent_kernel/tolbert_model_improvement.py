from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from evals.metrics import EvalMetrics

from .config import KernelConfig
from .improvement_common import retention_gate_preset
from .job_queue import DelegatedJobQueue, drain_delegated_jobs
from .modeling.training_backends import discover_training_backends
from .modeling.training.universal_dataset import materialize_universal_decoder_dataset
from .modeling.tolbert.delta import (
    create_tolbert_checkpoint_delta,
    load_tolbert_checkpoint_delta_metadata,
    resolve_tolbert_runtime_checkpoint_path,
)
from .schemas import TaskSpec
from .task_bank import load_discovered_tasks
from .tolbert_assets import build_agentkernel_tolbert_assets

_DEFAULT_TOLBERT_BUILD_POLICY: dict[str, object] = {
    "allow_kernel_autobuild": False,
    "allow_kernel_rebuild": False,
    "require_synthetic_dataset": True,
    "require_head_targets": True,
    "min_total_examples": 512,
    "min_synthetic_examples": 64,
    "min_policy_examples": 256,
    "min_transition_examples": 256,
    "min_value_examples": 256,
    "min_stop_examples": 128,
}
_TOLBERT_SHARED_STORE_GROUPS = (
    "assets",
    "dataset",
    "universal_dataset",
    "training",
    "retrieval_cache",
    "hybrid_runtime",
    "universal_runtime",
)
_TOLBERT_HARD_PROPOSAL_FAMILIES = {
    "integration",
    "project",
    "repository",
}
_TOLBERT_MEDIUM_PROPOSAL_FAMILIES = {
    "benchmark_candidate",
    "repo_chore",
    "repo_sandbox",
    "tooling",
    "verifier_candidate",
    "workflow",
}


def build_tolbert_model_candidate_artifact(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    metrics: EvalMetrics,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    dataset_dir = output_dir / "dataset"
    training_dir = output_dir / "training"
    cache_dir = output_dir / "retrieval_cache"
    assets = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=assets_dir,
        base_model_name=_baseline_training_controls(current_payload).get("base_model_name", "bert-base-uncased"),
        config=config,
    )
    dataset_manifest = build_tolbert_supervised_dataset_manifest(
        config=config,
        repo_root=repo_root,
        output_dir=dataset_dir,
    )
    universal_dataset_manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir / "universal_dataset",
    )
    universal_decoder_training_controls = _tolbert_universal_decoder_training_controls(universal_dataset_manifest)
    training_inputs = _write_training_inputs_manifest(
        output_dir=training_dir,
        dataset_manifest=dataset_manifest,
    )
    training_controls = tolbert_training_controls(
        metrics,
        focus=focus,
        baseline=retained_tolbert_model_training_controls(current_payload),
    )
    train_config_path = _write_training_config(
        base_config_path=assets.config_path,
        output_dir=training_dir,
        training_controls=training_controls,
        dataset_manifest=dataset_manifest,
        training_inputs=training_inputs,
    )
    (
        trained_checkpoint_path,
        cache_path,
        job_records,
        hybrid_manifest,
        hybrid_job_records,
        universal_manifest,
        universal_job_records,
    ) = _run_tolbert_generation_pipelines(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        train_config_path=train_config_path,
        model_spans_path=assets.model_spans_path,
        training_controls=training_controls,
        training_inputs=training_inputs,
        universal_dataset_manifest=universal_dataset_manifest,
        universal_decoder_training_controls=universal_decoder_training_controls,
        current_payload=current_payload,
    )
    hybrid_runtime = _tolbert_hybrid_runtime(
        output_dir=output_dir,
        dataset_manifest=dataset_manifest,
        manifest=hybrid_manifest,
        focus=focus,
    )
    hybrid_runtime_paths = _tolbert_hybrid_runtime_paths(output_dir=output_dir, manifest=hybrid_manifest)
    universal_runtime = _tolbert_universal_decoder_runtime(
        output_dir=output_dir,
        universal_dataset_manifest=universal_dataset_manifest,
        manifest=universal_manifest,
    )
    universal_runtime_paths = _tolbert_universal_decoder_runtime_paths(output_dir=output_dir, manifest=universal_manifest)
    storage_compaction = _compact_tolbert_model_candidate_output(
        output_dir,
        retained_checkpoint_path=trained_checkpoint_path,
    )
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tolbert_model_bundle",
        "lifecycle_state": "candidate",
        "generation_focus": (focus or "balanced").strip() or "balanced",
        "model_surfaces": _tolbert_model_surfaces(focus=focus),
        "runtime_policy": _tolbert_runtime_policy(
            dataset_manifest,
            focus=focus,
            current_payload=current_payload,
        ),
        "decoder_policy": _tolbert_decoder_policy(focus=focus),
        "action_generation_policy": _tolbert_action_generation_policy(
            dataset_manifest,
            current_payload=current_payload,
        ),
        "rollout_policy": _tolbert_rollout_policy(focus=focus),
        "liftoff_gate": _tolbert_liftoff_gate(dataset_manifest),
        "build_policy": _tolbert_build_policy(
            dataset_manifest,
            current_payload=current_payload,
        ),
        "retention_gate": _tolbert_retention_gate(
            dataset_manifest,
            current_payload=current_payload,
        ),
        "training_controls": training_controls,
        "dataset_manifest": dataset_manifest,
        "universal_dataset_manifest": universal_dataset_manifest,
        "training_inputs": training_inputs,
        "hybrid_runtime": hybrid_runtime,
        "universal_decoder_runtime": universal_runtime,
        "universal_decoder_training_controls": universal_decoder_training_controls,
        "external_training_backends": _external_training_backends(repo_root),
        "proposals": _tolbert_model_proposals(metrics, dataset_manifest, repo_root=repo_root, focus=focus),
        "job_records": job_records + hybrid_job_records + universal_job_records,
        "storage_compaction": storage_compaction,
        "runtime_paths": {
            "config_path": str(train_config_path),
            "checkpoint_path": str(trained_checkpoint_path),
            "nodes_path": str(assets.nodes_path),
            "label_map_path": str(assets.label_map_path),
            "source_spans_paths": [str(assets.source_spans_path)],
            "cache_paths": [str(cache_path)],
            **hybrid_runtime_paths,
            **universal_runtime_paths,
            "training_inputs_manifest_path": str(training_inputs.get("training_inputs_manifest_path", "")),
            "universal_train_dataset_path": str(universal_dataset_manifest.get("train_dataset_path", "")),
            "universal_eval_dataset_path": str(universal_dataset_manifest.get("eval_dataset_path", "")),
            "universal_decoder_vocab_path": str(universal_dataset_manifest.get("decoder_vocab_path", "")),
        },
        "output_dir": str(output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    parameter_delta = _build_tolbert_parameter_delta(
        current_payload=current_payload,
        output_dir=output_dir,
        trained_checkpoint_path=trained_checkpoint_path,
        storage_compaction=storage_compaction,
    )
    if parameter_delta:
        payload["parameter_delta"] = parameter_delta
        runtime_paths = payload.get("runtime_paths", {})
        if isinstance(runtime_paths, dict):
            runtime_paths["parent_checkpoint_path"] = str(parameter_delta.get("parent_checkpoint_path", "")).strip()
            runtime_paths["checkpoint_delta_path"] = str(parameter_delta.get("delta_checkpoint_path", "")).strip()
            runtime_paths["checkpoint_path"] = ""
    hybrid_runtime_delta = _build_runtime_bundle_checkpoint_delta(
        current_payload=current_payload,
        current_runtime_key="hybrid_runtime",
        bundle_dir=output_dir / "hybrid_runtime",
        manifest=hybrid_manifest,
    )
    if hybrid_runtime_delta:
        payload["hybrid_runtime_delta"] = hybrid_runtime_delta
    universal_runtime_delta = _build_runtime_bundle_checkpoint_delta(
        current_payload=current_payload,
        current_runtime_key="universal_decoder_runtime",
        bundle_dir=output_dir / "universal_runtime",
        manifest=universal_manifest,
    )
    if universal_runtime_delta:
        payload["universal_runtime_delta"] = universal_runtime_delta
    payload["lineage"] = _tolbert_candidate_lineage(
        config=config,
        current_payload=current_payload,
        payload=payload,
    )
    shared_store = _materialize_tolbert_model_shared_store(
        config=config,
        output_dir=output_dir,
        payload=payload,
    )
    payload["shared_store"] = shared_store
    payload["materialization_mode"] = str(shared_store.get("mode", "inline_bundle"))
    return payload


def _tolbert_candidate_lineage(
    *,
    config: KernelConfig,
    current_payload: object | None,
    payload: dict[str, object],
) -> dict[str, object]:
    parent_artifact_path = config.tolbert_model_artifact_path
    parent_artifact_sha256 = ""
    if parent_artifact_path.exists():
        parent_artifact_sha256 = hashlib.sha256(parent_artifact_path.read_bytes()).hexdigest()
    baseline = current_payload if isinstance(current_payload, dict) else {}
    proposals = payload.get("proposals", [])
    proposal_areas = sorted(
        {
            str(item.get("area", "")).strip()
            for item in proposals
            if isinstance(item, dict) and str(item.get("area", "")).strip()
        }
    )
    return {
        "mode": "canonical_parent_mutation",
        "parent_artifact_path": str(parent_artifact_path) if parent_artifact_sha256 else "",
        "parent_artifact_sha256": parent_artifact_sha256,
        "candidate_artifact_strategy": "manifest_first_shared_store",
        "checkpoint_materialization_policy": "evaluate_or_promote_only",
        "promotion_policy": "canonical_replace_on_retain",
        "rejected_candidate_policy": "metrics_and_mutation_record_only",
        "mutation_manifest": {
            "generation_focus": str(payload.get("generation_focus", "")).strip(),
            "proposal_areas": proposal_areas,
            "training_controls_delta": _mapping_delta(
                baseline.get("training_controls"),
                payload.get("training_controls"),
            ),
            "runtime_policy_delta": _mapping_delta(
                baseline.get("runtime_policy"),
                payload.get("runtime_policy"),
            ),
            "decoder_policy_delta": _mapping_delta(
                baseline.get("decoder_policy"),
                payload.get("decoder_policy"),
            ),
            "rollout_policy_delta": _mapping_delta(
                baseline.get("rollout_policy"),
                payload.get("rollout_policy"),
            ),
            "build_policy_delta": _mapping_delta(
                baseline.get("build_policy"),
                payload.get("build_policy"),
            ),
            "liftoff_gate_delta": _mapping_delta(
                baseline.get("liftoff_gate"),
                payload.get("liftoff_gate"),
            ),
            "model_surface_delta": _mapping_delta(
                baseline.get("model_surfaces"),
                payload.get("model_surfaces"),
            ),
        },
    }


def _mapping_delta(baseline: object, candidate: object) -> dict[str, object]:
    if not isinstance(candidate, dict):
        return {}
    baseline_mapping = baseline if isinstance(baseline, dict) else {}
    return {
        key: value
        for key, value in candidate.items()
        if baseline_mapping.get(key) != value
    }


def _build_tolbert_parameter_delta(
    *,
    current_payload: object | None,
    output_dir: Path,
    trained_checkpoint_path: Path,
    storage_compaction: dict[str, object],
) -> dict[str, object]:
    if not isinstance(current_payload, dict):
        return {}
    runtime_paths = current_payload.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        return {}
    parent_checkpoint_value = str(runtime_paths.get("checkpoint_path", "")).strip()
    if not parent_checkpoint_value:
        return {}
    parent_checkpoint_path = Path(parent_checkpoint_value)
    if not parent_checkpoint_path.exists():
        return {}
    delta_output_path = trained_checkpoint_path.with_name(f"{trained_checkpoint_path.stem}__delta.pt")
    if delta_output_path.exists():
        delta_metadata = load_tolbert_checkpoint_delta_metadata(delta_output_path)
        _record_removed_checkpoint_path(
            storage_compaction=storage_compaction,
            trained_checkpoint_path=trained_checkpoint_path,
            retained_checkpoint_path=delta_output_path,
        )
        return delta_metadata
    if not trained_checkpoint_path.exists():
        return {}
    try:
        delta_metadata = create_tolbert_checkpoint_delta(
            parent_checkpoint_path=parent_checkpoint_path,
            child_checkpoint_path=trained_checkpoint_path,
            delta_output_path=delta_output_path,
        )
    except Exception:
        return {}
    try:
        trained_checkpoint_path.unlink()
    except OSError:
        return {}
    _record_removed_checkpoint_path(
        storage_compaction=storage_compaction,
        trained_checkpoint_path=trained_checkpoint_path,
        retained_checkpoint_path=delta_output_path,
    )
    return delta_metadata


def _build_runtime_bundle_checkpoint_delta(
    *,
    current_payload: object | None,
    current_runtime_key: str,
    bundle_dir: Path,
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(manifest, dict) or not isinstance(current_payload, dict):
        return {}
    current_runtime = current_payload.get(current_runtime_key, {})
    if not isinstance(current_runtime, dict):
        return {}
    parent_checkpoint_value = str(current_runtime.get("checkpoint_path", "")).strip()
    if not parent_checkpoint_value:
        return {}
    parent_checkpoint_path = Path(parent_checkpoint_value)
    manifest_path = bundle_dir / "hybrid_bundle_manifest.json"
    existing_delta_value = str(manifest.get("checkpoint_delta_path", "")).strip()
    if existing_delta_value:
        existing_delta_path = Path(existing_delta_value)
        if not existing_delta_path.is_absolute():
            existing_delta_path = (manifest_path.parent / existing_delta_path).resolve()
        if parent_checkpoint_path.exists() and existing_delta_path.exists():
            delta_metadata = load_tolbert_checkpoint_delta_metadata(existing_delta_path)
            manifest["checkpoint_path"] = ""
            manifest["relative_checkpoint_path"] = ""
            manifest["parent_checkpoint_path"] = str(parent_checkpoint_path)
            manifest["checkpoint_delta_path"] = str(existing_delta_path)
            manifest.setdefault(
                "checkpoint_mutation",
                {
                    "mode": "parent_plus_low_rank_adapter",
                    "parent_checkpoint_path": str(parent_checkpoint_path),
                    "checkpoint_delta_path": str(existing_delta_path),
                    "stats": dict(delta_metadata.get("stats", {})),
                },
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return delta_metadata
    child_checkpoint_value = resolve_tolbert_runtime_checkpoint_path(manifest, artifact_path=manifest_path)
    if not child_checkpoint_value:
        return {}
    child_checkpoint_path = Path(child_checkpoint_value)
    if not parent_checkpoint_path.exists() or not child_checkpoint_path.exists() or not manifest_path.exists():
        return {}
    delta_output_path = child_checkpoint_path.with_name(f"{child_checkpoint_path.stem}__delta.pt")
    try:
        delta_metadata = create_tolbert_checkpoint_delta(
            parent_checkpoint_path=parent_checkpoint_path,
            child_checkpoint_path=child_checkpoint_path,
            delta_output_path=delta_output_path,
        )
    except Exception:
        return {}
    try:
        child_checkpoint_path.unlink()
    except OSError:
        return {}
    manifest["checkpoint_path"] = ""
    manifest["relative_checkpoint_path"] = ""
    manifest["parent_checkpoint_path"] = str(parent_checkpoint_path)
    manifest["checkpoint_delta_path"] = str(delta_output_path.name)
    manifest["checkpoint_mutation"] = {
        "mode": "parent_plus_low_rank_adapter",
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "checkpoint_delta_path": str(delta_output_path),
        "stats": dict(delta_metadata.get("stats", {})),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return delta_metadata


def _record_removed_checkpoint_path(
    *,
    storage_compaction: dict[str, object],
    trained_checkpoint_path: Path,
    retained_checkpoint_path: Path,
) -> None:
    removed_paths = storage_compaction.get("removed_paths", [])
    if isinstance(removed_paths, list) and str(trained_checkpoint_path) not in removed_paths:
        removed_paths.append(str(trained_checkpoint_path))
    storage_compaction["removed_paths"] = removed_paths
    storage_compaction["retained_checkpoint_path"] = str(retained_checkpoint_path)
    storage_compaction["checkpoint_materialization_policy"] = "parent_plus_delta"
    storage_compaction["delta_checkpoint_path"] = str(retained_checkpoint_path)


def _compact_tolbert_model_candidate_output(
    output_dir: Path,
    *,
    retained_checkpoint_path: Path | None,
) -> dict[str, object]:
    removed_paths: list[str] = []
    retained_checkpoint = ""
    retained_resolved: Path | None = None
    if retained_checkpoint_path is not None:
        retained_checkpoint = str(retained_checkpoint_path)
        try:
            retained_resolved = retained_checkpoint_path.resolve()
        except OSError:
            retained_resolved = retained_checkpoint_path

    checkpoints_dir = output_dir / "training" / "checkpoints"
    if checkpoints_dir.exists():
        for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
            try:
                checkpoint_resolved = checkpoint_path.resolve()
            except OSError:
                checkpoint_resolved = checkpoint_path
            if retained_resolved is not None and checkpoint_resolved == retained_resolved:
                continue
            try:
                checkpoint_path.unlink()
            except OSError:
                continue
            removed_paths.append(str(checkpoint_path))

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
        path = output_dir / relative
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except OSError:
            continue
        removed_paths.append(str(path))

    return {
        "removed_paths": removed_paths,
        "retained_checkpoint_path": retained_checkpoint,
    }


def _run_tolbert_generation_pipelines(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    train_config_path: Path,
    model_spans_path: Path,
    training_controls: dict[str, object],
    training_inputs: dict[str, object],
    universal_dataset_manifest: dict[str, object],
    universal_decoder_training_controls: dict[str, object],
    current_payload: object | None,
) -> tuple[
    Path,
    Path,
    list[dict[str, object]],
    dict[str, object] | None,
    list[dict[str, object]],
    dict[str, object] | None,
    list[dict[str, object]],
]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        finetune_future = executor.submit(
            run_tolbert_finetune_pipeline,
            config=config,
            repo_root=repo_root,
            output_dir=output_dir,
            train_config_path=train_config_path,
            model_spans_path=model_spans_path,
            num_epochs=int(training_controls.get("num_epochs", 1)),
            training_inputs=training_inputs,
            current_payload=current_payload,
        )
        hybrid_future = executor.submit(
            run_tolbert_hybrid_runtime_pipeline,
            config=config,
            repo_root=repo_root,
            output_dir=output_dir,
            current_payload=current_payload,
        )
        universal_future = executor.submit(
            run_tolbert_universal_decoder_pipeline,
            config=config,
            repo_root=repo_root,
            output_dir=output_dir,
            universal_dataset_manifest=universal_dataset_manifest,
            training_controls=universal_decoder_training_controls,
            current_payload=current_payload,
        )

        trained_checkpoint_path, cache_path, job_records = finetune_future.result()
        hybrid_manifest, hybrid_job_records = hybrid_future.result()
        universal_manifest, universal_job_records = universal_future.result()

    return (
        trained_checkpoint_path,
        cache_path,
        job_records,
        hybrid_manifest,
        hybrid_job_records,
        universal_manifest,
        universal_job_records,
    )


def _materialize_tolbert_model_shared_store(
    *,
    config: KernelConfig,
    output_dir: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    store_root = (config.tolbert_model_artifact_path.parent / "store").resolve()
    store_root.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, str] = {}
    entries: dict[str, object] = {}
    for group in _TOLBERT_SHARED_STORE_GROUPS:
        source_dir = output_dir / group
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        target_dir = _shared_store_directory_target(
            config=config,
            output_dir=output_dir,
            source_dir=source_dir,
            group=group,
        )
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            shutil.move(str(source_dir), str(target_dir))
        else:
            shutil.rmtree(source_dir, ignore_errors=True)
        for raw_old in _path_aliases(source_dir):
            mappings[raw_old] = str(target_dir)
        entries[group] = {
            "path": str(target_dir),
            "digest": target_dir.name,
        }
    if mappings:
        _rewrite_json_paths_in_shared_store(entries=entries, mappings=mappings)
        rebound = _rebase_paths_in_payload(payload, mappings)
        payload.clear()
        payload.update(rebound)
    try:
        output_dir.rmdir()
    except OSError:
        pass
    return {
        "mode": "content_addressed_shared_store" if entries else "inline_bundle",
        "root": str(store_root),
        "entries": entries,
    }


def _shared_store_directory_target(
    *,
    config: KernelConfig,
    output_dir: Path,
    source_dir: Path,
    group: str,
) -> Path:
    digest = _normalized_directory_digest(source_dir, output_dir=output_dir)
    return (config.tolbert_model_artifact_path.parent / "store" / group / digest).resolve()


def _normalized_directory_digest(path: Path, *, output_dir: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalized_file_bytes(file_path, output_dir=output_dir))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_file_bytes(path: Path, *, output_dir: Path) -> bytes:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    normalized = text
    for alias in _path_aliases(output_dir):
        normalized = normalized.replace(alias, "__TOLBERT_OUTPUT_DIR__")
    return normalized.encode("utf-8")


def _path_aliases(path: Path) -> tuple[str, ...]:
    raw = str(path)
    aliases = [raw]
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = raw
    if resolved not in aliases:
        aliases.append(resolved)
    return tuple(aliases)


def _rewrite_json_paths_in_shared_store(
    *,
    entries: dict[str, object],
    mappings: dict[str, str],
) -> None:
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        root = Path(str(entry.get("path", "")).strip())
        if not root.exists():
            continue
        for json_path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rebound = _rebase_paths_in_payload(payload, mappings)
            if rebound == payload:
                continue
            json_path.write_text(json.dumps(rebound, indent=2), encoding="utf-8")


def _rebase_paths_in_payload(payload: object, mappings: dict[str, str]) -> object:
    if isinstance(payload, dict):
        return {key: _rebase_paths_in_payload(value, mappings) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_rebase_paths_in_payload(value, mappings) for value in payload]
    if isinstance(payload, str):
        return _rebase_path_string(payload, mappings)
    return payload


def _rebase_path_string(value: str, mappings: dict[str, str]) -> str:
    rebound = value
    for old_prefix, new_prefix in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        if rebound == old_prefix:
            return new_prefix
        if rebound.startswith(old_prefix + os.sep):
            return new_prefix + rebound[len(old_prefix) :]
    return rebound


def build_tolbert_supervised_dataset_manifest(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    del repo_root
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    policy_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    value_records: list[dict[str, object]] = []
    stop_records: list[dict[str, object]] = []
    trajectory_examples = 0
    synthetic_trajectory_examples = 0
    transition_failure_examples = 0
    policy_examples = 0
    policy_positive_examples = 0
    transition_examples = 0
    transition_regression_examples = 0
    value_examples = 0
    stop_examples = 0
    stop_positive_examples = 0
    benchmark_families: list[str] = []
    trajectory_payloads = (
        config.sqlite_store().iter_trajectory_payloads()
        if config.uses_sqlite_storage()
        else []
    )
    if not trajectory_payloads:
        for path in sorted(config.trajectories_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                trajectory_payloads.append(payload)
    for payload in trajectory_payloads:
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        task_metadata = payload.get("task_metadata", {})
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        task_contract = payload.get("task_contract", {})
        if not isinstance(task_contract, dict):
            task_contract = {}
        if not task_metadata:
            task_contract_metadata = task_contract.get("metadata", {})
            if isinstance(task_contract_metadata, dict):
                task_metadata = dict(task_contract_metadata)
        world_model_summary = payload.get("world_model_summary", {})
        if not isinstance(world_model_summary, dict):
            world_model_summary = {}
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            continue
        benchmark_family = str(task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"
        final_completion_ratio = _safe_float(
            summary.get("final_completion_ratio", world_model_summary.get("completion_ratio", 0.0)),
        )
        records.append(
            {
                "source_type": "trajectory",
                "task_id": task_id,
                "success": bool(payload.get("success", False)),
                "failure_types": [str(value) for value in summary.get("failure_types", [])],
                "transition_failures": [str(value) for value in summary.get("transition_failures", [])],
                "benchmark_family": benchmark_family,
                "synthetic_worker": bool(task_metadata.get("synthetic_worker", False)),
                "prompt": str(task_contract.get("prompt", payload.get("prompt", ""))).strip(),
                "suggested_commands": _string_list(task_contract.get("suggested_commands", [])),
                "expected_files": _string_list(task_contract.get("expected_files", [])),
                "forbidden_files": _string_list(task_contract.get("forbidden_files", [])),
            }
        )
        step_payloads = payload.get("steps", [])
        cumulative_progress = 0.0
        if isinstance(step_payloads, list):
            total_steps = len(step_payloads)
            for raw_step in step_payloads:
                if not isinstance(raw_step, dict):
                    continue
                step_index = int(raw_step.get("index", 0) or 0)
                action = str(raw_step.get("action", "")).strip()
                content = str(raw_step.get("content", "")).strip()
                verification = raw_step.get("verification", {})
                if not isinstance(verification, dict):
                    verification = {}
                state_transition = raw_step.get("state_transition", {})
                if not isinstance(state_transition, dict):
                    state_transition = {}
                progress_delta = _safe_float(raw_step.get("state_progress_delta", state_transition.get("progress_delta", 0.0)))
                cumulative_progress = max(0.0, min(1.0, round(cumulative_progress + progress_delta, 3)))
                state_regression_count = int(raw_step.get("state_regression_count", 0) or 0)
                failure_signals = _string_list(raw_step.get("failure_signals", []))
                remaining_steps = max(0, total_steps - step_index)
                step_passed = bool(verification.get("passed", False))
                if content or action:
                    policy_record = {
                        "source_type": "policy_step",
                        "task_id": task_id,
                        "step_index": step_index,
                        "action": action,
                        "content": content,
                        "benchmark_family": benchmark_family,
                        "decision_source": str(raw_step.get("decision_source", "")).strip() or "unknown",
                        "tolbert_route_mode": str(raw_step.get("tolbert_route_mode", "")).strip(),
                        "selected_skill_id": str(raw_step.get("selected_skill_id", "")).strip(),
                        "selected_retrieval_span_id": str(raw_step.get("selected_retrieval_span_id", "")).strip(),
                        "retrieval_influenced": bool(raw_step.get("retrieval_influenced", False)),
                        "proposal_source": str(raw_step.get("proposal_source", "")).strip(),
                        "proposal_novel": bool(raw_step.get("proposal_novel", False)),
                        "proposal_metadata": dict(raw_step.get("proposal_metadata", {}))
                        if isinstance(raw_step.get("proposal_metadata", {}), dict)
                        else {},
                        "trust_retrieval": bool(raw_step.get("trust_retrieval", False)),
                        "path_confidence": _safe_float(raw_step.get("path_confidence", 0.0)),
                        "step_passed": step_passed,
                        "episode_success": bool(payload.get("success", False)),
                        "progress_delta": progress_delta,
                        "state_regression_count": state_regression_count,
                        "failure_signals": failure_signals,
                        "completion_ratio_estimate": cumulative_progress,
                        "final_completion_ratio": final_completion_ratio,
                        "expected_files": _string_list(task_contract.get("expected_files", [])),
                        "forbidden_files": _string_list(task_contract.get("forbidden_files", [])),
                    }
                    policy_records.append(policy_record)
                    policy_examples += 1
                    if step_passed or progress_delta > 0.0 or bool(payload.get("success", False)):
                        policy_positive_examples += 1
                if state_transition:
                    transition_record = {
                        "source_type": "transition_step",
                        "task_id": task_id,
                        "step_index": step_index,
                        "action": action,
                        "content": content,
                        "benchmark_family": benchmark_family,
                        "progress_delta": progress_delta,
                        "completion_ratio_before": max(0.0, min(1.0, round(cumulative_progress - progress_delta, 3))),
                        "completion_ratio_after": cumulative_progress,
                        "no_progress": bool(state_transition.get("no_progress", False)),
                        "regressions": _string_list(state_transition.get("regressions", [])),
                        "cleared_forbidden_artifacts": _string_list(state_transition.get("cleared_forbidden_artifacts", [])),
                        "newly_materialized_expected_artifacts": _string_list(
                            state_transition.get("newly_materialized_expected_artifacts", [])
                        ),
                        "newly_satisfied_expected_contents": _string_list(
                            state_transition.get("newly_satisfied_expected_contents", [])
                        ),
                        "failure_signals": failure_signals,
                        "state_regression_count": state_regression_count,
                    }
                    transition_records.append(transition_record)
                    transition_examples += 1
                    if transition_record["regressions"] or transition_record["no_progress"] or state_regression_count > 0:
                        transition_regression_examples += 1
                value_record = {
                    "source_type": "value_step",
                    "task_id": task_id,
                    "step_index": step_index,
                    "benchmark_family": benchmark_family,
                    "completion_ratio_estimate": cumulative_progress,
                    "progress_delta": progress_delta,
                    "remaining_steps": remaining_steps,
                    "episode_success": bool(payload.get("success", False)),
                    "final_completion_ratio": final_completion_ratio,
                    "future_progress_target": round(max(0.0, final_completion_ratio - cumulative_progress), 3),
                    "success_target": 1 if bool(payload.get("success", False)) else 0,
                    "value_target": round(
                        final_completion_ratio
                        + (1.0 if bool(payload.get("success", False)) else 0.0)
                        - (0.1 * remaining_steps)
                        - (0.2 * state_regression_count),
                        3,
                    ),
                }
                value_records.append(value_record)
                value_examples += 1
                stop_label = bool(
                    step_index == total_steps
                    and bool(payload.get("success", False))
                    and final_completion_ratio >= 0.95
                )
                stop_record = {
                    "source_type": "stop_step",
                    "task_id": task_id,
                    "step_index": step_index,
                    "benchmark_family": benchmark_family,
                    "completion_ratio_estimate": cumulative_progress,
                    "final_completion_ratio": final_completion_ratio,
                    "progress_delta": progress_delta,
                    "state_regression_count": state_regression_count,
                    "no_progress": bool(state_transition.get("no_progress", False)),
                    "missing_expected_artifacts_final": _string_list(world_model_summary.get("missing_expected_artifacts", [])),
                    "present_forbidden_artifacts_final": _string_list(world_model_summary.get("present_forbidden_artifacts", [])),
                    "changed_preserved_artifacts_final": _string_list(world_model_summary.get("changed_preserved_artifacts", [])),
                    "stop_target": stop_label,
                }
                stop_records.append(stop_record)
                stop_examples += 1
                if stop_label:
                    stop_positive_examples += 1
        trajectory_examples += 1
        if bool(task_metadata.get("synthetic_worker", False)):
            synthetic_trajectory_examples += 1
        if benchmark_family and benchmark_family not in benchmark_families:
            benchmark_families.append(benchmark_family)
        if summary.get("transition_failures"):
            transition_failure_examples += 1
    discovered_tasks = load_discovered_tasks(config.trajectories_root)
    discovered_examples = 0
    for task in discovered_tasks:
        records.append(
            {
                "source_type": "discovered_task",
                "task_id": task.task_id,
                "prompt": task.prompt,
                "expected_files": list(task.expected_files),
                "transition_failures": [
                    str(value).strip()
                    for value in task.metadata.get("discovery_transition_failures", [])
                    if str(value).strip()
                ],
            }
        )
        discovered_examples += 1
    verifier_examples = 0
    if config.verifier_contracts_path.exists():
        try:
            verifier_payload = json.loads(config.verifier_contracts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            verifier_payload = {}
        proposals = verifier_payload.get("proposals", []) if isinstance(verifier_payload, dict) else []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            contract = proposal.get("contract", {})
            if not isinstance(contract, dict):
                continue
            records.append(
                {
                    "source_type": "verifier_label",
                    "proposal_id": str(proposal.get("proposal_id", "")).strip(),
                    "task_id": str(proposal.get("source_task_id", "")).strip(),
                    "expected_files": list(contract.get("expected_files", [])),
                    "forbidden_files": list(contract.get("forbidden_files", [])),
                }
            )
            verifier_examples += 1
    examples_path = output_dir / "supervised_examples.jsonl"
    policy_examples_path = output_dir / "policy_examples.jsonl"
    transition_examples_path = output_dir / "transition_examples.jsonl"
    value_examples_path = output_dir / "value_examples.jsonl"
    stop_examples_path = output_dir / "stop_examples.jsonl"
    _write_jsonl(examples_path, records)
    _write_jsonl(policy_examples_path, policy_records)
    _write_jsonl(transition_examples_path, transition_records)
    _write_jsonl(value_examples_path, value_records)
    _write_jsonl(stop_examples_path, stop_records)
    action_generation_summary = _summarize_action_generation_examples(policy_records)
    manifest = {
        "dataset_kind": "tolbert_supervised_dataset",
        "supervised_examples_path": str(examples_path),
        "policy_examples_path": str(policy_examples_path),
        "transition_examples_path": str(transition_examples_path),
        "value_examples_path": str(value_examples_path),
        "stop_examples_path": str(stop_examples_path),
        "total_examples": len(records),
        "trajectory_examples": trajectory_examples,
        "synthetic_trajectory_examples": synthetic_trajectory_examples,
        "transition_failure_examples": transition_failure_examples,
        "verifier_label_examples": verifier_examples,
        "discovered_task_examples": discovered_examples,
        "policy_examples": policy_examples,
        "policy_positive_examples": policy_positive_examples,
        "transition_examples": transition_examples,
        "transition_regression_examples": transition_regression_examples,
        "value_examples": value_examples,
        "stop_examples": stop_examples,
        "stop_positive_examples": stop_positive_examples,
        "action_generation_examples": int(action_generation_summary.get("example_count", 0) or 0),
        "action_generation_positive_examples": int(action_generation_summary.get("positive_example_count", 0) or 0),
        "action_generation_summary": action_generation_summary,
        "benchmark_families": benchmark_families,
        "target_heads": {
            "policy": policy_examples > 0,
            "transition": transition_examples > 0,
            "value": value_examples > 0,
            "stop": stop_examples > 0,
        },
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if config.uses_sqlite_storage():
        config.sqlite_store().record_export_manifest(
            export_key=f"dataset:tolbert_supervised:{output_dir.resolve()}",
            export_kind="tolbert_supervised_dataset",
            payload=manifest,
        )
    return manifest


def _summarize_action_generation_examples(policy_records: list[dict[str, object]]) -> dict[str, object]:
    family_template_counts: dict[str, dict[str, int]] = {}
    family_template_successes: dict[str, dict[str, int]] = {}
    family_template_provenance: dict[str, dict[str, list[str]]] = {}
    example_count = 0
    positive_example_count = 0
    for record in policy_records:
        if not isinstance(record, dict):
            continue
        if str(record.get("action", "")).strip() != "code_execute":
            continue
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        example_count += 1
        benchmark_family = str(record.get("benchmark_family", "bounded")).strip() or "bounded"
        template_kind = _command_template_kind(
            content,
            proposal_source=str(record.get("proposal_source", "")).strip(),
        )
        family_counts = family_template_counts.setdefault(benchmark_family, {})
        family_counts[template_kind] = family_counts.get(template_kind, 0) + 1
        if bool(record.get("step_passed", False)) or bool(record.get("episode_success", False)):
            positive_example_count += 1
            family_successes = family_template_successes.setdefault(benchmark_family, {})
            family_successes[template_kind] = family_successes.get(template_kind, 0) + 1
        provenance = family_template_provenance.setdefault(benchmark_family, {}).setdefault(template_kind, [])
        task_id = str(record.get("task_id", "")).strip()
        if task_id and task_id not in provenance:
            provenance.append(task_id)
    template_preferences: dict[str, list[dict[str, object]]] = {}
    for family, counts in family_template_counts.items():
        family_preferences: list[dict[str, object]] = []
        successes = family_template_successes.get(family, {})
        provenance = family_template_provenance.get(family, {})
        for template_kind, support in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            success_count = successes.get(template_kind, 0)
            family_preferences.append(
                {
                    "template_kind": template_kind,
                    "support": support,
                    "success_count": success_count,
                    "pass_rate": round(success_count / max(1, support), 3),
                    "provenance": list(provenance.get(template_kind, []))[:8],
                }
            )
        if family_preferences:
            template_preferences[family] = family_preferences
    return {
        "example_count": example_count,
        "positive_example_count": positive_example_count,
        "template_preferences": template_preferences,
    }


def _command_template_kind(command: str, *, proposal_source: str = "") -> str:
    source = proposal_source.strip()
    if source:
        return source
    normalized = command.strip()
    if "cat <<'" in normalized and " > " in normalized:
        return "expected_file_content"
    if normalized.startswith("touch ") or "&& touch " in normalized:
        return "missing_expected_file"
    if normalized.startswith("rm -f ") or "&& rm -f " in normalized:
        return "cleanup_forbidden_file"
    if "printf " in normalized and " > " in normalized:
        return "expected_file_content"
    return "generic_command"


def tolbert_training_controls(
    metrics: EvalMetrics,
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    controls = {
        "base_model_name": "bert-base-uncased",
        "num_epochs": 1 if metrics.total <= 25 else 2,
        "lr": 5.0e-5,
        "batch_size": 8,
    }
    if isinstance(baseline, dict):
        controls.update({key: value for key, value in baseline.items() if key in controls})
    if metrics.low_confidence_episodes > 0 or focus == "recovery_alignment":
        controls["num_epochs"] = max(int(controls["num_epochs"]), 2)
    if focus == "discovered_task_adaptation":
        controls["lr"] = 3.0e-5
    return controls


def retained_tolbert_model_training_controls(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return {}
    controls = payload.get("training_controls", {})
    return dict(controls) if isinstance(controls, dict) else {}


def _tolbert_build_policy(
    dataset_manifest: dict[str, object],
    *,
    current_payload: object | None,
) -> dict[str, object]:
    controls = dict(_DEFAULT_TOLBERT_BUILD_POLICY)
    if isinstance(current_payload, dict):
        current = current_payload.get("build_policy", {})
        if isinstance(current, dict):
            for key in (
                "allow_kernel_rebuild",
                "require_synthetic_dataset",
                "require_head_targets",
                "min_total_examples",
                "min_synthetic_examples",
                "min_policy_examples",
                "min_transition_examples",
                "min_value_examples",
                "min_stop_examples",
            ):
                if key in current:
                    controls[key] = current[key]
    total_examples = int(dataset_manifest.get("total_examples", 0) or 0)
    synthetic_examples = int(dataset_manifest.get("synthetic_trajectory_examples", 0) or 0)
    policy_examples = int(dataset_manifest.get("policy_examples", 0) or 0)
    transition_examples = int(dataset_manifest.get("transition_examples", 0) or 0)
    value_examples = int(dataset_manifest.get("value_examples", 0) or 0)
    stop_examples = int(dataset_manifest.get("stop_examples", 0) or 0)
    controls["allow_kernel_autobuild"] = bool(total_examples >= int(controls["min_total_examples"]))
    if bool(controls.get("require_synthetic_dataset", True)):
        controls["allow_kernel_autobuild"] = bool(
            controls["allow_kernel_autobuild"] and synthetic_examples >= int(controls["min_synthetic_examples"])
        )
    if bool(controls.get("require_head_targets", True)):
        controls["allow_kernel_autobuild"] = bool(
            controls["allow_kernel_autobuild"]
            and policy_examples >= int(controls["min_policy_examples"])
            and transition_examples >= int(controls["min_transition_examples"])
            and value_examples >= int(controls["min_value_examples"])
            and stop_examples >= int(controls["min_stop_examples"])
        )
    controls["ready_total_examples"] = total_examples
    controls["ready_synthetic_examples"] = synthetic_examples
    controls["ready_policy_examples"] = policy_examples
    controls["ready_transition_examples"] = transition_examples
    controls["ready_value_examples"] = value_examples
    controls["ready_stop_examples"] = stop_examples
    return controls


def _tolbert_action_generation_policy(
    dataset_manifest: dict[str, object],
    *,
    current_payload: object | None,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "enabled": True,
        "max_candidates": 4,
        "proposal_score_bias": 1.5,
        "novel_command_bonus": 1.5,
        "verifier_alignment_bonus": 1.0,
        "expected_file_template_bonus": 0.75,
        "cleanup_template_bonus": 0.5,
        "min_family_support": 1,
        "template_preferences": {},
    }
    if isinstance(current_payload, dict):
        current = current_payload.get("action_generation_policy", {})
        if isinstance(current, dict):
            for key in (
                "enabled",
                "max_candidates",
                "proposal_score_bias",
                "novel_command_bonus",
                "verifier_alignment_bonus",
                "expected_file_template_bonus",
                "cleanup_template_bonus",
                "min_family_support",
            ):
                if key in current:
                    controls[key] = current[key]
    summary = dataset_manifest.get("action_generation_summary", {})
    if isinstance(summary, dict) and isinstance(summary.get("template_preferences", {}), dict):
        template_preferences = {}
        minimum_support = max(1, int(controls.get("min_family_support", 1) or 1))
        for family, items in summary.get("template_preferences", {}).items():
            if not isinstance(items, list):
                continue
            filtered = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                support = int(item.get("support", 0) or 0)
                if support < minimum_support:
                    continue
                filtered.append(
                    {
                        "template_kind": str(item.get("template_kind", "")).strip(),
                        "support": support,
                        "success_count": int(item.get("success_count", 0) or 0),
                        "pass_rate": float(item.get("pass_rate", 0.0) or 0.0),
                        "provenance": [
                            str(value).strip()
                            for value in item.get("provenance", [])
                            if str(value).strip()
                        ][:8],
                    }
                )
            if filtered:
                template_preferences[str(family).strip() or "bounded"] = filtered
        controls["template_preferences"] = template_preferences
    return controls


def _tolbert_retention_gate(
    dataset_manifest: dict[str, object],
    *,
    current_payload: object | None,
) -> dict[str, object]:
    gate = retention_gate_preset("tolbert_model")
    current_gate = {}
    if isinstance(current_payload, dict):
        raw_current_gate = current_payload.get("retention_gate", {})
        if isinstance(raw_current_gate, dict):
            current_gate = raw_current_gate
            gate.update(deepcopy(raw_current_gate))
    gate.setdefault("require_novel_command_signal", True)
    gate.setdefault("min_proposal_selected_steps_delta", 0)
    gate.setdefault("min_novel_valid_command_steps", 1)
    gate.setdefault("min_novel_valid_command_rate_delta", 0.0)
    gate.setdefault("required_confirmation_runs", 2)
    gate["proposal_gate_by_benchmark_family"] = _tolbert_proposal_gate_by_benchmark_family(
        dataset_manifest,
        current_gate=current_gate,
    )
    return gate


def _tolbert_proposal_gate_by_benchmark_family(
    dataset_manifest: dict[str, object],
    *,
    current_gate: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    existing = {}
    if isinstance(current_gate, dict):
        raw_existing = current_gate.get("proposal_gate_by_benchmark_family", {})
        if isinstance(raw_existing, dict):
            existing = raw_existing
    dataset_families = {
        str(value).strip()
        for value in dataset_manifest.get("benchmark_families", [])
        if str(value).strip()
    }
    configured_families = {
        str(value).strip()
        for value in existing
        if str(value).strip()
    }
    families = sorted(dataset_families | configured_families)
    policy: dict[str, dict[str, object]] = {}
    for family in families:
        family_gate = _tolbert_family_proposal_gate(family)
        current_family_gate = existing.get(family, {})
        if isinstance(current_family_gate, dict):
            family_gate.update(deepcopy(current_family_gate))
        policy[family] = family_gate
    return policy


def _tolbert_family_proposal_gate(family: str) -> dict[str, object]:
    normalized = str(family).strip().lower()
    if normalized in _TOLBERT_HARD_PROPOSAL_FAMILIES:
        return {
            "require_novel_command_signal": True,
            "min_proposal_selected_steps_delta": 1,
            "min_novel_valid_command_steps": 1,
            "min_novel_valid_command_rate_delta": 0.1,
        }
    if normalized in _TOLBERT_MEDIUM_PROPOSAL_FAMILIES or normalized.startswith("repo_"):
        return {
            "require_novel_command_signal": True,
            "min_proposal_selected_steps_delta": 0,
            "min_novel_valid_command_steps": 1,
            "min_novel_valid_command_rate_delta": 0.0,
        }
    return {
        "require_novel_command_signal": False,
        "min_proposal_selected_steps_delta": 0,
        "min_novel_valid_command_steps": 0,
        "min_novel_valid_command_rate_delta": 0.0,
    }


def run_tolbert_finetune_pipeline(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    train_config_path: Path,
    model_spans_path: Path,
    num_epochs: int,
    training_inputs: dict[str, object],
    current_payload: object | None = None,
) -> tuple[Path, Path, list[dict[str, object]]]:
    delegated_config = replace(
        config,
        use_tolbert_context=False,
        delegated_job_queue_path=output_dir / "jobs" / "queue.json",
        delegated_job_runtime_state_path=output_dir / "jobs" / "runtime_state.json",
        workspace_root=output_dir / "job_workspace",
        run_reports_dir=output_dir / "job_reports",
        run_checkpoints_dir=output_dir / "job_checkpoints",
    )
    delegated_config.ensure_directories()
    queue = DelegatedJobQueue(delegated_config.delegated_job_queue_path)
    checkpoint_path = output_dir / "training" / "checkpoints" / f"tolbert_epoch{max(1, num_epochs)}.pt"
    delta_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}__delta.pt")
    parent_checkpoint_path = _current_tolbert_checkpoint_path(
        current_payload=current_payload,
        artifact_path=config.tolbert_model_artifact_path,
    )
    cache_path = output_dir / "retrieval_cache" / f"{checkpoint_path.stem}.pt"
    cache_shard_size = max(0, int(config.tolbert_cache_shard_size))
    cache_artifact_path = cache_path.with_suffix(".json") if cache_shard_size > 0 else cache_path
    cache_command = [
        config.tolbert_python_bin,
        str(repo_root / "scripts" / "build_tolbert_cache.py"),
        "--config",
        str(train_config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--spans",
        str(model_spans_path),
        "--out",
        str(cache_path),
        "--device",
        config.tolbert_device,
    ]
    if parent_checkpoint_path is not None:
        cache_command.extend(
            [
                "--parent-checkpoint",
                str(parent_checkpoint_path),
                "--checkpoint-delta",
                str(delta_checkpoint_path),
            ]
        )
    if cache_shard_size > 0:
        cache_command.extend(["--shard-size", str(cache_shard_size)])
    train_success_target = delta_checkpoint_path if parent_checkpoint_path is not None else checkpoint_path
    train_job = queue.enqueue(
        task_id="tolbert_model_train",
        budget_group="tolbert_model",
        runtime_overrides={
            "task_payload": _worker_task_payload(
                task_id="tolbert_model_train",
                prompt="Run the delegated Tolbert fine-tune worker.",
                success_command=_shell_test_file_command(train_success_target),
            ),
            "worker_command": [
                config.tolbert_python_bin,
                str(repo_root / "other_repos" / "TOLBERT" / "scripts" / "train_tolbert.py"),
                "--config",
                str(train_config_path),
                "--device",
                config.tolbert_device,
            ],
            "worker_env": _tolbert_training_env(
                training_inputs,
                parent_checkpoint_path=parent_checkpoint_path,
                checkpoint_delta_path=delta_checkpoint_path if parent_checkpoint_path is not None else None,
            ),
            "worker_cwd": str(repo_root),
            "worker_timeout_seconds": max(config.command_timeout_seconds, 120),
        },
    )
    cache_job = queue.enqueue(
        task_id="tolbert_model_cache",
        budget_group="tolbert_model",
        runtime_overrides={
            "dependency_job_ids": [train_job.job_id],
            "task_payload": _worker_task_payload(
                task_id="tolbert_model_cache",
                prompt="Build the retrieval cache for the trained Tolbert checkpoint.",
                success_command=_shell_test_file_command(cache_artifact_path),
            ),
            "worker_command": cache_command,
            "worker_cwd": str(repo_root),
            "worker_timeout_seconds": max(config.command_timeout_seconds, 120),
        },
    )
    results = drain_delegated_jobs(
        queue,
        limit=4,
        base_config=delegated_config,
        repo_root=repo_root,
        enforce_preflight=False,
    )
    by_id = {job.job_id: job for job in results}
    for required_job in (train_job, cache_job):
        finished = by_id.get(required_job.job_id) or queue.get(required_job.job_id)
        if finished is None or finished.state != "completed" or finished.outcome != "success":
            raise RuntimeError(f"delegated Tolbert worker job failed: {required_job.job_id}")
    return checkpoint_path, cache_artifact_path, [
        _job_record(queue.get(train_job.job_id)),
        _job_record(queue.get(cache_job.job_id)),
    ]


def run_tolbert_hybrid_runtime_pipeline(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    current_payload: object | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    delegated_config = replace(
        config,
        use_tolbert_context=False,
        delegated_job_queue_path=output_dir / "hybrid_jobs" / "queue.json",
        delegated_job_runtime_state_path=output_dir / "hybrid_jobs" / "runtime_state.json",
        workspace_root=output_dir / "hybrid_job_workspace",
        run_reports_dir=output_dir / "hybrid_job_reports",
        run_checkpoints_dir=output_dir / "hybrid_job_checkpoints",
    )
    delegated_config.ensure_directories()
    queue = DelegatedJobQueue(delegated_config.delegated_job_queue_path)
    bundle_dir = output_dir / "hybrid_runtime"
    manifest_path = bundle_dir / "hybrid_bundle_manifest.json"
    parent_checkpoint_path = _current_runtime_checkpoint_path(
        current_payload=current_payload,
        runtime_key="hybrid_runtime",
    )
    job = queue.enqueue(
        task_id="tolbert_hybrid_runtime_train",
        budget_group="tolbert_model",
        runtime_overrides={
            "task_payload": _worker_task_payload(
                task_id="tolbert_hybrid_runtime_train",
                prompt="Train and materialize the retained Tolbert hybrid runtime bundle.",
                success_command=_shell_test_file_command(manifest_path),
            ),
            "worker_command": [
                config.tolbert_python_bin,
                str(repo_root / "scripts" / "train_hybrid_tolbert_runtime.py"),
                "--trajectories-root",
                str(config.trajectories_root),
                "--output-dir",
                str(bundle_dir),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--lr",
                "0.001",
                "--device",
                _hybrid_runtime_device(config),
            ],
            "worker_env": _runtime_training_env(parent_checkpoint_path=parent_checkpoint_path),
            "worker_cwd": str(repo_root),
            "worker_timeout_seconds": max(config.command_timeout_seconds, 240),
        },
    )
    results = drain_delegated_jobs(
        queue,
        limit=2,
        base_config=delegated_config,
        repo_root=repo_root,
        enforce_preflight=False,
    )
    by_id = {queued.job_id: queued for queued in results}
    finished = by_id.get(job.job_id) or queue.get(job.job_id)
    job_records = [_job_record(finished)]
    if finished is None or finished.state != "completed" or finished.outcome != "success":
        return None, job_records
    return _load_hybrid_runtime_manifest(manifest_path), job_records


def run_tolbert_universal_decoder_pipeline(
    *,
    config: KernelConfig,
    repo_root: Path,
    output_dir: Path,
    universal_dataset_manifest: dict[str, object],
    training_controls: dict[str, object],
    current_payload: object | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    dataset_manifest_path = str(universal_dataset_manifest.get("manifest_path", "")).strip()
    if not dataset_manifest_path:
        return None, []
    delegated_config = replace(
        config,
        use_tolbert_context=False,
        delegated_job_queue_path=output_dir / "jobs" / "universal_queue.json",
        delegated_job_runtime_state_path=output_dir / "jobs" / "universal_runtime_state.json",
        workspace_root=output_dir / "universal_job_workspace",
        run_reports_dir=output_dir / "universal_job_reports",
        run_checkpoints_dir=output_dir / "universal_job_checkpoints",
    )
    delegated_config.ensure_directories()
    queue = DelegatedJobQueue(delegated_config.delegated_job_queue_path)
    bundle_dir = output_dir / "universal_runtime"
    manifest_path = bundle_dir / "hybrid_bundle_manifest.json"
    train_config_path = bundle_dir / "universal_decoder_config.json"
    train_config_path.parent.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text(
        json.dumps(
            {
                "model_family": "tolbert_ssm_v1",
                "decoder_vocab_size": int(training_controls.get("decoder_vocab_size", 4096)),
                "max_command_tokens": int(training_controls.get("max_command_tokens", 16)),
                "sequence_length": int(training_controls.get("sequence_length", 6)),
                "hidden_dim": int(training_controls.get("hidden_dim", 96)),
                "d_state": int(training_controls.get("d_state", 24)),
                "dropout": float(training_controls.get("dropout", 0.0)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    job = queue.enqueue(
        task_id="tolbert_universal_decoder_train",
        budget_group="tolbert_model",
        runtime_overrides={
            "task_payload": _worker_task_payload(
                task_id="tolbert_universal_decoder_train",
                prompt="Train and materialize the retained Tolbert universal decoder runtime bundle.",
                success_command=_shell_test_file_command(manifest_path),
            ),
            "worker_command": [
                config.tolbert_python_bin,
                str(repo_root / "scripts" / "train_universal_tolbert_decoder.py"),
                "--dataset-manifest-path",
                dataset_manifest_path,
                "--output-dir",
                str(bundle_dir),
                "--config-path",
                str(train_config_path),
                "--epochs",
                str(int(training_controls.get("epochs", 1))),
                "--batch-size",
                str(int(training_controls.get("batch_size", 8))),
                "--lr",
                str(float(training_controls.get("lr", 0.001))),
                "--device",
                _hybrid_runtime_device(config),
            ],
            "worker_env": _runtime_training_env(
                parent_checkpoint_path=_current_runtime_checkpoint_path(
                    current_payload=current_payload,
                    runtime_key="universal_decoder_runtime",
                )
            ),
            "worker_cwd": str(repo_root),
            "worker_timeout_seconds": max(config.command_timeout_seconds, 240),
        },
    )
    results = drain_delegated_jobs(
        queue,
        limit=2,
        base_config=delegated_config,
        repo_root=repo_root,
        enforce_preflight=False,
    )
    by_id = {queued.job_id: queued for queued in results}
    finished = by_id.get(job.job_id) or queue.get(job.job_id)
    job_records = [_job_record(finished)]
    if finished is None or finished.state != "completed" or finished.outcome != "success":
        return None, job_records
    return _load_hybrid_runtime_manifest(manifest_path), job_records


def _write_training_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    training_controls: dict[str, object],
    dataset_manifest: dict[str, object],
    training_inputs: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    payload["base_model_name"] = str(training_controls.get("base_model_name", payload.get("base_model_name", "bert-base-uncased")))
    payload["num_epochs"] = int(training_controls.get("num_epochs", payload.get("num_epochs", 1)))
    payload["lr"] = float(training_controls.get("lr", payload.get("lr", 5.0e-5)))
    payload["batch_size"] = int(training_controls.get("batch_size", payload.get("batch_size", 8)))
    payload["output_dir"] = str(output_dir / "checkpoints")
    payload["agentkernel_supervision"] = {
        "dataset_manifest_path": str(training_inputs.get("dataset_manifest_path", "")),
        "training_inputs_manifest_path": str(training_inputs.get("training_inputs_manifest_path", "")),
        "supervised_examples_path": str(dataset_manifest.get("supervised_examples_path", "")),
        "policy_examples_path": str(dataset_manifest.get("policy_examples_path", "")),
        "transition_examples_path": str(dataset_manifest.get("transition_examples_path", "")),
        "value_examples_path": str(dataset_manifest.get("value_examples_path", "")),
        "stop_examples_path": str(dataset_manifest.get("stop_examples_path", "")),
        "policy_examples": int(dataset_manifest.get("policy_examples", 0) or 0),
        "transition_examples": int(dataset_manifest.get("transition_examples", 0) or 0),
        "value_examples": int(dataset_manifest.get("value_examples", 0) or 0),
        "stop_examples": int(dataset_manifest.get("stop_examples", 0) or 0),
        "enabled_heads": [
            str(value).strip()
            for value in training_inputs.get("enabled_heads", [])
            if str(value).strip()
        ] if isinstance(training_inputs.get("enabled_heads", []), list) else [],
    }
    train_config_path = output_dir / "tolbert_train_config.json"
    train_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return train_config_path


def _baseline_training_controls(payload: object | None) -> dict[str, object]:
    return retained_tolbert_model_training_controls(payload)


def _tolbert_model_proposals(
    metrics: EvalMetrics,
    dataset_manifest: dict[str, object],
    *,
    repo_root: Path,
    focus: str | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "area": (focus or "balanced").strip() or "balanced",
            "priority": 5 if metrics.low_confidence_episodes > 0 else 4,
            "reason": "Tolbert should train as the full encoder-latent-decoder family from discovered tasks, verifier labels, and transition outcomes rather than staying a retrieval-only compiler.",
            "dataset_examples": int(dataset_manifest.get("total_examples", 0)),
            "policy_examples": int(dataset_manifest.get("policy_examples", 0)),
            "transition_examples": int(dataset_manifest.get("transition_examples", 0)),
            "value_examples": int(dataset_manifest.get("value_examples", 0)),
            "stop_examples": int(dataset_manifest.get("stop_examples", 0)),
        },
        {
            "area": "external_training_backend",
            "priority": 3,
            "reason": "Vendored external training backends can improve efficient training or export surfaces without changing the retained Tolbert runtime artifact contract.",
            "backend_count": len(_external_training_backends(repo_root)),
        },
    ]


def _external_training_backends(repo_root: Path) -> list[dict[str, object]]:
    backends: list[dict[str, object]] = []
    for manifest in discover_training_backends(repo_root):
        backends.append(
            {
                "backend_id": str(manifest.get("backend_id", "")).strip(),
                "label": str(manifest.get("label", "")).strip(),
                "backend_kind": str(manifest.get("backend_kind", "")).strip(),
                "trainer_exists": bool(manifest.get("trainer_exists", False)),
                "default_launch": dict(manifest.get("default_launch", {}))
                if isinstance(manifest.get("default_launch", {}), dict)
                else {},
                "features": dict(manifest.get("features", {}))
                if isinstance(manifest.get("features", {}), dict)
                else {},
                "artifacts": dict(manifest.get("artifacts", {}))
                if isinstance(manifest.get("artifacts", {}), dict)
                else {},
                "notes": [
                    str(value).strip()
                    for value in manifest.get("notes", [])
                    if str(value).strip()
                ]
                if isinstance(manifest.get("notes", []), list)
                else [],
            }
        )
    return backends


def _tolbert_model_surfaces(*, focus: str | None = None) -> dict[str, object]:
    surfaces = {
        "encoder_surface": True,
        "latent_dynamics_surface": True,
        "decoder_surface": True,
        "world_model_surface": True,
        "retrieval_surface": True,
        "policy_head": True,
        "value_head": True,
        "transition_head": True,
        "risk_head": True,
        "stop_head": True,
        "latent_state": True,
        "universal_runtime": True,
    }
    if focus == "recovery_alignment":
        surfaces["transition_head"] = True
        surfaces["value_head"] = True
    if focus == "discovered_task_adaptation":
        surfaces["policy_head"] = True
        surfaces["latent_state"] = True
    return surfaces


def _tolbert_runtime_policy(
    dataset_manifest: dict[str, object],
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    current = current_payload if isinstance(current_payload, dict) else {}
    current_policy = current.get("runtime_policy", {}) if isinstance(current, dict) else {}
    benchmark_families = [
        str(value).strip()
        for value in dataset_manifest.get("benchmark_families", [])
        if str(value).strip()
    ]
    primary_families = []
    if isinstance(current_policy, dict):
        primary_families = [
            str(value).strip()
            for value in current_policy.get("primary_benchmark_families", [])
            if str(value).strip()
        ]
    shadow_families = list(benchmark_families)
    min_path_confidence = 0.8
    if focus == "recovery_alignment":
        min_path_confidence = 0.72
    elif focus == "discovered_task_adaptation":
        min_path_confidence = 0.7
    return {
        "shadow_benchmark_families": shadow_families,
        "primary_benchmark_families": primary_families,
        "min_path_confidence": min_path_confidence,
        "require_trusted_retrieval": True,
        "fallback_to_vllm_on_low_confidence": True,
        "allow_direct_command_primary": True,
        "allow_skill_primary": True,
        "primary_min_command_score": 2 if focus != "recovery_alignment" else 1,
        "use_encoder_context": True,
        "use_decoder_head": True,
        "use_value_head": True,
        "use_transition_head": True,
        "use_world_model_head": True,
        "use_risk_head": True,
        "use_stop_head": True,
        "use_policy_head": True,
        "use_latent_state": True,
    }


def _tolbert_liftoff_gate(dataset_manifest: dict[str, object]) -> dict[str, object]:
    return {
        "min_pass_rate_delta": 0.0,
        "max_step_regression": 0.0,
        "max_regressed_families": 0,
        "require_generated_lane_non_regression": True,
        "require_failure_recovery_non_regression": True,
        "require_shadow_signal": True,
        "min_shadow_episodes_per_promoted_family": 1,
        "require_family_novel_command_evidence": True,
        "proposal_gate_by_benchmark_family": _tolbert_proposal_gate_by_benchmark_family(
            dataset_manifest,
            current_gate={},
        ),
        "require_takeover_drift_eval": True,
        "takeover_drift_step_budget": 10000,
        "takeover_drift_wave_task_limit": 64,
        "takeover_drift_max_waves": 16,
        "max_takeover_drift_pass_rate_regression": 0.0,
        "max_takeover_drift_unsafe_ambiguous_rate_regression": 0.0,
        "max_takeover_drift_hidden_side_effect_rate_regression": 0.0,
        "max_takeover_drift_trust_success_rate_regression": 0.0,
        "max_takeover_drift_trust_unsafe_ambiguous_rate_regression": 0.0,
    }


def _tolbert_decoder_policy(*, focus: str | None = None) -> dict[str, object]:
    policy = {
        "allow_retrieval_guidance": True,
        "allow_skill_commands": True,
        "allow_task_suggestions": True,
        "allow_stop_decision": True,
        "min_stop_completion_ratio": 0.95,
        "max_task_suggestions": 3,
    }
    if focus == "discovered_task_adaptation":
        policy["max_task_suggestions"] = 4
    return policy


def _tolbert_rollout_policy(*, focus: str | None = None) -> dict[str, object]:
    policy = {
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
    }
    if focus == "recovery_alignment":
        policy["recover_from_stall_bonus_weight"] = 2.5
        policy["latent_risk_penalty_weight"] = 2.5
    return policy


def _tolbert_hybrid_runtime(
    *,
    output_dir: Path,
    dataset_manifest: dict[str, object],
    manifest: dict[str, object] | None = None,
    focus: str | None = None,
) -> dict[str, object]:
    ready = (
        int(dataset_manifest.get("policy_examples", 0) or 0) > 0
        and int(dataset_manifest.get("transition_examples", 0) or 0) > 0
        and int(dataset_manifest.get("value_examples", 0) or 0) > 0
        and int(dataset_manifest.get("stop_examples", 0) or 0) > 0
    )
    bundle_dir = output_dir / "hybrid_runtime"
    runtime = {
        "model_family": "tolbert_ssm_v1",
        "shadow_enabled": bool(ready and manifest is not None),
        "primary_enabled": False,
        "bundle_manifest_path": str(bundle_dir / "hybrid_bundle_manifest.json"),
        "checkpoint_path": str(bundle_dir / "hybrid_checkpoint.pt"),
        "config_path": str(bundle_dir / "hybrid_config.json"),
        "preferred_device": "cpu",
        "preferred_backend": "selective_scan",
        "scoring_policy": _tolbert_hybrid_scoring_policy(focus=focus),
        "supports_encoder_surface": True,
        "supports_latent_dynamics_surface": True,
        "supports_decoder_surface": True,
        "supports_world_model_surface": True,
        "supports_policy_head": True,
        "supports_value_head": True,
        "supports_transition_head": True,
        "supports_risk_head": True,
        "supports_stop_head": True,
        "supports_universal_runtime": True,
    }
    if isinstance(manifest, dict):
        runtime["model_family"] = str(manifest.get("model_family", runtime["model_family"])).strip() or str(runtime["model_family"])
        runtime["bundle_manifest_path"] = str(
            _resolve_hybrid_manifest_value(
                manifest_path=bundle_dir / "hybrid_bundle_manifest.json",
                raw_path=str(manifest.get("relative_bundle_manifest_path", "")).strip() or str(bundle_dir / "hybrid_bundle_manifest.json"),
            )
        )
        runtime["checkpoint_path"] = resolve_tolbert_runtime_checkpoint_path(
            manifest,
            artifact_path=bundle_dir / "hybrid_bundle_manifest.json",
        ) or runtime["checkpoint_path"]
        runtime["config_path"] = str(
            _resolve_hybrid_manifest_value(
                manifest_path=bundle_dir / "hybrid_bundle_manifest.json",
                raw_path=str(manifest.get("relative_config_path", "")).strip() or str(manifest.get("config_path", "")).strip(),
            )
        )
    return runtime


def _tolbert_hybrid_scoring_policy(*, focus: str | None = None) -> dict[str, float]:
    policy: dict[str, float] = {
        "learned_score_weight": 1.5,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "risk_penalty_weight": 1.0,
        "stop_weight": 1.0,
        "transition_progress_weight": 0.15,
        "transition_regression_penalty_weight": 0.20,
        "world_progress_weight": 0.20,
        "world_risk_penalty_weight": 0.20,
        "latent_bias_weight": 0.10,
        "decoder_logprob_weight": 0.10,
        "respond_learned_score_weight": 1.25,
        "respond_policy_weight": 0.0,
        "respond_value_weight": 1.0,
        "respond_risk_penalty_weight": 1.0,
        "respond_stop_weight": 1.0,
        "respond_transition_progress_weight": 0.0,
        "respond_transition_regression_penalty_weight": 0.0,
        "respond_world_progress_weight": 0.10,
        "respond_world_risk_penalty_weight": 0.20,
        "respond_latent_bias_weight": 0.0,
        "respond_decoder_logprob_weight": 0.05,
    }
    if focus == "recovery_alignment":
        policy["learned_score_weight"] = 1.8
        policy["transition_progress_weight"] = 0.30
        policy["transition_regression_penalty_weight"] = 0.35
        policy["risk_penalty_weight"] = 1.2
        policy["world_risk_penalty_weight"] = 0.30
    elif focus == "discovered_task_adaptation":
        policy["learned_score_weight"] = 1.7
        policy["latent_bias_weight"] = 0.18
        policy["decoder_logprob_weight"] = 0.14
        policy["world_progress_weight"] = 0.25
    return policy


def _tolbert_hybrid_runtime_paths(
    *,
    output_dir: Path,
    manifest: dict[str, object] | None,
) -> dict[str, str]:
    manifest_path = output_dir / "hybrid_runtime" / "hybrid_bundle_manifest.json"
    paths = {
        "hybrid_bundle_manifest_path": str(manifest_path),
        "hybrid_checkpoint_path": str(output_dir / "hybrid_runtime" / "hybrid_checkpoint.pt"),
        "hybrid_config_path": str(output_dir / "hybrid_runtime" / "hybrid_config.json"),
    }
    if not isinstance(manifest, dict):
        return paths
    paths["hybrid_bundle_manifest_path"] = str(manifest_path)
    paths["hybrid_checkpoint_path"] = resolve_tolbert_runtime_checkpoint_path(manifest, artifact_path=manifest_path) or paths[
        "hybrid_checkpoint_path"
    ]
    paths["hybrid_config_path"] = str(
        _resolve_hybrid_manifest_value(
            manifest_path=manifest_path,
            raw_path=str(manifest.get("relative_config_path", "")).strip() or str(manifest.get("config_path", "")).strip(),
        )
    )
    return paths


def _tolbert_universal_decoder_runtime(
    *,
    output_dir: Path,
    universal_dataset_manifest: dict[str, object],
    manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    ready = int(universal_dataset_manifest.get("train_examples", 0) or 0) > 0
    bundle_dir = output_dir / "universal_runtime"
    runtime = {
        "model_family": "tolbert_ssm_v1",
        "materialized": bool(ready and manifest is not None),
        "bundle_manifest_path": str(bundle_dir / "hybrid_bundle_manifest.json"),
        "checkpoint_path": str(bundle_dir / "hybrid_checkpoint.pt"),
        "config_path": str(bundle_dir / "hybrid_config.json"),
        "training_objective": "universal_decoder_only",
        "dataset_manifest_path": str(universal_dataset_manifest.get("manifest_path", "")),
    }
    if not isinstance(manifest, dict):
        return runtime
    runtime["model_family"] = str(manifest.get("model_family", runtime["model_family"])).strip() or str(runtime["model_family"])
    runtime["checkpoint_path"] = resolve_tolbert_runtime_checkpoint_path(
        manifest,
        artifact_path=bundle_dir / "hybrid_bundle_manifest.json",
    ) or runtime["checkpoint_path"]
    runtime["config_path"] = str(
        _resolve_hybrid_manifest_value(
            manifest_path=bundle_dir / "hybrid_bundle_manifest.json",
            raw_path=str(manifest.get("relative_config_path", "")).strip() or str(manifest.get("config_path", "")).strip(),
        )
    )
    return runtime


def _tolbert_universal_decoder_runtime_paths(
    *,
    output_dir: Path,
    manifest: dict[str, object] | None,
) -> dict[str, str]:
    manifest_path = output_dir / "universal_runtime" / "hybrid_bundle_manifest.json"
    paths = {
        "universal_bundle_manifest_path": str(manifest_path),
        "universal_checkpoint_path": str(output_dir / "universal_runtime" / "hybrid_checkpoint.pt"),
        "universal_config_path": str(output_dir / "universal_runtime" / "hybrid_config.json"),
    }
    if not isinstance(manifest, dict):
        return paths
    paths["universal_checkpoint_path"] = resolve_tolbert_runtime_checkpoint_path(manifest, artifact_path=manifest_path) or paths[
        "universal_checkpoint_path"
    ]
    paths["universal_config_path"] = str(
        _resolve_hybrid_manifest_value(
            manifest_path=manifest_path,
            raw_path=str(manifest.get("relative_config_path", "")).strip() or str(manifest.get("config_path", "")).strip(),
        )
    )
    return paths


def _tolbert_universal_decoder_training_controls(
    universal_dataset_manifest: dict[str, object],
) -> dict[str, object]:
    total_examples = int(universal_dataset_manifest.get("total_examples", 0) or 0)
    train_examples = int(universal_dataset_manifest.get("train_examples", 0) or 0)
    external_examples = int(universal_dataset_manifest.get("external_corpus_example_count", 0) or 0)
    tokenizer_stats = (
        dict(universal_dataset_manifest.get("decoder_tokenizer_stats", {}))
        if isinstance(universal_dataset_manifest.get("decoder_tokenizer_stats", {}), dict)
        else {}
    )
    unique_tokens = int(tokenizer_stats.get("unique_token_count", 0) or 0)
    controls = {
        "epochs": 1,
        "batch_size": 8,
        "lr": 1.0e-3,
        "decoder_vocab_size": max(256, min(4096, max(512, unique_tokens + 64))),
        "max_command_tokens": 24 if total_examples >= 512 else 16,
        "sequence_length": 8 if total_examples >= 512 else 6,
        "hidden_dim": 128 if train_examples >= 1024 else 96,
        "d_state": 32 if train_examples >= 1024 else 24,
        "dropout": 0.05 if external_examples > 0 else 0.0,
    }
    if train_examples >= 256:
        controls["epochs"] = 2
        controls["batch_size"] = 12
        controls["lr"] = 7.5e-4
    if train_examples >= 1024:
        controls["epochs"] = 3
        controls["batch_size"] = 16
        controls["lr"] = 5.0e-4
    return controls


def _write_training_inputs_manifest(
    *,
    output_dir: Path,
    dataset_manifest: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_heads = [
        head
        for head, count in (
            ("policy", int(dataset_manifest.get("policy_examples", 0) or 0)),
            ("transition", int(dataset_manifest.get("transition_examples", 0) or 0)),
            ("value", int(dataset_manifest.get("value_examples", 0) or 0)),
            ("stop", int(dataset_manifest.get("stop_examples", 0) or 0)),
        )
        if count > 0
    ]
    payload = {
        "manifest_kind": "tolbert_training_inputs",
        "dataset_manifest_path": str(output_dir.parent / "dataset" / "dataset_manifest.json"),
        "supervised_examples_path": str(dataset_manifest.get("supervised_examples_path", "")),
        "policy_examples_path": str(dataset_manifest.get("policy_examples_path", "")),
        "transition_examples_path": str(dataset_manifest.get("transition_examples_path", "")),
        "value_examples_path": str(dataset_manifest.get("value_examples_path", "")),
        "stop_examples_path": str(dataset_manifest.get("stop_examples_path", "")),
        "policy_examples": int(dataset_manifest.get("policy_examples", 0) or 0),
        "transition_examples": int(dataset_manifest.get("transition_examples", 0) or 0),
        "value_examples": int(dataset_manifest.get("value_examples", 0) or 0),
        "stop_examples": int(dataset_manifest.get("stop_examples", 0) or 0),
        "enabled_heads": enabled_heads,
    }
    manifest_path = output_dir / "training_inputs_manifest.json"
    payload["training_inputs_manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _tolbert_training_env(
    training_inputs: dict[str, object],
    *,
    parent_checkpoint_path: Path | None = None,
    checkpoint_delta_path: Path | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, env_name in (
        ("training_inputs_manifest_path", "AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH"),
        ("dataset_manifest_path", "AGENTKERNEL_TOLBERT_DATASET_MANIFEST_PATH"),
        ("supervised_examples_path", "AGENTKERNEL_TOLBERT_SUPERVISED_EXAMPLES_PATH"),
        ("policy_examples_path", "AGENTKERNEL_TOLBERT_POLICY_EXAMPLES_PATH"),
        ("transition_examples_path", "AGENTKERNEL_TOLBERT_TRANSITION_EXAMPLES_PATH"),
        ("value_examples_path", "AGENTKERNEL_TOLBERT_VALUE_EXAMPLES_PATH"),
        ("stop_examples_path", "AGENTKERNEL_TOLBERT_STOP_EXAMPLES_PATH"),
    ):
        value = str(training_inputs.get(key, "")).strip()
        if value:
            env[env_name] = value
    enabled_heads = training_inputs.get("enabled_heads", [])
    if isinstance(enabled_heads, list):
        env["AGENTKERNEL_TOLBERT_ENABLED_HEADS"] = ",".join(
            str(value).strip()
            for value in enabled_heads
            if str(value).strip()
        )
    if parent_checkpoint_path is not None and parent_checkpoint_path.exists():
        env["AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH"] = str(parent_checkpoint_path)
    if checkpoint_delta_path is not None:
        env["AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH"] = str(checkpoint_delta_path)
    return env


def _runtime_training_env(*, parent_checkpoint_path: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if parent_checkpoint_path is not None and parent_checkpoint_path.exists():
        env["AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH"] = str(parent_checkpoint_path)
    return env


def _current_tolbert_checkpoint_path(*, current_payload: object | None, artifact_path: Path) -> Path | None:
    if not isinstance(current_payload, dict):
        return None
    runtime_paths = current_payload.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        return None
    resolved = resolve_tolbert_runtime_checkpoint_path(runtime_paths, artifact_path=artifact_path)
    if resolved:
        path = Path(resolved)
        if path.exists():
            return path
    raw_path = str(runtime_paths.get("checkpoint_path", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (artifact_path.parent / path).resolve()
    return path if path.exists() else None


def _current_runtime_checkpoint_path(*, current_payload: object | None, runtime_key: str) -> Path | None:
    if not isinstance(current_payload, dict):
        return None
    runtime = current_payload.get(runtime_key, {})
    if not isinstance(runtime, dict):
        return None
    raw_path = str(runtime.get("checkpoint_path", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        return None
    return path


def _worker_task_payload(*, task_id: str, prompt: str, success_command: str) -> dict[str, object]:
    return TaskSpec(
        task_id=task_id,
        prompt=prompt,
        workspace_subdir=f"delegated/{task_id}",
        success_command=success_command,
        max_steps=1,
        metadata={"benchmark_family": "tooling", "capability": "python"},
    ).to_dict()


def _job_record(job: Any) -> dict[str, object]:
    if job is None:
        return {}
    return {
        "job_id": str(getattr(job, "job_id", "")).strip(),
        "state": str(getattr(job, "state", "")).strip(),
        "outcome": str(getattr(job, "outcome", "")).strip(),
        "last_error": str(getattr(job, "last_error", "")).strip(),
        "report_path": str(getattr(job, "report_path", "")).strip(),
        "checkpoint_path": str(getattr(job, "checkpoint_path", "")).strip(),
    }


def sh_quote(path: Path) -> str:
    return str(path).replace(" ", "\\ ")


def _shell_test_file_command(path: Path) -> str:
    return f"test -f {sh_quote(path.resolve())}"


def _hybrid_runtime_device(config: KernelConfig) -> str:
    device = str(config.tolbert_device).strip().lower()
    return "cuda" if device.startswith("cuda") else "cpu"


def _load_hybrid_runtime_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_hybrid_manifest_value(*, manifest_path: Path, raw_path: str) -> Path:
    if not raw_path:
        return manifest_path
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _string_list(values: object) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item:
            normalized.append(item)
    return normalized


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
