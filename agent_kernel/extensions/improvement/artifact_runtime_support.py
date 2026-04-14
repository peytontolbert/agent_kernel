from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ...config import KernelConfig
from .improvement_common import retained_artifact_payload
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER


def synchronize_retained_universe_artifacts(
    *,
    subsystem: str,
    payload: dict[str, object] | None,
    live_artifact_path: Path,
    runtime_config: KernelConfig | None,
) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    if subsystem not in {"universe", "universe_constitution", "operating_envelope"}:
        return {}
    paths = universe_sync_paths(live_artifact_path=live_artifact_path, runtime_config=runtime_config)
    if not paths:
        return {}

    constitution_path = paths["universe_constitution"]
    envelope_path = paths["operating_envelope"]
    contract_path = paths["universe"]

    if subsystem == "universe":
        bundle = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.compose_universe_bundle_payloads(
            constitution_payload=payload,
            operating_envelope_payload=payload,
            baseline_payload=payload,
            lifecycle_state="retained",
        )
    else:
        baseline_payload = load_first_retained_universe_payload(contract_path, artifact_kind="universe_contract")
        current_constitution = (
            payload
            if subsystem == "universe_constitution"
            else load_first_retained_universe_payload(
                constitution_path,
                artifact_kind="universe_constitution",
                fallback_paths=(contract_path,),
            )
        )
        current_envelope = (
            payload
            if subsystem == "operating_envelope"
            else load_first_retained_universe_payload(
                envelope_path,
                artifact_kind="operating_envelope",
                fallback_paths=(contract_path,),
            )
        )
        bundle = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.compose_universe_bundle_payloads(
            constitution_payload=current_constitution,
            operating_envelope_payload=current_envelope,
            baseline_payload=baseline_payload,
            lifecycle_state="retained",
        )
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.write_universe_bundle_files(
        contract_path=contract_path,
        constitution_path=constitution_path,
        operating_envelope_path=envelope_path,
        bundle=bundle,
    )


def universe_sync_paths(*, live_artifact_path: Path, runtime_config: KernelConfig | None) -> dict[str, Path]:
    if runtime_config is not None:
        return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.universe_bundle_paths(
            universe_contract_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(runtime_config, "universe"),
            universe_constitution_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(
                runtime_config, "universe_constitution"
            ),
            operating_envelope_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(
                runtime_config, "operating_envelope"
            ),
        )
    return DEFAULT_IMPROVEMENT_PLUGIN_LAYER.sibling_universe_bundle_paths(live_artifact_path)


def runtime_config_for_universe_sync(
    runtime_config: KernelConfig | None,
    live_artifact_path: Path,
) -> KernelConfig | None:
    if runtime_config is None:
        return None
    configured_paths = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.universe_bundle_paths(
        universe_contract_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(runtime_config, "universe"),
        universe_constitution_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(
            runtime_config, "universe_constitution"
        ),
        operating_envelope_path=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.active_artifact_path(runtime_config, "operating_envelope"),
    )
    if DEFAULT_IMPROVEMENT_PLUGIN_LAYER.universe_bundle_contains_path(configured_paths, live_artifact_path):
        return runtime_config
    return None


def load_first_retained_universe_payload(
    artifact_path: Path,
    *,
    artifact_kind: str,
    fallback_paths: tuple[Path, ...] = (),
) -> dict[str, object] | None:
    from ... import improvement as core

    for candidate_path in (artifact_path, *fallback_paths):
        if not candidate_path.exists():
            continue
        try:
            loaded = core._load_json_payload(candidate_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        loaded_kind = str(loaded.get("artifact_kind", "")).strip() or artifact_kind
        retained = retained_artifact_payload(loaded, artifact_kind=loaded_kind)
        if retained is not None:
            return retained
    return None


def stamp_tolbert_lineage_metadata(
    payload: dict[str, object],
    *,
    decision_state: str,
    cycle_id: str,
    live_artifact_path: Path,
    parent_artifact_sha256: str,
) -> None:
    lineage = payload.get("lineage", {})
    if not isinstance(lineage, dict):
        lineage = {}
    lineage["mode"] = str(lineage.get("mode", "canonical_parent_mutation")).strip() or "canonical_parent_mutation"
    lineage["canonical_artifact_path"] = str(live_artifact_path)
    if parent_artifact_sha256:
        lineage["parent_artifact_sha256"] = parent_artifact_sha256
        lineage["parent_artifact_path"] = str(live_artifact_path)
    lineage["checkpoint_materialization_policy"] = "evaluate_or_promote_only"
    lineage["promotion_policy"] = "canonical_replace_on_retain"
    lineage["rejected_candidate_policy"] = "metrics_and_mutation_record_only"
    if decision_state == "retain":
        lineage["promotion"] = {
            "cycle_id": cycle_id,
            "state": "promoted_to_canonical",
            "squash_strategy": "canonical_replace",
        }
    payload["lineage"] = lineage


def compact_rejected_tolbert_payload(payload: dict[str, object]) -> dict[str, object]:
    compact = {
        "spec_version": str(payload.get("spec_version", "asi_v1")).strip() or "asi_v1",
        "artifact_kind": str(payload.get("artifact_kind", "tolbert_model_bundle")).strip() or "tolbert_model_bundle",
        "lifecycle_state": str(payload.get("lifecycle_state", "rejected")).strip() or "rejected",
        "generation_focus": str(payload.get("generation_focus", "")).strip(),
        "model_surfaces": strip_pathlike_fields(payload.get("model_surfaces", {})),
        "runtime_policy": strip_pathlike_fields(payload.get("runtime_policy", {})),
        "decoder_policy": strip_pathlike_fields(payload.get("decoder_policy", {})),
        "action_generation_policy": strip_pathlike_fields(payload.get("action_generation_policy", {})),
        "rollout_policy": strip_pathlike_fields(payload.get("rollout_policy", {})),
        "liftoff_gate": strip_pathlike_fields(payload.get("liftoff_gate", {})),
        "build_policy": strip_pathlike_fields(payload.get("build_policy", {})),
        "retention_gate": strip_pathlike_fields(payload.get("retention_gate", {})),
        "training_controls": strip_pathlike_fields(payload.get("training_controls", {})),
        "dataset_manifest": strip_pathlike_fields(payload.get("dataset_manifest", {})),
        "universal_dataset_manifest": strip_pathlike_fields(payload.get("universal_dataset_manifest", {})),
        "hybrid_runtime": strip_pathlike_fields(payload.get("hybrid_runtime", {})),
        "universal_decoder_runtime": strip_pathlike_fields(payload.get("universal_decoder_runtime", {})),
        "universal_decoder_training_controls": strip_pathlike_fields(
            payload.get("universal_decoder_training_controls", {})
        ),
        "proposals": strip_pathlike_fields(payload.get("proposals", [])),
        "parameter_delta": strip_pathlike_fields(payload.get("parameter_delta", {})),
        "compatibility": strip_pathlike_fields(payload.get("compatibility", {})),
        "retention_decision": strip_pathlike_fields(payload.get("retention_decision", {})),
        "rollback_artifact_path": str(payload.get("rollback_artifact_path", "")).strip(),
        "lineage": strip_pathlike_fields(payload.get("lineage", {})),
        "generated_at": str(payload.get("generated_at", "")).strip(),
        "materialization_mode": "rejected_manifest_only",
    }
    compact["rejected_compaction"] = {
        "dropped_fields": [
            "runtime_paths",
            "output_dir",
            "job_records",
            "shared_store",
            "storage_compaction",
            "external_training_backends",
        ],
        "checkpoint_materialization_policy": "metrics_and_mutation_record_only",
    }
    return compact


def promote_tolbert_payload_to_canonical_checkpoint(
    payload: dict[str, object],
    *,
    live_artifact_path: Path,
    cycle_id: str,
) -> None:
    from ... import improvement as core

    runtime_paths = payload.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        return
    delta_path_value = str(runtime_paths.get("checkpoint_delta_path", "")).strip()
    parent_path_value = str(runtime_paths.get("parent_checkpoint_path", "")).strip()
    if not delta_path_value or not parent_path_value:
        return
    delta_path = Path(delta_path_value)
    parent_path = Path(parent_path_value)
    if not delta_path.exists() or not parent_path.exists():
        return
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    canonical_checkpoint_path = live_artifact_path.parent / "checkpoints" / f"tolbert_{safe_cycle_id}.pt"
    canonical_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    core.materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_path,
        delta_checkpoint_path=delta_path,
        output_checkpoint_path=canonical_checkpoint_path,
    )
    runtime_paths["checkpoint_path"] = str(canonical_checkpoint_path)
    runtime_paths.pop("checkpoint_delta_path", None)
    runtime_paths.pop("parent_checkpoint_path", None)
    parameter_delta = payload.get("parameter_delta", {})
    if isinstance(parameter_delta, dict):
        parameter_delta["promotion_applied"] = True
        parameter_delta["canonical_checkpoint_path"] = str(canonical_checkpoint_path)
        parameter_delta.pop("delta_checkpoint_path", None)
        parameter_delta.pop("parent_checkpoint_path", None)
        payload["parameter_delta"] = parameter_delta
    lineage = payload.get("lineage", {})
    if isinstance(lineage, dict):
        promotion = lineage.get("promotion", {})
        if not isinstance(promotion, dict):
            promotion = {}
        promotion["canonical_checkpoint_path"] = str(canonical_checkpoint_path)
        promotion["state"] = "promoted_to_canonical"
        promotion["squash_strategy"] = "materialized_parent_plus_delta"
        lineage["promotion"] = promotion
        payload["lineage"] = lineage


def strip_pathlike_fields(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in {
                "path",
                "paths",
                "root",
                "roots",
                "output_dir",
                "shared_store",
                "runtime_paths",
                "job_records",
                "storage_compaction",
                "external_training_backends",
            }:
                continue
            if lowered.endswith("_path") or lowered.endswith("_paths"):
                continue
            normalized[key] = strip_pathlike_fields(item)
        return normalized
    if isinstance(value, list):
        return [strip_pathlike_fields(item) for item in value]
    return value


def cleanup_rejected_tolbert_payload_artifacts(
    *,
    candidate_artifact_path: Path,
    active_artifact_path: Path,
    output_dir: str,
) -> dict[str, object]:
    removed_output_dir = ""
    candidate_output_dir = Path(output_dir) if str(output_dir).strip() else None
    if candidate_output_dir is not None and candidate_output_dir.exists():
        shutil.rmtree(candidate_output_dir, ignore_errors=True)
        if not candidate_output_dir.exists():
            removed_output_dir = str(candidate_output_dir)

    store_root = active_artifact_path.parent / "store"
    removed_shared_store = cleanup_unreferenced_tolbert_store(
        store_root=store_root,
        active_artifact_path=active_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
    )
    return {
        "removed_output_dir": removed_output_dir,
        "removed_shared_store": removed_shared_store,
    }


def cleanup_unreferenced_tolbert_store(
    *,
    store_root: Path,
    active_artifact_path: Path,
    candidate_artifact_path: Path,
) -> list[str]:
    if not store_root.exists():
        return []
    referenced = tolbert_store_references_from_paths(
        tolbert_reference_artifact_paths(
            active_artifact_path=active_artifact_path,
            candidate_artifact_path=candidate_artifact_path,
        )
    )
    removed: list[str] = []
    for group_dir in sorted(path for path in store_root.iterdir() if path.is_dir()):
        for digest_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            try:
                resolved = str(digest_dir.resolve())
            except OSError:
                resolved = str(digest_dir)
            if resolved in referenced:
                continue
            shutil.rmtree(digest_dir, ignore_errors=True)
            if not digest_dir.exists():
                removed.append(str(digest_dir))
        try:
            next(group_dir.iterdir())
        except StopIteration:
            try:
                group_dir.rmdir()
            except OSError:
                pass
    try:
        next(store_root.iterdir())
    except StopIteration:
        try:
            store_root.rmdir()
        except OSError:
            pass
    return removed


def tolbert_reference_artifact_paths(*, active_artifact_path: Path, candidate_artifact_path: Path) -> list[Path]:
    paths: list[Path] = []
    if active_artifact_path.exists():
        paths.append(active_artifact_path)
    if candidate_artifact_path.exists():
        paths.append(candidate_artifact_path)
    tolbert_candidates_root = None
    if len(candidate_artifact_path.parents) >= 2 and candidate_artifact_path.parents[1].name == "tolbert_model":
        tolbert_candidates_root = candidate_artifact_path.parents[1]
    if tolbert_candidates_root is not None and tolbert_candidates_root.exists():
        for path in sorted(tolbert_candidates_root.glob("*/*.json")):
            if path not in paths and path.exists():
                paths.append(path)
    return paths


def tolbert_store_references_from_paths(paths: list[Path]) -> set[str]:
    referenced: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        referenced.update(tolbert_shared_store_paths_from_payload(payload))
    return referenced


def tolbert_shared_store_paths_from_payload(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    shared_store = payload.get("shared_store", {})
    if not isinstance(shared_store, dict):
        return set()
    entries = shared_store.get("entries", {})
    if not isinstance(entries, dict):
        return set()
    referenced: set[str] = set()
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            referenced.add(str(Path(raw_path).resolve()))
        except OSError:
            referenced.add(raw_path)
    return referenced
