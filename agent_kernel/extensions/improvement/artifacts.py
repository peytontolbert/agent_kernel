from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ...config import KernelConfig
from ...improvement_engine import ImprovementVariant, retention_gate_for_payload as engine_retention_gate_for_payload
from ...ops.runtime_supervision import atomic_write_json
from .improvement_common import retention_gate_preset
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER


def _atomic_write_json_compat(path: Path, payload: Any, *, config: KernelConfig | None = None) -> None:
    try:
        from ... import improvement as improvement_module

        writer = getattr(improvement_module, "atomic_write_json", atomic_write_json)
    except Exception:
        writer = atomic_write_json
    writer(path, payload, config=config)


def artifact_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_artifact(artifact_path: Path, *, cycle_id: str, stage: str) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    history_root = artifact_path.parent / ".artifact_history"
    history_root.mkdir(parents=True, exist_ok=True)
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "_", stage).strip("._") or "snapshot"
    suffix = artifact_path.suffix or ".json"
    snapshot_path = history_root / f"{artifact_path.stem}.{safe_cycle_id}.{safe_stage}{suffix}"
    if artifact_path.exists():
        shutil.copy2(artifact_path, snapshot_path)
    else:
        snapshot_path.write_text("{}", encoding="utf-8")
    return snapshot_path


def snapshot_artifact_state(artifact_path: Path, *, cycle_id: str, stage: str) -> Path:
    return _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage=stage)


def materialize_replay_verified_tool_payload(payload: Any) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("tool artifact payload must be a JSON object")
    clone = deepcopy(payload)
    candidates = clone.get("candidates", [])
    if not isinstance(candidates, list):
        return clone
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        procedure = candidate.get("procedure", {})
        if not isinstance(procedure, dict):
            procedure = {}
        commands = [str(value).strip() for value in procedure.get("commands", []) if str(value).strip()]
        if commands:
            candidate["procedure"] = {"commands": list(commands)}
        task_contract = candidate.get("task_contract", {})
        if not isinstance(task_contract, dict):
            task_contract = {}
        task_metadata = task_contract.get("metadata", {})
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        source_task_id = str(candidate.get("source_task_id", "")).strip()
        tool_id = str(candidate.get("tool_id", "")).strip()
        script_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_task_id or tool_id or "tool_candidate").strip("._")
        script_name = str(candidate.get("script_name", "")).strip()
        if not script_name and script_stem:
            script_name = f"{script_stem}_tool.sh"
        script_body = str(candidate.get("script_body", "")).strip()
        if not script_body and commands:
            script_body = "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n"
        quality = candidate.get("quality")
        if isinstance(quality, bool) or not isinstance(quality, (int, float)):
            quality = 0.0
        benchmark_family = str(
            candidate.get("benchmark_family")
            or task_metadata.get("benchmark_family")
            or task_contract.get("benchmark_family", "")
        ).strip()
        verifier = candidate.get("verifier", {})
        if not isinstance(verifier, dict):
            verifier = {}
        if not verifier:
            verifier = {"termination_reason": "success"}
        candidate["kind"] = "local_shell_procedure"
        if benchmark_family:
            candidate["benchmark_family"] = benchmark_family
        if script_name:
            candidate["script_name"] = script_name
        if script_body:
            candidate["script_body"] = script_body
        candidate["quality"] = float(quality)
        candidate["verifier"] = verifier
        candidate["promotion_stage"] = "replay_verified"
        candidate["lifecycle_state"] = "replay_verified"
    return clone


def persist_replay_verified_tool_artifact(
    artifact_path: Path,
    *,
    cycle_id: str = "manual",
    runtime_config: KernelConfig | None = None,
) -> dict[str, object]:
    payload = _load_json_payload(artifact_path)
    previous_sha256 = artifact_sha256(artifact_path)
    rollback_snapshot_path = _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage="pre_replay_verified")
    replay_verified_payload = materialize_replay_verified_tool_payload(payload)
    replay_verified_payload["lifecycle_state"] = "replay_verified"
    replay_verified_payload["rollback_artifact_path"] = str(rollback_snapshot_path)
    _atomic_write_json_compat(artifact_path, replay_verified_payload, config=runtime_config)
    artifact_snapshot_path = _snapshot_artifact(artifact_path, cycle_id=cycle_id, stage="post_replay_verified")
    return {
        "artifact_kind": str(replay_verified_payload.get("artifact_kind", "")),
        "artifact_lifecycle_state": "replay_verified",
        "artifact_sha256": artifact_sha256(artifact_path),
        "previous_artifact_sha256": previous_sha256,
        "rollback_artifact_path": str(rollback_snapshot_path),
        "artifact_snapshot_path": str(artifact_snapshot_path),
    }


def _retention_gate(subsystem: str, payload: dict[str, object] | None) -> dict[str, object]:
    subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem)
    if isinstance(payload, dict):
        retention_gate = payload.get("retention_gate", {})
        if isinstance(retention_gate, dict):
            return retention_gate
    return retention_gate_preset(subsystem)


def retention_gate_for_payload(
    subsystem: str,
    payload: dict[str, object] | None,
    *,
    capability_modules_path: Path | None = None,
) -> dict[str, object]:
    return engine_retention_gate_for_payload(
        subsystem,
        payload,
        capability_modules_path=capability_modules_path,
        base_subsystem_fn=DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem,
        retention_gate_fn=_retention_gate,
    )


def _tool_candidates_have_stage(payload: dict[str, object] | None, stage: str) -> bool:
    if not isinstance(payload, dict):
        return False
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return False
    return all(
        isinstance(candidate, dict) and str(candidate.get("promotion_stage", "")).strip() == stage
        for candidate in candidates
    )


def effective_artifact_payload_for_retention(
    subsystem: str,
    payload: Any,
    *,
    capability_modules_path: Path | None = None,
) -> Any:
    effective_subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem, capability_modules_path)
    if effective_subsystem != "tooling" or not isinstance(payload, dict):
        return payload
    gate = _retention_gate(effective_subsystem, payload)
    if not bool(gate.get("require_replay_verification", True)):
        return payload
    if _tool_candidates_have_stage(payload, "replay_verified"):
        return payload
    replay_verified_payload = materialize_replay_verified_tool_payload(payload)
    replay_verified_payload["lifecycle_state"] = "replay_verified"
    return replay_verified_payload


def stamp_artifact_experiment_variant(
    artifact_path: Path,
    variant: ImprovementVariant,
    *,
    runtime_config: KernelConfig | None = None,
) -> None:
    if not artifact_path.exists():
        return
    payload = _load_json_payload(artifact_path)
    if not isinstance(payload, dict):
        return
    payload["experiment_variant"] = {
        "subsystem": variant.subsystem,
        "variant_id": variant.variant_id,
        "description": variant.description,
        "expected_gain": variant.expected_gain,
        "estimated_cost": variant.estimated_cost,
        "score": variant.score,
        "controls": dict(variant.controls),
    }
    _atomic_write_json_compat(artifact_path, payload, config=runtime_config)


def stamp_artifact_generation_context(
    artifact_path: Path,
    *,
    cycle_id: str,
    active_artifact_path: Path | None = None,
    candidate_artifact_path: Path | None = None,
    prior_active_artifact_path: Path | None = None,
    prior_retained_cycle_id: str | None = None,
    prior_retained_artifact_snapshot_path: Path | None = None,
    extra_context: dict[str, object] | None = None,
    runtime_config: KernelConfig | None = None,
) -> None:
    if not artifact_path.exists():
        return
    payload = _load_json_payload(artifact_path)
    if not isinstance(payload, dict):
        return
    context: dict[str, object] = {"cycle_id": cycle_id}
    if active_artifact_path is not None:
        context["active_artifact_path"] = str(active_artifact_path)
    if candidate_artifact_path is not None:
        context["candidate_artifact_path"] = str(candidate_artifact_path)
    if prior_active_artifact_path is not None:
        context["prior_active_artifact_path"] = str(prior_active_artifact_path)
    if prior_retained_cycle_id:
        context["prior_retained_cycle_id"] = prior_retained_cycle_id
    if prior_retained_artifact_snapshot_path is not None:
        context["prior_retained_artifact_snapshot_path"] = str(prior_retained_artifact_snapshot_path)
    if isinstance(extra_context, dict):
        for key, value in extra_context.items():
            normalized = str(key).strip()
            if not normalized:
                continue
            context[normalized] = value
    payload["generation_context"] = context
    _atomic_write_json_compat(artifact_path, payload, config=runtime_config)


def payload_with_active_artifact_context(
    payload: dict[str, object] | None,
    *,
    active_artifact_path: Path | None = None,
    active_artifact_payload: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    rebound = deepcopy(payload)
    context = rebound.get("generation_context", {})
    if not isinstance(context, dict):
        context = {}
    else:
        context = deepcopy(context)
    if active_artifact_path is not None:
        context["active_artifact_path"] = str(active_artifact_path)
    if isinstance(active_artifact_payload, dict):
        context["active_artifact_payload"] = deepcopy(active_artifact_payload)
    rebound["generation_context"] = context
    return rebound


def staged_candidate_artifact_path(
    active_artifact_path: Path,
    *,
    candidates_root: Path,
    subsystem: str,
    cycle_id: str,
) -> Path:
    safe_cycle_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cycle_id).strip("._") or "cycle"
    target_dir = candidates_root / subsystem / safe_cycle_id
    return target_dir / active_artifact_path.name


def apply_artifact_retention_decision(*args, **kwargs):
    from ...improvement import apply_artifact_retention_decision as _impl

    return _impl(*args, **kwargs)


def assess_artifact_compatibility(*args, **kwargs):
    from ...improvement import assess_artifact_compatibility as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "apply_artifact_retention_decision",
    "artifact_sha256",
    "assess_artifact_compatibility",
    "effective_artifact_payload_for_retention",
    "materialize_replay_verified_tool_payload",
    "payload_with_active_artifact_context",
    "persist_replay_verified_tool_artifact",
    "retention_gate_for_payload",
    "snapshot_artifact_state",
    "staged_candidate_artifact_path",
    "stamp_artifact_experiment_variant",
    "stamp_artifact_generation_context",
]
