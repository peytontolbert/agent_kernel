from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .artifact_support_evidence import normalized_tool_candidates_for_retention
from .improvement_common import retention_gate_preset
from .improvement_plugins import DEFAULT_IMPROVEMENT_PLUGIN_LAYER


def load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_artifact(artifact_path: Path, *, cycle_id: str, stage: str) -> Path:
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


def update_tool_candidate_states(
    payload: dict[str, object],
    *,
    decision_state: str,
    lifecycle_state: str,
) -> None:
    del lifecycle_state
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return
    if decision_state == "retain":
        payload["candidates"] = normalized_tool_candidates_for_retention(candidates)
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if decision_state == "retain":
            candidate["promotion_stage"] = "promoted_tool"
            candidate["lifecycle_state"] = "retained"
        else:
            candidate["promotion_stage"] = "rejected"
            candidate["lifecycle_state"] = "rejected"


def retention_gate(subsystem: str, payload: dict[str, object] | None) -> dict[str, object]:
    subsystem = DEFAULT_IMPROVEMENT_PLUGIN_LAYER.base_subsystem(subsystem)
    if isinstance(payload, dict):
        value = payload.get("retention_gate", {})
        if isinstance(value, dict):
            return value
    return retention_gate_preset(subsystem)


def verifier_contracts_are_strict(payload: dict[str, object] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list) or not proposals:
        return False
    return all(
        isinstance(proposal, dict)
        and isinstance(proposal.get("evidence", {}), dict)
        and bool(proposal.get("evidence", {}).get("strict_contract", False))
        for proposal in proposals
    )


def tool_candidates_have_stage(payload: dict[str, object] | None, stage: str) -> bool:
    if not isinstance(payload, dict):
        return False
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return False
    return all(
        isinstance(candidate, dict) and str(candidate.get("promotion_stage", "")).strip() == stage
        for candidate in candidates
    )


def prior_active_artifact_path(payload: Any) -> Path | None:
    if not isinstance(payload, dict):
        return None
    context = payload.get("generation_context", {})
    if not isinstance(context, dict):
        return None
    value = str(context.get("prior_active_artifact_path", "")).strip()
    if not value:
        return None
    return Path(value)


def average_metric_delta(
    records: list[dict[str, object]],
    *,
    baseline_key: str,
    candidate_key: str,
) -> float:
    deltas: list[float] = []
    for record in records:
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            continue
        baseline = metrics_summary.get(baseline_key)
        candidate = metrics_summary.get(candidate_key)
        if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
            deltas.append(float(candidate) - float(baseline))
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)
