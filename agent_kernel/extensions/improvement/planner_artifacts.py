from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def artifact_history(planner: Any, output_path: Path, artifact_path: Path | str) -> list[dict[str, object]]:
    normalized = str(artifact_path)
    return [
        record
        for record in planner.load_cycle_records(output_path)
        if str(record.get("artifact_path", "")) == normalized
    ]


def latest_artifact_record(
    planner: Any,
    output_path: Path,
    artifact_path: Path | str,
) -> dict[str, object] | None:
    history = artifact_history(planner, output_path, artifact_path)
    if not history:
        return None
    return history[-1]


def latest_artifact_decision(
    planner: Any,
    output_path: Path,
    artifact_path: Path | str,
) -> dict[str, object] | None:
    history = [
        record
        for record in artifact_history(planner, output_path, artifact_path)
        if str(record.get("state", "")) in {"retain", "reject"}
    ]
    if not history:
        return None
    return history[-1]


def artifact_rollback_metadata(planner: Any, output_path: Path, artifact_path: Path | str) -> dict[str, str]:
    latest = latest_artifact_record(planner, output_path, artifact_path)
    if latest is None:
        return {}
    return {
        "artifact_sha256": str(latest.get("artifact_sha256", "")),
        "previous_artifact_sha256": str(latest.get("previous_artifact_sha256", "")),
        "rollback_artifact_path": str(latest.get("rollback_artifact_path", "")),
        "artifact_snapshot_path": str(latest.get("artifact_snapshot_path", "")),
    }


def prior_retained_artifact_record(
    planner: Any,
    output_path: Path,
    subsystem: str,
    *,
    before_cycle_id: str | None = None,
) -> dict[str, object] | None:
    records = [
        record
        for record in planner.load_cycle_records(output_path)
        if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
        and str(record.get("state", "")) == "retain"
    ]
    if before_cycle_id is not None:
        all_records = planner.load_cycle_records(output_path)
        cutoff_index = next(
            (
                index
                for index, record in enumerate(all_records)
                if str(record.get("cycle_id", "")) == before_cycle_id
            ),
            None,
        )
        if cutoff_index is not None:
            eligible_cycle_ids = {
                str(record.get("cycle_id", ""))
                for record in all_records[:cutoff_index]
                if planner._subsystems_match(str(record.get("subsystem", "")), subsystem)
                and str(record.get("state", "")) == "retain"
            }
            records = [
                record for record in records if str(record.get("cycle_id", "")) in eligible_cycle_ids
            ]
    if not records:
        return None
    return records[-1]


def rollback_artifact(planner: Any, output_path: Path, artifact_path: Path | str) -> Path:
    from ... import improvement as core

    latest = latest_artifact_record(planner, output_path, artifact_path)
    if latest is None:
        raise ValueError("no cycle history exists for the requested artifact")
    snapshot_path = Path(str(latest.get("rollback_artifact_path", "")))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"rollback snapshot does not exist: {snapshot_path}")
    destination = Path(str(artifact_path))
    core.atomic_copy_file(snapshot_path, destination, config=planner.runtime_config)
    subsystem = str(latest.get("subsystem", "")).strip()
    if subsystem in {"universe", "universe_constitution", "operating_envelope"}:
        restored_payload = core._load_json_payload(destination)
        core._synchronize_retained_universe_artifacts(
            subsystem=subsystem,
            payload=restored_payload,
            live_artifact_path=destination,
            runtime_config=core._runtime_config_for_universe_sync(planner.runtime_config, destination),
        )
    write_rollback_revalidation_receipt(
        planner,
        destination=destination,
        snapshot_path=snapshot_path,
        latest_record=latest,
    )
    return destination


def write_rollback_revalidation_receipt(
    planner: Any,
    *,
    destination: Path,
    snapshot_path: Path,
    latest_record: dict[str, object],
) -> Path:
    from ... import improvement as core

    trust_summary = planner.trust_ledger_summary()
    phase_gate_failures = core._record_phase_gate_failures(latest_record)
    reasons = ["artifact_rollback_restores_files_not_world_state"]
    if phase_gate_failures:
        reasons.append("prior_phase_gate_failures_present")
    if float(trust_summary.get("hidden_side_effect_risk_rate", 0.0)) > 0.0:
        reasons.append("trust_hidden_side_effect_risk_present")
    if float(trust_summary.get("rollback_performed_rate", 0.0)) > 0.0:
        reasons.append("trust_history_requires_rollback")
    if float(trust_summary.get("false_pass_risk_rate", 0.0)) > 0.0:
        reasons.append("trust_false_pass_risk_present")
    if float(trust_summary.get("unexpected_change_report_rate", 0.0)) > 0.0:
        reasons.append("unexpected_change_reports_present")
    receipt_path = destination.with_name(f"{destination.name}.rollback_receipt.json")
    receipt_payload = {
        "receipt_kind": "artifact_rollback_revalidation_receipt",
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(destination),
        "rollback_snapshot_path": str(snapshot_path),
        "cycle_id": str(latest_record.get("cycle_id", "")).strip(),
        "subsystem": str(latest_record.get("subsystem", "")).strip(),
        "world_state_revalidation_required": True,
        "revalidation_scope": [
            "workspace_side_effects",
            "runtime_environment",
            "external_services_and_caches",
            "trust_ledger_alignment",
        ],
        "reasons": reasons,
        "phase_gate_failures": phase_gate_failures,
        "trust_summary": trust_summary,
    }
    core.atomic_write_json(receipt_path, receipt_payload, config=planner.runtime_config)
    return receipt_path
