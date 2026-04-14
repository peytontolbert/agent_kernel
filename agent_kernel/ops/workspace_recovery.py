from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any

from ..config import KernelConfig
from .preflight import capture_workspace_snapshot
from ..extensions.improvement.recovery_improvement import retained_recovery_controls


def snapshot_workspace_tree(
    workspace: Path,
    snapshot_root: Path,
    *,
    task_id: str,
    run_label: str,
) -> Path:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    safe_task_id = _safe_name(task_id)
    safe_run_label = _safe_name(run_label)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = snapshot_root / f"{safe_task_id}_{safe_run_label}_{timestamp}"
    if workspace.exists():
        shutil.copytree(workspace, target)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target


def restore_workspace_tree(snapshot_path: Path, workspace: Path) -> Path:
    if workspace.exists():
        shutil.rmtree(workspace)
    if snapshot_path.exists():
        shutil.copytree(snapshot_path, workspace)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def annotate_task_report_recovery(report_path: Path, recovery: dict[str, Any]) -> None:
    if not report_path.exists():
        return
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    payload["recovery"] = dict(recovery)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def recovery_policy_snapshot(config: KernelConfig) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "snapshot_before_execution": True,
        "rollback_on_runner_exception": bool(config.unattended_rollback_on_failure),
        "rollback_on_failed_outcome": bool(config.unattended_rollback_on_failure),
        "rollback_on_safe_stop": bool(config.unattended_rollback_on_failure),
        "verify_post_rollback_file_count": False,
        "max_post_rollback_file_count": 0,
    }
    retained = _retained_recovery_controls(config)
    if retained:
        for key in (
            "snapshot_before_execution",
            "rollback_on_runner_exception",
            "rollback_on_failed_outcome",
            "rollback_on_safe_stop",
            "verify_post_rollback_file_count",
        ):
            if key in retained:
                policy[key] = bool(retained[key])
        if "max_post_rollback_file_count" in retained:
            try:
                policy["max_post_rollback_file_count"] = max(0, int(retained["max_post_rollback_file_count"]))
            except (TypeError, ValueError):
                policy["max_post_rollback_file_count"] = 0
    return policy


def should_snapshot_workspace(config: KernelConfig) -> bool:
    return bool(recovery_policy_snapshot(config).get("snapshot_before_execution", True))


def should_restore_on_runner_exception(config: KernelConfig) -> bool:
    return bool(recovery_policy_snapshot(config).get("rollback_on_runner_exception", False))


def should_restore_on_outcome(config: KernelConfig, outcome: str) -> bool:
    policy = recovery_policy_snapshot(config)
    normalized = str(outcome).strip()
    if normalized == "safe_stop":
        return bool(policy.get("rollback_on_safe_stop", False))
    if normalized != "success":
        return bool(policy.get("rollback_on_failed_outcome", False))
    return False


def recovery_annotation(
    *,
    config: KernelConfig,
    workspace_snapshot_path: Path | None,
    rollback_performed: bool,
    rollback_reason: str,
    workspace_path: Path,
) -> dict[str, Any]:
    policy = recovery_policy_snapshot(config)
    post_rollback_file_count = len(capture_workspace_snapshot(workspace_path))
    verification_passed = True
    if rollback_performed and bool(policy.get("verify_post_rollback_file_count", False)):
        verification_passed = post_rollback_file_count <= int(policy.get("max_post_rollback_file_count", 0))
    return {
        "workspace_snapshot_path": "" if workspace_snapshot_path is None else str(workspace_snapshot_path),
        "snapshot_before_execution": bool(policy.get("snapshot_before_execution", True)),
        "rollback_enabled": any(
            bool(policy.get(key, False))
            for key in ("rollback_on_runner_exception", "rollback_on_failed_outcome", "rollback_on_safe_stop")
        ),
        "rollback_performed": bool(rollback_performed),
        "rollback_reason": rollback_reason,
        "verify_post_rollback_file_count": bool(policy.get("verify_post_rollback_file_count", False)),
        "max_post_rollback_file_count": int(policy.get("max_post_rollback_file_count", 0)),
        "rollback_verification_passed": bool(verification_passed),
        "post_rollback_file_count": post_rollback_file_count,
    }


def _retained_recovery_controls(config: KernelConfig) -> dict[str, object]:
    if not bool(config.use_recovery_proposals):
        return {}
    path = config.recovery_proposals_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return retained_recovery_controls(payload)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
