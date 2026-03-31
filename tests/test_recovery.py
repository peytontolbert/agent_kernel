import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.workspace_recovery import (
    recovery_annotation,
    recovery_policy_snapshot,
    should_restore_on_outcome,
)


def test_recovery_policy_snapshot_applies_retained_recovery_controls(tmp_path: Path):
    recovery_path = tmp_path / "recovery" / "recovery_proposals.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(
        json.dumps(
            {
                "artifact_kind": "recovery_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "snapshot_before_execution": True,
                    "rollback_on_runner_exception": True,
                    "rollback_on_failed_outcome": True,
                    "rollback_on_safe_stop": False,
                    "verify_post_rollback_file_count": True,
                    "max_post_rollback_file_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        recovery_proposals_path=recovery_path,
        unattended_rollback_on_failure=False,
    )

    snapshot = recovery_policy_snapshot(config)

    assert snapshot["snapshot_before_execution"] is True
    assert snapshot["rollback_on_runner_exception"] is True
    assert snapshot["rollback_on_failed_outcome"] is True
    assert snapshot["rollback_on_safe_stop"] is False
    assert snapshot["verify_post_rollback_file_count"] is True
    assert snapshot["max_post_rollback_file_count"] == 0
    assert should_restore_on_outcome(config, "safe_stop") is False
    assert should_restore_on_outcome(config, "unsafe_ambiguous") is True


def test_recovery_annotation_reports_post_rollback_verification(tmp_path: Path):
    recovery_path = tmp_path / "recovery" / "recovery_proposals.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(
        json.dumps(
            {
                "artifact_kind": "recovery_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "snapshot_before_execution": True,
                    "rollback_on_runner_exception": True,
                    "rollback_on_failed_outcome": True,
                    "rollback_on_safe_stop": True,
                    "verify_post_rollback_file_count": True,
                    "max_post_rollback_file_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "draft.txt").write_text("partial\n", encoding="utf-8")
    config = KernelConfig(recovery_proposals_path=recovery_path)

    annotation = recovery_annotation(
        config=config,
        workspace_snapshot_path=tmp_path / "snapshot",
        rollback_performed=True,
        rollback_reason="runner_exception",
        workspace_path=workspace,
    )

    assert annotation["rollback_performed"] is True
    assert annotation["verify_post_rollback_file_count"] is True
    assert annotation["post_rollback_file_count"] == 1
    assert annotation["rollback_verification_passed"] is False
