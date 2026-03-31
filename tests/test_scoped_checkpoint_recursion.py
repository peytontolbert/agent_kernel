from pathlib import Path

from agent_kernel.config import KernelConfig
from evals.harness import scoped_improvement_cycle_config


def test_scoped_improvement_cycle_config_does_not_seed_recursive_checkpoint_tree(tmp_path: Path) -> None:
    base_checkpoints = tmp_path / "trajectories" / "checkpoints"
    nested_scope_dir = base_checkpoints / "runner_a"
    nested_scope_dir.mkdir(parents=True, exist_ok=True)
    (base_checkpoints / "root_checkpoint.json").write_text("root\n", encoding="utf-8")
    (nested_scope_dir / "prior_scoped_checkpoint.json").write_text("nested\n", encoding="utf-8")

    base_snapshots = tmp_path / "trajectories" / "recovery" / "workspaces"
    nested_snapshot_dir = base_snapshots / "runner_a"
    nested_snapshot_dir.mkdir(parents=True, exist_ok=True)
    (base_snapshots / "base_snapshot.txt").write_text("snapshot\n", encoding="utf-8")
    (nested_snapshot_dir / "prior_scoped_snapshot.txt").write_text("nested snapshot\n", encoding="utf-8")

    base_config = KernelConfig(
        workspace_root=tmp_path / "workspace",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        improvement_reports_dir=tmp_path / "improvement" / "reports",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "learning_artifacts.json",
        run_checkpoints_dir=base_checkpoints,
        delegated_job_queue_path=tmp_path / "jobs" / "queue.json",
        delegated_job_runtime_state_path=tmp_path / "jobs" / "runtime_state.json",
        unattended_workspace_snapshot_root=base_snapshots,
        unattended_trust_ledger_path=tmp_path / "trust" / "ledger.json",
        tolbert_model_artifact_path=tmp_path / "tolbert" / "tolbert_model_artifact.json",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert" / "datasets",
        tolbert_liftoff_report_path=tmp_path / "tolbert" / "liftoff_gate_report.json",
    )

    scoped = scoped_improvement_cycle_config(base_config, "runner_a")

    assert scoped.run_checkpoints_dir == nested_scope_dir
    assert scoped.run_checkpoints_dir.exists()
    assert not (scoped.run_checkpoints_dir / "root_checkpoint.json").exists()
    assert (base_checkpoints / "root_checkpoint.json").exists()

    assert scoped.unattended_workspace_snapshot_root == nested_snapshot_dir
    assert scoped.unattended_workspace_snapshot_root.exists()
    assert not (scoped.unattended_workspace_snapshot_root / "base_snapshot.txt").exists()
    assert (base_snapshots / "base_snapshot.txt").exists()
