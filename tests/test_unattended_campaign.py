from pathlib import Path
import importlib.util
import json
from io import StringIO
import os
import sys
import threading
from types import SimpleNamespace

from agent_kernel.config import KernelConfig
from agent_kernel.sandbox import Sandbox
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.ops.unattended_controller import default_controller_state, normalize_controller_state, update_controller_state


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_runtime_state_removes_nested_snapshot_scopes(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    nested_roots = [
        config.unattended_workspace_snapshot_root / "generated_success" / "generated_success",
        config.unattended_workspace_snapshot_root / "generated_success" / "generated_failure",
        config.unattended_workspace_snapshot_root / "generated_failure" / "generated_success",
        config.unattended_workspace_snapshot_root / "generated_failure" / "generated_failure",
    ]
    for path in nested_roots:
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.txt").write_text("snapshot", encoding="utf-8")

    cleanup = module._cleanup_runtime_state(
        config,
        keep_reports=10,
        keep_candidate_dirs=10,
        keep_checkpoints=10,
        keep_snapshot_entries=10,
        keep_tolbert_dataset_dirs=10,
    )

    assert sorted(cleanup["removed_nested_scopes"]) == sorted(str(path) for path in nested_roots)
    for path in nested_roots:
        assert not path.exists()


def test_cleanup_runtime_state_prunes_unreferenced_tolbert_shared_store(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    tolbert_artifact_path = tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json"
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        tolbert_model_artifact_path=tolbert_artifact_path,
    )
    keep_dir = tolbert_artifact_path.parent / "store" / "dataset" / "keep_digest"
    drop_dir = tolbert_artifact_path.parent / "store" / "dataset" / "drop_digest"
    keep_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (drop_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    tolbert_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tolbert_artifact_path.write_text(
        json.dumps(
            {
                "shared_store": {
                    "entries": {
                        "dataset": {"path": str(keep_dir)}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cleanup = module._cleanup_runtime_state(
        config,
        keep_reports=10,
        keep_candidate_dirs=10,
        keep_checkpoints=10,
        keep_snapshot_entries=10,
        keep_tolbert_dataset_dirs=10,
    )

    assert keep_dir.exists()
    assert not drop_dir.exists()
    assert str(drop_dir) in cleanup["removed_tolbert_shared_store"]


def test_cleanup_tolbert_model_nested_storage_enforces_candidate_and_store_budgets(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    tolbert_artifact_path = tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json"
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        tolbert_model_artifact_path=tolbert_artifact_path,
    )
    store_root = tolbert_artifact_path.parent / "store" / "training"
    keep_store = store_root / "digest_keep"
    drop_store_1 = store_root / "digest_drop_1"
    drop_store_2 = store_root / "digest_drop_2"
    for path in (keep_store, drop_store_1, drop_store_2):
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoint.pt").write_bytes(b"x" * 64)

    tolbert_candidates_root = config.candidate_artifacts_root / "tolbert_model"
    cycle_paths = [
        tolbert_candidates_root / "cycle_oldest",
        tolbert_candidates_root / "cycle_middle",
        tolbert_candidates_root / "cycle_newest",
    ]
    store_paths = [drop_store_2, drop_store_1, keep_store]
    for index, (cycle_path, store_path) in enumerate(zip(cycle_paths, store_paths)):
        cycle_path.mkdir(parents=True, exist_ok=True)
        (cycle_path / "tolbert_model_artifact.json").write_text(
            json.dumps(
                {
                    "shared_store": {
                        "entries": {
                            "training": {"path": str(store_path)}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (cycle_path / "padding.bin").write_bytes(b"p" * 80)
        os.utime(cycle_path, (100 + index, 100 + index))

    cleanup = module._cleanup_tolbert_model_nested_storage(
        config,
        keep_candidate_dirs=2,
        candidate_budget_bytes=400,
        shared_store_budget_bytes=200,
    )

    assert not cycle_paths[0].exists()
    assert not cycle_paths[1].exists()
    assert cycle_paths[2].exists()
    assert keep_store.exists()
    assert not drop_store_1.exists()
    assert not drop_store_2.exists()
    assert cleanup["after_candidate_bytes"] < cleanup["before_candidate_bytes"]
    assert cleanup["after_shared_store_bytes"] < cleanup["before_shared_store_bytes"]


def test_cleanup_tolbert_promoted_checkpoints_prunes_unreferenced_files(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    tolbert_artifact_path = tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json"
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        tolbert_model_artifact_path=tolbert_artifact_path,
    )
    checkpoints_root = tolbert_artifact_path.parent / "checkpoints"
    keep_checkpoint = checkpoints_root / "tolbert_keep.pt"
    old_checkpoint = checkpoints_root / "tolbert_old.pt"
    newer_checkpoint = checkpoints_root / "tolbert_newer.pt"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    keep_checkpoint.write_bytes(b"k" * 64)
    old_checkpoint.write_bytes(b"o" * 64)
    newer_checkpoint.write_bytes(b"n" * 64)
    tolbert_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tolbert_artifact_path.write_text(
        json.dumps(
            {
                "runtime_paths": {
                    "checkpoint_path": str(keep_checkpoint)
                }
            }
        ),
        encoding="utf-8",
    )
    os.utime(keep_checkpoint, (300, 300))
    os.utime(newer_checkpoint, (200, 200))
    os.utime(old_checkpoint, (100, 100))

    cleanup = module._cleanup_tolbert_promoted_checkpoints(
        config,
        keep_checkpoints=1,
        budget_bytes=1000,
    )

    assert keep_checkpoint.exists()
    assert not old_checkpoint.exists()
    assert cleanup["after_bytes"] < cleanup["before_bytes"]


def test_directory_usage_bytes_tolerates_walk_time_file_not_found(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    real_walk = module.os.walk

    def flaky_walk(path, onerror=None):
        yielded = False
        for item in real_walk(path, onerror=onerror):
            if not yielded:
                yielded = True
                yield item
                raise FileNotFoundError("disappearing nested directory")
            yield item

    monkeypatch.setattr(module.os, "walk", flaky_walk)

    assert module._directory_usage_bytes(root) == len("hello")


def test_storage_governance_snapshot_falls_back_to_shallow_usage_on_du_timeout(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "top.txt").write_text("hello", encoding="utf-8")
    nested = reports_dir / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("world", encoding="utf-8")
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    def timeout_run(*args, **kwargs):
        raise module.subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0] if args else "du"), timeout=5.0)

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    snapshot = module._storage_governance_snapshot(config)
    tracked = snapshot["tracked_directories"]["reports"]

    assert tracked["bytes"] == len("hello")
    assert tracked["partial"] is True
    assert tracked["method"] == "shallow_timeout"


def test_global_storage_snapshot_marks_managed_entries_partial_on_du_timeout(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    global_root = tmp_path / "data"
    managed = global_root / "checkpoints"
    managed.mkdir(parents=True, exist_ok=True)
    (managed / "top.bin").write_bytes(b"hello")
    nested = managed / "nested"
    nested.mkdir()
    (nested / "deep.bin").write_bytes(b"world")

    def timeout_run(*args, **kwargs):
        raise module.subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0] if args else "du"), timeout=5.0)

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    snapshot = module._global_storage_snapshot(
        global_root,
        top_k=0,
        managed_entries=[
            {
                "label": "checkpoints",
                "path": str(managed),
                "keep_entries": 1,
                "enabled": True,
            }
        ],
    )
    managed_entry = snapshot["managed_entries"][0]

    assert managed_entry["bytes"] == len(b"hello")
    assert managed_entry["partial"] is True
    assert managed_entry["method"] == "shallow_timeout"


def test_global_storage_snapshot_uses_shallow_top_level_ranking_for_top_consumers(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    global_root = tmp_path / "data"
    global_root.mkdir()
    large_dir = global_root / "large"
    large_dir.mkdir()
    (large_dir / "top.bin").write_bytes(b"x" * 9)
    nested = large_dir / "nested"
    nested.mkdir()
    (nested / "deep.bin").write_bytes(b"y" * 50)
    small_dir = global_root / "small"
    small_dir.mkdir()
    (small_dir / "top.bin").write_bytes(b"z" * 3)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("top-consumer snapshot should not invoke subprocess.run")

    monkeypatch.setattr(module.subprocess, "run", unexpected_run)

    snapshot = module._global_storage_snapshot(global_root, top_k=2, managed_entries=[])

    assert [Path(item["path"]).name for item in snapshot["top_consumers"]] == ["large", "small"]
    assert snapshot["top_consumers"][0]["bytes"] == 9
    assert snapshot["top_consumers"][0]["partial"] is True
    assert snapshot["top_consumers"][0]["method"] == "shallow_top_level"


def test_governed_cleanup_runtime_state_skips_when_disk_is_already_above_target(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        workspace_root=tmp_path,
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 10**12,
            "free_gib": 500.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ample disk",
        },
    )
    monkeypatch.setattr(
        module,
        "_storage_governance_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("storage snapshot should be skipped")),
    )
    monkeypatch.setattr(
        module,
        "_cleanup_runtime_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cleanup should be skipped")),
    )

    result = module._governed_cleanup_runtime_state(
        config,
        keep_reports=50,
        keep_candidate_dirs=20,
        keep_checkpoints=20,
        keep_snapshot_entries=100,
        keep_tolbert_dataset_dirs=10,
        target_free_gib=75.0,
    )

    assert result["before_disk"]["free_gib"] == 500.0
    assert result["after_disk"]["free_gib"] == 500.0
    assert result["cleanup"]["skipped"] is True
    assert result["cleanup"]["reason"] == "disk already above target"
    assert result["before_storage"] == {}
    assert result["after_storage"] == {}


def test_governed_global_storage_cleanup_skips_when_disk_is_already_above_target(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    root = tmp_path / "global"
    root.mkdir()
    managed = root / "managed"
    managed.mkdir()

    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 10**12,
            "free_gib": 500.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ample disk",
        },
    )
    monkeypatch.setattr(
        module,
        "_global_storage_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("global snapshot should be skipped")),
    )
    monkeypatch.setattr(
        module,
        "_cleanup_global_storage_entries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("global cleanup should be skipped")),
    )

    result = module._governed_global_storage_cleanup(
        root,
        policy_entries=[
            {
                "label": "managed",
                "path": str(managed),
                "keep_entries": 1,
                "enabled": True,
            }
        ],
        target_free_gib=150.0,
        top_k=5,
    )

    assert result["before_disk"]["free_gib"] == 500.0
    assert result["after_disk"]["free_gib"] == 500.0
    assert result["cleanup"]["skipped"] is True
    assert result["cleanup"]["reason"] == "disk already above target"
    assert result["before_snapshot"] == {}
    assert result["after_snapshot"] == {}


def test_run_unattended_campaign_safe_stops_on_low_disk(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 0,
            "free_gib": 0.0,
            "min_free_gib": min_free_gib,
            "passed": False,
            "detail": "disk below threshold",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "unattended_campaign_report"
    assert payload["status"] == "safe_stop"
    assert payload["phase"] == "preflight"
    assert payload["reason"] == "disk below threshold"


def test_run_unattended_campaign_safe_stops_when_strict_live_llm_provider_is_bounded_decoder(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        provider="hybrid",
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 10**9,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["phase"] == "preflight"
    assert payload["runtime_claim"]["asi_coding_require_live_llm"] is True
    assert "requires external decoder authority" in payload["reason"]
    assert payload["rounds"][0]["preflight"]["decoder_contract"]["passed"] is False
    assert payload["rounds"][0]["preflight"]["decoder_contract"]["decoder_runtime_posture"]["provider"] == "hybrid"


def test_run_unattended_campaign_safe_stops_on_unexpected_cleanup_exception(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 10**9,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        module,
        "_governed_cleanup_runtime_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup exploded")),
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["phase"] == "cleanup"
    assert "cleanup exploded" in payload["reason"]
    assert payload["rounds"][0]["status"] == "safe_stop"
    assert payload["unexpected_exception"]["type"] == "RuntimeError"


def test_run_unattended_campaign_writes_status_and_alerts_on_safe_stop(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    status_path = reports_dir / "campaign.status.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    seen_alerts: list[dict[str, object]] = []

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 0,
            "free_gib": 0.0,
            "min_free_gib": min_free_gib,
            "passed": False,
            "detail": "disk below threshold",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        module,
        "_run_alert_command",
        lambda command, *, report_path, status_path, event_log_path, payload: (
            seen_alerts.append(
                {
                    "command": command,
                    "report_path": str(report_path),
                    "status_path": "" if status_path is None else str(status_path),
                    "event_log_path": "" if event_log_path is None else str(event_log_path),
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                }
            )
            or {"ran": True, "command": command, "returncode": 0}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--status-path",
            str(status_path),
            "--alert-command",
            "echo campaign-alert",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert status_payload["report_kind"] == "unattended_campaign_status"
    assert status_payload["status"] == "safe_stop"
    assert status_payload["reason"] == "disk below threshold"
    assert seen_alerts
    assert seen_alerts[0]["command"] == "echo campaign-alert"
    assert seen_alerts[0]["status"] == "safe_stop"


def test_run_unattended_campaign_defaults_global_storage_root_to_repo_root(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    seen_root: dict[str, str] = {}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 10**9,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_governed_global_storage_cleanup",
        lambda root, **kwargs: (
            seen_root.setdefault("value", str(root)),
            {"before_disk": {}, "after_disk": {}, "before_snapshot": {}, "after_snapshot": {}, "cleanup": {}},
        )[1],
    )
    monkeypatch.setattr(
        module,
        "_run_and_stream",
        lambda *args, **kwargs: {"returncode": 2, "stdout": "", "timed_out": False},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    assert seen_root["value"] == str(Path(module.__file__).resolve().parents[1])


def test_run_unattended_campaign_surfaces_unattended_evidence_in_report_and_status(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    status_path = reports_dir / "campaign.status.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(
        module,
        "build_unattended_trust_ledger",
        lambda cfg: {
            "generated_at": "2026-03-28T00:00:00+00:00",
            "reports_considered": 4,
            "policy": {"required_benchmark_families": ["repo_chore", "repo_sandbox"]},
            "overall_assessment": {
                "passed": False,
                "status": "restricted",
                "detail": "overall gated trust restricted unattended execution",
                "failing_thresholds": ["unsafe_ambiguous_rate=0.250 above max_unsafe_ambiguous_rate=0.100"],
            },
            "overall_summary": {
                "total": 4,
                "success_count": 3,
                "success_rate": 0.75,
                "unsafe_ambiguous_count": 1,
                "unsafe_ambiguous_rate": 0.25,
                "hidden_side_effect_risk_rate": 0.25,
                "false_pass_risk_rate": 0.25,
                "clean_success_streak": 1,
                "distinct_benchmark_families": 2,
                "benchmark_families": ["repo_chore", "repo_sandbox"],
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
                "external_benchmark_families": ["repo_sandbox"],
            },
            "gated_summary": {
                "total": 4,
                "success_count": 3,
                "success_rate": 0.75,
                "safe_stop_count": 0,
                "safe_stop_rate": 0.0,
                "unsafe_ambiguous_count": 1,
                "unsafe_ambiguous_rate": 0.25,
                "hidden_side_effect_risk_rate": 0.25,
                "success_hidden_side_effect_risk_rate": 0.0,
                "false_pass_risk_rate": 0.25,
                "clean_success_streak": 1,
                "distinct_benchmark_families": 2,
                "benchmark_families": ["repo_chore", "repo_sandbox"],
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
                "external_benchmark_families": ["repo_sandbox"],
            },
            "external_summary": {
                "total": 1,
                "success_count": 1,
                "success_rate": 1.0,
                "safe_stop_count": 0,
                "safe_stop_rate": 0.0,
                "unsafe_ambiguous_count": 0,
                "unsafe_ambiguous_rate": 0.0,
                "hidden_side_effect_risk_rate": 0.0,
                "success_hidden_side_effect_risk_rate": 0.0,
                "false_pass_risk_rate": 0.0,
                "clean_success_streak": 1,
                "distinct_benchmark_families": 1,
                "benchmark_families": ["repo_sandbox"],
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
                "external_benchmark_families": ["repo_sandbox"],
            },
            "family_summaries": {
                "repo_chore": {
                    "total": 3,
                    "success_rate": 2 / 3,
                    "unsafe_ambiguous_rate": 1 / 3,
                    "hidden_side_effect_risk_rate": 1 / 3,
                    "false_pass_risk_rate": 1 / 3,
                    "clean_success_streak": 0,
                    "distinct_clean_success_task_roots": 1,
                    "clean_success_task_roots": ["repo_sync_matrix_task"],
                },
                "repo_sandbox": {
                    "total": 1,
                    "success_rate": 1.0,
                    "unsafe_ambiguous_rate": 0.0,
                    "hidden_side_effect_risk_rate": 0.0,
                    "false_pass_risk_rate": 0.0,
                    "clean_success_streak": 1,
                    "distinct_clean_success_task_roots": 2,
                    "clean_success_task_roots": ["repo_sandbox_task_a", "repo_sandbox_task_b"],
                },
            },
            "family_assessments": {
                "repo_chore": {
                    "passed": False,
                    "status": "restricted",
                    "detail": "family trust restricted",
                    "failing_thresholds": ["unsafe_ambiguous_rate=0.333 above max_unsafe_ambiguous_rate=0.100"],
                },
                "repo_sandbox": {
                    "passed": True,
                    "status": "trusted",
                    "detail": "family trust passed",
                    "failing_thresholds": [],
                },
            },
            "coverage_summary": {
                "required_family_clean_task_root_counts": {
                    "repo_chore": 1,
                    "repo_sandbox": 2,
                }
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 0,
            "free_gib": 0.0,
            "min_free_gib": min_free_gib,
            "passed": False,
            "detail": "disk below threshold",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--status-path", str(status_path)])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["unattended_evidence"]["reports_considered"] == 4
    assert payload["unattended_evidence"]["overall_assessment"]["status"] == "restricted"
    assert payload["unattended_evidence"]["gated_summary"]["false_pass_risk_rate"] == 0.25
    assert payload["unattended_evidence"]["required_family_statuses"]["repo_chore"]["status"] == "restricted"
    assert payload["unattended_evidence"]["required_family_statuses"]["repo_sandbox"]["reports"] == 1
    assert payload["unattended_evidence"]["required_family_statuses"]["repo_chore"]["distinct_clean_success_task_roots"] == 1
    assert payload["unattended_evidence"]["required_family_statuses"]["repo_chore"]["clean_success_task_roots"] == [
        "repo_sync_matrix_task"
    ]
    assert payload["unattended_evidence"]["family_breadth_min_distinct_task_roots"] == 0
    assert payload["unattended_evidence"]["required_family_clean_task_root_counts"]["repo_chore"] == 1
    assert payload["unattended_evidence"]["required_families_missing_clean_task_root_breadth"] == []
    assert payload["unattended_evidence"]["semantic_hub_summary"]["reports"] == 0
    assert payload["unattended_evidence"]["replay_derived_summary"]["reports"] == 0
    assert payload["unattended_evidence"]["open_world_summary"]["external_report_count"] == 1
    assert payload["unattended_evidence"]["open_world_summary"]["open_world_report_count"] == 1
    assert status_payload["unattended_evidence"]["overall_summary"]["distinct_benchmark_families"] == 2
    assert status_payload["unattended_evidence"]["external_summary"]["total"] == 1
    assert status_payload["unattended_evidence"]["required_families_missing_reports"] == []


def test_run_unattended_campaign_runs_campaign_and_auto_liftoff(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        trajectories_root=tmp_path / "episodes",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    liftoff_report_path = reports_dir / "liftoff_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            }
        ),
        encoding="utf-8",
    )
    liftoff_report_path.write_text(
        json.dumps(
            {
                "report_kind": "tolbert_liftoff_loop_report",
                "status": "completed",
                "liftoff_report": {"state": "shadow_only", "reason": "candidate did not clear the liftoff pass-rate delta gate"},
                "dataset_readiness": {"allow_kernel_autobuild": True},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        joined = " ".join(cmd)
        if "run_repeated_improvement_cycles.py" in joined:
            return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}
        if "run_tolbert_liftoff_loop.py" in joined:
            assert "--skip-improvement" in cmd
            assert "--task-limit" in cmd
            assert "64" in cmd
            assert "--tolbert-device" in cmd
            assert "cuda:2" in cmd
            return {"returncode": 0, "stdout": f"{liftoff_report_path}\n"}
        raise AssertionError(joined)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--cycles",
            "2",
            "--task-limit",
            "64",
            "--tolbert-device",
            "cuda:2",
            "--apply-routing",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["campaign_report_path"] == str(campaign_report_path)
    assert payload["liftoff_report_path"] == str(liftoff_report_path)
    assert payload["liftoff_report"]["liftoff_report"]["state"] == "shadow_only"


def test_run_unattended_campaign_skips_liftoff_without_retention(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 0},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        joined = " ".join(cmd)
        assert "run_repeated_improvement_cycles.py" in joined
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--task-limit", "32"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["liftoff_skipped"]["reason"] == "campaign did not satisfy auto liftoff trigger"
    assert "liftoff_run" not in payload


def test_run_unattended_campaign_semantic_only_runtime_disables_tolbert_and_skips_liftoff(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            }
        ),
        encoding="utf-8",
    )
    seen_cmds = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, kwargs
        seen_cmds.append((cmd, env))
        joined = " ".join(cmd)
        if "run_repeated_improvement_cycles.py" in joined:
            assert "--semantic-only-runtime" in cmd
            exclude_indexes = [index for index, token in enumerate(cmd) if token == "--exclude-subsystem"]
            assert exclude_indexes
            assert any(cmd[index + 1] == "tolbert_model" for index in exclude_indexes)
            assert env["AGENT_KERNEL_USE_TOLBERT_CONTEXT"] == "0"
            assert env["AGENT_KERNEL_USE_TOLBERT_MODEL_ARTIFACTS"] == "0"
            return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}
        raise AssertionError(joined)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--semantic-only-runtime",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["campaign_report_path"] == str(campaign_report_path)
    assert len(seen_cmds) == 1
    assert payload["current_policy"]["semantic_only_runtime"] is True
    assert payload["current_policy"]["liftoff"] == "never"


def test_run_unattended_campaign_adapts_policy_between_rounds(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        joined = " ".join(cmd)
        assert "run_repeated_improvement_cycles.py" in joined
        seen_campaign_cmds.append(cmd)
        report_path = round_one_report if len(seen_campaign_cmds) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--campaign-width",
            "2",
            "--variant-width",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert len(seen_campaign_cmds) == 2
    assert seen_campaign_cmds[0][seen_campaign_cmds[0].index("--cycles") + 1] == "1"
    assert seen_campaign_cmds[1][seen_campaign_cmds[1].index("--cycles") + 1] == "2"
    assert seen_campaign_cmds[0][seen_campaign_cmds[0].index("--task-limit") + 1] == "32"
    assert seen_campaign_cmds[1][seen_campaign_cmds[1].index("--task-limit") + 1] == "128"


def test_run_unattended_campaign_adapts_policy_from_planner_pressure_summary(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.05,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                "planner_pressure_summary": {
                    "campaign_breadth_pressure_cycles": 1,
                    "variant_breadth_pressure_cycles": 1,
                    "recent_campaign_pressures": [
                        {"cycle_id": "cycle:retrieval:1", "subsystem": "retrieval", "campaign_breadth_pressure": 0.4}
                    ],
                    "recent_variant_pressures": [
                        {
                            "cycle_id": "cycle:retrieval:1",
                            "subsystem": "retrieval",
                            "variant_id": "routing_depth",
                            "selected_variant_breadth_pressure": 0.3,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        report_path = round_one_report if len(seen_campaign_cmds) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--campaign-width",
            "1",
            "--variant-width",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert len(seen_campaign_cmds) == 2
    assert seen_campaign_cmds[1][seen_campaign_cmds[1].index("--campaign-width") + 1] == "2"
    assert seen_campaign_cmds[1][seen_campaign_cmds[1].index("--variant-width") + 1] == "2"
    assert seen_campaign_cmds[1][seen_campaign_cmds[1].index("--task-limit") + 1] == "64"
    assert "--adaptive-search" in seen_campaign_cmds[1]
    exclude_indexes = [index for index, token in enumerate(seen_campaign_cmds[1]) if token == "--exclude-subsystem"]
    assert exclude_indexes
    assert seen_campaign_cmds[1][exclude_indexes[0] + 1] == "retrieval"

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["rounds_completed"] == 2
    assert payload["rounds"][0]["next_policy"]["focus"] == "balanced"
    assert "campaign_breadth_pressure" in payload["rounds"][0]["policy_shift_rationale"]["reason_codes"]
    assert payload["policy_shift_summary"]["reason_counts"]["campaign_breadth_pressure"] == 1


def test_run_unattended_campaign_surfaces_policy_shift_rationale_in_live_status_and_events(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    status_path = reports_dir / "campaign.status.json"
    event_log_path = reports_dir / "campaign.events.jsonl"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.05,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                "planner_pressure_summary": {
                    "campaign_breadth_pressure_cycles": 1,
                    "variant_breadth_pressure_cycles": 1,
                    "recent_campaign_pressures": [
                        {"cycle_id": "cycle:retrieval:1", "subsystem": "retrieval", "campaign_breadth_pressure": 0.4}
                    ],
                    "recent_variant_pressures": [
                        {
                            "cycle_id": "cycle:retrieval:1",
                            "subsystem": "retrieval",
                            "variant_id": "routing_depth",
                            "selected_variant_breadth_pressure": 0.3,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_campaign_cmds: list[list[str]] = []
    live_status_checks: list[dict[str, object]] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        if len(seen_campaign_cmds) == 2:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            event_payloads = [
                json.loads(line)
                for line in event_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            live_status_checks.append(
                {
                    "status": status_payload,
                    "events": event_payloads,
                }
            )
        report_path = round_one_report if len(seen_campaign_cmds) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--campaign-width",
            "1",
            "--variant-width",
            "1",
            "--status-path",
            str(status_path),
            "--event-log-path",
            str(event_log_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert live_status_checks
    latest_status = live_status_checks[0]["status"]
    assert latest_status["policy_shift_alert_subscriptions"] == ["all"]
    assert "campaign_breadth_pressure" in latest_status["latest_available_policy_shift_rationale"]["reason_codes"]
    assert latest_status["policy_shift_summary"]["reason_counts"]["campaign_breadth_pressure"] == 1
    assert latest_status["latest_policy_shift_alert"]["ran"] is False
    assert latest_status["latest_policy_shift_alert"]["detail"] == "no alert transport configured"
    assert latest_status["latest_policy_shift_alert_summary"].startswith(
        "not sent: no alert transport configured"
    )
    assert "campaign_breadth_pressure" in latest_status["latest_policy_shift_alert_summary"]
    policy_shift_events = [
        event for event in live_status_checks[0]["events"] if event.get("event_kind") == "policy_shift"
    ]
    assert policy_shift_events
    assert "campaign_breadth_pressure" in policy_shift_events[-1]["policy_shift_rationale"]["reason_codes"]


def test_run_unattended_campaign_emits_policy_shift_alert_for_breadth_pressure(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.05,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                "planner_pressure_summary": {
                    "campaign_breadth_pressure_cycles": 1,
                    "variant_breadth_pressure_cycles": 1,
                    "recent_campaign_pressures": [
                        {"cycle_id": "cycle:retrieval:1", "subsystem": "retrieval", "campaign_breadth_pressure": 0.4}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_alerts: list[dict[str, object]] = []
    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        report_path = round_one_report if len(seen_campaign_cmds) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        module,
        "_run_alert_command",
        lambda command, *, report_path, status_path, event_log_path, payload: (
            seen_alerts.append(
                {
                    "command": command,
                    "report_path": str(report_path),
                    "status_path": "" if status_path is None else str(status_path),
                    "event_log_path": "" if event_log_path is None else str(event_log_path),
                    "status": payload.get("status"),
                    "phase": payload.get("phase"),
                    "reason": payload.get("reason"),
                    "reason_codes": payload.get("policy_shift_rationale", {}).get("reason_codes", []),
                }
            )
            or {"ran": True, "command": command, "returncode": 0}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--campaign-width",
            "1",
            "--variant-width",
            "1",
            "--alert-command",
            "echo campaign-alert",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    policy_shift_alerts = [alert for alert in seen_alerts if alert["phase"] == "policy_shift"]
    assert policy_shift_alerts
    assert policy_shift_alerts[0]["status"] == "running"
    assert "breadth pressure" in policy_shift_alerts[0]["reason"]
    assert "campaign_breadth_pressure" in policy_shift_alerts[0]["reason_codes"]
    report_path = Path(stream.getvalue().strip())
    status_path = report_path.with_suffix(".status.json")
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["latest_policy_shift_alert_summary"].startswith("sent: severity=info")
    assert "campaign_breadth_pressure" in status_payload["latest_policy_shift_alert_summary"]


def test_run_unattended_campaign_filters_policy_shift_alerts_by_subscription(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_alerts: list[dict[str, object]] = []
    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        report_path = round_one_report if len(seen_campaign_cmds) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        module,
        "_run_alert_command",
        lambda command, *, report_path, status_path, event_log_path, payload: (
            seen_alerts.append(
                {
                    "command": command,
                    "phase": payload.get("phase"),
                    "reason": payload.get("reason"),
                    "reason_codes": payload.get("policy_shift_rationale", {}).get("reason_codes", []),
                }
            )
            or {"ran": True, "command": command, "returncode": 0}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--campaign-width",
            "1",
            "--variant-width",
            "1",
            "--alert-command",
            "echo campaign-alert",
            "--policy-shift-alert-subscriptions",
            "campaign_breadth_pressure,variant_breadth_pressure",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    policy_shift_alerts = [alert for alert in seen_alerts if alert["phase"] == "policy_shift"]
    assert not policy_shift_alerts
    report_path = Path(stream.getvalue().strip())
    status_path = report_path.with_suffix(".status.json")
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["policy_shift_alert_subscriptions"] == [
        "campaign_breadth_pressure",
        "variant_breadth_pressure",
    ]
    assert status_payload["latest_policy_shift_alert"]["suppressed"] is True
    assert status_payload["latest_policy_shift_alert"]["detail"] == "policy shift alert not subscribed"
    assert status_payload["latest_policy_shift_alert_summary"].startswith(
        "suppressed: policy shift alert not subscribed"
    )
    assert "no_yield_round" in status_payload["latest_policy_shift_alert_summary"]


def test_run_unattended_campaign_safe_stops_on_invalid_child_report(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    bad_campaign_report_path = reports_dir / "bad_campaign_report.json"
    bad_campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    bad_campaign_report_path.write_text(json.dumps({"report_kind": "wrong"}), encoding="utf-8")

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        return {"returncode": 0, "stdout": f"{bad_campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["reason"] == "campaign report kind was missing or invalid"
    controller_state_path = Path(payload["controller_state_path"])
    controller_payload = json.loads(controller_state_path.read_text(encoding="utf-8"))
    assert controller_payload["updates"] >= 1
    assert controller_payload["recent_rewards"][-1] < 0.0


def test_validate_campaign_report_includes_failed_run_and_decision_context():
    module = _load_script("run_unattended_campaign.py")

    validation = module._validate_campaign_report(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 2,
            "completed_runs": 2,
            "successful_runs": 1,
            "inheritance_summary": {"runtime_managed_decisions": 1},
            "runs": [
                {"index": 1, "returncode": 0, "stdout": "ok", "stderr": ""},
                {"index": 2, "returncode": 2, "stdout": "", "stderr": "worker command failed\ncheckpoint tail"},
            ],
            "recent_production_decisions": [
                {
                    "cycle_id": "cycle:tolbert_model:2",
                    "state": "reject",
                    "subsystem": "tolbert_model",
                    "reason": "phase gate failed",
                    "artifact_kind": "tolbert_model_bundle",
                    "artifact_path": "trajectories/tolbert_model_artifact.json",
                    "metrics_summary": {
                        "phase_gate_passed": False,
                        "baseline_pass_rate": 0.72,
                        "candidate_pass_rate": 0.41,
                    },
                }
            ],
        }
    )

    assert validation["passed"] is False
    assert validation["detail"].startswith("campaign report shows failed child cycles: run=2 returncode=2")
    assert validation["failure_context"]["failed_run"]["run_index"] == 2
    assert validation["failure_context"]["failed_run"]["stderr_tail"] == "worker command failed\ncheckpoint tail"
    assert validation["failure_context"]["failed_decision"]["subsystem"] == "tolbert_model"


def test_validate_campaign_report_accepts_intermediate_non_runtime_managed_decision_signal():
    module = _load_script("run_unattended_campaign.py")

    validation = module._validate_campaign_report(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 1,
            "completed_runs": 1,
            "successful_runs": 1,
            "inheritance_summary": {"runtime_managed_decisions": 0},
            "execution_source_summary": {
                "decoder_generated": 1,
                "llm_generated": 1,
                "bounded_decoder_generated": 0,
                "synthetic_plan": 0,
                "deterministic_or_other": 0,
                "total_executed_commands": 1,
            },
            "production_yield_summary": {"retained_cycles": 0},
            "recent_production_decisions": [
                {
                    "cycle_id": "cycle:retrieval:1",
                    "state": "reject",
                    "subsystem": "retrieval",
                    "reason": "retention gate failed",
                    "metrics_summary": {
                        "baseline_pass_rate": 0.9583,
                        "candidate_pass_rate": 0.9583,
                    },
                }
            ],
        }
    )

    assert validation["passed"] is True
    assert validation["detail"] == "campaign report is complete with intermediate decision evidence"


def test_campaign_signal_promotes_non_runtime_managed_decision_summary_without_runtime_managed_credit():
    module = _load_script("run_unattended_campaign.py")

    signal = module._campaign_signal(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 1,
            "completed_runs": 1,
            "successful_runs": 1,
            "inheritance_summary": {
                "runtime_managed_decisions": 0,
                "non_runtime_managed_decisions": 2,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 2,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 2,
            },
            "execution_source_summary": {
                "decoder_generated": 2,
                "llm_generated": 1,
                "synthetic_plan": 0,
                "deterministic_or_other": 1,
                "total_executed_commands": 3,
            },
            "production_yield_summary": {"retained_cycles": 0},
            "recent_non_runtime_decisions": [
                {"cycle_id": "cycle:retrieval:1", "state": "reject", "subsystem": "retrieval"},
            ],
        }
    )

    assert signal["runtime_managed_decisions"] == 0
    assert signal["non_runtime_managed_decisions"] == 2
    assert signal["decision_runs"] == 2
    assert signal["non_runtime_managed_runs"] == 2
    assert signal["partial_productive_without_decision_runs"] == 1
    assert signal["decoder_generated_commands"] == 2
    assert signal["bounded_decoder_generated_commands"] == 1
    assert signal["deterministic_or_other_commands"] == 1
    assert signal["decoder_control_active"] is True
    assert signal["llm_control_active"] is True
    assert signal["intermediate_decision_evidence"] is True


def test_validate_campaign_report_rejects_decision_signal_without_live_llm_generated_execution():
    module = _load_script("run_unattended_campaign.py")

    validation = module._validate_campaign_report(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 1,
            "completed_runs": 1,
            "successful_runs": 1,
            "inheritance_summary": {"runtime_managed_decisions": 0},
            "execution_source_summary": {
                "decoder_generated": 1,
                "llm_generated": 0,
                "bounded_decoder_generated": 1,
                "synthetic_plan": 0,
                "deterministic_or_other": 1,
                "total_executed_commands": 2,
            },
            "recent_production_decisions": [
                {
                    "cycle_id": "cycle:retrieval:1",
                    "state": "reject",
                    "subsystem": "retrieval",
                }
            ],
        }
    )

    assert validation["passed"] is False
    assert validation["detail"].startswith(
        "campaign report showed decision evidence without live LLM-generated execution"
    )


def test_validate_campaign_report_rejects_runtime_managed_credit_without_live_llm_generated_execution():
    module = _load_script("run_unattended_campaign.py")

    validation = module._validate_campaign_report(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 1,
            "completed_runs": 1,
            "successful_runs": 1,
            "inheritance_summary": {"runtime_managed_decisions": 1},
            "execution_source_summary": {
                "decoder_generated": 1,
                "llm_generated": 0,
                "bounded_decoder_generated": 1,
                "synthetic_plan": 0,
                "deterministic_or_other": 2,
                "total_executed_commands": 3,
            },
        }
    )

    assert validation["passed"] is False
    assert validation["detail"].startswith(
        "campaign report credited runtime-managed decisions without live LLM-generated execution"
    )


def test_merge_mirrored_child_status_fields_promotes_decision_conversion_summary():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {},
        {
            "campaign_records_considered": 3,
            "decision_records_considered": 1,
            "runtime_managed_decisions": 0,
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 1,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 1,
            },
        },
    )

    assert merged["campaign_records_considered"] == 3
    assert merged["decision_records_considered"] == 1
    assert merged["runtime_managed_decisions"] == 0
    assert merged["decision_conversion_summary"]["non_runtime_managed_runs"] == 1


def test_merge_mirrored_child_status_fields_promotes_nested_semantic_progress_state():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {},
        {
            "active_cycle_run": {
                "semantic_progress_state": {
                    "phase": "preview_candidate_eval",
                    "phase_family": "preview",
                    "progress_class": "healthy",
                    "decision_distance": "near",
                }
            }
        },
    )

    assert merged["semantic_progress_state"]["phase"] == "preview_candidate_eval"
    assert merged["semantic_progress_state"]["decision_distance"] == "near"


def test_merge_mirrored_child_status_fields_prefers_authoritative_child_inputs():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "selected_subsystem": "retrieval",
            "observe_summary": {"passed": 1, "total": 8},
            "pending_decision_state": "",
            "preview_state": "",
            "current_cognitive_stage": {"cognitive_stage": "memory_retrieved", "step_index": 1},
        },
        {
            "active_cycle_run": {
                "selected_subsystem": "policy",
                "observe_summary": {"passed": 4, "total": 4},
                "pending_decision_state": "reject",
                "preview_state": "reject",
                "current_cognitive_stage": {
                    "cognitive_stage": "verification_result",
                    "step_index": 2,
                    "verification_passed": False,
                },
            }
        },
    )

    assert merged["selected_subsystem"] == "policy"
    assert merged["observe_summary"] == {"passed": 4, "total": 4}
    assert merged["pending_decision_state"] == "reject"
    assert merged["preview_state"] == "reject"
    assert merged["current_cognitive_stage"]["cognitive_stage"] == "verification_result"


def test_merge_mirrored_child_status_fields_projects_child_phase_detail_when_parent_line_is_stale():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_line": "[cycle:test] phase=observe probe_task_limit=8 requested_task_limit=32",
            "last_output_line": "[cycle:test] phase=observe probe_task_limit=8 requested_task_limit=32",
        },
        {
            "selected_subsystem": "retrieval",
            "semantic_progress_state": {
                "phase": "preview_baseline_eval",
                "phase_family": "preview",
                "progress_class": "healthy",
                "decision_distance": "near",
                "detail": "preview task 5/8 is progressing",
            },
            "active_cycle_run": {
                "selected_subsystem": "retrieval",
                "current_task": {
                    "index": 5,
                    "total": 8,
                    "task_id": "deployment_manifest_task_project_recovery",
                    "phase": "preview_baseline_eval",
                },
                "last_progress_phase": "preview_baseline_eval",
            },
        },
    )

    assert merged["last_progress_line"].startswith("[child] phase=preview_baseline_eval task 5/8")
    assert "subsystem=retrieval" in merged["last_progress_line"]
    assert merged["last_output_line"] == merged["last_progress_line"]


def test_preferred_active_child_phase_detail_replaces_stale_same_phase_task_line():
    module = _load_script("run_unattended_campaign.py")

    detail = module._preferred_active_child_phase_detail(
        "[eval:test] phase=generated_success task 1/8 bridge_handoff_task_project_adjacent family=project",
        {
            "last_progress_line": (
                "[eval:test] phase=generated_success task 5/8 deployment_manifest_task_project_adjacent family=project"
            ),
            "current_task": {
                "index": 5,
                "total": 8,
                "task_id": "deployment_manifest_task_project_adjacent",
                "family": "project",
                "phase": "generated_success",
            },
            "semantic_progress_state": {
                "phase": "generated_success",
                "phase_family": "finalize",
                "progress_class": "healthy",
                "decision_distance": "near",
                "detail": "finalize task 5/8 is progressing",
            },
        },
    )

    assert "task 5/8 deployment_manifest_task_project_adjacent" in detail


def test_authoritative_round_phase_detail_prefers_live_child_progress_while_running():
    module = _load_script("run_unattended_campaign.py")

    detail = module._authoritative_round_phase_detail(
        existing_detail="runtime-managed reject recorded after productive work was already demonstrated",
        round_payload={
            "status": "running",
            "phase": "campaign",
            "active_child": {
                "state": "running",
                "runtime_managed_decisions": 1,
                "pending_decision_state": "reject",
                "last_progress_line": (
                    "[eval:test] phase=generated_failure task 8/8 repo_packet_review_task_safe_retry "
                    "family=repo_chore source_task=repo_packet_review_task"
                ),
                "last_output_line": (
                    "[eval:test] phase=generated_failure task 8/8 repo_packet_review_task_safe_retry "
                    "family=repo_chore source_task=repo_packet_review_task"
                ),
                "last_progress_phase": "generated_failure",
                "current_task": {
                    "index": 8,
                    "total": 8,
                    "task_id": "repo_packet_review_task_safe_retry",
                    "family": "repo_chore",
                    "source_task": "repo_packet_review_task",
                    "phase": "generated_failure",
                },
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "preview",
                    "status": "decision_emitted",
                    "progress_class": "healthy",
                    "decision_distance": "decision_emitted",
                    "detail": "decision has been emitted and is awaiting durable completion",
                },
            },
        },
        campaign_report={},
    )

    assert detail.startswith("[eval:test] phase=generated_failure task 8/8 repo_packet_review_task_safe_retry")


def test_authoritative_round_phase_detail_summarizes_runtime_managed_outcome_after_closeout():
    module = _load_script("run_unattended_campaign.py")

    detail = module._authoritative_round_phase_detail(
        existing_detail="[cycle:test] finalize phase=done subsystem=retrieval state=reject",
        round_payload={
            "status": "completed",
            "phase": "completed",
            "active_child": {
                "state": "completed",
                "runtime_managed_decisions": 1,
                "pending_decision_state": "reject",
                "last_progress_line": "[cycle:test] finalize phase=done subsystem=retrieval state=reject",
                "last_output_line": "[cycle:test] finalize phase=done subsystem=retrieval state=reject",
                "last_progress_phase": "done",
                "semantic_progress_state": {
                    "phase": "done",
                    "phase_family": "apply",
                    "status": "complete",
                    "progress_class": "complete",
                    "decision_distance": "complete",
                    "detail": "finalize completed",
                },
            },
        },
        campaign_report={},
    )

    assert detail == "runtime-managed reject recorded after productive work was already demonstrated"


def test_run_unattended_campaign_surfaces_child_failure_context_in_report_and_status(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    status_path = reports_dir / "campaign.status.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "cycles_requested": 1,
                "completed_runs": 1,
                "successful_runs": 0,
                "inheritance_summary": {"runtime_managed_decisions": 1},
                "runs": [
                    {
                        "index": 1,
                        "returncode": 2,
                        "stdout": "",
                        "stderr": "worker command failed\ncheckpoint tail",
                    }
                ],
                "recent_production_decisions": [
                    {
                        "cycle_id": "cycle:tolbert_model:1",
                        "state": "reject",
                        "subsystem": "tolbert_model",
                        "reason": "phase gate failed",
                        "artifact_kind": "tolbert_model_bundle",
                        "artifact_path": "trajectories/tolbert_model_artifact.json",
                        "metrics_summary": {
                            "phase_gate_passed": False,
                            "baseline_pass_rate": 0.72,
                            "candidate_pass_rate": 0.41,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unattended_campaign.py", "--liftoff", "never", "--status-path", str(status_path)],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["campaign_validation"]["failure_context"]["failed_run"]["run_index"] == 1
    assert payload["campaign_validation"]["failure_context"]["failed_decision"]["subsystem"] == "tolbert_model"
    assert status_payload["campaign_validation"]["failure_context"]["failed_run"]["returncode"] == 2
    assert status_payload["campaign_validation"]["failure_context"]["failed_decision"]["reason"] == "phase gate failed"


def test_run_unattended_campaign_persists_live_child_progress(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 0},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
            }
        ),
        encoding="utf-8",
    )

    seen_report_paths: list[Path] = []

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cwd, env, kwargs
        joined = " ".join(cmd)
        assert "run_repeated_improvement_cycles.py" in joined
        assert on_event is not None
        on_event(
            {
                "event": "start",
                "pid": 4242,
                "progress_label": "campaign_round_1",
                "started_at": 1.0,
            }
        )
        on_event(
            {
                "event": "output",
                "pid": 4242,
                "progress_label": "campaign_round_1",
                "timestamp": 2.0,
                "line": "[cycle:test] finalize phase=preview subsystem=retrieval",
            }
        )
        report_path = Path(
            next(path for path in reports_dir.glob("unattended_campaign_*.json") if not path.name.endswith(".status.json"))
        )
        seen_report_paths.append(report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["active_child"]["pid"] == 4242
        assert payload["active_child"]["last_progress_line"] == "[cycle:test] finalize phase=preview subsystem=retrieval"
        assert payload["phase_detail"] == "[cycle:test] finalize phase=preview subsystem=retrieval"
        assert payload["rounds"][0]["status"] == "running"
        assert payload["rounds"][0]["phase"] == "campaign"
        on_event(
            {
                "event": "exit",
                "pid": 4242,
                "progress_label": "campaign_round_1",
                "timestamp": 3.0,
                "returncode": 0,
            }
        )
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "1",
            "--cycles",
            "1",
            "--liftoff",
            "never",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert seen_report_paths
    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["active_child"]["state"] == "completed"


def test_run_unattended_campaign_resume_cools_down_interrupted_subsystem(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    report_path = reports_dir / "resume_campaign.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                "inheritance_summary": {"runtime_managed_decisions": 1},
                "cycles_requested": 1,
                "completed_runs": 1,
                "successful_runs": 1,
            }
        ),
        encoding="utf-8",
    )

    seen_campaign_cmds: list[list[str]] = []
    run_index = {"value": 0}

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cwd, env, kwargs
        run_index["value"] += 1
        if run_index["value"] == 1:
            assert on_event is not None
            on_event(
                {
                    "event": "start",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "started_at": 1.0,
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "timestamp": 2.0,
                    "line": "[cycle:test] finalize phase=preview subsystem=retrieval",
                }
            )
            raise KeyboardInterrupt("operator interrupted stuck retrieval lane")
        seen_campaign_cmds.append(cmd)
        assert "--exclude-subsystem" in cmd
        assert "retrieval" in cmd
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )

    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--report-path",
            str(report_path),
            "--liftoff",
            "never",
        ],
    )
    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("expected interrupted SystemExit")

    interrupted_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert interrupted_payload["status"] == "interrupted"
    assert interrupted_payload["rounds"][0]["status"] == "interrupted"
    assert interrupted_payload["rounds"][0]["active_child"]["last_progress_line"] == (
        "[cycle:test] finalize phase=preview subsystem=retrieval"
    )
    interrupted_controller_path = Path(interrupted_payload["controller_state_path"])
    interrupted_controller = json.loads(interrupted_controller_path.read_text(encoding="utf-8"))
    assert interrupted_controller["updates"] >= 1
    assert interrupted_controller["recent_rewards"][-1] < 0.0

    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--report-path",
            str(report_path),
            "--resume",
            "--liftoff",
            "never",
        ],
    )
    module.main()

    assert seen_campaign_cmds
    resumed_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert resumed_payload["status"] == "completed"
    assert resumed_payload["rounds"][1]["policy"]["excluded_subsystems"] == ["retrieval"]
    assert Path(resumed_payload["controller_state_path"]) == interrupted_controller_path


def test_run_unattended_campaign_initial_persist_skips_storage_governance(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    report_path = reports_dir / "bootstrap_interrupt.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    persisted_govern_storage: list[bool] = []
    original_persist_report_state = module._persist_report_state

    def tracking_persist_report_state(*args, **kwargs):
        persisted_govern_storage.append(bool(kwargs.get("govern_storage", True)))
        return original_persist_report_state(*args, **kwargs)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_persist_report_state", tracking_persist_report_state)
    monkeypatch.setattr(module, "_dispatch_alerts", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected alert dispatch")))
    monkeypatch.setattr(module, "_disk_preflight", lambda path, *, min_free_gib: {
        "path": str(path),
        "free_bytes": 0,
        "free_gib": 0.0,
        "min_free_gib": min_free_gib,
        "passed": False,
        "detail": "forced failure",
    })
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--report-path", str(report_path), "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected preflight SystemExit")

    assert persisted_govern_storage
    assert persisted_govern_storage[0] is False



def test_alerts_enabled_ignores_unset_transports():
    module = _load_script("run_unattended_campaign.py")
    args = SimpleNamespace(
        alert_command=None,
        slack_webhook_url=None,
        pagerduty_routing_key=None,
    )

    assert module._alerts_enabled(args) is False


def test_run_unattended_campaign_interrupt_refreshes_active_child_from_child_status_file(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    report_path = reports_dir / "interrupt_refresh.json"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cmd, cwd, kwargs
        assert on_event is not None
        on_event(
            {
                "event": "start",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "started_at": 1.0,
            }
        )
        on_event(
            {
                "event": "output",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "timestamp": 2.0,
                "line": "[cycle:test] phase=generated_success_schedule",
            }
        )
        child_reports_dir = Path(env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"])
        child_reports_dir.mkdir(parents=True, exist_ok=True)
        (child_reports_dir / "repeated_improvement_status.json").write_text(
            json.dumps(
                {
                    "report_kind": "repeated_improvement_status",
                    "state": "interrupted",
                    "status": "interrupted",
                    "reason": "received signal SIGHUP",
                    "selected_subsystem": "qwen_adapter",
                    "last_progress_phase": "holdout_eval",
                    "current_task": {
                        "index": 390,
                        "total": 390,
                        "task_id": "service_release_task",
                        "family": "repository",
                    },
                    "semantic_progress_state": {
                        "phase": "holdout_eval",
                        "phase_family": "holdout",
                        "status": "active",
                        "progress_class": "healthy",
                        "detail": "holdout task 390/390 reached the execution boundary and is awaiting verification",
                    },
                }
            ),
            encoding="utf-8",
        )
        raise KeyboardInterrupt("received signal SIGHUP")

    persisted_govern_storage: list[bool] = []
    original_persist_report_state = module._persist_report_state

    def tracking_persist_report_state(*args, **kwargs):
        persisted_govern_storage.append(bool(kwargs.get("govern_storage", True)))
        return original_persist_report_state(*args, **kwargs)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_dispatch_alerts", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected alert dispatch")))
    monkeypatch.setattr(module, "_persist_report_state", tracking_persist_report_state)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--report-path",
            str(report_path),
            "--liftoff",
            "never",
        ],
    )

    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("expected interrupted SystemExit")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(report_path.with_suffix(".status.json").read_text(encoding="utf-8"))

    assert payload["status"] == "interrupted"
    assert payload["rounds"][0]["status"] == "interrupted"
    assert payload["rounds"][0]["active_child"]["state"] == "interrupted"
    assert payload["rounds"][0]["active_child"]["last_progress_phase"] == "holdout_eval"
    assert payload["active_child"]["state"] == "interrupted"
    assert payload["active_child"]["last_progress_phase"] == "holdout_eval"
    assert status_payload["active_child"]["state"] == "interrupted"
    assert status_payload["active_run"]["child_status"]["state"] == "interrupted"
    assert persisted_govern_storage
    assert persisted_govern_storage[:2] == [False, False]
    assert persisted_govern_storage[-1] is False


def test_run_unattended_campaign_recovers_from_failed_child_with_adapted_next_round(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                "inheritance_summary": {"runtime_managed_decisions": 1},
                "cycles_requested": 1,
                "completed_runs": 1,
                "successful_runs": 1,
            }
        ),
        encoding="utf-8",
    )

    run_index = {"value": 0}
    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        run_index["value"] += 1
        if run_index["value"] == 1:
            assert on_event is not None
            on_event(
                {
                    "event": "start",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "started_at": 1.0,
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "timestamp": 2.0,
                    "line": "[cycle:test] finalize phase=preview subsystem=retrieval",
                }
            )
            return {"returncode": 2, "stdout": "", "stderr": "failed", "timed_out": False}
        assert "--exclude-subsystem" in cmd
        assert "retrieval" in cmd
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert len(seen_campaign_cmds) == 2
    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(report_path.with_suffix(".status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["child_failure_recovery_budget"] == 1
    assert payload["child_failure_recoveries_used"] == 1
    assert payload["rounds"][0]["status"] == "recovered"
    assert payload["rounds"][0]["recovery"]["stalled_subsystem"] == "retrieval"
    assert payload["rounds"][1]["policy"]["excluded_subsystems"] == ["retrieval"]
    assert status_payload["child_failure_recoveries_used"] == 1


def test_run_unattended_campaign_recovers_stalled_subsystem_after_generic_silence_output(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True},
                "inheritance_summary": {"runtime_managed_decisions": 1},
                "cycles_requested": 1,
                "completed_runs": 1,
                "successful_runs": 1,
            }
        ),
        encoding="utf-8",
    )

    run_index = {"value": 0}
    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        run_index["value"] += 1
        if run_index["value"] == 1:
            assert on_event is not None
            on_event(
                {
                    "event": "start",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "started_at": 1.0,
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "timestamp": 2.0,
                    "line": "[cycle:test] variant_search start subsystem=qwen_adapter selected_variants=1",
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 31337,
                    "progress_label": "campaign_round_1",
                    "timestamp": 3.0,
                    "line": "[repeated] child=campaign_round_1 still_running silence=120s",
                }
            )
            return {"returncode": 2, "stdout": "", "stderr": "failed", "timed_out": False}
        assert "--exclude-subsystem" in cmd
        assert "qwen_adapter" in cmd
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert len(seen_campaign_cmds) == 2
    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["rounds"][0]["recovery"]["stalled_subsystem"] == "qwen_adapter"
    assert payload["rounds"][1]["policy"]["excluded_subsystems"] == ["qwen_adapter"]


def test_run_unattended_campaign_recovery_from_no_runtime_managed_decisions_does_not_exclude_only_subsystem(
    tmp_path, monkeypatch
):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)

    report_without_runtime_decisions = {
        "report_kind": "improvement_campaign_report",
        "production_yield_summary": {"retained_cycles": 0},
        "phase_gate_summary": {"all_retained_phase_gates_passed": True},
        "inheritance_summary": {"runtime_managed_decisions": 0},
        "cycles_requested": 1,
        "completed_runs": 1,
        "successful_runs": 1,
    }
    report_with_runtime_decisions = {
        "report_kind": "improvement_campaign_report",
        "production_yield_summary": {"retained_cycles": 1},
        "phase_gate_summary": {"all_retained_phase_gates_passed": True},
        "inheritance_summary": {"runtime_managed_decisions": 1},
        "cycles_requested": 1,
        "completed_runs": 1,
        "successful_runs": 1,
    }

    run_index = {"value": 0}
    seen_campaign_cmds: list[list[str]] = []

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cwd, env, kwargs
        seen_campaign_cmds.append(cmd)
        run_index["value"] += 1
        if on_event is not None:
            on_event(
                {
                    "event": "start",
                    "pid": 31337 + run_index["value"],
                    "progress_label": f"campaign_round_{run_index['value']}",
                    "started_at": 1.0,
                }
            )
            on_event(
                {
                    "event": "output",
                    "pid": 31337 + run_index["value"],
                    "progress_label": f"campaign_round_{run_index['value']}",
                    "timestamp": 2.0,
                    "line": "[cycle:test] finalize phase=preview subsystem=retrieval",
                }
            )
        if run_index["value"] == 1:
            campaign_report_path.write_text(json.dumps(report_without_runtime_decisions), encoding="utf-8")
            return {"returncode": 0, "stdout": f"{campaign_report_path}\n", "stderr": "", "timed_out": False}
        campaign_report_path.write_text(json.dumps(report_with_runtime_decisions), encoding="utf-8")
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert len(seen_campaign_cmds) == 2
    exclude_indexes = [index for index, token in enumerate(seen_campaign_cmds[1]) if token == "--exclude-subsystem"]
    assert exclude_indexes
    assert seen_campaign_cmds[1][exclude_indexes[0] + 1] == "retrieval"
    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(report_path.with_suffix(".status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["child_failure_recoveries_used"] == 1
    assert payload["rounds"][0]["recovery"]["stalled_subsystem"] == "retrieval"
    assert payload["rounds"][0]["recovery"]["reason"] == "campaign report showed no runtime-managed decisions"
    assert payload["rounds"][1]["policy"]["excluded_subsystems"] == ["retrieval"]
    assert status_payload["child_failure_recoveries_used"] == 1


def test_run_unattended_campaign_accepts_intermediate_non_runtime_managed_decision_signal(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "cycles_requested": 1,
                "completed_runs": 1,
                "successful_runs": 1,
                "inheritance_summary": {"runtime_managed_decisions": 0},
                "execution_source_summary": {
                    "decoder_generated": 1,
                    "llm_generated": 0,
                    "synthetic_plan": 0,
                    "deterministic_or_other": 0,
                    "total_executed_commands": 1,
                },
                "production_yield_summary": {"retained_cycles": 0},
                "recent_production_decisions": [
                    {
                        "cycle_id": "cycle:retrieval:1",
                        "state": "reject",
                        "subsystem": "retrieval",
                        "reason": "retention gate failed",
                        "metrics_summary": {
                            "baseline_pass_rate": 0.9583,
                            "candidate_pass_rate": 0.9583,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cmd, cwd, env, on_event, kwargs
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["rounds"][0]["status"] == "completed"
    assert "recovery" not in payload["rounds"][0]


def test_child_failure_recovery_policy_cools_dominant_zero_yield_subsystem_after_no_runtime_managed_decisions():
    module = _load_script("run_unattended_campaign.py")
    next_policy = module._child_failure_recovery_policy(
        {
            "cycles": 1,
            "campaign_width": 2,
            "variant_width": 1,
            "adaptive_search": False,
            "task_limit": 64,
            "priority_benchmark_families": ["project", "repository", "integration"],
            "tolbert_device": "cuda",
            "focus": "balanced",
            "liftoff": "never",
            "excluded_subsystems": [],
            "subsystem_cooldowns": {},
        },
        round_payload={
            "campaign_report": {
                "recent_production_decisions": [
                    {
                        "subsystem": "retrieval",
                        "state": "reject",
                        "metrics_summary": {
                            "baseline_pass_rate": 0.95,
                            "candidate_pass_rate": 0.95,
                        },
                    },
                    {
                        "subsystem": "retrieval",
                        "state": "reject",
                        "metrics_summary": {
                            "baseline_pass_rate": 0.95,
                            "candidate_pass_rate": 0.95,
                        },
                    },
                    {
                        "subsystem": "retrieval",
                        "state": "reject",
                        "metrics_summary": {
                            "baseline_pass_rate": 0.9583,
                            "candidate_pass_rate": 0.9583,
                        },
                    },
                ]
            }
        },
        phase="campaign",
        reason="campaign report showed no runtime-managed decisions",
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    assert next_policy["excluded_subsystems"] == ["retrieval", "trust"]
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert next_policy["subsystem_cooldowns"]["trust"] >= 1


def test_initial_round_policy_bootstraps_coding_campaign_without_trust():
    module = _load_script("run_unattended_campaign.py")

    policy = module._initial_round_policy(
        SimpleNamespace(
            cycles=1,
            campaign_width=2,
            variant_width=1,
            adaptive_search=True,
            task_limit=10,
            task_step_floor=100,
            priority_benchmark_family=["integration", "project", "repository"],
            focus="discovered_task_adaptation",
            liftoff="never",
        ),
        KernelConfig(),
    )

    for subsystem in (
        "trust",
        "curriculum",
        "state_estimation",
        "transition_model",
        "recovery",
        "delegation",
        "operator_policy",
        "capabilities",
        "universe",
    ):
        assert subsystem in policy["excluded_subsystems"]
        assert policy["subsystem_cooldowns"][subsystem] == 1


def test_apply_semantic_policy_seed_merges_existing_exclusions_cooldowns_and_focus():
    module = _load_script("run_unattended_campaign.py")

    seeded = module._apply_semantic_policy_seed(
        {
            "excluded_subsystems": ["trust"],
            "subsystem_cooldowns": {"trust": 1},
            "priority_benchmark_families": ["integration", "project", "repository"],
            "adaptive_search": True,
            "focus": "balanced",
        },
        semantic_seed={
            "excluded_subsystems": ["retrieval"],
            "subsystem_cooldowns": {"retrieval": 2},
            "adaptive_search": True,
            "focus": "recovery_alignment",
        },
    )

    assert seeded["excluded_subsystems"] == ["trust", "retrieval"]
    assert seeded["subsystem_cooldowns"] == {"trust": 1, "retrieval": 2}
    assert seeded["focus"] == "recovery_alignment"



def test_semantic_policy_seed_surfaces_discovered_task_adaptation_focus_from_redirects(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "improvement_reports",
        semantic_hub_root=tmp_path / "semantic_hub",
    )
    redirects_dir = config.semantic_hub_root / "redirects"
    redirects_dir.mkdir(parents=True, exist_ok=True)
    (redirects_dir / "redirect.json").write_text(
        json.dumps(
            {
                "reason_codes": ["weak_retention_outcome"],
                "source_subsystems": ["retrieval", "retrieval"],
                "target_priority_families": ["integration", "project", "repository"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_codex_batch_policy_seed", lambda config: {})

    seed = module._semantic_policy_seed(config)

    assert seed["adaptive_search"] is True
    assert seed["focus"] == "discovered_task_adaptation"
    assert "excluded_subsystems" not in seed



def test_semantic_policy_seed_uses_latest_codex_closure_packet_for_runtime_first_campaigns(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        semantic_hub_root=tmp_path / "semantic_hub",
    )
    (reports_dir / "latest.status.json").write_text(
        json.dumps(
            {
                "asi_campaign_lane_recommendation": {
                    "primary_lane": "lane_decision_closure",
                    "primary_batch_goal": "retained_conversion",
                    "supporting_lanes": ["lane_trust_and_carryover"],
                },
                "coding_capability_ramp_summary": {
                    "primary_gap_id": "runtime_managed_conversion",
                    "semantic_only_runtime": True,
                    "specialization_mode": "semantic_only_runtime",
                    "required_families": ["integration", "project", "repository", "repo_chore"],
                },
            }
        ),
        encoding="utf-8",
    )

    seed = module._semantic_policy_seed(config)

    assert seed["adaptive_search"] is True
    assert seed["focus"] == "recovery_alignment"
    assert seed["focus_locked"] is True
    assert "qwen_adapter" in seed["excluded_subsystems"]
    assert seed["subsystem_cooldowns"]["qwen_adapter"] == 1
    assert seed["protected_subsystems"] == [
        "retrieval",
        "state_estimation",
        "recovery",
        "transition_model",
        "trust",
    ]
    assert seed["priority_benchmark_families"] == ["integration", "project", "repository", "repo_chore"]



def test_initial_round_policy_honors_latest_codex_closure_seed_for_semantic_runtime(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        semantic_hub_root=tmp_path / "semantic_hub",
    )
    (reports_dir / "latest.status.json").write_text(
        json.dumps(
            {
                "asi_campaign_lane_recommendation": {
                    "primary_lane": "lane_decision_closure",
                    "primary_batch_goal": "retained_conversion",
                    "supporting_lanes": ["lane_trust_and_carryover"],
                },
                "coding_capability_ramp_summary": {
                    "primary_gap_id": "runtime_managed_conversion",
                    "semantic_only_runtime": True,
                    "specialization_mode": "semantic_only_runtime",
                    "required_families": ["integration", "project", "repository", "repo_chore"],
                },
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        cycles=1,
        campaign_width=2,
        variant_width=1,
        adaptive_search=False,
        task_limit=8,
        task_step_floor=50,
        priority_benchmark_family=[],
        focus="balanced",
        liftoff="never",
        semantic_only_runtime=True,
        exclude_subsystem=[],
    )

    policy = module._initial_round_policy(args, config)
    policy = module._apply_semantic_policy_seed(policy, semantic_seed=module._semantic_policy_seed(config))
    policy = module._apply_semantic_only_runtime_policy(policy)

    assert policy["focus"] == "recovery_alignment"
    assert policy["adaptive_search"] is True
    assert "tolbert_model" in policy["excluded_subsystems"]
    assert "qwen_adapter" in policy["excluded_subsystems"]
    assert "retrieval" not in policy["excluded_subsystems"]
    assert "state_estimation" not in policy["excluded_subsystems"]
    assert "recovery" not in policy["excluded_subsystems"]
    assert "transition_model" not in policy["excluded_subsystems"]
    assert "trust" not in policy["excluded_subsystems"]
    assert policy["priority_benchmark_families"] == ["integration", "project", "repository", "repo_chore"]


def test_candidate_round_policies_preserve_locked_focus_and_protected_subsystems():
    module = _load_script("run_unattended_campaign.py")

    candidates = module._candidate_round_policies(
        {
            "cycles": 1,
            "campaign_width": 2,
            "variant_width": 1,
            "adaptive_search": True,
            "task_limit": 64,
            "task_step_floor": 50,
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            "focus": "recovery_alignment",
            "focus_locked": True,
            "protected_subsystems": ["retrieval", "state_estimation", "recovery", "transition_model", "trust"],
            "excluded_subsystems": ["qwen_adapter", "retrieval", "trust"],
            "subsystem_cooldowns": {"qwen_adapter": 1, "retrieval": 2, "trust": 2},
        },
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "runtime_managed_decisions": 0,
                "failed_decisions": 0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=4096,
    )

    assert candidates
    assert all(candidate["focus"] == "recovery_alignment" for candidate in candidates)
    assert all(
        candidate["priority_benchmark_families"] == ["integration", "project", "repository", "repo_chore"]
        for candidate in candidates
    )
    for subsystem in ("retrieval", "state_estimation", "recovery", "transition_model", "trust"):
        assert all(subsystem not in candidate["excluded_subsystems"] for candidate in candidates)



def test_next_round_policy_preserves_locked_focus_and_protected_subsystems(monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    monkeypatch.setattr(
        module,
        "plan_next_policy",
        lambda state, current_observation, candidate_policies: (
            {
                "cycles": 2,
                "campaign_width": 3,
                "variant_width": 2,
                "adaptive_search": True,
                "task_limit": 128,
                "task_step_floor": 50,
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "focus": "discovered_task_adaptation",
                "excluded_subsystems": ["retrieval", "trust", "qwen_adapter"],
                "subsystem_cooldowns": {"retrieval": 2, "trust": 2, "qwen_adapter": 1},
            },
            {"candidates": [{"policy": {"focus": "discovered_task_adaptation", "excluded_subsystems": ["retrieval", "trust", "qwen_adapter"]}, "score": 1.0, "action_key": "test"}]},
        ),
    )

    next_policy = module._next_round_policy(
        {
            "cycles": 1,
            "campaign_width": 2,
            "variant_width": 1,
            "adaptive_search": True,
            "task_limit": 64,
            "task_step_floor": 50,
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            "focus": "recovery_alignment",
            "focus_locked": True,
            "protected_subsystems": ["retrieval", "state_estimation", "recovery", "transition_model", "trust"],
            "excluded_subsystems": ["qwen_adapter"],
            "subsystem_cooldowns": {"qwen_adapter": 1},
        },
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "runtime_managed_decisions": 0,
                "failed_decisions": 0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        round_payload={"policy": {"focus": "recovery_alignment"}},
        controller_state=default_controller_state(),
        prior_rounds=[],
        planner_controls={},
        curriculum_controls={},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    assert next_policy["focus"] == "recovery_alignment"
    assert next_policy["priority_benchmark_families"] == ["integration", "project", "repository", "repo_chore"]
    assert "qwen_adapter" in next_policy["excluded_subsystems"]
    for subsystem in ("retrieval", "state_estimation", "recovery", "transition_model", "trust"):
        assert subsystem not in next_policy["excluded_subsystems"]


def test_resume_policy_from_partial_round_preserves_bootstrap_exclusions_without_prior_rounds():
    module = _load_script("run_unattended_campaign.py")

    resumed = module._resume_policy_from_partial_round(
        {
            "excluded_subsystems": ["trust", "retrieval"],
            "subsystem_cooldowns": {"trust": 1, "retrieval": 2},
            "priority_benchmark_families": ["integration", "project", "repository"],
            "cycles": 1,
            "campaign_width": 2,
            "task_limit": 10,
        },
        prior_rounds=[],
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
    )

    assert resumed["excluded_subsystems"] == ["trust", "retrieval"]
    assert resumed["subsystem_cooldowns"] == {"trust": 1, "retrieval": 2}


def test_child_failure_recovery_policy_keeps_budget_flat_for_decisionless_preview_timeout():
    module = _load_script("run_unattended_campaign.py")

    next_policy = module._child_failure_recovery_policy(
        {
            "cycles": 1,
            "campaign_width": 2,
            "variant_width": 2,
            "adaptive_search": True,
            "task_limit": 10,
            "priority_benchmark_families": ["integration", "project", "repository"],
            "tolbert_device": "cuda",
            "focus": "discovered_task_adaptation",
            "liftoff": "never",
            "excluded_subsystems": [],
            "subsystem_cooldowns": {},
        },
        round_payload={
            "active_child": {
                "selected_subsystem": "trust",
                "decision_records_considered": 0,
                "runtime_managed_decisions": 0,
                "retained_gain_runs": 0,
                "finalize_phase": "preview_baseline_eval",
                "semantic_progress_state": {
                    "phase_family": "preview",
                },
            }
        },
        phase="campaign",
        reason="child exceeded max runtime safety ceiling of 300 seconds",
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["cycles"] == 1
    assert next_policy["task_limit"] == 10
    assert next_policy["campaign_width"] == 2
    assert next_policy["variant_width"] == 2
    assert "trust" in next_policy["excluded_subsystems"]
    assert next_policy["subsystem_cooldowns"]["trust"] >= 2


def test_run_unattended_campaign_safe_stops_when_lock_is_held(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    lock_path = reports_dir / "campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 99999, "report_path": "other.json"}), encoding="utf-8")

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.fcntl, "flock", lambda fd, op: (_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unattended_campaign.py", "--lock-path", str(lock_path), "--attach-on-lock", "0"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert "lock already held" in payload["reason"]
    assert payload["lock"]["passed"] is False


def test_run_unattended_campaign_attaches_to_active_report_when_lock_is_held(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    active_report_path = reports_dir / "active_campaign.json"
    active_report_path.parent.mkdir(parents=True, exist_ok=True)
    active_report_path.write_text(
        json.dumps({"report_kind": "unattended_campaign_report", "status": "running", "phase": "campaign"}),
        encoding="utf-8",
    )
    lock_path = reports_dir / "campaign.lock"
    lock_path.write_text(
        json.dumps({"pid": 99999, "report_path": str(active_report_path), "status_path": "active.status.json"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module.fcntl, "flock", lambda fd, op: (_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--lock-path", str(lock_path)])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert Path(stream.getvalue().strip()) == active_report_path
    attached_reports = [
        path
        for path in reports_dir.glob("unattended_campaign_*.json")
        if ".status." not in path.name and ".alerts." not in path.name
    ]
    assert attached_reports == []


def test_governed_global_storage_cleanup_prunes_managed_entries_when_root_is_full(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    global_root = tmp_path / "data"
    managed = global_root / "checkpoints"
    old_dir = managed / "old"
    new_dir = managed / "new"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "blob.bin").write_bytes(b"a" * 32)
    (new_dir / "blob.bin").write_bytes(b"b" * 16)
    old_timestamp = 100
    new_timestamp = 200
    os.utime(old_dir, (old_timestamp, old_timestamp))
    os.utime(new_dir, (new_timestamp, new_timestamp))

    policy_path = tmp_path / "global_storage_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "label": "checkpoints",
                        "path": str(managed),
                        "keep_entries": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    disk_calls: list[float] = [10.0, 120.0]

    def fake_disk_preflight(path, *, min_free_gib):
        del min_free_gib
        if Path(path) == global_root:
            free_gib = disk_calls.pop(0)
            return {
                "path": str(path),
                "free_bytes": int(free_gib * (1024 ** 3)),
                "free_gib": free_gib,
                "min_free_gib": 0.0,
                "passed": True,
                "detail": "ok",
            }
        return {
            "path": str(path),
            "free_bytes": 0,
            "free_gib": 0.0,
            "min_free_gib": 0.0,
            "passed": True,
            "detail": "ok",
        }

    monkeypatch.setattr(module, "_disk_preflight", fake_disk_preflight)

    result = module._governed_global_storage_cleanup(
        global_root,
        policy_entries=module._load_global_storage_policy(policy_path),
        target_free_gib=100.0,
        top_k=5,
    )

    assert result["cleanup"]["removed_entries"]
    assert str(old_dir) in result["cleanup"]["removed_entries"][0]["removed"]
    assert not old_dir.exists()
    assert new_dir.exists()


def test_persist_report_state_writes_emergency_copy_when_primary_write_fails(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    emergency_path = tmp_path / "emergency" / "campaign.json"
    payload = {"status": "running", "phase": "campaign"}

    monkeypatch.setattr(
        module,
        "_write_report",
        lambda path, data, *, config=None, govern_storage=True: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(module, "_emergency_persist_path", lambda path: emergency_path)

    result = module._persist_report_state(report_path, payload, status_path=status_path, lock_path=None)

    assert result["emergency_report_path"] == str(emergency_path)
    assert emergency_path.exists()
    emergency_payload = json.loads(emergency_path.read_text(encoding="utf-8"))
    assert emergency_payload["status"] == "running"
    assert emergency_payload["emergency_persist"]["original_report_path"] == str(report_path)


def test_persist_report_state_projects_active_child_controller_intervention_to_status(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    intervention = {
        "triggered": True,
        "reason_code": "micro_step_verification_loop",
        "reason": "mid-round controller intervention: repeated step-level verification failures are looping",
    }

    module._persist_report_state(
        report_path,
        {
            "status": "running",
            "phase": "campaign",
            "active_child": {
                "controller_intervention": dict(intervention),
            },
        },
        status_path=status_path,
        lock_path=None,
    )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["active_child_controller_intervention"]["reason_code"] == "micro_step_verification_loop"
    assert status_payload["active_child"]["controller_intervention"]["reason_code"] == "micro_step_verification_loop"


def test_write_status_does_not_carry_prior_child_status_into_new_active_run(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"

    module._write_status(
        status_path,
        report_path=report_path,
        payload={
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 1,
                "run_match_id": "run-1",
                "phase": "campaign",
            },
            "active_child": {
                "current_task": {"task_id": "old_task", "phase": "preview_candidate_eval"},
                "controller_intervention": {
                    "triggered": True,
                    "reason_code": "productive_partial_preview_without_decision",
                },
            },
        },
    )

    module._write_status(
        status_path,
        report_path=report_path,
        payload={
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 2,
                "run_match_id": "run-2",
                "phase": "campaign",
            },
            "active_child": {},
        },
    )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["active_run"]["run_match_id"] == "run-2"
    assert status_payload["active_child"] == {}
    assert status_payload["active_child_controller_intervention"] == {}


def test_write_status_keeps_active_run_child_status_aligned_with_active_child(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"

    module._write_status(
        status_path,
        report_path=report_path,
        payload={
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 1,
                "run_match_id": "run-1",
                "phase": "campaign",
                "child_status": {
                    "current_task": {"task_id": "stale_task", "phase": "preview_candidate_eval"},
                    "last_progress_line": None,
                    "semantic_progress_state": {
                        "phase": "preview_candidate_eval",
                        "detail": "preview task 2/8 is progressing",
                    },
                },
            },
            "active_child": {
                "current_task": {
                    "task_id": "fresh_task",
                    "phase": "generated_failure",
                    "index": 8,
                    "total": 8,
                },
                "last_progress_line": (
                    "[eval:cycle:retrieval:cycle-1] phase=generated_failure complete total=8"
                ),
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "preview",
                    "status": "decision_emitted",
                    "detail": "decision has been emitted and is awaiting durable completion",
                },
            },
        },
    )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["active_run"]["child_status"] == status_payload["active_child"]
    assert status_payload["active_child"]["current_task"]["task_id"] == "fresh_task"
    assert status_payload["active_child"]["last_progress_line"].endswith("complete total=8")


def test_persist_report_state_writes_frozen_codex_input_packets(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    child_report_path = tmp_path / "improvement_reports" / "campaign_report.json"
    child_report_path.parent.mkdir(parents=True, exist_ok=True)
    child_report_path.write_text("{}", encoding="utf-8")
    payload = {
        "status": "running",
        "phase": "campaign",
        "event_log_path": str(tmp_path / "reports" / "campaign.events.jsonl"),
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
        },
        "active_run": {
            "run_index": 1,
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "semantic_only_runtime": True,
            },
        },
        "active_child": {
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(child_report_path),
            "verification_outcome_summary": {
                "successful_task_count": 1,
                "successful_families": ["integration"],
            },
            "sampled_families_from_progress": ["integration"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
        },
    }

    module._persist_report_state(report_path, payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    packet_paths = report_payload["codex_input_packets"]
    batch_packet_path = Path(packet_paths["batch_packet_path"])
    frontier_packet_path = Path(packet_paths["frontier_packet_path"])
    baseline_packet_path = Path(packet_paths["baseline_packet_path"])
    assert batch_packet_path.exists()
    assert frontier_packet_path.exists()
    assert baseline_packet_path.exists()
    assert status_payload["codex_input_packets"]["batch_packet_path"] == str(batch_packet_path)
    assert report_payload["coding_capability_ramp_summary"]["recommended_primary_lane"] == "lane_decision_closure"
    batch_packet = json.loads(batch_packet_path.read_text(encoding="utf-8"))
    assert batch_packet["packet_kind"] == "CodexBatchPacket"
    assert batch_packet["batch_goal"] == "retained_conversion"
    assert sorted(batch_packet["allowed_lanes"]) == ["lane_decision_closure", "lane_trust_and_carryover"]
    lane_packet_paths = packet_paths["lane_packet_paths"]
    assert Path(lane_packet_paths["lane_decision_closure"]).exists()
    assert Path(lane_packet_paths["lane_trust_and_carryover"]).exists()


def test_persist_report_state_prunes_stale_lane_packets_when_lane_recommendation_changes(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    child_report_path = tmp_path / "improvement_reports" / "campaign_report.json"
    child_report_path.parent.mkdir(parents=True, exist_ok=True)
    child_report_path.write_text("{}", encoding="utf-8")

    payload_runtime_authority = {
        "status": "running",
        "phase": "campaign",
        "event_log_path": str(tmp_path / "reports" / "campaign.events.jsonl"),
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
        },
        "active_run": {
            "run_index": 1,
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "semantic_only_runtime": True,
            },
        },
        "active_child": {
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(child_report_path),
            "verification_outcome_summary": {
                "successful_task_count": 0,
                "successful_families": [],
            },
            "sampled_families_from_progress": [],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": [],
                "missing_required_families": ["integration", "project", "repository", "repo_chore"],
            },
        },
    }

    module._persist_report_state(report_path, payload_runtime_authority, status_path=status_path)

    runtime_authority_packet_path = (
        report_path.parent / f"{report_path.stem}.lane_runtime_authority.codex_lane_packet.json"
    )
    assert runtime_authority_packet_path.exists()

    payload_decision_closure = {
        "status": "running",
        "phase": "campaign",
        "event_log_path": str(tmp_path / "reports" / "campaign.events.jsonl"),
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
        },
        "active_run": {
            "run_index": 1,
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "semantic_only_runtime": True,
            },
        },
        "active_child": {
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(child_report_path),
            "verification_outcome_summary": {
                "successful_task_count": 2,
                "successful_families": ["integration", "project"],
            },
            "sampled_families_from_progress": ["integration", "project"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
        },
    }

    module._persist_report_state(report_path, payload_decision_closure, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    lane_packet_paths = report_payload["codex_input_packets"]["lane_packet_paths"]
    assert "lane_runtime_authority" not in lane_packet_paths
    assert runtime_authority_packet_path.exists() is False
    assert Path(lane_packet_paths["lane_decision_closure"]).exists()
    assert Path(lane_packet_paths["lane_trust_and_carryover"]).exists()


def test_persist_report_state_refreshes_active_run_child_status_from_active_child(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    payload = {
        "status": "running",
        "phase": "campaign",
        "active_run": {
            "run_index": 1,
            "run_match_id": "run-1",
            "phase": "campaign",
            "child_status": {
                "current_task": {"task_id": "stale_task", "phase": "primary"},
                "last_progress_line": None,
                "semantic_progress_state": {
                    "phase": "primary",
                    "detail": "active task 2/4 is progressing",
                },
            },
        },
        "active_child": {
            "current_task": {},
            "last_progress_line": (
                "[cycle:unattended-campaign-round-1] observe complete passed=3/4 pass_rate=0.7500"
            ),
            "semantic_progress_state": {
                "phase": "observe",
                "phase_family": "observe",
                "status": "complete",
                "progress_class": "complete",
                "detail": "observe phase completed and is ready to hand off",
            },
        },
    }

    module._persist_report_state(report_path, payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert report_payload["active_run"]["child_status"] == report_payload["active_child"]
    assert status_payload["active_run"]["child_status"] == status_payload["active_child"]
    assert status_payload["active_child"]["semantic_progress_state"]["phase"] == "observe"



def test_persist_report_state_aligns_report_with_newer_existing_status_snapshot(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    newer_line = "[eval:round-1] phase=primary task 3/128 task-3 family=integration cognitive_stage=memory_update_written step=1 verification_passed=0"
    module._write_status(
        status_path,
        report_path=report_path,
        payload={
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 1,
                "run_match_id": "run-1",
                "phase": "campaign",
            },
            "active_child": {
                "current_task": {
                    "index": 3,
                    "total": 128,
                    "task_id": "task-3",
                    "family": "integration",
                    "phase": "primary",
                },
                "last_progress_phase": "primary",
                "last_progress_line": newer_line,
                "last_output_line": newer_line,
                "last_progress_at": 30.0,
                "semantic_progress_state": {
                    "phase": "primary",
                    "phase_family": "active",
                    "status": "active",
                    "progress_class": "degraded",
                    "decision_distance": "active",
                    "detail": "active task 3/128 failed verification and is awaiting recovery",
                    "recorded_at": 30.0,
                },
            },
            "phase_detail": newer_line,
        },
    )

    older_line = "[eval:round-1] phase=primary task 2/128 task-2 family=integration cognitive_stage=memory_retrieved subphase=graph_memory"
    payload = {
        "status": "running",
        "phase": "campaign",
        "phase_detail": older_line,
        "active_run": {
            "run_index": 1,
            "run_match_id": "run-1",
            "phase": "campaign",
        },
        "active_child": {
            "current_task": {
                "index": 2,
                "total": 128,
                "task_id": "task-2",
                "family": "integration",
                "phase": "primary",
            },
            "last_progress_phase": "primary",
            "last_progress_line": older_line,
            "last_output_line": older_line,
            "last_progress_at": 20.0,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 2/128 is progressing",
                "recorded_at": 20.0,
            },
        },
    }

    module._persist_report_state(report_path, payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert report_payload["active_child"]["current_task"]["task_id"] == "task-3"
    assert report_payload["active_run"]["child_status"] == report_payload["active_child"]
    assert status_payload["active_run"]["child_status"] == status_payload["active_child"]
    assert report_payload["active_child"]["current_task"] == status_payload["active_child"]["current_task"]
    assert report_payload["active_child"]["semantic_progress_state"] == status_payload["active_child"]["semantic_progress_state"]
    assert report_payload["active_child"]["phase"] == status_payload["active_child"]["phase"]
    assert report_payload["current_task"] == status_payload["current_task"]
    assert report_payload["last_progress_phase"] == status_payload["last_progress_phase"]
    assert report_payload["semantic_progress_state"] == status_payload["semantic_progress_state"]
    assert report_payload["canonical_progress_state"] == status_payload["canonical_progress_state"]
    assert report_payload["phase_detail"] == status_payload["phase_detail"]


def test_persist_report_state_projects_live_top_level_fields_without_existing_status_snapshot(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    live_line = (
        "[eval:round-1] phase=primary task 1/128 task-1 family=integration "
        "cognitive_stage=llm_request step=1"
    )
    payload = {
        "status": "running",
        "phase": "campaign",
        "phase_detail": live_line,
        "active_run": {
            "run_index": 1,
            "run_match_id": "run-1",
            "phase": "campaign",
        },
        "active_child": {
            "phase": "campaign",
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "task-1",
                "family": "integration",
            },
            "current_cognitive_stage": {
                "event": "cognitive_stage",
                "cognitive_stage": "llm_request",
                "phase": "primary",
                "step_index": 1,
            },
            "last_progress_phase": "observe",
            "last_progress_line": live_line,
            "last_output_line": live_line,
            "last_progress_at": 10.0,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 1/128 is progressing",
                "recorded_at": 10.0,
            },
        },
    }

    module._persist_report_state(report_path, payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert report_payload["current_task"] == status_payload["current_task"]
    assert report_payload["current_task"]["phase"] == "primary"
    assert report_payload["current_cognitive_stage"] == report_payload["active_child"]["current_cognitive_stage"]
    assert report_payload["last_progress_phase"] == "primary"
    assert report_payload["semantic_progress_state"]["phase"] == "primary"
    assert report_payload["canonical_progress_state"]["phase"] == "primary"
    assert report_payload["active_child"]["phase"] == "primary"
    assert report_payload["active_child"]["last_progress_phase"] == "primary"
    assert report_payload["phase_detail"] == status_payload["phase_detail"]
    assert report_payload["active_run"]["child_status"] == report_payload["active_child"]


def test_persist_report_state_does_not_pull_forward_child_snapshot_from_prior_status_run(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    newer_line = "[eval:round-1] phase=primary task 3/128 task-3 family=integration cognitive_stage=memory_update_written step=1 verification_passed=0"
    module._write_status(
        status_path,
        report_path=report_path,
        payload={
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 1,
                "run_match_id": "run-1",
                "phase": "campaign",
            },
            "active_child": {
                "current_task": {
                    "index": 3,
                    "total": 128,
                    "task_id": "task-3",
                    "family": "integration",
                    "phase": "primary",
                },
                "last_progress_phase": "primary",
                "last_progress_line": newer_line,
                "last_output_line": newer_line,
                "last_progress_at": 30.0,
                "semantic_progress_state": {
                    "phase": "primary",
                    "phase_family": "active",
                    "status": "active",
                    "progress_class": "degraded",
                    "decision_distance": "active",
                    "detail": "active task 3/128 failed verification and is awaiting recovery",
                    "recorded_at": 30.0,
                },
            },
            "phase_detail": newer_line,
        },
    )

    live_line = "[eval:round-2] phase=primary task 1/128 task-1 family=integration cognitive_stage=memory_retrieved subphase=graph_memory"
    payload = {
        "status": "running",
        "phase": "campaign",
        "phase_detail": live_line,
        "active_run": {
            "run_index": 2,
            "run_match_id": "run-2",
            "phase": "campaign",
        },
        "active_child": {
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "task-1",
                "family": "integration",
                "phase": "primary",
            },
            "last_progress_phase": "primary",
            "last_progress_line": live_line,
            "last_output_line": live_line,
            "last_progress_at": 40.0,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 1/128 is progressing",
                "recorded_at": 40.0,
            },
        },
    }

    module._persist_report_state(report_path, payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["active_child"]["current_task"]["task_id"] == "task-1"
    assert report_payload["phase_detail"] == live_line


def test_persist_report_state_uses_frozen_payload_snapshot(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    original_payload = {
        "status": "running",
        "phase": "campaign",
        "active_run": {
            "run_index": 1,
            "run_match_id": "run-1",
            "phase": "campaign",
            "child_status": {
                "current_task": {"task_id": "task-1", "phase": "primary"},
                "last_progress_line": "[eval:round-1] phase=primary task 1/4 task-1",
            },
        },
        "active_child": {
            "current_task": {"task_id": "task-1", "phase": "primary"},
            "last_progress_line": "[eval:round-1] phase=primary task 1/4 task-1",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "detail": "active task 1/4 is progressing",
            },
        },
    }

    def fake_write_report(path, payload, *, config=None, govern_storage=True):
        del govern_storage
        module.atomic_write_json(path, payload, config=config)
        original_payload["active_child"] = {
            "current_task": {"task_id": "task-2", "phase": "primary"},
            "last_progress_line": "[eval:round-1] phase=primary task 2/4 task-2",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "detail": "active task 2/4 is progressing",
            },
        }

    monkeypatch.setattr(module, "_write_report", fake_write_report)

    module._persist_report_state(report_path, original_payload, status_path=status_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert report_payload["active_child"]["current_task"]["task_id"] == "task-1"
    assert status_payload["active_child"]["current_task"]["task_id"] == "task-1"
    assert status_payload["active_run"]["child_status"] == status_payload["active_child"]


def test_persist_report_state_serializes_overlapping_writes(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "reports" / "campaign.json"
    status_path = tmp_path / "reports" / "campaign.status.json"
    first_report_written = threading.Event()
    allow_first_to_continue = threading.Event()
    original_write_report = module._write_report

    def payload_for(task_id: str) -> dict[str, object]:
        return {
            "status": "running",
            "phase": "campaign",
            "active_run": {
                "run_index": 1,
                "run_match_id": "run-1",
                "phase": "campaign",
            },
            "active_child": {
                "current_task": {"task_id": task_id, "phase": "primary"},
                "last_progress_line": f"[eval:round-1] phase=primary task {task_id}",
                "semantic_progress_state": {
                    "phase": "primary",
                    "phase_family": "active",
                    "status": "active",
                    "progress_class": "healthy",
                    "detail": f"active task {task_id} is progressing",
                },
            },
        }

    old_payload = payload_for("task-old")
    new_payload = payload_for("task-new")

    def fake_write_report(path, payload, *, config=None, govern_storage=True):
        del govern_storage
        original_write_report(path, payload, config=config)
        task_id = (
            payload.get("active_child", {}).get("current_task", {}).get("task_id")
            if isinstance(payload.get("active_child", {}), dict)
            else None
        )
        if task_id == "task-old":
            first_report_written.set()
            allow_first_to_continue.wait(timeout=5)

    monkeypatch.setattr(module, "_write_report", fake_write_report)

    def run_persist(payload):
        module._persist_report_state(report_path, payload, status_path=status_path)

    old_thread = threading.Thread(target=run_persist, args=(old_payload,))
    new_thread = threading.Thread(target=run_persist, args=(new_payload,))

    old_thread.start()
    assert first_report_written.wait(timeout=5)
    new_thread.start()
    allow_first_to_continue.set()
    old_thread.join(timeout=5)
    new_thread.join(timeout=5)
    assert not old_thread.is_alive()
    assert not new_thread.is_alive()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert report_payload["active_child"]["current_task"]["task_id"] == "task-new"
    assert status_payload["active_child"]["current_task"]["task_id"] == "task-new"
    assert status_payload["active_run"]["child_status"] == status_payload["active_child"]


def test_make_child_progress_callback_forwards_explicit_config(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {}
    round_payload: dict[str, object] = {}
    seen_event_configs: list[KernelConfig | None] = []
    seen_report_configs: list[KernelConfig | None] = []

    def fake_append_event(path, payload, *, config=None):
        assert path == event_log_path
        assert payload["event_kind"] == "child_event"
        seen_event_configs.append(config)

    def fake_persist_report_state(path, payload, *, config=None, status_path=None, lock_path=None, **kwargs):
        del kwargs
        assert path == report_path
        assert payload is report
        assert status_path == status_path_arg
        assert lock_path == lock_path_arg
        seen_report_configs.append(config)

    status_path_arg = status_path
    lock_path_arg = tmp_path / "campaign.lock"
    monkeypatch.setattr(module, "_append_event", fake_append_event)
    monkeypatch.setattr(module, "_persist_report_state", fake_persist_report_state)

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path_arg,
        lock_path=lock_path_arg,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
    )

    callback({"event": "start", "pid": 123, "started_at": 1.0, "timestamp": 1.0})

    assert seen_event_configs == [config]
    assert seen_report_configs == [config]


def test_make_child_progress_callback_clears_stale_intervention_on_new_child_start(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {
        "active_child": {
            "current_task": {
                "index": 8,
                "total": 8,
                "task_id": "stale_preview_task",
                "phase": "preview_candidate_eval",
            },
            "semantic_progress_state": {
                "phase": "preview_candidate_eval",
                "phase_family": "preview",
                "progress_class": "degraded",
                "detail": "preview task 8/8 failed verification and is awaiting recovery",
            },
            "controller_intervention": {
                "triggered": True,
                "reason_code": "productive_partial_preview_without_decision",
            },
        },
        "controller_intervention": {
            "triggered": True,
            "reason_code": "productive_partial_preview_without_decision",
        },
    }
    round_payload: dict[str, object] = {
        "controller_intervention": {
            "triggered": True,
            "reason_code": "productive_partial_preview_without_decision",
        }
    }

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mirrored_child_status_from_parent_status", lambda path: {})

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=tmp_path / "campaign.lock",
        config=config,
        round_payload=round_payload,
        round_index=2,
        phase="campaign",
        child_label="campaign_round_2",
        event_log_path=event_log_path,
    )

    callback(
        {
            "event": "start",
            "pid": 123,
            "started_at": 10.0,
            "timestamp": 10.0,
        }
    )

    active_child = report["active_child"]
    assert "controller_intervention" not in active_child
    assert active_child.get("current_task", {}) == {}
    assert "semantic_progress_state" not in active_child
    assert "controller_intervention" not in report
    assert "controller_intervention" not in round_payload


def test_make_child_progress_callback_ignores_long_non_path_output(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    report: dict[str, object] = {}
    round_payload: dict[str, object] = {}

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=None,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=None,
    )

    long_line = (
        "[cycle:test] finalize phase=preview_reject_reason subsystem=tooling "
        "reason_code=retention_reject_unknown reason="
        + ("archive_command_seed_task changes expected_files away from the source contract " * 8)
    )

    callback({"event": "output", "line": long_line, "pid": 123, "timestamp": 1.0})

    assert report["active_child"]["last_output_line"] == long_line.strip()
    assert report["active_child"].get("last_report_path", "") == ""


def test_mid_round_round_signal_detects_broad_observe_then_retrieval_first():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_round_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_baseline_eval",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["broad_observe_then_retrieval_first"] is True
    assert signal["definitive_decision_emitted"] is False
    assert signal["live_decision_credit_gap"] is False


def test_mid_round_round_signal_detects_live_decision_credit_gap_only_after_decision_emits():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_round_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_complete",
                "preview_state": "reject",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["broad_observe_then_retrieval_first"] is True
    assert signal["definitive_decision_emitted"] is True
    assert signal["decision_records_considered"] == 1
    assert signal["inferred_decision_credit"] is True
    assert signal["live_decision_credit_gap"] is False


def test_mid_round_controller_intervention_signal_detects_semantic_progress_drift_before_decision():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_candidate_eval",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
                "semantic_progress_state": {
                    "phase": "preview_candidate_eval",
                    "phase_family": "preview",
                    "progress_class": "stuck",
                    "progress_silence_seconds": 5.0,
                },
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "semantic_progress_drift"
    assert signal["round_signal"]["definitive_decision_emitted"] is False
    assert signal["round_signal"]["semantic_progress_drift"] is True


def test_mid_round_controller_intervention_signal_waits_for_sustained_semantic_progress_drift():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_candidate_eval",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
                "semantic_progress_state": {
                    "phase": "preview_candidate_eval",
                    "phase_family": "preview",
                    "progress_class": "degraded",
                    "progress_silence_seconds": 0.0,
                },
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["semantic_progress_drift"] is False


def test_mid_round_controller_intervention_signal_waits_for_stuck_observe_before_cutoff():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "current_task": {
                    "index": 1,
                    "total": 4,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "observe",
                },
                "observe_summary": {"passed": 0, "total": 0},
                "decision_records_considered": 0,
                "families_sampled": ["integration"],
                "semantic_progress_state": {
                    "phase": "observe",
                    "phase_family": "observe",
                    "progress_class": "degraded",
                    "progress_silence_seconds": 75.0,
                },
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"


def test_mid_round_controller_intervention_signal_waits_for_live_observe_task_progress_before_cutoff():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "current_task": {
                    "index": 10,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "observe",
                },
                "last_task_progress_at": 100.0,
                "observe_summary": {"passed": 0, "total": 0},
                "decision_records_considered": 0,
                "families_sampled": ["integration"],
                "semantic_progress_state": {
                    "phase": "observe",
                    "phase_family": "observe",
                    "progress_class": "stuck",
                    "progress_silence_seconds": 361.0,
                    "recorded_at": 100.0,
                },
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["observe_live_task_progress_fresh"] is True


def test_mid_round_controller_intervention_signal_allows_retrieval_first_to_reach_decision():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_baseline_eval",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["definitive_decision_emitted"] is False


def test_mid_round_controller_intervention_signal_cuts_off_preview_after_productive_generated_success():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "selected_subsystem": "tolbert_model",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
                },
                "semantic_progress_state": {
                    "phase": "preview_baseline_eval",
                    "phase_family": "preview",
                    "progress_class": "degraded",
                    "decision_distance": "near",
                    "progress_silence_seconds": 5.0,
                },
                "finalize_phase": "preview_baseline_eval",
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "productive_partial_preview_without_decision"
    assert signal["round_signal"]["generated_success_completed"] is True
    assert signal["round_signal"]["productive_partial"] is True


def test_mid_round_controller_intervention_signal_waits_for_preview_linger_after_phase_transition():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "selected_subsystem": "universe_constitution",
                "last_progress_phase": "preview_candidate_eval",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["integration"],
                },
                "semantic_progress_state": {
                    "phase": "preview_candidate_eval",
                    "phase_family": "preview",
                    "progress_class": "degraded",
                    "decision_distance": "near",
                    "progress_silence_seconds": 0.0,
                },
                "finalize_phase": "preview_baseline_eval",
                "families_sampled": ["integration"],
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["finalize_phase"] == "preview_candidate_eval"
    assert signal["round_signal"]["post_productive_predecision_linger"] is False


def test_mid_round_controller_intervention_signal_cuts_off_variant_generate_after_productive_generated_success():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "selected_subsystem": "tolbert_model",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
                },
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "recovery",
                    "progress_class": "healthy",
                    "decision_distance": "far",
                },
                "finalize_phase": "variant_generate",
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "post_productive_predecision_linger"
    assert signal["round_signal"]["post_productive_predecision_linger"] is True
    assert signal["round_signal"]["generated_success_completed"] is True
    assert signal["round_signal"]["productive_partial"] is True


def test_mid_round_controller_intervention_signal_cuts_off_late_generated_failure_recovery_without_decision():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "runtime_limits": {
                "max_child_runtime_seconds": 300,
            },
            "active_child": {
                "selected_subsystem": "retrieval",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "current_task": {
                    "phase": "generated_failure",
                    "index": 8,
                    "total": 8,
                },
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
                },
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "recovery",
                    "progress_class": "healthy",
                    "decision_distance": "far",
                    "runtime_elapsed_seconds": 252.0,
                },
                "finalize_phase": "generated_failure",
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "late_generated_failure_recovery_without_decision"
    assert signal["round_signal"]["late_generated_failure_recovery_without_decision"] is True


def test_mid_round_controller_intervention_signal_does_not_cut_off_generated_failure_recovery_before_decision_window():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "runtime_limits": {
                "max_child_runtime_seconds": 300,
            },
            "active_child": {
                "selected_subsystem": "retrieval",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "current_task": {
                    "phase": "generated_failure",
                    "index": 8,
                    "total": 8,
                },
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
                },
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "recovery",
                    "progress_class": "healthy",
                    "decision_distance": "far",
                    "runtime_elapsed_seconds": 180.0,
                },
                "finalize_phase": "generated_failure",
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["round_signal"]["late_generated_failure_recovery_without_decision"] is False


def test_mid_round_controller_intervention_signal_ignores_generated_failure_handoff_before_recovery_task_arrives():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "selected_subsystem": "operating_envelope",
                "last_progress_phase": "generated_failure",
                "decision_records_considered": 0,
                "partial_productive_runs": 1,
                "active_cycle_progress": {
                    "observe_completed": True,
                    "generated_success_completed": True,
                    "productive_partial": True,
                    "sampled_families_from_progress": ["integration"],
                },
                "semantic_progress_state": {
                    "phase": "generated_failure_seed",
                    "phase_family": "preview",
                    "progress_class": "degraded",
                    "decision_distance": "near",
                    "runtime_elapsed_seconds": 252.0,
                },
                "finalize_phase": "preview_baseline_eval",
                "families_sampled": ["integration"],
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["generated_failure_handoff_active"] is True
    assert signal["round_signal"]["post_productive_predecision_linger"] is False


def test_mid_round_controller_intervention_signal_ignores_stale_recovery_drift_before_decision():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "generated_failure",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
                "semantic_progress_state": {
                    "phase": "generated_failure",
                    "phase_family": "recovery",
                    "progress_class": "stuck",
                },
            },
        }
    )

    assert signal["triggered"] is False
    assert signal["reason"] == "no_live_decision_credit_gap"
    assert signal["round_signal"]["semantic_progress_drift"] is False


def test_merge_mirrored_child_status_fields_prefers_live_finalize_phase_over_stale_generated_phase():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "finalize_phase": "generated_failure",
            "last_progress_phase": "generated_failure",
            "current_task": {"phase": "generated_failure"},
        },
        {
            "active_cycle_run": {
                "finalize_phase": "preview_candidate_eval",
                "pending_decision_state": "",
                "preview_state": "",
            }
        },
    )

    assert merged["finalize_phase"] == "preview_candidate_eval"


def test_mid_round_controller_intervention_signal_detects_observe_progress_stall():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "selected_subsystem": "retrieval",
                "decision_records_considered": 0,
                "families_sampled": ["project"],
                "semantic_progress_state": {
                    "phase": "observe",
                    "phase_family": "observe",
                    "progress_class": "stuck",
                },
            },
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "observe_progress_stall"
    assert signal["round_signal"]["sampled_priority_family_count"] == 1


def test_mid_round_controller_intervention_signal_detects_repeated_verification_loop():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "active_child": {
                "selected_subsystem": "retrieval",
                "current_task_progress_timeline": [
                    {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                    {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                    {"stage": "context_compile", "step_index": 1},
                    {"stage": "world_model_updated", "step_index": 1},
                    {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                    {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                    {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                    {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                    {"stage": "context_compile", "step_index": 1},
                    {"stage": "world_model_updated", "step_index": 1},
                    {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                    {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                    {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                    {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                    {"stage": "context_compile", "step_index": 1},
                    {"stage": "world_model_updated", "step_index": 1},
                    {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                    {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                ],
            }
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "micro_step_verification_loop"
    assert signal["subsystem"] == "retrieval"
    assert signal["round_signal"]["micro_step_verification_loop"] is True


def test_mid_round_controller_intervention_signal_detects_generated_success_verification_loop_earlier():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_controller_intervention_signal(
        {
            "active_child": {
                "selected_subsystem": "retrieval",
                "last_progress_phase": "generated_success",
                "current_task": {
                    "index": 2,
                    "total": 8,
                    "task_id": "bridge_handoff_task_integration_recovery",
                    "family": "integration",
                    "phase": "generated_success",
                },
                "current_task_progress_timeline": [
                    {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                    {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                    {"stage": "context_compile", "step_index": 1},
                    {"stage": "world_model_updated", "step_index": 1},
                    {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                    {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                    {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                    {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                    {"stage": "context_compile", "step_index": 1},
                    {"stage": "world_model_updated", "step_index": 1},
                    {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                    {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                ],
            }
        }
    )

    assert signal["triggered"] is True
    assert signal["reason_code"] == "micro_step_verification_loop"
    assert signal["round_signal"]["micro_step_verification_loop_signal"]["phase"] == "generated_success"


def test_repeated_verification_loop_signal_defers_historical_generated_success_adjacent_task(monkeypatch):
    module = _load_script("run_unattended_campaign.py")

    monkeypatch.setattr(module, "_generated_success_historical_step_grace", lambda task_id: 8)

    signal = module._repeated_verification_loop_signal(
        {
            "current_task": {
                "task_id": "project_release_cutover_task_project_adjacent",
                "phase": "generated_success",
            },
            "current_task_progress_timeline": [
                {"stage": "world_model_updated", "step_index": 6},
                {"stage": "memory_update_written", "step_index": 6, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 6},
                {"stage": "verification_result", "step_index": 6, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 6},
                {"stage": "memory_update_written", "step_index": 6, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 6},
                {"stage": "verification_result", "step_index": 6, "verification_passed": False},
            ],
        }
    )

    assert signal == {"active": False}


def test_repeated_verification_loop_signal_still_triggers_historical_generated_success_adjacent_after_grace(
    monkeypatch,
):
    module = _load_script("run_unattended_campaign.py")

    monkeypatch.setattr(module, "_generated_success_historical_step_grace", lambda task_id: 8)

    signal = module._repeated_verification_loop_signal(
        {
            "current_task": {
                "task_id": "project_release_cutover_task_project_adjacent",
                "phase": "generated_success",
            },
            "current_task_progress_timeline": [
                {"stage": "world_model_updated", "step_index": 8},
                {"stage": "memory_update_written", "step_index": 8, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 8},
                {"stage": "verification_result", "step_index": 8, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 8},
                {"stage": "memory_update_written", "step_index": 8, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 8},
                {"stage": "verification_result", "step_index": 8, "verification_passed": False},
            ],
        }
    )

    assert signal["active"] is True
    assert signal["phase"] == "generated_success"
    assert signal["step_index"] == 8


def test_repeated_verification_loop_signal_defers_historical_primary_task(monkeypatch):
    module = _load_script("run_unattended_campaign.py")

    monkeypatch.setattr(module, "_historical_step_grace", lambda task_id, phase: 6)

    signal = module._repeated_verification_loop_signal(
        {
            "current_task": {
                "task_id": "project_release_cutover_task",
                "phase": "primary",
            },
            "current_task_progress_timeline": [
                {"stage": "world_model_updated", "step_index": 5},
                {"stage": "memory_update_written", "step_index": 5, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 5},
                {"stage": "verification_result", "step_index": 5, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 5},
                {"stage": "memory_update_written", "step_index": 5, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 5},
                {"stage": "verification_result", "step_index": 5, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 5},
                {"stage": "memory_update_written", "step_index": 5, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 5},
                {"stage": "verification_result", "step_index": 5, "verification_passed": False},
            ],
        }
    )

    assert signal == {"active": False}


def test_repeated_verification_loop_signal_triggers_historical_primary_task_after_grace(monkeypatch):
    module = _load_script("run_unattended_campaign.py")

    monkeypatch.setattr(module, "_historical_step_grace", lambda task_id, phase: 6)

    signal = module._repeated_verification_loop_signal(
        {
            "current_task": {
                "task_id": "project_release_cutover_task",
                "phase": "primary",
            },
            "current_task_progress_timeline": [
                {"stage": "world_model_updated", "step_index": 6},
                {"stage": "memory_update_written", "step_index": 6, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 6},
                {"stage": "verification_result", "step_index": 6, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 6},
                {"stage": "memory_update_written", "step_index": 6, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 6},
                {"stage": "verification_result", "step_index": 6, "verification_passed": False},
                {"stage": "world_model_updated", "step_index": 6},
                {"stage": "memory_update_written", "step_index": 6, "verification_passed": False},
                {"stage": "critique_reflected", "step_index": 6},
                {"stage": "verification_result", "step_index": 6, "verification_passed": False},
            ],
        }
    )

    assert signal["active"] is True
    assert signal["phase"] == "primary"
    assert signal["step_index"] == 6


def test_make_child_progress_callback_requests_mid_round_controller_intervention(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {}
    round_payload: dict[str, object] = {
        "policy": {
            "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
        }
    }
    seen_events: list[dict[str, object]] = []
    persist_calls: list[Path] = []

    monkeypatch.setattr(module, "_append_event", lambda path, payload, *, config=None: seen_events.append(dict(payload)))
    monkeypatch.setattr(
        module,
        "_persist_report_state",
        lambda path, payload, *, config=None, status_path=None, lock_path=None, **kwargs: persist_calls.append(path),
    )
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "decision_records_considered": 0,
            "families_sampled": ["project", "repository", "integration", "repo_chore"],
            "active_cycle_run": {
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_baseline_eval",
                "observe_summary": {"passed": 4, "total": 4},
                "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
            },
        },
    )

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=tmp_path / "campaign.lock",
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
    )

    action = callback(
        {
            "event": "heartbeat",
            "pid": 123,
            "timestamp": 1.0,
            "silence_seconds": 10,
        }
    )

    assert action is None
    assert "controller_intervention" not in round_payload
    assert not any(event.get("event_kind") == "controller_intervention" for event in seen_events)


def test_mid_round_round_signal_recovers_live_decision_credit_from_preview_state():
    module = _load_script("run_unattended_campaign.py")

    signal = module._mid_round_round_signal(
        {
            "policy": {
                "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
            },
            "active_child": {
                "observe_summary": {"passed": 4, "total": 4},
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_complete",
                "preview_state": "reject",
                "decision_records_considered": 0,
                "families_sampled": ["project", "repository", "integration", "repo_chore"],
            },
        }
    )

    assert signal["definitive_decision_emitted"] is True
    assert signal["decision_records_considered"] == 1
    assert signal["inferred_decision_credit"] is True
    assert signal["live_decision_credit_gap"] is False


def test_make_child_progress_callback_requests_mid_round_controller_intervention_on_semantic_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {}
    round_payload: dict[str, object] = {
        "policy": {
            "priority_benchmark_families": ["project", "repository", "integration", "repo_chore"],
        }
    }

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "decision_records_considered": 0,
            "families_sampled": ["project", "repository", "integration", "repo_chore"],
            "semantic_progress_state": {
                "phase": "preview_candidate_eval",
                "phase_family": "preview",
                "progress_class": "stuck",
                "progress_silence_seconds": 10.0,
            },
            "active_cycle_run": {
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_candidate_eval",
                "observe_summary": {"passed": 4, "total": 4},
                "sampled_families_from_progress": ["project", "repository", "integration", "repo_chore"],
            },
        },
    )

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=tmp_path / "campaign.lock",
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
    )

    action = callback(
        {
            "event": "heartbeat",
            "pid": 123,
            "timestamp": 1.0,
            "silence_seconds": 10,
        }
    )

    assert action == {
        "terminate": True,
        "reason": "mid-round controller intervention: broad observe covered priority families, preview progress drifted to stuck before decision emission",
    }
    assert round_payload["controller_intervention"]["reason_code"] == "semantic_progress_drift"


def test_merge_mirrored_child_status_fields_preserves_stuck_adaptive_state_over_healthy_child_state():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "semantic_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "stuck",
                "decision_distance": "far",
            },
            "adaptive_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "stuck",
                "decision_distance": "far",
            },
        },
        {
            "semantic_progress_state": {
                "phase": "preview_candidate_eval",
                "phase_family": "preview",
                "progress_class": "healthy",
                "decision_distance": "near",
            },
            "active_cycle_run": {
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_candidate_eval",
                "last_progress_phase": "generated_failure",
            },
        },
    )

    assert merged["semantic_progress_state"]["phase"] == "generated_failure"
    assert merged["semantic_progress_state"]["progress_class"] == "stuck"


def test_merge_mirrored_child_status_fields_prefers_fresher_adaptive_state_over_stale_semantic_state():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {},
        {
            "last_progress_phase": "generated_failure",
            "current_task": {
                "index": 3,
                "total": 8,
                "task_id": "incident_matrix_task_integration_recovery",
                "family": "integration",
                "phase": "generated_failure",
            },
            "semantic_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "degraded",
                "decision_distance": "far",
                "runtime_elapsed_seconds": 171.8,
                "detail": "recovery task 2/8 failed verification and is awaiting recovery",
            },
            "adaptive_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "healthy",
                "decision_distance": "far",
                "runtime_elapsed_seconds": 184.9,
                "detail": "recovery task 3/8 is progressing",
            },
        },
    )

    assert merged["semantic_progress_state"]["progress_class"] == "healthy"
    assert merged["semantic_progress_state"]["detail"] == "recovery task 3/8 is progressing"


def test_merge_mirrored_child_status_fields_prefers_adaptive_state_when_semantic_detail_lags_task_position():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {},
        {
            "last_progress_phase": "primary",
            "current_task": {
                "index": 4,
                "total": 8,
                "task_id": "repo_sync_matrix_task",
                "family": "repository",
                "phase": "primary",
            },
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "runtime_elapsed_seconds": 64.47,
                "detail": "active task 5/8 is progressing",
            },
            "adaptive_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "runtime_elapsed_seconds": 64.97,
                "detail": "active task 4/8 is progressing",
            },
        },
    )

    assert merged["semantic_progress_state"]["detail"] == "active task 4/8 is progressing"


def test_merge_mirrored_child_status_fields_preserves_newer_live_task_over_stale_mirrored_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 200.0,
            "last_progress_phase": "generated_failure",
            "last_progress_line": (
                "[eval:test] phase=generated_failure task 1/8 bridge_handoff_task_file_recovery family=transition_pressure"
            ),
            "current_task": {
                "index": 1,
                "total": 8,
                "task_id": "bridge_handoff_task_file_recovery",
                "family": "transition_pressure",
                "phase": "generated_failure",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "memory_retrieved",
                "phase": "generated_failure",
                "step_subphase": "graph_memory",
                "timestamp": 200.0,
            },
            "current_task_progress_timeline": [
                {
                    "cognitive_stage": "memory_retrieved",
                    "phase": "generated_failure",
                    "step_subphase": "graph_memory",
                    "timestamp": 200.0,
                }
            ],
            "semantic_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "healthy",
                "decision_distance": "far",
                "detail": "recovery task 1/8 is progressing",
                "recorded_at": 200.0,
            },
        },
        {
            "last_progress_at": 150.0,
            "last_progress_phase": "generated_failure_seed",
            "current_task": {
                "index": 8,
                "total": 8,
                "task_id": "archive_command_seed_task",
                "family": "bounded",
                "phase": "generated_failure_seed",
            },
            "semantic_progress_state": {
                "phase": "generated_failure_seed",
                "phase_family": "recovery",
                "progress_class": "healthy",
                "decision_distance": "far",
                "detail": "recovery task 8/8 reached the execution boundary and is awaiting verification",
                "recorded_at": 150.0,
            },
            "active_cycle_run": {
                "current_task": {
                    "index": 8,
                    "total": 8,
                    "task_id": "archive_command_seed_task",
                    "family": "bounded",
                    "phase": "generated_failure_seed",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "memory_update_written",
                    "phase": "generated_failure_seed",
                    "step_index": 2,
                    "verification_passed": True,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "memory_update_written",
                        "phase": "generated_failure_seed",
                        "step_index": 2,
                        "verification_passed": True,
                    }
                ],
            },
        },
    )

    assert merged["last_progress_phase"] == "generated_failure"
    assert merged["current_task"]["task_id"] == "bridge_handoff_task_file_recovery"
    assert merged["current_cognitive_stage"]["cognitive_stage"] == "memory_retrieved"
    assert merged["semantic_progress_state"]["phase"] == "generated_failure"
    assert merged["current_task_progress_timeline"][0]["phase"] == "generated_failure"


def test_merge_mirrored_child_status_fields_reconciles_semantic_phase_to_live_generated_task():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 220.0,
            "last_progress_phase": "generated_failure",
            "current_task": {
                "index": 1,
                "total": 8,
                "task_id": "bridge_handoff_task_file_recovery",
                "family": "transition_pressure",
                "phase": "generated_failure",
            },
            "semantic_progress_state": {
                "phase": "generated_failure_seed",
                "phase_family": "recovery",
                "progress_class": "healthy",
                "decision_distance": "far",
                "detail": "recovery task 8/8 reached the execution boundary and is awaiting verification",
                "recorded_at": 210.0,
            },
        },
        {},
    )

    assert merged["semantic_progress_state"]["phase"] == "generated_failure"


def test_merge_mirrored_child_status_fields_preserves_live_same_task_step_over_lagging_cycle_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 200.0,
            "last_progress_phase": "primary",
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "semantic_open_world_task",
                "family": "integration",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "llm_response",
                "phase": "primary",
                "step_index": 2,
            },
            "current_task_progress_timeline": [
                {
                    "cognitive_stage": "verification_result",
                    "phase": "primary",
                    "step_index": 1,
                    "verification_passed": False,
                },
                {
                    "cognitive_stage": "llm_response",
                    "phase": "primary",
                    "step_index": 2,
                },
            ],
            "current_task_verification_passed": None,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 1/128 is progressing",
                "recorded_at": 200.0,
            },
        },
        {
            "report_kind": "repeated_improvement_status",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "degraded",
                "decision_distance": "active",
                "detail": "active task 1/128 failed verification and is awaiting recovery",
                "recorded_at": 201.0,
            },
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "verification_result",
                    "phase": "primary",
                    "step_index": 1,
                    "verification_passed": False,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "verification_result",
                        "phase": "primary",
                        "step_index": 1,
                        "verification_passed": False,
                    }
                ],
                "current_task_verification_passed": False,
            },
        },
    )

    assert merged["current_task"]["task_id"] == "semantic_open_world_task"
    assert merged["current_cognitive_stage"]["cognitive_stage"] == "llm_response"
    assert merged["current_cognitive_stage"]["step_index"] == 2
    assert merged["current_task_progress_timeline"][-1]["cognitive_stage"] == "llm_response"
    assert merged["current_task_progress_timeline"][-1]["step_index"] == 2
    assert merged["current_task_verification_passed"] is None
    assert merged["semantic_progress_state"]["progress_class"] == "healthy"
    assert merged["semantic_progress_state"]["detail"] == "active task 1/128 is progressing"


def test_merge_mirrored_child_status_fields_preserves_live_same_step_over_ahead_of_pipe_cycle_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 200.0,
            "last_progress_phase": "primary",
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "semantic_open_world_task",
                "family": "integration",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "llm_response",
                "phase": "primary",
                "step_index": 2,
            },
            "current_task_progress_timeline": [
                {
                    "cognitive_stage": "context_compile",
                    "phase": "primary",
                    "step_index": 2,
                },
                {
                    "cognitive_stage": "llm_response",
                    "phase": "primary",
                    "step_index": 2,
                },
            ],
            "current_task_verification_passed": None,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 1/128 is progressing",
                "recorded_at": 200.0,
            },
        },
        {
            "report_kind": "repeated_improvement_status",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "degraded",
                "decision_distance": "active",
                "detail": "active task 1/128 failed verification and is awaiting recovery",
                "recorded_at": 201.0,
            },
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "memory_update_written",
                    "phase": "primary",
                    "step_index": 2,
                    "verification_passed": False,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "memory_update_written",
                        "phase": "primary",
                        "step_index": 2,
                        "verification_passed": False,
                    }
                ],
                "current_task_verification_passed": False,
            },
        },
    )

    assert merged["current_task"]["task_id"] == "semantic_open_world_task"
    assert merged["current_cognitive_stage"]["cognitive_stage"] == "llm_response"
    assert merged["current_cognitive_stage"]["step_index"] == 2
    assert merged["current_task_progress_timeline"][-1]["cognitive_stage"] == "llm_response"
    assert merged["current_task_progress_timeline"][-1]["step_index"] == 2
    assert merged["current_task_verification_passed"] is None
    assert merged["semantic_progress_state"]["progress_class"] == "healthy"
    assert merged["semantic_progress_state"]["detail"] == "active task 1/128 is progressing"


def test_merge_mirrored_child_status_fields_preserves_live_task_over_ahead_of_pipe_future_task_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 200.0,
            "last_progress_phase": "primary",
            "current_task": {
                "index": 1,
                "total": 4,
                "task_id": "semantic_open_world_20260411T225301327576Z_round_1_integration",
                "family": "integration",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "payload_build",
                "phase": "primary",
                "step_index": 2,
            },
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 1/4 is progressing",
                "recorded_at": 200.0,
            },
        },
        {
            "report_kind": "repeated_improvement_status",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 3/4 is progressing",
                "recorded_at": 201.0,
            },
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 3,
                    "total": 4,
                    "task_id": "semantic_open_world_20260411T225301327576Z_round_1_project",
                    "family": "project",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "transition_simulated",
                    "phase": "primary",
                    "step_index": 1,
                },
                "semantic_progress_state": {
                    "phase": "primary",
                    "phase_family": "active",
                    "progress_class": "healthy",
                    "decision_distance": "active",
                    "detail": "active task 3/4 is progressing",
                    "recorded_at": 201.0,
                },
            },
        },
    )

    assert merged["current_task"]["task_id"] == "semantic_open_world_20260411T225301327576Z_round_1_integration"
    assert merged["current_task"]["index"] == 1
    assert merged["semantic_progress_state"]["detail"] == "active task 1/4 is progressing"


def test_merge_mirrored_child_status_fields_prefers_healthier_same_step_mirrored_semantic_state():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_at": 200.0,
            "last_progress_phase": "primary",
            "current_task": {
                "index": 1,
                "total": 1,
                "task_id": "semantic_open_world_task",
                "family": "integration",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "llm_request",
                "phase": "primary",
                "step_index": 1,
            },
            "semantic_progress_state": {
                "phase": "preview_candidate_eval",
                "phase_family": "preview",
                "progress_class": "degraded",
                "decision_distance": "near",
                "detail": "preview work is advancing slowly relative to the runtime budget",
                "recorded_at": 200.0,
            },
        },
        {
            "report_kind": "repeated_improvement_status",
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 1,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "plan_candidates",
                    "phase": "primary",
                    "step_index": 1,
                },
                "semantic_progress_state": {
                    "phase": "preview_candidate_eval",
                    "phase_family": "preview",
                    "progress_class": "healthy",
                    "decision_distance": "near",
                    "detail": "preview task 1/1 reached the execution boundary and is awaiting verification",
                    "recorded_at": 199.0,
                },
            },
        },
    )

    assert merged["semantic_progress_state"]["phase"] == "preview_candidate_eval"
    assert merged["semantic_progress_state"]["progress_class"] == "healthy"
    assert (
        merged["semantic_progress_state"]["detail"]
        == "preview task 1/1 reached the execution boundary and is awaiting verification"
    )


def test_authoritative_child_supervision_snapshot_preserves_newer_live_task_over_stale_mirrored_status():
    module = _load_script("run_unattended_campaign.py")

    snapshot = module._authoritative_child_supervision_snapshot(
        {
            "report_kind": "repeated_improvement_status",
            "started_at": 100.0,
            "current_task": {
                "index": 4,
                "total": 4,
                "task_id": "repo_notice_review_task",
                "family": "repo_chore",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "memory_update_written",
                "phase": "primary",
                "step_index": 1,
                "verification_passed": False,
            },
            "current_task_verification_passed": False,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "degraded",
                "decision_distance": "active",
                "detail": "active task 4/4 failed verification and is awaiting recovery",
                "recorded_at": 150.0,
            },
        },
        last_progress_phase="primary",
        active_finalize_phase="preview_baseline_eval",
        last_progress_at=200.0,
        current_task={
            "index": 2,
            "total": 4,
            "task_id": "repository_guardrail_sync_task",
            "family": "repository",
            "phase": "primary",
        },
        current_cognitive_stage={
            "cognitive_stage": "memory_retrieved",
            "phase": "primary",
            "step_subphase": "graph_memory",
        },
        observe_summary={},
        pending_decision_state="",
        preview_state="",
        current_task_verification_passed=None,
    )

    assert snapshot["current_task"]["task_id"] == "repository_guardrail_sync_task"
    assert snapshot["current_task"]["index"] == 2
    assert snapshot["current_cognitive_stage"]["cognitive_stage"] == "memory_retrieved"
    assert snapshot["current_task_verification_passed"] is None


def test_merge_mirrored_child_status_fields_preserves_live_generated_success_over_stale_preview_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_phase": "generated_success",
            "current_task": {
                "index": 7,
                "total": 8,
                "task_id": "bridge_handoff_task_integration_recovery",
                "family": "integration",
                "phase": "generated_success",
            },
            "semantic_progress_state": {
                "phase": "generated_success",
                "phase_family": "finalize",
                "progress_class": "healthy",
                "decision_distance": "near",
                "detail": "finalize task 7/8 is progressing",
            },
        },
        {
            "semantic_progress_state": {
                "phase": "preview_candidate_eval",
                "phase_family": "preview",
                "progress_class": "healthy",
                "decision_distance": "near",
                "detail": "preview task 7/8 is progressing",
            },
            "active_cycle_run": {
                "selected_subsystem": "retrieval",
                "finalize_phase": "preview_candidate_eval",
                "last_progress_phase": "preview_candidate_eval",
                "current_task": {
                    "index": 7,
                    "total": 8,
                    "task_id": "bridge_handoff_task_integration_recovery",
                    "family": "integration",
                    "phase": "preview_candidate_eval",
                },
            },
        },
    )

    assert merged["last_progress_phase"] == "generated_success"
    assert merged["current_task"]["phase"] == "generated_success"
    assert merged["semantic_progress_state"]["phase"] == "generated_success"
    assert merged["semantic_progress_state"]["detail"] == "finalize task 7/8 is progressing"


def test_merge_mirrored_child_status_fields_canonicalizes_stale_adaptive_state_behind_live_task():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_phase": "generated_failure",
            "last_progress_at": 279.3,
            "last_progress_line": (
                "[eval:test] phase=generated_failure task 8/8 archive_command_seed_task_file_recovery family=bounded"
            ),
            "current_task": {
                "index": 8,
                "total": 8,
                "task_id": "archive_command_seed_task_file_recovery",
                "family": "bounded",
                "phase": "generated_failure",
            },
            "semantic_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "status": "active",
                "progress_class": "degraded",
                "decision_distance": "far",
                "runtime_elapsed_seconds": 279.36,
                "detail": "recovery task 8/8 failed verification and is awaiting recovery",
            },
            "adaptive_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "far",
                "runtime_elapsed_seconds": 278.71,
                "detail": "recovery task 7/8 is progressing",
                "recorded_at": 1775751943.57,
            },
        },
        {},
    )

    assert merged["canonical_progress_state"]["detail"] == "recovery task 8/8 failed verification and is awaiting recovery"
    assert merged["semantic_progress_state"]["detail"] == "recovery task 8/8 failed verification and is awaiting recovery"
    assert merged["adaptive_progress_state"]["detail"] == "recovery task 8/8 failed verification and is awaiting recovery"


def test_merge_mirrored_child_status_fields_preserves_live_variant_generate_over_stale_generated_failure_snapshot():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_phase": "variant_generate",
            "current_task": {
                "index": 8,
                "total": 8,
                "task_id": "archive_command_seed_task_file_recovery",
                "family": "bounded",
                "phase": "generated_failure",
            },
        },
        {
            "active_cycle_run": {
                "last_progress_phase": "generated_failure",
                "current_task": {
                    "index": 8,
                    "total": 8,
                    "task_id": "archive_command_seed_task_file_recovery",
                    "family": "bounded",
                    "phase": "generated_failure",
                },
            },
        },
    )

    assert merged["last_progress_phase"] == "variant_generate"


def test_merge_mirrored_child_status_fields_clears_stale_primary_task_after_campaign_plan_complete():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "last_progress_phase": "primary",
            "last_progress_line": (
                "[eval:test] phase=primary task 16/16 service_release_task "
                "family=repository cognitive_stage=memory_update_written step=1 verification_passed=1"
            ),
            "last_output_line": (
                "[eval:test] phase=primary task 16/16 service_release_task "
                "family=repository cognitive_stage=memory_update_written step=1 verification_passed=1"
            ),
            "current_task": {
                "index": 16,
                "total": 16,
                "task_id": "service_release_task",
                "family": "repository",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "memory_update_written",
                "phase": "primary",
                "step_index": 1,
                "verification_passed": True,
            },
            "current_task_progress_timeline": [
                {
                    "cognitive_stage": "memory_update_written",
                    "phase": "primary",
                    "step_index": 1,
                    "verification_passed": True,
                }
            ],
            "current_task_verification_passed": True,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 16/16 reached the execution boundary and is awaiting verification",
                "recorded_at": 500.0,
            },
            "adaptive_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 16/16 reached the execution boundary and is awaiting verification",
                "recorded_at": 500.0,
            },
        },
        {
            "report_kind": "repeated_improvement_status",
            "active_cycle_run": {
                "last_progress_line": (
                    "[cycle:test] campaign_plan complete subsystem=tolbert_model "
                    "variant_strategy=adaptive_history selected_variants=1"
                ),
                "last_output_line": (
                    "[cycle:test] campaign_plan complete subsystem=tolbert_model "
                    "variant_strategy=adaptive_history selected_variants=1"
                ),
                "last_progress_phase": "campaign_select",
                "current_task": {},
                "current_cognitive_stage": {},
                "current_task_progress_timeline": [],
                "semantic_progress_state": {
                    "phase": "campaign_select",
                    "phase_family": "active",
                    "status": "active",
                    "progress_class": "healthy",
                    "decision_distance": "active",
                    "detail": "active work is progressing",
                },
            },
            "active_cycle_progress": {
                "observe_completed": True,
                "last_progress_phase": "campaign_select",
            },
        },
    )

    assert merged["last_progress_phase"] == "campaign_select"
    assert merged["current_task"] == {}
    assert merged["current_cognitive_stage"] == {}
    assert merged["current_task_progress_timeline"] == []
    assert merged["current_task_verification_passed"] is None
    assert "phase=campaign_select" in merged["last_progress_line"]
    assert "task 16/16" not in merged["last_progress_line"]
    assert merged["semantic_progress_state"]["phase"] == "campaign_select"


def test_run_and_stream_terminates_child_on_interrupt(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_terminated: list[int] = []

    class FakeStdout:
        def fileno(self):
            return 0

    class FakeProcess:
        def __init__(self):
            self.pid = 2468
            self.stdout = FakeStdout()

        def poll(self):
            return None

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            raise KeyboardInterrupt("stop")

        def close(self):
            return None

    monkeypatch.setattr(module, "spawn_process_group", lambda cmd, *, cwd, env, text, bufsize: FakeProcess())
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: seen_terminated.append(process.pid))

    try:
        module._run_and_stream(
            ["python", "-u", "child.py"],
            cwd=tmp_path,
            env={},
            progress_label="child",
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert seen_terminated == [2468]


def test_run_and_stream_honors_controller_termination_action(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_terminated: list[int] = []

    class FakeStdout:
        def __init__(self):
            self._lines = ["[cycle:test] finalize phase=preview_baseline_eval subsystem=retrieval\n", ""]

        def fileno(self):
            return 0

        def readline(self):
            return self._lines.pop(0)

    class FakeProcess:
        def __init__(self):
            self.pid = 2468
            self.stdout = FakeStdout()

        def poll(self):
            return None

    class FakeSelector:
        def __init__(self, stdout):
            self.stdout = stdout

        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            return [(type("K", (), {"fileobj": self.stdout})(), None)]

        def close(self):
            return None

    monkeypatch.setattr(module, "spawn_process_group", lambda cmd, *, cwd, env, text, bufsize: FakeProcess())
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector(FakeStdout()))
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: seen_terminated.append(process.pid))

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="child",
        on_event=lambda event: {"terminate": True, "reason": "controller stop"} if event.get("event") == "output" else None,
    )

    assert result["timed_out"] is True
    assert result["timeout_reason"] == "controller stop"
    assert seen_terminated == [2468]


def test_dispatch_alerts_supports_concrete_transports(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "campaign.json"
    payload = {"status": "safe_stop", "phase": "campaign", "rounds_completed": 1, "reason": "failure"}
    seen_requests: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    def fake_urlopen(request, timeout=0):
        seen_requests.append(
            {
                "url": request.full_url,
                "body": request.data.decode("utf-8"),
                "timeout": timeout,
            }
        )
        return FakeResponse(202, b"ok")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    result = module._dispatch_alerts(
        alert_command="",
        slack_webhook_url="https://hooks.slack.test/services/T/B/X",
        pagerduty_routing_key="routing-key",
        report_path=report_path,
        status_path=None,
        event_log_path=None,
        payload=payload,
    )

    assert "slack" in result
    assert "pagerduty" in result
    assert seen_requests[0]["url"].startswith("https://hooks.slack.test/")
    assert seen_requests[1]["url"] == "https://events.pagerduty.com/v2/enqueue"


def test_dispatch_alerts_deduplicates_by_rate_limit(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "campaign.json"
    state_path = tmp_path / "alerts.json"
    payload = {"status": "safe_stop", "phase": "campaign", "rounds_completed": 1, "reason": "failure"}
    seen_commands: list[str] = []

    monkeypatch.setattr(
        module,
        "_run_alert_command",
        lambda command, **kwargs: (seen_commands.append(command) or {"ran": True, "command": command}),
    )

    first = module._dispatch_alerts(
        alert_command="echo alert",
        slack_webhook_url="",
        pagerduty_routing_key="",
        report_path=report_path,
        status_path=None,
        event_log_path=None,
        payload=payload,
        state_path=state_path,
        rate_limit_seconds=3600.0,
    )
    second = module._dispatch_alerts(
        alert_command="echo alert",
        slack_webhook_url="",
        pagerduty_routing_key="",
        report_path=report_path,
        status_path=None,
        event_log_path=None,
        payload=payload,
        state_path=state_path,
        rate_limit_seconds=3600.0,
    )

    assert first["command"]["ran"] is True
    assert second["suppressed"] is True
    assert seen_commands == ["echo alert"]


def test_acquire_external_campaign_lease_uses_http_endpoint(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    report_path = tmp_path / "campaign.json"
    seen_requests: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        module,
        "_post_json",
        lambda url, payload, headers=None: (
            seen_requests.append((url, payload))
            or {"acquired": True, "lease_id": "lease-123", "detail": "ok"}
        ),
    )

    result = module._acquire_external_campaign_lease(
        "https://lease.service.local",
        token="secret",
        report_path=report_path,
    )

    assert result["passed"] is True
    assert result["lease_id"] == "lease-123"
    assert seen_requests[0][0] == "https://lease.service.local/acquire"


def test_in_repo_lease_registry_acquires_heartbeats_and_releases(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_lease_server",
        Path(__file__).resolve().parents[1] / "scripts" / "run_lease_server.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = module.LeaseRegistry(
        state_path=tmp_path / "lease_state.json",
        ttl_seconds=60.0,
        bearer_token="",
    )
    acquired = registry.acquire(
        {"lease_kind": "agentkernel_unattended_campaign", "hostname": "host-a", "pid": 123, "report_path": "x.json"}
    )
    heartbeat = registry.heartbeat({"lease_id": acquired["lease_id"], "status": "running", "phase": "campaign"})
    released = registry.release({"lease_id": acquired["lease_id"]})

    assert acquired["acquired"] is True
    assert heartbeat["ok"] is True
    assert released["released"] is True


def test_next_round_policy_applies_multi_round_subsystem_cooldowns():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {"retrieval": 2},
    }
    campaign_report = {
        "production_yield_summary": {
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "average_retained_pass_rate_delta": 0.0,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        "recent_production_decisions": [
            {
                "subsystem": "policy",
                "state": "reject",
                "metrics_summary": {"baseline_pass_rate": 0.8, "candidate_pass_rate": 0.7},
            }
        ],
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report=campaign_report,
        liftoff_payload=None,
        round_payload={},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    assert "retrieval" in next_policy["excluded_subsystems"]
    assert "policy" in next_policy["excluded_subsystems"]


def test_next_round_policy_uses_learned_controller_signal():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state(exploration_bonus=0.0)
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    preferred_policy = {
        **current_policy,
        "focus": "discovered_task_adaptation",
        "adaptive_search": True,
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "task_limit": 256,
    }
    neutral_observation = module._controller_observation(campaign_report={}, liftoff_payload=None)
    strong_observation = module._controller_observation(
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "average_retained_pass_rate_delta": 0.09,
                "average_retained_step_delta": -0.5,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload={"liftoff_report": {"state": "shadow_only"}, "dataset_readiness": {"allow_kernel_autobuild": True}},
    )
    for _ in range(3):
        controller_state, _ = update_controller_state(
            controller_state,
            start_observation=neutral_observation,
            action_policy=preferred_policy,
            end_observation=strong_observation,
        )

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        round_payload={},
        controller_state=controller_state,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["adaptive_search"] is True


def test_controller_observation_derives_frontier_and_semantic_controller_features():
    module = _load_script("run_unattended_campaign.py")

    observation = module._controller_observation(
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "productive_depth_retained_cycles": 1,
                "long_horizon_retained_cycles": 1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "priority_families_with_retained_gain": ["project", "repository"],
            },
            "recent_production_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "retain",
                    "metrics_summary": {"pass_rate_delta": 0.05},
                },
                {
                    "subsystem": "tooling",
                    "state": "retain",
                    "metrics_summary": {"pass_rate_delta": 0.02},
                },
            ],
        },
        liftoff_payload=None,
        curriculum_controls={
            "frontier_failure_motif_priority_pairs": ["project:no_state_progress"],
            "frontier_repo_setting_priority_pairs": ["repository:worker_handoff"],
        },
        round_payload={
            "policy": {"priority_benchmark_families": ["project", "repository", "integration"]},
            "active_child": {
                "active_cycle_run": {
                    "semantic_progress_state": {
                        "phase": "preview_candidate_eval",
                        "phase_family": "preview",
                        "progress_class": "healthy",
                        "decision_distance": "near",
                    }
                }
            },
        },
    )

    assert observation["features"]["novel_subsystem_capability_gain"] > 0.0
    assert observation["features"]["strategy_hook_coverage"] > 0.0
    assert observation["round_signal"]["semantic_progress_state"]["phase_family"] == "preview"


def test_next_round_policy_records_policy_shift_rationale_and_historical_adjustment():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 32,
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    prior_rounds = [
        {
            "round_index": 1,
            "policy": dict(current_policy),
            "policy_shift_rationale": {
                "reason_codes": ["campaign_breadth_pressure", "no_yield_round", "stalled_subsystem_cooldown"],
                "cooled_subsystems": ["retrieval"],
                "widened_dimensions": ["task_limit", "campaign_width"],
            },
        }
    ]
    round_payload: dict[str, object] = {}
    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "planner_pressure_summary": {
                "campaign_breadth_pressure_cycles": 1,
                "variant_breadth_pressure_cycles": 1,
                "recent_campaign_pressures": [
                    {"subsystem": "retrieval", "campaign_breadth_pressure": 0.4},
                ],
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        controller_state=default_controller_state(exploration_bonus=0.0),
        prior_rounds=prior_rounds,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "campaign_breadth_pressure" in rationale["reason_codes"]
    assert "variant_breadth_pressure" in rationale["reason_codes"]
    assert "retrieval" in rationale["cooled_subsystems"]
    assert "campaign_width" in rationale["widened_dimensions"]
    assert "variant_width" in rationale["widened_dimensions"]
    assert round_payload["controller_planner"]["selected_historical_rationale_adjustment"] <= 0.0
    assert next_policy["adaptive_search"] is True


def test_subsystem_signal_marks_repeated_zero_yield_retrieval_for_longer_cooldown():
    module = _load_script("run_unattended_campaign.py")

    subsystem_signal = module._subsystem_signal(
        {
            "recent_production_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "tooling",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 0.99},
                },
            ]
        }
    )

    assert subsystem_signal["cooldown_candidates"][0] == "retrieval"
    assert subsystem_signal["cooldown_rounds_by_subsystem"]["retrieval"] == 3
    assert subsystem_signal["cooldown_rounds_by_subsystem"]["tooling"] == 2
    assert subsystem_signal["zero_yield_dominant_subsystems"] == ["retrieval"]


def test_subsystem_signal_falls_back_to_partial_productive_reject_runs_when_decisions_missing():
    module = _load_script("run_unattended_campaign.py")

    subsystem_signal = module._subsystem_signal(
        {
            "recent_production_decisions": [],
            "runs": [
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": (
                        "[cycle:demo] finalize phase=apply_decision subsystem=retrieval state=reject\n"
                        "[cycle:demo] finalized subsystem=retrieval state=reject\n"
                    ),
                },
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] finalized subsystem=retrieval state=reject\n",
                },
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] finalized subsystem=retrieval state=reject\n",
                },
                {
                    "subsystem": "tooling",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] finalized subsystem=tooling state=reject\n",
                },
                {
                    "subsystem": "planner",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] still running\n",
                },
            ],
        }
    )

    assert subsystem_signal["cooldown_candidates"][0] == "retrieval"
    assert subsystem_signal["cooldown_rounds_by_subsystem"]["retrieval"] == 3
    assert subsystem_signal["cooldown_rounds_by_subsystem"]["tooling"] == 2
    assert "planner" not in subsystem_signal["cooldown_rounds_by_subsystem"]
    assert subsystem_signal["zero_yield_dominant_subsystems"] == ["retrieval"]


def test_subsystem_monoculture_signal_detects_dominant_zero_yield_subsystem():
    module = _load_script("run_unattended_campaign.py")

    planner_pressure = module._planner_pressure_signal(
        {
            "planner_pressure_summary": {
                "campaign_breadth_pressure_cycles": 2,
                "variant_breadth_pressure_cycles": 1,
                "recent_campaign_pressures": [
                    {"subsystem": "retrieval"},
                    {"subsystem": "retrieval"},
                    {"subsystem": "retrieval"},
                    {"subsystem": "tooling"},
                ],
            }
        }
    )
    subsystem_signal = module._subsystem_signal(
        {
            "production_yield_summary": {"retained_cycles": 0},
            "recent_production_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "tooling",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 0.99},
                },
            ],
        }
    )

    monoculture = module._subsystem_monoculture_signal(
        {"production_yield_summary": {"retained_cycles": 0}},
        subsystem_signal=subsystem_signal,
        planner_pressure_signal=planner_pressure,
    )

    assert monoculture["active"] is True
    assert monoculture["severe"] is True
    assert monoculture["dominant_subsystem"] == "retrieval"
    assert monoculture["dominant_rejected"] == 3
    assert monoculture["dominant_share"] == 0.75


def test_next_round_policy_applies_longer_cooldown_to_repeated_zero_yield_retrieval():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "adaptive_search": True,
        "task_limit": 64,
        "task_step_floor": 50,
        "tolbert_device": "cuda:2",
        "focus": "discovered_task_adaptation",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 3,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "recent_production_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
            ],
            "planner_pressure_summary": {
                "recent_campaign_pressures": [
                    {"subsystem": "retrieval", "campaign_breadth_pressure": 0.3},
                ],
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert "retrieval" in next_policy["excluded_subsystems"]
    assert "dominant_zero_yield_subsystem_cooldown" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["cooldown_rounds_by_subsystem"]["retrieval"] == 3
    assert rationale["planner_pressure"]["zero_yield_dominant_subsystems"] == ["retrieval"]


def test_next_round_policy_flags_subsystem_monoculture_and_forces_broader_followup():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 2,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 32,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 4,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "recent_production_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 1.0},
                },
                {
                    "subsystem": "tooling",
                    "state": "reject",
                    "metrics_summary": {"baseline_pass_rate": 1.0, "candidate_pass_rate": 0.99},
                },
            ],
            "planner_pressure_summary": {
                "campaign_breadth_pressure_cycles": 3,
                "variant_breadth_pressure_cycles": 1,
                "recent_campaign_pressures": [
                    {"subsystem": "retrieval"},
                    {"subsystem": "retrieval"},
                    {"subsystem": "retrieval"},
                    {"subsystem": "tooling"},
                ],
            },
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": [],
                "missing_required_families": ["project", "repository", "integration"],
                "distinct_family_gap": 3,
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "subsystem_monoculture" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["subsystem_monoculture"]["active"] is True
    assert rationale["planner_pressure"]["subsystem_monoculture"]["dominant_subsystem"] == "retrieval"
    assert next_policy["adaptive_search"] is True
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["campaign_width"] == 4
    assert next_policy["variant_width"] >= 2
    assert next_policy["task_limit"] >= 128
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert "retrieval" in next_policy["excluded_subsystems"]


def test_next_round_policy_applies_cooldown_from_partial_productive_reject_runs_without_decisions():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "adaptive_search": True,
        "task_limit": 64,
        "task_step_floor": 50,
        "tolbert_device": "cuda:2",
        "focus": "discovered_task_adaptation",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 3,
                "decision_runs": 0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "recent_production_decisions": [],
            "runs": [
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] finalized subsystem=retrieval state=reject\n",
                },
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "[cycle:demo] finalized subsystem=retrieval state=reject\n",
                },
                {
                    "subsystem": "retrieval",
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": (
                        "[cycle:demo] finalize phase=apply_decision subsystem=retrieval state=reject\n"
                        "[cycle:demo] finalized subsystem=retrieval state=reject\n"
                    ),
                },
            ],
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert "retrieval" in next_policy["excluded_subsystems"]
    assert "dominant_zero_yield_subsystem_cooldown" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["cooldown_rounds_by_subsystem"]["retrieval"] == 3
    assert rationale["planner_pressure"]["zero_yield_dominant_subsystems"] == ["retrieval"]


def test_next_round_policy_records_external_family_coverage_gap():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "external_family_coverage_gap" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["external_report_count"] == 0
    assert rationale["planner_pressure"]["distinct_external_benchmark_families"] == 0
    assert next_policy["adaptive_search"] is True


def test_next_round_policy_cools_stalled_subsystem_after_productive_partial_family_sampling_without_decisions():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {
        "active_child": {
            "last_progress_line": "[cycle:test] finalize phase=preview_candidate_eval subsystem=retrieval",
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
                "sampled_families_from_progress": ["project", "repository", "integration"],
            },
        }
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "inheritance_summary": {"runtime_managed_decisions": 0},
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "priority_families_with_signal": [],
                "priority_families_without_signal": ["project", "repository", "integration"],
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert "retrieval" in next_policy["excluded_subsystems"]
    assert "productive_partial_conversion_gap" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["productive_partial_conversion_gap"] is True
    assert rationale["planner_pressure"]["productive_partial_sampled_priority_families"] == [
        "project",
        "repository",
        "integration",
    ]


def test_next_round_policy_treats_intermediate_decision_evidence_as_conversion_not_gap():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {
        "active_child": {
            "last_progress_line": "[cycle:test] finalize phase=preview_complete subsystem=retrieval state=reject",
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
                "sampled_families_from_progress": ["project", "repository", "integration"],
            },
        }
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "inheritance_summary": {
                "runtime_managed_decisions": 0,
                "non_runtime_managed_decisions": 1,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 1,
                "decision_runs": 1,
                "partial_productive_without_decision_runs": 0,
            },
            "recent_non_runtime_decisions": [
                {"cycle_id": "cycle:retrieval:1", "state": "reject", "subsystem": "retrieval"},
            ],
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "priority_families_with_signal": ["project", "repository", "integration"],
                "priority_families_without_signal": [],
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "productive_partial_conversion_gap" not in rationale["reason_codes"]
    assert rationale["planner_pressure"]["productive_partial_conversion_gap"] is False
    assert rationale["planner_pressure"]["intermediate_decision_evidence"] is True


def test_next_round_policy_prioritizes_missing_required_benchmark_families():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["repo_chore"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["repo_chore", "repo_sandbox", "project", "repository", "integration"],
                "required_families_with_reports": ["repo_chore", "repo_sandbox"],
                "missing_required_families": ["project", "repository", "integration"],
                "distinct_family_gap": 3,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "required_family_coverage_gap" in rationale["reason_codes"]
    assert set(rationale["planner_pressure"]["missing_required_families"]) == {
        "project",
        "repository",
        "integration",
    }
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert set(next_policy["priority_benchmark_families"]) == {"project", "repository", "integration"}


def test_next_round_policy_reprioritizes_priority_families_without_retained_gain():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.2,
                    },
                    "repository": {
                        "observed_decisions": 1,
                        "observed_estimated_cost": 2.0,
                        "retained_positive_delta_decisions": 0,
                    },
                    "integration": {
                        "observed_decisions": 0,
                        "observed_estimated_cost": 0.0,
                        "retained_positive_delta_decisions": 0,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "priority_family_under_sampled" in rationale["reason_codes"]
    assert "retained_gain_conversion_gap" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["priority_families_with_retained_gain"] == ["project"]
    assert rationale["planner_pressure"]["priority_families_without_signal"] == ["integration"]
    assert rationale["planner_pressure"]["priority_families_needing_retained_gain_conversion"] == [
        "repository",
    ]
    assert rationale["planner_pressure"]["priority_families_under_sampled"] == ["repository", "integration"]
    assert rationale["planner_pressure"]["priority_families_low_return"] == []
    assert next_policy["adaptive_search"] is True
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["priority_benchmark_families"] == ["repository", "integration", "project"]


def test_next_round_policy_keeps_trust_excluded_during_coding_first_bootstrap():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 1,
                "runtime_managed_decisions": 0,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": False, "failed_decisions": 1},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {"observed_decisions": 0, "observed_estimated_cost": 0.0, "retained_positive_delta_decisions": 0},
                    "repository": {"observed_decisions": 0, "observed_estimated_cost": 0.0, "retained_positive_delta_decisions": 0},
                    "integration": {"observed_decisions": 0, "observed_estimated_cost": 0.0, "retained_positive_delta_decisions": 0},
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    for subsystem in (
        "trust",
        "curriculum",
        "state_estimation",
        "transition_model",
        "recovery",
        "delegation",
        "operator_policy",
        "capabilities",
        "universe",
    ):
        assert subsystem in next_policy["excluded_subsystems"]
    assert "coding_first_bootstrap" in round_payload["policy_shift_rationale"]["reason_codes"]


def test_next_round_policy_keeps_support_lanes_excluded_until_first_retained_gain():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 2,
        "adaptive_search": True,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "discovered_task_adaptation",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 1,
                "runtime_managed_decisions": 3,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": False, "failed_decisions": 1},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {"observed_decisions": 2, "observed_estimated_cost": 4.0, "retained_positive_delta_decisions": 0},
                    "repository": {"observed_decisions": 2, "observed_estimated_cost": 4.0, "retained_positive_delta_decisions": 0},
                    "integration": {"observed_decisions": 2, "observed_estimated_cost": 4.0, "retained_positive_delta_decisions": 0},
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    for subsystem in (
        "trust",
        "curriculum",
        "state_estimation",
        "transition_model",
        "recovery",
        "delegation",
        "operator_policy",
        "capabilities",
        "universe",
    ):
        assert subsystem in next_policy["excluded_subsystems"]
    assert "coding_first_bootstrap" in round_payload["policy_shift_rationale"]["reason_codes"]


def test_next_round_policy_deprioritizes_costly_low_return_priority_families():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 9.0,
                        "retained_positive_delta_decisions": 0,
                        "retained_positive_pass_rate_delta_sum": 0.0,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 9.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.2,
                    },
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 9.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.1,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "priority_family_low_return" in rationale["reason_codes"]
    assert "retained_gain_conversion_gap" in rationale["reason_codes"]
    assert "priority_family_under_sampled" not in rationale["reason_codes"]
    assert rationale["planner_pressure"]["priority_families_needing_retained_gain_conversion"] == ["project"]
    assert rationale["planner_pressure"]["priority_families_low_return"] == ["project"]
    assert next_policy["priority_benchmark_families"] == ["project", "repository", "integration"]


def test_next_round_policy_prioritizes_retained_gain_conversion_before_productive_family_replay():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 2,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 2,
                        "rejected_decisions": 0,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.2,
                    },
                    "repository": {
                        "observed_decisions": 4,
                        "rejected_decisions": 4,
                        "observed_estimated_cost": 12.0,
                        "retained_positive_delta_decisions": 0,
                    },
                    "integration": {
                        "observed_decisions": 0,
                        "rejected_decisions": 0,
                        "observed_estimated_cost": 0.0,
                        "retained_positive_delta_decisions": 0,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_families_needing_retained_gain_conversion"] == ["repository"]
    assert next_policy["priority_benchmark_families"][0] == "repository"


def test_next_round_policy_cools_retrieval_when_broad_observe_replays_into_retrieval_without_gain():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload = {
        "policy": {"priority_benchmark_families": ["project", "repository", "integration"]},
        "active_child": {
            "observe_completed": True,
            "observe_summary": {"passed": 3, "total": 3},
            "selected_subsystem": "retrieval",
            "families_sampled": ["project", "repository", "integration"],
            "semantic_progress_state": {
                "phase_family": "preview",
                "progress_class": "degraded",
            },
        },
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 1,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 1,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.1,
                    },
                    "repository": {
                        "observed_decisions": 2,
                        "rejected_decisions": 2,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 0,
                    },
                    "integration": {
                        "observed_decisions": 0,
                        "rejected_decisions": 0,
                        "observed_estimated_cost": 0.0,
                        "retained_positive_delta_decisions": 0,
                    },
                },
            },
            "decision_conversion_summary": {
                "partial_productive_without_decision_runs": 1,
            },
            "partial_progress_summary": {
                "observed_primary_runs": 1,
                "generated_success_completed_runs": 1,
            },
            "incomplete_cycle_summary": {"count": 1},
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "broad_observe_then_retrieval_first" in rationale["reason_codes"]
    assert "semantic_progress_drift" in rationale["reason_codes"]
    assert "observability_gap" in rationale["reason_codes"]
    assert "subsystem_robustness_gap" in rationale["reason_codes"]
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["cycles"] == 3
    assert next_policy["campaign_width"] == 4
    assert next_policy["variant_width"] == 3
    assert next_policy["task_limit"] == 256
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert "retrieval" in next_policy["excluded_subsystems"]


def test_next_round_policy_escalates_retrieval_redirect_when_broad_observe_still_has_zero_decisions():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload = {
        "active_child": {
            "selected_subsystem": "retrieval",
            "active_cycle_progress": {
                "observe_completed": True,
                "sampled_families_from_progress": ["project", "repository", "integration"],
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 0,
            },
        },
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 1,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 1,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.1,
                    },
                    "repository": {
                        "observed_decisions": 2,
                        "rejected_decisions": 2,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 0,
                    },
                    "integration": {
                        "observed_decisions": 1,
                        "rejected_decisions": 1,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 0,
                    },
                },
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 0,
            },
            "partial_progress_summary": {
                "observed_primary_runs": 1,
                "generated_success_completed_runs": 1,
            },
            "incomplete_cycle_summary": {"count": 0},
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "broad_observe_then_retrieval_first" in rationale["reason_codes"]
    assert "retained_gain_conversion_gap" in rationale["reason_codes"]
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["cycles"] == 3
    assert next_policy["campaign_width"] == 4
    assert next_policy["variant_width"] == 3
    assert next_policy["task_limit"] == 256
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 3
    assert "retrieval" in next_policy["excluded_subsystems"]


def test_next_round_policy_cools_selected_subsystem_when_micro_step_verification_loop_detected():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 64,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload = {
        "active_child": {
            "selected_subsystem": "retrieval",
            "current_task_progress_timeline": [
                {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                {"stage": "context_compile", "step_index": 1},
                {"stage": "world_model_updated", "step_index": 1},
                {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                {"stage": "context_compile", "step_index": 1},
                {"stage": "world_model_updated", "step_index": 1},
                {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                {"stage": "verification_result", "step_index": 1, "verification_passed": False},
                {"stage": "memory_retrieved", "step_index": 1, "step_subphase": "graph_memory"},
                {"stage": "state_estimated", "step_index": 1, "step_subphase": "world_model_initial"},
                {"stage": "context_compile", "step_index": 1},
                {"stage": "world_model_updated", "step_index": 1},
                {"stage": "memory_update_written", "step_index": 1, "verification_passed": False},
                {"stage": "verification_result", "step_index": 1, "verification_passed": False},
            ],
        },
    }

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 1,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {},
            },
            "decision_conversion_summary": {},
            "partial_progress_summary": {},
            "incomplete_cycle_summary": {"count": 0},
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "micro_step_verification_loop" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["micro_step_verification_loop"] is True
    assert next_policy["variant_width"] >= 2
    assert next_policy["subsystem_cooldowns"]["retrieval"] == 2
    assert "retrieval" in next_policy["excluded_subsystems"]


def test_next_round_policy_ranks_priority_families_by_return_on_cost():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.07,
                "average_retained_step_delta": -0.1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 3,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.12,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.18,
                    },
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 5.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.15,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_families_ranked_by_selection_score"] == [
        "repository",
        "integration",
        "project",
    ]
    assert (
        rationale["planner_pressure"]["priority_family_history"]["priority_family_selection_scores"]["repository"]
        > rationale["planner_pressure"]["priority_family_history"]["priority_family_selection_scores"]["integration"]
        > rationale["planner_pressure"]["priority_family_history"]["priority_family_selection_scores"]["project"]
    )
    assert next_policy["priority_benchmark_families"] == ["repository", "integration", "project"]


def test_next_round_policy_applies_priority_family_scoring_controls():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.07,
                "average_retained_step_delta": -0.1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 3,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.12,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.18,
                    },
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 5.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.15,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={
            "priority_family_retained_gain_multiplier": {"project": 3.5},
            "priority_family_score_bias": {"project": 0.02},
            "priority_family_exploration_bonus": 0.08,
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_families_ranked_by_selection_score"] == [
        "project",
        "repository",
        "integration",
    ]
    project_summary = rationale["planner_pressure"]["priority_family_history"]["family_summaries"]["project"]
    assert project_summary["selection_gain_multiplier"] == 3.5
    assert project_summary["selection_score_bias"] == 0.02
    assert next_policy["priority_benchmark_families"] == ["project", "repository", "integration"]


def test_next_round_policy_prioritizes_curriculum_frontier_repo_setting_pressure():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 3,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.16,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.14,
                    },
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.08,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        curriculum_controls={
            "frontier_failure_motif_bonus": 2,
            "frontier_repo_setting_bonus": 3,
            "frontier_failure_motif_priority_pairs": ["workflow:no_state_progress"],
            "frontier_repo_setting_priority_pairs": [
                "integration:worker_handoff",
                "integration:validation_lane",
            ],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    integration_summary = rationale["planner_pressure"]["priority_family_history"]["family_summaries"]["integration"]
    assert rationale["planner_pressure"]["frontier_repo_setting_families"] == ["integration"]
    assert rationale["planner_pressure"]["frontier_repo_setting_priority_pairs"] == [
        "integration:worker_handoff",
        "integration:validation_lane",
    ]
    assert integration_summary["selection_frontier_repo_setting_bonus"] > 0.0
    assert rationale["planner_pressure"]["priority_families_ranked_by_selection_score"][0] == "integration"
    assert next_policy["priority_benchmark_families"][0] == "integration"


def test_next_round_policy_uses_worker_handoff_repo_setting_pressure_to_widen_campaign_width():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["integration"],
                "required_families_with_reports": ["integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["integration"],
                "family_summaries": {
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.09,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        controller_state=default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0),
        curriculum_controls={
            "frontier_repo_setting_bonus": 3,
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    pressure = rationale["planner_pressure"]["repo_setting_policy_pressure"]
    assert next_policy["campaign_width"] == 3
    assert next_policy["task_step_floor"] == 40
    assert next_policy["adaptive_search"] is True
    assert "frontier_repo_setting_campaign_width_pressure" in rationale["reason_codes"]
    assert "frontier_repo_setting_depth_pressure" not in rationale["reason_codes"]
    assert pressure["campaign_width_signals"] == ["worker_handoff"]
    assert pressure["task_step_floor_signals"] == []


def test_candidate_round_policies_includes_wider_repo_setting_candidates_for_worker_handoff():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }

    candidates = module._candidate_round_policies(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        curriculum_controls={
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=256,
    )

    assert all(candidate["adaptive_search"] is True for candidate in candidates)
    assert any(candidate["campaign_width"] == 3 for candidate in candidates)
    assert any(candidate["focus"] == "discovered_task_adaptation" for candidate in candidates)


def test_candidate_round_policies_uses_frontier_repo_setting_pressure_outside_current_priority_mix():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }

    candidates = module._candidate_round_policies(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository"],
                "required_families_with_reports": ["project", "repository"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 2,
            },
        },
        liftoff_payload=None,
        curriculum_controls={
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=256,
    )

    assert all(candidate["adaptive_search"] is True for candidate in candidates)
    assert any(candidate["campaign_width"] == 3 for candidate in candidates)
    assert any(candidate["cycles"] == 2 for candidate in candidates)


def test_repo_setting_candidate_score_adjustment_prefers_wider_worker_handoff_candidates():
    module = _load_script("run_unattended_campaign.py")

    narrow = module._repo_setting_candidate_score_adjustment(
        policy={
            "campaign_width": 1,
            "task_step_floor": 40,
            "adaptive_search": True,
            "focus": "balanced",
        },
        current_policy={
            "campaign_width": 1,
            "task_step_floor": 40,
        },
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
    )
    wide = module._repo_setting_candidate_score_adjustment(
        policy={
            "campaign_width": 3,
            "task_step_floor": 40,
            "adaptive_search": True,
            "focus": "discovered_task_adaptation",
        },
        current_policy={
            "campaign_width": 1,
            "task_step_floor": 40,
        },
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
    )

    assert wide["campaign_width_adjustment"] > narrow["campaign_width_adjustment"]
    assert wide["score_adjustment"] > narrow["score_adjustment"]
    assert wide["campaign_width_delta"] == 2


def test_learn_repo_setting_policy_priors_raise_worker_handoff_breadth_weight_from_retained_outcomes():
    module = _load_script("run_unattended_campaign.py")

    priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[
            {
                "policy": {
                    "campaign_width": 3,
                    "task_step_floor": 40,
                    "adaptive_search": True,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 2,
                        "rejected_cycles": 0,
                        "average_retained_pass_rate_delta": 0.12,
                        "average_retained_step_delta": 6.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 40,
                    "adaptive_search": False,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 0,
                        "rejected_cycles": 1,
                        "average_retained_pass_rate_delta": 0.0,
                        "average_retained_step_delta": 0.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
        ],
        repo_setting_policy_pressure={
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
    )

    assert priors["campaign_width_observations"] == 2
    assert priors["campaign_width_unit_weight"] > 0.3
    assert priors["adaptive_search_bonus"] > 0.25


def test_next_round_policy_uses_validation_lane_repo_setting_pressure_to_raise_task_step_floor():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["integration"],
                "required_families_with_reports": ["integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["integration"],
                "family_summaries": {
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.09,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        controller_state=default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0),
        curriculum_controls={
            "frontier_repo_setting_bonus": 3,
            "frontier_repo_setting_priority_pairs": ["integration:validation_lane"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    pressure = rationale["planner_pressure"]["repo_setting_policy_pressure"]
    assert next_policy["campaign_width"] == 1
    assert next_policy["task_step_floor"] == 60
    assert next_policy["adaptive_search"] is True
    assert "frontier_repo_setting_campaign_width_pressure" not in rationale["reason_codes"]
    assert "frontier_repo_setting_depth_pressure" in rationale["reason_codes"]
    assert pressure["campaign_width_signals"] == []
    assert pressure["task_step_floor_signals"] == ["validation_lane"]


def test_candidate_round_policies_includes_deeper_repo_setting_candidates_for_validation_lane():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }

    candidates = module._candidate_round_policies(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        curriculum_controls={
            "frontier_repo_setting_priority_pairs": ["integration:validation_lane"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=256,
    )

    task_step_floors = {candidate["task_step_floor"] for candidate in candidates}
    assert all(candidate["adaptive_search"] is True for candidate in candidates)
    assert 20 in task_step_floors
    assert 40 in task_step_floors
    assert 80 in task_step_floors


def test_repo_setting_candidate_score_adjustment_prefers_deeper_validation_lane_candidates():
    module = _load_script("run_unattended_campaign.py")

    shallow = module._repo_setting_candidate_score_adjustment(
        policy={
            "campaign_width": 1,
            "task_step_floor": 40,
            "adaptive_search": True,
            "focus": "balanced",
        },
        current_policy={
            "campaign_width": 1,
            "task_step_floor": 40,
        },
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:validation_lane"],
            "campaign_width_signals": [],
            "task_step_floor_signals": ["validation_lane"],
            "adaptive_search_signals": ["validation_lane"],
        },
    )
    deep = module._repo_setting_candidate_score_adjustment(
        policy={
            "campaign_width": 1,
            "task_step_floor": 80,
            "adaptive_search": True,
            "focus": "discovered_task_adaptation",
        },
        current_policy={
            "campaign_width": 1,
            "task_step_floor": 40,
        },
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:validation_lane"],
            "campaign_width_signals": [],
            "task_step_floor_signals": ["validation_lane"],
            "adaptive_search_signals": ["validation_lane"],
        },
    )

    assert deep["task_step_floor_adjustment"] > shallow["task_step_floor_adjustment"]
    assert deep["score_adjustment"] > shallow["score_adjustment"]
    assert deep["task_step_floor_delta"] == 40


def test_learn_repo_setting_policy_priors_raise_validation_depth_weight_from_retained_outcomes():
    module = _load_script("run_unattended_campaign.py")

    priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 160,
                    "adaptive_search": True,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 2,
                        "rejected_cycles": 0,
                        "average_retained_pass_rate_delta": 0.1,
                        "average_retained_step_delta": 10.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": [],
                            "task_step_floor_signals": ["validation_lane"],
                            "adaptive_search_signals": ["validation_lane"],
                        }
                    }
                },
            },
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 20,
                    "adaptive_search": False,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 0,
                        "rejected_cycles": 1,
                        "average_retained_pass_rate_delta": 0.0,
                        "average_retained_step_delta": -2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": [],
                            "task_step_floor_signals": ["validation_lane"],
                            "adaptive_search_signals": ["validation_lane"],
                        }
                    }
                },
            },
        ],
        repo_setting_policy_pressure={
            "campaign_width_signals": [],
            "task_step_floor_signals": ["validation_lane"],
            "adaptive_search_signals": ["validation_lane"],
        },
    )

    assert priors["task_step_floor_observations"] == 2
    assert priors["task_step_floor_unit_weight"] > 0.4
    assert priors["adaptive_search_bonus"] > 0.25


def test_next_round_policy_records_repo_setting_candidate_score_adjustment():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["integration"],
                "required_families_with_reports": ["integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["integration"],
                "family_summaries": {
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.09,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        controller_state=default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0),
        curriculum_controls={
            "frontier_repo_setting_bonus": 3,
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    planner = round_payload["controller_planner"]
    assert planner["selected_repo_setting_score_adjustment"] > 0.0
    assert planner["selected_repo_setting_score_rationale"]["campaign_width_signals"] == ["worker_handoff"]
    assert planner["candidates"][0]["repo_setting_score_adjustment"] > 0.0


def test_next_round_policy_promotes_frontier_repo_setting_family_outside_current_mix():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository"],
                "required_families_with_reports": ["project", "repository"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 2,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.12,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.14,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        controller_state=default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0),
        curriculum_controls={
            "frontier_repo_setting_bonus": 3,
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    pressure = rationale["planner_pressure"]["repo_setting_policy_pressure"]
    assert pressure["focused_families"] == ["integration"]
    assert pressure["campaign_width_signals"] == ["worker_handoff"]
    assert next_policy["cycles"] == 2
    assert next_policy["campaign_width"] == 3
    assert next_policy["priority_benchmark_families"][0] == "integration"
    assert next_policy["priority_benchmark_families"][1:] == ["repository", "project"]


def test_next_round_policy_expands_productive_repo_project_depth_into_integration_frontier():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 5,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.08,
                "average_retained_step_delta": 4.0,
                "productive_depth_retained_cycles": 1,
                "average_productive_depth_step_delta": 4.0,
                "long_horizon_retained_cycles": 1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository"],
                "required_families_with_reports": ["project", "repository"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 2,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.12,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.14,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={"priority_family_min_selection_score": 0.05},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=512,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "coding_frontier_expansion" in rationale["reason_codes"]
    assert rationale["planner_pressure"]["coding_frontier_expansion"] == {
        "triggered": True,
        "productive_priority_families": ["project", "repository"],
        "forced_priority_families": ["integration", "repository", "project"],
    }
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["adaptive_search"] is True
    assert next_policy["cycles"] == 2
    assert next_policy["task_limit"] == 10
    assert next_policy["campaign_width"] == 2
    assert next_policy["variant_width"] == 1
    assert next_policy["task_step_floor"] == 75
    assert next_policy["priority_benchmark_families"] == ["integration", "repository", "project"]


def test_next_round_policy_records_learned_repo_setting_priors():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 1,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 40,
        "priority_benchmark_families": ["integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.03,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["integration"],
                "required_families_with_reports": ["integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["integration"],
                "family_summaries": {
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.09,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        prior_rounds=[
            {
                "policy": {
                    "campaign_width": 3,
                    "task_step_floor": 40,
                    "adaptive_search": True,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 2,
                        "rejected_cycles": 0,
                        "average_retained_pass_rate_delta": 0.12,
                        "average_retained_step_delta": 6.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 40,
                    "adaptive_search": False,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 0,
                        "rejected_cycles": 1,
                        "average_retained_pass_rate_delta": 0.0,
                        "average_retained_step_delta": 0.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
        ],
        controller_state=default_controller_state(exploration_bonus=0.0, uncertainty_penalty=0.0),
        curriculum_controls={
            "frontier_repo_setting_bonus": 3,
            "frontier_repo_setting_priority_pairs": ["integration:worker_handoff"],
        },
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    planner = round_payload["controller_planner"]
    learned_priors = planner["selected_repo_setting_score_rationale"]["learned_priors"]
    assert planner["learned_repo_setting_priors"]["campaign_width_unit_weight"] > 0.3
    assert learned_priors["campaign_width_unit_weight"] > 0.3
    assert learned_priors["signal_observations"]["worker_handoff"] == 2


def test_update_repo_setting_policy_priors_persists_signal_state():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state()

    updated_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 3,
                "task_step_floor": 64,
                "adaptive_search": True,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 2,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.14,
                    "average_retained_step_delta": 8.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff", "integration:validation_lane"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": ["validation_lane"],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )

    normalized = normalize_controller_state(updated_state)
    priors = normalized["repo_setting_policy_priors"]
    assert priors["worker_handoff"]["observations"] == 1
    assert priors["worker_handoff"]["campaign_width_stats"]["observations"] == 1
    assert priors["worker_handoff"]["adaptive_search_stats"]["true_count"] == 1
    assert priors["worker_handoff"]["family_priors"]["integration"]["observations"] == 1
    assert priors["validation_lane"]["task_step_floor_stats"]["observations"] == 1
    assert priors["validation_lane"]["family_priors"]["integration"]["observations"] == 1


def test_learn_repo_setting_policy_priors_uses_persisted_controller_state():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state()
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 4,
                "task_step_floor": 32,
                "adaptive_search": True,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 2,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.12,
                    "average_retained_step_delta": 6.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 1,
                "task_step_floor": 32,
                "adaptive_search": False,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 0.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )

    priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[],
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
        persisted_priors=controller_state["repo_setting_policy_priors"],
    )

    assert priors["campaign_width_observations"] == 2
    assert priors["adaptive_search_observations"] == 2
    assert priors["persisted_signal_observations"]["worker_handoff"] == 2
    assert priors["persisted_family_signal_observations"]["integration:worker_handoff"] == 2
    assert priors["campaign_width_unit_weight"] > 0.3
    assert priors["adaptive_search_bonus"] > 0.25


def test_learn_repo_setting_policy_priors_specializes_by_family_before_shared_signal():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state()
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 1,
                "task_step_floor": 32,
                "adaptive_search": False,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": -1.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["workflow:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 4,
                "task_step_floor": 32,
                "adaptive_search": True,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 2,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.14,
                    "average_retained_step_delta": 7.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )

    priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[],
        repo_setting_policy_pressure={
            "focused_pairs": ["integration:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
        persisted_priors=controller_state["repo_setting_policy_priors"],
    )

    assert priors["persisted_signal_observations"]["worker_handoff"] == 2
    assert priors["persisted_family_signal_observations"]["integration:worker_handoff"] == 1
    assert priors["campaign_width_unit_weight"] > 0.3
    assert priors["adaptive_search_bonus"] > 0.25


def test_learn_repo_setting_policy_priors_borrows_neighbor_family_signal_from_persisted_state():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state()
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 4,
                "task_step_floor": 32,
                "adaptive_search": True,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 2,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.15,
                    "average_retained_step_delta": 7.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["tooling:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 1,
                "task_step_floor": 32,
                "adaptive_search": False,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": -2.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )

    workflow_priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[],
        repo_setting_policy_pressure={
            "focused_pairs": ["workflow:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
        persisted_priors=controller_state["repo_setting_policy_priors"],
    )
    assurance_priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[],
        repo_setting_policy_pressure={
            "focused_pairs": ["assurance:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
        persisted_priors=controller_state["repo_setting_policy_priors"],
    )

    assert workflow_priors["persisted_neighbor_signal_observations"]["workflow:worker_handoff"] > 0.0
    assert assurance_priors["persisted_neighbor_signal_observations"].get("assurance:worker_handoff", 0.0) == 0.0
    assert workflow_priors["campaign_width_neighbor_support"] > assurance_priors["campaign_width_neighbor_support"]
    assert workflow_priors["campaign_width_unit_weight"] > assurance_priors["campaign_width_unit_weight"]
    assert workflow_priors["adaptive_search_bonus"] >= assurance_priors["adaptive_search_bonus"]


def test_learn_repo_setting_policy_priors_borrows_neighbor_family_signal_from_prior_rounds():
    module = _load_script("run_unattended_campaign.py")

    workflow_priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[
            {
                "policy": {
                    "campaign_width": 4,
                    "task_step_floor": 32,
                    "adaptive_search": True,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 2,
                        "rejected_cycles": 0,
                        "average_retained_pass_rate_delta": 0.13,
                        "average_retained_step_delta": 6.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "focused_pairs": ["tooling:worker_handoff"],
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            }
            ,
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 32,
                    "adaptive_search": False,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 0,
                        "rejected_cycles": 1,
                        "average_retained_pass_rate_delta": 0.0,
                        "average_retained_step_delta": -2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "focused_pairs": ["integration:worker_handoff"],
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
        ],
        repo_setting_policy_pressure={
            "focused_pairs": ["workflow:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
    )
    assurance_priors = module._learn_repo_setting_policy_priors(
        prior_rounds=[
            {
                "policy": {
                    "campaign_width": 4,
                    "task_step_floor": 32,
                    "adaptive_search": True,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 2,
                        "rejected_cycles": 0,
                        "average_retained_pass_rate_delta": 0.13,
                        "average_retained_step_delta": 6.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "focused_pairs": ["tooling:worker_handoff"],
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            }
            ,
            {
                "policy": {
                    "campaign_width": 1,
                    "task_step_floor": 32,
                    "adaptive_search": False,
                },
                "campaign_report": {
                    "production_yield_summary": {
                        "retained_cycles": 0,
                        "rejected_cycles": 1,
                        "average_retained_pass_rate_delta": 0.0,
                        "average_retained_step_delta": -2.0,
                    },
                    "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
                },
                "policy_shift_rationale": {
                    "planner_pressure": {
                        "repo_setting_policy_pressure": {
                            "focused_pairs": ["integration:worker_handoff"],
                            "campaign_width_signals": ["worker_handoff"],
                            "task_step_floor_signals": [],
                            "adaptive_search_signals": ["worker_handoff"],
                        }
                    }
                },
            },
        ],
        repo_setting_policy_pressure={
            "focused_pairs": ["assurance:worker_handoff"],
            "campaign_width_signals": ["worker_handoff"],
            "task_step_floor_signals": [],
            "adaptive_search_signals": ["worker_handoff"],
        },
    )

    assert workflow_priors["neighbor_signal_observations"]["workflow:worker_handoff"] > 0.0
    assert assurance_priors["neighbor_signal_observations"].get("assurance:worker_handoff", 0.0) == 0.0
    assert workflow_priors["campaign_width_neighbor_support"] > assurance_priors["campaign_width_neighbor_support"]
    assert workflow_priors["campaign_width_unit_weight"] > assurance_priors["campaign_width_unit_weight"]


def test_update_repo_setting_policy_priors_decays_stale_signal_memory():
    module = _load_script("run_unattended_campaign.py")
    controller_state = default_controller_state()
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 4,
                "task_step_floor": 32,
                "adaptive_search": True,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 2,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.15,
                    "average_retained_step_delta": 8.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )
    first_sum_w = controller_state["repo_setting_policy_priors"]["worker_handoff"]["campaign_width_stats"]["sum_w"]
    controller_state = module._update_repo_setting_policy_priors(
        controller_state,
        round_payload={
            "policy": {
                "campaign_width": 1,
                "task_step_floor": 32,
                "adaptive_search": False,
            },
            "campaign_report": {
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": -2.0,
                },
                "phase_gate_summary": {
                    "all_retained_phase_gates_passed": True,
                    "failed_decisions": 0,
                },
            },
            "policy_shift_rationale": {
                "planner_pressure": {
                    "repo_setting_policy_pressure": {
                        "focused_pairs": ["integration:worker_handoff"],
                        "campaign_width_signals": ["worker_handoff"],
                        "task_step_floor_signals": [],
                        "adaptive_search_signals": ["worker_handoff"],
                    }
                }
            },
        },
    )
    updated = normalize_controller_state(controller_state)
    stats = updated["repo_setting_policy_priors"]["worker_handoff"]["campaign_width_stats"]
    assert stats["observations"] == 2
    assert stats["sum_w"] < (first_sum_w + 1.0)


def test_next_round_policy_filters_priority_families_below_min_selection_score():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 2,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.07,
                "average_retained_step_delta": -0.1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": ["project", "repository", "integration"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 3,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 6.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.12,
                    },
                    "repository": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.18,
                    },
                    "integration": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 20.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.02,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={"priority_family_min_selection_score": 0.015},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_family_history"]["priority_family_scoring_policy"][
        "priority_family_min_selection_score"
    ] == 0.015
    assert rationale["planner_pressure"]["priority_family_budget_warning"] == {
        "kind": "selection_score_floor",
        "detail": "priority family budget stayed narrow because the selection floor filtered lower-score families",
        "selected_count": 2,
        "selection_cap": 3,
        "filtered_families": ["integration"],
        "min_selection_score": 0.015,
    }
    assert "priority_family_selection_floor_narrowing" in rationale["reason_codes"]
    assert next_policy["priority_benchmark_families"] == ["repository", "project"]


def test_next_round_policy_keeps_missing_required_families_above_min_selection_score():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["repo_chore"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.2,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository", "integration"],
                "required_families_with_reports": [],
                "missing_required_families": ["project", "repository", "integration"],
                "distinct_family_gap": 3,
                "external_report_count": 1,
                "distinct_external_benchmark_families": 1,
            },
            "priority_family_yield_summary": {
                "priority_families": ["repo_chore"],
                "family_summaries": {
                    "repo_chore": {
                        "observed_decisions": 3,
                        "observed_estimated_cost": 30.0,
                        "retained_positive_delta_decisions": 0,
                        "retained_positive_pass_rate_delta_sum": 0.0,
                    }
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={"priority_family_min_selection_score": 0.2},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_family_budget_warning"] == {}
    assert set(next_policy["priority_benchmark_families"]) == {"project", "repository", "integration"}


def test_next_round_policy_ignores_repo_sandbox_only_gap_for_minimum_fresh_proof():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore", "repo_sandbox"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": ["repo_sandbox"],
                "distinct_family_gap": 1,
                "external_report_count": 4,
                "distinct_external_benchmark_families": 4,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository", "integration"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.08,
                    },
                    "repository": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.07,
                    },
                    "integration": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.06,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={"priority_family_min_selection_score": 0.0},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["missing_required_families"] == []
    assert rationale["planner_pressure"]["distinct_family_gap"] == 0
    assert "required_family_coverage_gap" not in rationale["reason_codes"]
    assert "repo_sandbox" not in next_policy["priority_benchmark_families"]


def test_next_round_policy_reports_limited_priority_family_availability():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.05,
                "average_retained_step_delta": -0.1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            "trust_breadth_summary": {
                "required_families": ["project", "repository"],
                "required_families_with_reports": ["project", "repository"],
                "missing_required_families": [],
                "distinct_family_gap": 0,
                "external_report_count": 2,
                "distinct_external_benchmark_families": 2,
            },
            "priority_family_yield_summary": {
                "priority_families": ["project", "repository"],
                "family_summaries": {
                    "project": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 4.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.08,
                    },
                    "repository": {
                        "observed_decisions": 2,
                        "observed_estimated_cost": 3.0,
                        "retained_positive_delta_decisions": 1,
                        "retained_positive_pass_rate_delta_sum": 0.06,
                    },
                },
            },
        },
        liftoff_payload=None,
        round_payload=round_payload,
        planner_controls={"priority_family_min_selection_score": 0.0},
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert rationale["planner_pressure"]["priority_family_budget_warning"] == {
        "kind": "limited_priority_families",
        "detail": "priority family budget stayed narrow because fewer than three ranked families were available",
        "selected_count": 2,
        "selection_cap": 3,
        "available_families": ["project", "repository"],
    }
    assert "priority_family_limited_availability" in rationale["reason_codes"]
    assert next_policy["priority_benchmark_families"] == ["project", "repository"]


def test_next_round_policy_raises_task_step_floor_for_productive_depth():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.08,
                "average_retained_step_delta": 4.0,
                "productive_depth_retained_cycles": 1,
                "average_productive_depth_step_delta": 4.0,
                "long_horizon_retained_cycles": 1,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=512,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "productive_depth_gain" in rationale["reason_codes"]
    assert rationale["task_step_floor_before"] == 50
    assert rationale["task_step_floor_after"] == 100
    assert next_policy["task_step_floor"] == 100


def test_next_round_policy_tightens_task_step_floor_on_depth_drift():
    module = _load_script("run_unattended_campaign.py")
    current_policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 80,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    round_payload: dict[str, object] = {}

    next_policy = module._next_round_policy(
        current_policy,
        campaign_report={
            "production_yield_summary": {
                "retained_cycles": 1,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 5.0,
                "depth_drift_cycles": 1,
                "average_depth_drift_step_delta": 5.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        },
        liftoff_payload=None,
        round_payload=round_payload,
        max_cycles=3,
        max_task_limit=1024,
        max_campaign_width=4,
        max_variant_width=3,
        max_task_step_floor=512,
    )

    rationale = round_payload["policy_shift_rationale"]
    assert "depth_drift_pressure" in rationale["reason_codes"]
    assert rationale["task_step_floor_before"] == 80
    assert rationale["task_step_floor_after"] == 40
    assert next_policy["task_step_floor"] == 40


def test_stop_counters_treat_productive_depth_continuation_as_non_stall():
    module = _load_script("run_unattended_campaign.py")
    policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    campaign_report = {
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.08,
            "average_retained_step_delta": 4.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 4.0,
            "long_horizon_retained_cycles": 1,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }

    no_yield_rounds = module._next_no_yield_rounds(1, campaign_report=campaign_report)
    policy_stall_rounds = module._next_policy_stall_rounds(
        1,
        current_policy=policy,
        next_policy=dict(policy),
        campaign_report=campaign_report,
    )

    assert no_yield_rounds == 0
    assert policy_stall_rounds == 0


def test_stop_counters_do_not_treat_non_retained_task_success_as_yield():
    module = _load_script("run_unattended_campaign.py")
    campaign_report = {
        "successful_runs": 1,
        "completed_runs": 1,
        "inheritance_summary": {
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 1,
        },
        "decision_conversion_summary": {
            "runtime_managed_runs": 0,
            "non_runtime_managed_runs": 1,
            "partial_productive_without_decision_runs": 0,
            "decision_runs": 1,
        },
        "execution_source_summary": {
            "decoder_generated": 1,
            "llm_generated": 1,
            "bounded_decoder_generated": 0,
            "synthetic_plan": 0,
            "deterministic_or_other": 0,
            "total_executed_commands": 1,
        },
        "production_yield_summary": {
            "retained_cycles": 0,
            "rejected_cycles": 1,
            "average_retained_pass_rate_delta": 0.0,
            "average_retained_step_delta": 0.0,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        "recent_non_runtime_decisions": [
            {"cycle_id": "cycle:retrieval:1", "state": "reject", "subsystem": "retrieval"},
        ],
    }

    signal = module._campaign_signal(campaign_report)
    stop_signal = module._round_stop_signal(campaign_report=campaign_report)
    no_yield_rounds = module._next_no_yield_rounds(1, campaign_report=campaign_report)

    assert signal["task_success_evidence"] is True
    assert stop_signal["yield_resets_no_yield"] is False
    assert no_yield_rounds == 2


def test_stop_counters_keep_same_policy_depth_drift_as_stall():
    module = _load_script("run_unattended_campaign.py")
    policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 80,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    campaign_report = {
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.0,
            "average_retained_step_delta": 5.0,
            "depth_drift_cycles": 1,
            "average_depth_drift_step_delta": 5.0,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }

    policy_stall_rounds = module._next_policy_stall_rounds(
        0,
        current_policy=policy,
        next_policy=dict(policy),
        campaign_report=campaign_report,
    )

    assert policy_stall_rounds == 1


def test_stop_counters_do_not_treat_failed_depth_as_nonstall():
    module = _load_script("run_unattended_campaign.py")
    policy = {
        "cycles": 1,
        "campaign_width": 2,
        "variant_width": 1,
        "adaptive_search": False,
        "task_limit": 128,
        "task_step_floor": 80,
        "priority_benchmark_families": ["project", "repository"],
        "tolbert_device": "cuda:2",
        "focus": "balanced",
        "liftoff": "auto",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    campaign_report = {
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.04,
            "average_retained_step_delta": 4.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 4.0,
            "long_horizon_retained_cycles": 1,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 1},
    }

    policy_stall_rounds = module._next_policy_stall_rounds(
        0,
        current_policy=policy,
        next_policy=dict(policy),
        campaign_report=campaign_report,
    )

    assert policy_stall_rounds == 1


def test_depth_runway_credit_accumulates_decays_and_resets():
    module = _load_script("run_unattended_campaign.py")
    productive_report = {
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.08,
            "average_retained_step_delta": 4.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 4.0,
            "long_horizon_retained_cycles": 1,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }
    neutral_report = {
        "production_yield_summary": {
            "retained_cycles": 0,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.0,
            "average_retained_step_delta": 0.0,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }
    drift_report = {
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.0,
            "average_retained_step_delta": 5.0,
            "depth_drift_cycles": 1,
            "average_depth_drift_step_delta": 5.0,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }

    credit = module._next_depth_runway_credit(0, campaign_report=productive_report)
    assert credit == 2

    budget = module._adaptive_stop_budget(
        base_max_no_yield_rounds=0,
        base_max_policy_stall_rounds=0,
        depth_runway_credit=credit,
    )
    assert budget["max_no_yield_rounds_effective"] == 2
    assert budget["max_policy_stall_rounds_effective"] == 2

    credit = module._next_depth_runway_credit(credit, campaign_report=neutral_report)
    assert credit == 1

    credit = module._next_depth_runway_credit(credit, campaign_report=drift_report)
    assert credit == 0


def test_run_unattended_campaign_same_policy_productive_depth_does_not_safe_stop(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    report_payload = {
        "report_kind": "improvement_campaign_report",
        "production_yield_summary": {
            "retained_cycles": 1,
            "rejected_cycles": 0,
            "average_retained_pass_rate_delta": 0.08,
            "average_retained_step_delta": 4.0,
            "productive_depth_retained_cycles": 1,
            "average_productive_depth_step_delta": 4.0,
            "long_horizon_retained_cycles": 1,
        },
        "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
    }
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(json.dumps(report_payload), encoding="utf-8")
    round_two_report.write_text(json.dumps(report_payload), encoding="utf-8")

    seen_calls = {"value": 0}

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        seen_calls["value"] += 1
        report_path = round_one_report if seen_calls["value"] == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    def same_policy(current_policy, **kwargs):
        del kwargs
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", same_policy)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--liftoff",
            "never",
            "--max-policy-stall-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["rounds"][0]["policy_stall_rounds"] == 0
    assert payload["rounds"][1]["policy_stall_rounds"] == 0


def test_run_unattended_campaign_productive_depth_credit_extends_same_policy_runway(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.08,
                    "average_retained_step_delta": 4.0,
                    "productive_depth_retained_cycles": 1,
                    "average_productive_depth_step_delta": 4.0,
                    "long_horizon_retained_cycles": 1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 0.0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_calls = {"value": 0}

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        seen_calls["value"] += 1
        report_path = round_one_report if seen_calls["value"] == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    def same_policy(current_policy, **kwargs):
        del kwargs
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", same_policy)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--liftoff",
            "never",
            "--max-no-yield-rounds",
            "0",
            "--max-policy-stall-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["rounds"][1]["no_yield_rounds"] == 1
    assert payload["rounds"][1]["policy_stall_rounds"] == 1
    assert payload["rounds"][1]["adaptive_stop_budget"]["depth_runway_credit"] == 1
    assert payload["rounds"][1]["adaptive_stop_budget"]["max_no_yield_rounds_effective"] == 1
    assert payload["rounds"][1]["adaptive_stop_budget"]["max_policy_stall_rounds_effective"] == 1


def test_run_unattended_campaign_same_policy_depth_drift_safe_stops(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    campaign_report_path = reports_dir / "campaign_round_1.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 5.0,
                    "depth_drift_cycles": 1,
                    "average_depth_drift_step_delta": 5.0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    def same_policy(current_policy, **kwargs):
        del kwargs
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", same_policy)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--liftoff",
            "never",
            "--max-policy-stall-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["reason"] == "unattended policy stalled without discovering a new outer-loop action"
    assert payload["rounds"][0]["policy_stall_rounds"] == 1


def test_run_unattended_campaign_depth_drift_clears_prior_depth_runway(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.08,
                    "average_retained_step_delta": 4.0,
                    "productive_depth_retained_cycles": 1,
                    "average_productive_depth_step_delta": 4.0,
                    "long_horizon_retained_cycles": 1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 5.0,
                    "depth_drift_cycles": 1,
                    "average_depth_drift_step_delta": 5.0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_calls = {"value": 0}

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        seen_calls["value"] += 1
        report_path = round_one_report if seen_calls["value"] == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    def same_policy(current_policy, **kwargs):
        del kwargs
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", same_policy)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--liftoff",
            "never",
            "--max-policy-stall-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["rounds"][1]["policy_stall_rounds"] == 1
    assert payload["rounds"][1]["adaptive_stop_budget"]["depth_runway_credit"] == 0
    assert payload["rounds"][1]["adaptive_stop_budget"]["max_policy_stall_rounds_effective"] == 0


def test_run_unattended_campaign_depth_runway_credit_exhausts_after_neutral_rounds(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    round_reports = [
        reports_dir / "campaign_round_1.json",
        reports_dir / "campaign_round_2.json",
        reports_dir / "campaign_round_3.json",
    ]
    round_reports[0].parent.mkdir(parents=True, exist_ok=True)
    round_reports[0].write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.08,
                    "average_retained_step_delta": 4.0,
                    "productive_depth_retained_cycles": 1,
                    "average_productive_depth_step_delta": 4.0,
                    "long_horizon_retained_cycles": 1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    neutral_payload = json.dumps(
        {
            "report_kind": "improvement_campaign_report",
            "production_yield_summary": {
                "retained_cycles": 0,
                "rejected_cycles": 0,
                "average_retained_pass_rate_delta": 0.0,
                "average_retained_step_delta": 0.0,
            },
            "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
        }
    )
    round_reports[1].write_text(neutral_payload, encoding="utf-8")
    round_reports[2].write_text(neutral_payload, encoding="utf-8")

    seen_calls = {"value": 0}

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        report_path = round_reports[seen_calls["value"]]
        seen_calls["value"] += 1
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    def same_policy(current_policy, **kwargs):
        del kwargs
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", same_policy)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "3",
            "--liftoff",
            "never",
            "--max-no-yield-rounds",
            "0",
            "--max-policy-stall-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "safe_stop"
    assert payload["rounds"][1]["adaptive_stop_budget"]["depth_runway_credit"] == 1
    assert payload["rounds"][2]["adaptive_stop_budget"]["depth_runway_credit"] == 0
    assert payload["rounds"][2]["no_yield_rounds"] == 2
    assert payload["rounds"][2]["policy_stall_rounds"] == 2


def test_write_status_preserves_mirrored_child_frontier_fields(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    status_path.write_text(
        json.dumps(
            {
                "active_run": {"run_index": 1, "child_status": {"state": "running"}},
                "families_sampled": ["integration"],
                "families_never_sampled": ["project"],
                "pressure_families_without_sampling": ["project"],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "campaign.json"
    payload = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {"priority_benchmark_families": ["repository", "project"]},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {},
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["active_run"] == {"run_index": 1, "child_status": {"state": "running"}}
    assert observed["families_sampled"] == ["integration"]
    assert observed["families_never_sampled"] == ["project"]
    assert observed["pressure_families_without_sampling"] == ["project"]
    assert observed["requested_priority_benchmark_families"] == ["repository", "project"]


def test_write_status_mirrors_active_run_policy_top_level(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    report_path = tmp_path / "campaign.json"
    active_policy = {
        "cycles": 2,
        "campaign_width": 3,
        "variant_width": 2,
        "adaptive_search": True,
        "task_limit": 4,
        "task_step_floor": 50,
        "priority_benchmark_families": ["project", "repository", "integration"],
        "tolbert_device": "cuda",
        "focus": "recovery_alignment",
        "liftoff": "never",
        "excluded_subsystems": [],
        "subsystem_cooldowns": {},
    }
    payload = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": dict(active_policy),
        "active_run": {"run_index": 1, "policy": dict(active_policy)},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {},
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["active_run_policy"] == active_policy
    assert observed["active_run"]["policy"] == active_policy


def test_write_status_surfaces_runtime_claim(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    report_path = tmp_path / "campaign.json"
    config = KernelConfig(
        provider="vllm",
        asi_coding_require_live_llm=True,
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    payload = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {"priority_benchmark_families": ["project", "repository"]},
        "active_run": {"run_index": 1, "provider": "vllm"},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {},
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload, config=config)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["claimed_runtime_shape"] == config.claimed_runtime_shape()
    assert observed["runtime_claim"]["asi_coding_require_live_llm"] is True
    assert observed["decoder_runtime_posture"]["authority"] == "external_decoder"
    assert observed["live_llm_provider_capable"] is True
    assert observed["active_run"]["runtime_claim"]["asi_coding_require_live_llm"] is True


def test_write_status_projects_live_child_signal_top_level(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    report_path = tmp_path / "campaign.json"
    status_path.write_text(
        json.dumps(
            {
                "active_run": {
                    "run_index": 2,
                    "policy": {"priority_benchmark_families": ["project", "repository", "integration", "repo_chore"]},
                    "child_status": {
                        "round_index": 2,
                        "last_progress_phase": "primary",
                        "runtime_managed_decisions": 1,
                        "retained_gain_runs": 1,
                        "decision_conversion_summary": {
                            "runtime_managed_runs": 1,
                            "non_runtime_managed_runs": 2,
                            "partial_productive_without_decision_runs": 1,
                            "decision_runs": 3,
                        },
                        "families_sampled": ["project", "repository", "integration"],
                        "current_task": {
                            "index": 3,
                            "total": 8,
                            "task_id": "bridge_handoff_task",
                            "family": "integration",
                            "phase": "primary",
                        },
                        "current_cognitive_stage": {"cognitive_stage": "context_compile"},
                        "last_report_path": str(tmp_path / "child-report.json"),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {"priority_benchmark_families": ["project", "repository", "integration", "repo_chore"]},
        "active_run": {"run_index": 2},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {},
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {"overall_assessment": {"status": "bootstrap"}},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["campaign_round_index"] == 2
    assert observed["runtime_managed_decisions"] == 1
    assert observed["non_runtime_managed_decisions"] == 2
    assert observed["sampled_families_from_progress"] == ["project", "repository", "integration"]
    assert observed["families_never_sampled"] == ["repo_chore"]
    assert observed["pressure_families_without_sampling"] == ["repo_chore"]
    assert observed["trust_status"] == "bootstrap"
    assert observed["latest_generated_run_report_path"] == str(tmp_path / "child-report.json")
    assert observed["decision_stream_summary"]["runtime_managed"]["retained_cycles"] == 1
    assert observed["trust_breadth_summary"]["missing_required_families"] == ["repo_chore"]


def test_write_status_reprojects_active_child_and_autonomy_from_live_projection(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    report_path = tmp_path / "campaign.json"
    payload = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {"priority_benchmark_families": ["integration", "project", "repository", "repo_chore"]},
        "active_run": {
            "run_index": 1,
            "policy": {"priority_benchmark_families": ["integration", "project", "repository", "repo_chore"]},
        },
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {
            "current_task": {
                "index": 1,
                "total": 8,
                "task_id": "semantic_task_1",
                "family": "integration",
                "phase": "observe",
            },
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "healthy",
                "detail": "active task 1/8 is progressing",
            },
            "verification_outcome_summary": {
                "successful_task_count": 1,
                "successful_families": ["integration"],
            },
            "sampled_families_from_progress": ["integration"],
        },
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["current_task"]["phase"] == "primary"
    assert observed["active_child"]["current_task"]["phase"] == "primary"
    assert observed["active_run"]["child_status"]["current_task"]["phase"] == "primary"
    assert observed["current_autonomy_level"] == "A3"
    assert observed["autonomy_level_assessment"]["successful_task_count"] == 1
    assert observed["autonomy_level_assessment"]["successful_task_count"] == observed["verification_outcome_summary"]["successful_task_count"]


def test_write_status_projects_completed_terminal_progress_state(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    status_path = tmp_path / "campaign.status.json"
    report_path = tmp_path / "campaign.json"
    payload = {
        "status": "completed",
        "phase": "completed",
        "reason": "requested unattended rounds completed",
        "rounds_requested": 1,
        "rounds_completed": 1,
        "child_failure_recovery_budget": 1,
        "child_failure_recoveries_used": 0,
        "active_child": {
            "current_task": {"index": 8, "total": 8, "task_id": "semantic_task", "family": "project", "phase": "primary"},
            "last_progress_phase": "variant_generate",
            "canonical_progress_state": {
                "phase": "variant_generate",
                "phase_family": "preview",
                "status": "active",
                "progress_class": "healthy",
                "detail": "preview work is progressing",
            },
            "semantic_progress_state": {
                "phase": "variant_generate",
                "phase_family": "preview",
                "status": "active",
                "progress_class": "healthy",
                "detail": "preview work is progressing",
            },
        },
        "current_policy": {"priority_benchmark_families": ["project", "repository"]},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }

    module._write_status(status_path, report_path=report_path, payload=payload)

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["status"] == "completed"
    assert observed["phase"] == "completed"
    assert observed["last_progress_phase"] == "done"
    assert observed["semantic_progress_state"]["phase"] == "done"
    assert observed["semantic_progress_state"]["status"] == "complete"
    assert observed["semantic_progress_state"]["detail"] == "campaign completed"
    assert observed["canonical_progress_state"]["phase"] == "done"
    assert observed["canonical_progress_state"]["detail"] == "campaign completed"


def test_run_unattended_campaign_passes_parent_status_bridge_to_child(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    campaign_report_path = reports_dir / "campaign_round_1.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {"retained_cycles": 1, "rejected_cycles": 0},
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    seen_env: dict[str, str] = {}

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, kwargs
        seen_env.update(
            {
                "AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH": str(env.get("AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH", "")),
                "AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX": str(env.get("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX", "")),
                "AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID": str(env.get("AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID", "")),
                "AGENT_KERNEL_RUN_REPORTS_DIR": str(env.get("AGENT_KERNEL_RUN_REPORTS_DIR", "")),
                "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH": str(env.get("AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH", "")),
                "AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR": str(env.get("AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR", "")),
                "AGENT_KERNEL_RUNTIME_DATABASE_PATH": str(env.get("AGENT_KERNEL_RUNTIME_DATABASE_PATH", "")),
            }
        )
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "1",
            "--liftoff",
            "never",
            "--status-path",
            str(reports_dir / "campaign.status.json"),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert seen_env["AGENT_KERNEL_AUTONOMOUS_PARENT_STATUS_PATH"].endswith("campaign.status.json")
    assert seen_env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_INDEX"] == "1"
    assert seen_env["AGENT_KERNEL_AUTONOMOUS_PARENT_RUN_MATCH_ID"].startswith("unattended:round:")
    assert seen_env["AGENT_KERNEL_RUN_REPORTS_DIR"].endswith("reports")
    assert seen_env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"].endswith("reports/unattended_trust_ledger.json")
    assert seen_env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"].endswith("improvement/improvement_reports")
    assert seen_env["AGENT_KERNEL_RUNTIME_DATABASE_PATH"].endswith("improvement/improvement_reports/agentkernel.sqlite3")


def test_initial_round_policy_preserves_requested_priority_family_order():
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(frontier_task_step_floor=50)

    class Args:
        cycles = 1
        campaign_width = 2
        variant_width = 1
        adaptive_search = True
        task_limit = 10
        task_step_floor = 75
        priority_benchmark_family = ["integration", "repository", "project"]
        focus = "discovered_task_adaptation"
        liftoff = "never"

    policy = module._initial_round_policy(Args(), config)

    assert policy["priority_benchmark_families"] == ["integration", "repository", "project"]


def test_initial_round_policy_bootstraps_minimum_fresh_proof_priority_families():
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(frontier_task_step_floor=50)

    class Args:
        cycles = 1
        campaign_width = 2
        variant_width = 1
        adaptive_search = True
        task_limit = 10
        task_step_floor = 75
        priority_benchmark_family = []
        focus = "discovered_task_adaptation"
        liftoff = "never"

    policy = module._initial_round_policy(Args(), config)

    assert policy["priority_benchmark_families"] == ["integration", "project", "repository", "repo_chore"]


def test_run_unattended_campaign_no_yield_waits_for_full_round_depth_signal(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    campaign_report_path = reports_dir / "campaign_round_1.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 0,
                    "rejected_cycles": 0,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, env, kwargs
        return {"returncode": 0, "stdout": f"{campaign_report_path}\n"}

    def inject_depth_continuation(current_policy, *, round_payload, **kwargs):
        del kwargs
        round_payload["policy_shift_rationale"] = {
            "reason_codes": ["productive_depth_gain"],
            "task_step_floor_before": int(current_policy["task_step_floor"]),
            "task_step_floor_after": int(current_policy["task_step_floor"]),
            "planner_pressure": {
                "productive_depth_retained_cycles": 1,
                "average_productive_depth_step_delta": 4.0,
                "long_horizon_retained_cycles": 1,
            },
        }
        return dict(current_policy)

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_next_round_policy", inject_depth_continuation)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "1",
            "--liftoff",
            "never",
            "--max-no-yield-rounds",
            "0",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["rounds"][0]["no_yield_rounds"] == 0
    assert payload["rounds"][0]["outer_stop_signal"]["productive_depth_continuation"] is True


def test_run_unattended_campaign_threads_task_step_floor_to_child_env(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(
        improvement_reports_dir=reports_dir,
        candidate_artifacts_root=tmp_path / "improvement" / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
        frontier_task_step_floor=50,
    )
    round_one_report = reports_dir / "campaign_round_1.json"
    round_two_report = reports_dir / "campaign_round_2.json"
    round_one_report.parent.mkdir(parents=True, exist_ok=True)
    round_one_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.08,
                    "average_retained_step_delta": 4.0,
                    "productive_depth_retained_cycles": 1,
                    "average_productive_depth_step_delta": 4.0,
                    "long_horizon_retained_cycles": 1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )
    round_two_report.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "production_yield_summary": {
                    "retained_cycles": 1,
                    "rejected_cycles": 0,
                    "average_retained_pass_rate_delta": 0.1,
                },
                "phase_gate_summary": {"all_retained_phase_gates_passed": True, "failed_decisions": 0},
            }
        ),
        encoding="utf-8",
    )

    seen_env_floors: list[str] = []

    def fake_run_and_stream(cmd, *, cwd, env, **kwargs):
        del cmd, cwd, kwargs
        seen_env_floors.append(str(env.get("AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR", "")))
        report_path = round_one_report if len(seen_env_floors) == 1 else round_two_report
        return {"returncode": 0, "stdout": f"{report_path}\n"}

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(module, "_gpu_preflight", lambda device: {"device": device, "passed": True, "detail": "ok"})
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--rounds",
            "2",
            "--cycles",
            "1",
            "--liftoff",
            "never",
            "--task-limit",
            "32",
            "--task-step-floor",
            "50",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert seen_env_floors[:2] == ["50", "100"]


def test_child_runtime_extension_seconds_grants_finalize_grace_after_generated_success_completion():
    module = _load_script("run_unattended_campaign.py")

    assert (
        module._child_runtime_extension_seconds(
            last_progress_phase="generated_success",
            current_task={"index": 10, "total": 10, "task_id": "integration_failover_drill_task", "family": "integration"},
        )
        == 120.0
    )
    assert (
        module._child_runtime_extension_seconds(
            last_progress_phase="generated_success",
            current_task={"index": 1, "total": 10, "task_id": "project_release_cutover_task", "family": "project"},
        )
        == 0.0
    )


def test_child_runtime_extension_seconds_grants_generated_failure_seed_active_grace():
    module = _load_script("run_unattended_campaign.py")

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="generated_failure_seed",
        current_task={
            "phase": "generated_failure_seed",
            "index": 1,
            "total": 10,
            "task_id": "bridge_handoff_task",
            "family": "integration",
        },
    )

    assert grace_key == "generated_failure_seed_active"
    assert grace_seconds == 240.0


def test_child_runtime_extension_seconds_grants_preview_completion_grace():
    module = _load_script("run_unattended_campaign.py")

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="preview_candidate_complete",
        current_task={
            "phase": "generated_failure",
            "index": 8,
            "total": 8,
            "task_id": "repo_cleanup_review_task_file_recovery",
            "family": "repo_chore",
        },
    )

    assert grace_key == "preview_completion"
    assert grace_seconds == 120.0


def test_child_runtime_extension_seconds_grants_preview_active_grace_from_finalize_phase():
    module = _load_script("run_unattended_campaign.py")

    grace_key, grace_seconds = module._child_runtime_extension_plan(
        last_progress_phase="primary",
        active_finalize_phase="preview_candidate_eval",
        current_task={
            "phase": "primary",
            "index": 1,
            "total": 1,
            "task_id": "semantic_open_world_task",
            "family": "integration",
        },
    )

    assert grace_key == "preview_active"
    assert grace_seconds == 120.0


def test_runtime_grace_marker_requires_progress_and_allows_reentry():
    module = _load_script("run_unattended_campaign.py")

    assert module._runtime_grace_marker(grace_key="", progress_epoch=1) is None
    assert module._runtime_grace_marker(grace_key="generated_success_completion", progress_epoch=0) is None
    assert module._runtime_grace_marker(
        grace_key="generated_success_completion",
        progress_epoch=3,
    ) == ("generated_success_completion", 3)
    assert module._runtime_grace_marker(
        grace_key="generated_success_completion",
        progress_epoch=4,
    ) != module._runtime_grace_marker(
        grace_key="generated_success_completion",
        progress_epoch=3,
    )


def test_child_runtime_extension_plan_skips_variant_generate_after_productive_work():
    module = _load_script("run_unattended_campaign.py")

    assert module._child_runtime_extension_plan(
        last_progress_phase="variant_generate",
        current_task={"phase": "generated_failure", "index": 8, "total": 8},
    ) == ("", 0.0)


def test_should_block_post_productive_runtime_grace_after_generated_success_without_decision():
    module = _load_script("run_unattended_campaign.py")

    assert module._should_block_post_productive_runtime_grace(
        {
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
        },
        last_progress_phase="generated_failure",
        current_task={"phase": "generated_failure", "index": 8, "total": 8},
    ) is True


def test_should_not_block_post_productive_runtime_grace_after_decision_credit_exists():
    module = _load_script("run_unattended_campaign.py")

    assert module._should_block_post_productive_runtime_grace(
        {
            "partial_productive_runs": 1,
            "decision_records_considered": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
        },
        last_progress_phase="generated_failure",
        current_task={"phase": "generated_failure", "index": 8, "total": 8},
    ) is False


def test_holdout_progress_budget_summary_projects_remaining_runtime():
    module = _load_script("run_unattended_campaign.py")

    summary = module._holdout_progress_budget_summary(
        [
            {"timestamp": 100.0, "index": 1, "total": 462},
            {"timestamp": 150.0, "index": 21, "total": 462},
            {"timestamp": 200.0, "index": 41, "total": 462},
        ]
    )

    assert summary["phase"] == "holdout_eval"
    assert summary["observed_tasks"] == 40
    assert summary["remaining_tasks"] == 421
    assert summary["seconds_per_task"] > 0.0
    assert summary["projected_total_seconds"] > summary["observed_seconds"]
    assert summary["projected_completion_seconds"] > 200.0


def test_holdout_budget_fit_distinguishes_within_budget_vs_over_budget():
    module = _load_script("run_unattended_campaign.py")
    summary = {
        "projected_total_seconds": 130.0,
    }

    within = module._holdout_budget_fit(summary, started_at=0.0, max_runtime_seconds=100.0)
    over = module._holdout_budget_fit(summary, started_at=0.0, max_runtime_seconds=60.0)

    assert within["status"] == "within_budget"
    assert over["status"] == "over_budget"


def test_adaptive_child_progress_state_classifies_holdout_health():
    module = _load_script("run_unattended_campaign.py")
    progress_samples = [
        {"timestamp": 100.0, "index": 1, "total": 100},
        {"timestamp": 140.0, "index": 8, "total": 100},
        {"timestamp": 180.0, "index": 16, "total": 100},
    ]

    healthy = module._adaptive_child_progress_state(
        last_progress_phase="holdout_eval",
        active_finalize_phase="holdout_eval",
        last_progress_at=180.0,
        progress_samples=progress_samples,
        started_at=100.0,
        now=185.0,
        max_runtime_seconds=400.0,
        max_progress_stall_seconds=300.0,
    )
    degraded = module._adaptive_child_progress_state(
        last_progress_phase="holdout_eval",
        active_finalize_phase="holdout_eval",
        last_progress_at=180.0,
        progress_samples=progress_samples,
        started_at=100.0,
        now=185.0,
        max_runtime_seconds=40.0,
        max_progress_stall_seconds=300.0,
    )

    assert healthy["progress_class"] == "healthy"
    assert degraded["progress_class"] == "degraded"
    assert degraded["status"] == "over_budget"


def test_adaptive_child_progress_state_prioritizes_primary_task_over_stale_preview_finalize_phase():
    module = _load_script("run_unattended_campaign.py")

    state = module._adaptive_child_progress_state(
        last_progress_phase="preview_baseline_eval",
        active_finalize_phase="preview_baseline_eval",
        current_task={
            "index": 1,
            "total": 4,
            "task_id": "semantic_open_world_20260411T225301327576Z_round_1_integration",
            "family": "integration",
            "phase": "primary",
            "task_origin": "semantic_hub",
        },
        current_cognitive_stage={
            "cognitive_stage": "plan_candidates",
            "step_index": 1,
            "phase": "primary",
        },
        last_progress_at=100.0,
        progress_samples=[],
        started_at=100.0,
        now=120.0,
        max_runtime_seconds=400.0,
        max_progress_stall_seconds=60.0,
    )

    assert state["phase"] == "primary"
    assert state["phase_family"] == "active"
    assert state["decision_distance"] == "active"
    assert state["detail"] == "active task 1/4 is progressing"


def test_adaptive_child_progress_state_classifies_preview_semantics():
    module = _load_script("run_unattended_campaign.py")

    healthy = module._adaptive_child_progress_state(
        last_progress_phase="preview_candidate_eval",
        active_finalize_phase="",
        last_progress_at=100.0,
        progress_samples=[],
        started_at=100.0,
        now=120.0,
        max_runtime_seconds=400.0,
        max_progress_stall_seconds=60.0,
    )
    stuck = module._adaptive_child_progress_state(
        last_progress_phase="preview_candidate_eval",
        active_finalize_phase="",
        last_progress_at=100.0,
        progress_samples=[],
        started_at=100.0,
        now=170.0,
        max_runtime_seconds=400.0,
        max_progress_stall_seconds=60.0,
    )

    assert healthy["phase_family"] == "preview"
    assert healthy["progress_class"] == "healthy"
    assert healthy["decision_distance"] == "near"
    assert stuck["progress_class"] == "stuck"
    assert stuck["status"] == "stalled"


def test_adaptive_child_progress_state_preserves_holdout_context_during_generated_subphases():
    module = _load_script("run_unattended_campaign.py")
    progress_samples = [
        {"timestamp": 100.0, "index": 20, "total": 165},
        {"timestamp": 140.0, "index": 40, "total": 165},
        {"timestamp": 180.0, "index": 70, "total": 165},
    ]

    state = module._adaptive_child_progress_state(
        last_progress_phase="generated_success",
        active_finalize_phase="holdout_eval",
        last_progress_at=180.0,
        progress_samples=progress_samples,
        started_at=100.0,
        now=185.0,
        max_runtime_seconds=800.0,
        max_progress_stall_seconds=300.0,
    )

    assert state["phase"] == "holdout_eval"
    assert state["phase_family"] == "holdout"
    assert state["progress_class"] == "healthy"
    assert state["holdout_subphase"] == "generated_success"


def test_run_and_stream_extends_runtime_deadline_for_progressing_holdout(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_events: list[dict[str, object]] = []

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 1357
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def __init__(self, stdout, clock):
            self._stdout = stdout
            self._clock = clock

        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            if self._stdout._lines:
                return [(type("Key", (), {"fileobj": self._stdout})(), None)]
            self._clock["now"] = 82.0
            return []

        def close(self):
            return None

    lines = [
        "[cycle:test] finalize phase=holdout_eval subsystem=qwen_adapter\n",
        "[eval:test] phase=generated_success task 1/100 holdout_task family=repository\n",
        "[eval:test] phase=generated_success task 6/100 holdout_task family=repository\n",
        "[eval:test] phase=generated_success task 12/100 holdout_task family=repository\n",
    ]
    clock = {"now": 0.0}
    stdout_holder: dict[str, object] = {}

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_monotonic():
        return float(clock["now"])

    def fake_time():
        return float(clock["now"])

    real_readline = FakeStdout.readline

    def timed_readline(self):
        line = real_readline(self)
        if "phase=holdout_eval" in line:
            clock["now"] = 1.0
        elif "phase=generated_success task 1/100" in line:
            clock["now"] = 1.0
        elif "phase=generated_success task 6/100" in line:
            clock["now"] = 41.0
        elif "phase=generated_success task 12/100" in line:
            clock["now"] = 81.0
        return line

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector(stdout_holder["stdout"], clock))
    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(module.time, "time", fake_time)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: (_ for _ in ()).throw(AssertionError(process.pid)))
    monkeypatch.setattr(FakeStdout, "readline", timed_readline)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        max_runtime_seconds=60.0,
        max_silence_seconds=300.0,
        max_progress_stall_seconds=300.0,
        on_event=seen_events.append,
    )

    assert result["timed_out"] is False
    adaptive_events = [event for event in seen_events if event.get("event") == "adaptive_runtime_budget"]
    assert adaptive_events
    assert int(adaptive_events[-1]["runtime_deadline_seconds"]) == 120


def test_run_and_stream_extends_runtime_deadline_for_active_preview_candidate(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_events: list[dict[str, object]] = []

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)
            self._idle_polls = 0

        def poll(self):
            if self.stdout._lines:
                return None
            self._idle_polls += 1
            return None if self._idle_polls == 1 else 0

        def wait(self):
            return 0

    class FakeSelector:
        def __init__(self, stdout, clock):
            self._stdout = stdout
            self._clock = clock

        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            if self._stdout._lines:
                return [(type("Key", (), {"fileobj": self._stdout})(), None)]
            self._clock["now"] = 61.0
            return []

        def close(self):
            return None

    lines = [
        "[cycle:test] finalize phase=preview_candidate_eval subsystem=operating_envelope\n",
        "[eval:test] phase=primary task 1/1 semantic_open_world_task family=integration cognitive_stage=world_model_updated step=1\n",
    ]
    clock = {"now": 0.0}
    stdout_holder: dict[str, object] = {}

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_monotonic():
        return float(clock["now"])

    def fake_time():
        return float(clock["now"])

    real_readline = FakeStdout.readline

    def timed_readline(self):
        line = real_readline(self)
        if "phase=preview_candidate_eval" in line:
            clock["now"] = 1.0
        elif "world_model_updated step=1" in line:
            clock["now"] = 59.0
        return line

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector(stdout_holder["stdout"], clock))
    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(module.time, "time", fake_time)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: (_ for _ in ()).throw(AssertionError(process.pid)))
    monkeypatch.setattr(FakeStdout, "readline", timed_readline)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        max_runtime_seconds=60.0,
        max_silence_seconds=300.0,
        max_progress_stall_seconds=300.0,
        on_event=seen_events.append,
    )

    assert result["timed_out"] is False
    grace_events = [event for event in seen_events if event.get("event") == "runtime_grace"]
    assert grace_events
    assert grace_events[-1]["grace_key"] == "preview_active"
    assert int(grace_events[-1]["runtime_deadline_seconds"]) == 181


def test_run_and_stream_exits_immediately_after_eof_when_child_already_terminated(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_events: list[dict[str, object]] = []

    class FakeStdout:
        def fileno(self):
            return 0

        def readline(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.pid = 97531
            self.stdout = FakeStdout()

        def poll(self):
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def __init__(self, stdout):
            self._stdout = stdout
            self._registered = True

        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj
            self._registered = False

        def select(self, timeout=None):
            del timeout
            if self._registered:
                return [(type("Key", (), {"fileobj": self._stdout})(), None)]
            raise AssertionError("selector should not be polled again after EOF closeout")

        def get_map(self):
            return {0: self._stdout} if self._registered else {}

        def close(self):
            return None

    stdout_holder: dict[str, object] = {}

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess()
        stdout_holder["stdout"] = process.stdout
        return process

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector(stdout_holder["stdout"]))

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        max_runtime_seconds=60.0,
        max_silence_seconds=300.0,
        max_progress_stall_seconds=300.0,
        on_event=seen_events.append,
    )

    assert result == {"returncode": 0, "stdout": "", "timed_out": False}
    assert seen_events[-1]["event"] == "exit"
    assert seen_events[-1]["returncode"] == 0


def test_run_and_stream_clears_stale_primary_task_before_observe_hypothesis(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen_events: list[dict[str, object]] = []

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 8642
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def __init__(self, stdout, clock):
            self._stdout = stdout
            self._clock = clock

        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            if self._stdout._lines:
                return [(type("Key", (), {"fileobj": self._stdout})(), None)]
            self._clock["now"] = 10.0
            return []

        def close(self):
            return None

    lines = [
        "[eval:test] phase=primary task 1/4 semantic_open_world_task family=integration cognitive_stage=verification_result step=5 verification_passed=0\n",
        "[eval:test] phase=metrics_finalize start\n",
        "[cycle:test] observe complete passed=3/4 pass_rate=0.7500 generated_pass_rate=0.0000\n",
        "[cycle:test] phase=observe_hypothesis start provider=vllm\n",
    ]
    clock = {"now": 0.0}
    stdout_holder: dict[str, object] = {}

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_monotonic():
        return float(clock["now"])

    def fake_time():
        return float(clock["now"])

    real_readline = FakeStdout.readline

    def timed_readline(self):
        line = real_readline(self)
        if "verification_result" in line:
            clock["now"] = 1.0
        elif "metrics_finalize start" in line:
            clock["now"] = 2.0
        elif "observe complete passed=3/4" in line:
            clock["now"] = 3.0
        elif "observe_hypothesis start" in line:
            clock["now"] = 4.0
        return line

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector(stdout_holder["stdout"], clock))
    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(module.time, "time", fake_time)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: (_ for _ in ()).throw(AssertionError(process.pid)))
    monkeypatch.setattr(FakeStdout, "readline", timed_readline)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        max_runtime_seconds=60.0,
        max_silence_seconds=300.0,
        max_progress_stall_seconds=300.0,
        on_event=seen_events.append,
    )

    assert result["timed_out"] is False
    adaptive_events = [event for event in seen_events if event.get("event") == "adaptive_progress_state"]
    assert adaptive_events
    assert any(
        event.get("phase") == "primary" and event.get("progress_class") == "degraded"
        for event in adaptive_events
    )
    assert adaptive_events[-1]["phase"] == "observe_hypothesis"
    assert adaptive_events[-1]["phase_family"] == "observe"
    assert adaptive_events[-1]["progress_class"] == "complete"
    assert adaptive_events[-1]["detail"] == "observe phase completed and is ready to hand off"


def test_make_child_progress_callback_records_adaptive_progress_state(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {}
    round_payload: dict[str, object] = {}

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=tmp_path / "campaign.lock",
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
    )

    callback(
        {
            "event": "adaptive_progress_state",
            "timestamp": 10.0,
            "phase": "holdout_eval",
            "status": "within_budget",
            "progress_class": "healthy",
            "progress_silence_seconds": 5.0,
            "detail": "holdout progressing within projected budget",
            "budget_fit": {"status": "within_budget"},
            "holdout_budget": {"total_tasks": 462, "completed_tasks": 180},
        }
    )

    active_child = report["active_child"]
    assert active_child["adaptive_progress_state"]["progress_class"] == "healthy"
    assert active_child["adaptive_progress_state"]["budget_fit"]["status"] == "within_budget"
    assert active_child["adaptive_progress_state"]["holdout_budget"]["total_tasks"] == 462
    assert active_child["semantic_progress_state"]["phase"] == "holdout_eval"
    assert active_child["semantic_progress_state"]["progress_class"] == "healthy"


def test_make_child_progress_callback_persists_adaptive_recovery_immediately(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    lock_path = tmp_path / "campaign.lock"
    report = {
        "status": "running",
        "phase": "campaign",
        "rounds_requested": 1,
        "rounds_completed": 0,
        "child_failure_recovery_budget": 0,
        "child_failure_recoveries_used": 0,
        "current_policy": {},
        "active_run": {"run_index": 1},
        "policy_shift_summary": {},
        "campaign_validation": {},
        "liftoff_validation": {},
        "active_child": {
            "round_index": 1,
            "phase": "campaign",
            "state": "running",
            "last_progress_phase": "primary",
            "current_task": {
                "index": 1,
                "total": 4,
                "task_id": "semantic_open_world_task",
                "family": "integration",
                "phase": "primary",
            },
            "current_cognitive_stage": {
                "cognitive_stage": "verification_result",
                "step_index": 1,
                "verification_passed": False,
            },
            "current_task_verification_passed": False,
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "status": "active",
                "progress_class": "degraded",
                "decision_distance": "active",
                "detail": "active verification failed on step 1 and is still awaiting recovery",
                "recorded_at": 10.0,
            },
        },
        "preflight": {},
        "cleanup": {},
        "unattended_evidence": {},
        "global_storage": {},
    }
    round_payload: dict[str, object] = {}

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    module._persist_report_state(
        report_path,
        report,
        config=config,
        status_path=status_path,
        lock_path=lock_path,
    )

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=lock_path,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=60.0,
    )

    callback(
        {
            "event": "adaptive_progress_state",
            "timestamp": 20.0,
            "phase": "primary",
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "progress_silence_seconds": 0.0,
            "runtime_elapsed_seconds": 120.0,
            "detail": "active task 1/4 is progressing",
        }
    )

    observed = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed["semantic_progress_state"]["progress_class"] == "healthy"
    assert observed["semantic_progress_state"]["detail"] == "active task 1/4 is progressing"
    assert observed["active_child"]["semantic_progress_state"]["progress_class"] == "healthy"


def test_make_child_progress_callback_refreshes_active_run_child_status(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(improvement_reports_dir=tmp_path / "reports")
    report_path = config.improvement_reports_dir / "campaign.json"
    status_path = config.improvement_reports_dir / "campaign.status.json"
    event_log_path = config.improvement_reports_dir / "campaign.events.jsonl"
    report: dict[str, object] = {"active_run": {"run_index": 1, "phase": "campaign"}}
    round_payload: dict[str, object] = {}

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=tmp_path / "campaign.lock",
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
    )

    callback(
        {
            "event": "output",
            "pid": 123,
            "timestamp": 10.0,
            "line": (
                "[eval:test] phase=generated_failure_seed task 4/4 repo_notice_review_task_file_recovery "
                "family=repo_chore source_task=repo_notice_review_task"
            ),
        }
    )

    active_child = report["active_child"]
    mirrored_child = report["active_run"]["child_status"]
    assert mirrored_child["last_progress_line"] == active_child["last_progress_line"]
    assert mirrored_child["last_output_line"] == active_child["last_output_line"]
    assert mirrored_child["current_task"] == active_child["current_task"]


def test_adaptive_progress_state_signature_ignores_silence_only_changes():
    module = _load_script("run_unattended_campaign.py")

    base_state = {
        "phase": "holdout_eval",
        "status": "within_budget",
        "progress_class": "healthy",
        "progress_silence_seconds": 5.0,
        "detail": "holdout progressing within projected budget",
        "budget_fit": {"status": "within_budget", "projected_total_seconds": 1350.0},
        "holdout_budget": {"completed_tasks": 200, "total_tasks": 462},
    }
    changed_only_silence = dict(base_state)
    changed_only_silence["progress_silence_seconds"] = 25.0

    assert module._adaptive_progress_state_signature(base_state) == module._adaptive_progress_state_signature(
        changed_only_silence
    )

    changed_budget = dict(base_state)
    changed_budget["holdout_budget"] = {"completed_tasks": 210, "total_tasks": 462}

    assert module._adaptive_progress_state_signature(base_state) != module._adaptive_progress_state_signature(
        changed_budget
    )


def test_phase_from_progress_line_recognizes_observe_completion_and_campaign_selection():
    module = _load_script("run_unattended_campaign.py")

    assert (
        module._phase_from_progress_line(
            "[cycle:cycle-1] observe complete passed=0/32 pass_rate=0.0000 generated_pass_rate=0.0000"
        )
        == "observe"
    )
    assert module._phase_from_progress_line("[cycle:cycle-1] campaign 1/1 select subsystem=retrieval") == (
        "campaign_select"
    )
    assert module._task_from_progress_line(
        "[cycle:cycle-1] campaign 1/1 select subsystem=retrieval"
    ) == {}
    assert (
        module._phase_from_progress_line(
            "[cycle:cycle-1] variant generate start subsystem=tolbert_model variant=recovery_alignment rank=1/2 expected_gain=0.0300 score=0.0075"
        )
        == "variant_generate"
    )
    assert (
        module._task_from_progress_line(
            "[cycle:cycle-1] variant generate heartbeat subsystem=tolbert_model variant=recovery_alignment stage=tolbert_pipeline_heartbeat pending=tolbert_finetune_pipeline"
        )
        == {}
    )


def test_adaptive_child_progress_state_uses_last_progress_timestamp_for_preview():
    module = _load_script("run_unattended_campaign.py")

    state = module._adaptive_child_progress_state(
        last_progress_phase="preview_baseline_eval",
        active_finalize_phase="",
        last_progress_at=1795.0,
        current_task={
            "index": 10,
            "total": 32,
            "task_id": "incident_matrix_task",
            "family": "integration",
            "phase": "preview_baseline_eval",
        },
        observe_summary={},
        pending_decision_state="",
        preview_state="",
        progress_samples=[],
        started_at=0.0,
        now=1800.0,
        max_runtime_seconds=0.0,
        max_progress_stall_seconds=1800.0,
    )

    assert state["phase"] == "preview_baseline_eval"
    assert state["phase_family"] == "preview"
    assert state["progress_class"] == "healthy"
    assert state["status"] == "active"
    assert state["progress_silence_seconds"] == 5.0


def test_adaptive_child_progress_state_prefers_generated_success_task_over_stale_finalize_phase():
    module = _load_script("run_unattended_campaign.py")

    state = module._adaptive_child_progress_state(
        last_progress_phase="generated_success",
        active_finalize_phase="preview_baseline_eval",
        last_progress_at=1795.0,
        current_task={
            "index": 20,
            "total": 32,
            "task_id": "integration_failover_drill_task",
            "family": "integration",
            "phase": "generated_success",
        },
        observe_summary={},
        pending_decision_state="",
        preview_state="",
        progress_samples=[],
        started_at=0.0,
        now=1800.0,
        max_runtime_seconds=3600.0,
        max_progress_stall_seconds=1800.0,
    )

    assert state["phase"] == "generated_success"
    assert state["phase_family"] == "finalize"
    assert state["progress_class"] == "healthy"
    assert state["detail"] == "finalize task 20/32 is progressing"


def test_adaptive_child_progress_state_marks_failed_generated_success_verification_as_degraded():
    module = _load_script("run_unattended_campaign.py")

    state = module._adaptive_child_progress_state(
        last_progress_phase="generated_success",
        active_finalize_phase="",
        last_progress_at=1795.0,
        current_task={
            "index": 8,
            "total": 8,
            "task_id": "repo_guardrail_review_task",
            "family": "repo_chore",
            "phase": "generated_success",
        },
        current_cognitive_stage={
            "cognitive_stage": "verification_result",
            "step_index": 1,
            "verification_passed": False,
            "phase": "generated_success",
        },
        observe_summary={},
        pending_decision_state="",
        preview_state="",
        progress_samples=[],
        started_at=0.0,
        now=1800.0,
        max_runtime_seconds=3600.0,
        max_progress_stall_seconds=1800.0,
    )

    assert state["phase"] == "generated_success"
    assert state["phase_family"] == "finalize"
    assert state["status"] == "active"
    assert state["progress_class"] == "degraded"
    assert state["detail"] == "finalize verification failed on step 1 and is still awaiting recovery"


def test_parse_child_semantic_progress_fields_extracts_observe_and_decision_state():
    module = _load_script("run_unattended_campaign.py")

    observe = module._parse_child_semantic_progress_fields(
        "[cycle:demo] observe complete passed=2/8 pass_rate=0.2500 generated_pass_rate=0.1250"
    )
    assert observe["observe_completed"] is True
    assert observe["observe_summary"] == {
        "passed": 2,
        "total": 8,
        "pass_rate": 0.25,
        "generated_pass_rate": 0.125,
    }

    preview = module._parse_child_semantic_progress_fields(
        "[cycle:demo] preview complete subsystem=retrieval variant=breadth_rebalance state=reject baseline_pass_rate=0.8"
    )
    assert preview["preview_state"] == "reject"
    assert preview["pending_decision_state"] == "reject"

    finalize = module._parse_child_semantic_progress_fields(
        "[cycle:demo] finalize phase=apply_decision subsystem=retrieval state=retain"
    )
    assert finalize["pending_decision_state"] == "retain"


def test_run_and_stream_feeds_child_semantic_fields_into_adaptive_state(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    lines = [
        "[cycle:test] observe complete passed=3/8 pass_rate=0.3750 generated_pass_rate=0.1250\n",
        "[cycle:test] preview complete subsystem=retrieval variant=breadth_rebalance state=reject baseline_pass_rate=0.8000\n",
        "[eval:test] phase=generated_success task 8/8 repo_guardrail_review_task family=repo_chore cognitive_stage=verification_result step=1 verification_passed=0\n",
    ]

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": "generated_success",
            "phase_family": "finalize",
            "status": "active",
            "progress_class": "degraded",
            "decision_distance": "near",
            "detail": "finalize verification failed on step 1 and is still awaiting recovery",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["observe_summary"] == {
        "passed": 3,
        "total": 8,
        "pass_rate": 0.375,
        "generated_pass_rate": 0.125,
    }
    assert latest["preview_state"] == "reject"
    assert latest["pending_decision_state"] == "reject"
    assert latest["current_task_verification_passed"] is False


def test_run_and_stream_uses_authoritative_mirrored_child_status_for_adaptive_state(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(["worker heartbeat\n"])
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": "preview_candidate_eval",
            "phase_family": "preview",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "near",
            "detail": "preview is progressing",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "active_cycle_run": {
                "observe_summary": {"passed": 4, "total": 4, "pass_rate": 1.0},
                "pending_decision_state": "reject",
                "preview_state": "reject",
                "current_task": {
                    "index": 7,
                    "total": 8,
                    "task_id": "bridge_handoff_task_integration_recovery",
                    "phase": "preview_candidate_eval",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "verification_result",
                    "step_index": 2,
                    "verification_passed": False,
                    "phase": "preview_candidate_eval",
                },
            }
        },
    )

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
        mirrored_status_path=tmp_path / "campaign.status.json",
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["observe_summary"] == {"passed": 4, "total": 4, "pass_rate": 1.0}
    assert latest["preview_state"] == "reject"
    assert latest["pending_decision_state"] == "reject"
    assert latest["current_task"]["phase"] == "preview_candidate_eval"
    assert latest["current_cognitive_stage"]["cognitive_stage"] == "verification_result"
    assert latest["current_task_verification_passed"] is False


def test_run_and_stream_clears_failed_verification_when_progress_moves_to_new_task(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    lines = [
        "[eval:test] phase=primary task 1/8 deployment_manifest_task family=project cognitive_stage=memory_update_written step=1 verification_passed=0\n",
        "[eval:test] phase=primary task 2/8 repo_sync_matrix_task family=repository cognitive_stage=memory_retrieved subphase=graph_memory\n",
    ]

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": str(kwargs.get("last_progress_phase", "")).strip(),
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "detail": "ok",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["current_task"]["task_id"] == "repo_sync_matrix_task"
    assert latest["current_task"]["index"] == 2
    assert latest["current_task_verification_passed"] is None


def test_run_and_stream_clears_failed_verification_when_progress_moves_to_new_step_same_task(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    lines = [
        "[eval:test] phase=primary task 3/128 semantic_open_world_task family=integration cognitive_stage=verification_result step=1 verification_passed=0\n",
        "[eval:test] phase=primary task 3/128 semantic_open_world_task family=integration cognitive_stage=llm_request step=3\n",
    ]

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": str(kwargs.get("last_progress_phase", "")).strip(),
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "detail": "ok",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["current_task"]["task_id"] == "semantic_open_world_task"
    assert latest["current_cognitive_stage"]["cognitive_stage"] == "llm_request"
    assert latest["current_cognitive_stage"]["step_index"] == 3
    assert latest["current_task_verification_passed"] is None


def test_run_and_stream_ignores_lagging_authoritative_child_verification_snapshot(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    lines = [
        "[eval:test] phase=primary task 1/128 semantic_open_world_task family=integration cognitive_stage=verification_result step=1 verification_passed=0\n",
        "[eval:test] phase=primary task 1/128 semantic_open_world_task family=integration cognitive_stage=llm_response step=2\n",
    ]

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": str(kwargs.get("last_progress_phase", "")).strip(),
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "detail": "ok",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "report_kind": "repeated_improvement_status",
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "verification_result",
                    "phase": "primary",
                    "step_index": 1,
                    "verification_passed": False,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "verification_result",
                        "phase": "primary",
                        "step_index": 1,
                        "verification_passed": False,
                    }
                ],
                "current_task_verification_passed": False,
            },
        },
    )

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
        mirrored_status_path=tmp_path / "campaign.status.json",
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["current_task"]["task_id"] == "semantic_open_world_task"
    assert latest["current_cognitive_stage"]["cognitive_stage"] == "llm_response"
    assert latest["current_cognitive_stage"]["step_index"] == 2
    assert latest["current_task_verification_passed"] is None


def test_run_and_stream_preserves_live_same_step_over_ahead_of_pipe_authoritative_snapshot(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    captured_calls: list[dict[str, object]] = []
    stdout_holder: dict[str, object] = {}

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def fileno(self):
            return 0

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakeProcess:
        def __init__(self, lines):
            self.pid = 2468
            self.stdout = FakeStdout(lines)

        def poll(self):
            if self.stdout._lines:
                return None
            return 0

        def wait(self):
            return 0

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def unregister(self, fileobj):
            del fileobj

        def select(self, timeout=None):
            del timeout
            stdout = stdout_holder["stdout"]
            if stdout._lines:
                return [(type("Key", (), {"fileobj": stdout})(), None)]
            return []

        def close(self):
            return None

    lines = [
        "[eval:test] phase=primary task 1/128 semantic_open_world_task family=integration cognitive_stage=verification_result step=1 verification_passed=0\n",
        "[eval:test] phase=primary task 1/128 semantic_open_world_task family=integration cognitive_stage=llm_response step=2\n",
    ]

    def fake_spawn_process_group(cmd, *, cwd, env, text, bufsize):
        del cmd, cwd, env, text, bufsize
        process = FakeProcess(lines)
        stdout_holder["stdout"] = process.stdout
        return process

    def fake_adaptive_child_progress_state(**kwargs):
        captured_calls.append(kwargs)
        return {
            "phase": str(kwargs.get("last_progress_phase", "")).strip(),
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "detail": "ok",
        }

    monkeypatch.setattr(module, "spawn_process_group", fake_spawn_process_group)
    monkeypatch.setattr(module.selectors, "DefaultSelector", lambda: FakeSelector())
    monkeypatch.setattr(module, "_adaptive_child_progress_state", fake_adaptive_child_progress_state)
    monkeypatch.setattr(module, "terminate_process_tree", lambda process: None)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "report_kind": "repeated_improvement_status",
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "memory_update_written",
                    "phase": "primary",
                    "step_index": 2,
                    "verification_passed": False,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "memory_update_written",
                        "phase": "primary",
                        "step_index": 2,
                        "verification_passed": False,
                    }
                ],
                "current_task_verification_passed": False,
            },
        },
    )

    result = module._run_and_stream(
        ["python", "-u", "child.py"],
        cwd=tmp_path,
        env={},
        progress_label="campaign_round_1",
        on_event=lambda event: None,
        mirrored_status_path=tmp_path / "campaign.status.json",
    )

    assert result["timed_out"] is False
    assert captured_calls
    latest = captured_calls[-1]
    assert latest["current_task"]["task_id"] == "semantic_open_world_task"
    assert latest["current_cognitive_stage"]["cognitive_stage"] == "llm_response"
    assert latest["current_cognitive_stage"]["step_index"] == 2
    assert latest["current_task_verification_passed"] is None


def test_validate_campaign_report_rejects_successful_runs_without_runtime_managed_decisions():
    module = _load_script("run_unattended_campaign.py")

    result = module._validate_campaign_report(
        {
            "report_kind": "improvement_campaign_report",
            "cycles_requested": 1,
            "completed_runs": 1,
            "successful_runs": 1,
            "inheritance_summary": {"runtime_managed_decisions": 0},
        }
    )

    assert result["passed"] is False
    assert result["detail"] == "campaign report showed no runtime-managed decisions"


def test_task_from_progress_line_clears_task_for_finalize_phase():
    module = _load_script("run_unattended_campaign.py")

    assert (
        module._task_from_progress_line(
            "[cycle:cycle-1] finalize phase=preview_baseline_eval subsystem=tolbert_model"
        )
        == {}
    )


def test_accept_productive_partial_child_timeout_accepts_generated_success_completion(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    child_report_path = tmp_path / "campaign_report.json"
    child_report_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")

    decision = module._accept_productive_partial_child_timeout(
        {"returncode": -9, "timeout_reason": "child exceeded max runtime of 300 seconds"},
        mirrored_child_status={
            "report_path": str(child_report_path),
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
        },
    )

    assert decision["accepted"] is True
    assert decision["report_path"] == str(child_report_path)


def test_merge_mirrored_child_status_fields_preserves_tolbert_runtime_summary():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {"semantic_progress_state": {"progress_class": "healthy"}},
        {
            "tolbert_runtime_summary": {
                "configured_runs": 1,
                "startup_failure_count": 1,
                "outcome_counts": {"failed_recovered": 1},
            },
            "active_cycle_run": {
                "tolbert_runtime_summary": {
                    "configured_to_use_tolbert": True,
                    "startup_failure_stages": ["preview_baseline"],
                    "recovered_without_tolbert_stages": ["preview_baseline"],
                    "bypassed_stages": ["preview_baseline"],
                    "outcome": "failed_recovered",
                }
            },
        },
    )

    assert merged["tolbert_runtime_summary"]["startup_failure_count"] == 1
    assert merged["tolbert_runtime_summary"]["outcome_counts"]["failed_recovered"] == 1


def test_accept_productive_partial_child_timeout_rejects_without_generated_success_completion(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    child_report_path = tmp_path / "campaign_report.json"
    child_report_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")

    decision = module._accept_productive_partial_child_timeout(
        {"returncode": -9, "timeout_reason": "child exceeded max runtime of 300 seconds"},
        mirrored_child_status={
            "report_path": str(child_report_path),
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": False,
            },
        },
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "generated_success_not_completed"


def test_accept_productive_partial_child_timeout_accepts_controller_intervention_shutdown(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    child_report_path = tmp_path / "campaign_report.json"
    child_report_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")

    decision = module._accept_productive_partial_child_timeout(
        {"returncode": -15, "timed_out": True, "timeout_reason": "controller stop"},
        mirrored_child_status={
            "report_path": str(child_report_path),
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
        },
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "generated_success_completed_before_controller_intervention"
    assert decision["report_path"] == str(child_report_path)


def test_accept_productive_partial_child_timeout_salvages_micro_step_verification_loop(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    child_report_path = tmp_path / "campaign_report.json"
    child_report_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")

    decision = module._accept_productive_partial_child_timeout(
        {"returncode": -15, "timed_out": True, "timeout_reason": "mid-round controller intervention"},
        mirrored_child_status={
            "report_path": str(child_report_path),
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": False,
                "sampled_families_from_progress": ["project"],
            },
        },
        round_payload={
            "controller_intervention": {
                "reason_code": "micro_step_verification_loop",
            }
        },
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "micro_step_verification_loop_salvaged_as_productive_partial"
    assert decision["report_path"] == str(child_report_path)


def test_accept_productive_partial_child_timeout_accepts_missing_report_path_for_micro_step_loop():
    module = _load_script("run_unattended_campaign.py")

    decision = module._accept_productive_partial_child_timeout(
        {"returncode": -15, "timed_out": True, "timeout_reason": "mid-round controller intervention"},
        mirrored_child_status={
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": False,
                "sampled_families_from_progress": ["project"],
            },
        },
        round_payload={
            "controller_intervention": {
                "reason_code": "micro_step_verification_loop",
            }
        },
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "micro_step_verification_loop_salvaged_without_report_path"
    assert decision["report_path"] == ""
    assert decision["requires_synthetic_report"] is True


def test_persist_partial_unattended_task_reports_from_event_log(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    event_log_path = reports_dir / "unattended_campaign.events.jsonl"
    event_log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_kind": "child_event",
                        "event": "cognitive_stage",
                        "round_index": 1,
                        "phase": "generated_success",
                        "timestamp": 100.0,
                        "step_index": 2,
                        "verification_passed": False,
                        "current_task": {
                            "task_id": "deployment_manifest_task_project_adjacent",
                            "family": "project",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_kind": "child_event",
                        "event": "cognitive_stage",
                        "round_index": 1,
                        "phase": "generated_success",
                        "timestamp": 110.0,
                        "step_index": 9,
                        "verification_passed": True,
                        "current_task": {
                            "task_id": "deployment_manifest_task_project_adjacent",
                            "family": "project",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_kind": "child_event",
                        "event": "cognitive_stage",
                        "round_index": 1,
                        "phase": "generated_success",
                        "timestamp": 115.0,
                        "step_index": 4,
                        "verification_passed": True,
                        "current_task": {
                            "task_id": "semantic_open_world_run123_round_1_project_project_recovery",
                            "family": "project",
                            "task_origin": "semantic_hub",
                            "source_task": "semantic_open_world_run123_round_1_project",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_kind": "child_event",
                        "event": "cognitive_stage",
                        "round_index": 1,
                        "phase": "generated_success",
                        "timestamp": 118.0,
                        "step_index": 1,
                        "verification_passed": True,
                        "current_task": {
                            "task_id": "deployment_manifest_task_project_adjacent",
                            "family": "project",
                            "task_origin": "external_manifest",
                            "source_task": "deployment_manifest_task",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_kind": "child_event",
                        "event": "cognitive_stage",
                        "round_index": 1,
                        "phase": "generated_failure",
                        "timestamp": 120.0,
                        "step_index": 1,
                        "verification_passed": True,
                        "current_task": {
                            "task_id": "bridge_handoff_task_integration_recovery",
                            "family": "integration",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = KernelConfig(run_reports_dir=reports_dir)
    paths = module._persist_partial_unattended_task_reports(
        reports_dir=reports_dir,
        event_log_path=event_log_path,
        round_index=1,
        mirrored_child_status=None,
        config=config,
    )

    assert len(paths) == 3
    payloads = {
        payload["task_id"]: payload
        for payload in (json.loads(Path(path).read_text(encoding="utf-8")) for path in paths)
    }
    success_payload = payloads["deployment_manifest_task_project_adjacent"]
    semantic_payload = payloads["semantic_open_world_run123_round_1_project_project_recovery"]
    recovery_payload = payloads["bridge_handoff_task_integration_recovery"]

    assert success_payload["report_kind"] == "unattended_task_report"
    assert success_payload["task_id"] == "deployment_manifest_task_project_adjacent"
    assert success_payload["success"] is True
    assert success_payload["task_metadata"]["task_origin"] == "external_manifest"
    assert success_payload["task_metadata"]["source_task"] == "deployment_manifest_task"
    assert success_payload["summary"]["command_steps"] == 1

    assert semantic_payload["report_kind"] == "unattended_task_report"
    assert semantic_payload["task_id"] == "semantic_open_world_run123_round_1_project_project_recovery"
    assert semantic_payload["success"] is True
    assert semantic_payload["task_metadata"]["task_origin"] == "semantic_hub"
    assert semantic_payload["task_metadata"]["source_task"] == "semantic_open_world_run123_round_1_project"

    assert recovery_payload["report_kind"] == "unattended_task_report"
    assert recovery_payload["task_id"] == "bridge_handoff_task_integration_recovery"
    assert recovery_payload["success"] is True
    assert recovery_payload["trust_scope"] == "contract_clean_failure_recovery"
    assert recovery_payload["task_metadata"]["curriculum_kind"] == "failure_recovery"
    assert recovery_payload["task_metadata"]["task_origin"] == "transition_pressure"
    assert recovery_payload["task_metadata"]["source_task"] == "bridge_handoff_task"
    assert recovery_payload["supervision"]["contract_clean_failure_recovery_candidate"] is True


def test_persist_partial_unattended_task_reports_falls_back_to_mirrored_child_status(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    config = KernelConfig(run_reports_dir=reports_dir)

    paths = module._persist_partial_unattended_task_reports(
        reports_dir=reports_dir,
        event_log_path=reports_dir / "missing.events.jsonl",
        round_index=2,
        mirrored_child_status={
            "current_task": {
                "task_id": "repository_guardrail_sync_task_repository_recovery",
                "family": "repository",
                "phase": "generated_failure",
            },
            "current_task_verification_passed": True,
            "current_task_progress_timeline": [
                {
                    "step_index": 3,
                    "verification_passed": True,
                    "timestamp": 200.0,
                }
            ],
        },
        config=config,
    )

    assert len(paths) == 1
    payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert payload["task_id"] == "repository_guardrail_sync_task_repository_recovery"
    assert payload["success"] is True
    assert payload["task_metadata"]["source_task"] == "repository_guardrail_sync_task"
    assert payload["summary"]["command_steps"] == 3


def test_persist_partial_unattended_task_reports_preserves_semantic_origin_from_mirrored_child_status(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    config = KernelConfig(run_reports_dir=reports_dir)

    paths = module._persist_partial_unattended_task_reports(
        reports_dir=reports_dir,
        event_log_path=reports_dir / "missing.events.jsonl",
        round_index=3,
        mirrored_child_status={
            "current_task": {
                "task_id": "semantic_open_world_run123_round_3_repository_repository_recovery",
                "family": "repository",
                "phase": "generated_failure",
            },
            "current_task_verification_passed": True,
            "current_task_progress_timeline": [
                {
                    "step_index": 2,
                    "verification_passed": True,
                    "timestamp": 300.0,
                }
            ],
        },
        config=config,
    )

    assert len(paths) == 1
    payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert payload["task_metadata"]["task_origin"] == "semantic_hub"
    assert payload["task_metadata"]["source_task"] == "semantic_open_world_run123_round_3_repository"


def test_persist_partial_unattended_task_reports_preserves_external_manifest_origin_from_mirrored_child_status(
    tmp_path,
):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    config = KernelConfig(run_reports_dir=reports_dir)

    paths = module._persist_partial_unattended_task_reports(
        reports_dir=reports_dir,
        event_log_path=reports_dir / "missing.events.jsonl",
        round_index=4,
        mirrored_child_status={
            "current_task": {
                "task_id": "deployment_manifest_task_project_recovery",
                "family": "project",
                "phase": "generated_failure",
                "task_origin": "external_manifest",
                "source_task": "deployment_manifest_task",
            },
            "current_task_verification_passed": True,
            "current_task_progress_timeline": [
                {
                    "step_index": 2,
                    "verification_passed": True,
                    "timestamp": 400.0,
                }
            ],
        },
        config=config,
    )

    assert len(paths) == 1
    payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert payload["task_metadata"]["task_origin"] == "external_manifest"
    assert payload["task_metadata"]["source_task"] == "deployment_manifest_task"


def test_synthetic_productive_partial_campaign_report_preserves_live_decision_and_retained_signals():
    module = _load_script("run_unattended_campaign.py")

    report = module._synthetic_productive_partial_campaign_report(
        mirrored_child_status={
            "partial_productive_runs": 1,
            "decision_records_considered": 0,
            "pending_decision_state": "retain",
            "retained_gain_runs": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
                "sampled_families_from_progress": ["project"],
            },
        },
        campaign_run={
            "returncode": -15,
            "timed_out": True,
            "timeout_reason": "controller stop",
        },
        round_payload={
            "active_child": {
                "current_task": {
                    "task_id": "project_release_cutover_task",
                    "phase": "generated_success",
                }
            }
        },
    )

    assert report["completed_runs"] == 1
    assert report["successful_runs"] == 1
    assert report["retained_gain_runs"] == 1
    assert report["decision_conversion_summary"]["decision_runs"] == 1
    assert report["runs"][0]["retained_gain"] is True
    assert report["runs"][0]["decision_records_considered"] == 1
    assert report["runs"][0]["decision_conversion_state"] == "non_runtime_managed"
    assert report["runs"][0]["decision_state"]["decision_owner"] == "child_native"
    assert report["runs"][0]["decision_state"]["closeout_mode"] == "child_native_before_partial_timeout"
    assert report["decision_state_summary"]["run_closeout_modes"]["child_native_before_partial_timeout"] == 1
    assert report["recent_production_decisions"][0]["decision_state"]["decision_credit"] == "child_emitted_decision"
    assert report["inferred_decision_credit"] is True


def test_synthetic_productive_partial_campaign_report_preserves_no_decision_when_live_decision_missing():
    module = _load_script("run_unattended_campaign.py")

    report = module._synthetic_productive_partial_campaign_report(
        mirrored_child_status={
            "partial_productive_runs": 1,
            "decision_records_considered": 0,
            "retained_gain_runs": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
                "sampled_families_from_progress": ["project"],
            },
        },
        campaign_run={
            "returncode": -15,
            "timed_out": True,
            "timeout_reason": "mid-round controller intervention",
        },
        round_payload={
            "active_child": {
                "current_task": {
                    "task_id": "project_release_cutover_task",
                    "phase": "generated_failure",
                }
            }
        },
    )

    assert report["runtime_managed_decisions"] == 0
    assert report["retained_gain_runs"] == 0
    assert report["decision_conversion_summary"]["runtime_managed_runs"] == 0
    assert report["decision_conversion_summary"]["partial_productive_without_decision_runs"] == 1
    assert report["runs"][0]["decision_records_considered"] == 0
    assert report["runs"][0]["retained_gain"] is False
    assert report["runs"][0]["decision_conversion_state"] == "partial_productive_without_decision"
    assert report["runs"][0]["decision_state"]["decision_owner"] == "none"
    assert report["runs"][0]["decision_state"]["closeout_mode"] == "partial_timeout_evidence_only"
    assert report["recent_runtime_managed_decisions"] == []


def test_credit_accepted_productive_partial_status_records_evidence_without_promoting_decision():
    module = _load_script("run_unattended_campaign.py")

    credited = module._credit_accepted_productive_partial_status(
        {
            "partial_productive_runs": 1,
            "decision_records_considered": 0,
            "retained_gain_runs": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
        }
    )

    assert credited["accepted_partial_timeout_evidence"] is True
    assert credited["decision_records_considered"] == 0
    assert credited["runtime_managed_decisions"] == 0
    assert credited["non_runtime_managed_decisions"] == 0
    assert credited["retained_gain_runs"] == 0
    assert credited["decision_conversion_summary"]["runtime_managed_runs"] == 0
    assert credited["decision_conversion_summary"]["partial_productive_without_decision_runs"] == 1


def test_normalize_accepted_productive_partial_campaign_report_preserves_missing_decision():
    module = _load_script("run_unattended_campaign.py")

    normalized = module._normalize_accepted_productive_partial_campaign_report(
        {
            "runtime_managed_decisions": 0,
            "retained_gain_runs": 0,
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 0,
            },
            "runs": [
                {
                    "partial_productive": True,
                    "generated_success_completed": True,
                    "decision_records_considered": 0,
                    "runtime_managed_decisions": 0,
                    "retained_gain": False,
                    "decision_conversion_state": "partial_productive_without_decision",
                }
            ],
        },
        campaign_run={
            "returncode": -9,
            "timed_out": True,
            "timeout_reason": "child exceeded max runtime safety ceiling of 3600 seconds",
        },
        round_payload={
            "campaign_partial_acceptance": {
                "accepted": True,
                "reason": "generated_success_completed_without_report_path",
            }
        },
    )

    assert normalized["runtime_managed_decisions"] == 0
    assert normalized["retained_gain_runs"] == 0
    assert normalized["decision_conversion_summary"]["runtime_managed_runs"] == 0
    assert normalized["decision_conversion_summary"]["partial_productive_without_decision_runs"] == 1
    assert normalized["runs"][0]["decision_records_considered"] == 0
    assert normalized["runs"][0]["runtime_managed_decisions"] == 0
    assert normalized["runs"][0]["non_runtime_managed_decisions"] == 0
    assert normalized["runs"][0]["retained_gain"] is False
    assert normalized["runs"][0]["decision_state"]["decision_owner"] == "none"
    assert normalized["runs"][0]["decision_state"]["closeout_mode"] == "partial_timeout_evidence_only"
    assert normalized["decision_state_summary"]["record_decisions"]["controller_runtime_manager"] == 0
    assert normalized["recent_runtime_managed_decisions"] == []


def test_synthetic_productive_partial_campaign_report_preserves_existing_non_runtime_credit():
    module = _load_script("run_unattended_campaign.py")

    report = module._synthetic_productive_partial_campaign_report(
        mirrored_child_status={
            "decision_records_considered": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 1,
            "retained_gain_runs": 0,
            "partial_productive_runs": 1,
            "active_cycle_progress": {
                "generated_success_completed": True,
                "productive_partial": True,
                "sampled_families_from_progress": ["integration", "repository"],
            },
        },
        campaign_run={"returncode": -9, "timed_out": True, "timeout_reason": "controller intervention"},
        round_payload={"active_child": {"current_task": {"task_id": "x", "phase": "generated_failure"}}},
    )

    assert report["runtime_managed_decisions"] == 0
    assert report["non_runtime_managed_decisions"] == 1
    assert report["retained_gain_runs"] == 0
    assert report["runs"][0]["decision_records_considered"] == 1
    assert report["runs"][0]["runtime_managed_decisions"] == 0
    assert report["runs"][0]["non_runtime_managed_decisions"] == 1
    assert report["runs"][0]["decision_conversion_state"] == "non_runtime_managed"


def test_credit_accepted_productive_partial_status_preserves_existing_decision_credit():
    module = _load_script("run_unattended_campaign.py")

    credited = module._credit_accepted_productive_partial_status(
        {
            "partial_productive_runs": 1,
            "decision_records_considered": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 1,
            "retained_gain_runs": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 1,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 1,
            },
        }
    )

    assert credited["decision_records_considered"] == 1
    assert credited["runtime_managed_decisions"] == 0
    assert credited["non_runtime_managed_decisions"] == 1
    assert credited["retained_gain_runs"] == 0
    assert credited["decision_conversion_summary"]["runtime_managed_runs"] == 0
    assert credited["decision_conversion_summary"]["non_runtime_managed_runs"] == 1


def test_finalize_accepted_productive_partial_child_status_converts_running_snapshot_into_finalized_status():
    module = _load_script("run_unattended_campaign.py")

    finalized = module._finalize_accepted_productive_partial_child_status(
        {
            "state": "running",
            "partial_productive_runs": 1,
            "decision_records_considered": 0,
            "retained_gain_runs": 0,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 0,
                "no_decision_runs": 0,
            },
        },
        campaign_report={
            "report_path": "/tmp/campaign_report.json",
            "completed_runs": 1,
            "successful_runs": 1,
            "productive_runs": 1,
            "retained_gain_runs": 1,
            "partial_productive_runs": 1,
            "partial_candidate_runs": 0,
            "runtime_managed_decisions": 1,
            "non_runtime_managed_decisions": 0,
            "decision_conversion_summary": {
                "runtime_managed_runs": 1,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 1,
                "no_decision_runs": 0,
            },
        },
        campaign_run={
            "returncode": -15,
            "timed_out": True,
            "timeout_reason": "mid-round controller intervention",
        },
    )

    assert finalized["state"] == "interrupted"
    assert finalized["report_kind"] == "repeated_improvement_status"
    assert finalized["report_path"] == "/tmp/campaign_report.json"
    assert finalized["completed_runs"] == 1
    assert finalized["successful_runs"] == 1
    assert finalized["productive_runs"] == 1
    assert finalized["runtime_managed_decisions"] == 1
    assert finalized["non_runtime_managed_decisions"] == 0
    assert finalized["retained_gain_runs"] == 1
    assert finalized["decision_conversion_summary"]["runtime_managed_runs"] == 1
    assert finalized["decision_conversion_summary"]["partial_productive_without_decision_runs"] == 0
    assert finalized["decision_conversion_summary"]["decision_runs"] == 1


def test_finalize_accepted_productive_partial_child_status_converts_timed_out_snapshot_into_interrupted():
    module = _load_script("run_unattended_campaign.py")

    finalized = module._finalize_accepted_productive_partial_child_status(
        {
            "state": "timed_out",
            "partial_productive_runs": 1,
            "decision_records_considered": 0,
            "retained_gain_runs": 0,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
            },
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 1,
                "decision_runs": 0,
                "no_decision_runs": 0,
            },
        },
        campaign_report={
            "report_path": "/tmp/campaign_report.json",
            "completed_runs": 1,
            "successful_runs": 1,
            "productive_runs": 1,
            "retained_gain_runs": 1,
            "partial_productive_runs": 1,
            "partial_candidate_runs": 0,
            "runtime_managed_decisions": 1,
            "non_runtime_managed_decisions": 0,
            "decision_conversion_summary": {
                "runtime_managed_runs": 1,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 1,
                "no_decision_runs": 0,
            },
        },
        campaign_run={
            "returncode": -9,
            "timed_out": True,
            "timeout_reason": "child exceeded max runtime safety ceiling of 300 seconds",
        },
    )

    assert finalized["state"] == "interrupted"
    assert finalized["runtime_managed_decisions"] == 1
    assert finalized["retained_gain_runs"] == 1


def test_write_controller_state_skips_storage_governance(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen: list[dict[str, object]] = []

    def fake_atomic_write_json(path, payload, *, config=None, govern_storage=True):
        seen.append(
            {
                "path": str(path),
                "payload": dict(payload),
                "config": config,
                "govern_storage": govern_storage,
            }
        )

    monkeypatch.setattr(module, "atomic_write_json", fake_atomic_write_json)

    module._write_controller_state(tmp_path / "controller.json", {"round": 1}, config=None)

    assert len(seen) == 1
    assert seen[0]["path"].endswith("controller.json")
    assert seen[0]["payload"]["round"] == 1
    assert seen[0]["govern_storage"] is False



def test_append_event_skips_storage_governance(monkeypatch, tmp_path):
    module = _load_script("run_unattended_campaign.py")
    seen: list[dict[str, object]] = []

    def fake_append_jsonl(path, payload, *, config=None, govern_storage=True):
        seen.append(
            {
                "path": str(path),
                "payload": dict(payload),
                "config": config,
                "govern_storage": govern_storage,
            }
        )

    monkeypatch.setattr(module, "append_jsonl", fake_append_jsonl)

    module._append_event(tmp_path / "events.jsonl", {"event_kind": "policy_shift"}, config=None)

    assert len(seen) == 1
    assert seen[0]["path"].endswith("events.jsonl")
    assert seen[0]["payload"]["event_kind"] == "policy_shift"
    assert seen[0]["govern_storage"] is False


def test_live_status_projection_prefers_normalized_campaign_report_for_authoritative_decision_state():
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "campaign_report_path": "/tmp/report.json",
            "campaign_report": {
                "runtime_managed_decisions": 1,
                "retained_gain_runs": 1,
                "runs": [
                    {
                        "decision_records_considered": 1,
                        "runtime_managed_decisions": 1,
                        "non_runtime_managed_decisions": 0,
                        "retained_gain": True,
                        "final_state": "retain",
                        "decision_conversion_state": "runtime_managed",
                    }
                ],
            },
        },
        active_run={},
        active_child={
            "decision_records_considered": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 0,
            "retained_gain_runs": 1,
            "sampled_families_from_progress": ["integration"],
        },
    )

    assert observed["runtime_managed_decisions"] == 1
    assert observed["decision_stream_summary"]["runtime_managed"]["total_decisions"] == 1
    assert observed["decision_stream_summary"]["runtime_managed"]["retained_cycles"] == 1
    assert observed["authoritative_decision_state"]["decision_conversion_state"] == "runtime_managed"
    assert observed["authoritative_decision_state"]["pending_decision_state"] == "retain"


def test_live_status_projection_surfaces_codex_input_packet_summary_for_closure_only_run(tmp_path):
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "report_path": str(tmp_path / "campaign.json"),
            "event_log_path": str(tmp_path / "campaign.events.jsonl"),
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            }
        },
        active_child={
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(tmp_path / "child_report.json"),
            "verification_outcome_summary": {
                "successful_task_count": 2,
                "successful_families": ["integration", "project"],
            },
            "sampled_families_from_progress": ["integration", "project"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
        },
    )

    summary = observed["codex_input_packet_summary"]
    assert summary["batch_packet"]["ready"] is False
    assert "missing_official_anchor_status_path" in summary["batch_packet"]["blockers"]
    assert summary["lane_packets"]["ready"] is True
    assert summary["frontier_packet"]["ready"] is False
    assert summary["baseline_packet"]["ready"] is False
    assert "missing_unfamiliar_domain_slice" in summary["frontier_packet"]["blockers"]
    assert "missing_strong_baseline_comparison_slice" in summary["frontier_packet"]["blockers"]
    assert "freeze_codex_batch_packet" in summary["suggested_actions"]
    assert "add_held_out_unfamiliar_domain_slice" in summary["suggested_actions"]
    assert "freeze_codex_frontier_packet" in summary["suggested_actions"]


def test_live_status_projection_surfaces_protocol_resource_lineage_summary(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    candidate_path = tmp_path / "candidates" / "policy.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "generation_context": {
                    "cycle_id": "cycle:policy:1",
                    "selection_id": "select:cycle:policy:1",
                    "resource_id": "subsystem:policy",
                    "active_artifact_path": str(tmp_path / "active" / "policy.json"),
                    "selection_record": {
                        "selection_id": "select:cycle:policy:1",
                        "resource_id": "subsystem:policy",
                        "edit_plan": {
                            "resource_id": "subsystem:policy",
                            "active_version_id": "sha256:baseline",
                            "active_resource_path": str(tmp_path / "runtime" / "policy.json"),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    observed = module._live_status_projection(
        payload={},
        active_run={},
        active_child={
            "active_cycle_run": {
                "last_candidate_artifact_path": str(candidate_path),
            },
        },
    )

    summary = observed["protocol_resource_lineage_summary"]
    assert summary["resource_id"] == "subsystem:policy"
    assert summary["active_version_id"] == "sha256:baseline"
    assert summary["candidate_artifact_path"] == str(candidate_path)
    assert summary["artifact_kind"] == "prompt_proposal_set"
    assert summary["versioned_resource"] is True
    assert summary["versioned_edit_plan"] is True
    assert "versioned_edit_plan" in observed["autonomy_level_assessment"]["higher_level_signals"]
    assert "no_retained_gain_runs" in observed["autonomy_level_assessment"]["level_gates"]["A6"]["blockers"]


def test_live_status_projection_autonomy_assessment_uses_minimum_fresh_proof_required_families():
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={},
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            }
        },
        active_child={
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore", "repo_sandbox"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": ["repo_sandbox"],
                "external_report_count": 4,
                "distinct_external_benchmark_families": 4,
            },
            "verification_outcome_summary": {
                "successful_task_count": 1,
                "successful_families": ["integration"],
            },
            "sampled_families_from_progress": ["integration"],
        },
    )

    required_families = observed["autonomy_level_assessment"]["required_families"]
    a4_blockers = observed["autonomy_level_assessment"]["level_gates"]["A4"]["blockers"]
    assert "repo_sandbox" not in required_families
    assert all("repo_sandbox" not in blocker for blocker in a4_blockers)


def test_live_status_projection_marks_codex_input_packet_set_ready_with_frontier_and_baseline_inputs(tmp_path):
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "report_path": str(tmp_path / "campaign.json"),
            "event_log_path": str(tmp_path / "campaign.events.jsonl"),
            "liftoff_report_path": str(tmp_path / "liftoff_report.json"),
            "campaign_report": {
                "recent_production_decisions": [
                    {
                        "subsystem": "retrieval",
                        "cycle_id": "retrieval:1",
                        "metrics_summary": {
                            "baseline_trusted_carryover_repair_rate": 0.25,
                            "trusted_carryover_repair_rate": 0.5,
                            "baseline_trusted_carryover_verified_steps": 2,
                            "trusted_carryover_verified_steps": 3,
                            "trusted_carryover_verified_step_delta": 1,
                        },
                    }
                ],
            },
            "unattended_evidence": {
                "semantic_hub_summary": {
                    "reports": 6,
                    "distinct_benchmark_families": 4,
                    "benchmark_families": ["integration", "project", "repository", "repo_chore"],
                },
                "external_summary": {
                    "total": 3,
                    "distinct_benchmark_families": 2,
                    "benchmark_families": ["integration", "repository"],
                    "success_rate_confidence_interval": {"level": 0.95, "lower": 0.3, "upper": 0.9},
                },
            },
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            }
        },
        active_child={
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(tmp_path / "child_report.json"),
            "verification_outcome_summary": {
                "successful_task_count": 4,
                "successful_families": ["integration", "project", "repository", "repo_chore"],
            },
            "sampled_families_from_progress": ["integration", "project", "repository", "repo_chore"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
            "semantic_progress_state": {
                "phase": "generated_success",
                "phase_family": "preview",
                "status": "active",
                "progress_class": "healthy",
                "detail": "preview work is progressing",
            },
            "last_progress_phase": "generated_success",
            "active_cycle_progress": {
                "productive_partial": True,
                "candidate_generated": True,
                "generated_success_started": True,
                "generated_success_completed": False,
            },
        },
    )

    summary = observed["codex_input_packet_summary"]
    assert summary["packet_set_ready"] is True
    assert summary["frontier_packet"]["ready"] is True
    assert summary["baseline_packet"]["ready"] is True
    assert summary["batch_packet"]["ready"] is True
    assert summary["lane_packets"]["ready"] is True
    assert summary["a8_minimum_frontier_inputs_ready"] is True


def test_live_status_projection_surfaces_coding_capability_ramp_summary_for_specialized_a3_run(tmp_path):
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "report_path": str(tmp_path / "campaign.json"),
            "event_log_path": str(tmp_path / "campaign.events.jsonl"),
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "semantic_only_runtime": True,
            }
        },
        active_child={
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(tmp_path / "child_report.json"),
            "verification_outcome_summary": {
                "successful_task_count": 2,
                "successful_families": ["integration", "project"],
            },
            "sampled_families_from_progress": ["integration", "project"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
        },
    )

    summary = observed["coding_capability_ramp_summary"]
    assert summary["declared_task_universe"] == "coding"
    assert summary["specialization_mode"] == "semantic_only_runtime"
    assert summary["current_transition"] == "A3->A4"
    assert summary["primary_gap_id"] == "runtime_managed_conversion"
    assert summary["recommended_primary_lane"] == "lane_decision_closure"
    assert summary["recommended_batch_goal"] == "retained_conversion"
    assert "freeze_codex_frontier_packet" in summary["suggested_actions"]


def test_live_status_projection_moves_coding_ramp_to_retrieval_after_conversion_and_trust_close(tmp_path):
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "report_path": str(tmp_path / "campaign.json"),
            "event_log_path": str(tmp_path / "campaign.events.jsonl"),
            "liftoff_report_path": str(tmp_path / "liftoff_report.json"),
            "campaign_report": {
                "runtime_managed_decisions": 1,
                "retained_gain_runs": 1,
                "runs": [
                    {
                        "decision_records_considered": 1,
                        "runtime_managed_decisions": 1,
                        "non_runtime_managed_decisions": 0,
                        "decision_owner": "child_native",
                        "closeout_mode": "natural",
                        "final_state": "retain",
                        "decision_conversion_state": "runtime_managed",
                    }
                ],
            },
            "unattended_evidence": {
                "semantic_hub_summary": {
                    "reports": 6,
                    "distinct_benchmark_families": 4,
                    "benchmark_families": ["integration", "project", "repository", "repo_chore"],
                },
                "external_summary": {
                    "total": 3,
                    "distinct_benchmark_families": 4,
                    "benchmark_families": ["integration", "project", "repository", "repo_chore"],
                    "success_rate_confidence_interval": {"level": 0.95, "lower": 0.3, "upper": 0.9},
                },
            },
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
                "semantic_only_runtime": True,
            }
        },
        active_child={
            "report_kind": "repeated_improvement_status",
            "pending_report_path": str(tmp_path / "child_report.json"),
            "runtime_managed_decisions": 1,
            "retained_gain_runs": 1,
            "verification_outcome_summary": {
                "successful_task_count": 4,
                "successful_families": ["integration", "project", "repository", "repo_chore"],
            },
            "sampled_families_from_progress": ["integration", "project", "repository", "repo_chore"],
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
                "external_report_count": 4,
                "distinct_external_benchmark_families": 4,
            },
            "semantic_progress_state": {
                "phase": "generated_success",
                "phase_family": "preview",
                "status": "active",
                "progress_class": "healthy",
                "detail": "preview work is progressing",
            },
            "last_progress_phase": "generated_success",
            "active_cycle_progress": {
                "productive_partial": True,
                "candidate_generated": True,
                "generated_success_started": True,
                "generated_success_completed": False,
            },
        },
    )

    summary = observed["coding_capability_ramp_summary"]
    assert summary["primary_gap_id"] == "retrieval_to_action_carryover"
    assert summary["recommended_batch_goal"] == "retrieval_and_memory_carryover"
    assert summary["recommended_primary_lane"] == "lane_trust_and_carryover"
    assert summary["a8_coding_frontier_ready"] is True


def test_live_status_projection_rates_bounded_success_as_a3():
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={},
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            }
        },
        active_child={
            "verification_outcome_summary": {
                "successful_task_count": 2,
                "successful_families": ["integration", "project"],
            },
            "sampled_families_from_progress": ["integration", "project"],
        },
    )

    assert observed["current_autonomy_level"] == "A3"
    assert observed["frontier_autonomy_level"] == "A3"
    assert observed["next_autonomy_level"] == "A4"
    assert "bounded_task_successes=2" in observed["autonomy_level_assessment"]["evidence"]
    assert observed["autonomy_level_assessment"]["level_gates"]["A3"]["met"] is True
    assert observed["autonomy_level_assessment"]["level_gates"]["A4"]["met"] is False
    assert "no_runtime_managed_decisions" in observed["autonomy_level_assessment"]["next_level_blockers"]
    assert observed["asi_campaign_lane_recommendation"]["primary_lane"] == "lane_decision_closure"
    assert observed["asi_campaign_lane_recommendation"]["primary_batch_goal"] == "retained_conversion"
    assert "lane_trust_and_carryover" in observed["asi_campaign_lane_recommendation"]["supporting_lanes"]



def test_live_status_projection_rates_child_native_natural_closure_with_breadth_as_a4():
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={
            "campaign_report": {
                "runtime_managed_decisions": 1,
                "retained_gain_runs": 0,
                "runs": [
                    {
                        "decision_records_considered": 1,
                        "runtime_managed_decisions": 1,
                        "non_runtime_managed_decisions": 0,
                        "decision_owner": "child_native",
                        "closeout_mode": "natural",
                        "final_state": "reject",
                        "decision_conversion_state": "runtime_managed",
                    }
                ],
            },
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            }
        },
        active_child={
            "runtime_managed_decisions": 1,
            "verification_outcome_summary": {
                "successful_task_count": 4,
                "successful_families": ["integration", "project", "repository", "repo_chore"],
            },
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": [],
            },
            "sampled_families_from_progress": ["integration", "project", "repository", "repo_chore"],
        },
    )

    assert observed["current_autonomy_level"] == "A4"
    assert observed["frontier_autonomy_level"] == "A4"
    assert observed["next_autonomy_level"] == "A5"
    assert observed["autonomy_level_assessment"]["decision_owner"] == "child_native"
    assert observed["autonomy_level_assessment"]["closeout_mode"] == "natural"
    assert observed["autonomy_level_assessment"]["level_gates"]["A4"]["met"] is True
    assert observed["autonomy_level_assessment"]["level_gates"]["A5"]["met"] is False
    assert "child_native_natural_decision_closure" in observed["autonomy_level_assessment"]["evidence"]
    assert "successful_resume_after_interruption_not_proven" in observed["autonomy_level_assessment"]["level_gates"]["A5"]["blockers"]
    assert observed["asi_campaign_lane_recommendation"]["primary_lane"] == "lane_runtime_authority"
    assert observed["asi_campaign_lane_recommendation"]["primary_batch_goal"] == "evidence_hygiene_handoff"


def test_live_status_projection_prefers_canonical_progress_state_over_stale_semantic_state():
    module = _load_script("run_unattended_campaign.py")

    observed = module._live_status_projection(
        payload={},
        active_run={},
        active_child={
            "current_task": {
                "index": 8,
                "total": 8,
                "task_id": "archive_command_seed_task_file_recovery",
                "family": "bounded",
                "phase": "generated_failure",
            },
            "semantic_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "degraded",
                "decision_distance": "far",
                "detail": "recovery task 8/8 failed verification and is awaiting recovery",
            },
            "adaptive_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "healthy",
                "decision_distance": "far",
                "detail": "recovery task 7/8 is progressing",
            },
            "canonical_progress_state": {
                "phase": "generated_failure",
                "phase_family": "recovery",
                "progress_class": "degraded",
                "decision_distance": "far",
                "detail": "recovery task 8/8 failed verification and is awaiting recovery",
            },
        },
    )

    assert observed["canonical_progress_state"]["detail"] == "recovery task 8/8 failed verification and is awaiting recovery"
    assert observed["semantic_progress_state"]["detail"] == "recovery task 8/8 failed verification and is awaiting recovery"
    assert observed["current_task"]["task_id"] == "archive_command_seed_task_file_recovery"


def test_run_unattended_campaign_accepted_productive_partial_timeout_runs_round_finalization(
    tmp_path, monkeypatch
):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(improvement_reports_dir=reports_dir)
    campaign_report_path = reports_dir / "campaign_report.json"
    campaign_report_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_report_path.write_text(
        json.dumps(
            {
                "report_kind": "improvement_campaign_report",
                "completed_runs": 1,
                "successful_runs": 1,
                "runtime_managed_decisions": 0,
                "retained_gain_runs": 0,
                "decision_conversion_summary": {
                    "runtime_managed_runs": 0,
                    "non_runtime_managed_runs": 0,
                    "partial_productive_without_decision_runs": 1,
                    "decision_runs": 0,
                    "no_decision_runs": 0,
                },
                "runs": [
                    {
                        "partial_productive": True,
                        "generated_success_completed": True,
                        "decision_records_considered": 0,
                        "runtime_managed_decisions": 0,
                        "retained_gain": False,
                        "decision_conversion_state": "partial_productive_without_decision",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cmd, cwd, env, kwargs
        assert on_event is not None
        on_event(
            {
                "event": "start",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "started_at": 1.0,
            }
        )
        on_event(
            {
                "event": "output",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "timestamp": 2.0,
                "line": (
                    "[eval:test] phase=generated_failure task 8/8 archive_command_seed_task_file_recovery "
                    "family=bounded family=bounded cognitive_stage=memory_update_written step=2 "
                    "verification_passed=1"
                ),
            }
        )
        return {
            "returncode": -15,
            "stdout": "",
            "stderr": "terminated",
            "timed_out": True,
            "timeout_reason": "mid-round controller intervention",
        }

    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "report_path": str(campaign_report_path),
            "state": "running",
            "partial_productive_runs": 1,
            "runtime_managed_decisions": 0,
            "retained_gain_runs": 0,
            "active_cycle_progress": {
                "productive_partial": True,
                "generated_success_completed": True,
                "sampled_families_from_progress": ["integration", "project", "repository"],
            },
        },
    )
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_load_retained_curriculum_controls", lambda cfg: {})
    monkeypatch.setattr(module, "_load_retained_improvement_planner_controls", lambda cfg: {})
    monkeypatch.setattr(module, "_next_round_policy", lambda current_policy, **kwargs: dict(current_policy))
    monkeypatch.setattr(module, "_persist_semantic_round_artifacts", lambda **kwargs: {})
    monkeypatch.setattr(module, "_unattended_evidence_snapshot", lambda cfg: {})
    monkeypatch.setattr(module, "_write_controller_state", lambda path, payload, *, config=None: None)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(sys, "argv", ["run_unattended_campaign.py", "--liftoff", "never"])
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status_payload = json.loads(report_path.with_suffix(".status.json").read_text(encoding="utf-8"))
    round_payload = payload["rounds"][0]

    assert round_payload["status"] == "completed"
    assert round_payload["phase"] == "completed"
    assert round_payload["campaign_validation"]["accepted_partial_timeout"] is True
    assert round_payload["campaign_validation"]["passed"] is False
    assert round_payload["campaign_validation"]["decision_credit"] == "none"
    assert (
        round_payload["liftoff_skipped"]["reason"]
        == "partial timeout incomplete child run; liftoff requires complete child report"
    )
    assert isinstance(round_payload["controller_update"], dict)
    assert round_payload["controller_observation"]["campaign_signal"]["runtime_managed_decisions"] == 0
    assert round_payload["controller_observation_start"]["campaign_signal"]["runtime_managed_decisions"] == 0
    assert round_payload["next_policy"] == payload["current_policy"]
    assert round_payload["authoritative_decision_state"]["runtime_managed_decisions"] == 0
    assert round_payload["authoritative_decision_state"]["retained_gain_runs"] == 0
    assert round_payload["runtime_managed_decisions"] == 0
    assert round_payload["retained_gain_runs"] == 0
    assert "runtime-managed retain" not in round_payload["phase_detail"]
    assert payload["rounds_completed"] == 1
    assert payload["authoritative_decision_state"]["runtime_managed_decisions"] == 0
    assert payload["runtime_managed_decisions"] == 0
    assert payload["retained_gain_runs"] == 0
    assert payload["decision_state_summary"]["run_closeout_modes"]["partial_timeout_evidence_only"] == 1
    assert payload["recent_runtime_managed_decisions"] == []
    assert len(payload["partial_task_report_paths"]) == 1
    partial_task_report = json.loads(Path(payload["partial_task_report_paths"][0]).read_text(encoding="utf-8"))
    assert partial_task_report["report_kind"] == "unattended_task_report"
    assert partial_task_report["task_id"] == "archive_command_seed_task_file_recovery"
    assert partial_task_report["benchmark_family"] == "bounded"
    assert partial_task_report["success"] is True
    assert partial_task_report["task_metadata"]["curriculum_kind"] == "failure_recovery"
    assert payload["rounds"][0]["decision_state_summary"]["run_decisions"]["controller_runtime_manager"] == 0
    assert payload["rounds"][0]["campaign_report"]["runs"][0]["decision_state"]["retention_state"] == "incomplete"
    assert "runtime-managed retain" not in payload["phase_detail"]
    assert status_payload["active_run"]["child_status"]["state"] == "interrupted"
    assert status_payload["active_run"]["child_status"]["runtime_managed_decisions"] == 0
    assert "runtime-managed retain" not in status_payload["phase_detail"]


def test_run_unattended_campaign_accepts_controller_intervention_from_child_mirrored_status_file(
    tmp_path, monkeypatch
):
    module = _load_script("run_unattended_campaign.py")
    reports_dir = tmp_path / "improvement" / "reports"
    config = KernelConfig(improvement_reports_dir=reports_dir)

    def fake_run_and_stream(cmd, *, cwd, env, on_event=None, **kwargs):
        del cmd, kwargs
        assert on_event is not None
        on_event(
            {
                "event": "start",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "started_at": 1.0,
            }
        )
        on_event(
            {
                "event": "output",
                "pid": 31337,
                "progress_label": "campaign_round_1",
                "timestamp": 2.0,
                "line": (
                    "[eval:test] phase=generated_failure task 8/8 archive_command_seed_task_file_recovery "
                    "family=bounded family=bounded cognitive_stage=memory_update_written step=2 "
                    "verification_passed=1"
                ),
            }
        )
        child_reports_dir = Path(env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"])
        child_reports_dir.mkdir(parents=True, exist_ok=True)
        campaign_report_path = child_reports_dir / "campaign_report.json"
        campaign_report_path.write_text(
            json.dumps(
                {
                    "report_kind": "improvement_campaign_report",
                    "report_path": str(campaign_report_path),
                    "completed_runs": 1,
                    "successful_runs": 1,
                    "runtime_managed_decisions": 0,
                    "retained_gain_runs": 0,
                    "decision_conversion_summary": {
                        "runtime_managed_runs": 0,
                        "non_runtime_managed_runs": 0,
                        "partial_productive_without_decision_runs": 1,
                        "decision_runs": 0,
                        "no_decision_runs": 0,
                    },
                    "runs": [
                        {
                            "partial_productive": True,
                            "generated_success_completed": True,
                            "decision_records_considered": 0,
                            "runtime_managed_decisions": 0,
                            "retained_gain": False,
                            "decision_conversion_state": "partial_productive_without_decision",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (child_reports_dir / "repeated_improvement_status.json").write_text(
            json.dumps(
                {
                    "report_kind": "repeated_improvement_status",
                    "report_path": str(campaign_report_path),
                    "state": "running",
                    "partial_productive_runs": 1,
                    "runtime_managed_decisions": 0,
                    "retained_gain_runs": 0,
                    "active_cycle_progress": {
                        "productive_partial": True,
                        "generated_success_completed": True,
                        "sampled_families_from_progress": ["integration", "project", "repository"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "returncode": -15,
            "stdout": "",
            "stderr": "terminated",
            "timed_out": True,
            "timeout_reason": "mid-round controller intervention",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "KernelConfig", lambda: config)
    monkeypatch.setattr(module, "_run_and_stream", fake_run_and_stream)
    monkeypatch.setattr(module, "_governed_cleanup_runtime_state", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_governed_global_storage_cleanup", lambda *args, **kwargs: {"cleanup": {}})
    monkeypatch.setattr(module, "_load_retained_curriculum_controls", lambda cfg: {})
    monkeypatch.setattr(module, "_load_retained_improvement_planner_controls", lambda cfg: {})
    monkeypatch.setattr(module, "_next_round_policy", lambda current_policy, **kwargs: dict(current_policy))
    monkeypatch.setattr(module, "_persist_semantic_round_artifacts", lambda **kwargs: {})
    monkeypatch.setattr(module, "_unattended_evidence_snapshot", lambda cfg: {})
    monkeypatch.setattr(module, "_write_controller_state", lambda path, payload, *, config=None: None)
    monkeypatch.setattr(
        module,
        "_disk_preflight",
        lambda path, *, min_free_gib: {
            "path": str(path),
            "free_bytes": 1024,
            "free_gib": 100.0,
            "min_free_gib": min_free_gib,
            "passed": True,
            "detail": "ok",
        },
    )
    monkeypatch.setattr(
        module,
        "_gpu_preflight",
        lambda device: {"device": device, "passed": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        module,
        "_trust_preflight",
        lambda cfg: {"passed": True, "status": "pass", "detail": "ok", "reports_considered": 1, "failing_thresholds": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unattended_campaign.py",
            "--liftoff",
            "never",
            "--rounds",
            "1",
            "--report-path",
            str(tmp_path / "campaign.json"),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    report_path = Path(stream.getvalue().strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["rounds_completed"] == 1
    assert payload["child_failure_recoveries_used"] == 0
    assert len(payload["rounds"]) == 1
    assert payload["rounds"][0]["status"] == "completed"
    assert payload["rounds"][0]["campaign_validation"]["accepted_partial_timeout"] is True
    assert payload["rounds"][0]["campaign_report_path"].endswith("campaign_report.json")


def test_merge_mirrored_child_status_fields_preserves_credited_runtime_managed_closure():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "decision_records_considered": 1,
            "runtime_managed_decisions": 1,
            "non_runtime_managed_decisions": 0,
            "retained_gain_runs": 1,
            "pending_decision_state": "retain",
            "decision_conversion_summary": {
                "runtime_managed_runs": 1,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 1,
                "no_decision_runs": 0,
            },
            "recent_structured_child_decisions": [{"state": "retain", "source": "parent_runtime_manager"}],
        },
        {
            "decision_records_considered": 1,
            "runtime_managed_decisions": 0,
            "non_runtime_managed_decisions": 0,
            "retained_gain_runs": 1,
            "pending_decision_state": "",
            "decision_conversion_summary": {
                "runtime_managed_runs": 0,
                "non_runtime_managed_runs": 0,
                "partial_productive_without_decision_runs": 0,
                "decision_runs": 0,
                "no_decision_runs": 0,
            },
            "recent_structured_child_decisions": [],
        },
    )

    assert merged["decision_records_considered"] == 1
    assert merged["runtime_managed_decisions"] == 1
    assert merged["non_runtime_managed_decisions"] == 0
    assert merged["retained_gain_runs"] == 1
    assert merged["pending_decision_state"] == "retain"
    assert merged["decision_conversion_summary"]["runtime_managed_runs"] == 1
    assert merged["decision_conversion_summary"]["decision_runs"] == 1
    assert merged["recent_structured_child_decisions"][0]["state"] == "retain"


def test_subsystem_signal_prefers_structured_child_decisions_over_stdout_inference():
    module = _load_script("run_unattended_campaign.py")

    signal = module._subsystem_signal(
        {
            "recent_non_runtime_decisions": [
                {
                    "subsystem": "retrieval",
                    "state": "reject",
                    "reason_code": "all_primary_verifications_failed",
                }
            ],
            "runs": [
                {
                    "subsystem": "retrieval",
                    "decision_conversion_state": "partial_productive_without_decision",
                    "stdout": "",
                }
            ],
        }
    )

    assert signal["rejected_by_subsystem"]["retrieval"] == 1
    assert signal["cooldown_candidates"] == ["retrieval"]


def test_live_status_projection_separates_bootstrap_sampling_from_trust_reports():
    module = _load_script("run_unattended_campaign.py")

    projected = module._live_status_projection(
        payload={
            "current_policy": {"priority_benchmark_families": ["project", "repository"]},
            "unattended_evidence": {"overall_assessment": {"status": "bootstrap"}},
        },
        active_run={"policy": {"priority_benchmark_families": ["project", "repository"]}},
        active_child={
            "sampled_families_from_progress": ["project", "repository"],
        },
    )

    assert projected["trust_status"] == "bootstrap"
    assert projected["sampled_families_from_progress"] == ["project", "repository"]
    assert projected["trust_breadth_summary"]["external_report_count"] == 0
    assert projected["trust_breadth_summary"]["distinct_external_benchmark_families"] == 0
    assert projected["trust_breadth_summary"]["bootstrap_sampled_required_families"] == ["project", "repository"]


def test_live_status_projection_surfaces_open_world_provider_carryover_finalize_and_handoff_summaries():
    module = _load_script("run_unattended_campaign.py")

    projected = module._live_status_projection(
        payload={
            "status": "interrupted",
            "reason": "runtime ceiling",
            "event_log_path": "/tmp/unattended.events.jsonl",
            "lock_path": "/tmp/unattended.lock",
            "unattended_evidence": {
                "external_summary": {
                    "total": 1,
                    "distinct_benchmark_families": 1,
                    "benchmark_families": ["repository"],
                },
                "semantic_hub_summary": {
                    "reports": 2,
                    "distinct_benchmark_families": 2,
                    "benchmark_families": ["integration", "project"],
                },
                "replay_derived_summary": {
                    "reports": 1,
                    "distinct_benchmark_families": 1,
                    "benchmark_families": ["repository"],
                },
            },
            "campaign_report": {
                "recent_production_decisions": [
                    {
                        "cycle_id": "cycle:retrieval:1",
                        "subsystem": "retrieval",
                        "state": "retain",
                        "metrics_summary": {
                            "baseline_trusted_carryover_repair_rate": 0.25,
                            "trusted_carryover_repair_rate": 0.5,
                            "baseline_trusted_carryover_verified_steps": 1,
                            "trusted_carryover_verified_steps": 2,
                            "trusted_carryover_verified_step_delta": 1,
                        },
                    },
                    {
                        "cycle_id": "cycle:tolbert:1",
                        "subsystem": "tolbert_model",
                        "state": "reject",
                        "reason": "Tolbert model candidate never entered retained Tolbert primary routing",
                    },
                ],
                "trust_breadth_summary": {
                    "required_families": ["integration", "project", "repository"],
                    "required_families_with_reports": ["integration", "project", "repository"],
                    "missing_required_families": [],
                    "external_report_count": 1,
                    "distinct_external_benchmark_families": 1,
                    "distinct_family_gap": 0,
                },
            },
        },
        active_run={
            "provider": "hybrid",
            "policy": {"priority_benchmark_families": ["integration", "project", "repository"]},
            "child_status": {"mirrored": True},
        },
        active_child={
            "last_progress_phase": "metrics_finalize",
            "last_report_path": "/tmp/campaign_report.json",
            "sampled_families_from_progress": ["integration", "project", "repository"],
            "active_cycle_progress": {
                "productive_partial": True,
                "candidate_generated": True,
                "generated_success_started": True,
                "generated_success_completed": False,
                "generated_failure_completed": True,
                "raw_phase": "metrics_finalize",
            },
            "pending_decision_state": "retain",
        },
    )

    assert projected["open_world_breadth_summary"]["semantic_hub_report_count"] == 2
    assert projected["open_world_breadth_summary"]["external_report_count"] == 1
    assert projected["open_world_breadth_summary"]["replay_derived_report_count"] == 1
    assert projected["provider_independence_summary"]["resolved_provider"] == "hybrid"
    assert projected["provider_independence_summary"]["provider_signal_state"] == "retained"
    assert projected["retrieval_carryover_summary"]["carryover_proven"] is True
    assert projected["tolbert_retained_routing_summary"]["primary_routing_failures"] == 1
    assert projected["long_horizon_finalize_summary"]["finalize_conversion_gap"] is False
    assert projected["handoff_summary"]["handoff_ready"] is True
    assert projected["closure_gap_summary"]["open_world_breadth"] == "partial"
    assert projected["closure_gap_summary"]["decoder_runtime_independence"] == "partial"
    assert projected["closure_gap_summary"]["retrieval_carryover"] == "closed"
    assert projected["closure_gap_summary"]["tolbert_retained_routing"] == "partial"


def test_live_status_projection_exposes_filtered_minimum_fresh_proof_gap():
    module = _load_script("run_unattended_campaign.py")

    projected = module._live_status_projection(
        payload={
            "current_policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore", "repo_sandbox"]
            },
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository", "repo_chore", "repo_sandbox"],
                "required_families_with_reports": ["integration", "project", "repository", "repo_chore"],
                "missing_required_families": ["repo_sandbox"],
                "external_report_count": 4,
                "distinct_external_benchmark_families": 4,
                "distinct_family_gap": 1,
            },
        },
        active_run={
            "policy": {
                "priority_benchmark_families": ["integration", "project", "repository", "repo_chore", "repo_sandbox"]
            }
        },
        active_child={
            "sampled_families_from_progress": ["integration", "project", "repository", "repo_chore"],
        },
    )

    assert projected["trust_breadth_summary"]["missing_required_families"] == ["repo_sandbox"]
    assert projected["trust_breadth_summary"]["proof_required_families"] == [
        "integration",
        "project",
        "repository",
        "repo_chore",
    ]
    assert projected["trust_breadth_summary"]["proof_required_families_with_reports"] == [
        "integration",
        "project",
        "repository",
        "repo_chore",
    ]
    assert projected["trust_breadth_summary"]["proof_missing_required_families"] == []
    assert projected["trust_breadth_summary"]["proof_distinct_family_gap"] == 0
    assert projected["closure_gap_summary"]["trust_breadth"] == "closed"
    assert projected["benchmark_dominance_summary"]["suite_kind"] == "minimum_fresh_proof_suite"
    assert projected["benchmark_dominance_summary"]["required_families"] == [
        "integration",
        "project",
        "repository",
        "repo_chore",
    ]
    assert projected["benchmark_dominance_summary"]["open_families"] == []
    assert "repo_sandbox" not in projected["benchmark_dominance_summary"]["required_family_statuses"]


def test_live_status_projection_prefers_live_child_breadth_summaries():
    module = _load_script("run_unattended_campaign.py")

    projected = module._live_status_projection(
        payload={
            "current_policy": {
                "priority_benchmark_families": ["integration", "project", "repository"]
            },
            "unattended_evidence": {
                "semantic_hub_summary": {
                    "reports": 1,
                    "distinct_benchmark_families": 1,
                    "benchmark_families": ["integration"],
                },
                "external_summary": {
                    "total": 0,
                    "distinct_benchmark_families": 0,
                    "benchmark_families": [],
                },
                "replay_derived_summary": {
                    "reports": 0,
                    "distinct_benchmark_families": 0,
                    "benchmark_families": [],
                },
            },
            "campaign_report": {
                "trust_breadth_summary": {
                    "required_families": ["integration", "project", "repository"],
                    "required_families_with_reports": ["integration"],
                    "missing_required_families": ["project", "repository"],
                    "external_report_count": 0,
                    "distinct_external_benchmark_families": 0,
                    "distinct_family_gap": 2,
                },
            },
        },
        active_run={
            "policy": {"priority_benchmark_families": ["integration", "project", "repository"]},
        },
        active_child={
            "sampled_families_from_progress": ["integration", "project", "repository"],
            "verification_outcome_summary": {
                "verified_task_count": 2,
                "failed_task_count": 0,
                "successful_task_count": 2,
                "verified_families": ["integration", "project"],
                "failed_families": [],
                "successful_families": ["integration", "project"],
            },
            "trust_breadth_summary": {
                "required_families": ["integration", "project", "repository"],
                "required_families_with_reports": ["integration", "project", "repository"],
                "missing_required_families": [],
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
                "distinct_family_gap": 0,
            },
            "open_world_breadth_summary": {
                "semantic_hub_report_count": 9,
                "distinct_semantic_hub_benchmark_families": 3,
                "semantic_hub_benchmark_families": ["integration", "project", "repository"],
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
                "external_benchmark_families": [],
                "replay_derived_report_count": 0,
                "distinct_replay_derived_benchmark_families": 0,
                "replay_derived_benchmark_families": [],
                "open_world_report_count": 9,
                "distinct_open_world_benchmark_families": 3,
                "benchmark_families": ["integration", "project", "repository"],
                "open_world_benchmark_families": ["integration", "project", "repository"],
                "retained_open_world_gain": False,
                "open_world_task_yield_present": True,
                "semantic_hub_summary": {
                    "reports": 9,
                    "distinct_benchmark_families": 3,
                    "benchmark_families": ["integration", "project", "repository"],
                },
                "external_summary": {"total": 0, "distinct_benchmark_families": 0, "benchmark_families": []},
                "replay_derived_summary": {
                    "reports": 0,
                    "distinct_benchmark_families": 0,
                    "benchmark_families": [],
                },
            },
        },
    )

    assert projected["verification_outcome_summary"]["successful_task_count"] == 2
    assert projected["trust_breadth_summary"]["required_families_with_reports"] == [
        "integration",
        "project",
        "repository",
    ]
    assert projected["trust_breadth_summary"]["missing_required_families"] == []
    assert projected["open_world_breadth_summary"]["semantic_hub_report_count"] == 9
    assert projected["open_world_breadth_summary"]["distinct_open_world_benchmark_families"] == 3


def test_merge_mirrored_child_status_fields_keeps_live_open_world_and_verification_summaries():
    module = _load_script("run_unattended_campaign.py")

    merged = module._merge_mirrored_child_status_fields(
        {
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "semantic_open_world_round_1_integration",
                "family": "integration",
                "phase": "primary",
            },
            "semantic_progress_state": {
                "phase": "primary",
                "progress_class": "healthy",
                "recorded_at": 10.0,
            },
        },
        {
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "semantic_open_world_round_1_integration",
                "family": "integration",
                "phase": "primary",
            },
            "semantic_progress_state": {
                "phase": "primary",
                "progress_class": "healthy",
                "recorded_at": 20.0,
            },
            "verification_outcome_summary": {
                "verified_task_count": 2,
                "failed_task_count": 1,
                "successful_task_count": 1,
                "verified_families": ["integration"],
                "failed_families": ["integration"],
                "successful_families": ["integration"],
            },
            "trust_breadth_summary": {
                "required_families_with_reports": ["integration"],
            },
            "open_world_breadth_summary": {
                "semantic_hub_report_count": 2,
                "distinct_open_world_benchmark_families": 1,
                "benchmark_families": ["integration"],
            },
        },
    )

    assert merged["verification_outcome_summary"]["verified_task_count"] == 2
    assert merged["trust_breadth_summary"]["required_families_with_reports"] == ["integration"]
    assert merged["open_world_breadth_summary"]["semantic_hub_report_count"] == 2


def test_controller_observation_includes_strategy_memory_and_trust_features():
    module = _load_script("run_unattended_campaign.py")

    observation = module._controller_observation(
        campaign_report={
            "production_yield_summary": {},
            "phase_gate_summary": {},
            "priority_family_yield_summary": {"priority_families": ["repository"]},
            "trust_breadth_summary": {
                "required_families": ["repository"],
                "required_families_with_reports": [],
                "missing_required_families": ["repository"],
                "distinct_family_gap": 1,
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
            },
        },
        liftoff_payload=None,
        controller_state={
            "repo_setting_policy_priors": {
                "repository": {
                    "campaign_width_unit_weight": 0.5,
                    "adaptive_search_bonus": 0.3,
                }
            }
        },
        curriculum_controls={"frontier_repo_setting_priority_pairs": ["repository:worker_handoff"]},
        round_payload={
            "policy": {"priority_benchmark_families": ["repository"]},
            "active_child": {"sampled_families_from_progress": ["repository"]},
        },
    )

    assert observation["features"]["strategy_memory_prior_strength"] > 0.0
    assert observation["features"]["trust_bootstrap_pressure"] > 0.0


def test_controller_observation_uses_retained_and_rejected_strategy_memory_priors():
    module = _load_script("run_unattended_campaign.py")

    observation = module._controller_observation(
        campaign_report={
            "production_yield_summary": {},
            "phase_gate_summary": {},
            "priority_family_yield_summary": {"priority_families": ["integration"]},
            "trust_breadth_summary": {
                "required_families": ["integration"],
                "required_families_with_reports": [],
                "missing_required_families": ["integration"],
                "distinct_family_gap": 1,
                "external_report_count": 0,
                "distinct_external_benchmark_families": 0,
            },
        },
        liftoff_payload=None,
        controller_state={
            "strategy_memory_priors": {
                "retained_count": 1,
                "best_retained_gain": 0.22,
                "recent_rejects": 2,
                "retained_family_breadth_gain": 2.0,
                "retained_closeout_gain": 1.0,
                "recent_reject_closeout_failures": 1,
                "retained_family_gains": {"integration": 0.22},
                "recent_rejects_by_family": {"integration": 2},
            }
        },
        curriculum_controls={"frontier_repo_setting_priority_pairs": ["integration:worker_handoff"]},
        round_payload={
            "policy": {"priority_benchmark_families": ["integration"]},
            "active_child": {"sampled_families_from_progress": ["integration"]},
        },
    )

    assert observation["features"]["strategy_memory_prior_strength"] > 0.0
    assert observation["features"]["strategy_memory_alignment"] > 0.0
    assert observation["features"]["strategy_memory_reject_pressure"] > 0.0
    assert observation["features"]["strategy_memory_family_breadth_gain"] > 0.0
    assert observation["features"]["strategy_memory_closeout_pressure"] > 0.0


def test_finalize_completed_campaign_round_refreshes_priors_before_controller_observation(tmp_path, monkeypatch):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        trajectories_root=tmp_path / "trajectories",
        semantic_hub_root=tmp_path / "trajectories/semantic_hub",
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    config.ensure_directories()

    observed_state: dict[str, object] = {}

    def fake_update_strategy_memory_priors(controller_state, *, config):
        return {
            **dict(controller_state or {}),
            "strategy_memory_priors": {"retained_count": 1, "best_retained_gain": 0.2},
        }

    def fake_update_repo_setting_policy_priors(controller_state, *, round_payload):
        return {
            **dict(controller_state or {}),
            "repo_setting_policy_priors": {
                "repository": {
                    "campaign_width_unit_weight": 0.5,
                    "adaptive_search_bonus": 0.25,
                }
            },
        }

    def fake_controller_observation(**kwargs):
        observed_state.update(dict(kwargs.get("controller_state", {})))
        return {"features": {"strategy_memory_prior_strength": 1.0}}

    monkeypatch.setattr(module, "_update_strategy_memory_priors", fake_update_strategy_memory_priors)
    monkeypatch.setattr(module, "_update_repo_setting_policy_priors", fake_update_repo_setting_policy_priors)
    monkeypatch.setattr(module, "_controller_observation", fake_controller_observation)
    monkeypatch.setattr(
        module,
        "update_controller_state",
        lambda controller_state, **kwargs: (
            normalize_controller_state(controller_state),
            {"reward": 0.0, "promotion_reward": 0.0, "steering_reward": 0.0},
        ),
    )
    monkeypatch.setattr(module, "_authoritative_status_decision_state", lambda **kwargs: {})
    monkeypatch.setattr(module, "_next_round_policy", lambda current_policy, **kwargs: dict(current_policy))
    monkeypatch.setattr(module, "_next_policy_stall_rounds", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_next_no_yield_rounds", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_next_depth_runway_credit", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_round_stop_signal", lambda **kwargs: {})
    monkeypatch.setattr(module, "_adaptive_stop_budget", lambda **kwargs: {})
    monkeypatch.setattr(module, "_mark_round", lambda round_payload, **kwargs: round_payload.update(kwargs))
    monkeypatch.setattr(module, "_persist_semantic_round_artifacts", lambda **kwargs: {})
    monkeypatch.setattr(module, "_count_completed_rounds", lambda rounds: 1)
    monkeypatch.setattr(module, "_policy_shift_summary", lambda rounds: {})
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)

    report_path = tmp_path / "reports" / "unattended_campaign.json"
    controller_state_path = tmp_path / "reports" / "controller_state.json"
    event_log_path = tmp_path / "reports" / "events.jsonl"
    report = {"rounds": [{}]}
    round_payload = {
        "policy": {"priority_benchmark_families": ["repository"]},
        "campaign_report": {},
    }
    campaign_report = {
        "production_yield_summary": {},
        "phase_gate_summary": {},
        "priority_family_yield_summary": {"priority_families": ["repository"]},
        "trust_breadth_summary": {
            "required_families": ["repository"],
            "required_families_with_reports": [],
            "missing_required_families": ["repository"],
            "distinct_family_gap": 1,
            "external_report_count": 0,
            "distinct_external_benchmark_families": 0,
        },
    }

    module._finalize_completed_campaign_round(
        args=type(
            "Args",
            (),
            {
                "max_no_yield_rounds": 1,
                "max_policy_stall_rounds": 1,
                "max_cycles_per_round": 1,
                "max_task_limit": 64,
                "max_campaign_width": 2,
                "max_variant_width": 2,
            },
        )(),
        config=config,
        report_path=report_path,
        status_path=None,
        lock_path=None,
        event_log_path=event_log_path,
        controller_state_path=controller_state_path,
        run_id="run-1",
        round_index=1,
        report=report,
        round_payload=round_payload,
        campaign_report=campaign_report,
        liftoff_payload=None,
        controller_state=default_controller_state(),
        current_policy={"priority_benchmark_families": ["repository"]},
        round_planner_controls={},
        round_curriculum_controls={"frontier_repo_setting_priority_pairs": ["repository:worker_handoff"]},
        no_yield_rounds=0,
        policy_stall_rounds=0,
        depth_runway_credit=0,
    )

    assert observed_state["strategy_memory_priors"]["retained_count"] == 1
    assert observed_state["repo_setting_policy_priors"]["repository"]["campaign_width_unit_weight"] == 0.5


def test_strategy_memory_candidate_score_adjustment_penalizes_replay_after_rejects():
    module = _load_script("run_unattended_campaign.py")

    unchanged = module._strategy_memory_candidate_score_adjustment(
        policy={"campaign_width": 1, "task_step_floor": 1, "adaptive_search": False},
        current_policy={"campaign_width": 1, "task_step_floor": 1, "adaptive_search": False},
        planner_pressure_signal={"priority_families": ["integration"]},
        strategy_memory_priors={
            "retained_family_gains": {"integration": 0.18},
            "retained_family_breadth_gain": 2.0,
            "retained_closeout_gain": 1.0,
            "recent_rejects_by_family": {"integration": 2},
        },
    )
    changed = module._strategy_memory_candidate_score_adjustment(
        policy={"campaign_width": 2, "task_step_floor": 2, "adaptive_search": True},
        current_policy={"campaign_width": 1, "task_step_floor": 1, "adaptive_search": False},
        planner_pressure_signal={"priority_families": ["integration"]},
        strategy_memory_priors={
            "retained_family_gains": {"integration": 0.18},
            "retained_family_breadth_gain": 2.0,
            "retained_closeout_gain": 1.0,
            "recent_rejects_by_family": {"integration": 2},
        },
    )

    assert unchanged["score_adjustment"] < changed["score_adjustment"]
    assert unchanged["reject_alignment_pressure"] > 0.0
    assert changed["policy_change_units"] > 0.0
    assert changed["retained_family_breadth_gain"] > 0.0
    assert changed["retained_closeout_gain"] > 0.0


def test_semantic_redirection_summary_requires_new_parent_after_stagnation():
    module = _load_script("run_unattended_campaign.py")

    summary = module._semantic_redirection_summary(
        {
            "policy_shift_rationale": {
                "reason_codes": [
                    "no_yield_round",
                    "stalled_subsystem_cooldown",
                    "subsystem_reject_cooldown",
                ],
                "cooled_subsystems": ["retrieval"],
                "stalled_subsystem": "retrieval",
                "priority_benchmark_families_after": ["integration", "repo_chore"],
                "planner_pressure": {
                    "missing_required_families": ["integration", "repo_chore"],
                },
            }
        }
    )

    assert summary["triggered"] is True
    assert summary["required_new_parent"] is True
    assert summary["source_subsystems"] == ["retrieval"]
    assert summary["target_priority_families"] == ["integration", "repo_chore"]
    assert summary["missing_required_families"] == ["integration", "repo_chore"]
    assert summary["semantic_hypotheses"]


def test_semantic_redirection_summary_uses_low_return_family_and_dominant_reject_subsystem():
    module = _load_script("run_unattended_campaign.py")

    summary = module._semantic_redirection_summary(
        {
            "policy_shift_rationale": {
                "reason_codes": ["weak_retention_outcome"],
            },
            "campaign_report": {
                "production_yield_summary": {
                    "rejected_by_subsystem": {
                        "tooling": 3,
                    }
                },
                "priority_family_yield_summary": {
                    "priority_families_with_signal_but_no_retained_gain": ["integration", "repository"],
                },
            },
        }
    )

    assert summary["triggered"] is True
    assert summary["required_new_parent"] is True
    assert summary["source_subsystems"] == ["tooling"]
    assert summary["target_priority_families"] == ["integration", "repository"]
    assert any("integration,repository" in item for item in summary["semantic_hypotheses"])


def test_persist_semantic_round_artifacts_writes_attempt_note_and_redirect(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        trajectories_root=tmp_path / "trajectories",
        semantic_hub_root=tmp_path / "trajectories/semantic_hub",
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    config.ensure_directories()

    paths = module._persist_semantic_round_artifacts(
        config=config,
        run_id="run-1",
        round_payload={
            "round_index": 2,
            "status": "completed",
            "phase": "completed",
            "policy": {"focus": "recovery_alignment"},
            "next_policy": {"focus": "discovered_task_adaptation"},
            "policy_shift_rationale": {"reason_codes": ["no_yield_round"]},
            "semantic_redirection": {
                "triggered": True,
                "reason_codes": ["no_yield_round"],
                "source_subsystems": ["retrieval"],
                "target_priority_families": ["integration"],
                "missing_required_families": ["integration"],
                "required_new_parent": True,
                "semantic_hypotheses": ["force descendants to adopt a new parent"],
            },
            "campaign_report": {
                "production_yield_summary": {
                    "rejected_cycles": 3,
                    "retained_cycles": 0,
                    "rejected_by_subsystem": {"retrieval": 3},
                },
                "priority_family_yield_summary": {
                    "priority_families_with_signal_but_no_retained_gain": ["integration"],
                },
                "retained_gain_runs": 0,
                "runtime_managed_decisions": 3,
            },
        },
        current_policy={"focus": "recovery_alignment"},
    )

    assert paths["semantic_attempt_path"].endswith(".json")
    assert paths["semantic_note_path"].endswith(".json")
    assert paths["semantic_redirect_path"].endswith(".json")
    assert paths["semantic_skill_path"].endswith(".json")
    assert paths["semantic_task_paths"]
    attempt = json.loads(Path(paths["semantic_attempt_path"]).read_text(encoding="utf-8"))
    assert attempt["semantic_redirection"]["required_new_parent"] is True
    redirect = json.loads(Path(paths["semantic_redirect_path"]).read_text(encoding="utf-8"))
    assert redirect["triggered"] is True
    skill = json.loads(Path(paths["semantic_skill_path"]).read_text(encoding="utf-8"))
    assert skill["priority_families"] == ["integration"]
    semantic_task = json.loads(Path(paths["semantic_task_paths"][0]).read_text(encoding="utf-8"))
    assert semantic_task["target_family"] == "integration"
    assert semantic_task["task"]["metadata"]["benchmark_family"] == "integration"
    assert semantic_task["task"]["metadata"]["open_world_candidate"] is True
    agent = json.loads(
        (tmp_path / "trajectories/semantic_hub/agents" / "unattended_agent_run-1.json").read_text(encoding="utf-8")
    )
    assert agent["last_attempt_id"] == "unattended_round:run-1:2"
    assert agent["last_skill_id"].startswith("skill:")
    assert agent["last_task_ids"][0].startswith("task:")


def test_preferred_unattended_provider_defaults_to_seed_runtime_even_when_retained_surfaces_exist(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    artifact_path = tmp_path / "trajectories" / "tolbert_model" / "tolbert_model_artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                    "universal_runtime": True,
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(tolbert_model_artifact_path=artifact_path)

    provider = module._preferred_unattended_provider(type("Args", (), {"provider": None})(), config=config)

    assert provider == "vllm"


def test_semantic_round_task_payloads_emit_shell_free_contract_commands(tmp_path):
    module = _load_script("run_unattended_campaign.py")

    tasks = module._semantic_round_task_payloads(
        run_id="run-1",
        round_payload={
            "round_index": 2,
            "semantic_redirection": {
                "target_priority_families": ["integration"],
            },
        },
        current_policy={"priority_benchmark_families": ["integration"]},
    )

    assert len(tasks) == 1
    task = tasks[0]["payload"]["task"]
    assert "python - <<'PY'" not in task["success_command"]
    assert all("python - <<'PY'" not in command for command in task["suggested_commands"])
    assert len(task["suggested_commands"]) == 1

    sandbox = Sandbox(timeout_seconds=10)
    for command in task["suggested_commands"]:
        result = sandbox.run(command, tmp_path)
        assert result.exit_code == 0, result.stderr

    success = sandbox.run(task["success_command"], tmp_path)
    assert success.exit_code == 0, success.stderr
    contract_payload = json.loads((tmp_path / "semantic_open_world" / "integration_task.json").read_text(encoding="utf-8"))
    assert contract_payload["metadata"]["benchmark_family"] == "integration"
    assert contract_payload["metadata"]["open_world_candidate"] is True


def test_task_bank_normalizes_legacy_semantic_hub_tasks_to_shell_free_contract_commands(tmp_path):
    semantic_hub_root = tmp_path / "semantic_hub"
    tasks_root = semantic_hub_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    legacy_task_path = tasks_root / "semantic_open_world_legacy_round_1_integration.json"
    legacy_task_path.write_text(
        json.dumps(
            {
                "task": {
                    "task_id": "semantic_open_world_legacy_round_1_integration",
                    "prompt": "Inspect this repository for a non-replay integration task surface.",
                    "workspace_subdir": "semantic_hub/semantic_open_world_legacy_round_1_integration",
                    "suggested_commands": [
                        "mkdir -p semantic_open_world && printf '# integration open-world evidence\\n' > semantic_open_world/integration_evidence.md",
                        "python - <<'PY'\nprint('legacy blocked command')\nPY",
                    ],
                    "success_command": "python - <<'PY'\nprint('legacy blocked success')\nPY",
                    "expected_files": [
                        "semantic_open_world/integration_task.json",
                        "semantic_open_world/integration_evidence.md",
                    ],
                    "metadata": {
                        "benchmark_family": "integration",
                        "capability": "semantic_open_world",
                        "open_world_candidate": True,
                        "semantic_parent_attempt_id": "unattended_round:legacy:1",
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = KernelConfig(semantic_hub_root=semantic_hub_root)

    task = TaskBank(config=config).get("semantic_open_world_legacy_round_1_integration")

    assert "python - <<'PY'" not in task.success_command
    assert all("python - <<'PY'" not in command for command in task.suggested_commands)
    assert len(task.suggested_commands) == 1

    sandbox = Sandbox(timeout_seconds=10)
    for command in task.suggested_commands:
        result = sandbox.run(command, tmp_path)
        assert result.exit_code == 0, result.stderr

    success = sandbox.run(task.success_command, tmp_path)
    assert success.exit_code == 0, success.stderr


def test_apply_semantic_policy_seed_uses_recent_redirects(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        trajectories_root=tmp_path / "trajectories",
        semantic_hub_root=tmp_path / "trajectories/semantic_hub",
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    config.ensure_directories()

    redirects = config.semantic_hub_root / "redirects"
    redirects.mkdir(parents=True, exist_ok=True)
    (redirects / "r1.json").write_text(
        json.dumps(
            {
                "reason_codes": ["weak_retention_outcome", "subsystem_reject_cooldown"],
                "required_new_parent": True,
                "source_subsystems": ["tooling"],
                "target_priority_families": ["integration", "repository"],
            }
        ),
        encoding="utf-8",
    )
    (redirects / "r2.json").write_text(
        json.dumps(
            {
                "reason_codes": ["no_yield_round"],
                "required_new_parent": True,
                "source_subsystems": ["tooling"],
                "target_priority_families": ["integration"],
            }
        ),
        encoding="utf-8",
    )

    seeded = module._apply_semantic_policy_seed(
        {
            "cycles": 3,
            "campaign_width": 2,
            "variant_width": 1,
            "adaptive_search": False,
            "task_limit": 64,
            "task_step_floor": 50,
            "priority_benchmark_families": ["project", "repository"],
            "tolbert_device": "cuda",
            "focus": "balanced",
            "liftoff": "never",
            "excluded_subsystems": [],
            "subsystem_cooldowns": {},
        },
        semantic_seed=module._semantic_policy_seed(config),
    )

    assert seeded["adaptive_search"] is True
    assert seeded["focus"] == "discovered_task_adaptation"
    assert seeded["excluded_subsystems"] == ["tooling"]
    assert seeded["subsystem_cooldowns"] == {"tooling": 2}
    assert seeded["priority_benchmark_families"] == ["integration", "repository"]
    assert seeded["variant_width"] == 1


def test_apply_semantic_policy_seed_preserves_bootstrap_minimum_fresh_proof_families():
    module = _load_script("run_unattended_campaign.py")

    seeded = module._apply_semantic_policy_seed(
        {
            "cycles": 3,
            "campaign_width": 2,
            "variant_width": 1,
            "adaptive_search": True,
            "task_limit": 128,
            "task_step_floor": 50,
            "priority_benchmark_families": ["integration", "project", "repository", "repo_chore"],
            "tolbert_device": "cuda",
            "focus": "balanced",
            "liftoff": "auto",
            "excluded_subsystems": [],
            "subsystem_cooldowns": {},
        },
        semantic_seed={
            "priority_benchmark_families": ["integration", "repository"],
        },
    )

    assert seeded["priority_benchmark_families"] == ["integration", "repository", "project", "repo_chore"]


def test_initial_round_policy_seeds_persistent_excluded_subsystems(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    args = SimpleNamespace(
        cycles=3,
        campaign_width=2,
        variant_width=1,
        adaptive_search=False,
        task_limit=128,
        task_step_floor=0,
        priority_benchmark_family=[],
        focus="balanced",
        liftoff="never",
        exclude_subsystem=["tolbert_model"],
    )

    policy = module._initial_round_policy(args, config)

    assert policy["persistent_excluded_subsystems"] == ["tolbert_model"]
    assert "tolbert_model" in policy["excluded_subsystems"]


def test_advance_subsystem_cooldowns_preserves_persistent_excluded_subsystems():
    module = _load_script("run_unattended_campaign.py")

    next_policy = module._advance_subsystem_cooldowns(
        {
            "persistent_excluded_subsystems": ["tolbert_model"],
            "excluded_subsystems": ["tolbert_model", "retrieval"],
            "subsystem_cooldowns": {"retrieval": 2},
        }
    )

    assert next_policy["persistent_excluded_subsystems"] == ["tolbert_model"]
    assert next_policy["excluded_subsystems"] == ["tolbert_model", "retrieval"]
    assert next_policy["subsystem_cooldowns"] == {"retrieval": 1}


def test_parse_cognitive_progress_fields_extracts_stage_payload():
    module = _load_script("run_unattended_campaign.py")

    payload = module._parse_cognitive_progress_fields(
        "[eval:campaign] phase=observe task 1/8 deployment_manifest_task "
        "family=project cognitive_stage=state_estimated step=1 subphase=world_model_initial"
    )

    assert payload["event"] == "cognitive_stage"
    assert payload["cognitive_stage"] == "state_estimated"
    assert payload["phase"] == "observe"
    assert payload["step_index"] == 1
    assert payload["step_subphase"] == "world_model_initial"
    assert payload["current_task"]["task_id"] == "deployment_manifest_task"


def test_child_progress_callback_emits_first_class_cognitive_stage_event(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {}
    round_payload = {}
    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=preview_candidate_eval task 1/8 deployment_manifest_task "
            "family=project cognitive_stage=memory_update_written step=2 verification_passed=1",
            "pid": 123,
            "timestamp": 1.0,
        }
    )

    lines = [json.loads(line) for line in event_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cognitive = [line for line in lines if line.get("event") == "cognitive_stage"]
    assert cognitive
    assert cognitive[-1]["cognitive_stage"] == "memory_update_written"
    assert report["active_child"]["current_cognitive_stage"]["cognitive_stage"] == "memory_update_written"
    assert report["active_child"]["current_task_progress_timeline"][-1]["cognitive_stage"] == "memory_update_written"


def test_child_progress_callback_clears_failed_verification_on_new_task(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {}
    round_payload = {}
    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 1/8 deployment_manifest_task "
            "family=project cognitive_stage=verification_result step=1 verification_passed=0",
            "pid": 123,
            "timestamp": 1.0,
        }
    )
    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 2/8 repo_sync_matrix_task "
            "family=repository cognitive_stage=memory_retrieved subphase=graph_memory",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert active_child["current_task"]["index"] == 2
    assert active_child["current_task_verification_passed"] is None
    assert active_child["current_cognitive_stage"]["cognitive_stage"] == "memory_retrieved"
    assert len(active_child["current_task_progress_timeline"]) == 1


def test_child_progress_callback_clears_failed_verification_on_new_step_same_task(tmp_path):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {}
    round_payload = {}
    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 3/128 semantic_open_world_task "
            "family=integration cognitive_stage=verification_result step=1 verification_passed=0",
            "pid": 123,
            "timestamp": 1.0,
        }
    )
    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 3/128 semantic_open_world_task "
            "family=integration cognitive_stage=llm_request step=3",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert active_child["current_task"]["task_id"] == "semantic_open_world_task"
    assert active_child["current_task"]["index"] == 3
    assert active_child["current_task_verification_passed"] is None
    assert active_child["current_cognitive_stage"]["cognitive_stage"] == "llm_request"
    assert active_child["current_cognitive_stage"]["step_index"] == 3


def test_child_progress_callback_preserves_live_task_over_ahead_of_pipe_future_task_snapshot(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {}
    round_payload = {}

    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "report_kind": "repeated_improvement_status",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "healthy",
                "decision_distance": "active",
                "detail": "active task 3/4 is progressing",
                "recorded_at": 3.0,
            },
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 3,
                    "total": 4,
                    "task_id": "semantic_open_world_20260411T225301327576Z_round_1_project",
                    "family": "project",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "transition_simulated",
                    "phase": "primary",
                    "step_index": 1,
                },
                "semantic_progress_state": {
                    "phase": "primary",
                    "phase_family": "active",
                    "progress_class": "healthy",
                    "decision_distance": "active",
                    "detail": "active task 3/4 is progressing",
                    "recorded_at": 3.0,
                },
            },
        },
    )

    callback = module._make_child_progress_callback(
        config=config,
        report=report,
        report_path=report_path,
        status_path=status_path,
        lock_path=None,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 1/4 semantic_open_world_20260411T225301327576Z_round_1_integration "
            "family=integration task_origin=semantic_hub family=integration cognitive_stage=llm_request step=2",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert active_child["current_task"]["task_id"] == "semantic_open_world_20260411T225301327576Z_round_1_integration"
    assert active_child["current_task"]["index"] == 1
    assert active_child.get("current_cognitive_stage", {}).get("cognitive_stage") == "llm_request"


def test_child_progress_callback_preserves_live_same_step_semantic_state_over_ahead_of_pipe_snapshot(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {}
    round_payload = {}

    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "report_kind": "repeated_improvement_status",
            "semantic_progress_state": {
                "phase": "primary",
                "phase_family": "active",
                "progress_class": "degraded",
                "decision_distance": "active",
                "detail": "active task 1/128 failed verification and is awaiting recovery",
                "recorded_at": 3.0,
            },
            "active_cycle_run": {
                "last_progress_phase": "primary",
                "current_task": {
                    "index": 1,
                    "total": 128,
                    "task_id": "semantic_open_world_task",
                    "family": "integration",
                    "phase": "primary",
                },
                "current_cognitive_stage": {
                    "cognitive_stage": "memory_update_written",
                    "phase": "primary",
                    "step_index": 2,
                    "verification_passed": False,
                },
                "current_task_progress_timeline": [
                    {
                        "cognitive_stage": "memory_update_written",
                        "phase": "primary",
                        "step_index": 2,
                        "verification_passed": False,
                    }
                ],
                "current_task_verification_passed": False,
            },
        },
    )

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=primary task 1/128 semantic_open_world_task "
            "family=integration cognitive_stage=llm_response step=2",
            "pid": 123,
            "timestamp": 1.0,
        }
    )
    callback(
        {
            "event": "adaptive_progress_state",
            "phase": "primary",
            "phase_family": "active",
            "status": "active",
            "progress_class": "healthy",
            "decision_distance": "active",
            "detail": "active task 1/128 is progressing",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert active_child["current_task"]["task_id"] == "semantic_open_world_task"
    assert active_child["current_cognitive_stage"]["cognitive_stage"] == "llm_response"
    assert active_child["current_cognitive_stage"]["step_index"] == 2
    assert active_child["current_task_verification_passed"] is None
    assert active_child["semantic_progress_state"]["progress_class"] == "healthy"
    assert active_child["semantic_progress_state"]["detail"] == "active task 1/128 is progressing"
    assert active_child["canonical_progress_state"]["progress_class"] == "healthy"
    assert "controller_intervention" not in report


def test_child_progress_callback_clears_stale_controller_intervention_after_progress_resumes(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {
        "active_child": {
            "controller_intervention": {
                "triggered": True,
                "reason_code": "productive_partial_preview_without_decision",
            }
        },
        "controller_intervention": {
            "triggered": True,
            "reason_code": "productive_partial_preview_without_decision",
        },
        "active_child_controller_intervention": {
            "triggered": True,
            "reason_code": "productive_partial_preview_without_decision",
        },
    }
    round_payload = {
        "controller_intervention": {
            "triggered": True,
            "reason_code": "productive_partial_preview_without_decision",
        }
    }

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mirrored_child_status_from_parent_status", lambda path: {})

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=generated_failure task 1/1 semantic_open_world_task_recovery "
            "family=integration source_task=semantic_open_world_task",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert "controller_intervention" not in active_child
    assert "controller_intervention" not in report
    assert "active_child_controller_intervention" not in report
    assert "controller_intervention" not in round_payload


def test_child_progress_callback_does_not_fire_observe_stall_intervention_when_task_progress_resumes(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    report = {"active_child": {}}
    round_payload = {}

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_mirrored_child_status_from_parent_status",
        lambda path: {
            "semantic_progress_state": {
                "phase": "observe",
                "phase_family": "observe",
                "progress_class": "stuck",
                "progress_silence_seconds": 361.0,
                "recorded_at": 1.0,
            },
            "current_task": {
                "index": 1,
                "total": 128,
                "task_id": "semantic_open_world_task",
                "family": "integration",
                "phase": "observe",
            },
        },
    )

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[eval:campaign] phase=observe task 10/128 semantic_open_world_task family=integration task_origin=semantic_hub",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    active_child = report["active_child"]
    assert active_child["current_task"]["index"] == 10
    assert active_child["last_task_progress_at"] == 2.0
    assert "controller_intervention" not in active_child
    assert "controller_intervention" not in report
    assert "active_child_controller_intervention" not in report
    assert "controller_intervention" not in round_payload


def test_child_progress_callback_does_not_clear_controller_intervention_on_still_running_heartbeat(
    monkeypatch, tmp_path
):
    module = _load_script("run_unattended_campaign.py")
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        candidate_artifacts_root=tmp_path / "candidates",
        run_checkpoints_dir=tmp_path / "checkpoints",
        unattended_workspace_snapshot_root=tmp_path / "snapshots",
        tolbert_supervised_datasets_dir=tmp_path / "tolbert_datasets",
    )
    report_path = tmp_path / "report.json"
    status_path = tmp_path / "status.json"
    event_log_path = tmp_path / "events.jsonl"
    intervention = {
        "triggered": True,
        "reason_code": "observe_progress_stall",
    }
    report = {
        "active_child": {"controller_intervention": dict(intervention)},
        "controller_intervention": dict(intervention),
        "active_child_controller_intervention": dict(intervention),
    }
    round_payload = {"controller_intervention": dict(intervention)}

    monkeypatch.setattr(module, "_append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_persist_report_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mirrored_child_status_from_parent_status", lambda path: {})

    callback = module._make_child_progress_callback(
        report_path=report_path,
        report=report,
        status_path=status_path,
        lock_path=None,
        config=config,
        round_payload=round_payload,
        round_index=1,
        phase="campaign",
        child_label="campaign_round_1",
        event_log_path=event_log_path,
        min_write_interval_seconds=0.0,
    )

    callback(
        {
            "event": "output",
            "line": "[repeated] child=campaign_round_1 still_running silence=120s",
            "pid": 123,
            "timestamp": 2.0,
        }
    )

    assert report["active_child"]["controller_intervention"]["reason_code"] == "observe_progress_stall"
    assert report["controller_intervention"]["reason_code"] == "observe_progress_stall"
    assert report["active_child_controller_intervention"]["reason_code"] == "observe_progress_stall"
    assert round_payload["controller_intervention"]["reason_code"] == "observe_progress_stall"
