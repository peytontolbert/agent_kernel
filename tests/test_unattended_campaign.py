from pathlib import Path
import importlib.util
import json
from io import StringIO
import os
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.unattended_controller import default_controller_state, update_controller_state


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
                },
                "repo_sandbox": {
                    "total": 1,
                    "success_rate": 1.0,
                    "unsafe_ambiguous_rate": 0.0,
                    "hidden_side_effect_risk_rate": 0.0,
                    "false_pass_risk_rate": 0.0,
                    "clean_success_streak": 1,
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

    monkeypatch.setattr(module, "_write_report", lambda path, data: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(module, "_emergency_persist_path", lambda path: emergency_path)

    result = module._persist_report_state(report_path, payload, status_path=status_path, lock_path=None)

    assert result["emergency_report_path"] == str(emergency_path)
    assert emergency_path.exists()
    emergency_payload = json.loads(emergency_path.read_text(encoding="utf-8"))
    assert emergency_payload["status"] == "running"
    assert emergency_payload["emergency_persist"]["original_report_path"] == str(report_path)


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
    assert rationale["planner_pressure"]["missing_required_families"] == ["project", "repository", "integration"]
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["priority_benchmark_families"] == ["project", "repository", "integration"]


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
    assert rationale["planner_pressure"]["priority_families_with_retained_gain"] == ["project"]
    assert rationale["planner_pressure"]["priority_families_without_signal"] == ["integration"]
    assert rationale["planner_pressure"]["priority_families_under_sampled"] == ["repository", "integration"]
    assert rationale["planner_pressure"]["priority_families_low_return"] == []
    assert next_policy["adaptive_search"] is True
    assert next_policy["focus"] == "discovered_task_adaptation"
    assert next_policy["priority_benchmark_families"] == ["repository", "integration", "project"]


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
    assert "priority_family_under_sampled" not in rationale["reason_codes"]
    assert rationale["planner_pressure"]["priority_families_low_return"] == ["project"]
    assert next_policy["priority_benchmark_families"] == ["repository", "integration", "project"]


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
    assert next_policy["priority_benchmark_families"] == ["project", "repository", "integration"]


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
