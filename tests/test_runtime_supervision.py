from pathlib import Path
import json
import os
import signal
import subprocess

import pytest

import agent_kernel.ops.runtime_supervision as runtime_supervision
from agent_kernel.config import KernelConfig
from agent_kernel.ops.runtime_supervision import (
    append_jsonl,
    atomic_copy_file,
    atomic_write_json,
    atomic_write_text,
    terminate_process_tree,
)


def test_atomic_write_text_replaces_existing_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"
    leftovers = [path for path in tmp_path.iterdir() if path.name != "state.json"]
    assert leftovers == []


def test_terminate_process_tree_uses_process_group_and_escalates(monkeypatch):
    seen_signals: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(self):
            self.pid = 4321
            self._poll = None
            self.wait_calls = 0

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            self._poll = -9
            return -9

        def terminate(self):
            raise AssertionError("terminate should not be used when killpg works")

        def kill(self):
            raise AssertionError("kill should not be used when killpg works")

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pgid, sig: seen_signals.append((pgid, sig)),
    )

    process = FakeProcess()
    terminate_process_tree(process, grace_seconds=0.01)

    assert seen_signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert process.poll() == -9


def test_atomic_write_json_surfaces_governance_failures_for_managed_paths(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()

    def explode(path: Path, *, config=None) -> None:
        raise RuntimeError("governance failed")

    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", explode)

    target = config.improvement_reports_dir / "report.json"
    with pytest.raises(RuntimeError, match="governance failed"):
        atomic_write_json(target, {"ok": True}, config=config)

    assert target.exists()


def test_atomic_write_json_uses_explicit_config_instead_of_env_lookup(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()

    def explode_env_lookup():
        raise AssertionError("env-derived config should not be consulted when explicit config is provided")

    seen: dict[str, object] = {}

    def capture(path: Path, *, config=None) -> None:
        seen["path"] = path
        seen["config"] = config

    monkeypatch.setattr(runtime_supervision, "current_storage_governance_config", explode_env_lookup)
    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", capture)

    target = config.improvement_reports_dir / "report.json"
    atomic_write_json(target, {"ok": True}, config=config)

    assert seen == {"path": target, "config": config}


def test_atomic_write_json_can_skip_governance(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()

    def explode(path: Path, *, config=None) -> None:
        raise AssertionError(f"governance should be skipped for {path}")

    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", explode)

    target = config.improvement_reports_dir / "report.json"
    atomic_write_json(target, {"ok": True}, config=config, govern_storage=False)

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_maybe_govern_improvement_exports_limits_checkpoint_scope(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
        run_reports_dir=tmp_path / "run_reports",
        run_checkpoints_dir=tmp_path / "run_checkpoints",
    )
    config.ensure_directories()
    target = config.run_checkpoints_dir / "job.json"
    target.write_text("{}", encoding="utf-8")
    config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.unattended_trust_ledger_path.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_govern(
        runtime_config: KernelConfig,
        *,
        preserve_paths=(),
        include_cycle_exports=True,
        include_report_exports=True,
        include_candidate_exports=True,
        include_run_reports=True,
        include_run_checkpoints=True,
    ) -> dict[str, object]:
        captured["config"] = runtime_config
        captured["preserve_paths"] = preserve_paths
        captured["include_cycle_exports"] = include_cycle_exports
        captured["include_report_exports"] = include_report_exports
        captured["include_candidate_exports"] = include_candidate_exports
        captured["include_run_reports"] = include_run_reports
        captured["include_run_checkpoints"] = include_run_checkpoints
        return {}

    monkeypatch.setattr(runtime_supervision, "govern_improvement_export_storage", fake_govern)

    runtime_supervision._maybe_govern_improvement_exports(target, config=config)

    assert captured["config"] == config
    assert captured["preserve_paths"] == (target, config.unattended_trust_ledger_path)
    assert captured["include_cycle_exports"] is False
    assert captured["include_report_exports"] is False
    assert captured["include_candidate_exports"] is False
    assert captured["include_run_reports"] is False
    assert captured["include_run_checkpoints"] is True


def test_maybe_govern_improvement_exports_skips_strategy_memory_snapshots_under_improvement_root(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
        strategy_memory_snapshots_path=tmp_path / "improvement" / "strategy_memory" / "snapshots.json",
    )
    config.ensure_directories()
    target = config.strategy_memory_snapshots_path
    target.write_text("{}", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_govern(
        runtime_config: KernelConfig,
        *,
        preserve_paths=(),
        include_cycle_exports=True,
        include_report_exports=True,
        include_candidate_exports=True,
        include_run_reports=True,
        include_run_checkpoints=True,
    ) -> dict[str, object]:
        calls.append(
            {
                "config": runtime_config,
                "preserve_paths": preserve_paths,
                "include_cycle_exports": include_cycle_exports,
                "include_report_exports": include_report_exports,
                "include_candidate_exports": include_candidate_exports,
                "include_run_reports": include_run_reports,
                "include_run_checkpoints": include_run_checkpoints,
            }
        )
        return {}

    monkeypatch.setattr(runtime_supervision, "govern_improvement_export_storage", fake_govern)

    runtime_supervision._maybe_govern_improvement_exports(target, config=config)

    assert calls == []


def test_maybe_govern_improvement_exports_matches_cycle_export_siblings_only(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()
    target = config.improvement_cycles_path.parent / "cycles.archive.jsonl"
    target.write_text("{}", encoding="utf-8")
    config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.unattended_trust_ledger_path.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_govern(
        runtime_config: KernelConfig,
        *,
        preserve_paths=(),
        include_cycle_exports=True,
        include_report_exports=True,
        include_candidate_exports=True,
        include_run_reports=True,
        include_run_checkpoints=True,
    ) -> dict[str, object]:
        captured["config"] = runtime_config
        captured["preserve_paths"] = preserve_paths
        captured["include_cycle_exports"] = include_cycle_exports
        captured["include_report_exports"] = include_report_exports
        captured["include_candidate_exports"] = include_candidate_exports
        captured["include_run_reports"] = include_run_reports
        captured["include_run_checkpoints"] = include_run_checkpoints
        return {}

    monkeypatch.setattr(runtime_supervision, "govern_improvement_export_storage", fake_govern)

    runtime_supervision._maybe_govern_improvement_exports(target, config=config)

    assert captured["config"] == config
    assert captured["preserve_paths"] == (target, config.unattended_trust_ledger_path)
    assert captured["include_cycle_exports"] is True
    assert captured["include_report_exports"] is False
    assert captured["include_candidate_exports"] is False
    assert captured["include_run_reports"] is False
    assert captured["include_run_checkpoints"] is False


def test_append_jsonl_surfaces_governance_failures_for_managed_paths(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()

    def explode(path: Path, *, config=None) -> None:
        raise RuntimeError("governance failed")

    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", explode)

    target = config.improvement_reports_dir / "history.jsonl"
    with pytest.raises(RuntimeError, match="governance failed"):
        append_jsonl(target, {"round": 1}, config=config)

    assert target.exists()


def test_atomic_copy_file_surfaces_governance_failures_for_managed_paths(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}', encoding="utf-8")

    def explode(path: Path, *, config=None) -> None:
        raise RuntimeError("governance failed")

    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", explode)

    target = config.improvement_reports_dir / "copied.json"
    with pytest.raises(RuntimeError, match="governance failed"):
        atomic_copy_file(source, target, config=config)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_copy_file_can_skip_governance(tmp_path, monkeypatch):
    config = KernelConfig(
        improvement_reports_dir=tmp_path / "reports",
        improvement_cycles_path=tmp_path / "improvement" / "cycles.jsonl",
        candidate_artifacts_root=tmp_path / "candidates",
    )
    config.ensure_directories()
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}', encoding="utf-8")

    def explode(path: Path, *, config=None) -> None:
        raise AssertionError(f"governance should be skipped for {path}")

    monkeypatch.setattr(runtime_supervision, "_maybe_govern_improvement_exports", explode)

    target = config.improvement_reports_dir / "copied.json"
    atomic_copy_file(source, target, config=config, govern_storage=False)

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
