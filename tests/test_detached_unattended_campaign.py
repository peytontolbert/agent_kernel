from pathlib import Path
import importlib.util
import json
import subprocess
import sys


def _load_script():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_detached_unattended_campaign.py"
    spec = importlib.util.spec_from_file_location("run_detached_unattended_campaign", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_writes_control_and_isolates_improvement_reports_dir(tmp_path, monkeypatch):
    module = _load_script()
    seen: dict[str, object] = {}

    class DummyProcess:
        pid = 43210

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["cwd"] = kwargs.get("cwd")
        seen["env"] = dict(kwargs.get("env", {}))
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_detached_unattended_campaign.py",
            "start",
            "--run-root",
            str(tmp_path / "run"),
            "--env",
            "AGENT_KERNEL_PROVIDER=vllm",
            "--",
            "--rounds",
            "1",
        ],
    )

    module.main()

    control = json.loads((tmp_path / "run" / "control.json").read_text(encoding="utf-8"))
    assert control["pid"] == 43210
    assert control["status"] == "running"
    assert control["status_path"].endswith("reports/unattended_campaign.status.json")
    assert control["improvement_reports_dir"].endswith("run/improvement_reports")
    assert seen["env"]["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"].endswith("run/improvement_reports")
    assert seen["env"]["AGENT_KERNEL_PROVIDER"] == "vllm"
    assert "--report-path" in seen["cmd"]
    assert "--status-path" in seen["cmd"]
    assert "--event-log-path" in seen["cmd"]
    assert "--" not in seen["cmd"]
    assert "--rounds" in seen["cmd"]


def test_status_missing_control_returns_not_found(tmp_path, monkeypatch, capsys):
    module = _load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_detached_unattended_campaign.py",
            "status",
            "--run-root",
            str(tmp_path / "missing-run"),
        ],
    )

    code = module.main()
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_found"
    assert payload["reason"] == "missing_or_invalid_control_file"
