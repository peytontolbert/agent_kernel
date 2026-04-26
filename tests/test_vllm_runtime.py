from __future__ import annotations

from urllib import error as url_error

import agent_kernel.ops.loop_runtime_support as loop_runtime_support
import agent_kernel.ops.preflight as preflight
import agent_kernel.ops.vllm_runtime as vllm_runtime
from agent_kernel.config import KernelConfig
from agent_kernel.ops.vllm_runtime import VLLMRuntimeStatus


class _FakeHTTPResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def _vllm_config(tmp_path, **overrides) -> KernelConfig:
    base = {
        "provider": "vllm",
        "model_name": "Qwen/Qwen3.5-9B",
        "vllm_host": "http://127.0.0.1:8000",
        "use_tolbert_context": False,
        "workspace_root": tmp_path / "workspace",
        "trajectories_root": tmp_path / "trajectories",
        "run_reports_dir": tmp_path / "reports",
        "runtime_database_path": tmp_path / "var" / "runtime" / "agentkernel.sqlite3",
        "vllm_runtime_log_path": tmp_path / "var" / "runtime" / "vllm_runtime.log",
        "vllm_runtime_state_path": tmp_path / "var" / "runtime" / "vllm_runtime.json",
        "vllm_runtime_lock_path": tmp_path / "var" / "runtime" / "vllm_runtime.lock",
        "vllm_python_bin": "/usr/bin/python3",
        "vllm_language_model_only": False,
        "vllm_autostart_poll_interval_seconds": 0.0,
    }
    base.update(overrides)
    return KernelConfig(**base)


def test_ensure_vllm_runtime_autostarts_local_runtime(tmp_path, monkeypatch):
    config = _vllm_config(tmp_path)
    seen: dict[str, object] = {"command": None, "spawn_count": 0}
    probe_attempt = {"count": 0}

    def fake_urlopen(req, timeout=0):
        del req, timeout
        probe_attempt["count"] += 1
        if probe_attempt["count"] < 3:
            raise url_error.URLError("connection refused")
        return _FakeHTTPResponse(200)

    class FakeProcess:
        pid = 43210

    def fake_popen(cmd, **kwargs):
        seen["command"] = list(cmd)
        seen["spawn_count"] = int(seen["spawn_count"] or 0) + 1
        del kwargs
        return FakeProcess()

    monkeypatch.setattr(vllm_runtime, "_pid_running", lambda pid: int(pid) == 43210)

    status = vllm_runtime.ensure_vllm_runtime(
        config,
        opener=fake_urlopen,
        popen_factory=fake_popen,
        sleep_fn=lambda _: None,
        monotonic_fn=iter([0.0, 0.0, 0.1]).__next__,
    )

    assert status.ready is True
    assert status.status == "autostarted"
    assert status.started is True
    assert status.pid == 43210
    assert "vllm autostarted and is now ready" in status.detail
    assert seen["spawn_count"] == 1
    assert seen["command"][:3] == ["/usr/bin/python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--enforce-eager" in seen["command"]
    assert config.vllm_runtime_state_path.exists()


def test_ensure_vllm_runtime_does_not_autostart_remote_host(tmp_path):
    config = _vllm_config(
        tmp_path,
        vllm_host="http://10.0.0.5:8000",
    )

    status = vllm_runtime.ensure_vllm_runtime(
        config,
        opener=lambda req, timeout=0: (_ for _ in ()).throw(url_error.URLError("connection refused")),
        popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn for remote host")),
        sleep_fn=lambda _: None,
        monotonic_fn=iter([0.0]).__next__,
    )

    assert status.ready is False
    assert "local hosts" in status.detail


def test_ensure_vllm_runtime_fails_fast_when_spawned_process_exits(tmp_path):
    config = _vllm_config(tmp_path)

    def fake_urlopen(req, timeout=0):
        del req, timeout
        raise url_error.URLError("connection refused")

    class FakeProcess:
        pid = 43211

        def poll(self):
            return 1

    status = vllm_runtime.ensure_vllm_runtime(
        config,
        opener=fake_urlopen,
        popen_factory=lambda *args, **kwargs: FakeProcess(),
        sleep_fn=lambda _: None,
        monotonic_fn=iter([0.0, 0.0]).__next__,
    )

    assert status.ready is False
    assert status.status == "failed"
    assert status.pid == 43211
    assert "exited before readiness" in status.detail
    assert "returncode=1" in status.detail


def test_provider_health_check_uses_vllm_runtime_manager(tmp_path, monkeypatch):
    config = _vllm_config(tmp_path)
    monkeypatch.setattr(
        preflight,
        "ensure_vllm_runtime",
        lambda config, opener=None: VLLMRuntimeStatus(
            ready=True,
            status="autostarted",
            detail="vllm autostarted and is now ready",
            started=True,
            pid=99,
        ),
    )

    check = preflight.provider_health_check(config)

    assert check.passed is True
    assert check.name == "provider_health"
    assert check.detail == "vllm autostarted and is now ready"


def test_provider_health_check_uses_model_stack_healthz(tmp_path):
    config = KernelConfig(
        provider="model_stack",
        model_stack_host="http://127.0.0.1:8001",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        runtime_database_path=tmp_path / "var" / "runtime" / "agentkernel.sqlite3",
    )
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _FakeHTTPResponse(200)

    check = preflight.provider_health_check(config, urlopen=fake_urlopen)

    assert check.name == "provider_health"
    assert check.passed is True
    assert check.detail == "model_stack responded with status 200"
    assert seen == {"url": "http://127.0.0.1:8001/healthz", "timeout": 5}


def test_build_default_policy_requires_vllm_runtime_ready(tmp_path, monkeypatch):
    config = _vllm_config(tmp_path)
    seen: dict[str, object] = {}

    class FakeSkillLibrary:
        def __init__(self, items):
            self.items = list(items)

        @classmethod
        def from_path(cls, path, *, min_quality):
            del path, min_quality
            return cls([])

    class FakePolicy:
        def __init__(self, client, *, context_provider, skill_library, config):
            seen["client"] = client
            seen["context_provider"] = context_provider
            seen["skill_library"] = skill_library
            seen["config"] = config

    class FakeVLLMClient:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = dict(kwargs)

    monkeypatch.setattr(
        loop_runtime_support,
        "ensure_vllm_runtime",
        lambda config: VLLMRuntimeStatus(
            ready=True,
            status="autostarted",
            detail="ok",
            started=True,
            pid=123,
        ),
    )

    policy = loop_runtime_support.build_default_policy(
        config=config,
        repo_root=tmp_path,
        skill_library_cls=FakeSkillLibrary,
        llm_decision_policy_cls=FakePolicy,
        context_provider_factory=lambda **kwargs: None,
        ollama_client_cls=lambda **kwargs: None,
        vllm_client_cls=FakeVLLMClient,
        model_stack_client_cls=lambda **kwargs: None,
        mock_client_factory=lambda: None,
        hybrid_client_factory=lambda: None,
    )

    assert isinstance(policy, FakePolicy)
    assert seen["client_kwargs"]["host"] == config.vllm_host
    assert seen["client_kwargs"]["model_name"] == config.model_name
    assert seen["config"] is config
