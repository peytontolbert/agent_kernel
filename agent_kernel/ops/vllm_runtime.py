from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Callable
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from ..config import KernelConfig
from .runtime_supervision import atomic_write_json

_LOCAL_VLLM_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(slots=True)
class VLLMRuntimeStatus:
    ready: bool
    status: str
    detail: str
    started: bool = False
    pid: int = 0
    log_path: str = ""
    state_path: str = ""
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_vllm_runtime(
    config: KernelConfig,
    *,
    opener: Callable[..., Any] | None = None,
) -> VLLMRuntimeStatus:
    request_open = opener or url_request.urlopen
    headers: dict[str, str] = {}
    if config.vllm_api_key.strip():
        headers["Authorization"] = f"Bearer {config.vllm_api_key.strip()}"
    req = url_request.Request(
        url=f"{config.vllm_host.rstrip('/')}/v1/models",
        headers=headers,
        method="GET",
    )
    try:
        with request_open(req, timeout=5) as response:
            status = int(getattr(response, "status", 200) or 200)
        return VLLMRuntimeStatus(
            ready=True,
            status="ready",
            detail=f"vllm responded with status {status}",
            log_path=str(config.vllm_runtime_log_path),
            state_path=str(config.vllm_runtime_state_path),
        )
    except (TimeoutError, url_error.URLError, OSError) as exc:
        return VLLMRuntimeStatus(
            ready=False,
            status="unavailable",
            detail=f"vllm probe failed: {exc}",
            log_path=str(config.vllm_runtime_log_path),
            state_path=str(config.vllm_runtime_state_path),
        )


def ensure_vllm_runtime(
    config: KernelConfig,
    *,
    opener: Callable[..., Any] | None = None,
    popen_factory: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> VLLMRuntimeStatus:
    if config.normalized_provider() != "vllm":
        return VLLMRuntimeStatus(
            ready=True,
            status="skipped",
            detail=f"provider {config.normalized_provider()} does not require vllm runtime",
            log_path=str(config.vllm_runtime_log_path),
            state_path=str(config.vllm_runtime_state_path),
        )
    initial = probe_vllm_runtime(config, opener=opener)
    if initial.ready:
        return initial
    if not bool(config.vllm_autostart):
        initial.detail = f"{initial.detail}; vllm autostart disabled"
        return initial
    autostart_detail = _autostart_support_detail(config)
    if autostart_detail:
        initial.detail = f"{initial.detail}; {autostart_detail}"
        return initial
    config.vllm_runtime_lock_path.parent.mkdir(parents=True, exist_ok=True)
    config.vllm_runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
    config.vllm_runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.vllm_runtime_lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        second = probe_vllm_runtime(config, opener=opener)
        if second.ready:
            return second
        state = _read_runtime_state(config.vllm_runtime_state_path)
        pid = int(state.get("pid", 0) or 0)
        command = [str(value) for value in state.get("command", []) if str(value).strip()]
        started = False
        process_handle: Any | None = None
        if pid <= 0 or not _pid_running(pid):
            command = build_vllm_start_command(config)
            process_handle = _spawn_vllm_runtime(
                config,
                command=command,
                popen_factory=popen_factory,
            )
            pid = int(getattr(process_handle, "pid", 0) or 0)
            started = True
            _write_runtime_state(
                config,
                pid=pid,
                command=command,
                status="starting",
            )
        deadline = monotonic_fn() + max(1, int(config.vllm_autostart_timeout_seconds))
        last_probe = second
        while monotonic_fn() < deadline:
            if process_handle is not None:
                poll = getattr(process_handle, "poll", None)
                if callable(poll):
                    returncode = poll()
                    if returncode is not None:
                        log_tail = _tail_file(config.vllm_runtime_log_path)
                        detail = (
                            "vllm process exited before readiness "
                            f"(pid={pid} returncode={returncode} log={config.vllm_runtime_log_path})"
                        )
                        if log_tail:
                            detail = f"{detail}: {log_tail}"
                        failed = VLLMRuntimeStatus(
                            ready=False,
                            status="failed",
                            detail=detail,
                            started=started,
                            pid=pid,
                            log_path=str(config.vllm_runtime_log_path),
                            state_path=str(config.vllm_runtime_state_path),
                            command=list(command),
                        )
                        _write_runtime_state(
                            config,
                            pid=pid,
                            command=command,
                            status="failed",
                            detail=failed.detail,
                        )
                        return failed
            last_probe = probe_vllm_runtime(config, opener=opener)
            if last_probe.ready:
                _write_runtime_state(
                    config,
                    pid=pid,
                    command=command,
                    status="ready",
                )
                detail_prefix = "vllm autostarted and is now ready" if started else "vllm runtime became ready"
                last_probe.status = "autostarted" if started else "ready_after_wait"
                last_probe.started = started
                last_probe.pid = pid
                last_probe.command = list(command)
                last_probe.detail = (
                    f"{detail_prefix}: {last_probe.detail} "
                    f"(pid={pid} log={config.vllm_runtime_log_path})"
                )
                return last_probe
            if pid > 0 and not _pid_running(pid):
                log_tail = _tail_file(config.vllm_runtime_log_path)
                detail = (
                    "vllm process exited before readiness "
                    f"(pid={pid} log={config.vllm_runtime_log_path})"
                )
                if log_tail:
                    detail = f"{detail}: {log_tail}"
                failed = VLLMRuntimeStatus(
                    ready=False,
                    status="failed",
                    detail=detail,
                    started=started,
                    pid=pid,
                    log_path=str(config.vllm_runtime_log_path),
                    state_path=str(config.vllm_runtime_state_path),
                    command=list(command),
                )
                _write_runtime_state(
                    config,
                    pid=pid,
                    command=command,
                    status="failed",
                    detail=failed.detail,
                )
                return failed
            sleep_fn(max(0.0, float(config.vllm_autostart_poll_interval_seconds)))
        timeout_detail = (
            "timed out waiting for vllm readiness "
            f"(pid={pid} log={config.vllm_runtime_log_path})"
        )
        log_tail = _tail_file(config.vllm_runtime_log_path)
        if log_tail:
            timeout_detail = f"{timeout_detail}: {log_tail}"
        failed = VLLMRuntimeStatus(
            ready=False,
            status="timeout",
            detail=timeout_detail,
            started=started,
            pid=pid,
            log_path=str(config.vllm_runtime_log_path),
            state_path=str(config.vllm_runtime_state_path),
            command=list(command),
        )
        _write_runtime_state(
            config,
            pid=pid,
            command=command,
            status="timeout",
            detail=failed.detail,
        )
        return failed


def build_vllm_start_command(config: KernelConfig) -> list[str]:
    explicit = str(config.vllm_start_command).strip()
    if explicit:
        return shlex.split(explicit)
    bind_host, bind_port = _vllm_bind_host_and_port(config)
    command = [
        str(config.vllm_python_bin).strip(),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config.model_name).strip(),
        "--host",
        bind_host,
        "--port",
        str(bind_port),
    ]
    if int(config.vllm_tensor_parallel_size) > 1:
        command.extend(["--tensor-parallel-size", str(int(config.vllm_tensor_parallel_size))])
    if int(config.vllm_max_model_len) > 0:
        command.extend(["--max-model-len", str(int(config.vllm_max_model_len))])
    if float(config.vllm_gpu_memory_utilization) > 0.0:
        command.extend(
            [
                "--gpu-memory-utilization",
                f"{float(config.vllm_gpu_memory_utilization):g}",
            ]
        )
    if bool(config.vllm_enforce_eager):
        command.append("--enforce-eager")
    reasoning_parser = str(config.vllm_reasoning_parser).strip()
    if not reasoning_parser and "qwen" in str(config.model_name).strip().lower():
        reasoning_parser = "qwen3"
    if reasoning_parser:
        command.extend(["--reasoning-parser", reasoning_parser])
    if bool(config.vllm_language_model_only):
        command.append("--language-model-only")
    if config.vllm_api_key.strip():
        command.extend(["--api-key", config.vllm_api_key.strip()])
    extra_args = str(config.vllm_start_extra_args).strip()
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def _spawn_vllm_runtime(
    config: KernelConfig,
    *,
    command: list[str],
    popen_factory: Callable[..., Any] | None = None,
) -> Any:
    env = dict(os.environ)
    env.update(config.to_env())
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runtime_log = config.vllm_runtime_log_path.open("ab")
    spawn = popen_factory or subprocess.Popen
    process = spawn(
        command,
        stdin=subprocess.DEVNULL,
        stdout=runtime_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    runtime_log.close()
    return process


def _write_runtime_state(
    config: KernelConfig,
    *,
    pid: int,
    command: list[str],
    status: str,
    detail: str = "",
) -> None:
    atomic_write_json(
        config.vllm_runtime_state_path,
        {
            "status": str(status).strip(),
            "pid": int(pid),
            "command": list(command),
            "host": str(config.vllm_host).strip(),
            "model_name": str(config.model_name).strip(),
            "log_path": str(config.vllm_runtime_log_path),
            "detail": str(detail).strip(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
        config=config,
        govern_storage=False,
    )


def _read_runtime_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _tail_file(path: Path, *, max_chars: int = 400) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    normalized = " ".join(line.strip() for line in content.splitlines() if line.strip())
    if len(normalized) > max_chars:
        normalized = normalized[-max_chars:].lstrip()
    return normalized


def _autostart_support_detail(config: KernelConfig) -> str:
    parsed = _parsed_vllm_host(config)
    scheme = parsed.scheme or "http"
    host = (parsed.hostname or "").strip().lower()
    if scheme != "http":
        return f"vllm autostart only supports local http hosts; configured scheme={scheme!r}"
    if host not in _LOCAL_VLLM_HOSTS:
        return f"vllm autostart only supports local hosts; configured host={host!r}"
    return ""


def _vllm_bind_host_and_port(config: KernelConfig) -> tuple[str, int]:
    parsed = _parsed_vllm_host(config)
    host = (parsed.hostname or "127.0.0.1").strip() or "127.0.0.1"
    port = int(parsed.port or 8000)
    return host, port


def _parsed_vllm_host(config: KernelConfig):
    normalized = str(config.vllm_host).strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return url_parse.urlparse(normalized)
