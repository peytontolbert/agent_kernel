from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import os
import signal
import subprocess
import tempfile


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def spawn_process_group(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    text: bool = True,
    bufsize: int = 1,
) -> subprocess.Popen[str]:
    kwargs = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": text,
        "bufsize": bufsize,
        "env": env,
    }
    try:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            **kwargs,
        )
    except TypeError:
        return subprocess.Popen(cmd, **kwargs)


def terminate_process_tree(process: subprocess.Popen[str], *, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    pid = int(getattr(process, "pid", 0) or 0)
    sent_group_signal = False
    if pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            sent_group_signal = True
        except OSError:
            sent_group_signal = False
    if not sent_group_signal:
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=max(0.0, float(grace_seconds)))
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    if pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            sent_group_signal = True
        except OSError:
            sent_group_signal = False
    if not sent_group_signal:
        try:
            process.kill()
        except OSError:
            return
    try:
        process.wait(timeout=1.0)
    except (subprocess.TimeoutExpired, OSError):
        return


def install_termination_handlers(handler: Callable[[int], None]) -> Callable[[], None]:
    managed_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        managed_signals.append(signal.SIGHUP)
    previous: dict[int, object] = {}

    def _signal_handler(signum, _frame) -> None:
        handler(int(signum))

    for signum in managed_signals:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)

    def _restore() -> None:
        for signum, prior in previous.items():
            signal.signal(signum, prior)

    return _restore
