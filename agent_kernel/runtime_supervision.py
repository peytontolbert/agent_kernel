from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import os
import signal
import shutil
import subprocess
import tempfile

from .config import KernelConfig, current_storage_governance_config
from .export_governance import govern_improvement_export_storage


def _supports_export_governance(config: object) -> bool:
    required_fields = (
        "improvement_cycles_path",
        "improvement_reports_dir",
        "candidate_artifacts_root",
        "run_reports_dir",
        "run_checkpoints_dir",
        "unattended_trust_ledger_path",
        "storage_max_cycle_export_records",
        "storage_keep_cycle_export_files",
        "storage_max_report_history_records",
        "storage_keep_report_export_files",
        "storage_keep_candidate_export_dirs",
        "storage_keep_namespace_candidate_dirs",
        "storage_keep_run_report_files",
        "storage_keep_run_checkpoint_files",
    )
    return all(hasattr(config, field) for field in required_fields)


def _maybe_govern_improvement_exports(path: Path, *, config: KernelConfig | None = None) -> None:
    resolved_config = config or current_storage_governance_config()
    if not _supports_export_governance(resolved_config):
        return
    target = path.resolve()
    root_matches: dict[str, Path] = {}

    improvement_cycles_path = getattr(resolved_config, "improvement_cycles_path", None)
    if improvement_cycles_path is not None:
        root_matches["cycle_exports"] = Path(improvement_cycles_path).parent.resolve()

    improvement_reports_dir = getattr(resolved_config, "improvement_reports_dir", None)
    if improvement_reports_dir is not None:
        root_matches["report_exports"] = Path(improvement_reports_dir).resolve()

    candidate_artifacts_root = getattr(resolved_config, "candidate_artifacts_root", None)
    if candidate_artifacts_root is not None:
        root_matches["candidate_exports"] = Path(candidate_artifacts_root).resolve()

    run_reports_dir = getattr(resolved_config, "run_reports_dir", None)
    if run_reports_dir is not None:
        root_matches["run_reports"] = Path(run_reports_dir).resolve()

    run_checkpoints_dir = getattr(resolved_config, "run_checkpoints_dir", None)
    if run_checkpoints_dir is not None:
        root_matches["run_checkpoints"] = Path(run_checkpoints_dir).resolve()

    matched_roots = {
        name
        for name, root in root_matches.items()
        if root == target or root in target.parents
    }
    if not matched_roots:
        return
    unattended_trust_ledger_path = getattr(resolved_config, "unattended_trust_ledger_path", None)
    preserve_paths = (path,)
    if unattended_trust_ledger_path is not None:
        preserve_paths = (path, Path(unattended_trust_ledger_path))
    govern_improvement_export_storage(
        resolved_config,
        preserve_paths=preserve_paths,
        include_cycle_exports="cycle_exports" in matched_roots,
        include_report_exports="report_exports" in matched_roots,
        include_candidate_exports="candidate_exports" in matched_roots,
        include_run_reports="run_reports" in matched_roots,
        include_run_checkpoints="run_checkpoints" in matched_roots,
    )


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


def atomic_write_json(
    path: Path,
    payload: dict[str, object],
    *,
    config: KernelConfig | None = None,
    govern_storage: bool = True,
) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")
    if govern_storage:
        _maybe_govern_improvement_exports(path, config=config)


def append_jsonl(
    path: Path,
    payload: dict[str, object],
    *,
    config: KernelConfig | None = None,
    govern_storage: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if govern_storage:
        _maybe_govern_improvement_exports(path, config=config)


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
    *,
    config: KernelConfig | None = None,
    govern_storage: bool = True,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=str(destination_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with source_path.open("rb") as source_handle, os.fdopen(fd, "wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(tmp_path, destination_path)
        try:
            directory_fd = os.open(str(destination_path.parent), os.O_RDONLY)
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
    if govern_storage:
        _maybe_govern_improvement_exports(destination_path, config=config)


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
