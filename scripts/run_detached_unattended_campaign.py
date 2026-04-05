from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _control_path(run_root: Path) -> Path:
    return run_root / "control.json"


def _default_paths(run_root: Path) -> dict[str, Path]:
    reports_dir = run_root / "reports"
    return {
        "reports_dir": reports_dir,
        "report_path": reports_dir / "unattended_campaign.json",
        "status_path": reports_dir / "unattended_campaign.status.json",
        "event_log_path": reports_dir / "unattended_campaign.events.jsonl",
        "stdout_log_path": reports_dir / "unattended_campaign.stdout.log",
        "improvement_reports_dir": run_root / "improvement_reports",
    }


def _load_control(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (IsADirectoryError, OSError, json.JSONDecodeError):
        return {}


def _write_control(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        token = str(raw).strip()
        if not token or "=" not in token:
            raise SystemExit(f"invalid --env override: {raw!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid --env override: {raw!r}")
        overrides[key] = value
    return overrides


def _signal_value(name: str) -> int:
    token = str(name).strip().upper()
    if not token:
        return signal.SIGTERM
    if not token.startswith("SIG"):
        token = f"SIG{token}"
    try:
        return int(getattr(signal, token))
    except AttributeError as exc:
        raise SystemExit(f"unknown signal: {name}") from exc


def _start(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    paths = _default_paths(run_root)
    control_path = _control_path(run_root)
    existing = _load_control(control_path)
    existing_pid = int(existing.get("pid", 0) or 0)
    if _pid_alive(existing_pid):
        print(json.dumps({"started": False, "reason": "already_running", **existing}, indent=2))
        return 0

    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    repo_root = _repo_root()
    run_script = repo_root / "scripts" / "run_unattended_campaign.py"
    env = dict(os.environ)
    env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"] = str(paths["improvement_reports_dir"])
    env.update(_parse_env_overrides(args.env))

    campaign_args = list(args.campaign_args)
    if campaign_args and campaign_args[0] == "--":
        campaign_args = campaign_args[1:]

    cmd = [
        sys.executable,
        "-u",
        str(run_script),
        *campaign_args,
        "--report-path",
        str(paths["report_path"]),
        "--status-path",
        str(paths["status_path"]),
        "--event-log-path",
        str(paths["event_log_path"]),
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    with paths["stdout_log_path"].open("ab") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    control = {
        "run_root": str(run_root),
        "pid": int(process.pid),
        "pgid": int(process.pid),
        "started_at": started_at,
        "status": "running",
        "cmd": cmd,
        "report_path": str(paths["report_path"]),
        "status_path": str(paths["status_path"]),
        "event_log_path": str(paths["event_log_path"]),
        "stdout_log_path": str(paths["stdout_log_path"]),
        "improvement_reports_dir": str(paths["improvement_reports_dir"]),
    }
    _write_control(control_path, control)
    print(json.dumps(control, indent=2))
    return 0


def _status(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    control_path = _control_path(run_root)
    control = _load_control(control_path)
    if not control:
        payload = {
            "run_root": str(run_root),
            "control_path": str(control_path),
            "alive": False,
            "status": "not_found",
            "reason": "missing_or_invalid_control_file",
        }
        print(json.dumps(payload, indent=2))
        return 1
    pid = int(control.get("pid", 0) or 0)
    control["alive"] = _pid_alive(pid)
    status_token = str(control.get("status_path", "")).strip()
    if status_token:
        status_path = Path(status_token)
    else:
        status_path = Path()
    if status_token and status_path.is_file():
        control["status_payload"] = _load_control(status_path)
    print(json.dumps(control, indent=2))
    return 0


def _stop(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    control_path = _control_path(run_root)
    control = _load_control(control_path)
    pid = int(control.get("pid", 0) or 0)
    if not _pid_alive(pid):
        control["alive"] = False
        control["status"] = "not_running"
        _write_control(control_path, control)
        print(json.dumps(control, indent=2))
        return 0

    sig = _signal_value(args.signal)
    os.killpg(pid, sig)
    deadline = time.monotonic() + max(0.0, float(args.wait_seconds))
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    control["alive"] = _pid_alive(pid)
    control["status"] = "stopped" if not control["alive"] else "signal_sent"
    control["stopped_at"] = datetime.now(timezone.utc).isoformat()
    control["signal"] = sig
    _write_control(control_path, control)
    print(json.dumps(control, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--run-root", required=True)
    start.add_argument("--env", action="append", default=[])
    start.add_argument("campaign_args", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-root", required=True)
    stop.add_argument("--signal", default="TERM")
    stop.add_argument("--wait-seconds", type=float, default=10.0)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "start":
        return _start(args)
    if args.command == "status":
        return _status(args)
    if args.command == "stop":
        return _stop(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
