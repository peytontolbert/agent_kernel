from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.ops.runtime_supervision import atomic_write_json


class LeaseRegistry:
    def __init__(self, *, state_path: Path, ttl_seconds: float, bearer_token: str = ""):
        self.state_path = state_path
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.bearer_token = str(bearer_token).strip()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {"leases": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"leases": {}}
        if not isinstance(payload, dict):
            return {"leases": {}}
        payload.setdefault("leases", {})
        return payload

    def _persist(self) -> None:
        atomic_write_json(self.state_path, self._state)

    def _lease_key(self, payload: dict[str, object]) -> str:
        return str(payload.get("lease_kind", "agentkernel_unattended_campaign")).strip() or "agentkernel_unattended_campaign"

    def _prune_expired(self) -> None:
        now = time.time()
        leases = self._state.get("leases", {})
        if not isinstance(leases, dict):
            leases = {}
        active = {
            key: value
            for key, value in leases.items()
            if isinstance(value, dict) and (now - float(value.get("heartbeat_at_epoch", 0.0) or 0.0)) < self.ttl_seconds
        }
        self._state["leases"] = active

    def verify_token(self, authorization_header: str) -> bool:
        if not self.bearer_token:
            return True
        return authorization_header.strip() == f"Bearer {self.bearer_token}"

    def acquire(self, payload: dict[str, object]) -> dict[str, object]:
        self._prune_expired()
        key = self._lease_key(payload)
        leases = self._state.setdefault("leases", {})
        now = time.time()
        lease_id = f"{key}:{payload.get('hostname', 'unknown')}:{payload.get('pid', 0)}"
        current = leases.get(key)
        if isinstance(current, dict):
            return {
                "acquired": False,
                "detail": f"lease already held by {current.get('hostname', 'unknown')}:{current.get('pid', 0)}",
                "lease_id": str(current.get("lease_id", "")).strip(),
            }
        lease_payload = {
            "lease_id": lease_id,
            "lease_kind": key,
            "hostname": str(payload.get("hostname", "")).strip(),
            "pid": int(payload.get("pid", 0) or 0),
            "report_path": str(payload.get("report_path", "")).strip(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_at_epoch": now,
        }
        leases[key] = lease_payload
        self._persist()
        return {"acquired": True, "detail": "lease acquired", "lease_id": lease_id}

    def heartbeat(self, payload: dict[str, object]) -> dict[str, object]:
        self._prune_expired()
        lease_id = str(payload.get("lease_id", "")).strip()
        leases = self._state.setdefault("leases", {})
        for lease_payload in leases.values():
            if isinstance(lease_payload, dict) and str(lease_payload.get("lease_id", "")).strip() == lease_id:
                lease_payload["heartbeat_at_epoch"] = time.time()
                lease_payload["status"] = str(payload.get("status", "")).strip()
                lease_payload["phase"] = str(payload.get("phase", "")).strip()
                lease_payload["reason"] = str(payload.get("reason", "")).strip()
                self._persist()
                return {"ok": True, "detail": "lease heartbeat accepted", "lease_id": lease_id}
        return {"ok": False, "detail": "lease id not found", "lease_id": lease_id}

    def release(self, payload: dict[str, object]) -> dict[str, object]:
        lease_id = str(payload.get("lease_id", "")).strip()
        leases = self._state.setdefault("leases", {})
        for key, lease_payload in list(leases.items()):
            if isinstance(lease_payload, dict) and str(lease_payload.get("lease_id", "")).strip() == lease_id:
                leases.pop(key, None)
                self._persist()
                return {"released": True, "detail": "lease released", "lease_id": lease_id}
        return {"released": False, "detail": "lease id not found", "lease_id": lease_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--ttl-seconds", type=float, default=600.0)
    parser.add_argument("--bearer-token", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    state_path = (
        Path(args.state_path)
        if args.state_path
        else repo_root / "var" / "lease_server_state.json"
    )
    registry = LeaseRegistry(
        state_path=state_path,
        ttl_seconds=float(args.ttl_seconds),
        bearer_token=str(args.bearer_token or ""),
    )

    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            authorization = self.headers.get("Authorization", "")
            if not registry.verify_token(authorization):
                self._json_response(403, {"detail": "forbidden"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(max(0, length))
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                self._json_response(400, {"detail": "invalid json"})
                return
            if not isinstance(payload, dict):
                self._json_response(400, {"detail": "json payload must be an object"})
                return
            if self.path == "/acquire":
                result = registry.acquire(payload)
                self._json_response(200 if bool(result.get("acquired", False)) else 409, result)
                return
            if self.path == "/heartbeat":
                result = registry.heartbeat(payload)
                self._json_response(200 if bool(result.get("ok", False)) else 404, result)
                return
            if self.path == "/release":
                result = registry.release(payload)
                self._json_response(200 if bool(result.get("released", False)) else 404, result)
                return
            self._json_response(404, {"detail": "unknown path"})

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(f"http://{args.host}:{int(args.port)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
