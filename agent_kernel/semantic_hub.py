from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .config import KernelConfig
from .runtime_supervision import atomic_write_json


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: object, *, default: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    return normalized or default


def _hub_dir(config: KernelConfig, category: str) -> Path:
    path = config.semantic_hub_root / _safe_token(category, default="misc")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hub_path(config: KernelConfig, category: str, item_id: str) -> Path:
    return _hub_dir(config, category) / f"{_safe_token(item_id, default=category)}.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_semantic_item(
    config: KernelConfig,
    *,
    category: str,
    item_id: str,
    payload: dict[str, Any],
) -> Path:
    path = _hub_path(config, category, item_id)
    body = dict(payload)
    body.setdefault("spec_version", "asi_v1")
    body.setdefault("updated_at", _utcnow())
    body.setdefault("item_id", str(item_id).strip())
    body.setdefault("category", str(category).strip())
    atomic_write_json(path, body, config=config)
    return path


def merge_semantic_item(
    config: KernelConfig,
    *,
    category: str,
    item_id: str,
    payload: dict[str, Any],
) -> Path:
    path = _hub_path(config, category, item_id)
    merged = _load_json(path)
    merged.update(dict(payload))
    merged.setdefault("created_at", merged.get("updated_at", _utcnow()))
    merged["updated_at"] = _utcnow()
    return write_semantic_item(config, category=category, item_id=item_id, payload=merged)


def record_semantic_attempt(
    config: KernelConfig,
    *,
    attempt_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_attempt")
    return merge_semantic_item(config, category="attempts", item_id=attempt_id, payload=payload)


def record_semantic_note(
    config: KernelConfig,
    *,
    note_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_note")
    return write_semantic_item(config, category="notes", item_id=note_id, payload=payload)


def record_semantic_redirect(
    config: KernelConfig,
    *,
    redirect_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_redirect")
    return write_semantic_item(config, category="redirects", item_id=redirect_id, payload=payload)


def record_semantic_skill(
    config: KernelConfig,
    *,
    skill_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_skill")
    return write_semantic_item(config, category="skills", item_id=skill_id, payload=payload)


def upsert_semantic_agent(
    config: KernelConfig,
    *,
    agent_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_agent")
    return merge_semantic_item(config, category="agents", item_id=agent_id, payload=payload)
