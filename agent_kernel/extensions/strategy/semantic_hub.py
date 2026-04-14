from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from ...config import KernelConfig
from ...ops.runtime_supervision import atomic_write_json

_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "candidate",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "strategy",
    "the",
    "to",
    "with",
}


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


def _flatten_semantic_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, nested in value.items():
            key_text = str(key).strip()
            if key_text:
                flattened.append(key_text)
            if isinstance(nested, (str, int, float, bool)) and key_text:
                flattened.append(f"{key_text}:{nested}")
            flattened.extend(_flatten_semantic_text(nested))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_semantic_text(item))
        return flattened
    return [str(value)]


def semantic_query_terms(*values: object) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for text in _flatten_semantic_text(values):
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower()):
            if len(token) < 2 or token in _SEMANTIC_STOPWORDS or token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms


def _semantic_haystack(payload: dict[str, Any]) -> str:
    return " ".join(_flatten_semantic_text(payload)).lower()


def _item_filter_matches(payload: dict[str, Any], filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        actual = payload.get(key)
        expected_values = {
            str(value).strip().lower()
            for value in (expected if isinstance(expected, (list, tuple, set)) else [expected])
            if str(value).strip()
        }
        if not expected_values:
            continue
        actual_values = {
            str(value).strip().lower()
            for value in (actual if isinstance(actual, (list, tuple, set)) else [actual])
            if str(value).strip()
        }
        if not actual_values.intersection(expected_values):
            return False
    return True


def _semantic_item_score(
    payload: dict[str, Any],
    *,
    query: str,
    query_terms: list[str],
    filters: dict[str, object],
) -> tuple[float, list[str]]:
    if filters and not _item_filter_matches(payload, filters):
        return 0.0, []
    if not query_terms and not filters:
        return 0.1, []
    item_terms = set(semantic_query_terms(payload))
    matched_terms = [term for term in query_terms if term in item_terms]
    score = float(len(matched_terms))
    if query_terms:
        score += len(matched_terms) / max(1, len(query_terms))
    haystack = _semantic_haystack(payload)
    phrase = str(query).strip().lower()
    if phrase and phrase in haystack:
        score += 2.0
    status = str(payload.get("retention_state", payload.get("status", ""))).strip().lower()
    if status == "retain":
        score += 0.25
    if matched_terms or not query_terms:
        score += 0.1 * len(filters)
    return round(score, 4), matched_terms


def iter_semantic_items(
    config: KernelConfig,
    *,
    categories: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    root = config.semantic_hub_root
    if not root.exists():
        return []
    category_tokens = {
        _safe_token(category, default="misc")
        for category in list(categories or [])
        if str(category).strip()
    }
    category_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (not category_tokens or path.name in category_tokens)
    ]
    items: list[dict[str, Any]] = []
    for category_dir in category_dirs:
        for path in sorted(category_dir.glob("*.json")):
            payload = _load_json(path)
            if not payload:
                continue
            category = str(payload.get("category", category_dir.name)).strip() or category_dir.name
            item_id = str(payload.get("item_id", path.stem)).strip() or path.stem
            items.append(
                {
                    "category": category,
                    "item_id": item_id,
                    "path": str(path),
                    "payload": payload,
                }
            )
    return items


def query_semantic_items(
    config: KernelConfig,
    *,
    query: str = "",
    categories: list[str] | tuple[str, ...] | set[str] | None = None,
    filters: dict[str, object] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    query_terms = semantic_query_terms(query)
    normalized_filters = dict(filters or {})
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in iter_semantic_items(config, categories=categories):
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        score, matched_terms = _semantic_item_score(
            payload,
            query=query,
            query_terms=query_terms,
            filters=normalized_filters,
        )
        if score <= 0.0:
            continue
        ranked.append(
            (
                score,
                str(item.get("path", "")),
                {
                    **item,
                    "score": score,
                    "matched_terms": matched_terms,
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item for _, _, item in ranked[: max(0, int(limit))]]


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


def record_semantic_task(
    config: KernelConfig,
    *,
    task_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_task")
    return write_semantic_item(config, category="tasks", item_id=task_id, payload=payload)


def upsert_semantic_agent(
    config: KernelConfig,
    *,
    agent_id: str,
    payload: dict[str, Any],
) -> Path:
    payload = dict(payload)
    payload.setdefault("artifact_kind", "semantic_agent")
    return merge_semantic_item(config, category="agents", item_id=agent_id, payload=payload)
