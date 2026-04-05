from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .schemas import TaskSpec


_CURRICULUM_TEMPLATE_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent / "datasets" / "curriculum_templates.json"
)


@lru_cache(maxsize=1)
def _catalog_payload() -> dict[str, Any]:
    try:
        payload = json.loads(_CURRICULUM_TEMPLATE_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to load curriculum template catalog from {_CURRICULUM_TEMPLATE_CATALOG_PATH}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Curriculum template catalog must be a JSON object")
    return payload


@lru_cache(maxsize=1)
def load_curriculum_template_catalog() -> dict[str, dict[str, Any]]:
    payload = _catalog_payload()
    templates = payload.get("templates", {})
    if not isinstance(templates, dict):
        raise RuntimeError("Curriculum template catalog must contain a top-level 'templates' object")
    return {
        str(template_id): dict(template_payload)
        for template_id, template_payload in templates.items()
        if isinstance(template_id, str) and isinstance(template_payload, dict)
    }


@lru_cache(maxsize=1)
def load_curriculum_metadata_catalog() -> dict[str, Any]:
    payload = _catalog_payload()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("Curriculum template catalog must contain a top-level 'metadata' object")
    return deepcopy(metadata)


def render_curriculum_template(
    template_id: str,
    *,
    replacements: dict[str, object],
    metadata_overrides: dict[str, object] | None = None,
    payload_overrides: dict[str, object] | None = None,
    prompt_suffix: str = "",
) -> TaskSpec:
    catalog = load_curriculum_template_catalog()
    try:
        template = deepcopy(catalog[template_id])
    except KeyError as exc:
        raise KeyError(f"Unknown curriculum template: {template_id}") from exc
    payload = _render_placeholders(template, replacements)
    payload["prompt"] = f"{payload.pop('prompt_template')}{prompt_suffix}"
    payload["task_id"] = payload.pop("task_id_template")
    payload["workspace_subdir"] = payload.pop("workspace_subdir_template")
    extra_payload = payload.pop("payload", {})
    extra_payload = dict(extra_payload) if isinstance(extra_payload, dict) else {}
    metadata = dict(payload.get("metadata", {}))
    for key in ("synthetic_edit_plan", "synthetic_edit_candidates"):
        if key in extra_payload and key not in metadata:
            metadata[key] = deepcopy(extra_payload[key])
    if metadata_overrides:
        metadata.update(dict(metadata_overrides))
    payload["metadata"] = metadata
    for key, value in extra_payload.items():
        payload.setdefault(str(key), deepcopy(value))
    if payload_overrides:
        payload.update(dict(payload_overrides))
    return TaskSpec.from_dict(payload)


def _render_placeholders(value: Any, replacements: dict[str, object]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace(f"{{{key}}}", str(replacement))
        return rendered
    if isinstance(value, list):
        return [_render_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {str(key): _render_placeholders(item, replacements) for key, item in value.items()}
    return value
