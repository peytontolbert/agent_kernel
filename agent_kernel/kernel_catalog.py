from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).resolve().parent.parent / "datasets" / "kernel_metadata.json"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, object]:
    try:
        loaded = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid kernel metadata dataset: {_CATALOG_PATH}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"kernel metadata dataset must be a JSON object: {_CATALOG_PATH}")
    return loaded


def _section(name: str) -> dict[str, object]:
    section = _catalog().get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f"missing kernel metadata section {name!r} in {_CATALOG_PATH}")
    return section


def kernel_catalog_mapping(section: str, key: str) -> dict[str, object]:
    value = _section(section).get(key, {})
    if not isinstance(value, dict):
        raise RuntimeError(f"kernel metadata {section!r}.{key!r} must be an object")
    return dict(value)


def kernel_catalog_list(section: str, key: str) -> list[Any]:
    value = _section(section).get(key, [])
    if not isinstance(value, list):
        raise RuntimeError(f"kernel metadata {section!r}.{key!r} must be a list")
    return list(value)


def kernel_catalog_string_list(section: str, key: str) -> list[str]:
    return [item for item in (str(entry).strip() for entry in kernel_catalog_list(section, key)) if item]


def kernel_catalog_string_set(section: str, key: str) -> set[str]:
    return set(kernel_catalog_string_list(section, key))
