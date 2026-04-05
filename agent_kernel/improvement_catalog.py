from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path


_CATALOG_PATH = Path(__file__).resolve().parent.parent / "datasets" / "improvement_catalog.json"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, object]:
    try:
        loaded = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid improvement catalog dataset: {_CATALOG_PATH}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"improvement catalog dataset must be a JSON object: {_CATALOG_PATH}")
    return loaded


def _section(name: str) -> dict[str, object]:
    section = _catalog().get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f"missing improvement catalog section {name!r} in {_CATALOG_PATH}")
    return section


def catalog_mapping(section: str, key: str) -> dict[str, object]:
    value = _section(section).get(key, {})
    if not isinstance(value, dict):
        raise RuntimeError(f"improvement catalog {section!r}.{key!r} must be an object")
    return dict(value)


def catalog_object(section: str, key: str) -> object:
    return deepcopy(_section(section).get(key))


def catalog_nested_string_sets(section: str, key: str) -> dict[str, set[str]]:
    value = catalog_mapping(section, key)
    normalized: dict[str, set[str]] = {}
    for field, raw_items in value.items():
        if not isinstance(raw_items, list):
            raise RuntimeError(f"improvement catalog {section!r}.{key!r}.{field!r} must be a list")
        normalized[str(field)] = {
            item for item in (str(entry).strip() for entry in raw_items) if item
        }
    return normalized


def catalog_string_list(section: str, key: str) -> list[str]:
    value = _section(section).get(key, [])
    if not isinstance(value, list):
        raise RuntimeError(f"improvement catalog {section!r}.{key!r} must be a list")
    return [item for item in (str(entry).strip() for entry in value) if item]


def catalog_string_set(section: str, key: str) -> set[str]:
    return set(catalog_string_list(section, key))
