from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).resolve().parent.parent / "datasets" / "kernel_metadata.json"
_RETAINED_EXTENSION_REGISTRY_KEY = "retained_extensions"
_RETAINED_EXTENSION_SCHEMA_VERSION = "retained_ontology_extensions_v1"
_RETAINED_EXTENSION_REGISTRY_FIELDS = frozenset({"schema_version", "entries"})
_RETAINED_EXTENSION_FIELDS = frozenset(
    {
        "extension_id",
        "namespace",
        "kind",
        "schema_version",
        "declared_fields",
        "validator_key",
        "retention_state",
        "provenance",
        "section",
        "key",
        "payload",
    }
)
_VALIDATOR_KIND_BY_KEY = {
    "mapping_v1": "mapping",
    "string_list_v1": "list",
    "record_list_v1": "list",
}


@dataclass(frozen=True, slots=True)
class RetainedOntologyExtension:
    extension_id: str
    namespace: str
    kind: str
    schema_version: str
    declared_fields: tuple[str, ...]
    validator_key: str
    retention_state: str
    provenance: dict[str, Any]
    section: str
    key: str
    payload: object


@lru_cache(maxsize=1)
def _raw_catalog() -> dict[str, object]:
    try:
        loaded = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid kernel metadata dataset: {_CATALOG_PATH}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"kernel metadata dataset must be a JSON object: {_CATALOG_PATH}")
    return loaded


@lru_cache(maxsize=1)
def _authored_catalog() -> dict[str, dict[str, object]]:
    authored: dict[str, dict[str, object]] = {}
    for name, value in _raw_catalog().items():
        if name == _RETAINED_EXTENSION_REGISTRY_KEY:
            continue
        if not isinstance(value, dict):
            raise RuntimeError(f"kernel metadata section {name!r} must be an object")
        authored[name] = deepcopy(value)
    return authored


def _normalize_declared_fields(raw_fields: object) -> tuple[str, ...]:
    if not isinstance(raw_fields, list):
        raise RuntimeError("retained ontology extension declared_fields must be a list")
    normalized = tuple(field for field in (str(item).strip() for item in raw_fields) if field)
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("retained ontology extension declared_fields must not contain duplicates")
    return normalized


def _validate_extension_mapping_payload(payload: object, declared_fields: tuple[str, ...]) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("retained ontology extension mapping payload must be an object")
    validated: dict[str, object] = {}
    for key, value in payload.items():
        normalized = str(key).strip()
        if not normalized:
            raise RuntimeError("retained ontology extension mapping payload keys must be non-empty")
        if declared_fields and normalized not in declared_fields:
            raise RuntimeError(f"retained ontology extension mapping payload contains undeclared field {normalized!r}")
        validated[normalized] = deepcopy(value)
    return validated


def _validate_extension_list_payload(
    payload: object,
    *,
    validator_key: str,
    declared_fields: tuple[str, ...],
) -> list[object]:
    if not isinstance(payload, list):
        raise RuntimeError("retained ontology extension list payload must be a list")
    validated: list[object] = []
    if validator_key == "string_list_v1":
        for item in payload:
            normalized = str(item).strip()
            if not normalized:
                raise RuntimeError("retained ontology extension string list payload must not contain empty entries")
            validated.append(normalized)
        return validated
    if validator_key != "record_list_v1":
        raise RuntimeError(f"unsupported retained ontology list validator {validator_key!r}")
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("retained ontology extension record list payload must contain only objects")
        record: dict[str, object] = {}
        for key, value in item.items():
            normalized = str(key).strip()
            if not normalized:
                raise RuntimeError("retained ontology extension record keys must be non-empty")
            if declared_fields and normalized not in declared_fields:
                raise RuntimeError(f"retained ontology extension record contains undeclared field {normalized!r}")
            record[normalized] = deepcopy(value)
        validated.append(record)
    return validated


def _validate_extension_payload(
    *,
    kind: str,
    validator_key: str,
    declared_fields: tuple[str, ...],
    payload: object,
) -> object:
    if kind == "mapping":
        return _validate_extension_mapping_payload(payload, declared_fields)
    if kind == "list":
        return _validate_extension_list_payload(
            payload,
            validator_key=validator_key,
            declared_fields=declared_fields,
        )
    raise RuntimeError(f"unsupported retained ontology extension kind {kind!r}")


def _validate_retained_extension(entry: object, authored_sections: set[str]) -> RetainedOntologyExtension:
    if not isinstance(entry, dict):
        raise RuntimeError("retained ontology extension entries must be objects")
    unknown_fields = set(entry) - _RETAINED_EXTENSION_FIELDS
    if unknown_fields:
        fields = ", ".join(sorted(str(field) for field in unknown_fields))
        raise RuntimeError(f"retained ontology extension contains unknown fields: {fields}")
    extension_id = str(entry.get("extension_id", "")).strip()
    namespace = str(entry.get("namespace", "")).strip()
    kind = str(entry.get("kind", "")).strip()
    schema_version = str(entry.get("schema_version", "")).strip()
    validator_key = str(entry.get("validator_key", "")).strip()
    retention_state = str(entry.get("retention_state", "")).strip()
    section = str(entry.get("section", "")).strip()
    key = str(entry.get("key", "")).strip()
    if not extension_id:
        raise RuntimeError("retained ontology extension entries must define extension_id")
    if not namespace:
        raise RuntimeError(f"retained ontology extension {extension_id!r} must define namespace")
    if not key.startswith(f"{namespace}__"):
        raise RuntimeError(
            f"retained ontology extension {extension_id!r} must use a namespaced key starting with {namespace}__"
        )
    if not kind:
        raise RuntimeError(f"retained ontology extension {extension_id!r} must define kind")
    if schema_version != _RETAINED_EXTENSION_SCHEMA_VERSION:
        raise RuntimeError(
            f"retained ontology extension {extension_id!r} must use schema_version {_RETAINED_EXTENSION_SCHEMA_VERSION!r}"
        )
    expected_kind = _VALIDATOR_KIND_BY_KEY.get(validator_key)
    if expected_kind is None:
        raise RuntimeError(f"retained ontology extension {extension_id!r} uses unsupported validator {validator_key!r}")
    if expected_kind != kind:
        raise RuntimeError(
            f"retained ontology extension {extension_id!r} kind {kind!r} does not match validator {validator_key!r}"
        )
    if retention_state != "retain":
        raise RuntimeError(
            f"retained ontology extension {extension_id!r} must be retained before it can be reloaded through the catalog"
        )
    if not section:
        raise RuntimeError(f"retained ontology extension {extension_id!r} must define section")
    if section not in authored_sections and not section.startswith(f"{namespace}__"):
        raise RuntimeError(
            f"retained ontology extension {extension_id!r} must target an authored section or a namespaced extension section"
        )
    declared_fields = _normalize_declared_fields(entry.get("declared_fields", []))
    provenance = entry.get("provenance", {})
    if not isinstance(provenance, dict):
        raise RuntimeError(f"retained ontology extension {extension_id!r} provenance must be an object")
    payload = _validate_extension_payload(
        kind=kind,
        validator_key=validator_key,
        declared_fields=declared_fields,
        payload=entry.get("payload"),
    )
    return RetainedOntologyExtension(
        extension_id=extension_id,
        namespace=namespace,
        kind=kind,
        schema_version=schema_version,
        declared_fields=declared_fields,
        validator_key=validator_key,
        retention_state=retention_state,
        provenance=deepcopy(provenance),
        section=section,
        key=key,
        payload=payload,
    )


@lru_cache(maxsize=1)
def _retained_extensions() -> tuple[RetainedOntologyExtension, ...]:
    registry = _raw_catalog().get(_RETAINED_EXTENSION_REGISTRY_KEY, {})
    if registry in ({}, None):
        return ()
    if not isinstance(registry, dict):
        raise RuntimeError(f"kernel metadata section {_RETAINED_EXTENSION_REGISTRY_KEY!r} must be an object")
    unknown_fields = set(registry) - _RETAINED_EXTENSION_REGISTRY_FIELDS
    if unknown_fields:
        fields = ", ".join(sorted(str(field) for field in unknown_fields))
        raise RuntimeError(f"kernel metadata retained extension registry contains unknown fields: {fields}")
    schema_version = str(registry.get("schema_version", "")).strip()
    if schema_version != _RETAINED_EXTENSION_SCHEMA_VERSION:
        raise RuntimeError(
            f"kernel metadata retained extension registry must use schema_version {_RETAINED_EXTENSION_SCHEMA_VERSION!r}"
        )
    raw_entries = registry.get("entries", [])
    if not isinstance(raw_entries, list):
        raise RuntimeError("kernel metadata retained extension registry entries must be a list")
    authored_sections = set(_authored_catalog())
    extensions = tuple(_validate_retained_extension(entry, authored_sections) for entry in raw_entries)
    seen_targets: set[tuple[str, str]] = set()
    for extension in extensions:
        target = (extension.section, extension.key)
        if target in seen_targets:
            raise RuntimeError(
                f"duplicate retained ontology extension target {extension.section!r}.{extension.key!r} in {_CATALOG_PATH}"
            )
        seen_targets.add(target)
    return extensions


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict[str, object]]:
    merged = deepcopy(_authored_catalog())
    for extension in _retained_extensions():
        section = merged.setdefault(extension.section, {})
        if not isinstance(section, dict):
            raise RuntimeError(f"kernel metadata section {extension.section!r} must be an object")
        if extension.key in section:
            raise RuntimeError(
                f"retained ontology extension {extension.extension_id!r} collides with existing catalog key "
                f"{extension.section!r}.{extension.key!r}"
            )
        section[extension.key] = deepcopy(extension.payload)
    return merged


def reset_kernel_catalog_cache() -> None:
    _raw_catalog.cache_clear()
    _authored_catalog.cache_clear()
    _retained_extensions.cache_clear()
    _catalog.cache_clear()


def kernel_catalog_retained_extensions() -> tuple[RetainedOntologyExtension, ...]:
    return tuple(_retained_extensions())


def _section(name: str) -> dict[str, object]:
    section = _catalog().get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f"missing kernel metadata section {name!r} in {_CATALOG_PATH}")
    return deepcopy(section)


def _section_key(section: str, key: str) -> object:
    resolved = _section(section)
    normalized_key = str(key).strip()
    if normalized_key not in resolved:
        raise RuntimeError(f"missing kernel metadata key {section!r}.{normalized_key!r} in {_CATALOG_PATH}")
    return resolved[normalized_key]


def kernel_catalog_mapping(section: str, key: str) -> dict[str, object]:
    value = _section_key(section, key)
    if not isinstance(value, dict):
        raise RuntimeError(f"kernel metadata {section!r}.{key!r} must be an object")
    return deepcopy(value)


def kernel_catalog_list(section: str, key: str) -> list[Any]:
    value = _section_key(section, key)
    if not isinstance(value, list):
        raise RuntimeError(f"kernel metadata {section!r}.{key!r} must be a list")
    return deepcopy(value)


def kernel_catalog_record_list(section: str, key: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in kernel_catalog_list(section, key):
        if not isinstance(item, dict):
            raise RuntimeError(f"kernel metadata {section!r}.{key!r} must contain only objects")
        records.append(deepcopy(item))
    return records


def kernel_catalog_string_list(section: str, key: str) -> list[str]:
    return [item for item in (str(entry).strip() for entry in kernel_catalog_list(section, key)) if item]


def kernel_catalog_string_set(section: str, key: str) -> set[str]:
    return set(kernel_catalog_string_list(section, key))
