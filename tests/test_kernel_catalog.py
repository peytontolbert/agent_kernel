import json
from pathlib import Path

import pytest

from agent_kernel.extensions.strategy import kernel_catalog


@pytest.fixture(autouse=True)
def _reset_kernel_catalog_cache():
    kernel_catalog.reset_kernel_catalog_cache()
    yield
    kernel_catalog.reset_kernel_catalog_cache()


def _write_catalog(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_kernel_catalog_reloads_retained_extension_records_through_catalog_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog_path = tmp_path / "kernel_metadata.json"
    _write_catalog(
        catalog_path,
        {
            "subsystems": {
                "base_flags": {},
            },
            "retained_extensions": {
                "schema_version": "retained_ontology_extensions_v1",
                "entries": [
                    {
                        "extension_id": "retained_extension:runtime_discovered:builtin_specs",
                        "namespace": "runtime_discovered",
                        "kind": "list",
                        "schema_version": "retained_ontology_extensions_v1",
                        "declared_fields": [
                            "subsystem",
                            "base_subsystem",
                            "artifact_path_attr",
                            "generator_kind",
                            "artifact_kind",
                            "action",
                        ],
                        "validator_key": "record_list_v1",
                        "retention_state": "retain",
                        "provenance": {
                            "source_run_id": "run:ontology:1",
                            "strategy_id": "strategy:kernel:ontology",
                        },
                        "section": "subsystems",
                        "key": "runtime_discovered__builtin_specs",
                        "payload": [
                            {
                                "subsystem": "runtime_discovered",
                                "base_subsystem": "runtime_discovered",
                                "artifact_path_attr": "runtime_discovered_artifact_path",
                                "generator_kind": "runtime_discovered",
                                "artifact_kind": "runtime_discovered_bundle",
                                "action": "observe",
                            }
                        ],
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(kernel_catalog, "_CATALOG_PATH", catalog_path)

    records = kernel_catalog.kernel_catalog_record_list("subsystems", "runtime_discovered__builtin_specs")

    assert records == [
        {
            "subsystem": "runtime_discovered",
            "base_subsystem": "runtime_discovered",
            "artifact_path_attr": "runtime_discovered_artifact_path",
            "generator_kind": "runtime_discovered",
            "artifact_kind": "runtime_discovered_bundle",
            "action": "observe",
        }
    ]
    assert kernel_catalog.kernel_catalog_mapping("subsystems", "base_flags") == {}
    assert kernel_catalog.kernel_catalog_retained_extensions()[0].extension_id == (
        "retained_extension:runtime_discovered:builtin_specs"
    )


def test_kernel_catalog_fails_closed_for_unknown_unapproved_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog_path = tmp_path / "kernel_metadata.json"
    _write_catalog(
        catalog_path,
        {
            "subsystems": {
                "base_flags": {},
            },
            "retained_extensions": {
                "schema_version": "retained_ontology_extensions_v1",
                "entries": [],
            },
        },
    )
    monkeypatch.setattr(kernel_catalog, "_CATALOG_PATH", catalog_path)

    with pytest.raises(RuntimeError, match="missing kernel metadata key 'subsystems'\\.'unsafe_unknown_key'"):
        kernel_catalog.kernel_catalog_list("subsystems", "unsafe_unknown_key")


def test_kernel_catalog_rejects_invalid_retained_extension_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog_path = tmp_path / "kernel_metadata.json"
    _write_catalog(
        catalog_path,
        {
            "subsystems": {
                "base_flags": {},
            },
            "retained_extensions": {
                "schema_version": "retained_ontology_extensions_v1",
                "entries": [
                    {
                        "extension_id": "retained_extension:runtime_discovered:builtin_specs",
                        "namespace": "runtime_discovered",
                        "kind": "list",
                        "schema_version": "retained_ontology_extensions_v1",
                        "declared_fields": ["subsystem"],
                        "validator_key": "record_list_v1",
                        "retention_state": "retain",
                        "provenance": {
                            "source_run_id": "run:ontology:2",
                        },
                        "section": "subsystems",
                        "key": "builtin_specs",
                        "payload": [
                            {
                                "subsystem": "runtime_discovered",
                            }
                        ],
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(kernel_catalog, "_CATALOG_PATH", catalog_path)

    with pytest.raises(RuntimeError, match="must use a namespaced key starting with runtime_discovered__"):
        kernel_catalog.kernel_catalog_retained_extensions()
