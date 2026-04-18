from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .extensions.improvement.improvement_common import artifact_payload_in_lifecycle_states
from .extensions.strategy.kernel_catalog import kernel_catalog_record_list
from .resource_types import (
    PROMPT_RESOURCE_DECISION,
    PROMPT_RESOURCE_REFLECTION,
    PROMPT_RESOURCE_SYSTEM,
    RESOURCE_KIND_ARTIFACT,
    RESOURCE_KIND_PROMPT_TEMPLATE,
    ResourceDescriptor,
    ResourceVersion,
    subsystem_resource_id,
)

_PROMOTED_OR_RETAINED_ARTIFACT_KINDS = {
    "benchmark_candidate_set",
    "operator_class_set",
    "skill_set",
    "tool_candidate_set",
    "verifier_candidate_set",
}
_LIFECYCLE_METADATA_FIELDS = {
    "lifecycle_state",
    "retention_decision",
    "retention_gate",
    "spec_version",
}


def _artifact_allowed_lifecycle_states(artifact_kind: str) -> tuple[str, ...]:
    if artifact_kind in _PROMOTED_OR_RETAINED_ARTIFACT_KINDS:
        return ("promoted", "retained")
    return ("retained",)


def _payload_has_lifecycle_metadata(payload: dict[str, object]) -> bool:
    return any(field in payload for field in _LIFECYCLE_METADATA_FIELDS)


def _file_version_id(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""
    return f"sha256:{digest}"


class ResourceRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, ResourceDescriptor] = {}

    def register(self, descriptor: ResourceDescriptor) -> None:
        resource_id = str(descriptor.resource_id).strip()
        if not resource_id:
            raise ValueError("resource descriptors require a non-empty resource_id")
        self._descriptors[resource_id] = descriptor

    def has(self, resource_id: str) -> bool:
        return str(resource_id).strip() in self._descriptors

    def descriptor(self, resource_id: str) -> ResourceDescriptor:
        normalized = str(resource_id).strip()
        if normalized not in self._descriptors:
            raise KeyError(f"unknown resource: {resource_id}")
        return self._descriptors[normalized]

    def active_path(self, resource_id: str) -> Path:
        return self.descriptor(resource_id).active_path

    def load_text(self, resource_id: str, *, default: str = "") -> str:
        version = self.resolve_active_version(resource_id)
        if version is None or not isinstance(version.payload, str):
            return default
        return version.payload

    def load_json(self, resource_id: str) -> dict[str, object] | None:
        version = self.resolve_active_version(resource_id)
        if version is None or not isinstance(version.payload, dict):
            return None
        return dict(version.payload)

    def resolve_active_version(self, resource_id: str) -> ResourceVersion | None:
        descriptor = self.descriptor(resource_id)
        path = descriptor.active_path
        if not path.exists():
            return None
        if descriptor.payload_format == "text":
            try:
                payload = path.read_text(encoding="utf-8")
            except OSError:
                return None
            return ResourceVersion(
                resource_id=descriptor.resource_id,
                kind=descriptor.kind,
                version_id=_file_version_id(path),
                path=path,
                payload=payload,
            )
        payload = self._load_json_payload(path)
        if payload is None:
            return None
        if descriptor.artifact_kind:
            if str(payload.get("artifact_kind", "")).strip() != descriptor.artifact_kind:
                return None
            if descriptor.allowed_lifecycle_states and _payload_has_lifecycle_metadata(payload):
                retained = artifact_payload_in_lifecycle_states(
                    payload,
                    artifact_kind=descriptor.artifact_kind,
                    allowed_states=set(descriptor.allowed_lifecycle_states),
                )
                if retained is None:
                    return None
                payload = retained
        lifecycle_state = str(payload.get("lifecycle_state", "")).strip()
        lineage = payload.get("lineage", {})
        return ResourceVersion(
            resource_id=descriptor.resource_id,
            kind=descriptor.kind,
            version_id=_file_version_id(path),
            path=path,
            payload=payload,
            artifact_kind=descriptor.artifact_kind,
            lifecycle_state=lifecycle_state,
            lineage=dict(lineage) if isinstance(lineage, dict) else {},
        )

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, object] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


def runtime_resource_registry(
    config: KernelConfig,
    *,
    repo_root: Path | None = None,
) -> ResourceRegistry:
    registry = ResourceRegistry()
    for item in kernel_catalog_record_list("subsystems", "builtin_specs"):
        subsystem = str(item.get("subsystem", "")).strip()
        artifact_path_attr = str(item.get("artifact_path_attr", "")).strip()
        artifact_kind = str(item.get("artifact_kind", "")).strip()
        if not subsystem or not artifact_path_attr or not hasattr(config, artifact_path_attr):
            continue
        registry.register(
            ResourceDescriptor(
                resource_id=subsystem_resource_id(subsystem),
                kind=RESOURCE_KIND_ARTIFACT,
                active_path=getattr(config, artifact_path_attr),
                payload_format="json",
                artifact_kind=artifact_kind,
                allowed_lifecycle_states=_artifact_allowed_lifecycle_states(artifact_kind),
                description=f"active artifact for subsystem {subsystem}",
            )
        )
    if repo_root is None:
        return registry
    prompts_dir = repo_root / "prompts"
    registry.register(
        ResourceDescriptor(
            resource_id=PROMPT_RESOURCE_SYSTEM,
            kind=RESOURCE_KIND_PROMPT_TEMPLATE,
            active_path=prompts_dir / "system.md",
            payload_format="text",
            description="base system prompt template",
        )
    )
    registry.register(
        ResourceDescriptor(
            resource_id=PROMPT_RESOURCE_DECISION,
            kind=RESOURCE_KIND_PROMPT_TEMPLATE,
            active_path=prompts_dir / "decision.md",
            payload_format="text",
            description="base decision prompt template",
        )
    )
    registry.register(
        ResourceDescriptor(
            resource_id=PROMPT_RESOURCE_REFLECTION,
            kind=RESOURCE_KIND_PROMPT_TEMPLATE,
            active_path=prompts_dir / "reflection.md",
            payload_format="text",
            description="base reflection prompt template",
        )
    )
    return registry


__all__ = [
    "ResourceRegistry",
    "ResourceDescriptor",
    "ResourceVersion",
    "runtime_resource_registry",
]
