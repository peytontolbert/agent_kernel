from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RESOURCE_KIND_ARTIFACT = "artifact"
RESOURCE_KIND_PROMPT_TEMPLATE = "prompt_template"

PROMPT_RESOURCE_SYSTEM = "prompt:system"
PROMPT_RESOURCE_DECISION = "prompt:decision"
PROMPT_RESOURCE_REFLECTION = "prompt:reflection"


def subsystem_resource_id(subsystem: str) -> str:
    normalized = str(subsystem).strip()
    if not normalized:
        raise ValueError("subsystem resource ids require a non-empty subsystem")
    return f"subsystem:{normalized}"


def prompt_resource_id(name: str) -> str:
    normalized = str(name).strip().lower()
    mapping = {
        "system": PROMPT_RESOURCE_SYSTEM,
        "decision": PROMPT_RESOURCE_DECISION,
        "reflection": PROMPT_RESOURCE_REFLECTION,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported prompt resource: {name}")
    return mapping[normalized]


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    resource_id: str
    kind: str
    active_path: Path
    payload_format: str = "json"
    artifact_kind: str = ""
    allowed_lifecycle_states: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class ResourceVersion:
    resource_id: str
    kind: str
    version_id: str
    path: Path
    payload: Any
    artifact_kind: str = ""
    lifecycle_state: str = ""
    lineage: dict[str, object] = field(default_factory=dict)
