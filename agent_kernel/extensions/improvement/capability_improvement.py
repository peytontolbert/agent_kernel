from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import EvalMetrics

from ...extensions.capabilities import load_capability_modules
from ...config import KernelConfig
from .improvement_common import build_standard_proposal_artifact, normalized_generation_focus, retention_gate_preset


def build_capability_module_artifact(
    config: KernelConfig,
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
    *,
    focus: str | None = None,
) -> dict[str, object]:
    modules = [_normalize_module(module) for module in load_capability_modules(config.capability_modules_path)]
    generation_focus = normalized_generation_focus(focus, default=_default_focus(metrics, failure_counts))
    if not modules:
        modules = [
            {
                "module_id": "kernel_self_extension",
                "enabled": True,
                "capabilities": [],
                "settings": {
                    "improvement_subsystems": [
                        _default_improvement_surface("kernel_self_extension", generation_focus, metrics, failure_counts)
                    ]
                },
            }
        ]
    else:
        modules = [
            _with_default_improvement_surface(module, generation_focus, metrics, failure_counts)
            for module in modules
        ]
    summary = capability_surface_summary({"modules": modules})
    return build_standard_proposal_artifact(
        artifact_kind="capability_module_set",
        generation_focus=generation_focus,
        retention_gate=retention_gate_preset("capabilities"),
        proposals=[],
        extra_sections={
            "summary": summary,
            "modules": modules,
        },
    )


def capability_surface_summary(payload: object) -> dict[str, int]:
    modules = _payload_modules(payload)
    enabled_modules = [module for module in modules if bool(module.get("enabled", False))]
    external_capabilities = {
        capability
        for module in enabled_modules
        for capability in module.get("capabilities", [])
        if str(capability).strip()
    }
    improvement_surfaces = [
        surface
        for module in enabled_modules
        for surface in _module_improvement_surfaces(module)
    ]
    return {
        "module_count": len(modules),
        "enabled_module_count": len(enabled_modules),
        "external_capability_count": len(external_capabilities),
        "improvement_surface_count": len(improvement_surfaces),
    }


def _payload_modules(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    modules = payload.get("modules", [])
    if not isinstance(modules, list):
        return []
    normalized: list[dict[str, object]] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        normalized.append(_normalize_module(module))
    return normalized


def _normalize_module(module: dict[str, object]) -> dict[str, object]:
    settings = module.get("settings", {})
    return {
        "module_id": str(module.get("module_id", "")).strip(),
        "enabled": bool(module.get("enabled", False)),
        "capabilities": sorted(
            {
                str(capability).strip()
                for capability in module.get("capabilities", [])
                if str(capability).strip()
            }
        ),
        "settings": dict(settings) if isinstance(settings, dict) else {},
    }


def _module_improvement_surfaces(module: dict[str, object]) -> list[dict[str, object]]:
    settings = module.get("settings", {})
    if not isinstance(settings, dict):
        return []
    surfaces = settings.get("improvement_subsystems", [])
    if not isinstance(surfaces, list):
        return []
    return [surface for surface in surfaces if isinstance(surface, dict)]


def _with_default_improvement_surface(
    module: dict[str, object],
    focus: str,
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
) -> dict[str, object]:
    normalized = _normalize_module(module)
    if not normalized["enabled"]:
        return normalized
    settings = dict(normalized.get("settings", {}))
    existing = _module_improvement_surfaces(normalized)
    if existing:
        settings["improvement_subsystems"] = existing
        normalized["settings"] = settings
        return normalized
    settings["improvement_subsystems"] = [
        _default_improvement_surface(str(normalized["module_id"]), focus, metrics, failure_counts)
    ]
    normalized["settings"] = settings
    return normalized


def _default_focus(metrics: EvalMetrics, failure_counts: dict[str, int]) -> str:
    if failure_counts.get("command_failure", 0) > failure_counts.get("missing_expected_file", 0):
        return "tooling_surface"
    if metrics.low_confidence_episodes > 0:
        return "retrieval_surface"
    if metrics.generated_total and metrics.generated_pass_rate < metrics.pass_rate:
        return "curriculum_surface"
    return "policy_surface"


def _default_improvement_surface(
    module_id: str,
    focus: str,
    metrics: EvalMetrics,
    failure_counts: dict[str, int],
) -> dict[str, object]:
    del metrics
    base_subsystem = "policy"
    reason = "enabled capability module should expose a retained policy-level improvement surface"
    if focus == "tooling_surface" or failure_counts.get("command_failure", 0) > failure_counts.get("missing_expected_file", 0):
        base_subsystem = "tooling"
        reason = "enabled capability module should expose a reusable tooling improvement surface"
    elif focus == "retrieval_surface":
        base_subsystem = "retrieval"
        reason = "enabled capability module should expose a retrieval improvement surface"
    elif focus == "curriculum_surface":
        base_subsystem = "curriculum"
        reason = "enabled capability module should expose a curriculum improvement surface"
    subsystem_id = f"{module_id}_{base_subsystem}"
    return {
        "subsystem_id": subsystem_id,
        "base_subsystem": base_subsystem,
        "reason": reason,
        "priority": 4,
        "expected_gain": 0.02,
        "estimated_cost": 2,
    }
