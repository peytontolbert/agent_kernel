"""Learned-world and latent-state helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CausalBeliefScanResult",
    "CausalWorldPrior",
    "CausalWorldProfile",
    "MODELING_COUNTERFACTUAL_GROUPS",
    "build_causal_state_signature",
    "causal_belief_scan",
    "causal_belief_scan_ref",
    "condition_causal_world_prior",
    "load_causal_world_profile",
    "parse_modeling_counterfactual_groups",
    "summarize_causal_machine_adoption",
]

_MODULE_MAP = {
    "CausalBeliefScanResult": ".belief_scan",
    "causal_belief_scan": ".belief_scan",
    "causal_belief_scan_ref": ".belief_scan",
    "CausalWorldPrior": ".causal_machine",
    "CausalWorldProfile": ".causal_machine",
    "build_causal_state_signature": ".causal_machine",
    "condition_causal_world_prior": ".causal_machine",
    "load_causal_world_profile": ".causal_machine",
    "summarize_causal_machine_adoption": ".causal_machine",
    "MODELING_COUNTERFACTUAL_GROUPS": ".counterfactual",
    "parse_modeling_counterfactual_groups": ".counterfactual",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
