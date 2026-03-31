"""Kernel boundary for learned world-model CUDA/Triton implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CudaBeliefScanStatus",
    "causal_belief_scan_cuda_fn",
    "causal_belief_scan_cuda_metadata",
    "cuda_belief_scan_status",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".cuda_belief_scan", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
