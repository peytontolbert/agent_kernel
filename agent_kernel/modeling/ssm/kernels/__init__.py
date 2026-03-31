"""CUDA-kernel boundary for state-space modeling backends.

This package is reserved for custom CUDA or Triton kernels used by future
TOLBERT-improved state-space models. Runtime agent code should never import
from here directly; access should flow through modeling-layer wrappers.
"""

from .cuda_selective_scan import (
    CudaSelectiveScanStatus,
    cuda_selective_scan_status,
    selective_scan_cuda_fn,
    selective_scan_cuda_metadata,
)

__all__ = [
    "CudaSelectiveScanStatus",
    "cuda_selective_scan_status",
    "selective_scan_cuda_fn",
    "selective_scan_cuda_metadata",
]
