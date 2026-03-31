from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - reduced environments
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from .build import build_command, build_extension, build_metadata, extension_name
from ...artifacts import load_model_artifact, tolbert_kernel_autobuild_ready

_SUPPORTED_DTYPES = set()
if torch is not None:
    _SUPPORTED_DTYPES = {torch.float32, torch.float64}

_LOADED_EXTENSION: Any | None = None
_LOAD_ERROR_DETAIL = ""


@dataclass(frozen=True, slots=True)
class CudaBeliefScanStatus:
    available: bool
    source: str
    cuda_available: bool
    compiled_extension: bool
    repo_root: str
    detail: str = ""


def cuda_belief_scan_status() -> CudaBeliefScanStatus:
    compiled = _load_extension(auto_build=False) is not None
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if not compiled:
        _, autobuild_detail = _autobuild_allowed()
        return CudaBeliefScanStatus(
            available=False,
            source="native",
            cuda_available=cuda_available,
            compiled_extension=False,
            repo_root="agent_kernel/modeling/world/kernels",
            detail=(
                autobuild_detail
                or _LOAD_ERROR_DETAIL
                or "native causal-belief CUDA extension is not importable. "
                f"Build it explicitly with: {build_command()}"
            ),
        )
    if not cuda_available:
        return CudaBeliefScanStatus(
            available=False,
            source="native",
            cuda_available=False,
            compiled_extension=True,
            repo_root="agent_kernel/modeling/world/kernels",
            detail="torch.cuda.is_available() is false",
        )
    return CudaBeliefScanStatus(
        available=True,
        source="native",
        cuda_available=True,
        compiled_extension=True,
        repo_root="agent_kernel/modeling/world/kernels",
        detail="",
    )


def causal_belief_scan_cuda_fn(
    local_logits: Tensor,
    transition_log_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    *,
    transition_gate: float | Tensor = 1.0,
    chunk_size: int = 128,
    auto_build: bool = False,
) -> tuple[Tensor, Tensor]:
    _require_torch()
    _validate_inputs(
        local_logits=local_logits,
        transition_log_probs=transition_log_probs,
        transition_context=transition_context,
        initial_log_belief=initial_log_belief,
        chunk_size=chunk_size,
    )
    gate_tensor = (
        transition_gate.to(device=local_logits.device, dtype=local_logits.dtype)
        if isinstance(transition_gate, Tensor)
        else local_logits.new_tensor(float(transition_gate))
    )
    transition_bandwidth = _transition_bandwidth(transition_log_probs)
    if gate_tensor.numel() != 1:
        raise ValueError(f"transition_gate must be scalar; got shape {tuple(gate_tensor.shape)}")
    beliefs, final_log_belief = _CudaBeliefScanFn.apply(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        gate_tensor.reshape(()),
        int(transition_bandwidth),
        int(chunk_size),
        bool(auto_build),
    )
    return beliefs, final_log_belief


def causal_belief_scan_cuda_metadata() -> dict[str, object]:
    metadata = build_metadata()
    return {
        "backend_source": "native",
        "repo_root": "agent_kernel/modeling/world/kernels",
        "compiled_extension_name": extension_name(),
        "compiled_extension_sources": [
            "agent_kernel/modeling/world/kernels/src/causal_belief_scan.cpp",
            "agent_kernel/modeling/world/kernels/src/causal_belief_scan_cuda.cu",
        ],
        "expected_entrypoint": extension_name(),
        "expected_wrapper": "agent_kernel.modeling.world.kernels.cuda_belief_scan.causal_belief_scan_cuda_fn",
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "compiled_extension_importable": _load_extension(auto_build=False) is not None,
        "recommended_build_command": build_command(),
        "build_metadata": metadata,
    }


if torch is not None and hasattr(torch, "autograd"):

    class _CudaBeliefScanFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            local_logits: Tensor,
            transition_log_probs: Tensor,
            transition_context: Tensor,
            initial_log_belief: Tensor,
            transition_gate: Tensor,
            transition_bandwidth: int,
            chunk_size: int,
            auto_build: bool,
        ) -> tuple[Tensor, Tensor]:
            module = _load_extension(auto_build=auto_build)
            if module is None:
                raise RuntimeError(
                    "native causal-belief CUDA extension is unavailable; "
                    f"build it explicitly with: {build_command()}"
                )
            beliefs, final_log_belief = module.fwd(
                local_logits.contiguous(),
                transition_log_probs.contiguous(),
                transition_context.contiguous(),
                initial_log_belief.contiguous(),
                float(transition_gate.detach().item()),
                int(transition_bandwidth),
                int(chunk_size),
            )
            ctx.save_for_backward(
                local_logits,
                transition_log_probs,
                transition_context,
                initial_log_belief,
                transition_gate.reshape(()),
                beliefs,
                final_log_belief,
            )
            ctx.transition_bandwidth = int(transition_bandwidth)
            ctx.chunk_size = int(chunk_size)
            return beliefs, final_log_belief

        @staticmethod
        def backward(ctx, grad_beliefs: Tensor, grad_final_log_belief: Tensor) -> tuple[Tensor | None, ...]:
            module = _load_extension(auto_build=False)
            if module is None:
                raise RuntimeError(
                    "native causal-belief CUDA extension became unavailable before backward"
                )
            (
                local_logits,
                transition_log_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                beliefs,
                final_log_belief,
            ) = ctx.saved_tensors
            (
                grad_local_logits,
                grad_transition_log_probs,
                grad_transition_context,
                grad_initial_log_belief,
                _grad_transition_gate,
            ) = module.bwd(
                grad_beliefs.contiguous(),
                grad_final_log_belief.contiguous(),
                local_logits,
                transition_log_probs,
                transition_context,
                initial_log_belief,
                beliefs,
                final_log_belief,
                float(transition_gate.detach().item()),
                int(ctx.transition_bandwidth),
                int(ctx.chunk_size),
            )
            return (
                grad_local_logits,
                grad_transition_log_probs,
                grad_transition_context,
                grad_initial_log_belief,
                _grad_transition_gate.reshape_as(transition_gate),
                None,
                None,
                None,
            )

else:  # pragma: no cover - reduced environments

    class _CudaBeliefScanFn:
        @staticmethod
        def apply(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
            del args, kwargs
            raise RuntimeError("torch.autograd is required for the native causal-belief CUDA path")


def _load_extension(*, auto_build: bool) -> Any | None:
    global _LOADED_EXTENSION, _LOAD_ERROR_DETAIL
    if _LOADED_EXTENSION is not None:
        return _LOADED_EXTENSION
    build_dir = str(build_metadata()["build_directory"])
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    if torch is None:
        return None
    try:
        _LOADED_EXTENSION = importlib.import_module(extension_name())
        _LOAD_ERROR_DETAIL = ""
        return _LOADED_EXTENSION
    except ImportError as exc:
        _LOAD_ERROR_DETAIL = str(exc).strip() or exc.__class__.__name__
        if not auto_build:
            return None
    except Exception as exc:
        _LOAD_ERROR_DETAIL = str(exc).strip() or exc.__class__.__name__
        if not auto_build:
            return None
    autobuild_allowed, _ = _autobuild_allowed()
    if not autobuild_allowed:
        return None
    try:
        _LOADED_EXTENSION = build_extension(verbose=False)
        _LOAD_ERROR_DETAIL = ""
    except Exception as exc:
        _LOAD_ERROR_DETAIL = str(exc).strip() or exc.__class__.__name__
        return None
    return _LOADED_EXTENSION


def _autobuild_allowed() -> tuple[bool, str]:
    payload = load_model_artifact(_active_tolbert_model_artifact_path())
    return tolbert_kernel_autobuild_ready(payload)


def _active_tolbert_model_artifact_path() -> Path:
    raw = Path(
        os.getenv(
            "AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH",
            "trajectories/tolbert_model/tolbert_model_artifact.json",
        )
    )
    if raw.is_absolute():
        return raw
    return _repo_root() / raw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _validate_inputs(
    *,
    local_logits: Tensor,
    transition_log_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    chunk_size: int,
) -> None:
    if local_logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"local_logits dtype must be one of {_SUPPORTED_DTYPES}; got {local_logits.dtype}")
    for name, value in (
        ("transition_log_probs", transition_log_probs),
        ("transition_context", transition_context),
        ("initial_log_belief", initial_log_belief),
    ):
        if value.dtype != local_logits.dtype:
            raise TypeError(f"{name} dtype must match local_logits dtype {local_logits.dtype}; got {value.dtype}")
        if value.device != local_logits.device:
            raise ValueError(f"{name} must be on device {local_logits.device}; got {value.device}")
    if local_logits.device.type != "cuda":
        raise ValueError(f"local_logits must be on CUDA; got {local_logits.device}")
    if local_logits.dim() != 3 or transition_context.shape != local_logits.shape:
        raise ValueError("expected local_logits and transition_context with shape (batch, length, states)")
    batch_size, _, num_states = [int(value) for value in local_logits.shape]
    if transition_log_probs.dim() == 1:
        if transition_log_probs.shape != (num_states,):
            raise ValueError(
                f"transition_log_probs must have shape ({num_states},); got {tuple(transition_log_probs.shape)}"
            )
    elif transition_log_probs.dim() == 2:
        if transition_log_probs.shape != (num_states, num_states):
            raise ValueError(
                f"transition_log_probs must have shape ({num_states}, {num_states}); got {tuple(transition_log_probs.shape)}"
            )
    else:
        raise ValueError(
            "transition_log_probs must have shape (states,) or (states, states); "
            f"got {tuple(transition_log_probs.shape)}"
        )
    if initial_log_belief.shape != (batch_size, num_states):
        raise ValueError(
            f"initial_log_belief must have shape ({batch_size}, {num_states}); got {tuple(initial_log_belief.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for world-model CUDA belief scan")


def _transition_bandwidth(transition_log_probs: Tensor) -> int:
    if transition_log_probs.dim() == 1:
        return 0
    num_states = int(transition_log_probs.shape[0])
    finite = torch.isfinite(transition_log_probs)
    if not bool(finite.any().item()):
        return 0
    positions = torch.arange(num_states, device=transition_log_probs.device)
    offsets = (positions[:, None] - positions[None, :]).abs()
    return int(offsets.masked_select(finite).max().item())
