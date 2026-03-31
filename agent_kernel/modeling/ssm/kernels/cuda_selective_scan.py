from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover - reduced environments
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from .build import build_command, build_directory, build_extension, build_metadata, extension_name
from ...artifacts import load_model_artifact, tolbert_kernel_autobuild_ready

_SUPPORTED_DTYPES = set()
if torch is not None:
    _SUPPORTED_DTYPES = {torch.float32, torch.float64}
_LOADED_EXTENSION: Any | None = None
_LOAD_ERROR_DETAIL = ""


@dataclass(frozen=True, slots=True)
class CudaSelectiveScanStatus:
    available: bool
    source: str
    cuda_available: bool
    compiled_extension: bool
    repo_root: str
    detail: str = ""


def cuda_selective_scan_status() -> CudaSelectiveScanStatus:
    compiled = _load_extension(auto_build=False) is not None
    cuda_available = _torch_cuda_available()
    if not compiled:
        _, autobuild_detail = _autobuild_allowed()
        return CudaSelectiveScanStatus(
            available=False,
            source="native",
            cuda_available=cuda_available,
            compiled_extension=False,
            repo_root="agent_kernel/modeling/ssm/kernels",
            detail=(
                autobuild_detail
                or _LOAD_ERROR_DETAIL
                or "native selective-scan CUDA extension is not importable. "
                f"Build it with: {build_command()}"
            ),
        )
    if not cuda_available:
        return CudaSelectiveScanStatus(
            available=False,
            source="native",
            cuda_available=False,
            compiled_extension=True,
            repo_root="agent_kernel/modeling/ssm/kernels",
            detail="torch.cuda.is_available() is false",
        )
    return CudaSelectiveScanStatus(
        available=True,
        source="native",
        cuda_available=True,
        compiled_extension=True,
        repo_root="agent_kernel/modeling/ssm/kernels",
        detail="",
    )


def selective_scan_cuda_fn(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    z: Tensor | None = None,
    delta_bias: Tensor | None = None,
    delta_softplus: bool = False,
    *,
    return_last_state: bool = False,
) -> tuple[Tensor, Tensor | None]:
    _require_torch()
    _validate_inputs(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z, delta_bias=delta_bias)
    output, last_state = _CudaSelectiveScanFn.apply(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
    )
    return (output, last_state) if return_last_state else (output, None)


def selective_scan_cuda_metadata() -> dict[str, object]:
    metadata = build_metadata()
    return {
        "backend_source": "native",
        "repo_root": "agent_kernel/modeling/ssm/kernels",
        "compiled_extension_name": extension_name(),
        "compiled_extension_sources": [
            "agent_kernel/modeling/ssm/kernels/src/selective_scan.cpp",
            "agent_kernel/modeling/ssm/kernels/src/selective_scan_cuda.cu",
        ],
        "expected_entrypoint": extension_name(),
        "expected_wrapper": "agent_kernel.modeling.ssm.kernels.cuda_selective_scan.selective_scan_cuda_fn",
        "cuda_available": _torch_cuda_available(),
        "compiled_extension_importable": _load_extension(auto_build=False) is not None,
        "recommended_build_command": build_command(),
        "build_metadata": metadata,
    }


if torch is not None:
    class _CudaSelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            u: Tensor,
            delta: Tensor,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            D: Tensor | None,
            z: Tensor | None,
            delta_bias: Tensor | None,
            delta_softplus: bool,
        ) -> tuple[Tensor, Tensor]:
            module = _load_extension(auto_build=True)
            if module is None:
                raise RuntimeError(
                    "native selective-scan CUDA extension is unavailable; "
                    f"build it with: {build_command()}"
                )

            target_dim = int(u.shape[1])
            B_norm, b_meta = _normalize_variable_tensor(B, target_dim=target_dim)
            C_norm, c_meta = _normalize_variable_tensor(C, target_dim=target_dim)

            D_value = None if D is None else D.contiguous()
            z_value = None if z is None else z.contiguous()
            delta_bias_value = None if delta_bias is None else delta_bias.contiguous()

            out, last_state, x_hist, y_base = module.fwd(
                u.contiguous(),
                delta.contiguous(),
                A.contiguous(),
                B_norm,
                C_norm,
                D_value,
                z_value,
                delta_bias_value,
                bool(delta_softplus),
            )

            empty = u.new_empty(0)
            ctx.save_for_backward(
                u,
                delta,
                A,
                B_norm,
                C_norm,
                x_hist,
                y_base,
                empty if D_value is None else D_value,
                empty if z_value is None else z_value,
                empty if delta_bias_value is None else delta_bias_value,
            )
            ctx.has_D = D is not None
            ctx.has_z = z is not None
            ctx.has_delta_bias = delta_bias is not None
            ctx.delta_softplus = bool(delta_softplus)
            ctx.b_meta = b_meta
            ctx.c_meta = c_meta
            return out, last_state

        @staticmethod
        def backward(ctx, dout: Tensor, dlast_state: Tensor | None) -> tuple[Tensor | None, ...]:
            module = _load_extension(auto_build=False)
            if module is None:
                raise RuntimeError(
                    "native selective-scan CUDA extension became unavailable before backward"
                )

            (
                u,
                delta,
                A,
                B_norm,
                C_norm,
                x_hist,
                y_base,
                D_saved,
                z_saved,
                delta_bias_saved,
            ) = ctx.saved_tensors

            D_value = D_saved if ctx.has_D else None
            z_value = z_saved if ctx.has_z else None
            delta_bias_value = delta_bias_saved if ctx.has_delta_bias else None
            dlast_state_value = (
                torch.zeros_like(x_hist[:, :, 0, :]) if dlast_state is None else dlast_state.contiguous()
            )

            (
                du,
                ddelta,
                dA_full,
                dB_full,
                dC_full,
                dD_full,
                dz,
                ddelta_bias_full,
            ) = module.bwd(
                dout.contiguous(),
                dlast_state_value,
                u,
                delta,
                A,
                B_norm,
                C_norm,
                x_hist,
                y_base,
                D_value,
                z_value,
                delta_bias_value,
                bool(ctx.delta_softplus),
            )

            dA = dA_full.sum(dim=0)
            dB = _reduce_grad_variable_tensor(dB_full, ctx.b_meta)
            dC = _reduce_grad_variable_tensor(dC_full, ctx.c_meta)
            dD = dD_full.sum(dim=0) if ctx.has_D else None
            dz_value = dz if ctx.has_z else None
            ddelta_bias = ddelta_bias_full.sum(dim=0) if ctx.has_delta_bias else None
            return du, ddelta, dA, dB, dC, dD, dz_value, ddelta_bias, None
else:  # pragma: no cover
    class _CudaSelectiveScanFn:  # type: ignore[no-redef]
        @staticmethod
        def apply(*args, **kwargs):
            raise RuntimeError("PyTorch is required for native selective-scan CUDA support")


def _normalize_variable_tensor(value: Tensor, *, target_dim: int) -> tuple[Tensor, dict[str, int | str]]:
    if value.dim() == 3:
        return value.unsqueeze(1).expand(-1, target_dim, -1, -1).contiguous(), {
            "kind": "shared",
            "groups": 1,
            "repeats": target_dim,
        }
    if value.dim() != 4:
        raise ValueError(
            f"expected B/C tensor with shape (batch, d_state, length) or (batch, groups, d_state, length); got {tuple(value.shape)}"
        )
    groups = int(value.shape[1])
    if groups <= 0 or target_dim % groups != 0:
        raise ValueError(f"groups={groups} must divide target_dim={target_dim}")
    repeats = target_dim // groups
    if repeats == 1:
        return value.contiguous(), {"kind": "grouped", "groups": groups, "repeats": repeats}
    return value.repeat_interleave(repeats, dim=1).contiguous(), {
        "kind": "grouped",
        "groups": groups,
        "repeats": repeats,
    }


def _reduce_grad_variable_tensor(grad_full: Tensor, meta: dict[str, int | str]) -> Tensor:
    kind = str(meta["kind"])
    if kind == "shared":
        return grad_full.sum(dim=1)
    groups = int(meta["groups"])
    repeats = int(meta["repeats"])
    if repeats == 1:
        return grad_full
    batch, _, d_state, length = grad_full.shape
    return grad_full.view(batch, groups, repeats, d_state, length).sum(dim=2)


def _load_extension(*, auto_build: bool) -> Any:
    global _LOADED_EXTENSION, _LOAD_ERROR_DETAIL
    if _LOADED_EXTENSION is not None:
        return _LOADED_EXTENSION
    module = _import_optional(extension_name())
    if module is not None:
        _LOAD_ERROR_DETAIL = ""
        _LOADED_EXTENSION = module
        return module
    if not auto_build:
        return module
    autobuild_allowed, _ = _autobuild_allowed()
    if not autobuild_allowed:
        return None
    try:
        _LOADED_EXTENSION = build_extension(verbose=False)
        _LOAD_ERROR_DETAIL = ""
        return _LOADED_EXTENSION
    except Exception as exc:
        _LOAD_ERROR_DETAIL = str(exc).strip() or exc.__class__.__name__
        return None


def _torch_cuda_available() -> bool:
    return bool(torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())


def _import_optional(name: str) -> Any:
    global _LOAD_ERROR_DETAIL
    build_dir = str(build_directory())
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    try:
        return importlib.import_module(name)
    except Exception as exc:
        _LOAD_ERROR_DETAIL = str(exc).strip() or exc.__class__.__name__
        return None


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch with torch.nn.functional is required for native selective-scan CUDA support"
        )


def _autobuild_allowed() -> tuple[bool, str]:
    payload = load_model_artifact(_active_tolbert_model_artifact_path())
    return tolbert_kernel_autobuild_ready(payload)


def _active_tolbert_model_artifact_path() -> Path:
    raw = Path(os.getenv("AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH", "trajectories/tolbert_model/tolbert_model_artifact.json"))
    if raw.is_absolute():
        return raw
    return _repo_root() / raw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _validate_inputs(
    *,
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    z: Tensor | None,
    delta_bias: Tensor | None,
) -> None:
    if not u.is_cuda or not delta.is_cuda or not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("native CUDA selective scan requires CUDA tensors for u, delta, A, B, and C")
    if _SUPPORTED_DTYPES and any(
        tensor.dtype not in _SUPPORTED_DTYPES for tensor in (u, delta, A, B, C)
    ):
        raise ValueError("native CUDA selective scan currently supports only float32 and float64 tensors")
    if u.dim() != 3:
        raise ValueError(f"u must have shape (batch, dim, length); got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(f"delta must match u shape; got {tuple(delta.shape)} vs {tuple(u.shape)}")
    if A.dim() != 2:
        raise ValueError(f"A must have shape (dim, d_state); got {tuple(A.shape)}")
    if A.shape[0] != u.shape[1]:
        raise ValueError(f"A first dimension must equal u dim; got {A.shape[0]} vs {u.shape[1]}")
    if B.dim() not in {3, 4} or C.dim() not in {3, 4}:
        raise ValueError("B and C must have 3 or 4 dimensions")
    if B.shape[0] != u.shape[0] or C.shape[0] != u.shape[0]:
        raise ValueError("B and C batch dimensions must match u")
    if B.shape[-1] != u.shape[-1] or C.shape[-1] != u.shape[-1]:
        raise ValueError("B and C sequence lengths must match u")
    if B.shape[-2] != A.shape[1] or C.shape[-2] != A.shape[1]:
        raise ValueError("B and C d_state dimensions must match A")
    if D is not None and (not D.is_cuda or D.shape != (u.shape[1],) or D.dtype != u.dtype):
        raise ValueError(f"D must be a CUDA tensor with shape ({u.shape[1]},)")
    if z is not None and (not z.is_cuda or z.shape != u.shape or z.dtype != u.dtype):
        raise ValueError("z must be a CUDA tensor matching u")
    if delta_bias is not None and (
        not delta_bias.is_cuda or delta_bias.shape != (u.shape[1],) or delta_bias.dtype != u.dtype
    ):
        raise ValueError(f"delta_bias must be a CUDA tensor with shape ({u.shape[1]},)")
