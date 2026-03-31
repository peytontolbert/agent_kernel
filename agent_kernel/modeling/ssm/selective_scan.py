from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover - exercised only in reduced environments
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from .kernels import cuda_selective_scan_status, selective_scan_cuda_fn


@dataclass(frozen=True, slots=True)
class SelectiveScanResult:
    output: Tensor
    last_state: Tensor | None = None
    backend: str = "python_ref"


def selective_scan(
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
    prefer_python_ref: bool = False,
    initial_state: Tensor | None = None,
) -> SelectiveScanResult:
    _require_torch()
    if prefer_python_ref:
        return selective_scan_ref(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
            initial_state=initial_state,
        )
    cuda_status = cuda_selective_scan_status()
    if u.device.type == "cuda" and initial_state is None:
        if not cuda_status.available:
            raise RuntimeError(
                cuda_status.detail
                or "native selective-scan CUDA extension is unavailable; "
                "build it explicitly or set prefer_python_ref=True"
            )
        output, last_state = selective_scan_cuda_fn(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
        return SelectiveScanResult(
            output=output,
            last_state=last_state,
            backend=f"native_cuda:{cuda_status.source}",
        )
    ref = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        initial_state=initial_state,
    )
    return SelectiveScanResult(output=ref.output, last_state=ref.last_state, backend="python_ref")


def selective_scan_ref(
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
    initial_state: Tensor | None = None,
) -> SelectiveScanResult:
    """Pure PyTorch selective scan.

    This is the development and testing fallback for future TOLBERT-family SSM
    work. It intentionally supports the real-valued layout used by the Mamba
    selective scan interface:

    - `u`: `(batch, dim, length)`
    - `delta`: `(batch, dim, length)`
    - `A`: `(dim, d_state)`
    - `B`: `(batch, d_state, length)` or `(batch, groups, d_state, length)`
    - `C`: `(batch, d_state, length)` or `(batch, groups, d_state, length)`
    """
    _require_torch()
    _validate_selective_scan_inputs(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        initial_state=initial_state,
    )

    dtype_in = u.dtype
    work_dtype = torch.float32 if not torch.is_floating_point(u) else u.dtype
    u_work = u.to(work_dtype)
    delta_work = delta.to(work_dtype)
    A_work = A.to(work_dtype)
    B_work = _normalize_variable_tensor(B, target_dim=u.shape[1], dtype=work_dtype)
    C_work = _normalize_variable_tensor(C, target_dim=u.shape[1], dtype=work_dtype)

    if delta_bias is not None:
        delta_work = delta_work + delta_bias.to(work_dtype).unsqueeze(-1)
    if delta_softplus:
        delta_work = F.softplus(delta_work)

    batch, dim, length = u_work.shape
    d_state = A_work.shape[1]
    x = (
        initial_state.to(device=u.device, dtype=work_dtype).clone()
        if initial_state is not None
        else torch.zeros((batch, dim, d_state), device=u.device, dtype=work_dtype)
    )
    outputs: list[Tensor] = []

    for index in range(length):
        delta_t = delta_work[:, :, index]
        u_t = u_work[:, :, index]
        b_t = B_work[:, :, :, index]
        c_t = C_work[:, :, :, index]
        deltaA = torch.exp(delta_t.unsqueeze(-1) * A_work.unsqueeze(0))
        deltaB_u = delta_t.unsqueeze(-1) * b_t * u_t.unsqueeze(-1)
        x = deltaA * x + deltaB_u
        y_t = torch.sum(x * c_t, dim=-1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + u_work * D.to(work_dtype).unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z.to(work_dtype))
    last_state = x if return_last_state else None
    return SelectiveScanResult(output=y.to(dtype_in), last_state=last_state, backend="python_ref")


def _validate_selective_scan_inputs(
    *,
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    z: Tensor | None,
    delta_bias: Tensor | None,
    initial_state: Tensor | None,
) -> None:
    if u.dim() != 3:
        raise ValueError(f"u must have shape (batch, dim, length); got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(f"delta must match u shape; got {tuple(delta.shape)} vs {tuple(u.shape)}")
    if A.dim() != 2:
        raise ValueError(f"A must have shape (dim, d_state); got {tuple(A.shape)}")
    if A.shape[0] != u.shape[1]:
        raise ValueError(f"A first dimension must equal u dim; got {A.shape[0]} vs {u.shape[1]}")
    if B.dim() not in {3, 4}:
        raise ValueError(f"B must have shape (batch, d_state, length) or (batch, groups, d_state, length); got {tuple(B.shape)}")
    if C.dim() not in {3, 4}:
        raise ValueError(f"C must have shape (batch, d_state, length) or (batch, groups, d_state, length); got {tuple(C.shape)}")
    if B.shape[0] != u.shape[0] or C.shape[0] != u.shape[0]:
        raise ValueError("B and C batch dimensions must match u")
    if B.shape[-1] != u.shape[-1] or C.shape[-1] != u.shape[-1]:
        raise ValueError("B and C sequence lengths must match u")
    if B.shape[-2] != A.shape[1] or C.shape[-2] != A.shape[1]:
        raise ValueError("B and C d_state dimensions must match A")
    if D is not None and D.shape != (u.shape[1],):
        raise ValueError(f"D must have shape ({u.shape[1]},); got {tuple(D.shape)}")
    if z is not None and z.shape != u.shape:
        raise ValueError(f"z must match u shape; got {tuple(z.shape)} vs {tuple(u.shape)}")
    if delta_bias is not None and delta_bias.shape != (u.shape[1],):
        raise ValueError(f"delta_bias must have shape ({u.shape[1]},); got {tuple(delta_bias.shape)}")
    if initial_state is not None and initial_state.shape != (u.shape[0], u.shape[1], A.shape[1]):
        raise ValueError(
            "initial_state must have shape "
            f"({u.shape[0]}, {u.shape[1]}, {A.shape[1]}); got {tuple(initial_state.shape)}"
        )
    if A.is_complex():
        raise NotImplementedError("Complex A is not yet supported in agent_kernel selective_scan_ref")


def _normalize_variable_tensor(value: Tensor, *, target_dim: int, dtype: torch.dtype) -> Tensor:
    if value.dim() == 3:
        return value.to(dtype).unsqueeze(1).expand(-1, target_dim, -1, -1)
    groups = value.shape[1]
    if groups <= 0 or target_dim % groups != 0:
        raise ValueError(f"groups={groups} must divide target_dim={target_dim}")
    repeats = math.ceil(target_dim / groups)
    expanded = value.to(dtype).repeat_interleave(repeats, dim=1)
    return expanded[:, :target_dim]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch with torch.nn.functional is required for agent_kernel.modeling.ssm selective_scan"
        )
