from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - reduced environments
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from .kernels import causal_belief_scan_cuda_fn, cuda_belief_scan_status


@dataclass(frozen=True, slots=True)
class CausalBeliefScanResult:
    beliefs: Tensor
    final_log_belief: Tensor
    backend: str


StructuredTransitionSpec = dict[str, object]


def causal_belief_scan_ref(
    local_logits: Tensor,
    transition_log_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    *,
    transition_structure: StructuredTransitionSpec | None = None,
    transition_gate: float | Tensor = 1.0,
    chunk_size: int = 128,
) -> CausalBeliefScanResult:
    _require_torch()
    _validate_inputs(
        local_logits=local_logits,
        transition_log_probs=transition_log_probs,
        transition_context=transition_context,
        initial_log_belief=initial_log_belief,
        transition_structure=transition_structure,
        chunk_size=chunk_size,
    )

    batch_size, seq_len, num_states = [int(value) for value in local_logits.shape]
    beliefs = torch.empty_like(local_logits)
    prev = initial_log_belief
    gate = (
        transition_gate.to(device=local_logits.device, dtype=local_logits.dtype)
        if torch is not None and isinstance(transition_gate, torch.Tensor)
        else local_logits.new_tensor(float(transition_gate))
    )
    if seq_len == 0:
        return CausalBeliefScanResult(
            beliefs=beliefs,
            final_log_belief=initial_log_belief.clone(),
            backend="python_ref",
        )
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(seq_len, chunk_start + chunk_size)
        for pos in range(chunk_start, chunk_end):
            if transition_structure is not None:
                pred = _structured_transition_prediction(
                    prev=prev,
                    transition_structure=transition_structure,
                    dtype=local_logits.dtype,
                )
            elif transition_log_probs.dim() == 1:
                pred = prev + transition_log_probs.unsqueeze(0)
            else:
                pred = torch.logsumexp(
                    prev.unsqueeze(-1) + transition_log_probs.unsqueeze(0),
                    dim=1,
                )
            obs = local_logits[:, pos, :] + gate * (pred + transition_context[:, pos, :])
            q = obs - torch.logsumexp(obs, dim=-1, keepdim=True)
            beliefs[:, pos, :] = q
            prev = q
    return CausalBeliefScanResult(
        beliefs=beliefs,
        final_log_belief=prev,
        backend="python_ref",
    )


def causal_belief_scan(
    local_logits: Tensor,
    transition_log_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    *,
    transition_structure: StructuredTransitionSpec | None = None,
    transition_gate: float | Tensor = 1.0,
    chunk_size: int = 128,
    prefer_python_ref: bool = False,
    auto_build_cuda: bool = False,
) -> CausalBeliefScanResult:
    _require_torch()
    _validate_inputs(
        local_logits=local_logits,
        transition_log_probs=transition_log_probs,
        transition_context=transition_context,
        initial_log_belief=initial_log_belief,
        transition_structure=transition_structure,
        chunk_size=chunk_size,
    )
    if prefer_python_ref or local_logits.device.type != "cuda":
        return causal_belief_scan_ref(
            local_logits,
            transition_log_probs,
            transition_context,
            initial_log_belief,
            transition_structure=transition_structure,
            transition_gate=transition_gate,
            chunk_size=chunk_size,
        )
    status = cuda_belief_scan_status()
    if not status.available and not auto_build_cuda:
        raise RuntimeError(
            status.detail
            or "native causal-belief CUDA extension is unavailable; "
            "build it explicitly or set prefer_python_ref=True"
        )
    beliefs, final_log_belief = causal_belief_scan_cuda_fn(
        local_logits,
        _materialize_transition_log_probs(
            transition_log_probs=transition_log_probs,
            transition_structure=transition_structure,
            dtype=local_logits.dtype,
        ),
        transition_context,
        initial_log_belief,
        transition_gate=transition_gate,
        chunk_size=chunk_size,
        auto_build=auto_build_cuda,
    )
    return CausalBeliefScanResult(
        beliefs=beliefs,
        final_log_belief=final_log_belief,
        backend="cuda_native",
    )


def _validate_inputs(
    *,
    local_logits: Tensor,
    transition_log_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_structure: StructuredTransitionSpec | None,
    chunk_size: int,
) -> None:
    if local_logits.dim() != 3:
        raise ValueError(
            f"expected local_logits with shape (batch, length, states); got {tuple(local_logits.shape)}"
        )
    if transition_log_probs.dim() not in {1, 2}:
        raise ValueError(
            "expected transition_log_probs with shape (states,) or (states, states); "
            f"got {tuple(transition_log_probs.shape)}"
        )
    if transition_context.shape != local_logits.shape:
        raise ValueError(
            f"transition_context must match local_logits; got {tuple(transition_context.shape)} "
            f"vs {tuple(local_logits.shape)}"
        )
    if initial_log_belief.dim() != 2:
        raise ValueError(
            f"expected initial_log_belief with shape (batch, states); got {tuple(initial_log_belief.shape)}"
        )
    batch_size, _, num_states = [int(value) for value in local_logits.shape]
    if transition_structure is None:
        if transition_log_probs.dim() == 1:
            if transition_log_probs.shape != (num_states,):
                raise ValueError(
                    f"transition_log_probs must have shape ({num_states},); got {tuple(transition_log_probs.shape)}"
                )
        elif transition_log_probs.shape != (num_states, num_states):
            raise ValueError(
                f"transition_log_probs must have shape ({num_states}, {num_states}); "
                f"got {tuple(transition_log_probs.shape)}"
            )
    else:
        _validate_transition_structure(transition_structure=transition_structure, num_states=num_states)
    if initial_log_belief.shape != (batch_size, num_states):
        raise ValueError(
            f"initial_log_belief must have shape ({batch_size}, {num_states}); "
            f"got {tuple(initial_log_belief.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for world belief-scan modeling")


def _validate_transition_structure(*, transition_structure: StructuredTransitionSpec, num_states: int) -> None:
    required_tensor_fields = (
        "base_transition_logits",
        "source_logits",
        "dest_logits",
        "stay_logits",
    )
    for field_name in required_tensor_fields:
        value = transition_structure.get(field_name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"transition_structure[{field_name!r}] must be a torch.Tensor")
    base_transition_logits = transition_structure["base_transition_logits"]
    source_logits = transition_structure["source_logits"]
    dest_logits = transition_structure["dest_logits"]
    stay_logits = transition_structure["stay_logits"]
    if base_transition_logits.shape != (num_states, num_states):
        raise ValueError(
            f"transition_structure['base_transition_logits'] must have shape ({num_states}, {num_states}); "
            f"got {tuple(base_transition_logits.shape)}"
        )
    if source_logits.dim() != 2 or source_logits.shape[0] != num_states:
        raise ValueError(
            f"transition_structure['source_logits'] must have shape ({num_states}, rank); "
            f"got {tuple(source_logits.shape)}"
        )
    if dest_logits.dim() != 2 or dest_logits.shape[1] != num_states or dest_logits.shape[0] != source_logits.shape[1]:
        raise ValueError(
            "transition_structure['dest_logits'] must have shape (rank, states) matching source rank; "
            f"got {tuple(dest_logits.shape)}"
        )
    if stay_logits.shape != (num_states,):
        raise ValueError(
            f"transition_structure['stay_logits'] must have shape ({num_states},); got {tuple(stay_logits.shape)}"
        )


def _structured_transition_prediction(
    *,
    prev: Tensor,
    transition_structure: StructuredTransitionSpec,
    dtype: torch.dtype,
) -> Tensor:
    family = str(transition_structure.get("family", "banded")).strip().lower() or "banded"
    if family == "diag":
        return prev + _structured_transition_diag_values(transition_structure=transition_structure, dtype=dtype).unsqueeze(0)
    num_states = int(prev.shape[-1])
    pred: Tensor | None = None
    for source_index in range(num_states):
        row_log_probs = _structured_transition_log_probs_row(
            transition_structure=transition_structure,
            source_index=source_index,
            dtype=dtype,
        ).unsqueeze(0)
        term = prev[:, source_index].unsqueeze(-1) + row_log_probs
        pred = term if pred is None else torch.logaddexp(pred, term)
    if pred is None:
        return prev
    return pred


def _structured_transition_log_probs_row(
    *,
    transition_structure: StructuredTransitionSpec,
    source_index: int,
    dtype: torch.dtype,
) -> Tensor:
    family = str(transition_structure.get("family", "banded")).strip().lower() or "banded"
    bandwidth = max(0, int(transition_structure.get("bandwidth", 0) or 0))
    base_transition_logits = transition_structure["base_transition_logits"]
    source_logits = transition_structure["source_logits"]
    dest_logits = transition_structure["dest_logits"]
    stay_logits = transition_structure["stay_logits"]
    row_logits = base_transition_logits[source_index].to(dtype=dtype)
    row_logits = row_logits + torch.matmul(source_logits[source_index].to(dtype=dtype), dest_logits.to(dtype=dtype))
    row_logits = row_logits.clone()
    row_logits[source_index] = row_logits[source_index] + stay_logits[source_index].to(dtype=dtype)
    if family == "banded":
        positions = torch.arange(int(row_logits.shape[0]), device=row_logits.device)
        mask = (positions - int(source_index)).abs() <= bandwidth
        row_logits = row_logits.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(row_logits, dim=-1)


def _materialize_transition_log_probs(
    *,
    transition_log_probs: Tensor,
    transition_structure: StructuredTransitionSpec | None,
    dtype: torch.dtype,
) -> Tensor:
    if transition_structure is None:
        return transition_log_probs
    family = str(transition_structure.get("family", "banded")).strip().lower() or "banded"
    if family == "diag":
        return _structured_transition_diag_values(transition_structure=transition_structure, dtype=dtype)
    num_states = int(transition_log_probs.shape[0])
    rows = [
        _structured_transition_log_probs_row(
            transition_structure=transition_structure,
            source_index=source_index,
            dtype=dtype,
        )
        for source_index in range(num_states)
    ]
    return torch.stack(rows, dim=0)


def _structured_transition_diag_values(
    *,
    transition_structure: StructuredTransitionSpec,
    dtype: torch.dtype,
) -> Tensor:
    base_transition_logits = transition_structure["base_transition_logits"]
    source_logits = transition_structure["source_logits"]
    dest_logits = transition_structure["dest_logits"]
    stay_logits = transition_structure["stay_logits"]
    diagonal = torch.diagonal(base_transition_logits, 0).to(dtype=dtype)
    interaction = (source_logits.to(dtype=dtype) * dest_logits.transpose(0, 1).to(dtype=dtype)).sum(dim=-1)
    return diagonal + interaction + stay_logits.to(dtype=dtype)
