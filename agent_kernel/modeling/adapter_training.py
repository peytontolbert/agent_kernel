from __future__ import annotations

import hashlib
from typing import Any

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.nn.utils import parametrize
    try:
        from torch.func import functional_call as _functional_call
    except Exception:  # pragma: no cover
        from torch.nn.utils.stateless import functional_call as _functional_call  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    parametrize = None  # type: ignore[assignment]
    _functional_call = None  # type: ignore[assignment]


_BASE_MODULE = nn.Module if nn is not None else object


class LowRankAdapterState(_BASE_MODULE):  # type: ignore[misc]
    def __init__(
        self,
        *,
        base_state_dict: dict[str, object],
        rank: int = 8,
        max_direct_elements: int = 4096,
    ) -> None:
        _require_torch()
        super().__init__()
        self.rank = max(1, int(rank))
        self.max_direct_elements = max(0, int(max_direct_elements))
        self.left_factors = nn.ParameterDict()
        self.right_factors = nn.ParameterDict()
        self.direct_deltas = nn.ParameterDict()
        self._adapter_specs: dict[str, dict[str, object]] = {}
        self._direct_specs: dict[str, dict[str, object]] = {}
        for key, value in base_state_dict.items():
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                continue
            safe_key = _safe_parameter_key(key)
            if value.ndim >= 2:
                matrix_rows = int(value.shape[0])
                matrix_cols = int(value.numel() // max(1, matrix_rows))
                effective_rank = min(self.rank, matrix_rows, matrix_cols)
                adapter_size = matrix_rows * effective_rank + effective_rank * matrix_cols
                if effective_rank > 0 and adapter_size < int(value.numel()):
                    self.left_factors[safe_key] = nn.Parameter(
                        torch.zeros(
                            matrix_rows,
                            effective_rank,
                            dtype=value.dtype,
                            device=value.device,
                        )
                    )
                    self.right_factors[safe_key] = nn.Parameter(
                        torch.zeros(
                            effective_rank,
                            matrix_cols,
                            dtype=value.dtype,
                            device=value.device,
                        )
                    )
                    self._adapter_specs[key] = {
                        "safe_key": safe_key,
                        "original_shape": list(value.shape),
                        "rank": effective_rank,
                    }
                    continue
            if int(value.numel()) <= self.max_direct_elements:
                self.direct_deltas[safe_key] = nn.Parameter(torch.zeros_like(value))
                self._direct_specs[key] = {
                    "safe_key": safe_key,
                    "original_shape": list(value.shape),
                }

    def effective_state_dict(self, base_state_dict: dict[str, object]) -> dict[str, object]:
        _require_torch()
        effective: dict[str, object] = {}
        for key, value in base_state_dict.items():
            if not isinstance(value, torch.Tensor):
                effective[key] = value
                continue
            tensor = value
            adapter_spec = self._adapter_specs.get(key)
            if adapter_spec is not None:
                safe_key = str(adapter_spec["safe_key"])
                left = self.left_factors[safe_key]
                right = self.right_factors[safe_key]
                tensor = tensor + (left @ right).reshape(tuple(int(item) for item in adapter_spec["original_shape"]))
            direct_spec = self._direct_specs.get(key)
            if direct_spec is not None:
                safe_key = str(direct_spec["safe_key"])
                tensor = tensor + self.direct_deltas[safe_key]
            effective[key] = tensor
        return effective

    def mutation_components(self) -> tuple[dict[str, object], dict[str, object]]:
        _require_torch()
        adapters: dict[str, object] = {}
        dense_deltas: dict[str, object] = {}
        for key, spec in self._adapter_specs.items():
            safe_key = str(spec["safe_key"])
            adapters[key] = {
                "kind": "low_rank_adapter",
                "original_shape": list(spec["original_shape"]),
                "rank": int(spec["rank"]),
                "left_factor": self.left_factors[safe_key].detach().cpu(),
                "right_factor": self.right_factors[safe_key].detach().cpu(),
            }
        for key, spec in self._direct_specs.items():
            safe_key = str(spec["safe_key"])
            dense_deltas[key] = self.direct_deltas[safe_key].detach().cpu()
        return adapters, dense_deltas

    def stats(self) -> dict[str, int]:
        return {
            "adapter_key_count": len(self._adapter_specs),
            "dense_delta_key_count": len(self._direct_specs),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


class LoRALinear(_BASE_MODULE):  # type: ignore[misc]
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        _require_torch()
        super().__init__()
        self.base_layer = base_layer
        self.rank = max(1, min(int(rank), base_layer.out_features, base_layer.in_features))
        self.alpha = float(alpha if alpha is not None else self.rank)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                dtype=base_layer.weight.dtype,
                device=base_layer.weight.device,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                dtype=base_layer.weight.dtype,
                device=base_layer.weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        lora_hidden = F.linear(self.dropout(inputs), self.lora_a)
        lora_output = F.linear(lora_hidden, self.lora_b)
        return base + (lora_output * self.scaling)

    def adapter_payload(self, *, original_shape: list[int] | None = None) -> dict[str, object]:
        return {
            "kind": "low_rank_adapter",
            "original_shape": list(original_shape or self.base_layer.weight.shape),
            "rank": self.rank,
            "left_factor": (self.lora_b.detach().cpu() * self.scaling),
            "right_factor": self.lora_a.detach().cpu(),
        }


class LowRankTensorParametrization(_BASE_MODULE):  # type: ignore[misc]
    def __init__(self, base_parameter: torch.Tensor, *, rank: int = 8) -> None:
        _require_torch()
        super().__init__()
        if base_parameter.ndim < 2:
            raise ValueError("LowRankTensorParametrization requires a tensor with ndim >= 2")
        rows = int(base_parameter.shape[0])
        cols = int(base_parameter.numel() // max(1, rows))
        self.original_shape = tuple(int(item) for item in base_parameter.shape)
        self.rank = max(1, min(int(rank), rows, cols))
        self.left_factor = nn.Parameter(
            torch.empty(
                rows,
                self.rank,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )
        self.right_factor = nn.Parameter(
            torch.zeros(
                self.rank,
                cols,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )
        nn.init.kaiming_uniform_(self.left_factor, a=5**0.5)

    def forward(self, base_value: torch.Tensor) -> torch.Tensor:
        delta = (self.left_factor @ self.right_factor).reshape(self.original_shape)
        return base_value + delta.to(dtype=base_value.dtype, device=base_value.device)

    def adapter_payload(self) -> dict[str, object]:
        return {
            "kind": "low_rank_adapter",
            "original_shape": list(self.original_shape),
            "rank": self.rank,
            "left_factor": self.left_factor.detach().cpu(),
            "right_factor": self.right_factor.detach().cpu(),
        }


class AdditiveDeltaParametrization(_BASE_MODULE):  # type: ignore[misc]
    def __init__(self, base_parameter: torch.Tensor) -> None:
        _require_torch()
        super().__init__()
        self.delta = nn.Parameter(torch.zeros_like(base_parameter))

    def forward(self, base_value: torch.Tensor) -> torch.Tensor:
        return base_value + self.delta.to(dtype=base_value.dtype, device=base_value.device)


class FixedBasisVectorParametrization(_BASE_MODULE):  # type: ignore[misc]
    def __init__(self, base_parameter: torch.Tensor, *, rank: int = 8) -> None:
        _require_torch()
        super().__init__()
        self.original_shape = tuple(int(item) for item in base_parameter.shape)
        self.numel = int(base_parameter.numel())
        self.rank = _effective_vector_basis_rank(numel=self.numel, requested_rank=rank)
        basis = _cosine_basis(
            numel=self.numel,
            rank=self.rank,
            device=base_parameter.device,
            dtype=base_parameter.dtype,
        )
        self.register_buffer("basis", basis, persistent=False)
        self.coefficients = nn.Parameter(
            torch.zeros(
                self.rank,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )

    def forward(self, base_value: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.coefficients.to(dtype=base_value.dtype), self.basis.to(dtype=base_value.dtype))
        return base_value + delta.reshape(self.original_shape).to(device=base_value.device, dtype=base_value.dtype)

    def adapter_payload(self) -> dict[str, object]:
        return {
            "kind": "fixed_basis_adapter",
            "basis_kind": "cosine_v1",
            "original_shape": list(self.original_shape),
            "basis_rank": self.rank,
            "coefficients": self.coefficients.detach().cpu(),
        }


class StructuredTransitionParametrization(_BASE_MODULE):  # type: ignore[misc]
    def __init__(self, base_parameter: torch.Tensor, *, rank: int = 8) -> None:
        _require_torch()
        super().__init__()
        if base_parameter.ndim != 2 or int(base_parameter.shape[0]) != int(base_parameter.shape[1]):
            raise ValueError("StructuredTransitionParametrization requires a square transition matrix")
        self.num_states = int(base_parameter.shape[0])
        self.rank = max(1, min(int(rank), self.num_states))
        self.source_logits = nn.Parameter(
            torch.zeros(
                self.num_states,
                self.rank,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )
        self.dest_logits = nn.Parameter(
            torch.zeros(
                self.rank,
                self.num_states,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )
        self.stay_logits = nn.Parameter(
            torch.zeros(
                self.num_states,
                dtype=base_parameter.dtype,
                device=base_parameter.device,
            )
        )

    def forward(self, base_value: torch.Tensor) -> torch.Tensor:
        delta = self.source_logits @ self.dest_logits
        delta = delta + torch.diag(self.stay_logits)
        return base_value + delta.to(dtype=base_value.dtype, device=base_value.device)

    def adapter_payload(self) -> dict[str, object]:
        return {
            "kind": "structured_transition_adapter",
            "transition_kind": "source_dest_stay_v1",
            "original_shape": [self.num_states, self.num_states],
            "rank": self.rank,
            "source_logits": self.source_logits.detach().cpu(),
            "dest_logits": self.dest_logits.detach().cpu(),
            "stay_logits": self.stay_logits.detach().cpu(),
        }


class InjectedLoRAState:
    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        module_filter: Any | None = None,
        direct_parameter_filter: Any | None = None,
    ) -> None:
        _require_torch()
        self.model = model
        self.rank = max(1, int(rank))
        self.alpha = alpha
        self.dropout = float(dropout)
        self.module_filter = module_filter
        self.direct_parameter_filter = direct_parameter_filter
        self.replaced_modules: dict[str, LoRALinear] = {}
        self.parameter_parametrizations: dict[str, nn.Module] = {}
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._inject(module=self.model, prefix="")
        self._inject_direct_parameters(module=self.model, prefix="")

    def parameters(self):
        for module in self.replaced_modules.values():
            for parameter in module.parameters():
                if parameter.requires_grad:
                    yield parameter
        for parametrization in self.parameter_parametrizations.values():
            for parameter in parametrization.parameters():
                if parameter.requires_grad:
                    yield parameter

    def mutation_components(self) -> tuple[dict[str, object], dict[str, object]]:
        adapters: dict[str, object] = {}
        dense_deltas: dict[str, object] = {}
        for module_path, module in self.replaced_modules.items():
            adapters[f"{module_path}.weight"] = module.adapter_payload()
        for parameter_path, parametrization in self.parameter_parametrizations.items():
            payload = getattr(parametrization, "adapter_payload", lambda: None)()
            if isinstance(payload, dict):
                adapters[parameter_path] = payload
                continue
            delta = getattr(parametrization, "delta", None)
            if isinstance(delta, torch.Tensor):
                dense_deltas[parameter_path] = delta.detach().cpu()
        return adapters, dense_deltas

    def stats(self) -> dict[str, int]:
        dense_delta_count = sum(
            1
            for parametrization in self.parameter_parametrizations.values()
            if isinstance(getattr(parametrization, "delta", None), torch.Tensor)
            and not isinstance(getattr(parametrization, "adapter_payload", lambda: None)(), dict)
        )
        return {
            "adapter_key_count": len(self.replaced_modules) + len(self.parameter_parametrizations) - dense_delta_count,
            "dense_delta_key_count": dense_delta_count,
            "structured_parameter_adapter_key_count": len(self.parameter_parametrizations) - dense_delta_count,
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def _inject(self, *, module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and self._should_wrap(child_path, child):
                wrapped = LoRALinear(
                    child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                setattr(module, child_name, wrapped)
                self.replaced_modules[child_path] = wrapped
                continue
            self._inject(module=child, prefix=child_path)

    def _inject_direct_parameters(self, *, module: nn.Module, prefix: str) -> None:
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            parameter_path = f"{prefix}.{parameter_name}" if prefix else parameter_name
            if not self._should_direct_wrap(parameter_path, parameter):
                continue
            parametrization = _build_parameter_parametrization(
                parameter_path=parameter_path,
                base_parameter=parameter.detach(),
                rank=self.rank,
            )
            parametrize.register_parametrization(module, parameter_name, parametrization)
            self.parameter_parametrizations[parameter_path] = parametrization
        for child_name, child in list(module.named_children()):
            if child_name == "parametrizations":
                continue
            child_path = f"{prefix}.{child_name}" if prefix else child_name
            self._inject_direct_parameters(module=child, prefix=child_path)

    def _should_wrap(self, module_path: str, module: nn.Module) -> bool:
        if self.module_filter is None:
            return True
        return bool(self.module_filter(module_path, module))

    def _should_direct_wrap(self, parameter_path: str, parameter: nn.Parameter) -> bool:
        if self.direct_parameter_filter is None:
            return False
        return bool(self.direct_parameter_filter(parameter_path, parameter))


def base_state_for_model(model: nn.Module) -> dict[str, object]:
    _require_torch()
    state: dict[str, object] = {}
    for key, value in model.named_parameters():
        state[key] = value.detach().clone()
    for key, value in model.named_buffers():
        state[key] = value.detach().clone()
    return state


def functional_model_call(
    model: nn.Module,
    *,
    base_state_dict: dict[str, object],
    adapter_state: LowRankAdapterState | None,
    kwargs: dict[str, object],
) -> Any:
    _require_torch()
    if adapter_state is None:
        return model(**kwargs)
    return _functional_call(model, adapter_state.effective_state_dict(base_state_dict), (), kwargs)


def _safe_parameter_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _build_parameter_parametrization(*, parameter_path: str, base_parameter: torch.Tensor, rank: int) -> nn.Module:
    if parameter_path == "world_transition_logits":
        return StructuredTransitionParametrization(base_parameter, rank=rank)
    if base_parameter.ndim >= 2:
        return LowRankTensorParametrization(base_parameter, rank=rank)
    if base_parameter.ndim in {0, 1}:
        return FixedBasisVectorParametrization(base_parameter, rank=rank)
    return AdditiveDeltaParametrization(base_parameter)


def _effective_vector_basis_rank(*, numel: int, requested_rank: int) -> int:
    if numel <= 1:
        return 1
    return max(1, min(int(requested_rank), max(1, numel // 4)))


def _cosine_basis(*, numel: int, rank: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if numel <= 0:
        return torch.empty((0, 0), device=device, dtype=dtype)
    positions = torch.arange(numel, device=device, dtype=torch.float32) + 0.5
    rows = []
    for basis_index in range(max(1, rank)):
        row = torch.cos(torch.pi * float(basis_index) * positions / float(numel))
        row = row / row.norm().clamp_min(1.0e-8)
        rows.append(row)
    return torch.stack(rows, dim=0).to(dtype=dtype)


def _require_torch() -> None:
    if torch is None or nn is None or F is None or _functional_call is None or parametrize is None:
        raise RuntimeError("Full PyTorch is required for adapter-native Tolbert training")
