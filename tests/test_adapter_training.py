from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from agent_kernel.modeling.adapter_training import InjectedLoRAState, LoRALinear


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 5),
        )
        self.head = torch.nn.Linear(5, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


def test_injected_lora_wraps_selected_linear_modules_and_exports_adapters() -> None:
    model = _ToyModel()

    lora_state = InjectedLoRAState(
        model,
        rank=2,
        alpha=4.0,
        module_filter=lambda module_path, module: module_path.startswith("encoder.") and isinstance(module, torch.nn.Linear),
    )

    assert isinstance(model.encoder[0], LoRALinear)
    assert isinstance(model.encoder[2], LoRALinear)
    assert not isinstance(model.head, LoRALinear)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)

    output = model(torch.randn(3, 4))
    assert output.shape == (3, 2)

    adapters, dense = lora_state.mutation_components()
    assert sorted(adapters) == ["encoder.0.weight", "encoder.2.weight"]
    assert dense == {}
    assert lora_state.stats()["adapter_key_count"] == 2
    assert lora_state.stats()["dense_delta_key_count"] == 0


class _HybridLikeToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.log_a = torch.nn.Parameter(torch.zeros(4, 2))
        self.d_skip = torch.nn.Parameter(torch.zeros(4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(inputs)
        return hidden + self.d_skip + self.log_a.mean(dim=-1)


def test_injected_lora_can_train_structured_parameter_adapters_without_functional_call() -> None:
    model = _HybridLikeToyModel()

    lora_state = InjectedLoRAState(
        model,
        rank=2,
        alpha=4.0,
        module_filter=lambda module_path, module: module_path == "encoder" and isinstance(module, torch.nn.Linear),
        direct_parameter_filter=lambda parameter_path, parameter: parameter_path in {"log_a", "d_skip"},
    )

    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert trainable == [
        "encoder.lora_a",
        "encoder.lora_b",
        "parametrizations.d_skip.0.coefficients",
        "parametrizations.log_a.0.left_factor",
        "parametrizations.log_a.0.right_factor",
    ]

    output = model(torch.randn(3, 4))
    assert output.shape == (3, 4)

    adapters, dense = lora_state.mutation_components()
    assert sorted(adapters) == ["d_skip", "encoder.weight", "log_a"]
    assert dense == {}
    assert lora_state.stats()["adapter_key_count"] == 3
    assert lora_state.stats()["dense_delta_key_count"] == 0
    assert lora_state.stats()["structured_parameter_adapter_key_count"] == 2


class _WorldTransitionToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.world_transition_logits = torch.nn.Parameter(torch.zeros(4, 4))


def test_injected_lora_uses_structured_transition_adapter_for_world_transition_logits() -> None:
    model = _WorldTransitionToyModel()

    lora_state = InjectedLoRAState(
        model,
        rank=2,
        direct_parameter_filter=lambda parameter_path, parameter: parameter_path == "world_transition_logits",
    )

    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert trainable == [
        "parametrizations.world_transition_logits.0.dest_logits",
        "parametrizations.world_transition_logits.0.source_logits",
        "parametrizations.world_transition_logits.0.stay_logits",
    ]

    adapters, dense = lora_state.mutation_components()
    assert dense == {}
    assert adapters["world_transition_logits"]["kind"] == "structured_transition_adapter"
    assert adapters["world_transition_logits"]["transition_kind"] == "source_dest_stay_v1"
