from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util


@dataclass(frozen=True, slots=True)
class HybridRuntimeStatus:
    available: bool
    torch_namespace_present: bool
    torch_nn_present: bool
    torch_nn_functional_present: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def hybrid_runtime_status() -> HybridRuntimeStatus:
    torch_spec = _safe_find_spec("torch")
    nn_spec = _safe_find_spec("torch.nn")
    nn_functional_spec = _safe_find_spec("torch.nn.functional")
    torch_namespace_present = torch_spec is not None
    torch_nn_present = nn_spec is not None
    torch_nn_functional_present = nn_functional_spec is not None
    available = bool(torch_namespace_present and torch_nn_present and torch_nn_functional_present)
    if available:
        reason = ""
    elif not torch_namespace_present:
        reason = "PyTorch is not installed"
    elif not torch_nn_present:
        reason = "PyTorch namespace is present but torch.nn is unavailable"
    else:
        reason = "PyTorch namespace is present but torch.nn.functional is unavailable"
    return HybridRuntimeStatus(
        available=available,
        torch_namespace_present=torch_namespace_present,
        torch_nn_present=torch_nn_present,
        torch_nn_functional_present=torch_nn_functional_present,
        reason=reason,
    )


def _safe_find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
