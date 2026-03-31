from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def create_tolbert_checkpoint_delta(
    *,
    parent_checkpoint_path: Path,
    child_checkpoint_path: Path,
    delta_output_path: Path,
) -> dict[str, Any]:
    _require_torch()
    parent_payload = _load_checkpoint_payload(parent_checkpoint_path)
    child_payload = _load_checkpoint_payload(child_checkpoint_path)
    parent_state = _state_dict(parent_payload)
    child_state = _state_dict(child_payload)
    adapter_state: dict[str, object] = {}
    delta_state: dict[str, object] = {}
    override_state: dict[str, object] = {}
    changed_keys: list[str] = []
    unchanged_keys = 0
    for key, child_value in child_state.items():
        parent_value = parent_state.get(key)
        if (
            isinstance(parent_value, torch.Tensor)
            and isinstance(child_value, torch.Tensor)
            and parent_value.shape == child_value.shape
            and parent_value.dtype == child_value.dtype
            and torch.is_floating_point(parent_value)
        ):
            child_cpu = child_value.detach().cpu()
            parent_cpu = parent_value.detach().cpu()
            if torch.equal(parent_cpu, child_cpu):
                unchanged_keys += 1
                continue
            delta_tensor = child_cpu - parent_cpu
            adapter_payload = _low_rank_adapter_payload(delta_tensor)
            if adapter_payload is not None:
                adapter_state[key] = adapter_payload
            else:
                delta_state[key] = delta_tensor
            changed_keys.append(key)
            continue
        if isinstance(parent_value, torch.Tensor) and isinstance(child_value, torch.Tensor):
            if torch.equal(parent_value.detach().cpu(), child_value.detach().cpu()):
                unchanged_keys += 1
                continue
        override_state[key] = child_value.detach().cpu() if isinstance(child_value, torch.Tensor) else child_value
        changed_keys.append(key)

    removed_keys = sorted(key for key in parent_state.keys() if key not in child_state)
    payload = {
        "artifact_kind": "tolbert_checkpoint_delta",
        "format_version": "tolbert_delta_v1",
        "checkpoint_format": _checkpoint_format(child_payload),
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "parent_checkpoint_sha256": _file_sha256(parent_checkpoint_path),
        "child_checkpoint_sha256": _file_sha256(child_checkpoint_path),
        "config": child_payload.get("config", parent_payload.get("config", {})),
        "state_dict_adapters": adapter_state,
        "state_dict_delta": delta_state,
        "override_state_dict": override_state,
        "removed_state_keys": removed_keys,
        "stats": {
            "changed_key_count": len(changed_keys),
            "unchanged_key_count": unchanged_keys,
            "adapter_key_count": len(adapter_state),
            "dense_delta_key_count": len(delta_state),
            "override_key_count": len(override_state),
            "removed_key_count": len(removed_keys),
            "total_key_count": len(child_state),
        },
    }
    delta_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, delta_output_path)
    return {
        "artifact_kind": "tolbert_checkpoint_delta",
        "delta_checkpoint_path": str(delta_output_path),
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "child_checkpoint_sha256": payload["child_checkpoint_sha256"],
        "stats": dict(payload["stats"]),
    }


def write_tolbert_checkpoint_delta(
    *,
    parent_checkpoint_path: Path,
    delta_output_path: Path,
    state_dict_adapters: dict[str, object] | None = None,
    state_dict_delta: dict[str, object] | None = None,
    override_state_dict: dict[str, object] | None = None,
    removed_state_keys: list[str] | None = None,
    config: object | None = None,
    checkpoint_format: str = "wrapped_state_dict",
    child_checkpoint_sha256: str = "",
) -> dict[str, Any]:
    _require_torch()
    payload = {
        "artifact_kind": "tolbert_checkpoint_delta",
        "format_version": "tolbert_delta_v1",
        "checkpoint_format": str(checkpoint_format).strip() or "wrapped_state_dict",
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "parent_checkpoint_sha256": _file_sha256(parent_checkpoint_path),
        "child_checkpoint_sha256": str(child_checkpoint_sha256).strip(),
        "config": config if isinstance(config, dict) else {},
        "state_dict_adapters": dict(state_dict_adapters or {}),
        "state_dict_delta": dict(state_dict_delta or {}),
        "override_state_dict": dict(override_state_dict or {}),
        "removed_state_keys": [str(key) for key in (removed_state_keys or []) if str(key).strip()],
    }
    payload["stats"] = {
        "changed_key_count": (
            len(payload["state_dict_adapters"]) + len(payload["state_dict_delta"]) + len(payload["override_state_dict"])
        ),
        "unchanged_key_count": 0,
        "adapter_key_count": len(payload["state_dict_adapters"]),
        "dense_delta_key_count": len(payload["state_dict_delta"]),
        "override_key_count": len(payload["override_state_dict"]),
        "removed_key_count": len(payload["removed_state_keys"]),
        "total_key_count": (
            len(payload["state_dict_adapters"])
            + len(payload["state_dict_delta"])
            + len(payload["override_state_dict"])
        ),
    }
    delta_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, delta_output_path)
    return {
        "artifact_kind": "tolbert_checkpoint_delta",
        "delta_checkpoint_path": str(delta_output_path),
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "child_checkpoint_sha256": payload["child_checkpoint_sha256"],
        "stats": dict(payload["stats"]),
    }


def load_tolbert_checkpoint_state(checkpoint_path: Path) -> tuple[dict[str, object], dict[str, Any], str]:
    _require_torch()
    payload = _load_checkpoint_payload(checkpoint_path)
    return _state_dict(payload), _config_dict(payload), _checkpoint_format(payload)


def load_tolbert_checkpoint_delta_metadata(delta_checkpoint_path: Path) -> dict[str, Any]:
    _require_torch()
    payload = torch.load(delta_checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid Tolbert delta payload: {delta_checkpoint_path}")
    return {
        "artifact_kind": "tolbert_checkpoint_delta",
        "delta_checkpoint_path": str(delta_checkpoint_path),
        "parent_checkpoint_path": str(payload.get("parent_checkpoint_path", "")).strip(),
        "parent_checkpoint_sha256": str(payload.get("parent_checkpoint_sha256", "")).strip(),
        "child_checkpoint_sha256": str(payload.get("child_checkpoint_sha256", "")).strip(),
        "stats": dict(payload.get("stats", {})) if isinstance(payload.get("stats", {}), dict) else {},
    }


def materialize_tolbert_checkpoint_from_delta(
    *,
    parent_checkpoint_path: Path,
    delta_checkpoint_path: Path,
    output_checkpoint_path: Path,
) -> Path:
    _require_torch()
    parent_payload = _load_checkpoint_payload(parent_checkpoint_path)
    delta_payload = torch.load(delta_checkpoint_path, map_location="cpu")
    parent_state = _state_dict(parent_payload)
    materialized_state = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in parent_state.items()
    }
    for key, adapter_payload in _state_dict_mapping(delta_payload.get("state_dict_adapters", {})).items():
        base = materialized_state.get(key)
        if not isinstance(base, torch.Tensor):
            raise RuntimeError(f"missing tensor base for adapter key: {key}")
        materialized_state[key] = base + _materialize_adapter_delta(adapter_payload, base_tensor=base)
    for key, delta_value in _state_dict_mapping(delta_payload.get("state_dict_delta", {})).items():
        base = materialized_state.get(key)
        if not isinstance(base, torch.Tensor):
            raise RuntimeError(f"missing tensor base for delta key: {key}")
        materialized_state[key] = base + delta_value.detach().cpu()
    for key, override_value in _state_dict_mapping(delta_payload.get("override_state_dict", {})).items():
        materialized_state[key] = override_value.detach().cpu() if isinstance(override_value, torch.Tensor) else override_value
    for key in delta_payload.get("removed_state_keys", []):
        materialized_state.pop(str(key), None)
    output_payload = _build_output_checkpoint_payload(
        state_dict=materialized_state,
        config=delta_payload.get("config", parent_payload.get("config", {})),
        checkpoint_format=str(delta_payload.get("checkpoint_format", _checkpoint_format(parent_payload))).strip(),
    )
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_checkpoint_path)
    return output_checkpoint_path


def resolve_tolbert_runtime_checkpoint_path(
    runtime_paths: object,
    *,
    artifact_path: Path | None = None,
) -> str:
    if not isinstance(runtime_paths, dict):
        return ""
    checkpoint_path = _resolved_existing_path(runtime_paths.get("checkpoint_path"), artifact_path=artifact_path)
    if checkpoint_path:
        return checkpoint_path
    parent_checkpoint_path = _resolved_existing_path(runtime_paths.get("parent_checkpoint_path"), artifact_path=artifact_path)
    delta_checkpoint_path = _resolved_existing_path(runtime_paths.get("checkpoint_delta_path"), artifact_path=artifact_path)
    if not parent_checkpoint_path or not delta_checkpoint_path:
        return ""
    output_path = _materialized_checkpoint_path(delta_checkpoint_path=Path(delta_checkpoint_path), artifact_path=artifact_path)
    if _needs_materialization(
        output_path=output_path,
        source_paths=(Path(parent_checkpoint_path), Path(delta_checkpoint_path)),
    ):
        materialize_tolbert_checkpoint_from_delta(
            parent_checkpoint_path=Path(parent_checkpoint_path),
            delta_checkpoint_path=Path(delta_checkpoint_path),
            output_checkpoint_path=output_path,
        )
    return str(output_path)


def _resolved_existing_path(raw: object, *, artifact_path: Path | None) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute() and artifact_path is not None:
        path = (artifact_path.parent / path).resolve()
    if not path.exists():
        return ""
    return str(path)


def _materialized_checkpoint_path(*, delta_checkpoint_path: Path, artifact_path: Path | None) -> Path:
    root = (
        artifact_path.parent / ".materialized_checkpoints"
        if artifact_path is not None
        else delta_checkpoint_path.parent / ".materialized_checkpoints"
    )
    digest = hashlib.sha256(str(delta_checkpoint_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return root / f"{delta_checkpoint_path.stem}__materialized_{digest}.pt"


def _needs_materialization(*, output_path: Path, source_paths: tuple[Path, ...]) -> bool:
    if not output_path.exists():
        return True
    try:
        output_mtime = output_path.stat().st_mtime
    except OSError:
        return True
    for path in source_paths:
        try:
            if path.stat().st_mtime > output_mtime:
                return True
        except OSError:
            return True
    return False


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    raw_payload = torch.load(path, map_location="cpu")
    if not isinstance(raw_payload, dict):
        raise RuntimeError(f"invalid Tolbert checkpoint payload: {path}")
    if "state_dict" in raw_payload and isinstance(raw_payload.get("state_dict"), dict):
        payload = dict(raw_payload)
        payload["_checkpoint_format"] = "wrapped_state_dict"
        return payload
    return {
        "state_dict": raw_payload,
        "config": {},
        "_checkpoint_format": "raw_state_dict",
    }


def _state_dict(payload: dict[str, Any]) -> dict[str, object]:
    state = payload.get("state_dict", {})
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint payload is missing a state_dict mapping")
    return state


def _state_dict_mapping(payload: object) -> dict[str, object]:
    return payload if isinstance(payload, dict) else {}


def _config_dict(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def _checkpoint_format(payload: dict[str, Any]) -> str:
    value = str(payload.get("_checkpoint_format", "wrapped_state_dict")).strip()
    return value or "wrapped_state_dict"


def _build_output_checkpoint_payload(
    *,
    state_dict: dict[str, object],
    config: object,
    checkpoint_format: str,
) -> dict[str, object]:
    normalized_format = str(checkpoint_format).strip() or "wrapped_state_dict"
    if normalized_format == "raw_state_dict":
        return dict(state_dict)
    return {
        "state_dict": state_dict,
        "config": config if isinstance(config, dict) else {},
    }


def _low_rank_adapter_payload(delta_tensor: object, *, rank_limit: int = 8) -> dict[str, object] | None:
    if not isinstance(delta_tensor, torch.Tensor):
        return None
    if delta_tensor.ndim < 2 or not torch.is_floating_point(delta_tensor):
        return None
    matrix = delta_tensor.reshape(delta_tensor.shape[0], -1)
    if min(matrix.shape) <= 1:
        return None
    if not hasattr(torch, "linalg") or not hasattr(torch.linalg, "svd"):
        return None
    try:
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return None
    if singular.numel() == 0:
        return None
    scale = float(torch.max(singular).item()) if singular.numel() > 0 else 0.0
    tolerance = max(1.0e-8, scale * 1.0e-6)
    nonzero = int(torch.count_nonzero(singular > tolerance).item())
    if nonzero <= 0:
        return None
    rank = min(nonzero, max(1, int(rank_limit)))
    if nonzero > rank:
        return None
    dense_size = int(matrix.numel())
    adapter_size = int(matrix.shape[0] * rank + rank * matrix.shape[1])
    if adapter_size >= dense_size:
        return None
    left = u[:, :rank] * singular[:rank]
    right = vh[:rank, :]
    return {
        "kind": "low_rank_adapter",
        "original_shape": list(delta_tensor.shape),
        "rank": rank,
        "left_factor": left.detach().cpu(),
        "right_factor": right.detach().cpu(),
    }


def _materialize_adapter_delta(adapter_payload: object, *, base_tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(adapter_payload, dict):
        raise RuntimeError("invalid adapter payload")
    kind = str(adapter_payload.get("kind", "")).strip()
    if kind == "structured_transition_adapter":
        return _materialize_structured_transition_adapter(adapter_payload, base_tensor=base_tensor)
    if kind == "fixed_basis_adapter":
        return _materialize_fixed_basis_adapter(adapter_payload, base_tensor=base_tensor)
    if kind != "low_rank_adapter":
        raise RuntimeError("unsupported adapter payload kind")
    left = adapter_payload.get("left_factor")
    right = adapter_payload.get("right_factor")
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise RuntimeError("adapter payload is missing tensor factors")
    original_shape = tuple(int(value) for value in adapter_payload.get("original_shape", []) if int(value) >= 0)
    if not original_shape:
        original_shape = tuple(base_tensor.shape)
    delta_matrix = left.detach().cpu() @ right.detach().cpu()
    return delta_matrix.reshape(original_shape).to(dtype=base_tensor.dtype)


def _materialize_fixed_basis_adapter(adapter_payload: dict[str, object], *, base_tensor: torch.Tensor) -> torch.Tensor:
    basis_kind = str(adapter_payload.get("basis_kind", "")).strip() or "cosine_v1"
    if basis_kind != "cosine_v1":
        raise RuntimeError(f"unsupported basis adapter kind: {basis_kind}")
    coefficients = adapter_payload.get("coefficients")
    if not isinstance(coefficients, torch.Tensor):
        raise RuntimeError("basis adapter payload is missing coefficients")
    original_shape = tuple(int(value) for value in adapter_payload.get("original_shape", []) if int(value) >= 0)
    if not original_shape:
        original_shape = tuple(base_tensor.shape)
    numel = 1
    for value in original_shape:
        numel *= max(1, int(value))
    basis = _cosine_basis(numel=numel, rank=int(coefficients.numel()))
    delta = torch.matmul(coefficients.detach().cpu().to(dtype=torch.float32), basis)
    return delta.reshape(original_shape).to(dtype=base_tensor.dtype)


def _materialize_structured_transition_adapter(adapter_payload: dict[str, object], *, base_tensor: torch.Tensor) -> torch.Tensor:
    transition_kind = str(adapter_payload.get("transition_kind", "")).strip() or "source_dest_stay_v1"
    if transition_kind != "source_dest_stay_v1":
        raise RuntimeError(f"unsupported structured transition kind: {transition_kind}")
    source_logits = adapter_payload.get("source_logits")
    dest_logits = adapter_payload.get("dest_logits")
    stay_logits = adapter_payload.get("stay_logits")
    if not isinstance(source_logits, torch.Tensor) or not isinstance(dest_logits, torch.Tensor) or not isinstance(stay_logits, torch.Tensor):
        raise RuntimeError("structured transition adapter payload is missing tensor factors")
    original_shape = tuple(int(value) for value in adapter_payload.get("original_shape", []) if int(value) >= 0)
    if not original_shape:
        original_shape = tuple(base_tensor.shape)
    if len(original_shape) != 2 or int(original_shape[0]) != int(original_shape[1]):
        raise RuntimeError("structured transition adapter requires a square original_shape")
    delta = source_logits.detach().cpu() @ dest_logits.detach().cpu()
    delta = delta + torch.diag(stay_logits.detach().cpu())
    return delta.reshape(original_shape).to(dtype=base_tensor.dtype)


def _cosine_basis(*, numel: int, rank: int) -> torch.Tensor:
    if numel <= 0 or rank <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    positions = torch.arange(numel, dtype=torch.float32) + 0.5
    rows = []
    for basis_index in range(rank):
        row = torch.cos(torch.pi * float(basis_index) * positions / float(numel))
        row = row / row.norm().clamp_min(1.0e-8)
        rows.append(row)
    return torch.stack(rows, dim=0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for Tolbert checkpoint delta operations")
