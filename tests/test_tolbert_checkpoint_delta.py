from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if not hasattr(torch, "tensor") or not hasattr(torch, "save") or not hasattr(torch, "load"):
    pytest.skip("full torch checkpoint APIs are unavailable", allow_module_level=True)

from agent_kernel.modeling.tolbert.delta import (  # noqa: E402
    create_tolbert_checkpoint_delta,
    materialize_tolbert_checkpoint_from_delta,
    resolve_tolbert_runtime_checkpoint_path,
    write_tolbert_checkpoint_delta,
)


def test_tolbert_checkpoint_delta_round_trip(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    child_checkpoint = tmp_path / "child.pt"
    delta_checkpoint = tmp_path / "child__delta.pt"
    materialized_checkpoint = tmp_path / "materialized.pt"

    parent_state = {
        "encoder.weight": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        "decoder.bias": torch.tensor([0.5, -0.5]),
    }
    child_state = {
        "encoder.weight": torch.tensor(
            [
                [2.0, 4.0, 6.0],
                [5.0, 7.0, 9.0],
                [8.0, 10.0, 12.0],
            ]
        ),
        "decoder.bias": torch.tensor([0.5, -0.25]),
    }
    torch.save({"state_dict": parent_state, "config": {"model_family": "tolbert_ssm_v1"}}, parent_checkpoint)
    torch.save({"state_dict": child_state, "config": {"model_family": "tolbert_ssm_v1"}}, child_checkpoint)

    metadata = create_tolbert_checkpoint_delta(
        parent_checkpoint_path=parent_checkpoint,
        child_checkpoint_path=child_checkpoint,
        delta_output_path=delta_checkpoint,
    )
    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_checkpoint_path=delta_checkpoint,
        output_checkpoint_path=materialized_checkpoint,
    )

    payload = torch.load(materialized_checkpoint, map_location="cpu")
    assert metadata["stats"]["changed_key_count"] == 2
    assert metadata["stats"]["adapter_key_count"] == 1
    assert metadata["stats"]["dense_delta_key_count"] == 1
    assert torch.allclose(payload["state_dict"]["encoder.weight"], child_state["encoder.weight"], atol=1.0e-6)
    assert torch.equal(payload["state_dict"]["decoder.bias"], child_state["decoder.bias"])


def test_resolve_tolbert_runtime_checkpoint_path_materializes_delta(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    child_checkpoint = tmp_path / "child.pt"
    delta_checkpoint = tmp_path / "child__delta.pt"
    artifact_path = tmp_path / "tolbert_model_artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    torch.save({"state_dict": {"weight": torch.tensor([1.0])}, "config": {}}, parent_checkpoint)
    torch.save({"state_dict": {"weight": torch.tensor([2.0])}, "config": {}}, child_checkpoint)
    create_tolbert_checkpoint_delta(
        parent_checkpoint_path=parent_checkpoint,
        child_checkpoint_path=child_checkpoint,
        delta_output_path=delta_checkpoint,
    )

    resolved = resolve_tolbert_runtime_checkpoint_path(
        {
            "checkpoint_path": "",
            "parent_checkpoint_path": str(parent_checkpoint),
            "checkpoint_delta_path": str(delta_checkpoint),
        },
        artifact_path=artifact_path,
    )

    assert Path(resolved).exists()
    payload = torch.load(resolved, map_location="cpu")
    assert float(payload["state_dict"]["weight"][0]) == 2.0


def test_tolbert_checkpoint_delta_round_trip_supports_raw_state_dict_format(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent_raw.pt"
    child_checkpoint = tmp_path / "child_raw.pt"
    delta_checkpoint = tmp_path / "child_raw__delta.pt"
    materialized_checkpoint = tmp_path / "materialized_raw.pt"

    torch.save(
        {
            "encoder.weight": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            )
        },
        parent_checkpoint,
    )
    torch.save(
        {
            "encoder.weight": torch.tensor(
                [
                    [2.0, 4.0, 6.0],
                    [5.0, 7.0, 9.0],
                    [8.0, 10.0, 12.0],
                ]
            )
        },
        child_checkpoint,
    )

    metadata = create_tolbert_checkpoint_delta(
        parent_checkpoint_path=parent_checkpoint,
        child_checkpoint_path=child_checkpoint,
        delta_output_path=delta_checkpoint,
    )
    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_checkpoint_path=delta_checkpoint,
        output_checkpoint_path=materialized_checkpoint,
    )

    payload = torch.load(materialized_checkpoint, map_location="cpu")
    assert metadata["stats"]["adapter_key_count"] == 1
    assert "state_dict" not in payload
    assert torch.allclose(
        payload["encoder.weight"],
        torch.tensor(
            [
                [2.0, 4.0, 6.0],
                [5.0, 7.0, 9.0],
                [8.0, 10.0, 12.0],
            ]
        ),
        atol=1.0e-6,
    )


def test_tolbert_checkpoint_delta_materializes_fixed_basis_adapter_payload(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    delta_checkpoint = tmp_path / "child__delta.pt"
    materialized_checkpoint = tmp_path / "materialized.pt"

    parent_state = {
        "d_skip": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    }
    torch.save({"state_dict": parent_state, "config": {"model_family": "tolbert_ssm_v1"}}, parent_checkpoint)
    coefficients = torch.tensor([0.5], dtype=torch.float32)
    write_tolbert_checkpoint_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_output_path=delta_checkpoint,
        state_dict_adapters={
            "d_skip": {
                "kind": "fixed_basis_adapter",
                "basis_kind": "cosine_v1",
                "original_shape": [4],
                "basis_rank": 1,
                "coefficients": coefficients,
            }
        },
        config={"model_family": "tolbert_ssm_v1"},
    )

    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_checkpoint_path=delta_checkpoint,
        output_checkpoint_path=materialized_checkpoint,
    )

    payload = torch.load(materialized_checkpoint, map_location="cpu")
    expected = torch.full((4,), 0.25, dtype=torch.float32)
    assert torch.allclose(payload["state_dict"]["d_skip"], expected, atol=1.0e-6)


def test_tolbert_checkpoint_delta_materializes_structured_transition_adapter_payload(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    delta_checkpoint = tmp_path / "child__delta.pt"
    materialized_checkpoint = tmp_path / "materialized.pt"

    parent_state = {
        "world_transition_logits": torch.zeros(3, 3),
    }
    torch.save({"state_dict": parent_state, "config": {"model_family": "tolbert_ssm_v1"}}, parent_checkpoint)
    write_tolbert_checkpoint_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_output_path=delta_checkpoint,
        state_dict_adapters={
            "world_transition_logits": {
                "kind": "structured_transition_adapter",
                "transition_kind": "source_dest_stay_v1",
                "original_shape": [3, 3],
                "rank": 1,
                "source_logits": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
                "dest_logits": torch.tensor([[0.5, -0.5, 1.0]], dtype=torch.float32),
                "stay_logits": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            }
        },
        config={"model_family": "tolbert_ssm_v1"},
    )

    materialize_tolbert_checkpoint_from_delta(
        parent_checkpoint_path=parent_checkpoint,
        delta_checkpoint_path=delta_checkpoint,
        output_checkpoint_path=materialized_checkpoint,
    )

    payload = torch.load(materialized_checkpoint, map_location="cpu")
    expected = torch.tensor(
        [
            [0.6, -0.5, 1.0],
            [1.0, -0.8, 2.0],
            [1.5, -1.5, 3.3],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(payload["state_dict"]["world_transition_logits"], expected, atol=1.0e-6)
