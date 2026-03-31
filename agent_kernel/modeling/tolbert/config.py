from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HybridTolbertSSMConfig:
    model_family: str = "tolbert_ssm_v1"
    token_vocab_size: int = 4096
    decoder_vocab_size: int = 4096
    decoder_pad_token_id: int = 0
    decoder_bos_token_id: int = 1
    decoder_eos_token_id: int = 2
    decoder_unk_token_id: int = 3
    family_vocab_size: int = 256
    path_vocab_size: int = 4096
    max_path_levels: int = 4
    max_command_tokens: int = 16
    sequence_length: int = 6
    scalar_feature_dim: int = 16
    hidden_dim: int = 96
    d_state: int = 24
    world_state_dim: int = 8
    use_world_model: bool = True
    use_dense_world_transition: bool = False
    world_transition_family: str = "banded"
    world_transition_bandwidth: int = 1
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HybridTolbertSSMConfig":
        dense = bool(payload.get("use_dense_world_transition", False))
        family = str(payload.get("world_transition_family", "banded")).strip().lower() or "banded"
        if dense:
            family = "dense"
        return cls(
            model_family=str(payload.get("model_family", "tolbert_ssm_v1")).strip() or "tolbert_ssm_v1",
            token_vocab_size=max(32, int(payload.get("token_vocab_size", 4096))),
            decoder_vocab_size=max(8, int(payload.get("decoder_vocab_size", payload.get("token_vocab_size", 4096)))),
            decoder_pad_token_id=max(0, int(payload.get("decoder_pad_token_id", 0))),
            decoder_bos_token_id=max(0, int(payload.get("decoder_bos_token_id", 1))),
            decoder_eos_token_id=max(0, int(payload.get("decoder_eos_token_id", 2))),
            decoder_unk_token_id=max(0, int(payload.get("decoder_unk_token_id", 3))),
            family_vocab_size=max(8, int(payload.get("family_vocab_size", 256))),
            path_vocab_size=max(64, int(payload.get("path_vocab_size", 4096))),
            max_path_levels=max(1, int(payload.get("max_path_levels", 4))),
            max_command_tokens=max(1, int(payload.get("max_command_tokens", 16))),
            sequence_length=max(2, int(payload.get("sequence_length", 6))),
            scalar_feature_dim=max(4, int(payload.get("scalar_feature_dim", 16))),
            hidden_dim=max(16, int(payload.get("hidden_dim", 96))),
            d_state=max(4, int(payload.get("d_state", 24))),
            world_state_dim=max(2, int(payload.get("world_state_dim", 8))),
            use_world_model=bool(payload.get("use_world_model", True)),
            use_dense_world_transition=dense,
            world_transition_family=family,
            world_transition_bandwidth=max(0, int(payload.get("world_transition_bandwidth", 1))),
            dropout=max(0.0, float(payload.get("dropout", 0.0))),
        )
