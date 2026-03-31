from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn.functional as F  # noqa: F401
    from torch import Tensor
    nn = torch.nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = Any  # type: ignore[misc,assignment]

from ..ssm import selective_scan
from ..world import causal_belief_scan
from .config import HybridTolbertSSMConfig

_BASE_MODULE = nn.Module if torch is not None else object


@dataclass(frozen=True, slots=True)
class HybridTolbertSSMOutput:
    score: Tensor
    policy_logits: Tensor
    value: Tensor
    stop_logits: Tensor
    risk_logits: Tensor
    transition: Tensor
    decoder_logits: Tensor
    pooled_state: Tensor
    ssm_backend: str
    decoder_diagnostics: dict[str, object] = field(default_factory=dict)
    ssm_last_state: Tensor | None = None
    ssm_diagnostics: dict[str, object] = field(default_factory=dict)
    world_final_belief: Tensor | None = None
    world_backend: str = ""
    world_diagnostics: dict[str, object] = field(default_factory=dict)


class HybridTolbertSSMModel(_BASE_MODULE):  # type: ignore[misc]
    def __init__(self, config: HybridTolbertSSMConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.hidden_dim, padding_idx=0)
        self.family_embedding = nn.Embedding(config.family_vocab_size, config.hidden_dim)
        self.path_embedding = nn.Embedding(config.path_vocab_size, config.hidden_dim)
        self.scalar_projection = nn.Linear(config.scalar_feature_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.delta_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.b_projection = nn.Linear(config.hidden_dim, config.d_state)
        self.c_projection = nn.Linear(config.hidden_dim, config.d_state)
        self.z_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.log_a = nn.Parameter(torch.zeros(config.hidden_dim, config.d_state))
        self.d_skip = nn.Parameter(torch.zeros(config.hidden_dim))
        self.delta_bias = nn.Parameter(torch.zeros(config.hidden_dim))
        self.world_local_projection = nn.Linear(config.hidden_dim, config.world_state_dim)
        self.world_context_projection = nn.Linear(config.hidden_dim, config.world_state_dim)
        self.world_belief_projection = nn.Linear(config.world_state_dim, config.hidden_dim)
        self.world_decoder_projection = nn.Linear(config.world_state_dim, config.hidden_dim)
        self.world_transition_logits = nn.Parameter(torch.zeros(config.world_state_dim, config.world_state_dim))
        self.world_transition_gate = nn.Parameter(torch.tensor(1.0))
        self.policy_head = nn.Linear(config.hidden_dim, 1)
        self.score_head = nn.Linear(config.hidden_dim, 1)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.stop_head = nn.Linear(config.hidden_dim, 1)
        self.risk_head = nn.Linear(config.hidden_dim, 1)
        self.transition_head = nn.Linear(config.hidden_dim, 2)
        self.decoder_token_embedding = nn.Embedding(
            config.decoder_vocab_size,
            config.hidden_dim,
            padding_idx=config.decoder_pad_token_id,
        )
        self.decoder_position_embedding = nn.Embedding(config.max_command_tokens, config.hidden_dim)
        self.decoder_input_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.decoder_delta_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.decoder_b_projection = nn.Linear(config.hidden_dim, config.d_state)
        self.decoder_c_projection = nn.Linear(config.hidden_dim, config.d_state)
        self.decoder_z_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.decoder_log_a = nn.Parameter(torch.zeros(config.hidden_dim, config.d_state))
        self.decoder_d_skip = nn.Parameter(torch.zeros(config.hidden_dim))
        self.decoder_delta_bias = nn.Parameter(torch.zeros(config.hidden_dim))
        self.decoder_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.decoder_head = nn.Linear(config.hidden_dim, config.decoder_vocab_size)

    def forward(
        self,
        *,
        command_token_ids: Tensor,
        scalar_features: Tensor,
        family_ids: Tensor,
        path_level_ids: Tensor,
        decoder_input_ids: Tensor | None = None,
        prefer_python_ref: bool = False,
        prefer_python_world_ref: bool = False,
        world_initial_log_belief: Tensor | None = None,
    ) -> HybridTolbertSSMOutput:
        _require_torch()
        encoded = self.encode_condition(
            command_token_ids=command_token_ids,
            scalar_features=scalar_features,
            family_ids=family_ids,
            path_level_ids=path_level_ids,
            prefer_python_ref=prefer_python_ref,
            prefer_python_world_ref=prefer_python_world_ref,
            world_initial_log_belief=world_initial_log_belief,
        )
        pooled = encoded["pooled_state"]
        decoder_logits, _decoder_cache = self.decode_decoder_tokens(
            pooled_state=pooled,
            decoder_input_ids=decoder_input_ids,
            prefer_python_ref=prefer_python_ref,
            initial_world_log_belief=encoded["world_final_belief"],
            prefer_python_world_ref=prefer_python_world_ref,
        )
        return HybridTolbertSSMOutput(
            score=self.score_head(pooled).squeeze(-1),
            policy_logits=self.policy_head(pooled).squeeze(-1),
            value=self.value_head(pooled).squeeze(-1),
            stop_logits=self.stop_head(pooled).squeeze(-1),
            risk_logits=self.risk_head(pooled).squeeze(-1),
            transition=self.transition_head(pooled),
            decoder_logits=decoder_logits,
            pooled_state=pooled,
            ssm_backend=str(encoded["ssm_backend"]),
            decoder_diagnostics=dict(_decoder_cache),
            ssm_last_state=encoded["ssm_last_state"],
            ssm_diagnostics=dict(encoded["ssm_diagnostics"]),
            world_final_belief=encoded["world_final_belief"],
            world_backend=str(encoded["world_backend"]),
            world_diagnostics=dict(encoded["world_diagnostics"]),
        )

    def encode_condition(
        self,
        *,
        command_token_ids: Tensor,
        scalar_features: Tensor,
        family_ids: Tensor,
        path_level_ids: Tensor,
        prefer_python_ref: bool = False,
        prefer_python_world_ref: bool = False,
        world_initial_log_belief: Tensor | None = None,
    ) -> dict[str, object]:
        _require_torch()
        token_hidden = self.token_embedding(command_token_ids).mean(dim=-2)
        scalar_hidden = self.scalar_projection(scalar_features)
        family_hidden = self.family_embedding(family_ids).unsqueeze(1)
        path_hidden = self.path_embedding(path_level_ids).sum(dim=1).unsqueeze(1)
        hidden = self.dropout(token_hidden + scalar_hidden + family_hidden + path_hidden)
        u = hidden.transpose(1, 2)
        delta = self.delta_projection(hidden).transpose(1, 2)
        b = self.b_projection(hidden).transpose(1, 2)
        c = self.c_projection(hidden).transpose(1, 2)
        z = self.z_projection(hidden).transpose(1, 2)
        a = -torch.exp(self.log_a)
        scan = selective_scan(
            u,
            delta,
            a,
            b,
            c,
            D=self.d_skip,
            z=z,
            delta_bias=self.delta_bias,
            delta_softplus=True,
            return_last_state=True,
            prefer_python_ref=prefer_python_ref,
        )
        latent = scan.output.transpose(1, 2)
        pooled = latent[:, -1, :] + latent.mean(dim=1)
        ssm_diagnostics = _summarize_recurrent_state(
            last_state=scan.last_state,
            latent=latent,
            pooled=pooled,
            backend=scan.backend,
        )
        world_final_belief: Tensor | None = None
        world_backend = ""
        world_diagnostics: dict[str, object] = {}
        if self.config.use_world_model:
            if world_initial_log_belief is None:
                init = torch.zeros(
                    hidden.shape[0],
                    self.config.world_state_dim,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                init = torch.log_softmax(init, dim=-1)
            else:
                if world_initial_log_belief.shape != (hidden.shape[0], self.config.world_state_dim):
                    raise ValueError(
                        "world_initial_log_belief must have shape "
                        f"({hidden.shape[0]}, {self.config.world_state_dim}); "
                        f"got {tuple(world_initial_log_belief.shape)}"
                    )
                init = world_initial_log_belief.to(device=hidden.device, dtype=hidden.dtype)
            world_scan, world_diagnostics = self._scan_world_hidden(
                hidden=hidden,
                initial_log_belief=init,
                prefer_python_ref=prefer_python_world_ref,
                chunk_size=max(1, self.config.sequence_length // 2),
            )
            world_probs = world_scan.beliefs.exp()
            world_final_belief = world_scan.final_log_belief
            world_backend = world_scan.backend
            pooled = pooled + self.world_belief_projection(world_probs.mean(dim=1))
            pooled = pooled + self.world_belief_projection(world_final_belief.exp())
        return {
            "pooled_state": pooled,
            "ssm_backend": scan.backend,
            "ssm_last_state": scan.last_state,
            "ssm_diagnostics": ssm_diagnostics,
            "world_final_belief": world_final_belief,
            "world_backend": world_backend,
            "world_diagnostics": world_diagnostics,
        }

    def decode_world_tokens(
        self,
        *,
        pooled_state: Tensor,
        decoder_input_ids: Tensor,
        initial_log_belief: Tensor | None,
        prefer_python_ref: bool = False,
    ) -> tuple[Tensor | None, dict[str, object]]:
        _require_torch()
        if not self.config.use_world_model:
            return initial_log_belief, {"backend": "disabled", "sequence_length": int(decoder_input_ids.shape[1])}
        if initial_log_belief is None:
            raise ValueError("initial_log_belief is required when use_world_model is enabled")
        decoder_hidden = self._encode_decoder_sequence_hidden(
            pooled_state=pooled_state,
            decoder_input_ids=decoder_input_ids,
        )
        world_scan, world_diagnostics = self._scan_world_hidden(
            hidden=decoder_hidden,
            initial_log_belief=initial_log_belief,
            prefer_python_ref=prefer_python_ref,
            chunk_size=max(1, int(decoder_input_ids.shape[1]) // 2),
        )
        world_diagnostics["sequence_length"] = int(decoder_input_ids.shape[1])
        return world_scan.final_log_belief, world_diagnostics

    def decode_decoder_tokens(
        self,
        *,
        pooled_state: Tensor,
        decoder_input_ids: Tensor | None = None,
        prefer_python_ref: bool = False,
        initial_world_log_belief: Tensor | None = None,
        prefer_python_world_ref: bool = False,
    ) -> tuple[Tensor, dict[str, object]]:
        _require_torch()
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (pooled_state.shape[0], self.config.max_command_tokens),
                self.config.decoder_pad_token_id,
                dtype=torch.long,
                device=pooled_state.device,
            )
            decoder_input_ids[:, 0] = self.config.decoder_bos_token_id
        decoder_hidden = self._encode_decoder_sequence_hidden(
            pooled_state=pooled_state,
            decoder_input_ids=decoder_input_ids,
        )
        world_final_belief: Tensor | None = None
        world_belief_sequence: Tensor | None = None
        world_backend = "disabled"
        if self.config.use_world_model and initial_world_log_belief is not None:
            world_scan, _world_diagnostics = self._scan_world_hidden(
                hidden=decoder_hidden,
                initial_log_belief=initial_world_log_belief,
                prefer_python_ref=prefer_python_world_ref,
                chunk_size=max(1, int(decoder_input_ids.shape[1]) // 2),
            )
            world_final_belief = world_scan.final_log_belief
            world_belief_sequence = world_scan.beliefs.exp()
            world_backend = world_scan.backend
        decoder_scan = selective_scan(
            decoder_hidden.transpose(1, 2),
            self.decoder_delta_projection(decoder_hidden).transpose(1, 2),
            -torch.exp(self.decoder_log_a),
            self.decoder_b_projection(decoder_hidden).transpose(1, 2),
            self.decoder_c_projection(decoder_hidden).transpose(1, 2),
            D=self.decoder_d_skip,
            z=self.decoder_z_projection(decoder_hidden).transpose(1, 2),
            delta_bias=self.decoder_delta_bias,
            delta_softplus=True,
            return_last_state=True,
            prefer_python_ref=prefer_python_ref,
        )
        decoder_out = self.dropout(
            self.decoder_projection(decoder_scan.output.transpose(1, 2) + pooled_state.unsqueeze(1))
        )
        if world_belief_sequence is not None:
            decoder_out = decoder_out + self.world_decoder_projection(world_belief_sequence)
        return self.decoder_head(decoder_out), {
            "backend": decoder_scan.backend,
            "last_state": decoder_scan.last_state,
            "sequence_length": int(decoder_input_ids.shape[1]),
            "world_backend": str(world_backend),
            "world_final_belief": world_final_belief,
        }

    def decode_decoder_step(
        self,
        *,
        pooled_state: Tensor,
        token_ids: Tensor,
        position: int,
        decoder_state: Tensor | None = None,
        prefer_python_ref: bool = False,
        world_log_belief: Tensor | None = None,
        prefer_python_world_ref: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, dict[str, object]]:
        _require_torch()
        if token_ids.dim() != 1 or token_ids.shape[0] != pooled_state.shape[0]:
            raise ValueError(
                f"token_ids must have shape ({pooled_state.shape[0]},); got {tuple(token_ids.shape)}"
            )
        position_ids = torch.full(
            (pooled_state.shape[0], 1),
            fill_value=int(position),
            dtype=torch.long,
            device=pooled_state.device,
        )
        decoder_hidden = self._encode_decoder_step_hidden(
            pooled_state=pooled_state,
            token_ids=token_ids,
            position_ids=position_ids,
        )
        next_world_belief = world_log_belief
        world_backend = "disabled"
        if self.config.use_world_model and world_log_belief is not None:
            world_scan, _world_diagnostics = self._scan_world_hidden(
                hidden=decoder_hidden,
                initial_log_belief=world_log_belief,
                prefer_python_ref=prefer_python_world_ref,
                chunk_size=1,
            )
            next_world_belief = world_scan.final_log_belief
            world_backend = world_scan.backend
        decoder_scan = selective_scan(
            decoder_hidden.transpose(1, 2),
            self.decoder_delta_projection(decoder_hidden).transpose(1, 2),
            -torch.exp(self.decoder_log_a),
            self.decoder_b_projection(decoder_hidden).transpose(1, 2),
            self.decoder_c_projection(decoder_hidden).transpose(1, 2),
            D=self.decoder_d_skip,
            z=self.decoder_z_projection(decoder_hidden).transpose(1, 2),
            delta_bias=self.decoder_delta_bias,
            delta_softplus=True,
            return_last_state=True,
            prefer_python_ref=prefer_python_ref,
            initial_state=decoder_state,
        )
        decoder_out = self.dropout(
            self.decoder_projection(decoder_scan.output.transpose(1, 2) + pooled_state.unsqueeze(1))
        )
        if next_world_belief is not None:
            decoder_out = decoder_out + self.world_decoder_projection(next_world_belief.exp()).unsqueeze(1)
        logits = self.decoder_head(decoder_out[:, 0, :])
        return logits, decoder_scan.last_state, next_world_belief, {
            "backend": decoder_scan.backend,
            "position": int(position),
            "world_backend": str(world_backend),
            "last_state_norm_mean": round(_tensor_norm_mean(decoder_scan.last_state), 4)
            if decoder_scan.last_state is not None
            else 0.0,
        }

    def advance_world_state(
        self,
        *,
        pooled_state: Tensor,
        token_ids: Tensor,
        position: int,
        world_log_belief: Tensor | None,
        prefer_python_ref: bool = False,
    ) -> tuple[Tensor | None, dict[str, object]]:
        _require_torch()
        if not self.config.use_world_model:
            return world_log_belief, {"backend": "disabled", "position": int(position)}
        if world_log_belief is None:
            raise ValueError("world_log_belief is required when use_world_model is enabled")
        position_ids = torch.full(
            (pooled_state.shape[0], 1),
            fill_value=int(position),
            dtype=torch.long,
            device=pooled_state.device,
        )
        decoder_hidden = self._encode_decoder_step_hidden(
            pooled_state=pooled_state,
            token_ids=token_ids,
            position_ids=position_ids,
        )
        world_scan, world_diagnostics = self._scan_world_hidden(
            hidden=decoder_hidden,
            initial_log_belief=world_log_belief,
            prefer_python_ref=prefer_python_ref,
            chunk_size=1,
        )
        world_diagnostics["position"] = int(position)
        return world_scan.final_log_belief, world_diagnostics

    def _encode_decoder_sequence_hidden(
        self,
        *,
        pooled_state: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        positions = torch.arange(int(decoder_input_ids.shape[1]), device=pooled_state.device)
        decoder_hidden = (
            self.decoder_token_embedding(decoder_input_ids)
            + self.decoder_position_embedding(positions).unsqueeze(0)
            + pooled_state.unsqueeze(1)
        )
        return self.dropout(self.decoder_input_projection(decoder_hidden))

    def _encode_decoder_step_hidden(
        self,
        *,
        pooled_state: Tensor,
        token_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        token_hidden = self.decoder_token_embedding(token_ids.unsqueeze(1))
        position_hidden = self.decoder_position_embedding(position_ids)
        decoder_hidden = token_hidden + position_hidden + pooled_state.unsqueeze(1)
        return self.dropout(self.decoder_input_projection(decoder_hidden))

    def _scan_world_hidden(
        self,
        *,
        hidden: Tensor,
        initial_log_belief: Tensor,
        prefer_python_ref: bool,
        chunk_size: int,
    ) -> tuple[object, dict[str, object]]:
        expected_shape = (hidden.shape[0], self.config.world_state_dim)
        if initial_log_belief.shape != expected_shape:
            raise ValueError(
                f"initial_log_belief must have shape {expected_shape}; got {tuple(initial_log_belief.shape)}"
            )
        initial = initial_log_belief.to(device=hidden.device, dtype=hidden.dtype)
        local_logits = self.world_local_projection(hidden)
        transition_context = self.world_context_projection(hidden)
        transition_structure = self._world_transition_structure()
        if transition_structure is None:
            transition_log_probs = _world_transition_log_probs(
                self.world_transition_logits,
                family=self.config.world_transition_family,
                bandwidth=self.config.world_transition_bandwidth,
            )
            transition_input = transition_log_probs
        else:
            transition_log_probs = self.world_transition_logits.new_empty(
                self.config.world_state_dim,
                self.config.world_state_dim,
            )
            transition_input = transition_structure["base_transition_logits"]
        world_gate = torch.sigmoid(self.world_transition_gate)
        world_scan = causal_belief_scan(
            local_logits,
            transition_input,
            transition_context,
            initial,
            transition_structure=transition_structure,
            transition_gate=world_gate,
            chunk_size=max(1, int(chunk_size)),
            prefer_python_ref=prefer_python_ref,
        )
        return world_scan, _summarize_world_state(
            beliefs=world_scan.beliefs,
            final_log_belief=world_scan.final_log_belief,
            initial_log_belief=initial,
            transition_log_probs=transition_log_probs,
            transition_structure=transition_structure,
            transition_family=self.config.world_transition_family,
            transition_bandwidth=self.config.world_transition_bandwidth,
            transition_gate=world_gate,
            backend=world_scan.backend,
        )

    def _world_transition_structure(self) -> dict[str, object] | None:
        parametrizations = getattr(getattr(self, "parametrizations", None), "world_transition_logits", None)
        if parametrizations is None:
            return None
        if len(parametrizations) <= 0:
            return None
        parametrization = parametrizations[0]
        source_logits = getattr(parametrization, "source_logits", None)
        dest_logits = getattr(parametrization, "dest_logits", None)
        stay_logits = getattr(parametrization, "stay_logits", None)
        base_transition_logits = getattr(parametrizations, "original", None)
        if not isinstance(source_logits, Tensor) or not isinstance(dest_logits, Tensor) or not isinstance(stay_logits, Tensor):
            return None
        if not isinstance(base_transition_logits, Tensor):
            return None
        return {
            "kind": "source_dest_stay_v1",
            "base_transition_logits": base_transition_logits,
            "source_logits": source_logits,
            "dest_logits": dest_logits,
            "stay_logits": stay_logits,
            "family": self.config.world_transition_family,
            "bandwidth": self.config.world_transition_bandwidth,
        }


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for HybridTolbertSSMModel")


def _world_transition_log_probs(
    transition_logits: Tensor,
    *,
    family: str,
    bandwidth: int,
) -> Tensor:
    normalized_family = str(family).strip().lower() or "banded"
    if normalized_family == "diag":
        return torch.diagonal(transition_logits)
    if normalized_family == "dense":
        return torch.log_softmax(transition_logits, dim=-1)
    if normalized_family == "banded":
        num_states = int(transition_logits.shape[0])
        positions = torch.arange(num_states, device=transition_logits.device)
        mask = (positions[:, None] - positions[None, :]).abs() <= max(0, int(bandwidth))
        masked = transition_logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(masked, dim=-1)
    raise ValueError(f"unsupported world_transition_family: {family!r}")


def _summarize_recurrent_state(
    *,
    last_state: Tensor | None,
    latent: Tensor,
    pooled: Tensor,
    backend: str,
) -> dict[str, object]:
    diagnostics = {
        "backend": str(backend),
        "sequence_length": int(latent.shape[1]),
        "hidden_dim": int(latent.shape[2]),
        "latent_norm_mean": round(_tensor_norm_mean(latent), 4),
        "pooled_norm_mean": round(_tensor_norm_mean(pooled), 4),
        "last_state_norm_mean": 0.0,
        "last_state_dim": 0,
    }
    if last_state is not None:
        diagnostics["last_state_norm_mean"] = round(_tensor_norm_mean(last_state), 4)
        diagnostics["last_state_dim"] = int(last_state.shape[-1])
    return diagnostics


def _summarize_world_state(
    *,
    beliefs: Tensor,
    final_log_belief: Tensor,
    initial_log_belief: Tensor,
    transition_log_probs: Tensor,
    transition_structure: dict[str, object] | None,
    transition_family: str,
    transition_bandwidth: int,
    transition_gate: Tensor,
    backend: str,
) -> dict[str, object]:
    final_probs = final_log_belief.exp()
    initial_probs = initial_log_belief.exp()
    final_top = torch.topk(final_probs[0], k=min(3, final_probs.shape[-1]))
    initial_top = torch.topk(initial_probs[0], k=min(3, initial_probs.shape[-1]))
    return {
        "backend": str(backend),
        "transition_family": str(transition_family),
        "transition_bandwidth": int(transition_bandwidth),
        "transition_gate": round(float(transition_gate.detach().item()), 4),
        "belief_entropy_mean": round(_entropy_mean(beliefs.exp()), 4),
        "initial_entropy_mean": round(_entropy_mean(initial_probs), 4),
        "final_entropy_mean": round(_entropy_mean(final_probs), 4),
        "final_top_states": [int(value) for value in final_top.indices.tolist()],
        "final_top_state_probs": [round(float(value), 4) for value in final_top.values.tolist()],
        "initial_top_states": [int(value) for value in initial_top.indices.tolist()],
        "initial_top_state_probs": [round(float(value), 4) for value in initial_top.values.tolist()],
        "structure": _world_transition_structure_summary(
            transition_log_probs,
            transition_structure=transition_structure,
            family=str(transition_family),
            bandwidth=int(transition_bandwidth),
        ),
    }


def _world_transition_structure_summary(
    transition_log_probs: Tensor,
    *,
    transition_structure: dict[str, object] | None,
    family: str,
    bandwidth: int,
) -> dict[str, object]:
    if transition_structure is not None:
        return _world_transition_structure_summary_from_structure(
            transition_structure=transition_structure,
            family=family,
            bandwidth=bandwidth,
            dtype=transition_log_probs.dtype,
        )
    if transition_log_probs.dim() == 1:
        probs = transition_log_probs.exp()
        return {
            "family": str(family),
            "bandwidth": int(bandwidth),
            "diag_mass_mean": round(float(probs.mean().item()), 4),
            "off_diag_mass_mean": 0.0,
            "band_mass_mean": round(float(probs.mean().item()), 4),
            "top_destinations": [int(index) for index in torch.topk(probs, k=min(3, probs.shape[0])).indices.tolist()],
        }
    probs = transition_log_probs.exp()
    num_states = int(probs.shape[0])
    positions = torch.arange(num_states, device=probs.device)
    diag_mask = positions[:, None] == positions[None, :]
    band_mask = (positions[:, None] - positions[None, :]).abs() <= max(0, int(bandwidth))
    diag_mass = probs.masked_select(diag_mask).mean() if bool(diag_mask.any().item()) else probs.new_tensor(0.0)
    off_diag_mask = ~diag_mask
    off_diag_mass = probs.masked_select(off_diag_mask).mean() if bool(off_diag_mask.any().item()) else probs.new_tensor(0.0)
    band_mass = probs.masked_select(band_mask).mean() if bool(band_mask.any().item()) else probs.new_tensor(0.0)
    first_row_top = torch.topk(probs[0], k=min(3, probs.shape[1]))
    return {
        "family": str(family),
        "bandwidth": int(bandwidth),
        "diag_mass_mean": round(float(diag_mass.item()), 4),
        "off_diag_mass_mean": round(float(off_diag_mass.item()), 4),
        "band_mass_mean": round(float(band_mass.item()), 4),
        "top_destinations": [int(index) for index in first_row_top.indices.tolist()],
    }


def _world_transition_structure_summary_from_structure(
    *,
    transition_structure: dict[str, object],
    family: str,
    bandwidth: int,
    dtype: torch.dtype,
) -> dict[str, object]:
    normalized_family = str(family).strip().lower() or "banded"
    if normalized_family == "diag":
        base_transition_logits = transition_structure.get("base_transition_logits")
        source_logits = transition_structure.get("source_logits")
        dest_logits = transition_structure.get("dest_logits")
        stay_logits = transition_structure.get("stay_logits")
        if not isinstance(base_transition_logits, Tensor):
            probs = torch.empty((0,), dtype=dtype)
        else:
            diag = torch.diagonal(base_transition_logits, 0).to(dtype=dtype)
            interaction = (source_logits.to(dtype=dtype) * dest_logits.transpose(0, 1).to(dtype=dtype)).sum(dim=-1)
            probs = (diag + interaction + stay_logits.to(dtype=dtype)).exp()
        top = torch.topk(probs, k=min(3, int(probs.shape[0]))) if probs.numel() > 0 else None
        return {
            "family": str(family),
            "bandwidth": int(bandwidth),
            "diag_mass_mean": round(float(probs.mean().item()), 4) if probs.numel() > 0 else 0.0,
            "off_diag_mass_mean": 0.0,
            "band_mass_mean": round(float(probs.mean().item()), 4) if probs.numel() > 0 else 0.0,
            "top_destinations": [int(index) for index in (top.indices.tolist() if top is not None else [])],
        }
    rows = _structured_transition_probability_rows(
        transition_structure=transition_structure,
        family=family,
        bandwidth=bandwidth,
        dtype=dtype,
    )
    if not rows:
        return {
            "family": str(family),
            "bandwidth": int(bandwidth),
            "diag_mass_mean": 0.0,
            "off_diag_mass_mean": 0.0,
            "band_mass_mean": 0.0,
            "top_destinations": [],
        }
    probs = torch.stack(rows, dim=0)
    num_states = int(probs.shape[0])
    positions = torch.arange(num_states, device=probs.device)
    diag_mask = positions[:, None] == positions[None, :]
    band_mask = (positions[:, None] - positions[None, :]).abs() <= max(0, int(bandwidth))
    diag_mass = probs.masked_select(diag_mask).mean() if bool(diag_mask.any().item()) else probs.new_tensor(0.0)
    off_diag_mask = ~diag_mask
    off_diag_mass = probs.masked_select(off_diag_mask).mean() if bool(off_diag_mask.any().item()) else probs.new_tensor(0.0)
    band_mass = probs.masked_select(band_mask).mean() if bool(band_mask.any().item()) else probs.new_tensor(0.0)
    first_row_top = torch.topk(probs[0], k=min(3, probs.shape[1]))
    return {
        "family": str(family),
        "bandwidth": int(bandwidth),
        "diag_mass_mean": round(float(diag_mass.item()), 4),
        "off_diag_mass_mean": round(float(off_diag_mass.item()), 4),
        "band_mass_mean": round(float(band_mass.item()), 4),
        "top_destinations": [int(index) for index in first_row_top.indices.tolist()],
    }


def _structured_transition_probability_rows(
    *,
    transition_structure: dict[str, object],
    family: str,
    bandwidth: int,
    dtype: torch.dtype,
) -> list[Tensor]:
    base_transition_logits = transition_structure.get("base_transition_logits")
    source_logits = transition_structure.get("source_logits")
    dest_logits = transition_structure.get("dest_logits")
    stay_logits = transition_structure.get("stay_logits")
    if not isinstance(base_transition_logits, Tensor):
        return []
    if not isinstance(source_logits, Tensor) or not isinstance(dest_logits, Tensor) or not isinstance(stay_logits, Tensor):
        return []
    normalized_family = str(family).strip().lower() or "banded"
    if normalized_family == "diag":
        diag = torch.diagonal(base_transition_logits, 0).to(dtype=dtype)
        interaction = (source_logits.to(dtype=dtype) * dest_logits.transpose(0, 1).to(dtype=dtype)).sum(dim=-1)
        probs = (diag + interaction + stay_logits.to(dtype=dtype)).exp()
        return [probs]
    num_states = int(base_transition_logits.shape[0])
    rows: list[Tensor] = []
    for source_index in range(num_states):
        row_logits = base_transition_logits[source_index].to(dtype=dtype)
        row_logits = row_logits + torch.matmul(source_logits[source_index].to(dtype=dtype), dest_logits.to(dtype=dtype))
        row_logits = row_logits.clone()
        row_logits[source_index] = row_logits[source_index] + stay_logits[source_index].to(dtype=dtype)
        if normalized_family == "banded":
            positions = torch.arange(num_states, device=row_logits.device)
            mask = (positions - int(source_index)).abs() <= max(0, int(bandwidth))
            row_logits = row_logits.masked_fill(~mask, float("-inf"))
        rows.append(torch.softmax(row_logits, dim=-1))
    return rows


def _tensor_norm_mean(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    flattened = value.float().reshape(value.shape[0], -1)
    return float(torch.linalg.vector_norm(flattened, dim=-1).mean().item())


def _entropy_mean(probs: Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    safe = probs.clamp_min(1.0e-8)
    entropy = -(safe * safe.log()).sum(dim=-1)
    return float(entropy.mean().item())
