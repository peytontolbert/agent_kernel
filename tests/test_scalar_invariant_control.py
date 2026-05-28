from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_STACK = ROOT / "other_repos" / "model-stack"
for path in (ROOT, MODEL_STACK):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from runtime.seq2seq import EncoderDecoderLM, ScalarInvariantControl
from specs.config import ModelConfig


def _tiny_config(**overrides) -> ModelConfig:
    values = {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 64,
        "vocab_size": 64,
        "attn_impl": "eager",
        "activation": "silu",
        "norm": "layer",
        "scalar_invariant_control": False,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_scalar_invariant_control_is_zero_initialized_baseline_preserving():
    torch.manual_seed(1)
    control = ScalarInvariantControl(d_model=16, rank=4, epsilon=0.1, smoothing_steps=1)
    hidden = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.long)

    out = control(hidden, mask)

    assert torch.equal(out, hidden)
    diagnostics = control.diagnostics()
    assert diagnostics["source_mean"] > 0.0
    assert diagnostics["update_norm"] == 0.0


def test_encoder_decoder_scalar_control_preserves_forward_shape_and_reports_diagnostics():
    torch.manual_seed(2)
    model = EncoderDecoderLM(
        _tiny_config(
            scalar_invariant_control=True,
            scalar_invariant_rank=4,
            scalar_invariant_epsilon=0.05,
            scalar_invariant_smoothing_steps=1,
            scalar_invariant_apply_encoder=True,
            scalar_invariant_apply_decoder=False,
        ),
        tie_embeddings=True,
    )
    enc = torch.randint(0, 64, (2, 7))
    dec = torch.randint(0, 64, (2, 4))
    mask = torch.ones_like(enc)

    logits = model(enc, dec, mask)

    assert tuple(logits.shape) == (2, 4, 64)
    diagnostics = model.scalar_control_diagnostics()
    assert "encoder" in diagnostics
    assert diagnostics["encoder"]["update_norm"] == 0.0
