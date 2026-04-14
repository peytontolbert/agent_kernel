# TOLBERT and Hybrid Runtime

In this repo, `TOLBERT` should be read as the seeded encoder/retrieval/compiler
system, not as the canonical name for every later retained runtime checkpoint.

The repo currently has two connected layers:

- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
  The seeded strict service wrapper for the original encoder/retrieval compiler.
- [`hybrid_model.py`](/data/agentkernel/agent_kernel/modeling/tolbert/hybrid_model.py)
  The retained modeled runtime candidate that adds latent dynamics, decoder
  control, world-model scoring, and kernel-facing decision heads.

The retained hybrid runtime contract includes:

- `encoder_surface`: hierarchy-aware representation and retrieval
- `latent_dynamics_surface`: recurrent or state-space task/world state updates
- `decoder_surface`: bounded and eventually free-form action generation
- `world_model_surface`: transition, recovery, and side-effect forecasting
- `policy/value/transition/risk/stop heads`: kernel-facing decision surfaces

The seed `vllm` runtime still owns free-form decoding today. The purpose of this
package is to host the retained hybrid runtime code next to the seeded TOLBERT
integration until retained checkpoints earn authority through the liftoff gate.
