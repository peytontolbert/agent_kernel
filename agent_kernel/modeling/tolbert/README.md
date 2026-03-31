# TOLBERT Universal Family

In this repo, `TOLBERT` names the full retained model family rather than only the
original ontology encoder.

The family contract is:

- `encoder_surface`: hierarchy-aware representation and retrieval
- `latent_dynamics_surface`: recurrent or state-space task/world state updates
- `decoder_surface`: bounded and eventually free-form action generation
- `world_model_surface`: transition, recovery, and side-effect forecasting
- `policy/value/transition/risk/stop heads`: kernel-facing decision surfaces

The current implementation is split across two layers:

- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
  The seed strict service wrapper for the original encoder/retrieval compiler.
- [`hybrid_model.py`](/data/agentkernel/agent_kernel/modeling/tolbert/hybrid_model.py)
  The first internal latent/runtime candidate using state-space modeling and
  learned control heads.

The seed `vllm` runtime still owns free-form decoding today. The purpose of this
package is to make retained TOLBERT-family checkpoints capable of taking over
family by family through the existing liftoff gate, not to bypass verifier,
trust, or retention controls.
