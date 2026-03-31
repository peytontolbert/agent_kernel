# World-Model Kernels

This directory contains owned CUDA or Triton kernels that support learned
world-model paths.

Current kernel:

- [`cuda_belief_scan.py`](/data/agentkernel/agent_kernel/modeling/world/kernels/cuda_belief_scan.py):
  explicit-build wrapper for the native causal-belief scan extension
- [`src/causal_belief_scan.cpp`](/data/agentkernel/agent_kernel/modeling/world/kernels/src/causal_belief_scan.cpp):
  local extension ABI and tensor validation
- [`src/causal_belief_scan_cuda.cu`](/data/agentkernel/agent_kernel/modeling/world/kernels/src/causal_belief_scan_cuda.cu):
  chunked latent-belief scan with carried recurrent state

Guardrails:

- no silent CUDA-to-Python downgrade unless the caller explicitly requests the Python reference path
- implicit kernel autobuild is gated by the retained Tolbert build-policy artifact
- the local build keeps IEEE-oriented math for belief normalization instead of `--use_fast_math`

Design responsibilities:

- chunked recurrent belief updates with carried latent state
- explicit forward/backward support for cached rollout mode
- parity and cache-equivalence smoke tests against a Python reference

Do not import external CUDA extensions from here. External repos are reference
material only.
