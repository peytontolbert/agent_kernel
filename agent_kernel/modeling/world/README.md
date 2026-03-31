# Learned World Modeling

This package is for learned world-model components that complement, but do not
replace, the symbolic runtime world model in
[`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py).

Current owned surfaces:

- [`latent_state.py`](/data/agentkernel/agent_kernel/modeling/world/latent_state.py): lightweight latent-state summaries for rollout scoring
- [`rollout.py`](/data/agentkernel/agent_kernel/modeling/world/rollout.py): learned-policy rollout heuristics over symbolic summaries
- [`causal_machine.py`](/data/agentkernel/agent_kernel/modeling/world/causal_machine.py): profile-loaded latent-state priors inspired by `StateSpace_CausalMachine`
- [`counterfactual.py`](/data/agentkernel/agent_kernel/modeling/world/counterfactual.py): counterfactual group parsing for future ablation/eval lanes
- [`belief_scan.py`](/data/agentkernel/agent_kernel/modeling/world/belief_scan.py): owned belief-scan API with Python reference and native CUDA dispatch

Current kernel boundary:

- [`kernels/cuda_belief_scan.py`](/data/agentkernel/agent_kernel/modeling/world/kernels/cuda_belief_scan.py): explicit-build native CUDA wrapper
- [`kernels/src/causal_belief_scan.cpp`](/data/agentkernel/agent_kernel/modeling/world/kernels/src/causal_belief_scan.cpp): extension bindings and validation
- [`kernels/src/causal_belief_scan_cuda.cu`](/data/agentkernel/agent_kernel/modeling/world/kernels/src/causal_belief_scan_cuda.cu): chunked carried-state belief update kernel

Planned next additions:

- incremental cache equivalence checks for learned world-state updates
- retained latent-state profile artifacts generated from synthetic and replay trajectories
- counterfactual ablation reports for `retrieval`, `state`, `policy`, `transition`, and `risk`

Non-goals:

- copying lexical shortcut or bigram trainer features from external repos
- mixing learned-world runtime code into the symbolic control plane

Operational guardrails:

- CUDA callers fail fast when the native world-model kernel is unavailable unless they explicitly set `prefer_python_ref=True`
- implicit CUDA autobuild follows the retained `tolbert_model_bundle` build-policy gate used by the `ssm` path
- the native world-model kernel build avoids `--use_fast_math` because belief updates run in log space
