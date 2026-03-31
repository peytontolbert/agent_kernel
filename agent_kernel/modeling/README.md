# Modeling Package

This directory is reserved for model-native code.

Use it for:

- TOLBERT-family architecture code
- checkpoint loaders and inference backends
- selective-scan and state-space modeling primitives
- training dataset builders
- external training-backend discovery and launch metadata
- learned policy heads
- learned transition and world-model code
- model evaluation and liftoff gating

Do not place the following here:

- task execution loop code
- sandbox policy code
- verifier orchestration
- unattended trust gating
- job queue control flow

Those remain in the main [`agent_kernel/`](/data/agentkernel/agent_kernel) runtime modules.

Intended subpackages:

- `backends/`
- `ssm/`
- `ssm/kernels/`
- `tolbert/`
- `policy/`
- `world/`
- `training/`
- `evaluation/`
- `artifacts/`

The purpose of this boundary is to let the model stack grow from TOLBERT seed usage to a full retained-checkpoint runtime without mixing model internals into the agentic control plane.

Model-native CUDA paths should not compile opportunistically just because a
wrapper was imported. Implicit builds should only be allowed after a retained
model artifact says the dataset threshold is high enough, including synthetic
dataset evidence for improved-model training readiness.

External model trainers that are not drop-in retained runtime artifacts should be vendored under [`other_repos/`](/data/agentkernel/other_repos) with an `agentkernel_backend.json` manifest. The `agent_kernel.modeling.training_backends` helpers and [`scripts/run_training_backend.py`](/data/agentkernel/scripts/run_training_backend.py) use that manifest to expose bounded launch metadata without forcing those trainers into the TOLBERT runtime contract.

Retained `tolbert_model_bundle` artifacts now also carry a `training_inputs`
manifest that points at the coarse supervised dataset plus the richer
policy/transition/value/stop JSONL targets. External trainers should consume
those paths from the artifact or from the exported `AGENTKERNEL_TOLBERT_*`
environment variables rather than rediscovering them ad hoc.
