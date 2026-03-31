# TOLBERT Universal Family and Liftoff Architecture

## Goal

The long-term goal is to establish TOLBERT as the full encoder-latent-decoder universal family for the kernel.

Before liftoff:

- `vllm` remains the authoritative free-form decoder and policy runtime
- the live TOLBERT service supplies the encoder/retrieval compiler slice
- retained hybrid TOLBERT-family checkpoints learn latent dynamics, transition/value/risk surfaces, and decoder control in shadow mode
- new TOLBERT-family checkpoints are trained and evaluated under the existing retain/reject loop

After liftoff:

- a retained TOLBERT-family checkpoint becomes the default runtime for approved task families
- `vllm` moves to fallback, arbitration, distillation, and exploration roles

Liftoff is not a naming change. It is the point where retained TOLBERT-family checkpoints measurably outperform the `vllm` baseline without weakening verifier, trust, or side-effect posture.

## Architectural rule

The kernel should not split into separate "agent code" and "model project" repositories. The model stack should live inside [`agent_kernel/`](/data/agentkernel/agent_kernel), but in a dedicated modeling subtree so checkpoint, training, and inference code do not tangle with the core runtime loop.

Planned package boundary:

- [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling): model code, checkpoint interfaces, training/eval utilities, and TOLBERT-family components
- [`agent_kernel/modeling/ssm/`](/data/agentkernel/agent_kernel/modeling/ssm): selective-scan and state-space modeling primitives for improved TOLBERT work
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py): task execution loop
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py): runtime routing between model capabilities and fallbacks
- [`agent_kernel/preflight.py`](/data/agentkernel/agent_kernel/preflight.py): trust and safety gates
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py): retain/reject lifecycle for runtime artifacts

The rule is:

- model training, checkpoint wiring, learned world-model code, and learned policy heads belong under `agent_kernel/modeling/`
- custom CUDA and Triton kernels for improved-model inference belong under `agent_kernel/modeling/ssm/kernels/`
- task execution, sandboxing, verifier orchestration, reporting, and unattended trust logic stay in the main kernel modules

## Current seed

Today the repo already has the seed of this architecture:

- the seed TOLBERT service compiles a strict `ContextPacket` in [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
- the runtime orders TOLBERT before policy decoding in [`docs/architecture.md`](/data/agentkernel/docs/architecture.md)
- retained retrieval assets already materialize a TOLBERT runtime bundle in [`agent_kernel/tolbert_assets.py`](/data/agentkernel/agent_kernel/tolbert_assets.py)
- the retained-artifact loop already decides whether runtime changes should survive in [`agent_kernel/subsystems.py`](/data/agentkernel/agent_kernel/subsystems.py)
- the first internal hybrid latent/runtime path now lives in [`agent_kernel/modeling/tolbert/`](/data/agentkernel/agent_kernel/modeling/tolbert)

That is enough to support staged expansion without inventing a second architecture.

## Staged expansion

### Stage 0: Seed

Current live role:

- path prediction
- branch/global retrieval
- typed context compilation
- bounded deterministic command and skill-ranking influence

Authority:

- `vllm` remains the sole authoritative free-form decoder
- the TOLBERT family is broader than the live service, but only its compiler slice is authoritative today

### Stage 1: TOLBERT Policy Heads

Add learned heads for:

- bounded command decoding
- stop vs continue decisions
- verifier-pass likelihood
- action risk scoring

At this stage, TOLBERT acts in shadow mode first, then in limited-authority lanes for narrow families where verifier outcomes are strong and action formats are constrained.

The repo now has the bounded runtime handoff substrate for this:

- retained `runtime_policy` inside `tolbert_model` artifacts
- shadow and primary family routing in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
- liftoff-gate reporting in [`agent_kernel/modeling/evaluation/liftoff.py`](/data/agentkernel/agent_kernel/modeling/evaluation/liftoff.py) and [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)

### Stage 2: TOLBERT Latent Dynamics, Transition, and World Modeling

Add learned heads for:

- next-state transition prediction
- no-progress and regression prediction
- side-effect risk prediction
- retrieval reranking by expected future utility
- learned latent state updates using state-space dynamics

This is also the point where improved TOLBERT variants can start using
selective-scan and broader state-space modeling machinery as first-class model
primitives rather than keeping them as external research dependencies.

This does not replace the current symbolic world model in [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py). The symbolic model remains the explicit state/accounting layer. Learned world modeling augments it with generalization and forecasting.

### Stage 3: TOLBERT Decoder Runtime

When retained checkpoints are non-regressive, the policy layer can route approved task families to TOLBERT-first decoding.

At this point:

- TOLBERT-family decoder control becomes primary for approved benchmark families
- `vllm` remains available for fallback, arbitration, and teacher generation
- trust and preflight gates still govern whether unattended execution is allowed

### Stage 4: Liftoff

Liftoff occurs only when the kernel can retain a TOLBERT-family runtime bundle that is stronger than the `vllm` baseline across repeated evals and unattended runs.

Liftoff criteria should include:

- higher or equal pass rate
- no safety regression
- no hidden side-effect regression
- no degradation in long-horizon or failure-recovery lanes
- stable retained gains across multiple cycles

## Runtime roles before and after liftoff

Before liftoff:

- `vllm` controls
- TOLBERT learns

During takeover:

- `vllm` arbitrates
- TOLBERT competes

After liftoff:

- TOLBERT controls
- `vllm` assists

## Planned model surfaces

TOLBERT should not be reduced to either "the old encoder" or "just a decoder." It is the full universal family with separate but coordinated surfaces:

- encoder and retrieval/path surface
- latent/state-space dynamics surface
- action decoding surface
- value and ranking surface
- transition and world-model surface
- stop/risk surface

This matches the kernel better than a monolithic replacement because the kernel is already structured around typed context, verifier outcomes, world-state summaries, and retained artifacts.

## Training loop

The kernel already emits the ingredients needed for a self-improving model loop:

- task prompt
- typed context packet
- workspace and world-model summary
- chosen action
- verifier result
- state transition
- recovery outcome
- trust outcome
- final task outcome

Those signals should feed supervised, ranking, and transition-model objectives for TOLBERT-family checkpoints.

The live kernel now emits these as separate dataset surfaces inside retained
`tolbert_model_bundle` artifacts:

- episode-level supervised examples
- policy/action examples
- transition examples
- value examples
- stop/continue examples

Kernel autobuild readiness should stay false unless those head-specific targets
clear their retained minimum coverage floors, not just the coarse total-example
floor.

Before any improved-model CUDA path is allowed to autobuild at runtime, the
retained `tolbert_model_bundle` should show that the kernel has accumulated a
meaningful supervised dataset and a meaningful synthetic-worker slice. In
practice this means:

- implicit kernel autobuild is disabled unless a retained `build_policy` allows it
- `build_policy` should require both a total-example floor and a synthetic-example floor
- synthetic dataset generation is treated as readiness evidence for improved-model work, not as proof of liftoff

Explicit operator builds are still allowed. The restriction is on unattended
or opportunistic runtime compilation.

## Planned artifact types

The retained-artifact loop should eventually grow model-native artifacts such as:

- `tolbert_policy_checkpoint`
- `tolbert_world_model_checkpoint`
- `tolbert_decoder_checkpoint`
- `tolbert_latent_runtime_checkpoint`
- `tolbert_runtime_bundle`
- `liftoff_gate_report`
- `build_policy` readiness inside retained `tolbert_model_bundle` artifacts

The important rule is that a checkpoint is not trusted because it exists. It is trusted only after the kernel retains it through the same measured candidate-vs-baseline process used elsewhere in the repo.

## Modeling package layout

The intended layout under [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling) is:

- `backends/`: inference backends and checkpoint loaders
- `ssm/`: selective-scan wrappers and state-space building blocks
- `ssm/kernels/`: custom CUDA and Triton kernels for improved-model paths
- `tolbert/`: TOLBERT-family architecture code
- `policy/`: action-decoding heads and routing adapters
- `world/`: learned transition and world-model components
- `training/`: training jobs, dataset builders, and loss wiring
- `evaluation/`: head-to-head checkpoint evaluation and liftoff gating
- `artifacts/`: checkpoint manifests and retained runtime bundle metadata

Exact filenames can evolve. The boundary should not.

The initial concrete implementation is now present:

- [`agent_kernel/modeling/artifacts.py`](/data/agentkernel/agent_kernel/modeling/artifacts.py)
- [`agent_kernel/modeling/policy/runtime.py`](/data/agentkernel/agent_kernel/modeling/policy/runtime.py)
- [`agent_kernel/modeling/world/latent_state.py`](/data/agentkernel/agent_kernel/modeling/world/latent_state.py)
- [`agent_kernel/modeling/evaluation/liftoff.py`](/data/agentkernel/agent_kernel/modeling/evaluation/liftoff.py)

## Non-goals

This architecture does not imply:

- removing verifier control
- removing the symbolic world model
- allowing checkpoints to bypass trust/preflight gates
- replacing measured retention with manual checkpoint promotion

Liftoff is a governed takeover, not an unconditional swap.
