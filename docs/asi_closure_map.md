# ASI Closure Map

## Purpose

The root `agent_kernel/` package is now close enough to the intended ASI-core boundary that the
main question is no longer "what should move out?" but "what is still missing from the core
machine?"

This document is the compact answer for the current codebase.

If you want the single current consolidated view across this document, the live
gap map, and the proof audit, start with
[`kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md).

## Current Core Surface

Current top-level core modules and approximate sizes:

- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1): `1217` lines
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1): `978` lines
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:1): `971` lines
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1): `955` lines
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:1): `910` lines
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:1): `876` lines
- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:1): `808` lines
- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1): `798` lines
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:1): `799` lines
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:1): `739` lines
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:1): `685` lines
- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:1): `660` lines

The main organizational work is done:

- root `agent_kernel/` is now mostly runtime and meta-loop code
- support logic has largely moved into `agent_kernel/extensions/`, `agent_kernel/ops/`, and `agent_kernel/tasking/`
- the remaining blockers are mostly substantive rather than filesystem-boundary problems

## What Is Now Mostly Complete

### 1. Executable agent floor

The repo now has a clean enough minimum closed-loop executor:

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:1)
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:1)
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:1)

The strict floor is exposed by
[`KernelConfig.executable_floor(...)`](/data/agentkernel/agent_kernel/config.py:603).

### 2. Core/meta-loop boundary

The self-improvement loop is no longer hidden inside one oversized root file.
The generic core is now split across:

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1)
- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1)
- [`agent_kernel/improvement_retention.py`](/data/agentkernel/agent_kernel/improvement_retention.py:1)
- [`agent_kernel/improvement_records.py`](/data/agentkernel/agent_kernel/improvement_records.py:1)
- [`agent_kernel/improvement_evidence.py`](/data/agentkernel/agent_kernel/improvement_evidence.py:1)
- [`agent_kernel/improvement_confirmation.py`](/data/agentkernel/agent_kernel/improvement_confirmation.py:1)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:1)

Subsystem scoring, compatibility checks, artifact lifecycle support, planner history, and strategy
surfaces are now mostly extension-side.

## Locked Closure Contract

The current repo now satisfies the locked ASI-core closure contract that this codebase has been
using for patch work. That contract is:

1. `memory.py`
   - semantic recall
   - reusable prototypes
   - prototype application semantics
2. `world_model.py`
   - sequence-aware prediction
   - multi-step rollout scoring
   - semantic-memory/prototype conditioning
3. `policy.py`
   - adaptive bounded sequence search
   - planner consumes rollout results directly
   - no shallow-preview-only path as the primary planner surface
4. `verifier.py`
   - behavior checks
   - differential checks
   - repo invariants
   - behavior-side-effect assertions
5. unattended closure/accounting surfaces
   - explicit machine-readable closure summaries for tracked gates

Status: `satisfied`

The rest of this document describes broader extension directions, but they are no longer blockers
for the locked closure contract above.

## Broader Extension Surface

### 1. Memory is now semantic, but still not fully abstract

[`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:57) now carries more than
descriptive counters. The core memory surface now includes:

- semantic episode recall over:
  - task ids
  - benchmark families
  - changed paths
  - verifier obligations
  - failure signals
- semantic prototypes that aggregate:
  - successful repair episodes
  - reusable application commands
  - repeated failure commands
  - short successful command sequences

The broader extension surface here is abstraction strength. Prototypes are now reusable repair
patterns, but they are still grouped summaries rather than a richer learned representation with
explicit transform semantics.

### 2. Verification is broader, but still rule-driven

[`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:10) now covers:

- exit code
- file existence
- exact file content
- output substrings
- behavior checks
- differential checks
- repo invariants
- regex output assertions
- JSON-field assertions
- behavior-side-effect assertions over created/changed files and repo invariants

The broader extension surface here is that these semantics are still contract/rule driven rather
than a generalized semantic correctness model.

### 3. The world model is predictive, but still short-horizon

[`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:60) now predicts with:

- task contract summaries
- workspace previews
- changed-path priors
- workflow-path expectations
- learned bootstrap and state-conditioned command patterns
- semantic-memory priors from recalled episodes
- semantic-prototype priors from reusable repair patterns
- explicit sequence rollout over short command sequences

The broader extension surface here is that rollout is still short-horizon and command-sequence
based, not a stronger latent causal transition model over repository state.

### 4. Planning is now adaptive short-horizon search

[`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:147) now does more than a fixed
tiny preview. It:

- reuses or compiles a context packet
- collects retrieval guidance and ranked skills
- builds a larger preview frontier when semantic memory or long-horizon pressure is present
- uses adaptive bounded search over short command sequences
- consumes semantic episodes and semantic prototypes during preview construction
- runs deterministic and TolBERT-assisted fallback/primary paths
- calls the LLM once on a compact payload

The broader extension surface here is hierarchy and depth. Search is now adaptive and
sequence-aware, but it is still bounded short-horizon search rather than a broader hierarchical
planner.

### 5. Retained broad gain is still not proven

The codebase is much cleaner, but the closure criterion is still empirical:

- can the child loop turn unattended work into retained gains
- across required benchmark families
- with counted external proof

The current proof gap is still tracked in
[`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md:1).

## Highest-ROI Closure Work

### 1. Stronger semantic verification

This is the highest-ROI substantive upgrade now.

Why:

- the current meta-loop can only retain what the verifier can truthfully detect
- better memory or planning without a stronger verifier risks optimizing the wrong targets
- the current root structure is clean enough that verifier expansion can now be evaluated as core progress rather than support sprawl

Primary surfaces:

- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:10)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:147)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:60)

Current floor:

- regex and JSON-field assertions
- behavior-side-effect assertions
- differential and invariant checks

Remaining direction:

- stronger correctness coverage for robustness, security, and performance
- richer machine-consumable semantic evidence beyond rule evaluation

### 2. Semantic episodic memory

Why second:

- current memory now retrieves semantic episodes and reusable prototypes
- the remaining step is stronger abstraction over those prototypes, not basic semantic recall

Primary surfaces:

- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:22)
- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:80)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:147)

Current floor:

- semantic episode indexing
- obligation-aware retrieval
- recovery and failure command recall
- prototype application commands and short command sequences

Remaining direction:

- stronger generalized prototype abstraction
- better transform semantics beyond grouped command patterns

### 3. Stronger predictive world model

Why third:

- the world model now uses semantic-memory and prototype priors plus explicit sequence rollout
- the remaining gain is stronger latent transition quality, not basic predictive scaffolding

Primary surfaces:

- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:60)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:147)

Current floor:

- edit-to-verifier-outcome priors
- semantic-memory-conditioned command prediction
- prototype-conditioned sequence rollout

Remaining direction:

- stronger latent transition state
- better uncertainty signals over longer horizons

### 4. Deeper search in policy

Why fourth:

- the policy now performs adaptive bounded sequence search
- the remaining gain is hierarchical depth, not the absence of search itself

Primary surfaces:

- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:147)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:60)

### 5. Retained-gain proof

Why fifth:

- proof is the outcome, not the first mechanism to improve
- the current kernel should strengthen its verifier, memory, and world-model substrate before treating live retained-gain widening as the main engineering loop

Primary surfaces:

- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:1)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1)
- [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md:1)

## Remaining Boundary Cleanup

The remaining largest root files are still:

- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:1)

But these are no longer the same kind of boundary problem that `improvement.py` used to be.
They are now mostly coordinator files for real core behavior.

That means the repo is at the point where ASI-closure work should bias toward substantive capability
gaps over more filesystem cleanup.

## Closure Rule

Treat a proposed task as ASI-closure work only if it improves one of these directly:

1. semantic memory
2. semantic verification
3. predictive world modeling
4. deeper planning over learned state
5. retained-gain and family-breadth proof

If it does not move one of those, it is probably auxiliary work even if it still improves the
machine.
