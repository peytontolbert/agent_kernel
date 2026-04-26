# Resource Protocol

## Purpose

The repo already had retained artifacts, rollback, compatibility checks, and subsystem metadata, but
those surfaces were still reached through subsystem-specific helpers and direct file reads.

This document defines the first `RSPL-lite` slice for `agent-kernel`: a small uniform resource layer
that treats selected runtime inputs as passive versioned resources instead of hidden runtime glue.

Current code:

- [`agent_kernel/resource_types.py`](/data/agentkernel/agent_kernel/resource_types.py)
- [`agent_kernel/resource_registry.py`](/data/agentkernel/agent_kernel/resource_registry.py)

## Scope

This first slice is intentionally narrow.

It covers:

- base prompt templates
  - `prompt:system`
  - `prompt:decision`
  - `prompt:reflection`
- builtin improvement artifacts loaded from the kernel subsystem registry
  - `subsystem:policy`
  - `subsystem:universe`
  - `subsystem:universe_constitution`
  - `subsystem:operating_envelope`
  - and the other builtin subsystem artifact paths declared in [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json)

It does not yet turn the whole runtime into a general AGP-style resource substrate.
Plans, agent-role policies, and episode memory are still not first-class resource kinds.

## Core Model

The protocol adds two data types:

- `ResourceDescriptor`
  - stable `resource_id`
  - `kind`
  - active path
  - payload format
  - artifact kind
  - allowed lifecycle states
- `ResourceVersion`
  - resolved active payload
  - file-backed `version_id`
  - lifecycle state
  - lineage metadata when present

`ResourceRegistry` is the loader and resolver:

- register descriptors
- resolve active paths
- load plain-text resources
- load JSON artifact resources
- filter artifact payloads to active lifecycle states when the file carries retention metadata

## Runtime Integration

The current runtime integration points are:

- [`agent_kernel/extensions/strategy/subsystems.py`](/data/agentkernel/agent_kernel/extensions/strategy/subsystems.py)
  - builtin subsystem artifact paths now resolve through the registry instead of direct `getattr(config, ...)`
- [`agent_kernel/extensions/policy_runtime_support.py`](/data/agentkernel/agent_kernel/extensions/policy_runtime_support.py)
  - prompt-policy artifacts now load through the registry
  - base prompt templates are exposed through `prompt_template(...)`
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
  - `LLMDecisionPolicy` now loads the base `system` and `decision` prompts through resource resolution
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py)
  - constitution, operating envelope, and legacy universe contract now load through resource ids rather than path-specialized helpers

## Why This Helps

This does three useful things immediately:

- it removes a few of the highest-value direct file reads from the runtime
- it gives prompt and governance inputs the same version/lineage vocabulary already used by retained artifacts
- it creates one place to extend active-resource loading instead of adding more subsystem-specific loaders

In AGP terms, this is closer to `RSPL-lite` than full RSPL:

- resources are passive
- active versions are explicit
- lifecycle-state filtering is standardized for the covered surfaces
- rollback and retention still live in the existing improvement engine

## Current Limits

The main gaps are still real:

- prompt templates are plain-file resources, not yet candidate/retained prompt objects
- agent-role policy is still code-level logic, not a registered resource
- memory is explicit but not yet loaded as a versioned resource family
- plans are still stateful runtime structures, not first-class plan resources
- `Improve`, `Evaluate`, and `Commit` still rely on the existing artifact engine rather than full first-class protocol objects

So this patch should be read as substrate preparation, not completion.

## Next Steps

The next useful upgrades are:

1. promote prompt templates into full prompt resources with candidate and retained versions
2. make planner state and role-routing policy resource-backed
3. decide whether episode-memory summaries should stay derived artifacts or become a first-class memory resource family
4. decide whether artifact generation and retention decisions should be reified as general resource mutations rather than artifact-specific helpers
