# Self-Evolution Protocol

## Purpose

The repo already had a retained-artifact improvement loop, but the `Reflect` and `Select` stages were
mostly implicit in planner heuristics, observe-phase notes, and variant-selection metadata.

This document defines the first `SEPL-lite` slice for `agent-kernel`: explicit reflection and selection
records that sit on top of the existing retention engine.

Current code:

- [`agent_kernel/reflection.py`](/data/agentkernel/agent_kernel/reflection.py)
- [`agent_kernel/selection.py`](/data/agentkernel/agent_kernel/selection.py)
- [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)

## Scope

This slice does not replace the existing compare-and-retain path.

It adds:

- `ReflectionRecord`
  - one explicit failure-localization record per selected cycle target
  - targets a concrete `resource_id`
  - captures the active resource version, observe-phase hypothesis, and the causal hypothesis used to justify mutation
- `SelectionRecord`
  - one explicit edit-planning record per selected cycle target
  - targets the same concrete `resource_id`
  - captures the chosen variant, strategy candidate, search budget, and the edit plan that will generate a candidate artifact
- stamped protocol lineage on generated artifacts
  - `reflection_id`
  - `selection_id`
  - the normalized reflection and selection payloads

It does not yet add:

- first-class `Improve`, `Evaluate`, and `Commit` protocol objects
- plan resources
- agent-role resources
- versioned memory resources

## Core Objects

`ReflectionRecord` contains:

- `reflection_id`
- `cycle_id`
- `subsystem`
- `resource_id`
- `resource_kind`
- `summary`
- `hypothesis`
  - confidence
  - rationale
  - source
  - failure modes
  - evidence labels
  - expected signals
- active resource version metadata
- observe-phase status/provider
- metrics digest

`SelectionRecord` contains:

- `selection_id`
- `cycle_id`
- `subsystem`
- `resource_id`
- `resource_kind`
- selected variant metadata
- strategy candidate metadata
- search-budget metadata
- `ResourceEditPlan`
  - mutation kind
  - active version metadata
  - expected artifact kind
  - controls
  - generation kwargs

## Runtime Integration

[`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py) now emits:

1. `observe`
2. `reflect`
3. `select`
4. `generate`
5. `evaluate`
6. `retain` or `reject`
7. `record`

The current integration points are:

- observe-phase ranking still produces `observe_hypothesis`
- [`build_reflection_record(...)`](/data/agentkernel/agent_kernel/reflection.py) converts that plus experiment evidence into a resource-scoped causal hypothesis
- [`build_selection_record(...)`](/data/agentkernel/agent_kernel/selection.py) converts the chosen variant and strategy candidate into a resource edit plan
- [`stamp_artifact_generation_context(...)`](/data/agentkernel/agent_kernel/extensions/improvement/artifacts.py) now carries reflection and selection lineage into the generated candidate artifact

## Why This Helps

This closes a real protocol gap without disturbing the stable retention path.

It helps because:

- the selected runtime mutation now points at a concrete resource id, not only a subsystem name
- reflection and selection can be audited from cycle history without reconstructing planner heuristics by hand
- generated artifacts now carry the causal and selection context that produced them
- the loop is closer to an explicit control protocol: `Reflect -> Select -> Improve -> Evaluate -> Commit`

## Current Limits

This is still `SEPL-lite`, not a complete self-evolution substrate.

Current limits:

- reflection is still heuristic; it does not yet run a deeper causal trace or counterfactual analysis pass
- selection still plans one primary resource mutation at a time rather than coordinated multi-resource commits
- `Improve`, `Evaluate`, and `Commit` still reuse existing artifact generation and retention machinery rather than their own first-class protocol records
- plans, agent profiles, and memory are still outside the explicit protocol surface

## Next Steps

The next useful upgrades are:

1. promote generated candidate artifacts and retention decisions into explicit `Improve` and `Commit` protocol records
2. turn plan state into a first-class versioned resource and attach selection records to plan lineage
3. turn agent-role policy into a resource family instead of code-only role wiring
4. decide whether memory should remain derived evidence or become a rollbackable resource family
