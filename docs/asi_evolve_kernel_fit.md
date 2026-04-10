# ASI-Evolve Fit For AgentKernel

## Goal

Add one structured layer to the kernel that improves unattended compounding without creating a parallel framework.

The best fit from [`other_repos/ASI-Evolve`](/data/agentkernel/other_repos/ASI-Evolve) is not its full agent split. The best fit is a persistent experiment-memory layer that unifies:

- attempted strategy
- rationale
- bounded code/action result
- analyzer lesson
- sampler state
- best retained snapshot

For AgentKernel, this should land as a new stack addition:

`strategy_memory`

not as a second orchestration system.

## Why This Is The Best Fit

Current kernel strengths:

- unattended controller already exists: [unattended_controller.py](/data/agentkernel/agent_kernel/unattended_controller.py)
- planner and cycle records already exist: [improvement.py](/data/agentkernel/agent_kernel/improvement.py)
- bounded execution harness already exists: [harness.py](/data/agentkernel/evals/harness.py)
- coding actor contract now exists: [coding.py](/data/agentkernel/agent_kernel/actors/coding.py)

Current kernel weakness:

- strategy evidence is fragmented across cycle reports, events, status snapshots, and controller state
- retained/rejected outcomes are counted, but lessons are not first-class reusable objects
- controller adaptation is mostly pressure-driven, not parent-selection-driven
- best retained outcomes are reconstructable, but not promoted as a canonical search surface

ASI-Evolve helps exactly here:

- durable `Node` object with rationale + code + results + analysis
- analyzer-written lesson after each trial
- persistent sampler over prior attempts
- explicit best snapshot management

Those four ideas directly address our compounding gap.

## Recommended Addition

Create a new subdirectory:

`agent_kernel/strategy_memory/`

Recommended files:

- `agent_kernel/strategy_memory/__init__.py`
- `agent_kernel/strategy_memory/node.py`
- `agent_kernel/strategy_memory/store.py`
- `agent_kernel/strategy_memory/sampler.py`
- `agent_kernel/strategy_memory/lesson.py`
- `agent_kernel/strategy_memory/snapshot.py`

This should be a kernel-native memory plane for unattended strategy evolution.

## Core Object

Introduce a durable `StrategyNode`.

Required fields:

- `strategy_node_id`
- `created_at`
- `parent_strategy_node_ids`
- `cycle_id`
- `subsystem`
- `selected_variant_id`
- `strategy_candidate_id`
- `strategy_candidate_kind`
- `motivation`
- `controls`
- `actor_summary`
- `results_summary`
- `retention_state`
- `retained_gain`
- `analysis_lesson`
- `score`
- `visit_count`
- `family_coverage`
- `artifact_paths`

This is the kernel equivalent of ASI-Evolve's experiment node, but grounded in our unattended loop.

## Exact Integration Points

### 1. Planner Write Path

In [`improvement.py`](/data/agentkernel/agent_kernel/improvement.py):

- when `select_portfolio_campaign()` chooses a subsystem, variant, or synthesized strategy, create a pending `StrategyNode`
- attach parent links from the best prior retained or most-relevant prior nodes
- store the exact rationale and controls that led to selection

Why:

- right now strategy choice is mostly ephemeral and flattened into cycle records

### 2. Execution Attachment

In [`harness.py`](/data/agentkernel/evals/harness.py) and [`coding.py`](/data/agentkernel/agent_kernel/actors/coding.py):

- attach coding-actor summary to the `StrategyNode`
- persist files touched, tests run, verifier status, runtime budget, and retrieval usage

Why:

- this turns actor execution into searchable strategy evidence instead of just episode metadata

### 3. Finalize / Retention Attachment

In [`cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py) and [`run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py):

- finalize the pending `StrategyNode`
- set `retention_state`
- set `retained_gain`
- write compact results summary
- preserve neutral-retained complementary support outcomes, not only positive pass-rate outcomes

Why:

- this is where the current kernel still loses compounding signal by scattering it across reports

### 4. Lesson Synthesis

Add a kernel-native lesson synthesizer.

It does not need a new heavyweight agent. It can be a structured summarizer that produces:

- what changed
- what signal improved
- what signal regressed
- when to reuse this strategy
- when to avoid this strategy

Input sources:

- cycle metrics
- actor summary
- retention reason
- family coverage deltas
- retrieval support deltas

Output:

- `analysis_lesson`
- `reuse_conditions`
- `avoid_conditions`

Why:

- our kernel counts decisions, but does not yet retain decision-shaped lessons as first-class policy priors

### 5. Strategy Sampling

Add a sampler over retained and rejected `StrategyNode`s.

Use simple kernel-native sampling first:

- UCB-style exploration bonus
- retained-gain reward
- neutral-retain reward floor
- reject penalty
- family-diversity bonus
- stale-strategy recency decay

Do not start with full island/MAP-Elites complexity unless needed.

Why:

- the unattended controller should not only shift policy weights
- it should also sample concrete prior strategies as parents or priors for new interventions

### 6. Best Snapshot Surface

Add best retained strategy snapshots.

Persist:

- best retained by subsystem
- best retained by family cluster
- best retained coding-actor outcome
- best retained retrieval-support strategy

Why:

- this gives the kernel a canonical high-signal reuse surface instead of reconstructing “best so far” from mixed reports

## What Not To Import

Do not import these ASI-Evolve concepts as-is:

- a separate manager/researcher/engineer/analyzer orchestration loop
- a duplicate cognition store
- a second experiment entrypoint outside the unattended kernel

Why:

- we already have stronger kernel-native orchestration
- duplicating orchestration would increase state incoherence
- the goal is compounding inside our stack, not alongside it

## How This Moves Us Closer To ASI

This addition improves the kernel in the exact place it is currently weak:

- from controller pressure to reusable strategy memory
- from counted decisions to retained lessons
- from one-run outcomes to parent-linked search
- from raw artifacts to best-snapshot reuse

That is a real step toward ASI-like compounding because it gives the system a durable memory of:

- what it tried
- why it tried it
- what happened
- what should be tried next

without requiring humans to reinterpret every unattended run.

## Minimal First Implementation

First wave only:

1. Add `StrategyNode` schema and JSON store.
2. Write nodes on strategy selection and finalize them on cycle closeout.
3. Add a compact lesson synthesizer from finalized cycle data.
4. Add a lightweight UCB-style sampler over retained/rejected nodes.
5. Expose best retained snapshots in artifacts.

This is enough to materially improve unattended compounding.

## Success Criteria

This addition is successful only if the next unattended waves show all of the following:

- controller or planner selection cites prior `StrategyNode` parents, not only abstract policy pressure
- retained and rejected cycles both produce reusable lessons
- a later run reuses a retained strategy node and improves time-to-decision or retained-gain conversion
- best retained strategy snapshots become a stable input surface for new unattended rounds

## Recommended Owned Paths

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- [`agent_kernel/actors/coding.py`](/data/agentkernel/agent_kernel/actors/coding.py)
- [`evals/harness.py`](/data/agentkernel/evals/harness.py)
- [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
- `agent_kernel/strategy_memory/`

## Bottom Line

If we take one structured addition from ASI-Evolve, it should be:

`persistent strategy experiment memory with lessons, sampling, and best retained snapshots`

That fits our kernel, strengthens unattended compounding, and moves us closer to ASI without splitting the stack into two competing systems.
