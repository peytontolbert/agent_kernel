# ASI Core

## Purpose

This repo has a large surface area, but not all of it is the same kind of work.
If we want autonomous self-improving intelligence rather than a growing pile of helpful machinery,
we need a stable definition of what counts as core and what counts as auxiliary.

This document defines that boundary so planning, reviews, and roadmaps can stay targeted.

## Thesis

There are three useful minimums in this repository, not one:

1. the smallest executable autonomous agent
2. the smallest runtime that actually looks ASI-shaped
3. the smallest self-improving machine layered on top of that runtime

That distinction matters because the repo contains a lot of useful machinery that improves the machine without defining its irreducible core.

## Three Minimums

### 1. Executable Agent Minimum

This is the smallest thing here that is still an autonomous coding agent.
Its irreducible contract is:

- `task -> state`
- `state -> action`
- `action -> bounded execution`
- `execution -> success or failure`
- `step updates -> stopping and persistence`

Primary modules:

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:192)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:61)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:63)
- [`agent_kernel/schemas.py`](/data/agentkernel/agent_kernel/schemas.py:13)
- [`agent_kernel/actions.py`](/data/agentkernel/agent_kernel/actions.py:1)
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28)
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:10)
- [`agent_kernel/llm.py`](/data/agentkernel/agent_kernel/llm.py:1)
- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:117)

This is the minimum closed-loop task executor.
Without it, there is no agent.

The strict executable floor is now available as
[`KernelConfig.executable_floor(...)`](/data/agentkernel/agent_kernel/config.py:603).
That preset disables TOLBERT context, skills, graph memory, world model, universe governance,
planner roles, role specialization, and post-episode learning-candidate compilation while still
leaving episode persistence on.

### 2. ASI-Shaped Runtime Minimum

The executable minimum is not yet the same thing as a runtime that looks like bounded autonomous intelligence.
The ASI-shaped runtime minimum adds:

- cross-step and cross-episode memory pressure
- an internal progress representation
- explicit governance structure above raw execution

Primary additions:

- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:21)
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:75)
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:14)

This is the right place to draw the runtime-side ASI line.
It is no longer just an LLM loop with a verifier.

Governance note:

- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28) is the hard enforcement floor
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:75) is the declarative governance layer

That split is now enforced in the live loop as a two-layer control surface:

- universe governance can reject a command before execution with an explicit `governance_rejected` step result
- the sandbox remains the lower hard containment boundary for commands that still reach execution

The live runtime now makes that distinction explicit too:

- `use_universe_model=0` removes declarative governance injection from the executor path
- the `bounded_autonomous` runtime claim now requires a retained native decoder posture together with
  `use_graph_memory=1` and `use_world_model=1`
- `persist_learning_candidates=0` lets the executable floor end at episode persistence instead of entering the self-improvement pipeline

### 3. Minimum Self-Improving ASI

The irreducible self-improving layer is smaller than the full improvement stack.
Its contract is:

- compile episodes into reusable evidence
- rank improvement targets and variants
- compare candidate and baseline behavior
- decide retain or reject
- apply that decision to live artifacts

Irreducible modules:

- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:80)
- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:2797)

This is the minimum compounding machine.
It is the point where the runtime can start changing itself under verifier pressure.

Meta-loop support structure lives around that minimum rather than inside it:

- [`agent_kernel/extensions/strategy/subsystems.py`](/data/agentkernel/agent_kernel/extensions/strategy/subsystems.py:40)
- [`agent_kernel/strategy_memory/`](/data/agentkernel/agent_kernel/strategy_memory)
- [`agent_kernel/extensions/improvement/improvement_plugins.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_plugins.py:1)
- [`agent_kernel/extensions/improvement/improvement_support_validation.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_support_validation.py:1)

Those pieces are valuable, but they are support structure for the meta-loop, not the meta-loop essence.

## Full ASI Loop

The full machine in this repo is two nested loops:

1. an inner bounded runtime loop that tries to solve one task
2. an outer self-improvement loop that changes the runtime through retained artifacts

In compact form:

```text
task
  -> workspace/checkpoint bootstrap
  -> setup commands
  -> memory + world + governance bootstrap
  -> policy decision
  -> governance check
  -> bounded execution
  -> verification
  -> state-transition estimation
  -> state + memory update
  -> stop or repeat
  -> episode persistence
  -> learning-candidate compilation
  -> eval-driven target selection
  -> candidate artifact generation
  -> baseline vs candidate comparison
  -> retain or reject
  -> retained artifact feeds the next task run
```

Runtime loop ownership:

- [`scripts/run_agent.py`](/data/agentkernel/scripts/run_agent.py:37): CLI/bootstrap shell around one task run
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:162): top-level task runner and step budget control
- [`agent_kernel/ops/loop_run_support.py`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:75): runtime-state bootstrap, per-step execution, and episode finalization
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:164): action selection
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:315): pre-execution governance rejection
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28): hard execution boundary
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:13): success/failure judgment
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:60): workspace-state summary and transition estimation
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:212): per-step state update, progress pressure, and stopping signals
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:22): episode persistence and graph-memory summaries

Self-improvement loop ownership:

- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:80): compile episodes into reusable learning candidates
- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:12): generic target, variant, cycle, and retention data model
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:332): rank targets, choose experiments, and choose variants
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:279): preview candidate-vs-baseline retention
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:443): finalize retain/reject and apply artifact lifecycle effects
- [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py:1608): outer-loop campaign/bootstrap entrypoint

The important boundary is:

- bootstrap and campaign scripts start or supervise the loop
- extensions can enrich the loop
- the ASI core is the closed path from `task` to `episode` to `learning evidence` to `retained artifact` and back into the next `task`

## What Is Auxiliary

Auxiliary layers can raise capability, reliability, or scale, but they are not minimum-defining.

The main auxiliary classes in this repo are:

- learned execution and retrieval specifics
  - [`agent_kernel/extensions/tolbert.py`](/data/agentkernel/agent_kernel/extensions/tolbert.py)
  - [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling)
- subsystem-specific proposal surfaces
  - [`agent_kernel/extensions/improvement/`](/data/agentkernel/agent_kernel/extensions/improvement) generators for prompts, retrieval, curriculum, trust, transition models, and related surfaces
- task and curriculum supply
  - [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
  - [`agent_kernel/tasking/curriculum.py`](/data/agentkernel/agent_kernel/tasking/curriculum.py)
- orchestration and operating machinery
  - [`agent_kernel/ops/unattended_controller.py`](/data/agentkernel/agent_kernel/ops/unattended_controller.py)
  - [`agent_kernel/ops/job_queue.py`](/data/agentkernel/agent_kernel/ops/job_queue.py)
  - [`agent_kernel/ops/preflight.py`](/data/agentkernel/agent_kernel/ops/preflight.py)

Important examples:

- TOLBERT, Qwen, and retrieval-stack specifics are implementation choices, not the abstract minimum
- planner roles, critic roles, and subgoal machinery are optional runtime enhancements
- world model and memory are optional for strict executability, but part of the ASI-shaped runtime line
- `extensions/strategy/subsystems.py` and `strategy_memory/` strengthen the meta-loop, but do not define its irreducible essence

## Targeting Rule

When deciding where work belongs, use this rule:

- `executable-agent core`: the change modifies the closed-loop task executor
- `ASI runtime core`: the change modifies memory, world-model, or governance structure that makes the runtime ASI-shaped
- `ASI self-improvement core`: the change modifies compile/rank/evaluate/retain/apply behavior for internal changes
- `auxiliary capability layer`: the change improves one modeling, retrieval, prompt, curriculum, or subsystem surface without changing one of those minimums
- `ops or environment layer`: the change improves controller flow, task supply, reporting, coordination, or deployment without changing one of those minimums

Useful review questions:

- Does this change alter how the agent turns state into verified action?
- Does this change alter how the machine selects and retains self-modifications?
- Or does it improve one capability surface, one controller, or one benchmark lane?

Only the middle two are ASI-core work.
The first is the executable floor that everything else depends on.

## What To Optimize For

For ASI-oriented progress, the most important work is:

1. tighter state, action, execution, verifier, and stopping integration in the executable agent floor
2. stronger memory, world-model, and governance structure in the ASI-shaped runtime
3. cleaner retained-vs-candidate search, retention, and artifact-application logic in the self-improvement minimum
4. clearer separation between generic self-improvement machinery and subsystem plugins

Important but secondary work includes:

- stronger TOLBERT or decoder surfaces
- better retrieval bundles
- broader task banks and curriculum generation
- better unattended controller ergonomics
- improved reporting and coordination

Those improvements help the machine.
They should not be mistaken for the machine's irreducible core.

## Current Refactor Direction

The biggest current boundary violation is [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py).
It mixes:

- generic search, retention, and campaign logic
- subsystem-specific proposal and scoring logic

The clean direction is:

- keep a generic self-improvement engine in the core
- move subsystem-specific proposal surfaces toward clearer plugins or adapters
- keep learned execution packages under modeling-oriented boundaries rather than pulling them into the runtime nucleus

That direction is now partially implemented:

- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1) holds shared engine data models plus generic target projection, variant orchestration, retention-gate application, and artifact lifecycle application helpers
- [`agent_kernel/extensions/improvement/improvement_plugins.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_plugins.py:1) is the current adapter seam for subsystem registry, default variants, external planner experiments, strategy-memory priors, TOLBERT delta helpers, and universe bundle helpers
- [`agent_kernel/extensions/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/extensions/runtime_modeling_adapter.py:1) is the runtime-side lazy seam for context compilation, model artifact loading, hybrid scoring, and learned world-signal enrichment
- [`agent_kernel/extensions/improvement/improvement_support_validation.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_support_validation.py:1) moves `TaskBank`-backed contract lookup behind a support validator instead of leaving it embedded in core retention code

The filesystem split now mirrors that boundary more clearly:

- [`agent_kernel/tasking/`](/data/agentkernel/agent_kernel/tasking): task-bank and curriculum support
- [`agent_kernel/ops/`](/data/agentkernel/agent_kernel/ops): preflight, runtime supervision, export governance, and recovery support
- [`agent_kernel/extensions/`](/data/agentkernel/agent_kernel/extensions): optional runtime/modeling adapters plus capability, policy-support, TOLBERT, trust, and extraction support
- [`agent_kernel/extensions/improvement/`](/data/agentkernel/agent_kernel/extensions/improvement): subsystem-specific improvement surfaces
- [`agent_kernel/extensions/strategy/`](/data/agentkernel/agent_kernel/extensions/strategy): subsystem registry and semantic support

One concrete shape would be:

- `improvement_engine.py`: target ranking, variant ranking, evidence collation, retention protocol, and lifecycle application
- `improvement_plugins.py`: subsystem registry adapters, strategy-prior access, model-specific helpers, and other support-layer bridges

That split makes future ASI-core work easier to identify and reduces the risk of treating every useful helper as a core intelligence advance.
