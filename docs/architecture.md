# Architecture

## Goal

The repository is a verifier-driven coding runtime for realistic local software work, not only a seed-task loop. The full repo surface currently combines core and support layers:

- one local policy runtime
- one bounded shell sandbox
- deterministic verification
- TOLBERT-backed encoder/retrieval compilation
- episode, skill, operator, tool, and verifier memory artifacts
- adjacent-task and failure-driven curriculum generation
- a retained-artifact improvement loop
- unattended and shared-repo workflow machinery for repo review, test repair, worker-branch coordination, and governed promotion

For the current evidence-ranked coding-AGI blockers, see
[`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md).

For the canonical boundary between the ASI-oriented kernel and auxiliary layers, see
[`asi_core.md`](/data/agentkernel/docs/asi_core.md).

## ASI Core Boundary

This repo has three useful minimums, not one:

- `executable agent minimum`: the closed-loop task executor
- `ASI-shaped runtime minimum`: the executable agent plus memory, world model, and governance structure
- `minimum self-improving ASI`: the ASI-shaped runtime plus the irreducible retain/reject meta-loop

That distinction matters because a lot of code improves the machine without defining the minimum machine.

For planning:

- if work changes schemas, state, policy, action execution, verification, or stopping behavior, it changes the executable agent minimum
- if work changes memory, world-model, or governance structure, it changes the ASI-shaped runtime minimum
- if work changes learning compilation, target ranking, candidate-vs-baseline evaluation, retention, or decision application, it changes the minimum self-improving ASI layer
- if work changes TOLBERT, retrieval, prompts, curriculum, task supply, controller ops, reporting, or deployment without changing those minimums, it is an auxiliary layer

That is not a value judgment. It is a scoping rule so the roadmap does not treat every useful improvement as equal evidence of ASI-core progress.

The repo now has an explicit strict-floor preset for this distinction:

- [`KernelConfig.executable_floor(...)`](/data/agentkernel/agent_kernel/config.py:603) disables TOLBERT context, skills, graph memory, world model, universe governance, planner roles, role specialization, and learning-candidate compilation while keeping the closed-loop executor and episode persistence intact

## Control flow

The main runtime path in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py) is:

`task -> setup commands -> graph/world/planner state -> TOLBERT context -> policy decision -> action -> verification -> episode save`

`loop.py` still owns the step loop, checkpointed episode state machine, and step-record closeout.
Support scaffolding that used to sit around that loop has been split out into:

- [`agent_kernel/ops/loop_runtime_support.py`](/data/agentkernel/agent_kernel/ops/loop_runtime_support.py:1) for provider construction, shared-repo materialization/publish flows, and post-episode persistence helpers
- [`agent_kernel/ops/loop_checkpointing.py`](/data/agentkernel/agent_kernel/ops/loop_checkpointing.py:1) for checkpoint payload serialization and resume reconstruction
- [`agent_kernel/ops/loop_progress.py`](/data/agentkernel/agent_kernel/ops/loop_progress.py:1) for progress-event shaping
- [`agent_kernel/extensions/planner_recovery.py`](/data/agentkernel/agent_kernel/extensions/planner_recovery.py:1) for planner-recovery artifact synthesis helpers

The intended inner control loop is:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

In the current implementation that maps to graph and retrieval context, latent/state estimation, world-model refresh, rollout-scored candidate selection, role-specialized planner/executor/critic policy routing, verifier checks, subgoal diagnosis, and episode-memory persistence.

The intended outer self-improvement loop is:

`run tasks -> collect outcomes -> compare against verifiers -> localize failure -> modify policy/planner/world-model/transition-model/prompts/tools -> retest -> retain only verified gains`

In this repo that outer loop is implemented through eval-driven subsystem selection, candidate artifact generation, candidate-vs-baseline comparison, retention gates, rollback, and unattended campaign control.

### Full ASI Loop

The current codebase now supports a cleaner full-loop description than the older bootstrap-heavy shape:

```text
scripts/run_agent.py
  -> AgentKernel.run_task(...)
     -> resume_or_initialize_run(...)
     -> execute_setup_commands(...)
     -> bootstrap_runtime_state(...)
     -> repeat execute_step(...):
        -> policy.decide(...)
        -> universe_model.should_block_command(...)
        -> sandbox.run(...)
        -> verifier.verify(...)
        -> summarize_transition(...)
        -> state.update_after_step(...)
     -> finalize_episode(...)
        -> episode save
        -> learning-candidate compilation

compiled learning candidates + eval metrics
  -> ImprovementPlanner.rank_targets(...)
  -> ImprovementPlanner.rank_variants(...)
  -> generate candidate artifact
  -> preview_candidate_retention(...)
  -> finalize_cycle(...)
  -> retained artifact becomes active runtime input for later task runs
```

That is the closed ASI loop in this repository:

`task -> verified episode -> reusable learning evidence -> evaluated candidate artifact -> retained runtime mutation -> next task`

### Runtime Phase Map

The inner runtime loop is concretely:

1. `resume/init`
   [`resume_or_initialize_run(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:75) materializes the workspace, restores checkpoints, and initializes [`AgentState`](/data/agentkernel/agent_kernel/state.py:63).
2. `setup`
   [`execute_setup_commands(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:133) performs task-specific setup before the agent starts acting.
3. `state bootstrap`
   [`bootstrap_runtime_state(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:182) loads graph memory, world-model state, universe governance state, latent controls, and the initial planner state.
4. `decision`
   [`LLMDecisionPolicy.decide(...)`](/data/agentkernel/agent_kernel/policy.py:164) compiles context, retrieves memory, scores options, and emits one action.
5. `governance`
   [`UniverseModel.should_block_command(...)`](/data/agentkernel/agent_kernel/universe_model.py:315) can reject a command before execution; [`simulate_command_governance(...)`](/data/agentkernel/agent_kernel/universe_model.py:244) records the governance view even when the command proceeds.
6. `bounded execution`
   [`Sandbox.run(...)`](/data/agentkernel/agent_kernel/sandbox.py:28) is the hard execution boundary.
7. `verification`
   [`Verifier.verify(...)`](/data/agentkernel/agent_kernel/verifier.py:13) checks deterministic and semantic contracts.
8. `transition estimation`
   [`summarize_transition(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:254) refreshes the world model, measures progress/regression, and updates latent state.
9. `state update`
   [`AgentState.update_after_step(...)`](/data/agentkernel/agent_kernel/state.py:212) records progress, repetition pressure, failure pressure, plan updates, and stopping signals.
10. `stop or repeat`
    [`execute_step(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:334) decides whether the episode has succeeded, terminated, or should continue.
11. `persist`
    [`finalize_episode(...)`](/data/agentkernel/agent_kernel/ops/loop_run_support.py:717) writes the episode through [`EpisodeMemory.save(...)`](/data/agentkernel/agent_kernel/memory.py:28) and optionally compiles learning candidates.

### Self-Improvement Phase Map

The outer improvement loop is concretely:

1. `compile evidence`
   [`compile_episode_learning_candidates(...)`](/data/agentkernel/agent_kernel/learning_compiler.py:80) converts episodes into success, failure, recovery, and gap artifacts.
2. `rank what to change`
   [`ImprovementPlanner.rank_targets(...)`](/data/agentkernel/agent_kernel/improvement.py:403) and [`rank_experiments(...)`](/data/agentkernel/agent_kernel/improvement.py:406) choose the subsystem pressure.
3. `reflect`
   [`agent_kernel/reflection.py`](/data/agentkernel/agent_kernel/reflection.py) converts observe-phase evidence into a resource-scoped causal hypothesis.
4. `select`
   [`ImprovementPlanner.rank_variants(...)`](/data/agentkernel/agent_kernel/improvement.py:699) chooses the candidate mutation family and [`agent_kernel/selection.py`](/data/agentkernel/agent_kernel/selection.py) records the planned resource edit.
5. `generate candidate artifact`
   [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py:1608) orchestrates artifact generation from the chosen subsystem and variant.
6. `measure candidate vs baseline`
   [`preview_candidate_retention(...)`](/data/agentkernel/agent_kernel/cycle_runner.py:279) runs the bounded comparison protocol.
7. `retain or reject`
   [`finalize_cycle(...)`](/data/agentkernel/agent_kernel/cycle_runner.py:443) applies the retention decision, rollback path, and cycle record.
8. `feed retained artifact back into runtime`
   The next [`AgentKernel.run_task(...)`](/data/agentkernel/agent_kernel/loop.py:162) sees the retained artifact through `KernelConfig` paths and the relevant runtime adapters.

Allowed actions remain:

- `respond`
- `code_execute`

In practice, task completion is driven by `code_execute`; `respond` is mainly used for policy errors or explicit termination.

## Main modules

- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py): runtime defaults, feature flags, and artifact paths
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py): task execution loop, step recording, and episode control state
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py): structured decision policy, skill loading, retrieval guidance use
- [`agent_kernel/extensions/tolbert.py`](/data/agentkernel/agent_kernel/extensions/tolbert.py): seed TOLBERT compiler wrapper
- [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py): subprocess TOLBERT service
- [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling): isolated modeling code for TOLBERT-family checkpoints, training, and evaluation
- [`agent_kernel/llm.py`](/data/agentkernel/agent_kernel/llm.py): Ollama and mock clients
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py): deterministic task verification and stricter-contract synthesis
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py): episode persistence and graph summaries
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py): lightweight world-model summaries
- [`agent_kernel/extensions/multi_agent.py`](/data/agentkernel/agent_kernel/extensions/multi_agent.py): planner/executor/reviewer role coordination
- [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py): built-in tasks plus replay-task loaders
- [`agent_kernel/tasking/curriculum.py`](/data/agentkernel/agent_kernel/tasking/curriculum.py): success-adjacent and failure-recovery followups
- [`datasets/curriculum_templates.json`](/data/agentkernel/datasets/curriculum_templates.json): data-backed curriculum task templates plus long-horizon routing metadata rendered by [`agent_kernel/tasking/curriculum_catalog.py`](/data/agentkernel/agent_kernel/tasking/curriculum_catalog.py)
- [`datasets/improvement_catalog.json`](/data/agentkernel/datasets/improvement_catalog.json): data-backed improvement schema, artifact-contract registry, artifact-validation profiles, retention-gate presets, and runtime-metadata catalog rendered by [`agent_kernel/extensions/improvement/improvement_catalog.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_catalog.py)
- [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json): data-backed kernel registry metadata for subsystem specs, capability adapters, task-bank lineage filters, unattended-controller feature catalogs, Tolbert policy thresholds, trust-family defaults, unattended runtime policy defaults, and frontier-family routing defaults rendered by [`agent_kernel/extensions/strategy/kernel_catalog.py`](/data/agentkernel/agent_kernel/extensions/strategy/kernel_catalog.py)
- [`agent_kernel/resource_types.py`](/data/agentkernel/agent_kernel/resource_types.py): descriptor and resolved-version data model for the first runtime resource substrate
- [`agent_kernel/resource_registry.py`](/data/agentkernel/agent_kernel/resource_registry.py): RSPL-lite registry for prompt templates and builtin retained artifacts
- [`agent_kernel/reflection.py`](/data/agentkernel/agent_kernel/reflection.py): `ReflectionRecord` builders and causal-hypothesis normalization for explicit `Reflect`
- [`agent_kernel/selection.py`](/data/agentkernel/agent_kernel/selection.py): `SelectionRecord` and `ResourceEditPlan` builders for explicit `Select`
- [`agent_kernel/extensions/extractors.py`](/data/agentkernel/agent_kernel/extensions/extractors.py): promoted skill and tool artifact extraction
- [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1): shared improvement-engine dataclasses plus generic target projection, variant orchestration, retention-gate application, and artifact lifecycle application helpers
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py): planner orchestration, subsystem-scored experiment ranking, evidence shaping, compatibility checks, and cycle records
- [`agent_kernel/extensions/improvement/improvement_plugins.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_plugins.py:1): plugin-layer adapters for subsystem registry access, strategy-memory priors, TOLBERT delta helpers, and universe bundle helpers
- [`agent_kernel/extensions/improvement/improvement_support_validation.py`](/data/agentkernel/agent_kernel/extensions/improvement/improvement_support_validation.py:1): support-side task-contract catalog used by artifact compatibility checks
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py): candidate-vs-baseline retention flow
- [`agent_kernel/extensions/improvement/prompt_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/prompt_improvement.py): prompt proposal generation
- [`agent_kernel/extensions/improvement/retrieval_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/retrieval_improvement.py): retrieval proposal generation
- [`agent_kernel/extensions/improvement/curriculum_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/curriculum_improvement.py): curriculum proposal generation
- [`agent_kernel/extensions/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/extensions/runtime_modeling_adapter.py:1): lazy adapter seam for context compilation, model artifact loading, hybrid scoring, and learned world-signal enrichment
- [`agent_kernel/ops/loop_runtime_support.py`](/data/agentkernel/agent_kernel/ops/loop_runtime_support.py:1): runtime support helpers for policy construction, shared-repo workspace flows, and post-episode persistence/publish closeout
- [`agent_kernel/ops/loop_checkpointing.py`](/data/agentkernel/agent_kernel/ops/loop_checkpointing.py:1): checkpoint serialization and resume helpers for the runtime loop
- [`agent_kernel/ops/loop_progress.py`](/data/agentkernel/agent_kernel/ops/loop_progress.py:1): shared progress-callback payload shaping
- [`agent_kernel/extensions/planner_recovery.py`](/data/agentkernel/agent_kernel/extensions/planner_recovery.py:1): extracted planner-recovery scoring and artifact-synthesis logic
- [`agent_kernel/extensions/context_budget.py`](/data/agentkernel/agent_kernel/extensions/context_budget.py): prompt-budget assembly for optional runtime context surfaces
- [`agent_kernel/extensions/delegation_policy.py`](/data/agentkernel/agent_kernel/extensions/delegation_policy.py): retained delegated-job runtime policy helpers
- [`agent_kernel/extensions/syntax_motor.py`](/data/agentkernel/agent_kernel/extensions/syntax_motor.py): Python edit-shape analysis and syntax-safe progress extraction
- [`agent_kernel/extensions/operator_policy.py`](/data/agentkernel/agent_kernel/extensions/operator_policy.py): retained unattended operator policy helpers
- [`agent_kernel/extensions/tolbert_assets.py`](/data/agentkernel/agent_kernel/extensions/tolbert_assets.py): retained TOLBERT asset materialization and runtime-path resolution
- [`agent_kernel/extensions/trust.py`](/data/agentkernel/agent_kernel/extensions/trust.py): unattended trust policy and ledger helpers
- [`agent_kernel/extensions/strategy/kernel_catalog.py`](/data/agentkernel/agent_kernel/extensions/strategy/kernel_catalog.py): kernel metadata registry and retained extension catalog
- [`agent_kernel/extensions/strategy/subsystems.py`](/data/agentkernel/agent_kernel/extensions/strategy/subsystems.py): subsystem registry, artifact paths, and generator dispatch
- [`agent_kernel/extensions/strategy/semantic_hub.py`](/data/agentkernel/agent_kernel/extensions/strategy/semantic_hub.py): semantic note, skill, attempt, and task indexing
- [`agent_kernel/ops/preflight.py`](/data/agentkernel/agent_kernel/ops/preflight.py): unattended preflight checks and task report shaping
- [`agent_kernel/ops/job_queue.py`](/data/agentkernel/agent_kernel/ops/job_queue.py): delegated worker queueing and runtime coordination
- [`agent_kernel/ops/roadmap_parallel.py`](/data/agentkernel/agent_kernel/ops/roadmap_parallel.py): ASI parallel-manifest generation helpers
- [`agent_kernel/ops/runtime_supervision.py`](/data/agentkernel/agent_kernel/ops/runtime_supervision.py): atomic writes, append-only histories, and managed export governance triggers
- [`agent_kernel/ops/workspace_recovery.py`](/data/agentkernel/agent_kernel/ops/workspace_recovery.py): snapshot and rollback support
- [`agent_kernel/ops/export_governance.py`](/data/agentkernel/agent_kernel/ops/export_governance.py): export pruning and retention hygiene
- [`evals/harness.py`](/data/agentkernel/evals/harness.py): multi-task eval and comparison modes
- [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md): compact evidence-backed map from the current unattended run to the next kernel surfaces to patch

In core-vs-auxiliary terms:

- executable agent minimum centers on `loop.py`, `policy.py`, `state.py`, `schemas.py`, `actions.py`, `sandbox.py`, `verifier.py`, `llm.py`, and `config.py`
- ASI-shaped runtime minimum adds `memory.py`, `world_model.py`, and `universe_model.py`
- minimum self-improving ASI centers on `learning_compiler.py`, `improvement_engine.py`, `improvement.py`, and `cycle_runner.py`
- `extensions/strategy/`, `strategy_memory/`, learned execution packages, `tasking/`, `ops/`, and `extensions/improvement/` are support layers around those minimums, now reached through `extensions/improvement/improvement_plugins.py`, `extensions/improvement/improvement_support_validation.py`, and `extensions/runtime_modeling_adapter.py` seams instead of raw direct imports in the main runtime and retention entrypoints

## Current Package Split

The filesystem now mirrors the core/support boundary more explicitly:

- `agent_kernel/`: minimum runtime and self-improvement core only
- [`agent_kernel/tasking/`](/data/agentkernel/agent_kernel/tasking): task-bank, curriculum, benchmark-synthesis, and task-budget support
- [`agent_kernel/ops/`](/data/agentkernel/agent_kernel/ops): preflight, runtime supervision, export governance, and workspace-recovery support
- [`agent_kernel/extensions/`](/data/agentkernel/agent_kernel/extensions): optional runtime/modeling adapters plus policy/runtime support helpers
- [`agent_kernel/extensions/improvement/`](/data/agentkernel/agent_kernel/extensions/improvement): subsystem-specific improvement surfaces
- [`agent_kernel/extensions/strategy/`](/data/agentkernel/agent_kernel/extensions/strategy): subsystem registry and semantic support

At the time of this update, the root contains `22` top-level Python files, and that root now matches the current ASI-core set exactly. The increase reflects the self-improvement core being split into explicit retention, artifact-lifecycle, record-normalization, and evidence/control-evidence modules instead of hiding those protocols inside `improvement.py`.

That means the shim-removal phase and the root cleanup phase are both complete.

Within that self-improvement minimum, the current boundary is:

- `learning_compiler.py`: compile episode outcomes into reusable learning candidates
- `improvement_engine.py`: generic ranking surfaces, cycle-record persistence, normalized learning-evidence attachment, measurable-runtime-signal checks, retention-evaluation scaffolding, and artifact lifecycle application
- `improvement.py`: subsystem-specific experiment scoring, evidence shaping, compatibility policy, and retention adapters registered through the plugin layer
- `extensions/improvement/improvement_plugins.py`: support-side registry adapters plus subsystem lifecycle hooks such as retrieval-bundle materialization and Tolbert liftoff report application
- `cycle_runner.py`: measured baseline-vs-candidate orchestration and cycle closeout, consuming generic lifecycle effects instead of owning retain-only side-effect branches itself
- `scripts/run_improvement_cycle.py`: campaign selection, generated-artifact compatibility preflight, and candidate-cycle bookkeeping for the bounded autonomous loop

## Runtime features

The code now has feature toggles for:

- TOLBERT context
- skill memory
- graph memory
- world model summaries
- planner-generated subgoals
- role specialization
- prompt proposal application
- curriculum proposal application
- retrieval proposal application

Those toggles are all enabled by default in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py), and most are surfaced in [`scripts/run_agent.py`](/data/agentkernel/scripts/run_agent.py) or [`scripts/run_eval.py`](/data/agentkernel/scripts/run_eval.py).

## Repository shape

```text
agent_kernel/
  extensions/
    improvement/
    strategy/
  ops/
  tasking/
docs/
evals/
prompts/
scripts/
tests/
trajectories/
workspace/
var/
```

## TOLBERT integration

The runtime compiles a typed `ContextPacket` before each policy call. That packet includes:

- a hierarchical path prediction
- retrieval bundles from branch-scoped and global search
- verifier contract details
- control metadata such as path confidence and retrieval guidance

That retrieval path is no longer strictly stateless across steps. The compiler
can now carry forward a previously selected or trusted retrieval command into
the next compile when the agent did not actually use it yet, so retrieval
selection has a direct path to later action influence instead of collapsing back
to a fresh threshold-only query on every step.

That same signal now persists past the episode boundary too. Learning artifacts
compiled from verified runs retain whether a reusable procedure was
retrieval-backed or trusted-retrieval-backed, and TolBERT reuses that stronger
provenance on the next similar task instead of flattening it back into ordinary
same-family memory.

The promoted artifact path now preserves that provenance as well. Extracted
skill and tool procedures keep retrieval-backed metadata and prefer
trusted-retrieval-backed variants during dedupe, so retained artifacts do not
erase the difference between ordinary replay and verifier-proven retrieval use.
The compare and frontier-reporting path now carries the same retrieval-reuse
summary forward, so promotion ranking and retained-baseline review can reward
verifier-proven retrieval reuse explicitly instead of only through extracted
procedure quality. The supervisor bootstrap-review path now carries that same
summary onto first-retain queue entries and gives it a small review-priority
bias, so verifier-proven retrieval reuse can surface earlier for operator
attention when trust and bootstrap gates are otherwise equal. That bootstrap
priority now persists briefly across supervisor rounds too, so a subsystem that
keeps surfacing verifier-proven retrieval-backed first-retain evidence does not
lose review priority after one quiet plan snapshot. Separately, when a policy
candidate is blocked specifically on missing generated evidence, the supervisor
now launches a targeted curriculum-backed discovery rerun for that same
subsystem and variant instead of only recycling the candidate through generic
bootstrap review.

The implementation is intentionally strict:

- it launches a real `tolbert-brain` subprocess service
- it treats that service as owned runtime state and closes it when the enclosing kernel is torn down
- it requires a checkpoint, ontology nodes, source spans, cache shards, and usually a label map
- it does not silently fall back when ontology, logits, or retrieval inputs are inconsistent

The current runtime contract is still seed-stage:

- the live TOLBERT service compiles context and can influence bounded direct decisions
- the retained TOLBERT family is defined more broadly as encoder, latent dynamics, decoder, and world-model surfaces
- the seed LLM remains the authoritative free-form decoder
- the runtime now reaches those learned/modeling surfaces through [`agent_kernel/extensions/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/extensions/runtime_modeling_adapter.py:1), so future model-native takeover work should stay under [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling) rather than re-expanding direct imports inside the core loop modules

The staged takeover plan is documented in [`tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md).

## Supporting subsystems

The repo now includes subsystems that earlier docs called non-goals:

- role-specialized decision routing
- lightweight planner/executor/reviewer coordination
- replay-task generation from skill, operator, tool, benchmark, verifier, and episode artifacts
- prompt/curriculum/retrieval/verifier/tooling improvement artifacts

What it still does not try to be:

- a general browser agent
- a remote execution platform
- a distributed multi-agent system
- an internet-enabled research stack
