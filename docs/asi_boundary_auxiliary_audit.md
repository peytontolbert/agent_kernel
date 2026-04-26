# Lane 4: Boundary And Auxiliary Audit

## Scope Reviewed

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:16)
- [`agent_kernel/subsystems.py`](/data/agentkernel/agent_kernel/subsystems.py:39)
- [`agent_kernel/strategy_memory/__init__.py`](/data/agentkernel/agent_kernel/strategy_memory/__init__.py:1)
- [`agent_kernel/strategy_memory/store.py`](/data/agentkernel/agent_kernel/strategy_memory/store.py:120)
- [`agent_kernel/strategy_memory/sampler.py`](/data/agentkernel/agent_kernel/strategy_memory/sampler.py:128)
- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py:1)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:19)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:17)
- [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py:320)
- [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py:52)
- [`agent_kernel/job_queue.py`](/data/agentkernel/agent_kernel/job_queue.py:16)
- [`docs/asi_core.md`](/data/agentkernel/docs/asi_core.md:1)
- [`docs/architecture.md`](/data/agentkernel/docs/architecture.md:22)
- [`docs/index.md`](/data/agentkernel/docs/index.md:1)

## Minimum Contract

The boundary contract from [`asi_core.md`](/data/agentkernel/docs/asi_core.md:21) is:

- executable-agent core is the closed-loop task executor
- ASI-runtime core is memory, world model, and governance layered onto that executor
- self-improvement core is compile, rank, compare, retain/reject, and apply
- TOLBERT/modeling, subsystem registries, strategy memory, task supply, curriculum, unattended control, and job orchestration are support layers unless they change one of those minimums directly

## Verdict

The repo mostly classifies the boundary correctly in docs, and the highest-ROI code-level boundary leaks identified in this lane are now patched.

The self-improvement layer now has a real engine/plugin/support split:

- generic engine dataclasses plus target projection, variant orchestration, retention-gate application, and artifact lifecycle helpers live in [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1)
- subsystem registry, strategy-memory priors, TOLBERT delta helpers, and universe bundle helpers are reached through [`agent_kernel/improvement_plugins.py`](/data/agentkernel/agent_kernel/improvement_plugins.py:1)
- task-supply-backed compatibility lookup is now wrapped by [`agent_kernel/improvement_support_validation.py`](/data/agentkernel/agent_kernel/improvement_support_validation.py:1)

The runtime minimum now also has an adapter seam for optional learned/modeling packages:

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1) and [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1) depend on [`agent_kernel/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/runtime_modeling_adapter.py:1) for context compilation, retained model artifact loading, hybrid scoring, latent-state enrichment, and bounded decoder helpers
- `loop.py` itself is now narrower on the runtime side too: provider/client construction, shared-repo materialization/publish flows, checkpoint serialization, progress-event shaping, and planner-recovery artifact synthesis have been moved into [`agent_kernel/ops/loop_runtime_support.py`](/data/agentkernel/agent_kernel/ops/loop_runtime_support.py:1), [`agent_kernel/ops/loop_checkpointing.py`](/data/agentkernel/agent_kernel/ops/loop_checkpointing.py:1), [`agent_kernel/ops/loop_progress.py`](/data/agentkernel/agent_kernel/ops/loop_progress.py:1), and [`agent_kernel/extensions/planner_recovery.py`](/data/agentkernel/agent_kernel/extensions/planner_recovery.py:1)

Residual boundary issue:

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1) still owns subsystem-scored experiment ranking and evidence shaping, so the engine split is real but not yet total. The generic retention/application seam is extracted; the experiment-ranking seam is only partially extracted.

## What Is Already Present

- The canonical docs now separate core from support explicitly. [`asi_core.md`](/data/agentkernel/docs/asi_core.md:81) classifies `subsystems.py`, `strategy_memory/`, learned execution packages, task supply, and controller machinery as auxiliary/support, and [`architecture.md`](/data/agentkernel/docs/architecture.md:93) repeats that split.
- Strategy memory is support that pressures planning, not a hard prerequisite for the meta-loop. When runtime config is absent, the planner returns an empty strategy-memory summary instead of failing in [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1832); only when config is present does it load nodes and summarize priors in [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1847).
- Task supply and controller machinery are still downstream of the runtime core rather than upstream of it. [`agent_kernel/job_queue.py`](/data/agentkernel/agent_kernel/job_queue.py:19) imports `AgentKernel`, and delegated execution calls `kernel.run_task(...)` in [`agent_kernel/job_queue.py`](/data/agentkernel/agent_kernel/job_queue.py:2110), so the queue wraps the kernel instead of defining it.
- Curriculum remains a support layer around episode generation rather than part of the executor minimum. [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py:58) synthesizes followup tasks from episode records, and [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py:320) annotates task contracts and manifests rather than shaping the executor loop directly.

## What Is Only Partial Or Optional

- TOLBERT, skills, graph memory, world model, planner, role specialization, and proposal families are behaviorally optional behind config in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:124). The loop only enables the TOLBERT context provider when `use_tolbert_context` is true in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:74), and it gates graph memory, world model, planner, and role specialization later in the live path in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:310), [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:330), [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:366), and [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:391).
- That optionality is still only partially packaging-clean. [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:45) and [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:17) now reach learned/modeling support through [`agent_kernel/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/runtime_modeling_adapter.py:1), but those adapter seams still live in the root package and the surrounding optional surfaces have not yet been moved into clearer support subdirectories.
- Strategy memory is no longer mere reporting. The planner uses prior nodes to influence selection in [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1848), and the sampler actively derives preferred and rejected control surfaces in [`agent_kernel/strategy_memory/sampler.py`](/data/agentkernel/agent_kernel/strategy_memory/sampler.py:42). That makes it an influential support layer, not meta-loop essence.

## What Was Missing And Is Now Present

- A real engine/plugin seam inside the self-improvement layer now exists through [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1) and [`agent_kernel/improvement_plugins.py`](/data/agentkernel/agent_kernel/improvement_plugins.py:1).
- A clean runtime adapter seam for learned execution/modeling packages now exists in [`agent_kernel/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/runtime_modeling_adapter.py:1).
- A stricter separation between core retention logic and support-side compatibility/task-supply logic now exists through [`agent_kernel/improvement_support_validation.py`](/data/agentkernel/agent_kernel/improvement_support_validation.py:1), which provides `TaskContractCatalog` instead of constructing `TaskBank` in the core retention path.

## What Still Remains

- The experiment-ranking and evidence-shaping portion of [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1) is still larger than the generic engine should ideally be.
- The plugin seam is adapter-based rather than a full plugin package with independent subsystem-owned ranking/evidence modules.
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1) is now mostly legitimately core, but some support-side orchestration still remains there around episode closeout and runtime sequencing even after the new helper split.

## Current Ranked Residual Violations

1. `improvement.py` still mixes subsystem-scored experiment ranking and evidence shaping with the narrower generic engine split.
   Evidence: the extracted engine now owns generic target projection, variant orchestration, retention-gate application, and artifact lifecycle application in [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1), but subsystem-specific ranking/evidence logic still lives in [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1).

2. The plugin seam is adapter-oriented rather than fully package-split by subsystem family.
   Evidence: [`agent_kernel/improvement_plugins.py`](/data/agentkernel/agent_kernel/improvement_plugins.py:1) centralizes support bridges, but subsystem-specific ranking/evidence logic has not been pushed into independently owned plugin modules.

3. Docs needed to be updated after the seam landed.
   Evidence: older versions of this page and nearby architecture docs still described the engine/plugin/adapter/support-validator seams as missing even after they were implemented.

## Refactor Map

### Keep In Minimums

- executable floor: [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:51), [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:61), [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28), [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:10), [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:63), [`agent_kernel/schemas.py`](/data/agentkernel/agent_kernel/schemas.py:13)
- ASI-runtime core: [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:14), [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:21), [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:75)
- self-improvement core: learning compilation, ranking, compare, retain/reject, and artifact application from [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:80), [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:247), and [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:17)

### Reclassify As Support Around The Minimums

- subsystem registry and proposal-family construction: [`agent_kernel/subsystems.py`](/data/agentkernel/agent_kernel/subsystems.py:39)
- strategy-memory storage, lessons, priors, and snapshots: [`agent_kernel/strategy_memory/store.py`](/data/agentkernel/agent_kernel/strategy_memory/store.py:120), [`agent_kernel/strategy_memory/sampler.py`](/data/agentkernel/agent_kernel/strategy_memory/sampler.py:128)
- learned execution/modeling stack: [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py:1), [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling)
- task supply and curriculum: [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py:320), [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py:52)
- unattended operations and job orchestration: [`agent_kernel/job_queue.py`](/data/agentkernel/agent_kernel/job_queue.py:16), [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py:1)

## Highest-ROI Next Patches

1. Continue shrinking [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1) by moving more experiment-ranking and evidence-collation logic into [`agent_kernel/improvement_engine.py`](/data/agentkernel/agent_kernel/improvement_engine.py:1) or clearer subsystem-owned adapters.
2. Decide whether the current adapter-style [`agent_kernel/improvement_plugins.py`](/data/agentkernel/agent_kernel/improvement_plugins.py:1) is sufficient, or whether subsystem-owned plugin modules should replace the centralized bridge.
3. Keep future learned/runtime integration behind [`agent_kernel/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/runtime_modeling_adapter.py:1) so optional modeling support does not re-expand into direct loop/policy imports.
