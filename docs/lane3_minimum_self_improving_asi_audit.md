# Lane 3 Audit: Minimum Self-Improving ASI

## Scope Reviewed

- learning compilation in [agent_kernel/learning_compiler.py](/data/agentkernel/agent_kernel/learning_compiler.py:56)
- planner ranking, retention evidence, lifecycle application, and rollback in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:398)
- measured finalize flow in [agent_kernel/cycle_runner.py](/data/agentkernel/agent_kernel/cycle_runner.py:1507)
- live runtime use of compiled learning artifacts in [agent_kernel/loop.py](/data/agentkernel/agent_kernel/loop.py:747) and [agent_kernel/tolbert.py](/data/agentkernel/agent_kernel/tolbert.py:1048)
- orchestration callers in [scripts/run_improvement_cycle.py](/data/agentkernel/scripts/run_improvement_cycle.py:1838) and [scripts/run_parallel_supervised_cycles.py](/data/agentkernel/scripts/run_parallel_supervised_cycles.py:206)
- proof tests in [tests/test_memory.py](/data/agentkernel/tests/test_memory.py:1065), [tests/test_improvement.py](/data/agentkernel/tests/test_improvement.py:8962), and [tests/test_finalize_cycle.py](/data/agentkernel/tests/test_finalize_cycle.py:1837)

## Minimum Contract

- episode outcomes compile into reusable evidence
- targets and variants can be ranked
- candidate and baseline behavior can be compared
- retain and reject decisions can be made under explicit gates
- those decisions can be applied back onto live artifacts

## Verdict

The repo has a real self-improvement loop, not just disconnected proposal helpers. The minimum is `present`, `wired`, and `tested` as a working retain/reject meta-loop. The boundary is now materially cleaner than the earlier audit state: generic engine helpers live in [agent_kernel/improvement_engine.py](/data/agentkernel/agent_kernel/improvement_engine.py:1), subsystem adapters and lifecycle hooks live in [agent_kernel/improvement_plugins.py](/data/agentkernel/agent_kernel/improvement_plugins.py:1), compiled learning evidence now feeds `retention_evidence`, and retained retrieval/Tolbert side effects now flow through generic lifecycle hooks rather than finalize-only branches.

## What Is Complete

- Episode outcomes do compile into reusable evidence. The live loop persists post-episode learning artifacts in [agent_kernel/loop.py](/data/agentkernel/agent_kernel/loop.py:747), and the compiler emits success, recovery, failure, negative-command, and benchmark-gap candidates in [agent_kernel/learning_compiler.py](/data/agentkernel/agent_kernel/learning_compiler.py:80). Tests cover retrieval-backed and syntax-enriched compilation in [tests/test_memory.py](/data/agentkernel/tests/test_memory.py:1065) and [tests/test_memory.py](/data/agentkernel/tests/test_memory.py:1121).

- That evidence is consumed by both the runtime and the planner. Runtime retrieval pulls matched learning candidates into recommended commands and avoidance notes in [agent_kernel/tolbert.py](/data/agentkernel/agent_kernel/tolbert.py:1048). The planner folds compiled candidates into subsystem summaries and experiment bonuses in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:3051) and [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:3127).

- Target and variant ranking are generic entrypoints and are used by the orchestration layer. The planner exposes `rank_experiments`, `rank_variants`, `choose_variant`, and campaign selection in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:408) and [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:2017). Those entrypoints drive autonomous and parallel cycle selection in [scripts/run_improvement_cycle.py](/data/agentkernel/scripts/run_improvement_cycle.py:1838), [scripts/run_improvement_cycle.py](/data/agentkernel/scripts/run_improvement_cycle.py:408), and [scripts/run_parallel_supervised_cycles.py](/data/agentkernel/scripts/run_parallel_supervised_cycles.py:206).

- Candidate-vs-baseline comparison is the central decision protocol. Preview evaluation, confirmation runs, holdout evaluation, prior-retained comparison, confidence aggregation, decision application, and cycle-history recording all sit inside [agent_kernel/cycle_runner.py](/data/agentkernel/agent_kernel/cycle_runner.py:2797). Tests cover the main finalize path in [tests/test_finalize_cycle.py](/data/agentkernel/tests/test_finalize_cycle.py:1837), required confirmation in [tests/test_finalize_cycle.py](/data/agentkernel/tests/test_finalize_cycle.py:2017), prior-retained rejection in [tests/test_finalize_cycle.py](/data/agentkernel/tests/test_finalize_cycle.py:5591), and bounded holdout behavior in [tests/test_finalize_cycle.py](/data/agentkernel/tests/test_finalize_cycle.py:9373).

- Retain/reject decisions are applied back onto live artifacts with snapshots and rollback metadata. `apply_artifact_retention_decision` stamps lifecycle state, compatibility, retention metadata, snapshots, and live-artifact promotion or restore paths in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:7276). Tests cover staged rejection without live mutation, restore-on-reject, universe synchronization, Tolbert rejection compaction, and Tolbert canonical promotion in [tests/test_improvement.py](/data/agentkernel/tests/test_improvement.py:8962), [tests/test_improvement.py](/data/agentkernel/tests/test_improvement.py:9104), and [tests/test_improvement.py](/data/agentkernel/tests/test_improvement.py:9244).

## What Is Partial

- The engine boundary is cleaner but not fully extracted. [agent_kernel/improvement_engine.py](/data/agentkernel/agent_kernel/improvement_engine.py:1) now owns generic ranking helpers, cycle-record persistence, retention-evaluation scaffolding, learning-evidence attachment, and lifecycle application, but subsystem-specific ranking, evidence shaping, and evaluator rules still largely live in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:1).

- Compiled learning evidence is now a first-class retention input, but only some subsystem evaluators currently act on it directly. `retention_evidence` now injects normalized learning-store summaries via the generic adapter path in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:5863), yet most retain/reject evaluators still primarily gate on eval deltas and payload-local evidence.

- External improvement surfaces are more pluggable than before, but not fully open-ended. The plugin layer now carries evaluator registration and post-apply hooks in [agent_kernel/improvement_plugins.py](/data/agentkernel/agent_kernel/improvement_plugins.py:40), but subsystem evidence builders are still mostly maintained inside [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:5863).

- Rollback is still artifact-level, not whole-world rollback. `rollback_artifact` restores the saved file snapshot, but the receipt explicitly marks `world_state_revalidation_required` in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:3866) and [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:3911). Tests confirm that limit in [tests/test_improvement.py](/data/agentkernel/tests/test_improvement.py:9438).

## What Is Missing

- A fuller evidence-adapter split. The learning adapter is generic, but most subsystem-specific evidence shaping still happens in one large `retention_evidence` function in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:5863).

- Wider retention use of direct learning-store evidence. The normalized `learning_evidence` block is now present, but only a subset of subsystem evaluators currently turns that support into explicit retain/reject policy.

- A broader plugin registry surface for subsystem evidence builders. Evaluators and lifecycle hooks are pluggable now; evidence shaping itself is not yet consistently delegated out of the core file.

## Top 3 Fact-Checked Remaining Gaps

- `Evidence shaping is still centralized`: the engine and plugin seams now exist, but subsystem evidence shaping still concentrates in [agent_kernel/improvement.py](/data/agentkernel/agent_kernel/improvement.py:5863).

- `Learning evidence is normalized before it is fully operationalized`: compiled candidates now feed a generic `learning_evidence` block, but only some subsystems currently turn that block into explicit retain/reject fallback policy.

- `Cycle orchestration is still heavier than the irreducible minimum`: finalize orchestration in [agent_kernel/cycle_runner.py](/data/agentkernel/agent_kernel/cycle_runner.py:2797) still owns comparison, confirmation, holdout, and recording policy even though post-retain side effects have moved behind lifecycle hooks.

## Top 3 Next Patches

- Push subsystem-specific evidence builders behind the plugin registry so the engine-facing retention scaffold stops depending on one monolithic `retention_evidence` function.

- Widen explicit use of `learning_evidence` inside subsystem evaluators, especially where flat benchmark deltas should still admit or reject candidates based on direct compiled support or transition-failure pressure.

- Continue shrinking finalize orchestration by extracting more generic confirmation/holdout/reporting helpers where that reduces cross-file coupling without obscuring the measured protocol.
