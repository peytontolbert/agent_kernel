# Supervised Probe History

## Purpose

This file holds the longer narrative history for supervised-loop probing and repair work.
It is intentionally separate from:

- [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md)
- [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)

Use this file for:

- durable historical notes
- probe summaries
- major closed gaps
- rationale that should survive queue churn

## Major Closed Gaps

- bounded observation now runs in a killable child process and terminates cleanly on budget
- timeout-budget provenance is now captured in observation payloads and cycle metrics
- context-provider hard-stop remediation closed the main first-task `context_compile` timing path for bounded single-task scoped runs
- planner scoring now penalizes repeated bounded observation timeouts and caps cold-start low-confidence pressure
- shared scoped-run evidence is now consolidated into [`supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
- finalize selection can now be derived from the frontier report through [`scripts/finalize_latest_candidate_from_cycles.py`](/data/agentkernel/scripts/finalize_latest_candidate_from_cycles.py)
- multi-candidate consolidation is now surfaced through [`supervised_frontier_promotion_plan.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_plan.json), which ranks frontier candidates and emits compare/finalize commands for the coordinator
- promotion execution now exists through [`supervised_frontier_promotion_pass.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_pass.json), which records compare and finalize outcomes across multiple ranked candidates in one coordinator pass
- promotion execution now routes `no prior retained baseline exists` into a `bootstrap_first_retain` path, so first-retain candidates are separated from true compare failures in the shared pass report
- supervisor policy now defines first-retain execution explicitly: bootstrap finalize remains review-only unless the supervisor is in `promote` mode and unattended trust status is `trusted`
- protected first-retain finalize now has an additional machine gate: bootstrap finalize is allowlisted per subsystem, and protected lanes require the higher clean-success threshold from [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json) before the supervisor includes them
- discovery now cools down on subsystems with bootstrap candidates pending review, so the supervisor stops wasting additional scoped loops on lanes that cannot yet advance through retention
- paused bootstrap lanes are now classified into remediation queues, separating baseline bootstrap work from trust-streak accumulation and protected-review-only lanes
- those remediation queues now emit queue-specific machine reports and execution records, so blocked lanes become explicit shared work items instead of only appearing as pause metadata inside supervisor history
- live `vllm` scoped runs exposed that scoped preview files were inheriting base `cycles.jsonl`; scoped cycle seeding is now disabled by default and frontier ingest prefers records whose `scope_id` matches the scoped filename, restoring scoped-lane isolation

## Current Design Boundary

Parallel supervised discovery is supported.
Parallel supervised consolidation is only partially supported.

What exists:

- scoped `--generate-only` runs
- lane manifest and coordination contracts
- machine-readable frontier report over scoped cycles
- frontier-to-finalize single-candidate bridge
- frontier-to-promotion-plan ranked multi-candidate queue

What still needs work:

- better candidate dedup semantics beyond path/hash identity
- stronger shared evidence ingestion for planner and retention decisions

## Current Shared Evidence

- frontier report: [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
- work queue: [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
- lane manifest: [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)

## Recommended Next History Entries

Add only durable high-signal updates here:

- new coordination or consolidation capability
- major planner-selection shift validated by live runs
- retention/promotion path upgrades
- systemic failure mode discovered across many scopes
