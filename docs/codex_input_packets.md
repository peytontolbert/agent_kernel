# Codex Input Packets

## Purpose

This document defines the minimum machine-readable input bundles that should be prepared for Codex
when it acts as the outer controller for `agent-kernel` improvement campaigns.

The goal is not to give Codex "more context."
The goal is to give Codex the highest-leverage context:

- fresh
- comparative
- causal
- versioned
- costed
- frontier-aware

Use this with:

- [`docs/codex_asi_campaign_lanes.md`](/data/agentkernel/docs/codex_asi_campaign_lanes.md)
- [`docs/kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md)
- [`docs/resource_protocol.md`](/data/agentkernel/docs/resource_protocol.md)
- [`docs/self_evolution_protocol.md`](/data/agentkernel/docs/self_evolution_protocol.md)

## Core Rule

Every official Codex batch should consume two classes of inputs:

1. `closure inputs`
   - the current proof anchor
   - the most recent non-upgrading or negative runs
   - the causal records and retention records that explain why
2. `frontier inputs`
   - held-out unfamiliar-domain tasks
   - strong baselines
   - long-horizon transfer slices
   - conservative comparison statistics

If the batch only consumes closure inputs, Codex can improve the current proof line but cannot
reliably steer toward `A7` or `A8`.

## Input Priority

Codex should prefer inputs in this order:

1. official proof anchors
2. anchor deltas versus non-upgrading runs
3. `ReflectionRecord`
4. `SelectionRecord`
5. retained candidate-vs-baseline reports
6. trust ledgers and supervision summaries
7. strategy memory snapshots and lessons
8. frontier packets
9. targeted tests
10. raw event logs and stdout

Raw logs are last on purpose.
Codex is strongest when the causal compression already exists and raw logs are only used for final
root-cause drilldown.

## `CodexBatchPacket`

This is the default packet for `Codex-Steward`.

Suggested fields:

- `packet_id`
- `generated_at`
- `batch_goal`
- `batch_scope`
- `official_anchor`
  - `status_path`
  - `report_path`
  - `repeated_status_path`
  - `anchor_date`
- `non_upgrading_runs`
  - list of the last `2-3` non-upgrading or negative runs
  - each with `status_path`, `report_path`, and `why_not_anchor`
- `top_live_bottlenecks`
  - copied from [`kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md)
- `top_substantive_gaps`
  - copied from [`kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md)
- `closure_scoreboard`
  - retained gains
  - counted trusted breadth
  - decision ownership
  - carryover
- `frontier_scoreboard`
  - held-out unfamiliar-domain pass rate
  - strong-baseline win rate
  - long-horizon transfer evidence
  - conservative bounds
- `reflection_records`
  - the top relevant reflection records by subsystem
- `selection_records`
  - the top relevant selection records by subsystem
- `top_retained_reports`
- `top_rejected_reports`
- `trust_summary`
- `external_task_manifest_paths`
- `external_task_manifest_status`
  - whether the batch is carrying a non-empty external task source into the runtime
  - which required benchmark families that source can exercise
- `strategy_memory_summary`
- `frontier_packet_path`
- `baseline_packet_path`
- `allowed_lanes`
- `merge_order`
- `compute_budget`
- `rerun_budget`

Purpose:

- let `Codex-Steward` choose the next batch goal without reconstructing context from chat
- keep one official outer-controller packet for every batch

## `CodexLanePacket`

This is the default packet for each lane worker.

Suggested fields:

- `packet_id`
- `lane_id`
- `generated_at`
- `question`
- `owned_paths`
- `explicitly_not_owned`
- `relevant_tests`
- `relevant_reports`
- `relevant_reflection_records`
- `relevant_selection_records`
- `relevant_strategy_memory_nodes`
- `failure_motifs`
- `expected_signals`
- `done_when`
- `verification_plan`
- `restart_requirement`

Purpose:

- reduce lane-local context search
- keep workers inside bounded ownership
- increase patch quality by surfacing the most relevant causal context first

## `CodexFrontierPacket`

This is the packet that prevents Codex from overfitting to the current closure surfaces.

Suggested fields:

- `packet_id`
- `generated_at`
- `frontier_version`
- `held_out_task_manifest`
  - tasks not used to justify current closure
  - tasks labeled by novelty, environment, family, and difficulty
- `max_packet_age`
- `slice_freeze_window`
- `unfamiliar_domain_slices`
  - repo novelty
  - validator novelty
  - topology novelty
  - workflow novelty
  - environment novelty
- `withheld_rotation_policy`
- `train_exclusion_manifest`
- `novelty_audit`
- `long_horizon_transfer_slices`
  - task subsets where long-horizon state quality matters
- `baseline_bundle`
  - `retained_kernel_baseline`
  - `current_codex_guided_baseline`
  - optional `expert_or_authored_baseline`
- `comparison_commands`
  - retained baseline compare commands
  - liftoff compare commands
  - frontier eval commands
- `family_conservative_bounds`
  - Wilson or other conservative bounds by family
- `paired_task_trace_report`
- `paired_trajectory_report`
- `transfer_alignment_evidence`
- `long_horizon_evidence`
- `frontier_acceptance_policy`
  - minimum non-regression rate
  - minimum baseline win rate
  - maximum allowed regressions

Purpose:

- keep every batch honest about `A7` and `A8`-relevant pressure
- prevent closure-only optimization

## `CodexBaselinePacket`

This is the packet that tells Codex what the candidate must beat.

Suggested fields:

- `packet_id`
- `generated_at`
- `baseline_type`
  - `retained_kernel`
  - `codex_guided_kernel`
  - `expert_or_authored`
- `artifact_paths`
- `compare_commands`
- `baseline_metrics`
- `baseline_task_outcomes`
- `baseline_trajectory_signature_summary`
- `baseline_trust_summary`
- `baseline_frontier_summary`
- `matched_budget_policy`
- `baseline_freeze_window`
- `tool_permission_profile`

Purpose:

- make "beat the baseline" concrete
- stop Codex from optimizing against an implicit moving target

## Input Hygiene Constraints

The packet layer should prevent benchmark gaming, not enable it.

Every official batch should follow these constraints:

1. freeze frontier slices before patching starts
2. freeze the active baseline bundle for the duration of the batch
3. compare candidate and baseline under matched budget, tool, and permission profiles
4. reject stale packets that exceed the declared `max_packet_age`
5. separate exploration sets from official scoring sets
6. record possible leakage or benchmark contamination in the packet itself
7. for `counted trust breadth` or `open_world_transfer` batches, reject packets whose
   `external_task_manifest_paths` are empty unless the batch explicitly declares that no counted
   external-breadth claim will be made

If these constraints are missing, Codex can still optimize useful local behavior.
But it can also optimize toward a moving or contaminated target.

## Existing Repo Inputs Codex Should Exploit

### 1. `ReflectionRecord`

Current code:

- [`agent_kernel/reflection.py`](/data/agentkernel/agent_kernel/reflection.py)

High-value fields:

- `resource_id`
- `summary`
- `hypothesis`
- `failure_modes`
- `evidence_labels`
- `expected_signals`
- `active_version_id`
- `metrics_digest`

Why this matters:

- it compresses why the batch thinks a subsystem is failing
- it points at a concrete active resource version

### 2. `SelectionRecord`

Current code:

- [`agent_kernel/selection.py`](/data/agentkernel/agent_kernel/selection.py)

High-value fields:

- selected variant id and description
- expected gain
- estimated cost
- variant score
- strategy metadata
- `ResourceEditPlan`
- generation kwargs

Why this matters:

- it exposes explicit gain-versus-cost information
- it tells Codex what mutation was actually chosen, not just what later existed

### 3. Resource Registry

Current code:

- [`agent_kernel/resource_registry.py`](/data/agentkernel/agent_kernel/resource_registry.py)
- [`docs/resource_protocol.md`](/data/agentkernel/docs/resource_protocol.md)

Why this matters:

- versioned resource ids are much better mutation targets than free-form file names

Current limitation:

- plans, memory, and role policies are not yet first-class resources

### 4. Retained Baseline Comparisons

Current code:

- [`scripts/compare_retained_baseline.py`](/data/agentkernel/scripts/compare_retained_baseline.py)
- [`agent_kernel/evals/retention_stats.py`](/data/agentkernel/agent_kernel/evals/retention_stats.py)

Why this matters:

- this is how Codex gets strong-baseline inputs without relying on narrative judgment
- conservative family bounds and paired task traces are especially valuable for frontier packets

### 5. Liftoff And Frontier Evaluation

Current code:

- [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)
- [`agent_kernel/modeling/evaluation/liftoff.py`](/data/agentkernel/agent_kernel/modeling/evaluation/liftoff.py)

Why this matters:

- the repo already has a comparison surface for family takeover, long-horizon evidence, transfer
  alignment, and autonomy-bridge evidence

## Packet Generation Policy

Every official batch should generate:

1. one `CodexBatchPacket`
2. one `CodexFrontierPacket`
3. one `CodexBaselinePacket`
4. one `CodexLanePacket` per opened lane

If any of those are missing:

- the batch may still produce useful code
- but Codex is operating below the repo's available outer-loop intelligence ceiling

## Minimum Frontier Inputs For A8-Oriented Progress

If the goal is "quickly bring it up to `A8`," the frontier packet must include at least:

- one held-out unfamiliar-domain task slice
- one strong-baseline comparison slice
- one long-horizon transfer slice
- one conservative comparison report, not just raw delta averages

Without those, the campaign can improve:

- retained conversion
- counted breadth
- local carryover

But it cannot honestly optimize for `A8`.

It can only optimize the current closure harness.
