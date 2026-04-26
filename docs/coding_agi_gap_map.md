# Coding AGI Gap Map

## Purpose

This is the compact map from the current unattended evidence to the highest-value remaining gaps on
the path to compounding coding agency.

If you want the single current consolidated view across live bottlenecks,
substantive kernel gaps, and proof gaps, start with
[`kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md).

It is anchored to the latest completed integrated unattended self-improvement run on `2026-04-10`
from
[`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json)
and the paired campaign report
[`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json).

Newer reruns on `2026-04-11` and `2026-04-12` were reviewed, but they do not upgrade the proof
state:

- [`2026-04-11 hybrid rerun`](/data/agentkernel/var/unattended_run_20260411T225300Z_asi_hybrid_evidence_rerun/reports/unattended_campaign.json)
  finished with `runtime_managed_decisions=0` and `closeout_mode=partial_timeout_evidence_only`
- [`2026-04-12 partial-report validation`](/data/agentkernel/var/unattended_run_20260412T004537Z_asi_partial_report_validation/reports/unattended_campaign.json)
  also finished with `runtime_managed_decisions=0` and `closeout_mode=partial_timeout_evidence_only`
- the newer `12h` evidence attempt on `2026-04-12` was interrupted before it could become a usable
  integrated proof anchor:
  [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260412T014423Z_12h_evidence_vllm/reports/unattended_campaign.status.json)

## Current Read

The current kernel is already beyond a toy seed-task loop. It now has:

- a verifier-driven task runtime in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
- an unattended outer loop in [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- retained-vs-candidate improvement flow in [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- trust and family-breadth accounting in [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
- bounded structured editing plus syntax-motor support in
  [`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py)
  and [`agent_kernel/syntax_motor.py`](/data/agentkernel/agent_kernel/syntax_motor.py)

The blocker is not "the runtime cannot do coding at all." The blocker is that the system still does
not convert unattended work into broad, child-native, repeatable coding gains with counted external
evidence.

## Evidence Snapshot

- the latest unattended parent finished with `status=completed` and `reason=requested unattended rounds completed`
- the integrated report shows `runtime_managed_decisions=3` and `retained_gain_runs=0`
- all three decisions are credited as `child_native` with `closeout_mode=natural`
- sampled priority families from the integrated run include `integration`, `project`, and
  `repository`
- the trust breadth summary still shows `external_report_count=0`,
  `distinct_external_benchmark_families=0`, and missing required-family proof for `integration`
  and `repo_chore`
- the main blocker has shifted from "can the child emit decisions at all" to
  "can child-native natural closeout convert into retained gains with counted breadth"

Non-upgrading follow-up evidence:

- the `2026-04-11` hybrid rerun did not strengthen the claim; it ended with
  `runtime_managed_decisions=0` and `closeout_mode=partial_timeout_evidence_only`
- the `2026-04-12` partial-report validation also stayed in evidence-only closeout and did not
  produce a retained or decision-bearing upgrade
- the `2026-04-12` 12-hour run was interrupted, so it should not be treated as a newer proof anchor

Primary artifacts:

- [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json)
- [`unattended_campaign.events.jsonl`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.events.jsonl)
- [`unattended_campaign.stdout.log`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.stdout.log)
- [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json)

## Priority Gaps

### 1. Runtime-managed conversion gap

The kernel can complete unattended rounds and now produce child-native runtime-managed decisions,
but it still fails to convert those natural closeouts into retained gains at a useful rate.

Why this is AGI-critical:

- compounding coding agency requires machine-owned decisions, not only machine activity
- controller-closed productive partial timeouts are better than pure churn, but they still do not
  prove that the child loop can carry decisions to a clean unattended closeout boundary

Primary surfaces:

- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
- [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)

What the live evidence says:

- the current integrated run cleared `runtime_managed_decisions=3`, so the zero-decision failure is
  no longer the immediate blocker
- all three decisions closed as `child_native` with `closeout_mode=natural`, which means decision
  ownership is materially cleaner than the April 9 wave
- the blocking gap is now retained conversion: those three natural closeouts still yielded
  `retained_gain_runs=0`
- the April 11 and April 12 follow-up reruns are negative evidence here because they dropped back to
  `partial_timeout_evidence_only` instead of upgrading the retained-conversion proof
- 2026-04-24 update: the direct five-family bounded `transition_model` route now has a narrow
  retained A4 crossing anchor in `cycle:transition_model:20260423T223800Z:604f5429r6`.
  That replay closed with `final_state=retain`, `decision_owner=child_native`,
  `decision_conversion_state=runtime_managed`, `closeout_mode=natural`, and
  non-regressive transition-model improvement. The remaining gap is no longer
  "can any A4-shaped route retain"; it is "can fresh official repeated campaigns anchor and repeat
  that crossing reliably while preserving counted breadth."

Next attack:

- prioritize `runtime_supervisor`, `cycle_finalize`, and `controller_policy` surfaces before adding
  more candidate-generation breadth

### 2. Trust-breadth and family-coverage gap

The kernel still lacks broad unattended proof across the required coding families, especially
`integration` and `repo_chore`, and it still lacks clean task-root breadth even where reports exist.

Why this is AGI-critical:

- coding AGI is not just "win one lane repeatedly"; it requires transfer across repo settings,
  workflows, and validation shapes
- without family breadth, promotion and supervisor widening remain governance-limited

Primary surfaces:

- [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
- [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
- [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
- [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
- [`scripts/run_autonomous_compounding_check.py`](/data/agentkernel/scripts/run_autonomous_compounding_check.py)

What the live evidence says:

- the campaign now samples `integration`, `project`, and `repository` in the live progress stream
- 2026-04-25 update: the live-vLLM delegated queue now has a clean required-family
  `A5_substrate` packet across two isolated proof roots. r5 covered `integration` and
  `repo_chore`; r6 covered `project`, `repository`, and `repo_sandbox`. Together they show
  15 separate `drain --limit 1` invocations, 15 successful jobs, five accepted parent
  promotions, zero blocked jobs at closeout, zero active leases, and zero hidden-side-effect risk
  in both packets:
  [`a5_substrate_required_family_persistent_queue_packet_20260425.json`](/data/agentkernel/docs/evidence/a5_substrate_required_family_persistent_queue_packet_20260425.json).
- 2026-04-25 r8 update: the r7 ordinary workspace failure mode is now patched and packeted.
  `workspace_contract_direct` executes exact non-retrieval, non-shared-repo workspace contracts
  before freeform synthesis, and the repaired live-vLLM r8 sweep completed 10 unique broader
  task roots with zero hidden-side-effect risk:
  [`a5_live_vllm_broader_workspace_contract_r8_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_broader_workspace_contract_r8_20260425.json).
- 2026-04-25 r9 update: the persistent queue resume/reaping substrate is now packeted.
  The isolated r9 proof resumed an interrupted `in_progress` job without an active lease, reaped a
  stale orphan lease for a second `in_progress` job, then completed six separate live-vLLM
  `drain --limit 1` invocations with `trust_status=trusted`, zero blocked jobs at closeout, zero
  active leases, and zero hidden-side-effect risk:
  [`a5_live_vllm_resume_recovery_r9_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_resume_recovery_r9_20260425.json).
- 2026-04-25 r10 update: real process restart/resume now has a bounded packet. The first probe
  exposed that `sleep` was blocked by sandbox executable policy, so bounded `sleep` was added with
  a timeout regression. The clean r10b proof interrupted a live delegated drain with SIGINT during
  an external-manifest task, left the job `in_progress`, then a fresh drain invocation resumed and
  completed the same job with `attempt_count=2` and final `active_leases=[]`:
  [`a5_live_vllm_real_restart_r10_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_real_restart_r10_20260425.json).
- 2026-04-25 r11 update: repeated real process restart/resume now has a bounded required-family
  packet. Five live delegated drains were interrupted with SIGINT during external-manifest tasks
  across `project`, `repository`, `integration`, `repo_chore`, and `repo_sandbox`; every job was
  left `in_progress` with `last_error=runner interrupted`, then completed by a fresh drain with
  `attempt_count=2`, final `active_leases=[]`, `trust_status=trusted`, and zero
  hidden-side-effect risk:
  [`a5_live_vllm_repeated_real_restart_r11_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_real_restart_r11_20260425.json).
- 2026-04-25 r12 update: ordinary built-in queue intake continuity now has a short persistent-role
  packet. Three waves of built-in task-bank jobs were enqueued over a 58-second live-vLLM role
  window, then drained through 12 fresh process invocations. All 12 completed successfully with
  `trust_status=trusted`, zero active leases, and zero hidden-side-effect risk. The packet covers
  `integration=3`, `project=3`, `repo_chore=3`, and `repository=3`; it intentionally does not
  include git-backed `repo_sandbox` jobs:
  [`a5_live_vllm_persistent_role_r12_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_persistent_role_r12_20260425.json).
- 2026-04-25 r13 update: git-backed `repo_sandbox` continuity now has a bounded packet. Five
  built-in `repo_sandbox` jobs completed under live vLLM with git enabled only for the isolated
  proof: two standalone git review/repair tasks plus a shared-repo worker/merge acceptance trio.
  The repo_sandbox family assessment is `trusted`, counted gated `repo_sandbox=5`, with zero
  active leases and zero hidden-side-effect risk:
  [`a5_live_vllm_repo_sandbox_r13_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repo_sandbox_r13_20260425.json).
- 2026-04-25 r14 update: the short combined A5-substrate proof now exists. One live queue window
  combined ordinary built-in intake, git-backed shared-repo work, and a real SIGINT
  interruption/resume of a git-backed external `repo_sandbox` task. All eight jobs completed with
  `trust_status=trusted`, counted coverage across all five required families, zero active leases,
  and zero hidden-side-effect risk:
  [`a5_live_vllm_combined_role_r14_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_combined_role_r14_20260425.json).
- 2026-04-25 r15 update: the stronger selected-intake mixed-role proof now exists. One 184-second
  live queue window selected ordinary built-in task-bank work by deterministic criteria, ran
  git-backed `repo_sandbox` work, and recovered two real SIGINT interruptions, one ordinary and one
  git-backed. All 15 jobs completed with `trust_status=trusted`, counted coverage across all five
  required families, zero active leases, and zero hidden-side-effect risk:
  [`a5_live_vllm_longer_mixed_role_r15_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_longer_mixed_role_r15_20260425.json).
- 2026-04-25 r16 update: repeated-window built-in role continuity now exists without external
  manifest scaffolding. One 294-second live queue role selected only built-in task-bank jobs,
  executed four separated intake/drain windows with 75-second pauses, included git-backed
  `repo_sandbox`, and closed queue-empty and lease-free. All 17 jobs completed with
  `trust_status=trusted`, counted coverage across all five required families, zero active leases,
  and zero hidden-side-effect risk:
  [`a5_live_vllm_repeated_window_r16_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_window_r16_20260425.json).
- 2026-04-25 r17 update: product-native queue role closeout status now exists. The status command
  derives `queue.role_closeout` from terminal job outcomes, active leases, blockers, and the
  unattended trust ledger. Replaying the r16 root through `scripts/run_job_queue.py status --json`
  produced `closeout_ready=true`, `closeout_mode=queue_empty_trusted`,
  `operator_steering_required=false`, 17 completed-success jobs, zero active leases, and
  `trust_status=trusted`:
  [`a5_live_vllm_product_native_role_closeout_r17_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_product_native_role_closeout_r17_20260425.json).
- 2026-04-25 r18 update: less-task-bank-bound external-manifest role continuity now exists with
  native closeout. One 769-second live queue role generated 10 fresh external-manifest tasks,
  executed four separated intake/drain windows with 240-second pauses, included git-backed
  external `repo_sandbox` tasks, and closed through `queue.role_closeout` as
  `queue_empty_trusted` with `operator_steering_required=false`. All 10 jobs completed with
  `trust_status=trusted`, `task_origins={"external_manifest":10}`, counted coverage of two clean
  task roots for each required family, zero active leases, and zero hidden-side-effect risk:
  [`a5_live_vllm_external_native_role_r18_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_external_native_role_r18_20260425.json).
- 2026-04-25 r19 update: product-native manifest intake now exists. The delegated queue CLI has
  `enqueue-manifest`, which loads external workstream manifests through `TaskBank`, enqueues through
  the real queue, infers family budget groups, and embeds resolved task payloads for portable
  drains. The isolated r19 live-vLLM proof used that command against the default external breadth
  manifest, enqueued 10 external jobs in one product command, drained all 10 successfully, and
  closed as `queue_empty_trusted` with two clean task roots counted for each required family:
  [`a5_live_vllm_enqueue_manifest_role_r19_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_enqueue_manifest_role_r19_20260425.json).
- this narrows the trust-breadth gap from "required-family live queue proof missing" to
  "bounded required-family, ordinary workspace, simulated resume/recovery, one real restart, and
  repeated real restart proof, short ordinary built-in intake continuity, git-backed
  `repo_sandbox` continuity, short combined-role continuity, and selected-intake mixed-role
  continuity plus repeated-window built-in continuity exist, with product-native closeout status
  and product-native external-manifest intake, but still need multi-hour production-duty
  persistence and live product/user workstream operation"
- sampled family signal should still not be treated as proof by itself; the upgrade here comes from
  accepted queue reports and static packet verification, not sampled coverage
- primary-only broad-observe runs can now be credited as discovered-strategy selections, but this
  still needs a retained integrated rerun before the gap can be closed from evidence
- detached validation `/data/agentkernel/var/unattended_run_20260410T174416Z_asi_origin_surface_validation`
  now shows clean 16/16 observe across `integration`, `project`, and `repository` and canonical
  cycle rows with `strategy_origin=discovered_strategy` for `strategy:broad_observe_diversification`;
- detached validation `/data/agentkernel/var/unattended_run_20260410T182000Z_asi_lineage_surface_validation`
  proved sibling preview/evaluation lineage promotion, then exposed a narrower closeout-row gap on
  `finalize_cycle` and `persist_retention_outcome`
- closeout-row lineage promotion is now patched in `agent_kernel/cycle_runner.py` and covered by
  `tests/test_finalize_cycle.py::test_finalize_cycle_promotes_strategy_lineage_to_closeout_records`;
  `/data/agentkernel/var/unattended_run_20260410T191100Z_asi_closeout_lineage_validation` is the
  active proof run for the fully patched path

Next attack:

- spend the next wave on multi-hour persistent-role operation over live product/user workstream work,
  using the native `queue.role_closeout` surface as the closeout gate instead of saturating
  retrieval-only probes

### 3. Retrieval-to-action gap

The retrieval subsystem is still selected as a likely improvement target, but improved retrieval
selection is not reliably becoming trusted repair behavior that survives retained gating.

Why this is AGI-critical:

- memory and retrieval only matter if they change later coding behavior
- a coding agent that repeatedly proposes retrieval changes without increasing trusted retrieval use
  or retained gains is not yet compounding

Primary surfaces:

- [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
- [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)

What the live evidence says:

- retrieval pressure is no longer the only visible family signal in the integrated run
- the remaining retrieval gap is carryover proof: the reports still do not broadly demonstrate that
  improved retrieval changed later repair or verification behavior in counted evidence

Next attack:

- focus on traces where trusted retrieval can be shown to change the next repair or verify step, not
  only retrieval selection metrics

### 4. Long-horizon software execution gap

The kernel now has meaningful planner, hotspot, structured-edit, and syntax-motor machinery, but it
still lacks the stronger repo-level execution quality needed for unattended long-horizon coding work.

Why this is AGI-critical:

- coding agency is won or lost in multi-step repo work: preserving working state, targeting the
  right symbol or file, sequencing validation, and recovering from failed plans
- syntax-aware editing helps, but the bigger remaining problem is durable repo-level implementation
  and recovery quality

Primary surfaces:

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py)
- [`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py)
- [`agent_kernel/syntax_motor.py`](/data/agentkernel/agent_kernel/syntax_motor.py)

What the live evidence says:

- the loop can progress through long preview/finalize phases without crashing
- the same loop still does not produce enough broad unattended gains afterward, which means the
  missing piece is no longer "add any syntax aid" but "improve repo-level execution, transfer, and
  recovery quality"

Next attack:

- use syntax-motor and structured-edit support as a motor aid, but rank repo-level recovery,
  verifier-obligation completion, and long-horizon state quality above further local edit-template
  expansion

## What Is Not The Main Gap

- not basic shell execution
- not lack of any planner or world-model machinery
- not lack of any syntax support at all
- not lack of any unattended controller or campaign wrapper

Those surfaces exist. The current failure is the gap between those ingredients and broad,
runtime-managed, family-spanning unattended coding yield.

## Recommended Next Replica Split

1. Runtime-managed decision yield:
   [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py),
   [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
2. Family-breadth and counted trust:
   [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py),
   [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py),
   [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
3. Retrieval carryover into trusted repair:
   [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py),
   [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
4. Long-horizon repo execution quality:
   [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py),
   [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py),
   [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py)

## Summary

The shortest honest statement is:

- the kernel already has many pieces of a coding-AGI runtime
- the current unattended evidence says the biggest remaining problem is not capability absence in
  the abstract, but weak conversion from those capabilities into retained, counted, runtime-managed
  coding gains
- the April 11 and April 12 reruns do not upgrade the proof state and should be treated as
  non-upgrading or negative evidence, not as newer proof by default
- the next highest-ROI work is retained conversion, counted trust breadth, retrieval carryover, and
  long-horizon repo execution quality, in that order
