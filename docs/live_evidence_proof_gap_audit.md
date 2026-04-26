# Live Evidence And Proof Gap Audit

## Purpose

This is the Lane 5 audit for the ASI-boundary review described in
[`minimal_asi_analysis_lanes.md`](/data/agentkernel/docs/minimal_asi_analysis_lanes.md:236).

It answers one narrow question:
which minimum contracts are only implemented or tested, and which are actually proven by live
integrated runtime evidence as of `2026-04-12`.

If you want the single current consolidated view across live bottlenecks,
substantive kernel gaps, and proof gaps, start with
[`kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md).

## Scope Reviewed

- ASI boundary and lane definitions in [`asi_core.md`](/data/agentkernel/docs/asi_core.md:15),
  [`minimal_asi_analysis_lanes.md`](/data/agentkernel/docs/minimal_asi_analysis_lanes.md:236),
  [`asi_closure_work_queue.md`](/data/agentkernel/docs/asi_closure_work_queue.md:25), and
  [`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md:8)
- executable runtime wiring in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:192)
  and [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2104)
- self-improvement closeout wiring in
  [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:2797)
- bounded tests for runtime progression, latent-state pressure, finalize closeout, and trust breadth
- live artifacts under [`var/`](/data/agentkernel/var), anchored primarily to the latest completed
  integrated unattended run on `2026-04-10`:
  [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json:6),
  [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json:5),
  [`unattended_campaign.events.jsonl`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.events.jsonl:22),
  and
  [`repeated_improvement_status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/repeated_improvement_status.json:10)

Later runs on `2026-04-11` and `2026-04-12` were reviewed as proof updates, not as the primary
anchor, because the newest 12-hour run was interrupted:
[`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260412T014423Z_12h_evidence_vllm/reports/unattended_campaign.status.json:6).

## Verdict

The repo has live proof of:

- a bounded executable coding agent floor
- a live reject-capable self-improvement loop with child-native natural closeout records
- runtime path presence for memory, world-model, and latent-state updates

The repo does not yet have live proof of:

- retained autonomous gains in the latest integrated ASI anchor run
- counted external trust breadth across the required families
- retrieval or Tolbert improvements converting into retained live gains
- stronger decoder-runtime independence from `vllm`

The ASI-shaped runtime minimum is therefore `partially proven live`.
The self-improvement minimum is `proven on the reject side` but still `unproven on the retained
compounding side`.

## Proof Matrix

| Contract | Implemented | Wired | Tested | Detached live | Proven live | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Executable agent floor: task -> state -> action -> bounded execution -> verify -> persist | Yes | Yes | Yes | Yes | Yes | `run_task` wires the loop in [`loop.py`](/data/agentkernel/agent_kernel/loop.py:192). A bounded run succeeds in [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:162). The integrated `2026-04-10` run shows `memory_retrieved`, `state_estimated`, `world_model_updated`, and `memory_update_written` events in the live path at [`events.jsonl`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.events.jsonl:22). |
| ASI-shaped runtime path: memory, world model, universe governance, latent state are consumed in decision scoring | Yes | Yes | Yes | Yes | Partial | The runtime builds world, universe, and latent summaries in [`loop.py`](/data/agentkernel/agent_kernel/loop.py:321) and refreshes them after each step in [`loop.py`](/data/agentkernel/agent_kernel/loop.py:555). Policy scoring consumes universe, world, graph, and latent signals in [`policy.py`](/data/agentkernel/agent_kernel/policy.py:2104). Local tests prove controller-belief pressure changes scores in [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:1612) and that learned world state is persisted in [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:172). The integrated run proves path presence, but not yet that these signals produced a better retained unattended outcome. |
| Self-improvement closeout: candidate vs baseline evaluation can emit child-native runtime-managed decisions | Yes | Yes | Yes | Yes | Yes | Finalize wiring starts in [`cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:2797). Tests cover strategy-lineage propagation in [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py:949) and child-native natural runtime-managed closeout in [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py:4692). The `2026-04-10` integrated run recorded `runtime_managed_decisions=3` with child-native natural closeout in [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json:103) and [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json:1206). |
| Retained-gain compounding: autonomous closeout converts into retained gains | Yes | Yes | Yes | Partial | No | Retain-path tests exist in [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py:1837), but the latest completed integrated ASI anchor still reports `retained_gain_runs=0` in [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json:105) and [`repeated_improvement_status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/repeated_improvement_status.json:13). |
| Trust breadth: required families and external evidence are counted, not just sampled | Yes | Yes | Yes | Partial | No | Trust-ledger coverage is tested in [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py:436) and runtime-managed family signal handling is tested in [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py:703). Live integrated evidence still shows `external_report_count=0`, `distinct_external_benchmark_families=0`, and missing required families in [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json:10749). |
| Stronger decoder-runtime independence: retained unattended runs start on hybrid or non-`vllm` control | Yes | Yes | Yes | Partial | No | Provider preference is tested in [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py:11307). The latest completed integrated anchor still starts `provider=vllm` in [`unattended_campaign.stdout.log`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.stdout.log:131). Later hybrid-evidence reruns did not strengthen this claim because they ended as `partial_timeout_evidence_only` with `runtime_managed_decisions=0` in [`2026-04-11` report](/data/agentkernel/var/unattended_run_20260411T225300Z_asi_hybrid_evidence_rerun/reports/unattended_campaign.json:3207). |

## What Is Complete

- The executable agent minimum is live, bounded, and evidenced in an integrated unattended run.
- The runtime really does execute graph-memory retrieval, world-model estimation, latent-state
  refresh, and memory-update writes on the live path.
- The self-improvement loop can emit child-native runtime-managed reject decisions with natural
  closeout and persisted decision records.

## What Is Partial

- The ASI-shaped runtime has live path evidence, but not yet live gain-attribution evidence.
- Detached validation and bounded tests prove more than the latest integrated unattended reruns do.
- Reject-side self-improvement is stronger than retain-side compounding evidence.

## What Is Missing

- A fresh integrated run with at least one child-native natural retained gain.
- Counted external breadth across `integration`, `project`, `repository`, and `repo_chore`.
- Live retained evidence that retrieval or Tolbert routing changed later repair or verification
  behavior.
- A fresh integrated anchor newer than `2026-04-10`; the `2026-04-12` 12-hour run was interrupted
  before it could upgrade the proof state.

## Doc Agreement Audit

- [`asi_closure_work_queue.md`](/data/agentkernel/docs/asi_closure_work_queue.md:25) is still a
  defensible primary evidence anchor because it openly anchors itself to the `2026-04-10`
  integrated run and states that stronger claims still need fresh proof.
- [`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md:8) is now aligned with the
  Lane 5 evidence posture. It anchors itself to the `2026-04-10` integrated run and explicitly
  marks the April 11–12 reruns as non-upgrading evidence rather than silently treating older proof
  as current.
- Later runs on `2026-04-11` and `2026-04-12` do not justify stronger claims. Both
  [`asi_hybrid_evidence_rerun`](/data/agentkernel/var/unattended_run_20260411T225300Z_asi_hybrid_evidence_rerun/reports/unattended_campaign.json:3277)
  and
  [`asi_partial_report_validation`](/data/agentkernel/var/unattended_run_20260412T004537Z_asi_partial_report_validation/reports/unattended_campaign.json:3481)
  ended with `runtime_managed_decisions=0`, and both record
  `closeout_mode=partial_timeout_evidence_only` at
  [`2026-04-11`](/data/agentkernel/var/unattended_run_20260411T225300Z_asi_hybrid_evidence_rerun/reports/unattended_campaign.json:3207)
  and
  [`2026-04-12`](/data/agentkernel/var/unattended_run_20260412T004537Z_asi_partial_report_validation/reports/unattended_campaign.json:3393).

## Top 3 Fact-Checked Gaps

1. Retained compounding is still a proof gap, not an implementation gap.
   The integrated `2026-04-10` anchor has `runtime_managed_decisions=3` but `retained_gain_runs=0`,
   and the April 11–12 reruns regressed to evidence-only partial timeout closeout instead of
   upgrading that proof.
2. Trust breadth is still sampled more strongly than it is counted.
   The integrated run observes family signal, but live evidence still shows `external_report_count=0`
   and missing required-family proof for `integration` and `repo_chore`.
3. Stronger independence and learned-routing claims remain unproven live.
   The integrated anchor still starts on `provider=vllm`, and Tolbert retain routing still fails in
   live evidence with `never entered retained Tolbert primary routing`.

## Top 3 Next Patches

1. Prioritize one merged-branch rerun whose acceptance gate requires:
   `runtime_managed_decisions>0`, `retained_gain_runs>0`, at least one
   `decision_owner=child_native` with `closeout_mode=natural`, no generated-lane or
   failure-recovery-lane phase-gate reject, and no `partial_timeout_evidence_only` closeout on the
   credited proof path.
2. Split trust-breadth closure into its own live proof gate.
   Do not treat sampled family signal as sufficient until the integrated report records counted
   external breadth for `integration`, `project`, `repository`, and `repo_chore`.
3. Split runtime-independence closure into its own live proof gate.
   Do not treat hybrid-preference code as sufficient until the integrated report records a retained
   or decision-bearing non-`vllm` child start path.

## Minimum Fresh Proof Needed

The smallest rerun that materially upgrades the proof state must show all of the following in one
integrated unattended artifact set:

- `runtime_managed_decisions >= 1`
- `retained_gain_runs >= 1`
- at least one `decision_owner=child_native` with `closeout_mode=natural`
- no `autonomous_generated_lane_missing`
- no `autonomous_failure_recovery_lane_missing`
- no `partial_timeout_evidence_only` closeout on the credited proof path
- nonzero counted breadth for `integration`, `project`, `repository`, and `repo_chore`

If the repo also wants the stronger decoder-independence claim, that same run must additionally
show a retained or at least decision-bearing child starting on `provider=hybrid` or another
non-`vllm` path.
