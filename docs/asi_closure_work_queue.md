# ASI Closure Work Queue

## Purpose

This document is the focused next-wave execution queue for closing:

- the exact blockers surfaced by the latest unattended validation run on `2026-04-10`
- the remaining explicit `no` verdicts in [`asi.md`](/data/agentkernel/asi.md)

It is not an ASI threshold claim.

Closing Stage 1 removes the latest run blockers.
Closing Stage 2 addresses the repo-level ASI substrate gaps that `asi.md` still marks as open.
Even if both stages land, the repo still needs fresh integrated evidence before any stronger claim is
defensible.

Proof-discipline rule:

- sampled family signal is not counted trust breadth
- hybrid-preference code is not runtime-independence proof
- detached validation is useful, but it does not replace one merged-branch integrated rerun

This queue now treats two ideas separately:

- baseline ASI-thesis closure
  - strategic memory, autonomous improvement, retained gains, and failure recovery can compound even
    if the decoder remains Qwen via `vllm`
- stronger independence closure
  - the repo can also reduce or remove dependence on the fixed external decoder/runtime path

## Evidence Anchor

Primary run anchor:

- [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.status.json)
- [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/reports/unattended_campaign.json)
- [`repeated_improvement_status.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/repeated_improvement_status.json)

Key facts from that run:

- `status=completed`, `rounds_completed=1`
- `completed_runs=3`, `runtime_managed_decisions=3`
- all three decisions were `child_native` with `closeout_mode=natural`
- `retained_gain_runs=0`
- `observe_summary.passed=16`, `observe_summary.total=16`
- sampled priority families were `integration`, `project`, and `repository`
- trust breadth remained `external_report_count=0`
- required-family coverage still missed `integration` and `repo_chore`

Exact reject artifacts:

- trust reject:
  [`cycle_report_cycle_trust_20260410T201850590627Z_f3649d03.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/cycle_report_cycle_trust_20260410T201850590627Z_f3649d03.json)
- Tolbert reject 1:
  [`cycle_report_cycle_tolbert_model_20260410T203216268958Z_47aa1de9.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/cycle_report_cycle_tolbert_model_20260410T203216268958Z_47aa1de9.json)
- Tolbert reject 2:
  [`cycle_report_cycle_tolbert_model_20260410T205839493078Z_2f0d19ef.json`](/data/agentkernel/var/unattended_run_20260410T201442Z_asi_reconcile_lineage_validation/improvement_reports/cycle_report_cycle_tolbert_model_20260410T205839493078Z_2f0d19ef.json)

ASI contract anchor:

- [`asi.md`](/data/agentkernel/asi.md)
- explicit open verdicts:
  - `open-world-engine-complete = no`
    - breadth milestone, not the sole ASI criterion
  - `fixed-model-dependence-removed = no`
    - stronger runtime-independence milestone, not a baseline disqualifier if autonomy still
      compounds through strategic memory on `qwen`/`vllm`
  - `validation-complete = no`

## Exact Gaps

### 1. Autonomous Phase-Gate Depth Gap

The `2026-04-10` anchor run rejected trust and Tolbert cycles on missing generated-task and
failure-recovery lanes, but the April 11 codepath reevaluation showed that this was a runtime
forwarding bug rather than a durable kernel capability limit.

Current status:

- `scripts/run_improvement_cycle.py` now forces generated-task and failure-recovery lanes back on
  for retention and sibling-preview evaluation instead of inheriting bounded-observation suppression
- `scripts/run_repeated_improvement_cycles.py` now clears stale decision state when a new `observe`
  cycle starts, so later cycles in the same child are no longer falsely killed as post-decision
  drift
- a fresh detached rerun at
  [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260411T165003Z_reeval_fix/reports/unattended_campaign.status.json)
  has not emitted `autonomous_generated_lane_missing`, `autonomous_failure_recovery_lane_missing`,
  or stale `mid-cycle controller intervention` signatures in its early live log

This lane is therefore reduced in the codebase: the forwarding and stale-decision faults are
patched, and the remaining open item is retained conversion rather than missing generated/failure
lanes.

Primary code surface:

- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)

### 2. Retained Conversion Gap

The latest run proves cleaner decision ownership than the `2026-04-09` wave, but it still converts
`3` productive child-native decisions into `0` retains.

This means the main blocker has shifted from "can the child close out naturally" to "can natural
closeout produce retained gains under autonomous phase gates."

Primary code surfaces:

- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)
- [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)

### 3. Tolbert Retained-Routing Gap

Both Tolbert cycle rejects failed on:

- `Tolbert model candidate never entered retained Tolbert primary routing`

One of them also failed on:

- `retrieval candidate showed no retrieval influence during autonomous evaluation`

Primary code surfaces:

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
- [`agent_kernel/tolbert_model_improvement.py`](/data/agentkernel/agent_kernel/tolbert_model_improvement.py)
- [`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)

### 4. Counted Trust-Breadth Gap

The latest run sampled `integration`, `project`, and `repository`, but the campaign report still
shows:

- `external_report_count=0`
- `distinct_external_benchmark_families=0`
- `missing_required_families=["integration","repo_chore"]`

Primary code surfaces:

- [`agent_kernel/extensions/trust.py`](/data/agentkernel/agent_kernel/extensions/trust.py)
- [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
- [`agent_kernel/tasking/curriculum.py`](/data/agentkernel/agent_kernel/tasking/curriculum.py)
- [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)

### 5. Retrieval Carryover Gap

The repo already treats retrieval as an improvement target, but the live gate still cannot show
that retrieval changed later repair or verification behavior in retained evidence.

Primary code surfaces:

- [`agent_kernel/extensions/improvement/retrieval_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/retrieval_improvement.py)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
- [`agent_kernel/extensions/context_budget.py`](/data/agentkernel/agent_kernel/extensions/context_budget.py)
- [`agent_kernel/extensions/tolbert.py`](/data/agentkernel/agent_kernel/extensions/tolbert.py)

### 6. Open-World Breadth Gap

`asi.md` still says discovered-task growth is mostly replay-derived rather than open-world.
The task bank now also accepts semantic-hub task entries and external manifests as first-class
runtime task sources, unattended rounds now persist runnable semantic-hub task candidates, and the
trust ledger plus unattended status now count semantic-hub, external-manifest, and replay-derived
task-yield buckets explicitly. This gap is now a partial-evidence gap: semantic-hub execution
exists in code and in reporting, but it has not yet converted into retained unattended gains.

This is important breadth evidence, but it is not the whole ASI thesis if the system can already
show strategic-memory-driven autonomous improvement on a fixed decoder runtime.

Primary code surfaces:

- [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
- [`agent_kernel/tasking/benchmark_synthesis.py`](/data/agentkernel/agent_kernel/tasking/benchmark_synthesis.py)
- [`agent_kernel/extensions/improvement/verifier_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/verifier_improvement.py)

### 7. Decoder Runtime Independence Gap

`asi.md` still says fixed-model dependence remains real in the live runtime.
The bounded unattended runner now defaults back to the configured seed provider for the baseline ASI proof path, and the policy layer no longer defaults retained routing to
`fallback_to_vllm_on_low_confidence`, but the real unattended runs on April 10, 2026 and
April 11, 2026 showed mixed evidence. The hybrid rerun did start on `provider=hybrid`, so the
gap is now reduced from "no live provider independence signal" to "live hybrid starts exist, but
they have not yet converted into retained unattended gains."

This lane is now treated as a stronger independence claim rather than the baseline ASI requirement.
The baseline requirement is that strategic memory, control, verification, and retention drive
autonomous self-improvement even if `qwen` via `vllm` remains the active decoder.

Primary code surfaces:

- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
- [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)
- [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)

### 8. Latent World-Model Gap

This lane is now closed for the bounded controller substrate.

Current status:

- the expected-belief substrate is now present in code: a profile-conditioned Tolbert prior feeds
  learned world-signal inference, the loop persists that signal into `latent_state_summary`, and
  policy plus rollout scoring consume it
- the live runtime now also carries explicit controller-belief and expected world-belief
  distributions through latent state and step records, and those belief distributions directly bias
  command scoring and rollout decisions
- remaining world-model work now belongs to the broader open-world-engine gap rather than the
  bounded latent-controller substrate

Primary code surfaces:

- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py)
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
- [`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
- [`agent_kernel/modeling/world/latent_state.py`](/data/agentkernel/agent_kernel/modeling/world/latent_state.py)
- [`agent_kernel/modeling/world/rollout.py`](/data/agentkernel/agent_kernel/modeling/world/rollout.py)

### 9. Evidence-Hygiene And Operational-Closure Gap

`asi.md` still says delegated-job postmortems are too thin and unattended restart semantics are not
yet autonomous handoff. The unattended status now carries explicit handoff and postmortem-readiness
fields, but this lane remains open until those fields are consistently complete across interrupted
and finalized runs.

Primary code surfaces:

- [`agent_kernel/ops/job_queue.py`](/data/agentkernel/agent_kernel/ops/job_queue.py)
- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)

## Stage 1: Latest-Run Closure

Stage 1 is the shortest path to closing the exact failures surfaced by the `2026-04-10` run.

### `lane_autonomous_phase_gate_depth`

- priority: `P0`
- objective:
  make autonomous candidate evaluation include generated-task and failure-recovery lanes whenever
  the phase gate requires them, and make those lanes emit measurable output
- owned_paths:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- done_when:
  - official rerun no longer emits `autonomous_generated_lane_missing`
  - official rerun no longer emits `autonomous_failure_recovery_lane_missing`
  - at least one retained-eligible cycle records nonzero generated-task output under autonomous eval

### `lane_tolbert_retained_routing`

- priority: `P0`
- objective:
  make Tolbert candidates reach retained primary routing under autonomous evaluation and emit
  measurable routing plus retrieval-influence evidence
- owned_paths:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`agent_kernel/tolbert_model_improvement.py`](/data/agentkernel/agent_kernel/tolbert_model_improvement.py)
  - [`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
  - [`tests/test_liftoff.py`](/data/agentkernel/tests/test_liftoff.py)
- done_when:
  - Tolbert cycle reports no longer fail on `never entered retained Tolbert primary routing`
  - retained or shadow-evaluated Tolbert runs record nonzero primary-routing evidence
  - retrieval-influenced steps become positive when retrieval is part of the winning path

### `lane_counted_trust_breadth`

- priority: `P0`
- objective:
  convert sampled family coverage into counted trusted breadth across the required families
- owned_paths:
  - [`agent_kernel/extensions/trust.py`](/data/agentkernel/agent_kernel/extensions/trust.py)
  - [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
  - [`agent_kernel/tasking/curriculum.py`](/data/agentkernel/agent_kernel/tasking/curriculum.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
  - [`scripts/run_autonomous_compounding_check.py`](/data/agentkernel/scripts/run_autonomous_compounding_check.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
  - [`tests/test_curriculum.py`](/data/agentkernel/tests/test_curriculum.py)
- done_when:
  - `integration`, `project`, `repository`, and `repo_chore` have counted trusted evidence
  - the campaign report exposes nonzero external family breadth
  - required-family coverage is no longer missing `integration` or `repo_chore`
  - sampled family signal alone is not being treated as sufficient closure

### `lane_retrieval_carryover`

- priority: `P1`
- objective:
  make retrieval improvements show up as later repair or verification behavior, not only as
  retrieval-selection activity
- owned_paths:
  - [`agent_kernel/extensions/improvement/retrieval_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/retrieval_improvement.py)
  - [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
  - [`agent_kernel/extensions/context_budget.py`](/data/agentkernel/agent_kernel/extensions/context_budget.py)
  - [`agent_kernel/extensions/tolbert.py`](/data/agentkernel/agent_kernel/extensions/tolbert.py)
  - [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
- done_when:
  - retained or trusted runs record positive retrieval-influenced downstream steps
  - reports can attribute a later repair or verify step to trusted retrieval carryover

### `lane_stage1_steward`

- priority: `P0`
- objective:
  integrate Stage 1 lanes and run the single official rerun used to re-evaluate the exact latest-run
  blockers
- owned_paths:
  - [`docs/asi_closure_work_queue.md`](/data/agentkernel/docs/asi_closure_work_queue.md)
  - [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
- done_when:
  - Stage 1 rerun completes from the merged branch
  - this document is updated with a pass/fail readout for each Stage 1 lane

## Stage 2: Remaining ASI-Spec Closure

Stage 2 starts only after Stage 1 produces a clean integrated rerun.

### `lane_strategic_memory_compounding`

- priority: `P0`
- objective:
  convert productive partial autonomous work into persisted task reports and later-round carryover
  so the strategic-memory substrate compounds even when preview or finalize phases run long
- owned_paths:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`agent_kernel/extensions/trust.py`](/data/agentkernel/agent_kernel/extensions/trust.py)
  - [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
- done_when:
  - productive partial runs emit persisted unattended task reports before preview or finalize timeouts
  - later rounds can consume that persisted output as strategic-memory carryover
  - repeated campaigns show that generated-success and generated-failure work changes later behavior

### `lane_open_world_task_acquisition`

- priority: `P1`
- objective:
  broaden discovered-task growth beyond replay-derived ecology into genuinely new environment
  families and verifier shapes
- owned_paths:
  - [`agent_kernel/tasking/task_bank.py`](/data/agentkernel/agent_kernel/tasking/task_bank.py)
  - [`agent_kernel/tasking/benchmark_synthesis.py`](/data/agentkernel/agent_kernel/tasking/benchmark_synthesis.py)
  - [`agent_kernel/extensions/improvement/verifier_improvement.py`](/data/agentkernel/agent_kernel/extensions/improvement/verifier_improvement.py)
  - [`tests/test_task_generation.py`](/data/agentkernel/tests/test_task_generation.py)
  - [`tests/test_generation_scripts.py`](/data/agentkernel/tests/test_generation_scripts.py)
- done_when:
  - new tasks can be enrolled from non-replay signals
  - official reports distinguish open-world discovered tasks from replay-adjacent tasks
  - this evidence expands breadth beyond replay-derived ecology, even when the active decoder
    remains external

### `lane_fixed_model_independence`

- priority: `P2`
- objective:
  reduce live dependence on external `vllm`/Qwen control paths outside the `hybrid` takeover lane
- owned_paths:
  - [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py)
  - [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
  - [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
  - [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
  - [`tests/test_cli_flags.py`](/data/agentkernel/tests/test_cli_flags.py)
  - [`tests/test_liftoff.py`](/data/agentkernel/tests/test_liftoff.py)
- done_when:
  - retained Tolbert routing no longer depends on fallback to the external base runtime
  - runtime docs and config defaults reflect the reduced fixed-model dependence honestly
  - a live integrated report, not just hybrid-preference code, shows a non-`vllm` retained or
    decision-bearing child start path
  - this lane is only required for claims about decoder-runtime independence, not for the baseline
    strategic-memory ASI thesis

### `lane_latent_world_model`

- priority: `P1`
- objective:
  move live planning and recovery from symbolic continuity summaries toward predictive latent-state
  control
- owned_paths:
  - [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py)
  - [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py)
  - [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
  - [`tests/test_modeling_world.py`](/data/agentkernel/tests/test_modeling_world.py)
  - [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py)
- done_when:
  - live planning uses predictive state or rollout features not reducible to static summaries alone
  - reports can show that predictive state changed later action selection or recovery

### `lane_evidence_hygiene`

- priority: `P1`
- objective:
  deepen durable delegated-job diagnostics and improve unattended restart and handoff behavior
- current progress:
  - delegated worker checkpoints now preserve structured verification payloads, failure codes,
    command and success-command summaries, and compact task-contract context for unattended triage
  - live unattended status bridging now reads the child repeated-status artifact directly, so parent
    status no longer lags behind child `campaign_select` progress with stale observe-task detail
  - transparent attach or supersede semantics for fresh unattended invocations are still open
- owned_paths:
  - [`agent_kernel/ops/job_queue.py`](/data/agentkernel/agent_kernel/ops/job_queue.py)
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_job_queue.py`](/data/agentkernel/tests/test_job_queue.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- done_when:
  - durable delegated checkpoints preserve enough stdout or stderr context for unattended diagnosis
  - fresh unattended invocation can resume or hand off instead of only safe-stopping

## Execution Order

1. `lane_autonomous_phase_gate_depth`
2. `lane_tolbert_retained_routing`
3. `lane_counted_trust_breadth`
4. `lane_retrieval_carryover`
5. `lane_stage1_steward` rerun
6. `lane_strategic_memory_compounding`
7. `lane_open_world_task_acquisition`
8. `lane_fixed_model_independence`
9. `lane_latent_world_model`
10. `lane_evidence_hygiene`

Reason:

- the latest run blockers must close before the broader `asi.md` gaps can be measured honestly
- Stage 2 should not be treated as solved if Stage 1 still cannot produce retained autonomous gains
- strategic-memory compounding comes before breadth or runtime-independence because the repo cannot
  claim self-improvement if productive internal work never crosses the reporting and carryover
  boundary

## Acceptance Gates

### Merged-Branch Rerun Acceptance

The next official merged-branch rerun is only a proof upgrade when one integrated unattended
artifact set shows all of the following:

1. `runtime_managed_decisions >= 1`
2. `retained_gain_runs >= 1`
3. at least one `decision_owner=child_native` with `closeout_mode=natural`
4. no `autonomous_generated_lane_missing`
5. no `autonomous_failure_recovery_lane_missing`
6. no `partial_timeout_evidence_only` closeout on the credited proof path
7. nonzero counted breadth for `integration`, `project`, `repository`, and `repo_chore`

This rerun gate intentionally does not treat either of the following as sufficient:

- sampled family signal without counted breadth
- hybrid-preference code or configuration without a live non-`vllm` decision-bearing start path

### Trust-Breadth Proof Gate

Trust-breadth closure is complete only when a live integrated report records all of the following:

1. `external_report_count > 0`
2. `distinct_external_benchmark_families > 0`
3. counted required-family proof for `integration`, `project`, `repository`, and `repo_chore`
4. required-family coverage is no longer missing `integration` or `repo_chore`

Observed or sampled family signal does not close this gate by itself.

### Runtime-Independence Proof Gate

Decoder-runtime independence is a separate stronger claim and is complete only when a live
integrated report records all of the following:

1. a retained or at least decision-bearing child start path on `provider=hybrid` or another
   non-`vllm` runtime
2. that path is part of the official merged-branch unattended evidence, not only a detached check
3. the claim no longer depends on hybrid-preference code existing in isolation

This gate is not required for the baseline strategic-memory ASI-thesis claim.

### Stage 1 Acceptance

Stage 1 is complete only when the merged-branch rerun acceptance gate passes and the Stage 1 lane
owners can point to that one official integrated artifact set as the closure anchor.

### Stage 2 Acceptance

Stage 2 is complete only when fresh integrated evidence supports all of the following:

1. productive partial autonomous work converts into persisted task reports and later-round carryover
2. failure-recovery and generated-success lanes remain viable on the active runtime, including
   `qwen` via `vllm` if that is still the decoder
3. open-world task acquisition is distinguishable from replay-derived growth when broader source
   ecology is present
4. predictive latent-state control affects live action selection
5. evidence hygiene is strong enough for unattended diagnosis and restart continuity
6. fixed-model dependence is materially reduced only if the repo wants to make the stronger
   decoder-runtime-independence claim

## Honest Claim Boundary

If this queue closes cleanly, the repo will be much closer to the `asi.md` engineering contract.
That still does not mean "we are at ASI."

The honest post-queue statement would be:

- the latest unattended blocker set is closed
- the explicit `asi.md` substrate gaps are narrower or closed where fresh evidence says so
- the baseline autonomy claim rests on strategic memory, verification, retention, and compounding
  behavior, not on decoder replacement by itself
- stronger autonomy language still depends on the next integrated proof run, not on the queue alone
