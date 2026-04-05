# Unattended Work Queue

## Purpose

This is the live human-readable claim ledger for parallel Codex work supporting the current
unattended campaign.

Use it for:

- active claims
- lane ownership
- bounded handoffs
- blocked items
- closeout summaries

Do not use this file for long historical analysis.

## Current Wave

- objective: `fill the highest-ROI kernel gaps toward coding AGI while the unattended campaign keeps running`
- run root:
  [`var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive)
- status source:
  [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json)
- event source:
  [`unattended_campaign.events.jsonl`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.events.jsonl)
- stdout source:
  [`unattended_campaign.stdout.log`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.stdout.log)
- coordination runbook:
  [`docs/unattended_parallel_codex_runbook.md`](/data/agentkernel/docs/unattended_parallel_codex_runbook.md)

## Live Snapshot

- unattended parent state: `running`
- active round: `2`
- active child pid: `3727427`
- current child phase: `campaign`
- last confirmed progress phase: `preview_baseline_eval`
- current concrete work:
  `recovery round 2 is alive inside retrieval breadth_rebalance preview baseline and is currently deep in a 340-task comparison sweep that has already drifted through bounded, repository, integration, episode_memory, and verifier_memory tasks`
- current campaign shape:
  `adaptive_search=true seed policy at campaign_width=3 cycles=2 task_limit=4, active recovery policy expanded to campaign_width=4 cycles=3 task_limit=8`
- current dominant subsystem:
  `retrieval`

## Post-Restart Read

- restart status:
  the unattended campaign was restarted again at `2026-04-05T02:30:19Z` so the current live run
  now includes the landed retrieval-retained-gate patch in addition to the earlier controller,
  Tolbert, finalize, trust-breadth, retrieval-carryover, and long-horizon patches
- current live run:
  `run_match_id=unattended:round:20260405T023047721303Z:1 parent_pid=3611403 child_pid=3611413 cycle_pid=3611463`
- latest completed report:
  [`campaign_report_20260405T023048457279Z.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T023048457279Z.json)
- current hard evidence:
  the latest completed report is still a decision-conversion miss:
  `productive_runs=2`, `retained_gain_runs=0`, `runtime_managed_decisions=0`, and
  `decision_conversion_summary.partial_productive_without_decision_runs=2`
- current pressure:
  the live run is no longer blocked on startup, status mirroring, or raw family sampling; the
  remaining pressure is that retrieval still monopolizes candidate work, preview baseline is now
  ballooning into long memory-heavy compare sweeps, and none of that activity is converting into a
  counted runtime-managed decision or retained gain
- implication for new terminals:
  new terminals should now open bounded post-restart lanes against the live run because the
  remaining gaps are already concrete: decision conversion, retrieval monopoly breaking, preview
  compare-shape control, and required-family decision yield

## Central Goal

The point of this wave is not only to rescue the current unattended run. It is to use the live run
as pressure to close the highest-value gaps between the current kernel and a more robust coding-AGI
operator loop:

1. adaptive supervisor timing that follows real progress
2. reliable child validation and recovery semantics
3. stronger unattended controller policy adaptation
4. stable TOLBERT context/runtime behavior
5. machine-readable coordination so multiple Codex workers can compound instead of collide

This wave assumes one unattended run can be more valuable than many concurrent unattended runs if
multiple Codex replicas use that single live run as a shared pressure source. The replicas should
split kernel gaps, communicate through this queue, and refine the operator loop in parallel instead
of multiplying unattended runtime noise.

## Lane Rules

Each worker must:

1. claim exactly one lane
2. stay inside owned paths for that lane
3. use the current unattended run artifacts as evidence
4. avoid touching the live run root except for diagnostics explicitly called out in the claim
5. leave a closeout block with verification and restart requirement

Do not let two workers edit the same shared unattended entrypoint at the same time.

## Fresh Client Ready

This queue is intended to be sufficient handoff context for a fresh Codex client when paired with:

- [`docs/unattended_parallel_codex_runbook.md`](/data/agentkernel/docs/unattended_parallel_codex_runbook.md)
- the live unattended artifacts linked in `Current Wave`

A fresh client should be able to:

1. identify the active run root
2. identify the active round and current runtime shape
3. see which lanes are already claimed
4. choose one unclaimed lane
5. append its claim block and begin bounded development work

If any of those steps require extra chat reconstruction, this file should be updated before
creating more replica workers.

## Priority Lanes

### `unattended_lane__runtime_supervisor`

- priority: `P0`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T00:45:00Z`
- question:
  refine the unattended supervisor against the current live run so productive children get time to
  finish, broken child outcomes are classified correctly, and status/event reporting remains a
  reliable coordination surface for replica workers
- why it matters:
  the supervisor owns the difference between productive drift and real stuckness
- owned_paths:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- target questions:
  - do runtime grace and timeout semantics match real progress across repeated preview/finalize re-entry
  - does child validation distinguish clean reject, broken report, timeout, and supervisor-managed recovery correctly
  - does live status reflect the active child cleanly even after prior child exits
- expected outputs:
  - supervisor patch or bounded root-cause report
  - targeted unattended-campaign regression tests
- verification_plan:
  - `python -m py_compile scripts/run_unattended_campaign.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_campaign.py -k "runtime_grace or recovery or adaptive_progress_state"`
- restart_requirement: `likely`

### `unattended_lane__controller_policy`

- priority: `P0`
- status: `claimed`
- agent_name: `Dewey`
- claimed_at: `2026-04-05T00:45:52Z`
- question:
  refine controller adaptation for repeated reject-only retrieval rounds so search shape broadens
  sooner when no retained gain is being produced
- why it matters:
  the unattended controller decides whether the outer loop compounds or churns
- owned_paths:
  - [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  - [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
- target questions:
  - are repeated reject-only retrieval rounds adapting search shape aggressively enough
  - should no-retained-gain on the same subsystem bias toward broader family exploration sooner
  - are stop budgets and support-aware priors aligned with current round evidence
- expected outputs:
  - controller-policy patch or scoring diagnosis
  - targeted controller tests
- verification_plan:
  - `python -m py_compile agent_kernel/unattended_controller.py tests/test_unattended_controller.py`
  - `pytest -q tests/test_unattended_controller.py`
- restart_requirement: `likely`
- released_at: `2026-04-05T00:50:51Z`
- result: `completed`
- findings:
  - live round 2 evidence showed repeated reject-only retrieval pressure with zero retained gain while the controller still lacked a strong bias toward broader adaptive exploration
  - the controller now penalizes subsystem reject pressure in round reward and adds a pressure-alignment bonus during policy selection for broader adaptive policies under reject-heavy retrieval and breadth pressure
- artifacts:
  - [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  - [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
- verification:
  - `python -m py_compile agent_kernel/unattended_controller.py tests/test_unattended_controller.py`
  - `pytest -q tests/test_unattended_controller.py`
- restart_requirement: `yes`
- next_recommendation: `restart the unattended campaign only after merging the controller and Tolbert patches so the next live run picks up both broader retrieval adaptation and improved Tolbert startup behavior`
- last_updated: `2026-04-05T00:54:55Z`
- current_progress:
  - the restarted unattended run is now reaching retrieval `preview_baseline_complete` and `preview_candidate_complete` for both `breadth_rebalance` and `confidence_gating`, so the old Tolbert startup failure is no longer blocking the same finalize path
  - controller-specific evidence is still pending because the current live round has not emitted an end-of-round policy-shift rationale yet; the decisive check is whether post-round adaptation broadens beyond repeated retrieval pressure rather than only inheriting the restart seed policy

### `unattended_lane__tolbert_runtime`

- priority: `P1`
- status: `claimed`
- agent_name: `Raman`
- claimed_at: `2026-04-05T00:47:16Z`
- question:
  harden TOLBERT startup and service reuse so cold starts stop failing preview/finalize evals while
  preserving bounded per-query runtime
- why it matters:
  TOLBERT startup 
- owned_paths:
  - [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
  - [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py)
  - [`tests/test_tolbert.py`](/data/agentkernel/tests/test_tolbert.py)
- target questions:
  - can service reuse be made more reliable across repeated preview/finalize evals
  - are startup budget and readiness semantics still too conservative under cold start
  - is there avoidable process churn during unattended recovery rounds
- expected outputs:
  - runtime hardening patch or startup/reuse diagnosis
  - targeted Tolbert tests
- verification_plan:
  - `python -m py_compile agent_kernel/tolbert.py tests/test_tolbert.py`
  - `pytest -q tests/test_tolbert.py -k "startup_ready or startup_timeout or service_client"`
- restart_requirement: `likely`
- released_at: `2026-04-05T00:49:26Z`
- result: `completed`
- findings:
  - live unattended failures were repeatedly caused by TOLBERT cold start missing the fixed 15s readiness budget during preview/finalize eval startup
  - the Tolbert client now defers process creation until first query and uses a separate startup readiness budget, preserving the tighter per-query timeout
  - structured Tolbert startup telemetry now includes `startup_elapsed_seconds`, `pid`, and `cache_shard_count`, so future unattended logs show whether cold-start cost or shard shape is driving readiness failures
- artifacts:
  - [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
  - [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py)
  - [`tests/test_tolbert.py`](/data/agentkernel/tests/test_tolbert.py)
- verification:
  - `python -m py_compile agent_kernel/tolbert.py scripts/tolbert_service.py tests/test_tolbert.py`
  - `pytest -q tests/test_tolbert.py -k "waits_for_startup_ready or retries_startup_ready_timeout_once or reports_attempt_count_after_repeated_startup_timeout or uses_distinct_startup_timeout_budget or close_terminates_live_process or main_emits_structured_startup_telemetry"`
- restart_requirement: `yes`
- next_recommendation: `restart the unattended campaign so live preview/finalize rounds pick up both the Tolbert cold-start hardening and the new startup telemetry fields in stderr`

### `unattended_lane__cycle_finalize`

- priority: `P1`
- status: `claimed`
- agent_name: `Pasteur`
- claimed_at: `2026-04-05T00:47:10Z`
- question:
  refine cycle finalize behavior so repeated preview/candidate/finalize work reuses safe results,
  preserves fallback quality, and emits reject reasons that improve controller adaptation
- why it matters:
  preview/finalize behavior is still where candidate quality, fallback, and retention gates converge
- owned_paths:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- target questions:
  - should preview baseline/candidate/finalize do better fallback or caching across same-cycle variants
  - are we repeating expensive eval phases that could be safely reused
  - are reject reasons machine-readable enough to steer controller adaptation
- expected outputs:
  - finalize-path patch or bounded analysis
  - targeted finalize tests
- verification_plan:
  - `python -m py_compile agent_kernel/cycle_runner.py tests/test_finalize_cycle.py`
  - `pytest -q tests/test_finalize_cycle.py -k "finalize or preview or reject"`
- restart_requirement: `likely`
- released_at: `2026-04-05T00:56:00Z`
- result: `completed`
- findings:
  - preview retention now exposes machine-readable reject `reason_code` values instead of prose-only reject diagnostics
  - final cycle reports now persist both `preview_reason_code` and `decision_reason_code`, and reject cases emit structured progress lines that controller logic can consume directly
  - confirmation and holdout finalize evals now reuse the preview Tolbert startup retry path so transient cold-start failures stop turning into false finalize rejects
  - confirmation rejects now preserve the nested underlying reject code instead of collapsing to the generic `confirmation_run_failed` label
- artifacts:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/cycle_runner.py /data/agentkernel/tests/test_finalize_cycle.py`
  - `pytest -q /data/agentkernel/tests/test_finalize_cycle.py -k "preview_candidate_retention_exposes_machine_readable_reject_reason_code or finalize_cycle_records_machine_readable_reject_reason_codes or finalize_cycle_emits_progress_heartbeat or finalize_cycle_propagates_protocol_match_id_to_finalize_records or preview_candidate_retention_retries_without_tolbert_context_after_startup_failure or preview_candidate_retention_does_not_retry_non_tolbert_failures"`
  - `pytest -q /data/agentkernel/tests/test_finalize_cycle.py -k "write_cycle_report_includes_validation_family_prior_retained_guard_code or write_cycle_report_includes_shared_repo_prior_retained_guard_code"`
  - `pytest -q /data/agentkernel/tests/test_finalize_cycle.py -k "preserves_nested_confirmation_reject_reason_code or confirmation_eval_without_tolbert_context or holdout_eval_without_tolbert_context"`
- restart_requirement: `yes`
- next_recommendation: `restart the unattended campaign with the controller patch so the next live round sees both structured reject codes and stronger no-retained-gain adaptation`

### `unattended_lane__coordination_surface`

- priority: `P1`
- status: `claimed`
- agent_name: `Kuhn`
- claimed_at: `2026-04-05T00:00:00Z`
- question:
  create the centralized coordination surface for this live unattended wave so replicas can claim
  non-colliding AGI-gap work immediately
- owned_paths:
  - [`docs/unattended_parallel_codex_runbook.md`](/data/agentkernel/docs/unattended_parallel_codex_runbook.md)
  - [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
  - [`docs/index.md`](/data/agentkernel/docs/index.md)
- status: `completed`
- result:
  added the unattended parallel runbook and this live work queue with claimable lanes tied to the
  active run
- verification:
  manual doc review against current unattended artifacts
- restart_requirement: `no`

### `unattended_lane__coding_agi_gap_map`

- priority: `P2`
- status: `claimed`
- agent_name: `Kuhn`
- claimed_at: `2026-04-05T00:47:29Z`
- question:
  map the highest-ROI coding-AGI capability gaps exposed by the current unattended safe-stop so the
  next replica wave can attack runtime-critical surfaces in priority order
- why it matters:
  not every unattended failure is equally relevant to coding AGI; the repo needs a compact map from
  observed failures to missing capability layers
- owned_paths:
  - [`docs/architecture.md`](/data/agentkernel/docs/architecture.md)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
  - [`docs/improvement.md`](/data/agentkernel/docs/improvement.md)
  - new doc under [`docs/`](/data/agentkernel/docs)
- target questions:
  - what capability gaps are still blocking compounding coding agency
  - which ones are now runtime-critical based on the unattended evidence
  - which surfaces should parallel replicas attack next after the current wave
- expected outputs:
  - compact AGI-gap map doc with direct code/file references
- verification_plan:
  - `python -m py_compile agent_kernel/unattended_controller.py agent_kernel/cycle_runner.py scripts/run_unattended_campaign.py`
  - `python - <<'PY'\nfrom pathlib import Path\nfor path in [Path('docs/architecture.md'), Path('docs/runtime.md'), Path('docs/improvement.md'), Path('docs/coding_agi_gap_map.md')]:\n    assert path.exists(), path\nPY`
- restart_requirement: `no`
- released_at: `2026-04-05T00:51:04Z`
- result: `completed`
- findings:
  - the live unattended safe-stop is a coding-AGI conversion failure, not a total runtime failure:
    round 2 finished cleanly but still produced `0` runtime-managed decisions
  - the highest-ROI remaining blockers are runtime-managed decision yield, trust/family breadth,
    retrieval carryover into trusted repair, and long-horizon repo execution quality, in that order
- artifacts:
  - [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
  - [`docs/architecture.md`](/data/agentkernel/docs/architecture.md)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
  - [`docs/improvement.md`](/data/agentkernel/docs/improvement.md)
- verification:
  - `python -m py_compile agent_kernel/unattended_controller.py agent_kernel/cycle_runner.py scripts/run_unattended_campaign.py`
  - `python - <<'PY'\nfrom pathlib import Path\nfor path in [Path('/data/agentkernel/docs/architecture.md'), Path('/data/agentkernel/docs/runtime.md'), Path('/data/agentkernel/docs/improvement.md'), Path('/data/agentkernel/docs/coding_agi_gap_map.md')]:\n    assert path.exists(), path\nprint('docs-ok')\nPY`

## Claim Template

Copy this block and fill it in before editing anything:

```md
### `<lane_id>`

- priority: `P?`
- status: `claimed`
- agent_name: `<worker name>`
- claimed_at: `<UTC timestamp>`
- question: `<one bounded question>`
- owned_paths:
  - `<path>`
  - `<path>`
- verification_plan:
  - `<command>`
  - `<command>`
- restart_requirement: `unknown`
```

## Closeout Template

When the lane ends, append:

```md
- released_at: `<UTC timestamp>`
- result: `completed|blocked|aborted|handed_off`
- findings:
  - `<high-signal finding>`
  - `<high-signal finding>`
- artifacts:
  - `<path>`
  - `<path>`
- verification:
  - `<command>`
  - `<command>`
- restart_requirement: `yes|no`
- next_recommendation: `<one next move>`
```

## Non-Collision Rules

- `runtime_supervisor` owns `run_unattended_campaign.py`
- `controller_policy` owns `unattended_controller.py`
- `tolbert_runtime` owns `tolbert.py` and `tolbert_service.py`
- `cycle_finalize` owns `cycle_runner.py`
- `coordination_surface` owns unattended coordination docs

If a lane needs a shared entrypoint outside its default ownership, it must either:

1. hand off to the owning lane, or
2. get explicit coordinator reassignment first

Replica workers should treat this queue as the shared lock table for the current unattended wave.
There is no value in creating multiple unattended campaigns when the same effort can be spent on
parallel kernel refinement against the evidence from the single live run.

## Parallel Execution Split

Use this split for the current coding-AGI gaps. Do not assign one worker the whole gap map.

### `execution_group__critical_path`

These lanes are the shortest path to converting live unattended activity into counted runtime-owned
outcomes. The current live evidence puts preview compare-shape control first because the active
round is spending budget inside a `340`-task retrieval preview baseline dominated by
`episode_memory` and `verifier_memory` replay families rather than decision-bearing evaluation.

1. `unattended_lane__runtime_supervisor`
2. `unattended_lane__preview_compare_shape`
3. `unattended_lane__retained_decision_conversion`
4. `unattended_lane__retrieval_monopoly_breaker`
5. `unattended_lane__cycle_finalize`
6. `unattended_lane__controller_policy`

Reason:

- the active child is not stuck; it is progressing through the wrong compare shape
- replay-heavy preview expansion is currently consuming the runtime budget before decision
  conversion can matter
- runtime-managed decision yield is still the top scoreboard blocker once preview slices are
  discriminative again
- controller adaptation is only worth as much as the decision stream it is steering

Restart guidance:

- batch compatible changes from these lanes into one restart window
- do not restart after each patch if the live run is still yielding fresh evidence

### `execution_group__next_parallel_wave`

These first-wave lanes are completed and are preserved here as the change log for the restart that
is now live. New terminals should not re-open them unless a fresh report directly regresses one of
their owned surfaces.

#### `unattended_lane__family_breadth_trust`

- priority: `P1`
- status: `completed`
- agent_name: `Dewey`
- claimed_at: `2026-04-05T01:00:06Z`
- question:
  convert the live unattended wave's narrow retrieval-heavy activity into counted cross-family
  runtime-managed signal by tightening trust/curriculum accounting and surfacing clean family
  breadth earlier
- why it matters:
  coding AGI cannot be claimed from repeated narrow success in one subsystem or family
- owned_paths:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
- target questions:
  - how do we turn `integration` and `repo_chore` into counted clean signal instead of unsampled or uncounted work
  - should curriculum and trust accounting bias more aggressively toward contract-clean family coverage before more retrieval churn
  - are light-supervision-clean opportunities being surfaced early enough in the unattended wave
- expected outputs:
  - family-breadth/trust patch or bounded diagnosis
  - targeted trust/curriculum tests
- verification_plan:
  - `python -m py_compile agent_kernel/trust.py agent_kernel/curriculum.py agent_kernel/task_bank.py scripts/run_supervisor_loop.py`
  - `pytest -q tests -k "trust or curriculum or supervisor_loop"`
- restart_requirement: `likely`
- released_at: `2026-04-05T01:09:01Z`
- result: `completed`
- findings:
  - the live unattended campaign report still showed zero runtime-managed decisions while `trust_breadth_summary` was missing `integration` and `repo_chore` clean task-root breadth entirely
  - the trust ledger now ingests the latest improvement campaign report and carries runtime-managed breadth signals from campaign `required_families_with_reports` plus sampled families seen in live partial progress
  - breadth-gap families were already visible in the trust ledger, but discovery priority only treated them as weak add-on pressure; missing zero-coverage families were not being pulled ahead of already-sampled families strongly enough
  - the supervisor loop now orders breadth-gap families first and gives zero-coverage breadth families stronger discovery weight so trust-breadth pressure becomes active sampling pressure instead of passive diagnostics
- artifacts:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
  - [`tests/test_generation_scripts.py`](/data/agentkernel/tests/test_generation_scripts.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/trust.py /data/agentkernel/scripts/run_supervisor_loop.py /data/agentkernel/tests/test_trust.py /data/agentkernel/tests/test_generation_scripts.py`
  - `pytest -q /data/agentkernel/tests/test_trust.py -k "campaign_runtime_managed_signals or summarize_recent_reports or light_supervision_independence"`
  - `pytest -q /data/agentkernel/tests/test_generation_scripts.py -k "prefers_runtime_managed_campaign_signal_over_clean_gap or blocks_bootstrap_finalize_when_required_task_root_breadth_is_missing or prioritizes_repository_breadth_trust_discovery_and_routes_to_repository_lane or prioritizes_zero_coverage_breadth_families_in_discovery or scales_trust_priority_discovery_budget_from_family_pressure or retains_mild_repository_breadth_focus_from_memory"`
- restart_requirement: `yes`
- next_recommendation: `pick this up in the next supervisor-loop or unattended restart window alongside the already-landed controller/finalize/Tolbert patches so missing families stop staying behind retrieval-heavy discovery pressure`

#### `unattended_lane__retrieval_carryover`

- priority: `P1`
- status: `claimed`
- agent_name: `Dewey`
- claimed_at: `2026-04-05T01:01:01Z`
- question:
  make trusted retrieval carryover measurably influence later repair and verify behavior instead of
  mostly improving retrieval-side metrics
- why it matters:
  retrieval only matters if it changes later repair and verify behavior that survives retained gates
- owned_paths:
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
  - [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
- target questions:
  - where is trusted retrieval failing to influence the next concrete repair or verify step
  - should policy and context-budget logic preserve retrieval-backed intent longer across steps
  - can retention compare paths reward repair-changing retrieval carryover instead of retrieval metrics alone
- expected outputs:
  - retrieval-carryover patch or bounded diagnosis
  - targeted retrieval/policy tests
- verification_plan:
  - `python -m py_compile agent_kernel/retrieval_improvement.py agent_kernel/policy.py agent_kernel/context_budget.py`
  - `pytest -q tests -k "retrieval or context_budget or policy"`
- restart_requirement: `likely`
- released_at: `2026-04-05T01:02:35Z`
- result: `completed`
- findings:
  - retrieval carryover credit in [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py) was diluted against `metrics.total`, so successful long-horizon carryover repairs could be undercounted whenever bounded or unrelated tasks dominated the eval mix
  - carryover scoring now normalizes against successful long-horizon tasks, which better matches the actual target behavior: trusted retrieval changing later repair/verify steps on repo-scale work
- artifacts:
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/retrieval_improvement.py /data/agentkernel/tests/test_improvement.py`
  - `pytest -q /data/agentkernel/tests/test_improvement.py -k "retrieval_confidence_gating_recognizes_verified_long_horizon_carryover_repairs or retrieval_confidence_gating_does_not_overcredit_short_horizon_carryover_repairs or retrieval_confidence_gating_credits_long_horizon_carryover_even_when_bounded_tasks_dominate"`
- restart_requirement: `yes`
- next_recommendation: `let the supervisor review this with the controller and finalize patches, then use the next live report to check whether retrieval proposals stop over-optimizing retrieval-side metrics relative to trusted repair carryover`

#### `unattended_lane__long_horizon_execution`

- priority: `P2`
- status: `completed`
- agent_name: `Curie`
- claimed_at: `2026-04-05T01:06:00Z`
- question:
  harden long-horizon workflow obligation tracking so repo-scale plans do not prematurely clear
  required test work after unrelated successful verification
- why it matters:
  the remaining gap is increasingly repo-level execution and recovery quality, not just local syntax help
- owned_paths:
  - [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
  - [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
  - [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py)
- target questions:
  - where is long-horizon repo execution still losing state, drifting plan quality, or failing recovery
  - should world-model or step-state carry stronger verifier-obligation tracking across longer runs
  - which repo-scale failure patterns are still invisible to the current loop state
- expected outputs:
  - long-horizon execution patch or bounded diagnosis
  - targeted loop/state/world-model tests
- verification_plan:
  - `python -m py_compile agent_kernel/loop.py agent_kernel/state.py agent_kernel/world_model.py`
  - `pytest -q tests -k "loop or state or world_model"`
- restart_requirement: `likely`
- current_progress:
  - live unattended evidence is still dominated by retrieval cycles with `long_horizon_retained_cycles=0`, so this lane is targeting a structural long-horizon state-tracking bug rather than an in-flight child crash
  - `agent_kernel/state.py` currently treats `run workflow test ...` as satisfied after any passed verification step, which can incorrectly drop required workflow-test obligations on repo-scale tasks
- released_at: `2026-04-05T01:10:00Z`
- result: `completed`
- findings:
  - live unattended reports still show `long_horizon_retained_cycles=0`, so the actionable gap for this lane was state fidelity rather than a currently failing long-horizon child
  - `agent_kernel/state.py` was clearing `run workflow test ...` subgoals after any successful verification step, even when the passing command was unrelated to the required workflow test
  - workflow-test subgoals now require matching successful execution evidence for the specific verifier test command, which prevents long-horizon plans from drifting past unresolved test obligations
- artifacts:
  - [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
  - [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/state.py /data/agentkernel/tests/test_loop.py`
  - `pytest -q /data/agentkernel/tests/test_loop.py -k "keeps_workflow_test_until_matching_test_command_passes or clears_workflow_test_after_matching_test_command_passes or advances_workflow_branch_subgoals or refresh_plan_progress_clears_planner_recovery_artifact_when_source_subgoal_is_satisfied"`
- restart_requirement: `no`
- next_recommendation: `keep the current unattended run alive for controller/runtime evidence; revisit long-horizon execution next when a live round starts surfacing repo-scale decisions instead of retrieval-only pressure`

### `execution_group__post_restart_claimable`

These are the remaining claimable lanes for fresh Codex terminals after the restart. Each lane is
defined to attack the current post-restart coding-AGI bottlenecks rather than to re-harden
already-patched infrastructure.

#### `unattended_lane__retained_decision_conversion`

- priority: `P0`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T04:29:59Z`
- question:
  why is the unattended wave still ending with `partial_productive_without_decision_runs` instead
  of a counted runtime-managed decision after clean `preview_complete`, `apply_decision`, and
  `done` paths
- why it matters:
  coding AGI cannot be claimed while productive machine work still fails to convert into explicit
  machine-owned decisions that the parent can trust and build on
- owned_paths:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- measurable_success:
  the next completed campaign report records `runtime_managed_decisions>=1` or at minimum
  `decision_conversion_summary.decision_runs>=1` for a clean machine reject/record path
- target questions:
  - where is a clean reject/apply-decision path still being classified as partial productive instead
    of decision-bearing
  - are reject-only machine decisions intentionally excluded from `runtime_managed_decisions`, and
    if so should there be a separate retained-safe machine-decision counter
  - does the repeated-improvement report need stronger cycle-to-decision stitching after finalize
- expected outputs:
  - decision-conversion patch or bounded decision-model diagnosis
  - targeted finalize/reporting regressions
- verification_plan:
  - `python -m py_compile agent_kernel/improvement.py agent_kernel/cycle_runner.py scripts/run_repeated_improvement_cycles.py tests/test_finalize_cycle.py`
  - `pytest -q tests/test_finalize_cycle.py -k "decision or apply_decision or runtime_managed or partial_productive"`
- restart_requirement: `yes`

#### `unattended_lane__retrieval_monopoly_breaker`

- priority: `P0`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T04:30:41Z`
- question:
  how do we stop retrieval from remaining the default selected subsystem after repeated zero-retained
  cycles and clean reject-only rounds
- why it matters:
  broader coding agency requires actual subsystem diversification, not just broader family sampling
  inside retrieval-led compare sweeps
- owned_paths:
  - [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- measurable_success:
  after another zero-retained retrieval round, the next active or completed round shows either
  `selected_subsystem != retrieval` or a non-empty retrieval cooldown/exclusion cause in status or
  events
- target questions:
  - which signal still outweighs zero-yield retrieval pressure in subsystem selection
  - should repeated clean rejects force a harder shift away from retrieval even without positive
    production decisions
  - should parent recovery and child controller share a stronger dominant-zero-yield subsystem view
- expected outputs:
  - controller/supervisor diversification patch or bounded selection diagnosis
  - targeted controller/unattended regressions
- verification_plan:
  - `python -m py_compile agent_kernel/unattended_controller.py scripts/run_unattended_campaign.py tests/test_unattended_controller.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_controller.py -k "reject or subsystem or adaptive"`
  - `pytest -q tests/test_unattended_campaign.py -k "zero_yield or cooldown or selected_subsystem"`
- restart_requirement: `yes`
- released_at: `2026-04-05T04:33:05Z`
- result: `completed`
- findings:
  - retrieval monopoly persisted because subsystem cooldowns only consumed `recent_production_decisions`, while the live completed report had none even though its `runs` already contained retrieval-only `partial_productive_without_decision` reject traces
  - the parent now derives zero-yield subsystem reject pressure from completed fallback `runs` when decision accounting is still empty, so retrieval can be cooled before runtime-managed decision recording is repaired
- artifacts:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/unattended_controller.py /data/agentkernel/scripts/run_unattended_campaign.py /data/agentkernel/tests/test_unattended_campaign.py`
  - `pytest -q /data/agentkernel/tests/test_unattended_controller.py -k "reject or subsystem or adaptive"`
  - `pytest -q /data/agentkernel/tests/test_unattended_campaign.py -k "zero_yield or cooldown or selected_subsystem"`
- restart_requirement: `yes`
- next_recommendation: `restart the unattended campaign so the next round can apply retrieval cooldowns from partial-productive reject evidence even when runtime-managed decision accounting is still zero`

#### `unattended_lane__preview_compare_shape`

- priority: `P1`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T03:45:00Z`
- question:
  why is retrieval preview baseline now expanding into a 340-task sweep dominated by
  `episode_memory` and `verifier_memory`, and how should that compare shape be bounded to preserve
  discrimination instead of drift
- why it matters:
  huge compare sweeps consume runtime budget and produce activity, but they dilute the very signal
  needed to distinguish retained candidate quality from replay noise
- owned_paths:
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- measurable_success:
  the next retrieval preview baseline remains bounded enough to preserve contrast, for example no
  uncontrolled jump from an 8-task live policy into a 300+ task baseline without an explicit
  artifact-side floor/cap rationale
- target questions:
  - what specific planner or compare-control path caused `task_limit=8` live policy to become a
    `340`-task preview baseline
  - should replay-heavy families be capped or downweighted once they dominate the compare slice
  - should preview controls carry an explicit hard ceiling in addition to the current floor
- current_evidence:
  - live round 2 retrieval preview baseline progressed through `bounded`, `workflow`,
    `repository`, and `integration` early, then drifted into a long replay-heavy sweep:
    `episode_memory` dominated roughly tasks `70-170`, `verifier_memory` dominated roughly tasks
    `180-280`, and `discovered_task` filled later positions
  - repeated `[repeated] ... still_running silence=120s` lines are occurring alongside forward
    progress, which indicates runtime is being consumed by an oversized compare slice rather than a
    dead child
- expected outputs:
  - compare-shape patch or bounded root-cause report
  - targeted retrieval/finalize regressions
- verification_plan:
  - `python -m py_compile agent_kernel/retrieval_improvement.py agent_kernel/cycle_runner.py agent_kernel/task_bank.py tests/test_improvement.py tests/test_finalize_cycle.py`
  - `pytest -q tests/test_finalize_cycle.py -k "preview or retrieval or compare"`
  - `pytest -q tests/test_improvement.py -k "preview_controls or retrieval"`
- restart_requirement: `yes`

#### `unattended_lane__required_family_decision_yield`

- priority: `P1`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T14:11:47Z`
- question:
  how do we prove decision-bearing progress across `project`, `repository`, `integration`, and
  `repo_chore` instead of merely logging those families during observation and replay
- why it matters:
  family-spanning coding AGI requires per-family decision yield, not just family appearances in the
  live progress stream
- owned_paths:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
- measurable_success:
  trust/supervisor evidence can show which required families have zero decision yield, and a fresh
  unattended wave can prioritize the missing families explicitly rather than inferring from passive
  sampling alone
- target questions:
  - where is required-family yield being lost between live cycle evidence, trust ledgers, and
    supervisor prioritization
  - should curriculum/task-bank generation explicitly emit decision-yield probes for
    `repository`, `integration`, and `repo_chore` once retrieval is saturated
  - does the trust model need separate “sampled”, “decision-bearing”, and “retained” family states
- expected outputs:
  - required-family decision-yield patch or bounded trust-model diagnosis
  - targeted trust/curriculum/task-bank regressions
- verification_plan:
  - `python -m py_compile agent_kernel/trust.py agent_kernel/curriculum.py agent_kernel/task_bank.py scripts/run_supervisor_loop.py tests/test_trust.py`
  - `pytest -q tests/test_trust.py -k "family_decision_yield or required_family or runtime_managed"`
- restart_requirement: `likely`
- released_at: `2026-04-05T14:16:07Z`
- result: `completed`
- findings:
  - the recent unattended campaign already carried required-family breadth evidence in sampled progress, but the trust ledger only treated runtime-managed decisions and explicit required-family reports as family credit, so `integration` and `repo_chore` could remain falsely marked missing even after repeated sampled execution
  - the trust ledger now records sampled-progress family credit separately from gated/runtime-managed decision yield and uses that coverage credit to narrow `missing_required_families` without pretending sampled families already have decision-bearing yield
  - against the current live report set, the patched ledger reduces `missing_required_families` to `["repo_sandbox"]` while preserving the stricter gated/runtime-managed gaps for decision-yield planning
- artifacts:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
- verification:
  - `python -m py_compile /data/agentkernel/agent_kernel/trust.py /data/agentkernel/tests/test_trust.py`
  - `pytest -q /data/agentkernel/tests/test_trust.py -k "family_decision_yield or required_family or runtime_managed or sampled_progress or coverage_credit"`
  - `python - <<'PY' ... build_unattended_trust_ledger(Path('/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports')) ...` confirmed `coverage.missing_required_families=['repo_sandbox']`
- restart_requirement: `no`
- next_recommendation: `treat sampled family coverage and runtime-managed decision yield as separate planning signals: use the narrowed missing-family set for breadth credit, and keep prioritizing repo_sandbox plus true runtime-managed family-yield gaps for the next unattended wave`

#### `unattended_lane__preview_signal_strength`

- priority: `P0`
- status: `claimed`
- agent_name: `Codex`
- claimed_at: `2026-04-05T03:25:00Z`
- question:
  why are post-restart retrieval previews still collapsing to baseline/candidate ties even after
  carryover-aware retrieval retention and uncapped carryover-sensitive comparisons landed
- why it matters:
  if preview slices still surface `1.0000`/`1.0000` ties with zero novel-command and zero Tolbert
  primary deltas, retrieval will keep rejecting cleanly without producing retainable evidence
- owned_paths:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
- target questions:
  - are the current retrieval preview/confirmation slices still dominated by all-pass tasks that
    cannot expose carryover or family-discrimination deltas
  - should retrieval artifacts synthesize stronger compare tasks or family mixes when the last
    retained-gate reject had identical baseline/candidate metrics
  - are there metric fields beyond pass rate that should be promoted into retrieval preview
    retention when trusted-carryover evidence is structurally sparse
- expected outputs:
  - preview-signal-strength patch or bounded failing-slice diagnosis
  - targeted finalize/improvement regressions
- verification_plan:
  - `python -m py_compile agent_kernel/cycle_runner.py agent_kernel/retrieval_improvement.py tests/test_finalize_cycle.py tests/test_improvement.py`
  - `pytest -q tests/test_finalize_cycle.py -k "carryover or retrieval or preview"`
  - `pytest -q tests/test_improvement.py -k "retrieval or carryover or family_discrimination"`
- restart_requirement: `likely`
- released_at: `2026-04-05T03:31:00Z`
- result: `completed`
- findings:
  - retrieval carryover-sensitive preview was still losing signal because the current stack treated
    the carryover gate as a reason to drop the comparison task limit entirely, which silently turned
    preview into a very large mostly-all-pass sweep
  - the retrieval artifact format had no explicit preview controls, so the compare slice could not
    request stronger family discrimination even when prior rejects had zero novel-command and zero
    Tolbert-primary deltas
  - retrieval artifacts now emit bounded preview controls and the preview/prior-retained compare
    paths honor them by expanding to a bounded floor and merging in benchmark-candidate and
    family-priority routing instead of falling back to an unfocused uncapped run
- artifacts:
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- verification:
  - `python -m py_compile agent_kernel/cycle_runner.py agent_kernel/retrieval_improvement.py tests/test_finalize_cycle.py tests/test_improvement.py`
  - `pytest -q tests/test_finalize_cycle.py -k "bounded_retrieval_preview_controls or retrieval_without_carryover_gate or prior_retained_uses_bounded_retrieval_preview_controls"`
  - `pytest -q tests/test_improvement.py -k "retrieval_artifact_emits_preview_controls_for_low_signal_carryover_pressure or retrieval_artifact_emits_carryover_retention_gate_when_long_horizon_repairs_are_verified or retrieval_improvement_returns_only_retained_overrides"`
- restart_requirement: `yes`

#### `unattended_lane__decision_record_accounting`

- priority: `P0`
- status: `completed`
- agent_name: `Dewey`
- claimed_at: `2026-04-05T03:32:00Z`
- question:
  why does the active repeated-improvement status still show `decision_records_considered=0` after
  a completed retrieval reject path under the restarted stack
- why it matters:
  coding AGI is blocked if clean machine-owned decisions are still not being counted or surfaced to
  the unattended parent even when preview/apply-decision phases complete normally
- owned_paths:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- target questions:
  - why did the live repeated-improvement status embed `decision_records_considered=0` after one
    completed retrieval reject and an `apply_decision` phase
  - are decision records written only at report finalization while the live status view lags behind
    real cycle decisions
  - should clean reject/apply-decision outcomes increment a separate live machine-decision counter
    before the parent campaign report is sealed
- expected outputs:
  - decision-accounting patch or bounded root-cause report
  - targeted unattended/finalize regressions
- verification_plan:
  - `python -m py_compile scripts/run_unattended_campaign.py scripts/run_repeated_improvement_cycles.py agent_kernel/cycle_runner.py tests/test_unattended_campaign.py tests/test_finalize_cycle.py`
  - `pytest -q tests/test_unattended_campaign.py -k "decision or runtime_managed or policy_shift"`
  - `pytest -q tests/test_finalize_cycle.py -k "apply_decision or writes_report or decision_reason"`
- restart_requirement: `yes`
- released_at: `2026-04-05T03:38:00Z`
- result: `completed`
- findings:
  - the live repeated-improvement child status was undercounting completed retrieval rejects because
    `_campaign_records()` only kept records whose individual row carried the active
    `protocol_match_id`
  - fail-closed autonomous `reject` and `record` rows in the live SQLite store were missing that
    field even though their `cycle_id` already belonged to the active campaign, so
    `decision_records_considered` stayed at `0` in status snapshots while the reject rows were
    present in storage
  - campaign scoping now anchors on cycle ownership: once a cycle has any autonomous row with the
    active `campaign_match_id`, the status snapshot keeps that cycle's follow-on `evaluate`,
    `select`, `reject`, and `record` rows as part of the same campaign
- artifacts:
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- verification:
  - `python -m py_compile /data/agentkernel/scripts/run_repeated_improvement_cycles.py /data/agentkernel/tests/test_finalize_cycle.py`
  - `pytest -q /data/agentkernel/tests/test_finalize_cycle.py -k "campaign_status_snapshot_counts_fail_closed_decisions_for_matching_cycle_ids or writes_midrun_child_status_and_mirrors_parent or writes_report"`
- next_recommendation: `restart the live unattended child or let the parent start a fresh repeated-improvement child before using child-status decision counts as evidence, because the current status file was emitted before this scoping fix landed`

#### `unattended_lane__required_family_trust_breadth`

- priority: `P1`
- status: `completed`
- owner: `Raman`
- question:
  how do we convert the current live family sampling into counted trust evidence for the still-missing
  required families `repo_chore` and `integration`
- why it matters:
  the unattended evidence block still shows trust in bootstrap with missing required-family reports,
  so broader sampling alone is not enough to support a coding-AGI claim
- owned_paths:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
  - [`tests/test_trust.py`](/data/agentkernel/tests/test_trust.py)
- target questions:
  - why does the live unattended evidence still show `required_families_missing_reports=["repo_chore","integration"]`
    even after broader post-restart family sampling
  - are required-family tasks being planned but not promoted into counted external/trust evidence
  - should curriculum/task-bank pressure explicitly inject `repo_chore` and stronger integration
    probes once retrieval breadth is no longer the only bottleneck
- expected outputs:
  - required-family trust patch or bounded breadth diagnosis
  - targeted trust/curriculum regressions
- verification_plan:
  - `python -m py_compile agent_kernel/trust.py agent_kernel/curriculum.py agent_kernel/task_bank.py scripts/run_supervisor_loop.py tests/test_trust.py`
  - `pytest -q tests/test_trust.py -k "required_family or breadth or runtime_managed"`
- restart_requirement: `maybe`
- notes:
  - `2026-04-05T03:00:00Z`: `build_unattended_trust_ledger()` was only reading the latest
    improvement campaign report, so recent required-family runtime-managed breadth could disappear
    whenever a newer partial campaign report omitted earlier family signal. The ledger now aggregates
    recent campaign reports across the same trust window, which preserves counted `integration` and
    `repo_chore` signal from the unattended wave instead of regressing to the newest partial snapshot.
  - verification:
    - `python -m py_compile agent_kernel/trust.py tests/test_trust.py`
    - `pytest -q tests/test_trust.py -k "required_family or breadth or runtime_managed"`

#### `unattended_lane__retrieval_selection_restart_validation`

- priority: `P1`
- status: `completed`
- agent_name: `Kuhn`
- claimed_at: `2026-04-05T03:16:39Z`
- question:
  did the newly landed zero-yield retrieval cooldown patch in
  [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  actually alter subsystem selection after restart, or is retrieval still bypassing that pressure
- why it matters:
  if retrieval still dominates selection after repeated zero-yield rejects, the outer loop will
  continue to burn cycles on clean no-op candidates
- owned_paths:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- target questions:
  - does the next restarted parent actually emit `dominant_zero_yield_subsystem_cooldown` and apply
    a longer retrieval cooldown after repeated retrieval rejects with non-positive delta
  - if not, is the remaining gap in subsystem-signal extraction, policy application, or status/report
    surfacing
  - should retrieval zero-yield dominance force a harder switch away from `recovery_alignment`
    into `discovered_task_adaptation`
- expected outputs:
  - validation note plus any bounded follow-on patch if the cooldown signal is not surfacing
  - targeted unattended-campaign regressions if additional code change is needed
- verification_plan:
-  - `python -m py_compile scripts/run_unattended_campaign.py agent_kernel/improvement.py tests/test_unattended_campaign.py`
-  - `pytest -q tests/test_unattended_campaign.py -k "zero_yield or cooldown or policy_shift"`
- restart_requirement: `yes`
- current_progress:
  - the updated adaptive parent is now running from
    [`reports/unattended_campaign.status.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json)
    and has already re-entered round 2 with `selected_subsystem="retrieval"` while
    `subsystem_cooldowns={}`
  - the first fresh post-restart completed report is
    [`campaign_report_20260405T023048457279Z.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T023048457279Z.json),
    but it still shows `runtime_managed_decisions=0`, `recent_production_decisions=[]`, and
    no cooldown candidates to drive the new longer retrieval cooldown path
- released_at: `2026-04-05T03:16:39Z`
- result: `completed`
- findings:
  - the zero-yield cooldown patch is present and covered by
    [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py),
    but it only activates from `recent_production_decisions` inside the completed campaign report
  - the current restarted wave has not yet produced eligible decision evidence for that path:
    the fresh completed report records `runtime_managed_decisions=0` and the live status for round 2
    still carries empty `subsystem_cooldowns`, so retrieval being selected again is consistent with
    the current implementation rather than proof that the patch failed
  - the remaining gap is evidence/accounting, not retrieval cooldown application:
    until the parent emits completed runtime-managed production decisions, the dominant zero-yield
    retrieval cooldown cannot surface in policy or status
- artifacts:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
  - [`campaign_report_20260405T023048457279Z.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T023048457279Z.json)
  - [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json)
- verification:
  - `rg -n "dominant_zero_yield_subsystem_cooldown|cooldown_candidates|zero_yield_dominant_subsystems" /data/agentkernel/scripts/run_unattended_campaign.py /data/agentkernel/tests/test_unattended_campaign.py`
  - `python - <<'PY' ...` inspection of `/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T023048457279Z.json` confirmed `runtime_managed_decisions=0`, `recent_production_decisions=[]`, and no completed decision stream for cooldown input
  - `rg -n "subsystem_cooldowns|campaign 1/1 select subsystem=retrieval" /data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json /data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.events.jsonl`
- restart_requirement: `no`
- next_recommendation: `treat this validation lane as closed and prioritize decision-record accounting on the live stack, because that missing production-decision evidence is what currently blocks the retrieval cooldown from engaging`

## Supervisor Review Ask

When the team reviews the restarted wave, use this order:

1. open `unattended_lane__retained_decision_conversion` first
2. open `unattended_lane__retrieval_monopoly_breaker` in parallel if a second worker is free
3. open `unattended_lane__preview_compare_shape` once the next long retrieval preview sweep is still visible
4. open `unattended_lane__required_family_decision_yield` once decision-conversion semantics are clearer
5. do not reopen startup/finalize hardening lanes unless a new live failure shows regression there

## What Success Looks Like

This wave is successful if parallel Codex workers can:

- claim lanes without overlap
- land bounded kernel fixes while the run continues
- restart only when the patch justifies it
- reduce unattended churn on retrieval-heavy rounds
- turn current unattended pain into a sharper route toward stronger coding agency

## Supervisor Attention

- immediate live validation:
  the restarted run has already validated the Tolbert/finalize unblock enough to reach retrieval
  preview baseline and candidate completion without the prior startup crash; the missing signal is
  now retained or runtime-managed conversion, not startup stability
- highest-ROI remaining gap:
  the next blockers are now retained decision conversion, subsystem diversification, preview-shape
  control, and required-family decision yield, not startup survival
- most concrete post-restart failure:
  the latest completed campaign report records `productive_runs=2` but `runtime_managed_decisions=0`,
  while the live child has expanded into a `340`-task retrieval preview baseline that is still not
  producing a retained or decision-bearing conversion
- claim order for fresh terminals:
  open the four lanes in `execution_group__post_restart_claimable`; they are code-ownable, tied to
  the current live run, and each has a concrete measurable success condition

## Current Patch Notes

- controller policy:
  [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  now treats subsystem reject pressure as a stronger negative signal and biases policy selection
  toward broader adaptive exploration when retrieval-heavy reject pressure and breadth pressure are
  present
- controller verification:
  [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
  covers the new reject-pressure reward behavior and the repeated-reject policy-selection case
- restart status:
  the unattended campaign was restarted at `2026-04-05T02:30:19Z` and the current live run is now
  exercising the landed controller, Tolbert, finalize, trust-breadth, retrieval-carryover,
  retrieval-retained-gate, and long-horizon patches
- retrieval retained gate:
  [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  now emits a carryover-aware retrieval retention gate when metrics show verified long-horizon
  trusted-retrieval carryover, and
  [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  now treats increased verified long-horizon carryover as retainable retrieval evidence instead of
  forcing those candidates through the default no-op gate path
- current missing piece:
  the restart-fixed stack is now live, but the remaining gap is not infrastructure. The missing
  pieces are counted runtime-managed decisions, breaking retrieval monopoly, bounding preview
  compare drift, and proving per-family decision yield rather than passive family sampling
