# Unattended Parallel Codex Runbook

## Purpose

This file defines how one Codex coordinator and multiple replica workers can support a live
unattended campaign in parallel without corrupting the run, duplicating work, or hiding evidence in
chat-only handoffs.

Use it when the unattended campaign is already running and the repo needs live operator-side kernel
work such as timeout-policy repair, TOLBERT startup hardening, report validation fixes, or
controller diagnostics.

This runbook is for outer-loop support work around the unattended campaign. It is not the inner
task policy, and it is not the supervised frontier runbook.

## Use With

- [`docs/unattended_controller.md`](/data/agentkernel/docs/unattended_controller.md)
- [`docs/scripts.md`](/data/agentkernel/docs/scripts.md)
- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- [`scripts/run_detached_unattended_campaign.py`](/data/agentkernel/scripts/run_detached_unattended_campaign.py)
- [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)

## Operating Model

One coordinator owns the live unattended run.

Replica workers do not steer the run directly. They:

1. inspect authoritative live artifacts
2. claim one bounded support lane
3. patch code or diagnostics inside that lane
4. verify locally
5. hand results back through repo artifacts, not only prose

The coordinator decides whether to:

- let the live run continue
- restart the run under patched code
- widen or narrow the next round policy
- merge multiple worker patches into one restart window

## Fresh Client Bootstrap

A fresh Codex client should be able to begin from docs alone.

Startup sequence:

1. read [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
2. read the active run artifacts listed there:
   `unattended_campaign.status.json`, `unattended_campaign.events.jsonl`,
   `unattended_campaign.stdout.log`
3. choose exactly one lane still marked `unclaimed`
4. append a claim block to the work queue before editing code
5. stay inside the lane's owned paths
6. verify locally
7. append a closeout block with restart requirement

If a fresh client cannot determine the current run root, active round, owned paths, or claim state
from repo docs and live artifacts, the coordination docs are incomplete and should be fixed before
more kernel work begins.

## Current Wave Notes

During the live wave rooted at
[`var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive):

- the Tolbert runtime lane landed a restart-required patch to reduce cold-start failures during
  preview/finalize startup in
  [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
- the controller-policy lane landed a restart-required patch in
  [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  so repeated reject-only retrieval pressure counts more strongly against narrow policies and pushes
  the planner toward broader adaptive exploration sooner
- fresh workers should read
  [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
  before claiming work so they do not duplicate already-landed controller or Tolbert changes

## Authority Order

Read these in order before claiming work:

1. detached run status:
   the active run root's `reports/unattended_campaign.status.json`
2. event log:
   the active run root's `reports/unattended_campaign.events.jsonl`
3. stdout log:
   the active run root's `reports/unattended_campaign.stdout.log`
4. latest campaign report under:
   the active run root's `improvement_reports/`
5. supervisor entrypoint:
   [`run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)

If those disagree, trust them in that order.

Active example run root while this document was written:

- [`var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive)

## Coordinator Contract

The coordinator must:

1. own the live run root and detached parent process
2. decide whether a change is safe to defer until the next restart
3. assign exactly one bounded lane per replica
4. prevent overlapping edits on shared unattended entrypoints unless the replica is explicitly
   patching that file
5. summarize current runtime state before asking replicas to work

Coordinator-owned surfaces by default:

- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- [`scripts/run_detached_unattended_campaign.py`](/data/agentkernel/scripts/run_detached_unattended_campaign.py)
- live run root under
  [`var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive)

## Replica Contract

Each replica must:

1. claim exactly one lane
2. stay inside owned files for that lane
3. use the current run artifacts as evidence, not stale memory
4. avoid mutating the live run root except for explicitly allowed diagnostic artifacts
5. deliver a patch plus verification, or a bounded root-cause report plus next action

Replica workers are support engineers for the live unattended loop. They are not independent outer
controllers.

## Lane Split

Use these lanes as the default parallel split:

1. `runtime_supervisor`
   - files:
     [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py),
     [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
   - scope:
     timeout grace, child validation, recovery policy, phase transitions, status/event emission
2. `controller_policy`
   - files:
     [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py),
     [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
   - scope:
     next-round policy selection, support-aware scoring, stop budgets, search adaptation
3. `tolbert_runtime`
   - files:
     [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py),
     [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py),
     [`tests/test_tolbert.py`](/data/agentkernel/tests/test_tolbert.py)
   - scope:
     startup readiness, retry, timeouts, process reuse, service diagnostics
4. `cycle_finalize`
   - files:
     [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py),
     [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
   - scope:
     preview/finalize fallback, candidate retention gates, per-phase recovery behavior
5. `run_observability`
   - files:
     [`scripts/report_unattended_run_metrics.py`](/data/agentkernel/scripts/report_unattended_run_metrics.py),
     [`docs/unattended_controller.md`](/data/agentkernel/docs/unattended_controller.md),
     new report helpers under [`scripts/`](/data/agentkernel/scripts)
   - scope:
     status summaries, report extraction, drift dashboards, operator diagnostics

## Claim Format

Record claims in one bounded block before editing anything. A claim should include:

- `worker_name`
- `lane_id`
- `claimed_at`
- `run_root`
- `current_round`
- `question`
- `owned_paths`
- `verification_plan`

If the repo later gets a dedicated unattended work queue, this same claim block should move there
without changing the contract.

## Live-Run Safe Actions

Safe while the unattended run is active:

- reading status, event, and stdout artifacts
- adding tests for already observed failures
- patching code that will only matter on the next restart
- writing new report or diagnostic scripts
- preparing restart commands

Unsafe while the unattended run is active unless the coordinator explicitly decides to restart:

- deleting or rewriting live status or event files
- changing the detached process environment in place
- killing the live child or parent without coordination
- mutating shared entrypoints from multiple workers at once
- editing retained artifacts under the active run root

## Restart Discipline

A live unattended run only picks up code changes after restart.

Before restart, the coordinator should ensure:

1. the patch is merged locally
2. targeted tests passed
3. the reason for restart is written in the handoff note
4. the exact restart command is preserved
5. the previous run's terminal status is captured from status and event artifacts

## Communication Contract

Parallel workers should communicate through repo artifacts first.

Required shared outputs:

- code patch in owned files
- targeted test coverage where practical
- one short handoff note containing:
  - observed failure
  - root cause
  - patch summary
  - verification summary
  - restart requirement: `yes` or `no`

Preferred evidence sources:

- status JSON snapshots
- filtered event-log excerpts
- exact file and line references
- deterministic test commands

Avoid using chat-only memory as the system of record.

The minimum handoff package to another fresh Codex client is:

1. [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
2. [`docs/unattended_parallel_codex_runbook.md`](/data/agentkernel/docs/unattended_parallel_codex_runbook.md)
3. the active run root referenced by the queue

That package should be sufficient for a new client to claim work and begin development without
extra operator reconstruction in chat.

## Gap-Filling Priorities

When the unattended campaign is running, the highest-value support work is usually:

1. fix supervisor false negatives:
   productive children killed as stalled or timed out
2. fix supervisor false positives:
   stuck children treated as healthy
3. harden child report validation:
   clean child exit with bad or empty report should not look successful
4. harden TOLBERT startup and fallback behavior
5. improve diagnostics so the next failure is machine-obvious instead of prose-only

Prefer kernel gaps that improve the next unattended round immediately over speculative redesign.

## Anti-Failure Rules

- do not let two replicas edit [`run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py) at the same time
- do not infer success from child exit code alone
- do not treat `campaign_complete` as proof the child succeeded
- do not cool down the only active subsystem without checking remaining portfolio coverage
- do not spend runtime grace without tying it to fresh progress
- do not rely on stdout alone when status or events disagree
- do not hide restart-critical reasoning only in chat

## Expected Outcome

If this runbook is followed, the unattended campaign can keep running while parallel Codex workers:

- diagnose failures from shared evidence
- patch bounded kernel gaps in parallel
- avoid overlapping edits
- hand fixes back to one coordinator
- restart only when the patch is verified and worth the interruption
