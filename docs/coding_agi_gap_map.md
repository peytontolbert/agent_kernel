# Coding AGI Gap Map

## Purpose

This is the compact map from the current unattended evidence to the highest-value remaining gaps on
the path to compounding coding agency.

It is anchored to the live unattended safe-stop on `2026-04-05T00:46:40Z` from
[`unattended_campaign.status.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json)
and the paired campaign report
[`campaign_report_20260405T002026515795Z.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T002026515795Z.json).

## Current Read

The current kernel is already beyond a toy seed-task loop. It has:

- a verifier-driven task runtime in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py)
- an unattended outer loop in [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- retained-vs-candidate improvement flow in [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- trust and family-breadth accounting in [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
- bounded structured editing plus syntax-motor support in
  [`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py)
  and [`agent_kernel/syntax_motor.py`](/data/agentkernel/agent_kernel/syntax_motor.py)

The blocker is not "the runtime cannot do coding at all." The blocker is that the system still does
not convert enough unattended work into broad, runtime-managed, repeatable coding gains.

## Evidence Snapshot

- live unattended parent reached `safe_stop` because `campaign report showed no runtime-managed decisions`
- round 2 stayed inside `retrieval`, finished cleanly, and still ended in `state=reject`
- the live campaign report shows `runtime_managed_decisions=0`
- the same report shows `priority_families_without_signal=["project","repository","integration"]`
- required-family coverage is still missing `integration` and `repo_chore`
- required-family clean task-root breadth is still zero for `integration`, `project`, `repo_chore`,
  `repo_sandbox`, and `repository`
- the event and stdout logs show repeated retrieval proposals with
  `reason=low-confidence retrieval remains common relative to trusted retrieval usage`
  and `final_reason=retrieval candidate did not satisfy the retained retrieval gate`

Primary artifacts:

- [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.status.json)
- [`unattended_campaign.events.jsonl`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.events.jsonl)
- [`unattended_campaign.stdout.log`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/reports/unattended_campaign.stdout.log)
- [`campaign_report_20260405T002026515795Z.json`](/data/agentkernel/var/unattended_repro_autonomyfull_20260404T1901Z_adaptive/improvement_reports/campaign_report_20260405T002026515795Z.json)

## Priority Gaps

### 1. Runtime-managed conversion gap

The kernel can complete unattended rounds and produce productive compare activity, but it still
fails to turn those rounds into runtime-managed retained or rejected decisions at a useful rate.

Why this is AGI-critical:

- compounding coding agency requires machine-owned decisions, not only machine activity
- a loop that generates artifacts but rarely crosses the runtime-managed decision boundary will keep
  spending compute without closing control

Primary surfaces:

- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
- [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)

What the live evidence says:

- the current safe-stop reason is directly tied to `runtime_managed_decisions=0`
- the system is willing to keep exploring retrieval variants, but the compare/finalize path still
  does not yield a runtime-managed result stream that the unattended parent considers meaningful

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

- required-family coverage is still missing `integration` and `repo_chore`
- `priority_families_without_signal` still includes every currently requested family
- the campaign sampled priority families, but sampled work did not turn into counted decision signal

Next attack:

- spend the next wave on counted proof for `integration`, `repository`, and `project` under
  contract-clean or light-supervision-clean conditions instead of saturating retrieval-only probes

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

- repeated unattended rounds kept emitting retrieval proposals with the same low-confidence reason
- retrieval candidates kept failing the retained retrieval gate
- the retrieval subsystem is acting more like a churn attractor than a compounding memory amplifier
  in the current unattended slice

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
  the abstract, but weak conversion from those capabilities into broad, runtime-managed coding gains
- the next highest-ROI work is decision yield, family breadth, retrieval carryover, and long-horizon
  repo execution quality, in that order
