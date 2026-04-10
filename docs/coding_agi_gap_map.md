# Coding AGI Gap Map

## Purpose

This is the compact map from the current unattended evidence to the highest-value remaining gaps on
the path to compounding coding agency.

It is anchored to the latest integrated unattended completion on `2026-04-09` from
[`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.status.json)
and the paired campaign report
[`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.json).

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
- the integrated report now shows `runtime_managed_decisions=1` and `retained_gain_runs=1`
- the credited retain is still described as
  `mid-round controller intervention closed as a runtime-managed retain after productive work was already demonstrated`
  rather than a clean child-native closeout stream
- sampled families from progress now include
  `["integration","project","repository","bounded","transition_pressure","discovered_task"]`
- the trust breadth summary still shows `required_families_with_reports=[]`,
  `distinct_external_benchmark_families=0`, and no counted external report breadth
- campaign validation still accepted a productive partial timeout rather than a completed child
  report path: `accepted productive partial child timeout: generated_success_completed_without_report_path`

Primary artifacts:

- [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.status.json)
- [`unattended_campaign.events.jsonl`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.events.jsonl)
- [`unattended_campaign.stdout.log`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.stdout.log)
- [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.json)

## Priority Gaps

### 1. Runtime-managed conversion gap

The kernel can complete unattended rounds and now produce at least one runtime-managed retain, but
it still fails to turn those rounds into a child-native retained or rejected decision stream at a
useful rate.

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

- the current integrated run did clear `runtime_managed_decisions=1`, so the zero-decision failure
  is no longer the immediate blocker
- the credited retain still came from a productive partial timeout that the controller accepted
  mid-round, which means decision ownership is improved but not yet robustly child-native

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
- the trust ledger still does not show counted external reports for those required families
- sampled family breadth is improving faster than counted trusted breadth

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
  the abstract, but weak conversion from those capabilities into broad, runtime-managed coding gains
- the next highest-ROI work is decision yield, family breadth, retrieval carryover, and long-horizon
  repo execution quality, in that order
