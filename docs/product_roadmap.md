# Product Roadmap

## Purpose

This roadmap is separate from [`asi.md`](/data/agentkernel/asi.md).

`asi.md` is the recursive-improvement and superintelligent-machine substrate contract. This document is a
product benchmark roadmap for two product questions:

- can `agent-kernel` reliably do useful local software tasks on your behalf while you are not watching?
- can the product climb from a `vllm`-led runtime into a retained TOLBERT-family runtime that eventually beats the initial `vllm+TOLBERT` product stack?

That benchmark is worth pursuing in parallel because it is:

- easier to measure than broad ASI language
- directly useful as a product outcome
- a practical forcing function for safety, verification, recovery, and operator trust
- the cleanest path to a governed TOLBERT liftoff product rather than a manual model-swap project

## Product Benchmark

The benchmark is not "general autonomy" by declaration.

The product benchmark is staged:

1. a user can hand the kernel a bounded local task
2. the kernel can run it end to end in the background
3. the kernel either finishes correctly or stops safely
4. the kernel leaves an auditable record of what it tried, why it stopped, and what changed
5. retained TOLBERT-family checkpoints can improve the runtime under the same verifier, trust, recovery, and retention system
6. a retained TOLBERT-family runtime can take over approved task families only after it beats the initial `vllm+TOLBERT` product

For product purposes, "does things on your behalf" means all of the following hold together:

- delegated execution without live supervision
- isolation strong enough that failure is containable
- verification strong enough that "passed" means useful, not merely format-matched
- recovery strong enough that rejected or failed attempts leave the system in a sane state
- reporting clear enough that a user can trust the result after the fact
- model-comparison discipline strong enough that a learned runtime only gains authority by beating the retained baseline

## Non-Goals

This roadmap does not by itself claim:

- AGI
- ASI
- unrestricted computer use
- safe operation on arbitrary machines with arbitrary privileges
- self-improvement competence measured against any external manual-steering baseline
- that a checkpoint becomes trusted merely because it trains successfully

It is a product benchmark that should tighten the runtime while the broader substrate work continues.

## Runtime Roadmap

The product should be understood as an authority-transfer roadmap:

- Seed product:
  - `vllm` is the primary actor
  - the live TOLBERT service is encoder, retrieval, path prediction, and typed-context support
- Takeover product:
  - retained TOLBERT-family checkpoints compete with the initial `vllm+TOLBERT` stack
  - family-by-family routing can shift toward TOLBERT when it wins
- Liftoff product:
  - TOLBERT-family checkpoints become the primary runtime for approved families
  - `vllm` remains fallback, arbitration, teacher generation, and exploration support

This mirrors [`docs/runtime.md`](/data/agentkernel/docs/runtime.md), [`docs/improvement.md`](/data/agentkernel/docs/improvement.md), and [`docs/tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md).

## Roadmap Shape

Three tracks should progress in parallel:

1. `ASI substrate track`
   - recursive improvement, retained artifacts, broader eval families, and long-run compounding
2. `delegation product track`
   - unattended execution, trust, isolation, semantic verification, and operator experience
3. `TOLBERT liftoff track`
   - checkpoint training, checkpoint comparison, learned transition/value/policy surfaces, and governed authority transfer

Success on the product track does not prove the ASI track.
Success on the ASI track does not automatically produce a trustworthy product.
Success on the liftoff track does not excuse regressions in verifier, trust, recovery, or operator clarity.

The goal is to make them reinforce each other without collapsing them into one document.

## Completeness Verdict

As of the current repo, the product track is materially further along than the first draft of this roadmap
claimed, but it is still not complete.

- supervised bounded reliability: mostly complete
- unattended single-task delegation: mostly implemented in substrate
- unattended repo workflows: materially implemented in substrate
- background work queue: materially implemented in substrate
- trusted personal kernel: partially implemented in substrate, not complete in evidence

The short version is:

- Phase 0 is largely in place
- Phase 1 is structurally strong but still open on OS-level containment and empirical walk-away proof
- Phases 2 and 3 are no longer just commitments; they now exist as real runtime behavior
- Phase 4 has real trust, policy, and audit infrastructure, but still needs repeated evidence and broader product polish
- the TOLBERT liftoff seed is now present, but learned policy/value/transition takeover is still earlier than the delegation product track

So the honest product status is:

- harness-complete: yes, with known containment limits
- trustworthy delegation product-complete: no
- phase-complete beyond Phase 0: partially yes in code, not yet yes in proof
- TOLBERT liftoff-complete: no

If this roadmap is weighted by implemented engine behavior rather than the age of its text, the repo is
closer to "Phase 1 mostly implemented, Phases 2 and 3 materially implemented, Phase 4 started" than to
"Phase 0 complete and Phase 1 started."

The correct model-runtime verdict is:

- seed `vllm+TOLBERT` product: real
- retained TOLBERT checkpoint loop: now real
- TOLBERT family takeover product: not complete
- universal encoder-latent-decoder TOLBERT runtime: not complete

## Status Matrix

### Phase 0: Supervised Bounded Reliability

Status: mostly complete

What exists in the repo now:

- deterministic bounded tasks with replayable evaluation
- stored episodes with steps, verification results, and termination reasons
- typed failure signals for policy/runtime failures
- retained-artifact evaluation infrastructure that already improves measurement discipline

What still weakens the phase:

- verification is mixed rather than uniformly semantic
- execution containment is still weak enough that some failures remain ambiguous

### Phase 1: Unattended Single-Task Delegation

Status: mostly implemented in substrate

What exists in the repo now:

- explicit task contracts, workspaces, and success constraints for bounded tasks
- unattended preflight checks for provider health, TOLBERT assets, verifier inputs, and workspace readiness
- a first unattended run outcome taxonomy with `success`, `safe_stop`, and `unsafe_ambiguous`
- a first unattended task report that records commands, verifier state, before/after workspace snapshots,
  changed-file hashes, contract-accounted side effects, and final workspace files
- rollback-aware workspace recovery for non-success outcomes
- bounded `http_request` support alongside bounded workspace execution
- operator policy and capability policy gates for unattended execution

What is still missing before Phase 1 exit:

- OS-level execution isolation beyond segmented `shell=False` execution, bounded host executable allowlists,
  workspace-local executables, and path/operator policy gates
- empirical proof that snapshot-based side-effect accounting catches failed unattended side effects with low false-negative rates
- empirical proof that failed unattended runs do not leave hidden side effects
- repeatable unattended benchmark evidence strong enough to claim reliable walk-away delegation

### Phase 2: Unattended Repo Workflows

Status: materially implemented in substrate

What exists in the repo now:

- shared-repo runtime with isolated worker clones and branch publication
- deterministic `repo_sandbox` benchmark lanes for review, test repair, parallel merge acceptance,
  and conflict-resolution workflows
- semantic acceptance packets and workflow-aware verification for repo tasks
- workflow guards for git, generated files, claimed paths, and rollback-aware recovery
- resumable checkpoints for longer unattended jobs

What is still missing:

- broader repo families beyond the current deterministic sandbox lanes
- stronger semantic validation for larger real-world diffs and regression-style behavior
- richer test selection and repo-environment coverage than the current local scripted workflows
- empirical proof that the strongest repo lanes remain reliable as breadth grows

### Phase 3: Background Work Queue

Status: materially implemented in substrate

What exists in the repo now:

- delegated job queue state with priorities, deadlines, cancellation, checkpoint/report paths, and history
- operator-visible runtime governance for concurrency, subprocesses, artifact budgets, budget groups,
  and anti-starvation scheduling
- shared-repo collision control through worker branches, claimed paths, isolated clones, and lease governance
- actionable queue surfaces for blocked jobs, next runnable jobs, promotable jobs, and acceptance-gated completion

What is still missing:

- stronger fairness and scheduling policy than the current lightweight runner heuristics
- stronger runtime isolation than workspace/path mediation alone
- richer operator-facing queue UX than the current CLI/status views
- repeated unattended proof that multi-job behavior stays reliable under pressure

### Phase 4: Trusted Personal Kernel

Status: partially implemented in substrate

What exists in the repo now:

- persistent unattended trust ledgers with family-level assessments
- stable operator-facing policy controls for benchmark families, git, generated paths, HTTP, and capability modules
- auditable acceptance packets with verifier results, tests, branches, edit plans, and capability usage

What is still missing:

- broader delegated task families that look more like routine user work than deterministic sandbox lanes
- repeated evidence that trust posture remains strong as breadth and run volume increase
- a stronger product-facing operator experience around trust, acceptance, and capability management
- proof that bounded autonomy remains reliable as external modules and richer environments are added

### Phase 5: TOLBERT Checkpoint Competition

Status: materially implemented in substrate

What exists in the repo now:

- retained Tolbert runtime bundle materialization
- first-class retained `tolbert_model` artifact generation and retention wiring
- delegated/offloaded TOLBERT training and retrieval-cache jobs
- supervised dataset synthesis from trajectories, transition failures, verifier labels, and discovered tasks
- candidate-vs-baseline evaluation through the same retention framework used by other subsystems
- first-class liftoff-gate reports at [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)
- retained `tolbert_model` runtime policy surfaces for shadow-mode and primary-family routing

What is still missing:

- broader learned decoder control beyond retrieval-guided bounded command selection
- stronger latent/world-model heads beyond the current bounded hybrid runtime
- automatic materialization of hybrid family bundles inside the main retained TOLBERT candidate pipeline

### Phase 6: TOLBERT Family Takeover

Status: materially implemented in bounded form

Target benchmark:

- a retained TOLBERT-family checkpoint beats the initial `vllm+TOLBERT` product on approved benchmark families without regressing verifier, trust, recovery, or side-effect posture

Required product capabilities:

- learned action-ranking or bounded decoding heads
- learned transition and value heads over kernel state
- shadow-mode comparison against the seed runtime
- family-level routing and fallback to `vllm`
- liftoff gate reports with explicit non-regression checks

What exists in the repo now:

- retained `runtime_policy` inside `tolbert_model` artifacts
- shadow and primary benchmark-family routing in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
- automatic family-routing updates when retained Tolbert candidates clear the liftoff gate in [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- isolated liftoff evaluation/reporting and routing application in [`scripts/evaluate_tolbert_liftoff.py`](/data/agentkernel/scripts/evaluate_tolbert_liftoff.py)

### Phase 7: Open-World Latent Runtime

Status: started in bounded form

Target benchmark:

- TOLBERT-family checkpoints act as the primary latent state, transition, value, and policy runtime for approved families and continue improving under the retained loop

Required product capabilities:

- latent long-horizon memory and state-space modeling under [`agent_kernel/modeling/ssm/`](/data/agentkernel/agent_kernel/modeling/ssm)
- learned world/transition modules that augment rather than remove the explicit symbolic world state
- open-world task discovery and environment clustering beyond the fixed seed bank
- family-by-family authority transfer with fallback, arbitration, and teacher use from `vllm`

What exists in the repo now:

- model-native package surfaces under [`agent_kernel/modeling/policy/`](/data/agentkernel/agent_kernel/modeling/policy), [`agent_kernel/modeling/world/`](/data/agentkernel/agent_kernel/modeling/world), and [`agent_kernel/modeling/evaluation/`](/data/agentkernel/agent_kernel/modeling/evaluation)
- latent-state summaries that sit on top of the symbolic world model and are consumed by runtime policy and checkpoint payloads
- open-ended discovered-task pressure in the evaluation loop

What is still missing:

- truly learned latent transition/value/policy checkpoints that dominate the heuristic bootstrap
- broader open-world environment discovery than the current bounded local-task ecology

## Milestone Status

Against the immediate milestones later in this document, the repo currently looks like this:

1. stronger execution containment: mostly done
2. first unattended-task outcome report: done in first useful form
3. delegated local repo-chore benchmark family: materially done in deterministic repo-sandbox form
4. unattended preflight gating: mostly done
5. false-pass and hidden-side-effect measurement: materially done in first useful form

The important nuance is that the product now has bounded containment, preflight gating, and contract-aware
side-effect reports, a real repo workflow lane, queue governance, and trust controls, but it still cannot
honestly claim trustworthy unattended delegation until OS-level isolation, broader task breadth, and stronger
repeated walk-away evidence are closed.

## Concrete Repo Gaps

The highest-signal remaining product gaps are:

- the sandbox now executes segmented commands with `shell=False`, bounded host executable allowlists,
  workspace-local executables, and path/operator policy gates, but it still lacks OS-level containment
- unattended reports now account for changed files with before/after hashes and contract status, but repeated
  empirical proof on hidden-side-effect coverage is still limited
- repo-workflow verification is stronger than shape checks, but still narrow compared with broader real-user workflows
- the queue/runtime substrate exists, but the product still lacks a fuller operator UX for acceptance, scheduling,
  trust posture, and capability management
- the capability system supports bounded core actions and scoped modules, but real external adapters are still sparse
- there is not yet enough empirical evidence on false passes, hidden side effects, and repeated walk-away success
- the current learned-runtime product is still mostly retrieval-centered rather than policy/value/transition-centered
- the modeling package boundary exists, but the planned `tolbert/`, `policy/`, `world/`, `training/`, and `evaluation/` liftoff surfaces are still thin
- there is not yet a family-routing and liftoff-gate product layer where retained TOLBERT checkpoints can become authoritative in a governed way
- the current world-state substrate is explicit and symbolic; the open-world latent runtime still needs learned state-space components on top

## TOLBERT Liftoff Path

The TOLBERT product roadmap should be read as a governed takeover path:

1. `Seed`
   - `vllm` controls
   - TOLBERT retrieves and predicts paths
2. `Learned-state`
   - TOLBERT checkpoints learn from verifier, transition, trust, and recovery signals
3. `Shadow policy`
   - TOLBERT ranks or proposes actions in shadow mode against the seed runtime
4. `Family takeover`
   - retained TOLBERT checkpoints become primary for approved families
5. `Open-world latent runtime`
   - TOLBERT-family checkpoints hold latent long-horizon state and beat the initial `vllm+TOLBERT` product as the primary self-improving machine runtime

The product rule is:

- a new checkpoint is not promoted because it trained
- a new checkpoint is promoted only because the kernel retained it against the active runtime under the same verifier, trust, recovery, and reporting system

## Phase 0: Supervised Bounded Reliability

Current benchmark:

- the kernel can solve bounded local tasks in a replayable harness with deterministic verification

Exit criteria:

- fixed-bank task success is stable under the preferred live runtime
- generated-task success is not collapsing relative to the fixed bank
- runtime failures are classified clearly as inference, retrieval, verifier, or controller failures
- every run leaves a replayable episode with steps, verification results, and termination reason

Why it matters:

- this is the minimum substrate for any later unattended product behavior

## Phase 1: Unattended Single-Task Delegation

Target benchmark:

- a user can queue one bounded local task, walk away, and return to either a correct result or a safe stop

Required product capabilities:

- task intake with explicit contract, workspace, and success conditions
- segmented `shell=False` command isolation with a bounded host executable allowlist, workspace-local executables,
  and path/operator policy gates
- preflight checks for model, retrieval assets, and required environment dependencies
- safe stop behavior on timeout, repeated failure, missing dependency, or verifier uncertainty
- end-of-run report summarizing actions, outputs, changed files, and verifier evidence
- delegated queue governance with budget-group caps so one autonomous campaign cannot monopolize all unattended slots

Exit criteria:

- unattended task success on a bounded benchmark family is repeatable
- failed runs do not escape the intended workspace or leave hidden side effects
- users can distinguish success, safe stop, and unsafe ambiguity from the final report

## Phase 2: Unattended Repo Workflows

Target benchmark:

- the kernel can handle common repository workflows rather than only tiny artifact tasks

Representative delegated jobs:

- implement a small scoped change in a repo sandbox
- run the relevant tests or checks
- repair a failing deterministic test
- prepare a branch, diff summary, and verification report for review

Required product capabilities:

- multi-step planning with longer horizons than the default short loop
- stronger semantic verifiers for code changes, not only file and substring checks
- workflow-specific guards for git, tests, generated files, and rollback
- resumable background execution with progress checkpoints

Exit criteria:

- the strongest repo-workflow lane stays reliable across multiple benchmark families
- semantic success rates are high enough that users would choose delegation over manual babysitting
- the system can stop cleanly and explain partial progress when a workflow cannot be completed

## Phase 3: Background Work Queue

Target benchmark:

- the kernel can manage multiple delegated jobs safely over time

Required product capabilities:

- a queue with per-task state, priorities, deadlines, and cancellation
- resource controls for model use, subprocesses, and disk artifacts
- concurrency rules so one job cannot silently corrupt another job's runtime inputs
- operator controls for approval policies, allowed task classes, and escalation behavior

Exit criteria:

- multiple queued tasks can run without state leakage or artifact collisions
- background jobs remain auditable and interruptible
- partial failures degrade gracefully rather than poisoning the whole runtime

## Phase 4: Trusted Personal Kernel

Target benchmark:

- the kernel becomes a tool a user trusts to routinely take local software work off their plate

Required product capabilities:

- benchmark families that resemble real user delegation, not only harness tasks
- persistent trust signals from repeated success, low false-pass rates, and strong rollback behavior
- stable operator-facing policy controls for what the kernel may or may not do unattended
- evidence that gains persist as task breadth expands

Exit criteria:

- repeated unattended runs show durable usefulness on real delegated tasks
- broadening the task set does not erase reliability
- the user can set clear autonomy boundaries and see that the kernel respects them

## Shared Dependencies With The ASI Track

The two tracks should share infrastructure where that creates real leverage:

- stronger verifier contracts
- broader deterministic benchmark families
- retained artifact evaluation and rollback
- failure clustering and recovery measurement
- world-model and planner improvements that raise real task success
- cycle logs that connect proposed changes to measured retained gains

## Convergence Principle

The right way to converge the product and ASI tracks is not to merge the claims.

The right way is to prioritize capabilities that satisfy both of these tests:

- they make the delegated-work product more trustworthy and useful
- they make the recursive-improvement substrate more measurable, safer, and more capable

In practice, the shared target is:

- a more capable agent that can act in broader local environments
- under stronger verification and containment
- while learning from retained evidence rather than one-off prompting

That means the product benchmark should not be treated as a distraction from the ASI roadmap.
It is one of the best empirical forcing functions for making the core agent stronger.

## Convergence Task Stack

The highest-leverage shared tasks are:

1. strengthen execution containment
   - replace weak shell blocking with a stronger bounded execution environment
   - track side effects explicitly so both unattended runs and improvement cycles can detect unsafe drift
2. upgrade verification from shape checks to semantic checks
   - keep deterministic file and output checks
   - add workflow-aware verifiers for repo changes, tests, regressions, and hidden side effects
3. build delegated-work benchmark families
   - add tasks that resemble real repo chores, not only micro artifacts
   - use those same families as broader empirical lanes for retained-artifact evaluation
4. harden candidate isolation and rollback
   - make sure proposed artifacts cannot silently become live runtime inputs
   - ensure both product runs and self-improvement cycles can recover cleanly from rejected changes
5. improve long-horizon control
   - extend planning, checkpointing, and resumability so the agent can complete multi-step work
   - use the same machinery to support longer improvement campaigns
6. add preflight and health gating
   - refuse to start unattended work or self-improvement runs when the model, retrieval stack, or verifier stack is unhealthy
7. improve reporting and auditability
   - produce user-facing reports for delegated jobs
   - produce cycle-facing evidence trails for retained artifact decisions
8. measure compounding, not just one-shot success
   - track whether capability gains persist across broader task families and later cycles
   - treat false-pass and hidden-side-effect rates as first-class regressions

## Current Capability Tasks

The next tasks should be ordered by how much they raise both product quality and core agent capability.

### Tier 1

- push the executor boundary from bounded process isolation to stronger OS-level containment
- broaden delegated benchmark families beyond the current deterministic repo-sandbox lanes
- improve acceptance UX so trust, blocker, verifier, and capability decisions are legible at a glance
- turn the capability/module system into a real product surface with manageable scopes and installable adapters

Why this tier matters:

- it closes the biggest remaining product gap between a strong engine and a tool users can routinely trust
- it creates the broader task lanes needed for real retained-evidence claims

### Tier 2

- add stronger semantic verifier coverage for larger repo changes, regression-style checks, and richer test behavior
- improve queue scheduling policy with better fairness, throughput, and operator control
- add richer capability adapters on top of the bounded core runtime
- expand failure clustering so repeated unattended failures can directly drive benchmark, verifier, and curriculum proposals

Why this tier matters:

- it raises task realism
- it turns product failures and blocked work into better improvement inputs

### Tier 3

- let the kernel compare multiple candidate plans or execution variants before committing to one path
- improve retrieval artifacts beyond threshold overrides into retained asset and routing changes
- strengthen planner and world-model evaluation on longer repo workflows and broader delegated tasks
- add portfolio-level experiment selection for improvement campaigns and delegated queues

Why this tier matters:

- it moves the agent from bounded replay competence toward more adaptive and compounding behavior

## What Capability Means Here

For this repo, "more capable" should be defined narrowly and empirically.

It means the agent can:

- solve a broader range of deterministic local software tasks
- do so with fewer unsafe side effects and fewer false passes
- recover better from failure
- preserve gains across later runs and later retained changes
- transfer improvements across benchmark families rather than overfitting one lane

If a change only makes the system sound smarter without improving those properties, it should not count as convergence progress.

## Product-Specific Gaps

The delegation roadmap also needs work that should not be hidden inside `asi.md`:

- real execution isolation
- broader delegated-work benchmark breadth
- stronger operator-facing trust and permission UX
- richer semantic success checks for user-facing work
- capability adapters and module lifecycle management
- packaging and environment preflight so unattended runs are operationally reliable at product depth

## Immediate Next Milestones

The next practical milestones for this roadmap are:

1. Push bounded execution into stronger OS-level containment without breaking the workspace-first product model.
2. Expand delegated benchmark breadth beyond the current deterministic repo-sandbox lanes.
3. Turn acceptance packets, trust posture, and queue blockers into a stronger operator experience.
4. Add concrete scoped capability adapters on top of the bounded core runtime.
5. Accumulate and analyze repeated unattended evidence across the broader delegated families.

## Near-Term Convergence Sequence

If the goal is to make the agent more capable as quickly as possible while preserving the product and
ASI boundary, the next sequence should be:

1. containment
   - make bounded execution stronger before expanding unattended reach further
2. broaden delegated benchmark families
   - create wider task lanes that resemble routine repo work and bounded external-tool use
3. acceptance and trust UX
   - make verifier decisions, blockers, capability use, and trust posture operator-legible
4. capability adapters
   - add scoped external modules on top of the bounded `workspace_fs`, `workspace_exec`, and `http_request` core
5. empirical trust closure
   - use the broader lanes to measure hidden-side-effect rates, false passes, and durable unattended usefulness
6. failure-driven improvement loop integration
   - use delegated-task failures to drive benchmark, verifier, retrieval, and curriculum proposals

## Relationship To `asi.md`

Keep the boundary clear:

- `asi.md` asks whether the repo contains the machinery for recursive self-improvement and retained better successors
- this roadmap asks whether the repo is becoming a trustworthy delegated-work product
- the convergence work above is the shared capabi