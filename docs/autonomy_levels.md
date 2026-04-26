# Autonomy Levels

## Purpose

This document defines a constrained autonomy ladder for `agent-kernel`.

The goal is not to invent a universal standard. The goal is to make autonomy claims in this repo
more measurable, more conservative, and easier to connect to the actual proof surfaces already used
elsewhere in the docs.

It complements, but does not replace:

- [`agi.md`](/data/agentkernel/agi.md), which is the capability-building roadmap
- [`asi.md`](/data/agentkernel/asi.md), which is the broader ASI substrate contract
- [`asi_core.md`](/data/agentkernel/docs/asi_core.md), which defines the kernel core boundary
- [`runtime.md`](/data/agentkernel/docs/runtime.md), which defines the runtime claim boundary
- [`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md), which defines the live
  proof gaps
- [`product_roadmap.md`](/data/agentkernel/docs/product_roadmap.md), which defines the delegated
  product benchmark

## Core Rule

Autonomy is the degree to which a system can turn goals into verified actions with decreasing human
supervision.

The key variable is control-loop ownership, not raw model IQ.

A strong model can still be low-autonomy if it only answers prompts.
A weaker model can be high-autonomy in a dangerous but untrustworthy way if it acts without
verification or containment.

For this repo, a level claim should therefore mean:

- the system owns a larger part of the control loop
- the claim is backed by stored evidence, not chat interpretation
- the lower levels remain stable while the higher-level behavior is exercised

## Important Separations

Several concepts in this repo are related, but they are not the same thing:

- `AGI roadmap layers` in [`agi.md`](/data/agentkernel/agi.md) describe the order in which
  capability-building work should land
- `autonomy levels` in this document describe who owns execution and what evidence is needed to
  claim that ownership
- supervisor `autonomy_mode` values such as `shadow`, `dry_run`, and `promote` are rollout
  permissions, not autonomy levels
- the `bounded_autonomous` runtime shape in
  [`runtime.md`](/data/agentkernel/docs/runtime.md) is a runtime contract, not proof that the
  product has reached a high autonomy level
- the ASI language in [`asi.md`](/data/agentkernel/asi.md) and
  [`minimal_asi_qwen_tolbert.md`](/data/agentkernel/docs/minimal_asi_qwen_tolbert.md) is a
  machine-contract and threshold discussion, not a claim that the current repo has already crossed
  that threshold

## Promotion Rule

The repo should only claim the highest level for which all of the following are true:

- the required surfaces exist in code
- the system repeatedly exercises them under bounded evaluation
- the evidence lands in durable reports or ledgers
- the next-level claim does not depend on hand-waving away lower-level instability

Implemented machinery above the current proof line is roadmap substrate, not current-level proof.

## What Does Not Count

These do not upgrade the autonomy level by themselves:

- one-off demos
- tool access by itself
- broad permissions without stronger verification
- human-authored step-by-step decomposition
- sampled breadth without counted trusted breadth
- controller-closed productive partial timeouts as proof of child-owned decision closure
- candidate generation without retained gains
- a single retained win without repeated isolated comparison
- a runtime flag or config preset that claims more than the evidence supports

## The Ladder

### A0: Non-Autonomous Model

The system is a reasoning primitive, not an operator.

Human ownership:

- supplies the prompt
- decides when the model runs
- interprets the output
- decides every next step

System ownership:

- generates a response for one turn

Required evidence:

- none beyond response quality

Disqualifiers:

- no persistent memory
- no external action
- no independent continuation

### A1: Reactive Assistant

The system can maintain short dialogue context and help with decomposition, drafting, explanation,
and coding suggestions, but it still only reacts turn by turn.

Human ownership:

- owns the task
- owns the control loop
- chooses all action-taking steps

System ownership:

- local reasoning within the current interaction

Required evidence:

- coherent multi-turn assistance

Disqualifiers:

- no tool-mediated action
- no bounded task ownership
- no independent execution record

### A2: Tool-Using Assistant

The system can use tools such as search, code execution, filesystems, APIs, or repo-local utilities.
This is the first meaningful action surface, but the human still owns the assignment.

Human ownership:

- defines the task
- sets the boundaries
- usually remains the outer control loop

System ownership:

- selects and sequences tools within a bounded turn or narrow workflow

Required evidence:

- tool calls succeed reliably enough to support task execution
- results are grounded by verifiers, checks, or deterministic outputs where possible

Disqualifiers:

- tool use alone is not workflow autonomy
- the system still stops when the turn ends
- no durable task record or safe-stop policy

### A3: Bounded Workflow Agent

The system can complete a bounded task end to end inside a constrained environment. It can decompose
the work, retry failed steps, verify outputs, and stop safely without needing a human after every
substep.

Human ownership:

- provides a bounded task
- defines the operating envelope

System ownership:

- owns most of the execution inside that task
- manages local retries, verification, and stop conditions

Required evidence:

- unattended or light-supervision task completion on bounded task families
- auditable task reports
- explicit `success`, `safe_stop`, and `unsafe_ambiguous` outcomes
- repeatable bounded pass-rate evidence

Repo-aligned measurement surfaces:

- `supervision.independent_execution`
- `supervision.mode`
- `light_supervision_candidate`
- `light_supervision_clean_success`
- bounded family pass rate and task completion

Disqualifiers:

- the human still decomposes the work between every step
- there is no trustworthy after-action record
- failures cannot stop safely or be audited

### A4: Goal-Directed Operator

The system moves from "finish this bounded task" to "pursue this bounded objective under stated
constraints." It can generate subgoals, notice when the plan is failing, and close decisions
naturally rather than only surfacing ambiguous partial work.

Human ownership:

- sets the objective
- sets constraints, budgets, and policy limits

System ownership:

- owns subgoal formation
- owns local decision closure
- owns replanning inside the bounded mission

Required evidence:

- runtime-managed decisions are produced by the child, not only by the controller
- decisions close with natural child-owned ownership
- breadth begins to appear in counted evidence, not only in sampled coverage

Repo-aligned measurement surfaces:

- `runtime_managed_decisions`
- `decision_owner=child_native`
- `closeout_mode=natural`
- counted trusted breadth and clean task-root breadth across required families

Disqualifiers:

- sampled family coverage without counted trust
- `controller_runtime_manager` closure as the main source of decisions
- `partial_timeout_evidence_only` as the main proof shape

### A5: Persistent Delegated Operator

The system now owns a continuing role rather than a single bounded mission. It can resume work,
prioritize multiple tasks, survive interruptions, and maintain continuity over time.

Human ownership:

- sets role-level direction
- sets mission priorities and policy boundaries

System ownership:

- owns queue-level continuity
- owns resume and interruption handling
- owns multi-goal local prioritization

Required evidence:

- persistent work queues or resumable task streams
- successful resume after interruption
- multi-job or multi-day continuity with stable trust posture
- operator input becomes supervisory rather than reconstructive

Repo-aligned measurement surfaces:

- job queue completion and resume artifacts
- stable unattended trust ledgers across repeated runs
- low intervention rate per completed delegated job

Disqualifiers:

- every run must be manually re-explained
- continuity exists only in scaffolding, not in live proof
- trust collapses as soon as workload volume rises

Current repo status:

- `agent-kernel` now has defended `A5_substrate` evidence rather than only A3/A4 evidence.
- The bounded A5 proof line is live-vLLM delegated queue operation with durable packets for
  required-family queue execution, ordinary workspace contracts, resume/reaping, real restart,
  repeated real restart, ordinary queue intake, git-backed `repo_sandbox` continuity, and a
  combined ordinary-plus-git-plus-restart queue window, plus a longer selected-intake mixed-role
  queue window and repeated-window built-in role continuity.
- This is enough to say the repo is operating in the A5 lane at the bounded-substrate level.
- It is not yet enough to claim full production-duty A5, because the remaining proof needs longer
  wall-clock continuity, less curated intake, and natural role closeout without operator steering.

### A6: Self-Improving Operator

The system can improve parts of the machinery by which it acts. It can propose, compare, retain, or
reject internal changes under verifier pressure, and those retained changes materially affect later
behavior.

Human ownership:

- sets governance boundaries
- controls the allowed mutation and retention envelope

System ownership:

- proposes internal changes
- evaluates candidate versus retained behavior
- applies only verified gains

Required evidence:

- retained-vs-candidate comparison exists and is active
- retained gains survive regression gates
- repeated isolated autonomous runs show non-collapsing improvement
- the retained changes alter later runtime behavior, not only reports

Repo-aligned measurement surfaces:

- `retained_gain_runs`
- retained candidate-vs-baseline reports
- autonomous compounding checks
- trusted carryover and retrieval carryover deltas that survive later runs

Disqualifiers:

- self-edits without stable retention gates
- more proposals without more retained gains
- one retained success without repeated isolated comparison
- human cherry-picking that substitutes for closed-loop retention evidence

### A7: AGI

AGI begins when the same system can enter unfamiliar domains inside its declared task universe,
acquire competence there with limited redesign, and continue operating robustly rather than
collapsing outside the training or benchmark shell.

Human ownership:

- supplies broad mission direction and governance

System ownership:

- acquires new competence across unfamiliar domains
- transfers what it has learned
- operates with strong breadth rather than lane-specific scripting

Required evidence:

- broad transfer across unfamiliar environments
- strong human-level or better performance across the declared task universe
- limited need for bespoke scaffolding per new domain

Disqualifiers:

- success only on curated narrow families
- each new domain requires new architecture
- tool breadth without robust unfamiliar-domain adaptation

Repo note:

- for this repository, the directly measurable near-term version of this threshold is coding-domain
  generality, not literature-wide human-domain generality

### A8: ASI

ASI begins when the system is decisively beyond the human frontier across the declared task
universe and the recursive-improvement machinery compounds that advantage under verification.

Human ownership:

- governance
- containment
- policy and constitutional limits

System ownership:

- superhuman task performance
- strategic planning beyond strong human baselines
- recursive capability growth that remains verifier-governed

Required evidence:

- decisive outperformance versus strong human or team baselines across the declared task universe
- recursive improvement that compounds rather than merely fluctuates
- strategically superior abstractions, plans, or inventions that survive verification

Disqualifiers:

- broad architecture without superhuman benchmark evidence
- autonomy without superhuman cognition
- local wins in one benchmark family treated as universal superiority

## Prompt Test

The ladder can be compressed into the kind of instruction the human gives:

- `A0`: answer this
- `A1`: help me think through this
- `A2`: use tools to solve this
- `A3`: complete this bounded task
- `A4`: achieve this bounded objective
- `A5`: keep handling this role over time
- `A6`: improve how you handle this role
- `A7`: handle unfamiliar domains in the declared task universe with limited redesign
- `A8`: outperform the best humans across that task universe

## Measurement Surfaces In This Repo

This repo should not collapse autonomy into one scalar. It should promote levels through gates.

The current high-value measurement surfaces are:

- `supervision.mode`
- `supervision.independent_execution`
- `light_supervision_candidate`
- `light_supervision_success`
- `light_supervision_clean_success`
- `runtime_managed_decisions`
- `decision_owner`
- `closeout_mode`
- counted trusted breadth by required family
- clean task-root breadth by required family
- `retained_gain_runs`
- candidate-vs-baseline retention reports
- autonomous compounding reports
- external breadth and unfamiliar-environment transfer evidence

These are more useful than a single "autonomy score" because they expose the actual failure mode:
insufficient independence, weak decision ownership, narrow breadth, weak retained conversion, or
weak cross-domain transfer.

## Current Honest Placement

The current repo is clearly beyond `A2`.

The current defended placement is:

- highest defended lane: `A6_bounded_self_improvement`
- plain-language status: bounded persistent delegated operator evidence exists, and a narrow
  isolated self-improvement scout has now retained repeated runtime-affecting gains
- lower anchors: A3 bounded task completion and a narrow A4 retained child-native route remain
  satisfied, but they are no longer the highest useful placement
- not-yet-claimed lane: `A7`

This document should not be read as placing the repo at A3. A3 is now a lower bound. The current
proof line has two defended layers. The A5 layer is bounded persistent delegated operation: the
queue can hold and complete delegated work, recover from interruption, preserve trust posture over
bounded live packets, and handle ordinary plus git-backed work. The A6 layer is a bounded
transition-model self-improvement scout: repeated isolated child runs retained non-regressing
runtime-control gains under autonomous compounding gates.

The defended A5-substrate evidence is:

- live-vLLM persistent delegated queue operation:
  r5 and r6 together cover `integration`, `repo_chore`, `project`, `repository`, and
  `repo_sandbox` across 15 separate `drain --limit 1` invocations, 15 successful jobs,
  five accepted parent promotions, zero blocked jobs at closeout, zero active leases, and zero
  hidden-side-effect risk in both isolated packets
- r8 broader ordinary workspace-contract control:
  [`a5_live_vllm_broader_workspace_contract_r8_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_broader_workspace_contract_r8_20260425.json)
  records 10 successful unique task roots under live vLLM after the
  `workspace_contract_direct` patch, with zero hidden-side-effect risk and a
  verified task-bank fix for the prior tooling quote defect
- r9 persistent queue resume/recovery:
  [`a5_live_vllm_resume_recovery_r9_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_resume_recovery_r9_20260425.json)
  records an interrupted in-progress job, a stale orphan lease reaped by the
  runtime controller, and six successful separate drain invocations with zero
  active leases at closeout
- r10 real process restart:
  [`a5_live_vllm_real_restart_r10_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_real_restart_r10_20260425.json)
  records a live delegated drain interrupted by SIGINT during a bounded
  external-manifest task, followed by a fresh drain invocation that resumed and
  completed the same job successfully
- r11 repeated real process restart:
  [`a5_live_vllm_repeated_real_restart_r11_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_real_restart_r11_20260425.json)
  records five live delegated drains interrupted by SIGINT across the required
  families, with every job left resumable, then completed by a fresh drain
  invocation with `attempt_count=2`, `trust_status=trusted`, zero active leases,
  and zero hidden-side-effect risk
- r12 ordinary persistent-role queue intake:
  [`a5_live_vllm_persistent_role_r12_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_persistent_role_r12_20260425.json)
  records three waves of ordinary built-in task-bank queue intake over a
  58-second role window, 12 fresh-process drains, 12 `completed/success` jobs,
  `trust_status=trusted`, zero active leases, and zero hidden-side-effect risk
- r13 git-backed `repo_sandbox` continuity:
  [`a5_live_vllm_repo_sandbox_r13_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repo_sandbox_r13_20260425.json)
  records five built-in `repo_sandbox` jobs, including standalone git review/repair
  and a shared-repo worker/merge acceptance trio, all completed with
  `repo_sandbox_family_status=trusted`, zero active leases, and zero
  hidden-side-effect risk
- r14 combined A5-substrate role window:
  [`a5_live_vllm_combined_role_r14_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_combined_role_r14_20260425.json)
  records one live queue window combining ordinary built-in intake, git-backed
  shared-repo work, and a real SIGINT interruption/resume of a git-backed
  `repo_sandbox` task, with 8/8 `completed/success`, all five required families
  counted, `trust_status=trusted`, zero active leases, and zero hidden-side-effect
  risk
- r15 longer selected-intake mixed-role window:
  [`a5_live_vllm_longer_mixed_role_r15_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_longer_mixed_role_r15_20260425.json)
  records a 184-second live queue window with deterministic task-bank selection,
  git-backed `repo_sandbox` work, and two real SIGINT interruption/resume events,
  with 15/15 `completed/success`, all five required families counted,
  `trust_status=trusted`, zero active leases, and zero hidden-side-effect risk
- r16 repeated-window built-in role continuity:
  [`a5_live_vllm_repeated_window_r16_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_window_r16_20260425.json)
  records a 294-second live queue role with four separated intake/drain windows,
  deterministic built-in task-bank selection only, ordinary plus git-backed work,
  final queue-empty closeout, 17/17 `completed/success`, all five required
  families counted, `trust_status=trusted`, zero active leases, and zero
  hidden-side-effect risk
- r17 product-native queue role closeout status:
  [`a5_live_vllm_product_native_role_closeout_r17_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_product_native_role_closeout_r17_20260425.json)
  records a production-code status surface where `scripts/run_job_queue.py
  status --json` emits `queue.role_closeout`; replaying the r16 root through
  that native surface returned `closeout_ready=true`,
  `closeout_mode=queue_empty_trusted`, `operator_steering_required=false`,
  `completed_success_jobs=17`, `active_leases=0`, and `trust_status=trusted`
- r18 external-manifest native-closeout role continuity:
  [`a5_live_vllm_external_native_role_r18_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_external_native_role_r18_20260425.json)
  records a 769-second live queue role with four separated external-manifest
  intake/drain windows, 10/10 `completed/success`, all five required families
  counted at two clean task roots each, `task_origins={"external_manifest":10}`,
  `trust_status=trusted`, zero active leases, zero hidden-side-effect risk, and
  native `queue.role_closeout` reporting `queue_empty_trusted` with
  `operator_steering_required=false`
- r19 product-native manifest intake:
  [`a5_live_vllm_enqueue_manifest_role_r19_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_enqueue_manifest_role_r19_20260425.json)
  records the new `scripts/run_job_queue.py enqueue-manifest` intake surface
  ingesting the default external breadth manifest, embedding portable
  `task_payload` records into all 10 queued jobs, draining all jobs to
  `completed/success`, and closing with native `queue.role_closeout` as
  `queue_empty_trusted`

The lower-level anchors remain:

- A3: real bounded coding task completion evidence with auditable reports
- A4: a narrow but defended retained crossing on the direct five-family bounded
  `transition_model` route, where `cycle:transition_model:20260423T223800Z:604f5429r6`
  retained after child-native runtime-managed natural closeout and non-regressive
  transition-model improvement
- A5: bounded persistent delegated queue operation with live-vLLM queue windows, resumable
  interruption handling, native queue closeout, and external-manifest intake

The defended A6 evidence is:

- r33 isolated transition-model autonomous compounding scout:
  `/tmp/agentkernel_a6_transition_scout_r33_20260426/trajectories/improvement/reports/autonomous_compounding_20260426T020354414563Z.json`
  records `autonomous_compounding_viable=true`,
  `claim_gate_summary.autonomous_compounding_claim_ready=true`, and no claim blockers.
  The run had `successful_runs=2`, `runs_with_retention=2`,
  `claim_gate_summary.min_runtime_managed_decisions=1`, a clean result-stream audit, and
  two retained child-native natural-closeout transition-model decisions.
- Both retained child cycle reports had `final_state=retain`, final reason
  `transition-model candidate improved retained bad-transition guidance without broader
  regression`, compatible transition-model artifacts, and
  `transition_model_scoring_control_delta_count=6`.
- The r33 parent transfer gate counted all seven target families as observed with retained gain:
  `project`, `repository`, `integration`, `repo_chore`, `repo_sandbox`, `workflow`, and
  `tooling`; `families_missing_clean_task_root_breadth=[]`.

The current blockers on an `A7` claim are:

- unfamiliar-environment transfer is not yet broad enough; the r33 A6 proof is still a
  transition-model scout, not a general coding-domain adaptation proof
- non-mock provider evidence exists mainly in A5 queue packets; A6 retained self-improvement still
  needs stronger live-provider repetition before treating it as production-duty self-improvement
- strong human-level or better performance across the declared coding task universe has not been
  established

The repo also contains meaningful `A5`, `A6`, and early `A7` machinery:

- persistent queue and supervision machinery now have a verified `A5_substrate` packet:
  [`a5_substrate_required_family_persistent_queue_packet_20260425.json`](/data/agentkernel/docs/evidence/a5_substrate_required_family_persistent_queue_packet_20260425.json)
- ordinary workspace-contract routing now has a verified live-vLLM companion packet:
  [`a5_live_vllm_broader_workspace_contract_r8_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_broader_workspace_contract_r8_20260425.json)
- persistent queue resume/reaping now has a verified live-vLLM companion packet:
  [`a5_live_vllm_resume_recovery_r9_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_resume_recovery_r9_20260425.json)
- real process restart/resume now has a verified live-vLLM companion packet:
  [`a5_live_vllm_real_restart_r10_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_real_restart_r10_20260425.json)
- repeated real process restart/resume now has a verified live-vLLM companion packet:
  [`a5_live_vllm_repeated_real_restart_r11_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_real_restart_r11_20260425.json)
- ordinary built-in queue intake continuity now has a verified live-vLLM companion packet:
  [`a5_live_vllm_persistent_role_r12_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_persistent_role_r12_20260425.json)
- git-backed repo_sandbox continuity now has a verified live-vLLM companion packet:
  [`a5_live_vllm_repo_sandbox_r13_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repo_sandbox_r13_20260425.json)
- combined ordinary intake, git-backed work, and real restart/resume now has a verified live-vLLM
  companion packet:
  [`a5_live_vllm_combined_role_r14_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_combined_role_r14_20260425.json)
- longer selected-intake mixed-role continuity now has a verified live-vLLM companion packet:
  [`a5_live_vllm_longer_mixed_role_r15_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_longer_mixed_role_r15_20260425.json)
- repeated-window built-in role continuity now has a verified live-vLLM companion packet:
  [`a5_live_vllm_repeated_window_r16_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_repeated_window_r16_20260425.json)
- product-native queue role closeout status now has a verified live-vLLM companion packet:
  [`a5_live_vllm_product_native_role_closeout_r17_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_product_native_role_closeout_r17_20260425.json)
- external-manifest native-closeout role continuity now has a verified live-vLLM companion packet:
  [`a5_live_vllm_external_native_role_r18_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_external_native_role_r18_20260425.json)
- product-native manifest intake now has a verified live-vLLM companion packet:
  [`a5_live_vllm_enqueue_manifest_role_r19_20260425.json`](/data/agentkernel/docs/evidence/a5_live_vllm_enqueue_manifest_role_r19_20260425.json)
- retained-vs-candidate self-improvement machinery exists
- repeated autonomous compounding checks exist
- held-out external manifests can now be used as the next A7 pressure surface; see
  [`a7_unfamiliar_transfer_manifest.json`](/data/agentkernel/config/a7_unfamiliar_transfer_manifest.json)

But those surfaces should still be treated conservatively. The r33 evidence places the repo in the
bounded A6 lane; it is not evidence for `A7`, because the next threshold requires robust unfamiliar
environment transfer with limited redesign.

AGI and ASI remain threshold terms, not current-state labels.

## Practical Path To ASI

If the goal is a defensible path rather than a slogan, the path is:

- `A3 -> A4`: increase child-native natural decision closure and counted family breadth
- `A4 -> A5`: stabilize persistent queue, resume, and multi-job delegated operation
- `A5 -> A6`: convert self-modification machinery into repeated retained gains under isolation
- `A6 -> A7`: widen transfer across unfamiliar environments in the declared task universe
- `A7 -> A8`: beat strong human baselines and keep recursive improvement compounding under
  verification

That path is strict on purpose.

It prevents the repo from calling implementation breadth "autonomy," calling runtime permissions
"agency," or calling self-improvement scaffolding "ASI" before the evidence is there.
