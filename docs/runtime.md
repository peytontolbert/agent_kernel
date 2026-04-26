# Runtime

## Default path

The default unattended proof path is the seeded `provider=vllm` runtime with TOLBERT context and
retrieval still enabled. The broader seed runtime for open-ended work is still local `vllm` plus a strict TOLBERT
service subprocess, so external `vllm`/`ollama` remains part of the overall live posture in
[`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py).

This is the current seed runtime posture:

- `vllm` is the authoritative free-form decoder
- the live TOLBERT service is the authoritative encoder/retrieval compiler slice
- `provider=hybrid` now binds the retained Tolbert-family decoder as the provider surface rather than
  falling back to deterministic task commands, and it prefers the retained universal-decoder bundle
  when that bundle is materialized
- retained Tolbert primary routing now includes state-conditioned decoder generation, not only
  bounded candidate enumeration plus hybrid rescoring
- `provider=tolbert` is retained only as a backward-compatible alias for `provider=hybrid`
- [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py) now
  defaults unattended child runs back to the configured seed provider unless a provider is explicitly requested
- the broader TOLBERT family in this repo is the target encoder-latent-decoder universal runtime documented in [`tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md)
- future TOLBERT-family checkpoint takeover is a retained-gate milestone, not a naming trick

Recommended `vllm` launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization 0.5
```

Recommended runtime env:

```bash
export AGENT_KERNEL_PROVIDER=vllm
export AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B
export AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000
```

Important `KernelConfig` defaults from [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py):

- provider: `ollama`
- provider: `vllm`
- model: `Qwen/Qwen3.5-9B`
- host: `http://127.0.0.1:8000`
- host: `http://127.0.0.1:11434`
- `use_tolbert_context=1`
- `tolbert_mode=full`
- `use_skills=1`
- `use_graph_memory=1`
- `use_world_model=1`
- `use_universe_model=1`
- `use_planner=1`
- `use_role_specialization=1`
- `persist_learning_candidates=1`
- `use_prompt_proposals=1`
- `use_curriculum_proposals=1`
- `use_retrieval_proposals=1`
- `max_steps=5`
- `max_task_steps_hard_cap=4096`
- `frontier_task_step_floor=50`
- `runtime_history_step_window=32`
- `payload_history_step_window=12`
- `checkpoint_history_step_window=24`
- `timeout_seconds=20`

## Resource Resolution

The runtime now has a small explicit resource substrate for selected active inputs.

- base prompt templates resolve through [`agent_kernel/resource_registry.py`](/data/agentkernel/agent_kernel/resource_registry.py)
- builtin subsystem artifacts are registered from [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py) now loads `system` and `decision` prompt templates through that registry
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py) now resolves `universe_constitution`, `operating_envelope`, and legacy `universe_contract` through resource ids instead of path-specialized loading

This is a first `RSPL-lite` slice rather than a complete resource substrate. The covered surfaces are documented in [`resource_protocol.md`](/data/agentkernel/docs/resource_protocol.md).

Runtime shape note:

- `KernelConfig.claimed_runtime_shape()` now treats `bounded_autonomous` as a decoder-native claim rather than a
  pure world/governance claim
- that claim now requires a retained native decoder posture in addition to `use_graph_memory=1` and
  `use_world_model=1`
- if you want a smaller runnable kernel without those surfaces, use [`KernelConfig.executable_floor(...)`](/data/agentkernel/agent_kernel/config.py:603) instead of weakening the bounded-autonomous mode

Strict executable floor:

- use [`KernelConfig.executable_floor(...)`](/data/agentkernel/agent_kernel/config.py:603) when you want the closed-loop executor without ASI-runtime enrichments or self-improvement closeout hooks
- that preset disables `use_tolbert_context`, `use_skills`, `use_graph_memory`, `use_world_model`, `use_universe_model`, `use_planner`, `use_role_specialization`, and `persist_learning_candidates`
- episode persistence remains on, so the floor still ends in a normal `EpisodeRecord` plus saved episode artifact

The loop in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py) enriches state before each decision:

- graph summary from prior episodes
- world-model summary
- short verifier-oriented plan
- current acting role
- TOLBERT `ContextPacket`

`loop.py` is now narrower than before. The live step loop still stays there, but support glue around it now lives in:

- [`agent_kernel/ops/loop_runtime_support.py`](/data/agentkernel/agent_kernel/ops/loop_runtime_support.py:1) for provider/client construction, shared-repo materialization and publish flows, and post-episode learning-candidate persistence helpers
- [`agent_kernel/ops/loop_checkpointing.py`](/data/agentkernel/agent_kernel/ops/loop_checkpointing.py:1) for checkpoint payload serialization, resume state reconstruction, and setup-history loading
- [`agent_kernel/ops/loop_progress.py`](/data/agentkernel/agent_kernel/ops/loop_progress.py:1) for progress-event shaping
- [`agent_kernel/extensions/planner_recovery.py`](/data/agentkernel/agent_kernel/extensions/planner_recovery.py:1) for planner-recovery artifact synthesis and ranking helpers

The runtime no longer reaches most learned/modeling surfaces directly from the loop and policy modules.
[`agent_kernel/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/runtime_modeling_adapter.py:1)
now provides the lazy seam for:

- context-provider construction
- retained model artifact loading
- hybrid scoring, world-signal inference, and retained decoder generation
- latent-state enrichment hooks
- bounded decoder/action-generation helpers plus the state-conditioned hybrid decoder path

That keeps optional TOLBERT/modeling support behaviorally available without making direct modeling imports the architectural boundary of the runtime core.

The canonical task loop for this repository is:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

Current code mapping:

- `observe`: inspect task contract, workspace, prior steps, and verifier-visible state
- `estimate state`: build latent state and transition summaries from retained state-estimation controls
- `retrieve memory`: graph memory, retrieval spans, skills, and replay-backed context
- `update world model`: refresh workspace and world-model summaries before and after steps
- `simulate likely transitions`: score stop and command candidates through world-model rollout
- `plan candidates`: maintain subgoals and planner recovery artifacts
- `choose via policy`: route through planner, executor, or critic policy paths
- `execute`: apply pre-execution universe governance, then run the chosen bounded command in the sandbox when not blocked
  Shared-repo-gated, task-scoped git mutations such as integrator merge steps are now treated as allowed bounded workflow actions rather than blanket `git_write_conflict` violations, while destructive git patterns remain blocked
- `verify`: apply deterministic verifier checks against the task contract
- `critique`: attach verifier/subgoal diagnoses and critic recovery state
- `update memory/models`: persist episode memory and learned candidates, then continue until termination

For the current unattended evidence-ranked runtime blockers, see
[`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md).

## Syntax Awareness Boundary

The current runtime now has a real syntax-motor surface, but it is still not a fully syntax-native
coding runtime.

- The live coding path is mostly built around retained text previews, command execution, verifier feedback, workflow state, and bounded structured-edit proposals.
- [`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py) can assemble localized `structured_edit:*` actions from retained preview windows, including multi-window and bridge-backed repairs, but much of that path is still text-region control rather than full AST/CST-native transformation.
- [`agent_kernel/syntax_motor.py`](/data/agentkernel/agent_kernel/syntax_motor.py) provides parser-backed edit targeting and syntax checks for structured edits, but it remains a motor aid rather than the whole software-work reasoning stack.
- The repo also contains Python AST usage in [`scripts/report_prompt_diagnostics.py`](/data/agentkernel/scripts/report_prompt_diagnostics.py), though that path is diagnostic rather than the main edit/execution loop.

This boundary matters, but it should be described precisely. Syntax-aware tooling is not the same thing as software understanding; human engineers mostly operate through broader semantic context, repo traversal, and workflow reasoning rather than explicit AST manipulation. For a digital agent, though, syntax-aware structure is still a useful motor aid. It can make symbol-targeted edits, import propagation, signature-aware repair, edited-region accounting, and pre-execution syntax validation cheaper and more reliable than repeated text traversal alone.

So the honest current posture is:

- the kernel already has nontrivial software-task cognition through context packing, retained workflow state, verifier-guided recovery, and bounded edit planning
- the kernel now has an early first-class syntax motor for structured edits
- the kernel still does not have a fully syntax-native end-to-end software runtime
- stronger repo execution, trust breadth, and verifier-backed recovery still matter more than syntax support alone

Step budgets are now layered instead of only inheriting the schema default:

- `TaskSpec.max_steps` remains the task-local baseline.
- built-in frontier tasks and memory-derived frontier tasks now serialize with a deeper contract ladder instead of defaulting every repo-scale task to `5`
- the loop clamps every task to `max_task_steps_hard_cap`
- frontier families such as `repo_sandbox`, `repository`, `integration`, `project`, and `tooling`, plus tasks tagged `difficulty=long_horizon` or `horizon=long_horizon`, automatically floor to `frontier_task_step_floor`
- task metadata can request a deeper explicit floor with `step_floor`, which still remains bounded by `max_task_steps_hard_cap`
- older steps now compact into `history_archive` metadata while the live state, payload path, and checkpoint path retain only a bounded recent window; that keeps deeper runs resumable without linearly growing prompt/checkpoint surfaces
- unattended campaigns can now carry a first-class `task_step_floor` policy dimension, and the parent runner threads that intent into child rounds with `AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR`
- unattended stop logic now waits for the full round-level depth signal before incrementing `no_yield_rounds` or `policy_stall_rounds`, so productive deep continuation is not mistaken for outer-loop stall while depth drift still is
- unattended stop limits are now adaptive as well: productive deep rounds earn bounded `depth_runway_credit`, ambiguous rounds spend it, and drift or failed retained gates zero it before `safe_stop` checks

That keeps seed tasks short by default while giving frontier tasks a real path to `50+` step rollouts. To reach `500` or `5000` steps intentionally, raise `AGENT_KERNEL_MAX_TASK_STEPS_HARD_CAP` and set task metadata or retained worker-depth controls accordingly.

Unattended task reports now also expose a first-class `supervision` summary for
the bridge between bounded competence and real delegated software work.
[`agent_kernel/preflight.py`](/data/agentkernel/agent_kernel/preflight.py) now
records `mode`, `operator_turns`, `independent_execution`, and
`light_supervision_*` outcomes per task report, and
[`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py) aggregates
those counts into family and overall trust summaries. That lets the kernel
measure whether a family is succeeding under objective-only or near-objective-only
guidance instead of treating all gated evidence as equally autonomous.
The task-selection path now uses the same bridge target proactively instead of
only measuring it after execution:
[`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
annotates verifier-clean primary contracts as `light_supervision_candidate`,
and [`evals/harness.py`](/data/agentkernel/evals/harness.py) biases bounded
compare selection toward those contract-clean tasks before retrieval or replay
tails when slots are scarce.
That same contract bias now continues into generated follow-ons too:
[`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
now gives adjacent-success seed priority to episodes that came from
`light_supervision_candidate` primaries, and the generated curriculum tasks are
re-annotated with the same contract metadata so later generated waves and
reports do not lose the supervision signal. The generated-seed scheduler now
also preserves the long-horizon structure around that bias: repeated
shared-repo integrator variants are diversified behind the first seed in a
bundle, complete shared-repo worker bundles stay together ahead of unrelated
tails, and late-wave gapfill seeds that expand stage/family coverage are not
crowded out by same-family phase transitions.
The same bridge now applies on failure waves too:
`failure_recovery` seed selection prefers failures from contract-clean primary
tasks over replay or retrieval-heavy tails, and the selected batch is
diversified across concrete recovery surfaces so one repeated file-recovery
pattern does not crowd out repository, workflow, or other verifier-visible
repair work in later generated waves.
That same contract-clean failure path now also survives into execution and
accounting instead of stopping at seed choice: generated failure tasks stamped
from light-supervision-clean primary failures now carry an explicit deeper
recovery step floor and budget floor, and unattended reports / trust summaries
count those runs separately as contract-clean failure-recovery evidence. That
lets the recovery wave run deeper and also makes it measurable whether those
deeper follow-ons are turning into real unattended clean-success breadth.
The eval harness now exposes that same lane directly in compare output:
`contract_clean_failure_recovery_summary` and the corresponding per-origin-
family breakdown track whether those deeper generated recovery tasks are
actually improving unattended long-horizon pass rate instead of just inflating
generic `failure_recovery` counts.
The supervisor now uses the same bridge for real rollout widening too instead
of requiring global trust posture to clear first:
[`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
can promote eligible non-protected lanes from `dry_run/compare_only` into
`canary` when the lane's required families have enough
`light_supervision_clean_success` evidence, with contract-clean
failure-recovery clean successes allowed to assist the threshold. Broad rollout
still requires trusted overall posture, but canary widening no longer waits on
unrelated global bootstrap debt when the lane-local evidence is already strong.

## Long-Horizon Orientation

Retained prompt-policy artifacts can now orient the runtime toward longer-horizon coding success instead of only short-step completion.
The `long_horizon_success` focus biases policy and rollout control toward:

- smaller reversible edits over optimistic rewrites
- preservation of already-working artifacts and preserved paths
- validation-backed stopping rather than early termination
- longitudinal retained evidence instead of single-run wins
- model-side Tolbert rollout and hybrid scoring that explicitly rewards longer-horizon progress signals
- learned world-prior conditioning that raises horizon bias on longer future profiles
- retained transition-model signatures and proposal controls that distinguish long-horizon stall patterns from bounded tasks
- training datasets and autobuild readiness that weight and count long-horizon supervision explicitly instead of treating it as generic task mass
- liftoff and retained-vs-baseline comparison slices that require explicit non-regression on long-horizon subsets before promotion

The single current kernel-gap dashboard now lives in
[`docs/kernel_gap_dashboard.md`](/data/agentkernel/docs/kernel_gap_dashboard.md).
Use it first when you want the current blocker ranking, then open
[`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
for the live bottleneck detail and
[`docs/live_evidence_proof_gap_audit.md`](/data/agentkernel/docs/live_evidence_proof_gap_audit.md)
for the proof-state detail.

When a retained `tolbert_model_bundle` exposes a hybrid runtime with world-model
support, the live loop now also probes that retained runtime before the first
step and after each state transition, then blends the returned learned world
signal into `latent_state_summary` instead of leaving that feedback path model-side only.
The live decoder and rollout scorer now also use that learned world signal
directly for long-horizon control: high-risk learned states suppress premature
stop decisions and add explicit bias toward corrective progress commands.
Role specialization now also reads the same signal, so a first risky long-horizon
stall can escalate into planner mode before the kernel burns extra executor loops,
while repeated risky failures still escalate to critic mode.
When planner mode activates under that high-risk signal, the loop now refreshes
the subgoal stack around ranked learned-world hotspots instead of blindly
keeping the stale original order. That ranking is no longer gated only by
`active_paths`: moderate-risk long-horizon states can now surface pending
workflow paths, missing workflow reports, pending generated artifacts,
preserved-path regressions, forbidden artifacts, and unsatisfied expected
artifacts directly from world-state summary plus the latest transition. The
same ranked hotspot summary now also feeds loop-level recovery role handoff, so
a stalled long-horizon executor can preempt back into planner mode, and
repeated hotspot pressure can escalate to critic mode, without waiting for the
older stricter learned-risk gate.
Critic turns now also attach explicit failure diagnoses to those hotspot subgoals,
so planner re-entry sees not just which path is risky but whether it stalled,
regressed, or failed at command execution.
Those diagnoses now also feed the live deterministic policy path: planner and
critic command scoring treat the diagnosed hotspot path as first-class evidence,
and critic mode will attempt a safe diagnosis-guided repair command before
giving up to generic synthesis or termination.
When that deterministic repair is already clear from state, planner and critic
recovery turns can now take the `plan_progress_direct` path before Tolbert
context compilation, which cuts `context_compile` cost on repeated long-horizon
repair steps instead of paying retrieval startup on every continuation turn.
That recovery path is now stricter about loop avoidance as well: previously
failed task-contract commands are skipped during active recovery, and critic
mode now terminates deterministically once the remaining task-contract repair
set is exhausted instead of paying another low-yield synthesis/context turn.
Planner mode now also defers fresh recovery synthesis until that exhaustion is
critic-proven: before that point planner still uses the bounded deterministic
repair surface, but once the critic has exhausted it the planner prompt stops
re-ranking the same stale repair commands and instead asks for a rewritten
verifier-facing subgoal or recovery contract before picking the next command.
That rewrite is now carried in live kernel state as a compact planner-recovery
artifact rather than existing only inside one prompt: the loop records the
critic-exhausted source subgoal, stale commands, focus paths, and a short
rewrite outline, preserves that artifact through checkpoints, and feeds it back
into planner payloads on later turns so harder long-horizon tasks can keep
reasoning from the rewritten objective instead of rediscovering it each step.
For workflow/repository tasks that artifact is now broader than one failed path:
it can bundle pending verifier-visible obligations across changed paths, report
outputs, generated artifacts, required merges, branch targets, and named test
checks into one rewritten planner objective, which gives the planner a more
global recovery target on less templated tasks.
That broader recovery state is now staged as an explicit plan update instead of
an unordered side note: the loop scores related verifier obligations against
live progress and recent attempt pressure, records a ranked recovery agenda plus
the next staged objective in the planner artifact, and the compact planner
payload now promotes that staged plan update into the model-facing `plan`
surface. Retained Tolbert hybrid scoring also reads the same ranked agenda and
adds a recovery-stage alignment term, so model-native primary routing can
prefer commands that advance the current staged verifier obligation rather than
only inheriting wrapper-side control flow.
Long-horizon software work now has a broader model-facing agenda even before
critic-proven recovery: [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
derives a `software_work_plan_update` from remaining subgoals, pending
synthetic edit-plan paths, and delayed verifier obligations such as reports,
generated artifacts, merges, and tests; [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
threads that agenda into planner payloads and state-context chunks; and
[`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
adds a software-work alignment term so retained Tolbert scoring can favor
commands that advance the current implementation, migration, testing, or
follow-up stage rather than scoring only local repair signals.
That agenda is now outcome-conditioned too instead of being a static list:
[`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py) persists
retained software-work stage outcomes across attempts and checkpoints,
including whether the current long-horizon objective advanced, stalled,
regressed, or completed; the agenda reorder now uses those retained outcomes so
repeatedly stalled implementation work yields priority to other verifier-visible
obligations such as reports or tests; [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
surfaces the same stage-state and recent outcomes in planner payloads; and
[`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
adds an explicit software-work transition term so retained model-side ranking
can penalize brittle repeats on stalled stages while still favoring commands
that continue an advanced stage.
The same retained lane now also derives explicit phase handoff boundaries across
implementation, migration, test, and follow-up-fix work: [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
compacts staged objectives into a `software_work_phase_state` with
`current_phase`, `suggested_phase`, and `handoff_ready`; [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
surfaces that phase state and phase chunks to planner payloads; [`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
adds a software-work phase-handoff alignment term so retained scoring can
penalize premature jumps into later phases and reward test/follow-up commands
once earlier phases are finished enough; and [`agent_kernel/modeling/training/universal_dataset.py`](/data/agentkernel/agent_kernel/modeling/training/universal_dataset.py)
now emits phase-state hints in long-horizon decoder prompts so training examples
teach when to hand off, not just which obligation names exist.
Unresolved earlier phases can now also act as an explicit phase gate instead of
just a weak ordering hint. [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
derives a `software_work_phase_gate_state` that names the unresolved gate phase,
the concrete gate objectives, and the later phases that should stay blocked;
[`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
promotes that gate state into planner payloads and context chunks;
[`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py) uses the
same gate to filter task-contract direct-path candidates toward unresolved
branch-acceptance and migration work; and
[`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
adds a software-work phase-gate term so retained model-native ranking can
penalize premature test/report moves while a merge-acceptance gate is still
open.
The same long-horizon lane now also carries a compact
`campaign_contract_state` so deep runs do not have to reconstruct the whole
campaign objective from the most recent subgoal. [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py)
derives unresolved anchor obligations, regressed or stalled objectives,
required paths, and aggregate drift pressure from retained plan, world, and
stage state; [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
surfaces those anchors as planner-visible context chunks and payload state;
[`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py) adds a
campaign-contract command bias so planner and critic turns reward commands that
reattach to anchor obligations and penalize off-contract mutations under drift
pressure; and [`agent_kernel/modeling/tolbert/runtime.py`](/data/agentkernel/agent_kernel/modeling/tolbert/runtime.py)
adds model-native campaign-contract alignment and drift-penalty terms so the
retained scorer can keep a long run attached to its unresolved obligations as
variables compound.
Eval and promotion now track whether those kernel-side signals actually improve
durable behavior instead of only improving headline pass rate.
[`evals/harness.py`](/data/agentkernel/evals/harness.py) summarizes
`long_horizon_persistence_summary` from step trajectories and
`transfer_alignment_summary` from learned `trusted_retrieval_alignment`,
`graph_environment_alignment`, and `transfer_novelty` signals, and
[`agent_kernel/modeling/evaluation/liftoff.py`](/data/agentkernel/agent_kernel/modeling/evaluation/liftoff.py)
uses those summaries in the retained liftoff gate. A candidate can now be
blocked for regressing productive long-horizon persistence, increasing
horizon-drop behavior, or getting less environment-aligned on unfamiliar
transfer steps even if raw pass rate still improves.
For the remaining long-horizon continuation turns that still need Tolbert
context, the policy now reuses the current `ContextPacket` when the
retrieval-relevant state fingerprint has not changed, so role or planner-state
updates alone do not force a fresh `context_compile`.
Trusted retrieval carryover now also survives the live payload path instead of
stopping at graph-memory summaries: retrieval-backed success counts and compact
trusted command histograms are preserved in the compacted `graph_summary`,
planner and critic context can see those commands as explicit state chunks, and
long-horizon continuation turns can reuse an aligned trusted retrieval-backed
repair command before `context_compile` when the active subgoal and world state
still point at the same hotspot. On missing-artifact recovery turns, that same
carryover path can now synthesize the current `printf %s ... > path` repair
from a trusted retrieval-backed write pattern, but it stays conservative and
defers to the workspace-preview structured-edit path when an existing file
should be repaired in place instead of overwritten wholesale. Eval summaries now
also persist verified `trusted_retrieval_carryover_direct` repair counts per
task, and the retrieval-improvement loop uses successful long-horizon carryover
repairs as stronger evidence than raw retrieval selection when deciding whether
to stay in activation bootstrap or rebuild around successful trace procedures.
Carryover is no longer limited to isolated commands either: graph memory now
also compacts short trusted retrieval-backed write/verify procedures, planner
and critic payloads surface them as `trusted_retrieval_procedures`, and the
live policy can continue the next verifier/report step of a previously
successful procedure before paying another `context_compile`.
Graph memory now also conditions live kernel behavior under unfamiliar
execution envelopes instead of only feeding reports: the policy preserves
dominant observed environment modes and repeated environment-alignment failures
in the compact `graph_summary`, and direct command scoring uses that evidence to
reward reconnaissance and verifier-backed commands while penalizing git/network
mutations that conflict with historically trusted environments. That gives the
kernel a more conservative first move when transfer lands in a different
network/git/write regime instead of treating every environment as equally known.
That same evidence now reaches the Tolbert hybrid scorer too: model-side
candidate ranking adds explicit bonuses for trusted retrieval-aligned repair
commands, bonuses for trusted retrieval procedure-stage continuation, and
explicit penalties for environment-conflicting mutations under transfer
novelty, so trusted carryover and unfamiliar-environment caution are no longer
only wrapper-side heuristics.
Those signals now also enter the learned hybrid-training path rather than
stopping at runtime scoring: [`agent_kernel/modeling/training/hybrid_dataset.py`](/data/agentkernel/agent_kernel/modeling/training/hybrid_dataset.py)
now preserves trusted-retrieval alignment, graph-environment alignment, and
transfer novelty inside per-step scalar features, example weights, world
targets, and dataset manifests. That means retained checkpoints can be trained
toward trusted repair carryover and transfer-safe first moves instead of only
being hand-steered by wrapper logic at inference time.
The planner also now treats failed verifier reasons as first-class long-horizon
hotspots rather than passive logs: missing git diff paths, report failures,
generated-artifact failures, unresolved conflicts, and required-branch failures
get promoted back into concrete workflow subgoals with verifier-sourced
diagnoses. Workflow branch setup and merge-acceptance subgoals also now have
real completion checks from recent command history, so repo-scale plans can
advance past branch setup instead of stalling on already-completed branch steps.

Use [`scripts/propose_prompt_update.py`](/data/agentkernel/scripts/propose_prompt_update.py) with
`--focus long_horizon_success` when you want the retained prompt-policy surface to explicitly optimize for durable coding progress.

## Retained Preview Windows

The bounded coding path now relies on retained workspace previews rather than jumping straight to full-file rewrites. For unsatisfied expected files, [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py) can expose `workspace_file_previews[path]` with:

- `edit_windows` for multiple retained localized windows in one large file
- `bridged_edit_windows` for compact hidden-gap proof between consecutive retained windows when the full merged region would be too large to keep as one preview
- `bridged_edit_window_runs` for canonical maximal consecutive bridge runs, with `bridge_segments` carrying ordered per-gap hidden current/target proof
- `retained_edit_window_count` and `total_edit_window_count` for whether all localized change windows fit in preview budget
- `partial_window_coverage` when the world model had to drop some change windows before policy context
- `line_start` and `line_end` for the current visible span
- `target_line_start` and `target_line_end` for the expected target span
- `line_delta` for the visible line-count change implied by that window
- `edit_content` and `target_edit_content` for the current and target window payloads

[`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py) consumes that metadata to prefer bounded structured edits, including bundled `structured_edit:multi_edit` commands across multiple retained windows. The decoder only batches windows when the retained target spans make the composition safe; exact target-span evidence is required for line-shifting chains such as delete-then-replace or insert-then-edit. When overlapping retained windows cannot be bundled safely as a `multi_edit` but their visible current/target spans can be stitched into one consistent contiguous region, the decoder can now emit a localized `structured_edit:block_replace` over that merged range instead of jumping directly to a whole-file rewrite. It also preserves exact single-window `block_replace` alternatives alongside cheaper token or line edits for the same retained window, and it now synthesizes exact contiguous-region `block_replace` candidates across adjacent retained windows when the visible current and target lines merge without hidden gaps. That exact-region path now extends two steps further: first, consecutive exact retained windows can widen into one larger exact hidden-gap `block_replace` by slicing the expected target over the bounded unseen region; second, when the world model emits `bridged_edit_windows` with explicit hidden-gap current proof, the decoder can safely bridge a larger bounded region even across mixed-precision retained windows instead of falling back to a fragmented plan or a bounded full rewrite. Those bridge-backed regions now also cover hidden-gap structural drift and consecutive bridge chains: explicit bridge payloads can insert or delete lines inside the unseen gap, and multiple adjacent bridge records can widen one bounded `block_replace` across more than two retained windows as long as every intermediate bridge is explicit and consecutive. The new `bridged_edit_window_runs` payload makes that bridge frontier canonical instead of pairwise-only: the world model now emits maximal consecutive bridge runs with ordered per-gap segments, and the decoder can consume the run directly instead of reconstructing longer chains itself. Proposal scoring now also treats bridge runs as their own frontier family: full-coverage maximal runs can beat shorter bridge runs cleanly, while partial retained bridge runs can lose to smaller exact localized `block_replace` repairs instead of winning by window count alone. The retained-vs-total window counts remain an explicit coverage signal, so partial retained coverage can still down-rank a preview bundle relative to a bounded full write instead of blindly assuming the visible windows cover the whole outstanding file delta.

## Providers

Supported providers are:

- `ollama`
- `vllm`
- `model_stack`
- `mock`

`ollama`, `vllm`, and `model_stack` are live paths. `mock` is mainly used by tests and skips retrieval-required tasks in eval.

## Provider behavior

For `ollama`, the client sends `think: false` to avoid structured output being placed only in Ollama’s thinking field. The parser still accepts JSON from either `response` or `thinking` as a fallback.

For `model_stack`, Agent Kernel talks to the local Model Stack serving API: `/healthz` for preflight and `/v1/generate` for token generation. Because that API is token-based rather than chat-compatible, configure a tokenizer root through `AGENT_KERNEL_MODEL_STACK_TOKENIZER_PATH` or `AGENT_KERNEL_MODEL_STACK_MODEL_DIR`; the client loads `data.tokenizer.get_tokenizer` from `AGENT_KERNEL_MODEL_STACK_REPO_PATH`.

Useful LLM settings:

- `AGENT_KERNEL_LLM_TIMEOUT_SECONDS`
- `AGENT_KERNEL_LLM_RETRY_ATTEMPTS`
- `AGENT_KERNEL_LLM_RETRY_BACKOFF_SECONDS`
- `AGENT_KERNEL_LLM_PLAN_MAX_ITEMS`
- `AGENT_KERNEL_LLM_SUMMARY_MAX_CHARS`
- `AGENT_KERNEL_OLLAMA_HOST`
- `AGENT_KERNEL_VLLM_HOST`
- `AGENT_KERNEL_MODEL_STACK_HOST`
- `AGENT_KERNEL_MODEL_STACK_MODEL_DIR`
- `AGENT_KERNEL_MODEL_STACK_TOKENIZER_PATH`
- `AGENT_KERNEL_MODEL_STACK_REPO_PATH`
- `AGENT_KERNEL_VLLM_API_KEY`
- `CUDA_VISIBLE_DEVICES`

## TOLBERT family context

The compiler path is:

1. send the current query to [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py)
2. embed and predict per-level logits with `tolbert-brain`
3. greedily decode a constrained path
4. select a branch depth based on confidence
5. retrieve branch-scoped, fallback-branch, and global evidence
6. compile a `ContextPacket` consumed by the policy

Lifecycle note:
- `TolbertServiceClient` owns the `scripts/tolbert_service.py` subprocess for a live kernel.
- `AgentKernel.close()` shuts that subprocess down through the policy/context-provider stack.
- `run_eval()` closes every kernel it creates, including generated-task and failure-seed kernels, so long improvement cycles do not leave stale TOLBERT workers behind.

Live configuration includes:

- `AGENT_KERNEL_TOLBERT_PYTHON_BIN`
- `AGENT_KERNEL_TOLBERT_CONFIG_PATH`
- `AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH`
- `AGENT_KERNEL_TOLBERT_NODES_PATH`
- `AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS`
- `AGENT_KERNEL_TOLBERT_CACHE_PATHS`
- `AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH`
- `AGENT_KERNEL_TOLBERT_DEVICE`
- `AGENT_KERNEL_TOLBERT_BRANCH_RESULTS`
- `AGENT_KERNEL_TOLBERT_GLOBAL_RESULTS`
- `AGENT_KERNEL_TOLBERT_TOP_BRANCHES`
- `AGENT_KERNEL_TOLBERT_ANCESTOR_BRANCH_LEVELS`
- `AGENT_KERNEL_TOLBERT_MAX_SPANS_PER_SOURCE`
- `AGENT_KERNEL_TOLBERT_CONTEXT_MAX_CHUNKS`
- `AGENT_KERNEL_TOLBERT_CONTEXT_CHAR_BUDGET`
- `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS`
- `AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD`
- `AGENT_KERNEL_TOLBERT_BRANCH_CONFIDENCE_MARGIN`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_WIDEN_THRESHOLD`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_BRANCH_MULTIPLIER`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_GLOBAL_MULTIPLIER`
- `AGENT_KERNEL_TOLBERT_DETERMINISTIC_COMMAND_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_FIRST_STEP_DIRECT_COMMAND_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_DIRECT_COMMAND_MIN_SCORE`
- `AGENT_KERNEL_TOLBERT_SKILL_RANKING_MIN_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_SEMANTIC_SCORE_WEIGHT`
- `AGENT_KERNEL_TOLBERT_TASK_MATCH_WEIGHT`
- `AGENT_KERNEL_TOLBERT_SOURCE_TASK_WEIGHT`
- `AGENT_KERNEL_TOLBERT_DISTRACTOR_PENALTY`

If `AGENT_KERNEL_TOLBERT_DEVICE=cuda` is set and CUDA is not actually visible in the TOLBERT env, the service fails fast.

## TOLBERT modes

[`scripts/run_eval.py`](/data/agentkernel/scripts/run_eval.py) exposes comparison modes for:

- `full`
- `path_only`
- `retrieval_only`
- `deterministic_command`
- `skill_ranking`

Those are eval-time analysis lanes, not separate product surfaces.

## Universal family definition

In this repo, seeded `TOLBERT` refers to the original ontology
encoder/retrieval/compiler path. The retained modeled runtime lane is the
`hybrid` path, which currently carries these coordinated surfaces:

- encoder surface for hierarchy-aware representation and retrieval
- latent/state-space surface for long-horizon task state
- decoder surface for bounded and eventually free-form action generation
- world-model surface for transition and risk forecasting
- policy/value/stop/risk heads that plug into kernel routing and acceptance

The current live service path in [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
is the seeded compiler path. The hybrid runtime under
[`agent_kernel/modeling/tolbert/`](/data/agentkernel/agent_kernel/modeling/tolbert)
is the first retained latent/runtime member. The seed `vllm` stack remains
active until retained hybrid checkpoints earn authority through the liftoff
gate.

`run_eval.py` also exposes memory/eval lanes that the older docs did not list:

- `--include-skill-transfer`
- `--include-operator-memory`
- `--include-benchmark-candidates`
- `--include-verifier-candidates`
- `--compare-abstractions`

## State and step records

Task state and stored steps carry more than the earlier docs described. A step record can include:

- selected skill ID
- retrieval candidate and evidence counts
- retrieval-selected span ID
- whether retrieval influenced the action
- path confidence
- whether retrieval was trusted
- active subgoal
- acting role
- world-model horizon

Episode documents also persist:

- task contract
- task metadata
- initial plan
- graph summary
- world-model summary
- termination reason

## Paths and artifacts

By default the runtime writes to:

- `workspace/`
- `trajectories/episodes/`
- `trajectories/skills/command_skills.json`
- `trajectories/operators/operator_classes.json`
- `trajectories/tools/tool_candidates.json`
- `trajectories/benchmarks/benchmark_candidates.json`
- `trajectories/verifiers/verifier_contracts.json`
- `trajectories/retrieval/retrieval_proposals.json`
- `trajectories/retrieval/retrieval_asset_bundle.json`
- `trajectories/prompts/prompt_proposals.json`
- `trajectories/curriculum/curriculum_proposals.json`
- `trajectories/improvement/cycles.jsonl`
- `trajectories/improvement/reports/`

## CLI entrypoints

Single task:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_agent.py --task-id hello_task
```

Eval:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_eval.py
```

Selfplay:

```bash
python scripts/run_selfplay.py --seed-task-id hello_task --seed-mode success
```

Native wrappers:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/run_native_agent.sh hello_task
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/run_native_eval.sh
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/verify_impl.sh
```
