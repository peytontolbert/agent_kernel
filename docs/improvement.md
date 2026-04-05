# Improvement

## Purpose

The repo includes an artifact-driven improvement loop on top of the core runtime. It uses eval metrics to decide which subsystem needs work, generates an artifact for that subsystem, and records cycle state transitions.

The canonical outer self-improvement loop for this repository is:

`run tasks -> collect outcomes -> compare against verifiers -> localize failure -> modify policy/planner/world-model/transition-model/prompts/tools -> retest -> retain only verified gains`

In repo terms:

- `run tasks`: execute bounded eval or unattended campaign rounds
- `collect outcomes`: record metrics, reports, family-level deltas, and cycle history
- `compare against verifiers`: compare baseline and candidate lanes under verifier contracts and retention gates
- `localize failure`: identify weak subsystems, regressed families, failure motifs, and trust gaps
- `modify ...`: generate candidate artifacts for policy, retrieval, verifier, world model, state estimation, transition model, prompts, curriculum, delegation, recovery, trust, and related control surfaces
- `retest`: rerun candidate-vs-baseline evaluation and confirmation paths
- `retain only verified gains`: finalize retained artifacts, otherwise reject and roll back

Seed and micro tasks still exist as smoke tests, but the intended optimization target is real-world local coding work: repository edits, git workflows, multi-file changes, tooling repair, integration repair, and long-horizon validation.

Primary subsystems are:

- `benchmark`
- `skills`
- `operators`
- `tooling`
- `verifier`
- `retrieval`
- `policy`
- `curriculum`

Static improvement schema metadata and retained default surfaces now live in
[`datasets/improvement_catalog.json`](/data/agentkernel/datasets/improvement_catalog.json),
which is loaded by [`agent_kernel/improvement_catalog.py`](/data/agentkernel/agent_kernel/improvement_catalog.py).
That dataset currently backs subsystem schema keys and default catalogs for improvement compatibility,
artifact-validation profiles, universe governance metadata, operator-policy defaults, transition-model parsing metadata,
and task-budget floors. It now also carries the artifact contract registry and default retention-gate presets used by
[`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
and [`agent_kernel/improvement_common.py`](/data/agentkernel/agent_kernel/improvement_common.py).
Broader kernel registries that are not specific to the improvement planner now live in
[`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json),
which backs subsystem registry entries, capability adapter catalogs, task-bank lineage filters,
unattended-controller feature catalogs, Tolbert build and proposal-family defaults, trust-family defaults,
unattended runtime policy defaults, and frontier-family step-floor defaults.

## Artifact outputs

Configured paths in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py):

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

## Cycle flow

[`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py) currently performs:

1. `observe`: run eval and record metrics
2. `select`: rank experiments, assemble a recent-history-aware portfolio campaign, and pick a variant
3. `generate`: snapshot any active artifact and write the corresponding candidate artifact
4. `evaluate`: compare baseline and candidate lanes
5. `retain` or `reject`: apply the retention gate, record compatibility, and restore the prior active artifact on rejection
6. `record`: append the final lifecycle outcome to cycle history

Cycle IDs are timestamped experiment IDs, for example `cycle:skills:20260319T123456789012Z:1a2b3c4d`.

By default the script runs the whole cycle end to end. Use `--generate-only` when you want to stop after artifact generation and finalize later.
Candidate stamping, replay-verified tooling updates, retention finalize writes, rollback restores, rollback receipts, and Tolbert liftoff gate reports now all flow through the runtime supervision atomic helpers, so non-protected lanes can perform real governed retention and rollback actions instead of only compare-only bookkeeping. The supervisor now also consumes candidate-observed benchmark families from the frontier report and treats missing counted gated evidence, bootstrap-only family posture, and missing clean task-root breadth as machine-readable autonomy blockers for the affected retained candidates instead of only reporting those gaps in the trust ledger. Those candidate-family blockers now also feed sticky discovery priority: the next supervisor rounds bias `launch_discovery` toward `trust`/`recovery` work, widen the trust-remediation batch budget when family pressure is higher, and forward both `--priority-benchmark-family` hints and per-family weights so unattended evidence collection chases the exact missing proof instead of rerunning a fixed small probe.
That closes an important control-path gap, but it does not mean broad coding autonomy is complete. The current kernel still gets a large share of its success from bounded task design, runtime guards, direct retained-vs-candidate paths, and family-specific recovery logic. The remaining product gap is broader unattended trust evidence and wider repeated proof on `repository`, `project`, and `integration`, so governed supervisor actions rest on deeper unattended evidence instead of narrow bootstrap-era coverage.
The coding path now has parser-backed syntax-motor support in [`agent_kernel/syntax_motor.py`](/data/agentkernel/agent_kernel/syntax_motor.py), but it still does not have a fully syntax-native end-to-end coding loop. The larger unattended blocker is conversion into broad runtime-managed coding gains, not mere parser absence. For the current evidence-ranked gap order, see [`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md).
Use `--adaptive-search` to let retained-history and score concentration widen campaign or sibling-variant search up to the requested width caps.
Campaign selection also uses recent cycle history so near-tied subsystems can rotate instead of letting one recently saturated subsystem dominate repeated autonomous runs.
For short bounded experiments, `--task-limit` and repeated `--priority-benchmark-family` flags let you constrain the observation pass before candidate generation.
When you care about real coding capability rather than smoke-test stability, prefer `repo_sandbox`, `repository`, `integration`, `tooling`, and `project` families in that order.
For parallel bounded autonomous previews, pair `--scope-id <runner-id>` with `--generate-only` so workspace, cycle history, candidate artifacts, reports, checkpoints, and learning artifacts stay isolated while the run still reads the shared retained artifact and replay-memory baseline. That `--generate-only` default is still the right worker discipline for parallel lanes, but it is no longer evidence that the underlying retention path is compare-only.

[`scripts/run_supervised_improvement_cycle.py`](/data/agentkernel/scripts/run_supervised_improvement_cycle.py) runs the supervised variant of the same loop.
It keeps the same observe/select/generate flow, but lets the operator pin or override `--subsystem`, `--variant-id`,
`--notes`, `--task-limit`, `--priority-benchmark-family`, and `--max-observation-seconds` while still using the standard planner and finalizer.
When `--max-observation-seconds` is greater than zero, the observation phase runs in a separate child process and is terminated before planning if the budget is exceeded; the supervised run records an `observe` timeout row and exits without generating an artifact unless there is remaining budget to retry once without generated curriculum.
That retry now consumes only the remaining observation budget instead of starting a fresh full-budget observation pass.
Timed-out supervised observation rows now also carry a compact partial snapshot from the child eval, including completed-task summaries when any tasks finished and the scheduled task manifest/current benchmark-family context when the active task itself consumed the whole budget.
Those partial snapshots now also include in-task step state for the active task, such as the current step index, completed-step count, and whether the task was still waiting on the first decision versus later command execution when the child was terminated.
Policy-side decision telemetry is now threaded into the same snapshot surface, so timed-out supervised runs can distinguish broad first-step delay from a narrower substage such as `context_compile`, `payload_build`, or `llm_request`.
Explicit supervised eval flags are also preserved when retained artifacts already exist on disk, so bounded observation can intentionally exclude candidate lanes instead of silently re-enabling them from file presence alone.
When bounded supervised observation injects its default priority families, it also enables low-cost task ordering so short `--task-limit` previews start with cheaper seed tasks inside those families instead of raw task-id order.
The staged follow-up budgets are now intended to be operator-facing controls: leave `--generated-curriculum-budget-seconds` and `--failure-curriculum-budget-seconds` at `0` when you want the fastest bounded observation path, or set a small positive budget when you want limited supplemental follow-up signal without reopening the full unbounded curriculum tail.
Retained curriculum behavior now also exposes an explicit coding-frontier surface through `curriculum_behavior_controls_v3`: `frontier_priority_families`, `frontier_missing_families`, and `frontier_retention_priority_families` steer adjacent-success generation toward missing coding families, harder long-horizon followups, and weak generated families that still need retained-baseline proof.
That control surface now also carries self-directed transfer pressure: `frontier_generalization_priority_families` highlights coding families where retained or observed evidence exists but generated curriculum still fails to generalize cleanly, while `frontier_outward_branch_bonus` and `frontier_lineage_breadth_bonus` bias seed scheduling toward outward cross-family followups instead of repeatedly replaying one saturated lane.
The same retained controls can now also carry concrete weakness signatures from prior coding episodes: `frontier_failure_motif_priority_pairs` records recurring `family:failure_motif` bottlenecks such as `workflow:no_state_progress`, and `frontier_repo_setting_priority_pairs` records `family:repo_setting_signature` weak spots such as `integration:worker_handoff` or `tooling:shared_repo`. The adjacent-success scheduler uses those retained pairs to spend self-generated followup budget on the repo settings and verifier-visible failure motifs that are still repeatedly failing, not only on broad family gaps.
Recent retain/reject cycle history now also feeds this same surface: `frontier_retained_gain_families` and `frontier_promotion_risk_families` are mined from recent family-level retained-baseline deltas, so the scheduler compounds harder tasks on families that already survive promotion and throttles escalation on families whose retained comparisons are still regressing.
That surface is meant to support a more open-ended self-directed coding curriculum: discover uncovered coding families, ratchet difficulty without reopening unbounded runtime tails, and keep spending generated-task budget where promotion evidence is still weak rather than repeatedly saturating already-healthy families.
For parallel bounded previews, pair `--scope-id <runner-id>` with `--generate-only` so observation, reports, cycle history, and candidate artifacts stay isolated from other supervised threads.
That scoped supervised path now also isolates Tolbert modeling previews by moving the active `tolbert_model` artifact path, derived shared-store/checkpoint parent, and liftoff report into the scope before candidate generation, while reusing the shared retained Tolbert dataset baseline instead of cloning the dataset tree into every scope.
[`scripts/run_parallel_supervised_cycles.py`](/data/agentkernel/scripts/run_parallel_supervised_cycles.py) now automates that pattern for `vllm` batching: it launches multiple scoped `--generate-only` supervised children concurrently, keeps retention disabled, streams each child's output with a stable `[parallel:<scope>]` prefix, auto-diversifies worker subsystems from planner plus recent batch outcomes when `--subsystem` is not supplied, can optionally assign per-worker `--variant-id` values with `--auto-diversify-variants`, and writes both a batch report and reusable batch-history log.
The older `run_human_guided_improvement_cycle.py` entrypoint remains for compatibility, but `run_supervised_improvement_cycle.py`
is the stable CLI to use because it no longer depends on importing a top-level `scripts` package name that may be shadowed by the Python environment.

### Supervised Observation Presets

Use one of these two presets unless you have a reason to tune the observation budget manually.

Fast bounded observation:

- use when you want the quickest operator-guided generate-only cycle
- keeps generated and failure follow-ups disabled so the run spends its budget on the primary observation slice only
- best default for parallel scoped probes and live coordination loops
- add explicit `--priority-benchmark-family` flags when the probe should stay focused on repository-scale work instead of low-cost smoke tests

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 5 \
  --max-observation-seconds 60 \
  --priority-benchmark-family repo_sandbox \
  --priority-benchmark-family repository \
  --priority-benchmark-family integration \
  --priority-benchmark-family tooling \
  --priority-benchmark-family project \
  --generate-only \
  --scope-id supervised_fast_1 \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory \
  --include-curriculum \
  --include-failure-curriculum \
  --generated-curriculum-budget-seconds 0 \
  --failure-curriculum-budget-seconds 0 \
  --progress-label supervised_fast_1 \
  --notes "fast bounded supervised iteration"
```

Bounded supplemental follow-up:

- use when you want a small amount of generated-success signal after the primary observation slice
- keeps the wall-clock bounded because the follow-up runs under its own small budget instead of reopening full curriculum expansion
- useful once the primary slice is already completing comfortably inside budget

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 1 \
  --max-observation-seconds 45 \
  --generate-only \
  --scope-id supervised_followup_1 \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory \
  --include-curriculum \
  --generated-curriculum-budget-seconds 2 \
  --failure-curriculum-budget-seconds 0 \
  --progress-label supervised_followup_1 \
  --notes "bounded generated-success follow-up probe"
```

## Finalization

[`scripts/finalize_improvement_cycle.py`](/data/agentkernel/scripts/finalize_improvement_cycle.py) compares:

- a baseline lane
- a candidate lane with the target subsystem enabled

It records whether the artifact should be retained based on subsystem-specific gates, can require confirmation runs, and stores rollback/snapshot metadata for retained or rejected artifacts.
Retained retrieval finalization also materializes a Tolbert runtime bundle manifest so runtime and preflight can follow rebuilt retrieval assets directly.

Related utilities:

- [`scripts/compare_retained_baseline.py`](/data/agentkernel/scripts/compare_retained_baseline.py): compare current artifact performance against the last retained snapshot
- [`scripts/rollback_artifact.py`](/data/agentkernel/scripts/rollback_artifact.py): restore a prior artifact snapshot
- [`scripts/materialize_retrieval_asset_bundle.py`](/data/agentkernel/scripts/materialize_retrieval_asset_bundle.py): rebuild a retained retrieval proposal into a Tolbert asset bundle manifest
- [`scripts/run_autonomous_compounding_check.py`](/data/agentkernel/scripts/run_autonomous_compounding_check.py): run repeated autonomous campaigns from isolated starting states and summarize compounding stability

## Parallel ASI Coordination

For parallel work derived from [`asi.md`](/data/agentkernel/asi.md), use the machine-readable manifest at
[`config/asi_parallel_development_manifest.json`](/data/agentkernel/config/asi_parallel_development_manifest.json).

Live AI-agent ownership and claim status now live in
[`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md).
Use that page as the current coordination source for:

- the supervised-loop coordinator and scoped-runner roles
- supervised cognitive-lane ownership for parallel Codex/ChatGPT work
- ASI worker-lane claim status
- fresh-agent handoff commands
- narrow file ownership for parallel Codex/ChatGPT contributors

Parallel supervised work also has a machine-readable lane manifest at
[`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json),
which is the source of truth for cognitive-lane boundaries, owned paths, suggested scoped-run commands, and the required claim-to-close lifecycle contract for parallel agents.
For the live unattended wave, the compact evidence-backed kernel gap map now
lives in [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
and should be used together with the unattended work queue as the first-pass
route for replica workers deciding which kernel surface to attack next.

## Typical commands

Select the next target:

```bash
python scripts/select_improvement_target.py --include-episode-memory --include-skill-memory
```

Generate one cycle:

```bash
python scripts/run_improvement_cycle.py \
  --adaptive-search \
  --variant-width 2 \
  --priority-benchmark-family repo_sandbox \
  --priority-benchmark-family repository \
  --priority-benchmark-family integration \
  --priority-benchmark-family tooling \
  --priority-benchmark-family project \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory
```

Run one bounded autonomous preview cycle in parallel with other preview cycles:

```bash
python scripts/run_improvement_cycle.py \
  --generate-only \
  --scope-id autonomous_preview_1 \
  --task-limit 5 \
  --priority-benchmark-family repo_sandbox \
  --priority-benchmark-family repository \
  --priority-benchmark-family integration \
  --priority-benchmark-family tooling \
  --priority-benchmark-family project \
  --adaptive-search \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory
```

Run one bounded supervised cycle without retention:

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 5 \
  --max-observation-seconds 60 \
  --priority-benchmark-family repo_sandbox \
  --priority-benchmark-family repository \
  --priority-benchmark-family integration \
  --priority-benchmark-family tooling \
  --priority-benchmark-family project \
  --generate-only \
  --scope-id supervised_preview_1 \
  --progress-label supervised_round_1 \
  --notes "operator-guided bounded iteration"
```

Run the fast bounded preset explicitly:

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 5 \
  --max-observation-seconds 60 \
  --priority-benchmark-family repo_sandbox \
  --priority-benchmark-family repository \
  --priority-benchmark-family integration \
  --priority-benchmark-family tooling \
  --priority-benchmark-family project \
  --generate-only \
  --scope-id supervised_fast_1 \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory \
  --include-curriculum \
  --include-failure-curriculum \
  --generated-curriculum-budget-seconds 0 \
  --failure-curriculum-budget-seconds 0 \
  --progress-label supervised_fast_1 \
  --notes "fast bounded supervised iteration"
```

Run the bounded supplemental-follow-up preset explicitly:

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 1 \
  --max-observation-seconds 45 \
  --generate-only \
  --scope-id supervised_followup_1 \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory \
  --include-curriculum \
  --generated-curriculum-budget-seconds 2 \
  --failure-curriculum-budget-seconds 0 \
  --progress-label supervised_followup_1 \
  --notes "bounded generated-success follow-up probe"
```

Run one supervised cycle with the operator pinning the target subsystem and variant:

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --subsystem retrieval \
  --variant-id confidence_gating \
  --task-limit 5 \
  --generate-only \
  --notes "inspect retrieval candidate before retention"
```

Finalize a generated artifact:

```bash
python scripts/finalize_improvement_cycle.py \
  --subsystem skills \
  --cycle-id cycle:skills:20260319T123456789012Z:1a2b3c4d \
  --artifact-path trajectories/skills/command_skills.json
```

Run multiple cycles and write a campaign report:

```bash
python scripts/run_repeated_improvement_cycles.py \
  --cycles 3 \
  --adaptive-search \
  --variant-width 2 \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory
```

Campaign reports split runtime-managed retained/rejected decisions from non-runtime or test-path decisions so artifact-path isolation stays auditable across repeated runs.
Repeated-cycle campaigns also accept `--priority-benchmark-family` so unattended rounds can reserve task-limit budget for broader required families such as `project`, `repository`, and `integration`.
Those reports now also summarize per-priority-family retained yield and estimated-cost spend so unattended selection can rank families by retained pass-rate gain per estimated cost, while still giving under-sampled families an exploration bonus instead of only using coarse under-sampled versus low-return buckets. The unattended scorer now reads the same retained `improvement_planner_controls` surface as subsystem and variant ranking, so family-level gain multipliers, cost multipliers, and score biases stay in one policy channel.
Improvement-planner policy now also separates score controls from explicit `search_guardrails`, so gain/cost/bias tuning and search-width floors or margins remain independently auditable.
Unattended family selection now also supports a narrower `priority_family_min_selection_score` guardrail: missing required families still override it, but otherwise low-score families can be dropped instead of always filling all three priority slots. Policy-shift reporting now distinguishes that guardrail-driven narrowing from cases where fewer than three ranked families were available at all.
Autonomous compounding reports now also extract comparable per-run result streams from repeated campaign reports, default to broader non-replay transfer families (`workflow`, `project`, `repository`, `tooling`, and `integration`) unless `--priority-benchmark-family` overrides them, rank those families by transfer investment over time, and reuse the latest ranking to seed the next batch’s family order and weighted task-budget allocation. Repeated campaign reports now also carry a `priority_family_allocation_summary`, and autonomous compounding rolls that up into a `priority_family_allocation_audit`, so planned family weights can be compared against realized sampled task spend instead of remaining implicit inside the eval harness. That allocation audit now also feeds back into planning: families with a material positive planned-versus-actual share gap in the latest allocation snapshot receive an explicit next-batch weight bonus, and that bonus now scales with the latest consecutive under-sampling streak so persistent under-allocation is treated as more urgent than a one-batch miss. Families whose latest run closed the gap are reported as recovered and stop carrying the old bonus forward. Material latest overshoot now also applies a smaller normalization penalty, and that penalty scales with the latest consecutive overshoot streak, so a one-batch sampling blip gets a lighter pull than a family that has stayed over-allocated across multiple batches. Both paths now also scale by allocation confidence from the latest batch, and that confidence is now family-specific and longitudinal rather than only batch-global, so a batch with plenty of total priority tasks but thin evidence for one family cannot overstate that family’s bonus or normalization, and one strong latest batch no longer fully resets a family whose recent sampling confidence was unstable. Bonus and normalization now use asymmetric longitudinal confidence blends, and the whole allocation-confidence surface is now retained prompt-policy under `improvement_planner_controls.priority_family_allocation_confidence`: minimum runs, target priority tasks, optional target family task overrides, history window, base longitudinal blend, and the bonus-versus-normalization history weights can all be tuned without changing code. The prompt-policy authoring loop now also learns those allocation-confidence settings from repeated evidence: low-confidence retrieval and weak generated-family transfer raise the required run/task evidence and make bonus expansion more conservative, while repeated command failures keep normalization more reactive. The applied bonus or normalization is recorded in `priority_benchmark_family_allocation_compensation` instead of being applied silently. Each isolated compounding run now also writes a scoped retained `curriculum_proposal_set` into its sandbox before launch, carrying `autonomous_frontier_curriculum_pressure` and derived frontier controls such as missing-family, generalization-priority, outward-branch, lineage-breadth, and harder-task bonuses. That turns prior transfer gaps into real next-run generated-task pressure instead of only reporting them after the batch. Autonomous compounding now also audits whether those pressure families were actually sampled: `frontier_expansion_summary` compares target and pressure families against realized per-family task counts, reports `families_never_sampled` and `pressure_families_without_sampling`, and feeds those unsampled pressure families back into the next batch’s frontier controls. That signal now affects the live launcher too, not just the post-run claim gate: unsampled or missing pressure families are reordered to the front of `--priority-benchmark-family`, receive a direct live weight bonus plus a pressure floor that keeps them above non-pressure families, and can raise a small task-limit floor when the requested run would otherwise be too narrow to exercise them at all. The parent wrapper now launches from that exact routed plan instead of recomputing it later in the child-launch path, so `active_run`, the forwarded child command, and the retained run manifest stay in lockstep under live pressure routing. The repeated child now uses the same signal inside one campaign too: when earlier cycles leave priority families unsampled, later cycles in that same repeated run reroute those families to the front, boost their forwarded family weights, and raise later child `task_limit` toward fuller priority-family coverage instead of waiting for the next top-level autonomous batch. Those reroutes are now only applied if another child cycle will actually run; failed child cycles still record the proposed `priority_family_rerouting` and `priority_family_budget_rerouting`, but the campaign’s final `effective_*` settings remain the ones that were truly executed. The isolated launcher now also bounds seed fingerprinting and trajectories-root seeding so autonomous runs do not spend startup time recursively hashing or copying the full historical episode tree before the first repeated-cycle child is even launched. It now also writes a stable `autonomous_compounding_status.json` while the wrapper is still running, including the active run, completed-run partial summaries, `families_sampled`, and `pressure_families_without_sampling`, so operators can inspect live frontier coverage before the full report lands. The repeated child now mirrors the same per-family live allocation counts into a local `repeated_improvement_status.json` and into `active_run.child_status` on the parent status artifact, so first-run family sampling is inspectable before the isolated campaign finishes. Claim-gate readiness is blocked if a run is missing scoped campaign record metadata, lacks runtime-managed decision streams, only shows too-narrow retained gain across those transfer families, fails to show at least one non-declining repeated retained-gain timeline across runs, only shows persistent transfer at too-low return on cost, or never actually exercises its own autonomous frontier pressure broadly enough to sample those families.

Run isolated autonomous compounding checks:

```bash
python scripts/run_autonomous_compounding_check.py \
  --runs 3 \
  --cycles 3 \
  --adaptive-search \
  --include-episode-memory \
  --include-skill-memory \
  --include-tool-memory \
  --include-verifier-memory
```

## Notes

- Skill artifacts are promoted objects with top-level metadata and a nested `skills` list.
- Operator artifacts are promoted objects with a nested `operators` list.
- Tool artifacts are candidate objects with a nested `candidates` list.
- Retrieval artifacts are `retrieval_policy_set` proposal objects with a nested `proposals` list and merged runtime `overrides`.
- Retained retrieval artifacts can also emit `tolbert_retrieval_asset_bundle` manifests that pin the rebuilt config, nodes, label map, and source spans used by Tolbert runtime.
- Benchmark and verifier artifacts are candidate sets with proposal-specific retention gates.
- Policy artifacts now carry retained decision controls, planner controls, and role directives, so they mutate both live prompting and initial plan construction.
- Retrieval, prompt, and curriculum proposal artifacts are applied by runtime feature toggles, not by direct task-bank replay.
- Replay tasks, eval comparison lanes, and retention logic consume these artifact files directly, so payload-shape drift shows up quickly in tests.
