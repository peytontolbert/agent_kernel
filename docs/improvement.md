# Improvement

## Purpose

The repo now includes a small artifact-driven improvement loop on top of the core runtime. It uses eval metrics to decide which subsystem needs work, generates an artifact for that subsystem, and records cycle state transitions.

Primary subsystems are:

- `benchmark`
- `skills`
- `operators`
- `tooling`
- `verifier`
- `retrieval`
- `policy`
- `curriculum`

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
Use `--adaptive-search` to let retained-history and score concentration widen campaign or sibling-variant search up to the requested width caps.
Campaign selection also uses recent cycle history so near-tied subsystems can rotate instead of letting one recently saturated subsystem dominate repeated autonomous runs.
For short bounded experiments, `--task-limit` and repeated `--priority-benchmark-family` flags let you constrain the observation pass before candidate generation.
For parallel bounded autonomous previews, pair `--scope-id <runner-id>` with `--generate-only` so workspace, cycle history, candidate artifacts, reports, checkpoints, and learning artifacts stay isolated while the run still reads the shared retained artifact and replay-memory baseline.

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

```bash
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 5 \
  --max-observation-seconds 60 \
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
Autonomous compounding reports now also extract comparable per-run result streams from repeated campaign reports, default to broader non-replay transfer families (`workflow`, `project`, `repository`, `tooling`, and `integration`) unless `--priority-benchmark-family` overrides them, rank those families by transfer investment over time, and reuse the latest ranking to seed the next batch’s family order and weighted task-budget allocation. Repeated campaign reports now also carry a `priority_family_allocation_summary`, and autonomous compounding rolls that up into a `priority_family_allocation_audit`, so planned family weights can be compared against realized sampled task spend instead of remaining implicit inside the eval harness. That allocation audit now also feeds back into planning: families with a material positive planned-versus-actual share gap in the latest allocation snapshot receive an explicit next-batch weight bonus, and that bonus now scales with the latest consecutive under-sampling streak so persistent under-allocation is treated as more urgent than a one-batch miss. Families whose latest run closed the gap are reported as recovered and stop carrying the old bonus forward. Material latest overshoot now also applies a smaller normalization penalty, and that penalty scales with the latest consecutive overshoot streak, so a one-batch sampling blip gets a lighter pull than a family that has stayed over-allocated across multiple batches. Both paths now also scale by allocation confidence from the latest batch, and that confidence is now family-specific and longitudinal rather than only batch-global, so a batch with plenty of total priority tasks but thin evidence for one family cannot overstate that family’s bonus or normalization, and one strong latest batch no longer fully resets a family whose recent sampling confidence was unstable. Bonus and normalization now use asymmetric longitudinal confidence blends, and the whole allocation-confidence surface is now retained prompt-policy under `improvement_planner_controls.priority_family_allocation_confidence`: minimum runs, target priority tasks, optional target family task overrides, history window, base longitudinal blend, and the bonus-versus-normalization history weights can all be tuned without changing code. The prompt-policy authoring loop now also learns those allocation-confidence settings from repeated evidence: low-confidence retrieval and weak generated-family transfer raise the required run/task evidence and make bonus expansion more conservative, while repeated command failures keep normalization more reactive. The applied bonus or normalization is recorded in `priority_benchmark_family_allocation_compensation` instead of being applied silently. Claim-gate readiness is blocked if a run is missing scoped campaign record metadata, lacks runtime-managed decision streams, only shows too-narrow retained gain across those transfer families, fails to show at least one non-declining repeated retained-gain timeline across runs, or only shows persistent transfer at too-low return on cost.

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
