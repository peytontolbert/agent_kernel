# Supervised Work Queue

## Purpose

This is the live human-readable claim ledger for parallel supervised Codex work.
Use it for:

- active claims
- handoffs
- blocked items
- closeout summaries

Do not use this file for long historical analysis.

## Current Wave

- objective: `maximize parallel supervised discovery while minimizing duplicate lane work`
- machine evidence source: [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
- consolidation queue: [`trajectories/improvement/reports/supervised_frontier_promotion_plan.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_plan.json)
- lane contract: [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
- run procedure: [`docs/supervised_agent_runbook.md`](/data/agentkernel/docs/supervised_agent_runbook.md)

## Active Focus

- `planner_selection`: frontier-driven consolidate/compare/promotion path
- `learning_memory`: make failure evidence and prior outcomes re-enter next-run decisions
- `task_ecology`: improve family mix and bounded probe coverage
- `world_state`: reduce bad first-task branching from weak state/world understanding
- `runtime_evidence`: preserve tight timeout and provider evidence where live loops still stall

## Runner Slots

| runner_slot | status | note |
| --- | --- | --- |
| `supervised_runner__codex_main` | `unclaimed` | shared coordinator slot only when entrypoints or retention path change |
| `supervised_runner__open` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_2` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_3` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_4` | `unclaimed` | general scoped worker slot |

## Lanes

| lane_id | status | owned paths summary |
| --- | --- | --- |
| `supervised_parallel_lane__parallel_orchestration` | `unclaimed` | coordination docs and manifest |
| `supervised_parallel_lane__observation_latency` | `unclaimed` | supervised timing and wrapper behavior |
| `supervised_parallel_lane__planner_selection` | `unclaimed` | planner ranking and target selection |
| `supervised_parallel_lane__learning_memory` | `unclaimed` | memory extraction and reuse |
| `supervised_parallel_lane__task_ecology` | `unclaimed` | task mix and curriculum |
| `supervised_parallel_lane__runtime_evidence` | `unclaimed` | runtime diagnostics and failure evidence |
| `supervised_parallel_lane__world_state` | `unclaimed` | loop, world model, state estimation |

## Claim Template

Copy this block when claiming work:

~~~md
### `<runner_slot>`

- runner_slot: `<supervised_runner__...>`
- agent_name: `<agent name>`
- lane_id: `<supervised_parallel_lane__...>`
- claimed_at: `<UTC timestamp>`
- scope_id: `<stable scope id>`
- progress_label: `<progress label>`
- status: `claimed`
- command:

```bash
<exact command>
```

- hypothesis: `<what this run/change is testing>`
- owned_paths:
  - `<path 1>`
  - `<path 2>`
- released_at: ``
- result: ``
- findings:
  - ``
- artifacts:
  - ``
- verification:
  - ``
- next_recommendation: ``
~~~

## Closeout Rules

- close every claim with one terminal status
- if a run times out, still close it
- if the work is handed off, set `status: handed_off`
- if no files changed, say so explicitly

## Current Coordination Notes

- refresh the frontier report before starting new loop batches
- if you are coordinating consolidation, refresh the promotion plan after refreshing the frontier
- if you are running a promotion pass, write the outcome and any baseline-readiness failures into the queue
- if the supervisor reports `canary_monitoring`, `rollback_pending`, or `rollback_validation_failed`, do not resume protected promotion outside the lifecycle gates recorded in `supervisor_loop_status.json`
- prefer frontier-backed candidate review over ad hoc cycle-file picking
- keep this file short and current; move long analysis into [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md)

### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__learning_memory`
- claimed_at: `2026-03-30T20:25:24Z`
- scope_id: `supervised_memory_codex`
- progress_label: `supervised_memory_codex`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex --notes "learning memory probe"
```

- hypothesis: `Prior failure and verifier evidence are still underweighted when choosing the next probe, so a memory-enabled scoped cycle should expose whether learning artifacts are re-entering the decision surface or remaining dead logs.`
- owned_paths:
  - `agent_kernel/episode_store.py`
  - `agent_kernel/memory.py`
  - `agent_kernel/extractors.py`
  - `agent_kernel/learning_compiler.py`
  - `agent_kernel/tolbert.py`
  - `tests/test_memory.py`
  - `tests/test_tolbert.py`
- released_at: `2026-03-30T20:31:00Z`
- result: `completed`
- findings:
  - `Scoped vLLM memory-enabled supervised run completed without timeout, but still selected retrieval/confidence_gating; that indicated the open gap was in how learned evidence re-entered future tasks rather than in observation/runtime plumbing.`
  - `Patched learning candidate matching so replay and transfer tasks receive strong matches from applicable task targets and lineage aliases like _episode_replay/_verifier_replay instead of relying on weak benchmark-family-only scoring.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_memory_codex.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex/retrieval/cycle_retrieval_20260330T202610485327Z_71447457_confidence_gating/retrieval_proposals.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex --notes "learning memory probe"`
  - `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_memory.py -k 'learning_candidates_record_and_match_memory_source or applicable_transfer or replay_task_lineage'`
  - `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_tolbert.py -k 'enriches_guidance_with_learning_candidates or surfaces_memory_source_on_learning_guidance or applicable_transfer_learning_candidates'`
  - `python -m py_compile agent_kernel/learning_compiler.py tests/test_memory.py tests/test_tolbert.py`
- next_recommendation: `Run a fresh memory-heavy scoped cycle after this patch and inspect whether episode/verifier/tool replay tasks now surface learned guidance before retrieval-only variants dominate planner output.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-03-31T02:46:11Z`
- scope_id: `supervised_ecology_codex`
- progress_label: `supervised_ecology_codex`
- status: `blocked`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex --notes "task ecology probe"
```

- hypothesis: `The frontier is over-concentrated on transition_model and observation, so a curriculum- and failure-aware ecology loop should surface a more revealing task family or candidate than another repeated context-compile branch.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-03-31T02:47:08Z`
- result: `blocked`
- findings:
  - `Scoped task-ecology loop stayed in observation instead of generating a task-ecology candidate, and terminated when the second repository task hit a 6.0-second budget overrun at the context_compile complete stage.`
  - `The scheduled mix did diversify into repository, project, bounded, and discovered-task families, but only the first repository retrieval task completed before the timeout, so the run yielded evidence about observation/runtime pressure rather than ecology candidate quality.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_ecology_codex.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex --notes "task ecology probe"`
  - `python scripts/report_supervised_frontier.py`
  - `python -m py_compile scripts/run_supervised_improvement_cycle.py`
- next_recommendation: `Route the next repair through the observation-latency lane or a reduced-task ecology retry, since this scope is still bottlenecked by observe-phase completion budgets before task-ecology generation can be evaluated.`

### `supervised_runner__open_3`

- runner_slot: `supervised_runner__open_3`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__observation_latency`
- claimed_at: `2026-03-31T02:50:12Z`
- scope_id: `supervised_latency_codex_fix`
- progress_label: `supervised_latency_codex_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 3 --max-observation-seconds 45 --generate-only --scope-id supervised_latency_codex_fix --progress-label supervised_latency_codex_fix --notes "observation latency repair"
```

- hypothesis: `The observe watchdog is misclassifying stale context-compile subphases after stage transitions, so tightening pre-step budget classification should stop false observation aborts and let bounded loops reach generation more often.`
- owned_paths:
  - `scripts/run_supervised_improvement_cycle.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `scripts/run_eval.py`
  - `tests/test_protocol_comparison.py`
- released_at: `2026-03-31T02:53:51Z`
- result: `completed`
- findings:
  - `Patched the observation watchdog so it only treats explicit prestep stages or active context-compile subphases as budgeted prestep work; stale subphases like complete no longer survive after a stage advance into llm_request, and arbitrary later stages no longer trigger the current-task prestep budget by default.`
  - `Mirrored the same guard in the autonomous observation runner and added regressions for both guided and autonomous paths.`
  - `Live validation succeeded: a fresh scoped observation-latency loop completed bounded observation without a timeout and produced a transition_model candidate instead of failing in observe.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_latency_codex_fix.jsonl`
  - `trajectories/improvement/candidates/supervised_latency_codex_fix/transition_model/cycle_transition_model_20260331T025342512029Z_9e5ac567_regression_guard/transition_model_proposals.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `scripts/run_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `tests/test_finalize_cycle.py`
- verification:
  - `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_protocol_comparison.py -k "stale_context_subphase or subphase_budget_ignores_stale or prestep_budget_stage_ignores_stale or terminates_child_when_context_compile_subphase_budget_exceeded or unclassified_prestep_budget_exceeded"`
  - `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_finalize_cycle.py -k "stale_context_subphase or context_compile_subphase_budget_exceeded"`
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py scripts/run_improvement_cycle.py`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 3 --max-observation-seconds 45 --generate-only --scope-id supervised_latency_codex_fix --progress-label supervised_latency_codex_fix --notes "observation latency repair"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `Re-run the previously blocked repository/project-scoped ecology probe or another higher-cost supervised lane now that observation no longer aborts on stale prestep classification.`
