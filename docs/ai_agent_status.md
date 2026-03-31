# AI Agent Status

## Purpose

This file is the thin coordinator index for parallel supervised Codex work on `agent-kernel`.
It is not the full run log.

Use it to answer only these questions:

1. where do I start
2. what is authoritative right now
3. where do I claim work
4. where is the shared machine-readable evidence
5. what is the current highest-ROI direction

## Current Status

- repo: `/data/agentkernel`
- primary supervised entrypoint: [`scripts/run_supervised_improvement_cycle.py`](/data/agentkernel/scripts/run_supervised_improvement_cycle.py)
- compatibility entrypoint: [`scripts/run_human_guided_improvement_cycle.py`](/data/agentkernel/scripts/run_human_guided_improvement_cycle.py)
- preferred live provider: `vllm`
- preferred model: `Qwen/Qwen3.5-9B`
- supervised coordination mode: `parallel supervised cycles`
- retention posture: `prefer --generate-only until the operator reviews the frontier`

## Authority Map

Read these in order:

1. [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md)
   This file. Start here.
2. [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
   Live human-readable claim ledger, active wave, and closeout queue.
3. [`docs/supervised_agent_runbook.md`](/data/agentkernel/docs/supervised_agent_runbook.md)
   Exact claim, run, update, and handoff procedure.
4. [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
   Machine-readable lane ownership and path boundaries.
5. [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
   Shared evidence bus and deduped frontier over scoped runs.
6. [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md)
   Human-readable historical notes and major probe outcomes.

## Start Here

If you are a fresh Codex or ChatGPT agent:

1. Read this file once.
2. Open [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md).
3. Claim exactly one runner slot and one lane.
4. Open [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json) only for your claimed lane.
5. Inspect [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json) before you run anything.
6. Follow [`docs/supervised_agent_runbook.md`](/data/agentkernel/docs/supervised_agent_runbook.md) exactly.

Do not treat this file as the place to append detailed run notes.

## Current Highest ROI

- Parallel supervised discovery is structurally supported.
- Shared evidence is now machine-readable through [`supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json).
- Frontier-to-finalize handoff now exists through [`scripts/finalize_latest_candidate_from_cycles.py`](/data/agentkernel/scripts/finalize_latest_candidate_from_cycles.py) with `--frontier-report`.
- Frontier-to-multi-candidate consolidation now exists through [`scripts/report_frontier_promotion_plan.py`](/data/agentkernel/scripts/report_frontier_promotion_plan.py).
- A first machine-owned supervisor loop now exists through [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py), with the human staying as the outer start/stop gate.
- Supervisor meta-policy is now machine-readable through [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json), so protected self-governance surfaces are gated by rollout stage instead of informal coordinator judgment alone.
- Protected-surface detection is now path-aware against the lane manifest, not only subsystem-tag based.
- Promotion passes now classify missing retained baselines as `bootstrap_first_retain` instead of generic compare failures.
- Supervisor first-retain policy is now explicit: bootstrap finalize defaults to operator review, and is only auto-allowed in `promote` mode when the bootstrap finalize policy is opened and unattended trust status is `trusted`.
- Protected first-retain candidates now need stronger evidence than ordinary trusted lanes: the supervisor only allowlists them for bootstrap finalize after the protected clean-success threshold in [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json) is met.
- The supervisor now pauses discovery on subsystems that already have bootstrap candidates pending review, which prevents duplicate scoped loops from compounding on blocked lanes.
- Paused bootstrap lanes are now classified into machine-readable remediation queues: `baseline_bootstrap`, `trust_streak_accumulation`, and `protected_review_only`.
- The supervisor now emits queue-specific remediation artifacts for those paused lanes:
  [`supervisor_baseline_bootstrap_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json),
  [`supervisor_trust_streak_recovery_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json),
  and [`supervisor_protected_review_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_protected_review_queue.json).
- The supervisor now owns a staged protected-canary lifecycle: a protected retain enters a one-round canary observation window, restricted trust triggers rollback plus automatic post-rollback validation, and promotion stays paused until the validation and trust-resume rules clear.
- Scoped supervised preview runs now keep isolated cycle history by default. New `cycles_<scope>.jsonl` files no longer inherit base `cycles.jsonl`, and frontier ingestion prefers records whose embedded `scope_id` matches the scoped filename.

Current next step:

- run ranked multi-candidate compare/promote passes from the frontier-backed promotion plan
- treat subsystems with no retained snapshot as first-retain bootstrap lanes, with protected lanes held out of bootstrap finalize until the stronger clean-success threshold is met
- keep parallel runners on scoped `--generate-only`
- treat scoped `cycles_<scope>.jsonl` as isolated preview evidence only; do not rely on inherited base cycle history
- keep protected subsystems on `compare_only` or `canary` rollout until trust streak and rollback posture justify broader finalize autonomy
- treat `supervisor_loop_status.json` as the machine-owned lane allocator, canary lifecycle ledger, and rollback planner for the next round, not just a passive status file
- require all loop results to land in the work queue and frontier, not in ad hoc prose

## Shared Artifacts

- work queue: [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
- runbook: [`docs/supervised_agent_runbook.md`](/data/agentkernel/docs/supervised_agent_runbook.md)
- lane manifest: [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
- supervisor meta-policy: [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json)
- shared frontier: [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
- frontier builder: [`scripts/report_supervised_frontier.py`](/data/agentkernel/scripts/report_supervised_frontier.py)
- frontier finalize bridge: [`scripts/finalize_latest_candidate_from_cycles.py`](/data/agentkernel/scripts/finalize_latest_candidate_from_cycles.py)
- frontier promotion plan: [`scripts/report_frontier_promotion_plan.py`](/data/agentkernel/scripts/report_frontier_promotion_plan.py)
- shared promotion plan: [`trajectories/improvement/reports/supervised_frontier_promotion_plan.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_plan.json)
- promotion execution pass: [`scripts/run_frontier_promotion_pass.py`](/data/agentkernel/scripts/run_frontier_promotion_pass.py)
- shared promotion pass report: [`trajectories/improvement/reports/supervised_frontier_promotion_pass.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_pass.json)
- supervisor loop: [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
- supervisor status: [`trajectories/improvement/reports/supervisor_loop_status.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_loop_status.json)
- supervisor history: [`trajectories/improvement/reports/supervisor_loop_history.jsonl`](/data/agentkernel/trajectories/improvement/reports/supervisor_loop_history.jsonl)
- baseline bootstrap queue: [`trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json)
- trust recovery queue: [`trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json)
- protected review queue: [`trajectories/improvement/reports/supervisor_protected_review_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_protected_review_queue.json)
- historical notes: [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md)

## Hard Rules

- one agent claims exactly one runner slot and one lane
- scoped supervised runs must use `--scope-id ... --generate-only`
- only the coordinator changes shared supervised entrypoints or runs retention/promotion
- protected meta-surfaces must follow the supervisor rollout stage and policy file, even when a candidate ranks first on the frontier
- findings go to the work queue and frontier-backed artifacts, not back into this index
- if a run ends, close it with a terminal status

## Coordinator Note

This file should stay short.
If you feel the need to append a long narrative here, put it in:

- [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md) for live claims and closeouts
- [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md) for historical analysis
- [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json) for machine-readable shared evidence
