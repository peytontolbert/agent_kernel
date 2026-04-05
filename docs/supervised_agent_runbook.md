# Supervised Agent Runbook

## Purpose

This file defines the exact operating procedure for one supervised Codex worker.
Use it with:

- [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md)
- [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
- [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
- [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)

## One Worker Contract

Each worker must:

1. claim exactly one runner slot
2. claim exactly one lane
3. use one stable `scope_id`
4. run only scoped `--generate-only` loops unless explicitly acting as coordinator
5. write closeout findings back to the work queue

That `--generate-only` rule is a coordination rule for parallel workers, not a statement that the kernel is still compare-only. Eligible non-protected lanes now have a real governed retain/canary/rollback path when the coordinator runs promotion in the appropriate mode.

## Read Order

1. [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md)
2. [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
3. [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
4. [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)

## Claim Procedure

1. Choose one `unclaimed` runner slot in the queue.
2. Choose one `unclaimed` lane in the queue.
3. Use the lane manifest to confirm owned paths.
4. Add a claim block in the queue before running anything.

Required claim fields:

- `runner_slot`
- `agent_name`
- `lane_id`
- `claimed_at`
- `scope_id`
- `progress_label`
- `status`
- `command`
- `hypothesis`
- `owned_paths`

## Execution Rules

- stay inside owned paths for your claimed lane
- treat shared supervised entrypoints as read-only unless you are the coordinator
- inspect the frontier report before deciding what to run
- prefer one bounded loop plus one repair over many blind loops
- do not leave claims open without a terminal status

## Ten-Agent Burst Mode

When ten Codex workers are available, use the dispatch targets in [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md) as the primary split:

1. `trust_repository`
2. `trust_project`
3. `trust_integration`
4. `supervisor_rollout`
5. `runtime_stability`
6. `world_state`
7. `planner_recovery`
8. `task_ecology`
9. `memory_retrieval`
10. `tolbert_runtime`

Burst-mode rules:

- claim one dispatch target and stay inside its preferred surfaces
- do not duplicate a live dispatch target unless the queue explicitly records a handoff or split
- use `supervised_runner__codex_main` for coordination only; worker claims should prefer `supervised_runner__open*`
- if your work changes the repo's measured posture, land that change in machine-readable frontier outputs and then summarize it in the queue

## Default Commands

Fresh bounded vLLM loop:

```bash
cd /data/agentkernel && \
python scripts/run_supervised_improvement_cycle.py \
  --provider vllm \
  --task-limit 5 \
  --max-observation-seconds 60 \
  --generate-only \
  --scope-id <scope_id> \
  --progress-label <scope_id> \
  --notes "parallel supervised loop"
```

Frontier refresh:

```bash
cd /data/agentkernel && \
python scripts/report_supervised_frontier.py
```

Promotion-plan refresh:

```bash
cd /data/agentkernel && \
python scripts/report_frontier_promotion_plan.py \
  --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

Promotion-pass execution:

```bash
cd /data/agentkernel && \
python scripts/run_frontier_promotion_pass.py \
  --promotion-plan trajectories/improvement/reports/supervised_frontier_promotion_plan.json \
  --limit 2
```

Trusted first-retain apply path for the coordinator only:

```bash
cd /data/agentkernel && \
python scripts/run_frontier_promotion_pass.py \
  --promotion-plan trajectories/improvement/reports/supervised_frontier_promotion_plan.json \
  --limit 2 \
  --apply-finalize \
  --allow-bootstrap-finalize
```

Coordinator rollout note:

- eligible non-protected lanes can now move past compare-only into governed finalize, one-round canary observation, rollback on trust regression, and post-rollback validation before promotion resumes
- protected lanes still keep the stricter review-only and rollout-stage gates from [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json)
- use the supervisor and promotion-pass reports as the authority for whether a lane is promotion-eligible; do not infer eligibility from the existence of a candidate artifact alone

Protected first-retain note:

- if the supervisor passes `--allow-bootstrap-subsystem`, only those subsystems are eligible for bootstrap finalize
- protected subsystems stay review-only until the stronger clean-success threshold in [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json) is satisfied
- if a subsystem already has `bootstrap_requires_review` or `bootstrap_subsystem_not_allowed` in the promotion pass report, the supervisor may pause further discovery on that subsystem until the review gate changes
- remediation queue meanings:
  `baseline_bootstrap` means the lane has a first-retain candidate and needs explicit baseline/bootstrap review
  `trust_streak_accumulation` means the lane should not be retried until unattended trust recovers to the required streak/status
  `protected_review_only` means the lane is blocked by protected-subsystem policy, not by missing candidate quality alone
- consume the queue-specific remediation reports before reopening a paused lane:
  [`supervisor_baseline_bootstrap_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json),
  [`supervisor_trust_streak_recovery_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json),
  [`supervisor_protected_review_queue.json`](/data/agentkernel/trajectories/improvement/reports/supervisor_protected_review_queue.json)
- `baseline_bootstrap` and `protected_review_only` entries now include `review_finalize_command` so a coordinator can inspect the exact dry-run finalize target without reconstructing it from frontier state
- scoped `cycles_<scope>.jsonl` files are now expected to contain only that scope's records; if a scoped file mixes other `scope_id` values, treat it as stale contamination and rebuild the frontier before making promotion decisions

Dry-run finalize from frontier:

```bash
cd /data/agentkernel && \
python scripts/finalize_latest_candidate_from_cycles.py \
  --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json \
  --subsystem <subsystem> \
  --variant-id <variant_id> \
  --dry-run
```

## Closeout Procedure

When the run ends, update the queue entry with:

- `released_at`
- `result`
- `findings`
- `artifacts`
- `verification`
- `next_recommendation`

Terminal statuses:

- `completed`
- `aborted`
- `blocked`
- `handed_off`

## What Goes Where

- coordination index: [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md)
- live claims and closeouts: [`docs/supervised_work_queue.md`](/data/agentkernel/docs/supervised_work_queue.md)
- lane ownership contract: [`config/supervised_parallel_work_manifest.json`](/data/agentkernel/config/supervised_parallel_work_manifest.json)
- shared machine-readable evidence: [`trajectories/improvement/reports/supervised_parallel_frontier.json`](/data/agentkernel/trajectories/improvement/reports/supervised_parallel_frontier.json)
- consolidation queue: [`trajectories/improvement/reports/supervised_frontier_promotion_plan.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_plan.json)
- execution report: [`trajectories/improvement/reports/supervised_frontier_promotion_pass.json`](/data/agentkernel/trajectories/improvement/reports/supervised_frontier_promotion_pass.json)
- historical narrative: [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md)

## Anti-Failure Rules

- do not append detailed long-form probe history to the status index
- do not claim more than one lane
- do not start an unscoped supervised run for parallel work
- do not promote directly from a scoped cycle file when the frontier report already covers it
- do not stop at frontier generation if you are the coordinator; refresh the promotion plan too
- if compare fails because no retained baseline exists, record that explicitly; that is now a retention-readiness gap, not a coordination gap
- do not apply first-retain finalize unless the supervisor or coordinator is explicitly running in trusted promotion mode
- do not make human parsing the only integration path between parallel runs
