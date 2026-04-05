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

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T22:11:00Z`
- scope_id: `delegated_queue_acceptance_review`
- progress_label: `delegated_queue_acceptance_review`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/run_job_queue.py tests/test_job_queue.py
cd /data/agentkernel && pytest -q tests/test_job_queue.py -k "acceptance_review or promotable_cli_excludes_synthetic_workers or promote_cli_marks_job_promoted_and_hides_from_promotable"
cd /data/agentkernel && AGENT_KERNEL_PROVIDER=mock AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE=disabled AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS=1 AGENT_KERNEL_WORKSPACE_ROOT=/data/agentkernel/trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z/workspace AGENT_KERNEL_TRAJECTORIES_ROOT=/data/agentkernel/trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z/trajectories/episodes AGENT_KERNEL_RUN_REPORTS_DIR=/data/agentkernel/trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z/trajectories/reports AGENT_KERNEL_RUN_CHECKPOINTS_DIR=/data/agentkernel/trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z/trajectories/checkpoints AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH=/data/agentkernel/trajectories/improvement/reports/delegated_queue_acceptance_review_20260403T221137Z/queue_source.json AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH=/data/agentkernel/trajectories/improvement/reports/delegated_queue_acceptance_review_20260403T221137Z/runtime_state_source.json python scripts/run_job_queue.py acceptance-review --json
```

- hypothesis: `Shared-repo acceptance concurrency is already proven. The next handoff ROI is not another scheduler change; it is an operator-facing review surface that can isolate completed acceptance-ready integrators, with their shared-repo and branch metadata intact, from the broader completed queue that still contains worker completions.`
- owned_paths:
  - `scripts/run_job_queue.py`
  - `tests/test_job_queue.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T22:18:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_job_queue.py with a new acceptance-review command. It filters to completed acceptance-ready integrators, emits a dedicated review rollup, and preserves shared_repo_id, target_branch, merged_branches, trusted-family status, and report_path per job instead of forcing review through the generic completed queue or the older promotable-only list.`
  - `Added focused regressions in tests/test_job_queue.py for both text and JSON review modes. The new cases lock default exclusion of already promoted integrators plus worker noise, and they verify include-promoted JSON rollups across repo_sandbox_parallel_merge and repo_sandbox_release_train.`
  - `The new evidence bundle trajectories/improvement/reports/delegated_queue_acceptance_review_20260403T221137Z replays the acceptance-review CLI against the raw completed queue snapshot from the earlier live shared-repo acceptance run. summary.json records the broad completed surface honestly: completed_queue_jobs=7, completed_queue_worker_jobs=5, completed_queue_integrator_jobs=2.`
  - `That same bundle shows the new handoff narrowing cleanly. acceptance_review_pre.json and acceptance_review_pre.txt reduce the same completed queue to the two promotable acceptance roots only: git_parallel_merge_acceptance_task and git_release_train_acceptance_task, each with shared_repo_id, target_branch=main, required merged worker branches, trust_status=trusted, and direct report_path references for operator review.`
- artifacts:
  - `scripts/run_job_queue.py`
  - `tests/test_job_queue.py`
  - `trajectories/improvement/reports/delegated_queue_acceptance_review_20260403T221137Z`
- verification:
  - `python -m py_compile scripts/run_job_queue.py tests/test_job_queue.py`
  - `pytest -q tests/test_job_queue.py -k "acceptance_review or promotable_cli_excludes_synthetic_workers or promote_cli_marks_job_promoted_and_hides_from_promotable"`
- next_recommendation: `Promotion/handoff isolation is no longer the blocker on this queue surface. The next runtime-evidence ROI here is harder generated/conflict acceptance roots on the same queue machinery, especially shared-repo acceptance tasks that carry generated-path or conflict-resolution risk rather than the now-clean merge/release acceptance pair.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T19:24:00Z`
- scope_id: `delegated_queue_integrator_acceptance_execution`
- progress_label: `delegated_queue_integrator_acceptance_execution`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_PROVIDER=mock AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE=disabled AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS=1 python scripts/run_job_queue.py enqueue --task-id git_parallel_merge_acceptance_task --decompose-workers 1
cd /data/agentkernel && AGENT_KERNEL_PROVIDER=mock AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE=disabled AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS=1 python scripts/run_job_queue.py enqueue --task-id git_release_train_acceptance_task --decompose-workers 1
cd /data/agentkernel && AGENT_KERNEL_PROVIDER=mock AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE=disabled AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS=1 python scripts/run_job_queue.py run-next
```

- hypothesis: `The queue lane no longer needs more worker-only breadth evidence. If the real delegated queue can drain worker dependencies and then hold two shared-repo acceptance integrators live at the same time under the isolated mock posture, we should be able to capture machine-readable in-flight integrator leases plus verified terminal acceptance outcomes from the real enqueue/run-next surface instead of only from direct kernel tests.`
- owned_paths:
  - `scripts/run_job_queue.py`
  - `tests/test_job_queue.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T19:27:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_job_queue.py so status/status --json now emit active_roles / active_lease_roles with worker-vs-integrator counts plus family, budget-group, and shared-repo rollups. Added focused coverage in tests/test_job_queue.py for both the new text output and JSON payload, including a mixed active worker+integrator lease case.`
  - `The fresh isolated bundle trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z shows the real queue decomposing two shared-repo acceptance roots into five worker dependencies plus two integrators, then draining to an integrator-only tail before the concurrent acceptance run.`
  - `The in-flight snapshot is the key evidence: queue_status_inflight.json records state_counts.completed=5 and state_counts.in_progress=2, while queue.active_lease_roles lands total=2 and integrator_jobs=2 across shared_repo_ids repo_sandbox_parallel_merge and repo_sandbox_release_train. queue_list_inflight.json shows both git_parallel_merge_acceptance_task and git_release_train_acceptance_task live in state=in_progress at the same time.`
  - `The same bundle closes the acceptance phase cleanly. inspect_git_parallel_merge_acceptance_task.json and inspect_git_release_train_acceptance_task.json both show outcome=success, verifier_passed=1, merged_branches populated with the required worker branches, and promotable=true after completion.`
- artifacts:
  - `scripts/run_job_queue.py`
  - `tests/test_job_queue.py`
  - `trajectories/improvement/reports/delegated_queue_integrator_acceptance_execution_20260403T192539Z`
- verification:
  - `python -m py_compile scripts/run_job_queue.py tests/test_job_queue.py`
  - `pytest -q tests/test_job_queue.py -k "run_job_queue_status_cli or run_job_queue_status_json_includes_queue_trust_and_capability_details or run_job_queue_status_json_reports_active_integrator_and_worker_leases"`
- next_recommendation: `Clean shared-repo acceptance concurrency is no longer the blocker on this queue surface. The next runtime-evidence ROI is harder acceptance execution and handoff hygiene: push the same queue flow onto generated/conflict shared-repo acceptance roots or tighten the promotion surface so operator handoff can isolate integrator-ready acceptance jobs from the broader completed queue.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T18:10:00Z`
- scope_id: `delegated_queue_repo_backed_execution`
- progress_label: `delegated_queue_repo_backed_execution`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_PROVIDER=mock AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 AGENT_KERNEL_SANDBOX_COMMAND_CONTAINMENT_MODE=disabled AGENT_KERNEL_UNATTENDED_ALLOW_GIT_COMMANDS=1 python scripts/run_job_queue.py run-next
```

- hypothesis: `The next queue-runtime gap is no longer synthetic worker-command breadth. If the delegated runner can execute real repo-backed jobs concurrently under the isolated mock posture, and if checkpoint export governance stops sweeping unrelated improvement roots on every checkpoint write, then two shared-repo worker branches plus an independent verifier-bearing repo review should hold live leases together long enough to capture non-empty in-flight kernel progress and terminal success outcomes.`
- owned_paths:
  - `agent_kernel/runtime_supervision.py`
  - `agent_kernel/export_governance.py`
  - `agent_kernel/job_queue.py`
  - `tests/test_runtime_supervision.py`
  - `tests/test_job_queue.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T18:23:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/runtime_supervision.py and agent_kernel/export_governance.py so managed writes govern only the touched export root. This removed the delegated repo-backed stall where every checkpoint write under run_checkpoints was recursively sizing unrelated improvement candidate trees before step 1.`
  - `Patched agent_kernel/job_queue.py so delegated kernel runs now write a per-job *.progress.json sidecar from the AgentKernel progress callback. That made mid-flight delegated progress visible without waiting for a full checkpoint step to finish.`
  - `The fresh isolated bundle trajectories/improvement/reports/delegated_queue_repo_backed_execution_20260403T182235Z now shows real concurrent repo-backed execution: runtime_state_inflight.json records active_leases=3, queue_status_inflight.json and queue_list_inflight.json show the two shared-repo worker branches plus repo_cleanup_review_task live at once, and progress_snapshot_inflight.json plus checkpoint_snapshot_inflight.json capture non-empty progress for all three jobs before completion.`
  - `The same bundle closed the run with three clean outcomes. queue_status_completed.json, queue_list_completed.json, and the per-job inspect_*.json artifacts all show completed/success for git_parallel_worker_api_task, git_parallel_worker_docs_task, and repo_cleanup_review_task.`
- artifacts:
  - `agent_kernel/runtime_supervision.py`
  - `agent_kernel/export_governance.py`
  - `agent_kernel/job_queue.py`
  - `tests/test_runtime_supervision.py`
  - `tests/test_job_queue.py`
  - `trajectories/improvement/reports/delegated_queue_repo_backed_execution_20260403T182235Z`
- verification:
  - `python -m py_compile agent_kernel/export_governance.py agent_kernel/runtime_supervision.py agent_kernel/job_queue.py tests/test_runtime_supervision.py tests/test_job_queue.py`
  - `pytest -q tests/test_runtime_supervision.py -k "atomic_write_json_can_skip_governance or maybe_govern_improvement_exports_limits_checkpoint_scope or atomic_write_json_uses_explicit_config_instead_of_env_lookup"`
  - `pytest -q tests/test_job_queue.py -k "writes_progress_sidecar or duplicate_claim_live_in_progress or skips_live_leased_in_progress_job or runtime_controller_snapshot_tolerates_invalid_json_state"`
- next_recommendation: `The queue lane now has real concurrent repo-backed worker/review evidence. The next runtime-evidence ROI is concurrent integrator/acceptance execution under the same queue surface, especially shared-repo merge/acceptance jobs that depend on live worker outputs rather than only worker/review tasks.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T17:30:00Z`
- scope_id: `liftoff_takeover_drift_reposandbox_task10`
- progress_label: `liftoff_takeover_drift_reposandbox_task10`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 10 --priority-benchmark-family repo_sandbox --preserve-report-history --takeover-drift-step-budget 20 --takeover-drift-wave-task-limit 4 --takeover-drift-max-waves 3 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json
```

- hypothesis: `The next blocker after repo_sandbox clean breadth is no longer family admission; it is missing drift evidence plus the unsafe acceptance tail. If the official drift pass inherits the same shadow-enrolled candidate artifact and repo-sandbox operator policy as the main evaluator, and low-cost selection postpones the unsafe acceptance roots, we should be able to validate a clean task_limit=10 repo_sandbox slice under real takeover-drift.`
- owned_paths:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `evals/harness.py`
  - `agent_kernel/trust.py`
  - `agent_kernel/modeling/evaluation/liftoff.py`
  - `tests/test_liftoff.py`
  - `tests/test_eval.py`
  - `tests/test_trust.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T18:30:00Z`
- result: `completed`
- findings:
  - `Patched scripts/evaluate_tolbert_liftoff.py so takeover-drift runs inherit priority-family hints, low-cost selection, repo-sandbox operator-policy overrides, and the shadow-enrolled candidate artifact instead of evaluating an unscoped generic drift path.`
  - `Patched evals/harness.py so the two known unsafe repo-sandbox acceptance roots are ranked behind the current worker/status tail; low-cost required-family selection now reaches a clean task_limit=10 repo-sandbox slice before those acceptance roots re-enter.`
  - `Patched agent_kernel/trust.py so dirty legacy executable reports without supervision metadata no longer count toward counted gated trust, while clean legacy executable reports still do. This repaired the repo-sandbox counted ledger without erasing older clean repository/project/repo_chore evidence.`
  - `Patched agent_kernel/modeling/evaluation/liftoff.py so trust non-regression compares counted_gated_summary before raw gated_summary, aligning the liftoff gate with the trust gate.`
  - `The fresh official report trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json now contains real takeover-drift evidence: step_budget=20, wave_task_limit=4, max_waves=3, budget_reached=true, worst_pass_rate_delta=0.0, worst_unsafe_ambiguous_rate_delta=0.0, worst_hidden_side_effect_rate_delta=0.0, and family_takeover_evidence.repo_sandbox.decision=promoted.`
  - `The bounded repo-sandbox slice widened cleanly to task_limit=10: candidate/baseline both held pass_rate=0.9 on the main compare, the counted repo_sandbox summary stayed clean with unsafe_ambiguous_rate=0.0 and hidden_side_effect_risk_rate=0.0, and the unsafe acceptance roots were no longer in the selected task roots.`
  - `The remaining blocker is now explicit and narrower than before: the scoped preserved-history candidate directory on this official path only contains repo_sandbox counted evidence, so overall_gated still fails on distinct_benchmark_families=1 below min_distinct_families=2 even though repo_sandbox itself is trusted and drift-backed. The next ROI is preserving or merging counted evidence across required families for official liftoff runs, not more repo-sandbox task triage.`
- artifacts:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `evals/harness.py`
  - `agent_kernel/trust.py`
  - `agent_kernel/modeling/evaluation/liftoff.py`
  - `tests/test_liftoff.py`
  - `tests/test_eval.py`
  - `tests/test_trust.py`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json`
- verification:
  - `python -m py_compile scripts/evaluate_tolbert_liftoff.py evals/harness.py agent_kernel/trust.py agent_kernel/modeling/evaluation/liftoff.py tests/test_liftoff.py tests/test_eval.py tests/test_trust.py`
  - `pytest -q tests/test_liftoff.py -k "repo_sandbox_operator_policy_for_required_family or scopes_takeover_drift_to_shadow_candidate_and_priority_family or rejects_trust_ledger_regression or prefers_counted_trust_summary_over_gated_attempts"`
  - `pytest -q tests/test_eval.py -k "cleaner_repo_sandbox_primaries_before_known_unstable_roots or widens_repo_sandbox_before_acceptance_tail"`
  - `pytest -q tests/test_trust.py -k "counts_only_executable_gated_reports_for_assessment or excludes_legacy_unsupervised_reports_from_counted_summary"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 10 --priority-benchmark-family repo_sandbox --preserve-report-history --takeover-drift-step-budget 20 --takeover-drift-wave-task-limit 4 --takeover-drift-max-waves 3 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json`
- next_recommendation: `Keep the new drift-backed repo_sandbox lane and switch ROI to official liftoff evidence aggregation: preserve or merge counted required-family history across family-specific official reruns so overall_gated can see repository/project/repo_chore/integration alongside repo_sandbox during liftoff promotion.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T02:26:00Z`
- scope_id: `trust_reposandbox_liftoff_cleanbreadth`
- progress_label: `trust_reposandbox_liftoff_cleanbreadth`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 16 --takeover-drift-step-budget 0 --priority-benchmark-family repo_chore --priority-benchmark-family repo_sandbox --preserve-report-history --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_repochore_reposandbox_counted_trust.json
cd /data/agentkernel && AGENT_KERNEL_RUN_REPORTS_DIR=trajectories/reports/liftoff_repo_sandbox_clean AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH=trajectories/reports/liftoff_repo_sandbox_clean/ledger.json AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 12 --takeover-drift-step-budget 0 --priority-benchmark-family repo_sandbox --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_fresh.json
cd /data/agentkernel && AGENT_KERNEL_RUN_REPORTS_DIR=trajectories/reports/liftoff_repo_sandbox_clean9 AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH=trajectories/reports/liftoff_repo_sandbox_clean9/ledger.json AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 9 --takeover-drift-step-budget 0 --priority-benchmark-family repo_sandbox --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task9.json
```

- hypothesis: `Repo-sandbox trust breadth is no longer blocked on missing tasks; it is blocked on two runtime-accounting defects and then on early-slot selection quality. If the official evaluator admits git/generated-path sandbox tasks, ignores `.git` bootstrap churn as expected git state, and leads with the cleaner sandbox roots first, repo_sandbox should become counted trusted evidence instead of remaining the last missing required family.`
- owned_paths:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `agent_kernel/preflight.py`
  - `evals/harness.py`
  - `tests/test_liftoff.py`
  - `tests/test_preflight.py`
  - `tests/test_eval.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T03:04:00Z`
- result: `completed`
- findings:
  - `The first official preserved-history rerun proved repo_sandbox was being filtered out before selection, not missing from the task bank: every repo-sandbox task had requires_git=True and the scoped liftoff configs still had unattended git disabled.`
  - `Patched scripts/evaluate_tolbert_liftoff.py so official liftoff configs enable unattended git commands plus generated-path mutations whenever repo_sandbox is in the required/requested family set, and added regression coverage in tests/test_liftoff.py.`
  - `The next blocker was report accounting, not task quality: fresh repo-sandbox reports were marking `.git/*` clone/bootstrap state as unexpected workspace changes, which forced hidden_side_effect_risk=true and false_pass_risk=true on otherwise verified successes. Patched agent_kernel/preflight.py so git-enabled tasks treat `.git/*` as expected internal state, with coverage in tests/test_preflight.py.`
  - `After the accounting repair, the fresh isolated task_limit=12 sandbox run in trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_fresh.json became clean on side effects but still landed restricted because two unsafe acceptance roots kept unsafe_ambiguous_rate at 0.1667.`
  - `Patched evals/harness.py so low-cost required-family selection pushes the four currently unstable repo-sandbox roots behind the cleaner executable roots, with coverage in tests/test_eval.py.`
  - `The fresh isolated official task_limit=9 rerun in trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task9.json closed the family: candidate_trust.family_assessments.repo_sandbox is now status=trusted with counted_gated_family_summaries.repo_sandbox.total=9, success_rate=1.0, clean_success_streak=9, distinct_clean_success_task_roots=9, and zero hidden-side-effect / unsafe / false-pass risk.`
  - `The wider boundary is now explicit too: task_limit=12 is still too wide for a clean official sandbox breadth sweep because the next acceptance roots remain unsafe, so the active repo-sandbox ROI has narrowed to widening that clean envelope past 9 tasks, not family admission or report accounting.`
- artifacts:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `agent_kernel/preflight.py`
  - `evals/harness.py`
  - `tests/test_liftoff.py`
  - `tests/test_preflight.py`
  - `tests/test_eval.py`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_repochore_reposandbox_counted_trust.json`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_fresh.json`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task9.json`
  - `trajectories/reports/liftoff_repo_sandbox_clean`
  - `trajectories/reports/liftoff_repo_sandbox_clean9`
- verification:
  - `python -m py_compile scripts/evaluate_tolbert_liftoff.py agent_kernel/preflight.py evals/harness.py tests/test_liftoff.py tests/test_preflight.py tests/test_eval.py`
  - `pytest -q tests/test_liftoff.py -k "shadow_enrolls_priority_families_for_candidate_probe or can_preserve_report_history or enables_repo_sandbox_operator_policy_for_required_family"`
  - `pytest -q tests/test_preflight.py -k "git_internal_changes_for_git_tasks or non_git_unexpected_changes_for_git_tasks or capture_workspace_snapshot_and_side_effect_summary_detect_changes"`
  - `pytest -q tests/test_eval.py -k "cleaner_repo_sandbox_primaries_before_known_unstable_roots or uses_five_low_cost_repo_chore_primaries_before_retrieval or uses_five_low_cost_repository_primaries_before_retrieval_or_long_horizon"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 16 --takeover-drift-step-budget 0 --priority-benchmark-family repo_chore --priority-benchmark-family repo_sandbox --preserve-report-history --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_repochore_reposandbox_counted_trust.json`
  - `AGENT_KERNEL_RUN_REPORTS_DIR=trajectories/reports/liftoff_repo_sandbox_clean AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH=trajectories/reports/liftoff_repo_sandbox_clean/ledger.json AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 12 --takeover-drift-step-budget 0 --priority-benchmark-family repo_sandbox --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_fresh.json`
  - `AGENT_KERNEL_RUN_REPORTS_DIR=trajectories/reports/liftoff_repo_sandbox_clean9 AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH=trajectories/reports/liftoff_repo_sandbox_clean9/ledger.json AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 9 --takeover-drift-step-budget 0 --priority-benchmark-family repo_sandbox --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task9.json`
- next_recommendation: `Required-family breadth is now closed across repository, project, repo_chore, and repo_sandbox. The next unattended-trust ROI is takeover-drift evidence for actual liftoff promotion, with a secondary runtime-evidence lane to widen clean repo-sandbox breadth beyond the current healthy task_limit=9 envelope without reintroducing unsafe acceptance roots.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T02:05:00Z`
- scope_id: `trust_repositorybreadth_liftoff_counted`
- progress_label: `trust_repositorybreadth_liftoff_counted`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 4 --takeover-drift-step-budget 0 --priority-benchmark-family repository --preserve-report-history --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_repository_counted_trust.json
```

- hypothesis: `Repository is no longer blocked on trust semantics; it is blocked on counted executable volume. If the official evaluator reuses the preserved report history and stays on low-cost repository primaries, repository should cross the 5-report counted threshold and move from bootstrap to trusted without another trust-stack patch.`
- owned_paths:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T02:24:34Z`
- result: `completed`
- findings:
  - `The current counted ledger was already close before the rerun: repository had 3 counted gated clean successes with task_roots=["repo_sync_matrix_task", "schema_alignment_task", "service_release_task"], so the real blocker was breadth accumulation, not another trust-policy defect.`
  - `A fresh official preserved-history rerun at trajectories/tolbert_model/liftoff_gate_report_20260403_repository_counted_trust.json kept the bounded evaluator on repository tasks and moved family:repository from bootstrap to trusted.`
  - `The candidate trust ledger now shows counted_gated_family_summaries.repository.total=6, success_rate=1.0, clean_success_streak=6, and distinct_clean_success_task_roots=3, with family_assessments.repository.status=trusted.`
  - `The same report also clarifies the next unattended-trust bottleneck: repository is no longer the missing required family on this stack. coverage_summary now leaves only missing_required_counted_gated_families=["repo_chore", "repo_sandbox"].`
  - `Liftoff itself is still shadow_only in this official report because takeover-drift evidence is still absent; this rerun closed repository trust breadth, not the liftoff promotion gate.`
- artifacts:
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_repository_counted_trust.json`
  - `trajectories/reports/tolbert_liftoff_candidate_unattended_trust_ledger.json`
- verification:
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 4 --takeover-drift-step-budget 0 --priority-benchmark-family repository --preserve-report-history --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_repository_counted_trust.json`
  - `python - <<'PY' ... inspected candidate_trust.family_assessments.repository, counted_gated_family_summaries.repository, and coverage_summary from trajectories/tolbert_model/liftoff_gate_report_20260403_repository_counted_trust.json ... PY`
- next_recommendation: `Repository counted unattended trust is now real and trusted. The next trust ROI on this same stack is missing required-family breadth: gather counted executable evidence for repo_chore and repo_sandbox so the unattended ledger can leave partial-family mode, then come back to takeover-drift evidence for actual liftoff promotion.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-03T01:37:00Z`
- scope_id: `trust_projectbreadth_liftoff_executablefirst`
- progress_label: `trust_projectbreadth_liftoff_executablefirst`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 12 --takeover-drift-step-budget 0 --priority-benchmark-family repository --priority-benchmark-family project --priority-benchmark-family workflow --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_bounded_projectbreadth.json
```

- hypothesis: `The official liftoff evaluator is still undercounting project-family trust breadth because retrieval companions are consuming the bounded family slots before executable project tasks can run. If the liftoff scripts force low-cost executable-first selection, the same bounded trust run should convert project from one counted executable report into multiple gated project task roots without changing the trust thresholds themselves.`
- owned_paths:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `scripts/run_tolbert_liftoff_loop.py`
  - `tests/test_eval.py`
  - `tests/test_liftoff.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T01:53:56Z`
- result: `completed`
- findings:
  - `Patched scripts/evaluate_tolbert_liftoff.py and scripts/run_tolbert_liftoff_loop.py so official trust-gathering runs always pass prefer_low_cost_tasks=True into run_eval. This keeps required-family slots on executable source tasks before retrieval companions.`
  - `Added regression coverage in tests/test_liftoff.py and tests/test_eval.py. The new selection-level test locks project-family low-cost selection to executable tasks over retrieval companions, and the official evaluator test now asserts prefer_low_cost_tasks=True is passed into run_eval.`
  - `A direct live task-bank probe now selects executable project tasks under the bounded trust recipe: ['report_rollup_task', 'deployment_manifest_task', 'release_packet_task'] instead of retrieval companions consuming those slots.`
  - `Fresh official evidence is in trajectories/tolbert_model/liftoff_gate_report_20260403_bounded_projectbreadth.json. The candidate trust ledger first moved project from 1 gated report to 3 gated clean successes, with task_roots=["deployment_manifest_task", "release_packet_task", "report_rollup_task"] and distinct_clean_success_task_roots=3.`
  - `Patched scripts/evaluate_tolbert_liftoff.py again with --preserve-report-history so official reruns can accumulate repeatable unattended evidence instead of deleting prior counted runs, and added regression coverage in tests/test_liftoff.py for preserved report markers.`
  - `Patched agent_kernel/trust.py so counted unattended-trust assessment is now based on executable gated reports only. Raw gated attempts stay visible in gated_summary / gated_family_summaries, while family_assessments and overall_assessment now use counted_gated_summary / counted_gated_family_summaries to exclude zero-step policy_terminated runs from counted trust evidence.`
  - `Fresh official counted-evidence proof is now in trajectories/tolbert_model/liftoff_gate_report_20260403_project_counted_trust.json. The candidate trust ledger now shows family:project status=trusted with counted_gated_family_summaries.project.total=6, success_rate=1.0, clean_success_streak=6, and task_roots=["deployment_manifest_task", "release_packet_task", "report_rollup_task"], while the raw attempt history remains separately visible at gated_family_summaries.project.total=10 with safe_stop_count=4.`
- artifacts:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `scripts/run_tolbert_liftoff_loop.py`
  - `agent_kernel/trust.py`
  - `tests/test_eval.py`
  - `tests/test_liftoff.py`
  - `tests/test_trust.py`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_bounded_projectbreadth.json`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_project_counted_trust.json`
  - `trajectories/reports/tolbert_liftoff_candidate_unattended_trust_ledger.json`
- verification:
  - `python -m py_compile scripts/evaluate_tolbert_liftoff.py scripts/run_tolbert_liftoff_loop.py agent_kernel/trust.py tests/test_eval.py tests/test_liftoff.py tests/test_trust.py`
  - `pytest -q tests/test_eval.py -k "prefers_executable_project_tasks_over_retrieval_companions or can_prefer_low_cost_tasks_within_family" --cache-clear`
  - `pytest -q tests/test_liftoff.py -k "shadow_enrolls_priority_families_for_candidate_probe or can_preserve_report_history" --cache-clear`
  - `pytest -q tests/test_trust.py -k "coverage_only_retrieval_reports_from_gated_summary or counts_only_executable_gated_reports_for_assessment or trusts_required_family_after_clean_task_root_breadth or bootstraps_required_family_with_narrow_clean_task_roots" --cache-clear`
  - `python - <<'PY' ... _limit_tasks_for_compare(project, 3, priority_families=['project'], prefer_low_cost_tasks=True) ... PY`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 12 --takeover-drift-step-budget 0 --priority-benchmark-family repository --priority-benchmark-family project --priority-benchmark-family workflow --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_bounded_projectbreadth.json`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 4 --takeover-drift-step-budget 0 --priority-benchmark-family project --preserve-report-history --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_project_counted_trust.json`
- next_recommendation: `Project counted unattended trust is now real and trusted. The next trust ROI is repository breadth and the remaining missing required families: apply the same counted-executable split to repository/integration evidence collection, then push repository from counted bootstrap to trusted and fill repo_chore/repo_sandbox so overall gated trust can leave partial-family mode.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-03T01:08:00Z`
- scope_id: `supervised_longhorizon_codex_apr03_followupcompletionflush`
- progress_label: `supervised_longhorizon_codex_apr03_followupcompletionflush`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_followupcompletionflush --scope-id supervised_longhorizon_codex_apr03_followupcompletionflush --notes "flush generated-success completion bookkeeping immediately after verified shared-repo followups return so long-horizon adjacent tasks do not disappear behind seed/report persistence"
```

- hypothesis: `The remaining long-horizon generated_success loss is inside the child after verifier success. If completion callbacks and partial snapshots flush before slower unattended-report or seed-bundle writes, the first shared-repo adjacent task should appear directly in the live partial summary instead of only through verifier-state salvage.`
- owned_paths:
  - `evals/harness.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_eval.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T01:35:56Z`
- result: `completed`
- findings:
  - `Patched evals/harness.py so _run_tasks_with_progress flushes task-complete/result callbacks before unattended-report persistence, and so generated_success emits its partial progress snapshot before seed-bundle persistence. The partial snapshot now also carries last_completed_generated_task_id and last_completed_generated_benchmark_family.`
  - `Added direct regressions in tests/test_eval.py: one locks callback-before-report ordering, and one proves a generated_success partial snapshot already shows completed_generated_tasks=1 while seed-bundle persistence is intentionally blocked.`
  - `Patched scripts/run_human_guided_improvement_cycle.py to flatten the new generated-task partial-summary fields into the scoped observation summary, and extended tests/test_protocol_comparison.py so timeout-merging followups can expose partial_last_completed_generated_task_id and family directly.`
  - `Fresh live evidence in trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_followupcompletionflush.jsonl shows the fix working on the real shared-repo path: the first adjacent followup still times out at the outer 80s guard, but its child partial summary now directly records completed_generated_tasks=1, generated_passed=1, generated_by_benchmark_family={\"repository\": 1}, and last_completed_generated_task_id=git_parallel_merge_acceptance_task__worker__worker_docs-status_repository_adjacent before shutdown.`
- artifacts:
  - `evals/harness.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_eval.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_followupcompletionflush.jsonl`
  - `trajectories/improvement/candidates/supervised_longhorizon_codex_apr03_followupcompletionflush/retrieval/cycle_retrieval_20260403T013410730123Z_1c79d849_confidence_gating/retrieval_proposals.json`
- verification:
  - `python -m py_compile evals/harness.py scripts/run_human_guided_improvement_cycle.py tests/test_eval.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_eval.py -k "flushes_callbacks_before_unattended_report_write or flushes_completion_snapshot_before_seed_bundle_persistence or persists_observed_runtime" --cache-clear`
  - `pytest -q tests/test_protocol_comparison.py::test_run_supervised_cycle_retries_generated_success_followup_with_reduced_task_limit --cache-clear`
  - `pytest -q tests/test_protocol_comparison.py::test_run_supervised_cycle_applies_long_horizon_runtime_controls_to_generated_success_followup --cache-clear`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_followupcompletionflush --scope-id supervised_longhorizon_codex_apr03_followupcompletionflush --notes "flush generated-success completion bookkeeping immediately after verified shared-repo followups return so long-horizon adjacent tasks do not disappear behind seed/report persistence"`
- next_recommendation: `The remaining long-horizon ROI is no longer missing first-wave completion bookkeeping. The next patch should carry the same shared-repo runtime guard and timeout-recovery path into wave2+ generated_success followups, which still execute on the raw late-wave path after the initial repository bridge succeeds.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-03T00:13:06Z`
- scope_id: `supervised_longhorizon_codex_apr03_integrator_tail_live`
- progress_label: `supervised_longhorizon_codex_apr03_integrator_tail_live`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_integrator_tail_live --scope-id supervised_longhorizon_codex_apr03_integrator_tail_live --notes "measure the deeper long-horizon integrator tail under the live bounded runtime envelope after governance/oversight integrator realism"
```

- hypothesis: `The late-wave integrator realism is proven only in generated-only seed probes today. A fresh bounded live long-horizon cycle should either carry the mixed integrator tail deeper through the real runtime envelope or expose the first concrete runtime/scheduling bottleneck to patch.`
- owned_paths:
  - `agent_kernel/curriculum.py`
  - `datasets/curriculum_templates.json`
  - `evals/harness.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_curriculum.py`
  - `tests/test_eval.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:32:59Z`
- result: `completed`
- findings:
  - `The first live rerun exposed a real wrapper bug instead of a task failure: the long-horizon observation completed and staged generated_success waves started, but the scoped generate-only wrapper threw the scope away at candidate selection because the ranked experiments did not include the explicitly requested subsystem tooling.`
  - `Patched scripts/run_human_guided_improvement_cycle.py so scoped/generate-only runs fall back to the top ranked experiment when the requested subsystem is unavailable, and so timed-out observations can salvage completed primary metrics from the child partial summary when all primary tasks already finished.`
  - `The repaired live scope trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_live.jsonl now preserves the real bounded bottleneck: primary completed 5 project tasks with 3/5 passed under the 80s shared-repo guard, the wrapper recovered those metrics from partial summary, generated_success launched, then the first followup timed out at 40s even after a reduced task_limit retry. The retained candidate for this evidence landed on retrieval/confidence_gating rather than tooling.`
  - `Patched scripts/run_human_guided_improvement_cycle.py again so long-horizon generated_success followups inherit the same shared-repo runtime guard as the primary observation: the first followup now runs under applied_max_observation_seconds=80.0, disables TolBERT context up front, and preserves the expanded git/path operator policy on the reduced task_limit retry too.`
  - `Fresh live evidence in trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_followupguard.jsonl shows that this moved the bottleneck forward: the first adjacent followup no longer died at the old 40s wall, started under the 80s followup guard, and reached the reduced task_limit=1/5 retry path before timing out.`
  - `The remaining loss turned out to be post-verifier bookkeeping, not search/runtime. The timed-out adjacent task was already reporting current_task_verification_passed=true with current_task_total=1 and step_complete in the partial summary, so scripts/run_human_guided_improvement_cycle.py now salvages that exact state as a real generated_success completion instead of dropping it because completed_generated_tasks never flushed.`
  - `Fresh live evidence in trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_followupsalvage.jsonl confirms the recovery: the same long-horizon scope still times out on the outer 80s followup envelope, but now retains generated_total=1 and generated_passed=1 for git_parallel_merge_acceptance_task__worker__worker_docs-status_repository_adjacent instead of leaving generated_success at zero.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_live.jsonl`
  - `trajectories/improvement/candidates/supervised_longhorizon_codex_apr03_integrator_tail_live/retrieval/cycle_retrieval_20260403T003033673912Z_afbd3758_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_followupguard.jsonl`
  - `trajectories/improvement/candidates/supervised_longhorizon_codex_apr03_integrator_tail_followupguard/retrieval/cycle_retrieval_20260403T010907451964Z_a52a5f2f_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/cycles_supervised_longhorizon_codex_apr03_integrator_tail_followupsalvage.jsonl`
  - `trajectories/improvement/candidates/supervised_longhorizon_codex_apr03_integrator_tail_followupsalvage/retrieval/cycle_retrieval_20260403T011458400759Z_1e01b42f_confidence_gating/retrieval_proposals.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py::test_run_supervised_cycle_scoped_generate_only_falls_back_when_requested_subsystem_is_unavailable --cache-clear`
  - `pytest -q tests/test_protocol_comparison.py::test_run_human_guided_improvement_cycle_expands_budget_for_long_horizon_generated_success_followup`
  - `pytest -q tests/test_protocol_comparison.py::test_run_supervised_cycle_applies_long_horizon_runtime_controls_to_generated_success_followup`
  - `pytest -q tests/test_protocol_comparison.py::test_generated_metrics_from_partial_summary_salvages_verified_single_generated_success`
  - `pytest -q tests/test_protocol_comparison.py::test_run_supervised_cycle_retries_generated_success_followup_with_reduced_task_limit`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_integrator_tail_live --scope-id supervised_longhorizon_codex_apr03_integrator_tail_live --notes "measure the deeper long-horizon integrator tail under the live bounded runtime envelope after governance/oversight integrator realism"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_integrator_tail_followupguard --scope-id supervised_longhorizon_codex_apr03_integrator_tail_followupguard --notes "carry the long-horizon shared-repo runtime guard into the first generated_success followup so the live integrator tail no longer times out on the first adjacent task"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --subsystem tooling --generate-only --task-limit 5 --max-observation-seconds 60 --include-curriculum --generated-curriculum-budget-seconds 40 --priority-benchmark-family project --progress-label supervised_longhorizon_codex_apr03_integrator_tail_followupsalvage --scope-id supervised_longhorizon_codex_apr03_integrator_tail_followupsalvage --notes "carry the long-horizon shared-repo runtime guard into the first generated_success followup and salvage verifier-complete single-task generated successes when bookkeeping times out"`
- next_recommendation: `The next ROI on task_ecology is no longer first-wave retry-policy drift either. The initial staged followups now share the same runtime-retry helper as wave2+, so long-horizon project followups keep the widened unattended git/path policy across TolBERT startup retries instead of falling back to the base config. The remaining gap is fresh live evidence on later-wave shared-repo followups under the 80s envelope, and then carrying any surviving long-horizon tail fixes into agent_kernel/cycle_runner.py if the offline runner still diverges.`

### `supervised_runner__open_3`

- runner_slot: `supervised_runner__open_3`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-03T00:00:50Z`
- scope_id: `supervised_longhorizon_codex_apr03_governance_oversight_integrator_tail`
- progress_label: `supervised_longhorizon_codex_apr03_governance_oversight_integrator_tail`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/curriculum.py tests/test_curriculum.py tests/test_curriculum_catalog.py tests/test_eval.py && pytest -q tests/test_curriculum.py -k 'integrator_lineage_into_governance_family or integrator_lineage_into_oversight_family or retargets_validation_cleanup_adjacent_to_governance_family or retargets_governance_cleanup_adjacent_to_oversight_family or preserves_integrator_audit_lineage_into_governance_family or preserves_integrator_audit_lineage_into_oversight_family' --cache-clear && pytest -q tests/test_curriculum_catalog.py -k 'validation_long_horizon_templates or long_horizon_metadata_catalogs' --cache-clear && pytest -q tests/test_eval.py -k 'generated_success_bridges_validation_integrator_into_governance or generated_success_bridges_governance_integrator_into_oversight' --cache-clear
```

- hypothesis: `The validation-stage realism fix still flattens one hop later. If governance and oversight preserve integrator lineage too, the late-wave tail can keep multi-repo control structure deeper into the chain, and a generated-only probe should emit those harder tasks instead of generic review/crosscheck bundles.`
- owned_paths:
  - `agent_kernel/curriculum.py`
  - `datasets/curriculum_templates.json`
  - `tests/test_curriculum.py`
  - `tests/test_curriculum_catalog.py`
  - `tests/test_eval.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:16:00Z`
- result: `completed`
- findings:
  - `Patched datasets/curriculum_templates.json with four harder late-wave integrator bundles: governance_integrator_cleanup_review_bundle, governance_integrator_audit_review_bundle, oversight_integrator_cleanup_crosscheck_bundle, and oversight_integrator_audit_crosscheck_bundle. Each carries a 14-step synthetic edit plan and explicit multi-repo mirror artifacts.`
  - `Patched agent_kernel/curriculum.py so governance and oversight preserve downstream integrator realism via lineage_surface_variants when the incoming tail already carries validation_integrator_* or governance_integrator_* lineage, while ordinary cleanup/audit tails still use the generic governance/oversight variants.`
  - `Validated the new path at two levels: focused curriculum/eval tests passed, and a generated-only harness probe under trajectories/improvement/reports/late_wave_governance_oversight_integrator_probe emitted wave7 governance integrator surfaces followed by wave8 oversight integrator surfaces from realistic validation-integrator seed bundles.`
- artifacts:
  - `agent_kernel/curriculum.py`
  - `datasets/curriculum_templates.json`
  - `tests/test_curriculum.py`
  - `tests/test_curriculum_catalog.py`
  - `tests/test_eval.py`
  - `trajectories/improvement/reports/late_wave_governance_oversight_integrator_probe/generated_success_wave6_integrator_seeds.json`
  - `trajectories/improvement/reports/late_wave_governance_oversight_integrator_probe/generated_success_wave7_integrator_seeds.json`
  - `trajectories/improvement/reports/late_wave_governance_oversight_integrator_probe/generated_success_wave8_integrator_seeds.json`
- verification:
  - `python -m py_compile agent_kernel/curriculum.py tests/test_curriculum.py tests/test_curriculum_catalog.py tests/test_eval.py`
  - `pytest -q tests/test_curriculum.py -k 'integrator_lineage_into_governance_family or integrator_lineage_into_oversight_family or retargets_validation_cleanup_adjacent_to_governance_family or retargets_governance_cleanup_adjacent_to_oversight_family or preserves_integrator_audit_lineage_into_governance_family or preserves_integrator_audit_lineage_into_oversight_family' --cache-clear`
  - `pytest -q tests/test_curriculum_catalog.py -k 'validation_long_horizon_templates or long_horizon_metadata_catalogs' --cache-clear`
  - `pytest -q tests/test_eval.py -k 'generated_success_bridges_validation_integrator_into_governance or generated_success_bridges_governance_integrator_into_oversight' --cache-clear`
  - `generated-only harness probe wrote wave7 governance-integrator and wave8 oversight-integrator seed artifacts under trajectories/improvement/reports/late_wave_governance_oversight_integrator_probe`
- next_recommendation: `Push the same realism one more hop into assurance or convert the generated-only probe into a fresh scoped long-horizon cycle so the deeper integrator tail is measured under a live provider/runtime envelope rather than only through generated-success harness execution.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-02T23:50:37Z`
- scope_id: `supervised_longhorizon_codex_apr02_validation_integrator_tail`
- progress_label: `supervised_longhorizon_codex_apr02_validation_integrator_tail`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/curriculum.py tests/test_curriculum.py tests/test_curriculum_catalog.py && pytest -q tests/test_curriculum.py -k 'retargets_repo_chore_cleanup_adjacent_to_validation_family or preserves_integrator_cleanup_lineage_into_validation_family or retargets_repo_chore_audit_adjacent_to_validation_family or preserves_integrator_audit_lineage_into_validation_family or retargets_validation_cleanup_adjacent_to_governance_family or rotates_adjudication_to_validation_after_saturated_late_wave or rotates_validation_to_governance_from_late_wave_cycle_without_origin_match or keeps_late_wave_family_when_rotation_coverage_is_balanced or branches_assurance_to_adjudication_when_late_wave_signal_is_saturated or keeps_assurance_when_late_wave_signal_is_not_saturated' --cache-clear && pytest -q tests/test_curriculum_catalog.py -k 'validation_long_horizon_templates or long_horizon_metadata_catalogs' --cache-clear
```

- hypothesis: `Later-wave long-horizon tasks are already deep, but they collapse away from shared-repo/integrator structure into generic packet families. If validation-stage variant selection preserves earlier integrator lineage, the kernel can generate a harder multi-repo validation gate instead of flattening the tail.`
- owned_paths:
  - `agent_kernel/curriculum.py`
  - `datasets/curriculum_templates.json`
  - `tests/test_curriculum.py`
  - `tests/test_curriculum_catalog.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:02:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/curriculum.py so long-horizon adjacent variant selection can preserve deeper integrator lineage through new lineage-surface-aware variant rules, and so the branch law distinguishes explicit pre-saturation bridges from the saturation-gated late-wave tail.`
  - `Added two harder validation-stage long-horizon templates in datasets/curriculum_templates.json: validation_integrator_cleanup_gate_bundle and validation_integrator_audit_gate_bundle. Both carry 14-step synthetic edit plans and explicit multi-repo mirror artifacts instead of the older generic validation packet surface.`
  - `Kept the long-horizon chain coherent while doing it: repo_chore->validation and validation->governance still branch on the normal chain, assurance->adjudication stays saturation-gated, and saturated adjudication can now rotate back to validation instead of dead-ending once coverage is balanced.`
- artifacts:
  - `agent_kernel/curriculum.py`
  - `datasets/curriculum_templates.json`
  - `tests/test_curriculum.py`
  - `tests/test_curriculum_catalog.py`
- verification:
  - `python -m py_compile agent_kernel/curriculum.py tests/test_curriculum.py tests/test_curriculum_catalog.py`
  - `pytest -q tests/test_curriculum.py -k 'retargets_repo_chore_cleanup_adjacent_to_validation_family or preserves_integrator_cleanup_lineage_into_validation_family or retargets_repo_chore_audit_adjacent_to_validation_family or preserves_integrator_audit_lineage_into_validation_family or retargets_validation_cleanup_adjacent_to_governance_family or rotates_adjudication_to_validation_after_saturated_late_wave or rotates_validation_to_governance_from_late_wave_cycle_without_origin_match or keeps_late_wave_family_when_rotation_coverage_is_balanced or branches_assurance_to_adjudication_when_late_wave_signal_is_saturated or keeps_assurance_when_late_wave_signal_is_not_saturated' --cache-clear`
  - `pytest -q tests/test_curriculum_catalog.py -k 'validation_long_horizon_templates or long_horizon_metadata_catalogs' --cache-clear`
  - `direct CurriculumEngine probes confirmed cleanup and audit repo_chore seeds with integrator lineage now emit validation_integrator_* bundles with 14-step plans`
- next_recommendation: `Push the same lineage-preserving realism one hop further downstream so governance/oversight can keep multi-repo integrator structure too, then validate it with a generated-only late-wave probe or a fresh scoped long-horizon cycle instead of catalog/probe evidence alone.`

### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `codex_main`
- lane_id: `supervised_parallel_lane__learning_memory`
- claimed_at: `2026-04-02T17:05:00Z`
- scope_id: `workspace_preview_composite_canonical_span_propagation`
- progress_label: `workspace_preview_composite_canonical_span_propagation`
- status: `completed`
- hypothesis: `canonical same-span block winners were still leaking back in through composite preview assembly, so decode could still see a localized multi_edit shape alongside the canonical bounded repair`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- findings:
  - `_workspace_preview_window_proposals(...)` now prunes cross-family bounded duplicates after composite proposal assembly, so canonical preview `block_replace` winners suppress same-span or same-command `structured_edit:multi_edit` re-entry before ambiguity/frontier annotation runs
  - cross-family dedup uses a generic structured-edit span signature, so localized `multi_edit` composites that cover the same current/target region as the canonical bounded repair no longer survive as a second repair shape
  - the composite-boundary prune leaves the downstream same-span and multi-hybrid ranking logic intact, so `multi_edit` remains available only when it represents a genuinely distinct localized fix
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'same_span_multi_edit_reentry_after_composite_assembly or same_command_multi_edit_reentry_after_composite_assembly or prefer_recovered_alias_same_span_variant_when_quality_ties or collapse_equal_rank_same_span_duplicates_to_single_canonical_block or prune_weaker_same_span_bounded_variants_upstream' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'same_span_multi_edit_reentry_after_composite_assembly or same_command_multi_edit_reentry_after_composite_assembly or prefer_recovered_alias_same_span_variant_when_quality_ties or collapse_equal_rank_same_span_duplicates_to_single_canonical_block or prune_weaker_same_span_bounded_variants_upstream or same_span_proof_complete_multi_hybrid_block_over_partial_current_proof_variant or same_span_proof_complete_multi_hybrid_block_over_lower_fidelity_bounded_variant or frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata or frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or multi_core_current_proof_multi_hybrid_block_over_single_core_variant or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_bridged_multi_hybrid_block_over_multi_edit' --cache-clear`
- next_recommendation:
  - `propagate the canonical bounded winner into hidden-gap ambiguity/frontier bookkeeping so annotation counts and fallback heuristics stop treating already-collapsed localized repairs as multiple live alternatives`

## Active Focus

- `planner_selection`: frontier-driven consolidate/compare/promotion path
- `learning_memory`: make failure evidence and prior outcomes re-enter next-run decisions
- `task_ecology`: improve family mix and bounded probe coverage
- `world_state`: reduce bad first-task branching from weak state/world understanding
- `runtime_evidence`: preserve tight timeout and provider evidence where live loops still stall

## Ten-Agent Dispatch Target

Use the current wave to attack the measured blockers in [`docs/ai_agent_status.md`](/data/agentkernel/docs/ai_agent_status.md) rather than scattering across convenience edits.

| dispatch_id | target | preferred surfaces | success signal |
| --- | --- | --- | --- |
| `trust_repository` | deepen unattended trust evidence on `repository` | `agent_kernel/trust.py`, `agent_kernel/preflight.py`, `agent_kernel/runtime_supervision.py`, `tests/test_trust.py`, `tests/test_preflight.py` | counted executable `repository` evidence increases and stops being shallow bootstrap-only |
| `trust_project` | deepen unattended trust evidence on `project` | `agent_kernel/trust.py`, `agent_kernel/preflight.py`, `agent_kernel/runtime_supervision.py`, `tests/test_trust.py`, `tests/test_preflight.py` | counted executable `project` evidence increases and survives gate accounting |
| `trust_integration` | deepen unattended trust evidence on `integration` | `agent_kernel/trust.py`, `agent_kernel/preflight.py`, `agent_kernel/runtime_supervision.py`, `tests/test_trust.py`, `tests/test_preflight.py` | `integration` stops being the obvious missing required-family evidence lane |
| `supervisor_rollout` | make non-protected supervisor action real | `scripts/run_supervisor_loop.py`, `agent_kernel/runtime_supervision.py`, `agent_kernel/improvement.py`, `tests/test_runtime_supervision.py` | canary/finalize/rollback moves beyond dry-run-only posture on eligible lanes |
| `runtime_stability` | remove long-horizon timeout and bookkeeping loss | `scripts/run_human_guided_improvement_cycle.py`, `agent_kernel/cycle_runner.py`, `tests/test_protocol_comparison.py` | long-horizon followups stop dying after verifier success and preserve generated metrics cleanly |
| `world_state` | improve learned hotspot detection and recovery state | `agent_kernel/world_model.py`, `agent_kernel/state.py`, `agent_kernel/loop.py`, `tests/test_modeling_world.py`, `tests/test_loop.py` | long-horizon hotspot detection becomes earlier and more specific |
| `planner_recovery` | reduce low-yield executor repetition and widen staged long-horizon software control | `agent_kernel/policy.py`, `agent_kernel/loop.py`, `agent_kernel/multi_agent.py`, `agent_kernel/context_budget.py`, `agent_kernel/state.py`, `agent_kernel/modeling/tolbert/runtime.py`, `agent_kernel/modeling/training/universal_dataset.py`, `tests/test_policy.py`, `tests/test_loop.py`, `tests/test_multi_agent.py`, `tests/test_hybrid_tolbert_model.py`, `tests/test_universal_dataset.py` | planner/critic recovery triggers sooner, chooses more specific repair actions, critic stops deterministically when task-contract recovery is exhausted, and planner only rewrites the subgoal/contract after critic-proven exhaustion instead of reworking the same stale repair surface; the rewritten recovery objective is now persisted in loop state and checkpoints as a planner artifact that can bundle related workflow/report/test obligations, score/order those obligations into an explicit staged plan update, and feed that staged agenda into model-facing planner payloads plus retained Tolbert hybrid scoring. Outside strict recovery mode, long-horizon tasks now also derive a software-work agenda from pending implementation/edit/report/test obligations and expose that agenda to both planner payloads and retained decoder/runtime training surfaces. That broader software-work lane is now outcome-conditioned too: retained state tracks whether each staged objective advanced, stalled, regressed, or completed, reorders the agenda accordingly, exposes recent stage outcomes to planner payloads, derives a phase-state handoff summary across implementation, migration, test, and follow-up-fix work, and teaches retained Tolbert/runtime plus universal decoder training to react to both stage transitions and phase handoff boundaries instead of only matching static obligation names. It now also derives an explicit software-work phase gate so unresolved merge/migration obligations can block later test/report moves in both deterministic routing and retained Tolbert scoring instead of relying only on soft agenda order. |
| `task_ecology` | deepen repo/workflow/integration task realism | `agent_kernel/curriculum.py`, `agent_kernel/curriculum_catalog.py`, `datasets/curriculum_templates.json`, `tests/test_curriculum.py` | harder long-horizon tasks are generated without losing verifier clarity |
| `memory_retrieval` | make prior evidence influence later coding steps | `agent_kernel/memory.py`, `agent_kernel/retrieval_improvement.py`, `agent_kernel/context_budget.py`, `tests/test_memory.py` | retrieval influence widens beyond selected-only metadata |
| `tolbert_runtime` | get model-native retained wins | `agent_kernel/modeling/`, `agent_kernel/tolbert_model_improvement.py`, `scripts/evaluate_tolbert_liftoff.py`, `tests/test_liftoff.py` | retained checkpoint evidence improves liftoff-relevant surfaces, not just wrappers |

## Runner Slots

| runner_slot | status | note |
| --- | --- | --- |
| `supervised_runner__codex_main` | `unclaimed` | shared coordinator slot only when queue/frontier/docs must move |
| `supervised_runner__open` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_2` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_3` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_4` | `claimed` | `supervised_world_codex_apr01_preview` world-state probe in progress; see claim block below |
| `supervised_runner__open_5` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_6` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_7` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_8` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_9` | `unclaimed` | general scoped worker slot |
| `supervised_runner__open_10` | `unclaimed` | general scoped worker slot |

Snapshot note: the slot table is advisory only while multiple Codex workers are active. Treat explicit claim blocks and machine-readable reports as authoritative before assuming a path is free. One additional Codex session is still assumed active beyond the explicitly recorded world-state claim.

## Lanes

| lane_id | status | owned paths summary |
| --- | --- | --- |
| `supervised_parallel_lane__parallel_orchestration` | `unclaimed` | coordination docs and manifest |
| `supervised_parallel_lane__observation_latency` | `unclaimed` | supervised timing and wrapper behavior |
| `supervised_parallel_lane__planner_selection` | `unclaimed` | planner ranking and target selection |
| `supervised_parallel_lane__learning_memory` | `unclaimed` | memory extraction and reuse |
| `supervised_parallel_lane__task_ecology` | `unclaimed` | task mix and curriculum |
| `supervised_parallel_lane__runtime_evidence` | `unclaimed` | runtime diagnostics and failure evidence |
| `supervised_parallel_lane__world_state` | `claimed` | `supervised_world_codex_apr01_preview` owns the world-model state-estimation surface |

Coordination note: at least two other Codex workers are active in parallel. One explicit claim is recorded below for `world_state`; any additional lane-owned modified files outside the orchestration surface should still be treated as in flight until their worker closes out in this ledger.

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
- status: `completed`
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
- during the ten-agent wave, prefer one claim per dispatch target from the `Ten-Agent Dispatch Target` table before opening secondary speculative work on the same surface
- keep `supervised_runner__codex_main` on orchestration only unless the queue is already current and another worker explicitly hands it an owned technical slice
- if the supervisor reports `canary_monitoring`, `rollback_pending`, or `rollback_validation_failed`, do not resume protected promotion outside the lifecycle gates recorded in `supervisor_loop_status.json`
- prefer frontier-backed candidate review over ad hoc cycle-file picking
- treat the top-level target fields in `supervisor_loop_status.json`, `supervisor_baseline_bootstrap_queue.json`, and `supervisor_trust_streak_recovery_queue.json` as the authoritative machine-owned queue pointer; do not scrape nested `entries` first
- keep this file short and current; move long analysis into [`docs/supervised_probe_history.md`](/data/agentkernel/docs/supervised_probe_history.md)

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T20:24:00Z`
- scope_id: `workspace_preview_shared_anchor_bridged_currentproof_multi_hybrid`
- progress_label: `workspace_preview_shared_anchor_bridged_currentproof_multi_hybrid`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'multi_core_current_proof_multi_hybrid_block_over_single_core_variant or frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata or shared_anchor_bridged_multi_hybrid_block_over_multi_edit' --cache-clear && pytest -q tests/test_policy.py -k 'frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata or multi_core_current_proof_multi_hybrid_block_over_single_core_variant or shared_anchor_bridged_multi_hybrid_block_over_multi_edit or frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or bridged_hidden_gap_region_block_task or current_proof_region_block or conflicting_alias_recovery_bundle' --cache-clear
```

- hypothesis: `The remaining shared-anchor decoder gap is the bridged/current-proof multi-hybrid tier: if current-proof and bridged hidden-gap repairs count as real hybrid families and keep their proof metadata through frontier composition, the strongest bounded block should preserve both provenance and score edge instead of flattening back toward generic block metadata or multi_edit.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T20:24:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so current_proof_region_block now counts as a shared-anchor hybrid family, frontier-composed bounded blocks retain current-proof provenance fields, and current-proof region scoring now carries the same shared-anchor core/hybrid bonuses as the other hidden-gap-proof paths.`
  - `Added focused tests/test_policy.py coverage for both sides of the ROI: one frontier test proves a shared-anchor overlap hybrid can merge with a current-proof region while preserving shared_anchor_core_count=2 and shared_anchor_hybrid_component_count=2, and one decode-time test proves the multi-core current-proof multi-hybrid bounded block outranks the single-core variant.`
  - `Added a bridged/current-proof decode-time fallback check: a shared-anchor bridged multi-hybrid bounded block still beats an equally localized multi_edit fallback on the same span.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'multi_core_current_proof_multi_hybrid_block_over_single_core_variant or frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata or shared_anchor_bridged_multi_hybrid_block_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata or multi_core_current_proof_multi_hybrid_block_over_single_core_variant or shared_anchor_bridged_multi_hybrid_block_over_multi_edit or frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or bridged_hidden_gap_region_block_task or current_proof_region_block or conflicting_alias_recovery_bundle' --cache-clear`
- next_recommendation: `The next decoder ROI is proof-quality canonicalization on same-span bounded competitors: when multiple localized block_replace candidates cover the same region, make the proof-complete multi-hybrid block outrank weaker partial-current-proof or lower-fidelity bounded alternatives without depending on a multi_edit fallback to expose the gap.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T20:13:00Z`
- scope_id: `workspace_preview_shared_anchor_multi_hybrid_core_fidelity`
- progress_label: `workspace_preview_shared_anchor_multi_hybrid_core_fidelity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_preserves_multiple_shared_anchor_cores or multi_core_shared_anchor_block_over_single_core_variant' --cache-clear && pytest -q tests/test_policy.py -k 'frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_preserves_multiple_shared_anchor_cores or multi_core_shared_anchor_block_over_single_core_variant or conflicting_alias_recovery_bundle or hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or exact_hidden_gap_region_block_probe' --cache-clear
```

- hypothesis: `The remaining shared-anchor decoder gap is not single-hybrid metadata anymore. It is true multi-hybrid canonicalization: when one bounded repair carries both overlap-recovery and hidden-gap-proof provenance, the frontier should preserve both hybrid families and the scorer should still let the multi-core bounded repair beat otherwise-equivalent weaker variants.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T20:13:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so shared_anchor_hybrid_component_count now counts hybrid families instead of collapsing them to a single boolean. A shared-anchor bounded repair can now simultaneously preserve overlap-recovery and hidden-gap-proof provenance.`
  - `Added focused tests/test_policy.py coverage for the true multi-hybrid path: one frontier test proves a composed shared-anchor block preserves shared_anchor_core_count=2 and shared_anchor_hybrid_component_count=2 while carrying both recovered_conflicting_alias and exact_hidden_gap_region_block, and one decode-time test proves the multi-core multi-hybrid bounded block outranks the otherwise-equivalent single-core variant.`
  - `Retest coverage confirmed the new path without regressing the adjacent single-hybrid overlap, exact-hidden-gap, bridged-hidden-gap, and exact multi-core shared-anchor slices.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_preserves_multiple_shared_anchor_cores or multi_core_shared_anchor_block_over_single_core_variant' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores or multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or frontier_preserves_multiple_shared_anchor_cores or multi_core_shared_anchor_block_over_single_core_variant or conflicting_alias_recovery_bundle or hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or exact_hidden_gap_region_block_probe' --cache-clear`
- next_recommendation: `The next decoder ROI is the bridged/current-proof multi-hybrid tier: preserve the same provenance and ranking when the hidden-gap side is a bridged or current-proof region rather than an exact hidden-gap block, and verify the best bounded multi-hybrid repair still beats an equally-localized multi_edit fallback on the same span.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T20:02:00Z`
- scope_id: `workspace_preview_shared_anchor_exact_hidden_gap_hybrid_fidelity`
- progress_label: `workspace_preview_shared_anchor_exact_hidden_gap_hybrid_fidelity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit' --cache-clear && pytest -q tests/test_policy.py -k 'shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or exact_hidden_gap_region_block_probe' --cache-clear
```

- hypothesis: `The remaining hybrid frontier gap is narrower than plain overlap or bridge fidelity: when a shared-anchor bounded block merges with an exact hidden-gap proof block, the canonical frontier repair should keep the expected-target proof metadata and retain explicit hybrid scoring instead of flattening back toward a generic exact hidden-gap block.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T20:02:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact hidden-gap proof metadata now survives frontier composition through hidden_gap_target_from_expected_content, and shared-anchor exact-hidden-gap cores now count as hybrid components instead of reporting shared_anchor_hybrid_component_count=0.`
  - `Strengthened hidden-gap scoring for the proof-bearing shared-anchor path: exact hidden-gap block_replace candidates now get an extra bounded bonus when they preserve shared-anchor lineage plus expected-target proof, which keeps the localized repair ahead of the fragmented multi_edit fallback on this hybrid lane.`
  - `Added focused tests/test_policy.py coverage for both layers: one frontier test proves a shared-anchor core plus exact hidden-gap proof block preserves hybrid metadata, and one decode-time test proves the corresponding bounded block_replace outranks multi_edit.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit or frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or exact_hidden_gap_region_block_probe' --cache-clear`
- next_recommendation: `The next decoder ROI is true multi-hybrid canonicalization: preserve provenance and ranking when one shared-anchor bounded repair absorbs both overlap/alias and hidden-gap proof components, and keep that fidelity when multiple shared-anchor cores compete for the same winning block.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:55:33Z`
- scope_id: `workspace_preview_shared_anchor_multicore_canonicalization`
- progress_label: `workspace_preview_shared_anchor_multicore_canonicalization`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'frontier_preserves_multiple_shared_anchor_cores or shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or shared_anchor_overlap_hybrid_block_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or multi_core_shared_anchor_block_over_single_core_variant' --cache-clear
```

- hypothesis: `The next decoder gap after single-hybrid recovery is not generic overlap scoring anymore. It is canonicalizing richer shared-anchor provenance: if one bounded block preserves two disjoint shared-anchor cores, the decoder should track that structure explicitly and prefer it over an otherwise similar single-core repair instead of tying them.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:55:33Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact-region and frontier-composed bounded repairs now carry shared_anchor_core_count in addition to shared_anchor_exact_neighbor_count and shared_anchor_hybrid_component_count.`
  - `Retuned shared-anchor block scoring so multi-core bounded repairs get explicit extra credit across exact, hidden-gap, bridged-hidden-gap, and overlap-recovery lanes instead of tying otherwise similar single-core variants.`
  - `Added focused tests/test_policy.py coverage proving frontier composition preserves two shared-anchor cores on one merged bounded block and proving decode-time ranking prefers the two-core bounded repair over a comparable single-core block.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'frontier_preserves_multiple_shared_anchor_cores or frontier_bridge_hybrid_preserves_shared_anchor_metadata or multi_core_shared_anchor_block_over_single_core_variant' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'frontier_preserves_multiple_shared_anchor_cores or shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or shared_anchor_overlap_hybrid_block_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or multi_core_shared_anchor_block_over_single_core_variant' --cache-clear`
- next_recommendation: `The remaining decoder ROI on this lane is full multi-hybrid canonicalization: one bounded repair that absorbs overlap recovery and hidden-gap proof at once while preserving the richer shared-anchor structure, rather than only preferring multi-core exact-region variants.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:28:00Z`
- scope_id: `workspace_preview_multi_window_modeling`
- progress_label: `workspace_preview_multi_window_modeling`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `When a large file has multiple distant localized edits, retaining and exposing all viable preview windows should improve coding quality more than forcing the stack to pick only one window per file.`
- owned_paths:
  - `agent_kernel/context_budget.py`
  - `agent_kernel/policy.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:28:00Z`
- result: `completed`
- findings:
  - `The world model already emitted multiple targeted preview windows under workspace_file_previews[path].edit_windows for distant edits, but downstream compaction and decoder logic still collapsed them to one retained window.`
  - `Patched agent_kernel/context_budget.py so each retained preview window becomes its own context chunk and selected windows are re-grouped back into workspace_file_previews[path].edit_windows; single-window paths keep the old chunk key for compatibility.`
  - `Patched agent_kernel/policy.py so compacted world_model_summary now preserves edit_windows in the LLM-facing payload instead of truncating each file preview to one visible window.`
  - `Patched agent_kernel/modeling/policy/decoder.py so TolBERT no longer picks only the single best retained window. It now emits per-window structured-edit proposals plus a bundled multi-window structured edit when non-overlapping windows can be applied safely in one command.`
  - `Fresh coverage now proves all three layers: tests/test_loop.py validates two retained world-model windows for one large file, tests/test_policy.py validates multi-window payload exposure, and TolBERT now selects a bundled multi-edit command for two distant localized replacements in one file.`
- artifacts:
  - `agent_kernel/context_budget.py`
  - `agent_kernel/policy.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
- verification:
  - `python -m py_compile agent_kernel/context_budget.py agent_kernel/policy.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `Use the new multi-window context surface to push on higher-order edit planning, especially when more than two retained windows exist or when bundled edits should be ranked against full-file rewrites under tighter budgets.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T03:02:00Z`
- scope_id: `workspace_preview_multi_window_planning`
- progress_label: `workspace_preview_multi_window_planning`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload or structured_edit_window_bonus"
```

- hypothesis: `Once multiple retained preview windows are visible, the next kernel gain is planning across 3+ windows: bundled structured edits should use the best safe subset, and larger safe bundles should outrank partial one-window edits.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so multi-window proposal composition is no longer all-or-nothing. The decoder now searches for the best non-overlapping subset of valid retained-window edits, which lets a 3-window file still emit a bundled multi_edit even when one candidate window overlaps another.`
  - `Structured-edit proposal scoring now scales with covered-window count and preview-window coverage, so 3+ safe bundles get materially more score than partial one-window proposals instead of flattening at the old 2-window cap.`
  - `Added tests/test_policy.py coverage for three-window bundled edits, best-subset bundling when one window overlaps, and the new 3+ window bonus scaling.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload or structured_edit_window_bonus"`
- next_recommendation: `Push the same bundle reasoning into line-shifting cases where adjacent inserts/deletes change downstream anchors, or teach ranking to compare partial preview bundles against bounded rewrite/block-replace alternatives using expected-coverage evidence.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T03:18:00Z`
- scope_id: `workspace_preview_line_shift_bundling`
- progress_label: `workspace_preview_line_shift_bundling`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or structured_edit_window_bonus or same_anchor_insert_and_replace or adjacent_delete_and_replace or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload"
```

- hypothesis: `The next preview-planning gain is line-shift awareness: a delete window with an explicit empty target span should still bundle safely with the adjacent retained edit or delete that shifts into that anchor, instead of dropping back to a partial one-window edit.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so truncated preview resolution preserves explicit empty target spans for delete windows instead of discarding them when target_edit_content is empty.`
  - `Extended bundled multi_edit safety so an adjacent delete can pair with the downstream retained edit or delete that shifts into the freed target anchor when exact target spans are available.`
  - `Adjusted action-generation scoring so bundled structural preview chains now carry component_line_deltas and structural_window_count, giving safe delete-shift multi_edits enough score to beat the partial single-window fallback in TolBERT primary selection.`
  - `Added tests/test_policy.py coverage for adjacent delete-plus-replace bundling, adjacent delete-plus-delete bundling, bundled structural metadata, and the full policy path that now selects the bundled delete-shift command.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or structured_edit_window_bonus or same_anchor_insert_and_replace or adjacent_delete_and_replace or includes_multi_window_workspace_file_previews_in_payload or includes_workspace_file_previews_in_payload"`
- next_recommendation: `The next ROI in this lane is the harder overlap class where multiple retained windows have ambiguous or partially inexact anchors, plus ranking work that uses expected-coverage evidence to decide between a partial preview bundle and a bounded rewrite/block-replace alternative.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T12:41:33Z`
- scope_id: `workspace_preview_partial_coverage_ranking`
- progress_label: `workspace_preview_partial_coverage_ranking`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "multiple_edit_windows or partial_retained_window_coverage or target_span_and_line_delta_for_insert_window" && pytest -q tests/test_policy.py -k "partial_preview_coverage_rank_task or multi_window_workspace_preview_can_emit_three_window_combined_structured_edit or multi_window_workspace_preview_can_emit_combined_structured_edit or adjacent_delete_and_replace or same_anchor_insert_and_replace"
```

- hypothesis: `When the world model can only retain a subset of a file's change windows, the decoder should treat that as explicit partial coverage and allow a bounded full write to outrank localized preview edits that no longer cover the whole outstanding delta.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so targeted preview windows now carry retained_edit_window_count, total_edit_window_count, and partial_window_coverage when the world model hits the retained-window cap.`
  - `Patched agent_kernel/modeling/policy/decoder.py so preview proposals preserve that coverage metadata and action-generation scoring applies a partial-coverage penalty when retained windows do not cover the full localized change set.`
  - `Added tests/test_loop.py coverage for the capped retained-window contract and tests/test_policy.py coverage proving a partial retained multi-window preview can correctly lose to expected_file_content while full-coverage preview bundles still win.`
- verification:
  - `pytest -q tests/test_loop.py -k "multiple_edit_windows or partial_retained_window_coverage or target_span_and_line_delta_for_insert_window"`
  - `pytest -q tests/test_policy.py -k "partial_preview_coverage_rank_task or multi_window_workspace_preview_can_emit_three_window_combined_structured_edit or multi_window_workspace_preview_can_emit_combined_structured_edit or adjacent_delete_and_replace or same_anchor_insert_and_replace"`
  - `python -m compileall agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The next ROI in this lane is no longer hidden-window coverage. The remaining hard case is ambiguous overlap and mixed exact/inexact retained anchors, especially when a partial structured bundle should compete against a localized block_replace rather than only against a bounded full write.`

### `supervised_runner__codex_main`
- scope_id: `workspace_preview_mixed_precision_ranking`
- progress_label: `workspace_preview_mixed_precision_ranking`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or structured_edit_window_bonus" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next preview-planning gain is not generic rewrite fallback. It is mixed-precision ranking: when a retained bundle mixes exact target anchors with inferred ones, the decoder should prefer exact-only safe subsets and only re-admit bounded rewrite fallback for that mixed-precision case or for explicit partial coverage, without breaking the existing all-inexact retained-window structured-edit path.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so preview-window step selection now prefers safer bounded block_replace-style edits for inexact multi-line windows, while keeping existing single-window inexact token edits intact.`
  - `Patched multi-window subset ranking so exact-only safe subsets beat mixed exact/inexact supersets, but all-inexact bundles still keep full retained-window coverage instead of collapsing to a smaller subset.`
  - `Patched action-generation fallback gating so expected_file_content is only re-admitted for explicit partial-window coverage or for mixed-precision retained bundles. A bad earlier retained window no longer forces rewrite fallback when a later valid localized preview edit still exists.`
  - `Added tests/test_policy.py coverage for exact-subset-over-mixed-superset ranking and for action-generation candidate lists where mixed-precision multi-window previews keep bounded rewrite fallback available and ranked above the risky mixed bundle.`
- verification:
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or structured_edit_window_bonus"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The next ROI in this lane is the real ambiguous-overlap class: partially inexact anchors that still collide or alias across windows, plus stronger ranking between mixed-precision partial bundles and localized block_replace alternatives inside the same file.`

- scope_id: `workspace_preview_overlap_recovery_block_replace`
- progress_label: `workspace_preview_overlap_recovery_block_replace`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or retained_preview_overlap_is_ambiguous or overlap_recovery_block_replace or window_spans_overlap" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py
```

- hypothesis: `The next preview-planning gain is the true overlap class: when retained windows overlap and a partial bundle is unsafe or incomplete, the decoder should synthesize a localized block_replace over the stitched visible region instead of competing only between a partial structured bundle and a whole-file rewrite.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so retained preview planning now synthesizes a merged structured_edit:block_replace candidate from overlapping or adjacent retained windows when their visible current and target spans can be stitched into one contiguous consistent region.`
  - `Patched decoder scoring so preview-derived multi-window block_replace candidates get explicit credit for overlap recovery, especially when they resolve mixed exact/inexact retained anchors without falling back to a whole-file write.`
  - `Added tests/test_policy.py coverage for overlap-recovery block_replace synthesis, for the positive policy decision where localized block_replace beats the partial bundle, and for the negative ambiguous-overlap case where inconsistent stitched content still falls back to expected_file_content.`
- verification:
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or retained_preview_overlap_is_ambiguous or overlap_recovery_block_replace or window_spans_overlap"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The next ROI in this lane is the remaining aliasing class: retained windows whose anchors still collide without yielding one consistent contiguous merged block, plus finer ranking between multiple bounded block-level alternatives when more than one stitched block_replace is plausible.`

- scope_id: `workspace_preview_alias_block_alternatives`
- progress_label: `workspace_preview_alias_block_alternatives`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or retained_preview_overlap_is_ambiguous or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or localized_block_replace or block_replace_rank_key or exact_localized_block_replace_for_alias_overlap" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py
```

- hypothesis: `The next aliasing gain is to preserve exact single-window block_replace alternatives when overlapping retained anchors make cheaper token edits brittle, then rank those bounded block candidates against stitched overlap candidates instead of leaving only one block-level option in play.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact retained-preview windows can emit both their cheapest local edit and an exact single-window structured_edit:block_replace, instead of suppressing the safer localized block candidate.`
  - `Annotated preview proposals with bounded block-alternative family metadata and adjusted action-generation scoring so exact localized block_replace candidates can outrank alias-prone token edits and weaker preview-derived block alternatives on the same file.`
  - `Added focused tests/test_policy.py coverage for exact single-window block preservation under alias overlap and for candidate ranking where the exact localized block_replace beats the same-window token edit.`
- verification:
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or partial_preview_coverage_rank_task or retained_preview_overlap_is_ambiguous or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or localized_block_replace or block_replace_rank_key or exact_localized_block_replace_for_alias_overlap"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The remaining ROI in this preview-modeling lane is bounded-region synthesis for conflicts that still cannot be resolved inside the retained windows at all, especially when the best safe edit needs a larger inferred block than any retained window or stitched overlap candidate currently exposes.`

- scope_id: `workspace_preview_alias_recovery_bundle_ranking`
- progress_label: `workspace_preview_alias_recovery_bundle_ranking`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "overlap_recovery_block_replace_with_distant_exact_window or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or structured_edit_window_bonus" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next ambiguous-alias gain is not another single block candidate. It is higher-order composition and ranking: when one retained region needs a localized overlap-recovery block_replace and a distant retained region is still exactly editable, the decoder should bundle them into one safe multi_edit and stop falling back to expected_file_content just because one component came from a mixed-precision overlap repair.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so overlap-recovery structured_edit:block_replace proposals can participate as composable units in higher-order multi-window planning instead of competing only as standalone alternatives.`
  - `Patched subset ranking and action-generation scoring so a hybrid preview-derived bundle can combine a localized alias-recovery block_replace with a distant exact edit, and mixed precision sourced only from that alias-recovery block no longer automatically forces expected_file_content fallback.`
  - `Added tests/test_policy.py coverage for a three-window hybrid multi_edit that emits block_replace plus token_replace in one command, and for action-generation candidate ranking where that hybrid bundle outranks both broader block_replace and bounded full-write fallback.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "overlap_recovery_block_replace_with_distant_exact_window or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or structured_edit_window_bonus"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this lane is the truly conflicting alias case: retained windows whose visible targets disagree enough that no localized stitched block is consistent, plus ranking between multiple plausible localized block regions when more than one hybrid overlap-recovery bundle exists.`

- scope_id: `workspace_preview_conflict_frontier_ranking`
- progress_label: `workspace_preview_conflict_frontier_ranking`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k "conflicting_alias_overlap_from_expected_target or propagate_recovered_conflicting_alias_into_hybrid_bundle or conflicting_alias_recovery_bundle_over_expected_write_fallback or frontier_localized_block_candidate_when_hybrid_bundles_tie or overlap_recovery_block_replace_with_distant_exact_window or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or structured_edit_window_bonus" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next overlap-recovery gain is to separate recovered conflicts from unresolved conflicts and to stop tie-like hybrid bundles from being ranked by incidental window order. Fully recovered alias bundles should still be viable localized edits, while unresolved alias components and weaker localized block regions should stay penalized.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so overlap-component conflict metadata now flows through both localized block_replace candidates and composed multi_edit bundles, including unresolved alias-pair counts, component frontier gaps, and recovered_conflicting_alias on hybrid bundles.`
  - `Adjusted subset ranking and action-generation scoring so fully recovered conflicting-alias bundles can still outrank broader fallbacks, while unresolved alias components and non-frontier localized block regions take explicit penalties instead of winning by incidental window-index order.`
  - `Added tests/test_policy.py coverage for recovered-conflict metadata propagation into hybrid bundles and for best-subset ranking where the stronger localized block frontier wins when two hybrid bundle shapes would otherwise tie.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "conflicting_alias_overlap_from_expected_target or propagate_recovered_conflicting_alias_into_hybrid_bundle or conflicting_alias_recovery_bundle_over_expected_write_fallback or frontier_localized_block_candidate_when_hybrid_bundles_tie or overlap_recovery_block_replace_with_distant_exact_window or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or overlap_recovery_block_replace or same_anchor_insert_and_replace or adjacent_delete_and_replace or structured_edit_window_bonus"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the no-safe-recovery alias component: cases where even expected-target conflict recovery cannot produce one localized bounded block, plus the larger hidden-gap class where the correct safe edit extends beyond every currently retained window union.`

- scope_id: `workspace_preview_exact_contiguous_region_block`
- progress_label: `workspace_preview_exact_contiguous_region_block`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "exact_contiguous_region_block or overlap_block_replace or skip_multi_edit_when_window_spans_overlap" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py
```

- hypothesis: `The safest next bounded-region gain is not hidden-gap inference. It is exact contiguous-region synthesis: when adjacent retained windows already provide a gap-free exact current/target region, the decoder should emit one localized block_replace and rank it against fragmented multi_edit alternatives.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact retained windows can synthesize a localized structured_edit:block_replace across adjacent windows when their visible current and target spans merge into one contiguous region without hidden gaps.`
  - `Patched action-generation scoring so exact contiguous-region block_replace candidates receive a narrow bounded-region bonus and can outrank fragmented multi_edit chains for that exact class.`
  - `Added focused tests/test_policy.py coverage for contiguous-region block synthesis, for the ranking case where the exact contiguous block beats the equivalent multi_edit, and for the negative guard that refuses hidden-gap region synthesis.`
- verification:
  - `pytest -q tests/test_policy.py -k "exact_contiguous_region_block or overlap_block_replace or skip_multi_edit_when_window_spans_overlap"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the hidden-gap region class: cases where the best safe bounded edit is larger than any visible exact retained-window union, and the decoder needs stronger evidence than the current window payloads provide.`

- scope_id: `workspace_preview_exact_hidden_gap_region_block`
- progress_label: `workspace_preview_exact_hidden_gap_region_block`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "exact_hidden_gap_region_block or exact_contiguous_region_block or skip_multi_edit_when_window_spans_overlap" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py
```

- hypothesis: `The narrow next hidden-gap gain does not need new world-model payloads. When consecutive retained windows are exact and share one expected target file, the decoder can synthesize one exact larger block_replace across the hidden gap by anchoring to the visible spans and slicing the task target over the full bounded region.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact retained-window region synthesis now has a hidden-gap branch: when consecutive exact windows cannot merge through visible current lines, the decoder can still emit one exact structured_edit:block_replace across the bounded hidden-gap region.`
  - `Patched scoring so exact hidden-gap region block_replace candidates get their own bounded-region bonus and can outrank fragmented multi_edit chains when that larger exact block is the safer local plan.`
  - `Added focused tests/test_policy.py coverage for hidden-gap block synthesis, for the ranking case where the exact hidden-gap block beats the equivalent multi_edit, and for the negative guard that refuses hidden-gap synthesis when the subset skips an intervening retained window.`
- verification:
  - `pytest -q tests/test_policy.py -k "exact_hidden_gap_region_block or exact_contiguous_region_block or skip_multi_edit_when_window_spans_overlap"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the stronger-evidence hidden-gap class: bounded regions that still cannot be justified from exact retained endpoints plus the task target alone, especially when the best local block would cross mixed-precision windows or require explicit unseen-current evidence.`

- scope_id: `workspace_preview_bridged_hidden_gap_region_block`
- progress_label: `workspace_preview_bridged_hidden_gap_region_block`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "hidden_gap_bridge" && pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or exact_hidden_gap_region_block or mixed_precision_preview_fallback" && python -m compileall agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py
```

- hypothesis: `The next hidden-gap ROI needs explicit world-model evidence, not more decoder-only inference. If the world model emits compact bridge payloads for the unseen current/target gap between consecutive retained windows, the decoder can safely recover a larger bounded block_replace even when the retained windows themselves are mixed-precision.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so retained workspace previews can now expose bridged_edit_windows: compact hidden-gap proof records carrying consecutive source window indices, exact hidden current/target gap spans, and explicit hidden-gap current/target content when that gap itself fits preview budget.`
  - `Patched agent_kernel/modeling/policy/decoder.py so those bridge records synthesize a stronger-evidence structured_edit:block_replace across the bounded hidden-gap region, with explicit proof metadata and a dedicated score lift over expected-write fallback and weaker fragmented alternatives.`
  - `Added focused tests/test_loop.py coverage for bridged hidden-gap world-model payloads, tests/test_policy.py coverage for positive bridged hidden-gap block synthesis and ranking, and a negative guard that refuses bridge records which skip an intervening retained window.`
- verification:
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge"`
  - `pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or exact_hidden_gap_region_block or mixed_precision_preview_fallback"`
  - `python -m compileall agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The remaining ROI in this preview-modeling lane is hidden-gap structural drift: bounded bridge payloads where the unseen gap itself has insert/delete line_delta or where more than one consecutive hidden-gap bridge must be chained without reopening skipped-window ambiguity.`

- scope_id: `workspace_preview_proof_only_current_bridge`
- progress_label: `workspace_preview_proof_only_current_bridge`
- status: `completed`
- branch: `main`
- mapped_test_commands:
```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "hidden_gap_bridge or current_only_hidden_gap_bridge or proof_only_hidden_gap_bridge" && pytest -q tests/test_policy.py -k "proof_only_current_bridge or bridged_hidden_gap_region_block or expected_target_reconstruction" && python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py
```

- hypothesis: `The next upstream hidden-gap ROI is proof-only current evidence. If the world model preserves exact hidden current line-span proof even when the hidden current payload itself is too large to inline, the decoder can still recover a bounded bridge-backed block_replace from exact spans plus task-target reconstruction instead of reopening proofless broader-current fallback.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so bridged_edit_windows and bridged_edit_window_runs survive when hidden current payloads exceed preview budget, carrying hidden_gap_current_from_line_span_proof plus exact gap line coordinates/counts instead of dropping the bridge entirely.`
  - `Patched agent_kernel/modeling/policy/decoder.py so bridge-backed bounded block_replace synthesis no longer requires inline hidden current content; when current proof is span-only, it builds the exact region directly from line spans and target reconstruction while preserving bridge metadata on the proposal.`
  - `Added focused tests/test_loop.py coverage for proof-only hidden-gap bridge emission and tests/test_policy.py coverage proving a proof-only current bridge outranks fallback as a bridged hidden-gap block candidate.`
- verification:
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge or current_only_hidden_gap_bridge or proof_only_hidden_gap_bridge or preview_window or multi_window_preview_task"`
  - `pytest -q tests/test_policy.py -k "proof_only_current_bridge or bridged_hidden_gap_region_block or expected_target_reconstruction or current_hidden_gap_requires_proof or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous"`
  - `pytest -q tests/test_policy.py -k "hidden_gap or bridged_hidden_gap or mixed_precision_preview_fallback or exact_hidden_gap_region_block or current_hidden_gap"`
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The remaining upstream ROI is broader current-state evidence beyond one consecutive bridge span: multi-segment or non-adjacent current-proof structures where the correct bounded block still cannot be justified from a single hidden-gap bridge record.`

- scope_id: `workspace_preview_bridge_runs_across_omitted_proofs`
- progress_label: `workspace_preview_bridge_runs_across_omitted_proofs`
- status: `completed`
- branch: `main`
- mapped_test_commands:
```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "hidden_gap_bridge or omitted_exact_proof_windows" && pytest -q tests/test_policy.py -k "proof_only_current_bridge or omitted_exact_proof_window" && python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py
```

- hypothesis: `The next upstream hidden-gap ROI is not another decoder-only penalty. If the world model emits bridge runs across the full diff-window sequence and exact-proof omitted windows carry reconstructed target slices, the decoder can promote one bounded bridge-backed block_replace across retained windows plus an omitted exact-proof tail instead of stopping at the retained-window cap.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so bridged_edit_windows and bridged_edit_window_runs are generated from the full diff-window list rather than only the retained preview subset, which keeps bridge segments alive when the rightmost endpoint is an omitted exact-proof window.`
  - `Patched agent_kernel/modeling/policy/decoder.py so exact_edit_window_proofs reconstruct target_content from the task target and can participate as bridge-run endpoints, allowing one bridge-backed block_replace to cover retained windows plus an omitted proof-backed tail.`
  - `Added tests/test_loop.py coverage for a compacted bridge run that extends through omitted exact-proof window index 3, and tests/test_policy.py coverage proving the decoder prefers that 4-window bridged block over expected-file fallback.`
- verification:
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge or omitted_exact_proof_windows"`
  - `pytest -q tests/test_policy.py -k "proof_only_current_bridge or omitted_exact_proof_window"`
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The remaining upstream ROI is the genuinely broader current-evidence case: files where the safe localized block depends on multiple disjoint unseen-current spans that do not collapse into one linear bridge chain even after omitted exact-proof windows re-enter the proof graph.`

- scope_id: `workspace_preview_multi_span_current_proof_regions`
- progress_label: `workspace_preview_multi_span_current_proof_regions`
- status: `completed`
- branch: `main`
- mapped_test_commands:
```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "omitted_exact_proof_windows or hidden_gap_bridge" && pytest -q tests/test_policy.py -k "omitted_exact_proof_window or multi_span_current_proof_region" && python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py
```

- hypothesis: `The next upstream hidden-gap ROI needs a region-level proof contract rather than more pairwise chaining. If the world model emits one bounded-region payload that lists every disjoint unseen-current proof span inside the region, the decoder can validate complete current/target coverage and emit a single exact block_replace even when there is no explicit bridge-chain payload to walk.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so compacted bridge runs also emit hidden_gap_current_proof_regions with multiple disjoint proof spans across one localized bounded region.`
  - `Patched agent_kernel/modeling/policy/decoder.py so those region payloads can produce exact structured_edit:block_replace candidates by validating that visible windows plus proof spans fully cover the region on both current and target sides, without depending on a bridge-chain walk.`
  - `Added tests/test_loop.py coverage for emitted current-proof regions and tests/test_policy.py coverage proving a multi-span current-proof region outranks expected-file fallback even without any bridged_edit_window_runs payload.`
- verification:
  - `pytest -q tests/test_loop.py -k "omitted_exact_proof_windows or hidden_gap_bridge"`
  - `pytest -q tests/test_policy.py -k "omitted_exact_proof_window or multi_span_current_proof_region"`
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The remaining upstream ROI is partial-coverage bounded regions: cases where the world model can prove several disjoint unseen-current spans inside the correct local block, but not every hidden line in that region, so the kernel needs a principled boundary between promotable bounded repair and fallback.`

- scope_id: `workspace_preview_chained_hidden_gap_bridges`
- progress_label: `workspace_preview_chained_hidden_gap_bridges`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block or mixed_precision_preview_fallback" && pytest -q tests/test_loop.py -k "hidden_gap_bridge" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py tests/test_loop.py
```

- hypothesis: `The next hidden-gap ROI does not need a broader world-model payload if the existing pairwise bridge records already carry exact hidden current/target gap spans and contents. The decoder can widen those into larger bounded block_replace regions by requiring a complete consecutive bridge chain and validating each bridge against the derived region offsets.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so bridge-backed hidden-gap regions now support structural drift inside the unseen gap itself: explicit bridge payloads can insert or delete lines and still synthesize one bounded structured_edit:block_replace when the bridge spans and contents validate exactly.`
  - `Patched agent_kernel/modeling/policy/decoder.py so multiple consecutive bridge records can chain into one larger bounded block_replace across more than two retained windows, with strict adjacency and per-bridge span validation to avoid reopening skipped-window ambiguity.`
  - `Added focused tests/test_policy.py coverage for bridge-backed gap insertion, multi-bridge chaining, a missing-intermediate-bridge negative guard, and top-level candidate selection of the chained bridge block over broader fallback.`
- verification:
  - `pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block or mixed_precision_preview_fallback"`
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py tests/test_loop.py`
- next_recommendation: `The remaining ROI in this preview-modeling lane is stronger world-model bridge compaction: emit maximal bridge runs or per-gap segment arrays directly from the world model so decoder-side chaining can stay simpler while preserving exact metadata for mixed-precision bridge frontiers and larger ambiguity classes.`

- scope_id: `workspace_preview_bridge_run_payload`
- progress_label: `workspace_preview_bridge_run_payload`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k "hidden_gap_bridge" && pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block or bridge_run_payload" && python -m compileall agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py
```

- hypothesis: `The next hidden-gap ROI needs a narrower world-model contract, not more decoder-side chain synthesis. If the world model emits canonical maximal bridge runs with explicit per-gap bridge segments, the decoder can consume larger mixed-precision hidden-gap frontiers directly and reduce ranking ambiguity without dropping legacy pairwise support.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so retained previews now expose bridged_edit_window_runs: canonical maximal consecutive bridge runs with ordered bridge_segments, cumulative hidden-gap counts, and aggregate current/target spans.`
  - `Patched agent_kernel/modeling/policy/decoder.py so bridge-run payloads are normalized directly from workspace_file_previews, validated against absolute run spans, and consumed before falling back to legacy pairwise bridge chaining.`
  - `Added focused tests/test_loop.py coverage for pairwise bridge runs and three-window maximal bridge runs, plus tests/test_policy.py coverage showing the decoder prefers a direct bridge-run payload over expected-write fallback.`
- verification:
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge"`
  - `pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block or bridge_run_payload"`
  - `python -m compileall agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
- next_recommendation: `The next ROI in this preview-modeling lane is bridge-run frontier ranking: use the run payload to compare multiple plausible bounded bridge frontiers directly, especially when one maximal run competes with smaller exact localized blocks or partial retained coverage.`

- scope_id: `workspace_preview_bridge_run_frontier_ranking`
- progress_label: `workspace_preview_bridge_run_frontier_ranking`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_policy.py -k "bridge_run_payload or bridge_run_frontier or partial_bridge_run_vs_localized_exact or mixed_precision_preview_fallback or bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block" && pytest -q tests/test_loop.py -k "hidden_gap_bridge" && python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py
```

- hypothesis: `The next hidden-gap ROI is ranking, not more synthesis. If the decoder annotates bridge-run block_replace proposals as their own frontier family and also tracks exact localized block competitors, maximal full-coverage bridge runs can beat shorter bridge alternatives cleanly while partial retained bridge runs can yield to smaller exact localized repairs without collapsing back to generic expected-write fallback.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/runtime.md`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so preview block families now annotate bridge-run alternatives, bridge-run frontier gaps, partial bridge-run counts, and exact localized block competitors.`
  - `Patched agent_kernel/modeling/policy/decoder.py scoring so maximal bridge-run frontiers get an explicit bonus, off-frontier or partial bridge runs get penalized against exact localized competitors, and exact localized block_replace repairs get a narrow counter-bonus when the competing bridge-run family is partial.`
  - `Added focused tests/test_policy.py coverage for maximal bridge-run frontier selection and for the safety case where a smaller exact localized block beats a partial maximal bridge run.`
- verification:
  - `pytest -q tests/test_policy.py -k "bridge_run_payload or bridge_run_frontier or partial_bridge_run_vs_localized_exact or mixed_precision_preview_fallback or bridged_hidden_gap_region_block or chained_bridged_hidden_gap_region_block"`
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge"`
  - `python -m compileall agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
- next_recommendation: `The next ROI in this preview-modeling lane is bridge-run alias resolution: bridge-run frontiers whose retained windows or hidden-gap segments still admit more than one exact bounded block family, especially when two maximal runs overlap but neither cleanly dominates on exactness plus retained coverage.`

- scope_id: `workspace_preview_unrecoverable_alias_hidden_gap_fallback`
- progress_label: `workspace_preview_unrecoverable_alias_hidden_gap_fallback`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k "hidden_gap_requires_broader_region or exact_overlap_component_has_no_localized_recovery or all_inexact_overlap_component_has_no_safe_recovery or conflicting_alias_recovery_bundle_over_expected_write_fallback or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback" && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next residual decoder gap is no longer recovered conflict ranking. It is the harder fallback boundary: when an overlap component has no safe localized recovery at all, or when hidden retained-window gaps imply the safe bounded edit extends beyond every visible union, preview-derived structured edits should stop outranking bounded expected-write fallback.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so workspace-preview proposals now carry file-level unrecoverable overlap-component counts and hidden-gap broader-region pressure, derived from overlap-component recovery failure and missing target/current lines across the retained-window union.`
  - `Patched action-generation scoring so preview-derived structured edits take explicit penalties when they coexist with an unrecoverable alias component or a hidden broader bounded region, which lets expected_file_content retake priority only in those genuinely under-justified cases instead of broadly suppressing valid localized recovery bundles.`
  - `Added focused tests/test_policy.py coverage for both new boundaries: one exact-plus-distant preview case where a conflicting overlap component has no safe localized recovery, and one retained hidden-gap case where the task target implies the safe bounded region is larger than any retained-window union.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "hidden_gap_requires_broader_region or exact_overlap_component_has_no_localized_recovery or all_inexact_overlap_component_has_no_safe_recovery or conflicting_alias_recovery_bundle_over_expected_write_fallback or alias_recovery_bundle_over_expected_write_fallback or mixed_precision_preview_fallback"`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the stronger-evidence hidden-gap class: cases where the safe localized block still cannot be justified from retained windows plus task target alone because unseen current-state lines or multiple mixed-precision anchor layouts leave several plausible bounded regions alive.`

- scope_id: `workspace_preview_hidden_gap_frontier_ambiguity`
- progress_label: `workspace_preview_hidden_gap_frontier_ambiguity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k "hidden_gap_frontier_is_ambiguous or hidden_gap_requires_broader_region or exact_hidden_gap_region_block or exact_overlap_component_has_no_localized_recovery or all_inexact_overlap_component_has_no_safe_recovery" && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next hidden-gap ROI is no longer basic broader-region synthesis. It is ambiguity detection: when multiple best-ranked broader block frontiers remain plausible after preview synthesis, the decoder should stop pretending one local block is justified and let bounded rewrite fallback compete again.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so workspace-preview proposals now carry hidden-gap frontier ambiguity metadata, based on tied best-ranked broader block_replace candidates that still leave more than one plausible local region alive.`
  - `Patched fallback and scoring so that ambiguity class re-enables expected_file_content fallback and penalizes preview-derived structured edits only for those underdetermined broader-region ties, without suppressing ordinary localized overlap recovery or exact hidden-gap blocks that remain uniquely justified.`
  - `Added tests/test_policy.py coverage for both pieces: a workspace-preview integration case where ambiguous broader hidden-gap frontiers now flip allow_expected_write_fallback back on, and an action-generation ranking case where expected_file_content retakes first place when the broader-region frontier remains ambiguous.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "hidden_gap_frontier_is_ambiguous or hidden_gap_requires_broader_region or exact_hidden_gap_region_block or exact_overlap_component_has_no_localized_recovery or all_inexact_overlap_component_has_no_safe_recovery"`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the stronger-evidence hidden-gap class that still survives after frontier-tie detection: cases where retained windows plus task target cannot choose among several broader local blocks because the missing evidence is about unseen current-state lines, not just target-side bounded regions.`

- scope_id: `workspace_preview_current_hidden_gap_ambiguity`
- progress_label: `workspace_preview_current_hidden_gap_ambiguity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k "current_hidden_gap_frontier_is_ambiguous or hidden_gap_frontier_is_ambiguous or workspace_preview_range_can_batch_same_anchor_insert_and_replace or chained_bridged_hidden_gap_region_block_over_expected_write" && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The remaining stronger-evidence hidden-gap class is not just tied target-side frontier synthesis. The decoder needs to represent when target evidence is already exhausted but unseen current-state lines still leave several broader local blocks plausible, and that current-dominant ambiguity should reopen expected-write fallback instead of letting one inferred block win on smaller current-span heuristics alone.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so workspace-preview proposals now carry current-side hidden-gap ambiguity metadata in addition to the earlier target-side frontier ambiguity: when broader block_replace candidates tie on target evidence but differ only in unseen current-state assumptions, the file is marked underdetermined and expected_file_content fallback is re-enabled.`
  - `Patched scoring so current-dominant hidden-gap ambiguity directly penalizes those preview-derived broader blocks, and tightened the bounded-alternative penalty so exact bounded repairs can outrank speculative exact hidden-gap blocks when a zero-gap bounded repair already exists.`
  - `Patched frontier/shared-anchor follow-on behavior in agent_kernel/modeling/policy/decoder.py so chained bridged hidden-gap proof metadata survives frontier block merges, and shared-anchor exact insert/replace recovery can validate against the final task target instead of being rejected by an intermediate window-local target mismatch.`
  - `Added tests/test_policy.py coverage for current-side hidden-gap ambiguity at both proposal-assembly and action-generation ranking layers, and repaired nearby hidden-gap/shared-anchor test fixtures so the focused decoder shard reflects live behavior instead of masked harness errors.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "current_hidden_gap_frontier_is_ambiguous or hidden_gap_frontier_is_ambiguous or workspace_preview_range_can_batch_same_anchor_insert_and_replace or chained_bridged_hidden_gap_region_block_over_expected_write"`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is the no-proof broader-current class beyond frontier ambiguity: cases where even target-equivalent ties plus retained previews are insufficient because choosing the right localized block would require additional unseen current-state evidence, not just better local ranking.`

- scope_id: `workspace_preview_current_hidden_gap_proof_required`
- progress_label: `workspace_preview_current_hidden_gap_proof_required`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k "current_hidden_gap_requires_proof or current_hidden_gap_frontier_is_ambiguous or workspace_preview_range_can_batch_same_anchor_insert_and_replace or chained_bridged_hidden_gap_region_block_over_expected_write or hidden_gap_bounded_alternative_task" && pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous or current_hidden_gap_requires_proof" && pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"
```

- hypothesis: `The next stronger-evidence hidden-gap gain is the post-ambiguity no-proof class: even when current-side ranking produces one apparent broader local block, that block should not be trusted if it still depends on unseen current-state evidence that the world model never supplied.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/modeling/policy/decoder.py so workspace-preview proposals now mark a hidden_gap_current_proof_required class when a proofless mixed-precision broader block_replace remains current-dominant after ranking but still lacks explicit hidden-gap current evidence.`
  - `Patched fallback and scoring so that unique no-proof broader-current block_replace candidates no longer survive on ranking alone: they reopen expected_file_content fallback and take a dedicated current-proof penalty until bridge-backed or other explicit current evidence exists.`
  - `Added tests/test_policy.py coverage for both layers: proposal assembly now marks proof-required broader-current files as fallback-eligible, and action-generation ranking now prefers expected_file_content over a unique proofless broader-current block.`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "current_hidden_gap_requires_proof or current_hidden_gap_frontier_is_ambiguous or workspace_preview_range_can_batch_same_anchor_insert_and_replace or chained_bridged_hidden_gap_region_block_over_expected_write or hidden_gap_bounded_alternative_task"`
  - `pytest -q tests/test_policy.py -k "workspace_preview_range or multi_window_workspace_preview or overlap_recovery_block_replace or conflicting_alias or hidden_gap or mixed_precision_preview_fallback or exact_subset_over_inexact_superset or alias_recovery_bundle_over_expected_write_fallback or structured_edit_window_bonus or exact_hidden_gap_region_block or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous or current_hidden_gap_requires_proof"`
  - `pytest -q tests/test_loop.py -k "workspace_file_previews or preview_window or multi_window_preview_task"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is upstream world-model evidence: bridge or broader current-state proof payloads for files where the right localized block is still impossible to justify from retained previews and task target alone.`

- scope_id: `workspace_preview_current_proof_bridge_reconstruction`
- progress_label: `workspace_preview_current_proof_bridge_reconstruction`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py && pytest -q tests/test_loop.py -k "hidden_gap_bridge or current_only_hidden_gap_bridge or preview_window or multi_window_preview_task" && pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or expected_target_reconstruction or current_hidden_gap_requires_proof or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous" && pytest -q tests/test_policy.py -k "hidden_gap or bridged_hidden_gap or mixed_precision_preview_fallback or exact_hidden_gap_region_block or current_hidden_gap"
```

- hypothesis: `The next upstream hidden-gap gain does not require a broader decoder heuristic. If the world model preserves explicit hidden-gap current proof even when the hidden target payload is too expensive to inline, the decoder can reconstruct that target span from the task target and still recover one bounded bridge-backed block_replace instead of reopening whole-file fallback.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- outcome:
  - `Patched agent_kernel/world_model.py so bridged hidden-gap preview payloads now keep explicit current proof even when the hidden target gap itself is omitted under preview budget, via hidden_gap_target_from_expected_content on both pairwise bridges and compacted bridge runs.`
  - `Patched agent_kernel/modeling/policy/decoder.py so bridge-backed block_replace recovery can reconstruct omitted hidden target spans from the shared full task target whenever the world model marked that bridge as target-reconstructible.`
  - `Added tests/test_loop.py coverage for current-only hidden-gap bridge payloads and tests/test_policy.py coverage proving the decoder prefers the bridge-backed bounded block when only current proof plus target-span coordinates are present.`
- verification:
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
  - `pytest -q tests/test_loop.py -k "hidden_gap_bridge or current_only_hidden_gap_bridge or preview_window or multi_window_preview_task"`
  - `pytest -q tests/test_policy.py -k "bridged_hidden_gap_region_block or expected_target_reconstruction or current_hidden_gap_requires_proof or hidden_gap_frontier_is_ambiguous or current_hidden_gap_frontier_is_ambiguous"`
  - `pytest -q tests/test_policy.py -k "hidden_gap or bridged_hidden_gap or mixed_precision_preview_fallback or exact_hidden_gap_region_block or current_hidden_gap"`
- next_recommendation: `The remaining ROI in this preview-modeling lane is broader current-state evidence beyond adjacent bridgeable gaps: files where the safe localized block still depends on unseen current lines that do not collapse into one consecutive hidden-gap bridge span.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:16:50Z`
- scope_id: `supervisor_bootstrap_evidence_policy`
- progress_label: `supervisor_bootstrap_evidence_policy`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode promote --rollout-stage compare_only --bootstrap-finalize-policy evidence --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `A new evidence-based bootstrap finalize policy should widen autonomy only when the unattended trust ledger shows a clean streak with no recent false-pass or unexpected-change risk; otherwise it should stay safely blocked even in promote mode.`
- owned_paths:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:16:50Z`
- result: `completed`
- findings:
  - `Patched scripts/run_supervisor_loop.py to support --bootstrap-finalize-policy evidence in addition to operator_review, trusted, and allow.`
  - `The new evidence policy allows non-protected bootstrap finalize only when the unattended trust ledger passes and shows clean_success_streak >= meta_trust_clean_success_streak with false_pass_risk_count=0 and unexpected_change_report_count=0; protected subsystems still require the stronger protected clean-streak gate.`
  - `Added regression coverage in tests/test_generation_scripts.py for both sides: clean bootstrap evidence enables non-protected finalize, and bootstrap evidence with low streak plus risk signals blocks finalize with explicit policy reasons.`
  - `Fresh live validation under promote/compare_only stayed safely blocked on current repo evidence. supervisor_loop_report.json now records bootstrap_finalize_policy=evidence with apply_finalize=true but allow_bootstrap_finalize=false and bootstrap_policy_reasons=[bootstrap_clean_success_streak:0<2], while the machine-owned target remains tooling scope supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert.`
- artifacts:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
- verification:
  - `python -m py_compile scripts/run_supervisor_loop.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k "allows_bootstrap_finalize_only_when_trusted or allows_bootstrap_finalize_with_clean_bootstrap_evidence or blocks_evidence_bootstrap_finalize_on_risk_signal or requires_stronger_clean_streak_for_protected_bootstrap_finalize"`
  - `python scripts/run_supervisor_loop.py --autonomy-mode promote --rollout-stage compare_only --bootstrap-finalize-policy evidence --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Move the next engineering cycle onto unattended trust evidence generation or trust-ledger quality so evidence mode can eventually unlock non-protected bootstrap finalize on real runs.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:12:17Z`
- scope_id: `supervisor_trust_artifact_policy_context`
- progress_label: `supervisor_trust_artifact_policy_context`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `If the supervisor queue/status reports carry trust-policy context at top level, bootstrap review and autonomy decisions can remain reliable without reading nested round policy payloads.`
- owned_paths:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:12:17Z`
- result: `completed`
- findings:
  - `Patched scripts/run_supervisor_loop.py so bootstrap remediation package actions carry queue_kind, autonomy_mode, rollout_stage, bootstrap_finalize_policy, and trust_status into the emitted queue reports.`
  - `supervisor_loop_status.json now exposes bootstrap_finalize_policy at top level instead of only under latest_round.policy.`
  - `Added regression coverage in tests/test_generation_scripts.py for the queue package shape and the top-level status policy field.`
  - `After a fresh dry-run supervisor round, supervisor_baseline_bootstrap_queue.json and supervisor_trust_streak_recovery_queue.json now carry top-level autonomy_mode=dry_run, rollout_stage=compare_only, bootstrap_finalize_policy=operator_review, trust_status=bootstrap, and queue_kind, while the machine-owned target remains tooling scope supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert.`
- artifacts:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `python -m py_compile scripts/run_supervisor_loop.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k "execute_action_writes_bootstrap_review_package_report or run_supervisor_loop_writes_status_history_and_report or run_supervisor_loop_rebuilds_lane_signal_from_fresh_promotion_pass"`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Keep the machine-owned tooling bootstrap target, but move the next engineering cycle onto unattended review quality or bootstrap-autonomy policy rather than more queue/report plumbing.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:12:00Z`
- scope_id: `supervised_coordination_codex_apr02_fanoutguard_budget20`
- progress_label: `supervised_coordination_codex_apr02_fanoutguard_budget20`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_fanoutguard_budget20 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_fanoutguard_budget20 --notes "guard bounded tooling repo/project fanout so 20-second generated-success budgets enter the proven 4-task primary slice up front"
```

- hypothesis: `If a bounded tooling repository/project run asks for a 5-task primary but only a 20-second generated-success budget, the wrapper should enter the proven 4-task primary slice immediately instead of timing out once and recovering on retry.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:18:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so bounded tooling repository/project runs can apply tooling_repository_project_fanout_guard: when task_limit=5 and the generated-success budget is too small for full fanout, the wrapper now drops to the proven 4-task primary slice before observation starts.`
  - `The change is narrow. It does not disturb the wider healthy TolBERT-backed 5-task path when the fanout budget is strong enough; the existing 40-second preserve behavior still holds.`
  - `Added protocol coverage in tests/test_protocol_comparison.py for both sides: the 5-task tooling primary is preserved under the strong-budget path, and reduced to 4 with TolBERT disabled under the 20-second bounded fanout path.`
  - `Fresh live evidence is now in trajectories/improvement/cycles_supervised_coordination_codex_apr02_fanoutguard_budget20.jsonl. The scope entered primary_task_limit=4/5 immediately, disabled TolBERT context up front, completed healthy in 41.0609 seconds with observation_budget_exceeded=false, merged generated_success at 2/2, and produced a fresh tooling candidate artifact.`
  - `The machine-owned queue target did not change after refresh. supervisor_loop_status.json still points at tooling scope supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert under trust_streak_accumulation, so this new 20-second scope is a runtime-stability gain rather than a new top review target.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_fanoutguard_budget20.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_fanoutguard_budget20/tooling/cycle_tooling_20260402T020510381568Z_70a64ba7_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "reduces_five_task_tooling_primary_when_generated_budget_cannot_support_full_fanout or preserves_five_task_tooling_primary_when_tolbert_budget_is_strong or run_supervised_cycle_disables_tolbert_context_for_bounded_transition_model_run or retries_generated_success_followup_with_reduced_task_limit or salvages_partial_generated_success_metrics_on_timeout or retries_with_reduced_primary_task_limit_after_timeout"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_fanoutguard_budget20 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_fanoutguard_budget20 --notes "guard bounded tooling repo/project fanout so 20-second generated-success budgets enter the proven 4-task primary slice up front"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `The bounded repository/project tooling wrapper is no longer the main blocker at either 40-second or 20-second generated-success budgets. The next ROI is review/autonomy work on the existing top tooling bootstrap target, or broader trust-quality improvements outside this already-stabilized tooling slice.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:05:00Z`
- scope_id: `supervisor_ingestion_top_level_target_sync`
- progress_label: `supervisor_ingestion_top_level_target_sync`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `If supervisor remediation reports surface their primary target fields at top level, the orchestration lane and parallel workers can consume the live machine-owned queue directly without ad hoc JSON digging, and the next ROI decision can shift back to runtime quality instead of ingestion repair.`
- owned_paths:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:10:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_supervisor_loop.py so remediation queue reports now emit primary target fields at top level: selected_subsystem, scope_id, cycle_id, candidate_artifact_path, review_finalize_command, and primary_entry.`
  - `Patched supervisor_loop_status.json generation so the live machine-owned queue target is also surfaced directly through trust_status, rollout_stage, selected_queue_kind, selected_queue_entry, selected_subsystem, scope_id, cycle_id, and candidate_artifact_path.`
  - `Added regression coverage in tests/test_generation_scripts.py for queue payload shape and for rebuilding the live lane signal from a fresh promotion pass.`
  - `After refreshing the promotion plan and supervisor round, the actionable machine-owned queue target is explicit again: tooling scope supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert under trust_streak_accumulation, mirrored into baseline_bootstrap for review packaging.`
  - `The sparse-artifact issue is resolved. The next ROI is runtime budget discipline on bounded repository/project tooling scopes, not promotion/supervisor ingestion.`
- artifacts:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `python -m py_compile scripts/run_supervisor_loop.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k "test_execute_action_writes_bootstrap_review_package_report or test_run_supervisor_loop_rebuilds_lane_signal_from_fresh_promotion_pass or test_run_supervisor_loop_writes_status_history_and_report"`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Keep the supervisor ingestion shape stable and move the next engineering cycle onto runtime budget discipline for bounded repository/project tooling scopes.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:35:00Z`
- scope_id: `supervised_coordination_codex_apr02_toolbridge_followup_retry`
- progress_label: `supervised_coordination_codex_apr02_toolbridge_followup_retry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_followup_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_followup_retry --notes "retry generated_success followup with reduced slice after partial timeout so tooling scopes can retain at least one adjacent-success result under bounded budget"
```

- hypothesis: `If a bounded generated_success tooling followup times out before finishing any adjacent task, retrying that followup once with a reduced task_limit and a fresh followup budget should preserve at least one adjacent-success result instead of leaving the scope as completed_with_warnings.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T01:48:50Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so staged generated_success followups can retry once with a reduced task_limit when they time out before completing any adjacent task, and so timeout partial summaries can salvage completed generated metrics instead of dropping them.`
  - `Added protocol coverage in tests/test_protocol_comparison.py for both cases: a reduced-slice generated_success retry after timeout with zero completed adjacent tasks, and partial generated-success metric salvage when one adjacent task completed before timeout.`
  - `The first live scope cycle:tooling:20260402T014138715355Z:192ffdd8 showed why salvage matters: supervised_coordination_codex_apr02_toolbridge_followup_retry still timed out at the followup boundary, but its partial summary already contained one completed adjacent-success result that older wrapper logic would have discarded.`
  - `The second live scope cycle:tooling:20260402T014734499997Z:444971dd validated the repaired path end-to-end. supervised_coordination_codex_apr02_toolbridge_followup_salvage completed the primary tooling pass at 4/4 success, merged generated_success at 2/2 success under the same 20-second followup budget, and produced a fresh tooling candidate artifact.`
  - `After refreshing the frontier, the new tooling scope now records generated_success_total=2 and generated_success_passed=2. The remaining warning is no longer followup loss; it is the over-budget primary runtime signature (elapsed 114.7s against the nominal 60s primary budget).`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_followup_retry.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_followup_salvage.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_toolbridge_followup_salvage/tooling/cycle_tooling_20260402T014734499997Z_444971dd_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "retries_generated_success_followup_with_reduced_task_limit or salvages_partial_generated_success_metrics_on_timeout or retries_with_reduced_primary_task_limit_after_timeout"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_followup_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_followup_retry --notes "retry generated_success followup with reduced slice after partial timeout so tooling scopes can retain at least one adjacent-success result under bounded budget"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_followup_salvage --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_followup_salvage --notes "salvage generated_success partial metrics on timeout so tooling scopes retain completed adjacent wins under bounded followup budgets"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `The generated_success followup path is no longer the highest-ROI tooling blocker. The next ROI is either reducing the over-budget primary runtime signature on these repository/project tooling scopes or fixing promotion/supervisor ingestion, because the refreshed queue/status artifacts remained sparse after the dry-run refresh even though the frontier now carries strong tooling evidence.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:17:00Z`
- scope_id: `supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry`
- progress_label: `supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry --notes "runtime retry shrinks bounded low-cost primary task_limit after partial timeout so tooling scopes survive contention"
```

- hypothesis: `If bounded low-cost repository/project tooling runs time out after partial progress, a fresh-budget retry with a reduced primary task_limit should convert that partial progress into a completed primary pass and let tooling candidate generation land instead of losing the whole scope.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T01:25:03Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so a bounded low-cost priority primary observation can retry once with a reduced task_limit and fresh observation budget after a partial timeout. The retry recomputes the current-task decision budget from the smaller slice instead of reusing the original 5-task budget.`
  - `Added protocol coverage in tests/test_protocol_comparison.py proving a timed-out 5-task repository/project run retries at 2 tasks with a recomputed 12.0-second decision budget and then records the reduced-slice retry in the observe metrics.`
  - `Fresh live tooling evidence is now in cycle:tooling:20260402T012440463371Z:b2c77c61. The new scope timed out once, retried at 4/5, completed the primary tooling pass at 4/4 success, and generated a compatible 43-candidate tooling artifact instead of failing before generate.`
  - `The remaining warning is narrower than before: generated_success still timed out inside its separate 20-second followup budget, so the scoped frontier entry is now completed_with_warnings rather than a hard observation failure.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry/tooling/cycle_tooling_20260402T012440463371Z_b2c77c61_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "retries_with_reduced_primary_task_limit_after_timeout or retries_without_tolbert_context_after_startup_failure or applies_multi_task_default_decision_budget"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_runtime_slice_retry --notes "runtime retry shrinks bounded low-cost primary task_limit after partial timeout so tooling scopes survive contention"`
  - `python scripts/report_supervised_frontier.py`
  - `python - <<'PY' ... assess_artifact_compatibility(subsystem='tooling', payload=payload) ... # compatible=True, candidate_count=43`
- next_recommendation: `The next ROI is to either finish the tooling generated_success followup within its separate 20-second budget or adjust frontier/promotion weighting so completed_with_warnings tooling runs with successful primary generation are treated as usable bootstrap evidence instead of being grouped with harder observation failures.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:59:58Z`
- scope_id: `supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget`
- progress_label: `supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 70 --generate-only --scope-id supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --progress-label supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget --notes "auto expand generated-success budget from primary coding wins with multi-task slack and fresh-budget tolbert startup fallback"
```

- hypothesis: `Once bounded explicit repository/project runs are already hitting 5/5 primary success, the next throughput loss is the manual 20-second adjacent-success cap. If the supervised wrapper auto-expands generated_success budget from the current-cycle primary win count when no explicit followup budget is set, Qwen should be able to replay the full coding seed pool in one scoped run instead of truncating followups at two tasks.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T01:14:47Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so generated_success budgets now auto-expand from the current-cycle primary win count when a bounded explicit priority run requests only generated_success and does not supply a manual followup budget. That converts a 5-win repository/project primary pass into a 5-task adjacent-success followup window by default instead of truncating at the prior manual 20-second cap.`
  - `Tightened bounded multi-task primary budgeting so the default per-task decision budget reserves one task-slot of slack. A 5-task, 60-second primary run now defaults to 10.0s instead of 12.0s per task, which reduces whole-run timeout risk from one slow llm_request.`
  - `Added a runtime fallback for shared TolBERT startup contention. When the observation child fails specifically because the TolBERT service did not become ready, the supervised wrapper now retries the primary observation once with use_tolbert_context=False and a fresh observation budget instead of aborting the whole scope.`
  - `Intermediate scopes made the blocker visible before the final recovery landed. The first 60-second auto-budget scope timed out at the primary wall-clock limit, and two follow-up scopes failed directly on TolBERT startup readiness. A 70-second retry proved the new TolBERT fallback path, and the fresh-budget retry finally completed cleanly enough to run the full adjacent-success sweep.`
  - `Fresh live evidence is now in cycle:retrieval:20260402T011432936117Z:6c8d78b9. Under scope supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget, the run retried without TolBERT context after startup failure, held primary repository/project pass at 5/5 success, auto-expanded generated_success to 43.9328s with task_limit=5, and passed all 5/5 adjacent-success followups. The last generated task was deployment_manifest_task_project_adjacent, so the full repository/project coding seed pool replayed instead of stopping at two tasks.`
  - `Patched frontier normalization so observation_budget_exceeded is no longer treated as a hard timeout. scripts/report_supervised_frontier.py and scripts/run_parallel_supervised_cycles.py now keep real child timeouts in observation_timed_out, expose observation_budget_exceeded separately, and classify successful over-budget runs as completed_with_warnings instead of failed.`
  - `After refreshing the frontier and promotion plan, the previously misclassified scope supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget now shows status=completed_with_warnings, run_succeeded=true, observation_timed_out=false, and observation_budget_exceeded=true in trajectories/improvement/reports/supervised_parallel_frontier.json.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget_slack.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget_retry.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget_fallback.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget_fallback70.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget/retrieval/cycle_retrieval_20260402T011432936117Z_6c8d78b9_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "retries_without_tolbert_context_after_startup_failure or applies_multi_task_default_decision_budget or auto_expands_generated_success_budget_from_primary_wins or auto_allocates_staged_curriculum_budget_for_explicit_priority_families or allows_two_generated_success_followups_when_budget_supports_it or preserves_explicit_priority_families or caps_generated_failure_followup_to_fit_budget"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_autogenbudget --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --progress-label supervised_coordination_codex_apr02_autogenbudget --notes "auto expand generated-success budget from primary coding wins"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_autogenbudget_slack --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --progress-label supervised_coordination_codex_apr02_autogenbudget_slack --notes "auto expand generated-success budget from primary coding wins with multi-task slack"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_autogenbudget_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --progress-label supervised_coordination_codex_apr02_autogenbudget_retry --notes "auto expand generated-success budget from primary coding wins with multi-task slack retry"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 70 --generate-only --scope-id supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --progress-label supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget --notes "auto expand generated-success budget from primary coding wins with multi-task slack and fresh-budget tolbert startup fallback"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next ROI is frontier ranking quality, not classification correctness. The repaired frontier now preserves completed_with_warnings runs, so the next useful change is to teach promotion scoring how to value over-budget-but-successful coding scopes relative to fully healthy runs instead of flattening them into the same generic warning bucket.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:28:29Z`
- scope_id: `supervised_coordination_codex_apr02_priority_lowcost`
- progress_label: `supervised_coordination_codex_apr02_priority_lowcost`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_priority_lowcost --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_priority_lowcost --notes "bounded explicit priority low-cost selection for repository/project coding tasks"
```

- hypothesis: `Repository/project bounded runs already focus the right coding families, but they still select tasks in default order, which front-loads retrieval-heavy tasks and wastes the fixed 60-second primary window. If explicit bounded priorities inherit low-cost selection by default, Qwen should spend the same budget on command-backed repository/project tasks and produce more coding successes plus richer adjacent-success seeds.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T00:32:18Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so explicit bounded priority families also default to prefer_low_cost_tasks=True unless a caller already set that flag. The supervised wrapper now keeps explicit repository/project focus while selecting the cheaper command-backed tasks first.`
  - `Validated the selection change against the real task bank before rerunning. Under default ordering, the 5-task repository/project slice began with retrieval-heavy tasks like repo_sync_matrix_retrieval_task and deployment_manifest_retrieval_task; under low_cost it switches to service_release_task, schema_alignment_task, report_rollup_task, repo_sync_matrix_task, and deployment_manifest_task.`
  - `Fresh scoped evidence materially improved Qwen's coding lane. The latest cycle record cycle:tooling:20260402T003148709863Z:66475c3a moved the primary repository/project pass from 2/5 to 5/5 success (pass_rate=1.0), with repository=3/3 and project=2/2. The staged generated_success followup still completed within the same 20-second window and passed 2/2 adjacent repository tasks, finishing on schema_alignment_task_repository_adjacent.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_priority_lowcost.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_priority_lowcost/tooling/cycle_tooling_20260402T003148709863Z_66475c3a_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "applies_bounded_default_priority_families or preserves_explicit_priority_families or generated_success_followups_when_budget_supports_it or generated_failure_followup_to_fit_budget"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_priority_lowcost --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_priority_lowcost --notes "bounded explicit priority low-cost selection for repository/project coding tasks"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `The next ROI is to keep compounding from this stronger seed pool. Either widen the staged generated_success budget so all five primary wins can spawn adjacent coding followups, or keep the 20-second window and add family-aware seed ranking so the highest-value repository/project wins are scheduled first when only two adjacent tasks can run.`


### `supervised_runner__open_3`

- runner_slot: `supervised_runner__open_3`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__observation_latency`
- claimed_at: `2026-04-02T00:22:41Z`
- scope_id: `supervised_latency_codex_apr02_successpair`
- progress_label: `supervised_latency_codex_apr02_successpair`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_latency_codex_apr02_successpair --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_latency_codex_apr02_successpair --notes "observation latency generated-success pair budget repair"
```

- hypothesis: `Generated-success is no longer empty, but the staged followup wrapper still inherits a per-task budget that only allows one adjacent-success task under a 20-second window. Reducing the generated-success decision budget to a bounded replay-specific cap should allow two current-cycle repository/project followups to execute in the same scoped run.`
- owned_paths:
  - `scripts/run_supervised_improvement_cycle.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `scripts/run_eval.py`
  - `tests/test_protocol_comparison.py`
- released_at: `2026-04-02T00:25:05Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so generated_success followups no longer inherit the full primary 12.0-second per-task budget. Staged adjacent-success now uses a bounded replay-specific decision cap, and the wrapper sets max_generated_success_schedule_tasks from the resolved followup task_limit instead of a hardcoded 1.`
  - `Added protocol coverage in tests/test_protocol_comparison.py proving a 20.0-second generated_success budget now resolves to task_limit=2, max_generated_success_schedule_tasks=2, and current_task_decision_budget_seconds=10.0 while the existing generated_failure budget-capping behavior still holds.`
  - `Fresh scoped evidence validated the live Qwen path. The latest cycle record cycle:tooling:20260402T002457194886Z:f895f9ec kept the primary repository/project pass at 2/5 success (pass_rate=0.4), but generated_success now executed both current-cycle adjacent tasks and both passed cleanly (generated_total=2, generated_passed=2, merged_generated_metrics=true). The followup finished on deployment_manifest_task_project_adjacent after also running the repository adjacent seed.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_latency_codex_apr02_successpair.jsonl`
  - `trajectories/improvement/candidates/supervised_latency_codex_apr02_successpair/tooling/cycle_tooling_20260402T002457194886Z_f895f9ec_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "generated_success_followups_when_budget_supports_it or generated_failure_followup_to_fit_budget"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_latency_codex_apr02_successpair --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_latency_codex_apr02_successpair --notes "observation latency generated-success pair budget repair"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `The bounded generated-success path is now materially better for coding: the same primary 2/5 repository/project window can produce two passing adjacent followups instead of one. The next ROI is upstream again, not wrapper mechanics: raise primary repository/project success yield above 2/5 so the expanded followup window compounds from a larger pool of coding wins, or shift coordinator attention to frontier-backed candidate review if this level of followup throughput is sufficient for the current wave.`



### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-02T00:08:51Z`
- scope_id: `supervised_ecology_codex_apr02_seedbundle`
- progress_label: `supervised_ecology_codex_apr02_seedbundle`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr02_seedbundle --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr02_seedbundle --notes "task ecology generated-success seed bundle loader repair"
```

- hypothesis: `The staged generated_success followup already emits a current-cycle seed bundle, but evals/harness.py only supports directory-style episode loading. Teaching the loader to consume the bundle file should let repository/project successes from the current scoped run produce at least one adjacent-success followup task for Qwen to execute.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-02T00:16:23Z`
- result: `completed`
- findings:
  - `Patched evals/harness.py so the generated_success fallback loader applies the scoped workspace filter to current-cycle seed bundle files, not just directory-style episode roots. The staged followup can now consume the explicit primary-run success bundle instead of silently treating it as empty.`
  - `Added regression coverage in tests/test_eval.py for bundle-file scoped filtering. That test proves a generated_success seed bundle containing mixed-scope successes yields only the current scope's repository seed.`
  - `Fresh scoped evidence moved the ecology bottleneck forward. The latest cycle record cycle:tooling:20260402T001556506989Z:58f71f8e kept the primary repository/project pass at 2/5 success (pass_rate=0.4), but generated_success advanced from zero scheduled work to one scheduled repository adjacent task and it passed cleanly (generated_total=1, generated_passed=1, merged_generated_metrics=true).`
- artifacts:
  - `evals/harness.py`
  - `tests/test_eval.py`
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr02_seedbundle.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr02_seedbundle/tooling/cycle_tooling_20260402T001556506989Z_58f71f8e_script_hardening/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `python -m py_compile evals/harness.py tests/test_eval.py`
  - `pytest -q tests/test_eval.py -k "generated_success_seed"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr02_seedbundle --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr02_seedbundle --notes "task ecology generated-success seed bundle loader repair"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `Generated-success seeding is now functional again, so the next task-ecology ROI is no longer loader plumbing. Push on seed ranking and budget use: decide whether the second current-cycle success seed should also run under the 20-second window, or whether repository/project weighting should explicitly prefer the higher-value family when only one adjacent-success followup is allowed.`




### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T23:49:00Z`
- scope_id: `supervised_ecology_codex_apr01_successseed_fix`
- progress_label: `supervised_ecology_codex_apr01_successseed_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successseed_fix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successseed_fix --notes "task ecology generated_success seed fallback repair"
```

- hypothesis: `After repairing generated_success bootstrap, the remaining zero-task result may be caused by the staged followup explicitly disabling fallback success-seed loading from trajectories/episodes.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T23:50:49Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so staged generated_success followups keep allow_generated_success_seed_fallback enabled when they run with include_primary_tasks=False.`
  - `That change reintroduced real seed-loading work, but it did not fully clear the ecology bottleneck. The repaired success-seed cycle still timed out at phase=generated_success_schedule under the same 20.0-second budget, so fallback loading or downstream adjacent-success seed construction is still too expensive for the staged window.`
  - `Combined with the prior harness repair, the current picture is tighter: empty-task bootstrap waste is gone, but historical success-seed fallback remains too slow. The next ROI is to bound or cache success-seed document loading rather than increasing the staged budget again.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `evals/harness.py`
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01_successseed_fix.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr01_successseed_fix/tooling/cycle_tooling_20260401T235023857847Z_053a4273_procedure_promotion/tool_candidates.json`
- verification:
  - `python -m py_compile evals/harness.py scripts/run_human_guided_improvement_cycle.py`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successseed_fix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successseed_fix --notes "task ecology generated_success seed fallback repair"`
- next_recommendation: `Next best ROI in task_ecology is to add a bounded or cached generated-success seed loader in evals/harness.py so staged followups do not scan or decode the full historical success corpus before scheduling.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T23:41:30Z`
- scope_id: `supervised_ecology_codex_apr01_successfocus_fix`
- progress_label: `supervised_ecology_codex_apr01_successfocus_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successfocus_fix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successfocus_fix --notes "task ecology generated_success scheduling repair"
```

- hypothesis: `The remaining generated_success bottleneck is still pre-execution bootstrap inside the followup schedule path, so deferring unused generated-kernel initialization should let the staged followup complete and emit its true task count.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T23:48:23Z`
- result: `completed`
- findings:
  - `Patched evals/harness.py so generated_success and generated_failure defer followup AgentKernel construction until after the followup task list is known. Empty followup lists now skip that bootstrap entirely and still emit the total/count progress line.`
  - `A fresh success-focused ecology cycle confirmed the repair: generated_success no longer times out inside generated_success_schedule. It now completes cleanly with phase=generated_success total=0, timed_out=false, and no supplemental warning.`
  - `The bottleneck is now semantic, not runtime. This ecology slice still produced zero adjacent-success followup tasks, so the next ROI is in success-seed eligibility/selection rather than more schedule budget or bootstrap shaving.`
- artifacts:
  - `evals/harness.py`
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01_successfocus_fix.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr01_successfocus_fix/tooling/cycle_tooling_20260401T234801328752Z_f70efc15_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile evals/harness.py`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successfocus_fix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successfocus_fix --notes "task ecology generated_success scheduling repair"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Next best ROI in task_ecology is to change adjacent-success seed selection so generated_success gets nonzero eligible seeds under repository/project-heavy scoped runs. Coordinator-side, the shared shortlist is still tooling-first bootstrap review.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T23:36:20Z`
- scope_id: `supervised_ecology_codex_apr01_successfocus`
- progress_label: `supervised_ecology_codex_apr01_successfocus`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successfocus --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successfocus --notes "task ecology generated_success scheduling probe"
```

- hypothesis: `With generated_failure startup waste already reduced elsewhere, the highest-ROI ecology probe is to give generated_success a dedicated staged budget and see whether it can get past generated_success_schedule and actually emit scheduled followup work.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T23:41:04Z`
- result: `completed`
- findings:
  - `The success-focused ecology cycle kept the exact intended shape: explicit repository/project priorities, no failure followup, and a dedicated 20.0-second generated_success budget with applied_task_limit=1.`
  - `That extra budget did not move the bottleneck. The generated_success followup still timed out at phase=generated_success_schedule with generated_tasks_scheduled=0 and merged_generated_metrics=False, so the failure is still in success-seed loading/scheduling before any generated task is emitted.`
  - `Primary observation remained stable: 5 repository/project tasks completed within the bounded observe pass, all failed with inference_failure after one step, and the scoped run still generated another tooling/procedure_promotion candidate. No source-code files changed in this cycle.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01_successfocus.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr01_successfocus/tooling/cycle_tooling_20260401T234044382786Z_7d120627_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_successfocus --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_ecology_codex_apr01_successfocus --notes "task ecology generated_success scheduling probe"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Do not spend the next cycle on more generated_success budget alone. The next best ROI inside task_ecology is to reduce or defer success-seed loading/scheduling work in evals/harness.py so generated_success can emit a nonzero task list before the staged budget expires. Coordinator-side, the promotion shortlist is still tooling-first bootstrap review rather than compare retry.`


### `supervised_runner__open_3`

- runner_slot: `supervised_runner__open_3`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__observation_latency`
- claimed_at: `2026-04-01T23:31:45Z`
- scope_id: `supervised_latency_codex_apr01_followup`
- progress_label: `supervised_latency_codex_apr01_followup`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_latency_codex_apr01_followup --include-failure-curriculum --failure-curriculum-budget-seconds 18.75 --progress-label supervised_latency_codex_apr01_followup --notes "failure followup budget repair"
```

- hypothesis: `The generated_failure staged followup still inherits an unrealistic task volume and per-task decision budget from primary observation, so a budget-aware cap should let recovery diagnostics finish and merge instead of timing out in the supplemental window.`
- owned_paths:
  - `scripts/run_supervised_improvement_cycle.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `scripts/run_eval.py`
  - `tests/test_protocol_comparison.py`
- released_at: `2026-04-01T23:33:46Z`
- result: `completed`
- findings:
  - `Staged curriculum followups are now budget-shaped before launch. For generated_failure, the supervised runner derives a smaller task_limit and per-task decision budget from the supplemental budget instead of reusing the primary 5-task / 12.0-second shape.`
  - `Targeted regressions passed, including a new case that forces a 5-task / 18.75-second generated_failure followup and verifies it is reduced to task_limit=1 with current_task_decision_budget_seconds=9.375.`
  - `A fresh scoped supervised run confirmed the runtime-cost repair live: generated_failure started with task_limit=1 and current_task_decision_budget_seconds=9.4, then completed without timeout. The remaining gap moved one layer deeper: the followup reported phase=generated_failure total=0 and merged_generated_metrics=False, so the next ROI is failure-seed/task synthesis or merge eligibility, not supplemental runtime cost.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_latency_codex_apr01_followup.jsonl`
  - `trajectories/improvement/candidates/supervised_latency_codex_apr01_followup/tooling/cycle_tooling_20260401T233254065158Z_1d2bdac9_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k 'stages_generated_curriculum_into_followup_observations or auto_allocates_staged_curriculum_budget_for_explicit_priority_families or caps_generated_failure_followup_to_fit_budget or preserves_partial_context_for_timed_out_curriculum_followup or does_not_retry_when_no_budget_remains'`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_latency_codex_apr01_followup --include-failure-curriculum --failure-curriculum-budget-seconds 18.75 --progress-label supervised_latency_codex_apr01_followup --notes "failure followup budget repair"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Do not spend another cycle on failure-followup runtime cost first. The timeout is gone; the next bottleneck is that generated_failure is scheduling zero followup tasks, so the best ROI is in failure curriculum seed/task generation or generated-metrics merge conditions. Coordinator-side, tooling is still a bootstrap-review path rather than a compare retry.`


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

### `supervised_runner__open_4`

- runner_slot: `supervised_runner__open_4`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__planner_selection`
- claimed_at: `2026-04-01T00:00:00Z`
- scope_id: `supervised_planner_codex_apr01`
- progress_label: `supervised_planner_codex_apr01`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_planner_codex_apr01 --progress-label supervised_planner_codex_apr01 --notes "planner selection probe"
```

- hypothesis: `The planner is still over-crediting benchmark and narrow retained wins relative to trust coverage gaps and learning-derived transition evidence, so stronger selector guardrails should shift scoped supervised runs toward higher-yield subsystems.`
- owned_paths:
  - `agent_kernel/improvement.py`
  - `scripts/select_improvement_target.py`
  - `tests/test_improvement.py`
- released_at: `2026-04-01T20:54:37Z`
- result: `completed`
- findings:
  - `Strengthened planner scoring so learning-artifact pressure can directly lift subsystems like transition_model/recovery/tooling, while trust coverage gaps, false-pass risk, and benchmark-shape weakness now actively penalize low-signal benchmark retries.`
  - `Retained universe feedback is now downweighted when support is narrow or phase-gate warnings/regressed families are present, which reduces over-credit for flashy but weak historical wins.`
  - `A fresh scoped supervised planner run completed within the observation budget and generated a new tooling candidate (procedure_promotion) instead of adding another transition_model retry, which confirms the selector moved the frontier.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_planner_codex_apr01.jsonl`
  - `trajectories/improvement/candidates/supervised_planner_codex_apr01/tooling/cycle_tooling_20260401T205350625447Z_1aa27836_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `agent_kernel/improvement.py`
  - `tests/test_improvement.py`
- verification:
  - `pytest -q tests/test_improvement.py`
  - `python -m py_compile agent_kernel/improvement.py tests/test_improvement.py scripts/select_improvement_target.py`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_planner_codex_apr01 --progress-label supervised_planner_codex_apr01 --notes "planner selection probe"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `Use the refreshed frontier/promotion path to compare the new tooling candidate against the existing transition_model-heavy set, and keep steering future scoped runs toward underrepresented high-yield subsystems instead of reopening benchmark-only retries.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T20:55:00Z`
- scope_id: `supervised_ecology_codex_apr01`
- progress_label: `supervised_ecology_codex_apr01`
- status: `handed_off`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex_apr01 --notes "task ecology retry after observation repair"
```

- hypothesis: `With observation stalls repaired and planner selection broadened, a fresh repository/project-weighted ecology run should reveal whether task-family mix can now generate a non-transition-model candidate instead of timing out before the useful part of the loop.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T21:15:00Z`
- result: `handed_off`
- findings:
  - `This first ecology retry scope was superseded by the follow-on retry scope supervised_ecology_codex_apr01_retry once the observation repair landed and the lane moved to a cleaner bounded rerun.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01.jsonl`
- verification:
  - `none; superseded by supervised_ecology_codex_apr01_retry before closeout evidence was written here`
- next_recommendation: `Use the later ecology retry and budget-fix closeouts as the authoritative task-ecology outcomes for this wave.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T21:15:00Z`
- scope_id: `supervised_ecology_codex_apr01_retry`
- progress_label: `supervised_ecology_codex_apr01_retry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex_apr01_retry --notes "task ecology retry after observation-latency repair"
```

- hypothesis: `Now that stale prestep observation aborts are fixed, the next high-ROI move is to rerun the previously blocked repository/project-scoped ecology lane and see whether the frontier can diversify beyond transition_model-heavy candidates.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/benchmark_synthesis.py`
  - `agent_kernel/curriculum.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T22:28:30Z`
- result: `completed`
- findings:
  - `The scoped ecology retry no longer timed out in observation; it completed all five primary tasks within the 60-second observation budget and reached generation cleanly.`
  - `The run generated a second tooling/procedure_promotion candidate instead of another transition_model retry, and the refreshed frontier/promotion plan now rank tooling above transition_model on the current shared shortlist.`
  - `The ecology probe still did not execute generated-success or generated-failure followups because both curriculum lanes were requested but skipped with no separate curriculum budget configured, so true curriculum-pressure remains a follow-on coordination gap rather than an observation blocker.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01_retry.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr01_retry/tooling/cycle_tooling_20260401T222630065813Z_2b49e425_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex_apr01_retry --notes "task ecology retry after observation-latency repair"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Take the refreshed tooling-first promotion shortlist through compare/dry-run finalize next, and separately decide whether supervised ecology runs should auto-allocate nonzero curriculum budgets when --include-curriculum flags are present.`

### `supervised_runner__open_2`

- runner_slot: `supervised_runner__open_2`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__task_ecology`
- claimed_at: `2026-04-01T22:42:00Z`
- scope_id: `supervised_ecology_codex_apr01_budgetfix`
- progress_label: `supervised_ecology_codex_apr01_budgetfix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_budgetfix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex_apr01_budgetfix --notes "task ecology retry after scheduling and curriculum budget fix"
```

- hypothesis: `If explicit family priorities survive scheduling and staged curriculum gets a real fallback budget, ecology-weighted observe passes should surface repository/project families instead of silently dropping the curriculum path.`
- owned_paths:
  - `evals/harness.py`
  - `agent_kernel/task_bank.py`
  - `agent_kernel/curriculum.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_eval.py`
  - `tests/test_protocol_comparison.py`
- released_at: `2026-04-01T23:01:49Z`
- result: `completed`
- findings:
  - `Primary task scheduling now preserved the explicit ecology priorities: the scoped run scheduled only repository/project tasks, and the observation partial summary recorded observed benchmark families ['project', 'repository'] instead of default-family drift.`
  - `Staged curriculum no longer disappeared behind zero configured budget. The supervised observe pass auto-allocated 10.5328s to both generated_success and generated_failure followups because explicit family priorities were present and observation time remained.`
  - `Both curriculum followups timed out inside their new budgets, so the bottleneck moved from missing scheduling/budget pressure to runtime cost inside the generated followup passes. The final selector still chose tooling/procedure_promotion, but that now happens after an ecology-shaped observation pass rather than before one.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_ecology_codex_apr01_budgetfix.jsonl`
  - `trajectories/improvement/candidates/supervised_ecology_codex_apr01_budgetfix/tooling/cycle_tooling_20260401T230130083638Z_ea7c7e9d_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `evals/harness.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_eval.py`
  - `tests/test_protocol_comparison.py`
- verification:
  - `pytest -q tests/test_eval.py -k 'keeps_prioritized_families_at_front_of_round_robin or weighted_benchmark_families or low_cost'`
  - `pytest -q tests/test_protocol_comparison.py -k 'curriculum_budget or staged_curriculum or priority_families'`
  - `python -m py_compile evals/harness.py scripts/run_human_guided_improvement_cycle.py tests/test_eval.py tests/test_protocol_comparison.py`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_ecology_codex_apr01_budgetfix --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --include-failure-curriculum --progress-label supervised_ecology_codex_apr01_budgetfix --notes "task ecology retry after scheduling and curriculum budget fix"`
  - `python scripts/report_supervised_frontier.py`
- next_recommendation: `Target the next repair at generated curriculum runtime cost or child timeout sizing, because ecology-weighted runs now reach the right repository/project families and launch both staged followups, but those followups still exhaust their supplemental budgets before contributing generated metrics.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T22:31:30Z`
- scope_id: `supervised_coordination_codex_apr01`
- progress_label: `supervised_coordination_codex_apr01`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_frontier_promotion_pass.py --promotion-plan trajectories/improvement/reports/supervised_frontier_promotion_plan.json --limit 2 --allow-subsystem tooling
```

- hypothesis: `With tooling now at the top of the refreshed promotion plan, the next best ROI move is to run a coordinator dry promotion pass on the tooling shortlist and classify whether the lane is compare-ready or still bootstrap-blocked.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/improvement.md`
  - `config/supervised_parallel_work_manifest.json`
- released_at: `2026-04-01T22:45:00Z`
- result: `completed`
- findings:
  - `The tooling-only dry promotion pass classified both top tooling candidates as bootstrap-first-retain and emitted ready dry-run finalize commands, confirming the next gating problem was baseline/bootstrap readiness rather than compare failures.`
  - `A supervisor bug was then fixed in scripts/run_supervisor_loop.py: dry-run bootstrap-first-retain results now count as bootstrap review pending instead of being ignored unless finalize_skip_reason was already populated, which restores first-round remediation queue emission.`
  - `After the fix, a one-round dry supervisor refresh emitted the trust recovery remediation artifact at trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json, so bootstrap candidates now re-enter machine-owned coordination instead of disappearing from the supervisor loop.`
- artifacts:
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
- verification:
  - `python scripts/run_frontier_promotion_pass.py --promotion-plan trajectories/improvement/reports/supervised_frontier_promotion_plan.json --limit 2 --allow-subsystem tooling`
  - `python -m py_compile scripts/run_supervisor_loop.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k 'bootstrap_review_pending_subsystem or trust_streak_accumulation_queue or dry_run_bootstrap_first_retain_into_trust_queue or protected_review_only_queue or emits_baseline_bootstrap_review_action or writes_status_history_and_report'`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Use the emitted trust recovery queue as the next machine-readable lane signal, and separately decide whether tooling bootstrap candidates should remain trust-gated or get a baseline_bootstrap review package even during compare-only dry runs.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T22:59:40Z`
- scope_id: `supervised_coordination_codex_apr01_followup`
- progress_label: `supervised_coordination_codex_apr01_followup`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `The supervisor should refresh remediation queues from the promotion pass it just ran, and compare-only bootstrap tooling candidates should still emit baseline review packages even when trust recovery remains the lane gate.`
- owned_paths:
  - `docs/supervised_work_queue.md`
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
- released_at: `2026-04-01T23:06:55Z`
- result: `completed`
- findings:
  - `Patched the supervisor loop so refresh and promotion actions run first, then round decisions are rebuilt from the fresh promotion-pass artifact before remediation queues, lane allocation, rollback planning, and discovery launch are computed. The queue is now same-round state instead of a one-round lagging reflection of the previous pass file.`
  - `Compare-only first-retain lanes that are still trust-gated now emit two machine-readable outputs: trust recovery remains the lane signal via remediation_queue=trust_streak_accumulation, and a separate baseline_bootstrap review package is emitted in the same round with finalize_gate_reason explaining that finalize still waits on trusted status.`
  - `Live dry-run verification succeeded, but the refreshed promotion plan no longer has tooling on top. The current plan ranks transition_model first at 6.5885 and 6.337, while tooling/procedure_promotion is history-penalized to 4.4668 and 4.4603, so the emitted live queues now point at transition_model rather than tooling.`
- artifacts:
  - `scripts/run_supervisor_loop.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `python -m py_compile scripts/run_supervisor_loop.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k 'bootstrap_review_pending_subsystem or trust_streak_accumulation_queue or dry_run_bootstrap_first_retain_into_trust_queue or protected_review_only_queue or emits_baseline_bootstrap_review_action or writes_status_history_and_report or rebuilds_lane_signal_from_fresh_promotion_pass'`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Use supervisor_trust_streak_recovery_queue.json as the live lane gate and supervisor_baseline_bootstrap_queue.json as the simultaneous coordinator review package. If you still want tooling to be the next operated lane, inspect whether its current history penalty is too aggressive or run a tooling-only compare pass before the next supervisor round.`


### `supervised_runner__open_4`

- runner_slot: `supervised_runner__open_4`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__world_state`
- claimed_at: `2026-04-01T23:12:58Z`
- scope_id: `supervised_world_codex_apr01_preview`
- progress_label: `supervised_world_codex_apr01_preview`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_world_codex_apr01_preview --progress-label supervised_world_codex_apr01_preview --notes "world-state probe after workspace-preview structured-edit path"
```

- hypothesis: `The new workspace-preview and structured-edit path should improve state-guided coding behavior enough for a fresh world-state scoped loop to generate a frontier candidate with better ROI than another tooling/procedure_promotion repeat.`
- owned_paths:
  - `agent_kernel/loop.py`
  - `agent_kernel/world_model.py`
  - `agent_kernel/universe_model.py`
  - `tests/test_loop.py`
- released_at: `2026-04-01T23:15:11Z`
- result: `completed`
- findings:
  - `The scoped world-state probe completed its 60-second bounded observation cleanly in 47.509s and generated a new candidate artifact, but subsystem selection still routed to tooling/procedure_promotion instead of surfacing a world-state-specific candidate.`
  - `The refreshed frontier absorbed the run as an eighth scoped candidate and a fifth tooling entry, which means the new workspace-preview path is now represented in machine-readable evidence, but it did not improve subsystem diversity.`
  - `ROI was weaker than the existing ecology/tooling leaders: the refreshed promotion plan still selects supervised_ecology_codex_apr01_retry and supervised_ecology_codex_apr01 ahead of this scope, so this run added evidence but not the top next action.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_world_codex_apr01_preview.jsonl`
  - `trajectories/improvement/candidates/supervised_world_codex_apr01_preview/tooling/cycle_tooling_20260401T231423997300Z_453911a9_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_world_codex_apr01_preview --progress-label supervised_world_codex_apr01_preview --notes "world-state probe after workspace-preview structured-edit path"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `If the goal is pure ROI rather than more evidence, stop spending near-term cycles on world_state selection pressure and either run the top-ranked tooling compare path or adjust selector/task-family pressure so world-state-oriented runs can escape the tooling/procedure_promotion attractor.`


### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__runtime_evidence`
- claimed_at: `2026-04-01T23:18:11Z`
- scope_id: `supervised_evidence_codex_apr01`
- progress_label: `supervised_evidence_codex_apr01`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_evidence_codex_apr01 --include-failure-curriculum --progress-label supervised_evidence_codex_apr01 --notes "runtime evidence ROI probe while parallel lanes are active"
```

- hypothesis: `With tooling and transition_model already dominating the frontier, the highest-ROI free lane is to improve structured runtime failure evidence so the next round can learn from richer diagnostics instead of replaying the same narrow candidate families.`
- owned_paths:
  - `agent_kernel/job_queue.py`
  - `scripts/run_selfplay.py`
  - `scripts/report_failure_recovery.py`
  - `tests/test_failure_recovery_report.py`
- released_at: `2026-04-01T23:20:05Z`
- result: `completed`
- findings:
  - `The scoped runtime-evidence cycle completed within the 60-second primary observation budget and generated a new tooling/procedure_promotion candidate, so it did not open a new subsystem family on the frontier even though it exercised a different lane.`
  - `The primary-task trace concentrated on bounded plus discovered-task coverage, and all five primary tasks terminated with inference_failure after a single step. That makes the run more useful as runtime-diagnostic evidence than as proof of a better candidate family.`
  - `The generated_failure curriculum follow-up was auto-budgeted at 18.7539 seconds and timed out before any generated metrics merged, so the next runtime-evidence ROI repair is still the same bottleneck: richer failure followups are being requested but not finishing inside the supplemental budget.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_evidence_codex_apr01.jsonl`
  - `trajectories/improvement/candidates/supervised_evidence_codex_apr01/tooling/cycle_tooling_20260401T231951684064Z_8943f559_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_evidence_codex_apr01 --include-failure-curriculum --progress-label supervised_evidence_codex_apr01 --notes "runtime evidence ROI probe while parallel lanes are active"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Keep the coordinator on the existing tooling-first promotion shortlist, and if you want another discovery cycle rather than a promotion step, target runtime-evidence followup cost next so generated_failure diagnostics can actually complete and change the planner signal instead of timing out.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T23:11:01Z`
- scope_id: `supervised_coordination_codex_apr01_queue_sync`
- progress_label: `supervised_coordination_codex_apr01_queue_sync`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `With two other Codex workers active, the highest-ROI coordinator move is to make the orchestration docs truthful again so lane ownership, stale claims, and current machine-owned next steps are visible without relying on chat context.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-01T23:11:01Z`
- result: `completed`
- findings:
  - `Cleaned the work-queue claim template after a completed ecology closeout was accidentally inserted inside it, and converted the stale supervised_ecology_codex_apr01 claimed block into an explicit handoff so the ledger no longer shows an open ghost claim.`
  - `Recorded the ecology budget-fix closeout in the normal ledger section and added concurrency notes near the slot and lane tables so readers do not assume unclaimed table rows mean the workspace is uncontended while two other Codex workers are active.`
  - `Updated the coordinator status index to reflect the current machine-owned next step: the latest supervisor dry run now emits both trust recovery and baseline bootstrap packages in the same compare-only round, and the current live lane signal has shifted back to transition_model rather than tooling.`
- artifacts:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `manual doc reconciliation against trajectories/improvement/reports/supervisor_loop_status.json, trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json, trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json, and trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- next_recommendation: `Keep the orchestration surface read-mostly while the parallel workers finish, and use the transition_model trust-recovery queue plus baseline review package as the live coordinator inputs unless a fresh frontier refresh changes the ranking again.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T23:18:42Z`
- scope_id: `supervised_coordination_codex_apr01_tooling_compare`
- progress_label: `supervised_coordination_codex_apr01_tooling_compare`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path trajectories/improvement/candidates/supervised_ecology_codex_apr01_retry/tooling/cycle_tooling_20260401T222630065813Z_2b49e425_procedure_promotion/tool_candidates.json --before-cycle-id cycle:tooling:20260401T222630065813Z:2b49e425
```

- hypothesis: `The highest-ROI coordinator action is to resolve whether the top-ranked tooling frontier candidate is truly compare-ready or still a bootstrap-first-retain lane, so the next step becomes concrete instead of speculative.`
- owned_paths:
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- released_at: `2026-04-01T23:19:14Z`
- result: `completed`
- findings:
  - `The top-ranked tooling compare path is not compare-ready yet. The retained-baseline comparison exited immediately with "no prior retained baseline exists for subsystem=tooling", so the lane remains a bootstrap-first-retain case rather than a true compare decision.`
  - `This confirms the promotion plan rank is useful for prioritization, but not sufficient as a readiness signal by itself. For tooling, the next ROI step is baseline/bootstrap review or dry-run finalize review, not another retained-baseline compare retry.`
- artifacts:
  - `none; compare command reported missing retained tooling baseline and emitted no new artifact`
- verification:
  - `python scripts/compare_retained_baseline.py --subsystem tooling --artifact-path trajectories/improvement/candidates/supervised_ecology_codex_apr01_retry/tooling/cycle_tooling_20260401T222630065813Z_2b49e425_procedure_promotion/tool_candidates.json --before-cycle-id cycle:tooling:20260401T222630065813Z:2b49e425`
- next_recommendation: `Use the existing dry-run finalize path or the supervisor baseline bootstrap package for tooling instead of repeating compare. If you want the highest-leverage compare-ready path right now, shift attention back to the transition_model entries that already dominate the live machine-owned lane signal.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T23:24:54Z`
- scope_id: `supervised_coordination_codex_apr01_tooling_finalize_review`
- progress_label: `supervised_coordination_codex_apr01_tooling_finalize_review`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/finalize_latest_candidate_from_cycles.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json --subsystem tooling --variant-id procedure_promotion --scope-id supervised_ecology_codex_apr01_retry --candidate-index 0 --dry-run
```

- hypothesis: `The highest-ROI coordinator follow-up is to turn the top tooling bootstrap candidate into an explicit dry-run finalize review package so the missing retained baseline becomes an operator-review decision instead of a repeated compare failure.`
- owned_paths:
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
- released_at: `2026-04-01T23:25:19Z`
- result: `completed`
- findings:
  - `The tooling dry-run finalize path resolved cleanly and emitted an explicit review command instead of failing on baseline readiness. That turns the bootstrap-first-retain case into a concrete operator-review action.`
  - `The emitted review command targets the top-ranked tooling scope supervised_ecology_codex_apr01_retry and its procedure_promotion candidate, which confirms the promotion plan and finalize bridge are aligned on the same candidate.`
- artifacts:
  - `dry-run finalize output: python scripts/finalize_improvement_cycle.py --subsystem tooling --cycle-id cycle:tooling:20260401T222630065813Z:2b49e425 --artifact-path trajectories/improvement/candidates/supervised_ecology_codex_apr01_retry/tooling/cycle_tooling_20260401T222630065813Z_2b49e425_procedure_promotion/tool_candidates.json`
- verification:
  - `python scripts/finalize_latest_candidate_from_cycles.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json --subsystem tooling --variant-id procedure_promotion --scope-id supervised_ecology_codex_apr01_retry --candidate-index 0 --dry-run`
- next_recommendation: `If you want to continue the tooling bootstrap path, review the emitted finalize_improvement_cycle command next. If you want a lane that is closer to compare-ready automation, switch coordinator attention back to the transition_model candidates instead of running more tooling compare passes.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T23:41:35Z`
- scope_id: `supervised_coordination_codex_apr01_supervisor_refresh`
- progress_label: `supervised_coordination_codex_apr01_supervisor_refresh`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `After the tooling compare and dry-run finalize review, the next coordinator move is to refresh machine-owned reports so the shared bootstrap and trust-recovery queues reflect the current tooling-first frontier instead of stale transition_model guidance.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
- released_at: `2026-04-01T23:41:43Z`
- result: `completed`
- findings:
  - `The refreshed supervisor round now points both machine-owned remediation queues at tooling rather than transition_model. baseline_bootstrap and trust_streak_accumulation both target supervised_ecology_codex_apr01_retry with compare_status=bootstrap_first_retain.`
  - `The promotion pass now records the exact evidence chain for the top two tooling candidates: retained compare fails with no prior tooling baseline, while dry-run finalize succeeds and emits the review command for each candidate.`
  - `Tooling is now the truthful live coordinator signal again, but bootstrap finalize remains gated by trust_status=bootstrap and bootstrap_finalize_policy=operator_review, so the next step is still review-oriented rather than unattended apply.`
- artifacts:
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
- verification:
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Treat tooling as the live machine-owned bootstrap lane again. If you want to keep moving without violating the trust gate, review the emitted finalize_improvement_cycle command for supervised_ecology_codex_apr01_retry; do not apply it unattended while trust_status remains bootstrap.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-01T23:50:24Z`
- scope_id: `supervised_coordination_codex_apr01_incompatible_filter`
- progress_label: `supervised_coordination_codex_apr01_incompatible_filter`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `If promotion planning filters incompatible candidate artifacts before ranking them, the machine-owned lane signal should stop treating empty tooling candidate sets as bootstrap-review-ready and revert to the highest-compatible frontier candidate instead.`
- owned_paths:
  - `scripts/report_frontier_promotion_plan.py`
  - `scripts/run_frontier_promotion_pass.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- released_at: `2026-04-01T23:50:35Z`
- result: `completed`
- findings:
  - `Found a concrete promotion bug: the top tooling bootstrap candidate had artifact_kind=tool_candidate_set with candidates=[], and assess_artifact_compatibility reported it incompatible with violation "artifact must contain a non-empty candidates list", yet it was still being ranked into the promotion plan and supervisor queues.`
  - `Patched scripts/report_frontier_promotion_plan.py to validate on-disk candidate artifacts and exclude incompatible ones from eligibility, and patched scripts/run_frontier_promotion_pass.py to skip incompatible artifacts defensively if one is injected into a plan.`
  - `After refreshing the live reports on the patched code, the machine-owned signal reverted from tooling back to transition_model. The new promotion plan has eligible_candidates=3, incompatible_candidates=13, and the supervisor baseline_bootstrap plus trust_streak_accumulation queues now both target transition_model again.`
- artifacts:
  - `scripts/report_frontier_promotion_plan.py`
  - `scripts/run_frontier_promotion_pass.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan or run_frontier_promotion_pass'`
  - `python -m py_compile scripts/report_frontier_promotion_plan.py scripts/run_frontier_promotion_pass.py tests/test_generation_scripts.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Treat transition_model as the truthful live bootstrap lane again. If you want the next highest-ROI coordinator move, inspect the transition_model dry-run finalize review package rather than continuing down the empty tooling candidate path.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:01:52Z`
- scope_id: `supervised_coordination_codex_apr02_transition_model_review`
- progress_label: `supervised_coordination_codex_apr02_transition_model_review`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/finalize_latest_candidate_from_cycles.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json --subsystem transition_model --variant-id regression_guard --scope-id codex_high_roi_20260330T2056Z --candidate-index 0 --dry-run
cd /data/agentkernel && python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path trajectories/improvement/candidates/codex_high_roi_20260330T2056Z/transition_model/cycle_transition_model_20260331T040430967602Z_4b84510d_regression_guard/transition_model_proposals.json --before-cycle-id cycle:transition_model:20260331T040430967602Z:4b84510d
```

- hypothesis: `The top compatible transition_model candidate should either collapse like the invalid tooling bootstrap path or resolve into a narrow first-retain review package that is worth operator attention under the current trust gate.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `trajectories/improvement/candidates/codex_high_roi_20260330T2056Z/transition_model/cycle_transition_model_20260331T040430967602Z_4b84510d_regression_guard/transition_model_proposals.json`
  - `trajectories/transition_model/transition_model_proposals.json`
- released_at: `2026-04-02T00:08:40Z`
- result: `completed`
- findings:
  - `The transition_model dry-run finalize bridge resolved cleanly and emitted a concrete operator-review target instead of failing on artifact readiness.`
  - `Retained compare is still bootstrap-first-retain here too: compare_retained_baseline exits with "no prior retained baseline exists for subsystem=transition_model", so this lane is review-ready but not compare-ready.`
  - `The candidate delta versus the active transition_model policy is narrow and auditable rather than noisy: generation_focus shifts from balanced to regression_guard, regressed_path_command_penalty increases from 3 to 5, and one new regression_guard proposal is added while the existing repeat/recovery controls remain unchanged.`
- artifacts:
  - `dry-run finalize output: python scripts/finalize_improvement_cycle.py --subsystem transition_model --cycle-id cycle:transition_model:20260331T040430967602Z:4b84510d --artifact-path trajectories/improvement/candidates/codex_high_roi_20260330T2056Z/transition_model/cycle_transition_model_20260331T040430967602Z_4b84510d_regression_guard/transition_model_proposals.json`
  - `candidate artifact: trajectories/improvement/candidates/codex_high_roi_20260330T2056Z/transition_model/cycle_transition_model_20260331T040430967602Z_4b84510d_regression_guard/transition_model_proposals.json`
  - `active artifact: trajectories/transition_model/transition_model_proposals.json`
- verification:
  - `python scripts/finalize_latest_candidate_from_cycles.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json --subsystem transition_model --variant-id regression_guard --scope-id codex_high_roi_20260330T2056Z --candidate-index 0 --dry-run`
  - `python scripts/compare_retained_baseline.py --subsystem transition_model --artifact-path trajectories/improvement/candidates/codex_high_roi_20260330T2056Z/transition_model/cycle_transition_model_20260331T040430967602Z_4b84510d_regression_guard/transition_model_proposals.json --before-cycle-id cycle:transition_model:20260331T040430967602Z:4b84510d`
- next_recommendation: `Keep transition_model as the live review target. Do not run finalize_improvement_cycle.py unattended while trust_status remains bootstrap, but if an operator wants the highest-ROI next decision, this candidate is the cleanest first-retain package to review now.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:10:16Z`
- scope_id: `supervised_coordination_codex_apr02_transition_model_regression_guard_fix`
- progress_label: `supervised_coordination_codex_apr02_transition_model_regression_guard_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_improvement.py -k 'transition_model_proposal_artifact or regression_guard_requires_regression_evidence or regression_guard_uses_regression_evidence'
cd /data/agentkernel && python -m py_compile agent_kernel/transition_model_improvement.py tests/test_improvement.py
```

- hypothesis: `The queued transition_model regression_guard candidate may be overstating regression evidence, so the right ROI move is to fix generation logic if focus-only variants can force unsupported regression penalties or proposal text.`
- owned_paths:
  - `agent_kernel/transition_model_improvement.py`
  - `tests/test_improvement.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T00:16:29Z`
- result: `completed`
- findings:
  - `Found and fixed an upstream generation bug in agent_kernel/transition_model_improvement.py: focus=regression_guard was forcing regressed_path_command_penalty>=5 and a regression_guard proposal even when the transition summary had no state_regression signal and regression_path_count=0.`
  - `Added targeted regressions in tests/test_improvement.py so regression_guard now requires actual regression evidence before it strengthens the regression penalty or emits regression-claiming proposal text, while still preserving the focused behavior when regressions are present.`
  - `A live non-mutating rebuild against trajectories/episodes confirms the queued first-retain candidate is now stale under corrected logic: the fixed generator still emits generation_focus=regression_guard, but it keeps regressed_path_command_penalty at 3 and proposal_areas=['repeat_avoidance', 'recovery_bias'] instead of the older penalty=5 plus regression_guard proposal.`
- artifacts:
  - `agent_kernel/transition_model_improvement.py`
  - `tests/test_improvement.py`
  - `comparison against stale candidate: old_penalty=5 old_areas=['repeat_avoidance', 'regression_guard', 'recovery_bias']; rebuilt_penalty=3 new_areas=['repeat_avoidance', 'recovery_bias']`
- verification:
  - `pytest -q tests/test_improvement.py -k 'transition_model_proposal_artifact or regression_guard_requires_regression_evidence or regression_guard_uses_regression_evidence'`
  - `python -m py_compile agent_kernel/transition_model_improvement.py tests/test_improvement.py`
  - `python - <<'PY' ... build_transition_model_proposal_artifact(KernelConfig().trajectories_root, focus='regression_guard', current_payload=...) ... PY`
- next_recommendation: `Do not finalize the old codex_high_roi_20260330T2056Z transition_model candidate as-is. Regenerate or refresh the transition_model candidate set under the corrected generator first, then re-run frontier/supervisor review on the refreshed artifact.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:17:26Z`
- scope_id: `supervised_coordination_codex_apr02_transition_refresh_cycle`
- progress_label: `supervised_coordination_codex_apr02_transition_refresh_cycle`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --subsystem transition_model --variant-id regression_guard --scope-id supervised_transition_codex_apr02_reguard_refresh --progress-label supervised_transition_codex_apr02_reguard_refresh --notes "refresh transition_model regression_guard candidate under corrected regression-evidence gating"
cd /data/agentkernel && pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'
cd /data/agentkernel && python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `A fresh transition_model cycle under corrected regression-evidence gating should produce a cleaner replacement for the stale regression_guard package, and the promotion planner should stop preferring superseded older same-variant previews once a newer compatible candidate exists.`
- owned_paths:
  - `scripts/report_frontier_promotion_plan.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_transition_codex_apr02_reguard_refresh.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- released_at: `2026-04-02T00:22:17Z`
- result: `completed`
- findings:
  - `Ran a new scoped generate-only cycle on transition_model/regression_guard. It completed cleanly in 48.956 seconds and produced a refreshed candidate at trajectories/improvement/candidates/supervised_transition_codex_apr02_reguard_refresh/transition_model/cycle_transition_model_20260402T001826877977Z_2fab19ed_regression_guard/transition_model_proposals.json. The refreshed artifact matches the corrected generator behavior: generation_focus stays regression_guard, but regressed_path_command_penalty remains 3 and proposal_areas are only repeat_avoidance plus recovery_bias.`
  - `Patched scripts/report_frontier_promotion_plan.py so older timestamped candidates in the same subsystem/variant family take a supersession penalty when a newer compatible candidate exists. That stops stale preview artifacts from outranking their own refreshes purely because the older run was shorter.`
  - `After rebuilding the frontier and rerunning the dry supervisor round, the machine-owned queue no longer points at the stale regression_guard package. The live bootstrap and trust-recovery queues now target transition_model scope supervised_vllm_live_20260331T0330Z variant repeat_avoidance, while the refreshed regression_guard scope remains on the frontier as valid secondary evidence.`
- artifacts:
  - `fresh cycle record: trajectories/improvement/cycles_supervised_transition_codex_apr02_reguard_refresh.jsonl`
  - `fresh candidate: trajectories/improvement/candidates/supervised_transition_codex_apr02_reguard_refresh/transition_model/cycle_transition_model_20260402T001826877977Z_2fab19ed_regression_guard/transition_model_proposals.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `bootstrap queue: trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trust queue: trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --subsystem transition_model --variant-id regression_guard --scope-id supervised_transition_codex_apr02_reguard_refresh --progress-label supervised_transition_codex_apr02_reguard_refresh --notes "refresh transition_model regression_guard candidate under corrected regression-evidence gating"`
  - `pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'`
  - `python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `Use the supervisor queue's current repeat_avoidance transition_model package as the live bootstrap review target. If you want another kernel-improving generation cycle instead of review, refresh transition_model/repeat_avoidance next so the active shortlist and the latest generated evidence stay aligned.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:25:44Z`
- scope_id: `supervised_coordination_codex_apr02_repeat_refresh_cycle`
- progress_label: `supervised_coordination_codex_apr02_repeat_refresh_cycle`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --subsystem transition_model --variant-id repeat_avoidance --scope-id supervised_transition_codex_apr02_repeat_refresh --progress-label supervised_transition_codex_apr02_repeat_refresh --notes "refresh transition_model repeat_avoidance candidate to align with live shortlist"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `If the live bootstrap shortlist already prefers transition_model/repeat_avoidance, a fresh scoped cycle in that exact variant family should both improve kernel evidence quality and potentially replace the older March repeat_avoidance scope in the supervisor queues.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_transition_codex_apr02_repeat_refresh.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- released_at: `2026-04-02T00:28:18Z`
- result: `completed`
- findings:
  - `Ran a new scoped generate-only cycle on transition_model/repeat_avoidance. It completed cleanly in 51.5867 seconds and produced a refreshed candidate at trajectories/improvement/candidates/supervised_transition_codex_apr02_repeat_refresh/transition_model/cycle_transition_model_20260402T002649776305Z_1647ac2c_repeat_avoidance/transition_model_proposals.json with repeat_command_penalty=6, regressed_path_command_penalty=3, and proposal_areas=['repeat_avoidance','recovery_bias'].`
  - `The observation result was materially stronger than the prior transition_model refreshes: the scoped primary pass finished at 5/5 success across bounded and discovered_task families, with no timeouts and all five tasks terminating in success after one step.`
  - `After rebuilding the frontier and rerunning the dry supervisor round, the fresh April scopes now own the top two promotion slots, but the machine-owned queue still points at supervised_transition_codex_apr02_reguard_refresh rather than the new repeat_avoidance scope. The refreshed repeat_avoidance candidate ranks second with promotion_score=4.6804, just behind refreshed regression_guard at 4.7681.`
- artifacts:
  - `fresh cycle record: trajectories/improvement/cycles_supervised_transition_codex_apr02_repeat_refresh.jsonl`
  - `fresh candidate: trajectories/improvement/candidates/supervised_transition_codex_apr02_repeat_refresh/transition_model/cycle_transition_model_20260402T002649776305Z_1647ac2c_repeat_avoidance/transition_model_proposals.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `bootstrap queue: trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trust queue: trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --subsystem transition_model --variant-id repeat_avoidance --scope-id supervised_transition_codex_apr02_repeat_refresh --progress-label supervised_transition_codex_apr02_repeat_refresh --notes "refresh transition_model repeat_avoidance candidate to align with live shortlist"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `If you want the current machine-owned next action, review the refreshed supervised_transition_codex_apr02_reguard_refresh bootstrap package. If you want the next kernel-improving implementation step instead, improve promotion scoring so transition_model candidates can account for strong observation outcomes like the 5/5 repeat_avoidance run instead of ranking almost entirely on generation/existence/elapsed heuristics.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:28:18Z`
- scope_id: `supervised_coordination_codex_apr02_passrate_scoring_fix`
- progress_label: `supervised_coordination_codex_apr02_passrate_scoring_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'
cd /data/agentkernel && python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2
```

- hypothesis: `The remaining ROI blocker is ranking bias, not evidence generation: the frontier already contains a 5/5 repeat_avoidance run, but the promotion plan still ignores observation pass-rate fields and overweights elapsed time.`
- owned_paths:
  - `scripts/report_frontier_promotion_plan.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- released_at: `2026-04-02T00:31:45Z`
- result: `completed`
- findings:
  - `Patched scripts/report_frontier_promotion_plan.py so candidate scoring now rewards successful observation outcomes through primary_pass_rate/observation_pass_rate instead of ranking almost entirely on candidate existence and elapsed seconds.`
  - `Added a regression in tests/test_generation_scripts.py proving a slower but successful candidate outranks a faster failed candidate once pass-rate evidence is present.`
  - `After rebuilding the live reports and rerunning the dry supervisor round, the machine-owned queue flipped to the fresh repeat_avoidance scope supervised_transition_codex_apr02_repeat_refresh. Both bootstrap review and trust-streak recovery now target the 5/5 repeat_avoidance run rather than the weaker regression_guard package.`
- artifacts:
  - `scripts/report_frontier_promotion_plan.py`
  - `tests/test_generation_scripts.py`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `bootstrap queue: trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trust queue: trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
- verification:
  - `pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'`
  - `python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 2`
- next_recommendation: `The current highest-ROI next action is operator review of supervised_transition_codex_apr02_repeat_refresh. If you want another kernel-improving implementation step instead of review, the next upstream target is frontier ingestion richness rather than planner ranking, because the pass-rate signal is now flowing through to supervisor selection.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T00:45:00Z`
- scope_id: `supervised_coordination_codex_apr02_successfanout40`
- progress_label: `supervised_coordination_codex_apr02_successfanout40`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40 --notes "low-cost repository/project coding run with wider generated_success fanout window"
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10 --notes "low-cost repository/project coding run with wider generated_success fanout window and tighter primary decision cap"
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10_retry --notes "retry low-cost repository/project coding run with wider generated_success fanout window and tighter primary decision cap after tolbert readiness miss"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10_timeout30 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10_timeout30 --notes "retry low-cost repository/project coding run with wider generated_success fanout window, tighter primary decision cap, and tolbert startup timeout 30s"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4 --notes "low-cost repository/project coding run with wider generated_success fanout window, tighter primary decision cap, startup timeout 30s, and task_limit=4 to stay within envelope"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 12 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4_cap12 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4_cap12 --notes "low-cost repository/project coding run with wider generated_success fanout window, startup timeout 30s, and task_limit=4 with 12s decision cap"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The low-cost repository/project path is already producing 5/5 primary success and 2/2 generated_success under a 20-second followup window. Doubling the staged generated_success budget should show whether the kernel can convert that stronger coding seed pool into more adjacent coding wins before another implementation step is needed.`
- owned_paths:
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10_retry.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10_timeout30.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4_cap12.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T01:02:45Z`
- result: `completed`
- findings:
  - `The intended ROI probe was to widen generated_success fanout beyond the prior 2/2 adjacent-task ceiling by keeping the low-cost repository/project slice and doubling the staged generated_success budget from 20s to 40s.`
  - `That probe did not produce a clean observation run under the current parallel load. The baseline 5-task fanout scope and the 30s-startup variant both exceeded the 60-second primary envelope before any generated_success followup began, while the 10s decision-cap and 4-task variants hit infrastructure and stage-budget failures instead of policy failures.`
  - `The dominant blocker is shared TolBERT startup pressure, not coding-lane regression. Two retry scopes failed before the first task because TOLBERT did not become ready within 15s, and the 30s-startup 4-task retry exposed the root cause directly: CUDA out of memory while loading the TolBERT model, with only 11.25 MiB free on GPU 0 and another process holding 22.84 GiB.`
  - `The shared frontier now records these as failed observation scopes rather than silently dropping them. After refresh, the frontier summary moved to 37 scoped runs with 8 observation-only failed runs, 4 timed-out runs, and 10 warning runs, so the machine-readable evidence bus now reflects that the current blocker is runtime contention.`
- artifacts:
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10_retry.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_cap10_timeout30.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4_cap12.jsonl`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40 --notes "low-cost repository/project coding run with wider generated_success fanout window"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10 --notes "low-cost repository/project coding run with wider generated_success fanout window and tighter primary decision cap"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10_retry --notes "retry low-cost repository/project coding run with wider generated_success fanout window and tighter primary decision cap after tolbert readiness miss"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_cap10_timeout30 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_cap10_timeout30 --notes "retry low-cost repository/project coding run with wider generated_success fanout window, tighter primary decision cap, and tolbert startup timeout 30s"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 10 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4 --notes "low-cost repository/project coding run with wider generated_success fanout window, tighter primary decision cap, startup timeout 30s, and task_limit=4 to stay within envelope"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 12 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4_cap12 --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4_cap12 --notes "low-cost repository/project coding run with wider generated_success fanout window, startup timeout 30s, and task_limit=4 with 12s decision cap"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Do not spend more supervised observation budget on wider generated_success fanout until TolBERT cold-start contention is reduced. The next ROI is runtime-evidence or orchestration work that serializes TolBERT-heavy supervised runs, warms the service once per wave, or shifts the parallel mix so the coding lane stops dying on startup OOM before policy evidence can accumulate.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:00:40Z`
- scope_id: `supervised_coordination_codex_apr02_toolbridge_fix`
- progress_label: `supervised_coordination_codex_apr02_toolbridge_fix`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_memory.py -k 'tool_candidate_extractor'
cd /data/agentkernel && python -m py_compile agent_kernel/extractors.py tests/test_memory.py
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_refresh --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_refresh --notes "tooling extractor now seeds from learning artifacts and admits compact repository/project procedures"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=45 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_retry --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_retry --notes "tooling extractor learning-artifact bridge retry with extended Tolbert startup timeout"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python - <<'PY'
import json, tempfile
from pathlib import Path
from agent_kernel.extractors import extract_tool_candidates
from agent_kernel.improvement import assess_artifact_compatibility
out = Path(tempfile.gettempdir()) / 'agentkernel_tool_candidates_probe.json'
extract_tool_candidates(Path('trajectories/episodes'), out)
payload = json.loads(out.read_text())
print(json.dumps(assess_artifact_compatibility(subsystem='tooling', payload=payload), indent=2))
PY
```

- hypothesis: `The coding lane now has strong repository/project wins, but tooling promotion still collapses because tool candidate extraction ignores fresh learning artifacts and emits empty or incompatible payloads. If tooling extraction bridges the learning candidates and stays TaskBank-anchored, the kernel can surface promotable coding procedures again without another ranking change.`
- owned_paths:
  - `agent_kernel/extractors.py`
  - `tests/test_memory.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_refresh.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_retry.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
- released_at: `2026-04-02T01:09:55Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/extractors.py so extract_tool_candidates now seeds from persisted success_skill_candidate learning artifacts, accepts compact repository/project procedures, and filters both learning-derived and episode-derived candidates through the current TaskBank.`
  - `Added extractor regressions in tests/test_memory.py covering single-command repository episodes, learning-artifact seeding, and rejection of unknown adjacent-task ids.`
  - `The direct live probe is now healthy. Against the current repo, extract_tool_candidates emits 43 tool candidates and assess_artifact_compatibility(subsystem='tooling', payload=...) returns compatible=true. The fresh repository/project wins for service_release_task, schema_alignment_task, report_rollup_task, repo_sync_matrix_task, and deployment_manifest_task are all present as compatible local_shell_procedure candidates.`
  - `Two fresh scoped refresh attempts still failed before generate, but for runtime reasons upstream of tooling extraction. supervised_coordination_codex_apr02_toolbridge_refresh died on TolBERT readiness after 15s, while supervised_coordination_codex_apr02_toolbridge_retry got past startup with a 45s service timeout and then exhausted the 60s observe window before candidate generation.`
  - `After refreshing trajectories/improvement/reports/supervised_parallel_frontier.json, both toolbridge scopes are now recorded explicitly as failed observation scopes under selected_subsystem=observation, which keeps the machine-readable evidence honest about the current blocker.`
- artifacts:
  - `agent_kernel/extractors.py`
  - `tests/test_memory.py`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_refresh.jsonl`
  - `failed scope record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_retry.jsonl`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
- verification:
  - `pytest -q tests/test_memory.py -k 'tool_candidate_extractor'`
  - `python -m py_compile agent_kernel/extractors.py tests/test_memory.py`
  - `python scripts/report_supervised_frontier.py`
  - `python - <<'PY' ... extract_tool_candidates(Path('trajectories/episodes'), out) ... assess_artifact_compatibility(subsystem='tooling', payload=payload) ... PY`
- next_recommendation: `Do not spend another cycle on tooling extractor logic. The next ROI is runtime/orchestration work that gets one repository/project tooling scope through bounded observation now that candidate generation is fixed, or otherwise an operator review of the existing transition-model bootstrap target while TolBERT contention is being reduced.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:14:00Z`
- scope_id: `supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun`
- progress_label: `supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 12 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun --notes "rerun low-cost repository/project coding fanout after separate training environment release"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The prior 4-task/12-second/40-second-fanout probe was blocked by external GPU contention, not by kernel behavior. If that training load is gone, the same scoped run should complete and reveal the true adjacent-success fanout yield.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T01:16:45Z`
- result: `completed`
- findings:
  - `The rerun validated the hypothesis. With GPU headroom restored, the same 4-task / 12-second decision-cap / 40-second generated_success configuration completed cleanly instead of dying on TolBERT startup.`
  - `The scoped run finished healthy with 4/4 primary success and 4/4 generated_success success, and the generated metrics merged successfully. This is the strongest direct coding-lane fanout evidence so far: the wider followup window produced four passing adjacent repository/project tasks rather than the previous 2/2 ceiling.`
  - `Observation completed without warning and produced a fresh tooling candidate at trajectories/improvement/candidates/supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun/tooling/cycle_tooling_20260402T011618751201Z_93191b23_procedure_promotion/tool_candidates.json.`
  - `The practical conclusion is narrower than the prior closeout implied: wider coding fanout was not fundamentally blocked. It was blocked by a separate training environment consuming the GPU. Under normal headroom, the widened fanout path is now a valid ROI-positive supervised configuration.`
- artifacts:
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun.jsonl`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun/tooling/cycle_tooling_20260402T011618751201Z_93191b23_procedure_promotion/tool_candidates.json`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 4 --max-observation-seconds 60 --max-current-task-decision-seconds 12 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun --notes "rerun low-cost repository/project coding fanout after separate training environment release"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Treat the 4-task / 12-second / 40-second fanout recipe as the current best coding-lane supervised run shape when GPU headroom is available. The next ROI is either another scoped run in this shape to compound tooling evidence, or candidate review/finalize work now that the fanout path is generating 4/4 adjacent successes instead of stalling.`

### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:18:58Z`
- scope_id: `supervised_coordination_codex_apr02_tooling_bootstrap_refresh`
- progress_label: `supervised_coordination_codex_apr02_tooling_bootstrap_refresh`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 3
```

- hypothesis: `The new 4/4 tooling fanout run may be strong enough to displace transition_model as the current machine-owned bootstrap review target, but the authoritative answer must come from a fresh supervisor round rather than stale queue state.`
- owned_paths:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
  - `trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
- released_at: `2026-04-02T01:19:10Z`
- result: `completed`
- findings:
  - `A fresh dry supervisor round promoted the new tooling candidate into the machine-owned bootstrap review queues. The baseline bootstrap and trust-streak recovery reports now both include tooling, retrieval, and transition_model, with tooling listed first from scope supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun.`
  - `The refreshed promotion pass scored the tooling bootstrap candidate at 7.5, ahead of retrieval at 7.5 tie-level placement but after ordering and ahead of transition_model at 7.1804. The compare result is still bootstrap_first_retain because no retained tooling baseline exists yet.`
  - `The supervisor-generated dry-run finalize path for tooling is now explicit and ready for operator review: python scripts/finalize_latest_candidate_from_cycles.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json --subsystem tooling --scope-id supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun --candidate-index 0 --dry-run`
  - `Finalize remains intentionally trust-gated. This round did not authorize unattended finalize; it prepared the bootstrap review package and recorded tooling as the current machine-owned review target.`
- artifacts:
  - `supervisor status: trajectories/improvement/reports/supervisor_loop_status.json`
  - `supervisor report: trajectories/improvement/reports/supervisor_loop_report.json`
  - `bootstrap queue: trajectories/improvement/reports/supervisor_baseline_bootstrap_queue.json`
  - `trust queue: trajectories/improvement/reports/supervisor_trust_streak_recovery_queue.json`
  - `promotion pass: trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
- verification:
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 3`
- next_recommendation: `If you want to stay on the review/finalize side, inspect the tooling bootstrap package next and decide whether to keep it on trust-gated review or open bootstrap finalize policy. If you want more evidence instead, run another scoped repository/project tooling cycle with the same 4-task / 12-second / 40-second recipe while GPU headroom is still clear.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:33:00Z`
- scope_id: `supervised_coordination_codex_apr02_warningaware_promotion`
- progress_label: `supervised_coordination_codex_apr02_warningaware_promotion`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py
cd /data/agentkernel && pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
cd /data/agentkernel && python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 3
```

- hypothesis: `Promotion ranking was still flattening healthy coding-lane fanout evidence together with weaker completed_with_warnings tooling scopes, and unconditional recency superseding could bury the stronger older candidate. If the scorer becomes warning-aware, generated-success-aware, and only supersedes when the newer same-variant candidate is not materially weaker, the machine-owned queue should swing back toward the healthy 4/4 tooling fanout scope.`
- owned_paths:
  - `scripts/report_frontier_promotion_plan.py`
  - `tests/test_generation_scripts.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
  - `trajectories/improvement/reports/supervisor_loop_report.json`
- released_at: `2026-04-02T01:38:51Z`
- result: `completed`
- findings:
  - `Patched scripts/report_frontier_promotion_plan.py so promotion base score now includes a healthy-run bonus, a generated-success bonus, and lighter warning penalties for supplemental followup overruns than for true budget overruns or timeouts. The plan output now exposes warning category and score-component fields directly.`
  - `Refined same-variant supersede logic so an older candidate is only history-penalized when the newest same-variant candidate is not materially weaker on the new base score. That stops a newer warning-only tooling refresh from automatically burying the stronger healthy fanout candidate.`
  - `Added regression coverage in tests/test_generation_scripts.py for both sides of that behavior: stronger newer candidates still supersede weaker older ones, while a healthy older tooling candidate with 4/4 generated-success evidence now stays ahead of a newer completed_with_warnings tooling run.`
  - `After refreshing the live reports, the machine-owned ranking moved back to the expected coding-lane winner: supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun now leads the promotion plan at 9.05, ahead of supervised_transition_codex_apr02_repeat_refresh at 7.5304 and supervised_coordination_codex_apr02_autogenbudget_fallback70_freshbudget at 7.35.`
- artifacts:
  - `scripts/report_frontier_promotion_plan.py`
  - `tests/test_generation_scripts.py`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_pass.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
- verification:
  - `python -m py_compile scripts/report_frontier_promotion_plan.py tests/test_generation_scripts.py`
  - `pytest -q tests/test_generation_scripts.py -k 'report_frontier_promotion_plan'`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 3`
- next_recommendation: `The next ROI is no longer scorer repair. Either inspect the top tooling bootstrap package for review/finalize quality, or run one more low-cost repository/project tooling scope with the current 4-task / 12-second / 40-second recipe so the top bootstrap lane accumulates another clean supervised win.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:20:00Z`
- scope_id: `supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert`
- progress_label: `supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py
cd /data/agentkernel && pytest -q tests/test_protocol_comparison.py -k "balances_primary_task_limit_for_wide_generated_success_fanout or retries_generated_success_without_tolbert_context_after_startup_failure or allows_two_generated_success_followups_when_budget_supports_it or retries_with_reduced_primary_task_limit_after_timeout or retries_without_tolbert_context_after_startup_failure"
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance --notes "auto-balance wide generated_success coding fanout to the strongest cross-family primary slice"
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert --notes "auto-balance wide generated_success coding fanout and retry staged followups without tolbert after startup failure"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The current best coding-lane recipe was still trapped behind manual coordinator tuning: the plain 5-task/40-second repository/project command only became healthy after manually dropping primary depth to 4 and surviving staged TolBERT startup noise. If the supervised wrapper auto-balances wide generated_success runs to the strongest cross-family primary slice and applies the existing TolBERT startup fallback to staged followups too, the kernel should recover the 4/4 + 4/4 coding result from the simpler 5-task command.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T01:40:27Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so explicit low-cost repository/project runs with wide generated_success fanout now auto-balance the primary slice to two tasks per family when the followup budget can replay the full seed pool. The plain 5-task/40-second coding command now resolves to primary_task_limit=4/5 source=generated_success_family_balance and recomputes the default primary decision budget to 12.0 seconds.`
  - `Extended the TolBERT startup fallback from the primary observation into staged curriculum followups. When a generated_success followup child fails specifically on TolBERT readiness, the wrapper now retries that followup once with use_tolbert_context=False and the same fresh followup budget instead of dropping the entire adjacent-success sweep.`
  - `Added protocol regressions in tests/test_protocol_comparison.py for both behaviors: the new auto-balance path on wide generated_success coding runs and the staged generated_success TolBERT retry path.`
  - `The first live validation scope supervised_coordination_codex_apr02_successfanout40_autobalance proved the primary auto-balance change immediately. It completed 4/4 primary success from the plain 5-task command, but its staged generated_success followup hit a separate TolBERT startup failure and finished as completed_with_warnings with generated_success=0/0.`
  - `The followup patch converted that warning-only run into the intended coding result. The new scope supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert completed 4/4 primary success plus 4/4 generated_success success under the same plain 5-task/40-second command, produced a fresh tooling candidate, and entered the refreshed promotion plan at score 7.35 despite the remaining budget_exceeded warning from the outer 60-second envelope.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert.jsonl`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert/tooling/cycle_tooling_20260402T013944601962Z_92a5344b_procedure_promotion/tool_candidates.json`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "balances_primary_task_limit_for_wide_generated_success_fanout or retries_generated_success_without_tolbert_context_after_startup_failure or allows_two_generated_success_followups_when_budget_supports_it or retries_with_reduced_primary_task_limit_after_timeout or retries_without_tolbert_context_after_startup_failure"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance --notes "auto-balance wide generated_success coding fanout to the strongest cross-family primary slice"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance_followuptolbert --notes "auto-balance wide generated_success coding fanout and retry staged followups without tolbert after startup failure"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next ROI is runtime-envelope cleanup rather than coding selection. The plain 5-task command now recovers the full 4/4 + 4/4 coding result automatically, but the scope still lands as completed_with_warnings because the fresh-budget primary TolBERT retry pushes total elapsed time past the outer 60-second envelope. Tighten that accounting or warm TolBERT once per scope so this recipe can become a healthy frontier win instead of a warning-scored one.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:41:00Z`
- scope_id: `supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert`
- progress_label: `supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py
cd /data/agentkernel && pytest -q tests/test_protocol_comparison.py -k "disables_tolbert_context_for_bounded_tooling_coding_run or disables_tolbert_context_for_bounded_transition_model_run or balances_primary_task_limit_for_wide_generated_success_fanout or retries_generated_success_without_tolbert_context_after_startup_failure or retries_without_tolbert_context_after_startup_failure or retries_with_reduced_primary_task_limit_after_timeout"
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert --notes "auto-balance wide generated_success coding fanout and disable tolbert up front for the bounded repository/project primary slice"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The coding lane already had a healthy 4/4 + 4/4 recipe, but it still required a manually tuned task_limit=4 run because the ordinary bounded 5-task command could spend its primary budget on a fresh TolBERT startup retry. If the wrapper disables TolBERT up front once the low-cost repository/project primary slice auto-balances down to the proven 4-task shape, the plain bounded command should become a healthy coding frontier run instead of a completed_with_warnings scope.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T01:49:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so bounded low-cost tooling observations on explicit repository/project families disable TolBERT context up front once the primary slice is at or below four tasks. The wrapper now records tooling_repository_project_bounded_startup_guard and keeps the chosen observation config across primary retries instead of reintroducing TolBERT later in the same scope.`
  - `Added a new protocol regression in tests/test_protocol_comparison.py covering the plain 5-task repository/project coding run. It now verifies that the wrapper auto-balances to task_limit=4, starts the primary observation with use_tolbert_context=False, records the new disable reason, and still merges 4/4 generated_success followup metrics.`
  - `Fresh live evidence upgraded the automated coding path from warning-only to healthy. The new scope supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert completed 4/4 primary success plus 4/4 generated_success success under the ordinary 5-task/60-second/40-second command, finished observation in 45.0989 seconds with observation_budget_exceeded=false, and produced a fresh tooling candidate.`
  - `After refreshing the reports, the new scope entered both the frontier and promotion plan as healthy and moved to the top of the promotion candidates at score 8.9467, ahead of the older manual rerun supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun at 8.45.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert.jsonl`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert/tooling/cycle_tooling_20260402T014742818833Z_7d385423_procedure_promotion/tool_candidates.json`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "disables_tolbert_context_for_bounded_tooling_coding_run or disables_tolbert_context_for_bounded_transition_model_run or balances_primary_task_limit_for_wide_generated_success_fanout or retries_generated_success_without_tolbert_context_after_startup_failure or retries_without_tolbert_context_after_startup_failure or retries_with_reduced_primary_task_limit_after_timeout"`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert --notes "auto-balance wide generated_success coding fanout and disable tolbert up front for the bounded repository/project primary slice"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next ROI on the coding lane is no longer startup recovery. The plain bounded command is now healthy, so the next useful step is either a second clean repository/project run to build trust streak on the top tooling scope or a bootstrap/finalize review pass against the new candidate while the evidence is fresh.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:52:00Z`
- scope_id: `supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve`
- progress_label: `supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py
cd /data/agentkernel && pytest -q tests/test_protocol_comparison.py -k "preserves_five_task_tooling_primary_when_tolbert_budget_is_strong or balances_primary_task_limit_for_wide_generated_success_fanout or disables_tolbert_context_for_bounded_tooling_coding_run or retries_generated_success_without_tolbert_context_after_startup_failure or retries_without_tolbert_context_after_startup_failure"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve --notes "preserve the wider 5-task repository/project TolBERT-backed primary when the bounded decision budget remains strong enough"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The current wrapper still collapsed the ordinary 5-task/40-second coding fanout command back to the 4-task fallback even though live evidence already showed the low-cost 5-task repository/project slice can complete cleanly with TolBERT under the 60-second bounded window. If the auto-balance logic preserves that requested fifth primary task whenever the bounded per-task decision budget remains strong, the wider TolBERT-backed path should become a real 5/5 + 5/5 coding run instead of an unreachable manual variant.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
  - `docs/ai_agent_status.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T01:59:30Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so generated-success auto-balance now preserves a requested fifth tooling primary task when the bounded observation window still yields a strong default decision budget. Concretely, the wrapper no longer shrinks the low-cost repository/project 5-task slice to 4 when the requested slice still gets a 10.0-second per-task decision budget under the 60-second window.`
  - `Kept the previous 4-task guard intact for genuinely wider fanout. The new regression coverage in tests/test_protocol_comparison.py proves both sides: a 6-task wide request still auto-balances down to 4 and disables TolBERT for the bounded fallback shape, while the 5-task repository/project request now stays at 5 with TolBERT enabled.`
  - `Fresh live evidence is now in trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve.jsonl. Under the wider TolBERT-backed path, the ordinary 5-task/60-second/40-second repository/project command completed healthy with 5/5 primary success plus 5/5 generated_success success, finished in 59.2232 seconds with observation_budget_exceeded=false, and produced a fresh tooling candidate.`
  - `After refreshing the frontier and promotion plan, the new scope entered as a healthy coding candidate with full 5/5 + 5/5 evidence. It ranks behind the leaner 4-task default scope at promotion_score=8.4759 versus 8.9467, but it is now the strongest validated larger-task-limit TolBERT-backed coding recipe on the frontier.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve.jsonl`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve/tooling/cycle_tooling_20260402T015848777834Z_51ae004e_procedure_promotion/tool_candidates.json`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "preserves_five_task_tooling_primary_when_tolbert_budget_is_strong or balances_primary_task_limit_for_wide_generated_success_fanout or disables_tolbert_context_for_bounded_tooling_coding_run or retries_generated_success_without_tolbert_context_after_startup_failure or retries_without_tolbert_context_after_startup_failure"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 40 --progress-label supervised_coordination_codex_apr02_successfanout40_task5_tolbertpreserve --notes "preserve the wider 5-task repository/project TolBERT-backed primary when the bounded decision budget remains strong enough"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next ROI for larger-task-limit coding is no longer proving the 5-task TolBERT path itself. Either build a trust streak on this new 5-task recipe with one more clean rerun, or improve promotion scoring so the frontier can value the wider 5/5 + 5/5 coding evidence more directly against the faster 4-task default winner.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T02:18:00Z`
- scope_id: `supervised_coordination_codex_apr02_tasklimit_scaling_tolbert`
- progress_label: `supervised_coordination_codex_apr02_tasklimit_scaling_tolbert`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py
cd /data/agentkernel && pytest -q tests/test_protocol_comparison.py -k "expands_tooling_priority_families_for_high_task_limit or enables_discovered_tasks_for_very_high_tooling_task_limit or preserves_five_task_tooling_primary_when_tolbert_budget_is_strong or reduces_five_task_tooling_primary_when_generated_budget_cannot_support_full_fanout or balances_primary_task_limit_for_wide_generated_success_fanout"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 10 --max-observation-seconds 120 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 80 --progress-label supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder --notes "expand tooling fanout to 10 tasks with workflow and repo_chore ahead of tooling, plus 8s primary TolBERT budget discipline"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 20 --max-observation-seconds 240 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8 --priority-benchmark-family repository --priority-benchmark-family project --progress-label supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8 --notes "scale tooling primary fanout to 20 tasks with adjacent coding families, repo_sandbox expansion, and 8s TolBERT budget discipline"
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 50 --max-observation-seconds 480 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8 --priority-benchmark-family repository --priority-benchmark-family project --progress-label supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8 --notes "scale tooling primary fanout to 50 tasks with discovered-task support, adjacent coding family expansion, and 8s TolBERT budget discipline"
cd /data/agentkernel && python scripts/report_supervised_frontier.py
cd /data/agentkernel && python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json
```

- hypothesis: `The coding lane had become healthy at 4-task and 5-task limits, but it was still structurally tiny because explicit repository/project focus supplied too few safe low-cost primaries. If the wrapper expands high task-limit tooling runs into adjacent coding families, enables discovered-task supply when the widened static pool is still too small, and caps primary decision budgets to 8 seconds for 10+ task TolBERT fanout, the kernel should scale beyond the tiny 5-task lane without immediately falling back to the small-slice runtime guard.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `docs/supervised_work_queue.md`
  - `docs/ai_agent_status.md`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8.jsonl`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8.jsonl`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- released_at: `2026-04-02T02:34:00Z`
- result: `completed`
- findings:
  - `Patched scripts/run_human_guided_improvement_cycle.py so tooling runs with task_limit >= 10 no longer stay trapped inside the tiny repository/project pool. The wrapper now auto-expands explicit repository/project focus into wider coding-adjacent families, and when a very large request still exceeds that widened low-cost pool it turns on include_discovered_tasks instead of silently pretending the larger slice exists.`
  - `Added a second runtime guard for these wide TolBERT-backed slices: when the run is in the new high-fanout mode and the operator did not set an explicit primary decision budget, the wrapper now caps the default primary decision budget at 8.0 seconds (`tooling_high_fanout_default`) so throughput scales with task_count instead of one late context-compile stall consuming the whole envelope.`
  - `Reordered the high-fanout family expansion to keep paper-research-heavy tooling tasks last. The first 10-task widened run proved that api_contract_task in the tooling family was the bad actor; after moving workflow and repo_chore ahead of tooling, the rerun stopped dying on tooling-paper-research and completed the full 10-task primary slice.`
  - `Fresh live evidence now spans three larger scales. Scope supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder completed 10/10 primary success and salvaged 2/2 generated_success before a late followup context-compile timeout, so it lands as completed_with_warnings. Scope supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8 completed healthy in 165.9708s with 15/20 primary pass rate 0.75 under the widened family set. Scope supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8 still fails: it made enough progress to retry at 22/50, but then hit another 8s context-compile timeout, so 50-task primary scale is still beyond the current runtime envelope.`
  - `The scaling result is no longer 'we only have five tasks'. The kernel now demonstrates real primary capacity at 10 and 20 tasks on the coding lane. The remaining blocker is runtime quality at the 50-task frontier, not lack of family supply.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_protocol_comparison.py`
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder.jsonl`
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8.jsonl`
  - `cycle record: trajectories/improvement/cycles_supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8.jsonl`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder/tooling/cycle_tooling_20260402T021932573126Z_6e2da9ac_procedure_promotion/tool_candidates.json`
  - `candidate: trajectories/improvement/candidates/supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8/tooling/cycle_tooling_20260402T022258131635Z_d4e1488e_procedure_promotion/tool_candidates.json`
  - `frontier: trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `promotion plan: trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_protocol_comparison.py -k "expands_tooling_priority_families_for_high_task_limit or enables_discovered_tasks_for_very_high_tooling_task_limit or preserves_five_task_tooling_primary_when_tolbert_budget_is_strong or reduces_five_task_tooling_primary_when_generated_budget_cannot_support_full_fanout or balances_primary_task_limit_for_wide_generated_success_fanout"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 10 --max-observation-seconds 120 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 80 --progress-label supervised_coordination_codex_apr02_tasklimit10_tolbertfanout_cap8_reorder --notes "expand tooling fanout to 10 tasks with workflow and repo_chore ahead of tooling, plus 8s primary TolBERT budget discipline"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 20 --max-observation-seconds 240 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8 --priority-benchmark-family repository --priority-benchmark-family project --progress-label supervised_coordination_codex_apr02_tasklimit20_tolbertprimary_cap8 --notes "scale tooling primary fanout to 20 tasks with adjacent coding families, repo_sandbox expansion, and 8s TolBERT budget discipline"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --subsystem tooling --task-limit 50 --max-observation-seconds 480 --generate-only --scope-id supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8 --priority-benchmark-family repository --priority-benchmark-family project --progress-label supervised_coordination_codex_apr02_tasklimit50_tolbertprimary_cap8 --notes "scale tooling primary fanout to 50 tasks with discovered-task support, adjacent coding family expansion, and 8s TolBERT budget discipline"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next ROI is specific 50-task runtime repair, not more family expansion. The 50-task probe now has enough supply and makes it to 22 completed tasks before failing in context_compile, so the next useful change is either a family-aware context-compile policy for high-fanout workflow/retrieval tasks or a larger-stage runtime budget policy that does not let one late task kill the whole 50-task slice.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T01:43:00Z`
- scope_id: `supervised_coordination_codex_apr02_toolbridge_partialsalvage`
- progress_label: `supervised_coordination_codex_apr02_toolbridge_partialsalvage`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_partialsalvage --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_partialsalvage --notes "preserve partial adjacent-success evidence when bounded generated_success followup times out"
```

- hypothesis: `If bounded generated_success followups preserve partial completed-adjacent-task metrics instead of flattening them to zero, tooling scopes can keep real coding evidence even when the staged child ends mid-followup; if runtime is already improved enough, the same path should simply finish cleanly inside the 20-second budget.`
- owned_paths:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `evals/harness.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T01:49:34Z`
- result: `completed`
- findings:
  - `Patched evals/harness.py so partial eval snapshots now carry generated benchmark-family counts, and patched scripts/run_human_guided_improvement_cycle.py so timed-out staged followups can salvage partial generated metrics with benchmark-family support instead of only keeping generated_by_kind.`
  - `Extended protocol coverage in tests/test_protocol_comparison.py for the timeout-salvage path so a timed-out generated_success followup can merge one completed adjacent-success result, including generated_by_benchmark_family, into the scoped observe metrics.`
  - `Fresh live tooling evidence is now in cycle:tooling:20260402T014807971722Z:c683b177. Under scope supervised_coordination_codex_apr02_toolbridge_partialsalvage, the bounded repository/project run completed healthy at 5/5 primary success plus 2/2 generated_success success within the separate 20-second followup budget, with no observation warning.`
  - `After refreshing the frontier and supervisor, the machine-owned bootstrap review target remained tooling-first: supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert now leads the bootstrap review queue at promotion_score=8.9467, ahead of supervised_coordination_codex_apr02_successfanout40_task4_cap12_rerun at 8.45 and retrieval at 7.95.`
- artifacts:
  - `scripts/run_human_guided_improvement_cycle.py`
  - `evals/harness.py`
  - `tests/test_protocol_comparison.py`
  - `trajectories/improvement/cycles_supervised_coordination_codex_apr02_toolbridge_partialsalvage.jsonl`
  - `trajectories/improvement/candidates/supervised_coordination_codex_apr02_toolbridge_partialsalvage/tooling/cycle_tooling_20260402T014807971722Z_c683b177_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
  - `trajectories/improvement/reports/supervisor_loop_status.json`
- verification:
  - `python -m py_compile scripts/run_human_guided_improvement_cycle.py evals/harness.py tests/test_protocol_comparison.py`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_coordination_codex_apr02_toolbridge_partialsalvage --priority-benchmark-family repository --priority-benchmark-family project --include-curriculum --generated-curriculum-budget-seconds 20 --progress-label supervised_coordination_codex_apr02_toolbridge_partialsalvage --notes "preserve partial adjacent-success evidence when bounded generated_success followup times out"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `python scripts/run_supervisor_loop.py --autonomy-mode dry_run --rollout-stage compare_only --max-rounds 1 --max-discovery-workers 0 --max-promotion-candidates 3`
- next_recommendation: `Do not spend more orchestration effort on the old toolbridge followup-repair shape. The live coding-lane ROI is now tooling bootstrap review quality around supervised_coordination_codex_apr02_successfanout40_autobalance_primarynotolbert, with retrieval second.`


### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__learning_memory`
- claimed_at: `2026-04-02T02:00:07Z`
- scope_id: `supervised_memory_codex_apr02_failure_reentry`
- progress_label: `supervised_memory_codex_apr02_failure_reentry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_failure_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_failure_reentry --notes "strengthen failure and recovery memory re-entry for repository/project coding decisions"
```

- hypothesis: `Failure and recovery artifacts are still under-reentering repository/project coding decisions, so a memory-heavy scoped run should reveal whether the kernel is reusing prior verifier/tool/episode evidence strongly enough on coding-focused tasks.`
- owned_paths:
  - `agent_kernel/episode_store.py`
  - `agent_kernel/memory.py`
  - `agent_kernel/extractors.py`
  - `agent_kernel/learning_compiler.py`
  - `agent_kernel/tolbert.py`
  - `tests/test_memory.py`
  - `tests/test_tolbert.py`
- released_at: `2026-04-02T02:16:13Z`
- result: `completed`
- findings:
  - `Patched the learning-memory path in agent_kernel/memory.py so EpisodeMemory now surfaces synthetic failure/recovery documents from compiled learning artifacts. Successful recovery cases now re-enter memory as curriculum-visible recovery documents, and negative command patterns re-enter memory as failure documents instead of remaining dead logs.`
  - `Patched agent_kernel/extractors.py so tooling extraction can promote successful recovery_case artifacts through source-task aliases such as parent_task/source_task instead of requiring the synthetic recovery task id itself to exist in the TaskBank.`
  - `Added focused regression coverage in tests/test_memory.py for both behaviors: failure/recovery learning documents now appear in EpisodeMemory.list_documents(), and a successful recovery_case can now emit a reusable tool candidate via its canonical source task.`
  - `The first fresh memory-heavy scoped run completed cleanly inside the bounded observation window and held the repository/project-focused primary pass at 5/5 success, but still routed to tooling/procedure_promotion, scheduled generated_failure with total=0 under the reduced one-task shape, and emitted an empty tool candidate artifact.`
  - `After the lane-local memory/extractor patch, rerunning the same scope changed the live outcome materially: the scope retried at 4/5 after timeout pressure, selected tooling/script_hardening, and emitted a non-empty tooling artifact with 64 candidates at trajectories/improvement/candidates/supervised_memory_codex_apr02_failure_reentry/tooling/cycle_tooling_20260402T021547106918Z_797e7c11_script_hardening/tool_candidates.json.`
  - `The remaining blocker is now wrapper-side rather than memory-side. On the rerun, generated_failure no longer reached the previous zero-task scheduled followup path; it was skipped entirely with skipped_reason=no separate curriculum budget configured after the reduced-slice retry, and the overall observe phase exceeded the nominal 60-second budget at 118.9s.`
- artifacts:
  - `docs/supervised_work_queue.md`
  - `agent_kernel/memory.py`
  - `agent_kernel/extractors.py`
  - `tests/test_memory.py`
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_failure_reentry.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_failure_reentry/tooling/cycle_tooling_20260402T020152214959Z_444e1a30_procedure_promotion/tool_candidates.json`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_failure_reentry/tooling/cycle_tooling_20260402T021547106918Z_797e7c11_script_hardening/tool_candidates.json`
- verification:
  - `python -m py_compile agent_kernel/memory.py agent_kernel/extractors.py tests/test_memory.py`
  - `pytest -q tests/test_memory.py -k 'includes_failure_recovery_learning_documents or uses_successful_recovery_case_via_source_task_alias or uses_learning_success_candidates or recurses_into_generated_phase_directories'`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_failure_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_failure_reentry --notes "strengthen failure and recovery memory re-entry for repository/project coding decisions"`
- next_recommendation: `The learning-memory lane no longer has the same flatten-to-zero extractor problem; the rerun proved that by emitting 64 tool candidates. The next ROI is outside this lane: fix the supervised wrapper/runtime path that drops generated_failure after a reduced-slice retry with no separate curriculum budget configured, then rerun this same scope to measure whether the repaired memory surface also restores nonzero failure-followup scheduling.`


### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__learning_memory`
- claimed_at: `2026-04-02T02:22:56Z`
- scope_id: `supervised_memory_codex_apr02_alias_reentry`
- progress_label: `supervised_memory_codex_apr02_alias_reentry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_alias_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_alias_reentry --notes "strengthen alias-aware learning-memory retrieval for repository/project coding decisions"
```

- hypothesis: `Recovery and negative-pattern artifacts still compile under synthetic or retrieval-specific task ids, so learning guidance under-matches future coding tasks unless the matcher resolves canonical source-task aliases like parent_task and task_metadata.source_task.`
- owned_paths:
  - `agent_kernel/memory.py`
  - `agent_kernel/extractors.py`
  - `agent_kernel/learning_compiler.py`
  - `agent_kernel/tolbert.py`
  - `tests/test_memory.py`
  - `tests/test_tolbert.py`
- released_at: `2026-04-02T02:33:58Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/learning_compiler.py so learning candidates now carry and match through canonical source-task aliases derived from parent_task, task_metadata.source_task, and task_contract metadata instead of relying only on exact source_task_id or prefilled applicable_tasks. Retrieval-style and recovery-style artifacts can now re-enter later matching even when they were compiled under synthetic or retrieval-specific task ids.`
  - `Patched agent_kernel/learning_compiler.py path resolution so an explicit learning-artifacts path under sqlite-backed configs no longer gets polluted by the global sqlite candidate store unless the path actually matches config.learning_artifacts_path.`
  - `Patched agent_kernel/tolbert.py so learned negative-command-pattern notes are prioritized ahead of generic retrieval avoidance notes when the avoidance budget is capped, preventing learned failure guidance from being trimmed out of the final control packet.`
  - `Added focused regressions in tests/test_memory.py and tests/test_tolbert.py covering alias-aware matching, explicit-path isolation under sqlite config, and TolBERT guidance recovery through source-task aliases.`
  - `The first live scope for supervised_memory_codex_apr02_alias_reentry was blocked by runtime readiness rather than memory logic: the observe pass retried without TolBERT context after startup failure, retried again at reduced 4/5 primary after timeout pressure, then closed with RuntimeError: TOLBERT service failed to become ready after 15.000 seconds. It emitted only an observation record and no candidate artifact.`
  - `A clean control rerun with AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 validated the lane-local memory changes without service variance. Scope supervised_memory_codex_apr02_alias_reentry_notolbert completed in 60.0967 seconds, executed generated_failure with a tiny 1.1-second/1-task budget, and selected retrieval/confidence_gating rather than falling back to the older tooling path. The emitted artifact is trajectories/improvement/candidates/supervised_memory_codex_apr02_alias_reentry_notolbert/retrieval/cycle_retrieval_20260402T023340403154Z_8bda24d9_confidence_gating/retrieval_proposals.json with one retrieval proposal explaining that low-confidence retrieval remains common relative to trusted retrieval usage.`
- artifacts:
  - `docs/supervised_work_queue.md`
  - `agent_kernel/learning_compiler.py`
  - `agent_kernel/tolbert.py`
  - `tests/test_memory.py`
  - `tests/test_tolbert.py`
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_alias_reentry.jsonl`
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_alias_reentry_notolbert.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_alias_reentry_notolbert/retrieval/cycle_retrieval_20260402T023340403154Z_8bda24d9_confidence_gating/retrieval_proposals.json`
- verification:
  - `python -m py_compile agent_kernel/learning_compiler.py agent_kernel/tolbert.py tests/test_memory.py tests/test_tolbert.py`
  - `pytest -q tests/test_memory.py -k 'respects_explicit_path_under_sqlite_config or match_source_task_aliases_from_metadata or match_replay_task_lineage or includes_failure_recovery_learning_documents'`
  - `pytest -q tests/test_tolbert.py -k 'matches_learning_guidance_via_source_task_alias or preserves_adjacent_success_task_contract_while_merging_learning_guidance'`
  - `python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_alias_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_alias_reentry --notes "strengthen alias-aware learning-memory retrieval for repository/project coding decisions"`
  - `AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_alias_reentry_notolbert --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_alias_reentry_notolbert --notes "validate alias-aware learning-memory retrieval for repository/project coding decisions without tolbert startup variance"`
- next_recommendation: `The memory lane now has stronger alias-aware reuse and cleaner path isolation. The next ROI inside this lane is propagating the same canonical alias preference into other memory consumers beyond TolBERT matching, while the separate blocker outside this lane remains TOLBERT service readiness on live supervised scopes.`


### `supervised_runner__open`

- runner_slot: `supervised_runner__open`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__learning_memory`
- claimed_at: `2026-04-02T02:36:30Z`
- scope_id: `supervised_memory_codex_apr02_recovery_skill_reentry`
- progress_label: `supervised_memory_codex_apr02_recovery_skill_reentry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_recovery_skill_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_recovery_skill_reentry --notes "promote successful recovery-case memory into reusable skill replay for repository/project coding decisions"
```

- hypothesis: `Recovery procedures now re-enter tool extraction and TolBERT guidance, but they still bypass skill-memory promotion. If successful recovery_case artifacts can promote into canonical postrun skills via source-task aliases, the skill-memory lane should reuse more failure-repair procedures and shift bounded live scopes toward stronger retrieval/skill selection without TolBERT variance.`
- owned_paths:
  - `agent_kernel/extractors.py`
  - `tests/test_memory.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T02:39:41Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/extractors.py so skill extraction now promotes successful recovery_case artifacts through canonical source-task aliases instead of only accepting success_skill_candidate records with exact source_task_id matches. Recovery procedures can now feed the skill-memory replay path the same way they already fed tool extraction and TolBERT guidance.`
  - `Added focused regression coverage in tests/test_memory.py proving extract_successful_command_skills(...) emits a canonical postrun skill from a successful recovery_case that only resolves through parent_task/task_metadata.source_task.`
  - `A fresh bounded no-TolBERT scope validated the broader memory path live. Scope supervised_memory_codex_apr02_recovery_skill_reentry completed cleanly under the widened shared-repo 80-second observation budget, executed generated_failure with a real 18.0-second/1-task supplemental window, and selected retrieval/confidence_gating again instead of collapsing back to the earlier empty tooling path.`
  - `The live retrieval artifact remained a single confidence-focused proposal rather than a skill artifact, so the memory lane is now re-entering recovery knowledge more broadly but the top bounded repository/project pressure is still retrieval-confidence quality rather than missing skill extraction.`
- artifacts:
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
  - `agent_kernel/extractors.py`
  - `tests/test_memory.py`
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_recovery_skill_reentry.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_recovery_skill_reentry/retrieval/cycle_retrieval_20260402T023900435453Z_7d402ffb_confidence_gating/retrieval_proposals.json`
- verification:
  - `python -m py_compile agent_kernel/extractors.py tests/test_memory.py`
  - `pytest -q tests/test_memory.py -k 'skill_extractor_includes_postrun_learning_candidates or skill_extractor_uses_successful_recovery_case_via_source_task_alias or tool_candidate_extractor_uses_successful_recovery_case_via_source_task_alias'`
  - `AGENT_KERNEL_USE_TOLBERT_CONTEXT=0 python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_recovery_skill_reentry --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_recovery_skill_reentry --notes "promote successful recovery-case memory into reusable skill replay for repository/project coding decisions"`
- next_recommendation: `The next ROI inside learning_memory is no longer basic recovery-case visibility. The lane now reaches tools, TolBERT guidance, and skill extraction. The best next step is either improving retrieval-confidence learning itself or threading the same canonical alias preference into any remaining memory consumers that still rank only exact source_task_id matches.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T17:40:52Z`
- scope_id: `workspace_preview_recovered_alias_frontier_extension`
- progress_label: `workspace_preview_recovered_alias_frontier_extension`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'extend_recovered_conflicting_alias_block_with_adjacent_exact_region or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset or action_generation_candidates_prefer_multi_block_overlap_repair_over_single_island_winner'
```

- hypothesis: `When a recovered conflicting-alias overlap block is already exact and an adjacent retained exact region touches it contiguously, the decoder should promote one larger bounded block_replace instead of keeping the weaker hybrid multi_edit frontier.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T17:40:52Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so exact block_replace frontier candidates can now merge across contiguous exact block regions, including the case where one side is a recovered conflicting-alias overlap block and the other is an adjacent exact retained region.`
  - `The merged frontier candidate carries one exact contiguous bounded block_replace instead of forcing the decoder to keep the older hybrid multi_edit bundle as the best localized repair.`
  - `Adjusted recovered-conflicting-alias scoring so an exact contiguous recovered block no longer keeps the old blanket penalty that was appropriate for weaker recovered blocks but not for the new safer frontier-extended block.`
  - `Added tests/test_policy.py coverage proving both proposal emission and ranking: the decoder now emits the larger [0,1,2,3] block_replace and ranks it above the hybrid multi_edit bundle while keeping nearby overlap and hidden-gap ranking tests green.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'extend_recovered_conflicting_alias_block_with_adjacent_exact_region or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or propagate_recovered_conflicting_alias_into_hybrid_bundle'`
  - `pytest -q tests/test_policy.py -k 'extend_recovered_conflicting_alias_block_with_adjacent_exact_region or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset or action_generation_candidates_prefer_multi_block_overlap_repair_over_single_island_winner'`
- next_recommendation: `The next ROI on the coding decoder lane is the remaining single-line exact-neighbor case: promote adjacent exact token-edit windows into the same frontier-extended bounded block path when they form one contiguous exact region beside a recovered alias block.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T18:08:00Z`
- scope_id: `workspace_preview_recovered_alias_single_line_neighbor_extension`
- progress_label: `workspace_preview_recovered_alias_single_line_neighbor_extension`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'single_line_exact_neighbor or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or extend_recovered_conflicting_alias_block_with_adjacent_exact_region' --cache-clear && pytest -q tests/test_policy.py -k 'single_line_exact_neighbor or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or extend_recovered_conflicting_alias_block_with_adjacent_exact_region or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset' --cache-clear
```

- hypothesis: `The remaining decoder gap is a single exact neighbor that never emits its own block_replace candidate. If the frontier combiner can promote that safe exact single-line replacement into a block fragment, a recovered conflicting-alias block should be able to absorb it into one larger bounded block_replace instead of stopping at a hybrid multi_edit bundle.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T18:08:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so the frontier block composer no longer requires every exact component to already be a block_replace. Safe exact single-line replacements with zero line delta now promote into synthetic block fragments for frontier merging.`
  - `That closes the residual recovered-alias case where a single exact token-edit neighbor could not be absorbed. The decoder now emits a single [0,1,2] bounded block_replace for recovered-alias-plus-single-neighbor previews instead of stopping at the hybrid multi_edit frontier.`
  - `Threaded synthetic_single_line_block_replace_count through proposal metadata so ranking tests can assert the new path cleanly at both proposal-emission and decode-candidate layers.`
  - `Focused and broader decoder slices stayed green, including the nearby exact contiguous region, hidden-gap region, localized overlap, and exact-overlap frontier tests.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'single_line_exact_neighbor or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or extend_recovered_conflicting_alias_block_with_adjacent_exact_region' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'single_line_exact_neighbor or extended_recovered_alias_block_over_hybrid_bundle or exact_contiguous_region_block_replace_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit or extend_recovered_conflicting_alias_block_with_adjacent_exact_region or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset' --cache-clear`
- next_recommendation: `The next decoder ROI is the remaining shared-anchor exact class: exact insert/replace or delete/replace neighbor pairs with fully known target spans still stay as multi_edit. If those can be proven safe to collapse into one bounded block, the coding path gets another step closer to preferring single localized repairs over fragmented edit schedules.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T18:20:11Z`
- scope_id: `workspace_preview_shared_anchor_delete_block_extension`
- progress_label: `workspace_preview_shared_anchor_delete_block_extension`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'adjacent_shared_anchor_deletes or shared_anchor_delete_block_replace_over_multi_edit or adjacent_delete_and_replace or same_anchor_insert_and_replace or hidden_gap_region_block_replace_over_multi_edit' --cache-clear && pytest -q tests/test_policy.py -k 'shared_anchor_delete_block_replace_over_multi_edit or adjacent_shared_anchor_deletes or same_anchor_insert_and_replace or adjacent_delete_and_replace or exact_hidden_gap_region_block_replace_over_multi_edit or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset or extended_recovered_alias_block_over_hybrid_bundle or single_line_exact_neighbor' --cache-clear
```

- hypothesis: `The remaining decoder gap on the shared-anchor lane is exact delete fragmentation. If shared-anchor exact-region recovery stops depending on full target snapshots and can synthesize target regions directly from exact preview windows, adjacent exact deletes should collapse into one bounded block_replace instead of staying as a two-step multi_edit bundle.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T18:20:11Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so shared-anchor exact-region safety and recovery now synthesize merged target regions from the exact preview windows themselves instead of requiring full_target_content for every case. That preserved the existing same-anchor insert/replace path while widening the safe exact-region lane to adjacent delete/delete pairs.`
  - `Extended the shared-anchor pair classifier to recognize adjacent exact delete chains, and taught the exact-region block builder to materialize delete-only bounded blocks without double-applying preview offsets. The decoder now emits a single shared-anchor block_replace for adjacent deletes with command sed -i 2,3d config.txt instead of only the fragmented delete bundle.`
  - `Added regression coverage in tests/test_policy.py at both layers: proposal emission now checks the new shared-anchor delete block, and decode candidate selection now proves that the bounded delete block outranks the old multi_edit bundle while nearby same-anchor insert/replace, adjacent delete/replace, hidden-gap, overlap, and single-line-neighbor slices stay green.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'adjacent_shared_anchor_deletes or shared_anchor_delete_block_replace_over_multi_edit or adjacent_delete_and_replace or same_anchor_insert_and_replace or hidden_gap_region_block_replace_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'shared_anchor_delete_block_replace_over_multi_edit or adjacent_shared_anchor_deletes or same_anchor_insert_and_replace or adjacent_delete_and_replace or exact_hidden_gap_region_block_replace_over_multi_edit or localized_block_replace_over_partial_bundle_for_overlap or exact_overlap_block_subset_over_mixed_superset or extended_recovered_alias_block_over_hybrid_bundle or single_line_exact_neighbor' --cache-clear`
- next_recommendation: `The next coding ROI on this decoder lane is the remaining anchor-local fragmentation beyond the simple delete pair: mixed insert/delete shared-anchor cases and longer exact delete chains still have room to collapse into one localized bounded block instead of surviving as multi_edit schedules.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T18:44:30Z`
- scope_id: `workspace_preview_shared_anchor_mixed_insert_delete_block_extension`
- progress_label: `workspace_preview_shared_anchor_mixed_insert_delete_block_extension`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'shared_anchor_delete_block_replace_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit or same_anchor_insert_and_delete or adjacent_delete_and_insert or shared_anchor_insert_delete_block_over_multi_edit or same_anchor_insert_replace_block_with_exact_neighbor_over_multi_edit or adjacent_delete_and_replace or adjacent_shared_anchor_deletes' --cache-clear
```

- hypothesis: `The remaining anchor-local decoder gap is mixed exact insert/delete recovery. If shared-anchor exact-region synthesis can use the fully known target file and recovered shared-anchor regions stay on the preview-region block path, same-anchor insert/delete and adjacent delete/insert pairs should collapse into one bounded block_replace instead of fragmenting back into line edits or multi_edit bundles.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T18:44:30Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so shared-anchor exact-region safety now distinguishes mixed-sign pair shapes and requires known full_target_content for the new mixed insert/delete variants. Exact target-region synthesis now uses the known target file to recover the final localized region for shared-anchor mixed cases instead of relying only on overlapping preview fragments.`
  - `Recovered shared-anchor exact regions now route through the preview-region block builder, not the smaller minimal-diff block derivation. That preserves a bounded local block even when the mixed case could otherwise collapse into a one-line replace and drop out of the shared-anchor block frontier.`
  - `The decoder now emits bounded shared-anchor block_replace commands for same-anchor insert/delete and adjacent delete/insert cases, and regression coverage now also locks in the longer three-delete shared-anchor chain at the decode-candidate layer.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'shared_anchor_delete_block_replace_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit or same_anchor_insert_and_delete or adjacent_delete_and_insert or shared_anchor_insert_delete_block_over_multi_edit or same_anchor_insert_replace_block_with_exact_neighbor_over_multi_edit or adjacent_delete_and_replace or adjacent_shared_anchor_deletes' --cache-clear`
- next_recommendation: `The next decoder ROI is the mixed anchor-local chain beyond these pairs: delete chains that terminate in insert/replace, and shared-anchor mixed subsets that should absorb one adjacent exact neighbor into the same localized bounded block.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:05:40Z`
- scope_id: `workspace_preview_shared_anchor_mixed_chain_canonicalization`
- progress_label: `workspace_preview_shared_anchor_mixed_chain_canonicalization`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'delete_chain_ending_in_replace or delete_chain_ending_in_insert or same_anchor_insert_delete_block_with_exact_neighbor or delete_chain_replace_block_over_multi_edit or delete_chain_insert_block_over_multi_edit or shared_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit or same_anchor_insert_delete_block_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or shared_anchor_delete_block_replace_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit' --cache-clear
```

- hypothesis: `The next decoder gap is no longer whether mixed exact chains can be formed; it is whether they stay canonical. If frontier block composition stops re-emitting signatures that exact-region recovery already produced, then delete-chain replace/insert blocks and same-anchor insert/delete plus exact-neighbor blocks should survive as one canonical localized repair instead of duplicate competing block variants.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:05:40Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so frontier block composition seeds its duplicate-signature filter from the block proposals already emitted by earlier exact-region passes. That removes redundant generic frontier block_replace variants when the same command/window signature already exists as the canonical shared-anchor exact-region candidate.`
  - `Locked the next mixed-chain decoder tier in tests/test_policy.py: delete chains ending in replace, delete chains ending in insert, and same-anchor insert/delete plus one adjacent exact neighbor now all emit exactly one shared-anchor [0,1,2] bounded block at the proposal layer and beat the equivalent fragmented multi_edit schedule at decode time.`
  - `The coding path is narrower and cleaner after this pass: these anchor-local mixed chains are no longer just possible, they are canonicalized into a single bounded repair candidate, which reduces noisy duplicate competition in the decoder frontier.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'delete_chain_ending_in_replace or delete_chain_ending_in_insert or same_anchor_insert_delete_block_with_exact_neighbor or delete_chain_replace_block_over_multi_edit or delete_chain_insert_block_over_multi_edit or shared_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit or same_anchor_insert_delete_block_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or shared_anchor_delete_block_replace_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit' --cache-clear`
- next_recommendation: `The next decoder ROI is the remaining higher-order mixed-sign frontier: longer shared-anchor runs that should absorb more than one adjacent exact neighbor, and any exact-region families where the canonical bounded block still drops metadata fidelity relative to the fragmented fallback path.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:17:11Z`
- scope_id: `workspace_preview_shared_anchor_longer_chain_score_fidelity`
- progress_label: `workspace_preview_shared_anchor_longer_chain_score_fidelity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or delete_chain_ending_in_insert or delete_chain_ending_in_replace or adjacent_delete_insert_block_over_multi_edit or shared_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit' --cache-clear
```

- hypothesis: `The next decoder gap is no longer exact-chain synthesis. It is score fidelity: if longer mixed-sign shared-anchor blocks explicitly track absorbed exact neighbors, the kernel can reward a 5-window safe localized block over the same 4-window chain instead of flattening them to the same score.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:17:11Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so shared-anchor exact-region proposals now carry shared_anchor_exact_neighbor_count, which counts exact retained neighbors absorbed outside the core shared-anchor overlap instead of losing that detail once the bounded block is formed.`
  - `Adjusted exact-region scoring so longer mixed-sign shared-anchor blocks get additional bounded credit for each absorbed exact neighbor. That fixes the live score-flattening bug where a 5-window mixed chain could tie the corresponding 4-window chain despite covering more exact local work safely.`
  - `Added focused tests/test_policy.py coverage for both layers: proposal emission now proves the 5-window mixed-sign chain reports shared_anchor_exact_neighbor_count=2, and decode-time scoring now proves the 5-window bounded block outranks the corresponding 4-window variant instead of tying it.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or delete_chain_ending_in_insert or delete_chain_ending_in_replace or adjacent_delete_insert_block_over_multi_edit or shared_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'same_anchor_insert_delete_block_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or delete_chain_insert_block_over_multi_edit or delete_chain_replace_block_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit or exact_hidden_gap_region_block_replace_over_multi_edit' --cache-clear`
- next_recommendation: `The next decoder ROI is no longer exact mixed-chain score fidelity. Move to the higher-order frontier where multiple shared-anchor cores, bridged hidden-gap proofs, or ambiguous overlap recoveries compete for one canonical bounded repair inside the same file.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:44:00Z`
- scope_id: `workspace_preview_frontier_shared_anchor_metadata_fidelity`
- progress_label: `workspace_preview_frontier_shared_anchor_metadata_fidelity`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or frontier_extends_mixed_core_with_multiple_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or same_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or delete_chain_insert_block_over_multi_edit or delete_chain_replace_block_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit' --cache-clear
```

- hypothesis: `The next decoder gap is metadata fidelity on the canonical frontier path. If a merged frontier block keeps the shared-anchor lineage from the exact core it absorbed, then longer mixed-sign repairs can stay both canonical and richly scored instead of degrading back to generic block_replace metadata once dedup keeps the frontier winner.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:44:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so frontier-composed exact contiguous block_replace candidates now aggregate shared-anchor metadata from the exact-region core they absorb. The canonical merged block now preserves shared_anchor_exact_region_block, shared_anchor_pair_kinds, shared_anchor_exact_neighbor_count, and shared_anchor_mixed_insert_delete instead of flattening those fields away.`
  - `The new aggregation also counts non-core absorbed exact neighbors on the frontier path, so a shared-anchor insert/delete core extended by three exact neighbors keeps shared_anchor_exact_neighbor_count=3 and synthetic_single_line_block_replace_count=3 on the final canonical bounded repair.`
  - `Added focused regression coverage in tests/test_policy.py that forces the frontier path directly: the proposal layer now proves a same-anchor insert/delete core can be extended by three exact neighbors without losing metadata, and the decode layer proves that richer frontier block still outranks the equivalent five-window multi_edit fallback.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'frontier_extends_mixed_core_with_multiple_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or longer_mixed_chain_with_more_exact_neighbors or same_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit or delete_chain_insert_block_over_multi_edit or delete_chain_replace_block_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'shared_anchor_exact_neighbor_count or longer_mixed_chain_with_more_exact_neighbors or frontier_extends_mixed_core_with_multiple_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or same_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or delete_chain_insert_block_over_multi_edit or delete_chain_replace_block_over_multi_edit or longer_shared_anchor_delete_chain_block_over_multi_edit' --cache-clear`
- next_recommendation: `The next decoder ROI is the hybrid higher-order frontier: preserve this same metadata and bounded-block score advantage when a shared-anchor exact core merges with recovered overlap/alias blocks or exact hidden-gap proof blocks, not just plain exact single-line neighbors.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:44:34Z`
- scope_id: `workspace_preview_shared_anchor_hybrid_frontier_scoring`
- progress_label: `workspace_preview_shared_anchor_hybrid_frontier_scoring`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit or conflicting_alias_recovery_bundle or bridged_hidden_gap_region_block_task or bridge_run_with_omitted_proof_task' --cache-clear
```

- hypothesis: `The next decoder gap is no longer plain exact-neighbor metadata. It is single-hybrid frontier fidelity: if a shared-anchor exact core keeps its lineage when it merges with one overlap/alias or hidden-gap proof block, and the recovered overlap hybrid gets explicit bounded-block credit, the decoder should keep preferring one localized repair instead of fragmenting back to multi_edit.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:44:34Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so frontier-composed shared-anchor metadata no longer disappears when a bridged or exact hidden-gap proof block joins the merge. The frontier path now carries shared_anchor_hybrid_component_count alongside the existing shared-anchor lineage fields.`
  - `Retuned conflicting-alias recovery scoring so a fully recovered shared-anchor overlap hybrid block_replace is treated as the canonical bounded repair rather than as a generic risky alias block. That restores the bounded localized path over the equivalent fragmented multi_edit schedule on this higher-order lane.`
  - `Added focused tests/test_policy.py coverage at both layers: one frontier test proves a shared-anchor core plus bridged hidden-gap block preserves shared-anchor metadata, and one decode-time test proves a shared-anchor overlap hybrid block_replace outranks the corresponding multi_edit fallback.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit' --cache-clear`
  - `pytest -q tests/test_policy.py -k 'same_anchor_insert_delete_block_over_multi_edit or adjacent_delete_insert_block_over_multi_edit or frontier_extends_mixed_core_with_multiple_exact_neighbors or preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors or conflicting_alias_recovery_bundle or all_inexact_conflicting_alias_fallback or hidden_gap_region_block_replace_over_multi_edit or bridged_hidden_gap_region_block_task or bridge_run_with_omitted_proof_task or frontier_bridge_hybrid_preserves_shared_anchor_metadata or shared_anchor_overlap_hybrid_block_over_multi_edit' --cache-clear`
- next_recommendation: `The next decoder ROI is multi-hybrid canonicalization inside one file: shared-anchor cores that need to absorb both overlap and hidden-gap components at once, or multiple shared-anchor cores that compete for the same canonical bounded repair without losing provenance or score fidelity.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:31:25Z`
- scope_id: `supervised_memory_codex_apr02_confidencegating_activationcarry_v3`
- progress_label: `supervised_memory_codex_apr02_confidencegating_activationcarry_v3`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_confidencegating_activationcarry_v3 --subsystem retrieval --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_confidencegating_activationcarry_v3 --notes "backfill partial task outcomes so retrieval activation-gap bootstrap stays widened when coding runs select retrieval more often than they trust it"
```

- hypothesis: `When repository/project coding runs keep selecting retrieval without converting it into trusted execution, retrieval/confidence_gating should stay in a stronger bootstrap-activation profile instead of de-escalating after the first trusted hit; the guided wrapper must preserve that per-task activation-gap evidence into candidate generation.`
- owned_paths:
  - `agent_kernel/retrieval_improvement.py`
  - `scripts/run_human_guided_improvement_cycle.py`
  - `tests/test_improvement.py`
  - `tests/test_protocol_comparison.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:31:25Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/retrieval_improvement.py so confidence_gating now distinguishes successful runs that merely select retrieval from runs that actually convert retrieval into trusted/influential execution. Activation-gap runs now stay on a stronger bootstrap profile with lower confidence gates and wider routing/context limits instead of falling back to the weaker generic bootstrap path.`
  - `Patched scripts/run_human_guided_improvement_cycle.py so the parent process backfills metrics.task_outcomes from observation partial summaries before candidate generation. That makes live retrieval proposal generation see per-task retrieval_selected_steps / retrieval_influenced_steps even when the child-process metrics object drops them.`
  - `Fresh live scope trajectories/improvement/cycles_supervised_memory_codex_apr02_confidencegating_activationcarry_v3.jsonl stayed healthy at 5/5 primary success with the real coding-lane activation-gap signature intact: retrieval_selected_steps=4, retrieval_influenced_steps=1, trusted_retrieval_steps=1.`
  - `The emitted retrieval candidate now shows the intended stronger activation-gap profile in trajectories/improvement/candidates/supervised_memory_codex_apr02_confidencegating_activationcarry_v3/retrieval/cycle_retrieval_20260402T193020766549Z_61326e87_confidence_gating/retrieval_proposals.json: confidence_threshold=0.22, skill_ranking_min_confidence=0.4, branch_results=4, context_max_chunks=7, top_branches=6, ancestor_branch_levels=5, and max_episode_step_spans_per_task=3.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_confidencegating_activationcarry_v3.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_confidencegating_activationcarry_v3/retrieval/cycle_retrieval_20260402T193020766549Z_61326e87_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile agent_kernel/retrieval_improvement.py scripts/run_human_guided_improvement_cycle.py tests/test_improvement.py tests/test_protocol_comparison.py`
  - `pytest -q tests/test_improvement.py -k "retrieval_confidence_gating_hardens_from_stalled_untrusted_tasks or retrieval_confidence_gating_bootstraps_from_success_without_guidance or retrieval_confidence_gating_bootstraps_after_successful_blind_high_low_confidence_run or retrieval_confidence_gating_keeps_bootstrap_profile_when_only_one_task_uses_retrieval or retrieval_confidence_gating_keeps_activation_bootstrap_when_selection_outpaces_influence"`
  - `pytest -q tests/test_protocol_comparison.py -k "backfills_task_outcomes_from_partial_summary"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `The next retrieval ROI on the coding lane is runtime conversion, not wider routing: increase repository/project retrieval_influenced_steps above 1 by improving how selected retrieval guidance survives into actual trusted execution on the three still-blind repository tasks.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T19:54:17Z`
- scope_id: `supervised_memory_codex_apr02_inferencefallback_retrievalcarry`
- progress_label: `supervised_memory_codex_apr02_inferencefallback_retrievalcarry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_inferencefallback_retrievalcarry --subsystem retrieval --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_inferencefallback_retrievalcarry --notes "preserve retrieval guidance through inference-failure fallback so repository/project coding tasks convert selected retrieval into retrieval-influenced execution"
```

- hypothesis: `When repository/project coding tasks hit inference failure after TolBERT has already compiled usable retrieval guidance, deterministic fallback should keep those retrieval spans alive instead of recomputing a blind fallback; that should convert the three still-blind repository tasks from retrieval_selected to retrieval_influenced execution.`
- owned_paths:
  - `agent_kernel/policy.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T19:54:17Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/policy.py so inference-failure fallback now preserves the active retrieval guidance and passes its recommended commands back into deterministic candidate selection instead of ranking against an empty retrieval surface.`
  - `Added tests/test_policy.py coverage proving that inference-failure fallback on a repository-style task now returns a retrieval-guided code_execute decision with selected_retrieval_span_id preserved.`
  - `Fresh live scope trajectories/improvement/cycles_supervised_memory_codex_apr02_inferencefallback_retrievalcarry.jsonl completed healthy at 4/5 primary success with the targeted runtime-conversion gain: retrieval_selected_steps stayed at 4, but retrieval_influenced_steps jumped from 1 to 4 while trusted_retrieval_steps stayed at 1.`
  - `The three repository tasks that had previously succeeded through blind inference-failure recovery now each carry learned retrieval spans and influenced execution: service_release_task -> learning:success_skill:service_release_task_discovered, schema_alignment_task -> learning:success_skill:schema_alignment_task, repo_sync_matrix_task -> learning:success_skill:repo_sync_matrix_task.`
  - `The remaining miss is now concentrated in the guarded project path: git_generated_conflict_resolution_task regressed to policy_terminated with no retrieval uptake, so the next cycle should repair that project runtime path without undoing the new repository fallback conversion.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_inferencefallback_retrievalcarry.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_inferencefallback_retrievalcarry/retrieval/cycle_retrieval_20260402T195417863009Z_455dde4c_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile agent_kernel/policy.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k "inference_failure_fallback_preserves_retrieval_guidance or keeps_trusted_workflow_guidance_under_low_confidence_for_screened_tolbert_candidates or executes_trusted_workflow_guidance_on_low_confidence_first_step or planner_role_prefers_subgoal_aligned_direct_command"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_inferencefallback_retrievalcarry --subsystem retrieval --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_inferencefallback_retrievalcarry --notes "preserve retrieval guidance through inference-failure fallback so repository/project coding tasks convert selected retrieval into retrieval-influenced execution"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Stay on the same coding-retrieval lane, but switch from repository inference-failure fallback to the remaining guarded project path: convert git_generated_conflict_resolution_task from policy_terminated/no_state_progress into retrieval-influenced execution without losing the new repo fallback gains.`


### `supervised_runner__codex_main`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T20:06:31Z`
- scope_id: `supervised_memory_codex_apr02_conflictbaselinefallback`
- progress_label: `supervised_memory_codex_apr02_conflictbaselinefallback`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_conflictbaselinefallback --subsystem retrieval --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_conflictbaselinefallback --notes "resolve missing shared-repo baseline refs in git_repo_review so guarded project conflict tasks verify successfully while keeping repo retrieval fallback conversion"
```

- hypothesis: `The remaining guarded project miss is not another retrieval-selection problem. The live workspace already contains the correct merged/generated outputs, so the failure is likely that git_repo_review loses its semantic diff base when the shared-repo clone lacks the named baseline ref. If verifier base resolution falls back to the repo root commit in that case, the project task should return to success without giving back the new repository retrieval-influenced fallback gains.`
- owned_paths:
  - `agent_kernel/verifier.py`
  - `tests/test_verifier.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T20:06:31Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/verifier.py so git_repo_review resolves a missing diff_base_ref=baseline to the repository root commit instead of treating the semantic diff as empty. That preserves the intended baseline semantics for shared-repo clones whose named baseline ref is absent.`
  - `Added tests/test_verifier.py coverage proving the generated-conflict workflow still fails when the conflict remains unresolved, but now also passes when the baseline tag is explicitly deleted before the accepted merge is verified.`
  - `Fresh live scope trajectories/improvement/cycles_supervised_memory_codex_apr02_conflictbaselinefallback.jsonl regained 5/5 primary success in 49.4121s with observation_budget_exceeded=false.`
  - `The repo-side runtime conversion gain held: retrieval_selected_steps=4, retrieval_influenced_steps=4, trusted_retrieval_steps=1. The three repository tasks still execute through learned retrieval spans after inference failure, and git_generated_conflict_resolution_task returned to success instead of policy_terminated.`
  - `The guarded project integrator still is not retrieval-influenced itself; success currently comes from the worker retrieval plus restored semantic verification. The next ROI is deeper project-side retrieval uptake, not more baseline/verifier repair.`
- artifacts:
  - `trajectories/improvement/cycles_supervised_memory_codex_apr02_conflictbaselinefallback.jsonl`
  - `trajectories/improvement/candidates/supervised_memory_codex_apr02_conflictbaselinefallback/retrieval/cycle_retrieval_20260402T200631421712Z_1492fcff_confidence_gating/retrieval_proposals.json`
  - `trajectories/improvement/reports/supervised_parallel_frontier.json`
  - `trajectories/improvement/reports/supervised_frontier_promotion_plan.json`
- verification:
  - `python -m py_compile agent_kernel/verifier.py tests/test_verifier.py`
  - `pytest -q tests/test_verifier.py -k "generated_conflict_repo_workflow"`
  - `pytest -q tests/test_loop.py -k "generated_conflict_task_when_git_and_generated_policy_enabled or segmented_direct_path_for_generated_conflict_integrator_after_inference_failure"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/run_human_guided_improvement_cycle.py --provider vllm --task-limit 5 --max-observation-seconds 60 --generate-only --scope-id supervised_memory_codex_apr02_conflictbaselinefallback --subsystem retrieval --priority-benchmark-family repository --priority-benchmark-family project --include-episode-memory --include-skill-memory --include-tool-memory --include-verifier-memory --include-failure-curriculum --progress-label supervised_memory_codex_apr02_conflictbaselinefallback --notes "resolve missing shared-repo baseline refs in git_repo_review so guarded project conflict tasks verify successfully while keeping repo retrieval fallback conversion"`
  - `python scripts/report_supervised_frontier.py`
  - `python scripts/report_frontier_promotion_plan.py --frontier-report trajectories/improvement/reports/supervised_parallel_frontier.json`
- next_recommendation: `Stay on the guarded project/runtime lane, but now target true project-side retrieval uptake: get git_generated_conflict_resolution_task itself to select and execute a trusted retrieval/procedure span, rather than succeeding only via the worker retrieval plus repaired semantic verification.`


### `workspace_preview_partial_current_proof_regions`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T20:24:00Z`
- scope_id: `workspace_preview_partial_current_proof_regions`
- progress_label: `workspace_preview_partial_current_proof_regions`
- status: `completed`
- command:

```bash
cd /data/agentkernel && pytest -q tests/test_loop.py -k 'partial_hidden_gap_current_proof_regions or omitted_exact_proof_windows' && pytest -q tests/test_policy.py -k 'partial_current_proof_region or multi_span_current_proof_region or proof_only_current_bridge'
```

- hypothesis: `When a workspace preview proves several disjoint hidden-current spans inside the correct local block but not every hidden line, the world model should preserve that partial evidence explicitly and the decoder should choose bounded expected-content fallback instead of fabricating an exact region block.`
- owned_paths:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T20:24:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/world_model.py so hidden-gap current-proof regions now record whether proof is complete or partial, plus covered/missing hidden-current line counts and missing-span counts.`
  - `Patched agent_kernel/modeling/policy/decoder.py so partial current-proof regions are resolved as first-class preview metadata, exact current_proof_region block synthesis is suppressed for those regions, and fallback/risk metadata carries the partial-proof signal through candidate scoring.`
  - `Added tests/test_loop.py coverage for world-model partial current-proof region accounting, and tests/test_policy.py coverage proving expected_file_content now outranks structured-edit proposals when only partial hidden-current region proof exists.`
- artifacts:
  - `agent_kernel/world_model.py`
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_loop.py`
  - `tests/test_policy.py`
- verification:
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
  - `pytest -q tests/test_loop.py -k 'partial_hidden_gap_current_proof_regions or omitted_exact_proof_windows'`
  - `pytest -q tests/test_policy.py -k 'partial_current_proof_region or multi_span_current_proof_region or proof_only_current_bridge'`
- next_recommendation: `The next upstream ROI is richer partial-region evidence itself: represent multiple partially proven subspans inside a bounded local block without depending on consecutive bridge segments, then let the decoder consume that richer proof map.`

- findings_update:
  - `Extended agent_kernel/world_model.py so hidden-gap current-proof regions can also be synthesized from sparse adjacent-pair proof evidence across zero-gap joins; a bounded local block no longer needs one consecutive bridge-run chain to surface multi-span current proof.`
  - `Added tests/test_loop.py coverage for sparse region synthesis over window indices [0,1,2,3,4] with proof spans separated by zero-gap joins, and tests/test_policy.py coverage proving decode now prefers an exact bounded block_replace from that sparse current-proof region without any bridge-run payload.`
- findings_update:
  - `Extended agent_kernel/world_model.py so partial current-proof regions now carry explicit current_proof_opaque_spans, including sparse regions where the opaque subspan is caused by missing adjacent-pair bridge evidence rather than a simple incomplete bridge payload.`
  - `Extended agent_kernel/modeling/policy/decoder.py so those opaque subspans survive preview resolution and risk annotation as hidden_gap_current_partial_proof_opaque_spans instead of collapsing immediately to aggregate missing counts.`
- verification_update:
  - `python -m py_compile agent_kernel/world_model.py tests/test_loop.py tests/test_policy.py`
  - `pytest -q tests/test_loop.py -k 'sparse_current_proof_region or partial_hidden_gap_current_proof_regions'`
  - `pytest -q tests/test_policy.py -k 'sparse_current_proof_region or multi_span_current_proof_region_without_bridge_chain'`
- verification_update:
  - `python -m py_compile agent_kernel/world_model.py agent_kernel/modeling/policy/decoder.py tests/test_loop.py tests/test_policy.py`
  - `python - <<'PY' ... WorldModel._hidden_gap_current_proof_regions(...) ... PY`
  - `python - <<'PY' ... WorldModel._sparse_hidden_gap_current_proof_regions(...) ... PY`
  - `python - <<'PY' ... decoder._resolve_workspace_preview_current_proof_region(...) / decoder._workspace_preview_risk_summary(...) ... PY`
  - `local compile sanity is green again, including python -m py_compile agent_kernel/task_bank.py`
- next_recommendation_update: `The next upstream ROI is evidence that is not expressible as adjacent-pair bridges at all: multiple partial subspans inferred inside one bounded local block where proof spans are sparse and some hidden lines remain unknown even after zero-gap closure.`
- findings_update:
  - `Extended agent_kernel/modeling/policy/decoder.py so a partial current-proof region can still synthesize a localized block_replace when exactly one opaque span is strictly interior to an otherwise exact bounded region. The opaque span now contributes line-span coverage for region assembly without pretending the current content itself is known.`
  - `Strengthened same-span proof-quality scoring for that path: partial current-proof bounded rewrites now preserve current_proof_region metadata, outrank generic same-span bounded competitors when the only opaque gap is internal, and still leave allow_expected_write_fallback enabled as a secondary safe option.`
  - `Boundary-touch partial regions remain conservative: if the opaque span reaches the bounded region edge, current_proof_region_block synthesis is still suppressed and expected_file_content remains the preferred action.`
- verification_update:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'prefer_localized_block_when_partial_current_proof_opaque_span_is_internal or keep_expected_write_when_partial_current_proof_opaque_span_touches_boundary or prefer_expected_write_when_current_proof_region_is_only_partial'`
  - `pytest -q tests/test_policy.py -k 'prefer_localized_block_when_partial_current_proof_opaque_span_is_internal or keep_expected_write_when_partial_current_proof_opaque_span_touches_boundary or prefer_expected_write_when_current_proof_region_is_only_partial or current_hidden_gap_requires_proof or proof_only_current_bridge or multi_span_current_proof_region'`
- next_recommendation_update: `The next ROI on this lane is no longer a single interior opaque span. It is multi-opaque partial current-proof regions inside one bounded local block, where several opaque islands may still justify a localized bounded rewrite or may need to fall back depending on their topology.`
- findings_update:
  - `Extended agent_kernel/modeling/policy/decoder.py so partial current-proof region admission is now topology-aware instead of single-span-only: sparse internal multi-opaque islands can still synthesize one localized block_replace when every opaque island is internal, single-line, and fully accounted for by the proof map.`
  - `Dense/coarse multi-opaque shapes remain conservative: if an opaque island is too wide or the missing topology is not sparse enough for the available proof segmentation, current_proof_region_block synthesis is suppressed and expected_file_content remains preferred.`
  - `Added topology metadata to the bounded proof path, including current_proof_partial_region_topology and current_proof_opaque_max_span_line_count, so later ranking/canonicalization can preserve the stronger localized proof-bearing block over weaker same-span alternatives.`
- verification_update:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'partial_current_proof_opaque_span_is_internal or partial_current_proof_opaque_span_touches_boundary or current_proof_region_is_only_partial or sparse_internal_opaque_islands or dense_internal_opaque_islands'`
  - `pytest -q tests/test_policy.py -k 'prefer_localized_block_when_partial_current_proof_opaque_span_is_internal or keep_expected_write_when_partial_current_proof_opaque_span_touches_boundary or prefer_expected_write_when_current_proof_region_is_only_partial or sparse_internal_opaque_islands or dense_internal_opaque_islands or current_hidden_gap_requires_proof or proof_only_current_bridge or multi_span_current_proof_region'`
- next_recommendation_update: `The next ROI on this lane is the mixed-topology multi-opaque case inside one bounded block: several opaque islands where some subshapes may still justify localized rewrite but denser or coarser shapes should fall back without losing the bounded proof map.`
- findings_update:
  - `Extended agent_kernel/modeling/policy/decoder.py so mixed multi-opaque topology is now preserved instead of flattened: partial-proof fallback proposals carry hidden_gap_current_partial_proof_topologies plus admissible/blocking/coarse opaque-island counts, so the bounded proof map survives even when localized current_proof_region_block synthesis is correctly suppressed.`
  - `Localized bounded rewrite admission still requires every opaque island in the chosen region to be admissible. Mixed internal topology now falls back on purpose when a coarse opaque island appears alongside otherwise admissible sparse islands.`
- verification_update:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'sparse_internal_opaque_islands or dense_internal_opaque_islands or partial_current_proof_opaque_span_is_internal or partial_current_proof_opaque_span_touches_boundary or current_proof_region_is_only_partial'`
  - `pytest -q tests/test_policy.py -k 'prefer_localized_block_when_partial_current_proof_opaque_span_is_internal or keep_expected_write_when_partial_current_proof_opaque_span_touches_boundary or prefer_expected_write_when_current_proof_region_is_only_partial or sparse_internal_opaque_islands or dense_internal_opaque_islands or current_hidden_gap_requires_proof or proof_only_current_bridge or multi_span_current_proof_region'`
- next_recommendation_update: `The next ROI on this lane is sub-region factorization inside one bounded local block: use the preserved mixed-topology proof map to carve out safe localized subregions instead of forcing all-or-nothing fallback across the whole bounded block.`
- findings_update:
  - `Extended agent_kernel/modeling/policy/decoder.py so mixed partial current-proof regions can emit factorized local subregions instead of staying all-or-nothing. When a parent mixed-topology proof map still contains a safe proof-bounded admissible chain, decode now synthesizes a localized current_proof_region_block for that subregion while preserving the parent mixed-topology risk metadata on the proposal frontier.`
  - `Kept the path conservative for coarse-only topology: if no admissible local chain exists, decode still prefers expected_file_content instead of inventing a localized bounded rewrite.`
- verification_update:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'partial_current_proof_opaque_span_is_internal or sparse_internal_opaque_islands or dense_internal_opaque_islands or factorize_safe_subregion_from_mixed_partial_current_proof or coarse_internal_opaque'`
  - `pytest -q tests/test_policy.py -k 'current_proof_region or partial_current_proof or multi_span_current_proof_region or hidden_gap_current_partial_proof or current_hidden_gap_requires_proof or proof_only_current_bridge'`
- next_recommendation_update: `The next ROI on this lane is multi-subregion canonicalization: when one file surfaces several safe factorized current-proof subregions, or when a factorized current-proof block competes with shared-anchor/overlap bounded repairs on the same span frontier, the decoder should choose one canonical localized repair instead of leaving several partially overlapping bounded winners alive.`


### `workspace_preview_hidden_gap_ambiguity_canonical_frontier`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-02T22:16:00Z`
- scope_id: `workspace_preview_hidden_gap_ambiguity_canonical_frontier`
- progress_label: `workspace_preview_hidden_gap_ambiguity_canonical_frontier`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'suppress_same_span_hidden_gap_region_duplicates_in_annotation_frontier or suppress_same_span_hidden_gap_current_duplicates_in_annotation_frontier or prune_same_span_multi_edit_reentry_after_composite_assembly or prune_same_command_multi_edit_reentry_after_composite_assembly or collapse_equal_rank_same_span_duplicates_to_single_canonical_block or prune_weaker_same_span_bounded_variants_upstream' --cache-clear
```

- hypothesis: `Same-span preview block_replace variants that survive proposal pruning with low proof quality are still being counted as multiple hidden-gap frontier alternatives, which biases expected-write fallback even when the localized repair family is already canonical.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-02T22:16:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so hidden-gap region/current ambiguity, current proof-required, current target-equivalent, and bounded-alternative annotation paths canonicalize same-span preview block_replace candidates with the existing same-span provenance/rank key before counting frontier alternatives.`
  - `That keeps hidden-gap bookkeeping aligned with the decoder's canonical bounded winner even when low-proof same-span variants are still present in raw proposal assembly, so ambiguity/frontier counts stop re-inflating to 2 for duplicate localized repairs.`
  - `Added tests/test_policy.py regressions proving same-span broader-region duplicates and same-span current-dominant duplicates both keep their raw proposals while reporting hidden-gap ambiguity count 0 and frontier count 1.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'suppress_same_span_hidden_gap_region_duplicates_in_annotation_frontier or suppress_same_span_hidden_gap_current_duplicates_in_annotation_frontier or prune_same_span_multi_edit_reentry_after_composite_assembly or prune_same_command_multi_edit_reentry_after_composite_assembly or collapse_equal_rank_same_span_duplicates_to_single_canonical_block or prune_weaker_same_span_bounded_variants_upstream' --cache-clear`
- verification_note:
  - `An older isolated fallback test, tests/test_policy.py::test_action_generation_candidates_prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous, still fails in this worktree while monkeypatching precomputed proposal metadata and does not exercise the new annotation path.`
- next_recommendation: `Carry the same canonical hidden-gap frontier into decode-time fallback/ranking so expected_file_content vs localized bounded repair decisions use the collapsed ambiguity family directly instead of only the raw metadata counts.`


### `workspace_preview_hidden_gap_decode_fallback_refresh`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-03T00:00:00Z`
- scope_id: `workspace_preview_hidden_gap_decode_fallback_refresh`
- progress_label: `workspace_preview_hidden_gap_decode_fallback_refresh`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous or prefer_expected_write_when_no_proof_broader_current_target_frontier_is_tied or prefer_expected_write_when_current_hidden_gap_requires_proof or prefer_bridged_hidden_gap_region_block_over_expected_write or prefer_chained_bridged_hidden_gap_region_block_over_expected_write or prefer_expected_write_when_hidden_gap_frontier_is_ambiguous or suppress_same_span_hidden_gap_region_duplicates_in_annotation_frontier or suppress_same_span_hidden_gap_current_duplicates_in_annotation_frontier or prune_same_span_multi_edit_reentry_after_composite_assembly or prune_same_command_multi_edit_reentry_after_composite_assembly' --cache-clear
```

- hypothesis: `Upstream same-span hidden-gap canonicalization is not enough if decode-time scoring consumes stale or incomplete preview metadata. Long-horizon bounded rewrite preference still regresses when proposal builders are monkeypatched or composed metadata drops proof/ambiguity fields before ranking.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:00:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so decode_action_generation_candidates now refreshes workspace-preview hidden-gap competition metadata from structured proposals before scoring, rather than trusting stale pre-annotated fields. Raw structured preview metadata now preserves proposal_source, command, and exact_target_span so the decode-time refresh can reconstruct current-ambiguity, target-equivalent, and proof-required fallback pressure even under monkeypatched proposal builders.`
  - `Expected-file-content proposals now carry a path-level workspace-preview fallback summary. That summary explicitly boosts bounded rewrite preference for broader hidden-gap ambiguity, broader-current ambiguity, target-equivalent broader-current ties, and proof-required broader-current blocks, while leaving bridged/current-proof bounded blocks unaffected because they do not emit those unresolved fallback signals.`
  - `This closes the older decoder gap: the expected-write fallback tests that previously failed with monkeypatched workspace-preview proposals now pass, and the bridged hidden-gap block preference tests still pass on the same slice.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous or prefer_expected_write_when_no_proof_broader_current_target_frontier_is_tied or prefer_expected_write_when_current_hidden_gap_requires_proof or prefer_bridged_hidden_gap_region_block_over_expected_write or prefer_chained_bridged_hidden_gap_region_block_over_expected_write or prefer_expected_write_when_hidden_gap_frontier_is_ambiguous or suppress_same_span_hidden_gap_region_duplicates_in_annotation_frontier or suppress_same_span_hidden_gap_current_duplicates_in_annotation_frontier or prune_same_span_multi_edit_reentry_after_composite_assembly or prune_same_command_multi_edit_reentry_after_composite_assembly' --cache-clear`
- next_recommendation: `Extend the same decode-time fallback-summary refresh to longer chained/composite localized repair families, especially multi-file or mixed composed preview proposals, so long-horizon bounded rewrite preference does not degrade when ambiguity/proof state is carried through deeper proposal assembly paths.`


### `workspace_preview_composite_fallback_summary_carry`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-03T00:30:00Z`
- scope_id: `workspace_preview_composite_fallback_summary_carry`
- progress_label: `workspace_preview_composite_fallback_summary_carry`
- status: `completed`
- command:

```bash
cd /data/agentkernel && python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py && pytest -q tests/test_policy.py -k 'attach_path_fallback_summary_to_composite_repairs or chained_preview_composite_carries_fallback_summary or carried_preview_fallback_summary_isolated_per_file or prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous or prefer_expected_write_when_current_hidden_gap_requires_proof or prefer_expected_write_when_hidden_gap_frontier_is_ambiguous' --cache-clear
```

- hypothesis: `Decode-time fallback preference is still brittle for long-horizon composed preview families because chained and multi-file structured proposals do not carry a reusable path-level hidden-gap fallback summary, so expected-write fallback still depends on direct block proposals surviving raw assembly.`
- owned_paths:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:30:00Z`
- result: `completed`
- findings:
  - `Patched agent_kernel/modeling/policy/decoder.py so workspace-preview proposals now receive a path-level fallback summary covering region ambiguity, broader-current ambiguity, target-equivalent ties, proof-required pressure, and best bounded proof quality. The summary is attached before proposals leave _workspace_preview_window_proposals and is propagated into raw structured metadata.`
  - `Decode-time fallback grouping now consumes both direct hidden-gap fields and carried workspace-preview summary fields, then reapplies the canonical per-path summary onto structured preview candidates before expected-file-content scoring. That keeps bounded rewrite preference stable even when a path only exposes chained or composite structured proposals at decode time.`
  - `Added regressions proving composite multi_edit proposals now carry the path-level fallback summary, that a chained monkeypatched preview composite can still drive expected-write fallback through carried ambiguity/proof fields, and that carried summaries remain isolated per target path across a multi-file decode.`
- artifacts:
  - `agent_kernel/modeling/policy/decoder.py`
  - `tests/test_policy.py`
- verification:
  - `python -m py_compile agent_kernel/modeling/policy/decoder.py tests/test_policy.py`
  - `pytest -q tests/test_policy.py -k 'attach_path_fallback_summary_to_composite_repairs or chained_preview_composite_carries_fallback_summary or carried_preview_fallback_summary_isolated_per_file or prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous or prefer_expected_write_when_current_hidden_gap_requires_proof or prefer_expected_write_when_hidden_gap_frontier_is_ambiguous' --cache-clear`
- next_recommendation: `Push the same carried-summary canonicalization into broader multi-stage localized repair families where preview compositions are stitched from several retained proposal tiers, so one canonical bounded proof signal survives deeper long-horizon decoder assembly without duplicate fallback pressure.`


### `liftoff_preserved_history_required_family_merge`

- runner_slot: `supervised_runner__codex_main`
- agent_name: `Codex`
- lane_id: `supervised_parallel_lane__parallel_orchestration`
- claimed_at: `2026-04-03T00:00:00Z`
- scope_id: `liftoff_preserved_history_required_family_merge`
- progress_label: `liftoff_preserved_history_required_family_merge`
- status: `completed`
- command:

```bash
cd /data/agentkernel && AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 10 --priority-benchmark-family repo_sandbox --preserve-report-history --takeover-drift-step-budget 20 --takeover-drift-wave-task-limit 4 --takeover-drift-max-waves 3 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json
```

- hypothesis: `The scoped preserved-history official rerun is not actually missing prior required-family evidence; the trust ledger is dropping it because repeated repo_sandbox reruns crowd the global 50-report recency window. If preserved-history trust switches to the full scoped history, overall_gated should see the earlier counted repository/project/repo_chore/integration reports and the official repo_sandbox liftoff gate should clear.`
- owned_paths:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `tests/test_liftoff.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- released_at: `2026-04-03T00:00:00Z`
- result: `completed`
- findings:
  - `Patched scripts/evaluate_tolbert_liftoff.py so --preserve-report-history now builds and writes trust ledgers with unattended_trust_recent_report_limit=0 while keeping the same scoped run_reports_dir and ledger output paths. That preserves counted required-family history across official family-specific reruns without broadening task selection or collapsing scoped artifacts.`
  - `Added targeted tests in tests/test_liftoff.py proving preserved-history trust switches to the full-history window and that an evaluator with unattended_trust_recent_report_limit=1 still recovers counted project/repository evidence from older scoped reports when preserve-report-history is enabled.`
  - `The live scoped candidate directory already contained cross-family evidence; the bug was the recency cap, not missing history. After the fix, trajectories/reports/tolbert_liftoff_candidate_unattended_trust_ledger.json now records policy.recent_report_limit=0 and required_families_with_counted_gated_reports=["integration","project","repo_chore","repo_sandbox","repository"].`
  - `The official rerun at trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json now lands state=retain instead of reject: candidate_trust.overall_assessment.status=trusted, baseline_trust.overall_assessment.status=trusted, primary_takeover_families=["repo_sandbox"], and the same report still carries the earlier clean takeover-drift evidence with zero drift regressions.`
- artifacts:
  - `scripts/evaluate_tolbert_liftoff.py`
  - `tests/test_liftoff.py`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json`
  - `trajectories/reports/tolbert_liftoff_candidate_unattended_trust_ledger.json`
- verification:
  - `python -m py_compile scripts/evaluate_tolbert_liftoff.py tests/test_liftoff.py`
  - `pytest -q tests/test_liftoff.py -k "can_preserve_report_history or preserved_history_uses_full_required_family_window or scopes_takeover_drift_to_shadow_candidate_and_priority_family or prefers_counted_trust_summary_over_gated_attempts"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 10 --priority-benchmark-family repo_sandbox --preserve-report-history --takeover-drift-step-budget 20 --takeover-drift-wave-task-limit 4 --takeover-drift-max-waves 3 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverdrift_task10.json`
- next_recommendation: `The family-history blocker is closed. The next ROI on official liftoff is no longer report aggregation; it is expanding actual primary takeover breadth beyond repo_sandbox and then validating larger clean repo_sandbox slices above task_limit=10 without reintroducing the unsafe acceptance tail.`

## 2026-04-03 - liftoff_takeover_breadth_and_repo_sandbox_width

- scope_id: `liftoff_takeover_breadth_and_repo_sandbox_width`
- lane: `official_liftoff`
- hypothesis: `The next official liftoff gain is twofold: keep repo_sandbox clean past task_limit=10 by inserting cleaner retrieval-backed sandbox roots before the unstable bridge/acceptance tail, and stop family-breadth official reruns from collapsing into one dirty global scope or synthetic shared-repo worker contamination so actual multi-family takeover can be measured cleanly.`
- owned_paths:
  - `evals/harness.py`
  - `scripts/evaluate_tolbert_liftoff.py`
  - `agent_kernel/modeling/evaluation/liftoff.py`
  - `tests/test_eval.py`
  - `tests/test_liftoff.py`
  - `docs/ai_agent_status.md`
- findings:
  - `Patched evals/harness.py so repo_sandbox required-family ordering now keeps the original 10 clean non-acceptance roots, then inserts git_repo_status_review_retrieval_task and git_repo_test_repair_retrieval_task before git_parallel_worker_api/docs, git_generated_conflict_resolution_task, and the release-train acceptance tail. This widens the clean sandbox slice without reintroducing the unsafe acceptance roots.`
  - `Patched scripts/evaluate_tolbert_liftoff.py so preserved-history official family reruns widen into a mixed required-family compare with explicit family weights, scope all baseline/candidate workspaces and report roots from the report stem, and disable synthetic shared-repo worker/integrator surfacing for official compare/drift evals. This turns official breadth probes into clean experiment-local family mixes instead of one shared dirty scope plus relabeled shared-repo worker spillover.`
  - `Patched agent_kernel/modeling/evaluation/liftoff.py to fix the latent _report(...) call that omitted insufficient_proposal_families on the takeover-drift budget gate.`
  - `Fresh official repo_sandbox breadth evidence is now positive. trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverbreadth_mixed_task10_step14.json lands state=retain with primary_takeover_families=["integration","project","repo_chore","repo_sandbox","repository"], candidate=10/10 pass_rate=1.0, candidate_trust.overall_assessment.status=trusted, and takeover_drift budget_reached=true at 14.0/14.0 estimated steps with zero regressions.`
  - `Fresh isolated repo_sandbox width evidence is wider but still globally trust-gated in isolation. trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task12_widened.json now reaches 11/12 with the retrieval-backed ordering and no acceptance-tail reintroduction, but it still rejects because one-family isolated reports cannot satisfy overall_gated min_distinct_families=2.`
- artifacts:
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverbreadth_mixed_task10_step14.json`
  - `trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task12_widened.json`
- verification:
  - `python -m py_compile evals/harness.py scripts/evaluate_tolbert_liftoff.py agent_kernel/modeling/evaluation/liftoff.py tests/test_eval.py tests/test_liftoff.py`
  - `pytest -q tests/test_eval.py::test_limit_tasks_for_compare_uses_cleaner_repo_sandbox_primaries_before_known_unstable_roots tests/test_eval.py::test_limit_tasks_for_compare_widens_repo_sandbox_before_acceptance_tail tests/test_eval.py::test_limit_tasks_for_compare_prefers_clean_repo_sandbox_retrieval_before_unstable_bridge_tail`
  - `pytest -q tests/test_liftoff.py -k "requires_takeover_drift_budget_reached or preserve_report_history or scopes_takeover_drift_to_shadow_candidate_and_priority_family or preserved_history_widens_eval_breadth_and_weights_drift or enables_repo_sandbox_operator_policy_for_required_family or scoped_liftoff_config_scopes_workspace_and_reports or liftoff_scope_name_uses_report_stem"`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 10 --priority-benchmark-family repo_sandbox --preserve-report-history --takeover-drift-step-budget 14 --takeover-drift-wave-task-limit 4 --takeover-drift-max-waves 3 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_takeoverbreadth_mixed_task10_step14.json`
  - `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 python scripts/evaluate_tolbert_liftoff.py --task-limit 12 --priority-benchmark-family repo_sandbox --takeover-drift-step-budget 0 --report-path trajectories/tolbert_model/liftoff_gate_report_20260403_reposandbox_cleantrust_task12_widened.json`
- next_recommendation: `The next ROI on this stack is no longer “expand beyond repo_sandbox” because the official gate now retains five primary families. It is either (1) push the clean isolated repo_sandbox slice from 11/12 toward 13+ without waking git_parallel_worker_api/docs or git_generated_conflict_resolution_task, or (2) convert the new five-family retain artifact into routing/application policy so that wider unattended operation actually uses the gained takeover breadth.`

## 2026-04-04 - unattended_repo_setting_reporoute_after_strong_round

- scope_id: `unattended_repo_setting_reporoute_after_strong_round`
- lane: `autonomy_runtime`
- hypothesis: `The live unattended gap is no longer early qwen stall; it is post-observation routing. If improvement ranking treats a fresh strong repository/project round as evidence to keep spending budget on coding-facing subsystems instead of a support-runtime-only Qwen preview, the same healthy unattended vllm round should route into retrieval/TolBERT/policy/tooling rather than qwen_adapter.`
- owned_paths:
  - `agent_kernel/improvement.py`
  - `tests/test_improvement.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- findings:
  - `Patched agent_kernel/improvement.py so support-runtime-only qwen_adapter proposals are suppressed after strong current-round repository/project/integration evidence, and so the runtime-only Qwen lane now pays a cold-start low-confidence cap like retrieval/TolBERT instead of remaining an uncapped top scorer on thin evidence.`
  - `Added focused regressions in tests/test_improvement.py proving two things: retrieval still wins the generic low-confidence cold-start case, and qwen_adapter disappears entirely after a strong 5/5 repository+project coding round.`
  - `Fresh live unattended vllm evidence is in var/unattended_repo_setting_live_20260404_vllm_reporoute/reports/unattended_campaign_repo_setting_live_20260404_vllm_reporoute.events.jsonl. The same bounded project+repository round again completed at observe complete passed=5/5 pass_rate=1.0000 generated_pass_rate=1.0000, but the first post-observation selection changed from qwen_adapter to tooling. The child then started tooling/procedure_promotion and emitted a candidate artifact at var/unattended_repo_setting_live_20260404_vllm_reporoute/candidates/tooling/cycle_tooling_20260404T003814645574Z_ec11e451_procedure_promotion/tool_candidates.json.`
  - `The top-level unattended report/status are interrupted because I terminated the process tree immediately after the selection evidence landed, to avoid leaving another long-running exec active while the machine is already over the unified-exec cap. The routing evidence itself is preserved in the event log before termination.`
- artifacts:
  - `agent_kernel/improvement.py`
  - `tests/test_improvement.py`
  - `var/unattended_repo_setting_live_20260404_vllm_reporoute/reports/unattended_campaign_repo_setting_live_20260404_vllm_reporoute.events.jsonl`
  - `var/unattended_repo_setting_live_20260404_vllm_reporoute/candidates/tooling/cycle_tooling_20260404T003814645574Z_ec11e451_procedure_promotion/tool_candidates.json`
- verification:
  - `python -m py_compile agent_kernel/improvement.py tests/test_improvement.py`
  - `pytest -q tests/test_improvement.py -k "prefers_retrieval_when_low_confidence_is_high or suppresses_qwen_adapter_after_strong_repo_project_round or caps_cold_start_qwen_support_runtime_score or caps_cold_start_low_confidence_retrieval_score or select_portfolio_campaign_prefers_underexplored_near_tie or select_portfolio_campaign_does_not_relax_scored_eligibility_under_stalled_lead_pressure"`
  - `env AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B ... python scripts/run_unattended_campaign.py --rounds 1 --cycles 1 --campaign-width 1 --variant-width 1 --task-limit 5 --task-step-floor 50 --priority-benchmark-family project --priority-benchmark-family repository --focus balanced --liftoff never --skip-trust-preflight ...`
- next_recommendation: `The next unattended ROI is no longer “stop qwen from winning after a strong coding round”; that is repaired. The next step is to convert this healthier routing into longer autonomous coding campaigns by biasing post-observation selection toward repo/project/integration-capable improvement lanes under wider task limits and repeated rounds, then validating that the same fix holds when the campaign is allowed to continue beyond the first selection.`

## 2026-04-04 - unattended_frontiercarry_productive_depth_widening

- scope_id: `unattended_frontiercarry_productive_depth_widening`
- lane: `autonomy_runtime`
- hypothesis: `If unattended next-round policy treats strong repository+project productive-depth yield as an explicit coding-frontier expansion signal, then healthy small coding rounds should widen into integration/repository/project followups instead of only proving a better first subsystem selection on the narrow 5-task slice.`
- owned_paths:
  - `scripts/run_unattended_campaign.py`
  - `tests/test_unattended_campaign.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- findings:
  - `Patched scripts/run_unattended_campaign.py so clean productive-depth repository+project evidence now triggers coding_frontier_expansion: the next unattended policy switches to discovered_task_adaptation, widens cycles/task_limit/campaign_width, optionally widens variant_width on long-horizon evidence, and forces the next priority-family mix to integration, repository, project instead of leaving the broader round on the older narrow family pair.`
  - `Added a focused regression in tests/test_unattended_campaign.py proving that a strong 5-task repository+project productive-depth round now expands into focus=discovered_task_adaptation, task_limit=10, campaign_width=2, variant_width=2, and priority_benchmark_families=[integration, repository, project] even when the selection floor would otherwise have filtered the newly added integration family.`
  - `The outer unattended parent was not a clean live artifact in this session: fresh isolated run_unattended_campaign.py invocations stalled before campaign launch while still showing the intended widened policy in status. I therefore validated the runtime effect directly at the child layer with scripts/run_repeated_improvement_cycles.py under the widened policy.`
  - `Fresh live widened child evidence is in var/unattended_repo_setting_live_20260404_vllm_frontierchild/improvement_reports/repeated_improvement_status.json and campaign_report_20260404T135831764936Z.json. The direct widened vllm child entered the harder coding lane immediately with [eval:frontiercarry_live] task 1/10 bridge_handoff_task family=integration, completed the primary slice through task 10/10 api_contract_retrieval_task_discovered, then ran generated_success total=10 and entered generated_failure total=10 before I interrupted it to avoid leaving another long-running exec active while the machine is over the unified-exec session cap.`
- artifacts:
  - `scripts/run_unattended_campaign.py`
  - `tests/test_unattended_campaign.py`
  - `var/unattended_repo_setting_live_20260404_vllm_frontierchild/improvement_reports/repeated_improvement_status.json`
  - `var/unattended_repo_setting_live_20260404_vllm_frontierchild/improvement_reports/campaign_report_20260404T135831764936Z.json`
- verification:
  - `python -m py_compile scripts/run_unattended_campaign.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_campaign.py -k "expands_productive_repo_project_depth_into_integration_frontier or promotes_frontier_repo_setting_family_outside_current_mix or reports_limited_priority_family_availability or raises_task_step_floor_for_productive_depth"`
  - `env AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B ... python scripts/run_repeated_improvement_cycles.py --cycles 2 --campaign-width 2 --variant-width 2 --adaptive-search --task-limit 10 --priority-benchmark-family integration --priority-benchmark-family repository --priority-benchmark-family project --campaign-label frontiercarry_live --campaign-match-id frontiercarry_live_20260404 ...`
- next_recommendation: `The next unattended ROI is to repair the fresh outer run_unattended_campaign.py pre-campaign stall so the same widened integration/repository/project policy can be observed end-to-end at the parent layer, then let the widened child continue through subsystem selection and retained candidate generation instead of stopping after confirming the broader primary/generated coding envelope.`

## 2026-04-04 - unattended_parent_cleanup_fastskip

- scope_id: `unattended_parent_cleanup_fastskip`
- lane: `autonomy_runtime`
- hypothesis: `The highest-ROI blocker after widened child proof is not coding selection anymore; it is parent launch overhead. If unattended parent cleanup fast-skips when disk headroom is already comfortably above target, fresh widened parent runs should stop stalling in cleanup and reach the actual coding child end-to-end.`
- owned_paths:
  - `scripts/run_unattended_campaign.py`
  - `tests/test_unattended_campaign.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- findings:
  - `Patched scripts/run_unattended_campaign.py so _governed_cleanup_runtime_state and _governed_global_storage_cleanup now return immediate skipped payloads when free disk is already above the configured target, instead of recursively snapshotting and pruning managed trees before every unattended round.`
  - `Added focused regressions in tests/test_unattended_campaign.py proving both cleanup surfaces skip cleanly without invoking the expensive snapshot/prune paths when disk headroom is already sufficient.`
  - `Fresh live parent evidence is now positive. var/unattended_repo_setting_live_20260404_vllm_parentfix/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentfix.events.jsonl records preflight -> cleanup -> campaign, then child start and observe start under the widened integration/repository/project policy. The paired status artifact shows both cleanup surfaces skipped with reason=disk already above target and the parent actively tracking task 1/10 deployment_manifest_task family=project before I terminated the run to avoid leaving another long-running exec open.`
- artifacts:
  - `scripts/run_unattended_campaign.py`
  - `tests/test_unattended_campaign.py`
  - `var/unattended_repo_setting_live_20260404_vllm_parentfix/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentfix.events.jsonl`
  - `var/unattended_repo_setting_live_20260404_vllm_parentfix/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentfix.status.json`
- verification:
  - `python -m py_compile scripts/run_unattended_campaign.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_campaign.py -k "governed_cleanup_runtime_state_skips_when_disk_is_already_above_target or governed_global_storage_cleanup_skips_when_disk_is_already_above_target or expands_productive_repo_project_depth_into_integration_frontier or recovers_stalled_subsystem_after_generic_silence_output"`
  - `env AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 ... python scripts/run_unattended_campaign.py --rounds 1 --cycles 1 --campaign-width 2 --variant-width 2 --adaptive-search --task-limit 10 --task-step-floor 100 --priority-benchmark-family integration --priority-benchmark-family repository --priority-benchmark-family project --focus discovered_task_adaptation --liftoff never --skip-trust-preflight --max-child-runtime-seconds 240 ...`
- next_recommendation: `The next unattended ROI is to let this repaired widened parent run continue long enough to capture post-observation subsystem selection and retained candidate generation at the parent layer, then reduce first-slice latency or stall risk inside that wider integration/repository/project child so multi-round autonomous coding can complete instead of only proving launch and early harder-task execution.`

## 2026-04-04 - unattended_parentbridge_frontiercarry_v4

- scope_id: `unattended_parentbridge_frontiercarry_v4`
- lane: `autonomy_runtime`
- hypothesis: `If the unattended parent preserves mirrored repeated-child status, the repeated child salvages productive interrupted runs, and the parent accepts runtime-capped children that already completed generated_success as completed warned rounds, widened autonomous coding runs should stop collapsing into empty safe_stop artifacts and instead become counted frontier evidence.`
- owned_paths:
  - `scripts/run_repeated_improvement_cycles.py`
  - `scripts/run_unattended_campaign.py`
  - `tests/test_finalize_cycle.py`
  - `tests/test_unattended_campaign.py`
  - `docs/ai_agent_status.md`
  - `docs/supervised_work_queue.md`
- findings:
  - `Patched scripts/run_repeated_improvement_cycles.py so repeated-child status now carries report_path, infers observe_completed once generated_success starts, salvages productive interrupted active runs into synthetic counted runs, and records partial_progress_summary instead of dropping widened runtime-capped work on KeyboardInterrupt.`
  - `Patched scripts/run_unattended_campaign.py so the outer wrapper parses child phase/task progress, grants one finalize grace after generated_success reaches its terminal task, and accepts runtime-capped children as completed warned rounds when the mirrored repeated-child status proves generated_success already completed productively.`
  - `Focused regressions now cover both autonomy edges: repeated-child interrupted salvage in tests/test_finalize_cycle.py, and parent-side generated_success runtime-grace plus productive-timeout acceptance helpers in tests/test_unattended_campaign.py.`
  - `Fresh live evidence is decisive. var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.events.jsonl shows the widened integration-first child complete the 10-task primary slice, complete generated_success total=10, reach generated_failure_seed total=10, and only then hit the outer 300s cap. The parent report now finishes completed instead of safe_stop, with round status completed and campaign_validation.accepted_partial_timeout=true.`
  - `The repeated child no longer loses that work: var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/improvement_reports/repeated_improvement_status.json now records state=interrupted but completed_runs=1, productive_runs=1, partial_productive_runs=1, active_cycle_progress.generated_success_completed=true, and sampled families integration/discovered_task/project.`
- artifacts:
  - `scripts/run_repeated_improvement_cycles.py`
  - `scripts/run_unattended_campaign.py`
  - `tests/test_finalize_cycle.py`
  - `tests/test_unattended_campaign.py`
  - `var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.json`
  - `var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.status.json`
  - `var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.events.jsonl`
  - `var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/improvement_reports/repeated_improvement_status.json`
  - `var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/improvement_reports/campaign_report_20260404T160048412773Z.json`
- verification:
  - `python -m py_compile scripts/run_repeated_improvement_cycles.py scripts/run_unattended_campaign.py tests/test_finalize_cycle.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_finalize_cycle.py -k "writes_midrun_child_status_and_mirrors_parent or counts_partial_productive_timeout_progress or salvages_partial_productive_interrupt"`
  - `pytest -q tests/test_unattended_campaign.py -k "child_runtime_extension_seconds_grants_finalize_grace_after_generated_success_completion or accept_productive_partial_child_timeout_accepts_generated_success_completion or accept_productive_partial_child_timeout_rejects_without_generated_success_completion or write_status_preserves_mirrored_child_frontier_fields or passes_parent_status_bridge_to_child or initial_round_policy_preserves_requested_priority_family_order"`
  - `env AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS=30 ... python scripts/run_unattended_campaign.py --rounds 1 --cycles 1 --campaign-width 2 --variant-width 1 --adaptive-search --task-limit 10 --task-step-floor 75 --priority-benchmark-family integration --priority-benchmark-family repository --priority-benchmark-family project --focus discovered_task_adaptation --liftoff never --skip-trust-preflight --max-child-runtime-seconds 300 --max-child-silence-seconds 600 --max-child-progress-stall-seconds 600 --child-heartbeat-seconds 30 --max-child-failure-recovery-rounds 0 --report-path var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.json --status-path var/unattended_repo_setting_live_20260404_vllm_parentbridge_v4/reports/unattended_campaign_repo_setting_live_20260404_vllm_parentbridge_v4.status.json`
- next_recommendation: `The next unattended autonomy ROI is no longer proving widened parent completion. The remaining live gap is runtime shaping inside the repeated child itself: generated_failure_seed already starts after generated_success, but the child still ages into the outer cap before clean campaign finalize. The next step is phase-budgeting generated_failure_seed/finalize inside scripts/run_repeated_improvement_cycles.py so the widened child can finish retained candidate generation without relying on parent-side accepted-partial fallback.`
