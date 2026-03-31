# Scripts

## Core runtime scripts

[`scripts/run_agent.py`](/data/agentkernel/scripts/run_agent.py)

- runs a single task from the task bank
- supports `--task-id`, provider/model overrides, feature toggles, and TOLBERT path overrides

Useful flags:

- `--use-tolbert-context {0,1}`
- `--use-skills {0,1}`
- `--use-graph-memory {0,1}`
- `--use-world-model {0,1}`
- `--use-planner {0,1}`
- `--use-role-specialization {0,1}`
- `--use-prompt-proposals {0,1}`
- `--use-curriculum-proposals {0,1}`
- `--use-retrieval-proposals {0,1}`
- `--tolbert-cache` can be passed multiple times

[`scripts/run_eval.py`](/data/agentkernel/scripts/run_eval.py)

- runs the local evaluation harness
- can include curriculum-generated tasks, replay tasks, transfer tasks, and candidate-task lanes
- can run comparison lanes for skills, operator abstractions, or TOLBERT modes

Important flags:

- `--include-curriculum`
- `--include-failure-curriculum`
- `--include-episode-memory`
- `--include-skill-memory`
- `--include-skill-transfer`
- `--include-operator-memory`
- `--include-tool-memory`
- `--include-verifier-memory`
- `--include-benchmark-candidates`
- `--include-verifier-candidates`
- `--use-prompt-proposals {0,1}`
- `--use-curriculum-proposals {0,1}`
- `--use-retrieval-proposals {0,1}`
- `--compare-skills`
- `--compare-abstractions`
- `--compare-tolbert`
- `--compare-tolbert-features`
- `--tolbert-mode {full,path_only,retrieval_only,deterministic_command,skill_ranking}`

[`scripts/run_selfplay.py`](/data/agentkernel/scripts/run_selfplay.py)

- runs a seed task and then a generated followup
- supports `--seed-task-id` and `--seed-mode {success,failure}`

[`scripts/replay_episode.py`](/data/agentkernel/scripts/replay_episode.py)

- prints a stored episode JSON by task ID

## Extraction and synthesis scripts

[`scripts/extract_skills.py`](/data/agentkernel/scripts/extract_skills.py)

- promotes successful command procedures into `trajectories/skills/command_skills.json`
- current payload is a structured artifact object with a top-level `skills` array, not a bare JSON list

[`scripts/extract_tools.py`](/data/agentkernel/scripts/extract_tools.py)

- extracts multi-command local shell procedures into `trajectories/tools/tool_candidates.json`

[`scripts/extract_operators.py`](/data/agentkernel/scripts/extract_operators.py)

- promotes reusable cross-task abstractions into `trajectories/operators/operator_classes.json`
- supports `--min-support` and `--cross-family-only`

[`scripts/synthesize_verifiers.py`](/data/agentkernel/scripts/synthesize_verifiers.py)

- writes stricter verifier contract candidates to `trajectories/verifiers/verifier_contracts.json`

[`scripts/synthesize_benchmarks.py`](/data/agentkernel/scripts/synthesize_benchmarks.py)

- writes retrieval/benchmark candidates to `trajectories/benchmarks/benchmark_candidates.json`

## Improvement scripts

[`scripts/select_improvement_target.py`](/data/agentkernel/scripts/select_improvement_target.py)

- runs eval and chooses the next subsystem to improve

[`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)

- runs observe/select/generate steps for one improvement cycle
- appends records to `trajectories/improvement/cycles.jsonl`
- supports `--adaptive-search` to widen campaign and sibling-variant search from retained-history signals
- supports bounded observation with `--task-limit`, repeated `--priority-benchmark-family`, and repeated `--priority-benchmark-family-weight`
- uses scored breadth eligibility for adaptive campaign and sibling-variant widening, so the same planner score surface decides both ordering and width instead of separate stalled-lead exceptions
- keeps search-width floors and margins explicit under planner `search_guardrails`, separate from gain/cost/bias score controls
- uses recent-history portfolio selection to diversify campaign subsystem choice across repeated autonomous runs

[`scripts/run_supervised_improvement_cycle.py`](/data/agentkernel/scripts/run_supervised_improvement_cycle.py)

- stable operator-facing alias for the guided improvement protocol
- delegates to `run_human_guided_improvement_cycle.py` without relying on the importable `scripts` package name
- supports the same bounded observation controls as the autonomous cycle: `--task-limit`, repeated `--priority-benchmark-family`, repeated `--priority-benchmark-family-weight`, and `--progress-label`
- supports `--max-observation-seconds` as a soft observation budget that records when the supervised loop is too slow for tight operator steering
- supports `--generate-only` so an operator can inspect the selected subsystem, variant, and candidate artifact before any retention run

[`scripts/run_parallel_supervised_cycles.py`](/data/agentkernel/scripts/run_parallel_supervised_cycles.py)

- launches multiple scoped `run_supervised_improvement_cycle.py --generate-only` children concurrently
- uses unique `--scope-id` and `--progress-label` values per worker so batch runs stay isolated while sharing the same retained baseline and provider
- auto-diversifies worker subsystems from planner plus recent batch-report outcomes when `--subsystem` is not provided
- streams child output with stable `[parallel:<scope>]` prefixes so one terminal can supervise the full batch
- writes one `parallel_supervised_preview_report` plus `parallel_supervised_preview_history.jsonl` so future batches can reuse preview-yield evidence instead of only single-cycle history

[`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)

- runs the machine-owned supervisor loop over the existing supervised discovery and promotion surfaces
- reads shared machine state from the frontier report, promotion-pass report, unattended trust ledger, delegated queue/runtime leases, and recent scoped supervised cycle outcomes
- writes `supervisor_loop_status.json`, `supervisor_loop_history.jsonl`, and `supervisor_loop_report.json` into `trajectories/improvement/reports`
- supports bounded discovery launches through `run_parallel_supervised_cycles.py`, promotion-plan refresh, and promotion-pass execution under explicit autonomy modes: `shadow`, `dry_run`, and `promote`
- loads [`config/supervisor_meta_policy.json`](/data/agentkernel/config/supervisor_meta_policy.json) to classify protected meta-surfaces from both subsystem and manifest path ownership, expose machine-owned claim and lane-allocation state, gate protected finalize paths by rollout stage: `shadow`, `compare_only`, `canary`, or `broad`, and run a staged canary lifecycle with rollback intent, automatic post-rollback validation, and promotion-resume gating when protected retains are followed by restricted trust
- keeps the human as the outer control plane by requiring the operator to start and stop the process while the kernel owns the inner round-by-round supervision logic

[`scripts/finalize_improvement_cycle.py`](/data/agentkernel/scripts/finalize_improvement_cycle.py)

- evaluates a generated artifact against a baseline lane and records a retention decision

[`scripts/propose_prompt_update.py`](/data/agentkernel/scripts/propose_prompt_update.py)

- writes prompt proposal artifacts to `trajectories/prompts/prompt_proposals.json`
- prompt artifacts now include decision controls, planner controls, and role-directive overrides

[`scripts/propose_curriculum_update.py`](/data/agentkernel/scripts/propose_curriculum_update.py)

- writes curriculum proposal artifacts to `trajectories/curriculum/curriculum_proposals.json`

[`scripts/propose_retrieval_update.py`](/data/agentkernel/scripts/propose_retrieval_update.py)

- writes retrieval proposal artifacts to `trajectories/retrieval/retrieval_proposals.json`

[`scripts/materialize_retrieval_asset_bundle.py`](/data/agentkernel/scripts/materialize_retrieval_asset_bundle.py)

- rebuilds a retained retrieval proposal into `trajectories/retrieval/retrieval_asset_bundle.json`
- emits Tolbert config/nodes/spans assets alongside a runtime bundle manifest

[`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)

- runs multiple improvement cycles back to back
- supports campaign labels and isolated-match metadata for autonomous campaign comparisons
- forwards adaptive search and sibling-variant width controls into each cycle run
- accepts repeated `--priority-benchmark-family` flags to reserve limited eval budget for specific benchmark families across child cycles
- reports per-priority-family retained yield plus estimated-cost spend so outer supervisors can rank families by retained pass-rate gain per estimated cost while still preserving an exploration bonus for under-sampled families
- shares the retained `improvement_planner_controls` surface with unattended family ranking, so family-level gain multipliers, cost multipliers, and score biases can be tuned through the same prompt-policy artifact path as subsystem and variant scoring

[`scripts/run_autonomous_compounding_check.py`](/data/agentkernel/scripts/run_autonomous_compounding_check.py)

- runs repeated autonomous improvement campaigns from isolated starting states
- forwards repeated `--priority-benchmark-family` flags into each isolated repeated-campaign run, and otherwise defaults to broader non-replay transfer families: `workflow`, `project`, `repository`, `tooling`, and `integration`
- resolves a batch-stable task budget for those runs from `--task-limit` or `compare_feature_max_tasks`, then forwards per-family weight hints so task-limited eval sampling can spend proportionally more budget on the strongest transfer-investment families instead of only ordering them
- records a planned-versus-realized priority-family allocation audit from repeated campaign reports, including the latest allocation snapshot, so the compounding summary can show whether the weighted family budget actually changed sampled task spend and clear stale compensation once the latest gap closes
- feeds material planned-versus-actual share gaps back into the next batch as explicit `priority_benchmark_family_allocation_compensation`, so persistently under-sampled high-priority families can receive a stronger streak-aware weight plan and over-sampled families can receive a smaller streak-aware normalization pull, both damped by longitudinal family-specific allocation confidence and allowed to use asymmetric bonus-versus-normalization confidence blends; the retained `improvement_planner_controls.priority_family_allocation_confidence` surface now also controls minimum runs, target priority tasks, optional target family task overrides, history window, base history blend, and the bonus-versus-normalization history weights instead of leaving those values hard-coded
- extracts a comparable per-run result stream from each repeated campaign report so runtime-managed outcomes stay auditable by run instead of only through nested raw payloads
- summarizes retained-gain transfer across those families and blocks compounding claim readiness when family observation or retained gain stays too narrow
- records per-family transfer timelines across runs, including retained return-on-cost, and blocks compounding claim readiness when none of those repeated retained-gain timelines stay non-declining at acceptable cost over time
- ranks benchmark families by cross-run transfer investment so reports can name the strongest longer-horizon transfer bets instead of only reporting which families cleared the gate
- reuses the latest autonomous-compounding family ranking to seed the next batch's default `--priority-benchmark-family` order and weighted family allocation when the CLI does not pin one explicitly
- writes a report summarizing cross-run retained gains, regression spread, and compounding viability

[`scripts/run_tolbert_liftoff_loop.py`](/data/agentkernel/scripts/run_tolbert_liftoff_loop.py)

- runs repeated improvement cycles to grow the episode corpus, then builds/trains a Tolbert candidate bundle
- evaluates the candidate against the baseline runtime under the liftoff gate
- writes a single readiness report with dataset volume, training artifacts, comparison deltas, and liftoff state
- forwards any `--priority-benchmark-family` selections into repeated-improvement runs and baseline/candidate eval sampling

[`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)

- runs the outer unattended supervisor loop around repeated improvement and optional Tolbert liftoff
- performs disk, GPU, and unattended-trust preflight checks before launching child campaigns
- prunes old reports, candidate artifacts, checkpoints, and scoped recovery snapshots before bounded autonomous runs
- validates child campaign/liftoff reports instead of trusting subprocess exit codes alone
- adapts cycles per round, task limits, campaign width, variant width, and focus across rounds from campaign/liftoff results
- turns missing required-family coverage into a concrete round policy by prioritizing broader families during task-limited repeated campaigns and liftoff evals
- reorders those priority families using cross-round return-on-cost ranking so budget shifts toward higher retained pass-rate gain per estimated cost while still keeping under-sampled families in the mix
- applies retained family scoring controls from `improvement_planner_controls` before choosing the next round's benchmark-family budget
- supports `priority_family_min_selection_score` so low-score families can be dropped from the selected priority set while missing required families still override the floor
- reports whether a narrowed priority-family set came from the score floor versus simple lack of additional ranked families
- embeds unattended trust, hidden-side-effect, false-pass, and required-family posture snapshots directly into campaign reports and live status
- auto-triggers `run_tolbert_liftoff_loop.py` only when the campaign retained artifacts and cleared retained phase gates
- supports local `flock` leases or an external HTTP lease backend, plus concrete Slack/PagerDuty alert delivery with dedup and rate limiting

[`scripts/run_lease_server.py`](/data/agentkernel/scripts/run_lease_server.py)

- runs a lightweight in-repo HTTP lease service for unattended campaign coordination
- exposes `/acquire`, `/heartbeat`, and `/release` endpoints with persisted lease state and TTL expiry

[`scripts/run_job_queue.py`](/data/agentkernel/scripts/run_job_queue.py)

- manages unattended delegated jobs
- supports queue portfolio controls including `--budget-group`, `--max-active-per-budget-group`, and `--max-queued-per-budget-group`

[`scripts/compare_retained_baseline.py`](/data/agentkernel/scripts/compare_retained_baseline.py)

- compares the current artifact against the most recent retained baseline for one subsystem

[`scripts/rollback_artifact.py`](/data/agentkernel/scripts/rollback_artifact.py)

- restores the last rollback snapshot recorded for an artifact path

[`scripts/validate_rollback_artifact.py`](/data/agentkernel/scripts/validate_rollback_artifact.py)

- verifies that a rolled-back artifact matches its recorded rollback snapshot before protected promotion is allowed to resume

[`scripts/report_tolbert_first_steps.py`](/data/agentkernel/scripts/report_tolbert_first_steps.py)

- summarizes first-step path confidence, retrieval trust, and retrieval-selected actions from stored episodes

[`scripts/report_failure_recovery.py`](/data/agentkernel/scripts/report_failure_recovery.py)

- summarizes failure-recovery curriculum episodes and their first actions

[`scripts/assert_verify_metrics.py`](/data/agentkernel/scripts/assert_verify_metrics.py)

- applies empirical gates to captured `verify_impl.sh` outputs

[`scripts/report_unattended_run_metrics.py`](/data/agentkernel/scripts/report_unattended_run_metrics.py)

- summarizes unattended task outcomes, hidden-side-effect and false-pass rates, and acceptance-packet activity
- prints required-family coverage, missing-family gaps, and per-required-family trust posture from the unattended trust ledger

Compatibility and protocol-comparison utilities still exist in `scripts/run_human_guided_improvement_cycle.py`,
`scripts/compare_improvement_protocols.py`, and `scripts/run_protocol_head_to_head.py`, but the recommended
operator entrypoint is `scripts/run_supervised_improvement_cycle.py` and none of those scripts are part of the
autonomous recursive-improvement closure path.

## TOLBERT asset scripts

[`scripts/build_agentkernel_tolbert_assets.py`](/data/agentkernel/scripts/build_agentkernel_tolbert_assets.py)

- builds repo-native ontology, spans, label map, config, and level-size assets
- is also used by retrieval-bundle materialization when retained retrieval controls request an asset rebuild

[`scripts/build_tolbert_cache.py`](/data/agentkernel/scripts/build_tolbert_cache.py)

- builds a retrieval shard from a trained checkpoint and spans file

[`scripts/build_tolbert_label_map.py`](/data/agentkernel/scripts/build_tolbert_label_map.py)

- builds a model-class to ontology-node ID map when training labels were remapped

[`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py)

- subprocess runtime used by the kernel for embedding, decode, and retrieval

[`scripts/run_training_backend.py`](/data/agentkernel/scripts/run_training_backend.py)

- lists or launches vendored external model-training backends from `other_repos/`
- supports `--tolbert-artifact` to export the retained Tolbert training-input manifest and per-head dataset paths into the backend environment

## Shell wrappers

[`scripts/run_native_agent.sh`](/data/agentkernel/scripts/run_native_agent.sh)

- exports the expected native asset env vars and runs `run_agent.py`
- still defaults to `ollama` unless `AGENT_KERNEL_PROVIDER`, `AGENT_KERNEL_MODEL`, and `AGENT_KERNEL_VLLM_HOST` are overridden in the shell

[`scripts/run_native_eval.sh`](/data/agentkernel/scripts/run_native_eval.sh)

- exports the same native asset env vars and runs `run_eval.py`
- also defaults to `ollama` unless the same provider env vars are overridden

[`scripts/verify_impl.sh`](/data/agentkernel/scripts/verify_impl.sh)

- builds an isolated verification workspace under `/tmp/agentkernel_verify`
- runs the live runtime and multiple eval comparison lanes
- refreshes skill, operator, and tool artifacts inside that isolated area
- archives per-stage outputs under `AGENT_KERNEL_VERIFY_ARCHIVE_ROOT`
- captures TOLBERT first-step and failure-recovery reports
- inherits `AGENT_KERNEL_PROVIDER`, `AGENT_KERNEL_MODEL`, and `AGENT_KERNEL_VLLM_HOST` from the shell, so use those when verifying against `vllm`

## Minimal examples

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 python scripts/run_agent.py --task-id hello_task
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 python scripts/run_eval.py --include-skill-memory --include-tool-memory
python scripts/run_selfplay.py --seed-task-id hello_task --seed-mode failure
python scripts/run_improvement_cycle.py --include-episode-memory --include-skill-memory
python scripts/extract_operators.py --cross-family-only
python scripts/propose_retrieval_update.py --focus confidence
```
