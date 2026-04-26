# Unattended Work Queue

## Purpose

This document is the canonical parallel execution plan for the current unattended kernel push.

Use it to:

- split work across multiple Codex terminals without write collisions
- define exact file ownership for each lane
- define interface contracts between lanes
- define merge order and restart points
- define the single end-to-end acceptance gate for the wave

Do not use this file as a historical journal. Replace it when the active wave changes.

## Active Wave

- wave_id: `unattended_kernel_full_gap_closure_2026_04_09`
- objective:
  `close the remaining unattended kernel gaps blocking runtime-managed compounding, open-ended strategy invention, ontology expansion, repo generalization, durable strategy memory, retrieval carryover, and trust breadth`
- latest validation run:
  [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.status.json)
- paired report:
  [`unattended_campaign.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.json)
- gap references:
  [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
  [`docs/asi_evolve_kernel_fit.md`](/data/agentkernel/docs/asi_evolve_kernel_fit.md)
  [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)

## Current Read

The latest unattended run is materially better than the earlier broken runs, but it does not close
the milestone set.

What improved:

- the latest run finished with `status=completed`
- top-level reports now preserve authoritative decision state
- the run recorded `runtime_managed_decisions=1`
- the sampled families now include `integration`, `project`, and `repository`
- the April 11, 2026 reevaluation closed two runtime wiring bugs that had been overstating the live
  unattended gap surface:
  - retention/finalization no longer inherits bounded-observation curriculum suppression, so
    autonomous phase gates are not falsely failed by missing generated lanes during priority-family
    campaigns
  - repeated-child supervision now clears stale decision state when a new `observe` cycle starts, so
    a prior-cycle reject no longer contaminates later generated-failure work in the same child

What is still not solved:

- the credited retain was controller-closed over an incomplete child cycle rather than a clean
  child-native decision stream
- a live retained `discovered_strategy` path is now wired but still needs an integrated retained rerun artifact
- runtime ontology expansion is not yet present in the catalogs
- discovered structural work classes are not yet first-class in reports or routing
- strategy memory now carries parent-linked lessons and reuse surfaces, but it still needs integrated retained evidence proving those priors improve unattended outcomes
- counted trust breadth and retrieval carryover are not yet broadly proven
- transparent unattended attach or supersede semantics are still not complete

## Active Steward Claim

- worker_name: `Codex`
- lane_id: `lane_integration_steward`
- claimed_at: `2026-04-09T17:55:45+00:00`
- run_root:
  `/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun`
- current_round: `1`
- question:
  `verify the merged branch, repair any integration regressions blocking steward acceptance, and produce the official rerun closeout`
- owned_paths:
  - [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
  - [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
- verification_plan:
  - `pytest -x tests/test_unattended_campaign.py tests/test_finalize_cycle.py tests/test_improvement.py tests/test_strategy_memory.py tests/test_curriculum.py tests/test_unattended_controller.py`
  - `python - <<'PY'\nfrom pathlib import Path\nfor path in [Path('docs/unattended_work_queue.md'), Path('docs/coding_agi_gap_map.md'), Path('docs/runtime.md')]:\n    assert path.exists(), path\nprint('docs-ok')\nPY`

## Non-Negotiable Rules

1. Do not let two terminals edit the same owned path at the same time.
2. If a lane needs a shared file outside its write scope, it must stage an interface request in this
   document instead of editing the file directly.
3. Every lane must add or update focused regression tests in its owned test files.
4. No lane may silently broaden its ownership. Ownership changes must be written here first.
5. No lane may launch a new unattended validation rerun until the merge steward says the integration
   branch is ready.
6. All lanes must preserve current report compatibility unless the interface contract below says a
   schema change is allowed.

## Canonical Milestones

This wave is only successful when one integrated unattended rerun shows all of the following:

1. runtime-managed decisions are mostly child-native rather than mostly controller-closed
2. progress state is coherent across the entire run and does not disagree on task identity
3. at least one selected intervention has origin `discovered_strategy`
4. at least one retained discovered ontology entry reloads through the catalog layer
5. reports and routing expose discovered structural work classes in addition to authored families
6. strategy memory is parent-linked and later selection can cite retained prior nodes
7. integration, project, repository, and repo_chore each show counted trusted evidence
8. retrieval improvements demonstrably change later repair or verification behavior

## Global Interface Contracts

These contracts exist so lanes can work in parallel without editing the same files.

### Contract A: `ProgressState`

Authoritative shape for live unattended child progress:

- `phase`
- `phase_family`
- `task_scope`
- `task_position`
- `status`
- `progress_class`
- `decision_distance`
- `detail`
- `recorded_at`
- `expected_next_transition_at`
- `confidence`

Notes:

- `task_scope` must be stable across mirrored status, live child memory, and final reports
- `task_position` must resolve `task X/Y` ambiguity into machine-comparable fields
- all consumers must treat the newest authoritative record as canonical

Owned by:

- `lane_runtime_authority`

Consumers:

- `lane_decision_closure`
- `lane_strategy_invention`
- `lane_repo_generalization`

### Contract B: `DecisionState`

Authoritative shape for unattended round closeout:

- `decision_owner`
- `decision_credit`
- `decision_conversion_state`
- `retention_state`
- `retention_basis`
- `closeout_mode`
- `controller_intervention_reason_code`
- `recorded_at`

Required conventions:

- `decision_owner` must distinguish `child_native`, `controller_runtime_manager`, and `none`
- `closeout_mode` must distinguish `natural`, `child_native_before_partial_timeout`, `partial_timeout_evidence_only`, and `forced_reject`
- reports must expose both raw child state and authoritative round state without conflating them

Owned by:

- `lane_decision_closure`

Consumers:

- all lanes

### Contract C: `DiscoveredStrategy`

Authoritative shape for runtime-invented intervention classes:

- `strategy_id`
- `origin`
- `generation_basis`
- `target_conditions`
- `controls`
- `expected_signals`
- `parent_strategy_node_ids`
- `retention_inputs`

Required conventions:

- `origin` must be `discovered_strategy`
- strategy ids must be stable enough to recur across rounds
- retained and rejected outcomes must both remain attributable to the strategy id

Owned by:

- `lane_strategy_invention`

Consumers:

- `lane_strategy_memory`
- `lane_ontology_expansion`

### Contract D: `RetainedOntologyExtension`

Authoritative shape for runtime-discovered ontology entries:

- `extension_id`
- `namespace`
- `kind`
- `schema_version`
- `declared_fields`
- `validator_key`
- `retention_state`
- `provenance`

Required conventions:

- extensions must be namespaced outside the core authored ontology
- unknown keys must still fail closed unless they are approved retained extensions

Owned by:

- `lane_ontology_expansion`

Consumers:

- `lane_strategy_invention`
- `lane_repo_generalization`

### Contract E: `DiscoveredStructuralClass`

Authoritative shape for repo-level generalization labels:

- `class_id`
- `class_kind`
- `derivation_basis`
- `repo_topology_signals`
- `verifier_shape_signals`
- `edit_pattern_signals`
- `workflow_shape_signals`

Required conventions:

- discovered classes supplement authored families and do not replace them
- derivation must be deterministic enough for regression tests

Owned by:

- `lane_repo_generalization`

Consumers:

- `lane_trust_and_carryover`
- `lane_strategy_memory`

### Contract F: `StrategyNode`

Authoritative shape for persistent strategy memory:

- `strategy_node_id`
- `parent_strategy_node_ids`
- `cycle_id`
- `strategy_id`
- `strategy_candidate_kind`
- `motivation`
- `controls`
- `actor_summary`
- `results_summary`
- `retention_state`
- `retained_gain`
- `analysis_lesson`
- `reuse_conditions`
- `avoid_conditions`
- `family_coverage`
- `artifact_paths`

Owned by:

- `lane_strategy_memory`

Consumers:

- `lane_strategy_invention`
- `lane_trust_and_carryover`

## Parallel Lanes

Each lane below is claimable by one Codex terminal. The write scopes are intentionally disjoint.

### `lane_runtime_authority`

- priority: `P0`
- suggested_agent_name: `Codex-Runtime`
- branch_name: `lane/runtime-authority`
- objective:
  make unattended runtime state authoritative, freshness-aware, and schema-stable before higher
  level autonomy work lands on top of it
- owned_paths:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- explicitly_not_owned:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
- required_outputs:
  - canonical `ProgressState`
  - task identity reconciliation across mirrored and live status
  - no stale semantic/adaptive/task-position disagreement
- first_patches:
  - centralize all child progress merging on one authority function
  - normalize `task_scope` and `task_position`
  - reconcile status/report phase detail to the newest authoritative progress record
  - make confidence and freshness explicit in emitted progress state
- verification_plan:
  - `python -m py_compile scripts/run_unattended_campaign.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_campaign.py -k "progress_state or semantic_progress or mirrored_child or freshness or task_scope"`
- done_when:
  - no final artifact can show conflicting task position for semantic vs adaptive progress
  - all emitted progress states carry comparable authoritative timestamps
- restart_requirement: `yes`
- lane: `lane_runtime_authority`
- branch: `main`
- files_changed:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py)
- interface_contracts_touched:
  - `Contract A: ProgressState`
- tests_run:
  - `python -m py_compile scripts/run_unattended_campaign.py tests/test_unattended_campaign.py`
  - `pytest -q tests/test_unattended_campaign.py -k "write_status_projects_live_child_signal_top_level or live_status_projection_prefers_normalized_campaign_report_for_authoritative_decision_state or accepted_productive_partial_timeout or canonicalizes_stale_adaptive_state_behind_live_task or prefers_canonical_progress_state_over_stale_semantic_state or prefers_fresher_adaptive_state_over_stale_semantic_state or prefers_adaptive_state_when_semantic_detail_lags_task_position or preserves_newer_live_task_over_stale_mirrored_snapshot or reconciles_semantic_phase_to_live_generated_task or preferred_active_child_phase_detail_replaces_stale_same_phase_task_line"`
  - `pytest -q tests/test_unattended_campaign.py -k "live_status_projection or write_status or merge_mirrored_child_status_fields or accepted_productive_partial_timeout"`
- restart_required: `no`
- blockers: `none`
- merge_notes:
  - status/report projection now derives a canonical child progress state before export instead of letting semantic and adaptive progress disagree on live task position
  - `_write_status` now uses mirrored child state for projection when no fresh `active_child` exists, without mutating passive mirrored snapshots inside `active_run.child_status`
  - decision summary merging now preserves historical non-runtime-managed counts alongside runtime-managed closure instead of forcing them to zero

### `lane_decision_closure`

- priority: `P0`
- suggested_agent_name: `Codex-Closure`
- branch_name: `lane/decision-closure`
- objective:
  make runtime-managed decisions child-native by default and controller-closed only as an explicit,
  auditable fallback
- owned_paths:
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
- explicitly_not_owned:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- required_outputs:
  - canonical `DecisionState`
  - explicit `child_native` vs `controller_runtime_manager` decision ownership
  - explicit `closeout_mode`
- first_patches:
  - make productive rounds emit retain, reject, or typed incomplete state from the child path
  - separate closeout basis from mere productive activity
  - record controller closure as fallback rather than defaulting it into the same bucket
- verification_plan:
  - `python -m py_compile agent_kernel/cycle_runner.py scripts/run_repeated_improvement_cycles.py tests/test_finalize_cycle.py`
  - `pytest -q tests/test_finalize_cycle.py -k "decision_state or child_native or controller_runtime_manager or closeout_mode"`
- completion_update:
  - accepted-partial timeout closeout now promotes canonical `decision_state` into final campaign report surfaces
  - explicit priority benchmark campaigns now keep preview retention on the same primary lane and suppress generated/recovery preview lanes
  - final retention preview now inherits explicit priority restrictions even when there is no sibling-preview reuse path
  - prior-retained retention comparisons now preserve explicit priority restrictions instead of letting artifact preview controls re-admit retrieval or candidate lanes
  - fallback campaign selection no longer re-admits explicitly excluded subsystems
  - verified with targeted closeout, priority-selection, and preview-wiring tests
- done_when:
  - round artifacts can report both raw child incompletion and authoritative round closeout without contradiction
  - metrics can distinguish child-native decisions from controller-closed ones
- restart_requirement: `yes`

### `lane_strategy_invention`

- priority: `P0`
- suggested_agent_name: `Codex-Strategy`
- branch_name: `lane/strategy-invention`
- objective:
  make the planner synthesize and select runtime-discovered intervention classes instead of only
  choosing among authored subsystem tactics
- owned_paths:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`agent_kernel/universe_improvement.py`](/data/agentkernel/agent_kernel/universe_improvement.py)
  - [`datasets/improvement_catalog.json`](/data/agentkernel/datasets/improvement_catalog.json)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
- explicitly_not_owned:
  - [`agent_kernel/strategy_memory/`](/data/agentkernel/agent_kernel/strategy_memory)
  - [`agent_kernel/kernel_catalog.py`](/data/agentkernel/agent_kernel/kernel_catalog.py)
- required_outputs:
  - canonical `DiscoveredStrategy`
  - planner portfolio that can rank authored and discovered strategies together
  - retention attribution for discovered strategies
- first_patches:
  - add a discovery path based on repeated failure motifs, transfer gaps, and repo signals
  - persist discovered strategy artifacts with stable ids and provenance
  - teach portfolio selection to choose a discovered strategy when fixed lanes are low-yield
- verification_plan:
  - `python -m py_compile agent_kernel/improvement.py agent_kernel/universe_improvement.py tests/test_improvement.py`
  - `pytest -q tests/test_improvement.py -k "discovered_strategy or select_portfolio_campaign or retention"`
- done_when:
  - a repeated cycle can emit a selected intervention with `origin=discovered_strategy`
  - retained and rejected outcomes remain attributable to the strategy id
- restart_requirement: `yes`
- completion_update:
  - fallback campaign selection now enriches raw ranked-experiment fallbacks with canonical strategy candidates before record/finalize
  - excluded-subsystem fallback paths no longer produce anonymous strategy-memory nodes with empty `strategy_candidate_id`/`strategy_candidate_kind`
  - primary-only broad-observe lanes can now emit `origin=discovered_strategy` instead of requiring generated-task coverage before strategy discovery is credited
  - strategy memory nodes and snapshots now preserve `strategy_origin`, so retained discovered strategies can be reloaded and counted by later selectors
  - canonical cycle records now promote `strategy_origin` at the top level and in `metrics_summary`, matching `strategy_candidate_id`/`strategy_candidate_kind` on observe, select, generate, sibling preview/evaluation, final closeout, retention-outcome, and final report records
  - live detached validation `/data/agentkernel/var/unattended_run_20260410T182000Z_asi_lineage_surface_validation` proved sibling preview/evaluation rows now carry `strategy_origin=discovered_strategy`; it also exposed the remaining closeout-row lineage gap on `finalize_cycle` and `persist_retention_outcome`
  - closeout-row lineage is patched and covered by `tests/test_finalize_cycle.py::test_finalize_cycle_promotes_strategy_lineage_to_closeout_records`; live validation is running under `/data/agentkernel/var/unattended_run_20260410T191100Z_asi_closeout_lineage_validation`
  - verified with fallback attribution, discovered-strategy planner, and strategy-memory tests

### `lane_ontology_expansion`

- priority: `P0`
- suggested_agent_name: `Codex-Ontology`
- branch_name: `lane/ontology-expansion`
- objective:
  make improvement meaning runtime-extensible so discovered categories can be retained and reused
  without hand-editing every catalog path
- owned_paths:
  - [`agent_kernel/kernel_catalog.py`](/data/agentkernel/agent_kernel/kernel_catalog.py)
  - [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json)
  - [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py)
- explicitly_not_owned:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`datasets/improvement_catalog.json`](/data/agentkernel/datasets/improvement_catalog.json)
- required_outputs:
  - canonical `RetainedOntologyExtension`
  - retained-extension namespace
  - validator behavior that distinguishes unsafe unknown keys from approved retained extensions
- first_patches:
  - separate core authored ontology from retained discovered ontology
  - add extension loading and validation hooks
  - add reversible retention metadata for discovered entries
- verification_plan:
  - `python -m py_compile agent_kernel/kernel_catalog.py tests/test_improvement.py`
  - `pytest -q tests/test_improvement.py -k "ontology or catalog or retained_extension"`
- done_when:
  - a runtime-generated ontology entry can be reloaded through the catalog layer
  - unsafe unknown keys still fail closed
- restart_requirement: `yes`

### `lane_repo_generalization`

- priority: `P1`
- suggested_agent_name: `Codex-Generalization`
- branch_name: `lane/repo-generalization`
- objective:
  add discovered structural work classes so routing and reporting can generalize beyond authored
  benchmark families
- owned_paths:
  - [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
  - [`agent_kernel/modeling/policy/decoder.py`](/data/agentkernel/agent_kernel/modeling/policy/decoder.py)
  - [`scripts/run_job_queue.py`](/data/agentkernel/scripts/run_job_queue.py)
  - [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
- explicitly_not_owned:
  - [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
- required_outputs:
  - canonical `DiscoveredStructuralClass`
  - controller features that can consume discovered classes
  - decoder routing that can fall back to discovered classes when family evidence is weak
- first_patches:
  - derive deterministic structural classes from repo topology, verifier shape, edit pattern, and workflow shape
  - add discovered classes to controller observations and task/run reports
  - let routing consume discovered classes without removing existing family logic
- verification_plan:
  - `python -m py_compile agent_kernel/unattended_controller.py agent_kernel/modeling/policy/decoder.py scripts/run_job_queue.py tests/test_unattended_controller.py`
  - `pytest -q tests/test_unattended_controller.py -k "generalization or discovered_class or reward"`
- done_when:
  - reports and controller state contain discovered classes alongside authored families
  - routing can use discovered classes when family evidence is sparse
- restart_requirement: `yes`

### `lane_strategy_memory`

- priority: `P1`
- suggested_agent_name: `Codex-Memory`
- branch_name: `lane/strategy-memory`
- objective:
  upgrade strategy memory from aggregate priors to parent-linked reusable experiment memory with
  lessons and best snapshots
- owned_paths:
  - [`agent_kernel/strategy_memory/__init__.py`](/data/agentkernel/agent_kernel/strategy_memory/__init__.py)
  - [`agent_kernel/strategy_memory/node.py`](/data/agentkernel/agent_kernel/strategy_memory/node.py)
  - [`agent_kernel/strategy_memory/store.py`](/data/agentkernel/agent_kernel/strategy_memory/store.py)
  - [`agent_kernel/strategy_memory/sampler.py`](/data/agentkernel/agent_kernel/strategy_memory/sampler.py)
  - [`agent_kernel/strategy_memory/lesson.py`](/data/agentkernel/agent_kernel/strategy_memory/lesson.py)
  - [`agent_kernel/strategy_memory/snapshot.py`](/data/agentkernel/agent_kernel/strategy_memory/snapshot.py)
  - [`agent_kernel/actors/coding.py`](/data/agentkernel/agent_kernel/actors/coding.py)
  - [`evals/harness.py`](/data/agentkernel/evals/harness.py)
  - [`tests/test_strategy_memory.py`](/data/agentkernel/tests/test_strategy_memory.py)
- explicitly_not_owned:
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
- required_outputs:
  - canonical `StrategyNode`
  - lesson synthesis
  - best retained snapshots
  - parent-linkable sampling surfaces
- first_patches:
  - enrich nodes with actor summary, results summary, reuse conditions, and avoid conditions
  - add best snapshot surfaces by subsystem, family cluster, and strategy kind
  - make sampler outputs usable by planner and controller without requiring shared-file edits
- verification_plan:
  - `python -m py_compile agent_kernel/strategy_memory/__init__.py agent_kernel/strategy_memory/node.py agent_kernel/strategy_memory/store.py agent_kernel/strategy_memory/sampler.py agent_kernel/strategy_memory/lesson.py agent_kernel/strategy_memory/snapshot.py agent_kernel/actors/coding.py evals/harness.py tests/test_strategy_memory.py`
  - `pytest -q tests/test_strategy_memory.py`
- done_when:
  - retained and rejected nodes both produce reusable lessons
  - snapshot surfaces expose best retained nodes without reconstructing them from mixed reports
- restart_requirement: `yes`
- lane: `lane_strategy_memory`
- branch: `lane/strategy-memory`
- files_changed:
  - [`agent_kernel/strategy_memory/__init__.py`](/data/agentkernel/agent_kernel/strategy_memory/__init__.py)
  - [`agent_kernel/strategy_memory/node.py`](/data/agentkernel/agent_kernel/strategy_memory/node.py)
  - [`agent_kernel/strategy_memory/store.py`](/data/agentkernel/agent_kernel/strategy_memory/store.py)
  - [`agent_kernel/strategy_memory/sampler.py`](/data/agentkernel/agent_kernel/strategy_memory/sampler.py)
  - [`agent_kernel/strategy_memory/snapshot.py`](/data/agentkernel/agent_kernel/strategy_memory/snapshot.py)
  - [`tests/test_strategy_memory.py`](/data/agentkernel/tests/test_strategy_memory.py)
- interface_contracts_touched:
  - `Contract F: StrategyNode`
- tests_run:
  - `python -m py_compile agent_kernel/strategy_memory/__init__.py agent_kernel/strategy_memory/node.py agent_kernel/strategy_memory/store.py agent_kernel/strategy_memory/sampler.py agent_kernel/strategy_memory/lesson.py agent_kernel/strategy_memory/snapshot.py agent_kernel/actors/coding.py evals/harness.py tests/test_strategy_memory.py`
  - `pytest -q tests/test_strategy_memory.py`
  - `pytest -q tests/test_actors.py`
  - `pytest -q tests/test_eval.py -k "run_kernel_task_attaches_coding_actor_summary_for_light_supervision_candidate"`
- restart_required: `yes`
- blockers: `none`
- merge_notes:
  - strategy memory now records canonical `strategy_id`, preserves richer retained snapshots, and links descendants across all declared parents
  - sampler outputs now expose direct parent-node payloads and best-retained snapshots without requiring consumers to reconstruct them from mixed reports

### `lane_trust_and_carryover`

- priority: `P1`
- suggested_agent_name: `Codex-Trust`
- branch_name: `lane/trust-and-carryover`
- objective:
  convert family sampling into counted trusted breadth and make retrieval improvements survive as
  later repair behavior
- owned_paths:
  - [`agent_kernel/trust.py`](/data/agentkernel/agent_kernel/trust.py)
  - [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
  - [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
  - [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py)
  - [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py)
  - [`agent_kernel/context_budget.py`](/data/agentkernel/agent_kernel/context_budget.py)
  - [`scripts/run_supervisor_loop.py`](/data/agentkernel/scripts/run_supervisor_loop.py)
  - [`scripts/run_autonomous_compounding_check.py`](/data/agentkernel/scripts/run_autonomous_compounding_check.py)
  - [`tests/test_curriculum.py`](/data/agentkernel/tests/test_curriculum.py)
- explicitly_not_owned:
  - [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
- required_outputs:
  - counted trust summaries that distinguish sampled, verified, retained, and clean-root breadth
  - retrieval carryover metrics tied to later repair or verification behavior
  - routing pressure toward underproven required families
- first_patches:
  - add per-family counted-evidence summaries with clean-root distinctions
  - add retrieval carryover accounting between intervention and later repair steps
  - feed missing-family and low-carryover pressure into supervision and compounding checks
- verification_plan:
  - `python -m py_compile agent_kernel/trust.py agent_kernel/task_bank.py agent_kernel/curriculum.py agent_kernel/retrieval_improvement.py agent_kernel/policy.py agent_kernel/context_budget.py scripts/run_supervisor_loop.py scripts/run_autonomous_compounding_check.py tests/test_curriculum.py`
  - `pytest -q tests/test_curriculum.py -k "trust or carryover or priority_family or retrieval"`
- done_when:
  - integration, project, repository, and repo_chore can be measured as counted trusted evidence
  - retrieval interventions can be tied to downstream repair or verify improvements
- restart_requirement: `yes`

### `lane_integration_steward`

- priority: `P0`
- suggested_agent_name: `Codex-Steward`
- branch_name: `lane/integration-steward`
- objective:
  keep the integration branch coherent, manage interface changes, and run the single official
  end-to-end unattended rerun after lane merges are complete
- owned_paths:
  - [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
  - [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
  - [`docs/runtime.md`](/data/agentkernel/docs/runtime.md)
- explicitly_not_owned:
  - all product code paths unless a lane explicitly hands them off
- required_outputs:
  - updated queue state
  - accepted interface changes
  - official integration rerun and final readout
- first_patches:
  - maintain claim status and blockers in this file
  - reject unsafe ownership creep
  - run the final integration rerun only after all lane acceptance checks pass
- verification_plan:
  - `python - <<'PY'\nfrom pathlib import Path\nfor path in [Path('docs/unattended_work_queue.md'), Path('docs/coding_agi_gap_map.md'), Path('docs/runtime.md')]:\n    assert path.exists(), path\nprint('docs-ok')\nPY`
- done_when:
  - final integration rerun is complete and milestone gate below is evaluated
- restart_requirement: `yes`
- lane: `lane_integration_steward`
- branch: `main`
- files_changed:
  - [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  - [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  - [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  - [`evals/harness.py`](/data/agentkernel/evals/harness.py)
  - [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)
  - [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  - [`docs/unattended_work_queue.md`](/data/agentkernel/docs/unattended_work_queue.md)
  - [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
- interface_contracts_touched:
  - `Contract A: ProgressState`
  - `Contract B: DecisionState`
- tests_run:
  - `pytest -x tests/test_unattended_campaign.py tests/test_finalize_cycle.py tests/test_improvement.py tests/test_strategy_memory.py tests/test_curriculum.py tests/test_unattended_controller.py`
  - `pytest -vv -x tests/test_unattended_controller.py`
  - `pytest -q tests/test_finalize_cycle.py -k "test_run_repeated_improvement_cycles_reconciles_incomplete_autonomous_cycle_before_report"`
  - `pytest -q tests/test_finalize_cycle.py -k "test_run_improvement_cycle_observation_eval_kwargs_caps_observe_probe_but_preserves_campaign_task_limit or test_run_improvement_cycle_observation_eval_enables_existing_runtime_surface or test_run_improvement_cycle_records_selected_variant or test_run_improvement_cycle_defaults_to_observing_present_abstraction_artifacts or test_run_improvement_cycle_executes_competitive_campaign"`
  - `pytest -q tests/test_finalize_cycle.py -k "test_run_improvement_cycle_uses_portfolio_campaign_selection or test_run_improvement_cycle_broadens_campaign_with_distinct_ranked_subsystems or test_run_improvement_cycle_executes_competitive_campaign or test_run_improvement_cycle_generates_unique_cycle_ids"`
  - `pytest -q tests/test_finalize_cycle.py -k "test_autonomous_phase_gate_report_rejects_missing_generated_lane_output or test_finalize_cycle_materializes_retrieval_asset_bundle_on_retain"`
  - `pytest -q tests/test_finalize_cycle.py -k "test_run_improvement_cycle_reconciles_prior_incomplete_autonomous_cycles"`
  - `pytest -x tests/test_curriculum.py`
  - `pytest -vv -x tests/test_curriculum.py -k "test_generated_eval_tracks_generated_benchmark_family"` did not complete within this session
  - `python - <<'PY'\nfrom pathlib import Path\nfor path in [Path('docs/unattended_work_queue.md'), Path('docs/coding_agi_gap_map.md'), Path('docs/runtime.md')]:\n    assert path.exists(), path\nprint('docs-ok')\nPY`
- restart_required: `yes`
- blockers:
  - `official integrated unattended rerun was not launched from this checkout because a full merged-branch pass of tests/test_curriculum.py was not completed; the isolated tail remained long-running at test_generated_eval_tracks_generated_benchmark_family`
- merge_notes:
  - steward repaired six narrow post-merge regressions across runtime progress mirroring, replay-verified tooling compatibility, repeated-cycle decision-state reconciliation, scoped eval seeding, planner-runtime path isolation and campaign budgeting, and retrieval-family autonomous phase gating
  - repeated broad sweeps were clean through `tests/test_unattended_campaign.py`, `tests/test_finalize_cycle.py`, `tests/test_improvement.py`, and `tests/test_strategy_memory.py`, and a full `tests/test_unattended_controller.py` pass completed green
  - the best integrated artifact remains [`unattended_campaign.status.json`](/data/agentkernel/var/unattended_run_20260409T1620Z_authority_bundle_final_rerun/reports/unattended_campaign.status.json) with `status=completed`, `runtime_managed_decisions=1`, and `retained_gain_runs=1`, but milestone closure is still blocked by controller-closed decision credit, missing counted external breadth, and the absence of a completed final steward rerun`

## Write-Collision Matrix

If a path is listed here, only the named lane may edit it during this wave.

| Path or subtree | Sole owner |
| --- | --- |
| `scripts/run_unattended_campaign.py` | `lane_runtime_authority` |
| `agent_kernel/cycle_runner.py` | `lane_decision_closure` |
| `scripts/run_repeated_improvement_cycles.py` | `lane_decision_closure` |
| `agent_kernel/improvement.py` | `lane_strategy_invention` |
| `agent_kernel/universe_improvement.py` | `lane_strategy_invention` |
| `datasets/improvement_catalog.json` | `lane_strategy_invention` |
| `agent_kernel/kernel_catalog.py` | `lane_ontology_expansion` |
| `datasets/kernel_metadata.json` | `lane_ontology_expansion` |
| `agent_kernel/unattended_controller.py` | `lane_repo_generalization` |
| `agent_kernel/modeling/policy/decoder.py` | `lane_repo_generalization` |
| `scripts/run_job_queue.py` | `lane_repo_generalization` |
| `agent_kernel/strategy_memory/` | `lane_strategy_memory` |
| `agent_kernel/actors/coding.py` | `lane_strategy_memory` |
| `evals/harness.py` | `lane_strategy_memory` |
| `agent_kernel/trust.py` | `lane_trust_and_carryover` |
| `agent_kernel/task_bank.py` | `lane_trust_and_carryover` |
| `agent_kernel/curriculum.py` | `lane_trust_and_carryover` |
| `agent_kernel/retrieval_improvement.py` | `lane_trust_and_carryover` |
| `agent_kernel/policy.py` | `lane_trust_and_carryover` |
| `agent_kernel/context_budget.py` | `lane_trust_and_carryover` |
| `scripts/run_supervisor_loop.py` | `lane_trust_and_carryover` |
| `scripts/run_autonomous_compounding_check.py` | `lane_trust_and_carryover` |
| `docs/unattended_work_queue.md` | `lane_integration_steward` |

## Temporary Ownership Handoffs

- effective_at: `2026-04-09T17:55:45+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow integration-only fix in
  [`scripts/run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
  because the merged branch currently fails
  `tests/test_unattended_campaign.py::test_merge_mirrored_child_status_fields_prefers_authoritative_child_inputs`
  and the queued runtime-authority lane is not actively resolving it in this checkout
- scope:
  `restore authoritative active-cycle child-status precedence without broadening runtime-authority ownership beyond this regression`
- effective_at: `2026-04-09T18:18:00+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow backward-compatibility fix in
  [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py)
  because the merged branch currently fails
  [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  at `test_finalize_cycle_main_persists_evaluate_retain_and_record`
  after tooling artifact validation began requiring normalized
  `local_shell_procedure` fields that older retained-candidate fixtures do not carry
- scope:
  `normalize legacy tooling candidates into the stricter replay-verified schema without weakening the tooling compatibility contract for newly generated candidates`
- effective_at: `2026-04-09T18:28:00+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow reporting fix in
  [`scripts/run_repeated_improvement_cycles.py`](/data/agentkernel/scripts/run_repeated_improvement_cycles.py)
  because the merged branch currently fails
  [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  at `test_run_repeated_improvement_cycles_writes_report`
  after record-level decision ownership drifted away from the run-level closeout classification
- scope:
  `restore record-level decision_state inference so non-runtime-managed rejects remain credited to controller_runtime_manager without changing lane-owned retention policy behavior`
- effective_at: `2026-04-09T18:36:00+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow scoped-eval filesystem fix in
  [`evals/harness.py`](/data/agentkernel/evals/harness.py)
  because the merged branch currently fails
  [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  at `test_finalize_cycle_candidate_eval_preserves_unrelated_retained_proposal_toggles`
  after scoped artifact seeding started failing hard on an already-missing destination path
- scope:
  `make scoped file seeding idempotent for missing destinations without changing the scoped-config contract or copy behavior when files exist`
- effective_at: `2026-04-09T18:46:00+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow integration repair in
  [`scripts/run_improvement_cycle.py`](/data/agentkernel/scripts/run_improvement_cycle.py)
  because the merged branch currently fails
  [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  at `test_run_improvement_cycle_executes_competitive_campaign` and
  `test_run_improvement_cycle_observation_eval_kwargs_caps_observe_probe_but_preserves_campaign_task_limit`
  after planner-memory rooting, observation-eval wiring, and diversity-fill budgeting drifted apart
  during lane merges
- scope:
  `restore merged-branch planner-runtime isolation, observation task-limit plumbing, and campaign diversification budgeting without broadening ownership into unrelated planning behavior`
- effective_at: `2026-04-09T19:12:00+00:00`
- steward_routing:
  `lane_integration_steward` may apply a narrow finalize-path repair in
  [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py)
  because the merged branch currently fails
  [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py)
  at `test_finalize_cycle_materializes_retrieval_asset_bundle_on_retain`
  after autonomous phase gating began requiring generated-lane output fields that retrieval-family
  fake evals do not emit
- scope:
  `restore retrieval-family finalize compatibility without weakening the stricter generated-lane gate for direct policy outputs`

## Shared Test Ownership

Shared test files are the main collision risk. Use these rules:

- [`tests/test_unattended_campaign.py`](/data/agentkernel/tests/test_unattended_campaign.py):
  `lane_runtime_authority` only
- [`tests/test_finalize_cycle.py`](/data/agentkernel/tests/test_finalize_cycle.py):
  `lane_decision_closure` only
- [`tests/test_improvement.py`](/data/agentkernel/tests/test_improvement.py):
  `lane_strategy_invention` owns direct edits; `lane_ontology_expansion` must request additions
  through the steward unless a temporary handoff is recorded here first
- [`tests/test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py):
  `lane_repo_generalization` only
- [`tests/test_strategy_memory.py`](/data/agentkernel/tests/test_strategy_memory.py):
  `lane_strategy_memory` only
- [`tests/test_curriculum.py`](/data/agentkernel/tests/test_curriculum.py):
  `lane_trust_and_carryover` only

## Dependency Order

Development can proceed in parallel, but merges must land in this order:

1. `lane_runtime_authority`
2. `lane_decision_closure`
3. `lane_strategy_invention`
4. `lane_ontology_expansion`
5. `lane_repo_generalization`
6. `lane_strategy_memory`
7. `lane_trust_and_carryover`
8. `lane_integration_steward` rerun and docs closeout

Reason:

- progress state authority must exist before decision closure can be trusted
- decision closure must exist before discovered strategies can be evaluated cleanly
- discovered strategies need ontology support before the new categories can become reusable
- repo generalization and strategy memory both consume those stabilized strategy and ontology
  surfaces
- trust breadth and retrieval carryover should land last so they can measure the final integrated
  behavior rather than an intermediate state

## Branch Workflow

Each terminal should use this workflow:

1. create its lane branch from the current integration base
2. edit only owned paths
3. run the lane verification plan
4. post a short closeout note with:
   - files changed
   - tests run
   - interface changes requested
   - restart requirement
5. wait for steward approval before rebasing or merging

## Required Closeout Template

Each lane must close out with this exact structure:

- lane:
- branch:
- files_changed:
- interface_contracts_touched:
- tests_run:
- restart_required:
- blockers:
- merge_notes:

## Blocker Handling

If a lane is blocked by a file it does not own:

1. do not edit the file
2. add a blocker note under the lane section in this document
3. specify:
   - exact path
   - exact requested change
   - why the interface contract is insufficient
4. wait for steward routing

## Lane Status Board

Update only this section while the wave is active.

### `lane_runtime_authority`

- status: `merged`
- blocker: `none`

### `lane_decision_closure`

- status: `merged`
- blocker: `none`

### `lane_strategy_invention`

- status: `merged`
- blocker: `none`

### `lane_ontology_expansion`

- status: `merged`
- blocker: `none`

### `lane_repo_generalization`

- status: `merged`
- blocker: `none`

### `lane_strategy_memory`

- status: `merged`
- blocker: `none`

### `lane_trust_and_carryover`

- status: `merged`
- blocker: `none`

### `lane_integration_steward`

- status: `merged_with_followup`
- blocker:
  `official integration rerun deferred pending a completed merged-branch pass of tests/test_curriculum.py`

## Final Integration Gate

The steward may run the official unattended rerun only when all of the following are true:

1. every lane reports `status=merged` or `status=merged_with_followup`
2. all lane verification plans passed on the integration branch
3. no blocker remains open in the status board
4. interface contracts in this file still match the shipped code

Official rerun acceptance requires:

- one integrated unattended rerun from the merged branch
- a completed status artifact
- final readout against the eight canonical milestones above
- explicit statement of any remaining misses

## Definition Of Failure For This Wave

This wave fails if any of the following happen:

- two lanes edit the same owned path without a documented handoff
- the integration branch requires ad hoc emergency edits in a lane-owned file because ownership was
  underspecified here
- the official rerun still relies mainly on controller-closed retains while claiming decision
  closure is solved
- discovered strategies, ontology extensions, or discovered structural classes exist only in tests
  and not in real unattended artifacts
- trust breadth is claimed from sampling alone rather than counted trusted evidence

## Steward Notes

- this file intentionally replaces the previous queue instead of extending it
- if a new lane is needed, add it here with a sole-owner write scope before any code changes start
- if a lane finishes early, do not reuse its terminal for unrelated edits until the steward marks the
  lane merged
