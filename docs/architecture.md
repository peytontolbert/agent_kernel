# Architecture

## Goal

The repository is a verifier-driven coding runtime for realistic local software work, not only a seed-task loop. The current kernel combines:

- one local policy runtime
- one bounded shell sandbox
- deterministic verification
- TOLBERT-backed encoder/retrieval compilation
- episode, skill, operator, tool, and verifier memory artifacts
- adjacent-task and failure-driven curriculum generation
- a retained-artifact improvement loop
- unattended and shared-repo workflow machinery for repo review, test repair, worker-branch coordination, and governed promotion

For the current evidence-ranked coding-AGI blockers, see
[`coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md).

## Control flow

The main runtime path in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py) is:

`task -> setup commands -> graph/world/planner state -> TOLBERT context -> policy decision -> action -> verification -> episode save`

The intended inner control loop is:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

In the current implementation that maps to graph and retrieval context, latent/state estimation, world-model refresh, rollout-scored candidate selection, role-specialized planner/executor/critic policy routing, verifier checks, subgoal diagnosis, and episode-memory persistence.

The intended outer self-improvement loop is:

`run tasks -> collect outcomes -> compare against verifiers -> localize failure -> modify policy/planner/world-model/transition-model/prompts/tools -> retest -> retain only verified gains`

In this repo that outer loop is implemented through eval-driven subsystem selection, candidate artifact generation, candidate-vs-baseline comparison, retention gates, rollback, and unattended campaign control.

Allowed actions remain:

- `respond`
- `code_execute`

In practice, task completion is driven by `code_execute`; `respond` is mainly used for policy errors or explicit termination.

## Main modules

- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py): runtime defaults, feature flags, and artifact paths
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py): task execution loop and step recording
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py): structured decision policy, skill loading, retrieval guidance use
- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py): seed TOLBERT compiler wrapper
- [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py): subprocess TOLBERT service
- [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling): isolated modeling code for TOLBERT-family checkpoints, training, and evaluation
- [`agent_kernel/llm.py`](/data/agentkernel/agent_kernel/llm.py): Ollama and mock clients
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py): deterministic task verification and stricter-contract synthesis
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py): episode persistence and graph summaries
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py): lightweight world-model summaries
- [`agent_kernel/multi_agent.py`](/data/agentkernel/agent_kernel/multi_agent.py): planner/executor/reviewer role coordination
- [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py): built-in tasks plus replay-task loaders
- [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py): success-adjacent and failure-recovery followups
- [`datasets/curriculum_templates.json`](/data/agentkernel/datasets/curriculum_templates.json): data-backed curriculum task templates plus long-horizon routing metadata rendered by [`agent_kernel/curriculum_catalog.py`](/data/agentkernel/agent_kernel/curriculum_catalog.py)
- [`datasets/improvement_catalog.json`](/data/agentkernel/datasets/improvement_catalog.json): data-backed improvement schema, artifact-contract registry, artifact-validation profiles, retention-gate presets, and runtime-metadata catalog rendered by [`agent_kernel/improvement_catalog.py`](/data/agentkernel/agent_kernel/improvement_catalog.py)
- [`datasets/kernel_metadata.json`](/data/agentkernel/datasets/kernel_metadata.json): data-backed kernel registry metadata for subsystem specs, capability adapters, task-bank lineage filters, unattended-controller feature catalogs, Tolbert policy thresholds, trust-family defaults, unattended runtime policy defaults, and frontier-family routing defaults rendered by [`agent_kernel/kernel_catalog.py`](/data/agentkernel/agent_kernel/kernel_catalog.py)
- [`agent_kernel/extractors.py`](/data/agentkernel/agent_kernel/extractors.py): promoted skill and tool artifact extraction
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py): target ranking, artifact lifecycle metadata, and cycle records
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py): candidate-vs-baseline retention flow
- [`agent_kernel/prompt_improvement.py`](/data/agentkernel/agent_kernel/prompt_improvement.py): prompt proposal generation
- [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py): retrieval proposal generation
- [`agent_kernel/curriculum_improvement.py`](/data/agentkernel/agent_kernel/curriculum_improvement.py): curriculum proposal generation
- [`evals/harness.py`](/data/agentkernel/evals/harness.py): multi-task eval and comparison modes
- [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md): compact evidence-backed map from the current unattended run to the next kernel surfaces to patch

## Runtime features

The code now has feature toggles for:

- TOLBERT context
- skill memory
- graph memory
- world model summaries
- planner-generated subgoals
- role specialization
- prompt proposal application
- curriculum proposal application
- retrieval proposal application

Those toggles are all enabled by default in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py), and most are surfaced in [`scripts/run_agent.py`](/data/agentkernel/scripts/run_agent.py) or [`scripts/run_eval.py`](/data/agentkernel/scripts/run_eval.py).

## Repository shape

```text
agent_kernel/
docs/
evals/
prompts/
scripts/
tests/
trajectories/
workspace/
var/
```

## TOLBERT integration

The runtime compiles a typed `ContextPacket` before each policy call. That packet includes:

- a hierarchical path prediction
- retrieval bundles from branch-scoped and global search
- verifier contract details
- control metadata such as path confidence and retrieval guidance

That retrieval path is no longer strictly stateless across steps. The compiler
can now carry forward a previously selected or trusted retrieval command into
the next compile when the agent did not actually use it yet, so retrieval
selection has a direct path to later action influence instead of collapsing back
to a fresh threshold-only query on every step.

That same signal now persists past the episode boundary too. Learning artifacts
compiled from verified runs retain whether a reusable procedure was
retrieval-backed or trusted-retrieval-backed, and TolBERT reuses that stronger
provenance on the next similar task instead of flattening it back into ordinary
same-family memory.

The promoted artifact path now preserves that provenance as well. Extracted
skill and tool procedures keep retrieval-backed metadata and prefer
trusted-retrieval-backed variants during dedupe, so retained artifacts do not
erase the difference between ordinary replay and verifier-proven retrieval use.
The compare and frontier-reporting path now carries the same retrieval-reuse
summary forward, so promotion ranking and retained-baseline review can reward
verifier-proven retrieval reuse explicitly instead of only through extracted
procedure quality. The supervisor bootstrap-review path now carries that same
summary onto first-retain queue entries and gives it a small review-priority
bias, so verifier-proven retrieval reuse can surface earlier for operator
attention when trust and bootstrap gates are otherwise equal. That bootstrap
priority now persists briefly across supervisor rounds too, so a subsystem that
keeps surfacing verifier-proven retrieval-backed first-retain evidence does not
lose review priority after one quiet plan snapshot. Separately, when a policy
candidate is blocked specifically on missing generated evidence, the supervisor
now launches a targeted curriculum-backed discovery rerun for that same
subsystem and variant instead of only recycling the candidate through generic
bootstrap review.

The implementation is intentionally strict:

- it launches a real `tolbert-brain` subprocess service
- it treats that service as owned runtime state and closes it when the enclosing kernel is torn down
- it requires a checkpoint, ontology nodes, source spans, cache shards, and usually a label map
- it does not silently fall back when ontology, logits, or retrieval inputs are inconsistent

The current runtime contract is still seed-stage:

- the live TOLBERT service compiles context and can influence bounded direct decisions
- the retained TOLBERT family is defined more broadly as encoder, latent dynamics, decoder, and world-model surfaces
- the seed LLM remains the authoritative free-form decoder
- future model-native takeover work should live under [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling) rather than inside the core loop modules

The staged takeover plan is documented in [`tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md).

## Supporting subsystems

The repo now includes subsystems that earlier docs called non-goals:

- role-specialized decision routing
- lightweight planner/executor/reviewer coordination
- replay-task generation from skill, operator, tool, benchmark, verifier, and episode artifacts
- prompt/curriculum/retrieval/verifier/tooling improvement artifacts

What it still does not try to be:

- a general browser agent
- a remote execution platform
- a distributed multi-agent system
- an internet-enabled research stack
