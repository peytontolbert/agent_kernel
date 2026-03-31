# Architecture

## Goal

The repository is still narrow in scope, but it is no longer just a two-task minimal loop. The current kernel combines:

- one local policy runtime
- one bounded shell sandbox
- deterministic verification
- TOLBERT-backed encoder/retrieval compilation
- episode, skill, operator, tool, and verifier memory artifacts
- adjacent-task and failure-driven curriculum generation
- a retained-artifact improvement loop

## Control flow

The main runtime path in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py) is:

`task -> setup commands -> graph/world/planner state -> TOLBERT context -> policy decision -> action -> verification -> episode save`

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
- [`agent_kernel/extractors.py`](/data/agentkernel/agent_kernel/extractors.py): promoted skill and tool artifact extraction
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py): target ranking, artifact lifecycle metadata, and cycle records
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py): candidate-vs-baseline retention flow
- [`agent_kernel/prompt_improvement.py`](/data/agentkernel/agent_kernel/prompt_improvement.py): prompt proposal generation
- [`agent_kernel/retrieval_improvement.py`](/data/agentkernel/agent_kernel/retrieval_improvement.py): retrieval proposal generation
- [`agent_kernel/curriculum_improvement.py`](/data/agentkernel/agent_kernel/curriculum_improvement.py): curriculum proposal generation
- [`evals/harness.py`](/data/agentkernel/evals/harness.py): multi-task eval and comparison modes

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
