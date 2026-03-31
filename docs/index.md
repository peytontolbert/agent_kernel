# Documentation

`agent-kernel` is a verifier-driven coding agent kernel built around a local LLM, a strict TOLBERT context compiler, and a replayable local task/eval harness.

Start here:

- `setup.md`: install, model setup, TOLBERT assets, and first commands
- `architecture.md`: core modules, control flow, and feature toggles
- `runtime.md`: policy execution, context packets, state shaping, and runtime flags
- `universe.md`: stable machine-level governance, retained universe contracts, and schema
- `tolbert_liftoff.md`: staged TOLBERT expansion, liftoff criteria, and modeling package boundary
- `tasks.md`: task schema, built-in task families, replay tasks, and curriculum behavior
- `scripts.md`: CLI entrypoints, extraction/proposal scripts, retention utilities, and eval helpers
- `improvement.md`: artifact extraction, proposal generation, retention, rollback, and campaign flow
- `ai_agent_status.md`: live AI agent ownership, supervised-loop status, and fresh-agent handoff instructions
- `product_roadmap.md`: the separate product benchmark for unattended delegated work
- `validation.md`: current test status, smoke checks, verification phases, and known failure points

Preferred live provider:

- provider: `vllm`
- model: `Qwen/Qwen3.5-9B`
- host: `http://127.0.0.1:8000`

Current code defaults from `KernelConfig`:

- provider: `ollama`
- provider: `vllm`
- model: `Qwen/Qwen3.5-9B`
- host: `http://127.0.0.1:11434`
- TOLBERT context: enabled
- skills: enabled
- graph memory: enabled
- world model: enabled
- planner: enabled
- role specialization: enabled
- prompt proposals: enabled
- curriculum proposals: enabled
- retrieval proposals: enabled

Current runtime loop:

`task -> optional setup -> state enrichment -> TOLBERT compile -> LLM decision -> sandbox execution -> verification -> episode memory`
