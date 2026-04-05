# Documentation

`agent-kernel` is a verifier-driven coding agent kernel built around a local LLM, a strict TOLBERT context compiler, and a replayable local task/eval harness.

Start here:

- `setup.md`: install, model setup, TOLBERT assets, and first commands
- `architecture.md`: core modules, control flow, and feature toggles
- `runtime.md`: policy execution, context packets, state shaping, and runtime flags
- `unattended_controller.md`: outer-loop round planning, support-aware exploration, and controller diagnostics
- `unattended_parallel_codex_runbook.md`: coordinator and replica workflow for parallel Codex support during live unattended campaigns
- `unattended_work_queue.md`: live lane claims and closeouts for the current unattended parallel Codex wave
- `coding_agi_gap_map.md`: compact evidence-backed map from unattended-run failures to the next kernel surfaces
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

- provider: `vllm`
- model: `Qwen/Qwen3.5-9B`
- host: `http://127.0.0.1:8000`
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

Canonical inner task loop:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

Canonical outer improvement loop:

`run tasks -> collect outcomes -> compare against verifiers -> localize failure -> modify policy/planner/world-model/transition-model/prompts/tools -> retest -> retain only verified gains`
