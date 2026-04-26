# Documentation

`agent-kernel` is a verifier-driven coding agent repo whose full surface includes a local LLM runtime, an optional TOLBERT context path, and a replayable local task/eval harness.

Start here:

- `asi_core.md`: canonical boundary between the ASI-oriented kernel and auxiliary enhancement layers
- `minimal_asi_qwen_tolbert.md`: paper-style definition of the ASI machine contract plus the Qwen + TOLBERT + vLLM reference implementation
- `kernel_gap_dashboard.md`: single current dashboard for live bottlenecks, substantive kernel gaps, and proof gaps
- `asi_closure_map.md`: current root-only closure map for the remaining substantive ASI-core gaps
- `asi_boundary_auxiliary_audit.md`: completed Lane 4 audit of core versus support boundaries
- `minimal_asi_analysis_lanes.md`: parallel lane split for fact-checking executable, runtime, meta-loop, and boundary completeness
- `live_evidence_proof_gap_audit.md`: completed Lane 5 audit of implemented, tested, detached-live, and proven-live evidence
- `setup.md`: install, model setup, TOLBERT assets, and first commands
- `architecture.md`: core modules, control flow, feature toggles, and current engine/plugin/adapter seams
- `resource_protocol.md`: RSPL-lite resource descriptors, registry loading, and the first runtime consumers
- `self_evolution_protocol.md`: SEPL-lite reflection and selection records layered on top of the retained-artifact cycle
- `runtime.md`: policy execution, context packets, state shaping, runtime flags, and the lazy modeling-adapter boundary
- `unattended_controller.md`: outer-loop round planning, support-aware exploration, and controller diagnostics
- `unattended_parallel_codex_runbook.md`: coordinator and replica workflow for parallel Codex support during live unattended campaigns
- `unattended_work_queue.md`: live lane claims and closeouts for the current unattended parallel Codex wave
- `coding_agi_gap_map.md`: compact evidence-backed map from unattended-run failures to the next kernel surfaces
- `universe.md`: stable machine-level governance, retained universe contracts, and schema
- `tolbert_liftoff.md`: staged TOLBERT expansion, liftoff criteria, and modeling package boundary
- `tasks.md`: task schema, built-in task families, replay tasks, and curriculum behavior
- `scripts.md`: CLI entrypoints, extraction/proposal scripts, retention utilities, and eval helpers
- `improvement.md`: artifact extraction, proposal generation, retention, rollback, campaign flow, and the current `improvement_engine` / plugin / support-validator split
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

`task -> workspace/checkpoint bootstrap -> optional setup -> memory/world/governance bootstrap -> policy decision -> governance check -> sandbox execution -> verification -> transition estimation -> state update -> episode memory`

Canonical inner task loop:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

Canonical outer improvement loop:

`run tasks -> persist episodes -> compile learning candidates -> rank improvement targets -> generate candidate artifacts -> compare candidate vs retained baseline -> retain or reject -> feed retained artifact back into the next task run`

Targeting rule:

- changes to schemas, state, policy, execution, verification, or stopping are `executable-agent core`
- changes to memory, world model, or governance are `ASI runtime core`
- changes to the retain/reject meta-loop are `ASI self-improvement core`
- changes to one modeling, retrieval, curriculum, controller, or task-supply surface are auxiliary unless they alter one of those minimums directly
