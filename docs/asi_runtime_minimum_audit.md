# Lane 2 Audit: ASI-Shaped Runtime Minimum

## Scope Reviewed

- [`docs/asi_core.md`](/data/agentkernel/docs/asi_core.md:43)
- [`docs/minimal_asi_analysis_lanes.md`](/data/agentkernel/docs/minimal_asi_analysis_lanes.md:80)
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:22)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:21)
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:75)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:310)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:338)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:212)
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28)
- [`agent_kernel/modeling/world/latent_state.py`](/data/agentkernel/agent_kernel/modeling/world/latent_state.py:8)
- [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:17)
- [`tests/test_memory.py`](/data/agentkernel/tests/test_memory.py:685)
- [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:1018)

## Minimum Contract

Per [`asi_core.md`](/data/agentkernel/docs/asi_core.md:52), the ASI-shaped runtime minimum adds three things above the executable floor:

- cross-step and cross-episode memory pressure
- an internal progress representation
- explicit governance structure above raw execution

For this audit, the bar is not just `present in code`. The bar is:

- `present`: the structure exists as a maintained runtime surface
- `wired`: the live loop or policy consumes it while choosing or evaluating actions
- `proven`: tests show it changes runtime behavior, not just reporting payloads

## Verdict

The repository does clear the `ASI-shaped runtime minimum`, but narrowly.

It is no longer just a bare executor plus verifier. Cross-episode memory, world-state summaries, transition tracking, latent state estimation, and declarative governance all exist and are wired into live decision pressure through [`AgentKernel.run_task()`](/data/agentkernel/agent_kernel/loop.py:192), [`LLMDecisionPolicy.decide()`](/data/agentkernel/agent_kernel/policy.py:338), and [`LLMDecisionPolicy._command_control_score()`](/data/agentkernel/agent_kernel/policy.py:2104).

The main caveat is narrower now. The governance layer is no longer only scoring-and-attestation: the loop now performs a deterministic pre-execution universe-governance block before `Sandbox.run()`, and blocked commands fail the step as `governance_rejected`. The hard containment floor still remains the sandbox in [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:80), so the current design is a two-layer control surface rather than universe governance replacing containment.

## What Is Already Present

### Memory

`EpisodeMemory` persists episode documents and computes graph summaries over prior runs in [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:28) and [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:57). Those summaries include failure signals, retrieval-backed successes, trusted retrieval procedures, and environment priors rather than only raw episode counts.

The loop actually loads that memory into runtime state before stepping. See [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:310), where `graph_memory.summarize()` populates `state.graph_summary`.

### World Model

`WorldModel.summarize()` builds an internal progress representation over expected artifacts, forbidden artifacts, preserved artifacts, workflow obligations, and workspace state in [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:27). It also exposes command scoring in [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:70), transition extraction in [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:229), and long-horizon hotspot prioritization in [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:278).

The live loop refreshes that world model after execution, derives a transition object, and stores it on state in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:498), [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:520), and [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:244).

### Governance

`UniverseModel` defines a declarative constitution and operating envelope in [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:22) and [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:62). It merges retained runtime artifacts into those defaults in [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:391) and [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:434).

Its live outputs include invariants, forbidden command patterns, environment assumptions, risk controls, environment alignment, runtime attestation, and plan risk summaries in [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:81) and [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:315).

### Sandbox As Hard Floor

The actual hard containment layer is still the sandbox. `Sandbox.run()` blocks prohibited command classes, rejects unsupported shell operators, applies path policy checks, and runs commands in bounded subprocess execution in [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:80), [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:201), and [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:108).

That matches the contract in [`asi_core.md`](/data/agentkernel/docs/asi_core.md:62): declarative governance sits above execution, while the sandbox remains the hard enforcement floor.

## What Is Wired Into The Live Path

### Memory Changes Later Choices

Memory is not just recorded. It changes later command selection in at least two live ways.

First, graph-memory environment priors change command scores through [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2104) and the graph-memory-specific bias in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2255).

Second, trusted retrieval carryover can short-circuit normal context compilation and directly emit the next action from prior successful episodes. See [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1489), [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:3760), and [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:3897).

That is strong evidence of cross-episode behavioral pressure rather than passive logging.

### World State Changes Later Choices

World-model output directly affects policy scoring in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2108), planner progress reconciliation in [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:270), and subgoal satisfaction checks in [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:973).

The loop also derives a step-to-step transition summary and converts it into latent state used by the policy in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:520), [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:564), and [`agent_kernel/modeling/world/latent_state.py`](/data/agentkernel/agent_kernel/modeling/world/latent_state.py:184).

This means the runtime does not only know whether verification passed. It tracks whether state progressed, regressed, or stalled.

### Governance Changes Later Choices

Universe governance affects live command ranking through [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2107), via explicit governance simulation in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:2233), and through the compact universe payload passed into the model in [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:486) and [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:4155).

The loop now also enforces a deterministic pre-execution governance gate through [`UniverseModel.should_block_command()`](/data/agentkernel/agent_kernel/universe_model.py:315) and [`AgentKernel.run_task()`](/data/agentkernel/agent_kernel/loop.py:486). On block-worthy commands, the step fails before sandbox execution with explicit `governance_rejected` verification output and command-governance metadata.

So governance is not passive metadata. It now shapes both ranking and execution eligibility. The remaining limitation is coverage breadth, not whether it can bind behavior at all.

## What Is Proven By Tests

The existence of the runtime-shaped surfaces is covered in [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:17), which asserts that a solved seed task carries graph memory, universe summary, and world-model summary on the resulting episode.

Cross-episode memory evidence is proven in [`tests/test_memory.py`](/data/agentkernel/tests/test_memory.py:685), [`tests/test_memory.py`](/data/agentkernel/tests/test_memory.py:729), and [`tests/test_memory.py`](/data/agentkernel/tests/test_memory.py:788), which verify memory-source tracking, trusted retrieval carryover signals, and trusted retrieval procedures.

Governance evidence is proven in [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:1018), [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:1049), and [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:2720), which verify universe summary payload content and show destructive or network-heavy commands score worse than verification-aligned commands.

The strongest memory-to-action proof is in [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:14990), [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:15062), and [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:15194), where graph memory directly changes the chosen command before context compilation.

Latent world-state shaping is proven in [`tests/test_policy.py`](/data/agentkernel/tests/test_policy.py:1462) and [`tests/test_loop.py`](/data/agentkernel/tests/test_loop.py:106), which verify latent state construction and learned world-signal propagation from retained Tolbert runtime artifacts.

## What Is Only Partial Or Optional

`GraphMemory` only enters the loop when `use_graph_memory` is enabled in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:310). That means cross-episode memory pressure is still feature-gated at the implementation level.

`WorldModel` refresh and latent-state synthesis are gated behind `use_world_model` in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:330) and [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:498). The stronger ASI-shaped runtime is therefore still a configurable surface in code.

What changed is the runtime claim boundary. `KernelConfig.validate()` now rejects a weakened `bounded_autonomous` shape if either `use_graph_memory` or `use_world_model` is disabled in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:622) and [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:704). So these surfaces are optional for the strict executable floor, but no longer optional for the mode that claims bounded autonomy.

The richer learned world signal is even more optional. It only appears when retained Tolbert model artifacts expose a world-model surface in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:925). The heuristic world model is core; the learned world model is an optional strengthening layer.

Much of graph memory is still report-side. Fields like benchmark-family counts, failure-type histograms, and related-task lists are useful runtime context, but the strongest live effects currently come from trusted retrieval carryover, environment-prior scoring, graph-memory-derived failure pressure in command scoring, and recovery-role escalation rather than from the full memory graph.

## What Is Missing

The previous biggest missing contract, a hard universe-governance pre-execution gate, is now closed.

`UniverseModel.should_block_command()` now wraps `simulate_command_governance()` and derives deterministic block reasons such as `forbidden_pattern`, `network_access_conflict`, and `path_scope_conflict` in [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:315). The loop consults it before `Sandbox.run()` in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:486), emitting a synthetic failed step with `outcome_label="governance_rejected"` when the command is blocked.

The next remaining gap is broader governance coverage. Blocking is deterministic and real, but it is still focused on a finite rule set around forbidden patterns, path scope, network conflicts, git-write conflicts, destructive mutations, privileged or remote execution, and unbounded execution. Plan-level vetoes, multi-step intent controls, and richer rewrite-or-repair behavior are still thinner than the command-level gate.

The second previously missing contract, a clearer bounded-autonomy floor for memory and world-model pressure, is also closed at config-validation time. A runtime can still opt into the strict executable floor, but it cannot silently claim `bounded_autonomous` while dropping graph memory or world-model support.

## Highest-ROI Next Patches

1. Broaden universe-governance coverage beyond deterministic command heuristics.
   The new gate is real, but it still operates on a finite command-risk surface. The next step is plan-aware blocking and repair guidance when a sequence of individually-legal commands composes into a constitution conflict.

2. Expand graph-memory pressure beyond the current recovery signals.
   Failure and environment priors now bias command scoring and long-horizon role routing, but more of the graph summary should directly affect planner recovery structure, verifier sequencing, and task-local rollback choice.

3. Add rewrite-or-repair governance responses alongside hard block decisions.
   Some blocked commands should be transformed into bounded safe alternatives instead of only failing with `governance_rejected`.

4. Add broader integration tests around recovery after governance rejection.
   The new tests prove pre-execution blocking, but the next useful proof is that the runtime recovers into safer next actions rather than stalling after the rejection.

5. Push the ASI-runtime minimum from config enforcement into clearer runtime posture reporting.
   The runtime should expose the claimed shape and missing prerequisites explicitly in status/report surfaces so audits do not need to infer them from config validation rules.
