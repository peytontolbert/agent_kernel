# Minimal ASI Analysis Lanes

## Purpose

This document splits the minimal-ASI audit into parallel analysis lanes.
Each lane is meant to answer one bounded question about ASI completeness using code evidence, not intuition.

Use this when multiple people are reviewing the codebase at the same time and need clean boundaries.

Canonical boundary:

- [`asi_core.md`](/data/agentkernel/docs/asi_core.md)

The three minimums being audited are:

1. `executable agent minimum`
2. `ASI-shaped runtime minimum`
3. `minimum self-improving ASI`

This lane plan adds one more cross-cutting lane:

4. `boundary and auxiliary audit`

## Shared Rules

Every lane should follow the same discipline:

- cite code, not just prose docs
- treat line references as the primary evidence
- distinguish `present in code`, `wired into live path`, and `proven by tests or reports`
- separate `minimum-defining` logic from `support` logic
- record gaps as concrete missing contracts, not vague concerns

Every lane output should include:

- `scope reviewed`
- `minimum contract`
- `what is already present`
- `what is only partial or optional`
- `what is missing`
- `highest-ROI next patches`

Every claimed gap should answer:

- what contract is supposed to hold
- where the contract is implemented now
- what evidence says it is incomplete
- whether the gap is code absence, wiring absence, or evidence absence

## Lane 1: Executable Agent Minimum

### Goal

Audit the smallest closed-loop executor and decide whether it is complete as an autonomous coding agent floor.

### Primary surfaces

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:192)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:61)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:63)
- [`agent_kernel/schemas.py`](/data/agentkernel/agent_kernel/schemas.py:13)
- [`agent_kernel/actions.py`](/data/agentkernel/agent_kernel/actions.py:1)
- [`agent_kernel/sandbox.py`](/data/agentkernel/agent_kernel/sandbox.py:28)
- [`agent_kernel/verifier.py`](/data/agentkernel/agent_kernel/verifier.py:10)
- [`agent_kernel/llm.py`](/data/agentkernel/agent_kernel/llm.py:1)
- [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py:117)

### Contract to verify

- task representation exists and is stable
- runtime state can advance step by step
- policy can turn state into an action decision
- action execution is bounded by the sandbox
- verification can produce success or failure
- loop can stop, persist, and return a result

### Questions to answer

- Is there any required executor behavior that still lives outside this minimum?
- Are any parts of this minimum only conditionally active behind flags that break the floor?
- Is the action model still truly minimal, or are hidden planner or retrieval dependencies required in practice?
- Does stopping behavior depend on optional runtime enrichments rather than the minimum executor contract?

### Evidence to collect

- key state transition functions
- policy decision entrypoints and fallback behavior
- action execution path from policy output into sandbox
- verifier entrypoints and result-shaping path
- loop termination and episode-write path
- tests that exercise this floor without optional ASI-shaped additions

### Expected output

- a verdict on whether the executable floor is complete
- a list of any floor violations or hidden dependencies
- exact missing contracts if the floor is not actually self-sufficient

## Lane 2: ASI-Shaped Runtime Minimum

### Goal

Audit whether the runtime rises above a bare executor into a bounded autonomous intelligence shape.

### Primary surfaces

- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:14)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:21)
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:75)
- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:192)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:338)
- [`agent_kernel/state.py`](/data/agentkernel/agent_kernel/state.py:212)

### Contract to verify

- memory changes later action selection or evaluation
- world-model state represents progress, uncertainty, or likely transitions in a way the runtime actually consumes
- governance structure exists above raw shell execution
- sandbox remains the hard enforcement layer
- universe-level governance is declarative and meaningfully wired into live decisions

### Questions to answer

- Is memory merely recorded, or does it reliably pressure later choices?
- Is the world model a real control input or mostly a report-side artifact?
- Does governance actually change behavior, or is it passive metadata?
- Which ASI-runtime pieces are wired into the live path, and which are still optional ornaments?
- What evidence proves cross-step and cross-episode learning pressure rather than just extra summaries?

### Evidence to collect

- memory read and write sites in the runtime path
- world-model update sites and downstream consumption sites
- universe-model construction and policy/loop enforcement points
- interaction boundaries between `UniverseModel` and `Sandbox`
- tests or reports showing these structures affect decisions

### Expected output

- a verdict on whether the repo truly has an ASI-shaped runtime minimum
- any gaps where memory, world model, or governance exist in code but do not yet govern behavior
- a strict split between `present`, `wired`, and `proven`

## Lane 3: Minimum Self-Improving ASI

### Goal

Audit the irreducible meta-loop that lets the runtime change itself under verifier pressure.

### Primary surfaces

- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:80)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:247)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:2797)

### Contract to verify

- episode outcomes compile into reusable evidence
- targets and variants can be ranked
- candidate and baseline behavior can be compared
- retain and reject decisions can be made under explicit gates
- those decisions can be applied back onto live artifacts

### Questions to answer

- Is the learning compiler producing evidence that the improvement engine actually consumes?
- Are ranking and retention generic engine behavior, or are they entangled with subsystem-specific logic?
- Is candidate-vs-baseline comparison truly the central decision protocol?
- Does live artifact application happen inside the meta-loop minimum, or does it depend on extra orchestration layers?
- Which parts of the current self-improvement stack are essence versus support structure?

### Evidence to collect

- learning-compiler outputs and downstream consumers
- target-ranking and variant-ranking entrypoints
- retain/reject gate logic
- artifact-application and rollback paths
- tests for finalize and lifecycle application
- concrete examples where the engine retained or rejected based on evidence

### Expected output

- a verdict on whether the minimum self-improving loop is complete in code
- a list of entanglements between generic engine logic and subsystem plugins
- exact candidates for `improvement_engine.py` versus `improvement_plugins/`

## Lane 4: Boundary And Auxiliary Audit

Current output:

- [`asi_boundary_auxiliary_audit.md`](/data/agentkernel/docs/asi_boundary_auxiliary_audit.md)

### Goal

Audit whether the codebase actually respects the minimum boundaries defined in [`asi_core.md`](/data/agentkernel/docs/asi_core.md), or whether core and auxiliary concerns are still mixed together.

### Primary surfaces

- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:247)
- [`agent_kernel/subsystems.py`](/data/agentkernel/agent_kernel/subsystems.py:40)
- [`agent_kernel/strategy_memory/`](/data/agentkernel/agent_kernel/strategy_memory)
- [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
- [`agent_kernel/modeling/`](/data/agentkernel/agent_kernel/modeling)
- [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py)
- [`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py)
- [`agent_kernel/unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
- [`agent_kernel/job_queue.py`](/data/agentkernel/agent_kernel/job_queue.py)

### Contract to verify

- minimum-defining code is clearly separable from support code
- auxiliary layers improve the machine without silently becoming hard requirements for the minimums
- improvement support structure is not being misdescribed as the irreducible meta-loop
- learned execution packages and controller machinery are not incorrectly treated as ASI-core evidence

### Questions to answer

- Which auxiliary layers are incorrectly acting like required dependencies?
- Which modules should be reclassified from `core` to `support` in docs and architecture notes?
- Where is the biggest current boundary violation between generic engine logic and subsystem plugins?
- Which refactor seams are most realistic right now?

### Evidence to collect

- imports that pull auxiliary surfaces into minimum-defining codepaths
- config flags that reveal optional versus hard-required layers
- places where docs still overclaim auxiliary layers as core
- concrete engine-versus-plugin logic inside `improvement.py`

### Expected output

- a refactor map separating minimums from support layers
- a ranked list of boundary violations
- concrete file-level candidates for engine/plugin extraction

## Optional Lane 5: Live Evidence And Proof Gap Audit

### Goal

Audit whether the minimums that appear present in code are actually proven by runtime evidence,
and separate real implementation gaps from missing proof.

### Primary surfaces

- [`docs/asi_closure_work_queue.md`](/data/agentkernel/docs/asi_closure_work_queue.md)
- [`docs/coding_agi_gap_map.md`](/data/agentkernel/docs/coding_agi_gap_map.md)
- [`docs/asi_core.md`](/data/agentkernel/docs/asi_core.md)
- [`docs/minimal_asi_analysis_lanes.md`](/data/agentkernel/docs/minimal_asi_analysis_lanes.md)
- unattended and repeated-improvement reports under [`var/`](/data/agentkernel/var)
- key tests for finalize, retention, and runtime progression

### Contract to verify

- minimum-defining code has live evidence, not just implementation presence
- code and docs agree on what has been proven
- evidence gaps are not mislabeled as code gaps
- "detached validation", "integrated rerun", and "live unattended proof" are not conflated

### Questions to answer

- Which minimum contracts are implemented but not yet proven in integrated runs?
- Which conclusions in the docs are still relying on detached or partial evidence?
- Where do we need fresh runs instead of more refactoring?
- Which gaps are now proof-hygiene problems rather than missing runtime machinery?

### Evidence to collect

- the latest integrated unattended parent status and campaign report
- repeated-improvement status, retain or reject counts, and closeout-mode evidence
- any detached validation runs currently cited as proof for a stronger claim
- exact report fields that back each claim in `asi_closure_work_queue.md` and
  `coding_agi_gap_map.md`
- tests that prove bounded contracts locally but still lack live integrated confirmation
- a per-contract classification of `implemented`, `wired`, `tested`, `detached-live`, and
  `proven-live`

### Expected output

- a matrix of `implemented`, `wired`, `tested`, and `proven-live`
- a ranked list of proof gaps that block stronger ASI claims
- a short rerun plan that names the minimum fresh artifacts needed to close the top proof gaps

## Recommended Parallel Split

If there are four analysts, use:

1. `Lane 1`: executable agent minimum
2. `Lane 2`: ASI-shaped runtime minimum
3. `Lane 3`: minimum self-improving ASI
4. `Lane 4`: boundary and auxiliary audit

If there are five analysts, add:

5. `Lane 5`: live evidence and proof gap audit

## Final Deliverable Format

Each lane should end with the same compact closeout:

- `Verdict`
- `What is complete`
- `What is partial`
- `What is missing`
- `Top 3 fact-checked gaps`
- `Top 3 next patches`

Do not close with generic opinions.
Close with code-backed gaps that can be turned into work items immediately.
