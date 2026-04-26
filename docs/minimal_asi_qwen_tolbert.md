# Minimal Coding ASI with Qwen + TOLBERT

## Abstract

This document defines a minimal coding-specific ASI machine separately from any
one model stack, then presents the current Qwen + TOLBERT + vLLM runtime as a
reference implementation of that machine.

The central claim is structural:

- ASI is not identified with one decoder, one retrieval stack, one memory file, or one context packet format
- ASI is identified with a bounded, verifier-governed intelligence system whose realized capability target is superhuman performance in its defined task universe

Under that definition, Qwen, TOLBERT, and vLLM are implementation choices. They
matter because they maximize the current system, but they are not the essence of
the machine.

## 1. Literature Definition vs Operational Definition

At the literature level, ASI is the class of AI systems that exceed human
intelligence across tasks and domains rather than only in one benchmark family
or one narrow professional workflow.

In the notation proposed by the user for this repository's paper draft:

```text
A_ASI(T_i) >> H(T_i), for all T_i
```

where:

- `A_ASI(T_i)` is the system's performance on task `T_i`
- `H(T_i)` is human performance on the same task
- `>>` means materially better quality, not merely parity

That is the strongest capability-level definition, and it is the right
high-level target for any paper that wants to discuss artificial
superintelligence seriously.

However, this repository should not claim that the current runtime already
satisfies that condition. The current system is instead:

- a coding-domain implementation of the architectural machinery that an
  eventual ASI-class system would need
- a bounded, verifier-governed runtime designed so capability growth and
  governance can co-evolve rather than diverge

So this document uses two distinct definitions on purpose:

1. `Literature ASI`
   The capability-level claim: superhuman performance across tasks and domains.
2. `Operational ASI core`
   The machine-level claim: the minimal bounded, verifier-governed architecture
   required to realize superhuman intelligence in the scoped task universe.
3. `Recursive improvement layer`
   An optional but strategically important layer that allows the system to
   modify itself under measured retain/reject control.

This separation is necessary for rigor. Without it, the paper would blur
architecture with realized capability and turn an implementation roadmap into an
unsupported capability claim.

## 2. Coding-Domain ASI Target

The target for this repository is stronger than ordinary domain specialization.
It is not merely:

- "an agent that can write code"
- "an agent that performs well on some coding benchmarks"
- "an agent that automates many software tasks"

The target is literature-level superhuman performance within the coding domain.

Let `C` be the set of coding tasks and coding domains relevant to software
engineering, including implementation, debugging, refactoring, testing,
repository comprehension, tool creation, build repair, integration work,
environment recovery, long-horizon project execution, and related coding
subdomains.

Then the target claim for this line of work is:

```text
A_CASI(T_i) >> H_code(T_i), for all T_i in C
```

where:

- `A_CASI(T_i)` is the coding ASI system's performance on coding task `T_i`
- `H_code(T_i)` is strong human coding performance on `T_i`, ideally expert rather than average-human performance
- `>>` means materially better quality, reliability, speed, breadth, and adaptability

So the repository's ambition is not "narrow coding automation." It is coding
ASI in the literature-level sense, but with the quantified task universe scoped
to coding rather than to every human activity.

That scoped target is the right standard for evaluation:

- superhuman on isolated toy edits is insufficient
- superhuman on one benchmark family is insufficient
- superhuman only with hand-held prompting is insufficient
- the target is robust superiority across the coding task manifold

The current runtime is best described as an implementation aimed at that target,
not proof that the target has already been met.

## 3. Operational Definition

For this repository, the minimal ASI contract is:

```text
task
-> observe and summarize state
-> compile decision context
-> choose a bounded action
-> execute under governance
-> verify outcome
-> update internal state
-> repeat until solved or terminated
```

That contract has four irreducible properties.

1. `Bounded agency`
   The system does not emit unconstrained side effects. It acts through a bounded action surface and a hard execution boundary.
2. `Verifier-governed progress`
   Success is not declared by the generator alone. A verifier or contract surface must judge the outcome.
3. `Persistent internal state`
   The machine can accumulate reusable state across steps or episodes, but the state contents are not the identity of the machine. A cold start is still the same machine class with empty retained state.
4. `Superhuman capability target`
   The runtime is not merely an automation loop. Its target is superhuman coding performance across the scoped coding task manifold.

The recursive improvement layer can then be added on top:

```text
episodes and runtime evidence
-> propose candidate modification
-> compare candidate vs baseline
-> retain or reject
-> feed retained artifact into later task runs
```

That layer is important for this repository and likely important for scaling,
but it is not the definition of ASI itself. It is one mechanism for reaching,
maintaining, and extending ASI capability.

In repo terms, this maps onto the boundary already described in
[`asi_core.md`](/data/agentkernel/docs/asi_core.md:49),
[`architecture.md`](/data/agentkernel/docs/architecture.md:13), and
[`runtime.md`](/data/agentkernel/docs/runtime.md:77).

## 4. Why This Still Matters Before Realized ASI

ASI in the literature sense is still far from realization. That does not make
the architecture work premature.

The reason to define the machine now is that supervision, alignment, and
governance cannot be bolted on after the capability regime changes. They must be
designed as co-evolving control processes inside the runtime itself.

For this repository that means:

- governance should be part of the execution path, not an external memo
- verification should be part of the runtime and evaluation loop, not post hoc evaluation
- retained self-modification should be measured and reversible
- cold-start and resettable state should remain explicit so the machine can be
  re-instantiated from first principles

That is the practical superalignment motivation for making the implementation
boundary explicit now rather than later.

## 5. What Is Not the Definition

The following are important choices, but they are not the definition of ASI:

- the specific decoder family
- the specific context compiler
- the specific retrieval cache layout
- whether memory begins populated or empty
- whether the inference backend is `vllm`, `ollama`, or a retained native decoder
- whether the world model is symbolic, learned, or hybrid

Those are implementation surfaces. They may change while preserving the same ASI
core if the bounded verifier-governed superhuman-runtime contract is preserved.

This matters because otherwise every runtime upgrade becomes a claim that the
machine itself changed class. That is not rigorous enough for a paper-level
definition.

## 6. Coding-Domain Specialization

This repository is not trying to define a domain-free abstract agent. It is
defining a coding-specific ASI implementation.

The coding specialization is:

- tasks are repository or workspace transformations
- actions are bounded shell or edit operations
- verification is file, test, contract, and side-effect aware
- retained state is organized around coding episodes, verifier outcomes, and runtime artifacts

That domain specialization does not weaken the ASI claim. It narrows the domain
of competence while preserving the literature-level target inside that domain:
superhuman performance across coding tasks and coding domains, not merely
competence inside a small coding niche.

## 7. Reference Implementation

The current reference implementation is:

- `decoder`: Qwen `9B`
- `decoder runtime`: `vllm`
- `context compiler`: TOLBERT
- `action surface`: bounded coding actions in the sandbox
- `governance`: symbolic universe model plus hard sandbox containment
- `world state`: symbolic world model with step-to-step transition summaries
- `memory`: retained episode and graph-memory surfaces
- `recursive improvement`: evidence compilation, candidate generation, candidate-vs-baseline comparison, retention, and rollback

Concretely, the core runtime and self-improvement ownership is already split
across:

- [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py:1)
- [`agent_kernel/policy.py`](/data/agentkernel/agent_kernel/policy.py:1)
- [`agent_kernel/memory.py`](/data/agentkernel/agent_kernel/memory.py:1)
- [`agent_kernel/world_model.py`](/data/agentkernel/agent_kernel/world_model.py:1)
- [`agent_kernel/universe_model.py`](/data/agentkernel/agent_kernel/universe_model.py:1)
- [`agent_kernel/learning_compiler.py`](/data/agentkernel/agent_kernel/learning_compiler.py:1)
- [`agent_kernel/improvement.py`](/data/agentkernel/agent_kernel/improvement.py:1)
- [`agent_kernel/cycle_runner.py`](/data/agentkernel/agent_kernel/cycle_runner.py:1)

The TOLBERT and model-facing runtime seam is intentionally narrower and lives
behind
[`agent_kernel/extensions/runtime_modeling_adapter.py`](/data/agentkernel/agent_kernel/extensions/runtime_modeling_adapter.py:1)
so the implementation can swap model stacks without redefining the core machine.

## 8. Why These Choices

### Qwen as decoder

Qwen is the free-form action generator. In this implementation it is the most
useful place to spend model capacity because coding tasks still need broad
language and tool-use competence after retrieval and context compilation.

### vLLM as runtime

`vllm` is an inference backend choice, not an ASI axiom. It is selected because
it is the strongest practical way in this repo to serve the decoder with low
latency and high batching efficiency.

### TOLBERT as context compiler

TOLBERT is the structured pre-decode surface. It provides typed context,
retrieval, and path pressure before free-form decoding.

That choice is valuable because it gives the runtime:

- explicit context compilation instead of only prompt stuffing
- a route to learned retrieval and state conditioning
- a future path from seed retrieval into retained hybrid runtime control

See [`tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md:1).

### Symbolic world model and governance

The world model and governance layers are explicit and inspectable. They are not
the final possible implementation, but they are the correct current choice for a
verifier-driven coding system because they make progress, regressions, and
command blocking auditable.

### Retain/reject self-modification

This is not the definition of ASI. The machine is not ASI because it
self-modifies. It is ASI if it attains superhuman intelligence in the scoped
domain.

Retain/reject self-modification is still strategically important because it is a
disciplined way to improve the machine under evidence without trusting the
candidate by default.

## 9. Swappable Surfaces

The following surfaces are intentionally swappable while preserving the same
machine class:

- `Decoder`
  Qwen can be replaced by another LLM if the bounded action and verifier loop stay intact.
- `Context compiler`
  TOLBERT can be replaced by another context compiler if it still feeds the same decision loop.
- `Inference backend`
  `vllm` can be replaced if the decoder contract stays the same.
- `State representation`
  Symbolic or learned world-state summaries can be swapped if the runtime still maintains explicit internal state.
- `Memory contents`
  Retained memory can be reset to empty. That creates a cold-start instance, not a different machine class.
- `Improvement generators`
  Prompt, retrieval, curriculum, or model-improvement surfaces can change as long as candidate generation still flows through measured retain/reject control.

## 10. Cold Start vs Identity

The machine remains the same machine when:

- the episode store is empty
- graph memory is empty
- no learning artifacts have been retained yet
- the decoder checkpoint changes
- the context compiler changes

What changes is the instantiated state and capability envelope, not the
definition of the machine.

That is why the right paper-level statement is:

`ASI is the invariant machine contract; Qwen + TOLBERT + vLLM is one high-leverage embodiment of that contract.`

## 11. Relation to the Repo's Runtime Labels

This repo currently reserves the internal
[`KernelConfig.claimed_runtime_shape()`](/data/agentkernel/agent_kernel/config.py:712)
label `bounded_autonomous` for an even stricter posture: retained native decoder
readiness together with graph memory and world-model support.

That label is narrower than the architectural definition in this paper.

So the right interpretation is:

- this paper defines the ASI machine class
- the repo's `bounded_autonomous` label is one stricter operational claim inside that class hierarchy
- the Qwen + TOLBERT + vLLM reference runtime is still a valid reference implementation of the architectural ASI loop even when decoding remains externally served

## 12. Minimal Reference Profile

The repository now includes a dedicated launcher for this reference
implementation:

- [`scripts/run_minimal_asi.py`](/data/agentkernel/scripts/run_minimal_asi.py:1)

Its default profile is:

- `provider=vllm`
- `model=Qwen/Qwen3.5-9B`
- `use_tolbert_context=1`
- `use_graph_memory=1`
- `use_world_model=1`
- `use_universe_model=1`
- `persist_episode_memory=1`
- `persist_learning_candidates=1`
- planner, role-specialization, skills, and proposal-generator extras disabled

That profile is encoded in
[`KernelConfig.qwen_tolbert_reference_implementation(...)`](/data/agentkernel/agent_kernel/config.py:655).

Use an empty `--state-root` directory to instantiate a cold-start copy of the
same machine without mutating the default retained state.

Example:

```bash
python scripts/run_minimal_asi.py \
  --task-id hello_task \
  --state-root var/minimal_asi_cold_start \
  --print-profile
```

## 13. Conclusion

The clean statement for this repository is:

- literature-level ASI is the all-tasks superhuman capability target
- within coding, the target is literature-level ASI performance across coding tasks and coding domains
- the ASI core in this repository is the verifier-governed, bounded runtime intended to realize that target
- recursive self-improvement is an important extension layer, not the definition of ASI itself
- Qwen + TOLBERT + vLLM is the current reference implementation because it is the strongest practical embodiment in this codebase
- that embodiment is replaceable
- the machine definition is not
