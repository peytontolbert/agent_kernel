The paper’s central thesis is that self-improving agents should not be built as monolithic prompt-and-tool glue code. Instead, the agent system should expose its internal building blocks as versioned protocol resources, and all self-modification should pass through a closed-loop mutation interface with evaluation, auditability, and rollback. In the paper’s language, this means separating what evolves from how evolution happens. That is the conceptual heart of the system.

For an ASI-kernel interpretation, the cleanest mapping is:

AGP = the kernel protocol
RSPL = the kernel substrate / resource space
SEPL = the kernel mutation and improvement controller
AGS = a reference runtime built on top of the kernel

That mapping is faithful to the paper’s own structure, where RSPL handles the managed objects and SEPL handles the closed-loop update logic. The architecture diagram on page 4 explicitly visualizes this separation between the protocol layers and the multi-agent application layer.

2. Normative design goals from the paper

The paper says a self-evolution protocol must solve three problems. First, decoupling: prompts, tools, and memory cannot remain buried inside agent code; they must become independently managed entities. Second, safety and auditability: every update must be versioned and reversible. Third, formalism: the evolution process cannot remain ad hoc; it needs explicit operators such as reflection, proposal, verification, and commit. These three goals are the closest thing the paper gives to kernel requirements.

For an ASI kernel, this means your “kernel contract” should not be “run agent loop forever.” It should be:

register resources,
expose controlled mutation surfaces,
trace behavior,
propose and evaluate changes,
commit only under invariant-preserving rules.

That is exactly the paper’s philosophy.

3. Formal object model

The paper gives a surprisingly clean formal resource model. A resource entity is defined as a tuple containing a unique name, short description, input-output mapping, trainable marker, and metadata. The allowed core types are {PROMPT, AGENT, TOOL, ENV, MEM}. This is important because it means the protocol is not only about prompts or tool calls; it is about a uniform object space for heterogeneous agent components.

A registration record wraps the entity with version, implementation descriptor, instantiation parameters, and exported representations for LLM interaction. That exported representation field is especially important for a kernel because it means the same underlying resource can expose a callable schema, natural-language contract, or structured argument interface to the runtime.

A protocol-registered resource is then modeled as a triple of (records, context manager, server interface). This is the kernel-level abstraction: the resource is not just code or data; it is code/data plus lifecycle manager plus externally callable interface.

A documentation-grade encoding of the paper’s model would look like this:

ResourceEntity:
  type: PROMPT | AGENT | TOOL | ENV | MEM
  name: unique string
  description: short text
  mapping: input -> output contract
  trainable: bool
  metadata: dict

RegistrationRecord:
  entity: ResourceEntity
  version: semantic or monotonic version id
  implementation: import path | class | source code | remote binding
  init_params: dict
  exports: [tool schema | NL contract | structured argument schema]

ProtocolRegisteredResource:
  records: registry of RegistrationRecord
  manager: typed ContextManager
  api: server-exposed interface

For an ASI kernel, I would treat this object model as kernel-space truth. The runtime should never mutate raw prompt files or tool code directly; it should mutate only the registered record via the manager interface. That is a direct implication of the paper’s design.

4. Resource substrate layer as kernel-space

RSPL is effectively the paper’s kernel substrate. It makes all resources passive. That word matters: resources do not self-modify, do not run their own optimizers, and do not bypass the protocol. All observations and state transitions happen through higher-layer controlled interfaces. This is one of the cleanest safety ideas in the paper because it prevents “rogue self-editing” at the resource level.

The paper also defines the context manager as the real management plane for each resource type. It maintains the active registry, version history, lifecycle routines, and restoration capability. It also explicitly supports contract generation, which is a major systems insight: instead of bloating prompts with informal tool descriptions, the manager can emit a canonical capability contract, such as a skills.md-style specification for tools and skills.

Table 7 on page 19 is especially valuable because it is almost a minimal kernel API. The context manager/server interface supports lifecycle registration, retrieval and inspection, evolution and versioning, execution and contract, and serialization/deserialization. A direct paper-derived API surface is:

Lifecycle:
  init
  build
  register
  unregister

Retrieval:
  get
  get_info
  list
  retrieve
  get_state

Evolution:
  update
  copy
  restore
  get_variables
  set_variables

Execution:
  run
  save_contract
  load_contract

Persistence:
  save_to_json
  load_from_json

That API list is one of the most important parts of the paper if your goal is to build a reusable kernel instead of a one-off agent.

5. Infrastructure services you should treat as mandatory

The paper includes four cross-cutting services in RSPL. The Model Manager standardizes model access across providers and supports routing and fallback. The Version Manager owns lineage, diffing, rollback, and branching. The Dynamic Manager handles serialization/deserialization and enables hot-swapping resources at runtime. The Tracer Module captures inputs, outputs, decisions, tool interactions, and failures for debugging and future training/improvement. These are not optional add-ons in the paper; they are the infrastructure required to make evolution reproducible and safe.

For an ASI kernel, I would interpret these as hard kernel services:

model virtualization,
versioned resource journal,
live hot-swap deployment,
immutable trace ledger.

That is a reasonable systems-level reading of the paper’s infrastructure section.

6. The mutation model: SEPL

SEPL is the paper’s self-evolution controller. It introduces variable lifting, which maps heterogeneous resources into a common evolvable variable space Vevo. Importantly, Vevo is not limited to prompts or tool code; it includes managed resource entities plus execution artifacts like outputs and reasoning traces. The paper then defines a trainable subspace with a binary learnability mask. In other words, the system can know not only what exists, but also what is legally mutable. That is extremely relevant for an ASI kernel because it gives you a principled mutation permission model.

The core operator algebra is:

Reflect: map traces to failure hypotheses
Select: map hypotheses to proposed edits
Improve: apply edits to produce a candidate state
Evaluate: score the candidate against objective and safety
Commit: accept or reject the candidate state

This is the paper’s main formal contribution. The operators are typed over trace space Z, hypothesis space H, modification space D, objective G, and evaluation space S.

The key kernel interpretation is that commit is a transaction boundary. The paper explicitly says evolution must be traceable, reversible, and safe-by-construction because all mutations are mediated through standardized RSPL interfaces. That is exactly how a self-modifying kernel should think about updates.

7. Runtime architecture: AGS as reference implementation

AGS is the paper’s concrete runtime. It is a multi-agent system organized around an Agent Bus. The orchestrator does planning and coordination, not subtask execution. It creates a structured plan.md artifact containing task decomposition, flow structure, and sub-agent assignments. The paper says this plan is registered as a versioned RSPL resource, which means even the plan itself becomes inspectable and evolvable. That is a very important idea for your kernel.

Each specialist sub-agent retrieves prompts and tools from the RSPL registry, executes against the environment, writes intermediate results and traces into shared memory, and can operate concurrently. The orchestrator later collects results, updates the plan, and decides whether to terminate or replan. The paper also supports agent-as-tool composition, where an agent is wrapped as a tool behind a standard schema. That means the runtime can fluidly treat agents either as autonomous workers or callable capabilities.

One subtle but important systems observation is that the formal core entity types are only prompt, agent, tool, environment, and memory, but the concrete runtime elevates plan.md into a versioned resource. That suggests a production kernel should probably add artifact-like resource classes beyond the core five, such as PLAN, POLICY, DATASET, EVAL_SUITE, or SKILL. That extension is not formally defined in the paper, but it is strongly implied by the AGS design.

8. Optimizer backends

The default optimizer is reflection-driven. The LLM analyzes execution traces, writes structured failure hypotheses, proposes edits, applies them through set_variables, re-runs the task, and commits only if performance improves or invariants hold. This is the cleanest backend for a first implementation because it is human-readable and aligns with the paper’s main loop.

The paper also plugs in TextGrad, Reinforce++, and GRPO under the same mutation surface. That matters because it means the kernel is optimizer-agnostic. You can hold RSPL/SEPL steady and swap the optimizer family. TextGrad is essentially critique-driven prompt rewriting; Reinforce++ and GRPO reinterpret the same mutation surface with reward-style policy improvement, including solution refinement and updates to prompts or tools. This is one of the paper’s most reusable ideas: one substrate, many optimizers.

9. Benchmark evidence you can actually use

On reasoning-heavy benchmarks like GPQA and AIME, the paper mainly evolves prompts and outputs. The main finding is that weaker models gain more, stronger models gain less, and combined prompt + solution evolution usually beats evolving only one of them. That tells us the kernel should support simultaneous edits across multiple resource classes, not a single mutation target.

On GAIA, the strongest result comes from tool evolution. The paper reports a vanilla average of 79.07 versus 89.04 for evolve-tool, with the hardest Level 3 tasks improving from 61.22 to 81.63. That is the most compelling practical evidence in the paper because it suggests that as tasks become longer and more tool-dependent, resource evolution matters more. For an ASI kernel, this strongly argues that tool synthesis, repair, and reuse must be kernel-native rather than bolted on later.

On the coding benchmark, the paper reports consistent gains across Python, C++, Java, Go, and Kotlin, with especially strong results in compiled languages. The chart on page 11 shows the evolving agent outperforming the vanilla agent not just at endpoints but over the whole evaluation trajectory, which the authors interpret as compounding improvement within inference. This supports a kernel design where iterative repair is treated as a first-class control loop rather than an afterthought.

10. What the paper gives you, and what it does not

What the paper gives you is a protocol kernel for safe self-evolution: typed resources, version lineage, rollback, trace-based evaluation, a mutation operator algebra, and a reference multi-agent runtime. That is strong enough to serve as the architectural backbone of an ASI-kernel-style system.

What the paper does not fully specify is equally important. It does not define a rigorous security model for sandboxing self-generated tools. It does not specify distributed consistency rules for concurrent mutations. It does not define a full invariant language for the Commit gate. It does not deeply formalize conflict resolution when multiple sub-agents want to mutate overlapping resources. It also does not fully isolate protocol benefit from simple extra inference-time budget in the experiments. Those are the main gaps I would close before calling this an ASI kernel rather than a strong self-evolving agent framework.

11. My recommended adaptation into an ASI kernel

If I were converting this paper into a repo-grade kernel spec, I would keep the paper’s two-layer division exactly, but I would extend the substrate into the following first-class resource taxonomy:

PROMPT
AGENT
TOOL
ENV
MEM
PLAN
SKILL
POLICY_ADAPTER
EVAL_SUITE
DATASET
TRACESET

That extension is not verbatim from the paper, but it follows directly from the paper’s logic that heterogeneous agent internals should be protocol resources, and it resolves the paper’s own practical move of treating plan.md as a registered resource.

I would also harden the commit path into a promotion ladder:

candidate -> sandbox eval -> invariant check -> canary deployment -> promoted version -> rollback on regression

This is a production-grade elaboration of the paper’s evaluate-and-commit semantics. The paper already requires evaluation, safety invariants, lineage, and rollback; this just operationalizes them more aggressively for a kernel setting.

12. Short kernel definition

If you want a single sentence you can reuse in your internal docs:

An ASI kernel derived from Autogenesis is a protocol-managed substrate of versioned agent resources whose mutation is only allowed through a traced, evaluated, rollback-capable self-evolution loop.

A good next document from this would be a repo-ready ARCHITECTURE.md that turns RSPL, SEPL, the bus, and the commit gate into Python interfaces and file structure.