# Runtime

## Default path

The preferred live runtime is local `vllm` plus a strict TOLBERT service subprocess. The current code defaults in [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py) and the native shell wrappers still point at `ollama` unless overridden through environment variables or CLI flags.

This is the current seed runtime posture:

- `vllm` is the authoritative free-form decoder
- the live TOLBERT service is the authoritative encoder/retrieval compiler slice
- the broader TOLBERT family in this repo is the target encoder-latent-decoder universal runtime documented in [`tolbert_liftoff.md`](/data/agentkernel/docs/tolbert_liftoff.md)
- future TOLBERT-family checkpoint takeover is a retained-gate milestone, not a naming trick

Recommended `vllm` launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization 0.6
```

Recommended runtime env:

```bash
export AGENT_KERNEL_PROVIDER=vllm
export AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B
export AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000
```

Important `KernelConfig` defaults from [`agent_kernel/config.py`](/data/agentkernel/agent_kernel/config.py):

- provider: `ollama`
- provider: `vllm`
- model: `Qwen/Qwen3.5-9B`
- host: `http://127.0.0.1:8000`
- host: `http://127.0.0.1:11434`
- `use_tolbert_context=1`
- `tolbert_mode=full`
- `use_skills=1`
- `use_graph_memory=1`
- `use_world_model=1`
- `use_planner=1`
- `use_role_specialization=1`
- `use_prompt_proposals=1`
- `use_curriculum_proposals=1`
- `use_retrieval_proposals=1`
- `max_steps=5`
- `timeout_seconds=20`

The loop in [`agent_kernel/loop.py`](/data/agentkernel/agent_kernel/loop.py) enriches state before each decision:

- graph summary from prior episodes
- world-model summary
- short verifier-oriented plan
- current acting role
- TOLBERT `ContextPacket`

## Providers

Supported providers are:

- `ollama`
- `vllm`
- `mock`

`ollama` and `vllm` are the live paths. `mock` is mainly used by tests and skips retrieval-required tasks in eval.

## Provider behavior

For `ollama`, the client sends `think: false` to avoid structured output being placed only in Ollama’s thinking field. The parser still accepts JSON from either `response` or `thinking` as a fallback.

Useful LLM settings:

- `AGENT_KERNEL_LLM_TIMEOUT_SECONDS`
- `AGENT_KERNEL_LLM_RETRY_ATTEMPTS`
- `AGENT_KERNEL_LLM_RETRY_BACKOFF_SECONDS`
- `AGENT_KERNEL_LLM_PLAN_MAX_ITEMS`
- `AGENT_KERNEL_LLM_SUMMARY_MAX_CHARS`
- `AGENT_KERNEL_OLLAMA_HOST`
- `AGENT_KERNEL_VLLM_HOST`
- `AGENT_KERNEL_VLLM_API_KEY`
- `CUDA_VISIBLE_DEVICES`

## TOLBERT family context

The compiler path is:

1. send the current query to [`scripts/tolbert_service.py`](/data/agentkernel/scripts/tolbert_service.py)
2. embed and predict per-level logits with `tolbert-brain`
3. greedily decode a constrained path
4. select a branch depth based on confidence
5. retrieve branch-scoped, fallback-branch, and global evidence
6. compile a `ContextPacket` consumed by the policy

Lifecycle note:
- `TolbertServiceClient` owns the `scripts/tolbert_service.py` subprocess for a live kernel.
- `AgentKernel.close()` shuts that subprocess down through the policy/context-provider stack.
- `run_eval()` closes every kernel it creates, including generated-task and failure-seed kernels, so long improvement cycles do not leave stale TOLBERT workers behind.

Live configuration includes:

- `AGENT_KERNEL_TOLBERT_PYTHON_BIN`
- `AGENT_KERNEL_TOLBERT_CONFIG_PATH`
- `AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH`
- `AGENT_KERNEL_TOLBERT_NODES_PATH`
- `AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS`
- `AGENT_KERNEL_TOLBERT_CACHE_PATHS`
- `AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH`
- `AGENT_KERNEL_TOLBERT_DEVICE`
- `AGENT_KERNEL_TOLBERT_BRANCH_RESULTS`
- `AGENT_KERNEL_TOLBERT_GLOBAL_RESULTS`
- `AGENT_KERNEL_TOLBERT_TOP_BRANCHES`
- `AGENT_KERNEL_TOLBERT_ANCESTOR_BRANCH_LEVELS`
- `AGENT_KERNEL_TOLBERT_MAX_SPANS_PER_SOURCE`
- `AGENT_KERNEL_TOLBERT_CONTEXT_MAX_CHUNKS`
- `AGENT_KERNEL_TOLBERT_CONTEXT_CHAR_BUDGET`
- `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS`
- `AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD`
- `AGENT_KERNEL_TOLBERT_BRANCH_CONFIDENCE_MARGIN`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_WIDEN_THRESHOLD`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_BRANCH_MULTIPLIER`
- `AGENT_KERNEL_TOLBERT_LOW_CONFIDENCE_GLOBAL_MULTIPLIER`
- `AGENT_KERNEL_TOLBERT_DETERMINISTIC_COMMAND_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_FIRST_STEP_DIRECT_COMMAND_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_DIRECT_COMMAND_MIN_SCORE`
- `AGENT_KERNEL_TOLBERT_SKILL_RANKING_MIN_CONFIDENCE`
- `AGENT_KERNEL_TOLBERT_SEMANTIC_SCORE_WEIGHT`
- `AGENT_KERNEL_TOLBERT_TASK_MATCH_WEIGHT`
- `AGENT_KERNEL_TOLBERT_SOURCE_TASK_WEIGHT`
- `AGENT_KERNEL_TOLBERT_DISTRACTOR_PENALTY`

If `AGENT_KERNEL_TOLBERT_DEVICE=cuda` is set and CUDA is not actually visible in the TOLBERT env, the service fails fast.

## TOLBERT modes

[`scripts/run_eval.py`](/data/agentkernel/scripts/run_eval.py) exposes comparison modes for:

- `full`
- `path_only`
- `retrieval_only`
- `deterministic_command`
- `skill_ranking`

Those are eval-time analysis lanes, not separate product surfaces.

## Universal family definition

In this repo, `TOLBERT` no longer means only the original ontology encoder.
It means the full retained model family with these coordinated surfaces:

- encoder surface for hierarchy-aware representation and retrieval
- latent/state-space surface for long-horizon task state
- decoder surface for bounded and eventually free-form action generation
- world-model surface for transition and risk forecasting
- policy/value/stop/risk heads that plug into kernel routing and acceptance

The current live service path in [`agent_kernel/tolbert.py`](/data/agentkernel/agent_kernel/tolbert.py)
is the seed compiler member of that family. The hybrid runtime under
[`agent_kernel/modeling/tolbert/`](/data/agentkernel/agent_kernel/modeling/tolbert)
is the first internal latent/runtime member. The seed `vllm` stack remains active
until retained TOLBERT-family checkpoints earn authority through the liftoff gate.

`run_eval.py` also exposes memory/eval lanes that the older docs did not list:

- `--include-skill-transfer`
- `--include-operator-memory`
- `--include-benchmark-candidates`
- `--include-verifier-candidates`
- `--compare-abstractions`

## State and step records

Task state and stored steps carry more than the earlier docs described. A step record can include:

- selected skill ID
- retrieval candidate and evidence counts
- retrieval-selected span ID
- whether retrieval influenced the action
- path confidence
- whether retrieval was trusted
- active subgoal
- acting role
- world-model horizon

Episode documents also persist:

- task contract
- task metadata
- initial plan
- graph summary
- world-model summary
- termination reason

## Paths and artifacts

By default the runtime writes to:

- `workspace/`
- `trajectories/episodes/`
- `trajectories/skills/command_skills.json`
- `trajectories/operators/operator_classes.json`
- `trajectories/tools/tool_candidates.json`
- `trajectories/benchmarks/benchmark_candidates.json`
- `trajectories/verifiers/verifier_contracts.json`
- `trajectories/retrieval/retrieval_proposals.json`
- `trajectories/retrieval/retrieval_asset_bundle.json`
- `trajectories/prompts/prompt_proposals.json`
- `trajectories/curriculum/curriculum_proposals.json`
- `trajectories/improvement/cycles.jsonl`
- `trajectories/improvement/reports/`

## CLI entrypoints

Single task:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_agent.py --task-id hello_task
```

Eval:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_eval.py
```

Selfplay:

```bash
python scripts/run_selfplay.py --seed-task-id hello_task --seed-mode success
```

Native wrappers:

```bash
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/run_native_agent.sh hello_task
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/run_native_eval.sh
AGENT_KERNEL_PROVIDER=vllm AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 ./scripts/verify_impl.sh
```
