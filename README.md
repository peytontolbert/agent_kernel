# agent-kernel

`agent-kernel` is a verifier-driven coding agent with a strict TOLBERT context layer in front of the LLM policy.
It uses one LLM-backed policy, two actions (`respond` and `code_execute`), deterministic and semantic verification, and selfplay-oriented task generation to solve local software tasks ranging from boot-check file edits to multi-file repo workflows, test repair, and shared-repo coordination.

The repository is organized around two core loops.

Inner task loop:

`observe -> estimate state -> retrieve memory -> update world model -> simulate likely transitions -> plan candidates -> choose via policy -> execute -> verify -> critique -> update memory/models -> repeat`

Outer improvement loop:

`run tasks -> collect outcomes -> compare against verifiers -> localize failure -> modify policy/planner/world-model/transition-model/prompts/tools -> retest -> retain only verified gains`

## Documentation

- [docs/index.md](docs/index.md)
- [docs/setup.md](docs/setup.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/runtime.md](docs/runtime.md)
- [docs/tasks.md](docs/tasks.md)
- [docs/scripts.md](docs/scripts.md)
- [docs/ai_agent_status.md](docs/ai_agent_status.md)
- [docs/product_roadmap.md](docs/product_roadmap.md)
- [docs/validation.md](docs/validation.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
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
export AGENT_KERNEL_PROVIDER=vllm
export AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B
export AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000
python scripts/build_agentkernel_tolbert_assets.py --output-dir var/tolbert/agentkernel
./scripts/run_native_agent.sh hello_task
./scripts/run_native_agent.sh git_repo_test_repair_task
python scripts/run_supervised_improvement_cycle.py --provider vllm --task-limit 5 --generate-only
```

Treat `hello_task` as a liveness probe only. The main product target is reliable work on `repository`, `project`, `tooling`, `integration`, and `repo_sandbox` families such as `git_repo_test_repair_task`, `git_repo_status_review_task`, and the supervised coding lanes that bias toward repository and project tasks.

`vllm` is the preferred live provider. The current code defaults still fall back to `ollama` unless `AGENT_KERNEL_PROVIDER`, `AGENT_KERNEL_MODEL`, and `AGENT_KERNEL_VLLM_HOST` are set in the shell or passed on the CLI.

For implementation verification against the real runtime instead of only mock tests:

```bash
AGENT_KERNEL_PROVIDER=vllm \
AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B \
AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
./scripts/verify_impl.sh
```

That sequence treats the full Tolbert-enabled runtime as the primary verification path, then runs skill and Tolbert ablations as secondary diagnostics.

The default runtime now compiles a `ContextPacket` before each LLM decision using a real `tolbert-brain` subprocess service. That packet contains:

- a TOLBERT-style path prediction
- branch-scoped retrieval results
- global retrieval results
- the verifier contract for the current task

This path requires real TOLBERT assets: a native agent-kernel checkpoint, `nodes.jsonl`, retrieval cache shards, and an explicit label-map artifact for the remapped training labels.
`AgentKernel` now closes that helper subprocess explicitly, and eval/improvement flows are expected to tear kernels down after each run so TOLBERT workers do not accumulate on the GPU.

## Project layout

```text
agent_kernel/
  config.py
  schemas.py
  state.py
  actions.py
  policy.py
  sandbox.py
  verifier.py
  memory.py
  task_bank.py
  curriculum.py
  loop.py
  extractors.py
evals/
  harness.py
  metrics.py
prompts/
  system.md
  decision.md
  reflection.md
scripts/
  run_agent.py
  run_selfplay.py
  replay_episode.py
  extract_skills.py
  run_eval.py
docs/
  index.md
  setup.md
  architecture.md
  runtime.md
  tasks.md
  scripts.md
  validation.md
tests/
```
