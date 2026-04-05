# Setup

## Requirements

- Python 3.10+
- a local `vllm` install for the preferred live provider path
- a locally available `Qwen/Qwen3.5-9B` model served through `vllm`
- TOLBERT assets if you want the real context compiler enabled

Real TOLBERT runs require:

- checkpoint file
- ontology nodes JSONL
- one or more retrieval cache shards
- source spans JSONL
- label map JSON when class IDs were remapped during training
- a working TOLBERT Python environment

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the test suite

```bash
pytest -q
```

Use the live output in your environment as the source of truth instead of relying on a hard-coded count in the docs.

## Model setup

The preferred live provider is `vllm` using its OpenAI-compatible server. Start it with the current recommended launch command:

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

Point `agent-kernel` at that server:

```bash
export AGENT_KERNEL_PROVIDER=vllm
export AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B
export AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000
```

The current code defaults in `KernelConfig` and the shell wrappers still use `ollama` unless those variables are overridden.

## Build native agent-kernel TOLBERT assets

Generate repo-native ontology and span inputs:

```bash
python scripts/build_agentkernel_tolbert_assets.py --output-dir var/tolbert/agentkernel
```

This writes:

- `nodes_agentkernel.jsonl`
- `source_spans_agentkernel.jsonl`
- `model_spans_agentkernel.jsonl`
- `label_map_agentkernel.json`
- `config_agentkernel.json`
- `level_sizes_agentkernel.json`

Train a checkpoint in the TOLBERT env:

```bash
PYTHONPATH=/data/agentkernel/other_repos/TOLBERT \
/home/peyton/miniconda3/envs/ai/bin/python /data/agentkernel/other_repos/tolbert-brain/scripts/train_tolbert.py \
  --config /data/agentkernel/var/tolbert/agentkernel/config_agentkernel.json \
  --device cuda
```

Build a retrieval shard for that checkpoint:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/build_tolbert_cache.py \
  --config /data/agentkernel/var/tolbert/agentkernel/config_agentkernel.json \
  --checkpoint /data/agentkernel/var/tolbert/agentkernel/checkpoints/tolbert_epoch10.pt \
  --spans /data/agentkernel/var/tolbert/agentkernel/model_spans_agentkernel.jsonl \
  --out /data/agentkernel/var/tolbert/agentkernel/retrieval_cache/agentkernel__tolbert_epoch10.pt \
  --device cuda
```

## Environment variables

Most useful runtime settings:

- `AGENT_KERNEL_PROVIDER`
- `AGENT_KERNEL_MODEL`
- `AGENT_KERNEL_OLLAMA_HOST`
- `AGENT_KERNEL_VLLM_HOST`
- `AGENT_KERNEL_VLLM_API_KEY`
- `CUDA_VISIBLE_DEVICES`
- `AGENT_KERNEL_USE_TOLBERT_CONTEXT`
- `AGENT_KERNEL_TOLBERT_MODE`
- `AGENT_KERNEL_USE_SKILLS`
- `AGENT_KERNEL_USE_GRAPH_MEMORY`
- `AGENT_KERNEL_USE_WORLD_MODEL`
- `AGENT_KERNEL_USE_PLANNER`
- `AGENT_KERNEL_USE_ROLE_SPECIALIZATION`
- `AGENT_KERNEL_USE_PROMPT_PROPOSALS`
- `AGENT_KERNEL_USE_CURRICULUM_PROPOSALS`
- `AGENT_KERNEL_USE_RETRIEVAL_PROPOSALS`
- `AGENT_KERNEL_MAX_STEPS`
- `AGENT_KERNEL_COMMAND_TIMEOUT_SECONDS`
- `AGENT_KERNEL_LLM_TIMEOUT_SECONDS`
- `AGENT_KERNEL_WORKSPACE_ROOT`
- `AGENT_KERNEL_TRAJECTORIES_ROOT`
- `AGENT_KERNEL_SKILLS_PATH`
- `AGENT_KERNEL_OPERATOR_CLASSES_PATH`
- `AGENT_KERNEL_TOOL_CANDIDATES_PATH`
- `AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH`
- `AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH`
- `AGENT_KERNEL_VERIFIER_CONTRACTS_PATH`
- `AGENT_KERNEL_PROMPT_PROPOSALS_PATH`
- `AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH`
- `AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH`
- `AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR`

TOLBERT-specific settings:

- `AGENT_KERNEL_TOLBERT_PYTHON_BIN`
- `AGENT_KERNEL_TOLBERT_CONFIG_PATH`
- `AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH`
- `AGENT_KERNEL_TOLBERT_NODES_PATH`
- `AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS`
- `AGENT_KERNEL_TOLBERT_CACHE_PATHS`
- `AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH`
- `AGENT_KERNEL_TOLBERT_DEVICE`
- `AGENT_KERNEL_TOLBERT_MAX_SPANS_PER_SOURCE`
- `AGENT_KERNEL_TOLBERT_SERVICE_TIMEOUT_SECONDS`

## First run

Boot sanity check:

```bash
AGENT_KERNEL_PROVIDER=vllm \
AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B \
AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_agent.py --task-id hello_task
```

Expected shape:

```text
task=hello_task success=True
```

Representative coding task:

```bash
AGENT_KERNEL_PROVIDER=vllm \
AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B \
AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
python scripts/run_agent.py --task-id git_repo_test_repair_task
```

Native wrapper with preconfigured asset paths:

```bash
AGENT_KERNEL_PROVIDER=vllm \
AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B \
AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
./scripts/run_native_agent.sh git_repo_test_repair_task
```

Expected shape:

```text
task=git_repo_test_repair_task success=True
```

## Full runtime verification

For a more realistic end-to-end check:

```bash
AGENT_KERNEL_PROVIDER=vllm \
AGENT_KERNEL_MODEL=Qwen/Qwen3.5-9B \
AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000 \
./scripts/verify_impl.sh
```

That script runs:

- a live boot-check task
- a higher-value repo workflow task
- full eval with TOLBERT and skills
- TOLBERT comparison
- TOLBERT feature-mode comparison
- skill, operator, and tool extraction refresh
- skill comparison
- a baseline eval without TOLBERT or skills
- empirical gate assertions over the captured outputs
