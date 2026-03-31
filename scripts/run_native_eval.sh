#!/usr/bin/env bash
set -euo pipefail

cd /data/agentkernel

export AGENT_KERNEL_PROVIDER="${AGENT_KERNEL_PROVIDER:-vllm}"
export AGENT_KERNEL_MODEL="${AGENT_KERNEL_MODEL:-Qwen/Qwen3.5-9B}"
export AGENT_KERNEL_VLLM_HOST="${AGENT_KERNEL_VLLM_HOST:-http://127.0.0.1:8000}"
export AGENT_KERNEL_USE_TOLBERT_CONTEXT="${AGENT_KERNEL_USE_TOLBERT_CONTEXT:-1}"
export AGENT_KERNEL_TOLBERT_PYTHON_BIN="${AGENT_KERNEL_TOLBERT_PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
export AGENT_KERNEL_TOLBERT_CONFIG_PATH="${AGENT_KERNEL_TOLBERT_CONFIG_PATH:-/data/agentkernel/var/tolbert/agentkernel/config_agentkernel.json}"
export AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH="${AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH:-/data/agentkernel/var/tolbert/agentkernel/checkpoints/tolbert_epoch10.pt}"
export AGENT_KERNEL_TOLBERT_NODES_PATH="${AGENT_KERNEL_TOLBERT_NODES_PATH:-/data/agentkernel/var/tolbert/agentkernel/nodes_agentkernel.jsonl}"
export AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH="${AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH:-/data/agentkernel/var/tolbert/agentkernel/label_map_agentkernel.json}"
export AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS="${AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS:-/data/agentkernel/var/tolbert/agentkernel/source_spans_agentkernel.jsonl}"
export AGENT_KERNEL_TOLBERT_CACHE_PATHS="${AGENT_KERNEL_TOLBERT_CACHE_PATHS:-/data/agentkernel/var/tolbert/agentkernel/retrieval_cache/agentkernel__tolbert_epoch10.pt}"
export AGENT_KERNEL_TOLBERT_DEVICE="${AGENT_KERNEL_TOLBERT_DEVICE:-cuda}"

python scripts/run_eval.py --provider "$AGENT_KERNEL_PROVIDER" --model "$AGENT_KERNEL_MODEL" "$@"
