#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"

DATASET_MANIFEST="${1:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset/agentkernel_harness_skill_retrieval_dataset_manifest.json}"
OUTPUT_DIR="${2:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/harness_skill_retriever}"
INIT_CHECKPOINT="${3:-}"
TOKENIZER_DIR="${4:-}"

ARGS=(
  "$ROOT_DIR/scripts/train_agentkernel_lite_encdec.py"
  --dataset-manifest "$DATASET_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --preset agentkernel-lite-100m
  --encoder-position-embeddings 1
  --retrieval-contrastive-weight "${RETRIEVAL_CONTRASTIVE_WEIGHT:-0.12}"
  --retrieval-ternary-aware-weight "${RETRIEVAL_TERNARY_AWARE_WEIGHT:-0.12}"
  --retrieval-hard-negative-weight "${RETRIEVAL_HARD_NEGATIVE_WEIGHT:-0.15}"
  --retrieval-hard-negative-ternary "${RETRIEVAL_HARD_NEGATIVE_TERNARY:-1}"
  --retrieval-temperature "${RETRIEVAL_TEMPERATURE:-0.05}"
  --decoder-loss-weight "${DECODER_LOSS_WEIGHT:-0}"
  --parquet-require-retrieval-pair 1
  --parquet-task-type-include "${PARQUET_TASK_TYPE_INCLUDE:-harness_skill_retrieval,swe_live_failure_retrieval}"
  --max-encoder-tokens "${MAX_ENCODER_TOKENS:-256}"
  --max-decoder-tokens "${MAX_DECODER_TOKENS:-64}"
  --max-retrieval-query-tokens "${MAX_RETRIEVAL_QUERY_TOKENS:-160}"
  --max-retrieval-doc-tokens "${MAX_RETRIEVAL_DOC_TOKENS:-384}"
  --max-retrieval-negatives "${MAX_RETRIEVAL_NEGATIVES:-8}"
  --batch-size "${BATCH_SIZE:-16}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}"
  --max-steps "${MAX_STEPS:-1000}"
  --learning-rate "${LEARNING_RATE:-5e-5}"
  --log-every "${LOG_EVERY:-10}"
  --eval-every "${EVAL_EVERY:-100}"
  --checkpoint-every "${CHECKPOINT_EVERY:-250}"
  --dry-run "${DRY_RUN:-0}"
  --export-browser-bitnet "${EXPORT_BROWSER_BITNET:-0}"
  --device "${DEVICE:-cuda}"
)

if [[ -n "$TOKENIZER_DIR" ]]; then
  ARGS+=(--tokenizer-kind agentkernel-bpe --tokenizer-source-dir "$TOKENIZER_DIR" --agentkernel-special-tokens 1)
fi

if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARGS+=(--init-from-checkpoint "$INIT_CHECKPOINT" --checkpoint-vocab-mismatch strict)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
