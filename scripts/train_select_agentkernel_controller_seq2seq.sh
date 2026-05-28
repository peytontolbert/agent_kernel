#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

DATASET_MANIFEST="${DATASET_MANIFEST:-${1:-}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${2:-}}"
TOKENIZER_SOURCE_DIR="${TOKENIZER_SOURCE_DIR:-${3:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${4:-artifacts/agentkernel_controller/seq2seq_controller_family_gate_run}}"
BASELINE_REPORT="${BASELINE_REPORT:-trajectories/neural_controller/replay_default_preserve_repairs_v6_on_defended_v31/report.json}"
SELECTION_DATASET="${SELECTION_DATASET:-}"
SELECTION_OUTPUT_DIR="${SELECTION_OUTPUT_DIR:-}"
SELECTION_LIMIT="${SELECTION_LIMIT:-132}"
MAX_STEPS="${MAX_STEPS:-90}"
SAVE_EVERY="${SAVE_EVERY:-30}"
EVAL_EVERY="${EVAL_EVERY:-30}"

if [[ -z "$DATASET_MANIFEST" ]]; then
  echo "DATASET_MANIFEST or arg1 is required" >&2
  exit 2
fi
if [[ -z "$INIT_CHECKPOINT" ]]; then
  echo "INIT_CHECKPOINT or arg2 is required" >&2
  exit 2
fi
if [[ -z "$TOKENIZER_SOURCE_DIR" ]]; then
  echo "TOKENIZER_SOURCE_DIR or arg3 is required" >&2
  exit 2
fi
if [[ -z "$SELECTION_DATASET" ]]; then
  SELECTION_DATASET="$("$PYTHON_BIN" - "$DATASET_MANIFEST" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(manifest.get("eval_dataset_path", "")))
PY
)"
fi
if [[ -z "$SELECTION_OUTPUT_DIR" ]]; then
  SELECTION_OUTPUT_DIR="${OUTPUT_DIR%/}/checkpoint_selection"
fi

cd "$ROOT"

DATASET_MANIFEST="$DATASET_MANIFEST" \
INIT_CHECKPOINT="$INIT_CHECKPOINT" \
TOKENIZER_SOURCE_DIR="$TOKENIZER_SOURCE_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
MAX_STEPS="$MAX_STEPS" \
SAVE_EVERY="$SAVE_EVERY" \
EVAL_EVERY="$EVAL_EVERY" \
SCALAR_INVARIANT_CONTROL="${SCALAR_INVARIANT_CONTROL:-1}" \
SCALAR_INVARIANT_APPLY_ENCODER="${SCALAR_INVARIANT_APPLY_ENCODER:-1}" \
SCALAR_INVARIANT_APPLY_DECODER="${SCALAR_INVARIANT_APPLY_DECODER:-0}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
scripts/train_agentkernel_controller_seq2seq.sh

mapfile -t CHECKPOINTS < <(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' | sort)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "no checkpoints found in $OUTPUT_DIR/checkpoints" >&2
  exit 3
fi

SELECT_ARGS=(
  scripts/select_neural_controller_checkpoints.py
  --baseline-report "$BASELINE_REPORT"
  --template-manifest "$OUTPUT_DIR/agentkernel_controller_manifest.json"
  --dataset-path "$SELECTION_DATASET"
  --output-dir "$SELECTION_OUTPUT_DIR"
  --repo-root "$ROOT"
  --device "${SELECTION_DEVICE:-cuda}"
  --limit "$SELECTION_LIMIT"
  --max-new-tokens "${SELECTION_MAX_NEW_TOKENS:-192}"
  --max-encoder-tokens "${SELECTION_MAX_ENCODER_TOKENS:-1024}"
  --progress-every "${SELECTION_PROGRESS_EVERY:-16}"
)
if [[ "${SELECTION_RESUME_PARTIAL:-1}" != "0" ]]; then
  SELECT_ARGS+=(--resume-partial)
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
  SELECT_ARGS+=(--checkpoint "$checkpoint")
done

"$PYTHON_BIN" "${SELECT_ARGS[@]}"
