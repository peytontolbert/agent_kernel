#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

DATASET_MANIFEST="${DATASET_MANIFEST:-${1:-artifacts/agentkernel_lite_encdec/controller_trace_x5_plus_retrieval_chatmix_v1/agentkernel_lite_encdec_dataset_manifest.json}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${2:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${3:-artifacts/agentkernel_controller/seq2seq_controller_stage1}}"
TOKENIZER_SOURCE_DIR="${TOKENIZER_SOURCE_DIR:-}"
MAX_STEPS="${MAX_STEPS:-3000}"
PRESET="${PRESET:-agentkernel-lite-100m}"

cd "$ROOT"

ARGS=(
  scripts/train_agentkernel_lite_encdec.py
  --dataset-manifest "$DATASET_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --preset "$PRESET"
  --tokenizer-kind agentkernel-bpe
  --agentkernel-special-tokens 1
  --retrieval-head-dim 256
  --agent-policy-heads 1
  --policy-head-loss-weight 0.12
  --scalar-invariant-control "${SCALAR_INVARIANT_CONTROL:-0}"
  --scalar-invariant-rank "${SCALAR_INVARIANT_RANK:-32}"
  --scalar-invariant-epsilon "${SCALAR_INVARIANT_EPSILON:-0.05}"
  --scalar-invariant-smoothing-steps "${SCALAR_INVARIANT_SMOOTHING_STEPS:-1}"
  --scalar-invariant-apply-encoder "${SCALAR_INVARIANT_APPLY_ENCODER:-1}"
  --scalar-invariant-apply-decoder "${SCALAR_INVARIANT_APPLY_DECODER:-0}"
  --retrieval-contrastive-weight 0.08
  --retrieval-temperature 0.05
  --decoder-loss-weight 1.0
  --teacher-distill-weight "${TEACHER_DISTILL_WEIGHT:-0.0}"
  --teacher-distill-temperature "${TEACHER_DISTILL_TEMPERATURE:-1.0}"
  --max-encoder-tokens "${MAX_ENCODER_TOKENS:-1024}"
  --max-decoder-tokens "${MAX_DECODER_TOKENS:-512}"
  --max-retrieval-query-tokens "${MAX_RETRIEVAL_QUERY_TOKENS:-96}"
  --max-retrieval-doc-tokens "${MAX_RETRIEVAL_DOC_TOKENS:-256}"
  --batch-size "${BATCH_SIZE:-4}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}"
  --max-steps "$MAX_STEPS"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --weight-decay 0.01
  --clip-grad-norm 1.0
  --log-every 25
  --eval-every "${EVAL_EVERY:-250}"
  --max-eval-batches 64
  --checkpoint-every "${SAVE_EVERY:-500}"
  --checkpoint-include-optimizer "${CHECKPOINT_INCLUDE_OPTIMIZER:-1}"
  --dry-run 0
  --export-browser-bitnet 0
)

if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARGS+=(--init-from-checkpoint "$INIT_CHECKPOINT" --checkpoint-vocab-mismatch expand)
fi
if [[ -n "$TOKENIZER_SOURCE_DIR" ]]; then
  ARGS+=(--tokenizer-source-dir "$TOKENIZER_SOURCE_DIR")
fi

"$PYTHON_BIN" "${ARGS[@]}"

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
manifest_path = output_dir / "agentkernel_lite_encdec_manifest.json"
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
payload["artifact_kind"] = "agentkernel_controller_seq2seq_bundle"
payload["model_family"] = "agentkernel_controller_seq2seq_v1"
payload["full_agent_kernel_controller"] = True
payload["source_scaffold"] = "agentkernel_lite_encdec_trainer"
payload["runtime_targets"] = {
    "kernel": "full_agent_kernel_shadow_advisory_neural_controller",
    "authority": "shadow_advisory_until_retained_promotion",
}
payload.setdefault("training_summary", {})["runtime_target"] = "full_agent_kernel_neural_controller"
payload.setdefault("training_summary", {})["primary_authority_allowed"] = False
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output_dir / "agentkernel_controller_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"full_kernel_controller_manifest={output_dir / 'agentkernel_controller_manifest.json'}")
PY
