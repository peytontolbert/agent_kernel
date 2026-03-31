#!/usr/bin/env bash
set -euo pipefail

cd /data/agentkernel

provider="${AGENT_KERNEL_PROVIDER:-vllm}"
model="${AGENT_KERNEL_MODEL:-Qwen/Qwen3.5-9B}"
runtime_root="${AGENT_KERNEL_VERIFY_ROOT:-/tmp/agentkernel_verify}"
seed_skills_path="${AGENT_KERNEL_VERIFY_SEED_SKILLS_PATH:-/data/agentkernel/var/skills/bootstrap_command_skills.json}"
verify_capture_root="$(mktemp -d "${TMPDIR:-/tmp}/agentkernel_verify_capture.XXXXXX")"
archive_root="${AGENT_KERNEL_VERIFY_ARCHIVE_ROOT:-${runtime_root}_archive}"
archive_run_dir="$archive_root/$(date -u +%Y%m%dT%H%M%SZ)"

cleanup() {
  rm -rf "$verify_capture_root"
}

trap cleanup EXIT

run_eval() {
  python scripts/run_eval.py --provider "$provider" --model "$model" "$@"
}

archive_stage() {
  local stage="$1"
  local stage_dir="$archive_run_dir/$stage"
  mkdir -p "$stage_dir"
  if [[ -d "$runtime_root/trajectories" ]]; then
    cp -R "$runtime_root/trajectories" "$stage_dir/trajectories"
  fi
  if [[ -f "$verify_capture_root/full_eval.txt" ]]; then
    cp "$verify_capture_root/full_eval.txt" "$stage_dir/full_eval.txt"
  fi
  if [[ -f "$verify_capture_root/tolbert_compare.txt" ]]; then
    cp "$verify_capture_root/tolbert_compare.txt" "$stage_dir/tolbert_compare.txt"
  fi
  if [[ -f "$verify_capture_root/tolbert_features.txt" ]]; then
    cp "$verify_capture_root/tolbert_features.txt" "$stage_dir/tolbert_features.txt"
  fi
  if [[ -f "$verify_capture_root/skill_compare.txt" ]]; then
    cp "$verify_capture_root/skill_compare.txt" "$stage_dir/skill_compare.txt"
  fi
  if [[ -f "$verify_capture_root/baseline_eval.txt" ]]; then
    cp "$verify_capture_root/baseline_eval.txt" "$stage_dir/baseline_eval.txt"
  fi
  if [[ -f "$verify_capture_root/tolbert_first_steps.txt" ]]; then
    cp "$verify_capture_root/tolbert_first_steps.txt" "$stage_dir/tolbert_first_steps.txt"
  fi
  if [[ -f "$verify_capture_root/failure_recovery.txt" ]]; then
    cp "$verify_capture_root/failure_recovery.txt" "$stage_dir/failure_recovery.txt"
  fi
}

run_eval_main() {
  run_eval \
    --use-tolbert-context 1 \
    --use-skills 1 \
    --include-episode-memory \
    --include-skill-memory \
    --include-tool-memory \
    --include-verifier-memory \
    --include-curriculum \
    --include-failure-curriculum
}

run_eval_compare_tolbert() {
  run_eval \
    --use-tolbert-context 0 \
    --use-skills 1 \
    --include-episode-memory \
    --include-skill-memory \
    --include-tool-memory \
    --include-verifier-memory \
    --include-curriculum \
    --include-failure-curriculum \
    --compare-tolbert
}

run_eval_compare_tolbert_features() {
  run_eval \
    --use-tolbert-context 1 \
    --use-skills 1 \
    --include-episode-memory \
    --include-skill-memory \
    --include-tool-memory \
    --include-verifier-memory \
    --include-curriculum \
    --include-failure-curriculum \
    --compare-tolbert-features
}

run_eval_compare_skills() {
  run_eval \
    --use-tolbert-context 1 \
    --include-episode-memory \
    --include-skill-memory \
    --include-tool-memory \
    --include-verifier-memory \
    --include-curriculum \
    --include-failure-curriculum \
    --compare-skills
}

run_eval_baseline() {
  run_eval \
    --use-tolbert-context 0 \
    --use-skills 0 \
    --include-episode-memory \
    --include-skill-memory \
    --include-tool-memory \
    --include-verifier-memory \
    --include-curriculum \
    --include-failure-curriculum
}

rm -rf "$runtime_root"
mkdir -p "$runtime_root"/workspace "$runtime_root"/trajectories/episodes "$runtime_root"/trajectories/skills
mkdir -p "$runtime_root"/trajectories/operators "$runtime_root"/trajectories/tools "$runtime_root"/trajectories/benchmarks
mkdir -p "$runtime_root"/trajectories/retrieval "$runtime_root"/trajectories/verifiers
mkdir -p "$runtime_root"/trajectories/prompts "$runtime_root"/trajectories/curriculum
mkdir -p "$runtime_root"/trajectories/improvement "$runtime_root"/trajectories/improvement/reports
mkdir -p "$archive_run_dir"

export AGENT_KERNEL_WORKSPACE_ROOT="$runtime_root/workspace"
export AGENT_KERNEL_TRAJECTORIES_ROOT="$runtime_root/trajectories/episodes"
export AGENT_KERNEL_SKILLS_PATH="$runtime_root/trajectories/skills/command_skills.json"
export AGENT_KERNEL_OPERATOR_CLASSES_PATH="$runtime_root/trajectories/operators/operator_classes.json"
export AGENT_KERNEL_TOOL_CANDIDATES_PATH="$runtime_root/trajectories/tools/tool_candidates.json"
export AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH="$runtime_root/trajectories/benchmarks/benchmark_candidates.json"
export AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH="$runtime_root/trajectories/retrieval/retrieval_proposals.json"
export AGENT_KERNEL_VERIFIER_CONTRACTS_PATH="$runtime_root/trajectories/verifiers/verifier_contracts.json"
export AGENT_KERNEL_PROMPT_PROPOSALS_PATH="$runtime_root/trajectories/prompts/prompt_proposals.json"
export AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH="$runtime_root/trajectories/curriculum/curriculum_proposals.json"
export AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH="$runtime_root/trajectories/improvement/cycles.jsonl"
export AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR="$runtime_root/trajectories/improvement/reports"

if [[ -f "$seed_skills_path" ]]; then
  cp "$seed_skills_path" "$AGENT_KERNEL_SKILLS_PATH"
fi

printf '{\n  "provider": "%s",\n  "model": "%s",\n  "runtime_root": "%s",\n  "archive_run_dir": "%s"\n}\n' \
  "$provider" \
  "$model" \
  "$runtime_root" \
  "$archive_run_dir" > "$archive_run_dir/manifest.json"

echo "[1/6] real agent with Tolbert and skills"
python scripts/run_agent.py \
  --task-id hello_task \
  --provider "$provider" \
  --model "$model" \
  --use-tolbert-context 1 \
  --use-skills 1
archive_stage "step1_agent"

echo "[2/6] real eval with Tolbert and skills"
run_eval_main | tee "$verify_capture_root/full_eval.txt"
archive_stage "step2_full_eval"

echo "[3/6] compare Tolbert"
run_eval_compare_tolbert | tee "$verify_capture_root/tolbert_compare.txt"
run_eval_compare_tolbert_features | tee "$verify_capture_root/tolbert_features.txt"
python scripts/report_tolbert_first_steps.py \
  --trajectories-root "$runtime_root/trajectories/episodes" \
  --benchmark-family project \
  --benchmark-family repository \
  --benchmark-family tooling \
  --benchmark-family workflow \
  --failures-only > "$verify_capture_root/tolbert_first_steps.txt"
archive_stage "step3_tolbert_compare"

echo "[4/6] refresh local skills from successful episodes"
python scripts/extract_skills.py
python scripts/extract_operators.py
python scripts/extract_tools.py
archive_stage "step4_refresh_memory"

echo "[5/6] compare skills"
run_eval_compare_skills | tee "$verify_capture_root/skill_compare.txt"
archive_stage "step5_skill_compare"

echo "[6/6] diagnostic baseline without Tolbert or skills"
run_eval_baseline | tee "$verify_capture_root/baseline_eval.txt"
python scripts/report_failure_recovery.py \
  --trajectories-root "$runtime_root/trajectories/episodes" > "$verify_capture_root/failure_recovery.txt"
archive_stage "step6_baseline"

python scripts/assert_verify_metrics.py \
  --full-eval "$verify_capture_root/full_eval.txt" \
  --tolbert-compare "$verify_capture_root/tolbert_compare.txt" \
  --tolbert-features "$verify_capture_root/tolbert_features.txt" \
  --skill-compare "$verify_capture_root/skill_compare.txt" \
  --baseline-eval "$verify_capture_root/baseline_eval.txt" \
  --tolbert-first-steps "$verify_capture_root/tolbert_first_steps.txt" \
  --failure-recovery "$verify_capture_root/failure_recovery.txt"

echo "verify_archive=$archive_run_dir"
