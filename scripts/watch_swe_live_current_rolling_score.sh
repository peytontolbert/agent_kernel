#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/data/agentkernel}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"

cd "$ROOT_DIR"

while true; do
  date -Is
  "$PYTHON_BIN" scripts/refresh_swe_live_official_rolling_score.py \
    --queue-json benchmarks/swe_bench_live/autonomous_harness_runs/raw_completed_verifier_hardening_r1_20260519/queue_raw_completed_verifier_hardening_r1_20260519/queue.json \
    --queue-manifest benchmarks/swe_bench_live/autonomous_harness_runs/raw_completed_verifier_hardening_r1_20260519/queue_manifest_raw_completed_verifier_hardening_r1_20260519.json \
    --prediction-task-manifest benchmarks/swe_bench_live/autonomous_harness_runs/raw_completed_verifier_hardening_r1_20260519/prediction_tasks_raw_completed_verifier_hardening_r1_20260519.json \
    --workspace-root /data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs/verified_leaderboard/workspaces \
    --output-root benchmarks/swe_bench_live/rolling_score/raw_completed_verifier_hardening_r1_20260519 \
    --swe-bench-live-root /data/agentkernel/other_repos/SWE-bench-Live \
    --python "$PYTHON_BIN" \
    --workers 2 \
    --launch-evaluator
  sleep "$INTERVAL_SECONDS"
done
