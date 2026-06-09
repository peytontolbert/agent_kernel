# AgentKernel Benchmark Supervision Runbook

This runbook describes how to manage AgentKernel benchmark campaigns, diagnose failures, keep retrieval datasets current, and use the AgentKernel plus OpenClaw/Hermes skill datasets to improve benchmark score.

## 1. Keep The Three Data Sources Separate

Use three distinct stores during diagnosis and improvement:

- AgentKernel source skills:
  `/data/repo_skills_miner/artifacts/hf_agentkernel_source_skills/data/train.parquet`
- OpenClaw/Hermes harness skills:
  `/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet`
- AgentKernel benchmark traces and reports:
  `/data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs/`

AgentKernel source skills explain what the local code can do. OpenClaw/Hermes skills provide outside harness patterns. Benchmark traces are the ground truth for what actually failed.

## 2. Start Every Benchmark Round With A Manifest

Before launching a run, create or refresh the task manifest:

```bash
cd /data/agentkernel
python scripts/prepare_swe_bench_queue_manifest.py --help
python scripts/prepare_autonomous_benchmark_harness.py --help
```

Record:

- run name
- benchmark family
- task list
- model/controller config
- retrieval dataset versions
- git commit of `/data/agentkernel`
- git commit of `/data/repo_skills_miner`

Do not overwrite old run directories. Use timestamped names so failures remain inspectable.

## 3. Launch Runs Under Supervision

Use the autonomous harness or job queue entrypoints:

```bash
cd /data/agentkernel
python scripts/run_autonomous_benchmark_harness.py --help
python scripts/run_job_queue.py --help
```

Run long campaigns in `tmux`, and keep one terminal open for progress checks:

```bash
tmux new -s agentkernel-bench
```

The live state to watch is usually under:

```text
/data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs/<run_name>/checkpoints/
/data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs/<run_name>/reports/
```

## 4. Monitor Progress

Check active jobs first:

```bash
find /data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs \
  -path '*/checkpoints/*.progress.json' -type f -printf '%T@ %p\n' \
  | sort -n | tail -20
```

For a specific task:

```bash
python - <<'PY'
import json
from pathlib import Path

task = "python-babel__babel-1131"
root = Path("/data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs")
for p in sorted(root.glob(f"**/checkpoints/*{task}*.progress.json*")):
    try:
        obj = json.loads(p.read_text())
    except Exception:
        continue
    print("\n", p)
    print(json.dumps(obj, indent=2)[:2000])
PY
```

## 5. Diagnose A Failed Task

For each failure, inspect these files in order:

1. Progress file: current phase, step index, terminal state.
2. Report file: outcome, failure reason, retrieval counts, verifier result.
3. Checkpoint file: full history, decisions, command results, policy metadata.
4. Workspace: `patch.diff`, candidate source files, `source_lines/`, tests.

Use this extractor:

```bash
python - <<'PY'
import json
from pathlib import Path

task = "python-babel__babel-1131"
root = Path("/data/agentkernel/benchmarks/swe_bench_live/autonomous_harness_runs")
for p in sorted(root.glob(f"**/reports/*{task}*.json*")):
    obj = json.loads(p.read_text())
    print("\n###", p)
    print("generated:", obj.get("generated_at"))
    print("outcome:", obj.get("outcome"))
    print("termination:", obj.get("termination_reason"))
    print("failure:", obj.get("failure_reason"))
    print("summary:", obj.get("summary"))
    print("artifact_contract_failure:", obj.get("artifact_contract_failure"))
PY
```

Then inspect the checkpoint:

```bash
python - <<'PY'
import json
from pathlib import Path

checkpoint = Path("REPLACE_WITH_CHECKPOINT_JSON")
obj = json.loads(checkpoint.read_text())
print("status", obj.get("status"), "success", obj.get("success"), "term", obj.get("termination_reason"))
for step in obj.get("history", []):
    print("\n## step", step.get("index"), step.get("action"), step.get("decision_source"))
    print("content:", str(step.get("content", "")).replace("\n", " ")[:800])
    print("verification:", step.get("verification", {}).get("reasons"))
    print("metadata:", step.get("proposal_metadata"))
PY
```

## 6. Classify Root Cause

Every failed task should receive one primary root-cause label:

- `retrieval_absent`: no relevant retrieval candidates, or `trusted_retrieval_steps == 0`
- `retrieval_wrong`: retrieved skills are semantically related but operationally wrong
- `localization_wrong`: candidate files/functions are wrong
- `patch_materialization_failed`: no valid `patch.diff` was created
- `invalid_patch_syntax`: unified diff, Python syntax, or apply check failed
- `semantic_patch_wrong`: patch applies but changes the wrong behavior
- `verification_gap`: verifier accepted too little or did not run targeted tests
- `policy_loop_failure`: repeated no-progress, policy termination, or guard backoff
- `sandbox/tool_failure`: command blocked, timed out, or ran in wrong environment

For the Babel 1131 failure, the primary label is:

```text
patch_materialization_failed + policy_loop_failure
```

The contributing label is:

```text
retrieval_absent
```

because the completed reports showed `trusted_retrieval_steps: 0`.

## 7. Compare Against The Skill Datasets

For each root cause, query both skill datasets.

Use AgentKernel source skills when the failure is inside local code:

```bash
cd /data/repo_skills_miner
python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("artifacts/hf_agentkernel_source_skills/data/train.parquet")
df = pd.read_parquet(path)
query = "artifact materialization guard patch diff fail to pass retrieval"
mask = False
for col in ["source_path", "qualname", "source_excerpt", "summary", "labels"]:
    if col in df:
        mask = mask | df[col].astype(str).str.contains(query.replace(" ", "|"), case=False, regex=True, na=False)
print(df[mask].head(20)[["source_path", "qualname", "primitive_type"]])
PY
```

Use OpenClaw/Hermes when the failure is a harness pattern:

```bash
cd /data/repo_skills_miner
python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("artifacts/hf_openclaw_hermes_skills/data/train.parquet")
df = pd.read_parquet(path)
terms = "self improve memory skill recovery policy sandbox tool permission"
mask = False
for col in ["source_path", "qualname", "source_excerpt", "llm_summary", "labels"]:
    if col in df:
        mask = mask | df[col].astype(str).str.contains(terms.replace(" ", "|"), case=False, regex=True, na=False)
print(df[mask].head(30)[["source_repo", "source_path", "qualname", "primitive_type"]])
PY
```

The expected output is not an auto-patch. It is a short list of transferable patterns to implement in AgentKernel.

## 8. Update The AgentKernel Dataset After Code Changes

After changing AgentKernel, regenerate the source skills dataset from `/data/agentkernel` while excluding heavy benchmark/cache/generated data.

Use repo-skills-miner and keep output in Parquet:

```bash
cd /data/repo_skills_miner
python -m skill_engine.cli ingest /data/agentkernel \
  --repo-label agentkernel \
  --out artifacts/agentkernel_source_run_$(date -u +%Y%m%dT%H%M%SZ) \
  --exclude benchmarks/repo_cache \
  --exclude benchmarks/swe_bench_live/autonomous_harness_runs \
  --exclude workspace \
  --exclude var \
  --exclude .git
```

Then build or refresh the Hugging Face style Parquet dataset:

```bash
python scripts/build_hf_skills_dataset.py --help
```

Do not use JSONL for large runs. Keep intermediate worker outputs and final datasets as Parquet.

## 9. Build Retrieval And Action Datasets From Failures

For benchmark failures, build a failure retrieval dataset:

```bash
cd /data/agentkernel
python scripts/build_swe_live_failure_retrieval_dataset.py --help
```

For patch-diff action learning, build patch action data:

```bash
python scripts/build_patch_action_dataset.py --help
```

For harness skill retrieval, connect the OpenClaw/Hermes dataset:

```bash
python scripts/build_harness_skill_retrieval_dataset.py --help
```

Each training row should include:

- task prompt
- repo context
- failing trace
- retrieved skill ids
- patch attempted
- verifier result
- root-cause label
- final fix if known
- skill credit assignment

## 10. Train Retrieval Before Training Patch Generation

Train or refresh retrieval first:

```bash
cd /data/agentkernel
python scripts/train_agentkernel_lite_encdec.py --help
python scripts/train_agentkernel_lite_ternary_retrieval.sh --help
python scripts/train_agentkernel_lite_harness_skill_retrieval.sh --help
```

Gate promotion with retrieval metrics:

- top-1 accuracy
- MRR
- hard-negative accuracy
- task-level retrieval influence
- trusted retrieval step count
- downstream patch success

Do not promote a retriever only because embeddings look semantically close. Promote it only when benchmark traces show it changes the patch outcome.

## 11. Improve AgentKernel From Diagnosed Failures

For each root cause:

1. Write a minimal failing regression test in `/data/agentkernel/tests/`.
2. Patch AgentKernel policy/runtime code.
3. Run the targeted test.
4. Replay the failed benchmark task.
5. Rebuild AgentKernel source skills.
6. Rebuild failure retrieval data.
7. Retrain or fine-tune the retrieval/action model.
8. Re-run the benchmark subset.
9. Promote only if score improves without regressions.

Useful test areas:

- `tests/test_policy.py`
- `tests/test_recovery.py`
- `tests/test_verifier.py`
- `tests/test_build_swe_live_failure_retrieval_dataset.py`
- `tests/test_build_harness_skill_retrieval_dataset.py`

## 12. Promotion Gates

A fix is not ready for full benchmark promotion until all are true:

- targeted regression passes
- failed benchmark task improves or reaches a new failure mode
- no increase in policy termination rate
- no increase in hidden side-effect risk
- `patch.diff` materialization rate improves
- retrieval influence increases only when retrieval is relevant
- generated patches apply cleanly
- verifier catches known bad patches

For SWE-style tasks, additionally require:

- no comment/docstring-only patches
- no definition-header removal
- no fake test stubs
- no constant placeholder returns
- no edits outside likely relevant files unless justified by retrieval evidence

## 13. Root-Cause Report Template

Every failed task should get this report:

```text
Task:
Run:
Status:
Primary root cause:
Contributing causes:
Patch status:
Retrieval status:
Trusted retrieval steps:
Relevant files:
Bad decision source:
Verifier result:
What the correct strategy should have been:
Missing skill/dataset coverage:
AgentKernel code to patch:
Regression test to add:
Replay command:
Promotion gate:
```

## 14. Practical Target For Perfect Scoring

Perfect benchmark scoring requires a closed loop:

```text
failure trace
-> root-cause label
-> missing skill or policy weakness
-> regression test
-> AgentKernel patch
-> updated AgentKernel skills dataset
-> updated failure/action retrieval dataset
-> retriever/action model refresh
-> replay failed task
-> subset benchmark
-> full benchmark
```

The highest-priority current gaps are:

- fail-to-pass scoped patch materialization
- trusted retrieval activation on SWE failures
- hard-negative retrieval for semantically similar but operationally wrong patches
- semantic patch sanity checks
- verifier tests that reject placeholder constants and off-target edits
- automatic trace-to-training-row credit assignment

The OpenClaw/Hermes dataset should be used mainly for harness behaviors: self-improvement loops, memory update, skill mutation, tool boundary design, and recovery policy. AgentKernel source skills should be used for local implementation details.

## 15. Backtest The Runbook Against Failed Tasks

After changing the supervision process, run the failure backtest:

```bash
cd /data/agentkernel
python scripts/backtest_benchmark_failure_runbook.py \
  --output-dir artifacts/benchmark_failure_backtests/runbook_backtest_$(date -u +%Y%m%dT%H%M%SZ)
```

Outputs:

- `summary.md`
- `summary.json`
- `all_failed_report_backtest.parquet`
- `latest_failed_task_backtest.parquet`

Default mode uses compact skill metadata so it is fast enough to run interactively. For a slower offline pass that includes source excerpts in skill coverage checks:

```bash
python scripts/backtest_benchmark_failure_runbook.py \
  --include-source-excerpt \
  --output-dir artifacts/benchmark_failure_backtests/runbook_backtest_deep_$(date -u +%Y%m%dT%H%M%SZ)
```

Interpretation rules:

- If most failures are `patch_materialization_failed`, fix artifact generation and patch builder policy before training a larger model.
- If `trusted_retrieval_steps` is zero across failures, retrieval is not reaching the control path; fix retrieval activation and trace-to-skill routing.
- If skill hits are nonzero but retrieval is absent, the dataset has useful material but AgentKernel is not using it.
- If failures are mostly `semantic_patch_wrong`, add verifier and semantic sanity checks before accepting generated diffs.
- If failures are mostly `localization_wrong`, improve fail-to-pass function/file scoping before widening search.

The latest backtest at `artifacts/benchmark_failure_backtests/runbook_backtest_20260518` found:

- 445 failed report artifacts
- 12 unique failed tasks
- 445/445 reports with zero trusted retrieval
- latest failed task root cause: `patch_materialization_failed` for all 12 tasks

That result means the immediate benchmark bottleneck is not lack of generic code knowledge. It is that failed tasks are not reliably materializing a valid `patch.diff`, and trusted retrieval is not being activated before the policy loop terminates.
