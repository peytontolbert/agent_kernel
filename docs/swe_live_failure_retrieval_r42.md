# SWE-live r42 Failure Retrieval

Analysis target:

```text
benchmarks/swe_bench_live/autonomous_harness_runs/official_score_feedback_r42_20260514
```

Retrieval artifact:

```text
artifacts/swe_live_failure_retrieval_r42.json
```

Fresh verification artifact:

```text
artifacts/swe_live_r42_fresh_patch_jobs_verification.json
artifacts/swe_live_r42_fresh_predictions.jsonl
```

## Findings

The published r42 prediction file contains four empty `model_patch` values.
That was a collection-timing failure, not a total patch-generation failure.

The stale verification file in the run directory was created at
`2026-05-14T17:58:23Z` and marked all four tasks failed/nonterminal. Later job
reports show verified patches for:

- `pvlib__pvlib-python-2190`
- `pvlib__pvlib-python-2249`
- `pytorch__torchtune-1697`

A fresh verification now finds two valid patches:

- `pvlib__pvlib-python-2190`
- `pvlib__pvlib-python-2249`

It marks `pytorch__torchtune-1697` as an abstention because the semantic
artifact verifier rejects the one-line replacement as too weak, and
`python-babel__babel-1131` is still `in_progress`.

## Retrieved Skill Signals

For the stale queue/restart failure, the harness-skill retriever returned:

- `docs/concepts/queue.md`: queue troubleshooting
- `docs/automation/cron-jobs.md`: job management
- `src/process/command-queue.ts`: command queue symbol index
- `docs/cli/gateway.md`: restart flow

This points to a queue/restart-state bug: `verify_patch_jobs` and
`collect_predictions` consumed a stale queue/verification snapshot while later
job attempts continued and eventually produced patches.

For the official SWE-bench-Live scoring failure, retrieval returned Docker
sandbox/setup skills:

- `scripts/e2e/docker-openai-seed.ts`
- `docs/install/docker.md`
- `src/agents/sandbox/docker-backend.ts`
- `docs/gateway/sandboxing.md`

The concrete error was:

```text
ModuleNotFoundError: No module named 'docker'
```

The failing official harness used:

```text
/home/peyton/miniconda3/bin/python3
```

But the working environment is:

```text
/home/peyton/miniconda3/envs/ai/bin/python
```

The `ai` environment imports `docker` successfully.

## Root Causes

1. `collect_predictions` can validly produce empty no-op predictions when
   `patch_jobs_verification.json` is stale or generated before later successful
   patch reports.

2. `verify_patch_jobs` trusts `queue.json` state at the time it runs. In r42,
   queue/restart behavior left stale queued/in-progress state in the verifier
   output while subsequent manual fresh attempts produced patches.

3. `python-babel__babel-1131` repeatedly reaches
   `artifact_materialization_guard` with `artifact_missing_after_response`.
   The rejected retry command uses `--path patch.diff`, which is blocked because
   `patch.diff` cannot be the source path:

```text
swe_patch_builder --path patch.diff --replace-line 660 --with "d = date(2007, 4, 1)" --source-lines babel/dates.py.lines.2_format > patch.diff
```

4. SWE-bench-Live official scoring runs under the wrong Python interpreter for
   this machine. The base conda interpreter lacks the `docker` package.

## Repair Targets

1. Add a freshness guard before `collect_predictions`: reject
   `patch_jobs_verification.json` when it is older than any queue/job report or
   when `queue.json` currently contains newer successful jobs.

2. Add a success-continuation collector mode: if verification is stale, rerun
   `verify_patch_jobs` or collect only currently verified patches into a new
   output path instead of emitting empty no-op predictions.

3. Fix SWE-live retry harness generation so it preserves the configured
   `python_bin` instead of forcing `sys.executable` for live retries.

4. Add an official harness preflight for:

```text
python -c "import docker"
```

5. For Babel artifact repair, reject builder plans where `--path patch.diff`
   appears before the command reaches the sandbox, and route to the candidate
   source path from `--source-lines` instead.

## Follow-up Training

The exact SWE-live failure operators are now materialized as retrieval rows:

```text
artifacts/agentkernel_lite_encdec/swe_live_failure_retrieval_dataset/
```

This dataset contains five operators repeated into 1,000 training/eval rows:

- stale verification freshness guard
- late successful patch collection
- official harness Python/Docker preflight
- `patch.diff` source-path repair
- semantic artifact abstention

The targeted retriever was trained from the OpenClaw/Hermes harness-skill
retriever:

```text
artifacts/agentkernel_lite_encdec/swe_live_failure_retriever_r1/
```

Evaluation:

| Bundle | Dataset | Top-1 | MRR |
| --- | --- | ---: | ---: |
| `harness_skill_retriever_r1` | SWE-live failure operators | 0.8000 | 0.8500 |
| `swe_live_failure_retriever_r1` | SWE-live failure operators | 1.0000 | 1.0000 |
| `swe_live_failure_retriever_r1` | held-out mined harness skills, 1,024 pairs | 0.9922 | 0.9961 |

The broader merged continuation is also available, but a short sequential run
mostly saw the original mined-skill rows before reaching the SWE-live examples:

```text
artifacts/agentkernel_lite_encdec/harness_skill_retriever_r2_swe_live/
```

Use the targeted `swe_live_failure_retriever_r1` bundle for diagnosing these
exact AgentKernel SWE-live harness failures.
