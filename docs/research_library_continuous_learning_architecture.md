# Research Library Continuous Learning Architecture

Status: deeper design  
Date: 2026-04-27

For the full 1M-paper parquet source-of-truth, streaming TOLBERT training, and
incremental code-span sync design, see
`docs/research_library_full_corpus_tolbert_sync.md`.

## Executive Summary

The right integration is a sidecar knowledge system, not a direct rewrite of the kernel. The kernel should keep its current planner, verifier, policy, and Agent Kernel TOLBERT assets stable. A separate research-library retrieval and learning layer should grow continuously from the 1M paper corpus, repository graph library, algorithms library, and benchmark run outcomes.

The model should gain a growing knowledge base primarily through external memory and retrieval indexes. We should update model weights only through versioned, gated promotions. This avoids corrupting the kernel while still giving it much larger context reach.

The target architecture has three layers:

1. Stable kernel layer: current `agent_kernel` planner, policy, verifier, world model, graph memory, and Agent Kernel TOLBERT runtime.
2. Research knowledge layer: immutable generations of papers, repos, algorithms, graphs, embeddings, TOLBERT spans, and retrieval caches.
3. Learning layer: retrieval logs, benchmark outcomes, ranker training, TOLBERT retraining, and artifact promotion gates.

## Current State

`agent_kernel` already has two TOLBERT surfaces:

- Primary Agent Kernel TOLBERT:
  - config fields: `AGENT_KERNEL_TOLBERT_*`
  - default assets under `/data/agentkernel/var/tolbert/agentkernel`
  - used for task memory, command/procedure retrieval, and kernel-local context compilation.

- Auxiliary paper research TOLBERT:
  - config fields: `AGENT_KERNEL_PAPER_RESEARCH_*`
  - wired inside `agent_kernel/extensions/tolbert.py`
  - queried by `TolbertContextCompiler._query_paper_research`
  - merged into normal retrieval with `retrieval_source=paper_research`.

The auxiliary path is a good extension point. It already uses the same `TolbertQueryClient` protocol and can merge results into the existing `ContextPacket` without changing planner or policy contracts.

### Sync Gap

The full 1M-paper corpus exists locally as parquet and is the source of truth:

- `/arxiv/huggingface/paper_text_1m_dedup_v1`: exactly `1,000,000` paper rows, `21G`
- `/data/tmp/p1_full_paper_lm_hf_all_chunks`: `17,117,443` chunk examples from the 1M corpus, `35G`
- `/data/repository_library/exports/_paper_universe`: `1,000,000` paper nodes, metadata embeddings, full-text embeddings, KNN edges, and topic edges

The newer TOLBERT joint v2 projection exists locally:

- `/data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/nodes_joint_v2.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/label_map_joint_v2.json`
- `/data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml`

Counts observed:

- code spans: 101,118
- paper paragraph spans: 3,939,404
- nodes: 336,480

But the expected v2 runtime checkpoint/cache directory is absent:

- missing: `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2`

The available trained research bundle is older:

- checkpoints: `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/tolbert_epoch1.pt` through `tolbert_epoch5.pt`
- cache: `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/retrieval_cache/paper_spans_joint_mapped__tolbert_epoch3.pt`
- old paper spans: 234,986
- old code spans: 101,118

So the kernel can use an older paper research cache, but it is not yet synchronized with the 3.9M paragraph-span joint v2 projection or the full 1M paper system. The 3.9M paper spans are derived TOLBERT training/runtime artifacts, not the source-of-truth paper corpus.

### Important Constraint

The old checkpoint cannot simply be pointed at the v2 config as a full model load. The old and v2 configs have different hierarchy head shapes:

- old config includes levels 1..5 with level sizes like `1:2, 2:12, 3:28, 4:354, 5:336104`
- v2 config includes levels 1..4 with sizes `1:2, 2:2, 3:340, 4:336104`

Because the classification heads differ, a safe v2 update is either:

- train v2 from scratch using the v2 config; or
- partial warm-start only the shared encoder/projection weights, reinitialize hierarchy heads, then train v2.

Do not overwrite the current Agent Kernel TOLBERT checkpoint or switch runtime paths to v2 until a promoted artifact passes gates.

## Principle: Index First, Train Second

For a growing corpus, putting everything into model weights is the wrong first move. Model weights should learn routing, alignment, ranking, and compression. The full corpus should live in:

- raw source snapshots;
- graph exports;
- span files;
- embedding shards;
- lexical indexes;
- vector indexes;
- cross-domain alignment edges;
- retrieval logs.

The model should query this external memory. Periodic retraining should improve retrieval and ranking, not replace the knowledge store.

This gives three benefits:

- growth is cheap: new papers/repos can be indexed without retraining the whole model;
- failure isolation is strong: a bad index generation can be rolled back;
- benchmark evidence stays clean: every retrieved item has provenance and can be audited.

## Target Knowledge Generation Layout

Create immutable generations under `var/research_library/generations/`.

Example:

```text
var/research_library/
  active_generation.json
  generations/
    20260427T000000Z/
      generation_manifest.json
      sources/
        repository_library_status.json
        paper_universe_status.json
        tolbert_brain_status.json
        algorithms_status.json
      spans/
        repo_spans_manifest.json
        paper_spans_manifest.json
        algorithm_spans.jsonl
      indexes/
        lexical/
        vector/
        graph/
        tolbert/
      models/
        reranker/
        tolbert_research/
      evals/
        retrieval_eval.json
        benchmark_shadow_eval.json
      promotion/
        gate_report.json
```

`active_generation.json` should be a small pointer to the currently promoted generation. Runtime services read only that pointer and generation manifests. New generations are built in a staging directory and promoted atomically.

## Research Retriever Design

Add a `ResearchLibraryQueryClient` that implements the existing `TolbertQueryClient` protocol. That keeps integration non-invasive because `TolbertContextCompiler` already knows how to query an auxiliary research client and merge results.

The client should return the same shape as a TOLBERT query:

```json
{
  "backend": "research_library_ensemble",
  "device": "cpu|cuda|mixed",
  "index_shards": ["paper_universe:generation", "repo_graph:generation"],
  "level_focus": "corpus",
  "path_prediction": {
    "tree_version": "research_library_v1",
    "decode_mode": "ensemble_route",
    "confidence_by_level": {"1": 0.0}
  },
  "retrieval": {
    "branch_scoped": [],
    "fallback_scoped": [],
    "global": []
  }
}
```

Each retrieved item should conform to the current span shape used by `ContextPacket.retrieval`:

```json
{
  "span_id": "paper:arxiv_id:paragraph:123",
  "text": "short preview",
  "source_id": "paper or repo id",
  "span_type": "doc:paper_paragraph|repo:symbol|algorithm:implementation",
  "score": 0.0,
  "node_path": [],
  "metadata": {
    "retrieval_source": "research_library",
    "source_kind": "paper|repo|algorithm",
    "generation_id": "20260427T000000Z",
    "path": "/data/...",
    "locator": "paragraph/span/symbol",
    "dataset": "PeytonT/1m_papers_text|PeytonT/repo_graph|local"
  }
}
```

## Multi-Source Retrieval Pipeline

Use a retrieval ensemble with late fusion:

1. Query analysis:
   - benchmark family: SWE-Bench, SWE-ReBench, RE-Bench, MLE-Bench, Codeforces;
   - task type: bug fix, algorithm design, ML experiment, systems optimization, proof/research;
   - known repo/package names;
   - error messages, symbols, files, failing tests;
   - algorithm constraints.

2. Candidate generation:
   - Agent Kernel TOLBERT for kernel memory and prior successful workflows;
   - research TOLBERT for cross-domain code/paper spans;
   - repository graph library for repos, files, symbols, source snippets, QA;
   - paper universe for 1M-paper metadata, embeddings, topics, KNN neighborhoods;
   - algorithms catalog for problem/constraint/complexity mapping;
   - lexical fallback for exact error messages and symbol names.

3. Normalization:
   - convert all candidates into common retrieved-span records;
   - attach source kind, generation id, dataset id, path, locator, and contamination policy.

4. Reranking:
   - semantic score;
   - exact symbol/error overlap;
   - benchmark family match;
   - repository match;
   - algorithm constraint match;
   - prior outcome labels from kernel traces;
   - freshness and source reliability.

5. Context packaging:
   - give planner compact candidates;
   - give editor source snippets only for selected candidates;
   - give verifier constraints, citations, tests, algorithm complexity, and source provenance.

6. Logging:
   - log every candidate shown, injected, selected, ignored, and associated outcome.

## Safe Integration Into Agent Kernel

The lowest-risk code path is:

1. Keep `TolbertContextCompiler.compile` as the main context compiler.
2. Replace or extend `_ensure_research_client` so it can instantiate `ResearchLibraryQueryClient` when `AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT=1`.
3. Keep the current `paper_research` path working as a fallback.
4. Mark all new records with `retrieval_source=research_library`.
5. Let `_merge_research_retrieval`, `_dedupe_and_rank`, `_apply_source_diversity`, and `_select_context_chunks` continue doing normal kernel-safe budgeting.

This avoids touching the planner and policy first. The research library becomes another retrieval backend, not a new control system.

Recommended config additions:

```text
AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT=0
AGENT_KERNEL_RESEARCH_LIBRARY_GENERATION_MANIFEST=var/research_library/active_generation.json
AGENT_KERNEL_RESEARCH_LIBRARY_QUERY_MODE=auto
AGENT_KERNEL_RESEARCH_LIBRARY_MAX_REPOS=5
AGENT_KERNEL_RESEARCH_LIBRARY_MAX_PAPERS=8
AGENT_KERNEL_RESEARCH_LIBRARY_MAX_ALGORITHMS=5
AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_CHAR_BUDGET=4000
AGENT_KERNEL_RESEARCH_LIBRARY_SHADOW_ONLY=1
```

Start disabled or shadow-only. Promote to active only after the retrieval logs show it helps.

## TOLBERT Research Model Update Plan

### Stage 1: Make The Full Corpus Indexable Without Training

Before training full-corpus research TOLBERT, expose the already-built 1M paper universe and repository graph through retrieval clients:

- paper universe vector index from existing parquet embeddings;
- paper universe full-text embedding index;
- paper KNN/topic/category/year graph expansion;
- repository graph lookup from `/data/repository_library/exports`;
- algorithm in-memory index from `/data/algorithms`.
- parquet-backed text fetch by paper/chunk pointer.

This immediately exposes more data without risking a broken TOLBERT checkpoint.

### Stage 2: Build Full-Corpus Research TOLBERT

Train a research TOLBERT sidecar over the full paper/code projection. The current JSONL `TreeOfLifeDataset` loads records into memory, so full 1M/17M training needs a streaming or sharded dataset before using the whole corpus.

Initial command shape after the streaming dataset exists:

```bash
cd /data/TOLBERT_BRAIN
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_tolbert.py \
  --config configs/tolbert_brain_full_corpus.yaml \
  --device cuda
```

Training options:

- safest: train from `bert-base-uncased` using bounded hierarchy heads;
- faster but requires code support: partial warm-start encoder/projection from the old checkpoint, skip mismatched hierarchy heads.

Do not make this the kernel runtime just because a checkpoint exists. It must get a retrieval cache and pass gates first.

### Stage 3: Build Sharded Retrieval Caches

Use `scripts/build_tolbert_cache.py` from Agent Kernel for JSONL span projections, or add a parquet-native cache builder for the full paper chunk dataset. Output to the research generation directory or the TOLBERT_BRAIN checkpoint cache dir.

The 3.9M paper-span cache must be sharded. Use the existing manifest format because `scripts/tolbert_service.py` already supports JSON cache manifests with shard paths and branch-presence metadata.

Command shape:

```bash
python /data/agentkernel/scripts/build_tolbert_cache.py \
  --config /data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml \
  --checkpoint /data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/tolbert_epoch5.pt \
  --spans /data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl \
  --out /data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/retrieval_cache/paper_spans_paragraphs_joint_v2__tolbert_epoch5.pt \
  --shard-size 50000 \
  --batch-size 64 \
  --device cuda
```

Also build a code-span cache:

```bash
python /data/agentkernel/scripts/build_tolbert_cache.py \
  --config /data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml \
  --checkpoint /data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/tolbert_epoch5.pt \
  --spans /data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl \
  --out /data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/retrieval_cache/code_spans_joint_v2__tolbert_epoch5.pt \
  --shard-size 50000 \
  --batch-size 64 \
  --device cuda
```

The service can load only a small number of shards at once with `AGENT_KERNEL_TOLBERT_MAX_LOADED_SHARDS`, so sharding is required for memory stability.

### Stage 4: Shadow Runtime

Point only the research sidecar at v2:

```text
AGENT_KERNEL_PAPER_RESEARCH_CONFIG_PATH=/data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml
AGENT_KERNEL_PAPER_RESEARCH_CHECKPOINT_PATH=/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/tolbert_epoch5.pt
AGENT_KERNEL_PAPER_RESEARCH_NODES_PATH=/data/TOLBERT_BRAIN/data/joint_v2/nodes_joint_v2.jsonl
AGENT_KERNEL_PAPER_RESEARCH_LABEL_MAP_PATH=/data/TOLBERT_BRAIN/data/joint_v2/label_map_joint_v2.json
AGENT_KERNEL_PAPER_RESEARCH_SOURCE_SPANS_PATHS=/data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl:/data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl
AGENT_KERNEL_PAPER_RESEARCH_CACHE_PATHS=/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/retrieval_cache/paper_spans_paragraphs_joint_v2__tolbert_epoch5.json:/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/retrieval_cache/code_spans_joint_v2__tolbert_epoch5.json
AGENT_KERNEL_PAPER_RESEARCH_QUERY_MODE=always
```

Run in shadow mode first. Log candidates but do not inject them into decisive planner/editor context until latency and quality are verified.

### Stage 5: Promotion

Promote v2 only if it passes:

- service starts reliably;
- query latency stays under budget;
- cache memory stays bounded;
- retrieved items have valid provenance;
- no benchmark contamination;
- retrieval helps on held-out benchmark-like tasks;
- kernel baseline behavior does not regress.

## Continuous Learning Loop

Continuous learning should have two loops: fast index learning and slow model learning.

### Fast Loop: Index Growth

Runs frequently, possibly after new corpora arrive:

1. Detect new papers, repos, algorithms, or repository-library exports.
2. Extract spans and metadata.
3. Build or update lexical/vector/graph indexes.
4. Build a new immutable generation.
5. Run smoke retrieval tests.
6. Promote only the generation pointer if safe.

This loop does not modify model weights.

### Slow Loop: Model And Ranker Learning

Runs after enough supervised signal accumulates:

1. Collect retrieval impressions from benchmark and coding runs:
   - query text;
   - candidates generated;
   - candidates injected into context;
   - selected snippets;
   - patch/test outcome;
   - benchmark resolution outcome.

2. Convert logs into labels:
   - positive: retrieved source was cited, selected, used in a patch, or preceded a passing verifier result;
   - weak positive: source was injected in a successful run;
   - hard negative: high-ranked source was ignored in a successful run;
   - negative: source was selected before a failed or reverted attempt.

3. Train small components first:
   - reranker;
   - query router;
   - source-type selector;
   - paper/repo alignment scorer.

4. Train TOLBERT updates only after ranker learning proves value:
   - v2 research TOLBERT;
   - benchmark-family routing calibration;
   - repository-code/paper span alignment.

5. Promote through a retention gate.

### Learning Artifacts

Store learning outputs as artifacts:

```text
trajectories/research_library/
  retrieval_impressions.jsonl
  retrieval_outcomes.jsonl
  training_sets/
  candidate_artifacts/
  retained/
```

Each retained artifact must include:

- parent generation id;
- source manifest hashes;
- training data manifest;
- model/checkpoint paths;
- eval report;
- promotion gate report;
- rollback pointer.

## A8 Benchmark Usage

The retrieval layer should be tuned to A8 benchmark needs.

### SWE-Bench / SWE-ReBench

Use:

- repository graph library for source localization;
- repo QA indexes for package-specific conventions;
- exact symbol/error lexical retrieval;
- prior successful kernel episode retrieval;
- public upstream docs and issue-like patterns when available.

Avoid:

- generated predictions;
- known target patches;
- evaluator answers;
- hidden tests.

### RE-Bench

Use:

- systems and ML systems papers;
- repository graph examples;
- optimization and experiment infrastructure patterns;
- algorithms library for core methods.

The verifier should receive compact claims and constraints, not long papers.

### MLE-Bench

Use:

- paper universe for model families, metrics, loss functions, feature engineering, and ablation patterns;
- repo graph for training loops and evaluation code;
- algorithms for optimization and data-processing constraints.

Kaggle raw data remains credential-gated. The research library can still provide methods and code patterns.

### Codeforces / CodeContests

Use:

- `/data/algorithms/algorithms.jsonl`;
- `/data/algorithms/problems.jsonl`;
- `/data/algorithms_library/Python`;
- complexity and invalid-use constraints;
- generated tests based on problem constraints.

This can wait for account integration, but the local adapter should be ready.

## Contamination Controls

Every benchmark run should write a retrieval audit record:

```json
{
  "run_id": "string",
  "task_id": "string",
  "benchmark_family": "swe_bench|re_bench|mle_bench|codeforces",
  "generation_id": "string",
  "query": "string",
  "candidates": [
    {
      "span_id": "string",
      "source_kind": "paper|repo|algorithm|kernel_memory",
      "rank": 1,
      "score": 0.0,
      "injected": true,
      "selected": false,
      "contamination_class": "public_ok|blocked|unknown"
    }
  ]
}
```

Block sources with:

- benchmark solution patches;
- generated predictions for the same task;
- evaluator logs containing expected final answers;
- hidden tests;
- private benchmark answer keys.

Allow sources with:

- public repository code;
- public papers;
- public docs;
- local algorithm library;
- task-visible problem statements;
- prior kernel process knowledge that does not include answer leakage.

## Promotion Gates

No new generation or model should become active unless it passes:

- source integrity: all manifest hashes valid;
- availability: required files exist and are readable;
- latency: p50 and p95 query latency within configured budgets;
- memory: cache shard loading stays below process/GPU budget;
- retrieval quality: improves or matches baseline on held-out queries;
- benchmark safety: contamination audit passes;
- kernel safety: no regression in existing tests and smoke runs;
- rollback: previous active generation remains available.

## Implementation Plan

### PR 1: Source And Generation Registry

Files:

- `config/research_library_sources.json`
- `agent_kernel/research_library/sources.py`
- `agent_kernel/research_library/generations.py`
- `scripts/build_research_library_status.py`
- `tests/test_research_library_sources.py`

Purpose:

- validate paths;
- count assets;
- fingerprint manifests;
- produce `var/research_library/status.json`;
- create generation manifests.

### PR 2: Read-Only Retrieval Clients

Files:

- `agent_kernel/research_library/repository_client.py`
- `agent_kernel/research_library/paper_client.py`
- `agent_kernel/research_library/algorithm_client.py`
- `scripts/query_research_library.py`

Purpose:

- query repository graph exports;
- query paper universe metadata and embeddings;
- query algorithms and implementations;
- output normalized retrieved-span records.

### PR 3: Ensemble Query Client

Files:

- `agent_kernel/research_library/query_client.py`
- `agent_kernel/research_library/rerank.py`
- tests for the `TolbertQueryClient` response contract.

Purpose:

- implement the existing `TolbertQueryClient` protocol;
- return retrieval buckets compatible with `TolbertContextCompiler`;
- support shadow mode.

### PR 4: Kernel Hook

Minimal edit:

- update `agent_kernel/extensions/tolbert.py` so `_ensure_research_client` can instantiate `ResearchLibraryQueryClient` when enabled.

Purpose:

- preserve all current context budgeting and ranking behavior;
- add `research_library` as an auxiliary backend, not a control-plane replacement.

### PR 5: TOLBERT v2 Sync

Files/scripts:

- training config validation for `/data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml`;
- cache build orchestration for v2 code and paper spans;
- status checks for v2 checkpoint/cache presence;
- optional partial warm-start script if full training is too slow.

Purpose:

- make the 3.9M paragraph-span corpus usable by the sidecar runtime.

### PR 6: Continuous Learning

Files:

- `agent_kernel/research_library/logging.py`
- `agent_kernel/research_library/labels.py`
- `scripts/build_research_retrieval_training_set.py`
- `scripts/train_research_reranker.py`
- promotion gate tests.

Purpose:

- turn benchmark runs into retrieval labels;
- train small rankers first;
- promote only gated artifacts.

## Do Not Do

- Do not overwrite `/data/agentkernel/var/tolbert/agentkernel` with research TOLBERT assets.
- Do not point the primary Agent Kernel TOLBERT runtime at `/data/TOLBERT_BRAIN` directly.
- Do not load all 3.9M paper spans into the kernel process.
- Do not inject long paper text into every task.
- Do not retrain weights on every new paper/repo arrival.
- Do not promote a retriever without contamination logs.
- Do not use benchmark predictions or solution artifacts as retrieval sources.

## Desired End State

The kernel has a continuously growing knowledge base with stable runtime behavior:

- new papers/repos are indexed into immutable generations;
- the active generation can be queried through a single research client;
- retrieval spans are provenance-rich and budgeted;
- A8 benchmark runs log every retrieval influence;
- rankers improve from benchmark outcomes;
- TOLBERT research models are trained and promoted safely;
- rollback is always possible;
- the core kernel remains stable while retrieval capacity keeps scaling.
