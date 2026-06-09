# Full-Corpus TOLBERT Sync Plan

Status: design correction and implementation plan  
Date: 2026-04-27

## Core Correction

The current TOLBERT paper spans are not the full 1M-paper corpus. They are a derived training/runtime projection. The full 1M-paper corpus already exists locally as compressed parquet and should be treated as the source of truth.

The right architecture is:

1. Source of truth: full paper text parquet and repository library exports.
2. Derived projections: paper chunks, code spans, algorithm spans, ontology paths.
3. Learned models: TOLBERT, rerankers, embedding adapters.
4. Runtime indexes: dense shards, lexical indexes, graph neighborhoods, and provenance pointers.

TOLBERT can be trained over all 1M paper texts. Runtime retrieval should still fetch only the exact top-k text chunks needed for a task. That is not a limitation; it is the serving pattern that lets training scale and keeps the kernel stable.

## Verified Local Assets

### Full 1M Paper Text Source

Path:

- `/arxiv/huggingface/paper_text_1m_dedup_v1`

Observed:

- size: `21G`
- files: `train_00000.parquet` through `train_00009.parquet`
- rows: exactly `1,000,000`
- unique canonical papers: exactly `1,000,000`
- base rows considered: `124,036`
- backfill rows considered: `875,968`
- merged rows before cap: `1,000,004`

Schema includes:

- `paper_id`
- `canonical_paper_id`
- `paper_version`
- `pdf_path`
- `title`
- `abstract`
- `authors`
- `categories`
- `license`
- `update_date`
- `text`
- `text_source`
- `text_is_partial`
- `text_char_count`
- `text_line_count`
- `token_count`
- `page_count`
- `token_type_counts_json`

This is the correct paper source for full-corpus TOLBERT training and retrieval.

### Derived P1 Full-Paper LM Chunks

Path:

- `/data/tmp/p1_full_paper_lm_hf_all_chunks`

Observed:

- size: `35G`
- parquet shards: `172`
- examples: `17,117,443`
- seen papers: `1,000,000`
- kept papers: `965,038`
- chunk size: `3000` chars
- chunk overlap: `300` chars

Schema:

- `text`: current paper chunk
- `target`: following paper chunk
- `paper_id`
- `title`
- `categories`
- `year`
- `offset`
- `chunk_index`

This is a strong training source for next-chunk, adjacent-chunk contrastive learning, and retrieval embedding training. It is not the source of truth; it is a derived chunk training set.

### Paper Universe

Path:

- `/data/repository_library/exports/_paper_universe`

Observed:

- `paper_nodes.parquet`: `1,000,000` rows
- `paper_embeddings.parquet`: `1,000,000` rows
- `paper_fulltext_embeddings.parquet`: `1,000,000` rows
- `paper_knn_edges.parquet`: `20,000,062` rows
- `topic_nodes.parquet`: `1,483,957` rows
- `paper_topic_edges.parquet`: `3,000,000` rows

This already covers the full 1M paper set at the graph and embedding level. It does not duplicate full paper text; it stores metadata, embeddings, KNN edges, topics, categories, years, and coordinates. This should be the immediate retrieval backbone while TOLBERT full-corpus training is built.

### Current TOLBERT Paper/Code Spans

Current v2 paths:

- `/data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/nodes_joint_v2.jsonl`

Observed:

- code spans: `101,118`
- paper paragraph spans: `3,939,404`
- nodes: `336,480`

These are useful but not complete. They do not represent the full recent 1M-paper text dataset. They should be considered a v2 projection that needs to be rebuilt from the full source corpus.

## Training Versus Serving

Training on all 1M papers means TOLBERT sees the full text corpus during optimization. It does not mean the kernel loads all full text into memory during every task.

Serving should work like this:

1. The task query is embedded and routed.
2. TOLBERT, paper universe embeddings, lexical search, and repo graph search generate candidates.
3. The top candidates contain stable pointers to paper/chunk/repo locations.
4. The runtime fetches only the selected text snippets from parquet/source files.
5. The kernel injects bounded snippets into context.

This is the only practical way to combine full-corpus training with stable benchmark execution.

## Why Not One Class Per Paper Or Chunk

TOLBERT has hierarchical classification heads:

```python
self.level_heads[str(level)] = nn.Linear(hidden_dim, size)
```

A one-million-class paper head would add roughly `hidden_dim * 1,000,000` parameters for a single level. With a 768-dimensional encoder, that is about 768M weights before bias and optimizer state. A 17M-chunk head is much worse.

We should still train on every paper/chunk, but exact paper/chunk identity should be represented by embeddings and metadata pointers, not a giant softmax class.

Recommended classification levels:

- L1: modality, such as `Papers`, `Code`, `Algorithms`, `KernelMemory`
- L2: broad domain, such as arXiv area or software pillar
- L3: category, such as `cs.LG`, `cs.SE`, repo family, or benchmark family
- L4: topic or cluster id from paper universe / repo universe
- L5: bounded source bucket or artifact family, not every paper/chunk id

Exact identifiers remain metadata:

- `paper_id`
- `canonical_paper_id`
- `parquet_shard`
- `row_group`
- `row_index`
- `offset`
- `chunk_index`
- `repo_id`
- `symbol_id`
- `source_path`

## Full-Corpus TOLBERT Training Design

### Dataset Path

Do not convert the 17M chunk dataset into one giant JSONL and feed it to the current `TreeOfLifeDataset`.

The current TOLBERT dataset class loads all JSONL records into memory:

```python
self._records: List[SpanRecord] = []
...
self._records.append(SpanRecord(...))
```

That is fine for small span sets and not acceptable for 1M papers or 17M chunks.

Add one of these:

- `ParquetTreeOfLifeIterableDataset`: streams rows from parquet shards and tokenizes on the fly.
- `ShardedTreeOfLifeDataset`: reads shard manifests and memory-maps or page-loads shard subsets.
- `ResearchSpanDataPipe`: unified iterator over paper chunks, code spans, algorithm spans, and retained kernel spans.

The first implementation should be parquet streaming because the full paper sources are already parquet.

### Paper Training Rows

Use two complementary row sources:

1. `/arxiv/huggingface/paper_text_1m_dedup_v1`
   - one row per paper;
   - best for metadata, whole-paper sampling, title/abstract/category objectives, and provenance.

2. `/data/tmp/p1_full_paper_lm_hf_all_chunks`
   - one row per adjacent chunk pair;
   - best for chunk-level MLM, adjacent-chunk contrastive learning, and retrieval cache building.

### Objectives

Use TOLBERT for encoder/retrieval learning:

- MLM over paper chunks and code chunks.
- Hierarchical route prediction over bounded levels.
- Same-paper contrastive positives.
- Adjacent-chunk contrastive positives.
- Same-topic and same-category weak positives.
- Paper-universe KNN positives.
- Paper-to-repo alignment positives from `paper_repo_align.jsonl` and `paper_repo_span_align.jsonl`.
- Hard negatives from different categories, nearby but wrong KNN nodes, and failed benchmark retrieval logs.

The P1 next-chunk `text -> target` objective is useful, but it is not currently native to the TOLBERT encoder. Use it in one of two ways:

- train/keep a separate P1 seq2seq model for next-chunk modeling;
- convert adjacent chunks into a contrastive encoder objective for TOLBERT.

### Training Phases

Phase A: metadata and route warmup

- Source: 1M paper rows.
- Text: title + abstract + compact metadata card.
- Targets: modality, arXiv area, primary category, year bucket, topic clusters.
- Purpose: stabilize hierarchy heads and broad routing.

Phase B: full-text chunk training

- Source: P1 chunk parquet.
- Text: chunk text.
- Targets: same hierarchy as metadata plus topic/cluster/bucket labels.
- Objectives: MLM, adjacent-chunk contrastive, same-paper contrastive.
- Purpose: make TOLBERT encode real full-text content.

Phase C: code + paper joint training

- Source: repository code spans plus paper chunks.
- Add positives from paper-to-repo alignment and benchmark traces.
- Purpose: make papers useful for code repair, systems tasks, MLE methods, and algorithmic work.

Phase D: benchmark-aware calibration

- Source: A8 benchmark runs and retrieval logs.
- Train lightweight rerankers first.
- Fine-tune TOLBERT only after reranker labels prove value.

## Full-Corpus Runtime Indexes

Training produces a better encoder. Serving still needs indexes.

Recommended runtime indexes:

- paper metadata vector index over `paper_embeddings.parquet`;
- paper full-text vector index over `paper_fulltext_embeddings.parquet`;
- chunk-level vector shards from P1 chunks;
- lexical/BM25 index over titles, abstracts, full-text chunks, symbols, errors;
- paper KNN graph from `paper_knn_edges.parquet`;
- topic graph from `topic_nodes.parquet` and `paper_topic_edges.parquet`;
- repo graph exports from `/data/repository_library/exports`;
- code span vector shards from repository exports;
- algorithm in-memory/vector index from `/data/algorithms` and `/data/algorithms_library`.

For each chunk embedding, store:

```json
{
  "source_kind": "paper_chunk",
  "paper_id": "1001.0115",
  "canonical_paper_id": "1001.0115",
  "dataset": "/data/tmp/p1_full_paper_lm_hf_all_chunks",
  "parquet_shard": "train-00000.parquet",
  "row_group": 0,
  "row_index": 123,
  "offset": 9000,
  "chunk_index": 3,
  "categories": "cs.MA",
  "year": 2010
}
```

The runtime can then fetch the exact `text` only when the chunk survives top-k retrieval/reranking.

## Incremental Code Span Sync

Code spans also need to grow continuously. The source of truth for code should be the repository library, not a static TOLBERT JSONL.

Existing repo export behavior is already incremental:

- entry point: `/data/repository_library/scripts/library_repo_graph_export.py`
- root: `/data/repositories`
- manifest: `/data/repository_library/exports/_manifest.json`
- state: git HEAD when available, otherwise file mtime snapshot
- if schema and repo state match, the exporter skips the repo as up to date

Code sync flow:

1. Add or update repo under `/data/repositories` or an extra library root.
2. Run repository graph export.
3. Detect changed repos from the manifest.
4. Project changed repo exports into code spans.
5. Build/update lexical and vector indexes for changed spans.
6. Add changed spans to a new immutable research generation.
7. Train or fine-tune TOLBERT later when enough new code data accumulates.

The code span projector should prefer repository exports over raw file walking:

- entities: functions, classes, files, modules;
- edges: ownership, references, imports;
- artifacts: source file hashes;
- QA indexes where present;
- repo skills where present.

Span granularity should be:

- function/class spans where available;
- file chunks for files without parsed symbols;
- docstring/readme/test chunks;
- config/build/test workflow chunks.

Avoid whole-file spans as the only representation. Whole-file spans are useful fallback context, but symbol-level and test-level spans are better for SWE-Bench-style localization.

## Unified Paper + Code Ontology

The ontology must support growth. Do not let exact corpus cardinality force model-head cardinality.

Recommended high-level tree:

```text
SoftwareEngineeringAI
  Papers
    arxiv_area
      primary_category
        topic_cluster
          source_bucket
  Code
    repo_family_or_language
      repo_id_or_repo_cluster
        artifact_kind
          symbol_or_path_bucket
  Algorithms
    topic
      problem_family
        complexity_class
          implementation_family
  KernelMemory
    benchmark_family
      task_family
        artifact_class
          retained_skill_or_episode_bucket
```

Exact paper ids, repo ids, chunk ids, and symbol ids remain in metadata and retrieval indexes. They do not all need to be classification classes.

## Continuous Learning Update Loop

### Fast Loop: Index Growth

Run this whenever data changes:

1. Scan source manifests:
   - paper dataset stats;
   - paper universe manifest;
   - repo export manifest;
   - algorithm catalog hashes.
2. Build changed projections:
   - new paper chunks;
   - new code spans;
   - new algorithm spans.
3. Embed changed spans.
4. Update retrieval shards.
5. Create a new immutable generation.
6. Run retrieval smoke tests.
7. Promote generation pointer if gates pass.

This loop should not retrain TOLBERT weights every time.

### Slow Loop: Model Learning

Run after enough signal accumulates:

1. Aggregate benchmark/coding retrieval logs.
2. Build labels:
   - positive: source cited or used before success;
   - weak positive: injected into a successful run;
   - hard negative: high-ranked but unused;
   - negative: selected before failed/reverted attempt.
3. Train rerankers and routers first.
4. Periodically train TOLBERT on:
   - all paper chunks;
   - changed code spans;
   - retained benchmark labels;
   - hard negatives.
5. Promote only through a gate.

## A8 Benchmark Integration

Full-corpus retrieval helps A8 only if it is targeted.

SWE-Bench / SWE-ReBench:

- prioritize repo graph, symbol spans, tests, docs, exact errors;
- use papers only for library design, algorithms, numerical methods, or framework behavior when relevant.

RE-Bench:

- prioritize systems papers, ML systems papers, optimization examples, repo implementation patterns.

MLE-Bench:

- prioritize methods, metrics, feature engineering, training loops, ablation patterns.

Codeforces / CodeContests:

- prioritize algorithm catalog and implementation library;
- use papers only for advanced algorithms or proofs.

All benchmark runs must log retrieval provenance and contamination class.

## Immediate Implementation Order

1. Add source registry entries for:
   - `/arxiv/huggingface/paper_text_1m_dedup_v1`
   - `/data/tmp/p1_full_paper_lm_hf_all_chunks`
   - `/data/repository_library/exports/_paper_universe`
   - `/data/repository_library/exports`

2. Implement a paper parquet fetcher:
   - fetch by `paper_id`;
   - fetch by parquet shard/row;
   - fetch by chunk pointer;
   - return bounded snippets with provenance.

3. Implement paper universe retrieval first:
   - metadata embedding search;
   - full-text embedding search;
   - KNN/topic expansion;
   - no TOLBERT retraining required yet.

4. Implement code export-to-span projection:
   - changed repo detection from manifest;
   - function/class/file/doc/test spans;
   - source hashes and provenance.

5. Implement full-corpus TOLBERT streaming dataset:
   - direct parquet iterator;
   - bounded hierarchy labels;
   - contrastive pair sampler.

6. Train research TOLBERT sidecar:
   - do not replace primary Agent Kernel TOLBERT;
   - build sharded retrieval caches;
   - run in shadow mode.

7. Promote only after:
   - retrieval quality improves;
   - latency and memory are stable;
   - contamination logs pass;
   - benchmark smoke runs do not regress.

## Concrete New Components

Recommended files inside `agent_kernel`:

- `agent_kernel/research_library/paper_parquet.py`
- `agent_kernel/research_library/paper_universe.py`
- `agent_kernel/research_library/repo_spans.py`
- `agent_kernel/research_library/ontology.py`
- `agent_kernel/research_library/query_client.py`
- `agent_kernel/research_library/retrieval_log.py`

Recommended scripts:

- `scripts/build_research_library_status.py`
- `scripts/query_research_library.py`
- `scripts/build_paper_chunk_pointer_index.py`
- `scripts/project_repo_exports_to_research_spans.py`
- `scripts/build_research_generation.py`
- `scripts/train_research_tolbert_from_parquet.py`
- `scripts/build_research_tolbert_parquet_cache.py`

Some of these may later move into `/data/TOLBERT_BRAIN`, but the first implementation should live in Agent Kernel as an adapter layer so the benchmark kernel remains stable.

## Non-Negotiable Safety Rules

- Do not overwrite the primary Agent Kernel TOLBERT runtime.
- Do not require all 1M papers or 17M chunks to load into memory.
- Do not create a 17M-class TOLBERT head.
- Do not promote a new retrieval generation without provenance logs.
- Do not use benchmark solution artifacts, predictions, hidden tests, or evaluator answers as retrieval sources.
- Do not train directly from generated benchmark patches without marking them as outcome labels and contamination-sensitive.

## Bottom Line

Yes, TOLBERT should be trained over the full 1M paper corpus. The current TOLBERT span set is behind the data we have. The fix is not to manually point the existing runtime at the 1M parquet files. The fix is to build a full-corpus streaming training and retrieval pipeline:

- 1M paper parquet as source of truth;
- P1 chunks as full-text training examples;
- paper universe as immediate 1M-scale graph/vector retrieval;
- repo graph exports as source of truth for growing code spans;
- TOLBERT as the learned cross-domain router and embedding model;
- sharded runtime indexes with exact source pointers;
- benchmark retrieval logs feeding continuous learning.

