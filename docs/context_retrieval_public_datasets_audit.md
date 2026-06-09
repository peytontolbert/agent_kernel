# Context Retrieval Public Dataset Audit

Status: audit and integration plan  
Date: 2026-04-28

This is focused on whether Agent Kernel can use the public paper and code
datasets to answer the user, both in the Python kernel and in AgentKernel Lite.
It does not change the adapter contract: the research-library adapter remains
detachable and off by default.

## Current Coverage

The local source snapshot is strong. `var/research_library/status.json` reports:

- `1,000,000` full paper rows from `/arxiv/huggingface/paper_text_1m_dedup_v1`
- `17,117,443` paper chunk examples from `/data/tmp/p1_full_paper_lm_hf_all_chunks`
- `1,000,000` paper-universe rows, `20,000,062` paper KNN edges, and `3,000,000` paper-topic edges in `/data/repository_library/exports/_paper_universe`
- `178` repository exports and `74,472` mined repository skills in `/data/repository_library/exports`
- `1,942` algorithm catalog rows and `1,373` Python implementation files
- `35` trained model assets across repository models, digital-world-model assets, and TOLBERT assets

The public browser-facing paper path is also present:

- paper pack root: `https://huggingface.co/datasets/PeytonT/paper_universe_interactive/resolve/main/interactive`
- full text dataset id: `PeytonT/1m_papers_text`
- dataset server: `https://datasets-server.huggingface.co`
- arXiv PDF links are constructed from retrieved paper ids

## Current Retrieval Path

Python kernel:

- normal TOLBERT context remains the primary compiler when enabled;
- auxiliary `paper_research` TOLBERT is queried for research-like prompts;
- standalone research-library context is off by default and only attaches when `AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT=1`;
- code retrieval already uses repository manifest ranking, mined repo-skill snippets, and entity-to-source spans;
- algorithm retrieval uses the small JSONL algorithm catalog;
- paper retrieval in the standalone adapter currently uses a bounded parquet scan over `paper_chunks_p1`.

Browser AgentKernel Lite:

- can search local paper packs or Hugging Face dataset rows;
- can enrich paper metadata rows with full-text rows by offset or id;
- can open arXiv PDFs;
- does not yet expose repo graph/code retrieval datasets.

## Audit Result

I added a repeatable read-only audit:

```bash
python scripts/audit_context_retrieval_public_datasets.py --run-probes \
  --output var/research_library/context_retrieval_public_dataset_audit.json
```

Current probe behavior:

- `repo_code_transformers` retrieves the concrete `transformers` implementation for `flash_attn_supports_top_left_mask`, plus related attention snippets.
- `paper_attention_kernel` gets paper hits, but they are false-positive lexical matches because the standalone adapter scans only a tiny prefix of the 17M chunk dataset.
- `algorithm_shortest_path` gets algorithm evidence, but the ranking prefers BFS-style shortest path entries even when the prompt says non-negative weighted graph, so the algorithm ranker needs constraint-aware scoring.

The important conclusion: code retrieval is already useful server-side; paper retrieval has the data but lacks the right index path in the standalone adapter; browser retrieval is paper-only.

## Active TOLBERT State

The newer joint-v2 projection exists:

- `/data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/nodes_joint_v2.jsonl`
- `/data/TOLBERT_BRAIN/data/joint_v2/label_map_joint_v2.json`

Observed line counts:

- old joint paper spans: `234,986`
- joint-v2 paper paragraph spans: `3,939,404`
- old and v2 code spans: `101,118`
- joint-v2 data directory size: `6.5G`

But the active `paper_research` runtime is still the older joint bundle:

- checkpoint: `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/tolbert_epoch3.pt`
- cache: `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/retrieval_cache/paper_spans_joint_mapped__tolbert_epoch3.pt`

The promoted v2 runtime directory is missing:

- `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2`

So the kernel can use paper research TOLBERT today, but not the newer full joint-v2 projection as an active runtime.

## Gaps To Fix

1. Standalone paper retrieval should stop scanning only `4096` rows from one shard by default. That is about `0.024%` of `paper_chunks_p1` and explains false positives.
2. The paper adapter should generate candidates from `paper_universe` metadata/full-text embeddings, then fetch snippets from `paper_chunks_p1` or `paper_text_1m` by id/offset.
3. Repository code retrieval should become a public dataset adapter for the browser, not only a server-side local export reader.
4. Algorithm retrieval should score constraint compatibility, not only token overlap. For example, `non-negative weighted` should push Dijkstra above BFS.
5. Probe-specific applied facts and exact-command generation should be replaced by a general evidence-normalization layer before broad benchmark use.
6. Context budgeting boosts `research:applied_guidance` and `research:paper_hits`, but there is no explicit diversity floor across paper/code/algorithm evidence.

## Integration Shape

Add a unified evidence schema used by Python, Rust/WASM, browser JS, and traces:

```json
{
  "evidence_id": "stable id",
  "source_kind": "paper|repo_code|repo_qa|algorithm|artifact|trace",
  "source_id": "arxiv id, repo id, algorithm id, or artifact id",
  "title": "human readable title",
  "uri": "public or local source uri",
  "locator": "page, paragraph, line range, symbol, or row offset",
  "snippet": "bounded source text",
  "score": 0.0,
  "provenance": {
    "dataset": "PeytonT/1m_papers_text|PeytonT/repo_graph|local",
    "retriever": "paper_universe|repo_graph|algorithm_catalog|tolbert",
    "generation_id": "active generation id",
    "contamination_class": "public_ok|blocked|unknown"
  }
}
```

Then implement one query client that returns normal `ContextPacket.retrieval`
spans:

- paper candidates: paper universe embedding/metadata search, KNN/topic expansion, snippet fetch from full text/chunks, arXiv PDF link;
- code candidates: repo manifest, repo-skill snippets, entity spans, QA indexes, public repo graph dataset shards;
- algorithm candidates: aliases, constraints, complexity, reference implementation path;
- trace/artifact candidates: user-approved local memory and previous successful run evidence.

## Recommended Next Work

1. Build `ResearchLibraryQueryClient` as a retrieval backend, not a prompt mutator.
2. Add paper-universe candidate generation and full-text snippet fetch.
3. Add repo graph/code retrieval to the browser using Hugging Face dataset shards plus IndexedDB/OPFS cache.
4. Add a source-kind diversity floor in context selection so a user query can receive one strong paper, code, and algorithm candidate when each is relevant.
5. Build/promote joint-v2 TOLBERT checkpoint and sharded caches in shadow mode before enabling active runtime use.
6. Log every retrieved, injected, selected, and ignored evidence record into run artifacts for benchmark auditability.

