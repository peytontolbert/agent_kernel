# AgentKernel Harness Skill Retrieval

AgentKernel consumes repo-skills-miner output as retrieval-time operators, not
as direct patch generators. The skill retriever learns to map a task, failure
trace, or patch situation to compact harness skills that can guide a later diff
planner and verifier.

The repo-skills-miner side stays repository-generic. Patch-specific supervision
belongs in AgentKernel and starts with:

```bash
python scripts/build_harness_skill_retrieval_dataset.py \
  --skills-parquet /data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet \
  --output-dir artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset
```

The builder writes compressed Parquet shards and a manifest:

```text
artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset/
  agentkernel_harness_skill_retrieval_dataset_manifest.json
  train/part-00000.parquet
  eval/part-00000.parquet
```

Each training row includes:

- `retrieval_query_text`: task-like query built from summary, use-when,
  patch-relevance, verification hints, path, repo, and labels.
- `retrieval_doc_text`: skill operator document with provenance, annotation
  fields, risk/permission hints, and source excerpt.
- `retrieval_negative_doc_texts`: JSON-encoded hard negatives selected from
  nearby labels, language, primitive type, and skill kind.
- AgentKernel action targets that route the learned behavior through
  `gather_context` and `<AK_RET_SKILLS>`.

Train the first retrieval head with:

```bash
scripts/train_agentkernel_lite_harness_skill_retrieval.sh
```

The wrapper uses the existing `train_agentkernel_lite_encdec.py` retrieval
losses and keeps decoder loss disabled by default. For a retained checkpoint,
pass the manifest, output directory, checkpoint, and tokenizer directory:

```bash
scripts/train_agentkernel_lite_harness_skill_retrieval.sh \
  artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset/agentkernel_harness_skill_retrieval_dataset_manifest.json \
  artifacts/agentkernel_lite_encdec/harness_skill_retriever_r1 \
  artifacts/agentkernel_lite_encdec/<prior>/checkpoints/step_00002000.pt \
  artifacts/agentkernel_lite_encdec/<prior>/tokenizer
```

This stage optimizes skill retrieval quality. Patch generation should consume
the retrieved operators, anti-patterns, and verification hints as planning
context, then save trace outcomes for later credit assignment.

## First OpenClaw/Hermes Run

The first full dataset was built from the repo-skills-miner OpenClaw/Hermes
Parquet export:

- examples: 93,039
- train examples: 88,406
- eval examples: 4,633
- negatives per example: 8 in the dataset, 4 used in the first local run
- train shards: 2 compressed Parquet files
- eval shards: 1 compressed Parquet file

The first local training run fine-tuned from:

```text
artifacts/agentkernel_lite_encdec/encoder_retrieval_ternary_aware_from_1mabs_train_01000/checkpoints/step_00001000.pt
```

Output:

```text
artifacts/agentkernel_lite_encdec/harness_skill_retriever_r1/
```

Held-out retrieval evaluation on 1,024 pairs:

| Bundle | Top-1 | MRR |
| --- | ---: | ---: |
| Base ternary-aware retrieval checkpoint | 0.8506 | 0.8893 |
| Harness-skill retriever r1 | 0.9932 | 0.9964 |

Evaluation files:

```text
artifacts/agentkernel_lite_encdec/harness_skill_retriever_r1/base_retrieval_eval_1024.json
artifacts/agentkernel_lite_encdec/harness_skill_retriever_r1/retrieval_eval_1024.json
```
