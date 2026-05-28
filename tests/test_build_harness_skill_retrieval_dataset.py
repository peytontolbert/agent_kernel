from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_harness_skill_retrieval_dataset.py"
    spec = importlib.util.spec_from_file_location("build_harness_skill_retrieval_dataset", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_harness_skill_retrieval_dataset_writes_parquet_manifest(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    module = _load_module()

    source = tmp_path / "skills.parquet"
    rows = [
        {
            "id": "skill-runtime-timeout",
            "dataset_repo_label": "openclaw",
            "source_repo": "openclaw/openclaw",
            "source_path": "src/runtime.ts",
            "language": "TypeScript",
            "primitive_type": "code_symbol",
            "primitive_subtype": "function",
            "skill_kind": "function",
            "qualname": "runShell",
            "side_effects": ["shell"],
            "required_permissions": ["run_tests"],
            "source_excerpt": "export async function runShell(cmd) { return exec(cmd) }",
            "annotation_summary": "Runs a shell command for the agent runtime.",
            "annotation_primitive_labels": ["runtime.command_execution", "behavior.timeout_handling"],
            "annotation_use_when": ["Use when a patch needs command execution boundaries."],
            "annotation_patch_relevance": ["Add timeout handling around command execution."],
            "annotation_risks": ["Unbounded shell execution can hang."],
            "annotation_verification_hints": ["Run a unit test that simulates a timeout."],
            "annotation_confidence": 0.9,
        },
        {
            "id": "skill-browser-timeout",
            "dataset_repo_label": "openclaw",
            "source_repo": "openclaw/openclaw",
            "source_path": "src/browser.ts",
            "language": "TypeScript",
            "primitive_type": "code_symbol",
            "primitive_subtype": "method",
            "skill_kind": "method",
            "qualname": "waitForLoad",
            "source_excerpt": "async function waitForLoad(page) { await page.waitForLoadState() }",
            "annotation_summary": "Waits for browser page readiness.",
            "annotation_primitive_labels": ["browser.runtime", "behavior.timeout_handling"],
            "annotation_use_when": ["Use for browser automation waits."],
            "annotation_patch_relevance": ["Bound browser page waits."],
            "annotation_confidence": 0.8,
        },
        {
            "id": "skill-memory",
            "dataset_repo_label": "hermes-agent",
            "source_repo": "NousResearch/hermes-agent",
            "source_path": "memory.py",
            "language": "Python",
            "primitive_type": "code_symbol",
            "primitive_subtype": "class",
            "skill_kind": "class",
            "qualname": "MemoryStore",
            "source_excerpt": "class MemoryStore: pass",
            "annotation_summary": "Stores reusable agent memory.",
            "annotation_primitive_labels": ["memory.state"],
            "annotation_use_when": ["Use when persisting agent experience."],
            "annotation_patch_relevance": ["Add durable state around remembered facts."],
            "annotation_confidence": 0.7,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), source, compression="zstd")

    manifest = module.build(
        type(
            "Args",
            (),
            {
                "skills_parquet": str(source),
                "output_dir": str(tmp_path / "out"),
                "max_rows": 0,
                "negative_count": 2,
                "max_excerpt_chars": 200,
                "shard_size": 2,
                "eval_fraction": 0.34,
                "seed": 1,
            },
        )()
    )

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["dataset_format"] == "parquet"
    assert loaded["retrieval_pair_count"] == 3
    assert Path(loaded["train_dataset_path"]).is_dir()
    assert Path(loaded["eval_dataset_path"]).is_dir()

    train_rows = []
    for shard in Path(loaded["train_dataset_path"]).glob("*.parquet"):
        train_rows.extend(pq.read_table(shard).to_pylist())
    eval_rows = []
    for shard in Path(loaded["eval_dataset_path"]).glob("*.parquet"):
        eval_rows.extend(pq.read_table(shard).to_pylist())
    assert len(train_rows) + len(eval_rows) == 3
    sample = (train_rows + eval_rows)[0]
    assert sample["task_type"] == "harness_skill_retrieval"
    assert sample["retrieval_query_text"]
    assert sample["retrieval_doc_text"]
    assert len(json.loads(sample["retrieval_negative_doc_texts"])) == 2
