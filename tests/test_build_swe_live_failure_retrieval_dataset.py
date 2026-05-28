from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_swe_live_failure_retrieval_dataset.py"
    spec = importlib.util.spec_from_file_location("build_swe_live_failure_retrieval_dataset", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_swe_live_failure_retrieval_dataset_writes_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    module = _load_module()

    manifest = module.build(
        type(
            "Args",
            (),
            {
                "source_run": "benchmarks/swe_bench_live/autonomous_harness_runs/r42",
                "output_dir": str(tmp_path / "out"),
                "repeat": 3,
            },
        )()
    )

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["dataset_format"] == "parquet"
    assert loaded["operator_count"] == 5
    assert loaded["retrieval_pair_count"] == 15

    train_rows = []
    for shard in Path(loaded["train_dataset_path"]).glob("*.parquet"):
        train_rows.extend(pq.read_table(shard).to_pylist())
    eval_rows = []
    for shard in Path(loaded["eval_dataset_path"]).glob("*.parquet"):
        eval_rows.extend(pq.read_table(shard).to_pylist())

    assert len(train_rows) == 10
    assert len(eval_rows) == 5
    assert {row["task_type"] for row in train_rows + eval_rows} == {"swe_live_failure_retrieval"}
    assert any("patch.diff" in row["retrieval_doc_text"] for row in train_rows + eval_rows)
