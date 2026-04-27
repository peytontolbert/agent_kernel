from __future__ import annotations

import json

from agent_kernel.research_library.sources import (
    build_research_library_status,
    resolve_source_path,
    write_research_library_status,
)
from agent_kernel.research_library.models import (
    iter_trained_model_assets,
    trained_model_asset_catalog,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _source_config():
    return {
        "schema_version": 1,
        "sources": [
            {
                "id": "paper_text_1m",
                "label": "Papers",
                "kind": "paper_parquet_dataset",
                "role": "source_of_truth",
                "path": "/arxiv/huggingface/paper_text_1m_dedup_v1",
                "stats_path": "stats.json",
                "row_metric": "merged_rows",
                "expected_rows": 3,
                "file_glob": "train_*.parquet",
            },
            {
                "id": "paper_chunks_p1",
                "label": "Chunks",
                "kind": "paper_chunk_parquet_dataset",
                "role": "derived_training_chunks",
                "path": "/data/tmp/p1_full_paper_lm_hf_all_chunks",
                "stats_path": "dataset_stats.json",
                "row_metric": "examples",
                "file_glob": "train-*.parquet",
            },
            {
                "id": "paper_universe",
                "label": "Paper Universe",
                "kind": "paper_graph",
                "role": "retrieval_backbone",
                "path": "/data/repository_library/exports/_paper_universe",
                "manifest_path": "manifest.json",
                "row_metric": "paper_count",
                "expected_rows": 3,
                "file_glob": "*.parquet",
            },
            {
                "id": "repository_exports",
                "label": "Repositories",
                "kind": "repo_graph_exports",
                "role": "code_source_of_truth",
                "path": "/data/repository_library/exports",
                "manifest_path": "_manifest.json",
            },
            {
                "id": "tolbert_joint_v2",
                "label": "TOLBERT",
                "kind": "tolbert_projection",
                "role": "derived_projection",
                "path": "/data/TOLBERT_BRAIN/data/joint_v2",
                "required_files": ["code.jsonl", "paper.jsonl", "nodes.jsonl"],
            },
            {
                "id": "algorithms",
                "label": "Algorithms",
                "kind": "jsonl_catalog",
                "role": "algorithm_retrieval",
                "path": "/data/algorithms",
                "jsonl_counts": {
                    "algorithms": "algorithms.jsonl",
                    "problems": "problems.jsonl",
                },
            },
            {
                "id": "algorithms_library",
                "label": "Algorithm Implementations",
                "kind": "source_tree",
                "role": "algorithm_implementations",
                "path": "/data/algorithms_library/Python",
                "file_glob": "**/*.py",
            },
            {
                "id": "repository_models",
                "label": "Repository Models",
                "kind": "checkpoint_tree",
                "role": "trained_repository_paper_models",
                "path": "/data/repository_library/models/checkpoints",
                "file_glob": "**/*",
            },
        ],
    }


def _seed_sources(root):
    _write_json(
        root / "arxiv/huggingface/paper_text_1m_dedup_v1/stats.json",
        {"merged_rows": 3, "merged_unique_canonical_ids": 3},
    )
    _write_text(root / "arxiv/huggingface/paper_text_1m_dedup_v1/train_00000.parquet")

    _write_json(
        root / "data/tmp/p1_full_paper_lm_hf_all_chunks/dataset_stats.json",
        {"examples": 5, "kept_papers": 2},
    )
    _write_text(root / "data/tmp/p1_full_paper_lm_hf_all_chunks/train-00000-of-00001.parquet")

    _write_json(
        root / "data/repository_library/exports/_paper_universe/manifest.json",
        {
            "paper_count": 3,
            "paper_knn": {"edge_count": 6},
            "paper_topic_edges": 9,
        },
    )
    _write_text(root / "data/repository_library/exports/_paper_universe/paper_nodes.parquet")

    _write_json(
        root / "data/repository_library/exports/_manifest.json",
        {
            "export_schema_version": 2,
            "manifest_version": 1,
            "repos": {
                "repo-a": {
                    "languages": ["Python"],
                    "indices": {"qa": {"size": 10}},
                    "extensions": {"repo_skills_miner": {"counts": {"skills": 2}}},
                },
                "repo-b": {
                    "languages": ["Rust"],
                    "indices": {},
                    "extensions": {"repo_skills_miner": {"counts": {"skills": 3}}},
                },
            },
        },
    )
    (root / "data/repository_library/exports/repo-a").mkdir(parents=True)
    (root / "data/repository_library/exports/repo-b").mkdir(parents=True)

    _write_text(root / "data/TOLBERT_BRAIN/data/joint_v2/code.jsonl")
    _write_text(root / "data/TOLBERT_BRAIN/data/joint_v2/paper.jsonl")
    _write_text(root / "data/TOLBERT_BRAIN/data/joint_v2/nodes.jsonl")
    _write_json(root / "data/TOLBERT_BRAIN/data/joint_v2/level_sizes_train_joint_v2.json", {"4": 7})

    _write_text(root / "data/algorithms/algorithms.jsonl", "{}\n{}\n")
    _write_text(root / "data/algorithms/problems.jsonl", "{}\n")
    _write_text(root / "data/algorithms_library/Python/dp/example.py", "print('ok')\n")

    model_root = root / "data/repository_library/models/checkpoints/A3/checkpoint-10"
    _write_json(
        model_root / "adapter_config.json",
        {
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        },
    )
    _write_json(model_root / "trainer_state.json", {"global_step": 10, "epoch": 1.0})
    _write_text(model_root / "adapter_model.safetensors", "weights")


def test_resolve_source_path_maps_absolute_paths_under_test_root(tmp_path):
    expected = tmp_path / "data/example"

    assert resolve_source_path("/data/example", root=tmp_path) == expected


def test_build_research_library_status_summarizes_registered_sources(tmp_path):
    _seed_sources(tmp_path)
    config_path = tmp_path / "config/research_library_sources.json"
    _write_json(config_path, _source_config())

    status = build_research_library_status(
        config_path=config_path,
        root=tmp_path,
        generated_at="2026-04-27T00:00:00+00:00",
    )

    assert status["summary"]["source_count"] == 8
    assert status["summary"]["available_count"] == 8
    assert status["summary"]["paper_rows"] == 3
    assert status["summary"]["paper_chunk_examples"] == 5
    assert status["summary"]["paper_knn_edges"] == 6
    assert status["summary"]["repository_count"] == 2
    assert status["summary"]["repository_qa_index_count"] == 1
    assert status["summary"]["repository_mined_skill_count"] == 5
    assert status["summary"]["algorithm_catalog_rows"] == 3
    assert status["summary"]["algorithm_implementation_files"] == 1
    assert status["summary"]["repository_model_assets"] == 1
    assert status["summary"]["trained_adapter_assets"] == 1
    repository_models = next(source for source in status["sources"] if source["id"] == "repository_models")
    assert repository_models["model_assets"][0]["asset_type"] == "peft_adapter"
    assert repository_models["model_assets"][0]["metadata"]["adapter_config.json"]["peft_type"] == "LORA"

    adapters = iter_trained_model_assets(status, asset_type="peft_adapter")
    catalog = trained_model_asset_catalog(status)

    assert adapters[0]["source_id"] == "repository_models"
    assert catalog["summary"]["asset_count"] == 1
    assert catalog["summary"]["by_type"] == {"peft_adapter": 1}


def test_build_research_library_status_marks_incomplete_sources_partial(tmp_path):
    config_path = tmp_path / "config/research_library_sources.json"
    config = _source_config()
    config["sources"] = [
        {
            "id": "tolbert_joint_v2",
            "label": "TOLBERT",
            "kind": "tolbert_projection",
            "role": "derived_projection",
            "path": "/data/TOLBERT_BRAIN/data/joint_v2",
            "required_files": ["present.jsonl", "missing.jsonl"],
        }
    ]
    _write_json(config_path, config)
    _write_text(tmp_path / "data/TOLBERT_BRAIN/data/joint_v2/present.jsonl")

    status = build_research_library_status(config_path=config_path, root=tmp_path)

    assert status["summary"]["partial_count"] == 1
    assert status["sources"][0]["missing_files"] == ["missing.jsonl"]


def test_write_research_library_status(tmp_path):
    _seed_sources(tmp_path)
    config_path = tmp_path / "config/research_library_sources.json"
    output_path = tmp_path / "var/research_library/status.json"
    _write_json(config_path, _source_config())

    payload = write_research_library_status(
        output_path,
        config_path=config_path,
        root=tmp_path,
        generated_at="2026-04-27T00:00:00+00:00",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["summary"] == payload["summary"]
    assert written["summary"]["available_count"] == 8
