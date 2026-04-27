from __future__ import annotations

import json

from agent_kernel.schemas import TaskSpec
from scripts.evaluate_research_library_ab import build_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _status(tmp_path):
    repo_exports = tmp_path / "data/repository_library/exports"
    algorithms = tmp_path / "data/algorithms"
    _write_json(
        repo_exports / "_manifest.json",
        {
            "repos": {
                "AgentLab": {
                    "repo_root": "/data/repositories/AgentLab",
                    "languages": ["python"],
                    "indices": {"qa": {"size": 8}},
                    "extensions": {"repo_skills_miner": {"counts": {"skills": 5}}},
                }
            }
        },
    )
    _write_text(
        algorithms / "algorithms.jsonl",
        json.dumps(
            {
                "algo_id": "dijkstra",
                "names": ["Dijkstra"],
                "category": "shortest_path",
                "topics": ["graphs", "shortest_path"],
                "time_complexity": {"worst": "O(E + V log V)"},
                "notes": "Shortest path algorithm for non-negative weighted edges.",
            }
        )
        + "\n",
    )
    return {
        "schema_version": 1,
        "generated_at": "2026-04-27T00:00:00+00:00",
        "summary": {
            "paper_rows": 1000000,
            "paper_chunk_examples": 17117443,
            "paper_knn_edges": 20000062,
            "paper_topic_edges": 3000000,
            "repository_count": 1,
            "repository_mined_skill_count": 5,
            "algorithm_catalog_rows": 1,
            "algorithm_implementation_files": 1,
            "trained_model_assets": 2,
            "trained_adapter_assets": 1,
            "full_model_assets": 0,
            "faiss_index_assets": 0,
            "tolbert_checkpoint_assets": 1,
        },
        "sources": [
            {
                "id": "paper_text_1m",
                "label": "Papers",
                "kind": "paper_parquet_dataset",
                "path": "/arxiv/huggingface/paper_text_1m_dedup_v1",
                "status": "available",
            },
            {
                "id": "paper_chunks_p1",
                "label": "Chunks",
                "kind": "paper_chunk_parquet_dataset",
                "path": "/data/tmp/p1_full_paper_lm_hf_all_chunks",
                "status": "available",
            },
            {
                "id": "paper_universe",
                "label": "Paper Universe",
                "kind": "paper_graph",
                "path": "/data/repository_library/exports/_paper_universe",
                "status": "available",
            },
            {
                "id": "repository_exports",
                "label": "Repos",
                "kind": "repo_graph_exports",
                "path": str(repo_exports),
                "status": "available",
            },
            {
                "id": "algorithms",
                "label": "Algorithms",
                "kind": "jsonl_catalog",
                "path": str(algorithms),
                "status": "available",
            },
            {
                "id": "repository_models",
                "label": "Repo Models",
                "kind": "checkpoint_tree",
                "role": "trained_repository_paper_models",
                "status": "available",
                "model_assets": [
                    {
                        "group": "A3",
                        "asset_type": "peft_adapter",
                        "relative_path": "/data/repository_library/models/checkpoints/A3/checkpoint-3895",
                        "checkpoint_step": 3895,
                    }
                ],
            },
            {
                "id": "tolbert_checkpoints",
                "label": "TOLBERT",
                "kind": "checkpoint_tree",
                "role": "trained_hierarchical_retrieval_model",
                "status": "available",
                "model_assets": [
                    {
                        "group": "tolbert_epoch5.pt",
                        "asset_type": "tolbert_checkpoint",
                        "relative_path": "/data/TOLBERT_BRAIN/checkpoints/tolbert_brain/tolbert_epoch5.pt",
                    }
                ],
            },
        ],
    }


def test_research_library_ab_report_measures_candidate_signal_lift(tmp_path):
    status_path = tmp_path / "var/research_library/status.json"
    _write_json(status_path, _status(tmp_path))
    tasks = [
        TaskSpec(
            task_id="repo_fix",
            prompt="Fix a Python bug in the AgentLab repository with trained repair adapters.",
            workspace_subdir="work/repo_fix",
            metadata={
                "benchmark_family": "swe_bench_lite",
                "repo": "AgentLab",
                "expected_research_signals": [
                    "trained_models",
                    "paper_backbone",
                    "repositories",
                    "retrieval_evidence",
                ],
            },
        ),
        TaskSpec(
            task_id="graph_shortest_path",
            prompt="Solve Dijkstra shortest path for non-negative weighted edges.",
            workspace_subdir="work/graph",
            metadata={
                "benchmark_family": "codeforces",
                "expected_research_signals": ["trained_models", "algorithms", "retrieval_evidence"],
            },
        ),
    ]

    report = build_report(
        repo_root=tmp_path,
        tasks=tasks,
        status_path=status_path,
        config_path=tmp_path / "missing_sources.json",
    )

    assert report["measurement_type"] == "context_packet_ab"
    assert report["baseline"]["aggregate"]["expected_signal_hit_count"] == 0
    assert report["candidate"]["aggregate"]["expected_signal_hit_count"] == 7
    assert report["candidate"]["aggregate"]["model_asset_count"] > 0
    assert report["candidate"]["aggregate"]["repository_match_count"] >= 1
    assert report["candidate"]["aggregate"]["algorithm_match_count"] == 1
    assert report["delta"]["helps_context"] is True
