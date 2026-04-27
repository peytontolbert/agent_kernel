from __future__ import annotations

import json

import pytest

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.runtime_modeling_adapter import build_context_provider
from agent_kernel.schemas import TaskSpec
from agent_kernel.state import AgentState


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _research_status(tmp_path):
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
                },
                "OtherRepo": {
                    "repo_root": "/data/repositories/OtherRepo",
                    "languages": ["rust"],
                    "indices": {},
                },
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
                "notes": "Shortest path algorithm for non-negative edges.",
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
            "repository_count": 2,
            "repository_mined_skill_count": 5,
            "algorithm_catalog_rows": 1,
            "algorithm_implementation_files": 1,
            "trained_model_assets": 3,
            "trained_adapter_assets": 1,
            "full_model_assets": 1,
            "faiss_index_assets": 1,
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
                "id": "digital_world_model_checkpoints",
                "label": "DWM",
                "kind": "checkpoint_tree",
                "role": "trained_planner_verifier_coder_memory_models",
                "status": "available",
                "model_assets": [
                    {
                        "group": "memory",
                        "asset_type": "faiss_index",
                        "relative_path": "/data/checkpoints/digital_world_model/memory",
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


def test_research_library_context_provider_adds_capability_chunks(tmp_path):
    status_path = tmp_path / "var/research_library/status.json"
    _write_json(status_path, _research_status(tmp_path))
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_research_library_context=True,
        research_library_standalone_context=True,
        research_library_status_path=status_path,
        research_library_context_max_chunks=6,
        research_library_context_max_models=4,
    )
    provider = build_context_provider(config=config, repo_root=tmp_path)
    assert provider is not None
    state = AgentState(
        task=TaskSpec(
            task_id="codeforces_shortest_path",
            prompt="Fix a Python shortest path algorithm bug in the AgentLab repository tests.",
            workspace_subdir="work",
            metadata={"benchmark_family": "codeforces", "repo": "AgentLab"},
        )
    )

    packet = provider.compile(state)

    research = packet.control["research_library"]
    assert research["enabled"] is True
    assert research["summary"]["paper_rows"] == 1000000
    span_ids = [chunk["span_id"] for chunk in packet.control["selected_context_chunks"]]
    assert "research:inventory" in span_ids
    assert "research:trained_models" in span_ids
    assert "research:repositories" in span_ids
    assert "research:algorithms" in span_ids
    model_chunk = next(chunk for chunk in packet.control["selected_context_chunks"] if chunk["span_id"] == "research:trained_models")
    assert any(asset["asset_type"] == "tolbert_checkpoint" for asset in model_chunk["metadata"]["assets"])
    algorithm_chunk = next(chunk for chunk in packet.control["selected_context_chunks"] if chunk["span_id"] == "research:algorithms")
    assert "notes=Shortest path algorithm for non-negative edges." in algorithm_chunk["text"]
    assert any(span["span_id"] == "research:inventory" for span in packet.retrieval["global"])
    assert any("prefer existing trained adapters" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_research_library_context_adds_paper_content_hits(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    chunks_dir = tmp_path / "paper_chunks"
    chunks_dir.mkdir(parents=True)
    paper_text = (
        "Title: Developing Artificial Herders Using Jason\n\n"
        "Abstract: This paper gives an overview of a proposed strategy for the Cows and Herders scenario. "
        "The strategy is to be implemented using the Jason platform, based on the agent-oriented "
        "programming language Agent-Speak. The paper describes the agents, their goals and strategies."
    )
    table = pa.Table.from_pylist(
        [
            {
                "id": "1001.0115:0:0",
                "paper_id": "1001.0115",
                "title": "Developing Artificial Herders Using Jason",
                "categories": "cs.MA",
                "year": 2010,
                "chunk_index": 0,
                "text": paper_text,
            }
        ]
    )
    pq.write_table(table, chunks_dir / "train-00000.parquet")
    status = _research_status(tmp_path)
    for source in status["sources"]:
        if source["id"] == "paper_chunks_p1":
            source["path"] = str(chunks_dir)
    status_path = tmp_path / "var/research_library/status.json"
    _write_json(status_path, status)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_research_library_context=True,
        research_library_standalone_context=True,
        research_library_status_path=status_path,
        research_library_context_max_chunks=6,
        research_library_context_max_paper_hits=1,
        research_library_paper_scan_file_limit=1,
        research_library_paper_scan_row_limit=8,
    )
    provider = build_context_provider(config=config, repo_root=tmp_path)
    assert provider is not None
    state = AgentState(
        task=TaskSpec(
            task_id="paper_content_probe",
            prompt=(
                "Find the local paper chunk for Developing Artificial Herders Using Jason. "
                "What sentence says how the strategy is implemented? Create answer.txt."
            ),
            workspace_subdir="work",
            expected_file_contents={"answer.txt": "placeholder"},
            metadata={
                "benchmark_family": "research_library_hard_probe",
                "paper_id": "1001.0115",
                "paper_title": "Developing Artificial Herders Using Jason",
            },
        )
    )

    packet = provider.compile(state)

    paper_chunk = next(chunk for chunk in packet.control["selected_context_chunks"] if chunk["span_id"] == "research:paper_hits")
    assert "The strategy is to be implemented using the Jason platform" in paper_chunk["text"]
    assert "evidence_sentence=The strategy is to be implemented using the Jason platform" in paper_chunk["text"]
    assert paper_chunk["metadata"]["papers"][0]["paper_id"] == "1001.0115"
    assert any("research_library: paper_hit paper_id=1001.0115" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_research_library_context_promotes_content_derived_applied_guidance(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    chunks_dir = tmp_path / "paper_chunks"
    chunks_dir.mkdir(parents=True)
    paper_text = (
        "Title: A Note on Leftist Grammars\n\n"
        "The rules of P define a one-step rewrite relation: for rule alpha -> beta, "
        "u is some u1 alphau2 with |u1 alpha| = p and u_prime = u1 betau2. "
        "This convention means the position p is measured at the end of alpha, not its start. "
        "Given a step u => r,p v, the p-th letter in u is the active letter."
    )
    table = pa.Table.from_pylist(
        [
            {
                "id": "leftist:0:0",
                "paper_id": "leftist",
                "title": "A Note on Leftist Grammars",
                "categories": "cs.FL",
                "year": 2011,
                "chunk_index": 0,
                "text": paper_text,
            }
        ]
    )
    pq.write_table(table, chunks_dir / "train-00000.parquet")
    status = _research_status(tmp_path)
    for source in status["sources"]:
        if source["id"] == "paper_chunks_p1":
            source["path"] = str(chunks_dir)
    status_path = tmp_path / "var/research_library/status.json"
    _write_json(status_path, status)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_research_library_context=True,
        research_library_standalone_context=True,
        research_library_status_path=status_path,
        research_library_context_max_chunks=6,
        research_library_context_max_paper_hits=1,
        research_library_paper_scan_file_limit=1,
        research_library_paper_scan_row_limit=8,
    )
    provider = build_context_provider(config=config, repo_root=tmp_path)
    assert provider is not None
    state = AgentState(
        task=TaskSpec(
            task_id="leftist_guidance_probe",
            prompt=(
                "Implement replacement_start(alpha, p) for the one-step rewrite relation "
                "from local research context about leftist grammars and semi-Thue systems."
            ),
            workspace_subdir="work",
            expected_files=["leftist_window.py"],
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "research_topic": "leftist grammars semi-Thue one-step rewrite relation alpha beta u1 u2 position p",
            },
        )
    )

    packet = provider.compile(state)

    span_ids = [chunk["span_id"] for chunk in packet.control["selected_context_chunks"]]
    assert span_ids.index("research:applied_guidance") < span_ids.index("research:paper_hits")
    guidance_chunk = next(
        chunk for chunk in packet.control["selected_context_chunks"] if chunk["span_id"] == "research:applied_guidance"
    )
    assert "replacement_start(alpha, p): start = p - len(alpha)" in guidance_chunk["text"]
    assert "rewrite_at_position(word, alpha, beta, p)" in guidance_chunk["text"]
    assert "active_index = p - 1" in guidance_chunk["text"]
    assert any(
        "replacement_start(alpha, p): start = p - len(alpha)" in item
        for item in packet.control["research_library"]["applied_guidance"]
    )
    assert any(
        "research_library: applied_guidance replacement_start(alpha, p): start = p - len(alpha)" in item
        for item in packet.control["retrieval_guidance"]["evidence"]
    )


def test_research_library_context_expands_visible_chunk_budget(tmp_path):
    status_path = tmp_path / "var/research_library/status.json"
    _write_json(status_path, _research_status(tmp_path))
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        use_research_library_context=True,
        research_library_standalone_context=False,
        research_library_status_path=status_path,
        tolbert_context_max_chunks=2,
        research_library_context_max_chunks=4,
    )
    provider = build_context_provider(config=config, repo_root=tmp_path)
    assert provider is not None
    state = AgentState(
        task=TaskSpec(
            task_id="budgeted_context",
            prompt="Fix a Python shortest path algorithm bug in the AgentLab repository tests.",
            workspace_subdir="work",
            suggested_commands=["printf one", "printf two"],
            metadata={"benchmark_family": "codeforces", "repo": "AgentLab"},
        )
    )

    packet = provider.compile(state)

    budget = packet.control["context_chunk_budget"]
    span_ids = [chunk["span_id"] for chunk in packet.control["selected_context_chunks"]]
    visible_span_ids = span_ids[: budget["max_chunks"]]
    assert budget["base_max_chunks"] == 2
    assert budget["research_library_chunks"] == 4
    assert budget["max_chunks"] == 6
    assert "task:budgeted_context:suggested:1" in visible_span_ids
    assert "research:inventory" in visible_span_ids
    assert "research:trained_models" in visible_span_ids
