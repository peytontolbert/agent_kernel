from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.runtime_modeling_adapter import build_context_provider
from agent_kernel.extensions.tolbert import _paper_research_runtime_paths
from agent_kernel.schemas import TaskSpec
from agent_kernel.state import AgentState


DEFAULT_STATUS = Path("var/research_library/status.json")
DEFAULT_BROWSER_APP = Path("/data/repository_library/exports/agent_kernel/js/agent-kernel-app.js")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_V2_CHECKPOINT_DIR = Path("/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_map(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = status.get("sources", [])
    if not isinstance(sources, list):
        return {}
    return {
        str(source.get("id", "")): dict(source)
        for source in sources
        if isinstance(source, dict) and str(source.get("id", "")).strip()
    }


def _source_brief(source: dict[str, Any]) -> dict[str, Any]:
    metrics = source.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "status": source.get("status", ""),
        "kind": source.get("kind", ""),
        "path": source.get("path", ""),
        "rows": source.get("rows"),
        "file_count": source.get("file_count"),
        "size_bytes": source.get("size_bytes"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "expected_rows",
                "row_coverage",
                "paper_count",
                "examples",
                "repository_count",
                "mined_skill_count",
                "qa_index_count",
                "model_asset_count",
                "tolbert_checkpoint_count",
                "retrieval_cache_count",
            )
            if key in metrics
        },
    }


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 0


def _path_exists(raw_path: object) -> bool:
    text = str(raw_path or "").strip()
    return bool(text) and Path(text).exists()


def _browser_retrieval_audit(browser_app: Path) -> dict[str, Any]:
    try:
        text = browser_app.read_text(encoding="utf-8")
    except OSError:
        return {
            "browser_app": str(browser_app),
            "available": False,
            "paper_dataset": "",
            "paper_pack_root": "",
            "dataset_server": "",
            "paper_retrieval": False,
            "code_retrieval": False,
            "repository_retrieval": False,
        }
    paper_dataset = _first_js_string(text, "paperTextDataset")
    paper_pack_root = _first_js_string(text, "paperInteractiveRoot")
    dataset_server = _first_js_string(text, "datasetServer")
    return {
        "browser_app": str(browser_app),
        "available": True,
        "paper_dataset": paper_dataset,
        "paper_pack_root": paper_pack_root,
        "dataset_server": dataset_server,
        "paper_retrieval": bool("retrieveContext" in text and paper_dataset),
        "hf_dataset_search": bool("/search" in text and "hfSearchRows" in text),
        "local_paper_pack_search": bool("state.packRows" in text and "scorePaper" in text),
        "arxiv_pdf_links": bool("arxivPdfUrl" in text and "https://arxiv.org/pdf/" in text),
        "code_retrieval": bool(re.search(r"code(Text|Graph|Dataset|Retrieval)", text)),
        "repository_retrieval": bool(re.search(r"repo(Graph|Dataset|Retrieval|sitory)", text)),
    }


def _first_js_string(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}:\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else ""


def _paper_runtime_audit(config: KernelConfig, repo_root: Path) -> dict[str, Any]:
    runtime_paths = _paper_research_runtime_paths(config, repo_root=repo_root)
    if not runtime_paths:
        return {
            "available": False,
            "runtime_paths": {},
            "uses_joint_v2": False,
            "joint_v2_checkpoint_dir_exists": JOINT_V2_CHECKPOINT_DIR.exists(),
        }
    flattened = " ".join(
        str(value)
        for value in [
            runtime_paths.get("tolbert_config_path"),
            runtime_paths.get("tolbert_checkpoint_path"),
            runtime_paths.get("tolbert_nodes_path"),
            runtime_paths.get("tolbert_label_map_path"),
            *list(runtime_paths.get("tolbert_source_spans_paths", []) or []),
            *list(runtime_paths.get("tolbert_cache_paths", []) or []),
        ]
    )
    return {
        "available": True,
        "runtime_paths": runtime_paths,
        "uses_joint_v2": "joint_v2" in flattened or "tolbert_brain_joint_v2" in flattened,
        "joint_v2_checkpoint_dir_exists": JOINT_V2_CHECKPOINT_DIR.exists(),
        "all_runtime_files_exist": all(
            _path_exists(path)
            for path in [
                runtime_paths.get("tolbert_config_path"),
                runtime_paths.get("tolbert_checkpoint_path"),
                runtime_paths.get("tolbert_nodes_path"),
                runtime_paths.get("tolbert_label_map_path"),
                *list(runtime_paths.get("tolbert_source_spans_paths", []) or []),
                *list(runtime_paths.get("tolbert_cache_paths", []) or []),
            ]
        ),
    }


def _standalone_probe(repo_root: Path, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_research_library_context=True,
        research_library_standalone_context=True,
        research_library_context_max_chunks=8,
        research_library_context_max_models=8,
        research_library_context_max_repositories=4,
        research_library_context_max_algorithms=4,
    )
    provider = build_context_provider(config=config, repo_root=repo_root)
    if provider is None:
        return {"provider_available": False}
    try:
        packet = provider.compile(
            AgentState(
                task=TaskSpec(
                    task_id=str(metadata.get("task_id", "retrieval_probe")),
                    prompt=prompt,
                    workspace_subdir="var/research_library/audit_probe",
                    metadata={key: value for key, value in metadata.items() if key != "task_id"},
                )
            )
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    chunks = [
        dict(chunk)
        for chunk in list(packet.control.get("selected_context_chunks", []) or [])
        if isinstance(chunk, dict)
    ]
    guidance = packet.control.get("retrieval_guidance", {})
    guidance = guidance if isinstance(guidance, dict) else {}
    evidence = [str(item) for item in list(guidance.get("evidence", []) or []) if str(item).strip()]
    return {
        "provider_available": True,
        "selected_chunk_ids": [str(chunk.get("span_id", "")) for chunk in chunks],
        "research_evidence_count": sum(1 for item in evidence if "research_library" in item),
        "evidence_preview": evidence[:5],
        "repository_code_hit_count": sum(
            len(list(dict(chunk.get("metadata", {}) or {}).get("repository_code_hits", []) or []))
            for chunk in chunks
            if str(chunk.get("span_id", "")) == "research:repository_code"
        ),
        "paper_hit_count": sum(
            len(list(dict(chunk.get("metadata", {}) or {}).get("papers", []) or []))
            for chunk in chunks
            if str(chunk.get("span_id", "")) == "research:paper_hits"
        ),
        "algorithm_hit_count": sum(
            len(list(dict(chunk.get("metadata", {}) or {}).get("algorithms", []) or []))
            for chunk in chunks
            if str(chunk.get("span_id", "")) == "research:algorithms"
        ),
    }


def _probe_report(repo_root: Path) -> dict[str, Any]:
    probes = [
        (
            "repo_code_transformers",
            "In transformers, how does flash_attn_supports_top_left_mask decide top-left mask support?",
            {
                "task_id": "repo_code_transformers",
                "benchmark_family": "analysis",
                "repo": "transformers",
                "code_symbol": "flash_attn_supports_top_left_mask",
            },
        ),
        (
            "paper_attention_kernel",
            "Find public paper evidence about exact attention kernel implementation details for FlashAttention.",
            {
                "task_id": "paper_attention_kernel",
                "benchmark_family": "analysis",
                "research_topic": "FlashAttention exact attention kernel implementation details",
            },
        ),
        (
            "algorithm_shortest_path",
            "Solve a shortest path task with non-negative weighted graph constraints and explain complexity.",
            {
                "task_id": "algorithm_shortest_path",
                "benchmark_family": "analysis",
                "research_topic": "shortest path non-negative weighted graph complexity",
            },
        ),
    ]
    return {
        name: _standalone_probe(repo_root, prompt, metadata)
        for name, prompt, metadata in probes
    }


def build_audit(
    *,
    repo_root: Path,
    status_path: Path,
    browser_app: Path,
    run_probes: bool,
) -> dict[str, Any]:
    status = _load_json(status_path)
    sources = _source_map(status)
    summary = status.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    config = KernelConfig()
    paper_chunks = sources.get("paper_chunks_p1", {})
    paper_chunk_rows = _safe_int(paper_chunks.get("rows")) or _safe_int(
        dict(paper_chunks.get("metrics", {}) or {}).get("examples")
    )
    paper_scan_rows = max(0, int(config.research_library_paper_scan_row_limit))
    paper_scan_files = max(0, int(config.research_library_paper_scan_file_limit))
    audit = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "status_path": str(status_path),
        "public_dataset_coverage": {
            "paper_rows": summary.get("paper_rows", 0),
            "paper_chunk_examples": summary.get("paper_chunk_examples", 0),
            "paper_universe_rows": summary.get("paper_universe_rows", 0),
            "paper_knn_edges": summary.get("paper_knn_edges", 0),
            "paper_topic_edges": summary.get("paper_topic_edges", 0),
            "repository_count": summary.get("repository_count", 0),
            "repository_mined_skill_count": summary.get("repository_mined_skill_count", 0),
            "repository_qa_index_count": summary.get("repository_qa_index_count", 0),
            "algorithm_catalog_rows": summary.get("algorithm_catalog_rows", 0),
            "algorithm_implementation_files": summary.get("algorithm_implementation_files", 0),
            "trained_model_assets": summary.get("trained_model_assets", 0),
        },
        "source_briefs": {
            source_id: _source_brief(source)
            for source_id, source in sources.items()
            if source_id
            in {
                "paper_text_1m",
                "paper_chunks_p1",
                "paper_universe",
                "repository_exports",
                "tolbert_joint_v2",
                "algorithms",
                "algorithms_library",
                "repository_models",
                "digital_world_model_checkpoints",
                "tolbert_checkpoints",
            }
        },
        "runtime_flags": {
            "use_tolbert_context_default": config.use_tolbert_context,
            "use_research_library_context_default": config.use_research_library_context,
            "research_library_standalone_context_default": config.research_library_standalone_context,
            "use_paper_research_context_default": config.use_paper_research_context,
            "paper_research_query_mode": config.paper_research_query_mode,
            "research_library_paper_scan_file_limit": paper_scan_files,
            "research_library_paper_scan_row_limit": paper_scan_rows,
            "standalone_adapter_scan_fraction_of_chunks": (
                0.0 if paper_chunk_rows <= 0 else paper_scan_rows / paper_chunk_rows
            ),
        },
        "paper_research_sidecar": _paper_runtime_audit(config, repo_root),
        "browser_retrieval": _browser_retrieval_audit(browser_app),
        "current_paths": {
            "python_code_retrieval": [
                "repository manifest ranking",
                "repo_skills_miner.skills.jsonl snippets",
                "repo graph entity span fallback against local repo roots",
            ],
            "python_paper_retrieval": [
                "TOLBERT paper_research sidecar when prompt is research-like",
                "standalone research adapter bounded parquet scan over paper_chunks_p1",
                "paper_backbone inventory chunk for source paths and counts",
            ],
            "browser_retrieval": [
                "paper universe interactive JSON packs",
                "Hugging Face dataset-server /search and /rows for PeytonT/1m_papers_text",
                "arXiv PDF URL construction from paper ids",
            ],
        },
        "gaps": [
            (
                "Standalone paper hits do not use the 1M paper universe embeddings or a full-text vector/lexical index; "
                "the default bounded scan covers only a tiny prefix of paper_chunks_p1."
            ),
            (
                "The active paper_research sidecar falls back to the older joint bundle unless explicit v2 "
                "checkpoint/cache paths are provided; the joint_v2 span projection exists but its promoted runtime "
                "checkpoint/cache directory is absent."
            ),
            (
                "Repository code retrieval is useful server-side, but the browser app is currently paper-only and "
                "does not expose public repo graph/code datasets to AgentKernel Lite."
            ),
            (
                "The adapter can emit probe-specific applied facts and exact commands. That is useful for diagnostics "
                "but should be replaced by a general evidence-normalization and reranking layer before broad benchmark use."
            ),
            (
                "The context budgeter boosts paper hits/applied guidance, but there is no explicit source-kind diversity "
                "floor for paper/code/algorithm evidence."
            ),
        ],
        "recommended_sequence": [
            "Add a common EvidenceRecord schema for paper, repo_code, repo_qa, algorithm, artifact, and trace candidates.",
            "Implement a ResearchLibraryQueryClient that returns normal ContextPacket retrieval spans from public paper/code datasets.",
            "Use paper_universe metadata/full-text embeddings for candidate generation, then fetch snippets from paper_chunks_p1 or paper_text_1m by id/offset.",
            "Extend browser retrieval with a repo graph/code dataset manifest and local IndexedDB/OPFS cache, mirroring the paper pack path.",
            "Promote joint_v2 only after training/building compatible checkpoint and sharded retrieval caches, then run it as shadow-only before active use.",
            "Log retrieved, injected, selected, and ignored evidence in run artifacts so benchmark gains can be audited.",
        ],
    }
    if run_probes:
        audit["standalone_probe_results"] = _probe_report(repo_root)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit AgentKernel context retrieval coverage over public code and paper datasets."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--browser-app", type=Path, default=DEFAULT_BROWSER_APP)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-probes", action="store_true")
    args = parser.parse_args()

    repo_root = args.root.resolve()
    status_path = args.status if args.status.is_absolute() else repo_root / args.status
    browser_app = args.browser_app
    if not browser_app.is_absolute():
        browser_app = repo_root / browser_app
    audit = build_audit(
        repo_root=repo_root,
        status_path=status_path,
        browser_app=browser_app,
        run_probes=bool(args.run_probes),
    )
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"context_retrieval_public_dataset_audit output={output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
