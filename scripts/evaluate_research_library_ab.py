from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.runtime_modeling_adapter import build_context_provider
from agent_kernel.research_library import (
    DEFAULT_RESEARCH_LIBRARY_CONFIG,
    write_research_library_status,
)
from agent_kernel.schemas import ContextPacket, TaskSpec
from agent_kernel.state import AgentState


DEFAULT_OUTPUT = Path("var/research_library/ab_report.json")
DEFAULT_STATUS = Path("var/research_library/status.json")
SIGNAL_KEYS = (
    "inventory",
    "trained_models",
    "paper_backbone",
    "repositories",
    "algorithms",
    "retrieval_evidence",
)


def default_ab_tasks() -> list[TaskSpec]:
    """Small deterministic prompts that exercise the research-library adapter surfaces."""
    return [
        TaskSpec(
            task_id="research_ab_swe_bench_repo_fix",
            prompt=(
                "Fix a SWE-Bench Lite style Python repository bug in AgentLab tests. "
                "Use repository history, trained adapters, and prior code repair knowledge."
            ),
            workspace_subdir="var/research_library/ab/swe_bench",
            metadata={
                "benchmark_family": "swe_bench_lite",
                "repo": "AgentLab",
                "expected_research_signals": ["trained_models", "repositories", "retrieval_evidence"],
            },
        ),
        TaskSpec(
            task_id="research_ab_swe_rebench_regression",
            prompt=(
                "Diagnose a SWE-ReBench regression in a Python project and recover the failing "
                "test with repository exports, learned repair skills, and trained checkpoints."
            ),
            workspace_subdir="var/research_library/ab/swe_rebench",
            metadata={
                "benchmark_family": "swe_rebench",
                "repo": "AgentLab",
                "expected_research_signals": ["trained_models", "repositories", "retrieval_evidence"],
            },
        ),
        TaskSpec(
            task_id="research_ab_mle_bench_pipeline",
            prompt=(
                "Improve an MLE-Bench modeling pipeline by selecting relevant papers, retrieval "
                "memory, and trained model assets before deciding whether retraining is needed."
            ),
            workspace_subdir="var/research_library/ab/mle_bench",
            metadata={
                "benchmark_family": "mle_bench",
                "expected_research_signals": ["trained_models", "paper_backbone", "retrieval_evidence"],
            },
        ),
        TaskSpec(
            task_id="research_ab_rebench_paper_to_code",
            prompt=(
                "Plan a ReBench paper-to-code alignment task using the 1M full-text paper corpus, "
                "paper graph, TOLBERT checkpoints, and trained retrieval adapters."
            ),
            workspace_subdir="var/research_library/ab/rebench",
            metadata={
                "benchmark_family": "re_bench",
                "expected_research_signals": ["trained_models", "paper_backbone", "retrieval_evidence"],
            },
        ),
        TaskSpec(
            task_id="research_ab_codeforces_dijkstra",
            prompt=(
                "Solve a Codeforces shortest path graph problem with non-negative weighted edges. "
                "Use Dijkstra and check complexity against the algorithm library."
            ),
            workspace_subdir="var/research_library/ab/codeforces",
            metadata={
                "benchmark_family": "codeforces",
                "expected_research_signals": ["trained_models", "algorithms", "retrieval_evidence"],
            },
        ),
        TaskSpec(
            task_id="research_ab_kernel_adapter",
            prompt=(
                "Extend agent_kernel with a retrieval adapter that routes between papers, "
                "repository exports, algorithms, and trained local model assets."
            ),
            workspace_subdir="var/research_library/ab/agent_kernel",
            metadata={
                "benchmark_family": "agent_kernel",
                "repo": "agentkernel",
                "expected_research_signals": [
                    "trained_models",
                    "paper_backbone",
                    "retrieval_evidence",
                ],
            },
        ),
    ]


def _load_tasks(path: Path | None) -> list[TaskSpec]:
    if path is None:
        return default_ab_tasks()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_tasks = payload.get("tasks", [])
    else:
        raw_tasks = payload
    if not isinstance(raw_tasks, list):
        raise ValueError(f"task file must contain a list or an object with tasks: {path}")
    return [TaskSpec.from_dict(task) for task in raw_tasks if isinstance(task, dict)]


def _make_config(
    *,
    research_enabled: bool,
    status_path: Path,
    config_path: Path,
    standalone: bool,
    max_chunks: int,
    max_models: int,
    max_repositories: int,
    max_algorithms: int,
) -> KernelConfig:
    return KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        use_research_library_context=research_enabled,
        research_library_standalone_context=standalone,
        research_library_config_path=config_path,
        research_library_status_path=status_path,
        research_library_context_max_chunks=max_chunks,
        research_library_context_max_models=max_models,
        research_library_context_max_repositories=max_repositories,
        research_library_context_max_algorithms=max_algorithms,
    )


def _compile_packet(*, config: KernelConfig, repo_root: Path, task: TaskSpec) -> ContextPacket | None:
    provider = build_context_provider(config=config, repo_root=repo_root)
    if provider is None:
        return None
    try:
        return provider.compile(AgentState(task=task))
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _list_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _packet_metrics(packet: ContextPacket | None) -> dict[str, Any]:
    if packet is None:
        return {
            "provider_available": False,
            "total_context_chunks": 0,
            "research_context_chunks": 0,
            "llm_visible_context_chunks": 0,
            "llm_visible_research_context_chunks": 0,
            "research_global_spans": 0,
            "evidence_count": 0,
            "research_evidence_count": 0,
            "model_asset_count": 0,
            "repository_match_count": 0,
            "algorithm_match_count": 0,
            "signals": {key: False for key in SIGNAL_KEYS},
            "selected_chunk_ids": [],
            "model_assets": [],
            "repositories": [],
            "algorithms": [],
        }
    control = dict(packet.control or {})
    retrieval = dict(packet.retrieval or {})
    chunks = _list_dicts(control.get("selected_context_chunks", []))
    chunk_budget = dict(control.get("context_chunk_budget", {}) or {}) if isinstance(control.get("context_chunk_budget", {}), dict) else {}
    max_visible_chunks = int(chunk_budget.get("max_chunks", 8) or 8)
    visible_chunks = chunks[:max_visible_chunks]
    research_chunks = [
        chunk
        for chunk in chunks
        if str(chunk.get("span_id", "")).startswith("research:")
        or str(chunk.get("source_id", "")) == "research_library"
    ]
    research_chunk_ids = [str(chunk.get("span_id", "")) for chunk in research_chunks if str(chunk.get("span_id", ""))]
    visible_research_chunks = [
        chunk
        for chunk in visible_chunks
        if str(chunk.get("span_id", "")).startswith("research:")
        or str(chunk.get("source_id", "")) == "research_library"
    ]
    model_assets: list[dict[str, Any]] = []
    repositories: list[dict[str, Any]] = []
    algorithms: list[dict[str, Any]] = []
    for chunk in research_chunks:
        metadata = dict(chunk.get("metadata", {}) or {}) if isinstance(chunk.get("metadata", {}), dict) else {}
        model_assets.extend(_list_dicts(metadata.get("assets", [])))
        repositories.extend(_list_dicts(metadata.get("repositories", [])))
        algorithms.extend(_list_dicts(metadata.get("algorithms", [])))
    guidance = dict(control.get("retrieval_guidance", {}) or {})
    evidence = [str(item) for item in guidance.get("evidence", []) if str(item).strip()]
    research_evidence = [item for item in evidence if "research_library" in item]
    global_spans = _list_dicts(retrieval.get("global", []))
    research_global_spans = [
        span
        for span in global_spans
        if str(span.get("span_id", "")).startswith("research:")
        or str(span.get("source_id", "")) == "research_library"
    ]
    signals = {
        "inventory": "research:inventory" in research_chunk_ids,
        "trained_models": bool(model_assets),
        "paper_backbone": "research:paper_backbone" in research_chunk_ids,
        "repositories": bool(repositories),
        "algorithms": bool(algorithms),
        "retrieval_evidence": any("research_library" in item for item in evidence),
    }
    return {
        "provider_available": True,
        "total_context_chunks": len(chunks),
        "research_context_chunks": len(research_chunks),
        "llm_visible_context_chunks": len(visible_chunks),
        "llm_visible_research_context_chunks": len(visible_research_chunks),
        "research_global_spans": len(research_global_spans),
        "evidence_count": len(evidence),
        "research_evidence_count": len(research_evidence),
        "model_asset_count": len(model_assets),
        "repository_match_count": len(repositories),
        "algorithm_match_count": len(algorithms),
        "signals": signals,
        "selected_chunk_ids": research_chunk_ids,
        "model_assets": model_assets,
        "repositories": repositories,
        "algorithms": algorithms,
    }


def _task_report(*, task: TaskSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    expected = [
        str(value)
        for value in task.metadata.get("expected_research_signals", [])
        if str(value).strip()
    ]
    signals = dict(metrics.get("signals", {}) or {})
    signal_hits = {signal: bool(signals.get(signal, False)) for signal in expected}
    hit_count = sum(1 for hit in signal_hits.values() if hit)
    return {
        "task_id": task.task_id,
        "benchmark_family": str(task.metadata.get("benchmark_family", "")),
        "expected_signals": expected,
        "expected_signal_hits": signal_hits,
        "expected_signal_hit_count": hit_count,
        "expected_signal_hit_rate": 0.0 if not expected else hit_count / len(expected),
        "metrics": metrics,
    }


def _aggregate(task_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_expected = 0
    total_hits = 0
    aggregate = {
        "task_count": len(task_reports),
        "provider_available_tasks": 0,
        "total_context_chunks": 0,
        "research_context_chunks": 0,
        "llm_visible_context_chunks": 0,
        "llm_visible_research_context_chunks": 0,
        "research_global_spans": 0,
        "evidence_count": 0,
        "research_evidence_count": 0,
        "model_asset_count": 0,
        "repository_match_count": 0,
        "algorithm_match_count": 0,
        "signal_hits_by_key": {key: 0 for key in SIGNAL_KEYS},
    }
    for report in task_reports:
        metrics = dict(report.get("metrics", {}) or {})
        if bool(metrics.get("provider_available", False)):
            aggregate["provider_available_tasks"] += 1
        for key in (
            "total_context_chunks",
            "research_context_chunks",
            "llm_visible_context_chunks",
            "llm_visible_research_context_chunks",
            "research_global_spans",
            "evidence_count",
            "research_evidence_count",
            "model_asset_count",
            "repository_match_count",
            "algorithm_match_count",
        ):
            aggregate[key] += int(metrics.get(key, 0) or 0)
        signals = dict(metrics.get("signals", {}) or {})
        for key in SIGNAL_KEYS:
            if bool(signals.get(key, False)):
                aggregate["signal_hits_by_key"][key] += 1
        expected = list(report.get("expected_signals", []) or [])
        total_expected += len(expected)
        total_hits += int(report.get("expected_signal_hit_count", 0) or 0)
    aggregate["expected_signal_count"] = total_expected
    aggregate["expected_signal_hit_count"] = total_hits
    aggregate["expected_signal_hit_rate"] = 0.0 if total_expected == 0 else total_hits / total_expected
    return aggregate


def _variant_report(
    *,
    name: str,
    research_enabled: bool,
    config: KernelConfig,
    repo_root: Path,
    tasks: list[TaskSpec],
) -> dict[str, Any]:
    reports = []
    for task in tasks:
        packet = _compile_packet(config=config, repo_root=repo_root, task=task)
        reports.append(_task_report(task=task, metrics=_packet_metrics(packet)))
    return {
        "name": name,
        "research_enabled": research_enabled,
        "aggregate": _aggregate(reports),
        "tasks": reports,
    }


def _delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = dict(baseline.get("aggregate", {}) or {})
    cand = dict(candidate.get("aggregate", {}) or {})
    numeric_keys = (
        "provider_available_tasks",
        "total_context_chunks",
        "research_context_chunks",
        "llm_visible_context_chunks",
        "llm_visible_research_context_chunks",
        "research_global_spans",
        "evidence_count",
        "research_evidence_count",
        "model_asset_count",
        "repository_match_count",
        "algorithm_match_count",
        "expected_signal_count",
        "expected_signal_hit_count",
        "expected_signal_hit_rate",
    )
    result: dict[str, Any] = {}
    for key in numeric_keys:
        result[key] = cand.get(key, 0) - base.get(key, 0)
    base_hits = dict(base.get("signal_hits_by_key", {}) or {})
    cand_hits = dict(cand.get("signal_hits_by_key", {}) or {})
    result["signal_hits_by_key"] = {
        key: int(cand_hits.get(key, 0) or 0) - int(base_hits.get(key, 0) or 0)
        for key in SIGNAL_KEYS
    }
    result["helps_context"] = (
        float(result.get("expected_signal_hit_rate", 0.0) or 0.0) > 0.0
        and int(result.get("research_context_chunks", 0) or 0) > 0
    )
    return result


def build_report(
    *,
    repo_root: Path,
    tasks: list[TaskSpec],
    status_path: Path,
    config_path: Path,
    standalone: bool = False,
    max_chunks: int = 6,
    max_models: int = 8,
    max_repositories: int = 4,
    max_algorithms: int = 4,
) -> dict[str, Any]:
    baseline_config = _make_config(
        research_enabled=False,
        status_path=status_path,
        config_path=config_path,
        standalone=standalone,
        max_chunks=max_chunks,
        max_models=max_models,
        max_repositories=max_repositories,
        max_algorithms=max_algorithms,
    )
    candidate_config = _make_config(
        research_enabled=True,
        status_path=status_path,
        config_path=config_path,
        standalone=standalone,
        max_chunks=max_chunks,
        max_models=max_models,
        max_repositories=max_repositories,
        max_algorithms=max_algorithms,
    )
    baseline = _variant_report(
        name="baseline_without_research_library",
        research_enabled=False,
        config=baseline_config,
        repo_root=repo_root,
        tasks=tasks,
    )
    candidate = _variant_report(
        name="candidate_with_research_library",
        research_enabled=True,
        config=candidate_config,
        repo_root=repo_root,
        tasks=tasks,
    )
    delta = _delta(baseline, candidate)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "measurement_type": "context_packet_ab",
        "note": (
            "This deterministic A/B measures research-library context contribution. "
            "It does not claim live benchmark solve-rate lift without a live LLM eval run."
        ),
        "status_path": str(status_path),
        "config_path": str(config_path),
        "task_count": len(tasks),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B test research-library context contribution on representative benchmark prompts."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_RESEARCH_LIBRARY_CONFIG)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--refresh-status", action="store_true")
    parser.add_argument("--standalone-context", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--max-models", type=int, default=8)
    parser.add_argument("--max-repositories", type=int, default=4)
    parser.add_argument("--max-algorithms", type=int, default=4)
    args = parser.parse_args()

    repo_root = args.root.resolve()
    status_path = args.status if args.status.is_absolute() else repo_root / args.status
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    if args.refresh_status or not status_path.exists():
        write_research_library_status(status_path, config_path=config_path, root=repo_root)
    tasks = _load_tasks(args.tasks)
    report = build_report(
        repo_root=repo_root,
        tasks=tasks,
        status_path=status_path,
        config_path=config_path,
        standalone=bool(args.standalone_context),
        max_chunks=args.max_chunks,
        max_models=args.max_models,
        max_repositories=args.max_repositories,
        max_algorithms=args.max_algorithms,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    delta = report["delta"]
    candidate = report["candidate"]["aggregate"]
    baseline = report["baseline"]["aggregate"]
    print(
        "research_library_ab "
        f"tasks={len(tasks)} "
        f"baseline_expected_hits={baseline['expected_signal_hit_count']} "
        f"candidate_expected_hits={candidate['expected_signal_hit_count']} "
        f"hit_rate_delta={delta['expected_signal_hit_rate']:.2f} "
        f"research_chunk_delta={delta['research_context_chunks']} "
        f"llm_visible_research_chunk_delta={delta['llm_visible_research_context_chunks']} "
        f"research_evidence_delta={delta['research_evidence_count']} "
        f"model_asset_delta={delta['model_asset_count']} "
        f"repo_match_delta={delta['repository_match_count']} "
        f"algorithm_match_delta={delta['algorithm_match_count']} "
        f"helps_context={str(delta['helps_context']).lower()} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
