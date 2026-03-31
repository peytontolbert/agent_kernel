from __future__ import annotations

import json
from pathlib import Path

from .config import KernelConfig
from .episode_store import iter_episode_documents, load_episode_document
from .extractors import render_episode_document
from .schemas import EpisodeRecord


class GraphMemory:
    def __init__(self, episode_memory: "EpisodeMemory") -> None:
        self.episode_memory = episode_memory

    def summarize(self, task_id: str = "") -> dict[str, object]:
        return self.episode_memory.graph_summary(task_id)


class EpisodeMemory:
    def __init__(self, root: Path, *, config: KernelConfig | None = None) -> None:
        self.root = root
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, episode: EpisodeRecord) -> Path:
        path = self.root / f"{episode.task_id}.json"
        document = render_episode_document(episode)
        if self.config is not None and self.config.uses_sqlite_storage():
            storage = {
                "relative_path": path.relative_to(self.root).as_posix(),
                "phase": "primary",
                "source_group": "",
                "depth": 0,
                "is_generated": False,
            }
            self.config.sqlite_store().upsert_episode_document(payload=document, storage=storage)
            if self.config.storage_write_episode_exports:
                path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            return path
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    def load(self, task_id: str) -> dict:
        return load_episode_document(self.root, task_id, config=self.config)

    def list_documents(self) -> list[dict]:
        return iter_episode_documents(self.root, config=self.config)

    def graph_summary(self, task_id: str = "") -> dict[str, object]:
        documents = self.list_documents()
        benchmark_families: dict[str, int] = {}
        memory_sources: dict[str, int] = {}
        source_tasks: set[str] = set()
        failure_types: dict[str, int] = {}
        failure_signals: dict[str, int] = {}
        memory_source_failure_signals: dict[str, dict[str, int]] = {}
        transition_failures: dict[str, int] = {}
        environment_violation_counts: dict[str, int] = {}
        environment_alignment_failures: dict[str, int] = {}
        observed_environment_modes: dict[str, dict[str, int]] = {
            "network_access_mode": {},
            "git_write_mode": {},
            "workspace_write_scope": {},
        }
        for document in documents:
            task_metadata = document.get("task_metadata", {})
            memory_source = ""
            if isinstance(task_metadata, dict):
                family = str(task_metadata.get("benchmark_family", "bounded"))
                benchmark_families[family] = benchmark_families.get(family, 0) + 1
                memory_source = str(task_metadata.get("memory_source", "")).strip()
                if memory_source:
                    memory_sources[memory_source] = memory_sources.get(memory_source, 0) + 1
            source_id = str(document.get("task_id", "")).strip()
            if source_id:
                source_tasks.add(source_id)
            summary = document.get("summary", {})
            if isinstance(summary, dict):
                for failure_type in summary.get("failure_types", []):
                    key = str(failure_type).strip()
                    if key:
                        failure_types[key] = failure_types.get(key, 0) + 1
                for failure_signal in summary.get("failure_signals", []):
                    key = str(failure_signal).strip()
                    if key:
                        failure_signals[key] = failure_signals.get(key, 0) + 1
                        if memory_source:
                            source_counts = memory_source_failure_signals.setdefault(memory_source, {})
                            source_counts[key] = source_counts.get(key, 0) + 1
                        if key in {"no_state_progress", "state_regression"}:
                            transition_failures[key] = transition_failures.get(key, 0) + 1
                for label, value in dict(summary.get("environment_violation_counts", {})).items():
                    key = str(label).strip()
                    if key:
                        try:
                            environment_violation_counts[key] = environment_violation_counts.get(key, 0) + int(value)
                        except (TypeError, ValueError):
                            continue
                for label in summary.get("environment_alignment_failures", []):
                    key = str(label).strip()
                    if key:
                        environment_alignment_failures[key] = environment_alignment_failures.get(key, 0) + 1
                snapshot = summary.get("environment_snapshot", {})
                if isinstance(snapshot, dict):
                    for field in observed_environment_modes:
                        value = str(snapshot.get(field, "")).strip().lower()
                        if value:
                            observed_environment_modes[field][value] = observed_environment_modes[field].get(value, 0) + 1
        neighbors = sorted(
            source
            for source in source_tasks
            if task_id and (source.startswith(f"{task_id}_") or task_id.startswith(f"{source}_"))
        )[:5]
        return {
            "document_count": len(documents),
            "benchmark_families": benchmark_families,
            "memory_sources": memory_sources,
            "failure_types": failure_types,
            "failure_type_counts": failure_types,
            "failure_signals": failure_signals,
            "memory_source_failure_signals": memory_source_failure_signals,
            "transition_failures": transition_failures,
            "environment_violation_counts": environment_violation_counts,
            "environment_alignment_failures": environment_alignment_failures,
            "observed_environment_modes": observed_environment_modes,
            "related_tasks": neighbors,
            "neighbors": neighbors,
        }
