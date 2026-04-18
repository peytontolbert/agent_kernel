from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .config import KernelConfig
from .ops.episode_store import iter_episode_documents, load_episode_document
from .ops.episode_store import episode_storage_metadata
from .extensions.extractors import render_episode_document
from .learning_compiler import load_learning_candidates
from .schemas import EpisodeRecord


class GraphMemory:
    def __init__(self, episode_memory: "EpisodeMemory") -> None:
        self.episode_memory = episode_memory

    def summarize(self, task_id: str = "") -> dict[str, object]:
        return self.episode_memory.graph_summary(task_id)

    def recall(
        self,
        *,
        task_id: str = "",
        benchmark_family: str = "",
        changed_paths: list[str] | None = None,
        verifier_obligations: list[str] | None = None,
        failure_signals: list[str] | None = None,
        require_success: bool | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        return self.episode_memory.semantic_recall(
            task_id=task_id,
            benchmark_family=benchmark_family,
            changed_paths=changed_paths,
            verifier_obligations=verifier_obligations,
            failure_signals=failure_signals,
            require_success=require_success,
            limit=limit,
        )

    def prototype_recall(
        self,
        *,
        task_id: str = "",
        benchmark_family: str = "",
        changed_paths: list[str] | None = None,
        verifier_obligations: list[str] | None = None,
        failure_signals: list[str] | None = None,
        require_success: bool | None = True,
        limit: int = 4,
    ) -> list[dict[str, object]]:
        return self.episode_memory.semantic_prototype_recall(
            task_id=task_id,
            benchmark_family=benchmark_family,
            changed_paths=changed_paths,
            verifier_obligations=verifier_obligations,
            failure_signals=failure_signals,
            require_success=require_success,
            limit=limit,
        )


class EpisodeMemory:
    def __init__(self, root: Path, *, config: KernelConfig | None = None) -> None:
        self.root = root
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, episode: EpisodeRecord) -> Path:
        path = self.root / f"{episode.task_id}.json"
        document = render_episode_document(episode)
        if self.config is not None and self.config.uses_sqlite_storage():
            storage_root = self.root
            try:
                trajectories_root = self.config.trajectories_root.resolve()
                if path.resolve().is_relative_to(trajectories_root):
                    storage_root = trajectories_root
            except (AttributeError, OSError, ValueError):
                storage_root = self.root
            storage = _episode_storage_metadata_for_sqlite(storage_root, episode, fallback_path=path)
            self.config.sqlite_store().upsert_episode_document(payload=document, storage=storage)
            if self.config.storage_write_episode_exports:
                path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            return path
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    def load(self, task_id: str) -> dict:
        return load_episode_document(self.root, task_id, config=self.config)

    def list_documents(self) -> list[dict]:
        documents = iter_episode_documents(self.root, config=self.config)
        synthetic = _synthetic_learning_documents(self.root, config=self.config, existing_task_ids={
            str(document.get("task_id", "")).strip() for document in documents
        })
        return [*documents, *synthetic]

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
        retrieval_backed_successes = 0
        retrieval_influenced_successes = 0
        trusted_retrieval_successes = 0
        retrieval_backed_command_counts: dict[str, int] = {}
        trusted_retrieval_command_counts: dict[str, int] = {}
        trusted_retrieval_procedure_counts: dict[tuple[str, ...], int] = {}
        verifier_obligation_counts: dict[str, int] = {}
        changed_path_counts: dict[str, int] = {}
        edit_patch_path_counts: dict[str, int] = {}
        recovery_command_counts: dict[str, int] = {}
        semantic_episodes: list[dict[str, object]] = []
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
                summary_failure_types = [
                    str(failure_type).strip()
                    for failure_type in summary.get("failure_types", [])
                    if str(failure_type).strip()
                ]
                if not summary_failure_types:
                    has_episode_steps = isinstance(document.get("steps", []), list) and bool(document.get("steps", []))
                    has_executed_commands = bool(summary.get("executed_commands", []))
                    if has_episode_steps or has_executed_commands:
                        summary_failure_types = ["other"]
                for failure_type in summary_failure_types:
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
                retrieval_summary = _document_retrieval_summary(document)
                if bool(document.get("success", False)) and retrieval_summary["retrieval_backed"]:
                    retrieval_backed_successes += 1
                    if retrieval_summary["retrieval_influenced_steps"] > 0:
                        retrieval_influenced_successes += 1
                    if retrieval_summary["trusted_retrieval_steps"] > 0:
                        trusted_retrieval_successes += 1
                    for command in retrieval_summary["retrieval_backed_commands"]:
                        retrieval_backed_command_counts[command] = (
                            retrieval_backed_command_counts.get(command, 0) + 1
                        )
                    if retrieval_summary["trusted_retrieval_steps"] > 0:
                        for command in retrieval_summary["retrieval_backed_commands"]:
                            trusted_retrieval_command_counts[command] = (
                                trusted_retrieval_command_counts.get(command, 0) + 1
                            )
                for procedure in _document_trusted_retrieval_procedures(document):
                    key = tuple(procedure)
                    if key:
                        trusted_retrieval_procedure_counts[key] = trusted_retrieval_procedure_counts.get(key, 0) + 1
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
                verifier_obligations = _document_verifier_obligation_texts(document)
                for obligation in verifier_obligations:
                    verifier_obligation_counts[obligation] = verifier_obligation_counts.get(obligation, 0) + 1
                changed_paths = _document_changed_paths(document)
                for changed_path in changed_paths:
                    changed_path_counts[changed_path] = changed_path_counts.get(changed_path, 0) + 1
                edit_patches = _document_edit_patches(document)
                for patch in edit_patches:
                    edit_path = str(patch.get("path", "")).strip()
                    if edit_path:
                        edit_patch_path_counts[edit_path] = edit_patch_path_counts.get(edit_path, 0) + 1
                recovery_traces = _document_recovery_traces(document)
                for trace in recovery_traces:
                    recovery_command = str(trace.get("recovery_command", "")).strip()
                    if recovery_command:
                        recovery_command_counts[recovery_command] = recovery_command_counts.get(recovery_command, 0) + 1
                semantic_episode = _document_semantic_episode(document)
                if semantic_episode is not None:
                    semantic_episodes.append(semantic_episode)
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
            "retrieval_backed_successes": retrieval_backed_successes,
            "retrieval_influenced_successes": retrieval_influenced_successes,
            "trusted_retrieval_successes": trusted_retrieval_successes,
            "retrieval_backed_command_counts": _sorted_count_mapping(retrieval_backed_command_counts),
            "trusted_retrieval_command_counts": _sorted_count_mapping(trusted_retrieval_command_counts),
            "trusted_retrieval_procedures": _sorted_procedure_counts(trusted_retrieval_procedure_counts),
            "verifier_obligation_counts": _sorted_count_mapping(verifier_obligation_counts),
            "changed_path_counts": _sorted_count_mapping(changed_path_counts),
            "edit_patch_path_counts": _sorted_count_mapping(edit_patch_path_counts),
            "recovery_command_counts": _sorted_count_mapping(recovery_command_counts),
            "semantic_episodes": _sorted_semantic_episodes(semantic_episodes, task_id=task_id),
            "semantic_prototypes": _semantic_prototypes(semantic_episodes),
            "observed_environment_modes": observed_environment_modes,
            "related_tasks": neighbors,
            "neighbors": neighbors,
        }

    def semantic_recall(
        self,
        *,
        task_id: str = "",
        benchmark_family: str = "",
        changed_paths: list[str] | None = None,
        verifier_obligations: list[str] | None = None,
        failure_signals: list[str] | None = None,
        require_success: bool | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        documents = self.list_documents()
        candidate_episodes = [
            semantic_episode
            for semantic_episode in (_document_semantic_episode(document) for document in documents)
            if semantic_episode is not None
        ]
        if not candidate_episodes:
            return []

        raw_requested_paths = [str(value).strip() for value in changed_paths or [] if str(value).strip()]
        requested_paths = _normalized_semantic_strings(raw_requested_paths)
        requested_obligations = _normalized_semantic_strings(verifier_obligations or [])
        requested_failures = _normalized_semantic_strings(failure_signals or [])
        requested_family = str(benchmark_family).strip().lower()
        requested_task = str(task_id).strip()

        scored: list[tuple[int, dict[str, object]]] = []
        for episode in candidate_episodes:
            if require_success is not None and bool(episode.get("success", False)) is not require_success:
                continue
            episode_family = str(episode.get("benchmark_family", "bounded")).strip().lower()
            if requested_family and episode_family != requested_family:
                continue
            score = _semantic_episode_query_score(
                episode,
                task_id=requested_task,
                changed_paths=requested_paths,
                verifier_obligations=requested_obligations,
                failure_signals=requested_failures,
            )
            if requested_task or requested_paths or requested_obligations or requested_failures or requested_family:
                if score <= 0:
                    continue
            scored.append((score, episode))

        return [
            episode
            for _, episode in sorted(
                scored,
                key=lambda item: (
                    -item[0],
                    -int(bool(item[1].get("success", False))),
                    str(item[1].get("task_id", "")).strip(),
                ),
            )[: max(0, int(limit))]
        ]

    def semantic_prototype_recall(
        self,
        *,
        task_id: str = "",
        benchmark_family: str = "",
        changed_paths: list[str] | None = None,
        verifier_obligations: list[str] | None = None,
        failure_signals: list[str] | None = None,
        require_success: bool | None = True,
        limit: int = 4,
    ) -> list[dict[str, object]]:
        del task_id
        documents = self.list_documents()
        candidate_episodes = [
            semantic_episode
            for semantic_episode in (_document_semantic_episode(document) for document in documents)
            if semantic_episode is not None
        ]
        if not candidate_episodes:
            return []
        raw_requested_paths = [str(value).strip() for value in changed_paths or [] if str(value).strip()]
        requested_paths = _normalized_semantic_strings(raw_requested_paths)
        requested_obligations = _normalized_semantic_strings(verifier_obligations or [])
        requested_failures = _normalized_semantic_strings(failure_signals or [])
        requested_family = str(benchmark_family).strip().lower()
        scored: list[tuple[int, dict[str, object]]] = []
        for prototype in _semantic_prototypes(candidate_episodes):
            if require_success is True and int(prototype.get("success_count", 0) or 0) <= 0:
                continue
            if require_success is False and int(prototype.get("success_count", 0) or 0) > 0:
                continue
            prototype_family = str(prototype.get("benchmark_family", "bounded")).strip().lower()
            if requested_family and prototype_family != requested_family:
                continue
            score = _semantic_prototype_query_score(
                prototype,
                changed_paths=requested_paths,
                verifier_obligations=requested_obligations,
                failure_signals=requested_failures,
            )
            if requested_paths or requested_obligations or requested_failures or requested_family:
                if score <= 0:
                    continue
            scored.append((score, prototype))
        recalled: list[dict[str, object]] = []
        for _, prototype in sorted(
                scored,
                key=lambda item: (
                    -item[0],
                    -int(item[1].get("success_count", 0) or 0),
                    -int(item[1].get("episode_count", 0) or 0),
                    str(item[1].get("benchmark_family", "bounded")).strip(),
                ),
            )[: max(0, int(limit))]:
            payload = dict(prototype)
            if requested_paths:
                payload["instantiated_application_commands"] = _prototype_instantiated_application_commands(
                    prototype,
                    requested_paths=raw_requested_paths,
                )
            recalled.append(payload)
        return recalled


def _episode_storage_metadata_for_sqlite(
    root: Path,
    episode: EpisodeRecord,
    *,
    fallback_path: Path,
) -> dict[str, object]:
    task_metadata = dict(episode.task_metadata) if isinstance(episode.task_metadata, dict) else {}
    phase = str(task_metadata.get("episode_phase", "")).strip()
    source_group = str(task_metadata.get("episode_source_group", "")).strip()
    relative_path = str(task_metadata.get("episode_relative_path", "")).strip()
    if not phase:
        workspace = str(episode.workspace).strip()
        workspace_parts = PurePosixPath(workspace).parts
        if workspace_parts[:2] and workspace_parts[0] == "workspace":
            if len(workspace_parts) > 2 and workspace_parts[1].startswith("generated_"):
                phase = str(workspace_parts[1]).strip()
                source_group = source_group or phase
                relative_path = relative_path or f"{phase}/{episode.task_id}.json"
            elif len(workspace_parts) >= 2:
                phase = "primary"
                relative_path = relative_path or f"{episode.task_id}.json"
    if not phase:
        return episode_storage_metadata(root, fallback_path)
    normalized_relative_path = PurePosixPath(relative_path) if relative_path else PurePosixPath(f"{episode.task_id}.json")
    if not source_group and len(normalized_relative_path.parts) > 1:
        source_group = PurePosixPath(*normalized_relative_path.parts[:-1]).as_posix()
    return {
        "relative_path": normalized_relative_path.as_posix(),
        "phase": phase,
        "source_group": source_group,
        "depth": max(0, len(normalized_relative_path.parts) - 1),
        "is_generated": phase.startswith("generated_"),
    }


def _document_retrieval_summary(document: dict[str, object]) -> dict[str, object]:
    summary = document.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    selected_steps = _safe_int(summary.get("retrieval_selected_steps", 0))
    influenced_steps = _safe_int(summary.get("retrieval_influenced_steps", 0))
    trusted_steps = _safe_int(summary.get("trusted_retrieval_steps", 0))
    commands = [
        str(value).strip()
        for value in summary.get("retrieval_backed_commands", [])
        if str(value).strip()
    ]
    if (
        selected_steps > 0
        or influenced_steps > 0
        or trusted_steps > 0
        or commands
        or bool(summary.get("retrieval_backed", False))
    ):
        return {
            "retrieval_selected_steps": selected_steps,
            "retrieval_influenced_steps": influenced_steps,
            "trusted_retrieval_steps": trusted_steps,
            "retrieval_backed_commands": commands,
            "retrieval_backed": bool(
                summary.get("retrieval_backed", False) or commands or influenced_steps > 0 or trusted_steps > 0
            ),
        }

    selected_steps = 0
    influenced_steps = 0
    trusted_steps = 0
    commands = []
    for step in document.get("steps", []) if isinstance(document.get("steps", []), list) else []:
        if not isinstance(step, dict):
            continue
        selected_span_id = str(step.get("selected_retrieval_span_id", "")).strip()
        retrieval_influenced = bool(step.get("retrieval_influenced", False))
        trusted_retrieval = bool(step.get("trust_retrieval", False))
        if selected_span_id:
            selected_steps += 1
        if retrieval_influenced:
            influenced_steps += 1
        if trusted_retrieval:
            trusted_steps += 1
        verification = step.get("verification", {})
        if not isinstance(verification, dict) or not bool(verification.get("passed", False)):
            continue
        if not (selected_span_id or retrieval_influenced or trusted_retrieval):
            continue
        command = str(step.get("content", "")).strip()
        if command and command not in commands:
            commands.append(command)
    return {
        "retrieval_selected_steps": selected_steps,
        "retrieval_influenced_steps": influenced_steps,
        "trusted_retrieval_steps": trusted_steps,
        "retrieval_backed_commands": commands,
        "retrieval_backed": bool(commands or influenced_steps > 0 or trusted_steps > 0),
    }


def _document_semantic_episode(document: dict[str, object]) -> dict[str, object] | None:
    source_id = str(document.get("task_id", "")).strip()
    task_metadata = document.get("task_metadata", {})
    summary = document.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    verifier_obligations = _document_verifier_obligation_texts(document)
    changed_paths = _document_changed_paths(document)
    edit_patches = _document_edit_patches(document)
    recovery_traces = _document_recovery_traces(document)
    failure_signals = [
        str(value).strip()
        for value in summary.get("failure_signals", [])
        if str(value).strip()
    ]
    if not (verifier_obligations or changed_paths or recovery_traces or edit_patches or failure_signals):
        return None
    return {
        "task_id": source_id,
        "benchmark_family": str(task_metadata.get("benchmark_family", "bounded")).strip()
        if isinstance(task_metadata, dict)
        else "bounded",
        "memory_source": str(task_metadata.get("memory_source", "")).strip() if isinstance(task_metadata, dict) else "",
        "success": bool(document.get("success", False)),
        "verifier_obligations": verifier_obligations[:4],
        "changed_paths": changed_paths[:6],
        "failure_signals": failure_signals[:4],
        "command_sequence": _document_success_command_sequence(document)[:3],
        "edit_patches": [
            {
                "path": str(item.get("path", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "patch_summary": str(item.get("patch_summary", "")).strip(),
                "patch_excerpt": str(item.get("patch", "")).strip()[:160],
            }
            for item in edit_patches[:2]
            if str(item.get("path", "")).strip()
        ],
        "recovery_trace": recovery_traces[0] if recovery_traces else {},
    }


def _sorted_count_mapping(counts: dict[str, int]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    }


def _sorted_procedure_counts(counts: dict[tuple[str, ...], int]) -> list[dict[str, object]]:
    return [
        {"commands": list(commands), "count": int(count)}
        for commands, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), len(item[0]), list(item[0])),
        )[:4]
        if commands and count > 0
    ]


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _document_trusted_retrieval_procedures(document: dict[str, object]) -> list[list[str]]:
    summary = document.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    if not bool(document.get("success", summary.get("success", False))):
        return []
    retrieval_summary = _document_retrieval_summary(document)
    if retrieval_summary["trusted_retrieval_steps"] <= 0:
        return []

    procedures: list[list[str]] = []
    explicit_procedure = _document_explicit_procedure(document)
    if len(explicit_procedure) >= 2:
        procedures.append(explicit_procedure)

    command_sequence = _document_success_command_sequence(document)
    if len(command_sequence) >= 2 and command_sequence not in procedures:
        procedures.append(command_sequence)
    return procedures


def _document_explicit_procedure(document: dict[str, object]) -> list[str]:
    procedure = document.get("procedure", {})
    if not isinstance(procedure, dict):
        return []
    return [str(value).strip() for value in procedure.get("commands", []) if str(value).strip()]


def _document_success_command_sequence(document: dict[str, object]) -> list[str]:
    steps = document.get("steps", [])
    if not isinstance(steps, list):
        return []
    commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).strip() != "code_execute":
            continue
        command = str(step.get("content", "")).strip()
        if not command:
            continue
        verification = step.get("verification", {})
        if isinstance(verification, dict) and bool(verification.get("passed", False)):
            commands.append(command)
            continue
        command_result = step.get("command_result", {})
        if isinstance(command_result, dict) and not bool(command_result.get("timed_out", False)):
            try:
                exit_code = int(command_result.get("exit_code", 1))
            except (TypeError, ValueError):
                exit_code = 1
            if exit_code == 0:
                commands.append(command)
    return commands


def _document_verifier_obligation_texts(document: dict[str, object]) -> list[str]:
    summary = document.get("summary", {})
    if not isinstance(summary, dict):
        return []
    obligations = summary.get("verifier_obligations", [])
    texts: list[str] = []
    for item in obligations if isinstance(obligations, list) else []:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _document_changed_paths(document: dict[str, object]) -> list[str]:
    summary = document.get("summary", {})
    if isinstance(summary, dict):
        changed_paths = summary.get("changed_paths", [])
        if isinstance(changed_paths, list):
            ordered = [str(value).strip() for value in changed_paths if str(value).strip()]
            if ordered:
                return ordered
    changed: list[str] = []
    for fragment in document.get("fragments", []) if isinstance(document.get("fragments", []), list) else []:
        if not isinstance(fragment, dict):
            continue
        if str(fragment.get("kind", "")).strip() not in {"command_outcome", "recovery_trace"}:
            continue
        for value in fragment.get("changed_paths", fragment.get("recovered_changed_paths", [])):
            normalized = str(value).strip()
            if normalized and normalized not in changed:
                changed.append(normalized)
    return changed


def _document_recovery_traces(document: dict[str, object]) -> list[dict[str, object]]:
    summary = document.get("summary", {})
    if isinstance(summary, dict):
        traces = summary.get("recovery_traces", [])
        if isinstance(traces, list):
            normalized = [dict(item) for item in traces if isinstance(item, dict)]
            if normalized:
                return normalized
    traces: list[dict[str, object]] = []
    for fragment in document.get("fragments", []) if isinstance(document.get("fragments", []), list) else []:
        if not isinstance(fragment, dict):
            continue
        if str(fragment.get("kind", "")).strip() != "recovery_trace":
            continue
        traces.append(dict(fragment))
    return traces


def _document_edit_patches(document: dict[str, object]) -> list[dict[str, object]]:
    summary = document.get("summary", {})
    if isinstance(summary, dict):
        patches = summary.get("edit_patches", [])
        if isinstance(patches, list):
            normalized = [dict(item) for item in patches if isinstance(item, dict)]
            if normalized:
                return normalized
    patches: list[dict[str, object]] = []
    for fragment in document.get("fragments", []) if isinstance(document.get("fragments", []), list) else []:
        if not isinstance(fragment, dict):
            continue
        if str(fragment.get("kind", "")).strip() != "edit_patch":
            continue
        patches.append(dict(fragment))
    return patches


def _sorted_semantic_episodes(
    episodes: list[dict[str, object]],
    *,
    task_id: str = "",
) -> list[dict[str, object]]:
    if not episodes:
        return []

    def _episode_priority(payload: dict[str, object]) -> tuple[int, int, int, str]:
        current_task = str(payload.get("task_id", "")).strip()
        related_bonus = 1 if task_id and (current_task.startswith(f"{task_id}_") or task_id.startswith(f"{current_task}_")) else 0
        success_bonus = 1 if bool(payload.get("success", False)) else 0
        semantic_pressure = len(list(payload.get("verifier_obligations", []))) + len(list(payload.get("changed_paths", [])))
        return (-related_bonus, -success_bonus, -semantic_pressure, current_task)

    return [
        {
            "task_id": str(item.get("task_id", "")).strip(),
            "benchmark_family": str(item.get("benchmark_family", "bounded")).strip() or "bounded",
            "memory_source": str(item.get("memory_source", "")).strip(),
            "success": bool(item.get("success", False)),
            "verifier_obligations": [
                str(value).strip()
                for value in item.get("verifier_obligations", [])
                if str(value).strip()
            ][:4],
            "changed_paths": [
                str(value).strip()
                for value in item.get("changed_paths", [])
                if str(value).strip()
            ][:6],
            "failure_signals": [
                str(value).strip()
                for value in item.get("failure_signals", [])
                if str(value).strip()
            ][:4],
            "edit_patches": [
                {
                    "path": str(dict(patch).get("path", "")).strip(),
                    "status": str(dict(patch).get("status", "")).strip(),
                    "patch_summary": str(dict(patch).get("patch_summary", "")).strip(),
                    "patch_excerpt": str(dict(patch).get("patch_excerpt", dict(patch).get("patch", ""))).strip()[:160],
                }
                for patch in item.get("edit_patches", [])[:2]
                if isinstance(patch, dict) and str(dict(patch).get("path", "")).strip()
            ],
            "recovery_trace": dict(item.get("recovery_trace", {}))
            if isinstance(item.get("recovery_trace", {}), dict)
            else {},
        }
        for item in sorted(episodes, key=_episode_priority)[:6]
        if str(item.get("task_id", "")).strip()
    ]


def _normalized_semantic_strings(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _semantic_prototypes(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    prototypes: dict[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], dict[str, object]] = {}
    for episode in episodes:
        family = str(episode.get("benchmark_family", "bounded")).strip() or "bounded"
        changed_paths = tuple(
            _normalized_semantic_strings(
                [str(value).strip() for value in episode.get("changed_paths", []) if str(value).strip()]
            )[:3]
        )
        obligations = tuple(
            _normalized_semantic_strings(
                [str(value).strip() for value in episode.get("verifier_obligations", []) if str(value).strip()]
            )[:2]
        )
        failure_signals = tuple(
            _normalized_semantic_strings(
                [str(value).strip() for value in episode.get("failure_signals", []) if str(value).strip()]
            )[:2]
        )
        key = (family, changed_paths, obligations, failure_signals)
        prototype = prototypes.setdefault(
            key,
            {
                "benchmark_family": family,
                "changed_paths": list(changed_paths),
                "verifier_obligations": list(obligations),
                "failure_signals": list(failure_signals),
                "episode_count": 0,
                "success_count": 0,
                "memory_sources": {},
                "task_ids": [],
                "recovery_commands": {},
                "failed_commands": {},
                "command_sequences": {},
                "application_templates": {},
                "sequence_templates": {},
                "path_kinds": {},
                "path_extensions": {},
                "path_directories": {},
                "operation_counts": {},
            },
        )
        prototype["episode_count"] = int(prototype.get("episode_count", 0) or 0) + 1
        if bool(episode.get("success", False)):
            prototype["success_count"] = int(prototype.get("success_count", 0) or 0) + 1
        memory_source = str(episode.get("memory_source", "")).strip()
        if memory_source:
            source_counts = prototype.setdefault("memory_sources", {})
            if isinstance(source_counts, dict):
                source_counts[memory_source] = int(source_counts.get(memory_source, 0) or 0) + 1
        task_id = str(episode.get("task_id", "")).strip()
        if task_id:
            task_ids = prototype.setdefault("task_ids", [])
            if isinstance(task_ids, list) and task_id not in task_ids:
                task_ids.append(task_id)
        recovery_trace = episode.get("recovery_trace", {})
        recovery_trace = dict(recovery_trace) if isinstance(recovery_trace, dict) else {}
        recovery_command = str(recovery_trace.get("recovery_command", "")).strip()
        if recovery_command:
            recovery_commands = prototype.setdefault("recovery_commands", {})
            if isinstance(recovery_commands, dict):
                recovery_commands[recovery_command] = int(recovery_commands.get(recovery_command, 0) or 0) + 1
        failed_command = str(recovery_trace.get("failed_command", "")).strip()
        if failed_command:
            failed_commands = prototype.setdefault("failed_commands", {})
            if isinstance(failed_commands, dict):
                failed_commands[failed_command] = int(failed_commands.get(failed_command, 0) or 0) + 1
        command_sequence = [
            str(value).strip()
            for value in episode.get("command_sequence", [])
            if str(value).strip()
        ]
        if command_sequence:
            sequence_key = " || ".join(command_sequence[:3])
            command_sequences = prototype.setdefault("command_sequences", {})
            if isinstance(command_sequences, dict):
                command_sequences[sequence_key] = int(command_sequences.get(sequence_key, 0) or 0) + 1
        primary_target_path = _prototype_primary_target_path(episode)
        if primary_target_path:
            path_kinds = prototype.setdefault("path_kinds", {})
            if isinstance(path_kinds, dict):
                kind = _prototype_path_kind(primary_target_path)
                if kind:
                    path_kinds[kind] = int(path_kinds.get(kind, 0) or 0) + 1
            path_extensions = prototype.setdefault("path_extensions", {})
            if isinstance(path_extensions, dict):
                extension = PurePosixPath(primary_target_path).suffix.strip().lower()
                if extension:
                    path_extensions[extension] = int(path_extensions.get(extension, 0) or 0) + 1
            path_directories = prototype.setdefault("path_directories", {})
            if isinstance(path_directories, dict):
                parent = PurePosixPath(primary_target_path).parent.as_posix()
                if parent and parent != ".":
                    path_directories[parent] = int(path_directories.get(parent, 0) or 0) + 1
        for patch in episode.get("edit_patches", []):
            if not isinstance(patch, dict):
                continue
            status = str(patch.get("status", "")).strip().lower()
            if status:
                operation_counts = prototype.setdefault("operation_counts", {})
                if isinstance(operation_counts, dict):
                    operation_counts[status] = int(operation_counts.get(status, 0) or 0) + 1
        application_templates = prototype.setdefault("application_templates", {})
        if isinstance(application_templates, dict):
            for command in _prototype_application_commands(prototype):
                template = _generalize_command_template(command, primary_target_path)
                if template:
                    application_templates[template] = int(application_templates.get(template, 0) or 0) + 1
        sequence_templates = prototype.setdefault("sequence_templates", {})
        if isinstance(sequence_templates, dict):
            for serialized, count in dict(prototype.get("command_sequences", {})).items():
                sequence_commands = [part.strip() for part in str(serialized).split(" || ") if part.strip()]
                if not sequence_commands:
                    continue
                template = " || ".join(
                    _generalize_command_template(command, primary_target_path) or command
                    for command in sequence_commands
                )
                if template:
                    sequence_templates[template] = int(sequence_templates.get(template, 0) or 0) + int(count or 0)

    return [
        {
            "benchmark_family": str(item.get("benchmark_family", "bounded")).strip() or "bounded",
            "changed_paths": [
                str(value).strip()
                for value in item.get("changed_paths", [])
                if str(value).strip()
            ][:3],
            "verifier_obligations": [
                str(value).strip()
                for value in item.get("verifier_obligations", [])
                if str(value).strip()
            ][:2],
            "failure_signals": [
                str(value).strip()
                for value in item.get("failure_signals", [])
                if str(value).strip()
            ][:2],
            "episode_count": int(item.get("episode_count", 0) or 0),
            "success_count": int(item.get("success_count", 0) or 0),
            "memory_sources": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("memory_sources", {})).items()
                    if str(key).strip()
                }
            ),
            "task_ids": [
                str(value).strip()
                for value in item.get("task_ids", [])
                if str(value).strip()
            ][:4],
            "recovery_commands": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("recovery_commands", {})).items()
                    if str(key).strip()
                }
            ),
            "failed_commands": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("failed_commands", {})).items()
                    if str(key).strip()
                }
            ),
            "command_sequences": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("command_sequences", {})).items()
                    if str(key).strip()
                }
            ),
            "application_commands": _prototype_application_commands(item),
            "application_command_templates": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("application_templates", {})).items()
                    if str(key).strip()
                }
            ),
            "command_sequence_templates": _sorted_count_mapping(
                {
                    str(key).strip(): int(value or 0)
                    for key, value in dict(item.get("sequence_templates", {})).items()
                    if str(key).strip()
                }
            ),
            "transform_semantics": _prototype_transform_semantics(item),
        }
        for item in sorted(
            prototypes.values(),
            key=lambda payload: (
                -int(payload.get("success_count", 0) or 0),
                -int(payload.get("episode_count", 0) or 0),
                str(payload.get("benchmark_family", "bounded")).strip(),
                list(payload.get("changed_paths", [])),
            ),
        )[:6]
    ]


def _semantic_episode_query_score(
    episode: dict[str, object],
    *,
    task_id: str,
    changed_paths: list[str],
    verifier_obligations: list[str],
    failure_signals: list[str],
) -> int:
    score = 0
    current_task = str(episode.get("task_id", "")).strip()
    if task_id:
        if current_task == task_id:
            score += 8
        elif current_task.startswith(f"{task_id}_") or task_id.startswith(f"{current_task}_"):
            score += 4
    episode_paths = _normalized_semantic_strings(
        [str(value).strip() for value in episode.get("changed_paths", []) if str(value).strip()]
    )
    episode_obligations = _normalized_semantic_strings(
        [str(value).strip() for value in episode.get("verifier_obligations", []) if str(value).strip()]
    )
    episode_failures = _normalized_semantic_strings(
        [str(value).strip() for value in episode.get("failure_signals", []) if str(value).strip()]
    )
    for path in changed_paths:
        if any(candidate == path or candidate.endswith(path) or path.endswith(candidate) for candidate in episode_paths):
            score += 4
    for obligation in verifier_obligations:
        if any(
            obligation == candidate
            or obligation in candidate
            or candidate in obligation
            for candidate in episode_obligations
        ):
            score += 4
    for failure_signal in failure_signals:
        if failure_signal in episode_failures:
            score += 3
    if bool(episode.get("success", False)):
        score += 1
    score += min(3, len(episode_paths) + len(episode_obligations))
    return score


def _semantic_prototype_query_score(
    prototype: dict[str, object],
    *,
    changed_paths: list[str],
    verifier_obligations: list[str],
    failure_signals: list[str],
) -> int:
    score = 0
    prototype_paths = _normalized_semantic_strings(
        [str(value).strip() for value in prototype.get("changed_paths", []) if str(value).strip()]
    )
    prototype_obligations = _normalized_semantic_strings(
        [str(value).strip() for value in prototype.get("verifier_obligations", []) if str(value).strip()]
    )
    prototype_failures = _normalized_semantic_strings(
        [str(value).strip() for value in prototype.get("failure_signals", []) if str(value).strip()]
    )
    for path in changed_paths:
        if any(candidate == path or candidate.endswith(path) or path.endswith(candidate) for candidate in prototype_paths):
            score += 5
    score += _prototype_transform_query_score(prototype, changed_paths)
    for obligation in verifier_obligations:
        if any(
            obligation == candidate
            or obligation in candidate
            or candidate in obligation
            for candidate in prototype_obligations
        ):
            score += 4
    for failure_signal in failure_signals:
        if failure_signal in prototype_failures:
            score += 3
    score += min(4, int(prototype.get("success_count", 0) or 0))
    score += min(2, int(prototype.get("episode_count", 0) or 0))
    return score


def _prototype_application_commands(prototype: dict[str, object]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for source in ("recovery_commands", "command_sequences"):
        payload = prototype.get(source, {})
        payload = dict(payload) if isinstance(payload, dict) else {}
        for key, count in payload.items():
            normalized = str(key).strip()
            if not normalized:
                continue
            if source == "command_sequences":
                commands = [part.strip() for part in normalized.split(" || ") if part.strip()]
                if not commands:
                    continue
                normalized = commands[-1]
            ranked.append((int(count or 0), normalized))
    seen: list[str] = []
    for _, command in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if command not in seen:
            seen.append(command)
    return seen[:4]


def _prototype_primary_target_path(payload: dict[str, object]) -> str:
    for value in payload.get("changed_paths", []):
        normalized = str(value).strip()
        if normalized:
            return normalized
    for patch in payload.get("edit_patches", []):
        if not isinstance(patch, dict):
            continue
        normalized = str(patch.get("path", "")).strip()
        if normalized:
            return normalized
    return ""


def _prototype_path_kind(path: str) -> str:
    normalized = str(path).strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("reports/") or "/reports/" in normalized or "report" in normalized:
        return "report"
    if normalized.startswith("generated/") or "/generated/" in normalized:
        return "generated"
    if normalized.startswith("tests/") or normalized.endswith("_test.py") or "/tests/" in normalized:
        return "test"
    if normalized.startswith("src/") or "/src/" in normalized:
        return "source"
    if normalized.startswith("docs/") or "/docs/" in normalized:
        return "docs"
    if normalized.endswith(".json"):
        return "json"
    if normalized.endswith(".md"):
        return "markdown"
    if normalized.endswith(".py"):
        return "python"
    return "file"


def _path_role_bindings(path: str) -> dict[str, str]:
    target_path = str(path).strip()
    if not target_path:
        return {}
    pure = PurePosixPath(target_path)
    parent = pure.parent.as_posix()
    parent = "" if parent == "." else parent
    stem = pure.stem.strip()
    suffix = pure.suffix.strip()
    return {
        "target_path": target_path,
        "target_dir": parent,
        "target_file": pure.name,
        "target_stem": stem,
        "target_ext": suffix,
    }


def _generalize_command_template(command: str, target_path: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    bindings = _path_role_bindings(target_path)
    if not bindings:
        return normalized
    template = normalized
    for key, placeholder in (
        ("target_path", "{target_path}"),
        ("target_dir", "{target_dir}"),
        ("target_file", "{target_file}"),
    ):
        value = str(bindings.get(key, "")).strip()
        if value and value in template:
            template = template.replace(value, placeholder)
    return template


def _instantiate_command_template(template: str, target_path: str) -> str:
    normalized = str(template).strip()
    if not normalized:
        return ""
    bindings = _path_role_bindings(target_path)
    if not bindings:
        return normalized
    instantiated = normalized
    for key, placeholder in (
        ("target_path", "{target_path}"),
        ("target_dir", "{target_dir}"),
        ("target_file", "{target_file}"),
        ("target_stem", "{target_stem}"),
        ("target_ext", "{target_ext}"),
    ):
        value = str(bindings.get(key, "")).strip()
        if placeholder in instantiated and value:
            instantiated = instantiated.replace(placeholder, value)
    return instantiated


def _prototype_instantiated_application_commands(
    prototype: dict[str, object],
    *,
    requested_paths: list[str],
) -> list[str]:
    target_path = next((str(path).strip() for path in requested_paths if str(path).strip()), "")
    if not target_path:
        return []
    templates = [
        str(value).strip()
        for value in dict(prototype.get("application_command_templates", {})).keys()
        if str(value).strip()
    ]
    commands: list[str] = []
    for template in templates:
        instantiated = _instantiate_command_template(template, target_path)
        if instantiated and instantiated not in commands:
            commands.append(instantiated)
    for command in prototype.get("application_commands", []):
        normalized = str(command).strip()
        if normalized and normalized not in commands:
            commands.append(normalized)
    return commands[:4]


def _prototype_transform_semantics(prototype: dict[str, object]) -> dict[str, object]:
    return {
        "target_kinds": _sorted_count_mapping(
            {
                str(key).strip(): int(value or 0)
                for key, value in dict(prototype.get("path_kinds", {})).items()
                if str(key).strip()
            }
        ),
        "target_extensions": _sorted_count_mapping(
            {
                str(key).strip(): int(value or 0)
                for key, value in dict(prototype.get("path_extensions", {})).items()
                if str(key).strip()
            }
        ),
        "target_directories": _sorted_count_mapping(
            {
                str(key).strip(): int(value or 0)
                for key, value in dict(prototype.get("path_directories", {})).items()
                if str(key).strip()
            }
        ),
        "operation_counts": _sorted_count_mapping(
            {
                str(key).strip(): int(value or 0)
                for key, value in dict(prototype.get("operation_counts", {})).items()
                if str(key).strip()
            }
        ),
    }


def _prototype_transform_query_score(
    prototype: dict[str, object],
    changed_paths: list[str],
) -> int:
    semantics = prototype.get("transform_semantics", {})
    semantics = dict(semantics) if isinstance(semantics, dict) else {}
    kinds = {
        str(key).strip()
        for key in dict(semantics.get("target_kinds", {})).keys()
        if str(key).strip()
    }
    extensions = {
        str(key).strip()
        for key in dict(semantics.get("target_extensions", {})).keys()
        if str(key).strip()
    }
    directories = {
        str(key).strip()
        for key in dict(semantics.get("target_directories", {})).keys()
        if str(key).strip()
    }
    score = 0
    for path in changed_paths:
        normalized = str(path).strip()
        if not normalized:
            continue
        pure = PurePosixPath(normalized)
        extension = pure.suffix.strip().lower()
        directory = pure.parent.as_posix()
        directory = "" if directory == "." else directory
        kind = _prototype_path_kind(normalized)
        if kind and kind in kinds:
            score += 2
        if extension and extension in extensions:
            score += 2
        if directory and directory in directories:
            score += 1
    return score


def _learning_artifacts_path_for_memory(root: Path, *, config: KernelConfig | None = None) -> Path:
    if config is not None:
        path = config.learning_artifacts_path
        if path.is_absolute():
            return path
        try:
            if root.resolve() == config.trajectories_root.resolve():
                parts = path.parts
                if parts[:1] == ("trajectories",):
                    remainder = Path(*parts[1:]) if len(parts) > 1 else Path(path.name)
                    return root.parent / remainder
                return root.parent / path
        except OSError:
            pass
        return path
    try:
        default_config = KernelConfig()
        if root.resolve() == default_config.trajectories_root.resolve():
            return default_config.learning_artifacts_path
    except OSError:
        pass
    return root.parent / "learning" / "run_learning_artifacts.json"


def _synthetic_learning_documents(
    root: Path,
    *,
    config: KernelConfig | None = None,
    existing_task_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    path = _learning_artifacts_path_for_memory(root, config=config)
    if not path.exists():
        return []
    seen_task_ids = {task_id for task_id in (existing_task_ids or set()) if task_id}
    documents: list[dict[str, object]] = []
    for candidate in load_learning_candidates(path, config=config):
        document = _synthetic_document_from_learning_candidate(candidate)
        if document is None:
            continue
        task_id = str(document.get("task_id", "")).strip()
        if not task_id or task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        documents.append(document)
    return documents


def _synthetic_document_from_learning_candidate(candidate: dict[str, object]) -> dict[str, object] | None:
    artifact_kind = str(candidate.get("artifact_kind", "")).strip()
    source_task_id = str(candidate.get("source_task_id", "")).strip()
    if artifact_kind not in {"recovery_case", "failure_case", "negative_command_pattern"} or not source_task_id:
        return None
    task_contract = dict(candidate.get("task_contract", {})) if isinstance(candidate.get("task_contract", {}), dict) else {}
    task_metadata = dict(candidate.get("task_metadata", {})) if isinstance(candidate.get("task_metadata", {}), dict) else {}
    benchmark_family = str(candidate.get("benchmark_family", task_metadata.get("benchmark_family", "bounded"))).strip() or "bounded"
    memory_source = str(candidate.get("memory_source", task_metadata.get("memory_source", "learning_artifact"))).strip() or "learning_artifact"
    failure_types = [str(value).strip() for value in candidate.get("failure_types", []) if str(value).strip()]
    transition_failures = [str(value).strip() for value in candidate.get("transition_failures", []) if str(value).strip()]
    failure_signals = [*failure_types, *[value for value in transition_failures if value not in failure_types]]
    commands: list[str] = []
    if artifact_kind == "recovery_case":
        commands = [str(value).strip() for value in candidate.get("recovery_commands", []) if str(value).strip()]
        task_id = source_task_id
        success = bool(candidate.get("success", False)) and bool(commands)
    elif artifact_kind == "failure_case":
        commands = [str(value).strip() for value in candidate.get("executed_commands", []) if str(value).strip()]
        task_id = f"{source_task_id}__failure_memory"
        success = False
    else:
        command = str(candidate.get("command", "")).strip()
        commands = [command] if command else []
        task_id = f"{source_task_id}__negative_command_pattern"
        success = False
    if not commands and artifact_kind == "recovery_case":
        return None
    fragments: list[dict[str, object]] = []
    if success:
        fragments.extend(
            {
                "kind": "command",
                "task_id": task_id,
                "step_index": index,
                "command": command,
                "passed": True,
            }
            for index, command in enumerate(commands, start=1)
        )
    verification_reasons = [
        str(value).strip()
        for value in candidate.get("verification_reasons", [])
        if str(value).strip()
    ]
    fragments.extend(
        {
            "kind": "failure",
            "task_id": task_id,
            "step_index": 0,
            "reason": reason,
            "failure_types": failure_types,
            "failure_signals": failure_signals,
        }
        for reason in verification_reasons
    )
    episode_storage = {
        "relative_path": f"learning/{task_id}.json",
        "phase": "learning_artifacts",
        "source_group": "learning_artifacts",
        "depth": 1,
        "is_generated": False,
    }
    task_metadata.setdefault("benchmark_family", benchmark_family)
    task_metadata.setdefault("memory_source", memory_source)
    task_metadata.setdefault("episode_phase", "learning_artifacts")
    task_metadata.setdefault("episode_source_group", "learning_artifacts")
    task_metadata.setdefault("episode_relative_path", episode_storage["relative_path"])
    if artifact_kind == "recovery_case":
        task_metadata.setdefault("curriculum_kind", "failure_recovery")
        parent_task = str(candidate.get("parent_task", "")).strip()
        if parent_task:
            task_metadata.setdefault("parent_task", parent_task)
    summary = {
        "task_id": task_id,
        "success": success,
        "termination_reason": str(candidate.get("termination_reason", "success" if success else "learning_artifact")).strip(),
        "step_count": len(commands),
        "executed_command_count": len(commands),
        "executed_commands": commands,
        "execution_source_summary": {
            "decoder_generated": 0,
            "llm_generated": 0,
            "synthetic_plan": 0,
            "deterministic_or_other": len(commands),
            "total_executed_commands": len(commands),
        },
        "failure_types": failure_types,
        "failure_signals": failure_signals,
        "transition_failures": transition_failures,
        "state_progress_gain_steps": 0,
        "state_regression_steps": 0,
        "final_completion_ratio": 1.0 if success else 0.0,
        "net_state_progress_delta": 0.0,
        "final_action": "code_execute" if commands else "",
        "environment_violation_counts": {},
        "environment_violation_flags": [],
        "environment_alignment_failures": [],
        "environment_snapshot": {},
    }
    return {
        "task_id": task_id,
        "success": success,
        "task_metadata": task_metadata,
        "task_contract": task_contract,
        "summary": summary,
        "fragments": fragments,
        "episode_storage": episode_storage,
    }
