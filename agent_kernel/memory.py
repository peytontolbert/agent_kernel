from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .config import KernelConfig
from .episode_store import iter_episode_documents, load_episode_document
from .episode_store import episode_storage_metadata
from .extractors import render_episode_document
from .learning_compiler import load_learning_candidates
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
            "observed_environment_modes": observed_environment_modes,
            "related_tasks": neighbors,
            "neighbors": neighbors,
        }


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


def _learning_artifacts_path_for_memory(root: Path, *, config: KernelConfig | None = None) -> Path:
    if config is not None:
        return config.learning_artifacts_path
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
