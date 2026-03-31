from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import KernelConfig


def _classify_failure_reason(reason: object) -> str:
    normalized = str(reason).strip()
    if not normalized or normalized.lower() == "verification passed":
        return ""
    lowered = normalized.lower()
    if "timed out" in lowered:
        return "timeout"
    if "exit code" in lowered:
        return "command_failure"
    if "missing expected file" in lowered:
        return "missing_expected_file"
    if "missing expected output" in lowered:
        return "missing_expected_output"
    if "forbidden file present" in lowered:
        return "forbidden_file_present"
    if "unexpected file content" in lowered:
        return "unexpected_file_content"
    if "forbidden output present" in lowered:
        return "forbidden_output_present"
    if "policy terminated" in lowered:
        return "policy_terminated"
    if "repeated failed action" in lowered:
        return "repeated_failed_action"
    if normalized in {"no_state_progress", "state_regression", "setup_failed"}:
        return normalized
    return "other"


def _normalize_summary_from_steps(document: dict[str, object]) -> dict[str, object]:
    normalized = dict(document)
    steps = normalized.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return normalized
    summary = dict(normalized.get("summary", {})) if isinstance(normalized.get("summary", {}), dict) else {}
    executed_commands: list[str] = []
    failure_types: set[str] = set()
    failure_signals: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).strip() == "code_execute":
            command = str(step.get("content", "")).strip()
            if command:
                executed_commands.append(command)
        verification = step.get("verification", {})
        if isinstance(verification, dict):
            for reason in verification.get("reasons", []):
                classified = _classify_failure_reason(reason)
                if classified:
                    failure_types.add(classified)
        for signal in step.get("failure_signals", []):
            normalized_signal = str(signal).strip()
            if normalized_signal:
                failure_signals.add(normalized_signal)
    summary["executed_command_count"] = len(executed_commands)
    summary["executed_commands"] = executed_commands
    summary["failure_types"] = sorted(failure_types)
    summary["failure_signals"] = sorted(failure_signals)
    summary["transition_failures"] = sorted(
        signal for signal in failure_signals if signal in {"no_state_progress", "state_regression"}
    )
    normalized["summary"] = summary
    return normalized


def _parsed_episode_document(root: Path, path: Path) -> dict[str, object] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed = _normalize_summary_from_steps(parsed)
    task_id = str(parsed.get("task_id", "")).strip()
    if not task_id:
        return None
    return annotate_episode_document(parsed, root=root, path=path)


def _iter_file_episode_documents(root: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        parsed = _parsed_episode_document(root, path)
        if parsed is not None:
            documents.append(parsed)
    return documents


def _load_file_episode_document(root: Path, task_id: str) -> dict[str, object]:
    direct_path = root / f"{task_id}.json"
    if direct_path.exists():
        parsed = _parsed_episode_document(root, direct_path)
        if parsed is not None:
            return parsed
    for path in sorted(root.rglob(f"{task_id}.json")):
        if not path.is_file():
            continue
        parsed = _parsed_episode_document(root, path)
        if parsed is not None:
            return parsed
    raise FileNotFoundError(f"episode document not found for task_id={task_id}")


def episode_storage_metadata(root: Path, path: Path) -> dict[str, object]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        relative_path = resolved_path
    parts = relative_path.parts
    phase = "primary"
    source_group = ""
    if len(parts) > 1:
        phase = str(parts[0]).strip() or "primary"
        source_group = str(Path(*parts[:-1])).strip()
    return {
        "relative_path": relative_path.as_posix(),
        "phase": phase,
        "source_group": source_group,
        "depth": max(0, len(parts) - 1),
        "is_generated": phase.startswith("generated_"),
    }


def annotate_episode_document(document: dict[str, object], *, root: Path, path: Path) -> dict[str, object]:
    annotated = dict(document)
    storage = episode_storage_metadata(root, path)
    annotated["episode_storage"] = storage
    task_metadata = dict(annotated.get("task_metadata", {})) if isinstance(annotated.get("task_metadata", {}), dict) else {}
    task_metadata.setdefault("episode_phase", str(storage.get("phase", "")))
    task_metadata.setdefault("episode_source_group", str(storage.get("source_group", "")))
    task_metadata.setdefault("episode_relative_path", str(storage.get("relative_path", "")))
    annotated["task_metadata"] = task_metadata
    return annotated


def _resolved_storage_config(root: Path, config: KernelConfig | None = None) -> KernelConfig | None:
    if config is not None:
        return config if config.uses_sqlite_storage() else None
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and root.resolve() == default_config.trajectories_root.resolve():
            return default_config
    except OSError:
        return None
    return None


def iter_episode_documents(root: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    storage_config = _resolved_storage_config(root, config=config)
    file_documents = _iter_file_episode_documents(root)
    if storage_config is not None:
        documents: list[dict[str, object]] = []
        seen_task_ids: set[str] = set()
        for payload in storage_config.sqlite_store().iter_episode_documents():
            if not isinstance(payload, dict):
                continue
            normalized = _normalize_summary_from_steps(payload)
            task_id = str(normalized.get("task_id", "")).strip()
            if not task_id:
                continue
            seen_task_ids.add(task_id)
            documents.append(normalized)
        for document in file_documents:
            task_id = str(document.get("task_id", "")).strip()
            if not task_id or task_id in seen_task_ids:
                continue
            documents.append(document)
        return documents
    return file_documents


def load_episode_document(root: Path, task_id: str, *, config: KernelConfig | None = None) -> dict[str, object]:
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        raise FileNotFoundError("episode task_id must not be empty")
    storage_config = _resolved_storage_config(root, config=config)
    if storage_config is not None:
        try:
            payload = storage_config.sqlite_store().load_episode_document(normalized_task_id)
        except FileNotFoundError:
            pass
        else:
            return _normalize_summary_from_steps(payload)
    return _load_file_episode_document(root, normalized_task_id)
