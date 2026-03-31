from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .schemas import EpisodeRecord, StepRecord


def compiled_learning_artifacts_path(config: KernelConfig) -> Path:
    return config.learning_artifacts_path


def _resolved_storage_config(path: Path, config: KernelConfig | None = None) -> KernelConfig | None:
    if config is not None:
        return config if config.uses_sqlite_storage() else None
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and path.resolve() == default_config.learning_artifacts_path.resolve():
            return default_config
    except OSError:
        return None
    return None


def _load_json_learning_candidates(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    normalized: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        repaired = _normalize_learning_candidate(dict(candidate))
        if repaired is not None:
            normalized.append(repaired)
    return normalized


def load_learning_candidates(path: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    storage_config = _resolved_storage_config(path, config=config)
    file_candidates = _load_json_learning_candidates(path)
    if storage_config is not None:
        normalized: list[dict[str, object]] = []
        seen_candidate_ids: set[str] = set()
        for candidate in storage_config.sqlite_store().load_learning_candidates():
            if not isinstance(candidate, dict):
                continue
            repaired = _normalize_learning_candidate(dict(candidate))
            if repaired is not None:
                candidate_id = str(repaired.get("candidate_id", "")).strip()
                if candidate_id:
                    seen_candidate_ids.add(candidate_id)
                normalized.append(repaired)
        for candidate in file_candidates:
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            if candidate_id and candidate_id in seen_candidate_ids:
                continue
            normalized.append(candidate)
        return normalized
    return file_candidates


def compile_episode_learning_candidates(
    episode: EpisodeRecord,
    *,
    episode_storage: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    storage = dict(episode_storage or {})
    benchmark_family = str(episode.task_metadata.get("benchmark_family", "bounded")).strip() or "bounded"
    memory_source = str(episode.task_metadata.get("memory_source", "")).strip()
    commands = _executed_commands(episode.steps)
    failure_types = _failure_types(episode.steps, termination_reason=episode.termination_reason)
    failure_signals = sorted(
        {
            str(signal).strip()
            for step in episode.steps
            for signal in step.failure_signals
            if str(signal).strip()
        }
    )
    transition_failures = sorted(
        signal for signal in failure_signals if signal in {"no_state_progress", "state_regression"}
    )
    timestamp = datetime.now(UTC).isoformat()
    base = {
        "source_task_id": episode.task_id,
        "benchmark_family": benchmark_family,
        "task_contract": dict(episode.task_contract),
        "task_metadata": dict(episode.task_metadata),
        "memory_source": memory_source,
        "memory_sources": [memory_source] if memory_source else [],
        "episode_storage": storage,
        "termination_reason": episode.termination_reason,
        "compiled_at": timestamp,
        "support_count": 1,
    }
    candidates: list[dict[str, object]] = []
    if episode.success and commands:
        candidates.append(
            {
                **base,
                "candidate_id": f"learning:success_skill:{episode.task_id}",
                "artifact_kind": "success_skill_candidate",
                "procedure": {"commands": list(commands)},
                "applicable_tasks": [episode.task_id],
                "quality": 0.85 if len(commands) <= 2 else 0.8,
                "known_failure_types": failure_types,
            }
        )
    if str(episode.task_metadata.get("curriculum_kind", "")).strip() == "failure_recovery":
        candidates.append(
            {
                **base,
                "candidate_id": f"learning:recovery_case:{episode.task_id}",
                "artifact_kind": "recovery_case",
                "parent_task": str(episode.task_metadata.get("parent_task", "")).strip(),
                "failed_command": str(episode.task_metadata.get("failed_command", "")).strip(),
                "failure_types": failure_types,
                "recovery_commands": list(commands),
                "success": bool(episode.success),
            }
        )
    if not episode.success:
        candidates.append(
            {
                **base,
                "candidate_id": f"learning:failure_case:{episode.task_id}",
                "artifact_kind": "failure_case",
                "executed_commands": list(commands),
                "failure_types": failure_types,
                "failure_signals": failure_signals,
                "transition_failures": transition_failures,
            }
        )
        for command, reasons in _negative_command_patterns(episode.steps).items():
            candidates.append(
                {
                    **base,
                    "candidate_id": f"learning:negative_command:{episode.task_id}:{_short_hash(command)}",
                    "artifact_kind": "negative_command_pattern",
                    "command": command,
                    "failure_types": failure_types,
                    "failure_signals": failure_signals,
                    "verification_reasons": reasons,
                }
            )
    if failure_types or transition_failures:
        candidates.append(
            {
                **base,
                "candidate_id": f"learning:benchmark_gap:{episode.task_id}",
                "artifact_kind": "benchmark_gap",
                "failure_types": failure_types,
                "transition_failures": transition_failures,
                "executed_command_count": len(commands),
                "gap_kind": _benchmark_gap_kind(failure_types, transition_failures, command_count=len(commands)),
            }
        )
    if not candidates:
        candidates.append(
            {
                **base,
                "candidate_id": f"learning:failure_case:{episode.task_id}:empty",
                "artifact_kind": "failure_case",
                "executed_commands": [],
                "failure_types": failure_types,
                "failure_signals": failure_signals,
                "transition_failures": transition_failures,
            }
        )
    return candidates


def persist_episode_learning_candidates(
    episode: EpisodeRecord,
    *,
    config: KernelConfig,
    episode_storage: dict[str, object] | None = None,
) -> Path:
    path = compiled_learning_artifacts_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "run_learning_candidate_set",
        "lifecycle_state": "candidate",
        "retention_gate": {
            "require_improvement_cycle_promotion": True,
            "require_non_regression": True,
        },
        "candidates": [],
    }
    new_candidates = compile_episode_learning_candidates(episode, episode_storage=episode_storage)
    if config.uses_sqlite_storage():
        existing = config.sqlite_store().load_learning_candidates_by_ids(
            [str(candidate.get("candidate_id", "")).strip() for candidate in new_candidates]
        )
        merged_candidates = _merge_candidates(existing, new_candidates)
        config.sqlite_store().upsert_learning_candidates(merged_candidates)
        payload["candidates"] = config.sqlite_store().load_learning_candidates()
        if config.storage_write_learning_exports:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    existing = load_learning_candidates(path, config=config)
    merged = _merge_candidates(existing, new_candidates)
    payload["candidates"] = merged
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def matching_learning_candidates(
    path: Path,
    *,
    config: KernelConfig | None = None,
    task_id: str,
    source_task_id: str = "",
    benchmark_family: str = "",
    curriculum_kind: str = "",
    memory_source: str = "",
) -> list[dict[str, object]]:
    task_tokens = _task_match_tokens(
        str(task_id).strip(),
        str(source_task_id).strip(),
    )
    task_tokens.discard("")
    normalized_family = str(benchmark_family).strip()
    normalized_curriculum = str(curriculum_kind).strip()
    normalized_memory_source = str(memory_source).strip()
    ranked: list[tuple[tuple[int, int, str], dict[str, object]]] = []
    for candidate in load_learning_candidates(path, config=config):
        score = 0
        candidate_source = str(candidate.get("source_task_id", "")).strip()
        candidate_family = str(candidate.get("benchmark_family", "")).strip()
        candidate_memory_source = str(candidate.get("memory_source", "")).strip()
        candidate_memory_sources = _merged_unique_list(candidate.get("memory_sources"), [candidate_memory_source])
        candidate_meta = candidate.get("task_metadata", {})
        candidate_curriculum = ""
        if isinstance(candidate_meta, dict):
            candidate_curriculum = str(candidate_meta.get("curriculum_kind", "")).strip()
            if not candidate_memory_source:
                candidate_memory_source = str(candidate_meta.get("memory_source", "")).strip()
                candidate_memory_sources = _merged_unique_list(candidate_memory_sources, [candidate_memory_source])
        if candidate_source and candidate_source in task_tokens:
            score += 6
        applicable_tasks = _merged_unique_list(candidate.get("applicable_tasks"), [])
        if task_tokens and any(applicable_task in task_tokens for applicable_task in applicable_tasks):
            score += 5
        if normalized_family and candidate_family == normalized_family:
            score += 2
        if normalized_curriculum and candidate_curriculum == normalized_curriculum:
            score += 2
        if normalized_memory_source and normalized_memory_source in candidate_memory_sources:
            score += 3
        if str(candidate.get("artifact_kind", "")) == "recovery_case" and normalized_curriculum == "failure_recovery":
            score += 2
        if score <= 0:
            continue
        ranked.append(((-score, -int(candidate.get("support_count", 1) or 1), str(candidate.get("candidate_id", ""))), candidate))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked]


def _executed_commands(steps: list[StepRecord]) -> list[str]:
    commands: list[str] = []
    for step in steps:
        if step.action != "code_execute":
            continue
        command = str(step.content).strip()
        if command:
            commands.append(command)
    return commands


def _negative_command_patterns(steps: list[StepRecord]) -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {}
    for step in steps:
        if step.action != "code_execute" or not step.content or step.verification.get("passed", False):
            continue
        command = str(step.content).strip()
        if not command:
            continue
        reasons = [
            str(reason).strip()
            for reason in step.verification.get("reasons", [])
            if str(reason).strip()
        ]
        existing = patterns.setdefault(command, [])
        for reason in reasons:
            if reason not in existing:
                existing.append(reason)
    return patterns


def _failure_types(steps: list[StepRecord], *, termination_reason: str) -> list[str]:
    failure_types: set[str] = set()
    for step in steps:
        for reason in step.verification.get("reasons", []):
            normalized = _classify_failure_reason(str(reason))
            if normalized:
                failure_types.add(normalized)
        for signal in step.failure_signals:
            normalized = str(signal).strip()
            if normalized:
                failure_types.add(normalized)
    normalized_termination = str(termination_reason).strip()
    if normalized_termination in {"repeated_failed_action", "no_state_progress", "setup_failed"}:
        failure_types.add(normalized_termination)
    return sorted(failure_types)


def _classify_failure_reason(reason: str) -> str:
    normalized = reason.strip()
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
    return "other" if normalized else ""


def _benchmark_gap_kind(
    failure_types: list[str],
    transition_failures: list[str],
    *,
    command_count: int,
) -> str:
    if transition_failures:
        return "transition_pressure"
    if failure_types and command_count >= 2:
        return "recovery_path"
    if failure_types:
        return "failure_cluster"
    return "environment_pattern"


def _merge_candidates(
    existing: list[dict[str, object]],
    new_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for candidate in existing:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if candidate_id:
            by_id[candidate_id] = dict(candidate)
    for candidate in new_candidates:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        current = by_id.get(candidate_id)
        if current is None:
            by_id[candidate_id] = dict(candidate)
            continue
        merged = dict(current)
        merged["support_count"] = int(current.get("support_count", 1) or 1) + int(candidate.get("support_count", 1) or 1)
        merged["compiled_at"] = str(candidate.get("compiled_at", current.get("compiled_at", ""))).strip()
        for key in (
            "failure_types",
            "failure_signals",
            "transition_failures",
            "applicable_tasks",
            "verification_reasons",
            "recovery_commands",
            "executed_commands",
            "memory_sources",
        ):
            merged[key] = _merged_unique_list(current.get(key), candidate.get(key))
        merged["memory_source"] = (
            str(candidate.get("memory_source", "")).strip()
            or str(current.get("memory_source", "")).strip()
        )
        if isinstance(current.get("procedure"), dict) or isinstance(candidate.get("procedure"), dict):
            current_commands = dict(current.get("procedure", {})).get("commands", [])
            new_commands = dict(candidate.get("procedure", {})).get("commands", [])
            merged["procedure"] = {"commands": _merged_unique_list(current_commands, new_commands)}
        by_id[candidate_id] = merged
    return [by_id[key] for key in sorted(by_id)]


def _normalize_learning_candidate(candidate: dict[str, object]) -> dict[str, object] | None:
    normalized = dict(candidate)
    termination_reason = str(normalized.get("termination_reason", "")).strip()
    if termination_reason != "success":
        return normalized
    for field in ("failure_types", "known_failure_types"):
        values = normalized.get(field, [])
        if not isinstance(values, list):
            continue
        normalized[field] = [
            item
            for item in (str(value).strip() for value in values)
            if item and item != "other"
        ]
    transition_failures = normalized.get("transition_failures", [])
    if isinstance(transition_failures, list):
        normalized["transition_failures"] = [
            item
            for item in (str(value).strip() for value in transition_failures)
            if item
        ]
    else:
        normalized["transition_failures"] = []
    if str(normalized.get("artifact_kind", "")).strip() == "benchmark_gap":
        if not normalized.get("failure_types") and not normalized.get("transition_failures"):
            return None
    return normalized


def _merged_unique_list(left: object, right: object) -> list[str]:
    values: list[str] = []
    for source in (left, right):
        if not isinstance(source, list):
            continue
        for item in source:
            normalized = str(item).strip()
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def _task_match_tokens(task_id: str, source_task_id: str) -> set[str]:
    tokens: set[str] = set()
    for raw_value in (task_id, source_task_id):
        normalized = str(raw_value).strip()
        if not normalized:
            continue
        tokens.add(normalized)
        base = normalized
        while True:
            reduced = _strip_task_lineage_suffix(base)
            if not reduced or reduced == base:
                break
            tokens.add(reduced)
            base = reduced
    return tokens


def _strip_task_lineage_suffix(task_id: str) -> str:
    normalized = str(task_id).strip()
    for suffix in (
        "_verifier_replay",
        "_tool_replay",
        "_episode_replay",
        "_transition_pressure",
        "_discovered",
        "_skill_replay",
        "_skill_transfer",
        "_operator_replay",
    ):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
