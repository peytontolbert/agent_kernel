from __future__ import annotations

import json
from pathlib import Path
import re

from .config import KernelConfig
from .episode_store import iter_episode_documents
from .learning_compiler import load_learning_candidates
from .schemas import EpisodeRecord


def render_episode_document(episode: EpisodeRecord) -> dict[str, object]:
    document = episode.to_dict()
    summary = build_episode_summary(episode)
    document["summary"] = summary
    document["fragments"] = build_episode_fragments(episode, summary["failure_types"])
    return document


def build_episode_summary(episode: EpisodeRecord) -> dict[str, object]:
    executed_commands = [
        step.content
        for step in episode.steps
        if step.action == "code_execute" and step.content
    ]
    failure_types = sorted(
        {
            failure_type
            for step in episode.steps
            for failure_type in classify_failure_reasons(step.verification.get("reasons", []))
        }
    )
    failure_signals = sorted(
        {
            signal.strip()
            for step in episode.steps
            for signal in step.failure_signals
            if signal.strip()
        }
    )
    transition_failures = sorted(
        signal
        for signal in failure_signals
        if signal in {"no_state_progress", "state_regression"}
    )
    progress_gains = [float(step.state_progress_delta) for step in episode.steps if float(step.state_progress_delta) > 0]
    completion_trace = [
        float(step.state_transition.get("progress_delta", 0.0))
        for step in episode.steps
        if isinstance(step.state_transition, dict)
    ]
    final_completion_ratio = float(episode.world_model_summary.get("completion_ratio", 0.0))
    environment_violation_counts = _environment_violation_counts(episode)
    environment_alignment_failures = sorted(
        key
        for key, value in dict(episode.universe_summary.get("environment_alignment", {})).items()
        if key and value is False
    )
    environment_snapshot = {
        key: value
        for key, value in dict(episode.universe_summary.get("environment_snapshot", {})).items()
        if key in {"network_access_mode", "git_write_mode", "workspace_write_scope"}
    }
    return {
        "task_id": episode.task_id,
        "success": episode.success,
        "termination_reason": episode.termination_reason,
        "step_count": len(episode.steps),
        "executed_command_count": len(executed_commands),
        "executed_commands": executed_commands,
        "failure_types": failure_types,
        "failure_signals": failure_signals,
        "transition_failures": transition_failures,
        "state_progress_gain_steps": len(progress_gains),
        "state_regression_steps": sum(1 for step in episode.steps if step.state_regression_count > 0),
        "final_completion_ratio": final_completion_ratio,
        "net_state_progress_delta": round(sum(completion_trace), 3),
        "final_action": episode.steps[-1].action if episode.steps else "",
        "environment_violation_counts": environment_violation_counts,
        "environment_violation_flags": sorted(environment_violation_counts),
        "environment_alignment_failures": environment_alignment_failures,
        "environment_snapshot": environment_snapshot,
    }


def build_episode_fragments(
    episode: EpisodeRecord,
    failure_types: list[str] | None = None,
) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    normalized_failure_types = failure_types or []
    for step in episode.steps:
        if step.action == "code_execute" and step.content:
            fragments.append(
                {
                    "kind": "command",
                    "task_id": episode.task_id,
                    "step_index": step.index,
                    "command": step.content,
                    "passed": bool(step.verification.get("passed", False)),
                }
            )
        for reason in step.verification.get("reasons", []):
            if reason == "verification passed":
                continue
            fragments.append(
                {
                    "kind": "failure",
                    "task_id": episode.task_id,
                    "step_index": step.index,
                    "reason": reason,
                    "failure_types": normalized_failure_types,
                    "failure_signals": list(step.failure_signals),
                }
            )
        if step.state_transition:
            fragments.append(
                {
                    "kind": "state_transition",
                    "task_id": episode.task_id,
                    "step_index": step.index,
                    "progress_delta": step.state_progress_delta,
                    "regressions": list(step.state_transition.get("regressions", [])),
                    "cleared_forbidden_artifacts": list(step.state_transition.get("cleared_forbidden_artifacts", [])),
                    "newly_materialized_expected_artifacts": list(
                        step.state_transition.get("newly_materialized_expected_artifacts", [])
                    ),
                }
            )
        if step.command_governance:
            fragments.append(
                {
                    "kind": "governance",
                    "task_id": episode.task_id,
                    "step_index": step.index,
                    "score": int(step.command_governance.get("score", 0)),
                    "risk_flags": list(step.command_governance.get("risk_flags", [])),
                    "action_categories": list(step.command_governance.get("action_categories", [])),
                    "environment_alignment": dict(step.command_governance.get("environment_alignment", {})),
                    "network_host": str(step.command_governance.get("network_host", "")).strip(),
                }
            )
    return fragments


def classify_failure_reasons(reasons: list[str]) -> list[str]:
    failure_types: list[str] = []
    for reason in reasons:
        normalized = reason.strip()
        if not normalized or normalized.lower() == "verification passed":
            continue
        lowered = normalized.lower()
        if "timed out" in lowered:
            failure_types.append("timeout")
        elif "exit code" in lowered:
            failure_types.append("command_failure")
        elif "missing expected file" in lowered:
            failure_types.append("missing_expected_file")
        elif "missing expected output" in lowered:
            failure_types.append("missing_expected_output")
        elif "forbidden file present" in lowered:
            failure_types.append("forbidden_file_present")
        elif "unexpected file content" in lowered:
            failure_types.append("unexpected_file_content")
        elif "forbidden output present" in lowered:
            failure_types.append("forbidden_output_present")
        elif "policy terminated" in lowered:
            failure_types.append("policy_terminated")
        elif "repeated failed action" in lowered:
            failure_types.append("repeated_failed_action")
        else:
            failure_types.append("other")
    return failure_types


def _environment_violation_counts(episode: EpisodeRecord) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in episode.steps:
        governance = step.command_governance if isinstance(step.command_governance, dict) else {}
        for flag in governance.get("risk_flags", []):
            label = str(flag).strip()
            if not label:
                continue
            if label.endswith("_conflict"):
                counts[label] = counts.get(label, 0) + 1
        alignment = governance.get("environment_alignment", {})
        if isinstance(alignment, dict):
            for key, value in alignment.items():
                label = str(key).strip()
                if label and value is False:
                    counts[label] = counts.get(label, 0) + 1
    episode_alignment = episode.universe_summary.get("environment_alignment", {})
    if isinstance(episode_alignment, dict):
        for key, value in episode_alignment.items():
            label = str(key).strip()
            if label and value is False:
                counts[label] = counts.get(label, 0) + 1
    return counts


def _learning_artifacts_path_for_episodes_root(episodes_root: Path) -> Path:
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and episodes_root.resolve() == default_config.trajectories_root.resolve():
            return default_config.learning_artifacts_path
    except OSError:
        pass
    return episodes_root.parent / "learning" / "run_learning_artifacts.json"


def _iter_episode_documents_for_root(episodes_root: Path) -> list[dict[str, object]]:
    return iter_episode_documents(episodes_root)


def _success_skill_candidates_from_learning_artifacts(episodes_root: Path) -> list[dict[str, object]]:
    path = _learning_artifacts_path_for_episodes_root(episodes_root)
    skills: list[dict[str, object]] = []
    for candidate in load_learning_candidates(path):
        if str(candidate.get("artifact_kind", "")).strip() != "success_skill_candidate":
            continue
        procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
        commands = [str(command) for command in procedure.get("commands", []) if str(command).strip()]
        if not commands:
            continue
        source_task_id = str(candidate.get("source_task_id", "")).strip()
        if not source_task_id:
            continue
        skills.append(
            {
                "skill_id": f"skill:{source_task_id}:postrun",
                "kind": "command_sequence",
                "source_task_id": source_task_id,
                "applicable_tasks": [
                    str(value)
                    for value in candidate.get("applicable_tasks", [source_task_id])
                    if str(value).strip()
                ],
                "procedure": {"commands": commands},
                "verifier": {
                    "termination_reason": str(candidate.get("termination_reason", "")).strip(),
                },
                "known_failure_types": [
                    str(value)
                    for value in candidate.get("known_failure_types", [])
                    if str(value).strip()
                ],
                "task_contract": dict(candidate.get("task_contract", {}))
                if isinstance(candidate.get("task_contract", {}), dict)
                else {},
                "reuse_scope": "task_specific",
                "skill_signature": build_skill_signature(commands),
                "quality": float(candidate.get("quality", 0.8) or 0.8),
                "benchmark_family": str(candidate.get("benchmark_family", "bounded")).strip() or "bounded",
            }
        )
    return skills


def extract_successful_command_skills(
    episodes_root: Path,
    output_path: Path,
    *,
    min_quality: float = 0.0,
    transfer_only: bool = False,
) -> Path:
    skills: list[dict[str, object]] = _success_skill_candidates_from_learning_artifacts(episodes_root)
    for data in _iter_episode_documents_for_root(episodes_root):
        if not data.get("success"):
            continue
        workspace_name = Path(str(data.get("workspace", ""))).name
        commands = [
            _normalize_command_for_workspace(fragment["command"], workspace_name)
            for fragment in data.get("fragments", [])
            if fragment.get("kind") == "command" and fragment.get("passed")
        ]
        if not commands:
            commands = [
                _normalize_command_for_workspace(step["content"], workspace_name)
                for step in data.get("steps", [])
                if step.get("action") == "code_execute"
            ]
        if commands:
            quality = score_skill_quality(data, commands)
            skills.append(
                {
                    "skill_id": f"skill:{data['task_id']}:primary",
                    "kind": "command_sequence",
                    "source_task_id": data["task_id"],
                    "applicable_tasks": [data["task_id"]],
                    "procedure": {"commands": commands},
                    "verifier": {
                        "termination_reason": data.get("termination_reason", ""),
                    },
                    "known_failure_types": data.get("summary", {}).get("failure_types", []),
                    "task_contract": dict(data.get("task_contract", {})),
                    "reuse_scope": "workflow_specific"
                    if str(data.get("task_metadata", {}).get("benchmark_family", "")) == "workflow"
                    else "task_specific",
                    "skill_signature": build_skill_signature(commands),
                    "quality": quality,
                    "benchmark_family": str(
                        data.get("task_metadata", {}).get("benchmark_family", "bounded")
                    ),
                }
            )

    skills = dedupe_skills(skills)
    if min_quality > 0.0:
        skills = [skill for skill in skills if float(skill.get("quality", 0.0)) >= min_quality]
    if transfer_only:
        skills = [skill for skill in skills if _skill_has_transfer_potential(skill)]
        for skill in skills:
            skill["reuse_scope"] = "transfer_candidate"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "promoted",
        "retention_gate": {
            "min_quality": max(0.75, min_quality),
            "require_replay_transfer": True,
            "require_non_regression": True,
        },
        "generation_strategy": "cross_task_transfer" if transfer_only else "task_specific_replay",
        "skills": skills,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def extract_tool_candidates(
    episodes_root: Path,
    output_path: Path,
    *,
    min_quality: float = 0.0,
    replay_hardening: bool = False,
) -> Path:
    candidates: list[dict[str, object]] = []
    for data in _iter_episode_documents_for_root(episodes_root):
        if not data.get("success"):
            continue
        workspace_name = Path(str(data.get("workspace", ""))).name
        commands = [
            _normalize_command_for_workspace(fragment["command"], workspace_name)
            for fragment in data.get("fragments", [])
            if fragment.get("kind") == "command" and fragment.get("passed")
        ]
        if len(commands) < 2:
            continue
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            continue
        command_count = len(commands)
        benchmark_family = str(data.get("task_metadata", {}).get("benchmark_family", "bounded"))
        candidate_id = f"tool:{task_id}:primary"
        script_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        if replay_hardening:
            script_lines.extend(["IFS=$'\\n\\t'", "trap 'exit 1' ERR"])
        candidates.append(
            {
                "spec_version": "asi_v1",
                "tool_id": candidate_id,
                "kind": "local_shell_procedure",
                "lifecycle_state": "candidate",
                "promotion_stage": "candidate_procedure",
                "source_task_id": task_id,
                "benchmark_family": benchmark_family,
                "quality": score_tool_candidate(data, commands),
                "script_name": f"{task_id}_tool.sh",
                "script_body": "\n".join([*script_lines, *commands, ""]),
                "procedure": {"commands": commands},
                "task_contract": dict(data.get("task_contract", {})),
                "verifier": {"termination_reason": data.get("termination_reason", "")},
            }
        )
    deduped = dedupe_tool_candidates(candidates)
    if min_quality > 0.0:
        deduped = [candidate for candidate in deduped if float(candidate.get("quality", 0.0)) >= min_quality]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "candidate",
        "retention_gate": {
            "min_quality": max(0.75, min_quality),
            "require_replay_verification": True,
            "require_future_task_gain": True,
        },
        "generation_strategy": "script_hardening" if replay_hardening else "procedure_promotion",
        "candidates": deduped,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def extract_operator_classes(
    episodes_root: Path,
    output_path: Path,
    *,
    min_support: int = 2,
    cross_family_only: bool = False,
) -> Path:
    grouped: dict[str, list[dict[str, object]]] = {}
    for data in _iter_episode_documents_for_root(episodes_root):
        if not data.get("success"):
            continue
        task_id = str(data.get("task_id", "")).strip()
        contract = data.get("task_contract", {})
        if not task_id or not isinstance(contract, dict) or not contract:
            continue
        commands = _successful_commands(data)
        if not commands:
            continue
        signature = build_operator_signature(data, commands, include_benchmark_family=not cross_family_only)
        grouped.setdefault(signature, []).append(
            {
                "task_id": task_id,
                "commands": commands,
                "task_contract": dict(contract),
                "task_metadata": dict(data.get("task_metadata", {})),
                "quality": score_skill_quality(data, commands),
            }
        )

    operators: list[dict[str, object]] = []
    for signature, records in sorted(grouped.items()):
        source_task_ids = sorted({str(record["task_id"]) for record in records})
        if len(source_task_ids) < min_support:
            continue
        exemplar = max(records, key=lambda record: float(record.get("quality", 0.0)))
        metadata = exemplar.get("task_metadata", {})
        contract = exemplar.get("task_contract", {})
        applicable_benchmark_families = sorted(
            {
                str(record.get("task_metadata", {}).get("benchmark_family", "bounded"))
                for record in records
            }
        )
        if cross_family_only and len(applicable_benchmark_families) < 2:
            continue
        operators.append(
            {
                "operator_id": f"operator:{signature}:{source_task_ids[0]}",
                "kind": "operator_class",
                "operator_signature": signature,
                "operator_kind": infer_operator_kind(contract, exemplar["commands"]),
                "source_task_ids": source_task_ids,
                "support": len(source_task_ids),
                "support_count": len(source_task_ids),
                "benchmark_families": applicable_benchmark_families,
                "applicable_benchmark_families": applicable_benchmark_families,
                "applicable_capabilities": sorted(
                    {
                        str(record.get("task_metadata", {}).get("capability", "unknown"))
                        for record in records
                    }
                ),
                "steps": list(exemplar["commands"]),
                "task_contract": dict(contract),
                "template_procedure": {"commands": list(exemplar["commands"])},
                "template_contract": dict(contract),
                "quality": round(
                    sum(float(record.get("quality", 0.0)) for record in records) / len(records),
                    4,
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "operator_class_set",
        "lifecycle_state": "promoted",
        "retention_gate": {
            "min_transfer_pass_rate_delta_abs": 0.05,
            "require_cross_task_support": True,
            "min_support": min_support,
        },
        "generation_strategy": "cross_family_operator" if cross_family_only else "single_family_operator",
        "operators": operators,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _normalize_command_for_workspace(command: str, workspace_name: str) -> str:
    normalized = str(command).strip()
    workspace_name = workspace_name.strip().strip("/")
    if not workspace_name:
        return normalized
    mkdir_prefix = f"mkdir -p {workspace_name} && "
    if normalized.startswith(mkdir_prefix):
        normalized = normalized[len(mkdir_prefix):].strip()
    return normalized.replace(f"{workspace_name}/", "")


def _successful_commands(data: dict[str, object]) -> list[str]:
    workspace_name = Path(str(data.get("workspace", ""))).name
    commands = [
        _normalize_command_for_workspace(fragment["command"], workspace_name)
        for fragment in data.get("fragments", [])
        if fragment.get("kind") == "command" and fragment.get("passed")
    ]
    if commands:
        return commands
    return [
        _normalize_command_for_workspace(step["content"], workspace_name)
        for step in data.get("steps", [])
        if step.get("action") == "code_execute"
    ]


def build_operator_signature(
    data: dict[str, object],
    commands: list[str],
    *,
    include_benchmark_family: bool = True,
) -> str:
    contract = data.get("task_contract", {})
    benchmark_family = str(data.get("task_metadata", {}).get("benchmark_family", "bounded"))
    capability = str(data.get("task_metadata", {}).get("capability", "unknown"))
    expected_files = list(contract.get("expected_files", [])) if isinstance(contract, dict) else []
    forbidden_files = list(contract.get("forbidden_files", [])) if isinstance(contract, dict) else []
    verbs = sorted({_command_verb(command) for command in commands if command.strip()})
    path_shape = sorted(_path_shape(path) for path in expected_files)
    parts = [capability]
    if include_benchmark_family:
        parts.append(benchmark_family)
    parts.extend(
        [
            infer_operator_kind(contract, commands),
            f"expected{len(expected_files)}",
            f"forbidden{len(forbidden_files)}",
            f"verbs-{','.join(verbs)}",
            f"paths-{','.join(path_shape)}",
        ]
    )
    return ":".join(parts)


def infer_operator_kind(contract: dict[str, object], commands: list[str]) -> str:
    expected_files = list(contract.get("expected_files", [])) if isinstance(contract, dict) else []
    forbidden_files = list(contract.get("forbidden_files", [])) if isinstance(contract, dict) else []
    if forbidden_files and expected_files:
        return "cleanup_write"
    if len(expected_files) > 1:
        return "multi_emit"
    if len(expected_files) == 1:
        return "single_emit"
    if any(_command_verb(command) == "mv" for command in commands):
        return "rename"
    return "shell_procedure"


def _command_verb(command: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", str(command))
    return match.group(1) if match else "unknown"


def _path_shape(path: str) -> str:
    parts = [part for part in str(path).split("/") if part]
    if not parts:
        return "flat"
    return "nested" if len(parts) > 1 else "flat"


def build_skill_signature(commands: list[str]) -> str:
    return " && ".join(command.strip() for command in commands if command.strip())


def dedupe_skills(skills: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for skill in skills:
        source_task_id = str(skill.get("source_task_id", skill.get("task_id", "")))
        commands = list(skill.get("procedure", {}).get("commands", []))
        signature = str(skill.get("skill_signature") or build_skill_signature(commands))
        candidate = dict(skill)
        candidate["skill_signature"] = signature
        key = (source_task_id, signature)
        incumbent = best_by_key.get(key)
        if incumbent is None or _prefer_skill(candidate, incumbent):
            best_by_key[key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda skill: (
            -float(skill.get("quality", 0.0)),
            str(skill.get("source_task_id", skill.get("task_id", ""))),
            str(skill.get("skill_id", "")),
        ),
    )


def _prefer_skill(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    candidate_quality = float(candidate.get("quality", 0.0))
    incumbent_quality = float(incumbent.get("quality", 0.0))
    if candidate_quality != incumbent_quality:
        return candidate_quality > incumbent_quality

    candidate_commands = list(candidate.get("procedure", {}).get("commands", []))
    incumbent_commands = list(incumbent.get("procedure", {}).get("commands", []))
    if len(candidate_commands) != len(incumbent_commands):
        return len(candidate_commands) < len(incumbent_commands)

    candidate_id = str(candidate.get("skill_id", ""))
    incumbent_id = str(incumbent.get("skill_id", ""))
    return candidate_id < incumbent_id


def score_skill_quality(data: dict[str, object], commands: list[str]) -> float:
    score = 0.0
    summary = data.get("summary", {})
    failure_types = list(summary.get("failure_types", []))
    if data.get("success"):
        score += 0.35
    if data.get("termination_reason") == "success":
        score += 0.15
    if not failure_types or failure_types == ["other"]:
        score += 0.1
    step_count = int(summary.get("step_count", len(data.get("steps", [])) or 0))
    if step_count <= 1:
        score += 0.2
    elif step_count <= 2:
        score += 0.1
    if len(commands) == 1:
        score += 0.1
    normalized_commands = [command.strip() for command in commands if command.strip()]
    if normalized_commands and all("&& echo " not in command for command in normalized_commands):
        score += 0.05
    if normalized_commands and all("echo '" not in command or "printf" in command for command in normalized_commands):
        score += 0.05
    return round(min(score, 1.0), 2)


def score_tool_candidate(data: dict[str, object], commands: list[str]) -> float:
    score = score_skill_quality(data, commands)
    if len(commands) >= 2:
        score += 0.05
    if len(commands) <= 4:
        score += 0.05
    return round(min(score, 1.0), 2)


def dedupe_tool_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_signature: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        commands = list(candidate.get("procedure", {}).get("commands", []))
        signature = build_skill_signature(commands)
        incumbent = best_by_signature.get(signature)
        if incumbent is None or _prefer_skill(candidate, incumbent):
            best_by_signature[signature] = dict(candidate)
    return sorted(best_by_signature.values(), key=lambda item: (-float(item.get("quality", 0.0)), str(item.get("tool_id", ""))))


def _skill_has_transfer_potential(skill: dict[str, object]) -> bool:
    commands = list(skill.get("procedure", {}).get("commands", []))
    benchmark_family = str(skill.get("benchmark_family", "bounded"))
    return len(commands) >= 2 or benchmark_family in {
        "workflow",
        "tooling",
        "integration",
        "project",
        "repository",
    }
