from __future__ import annotations

import json
from pathlib import Path
import re

from ..config import KernelConfig, current_external_task_manifests_paths
from ..ops.episode_store import iter_episode_documents
from ..learning_compiler import load_learning_candidates
from ..schemas import EpisodeRecord
from ..tasking.task_bank import TaskBank


def _is_llm_generated_decision_source(decision_source: str) -> bool:
    normalized = str(decision_source).strip()
    return normalized == "llm"


def _is_bounded_decoder_generated_decision_source(decision_source: str) -> bool:
    normalized = str(decision_source).strip()
    return normalized.endswith("_decoder")


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
    execution_source_counts = {
        "decoder_generated": 0,
        "llm_generated": 0,
        "bounded_decoder_generated": 0,
        "synthetic_plan": 0,
        "deterministic_or_other": 0,
    }
    for step in episode.steps:
        if step.action != "code_execute" or not step.content:
            continue
        decision_source = str(step.decision_source).strip()
        if _is_llm_generated_decision_source(decision_source):
            execution_source_counts["decoder_generated"] += 1
            execution_source_counts["llm_generated"] += 1
        elif _is_bounded_decoder_generated_decision_source(decision_source):
            execution_source_counts["decoder_generated"] += 1
            execution_source_counts["bounded_decoder_generated"] += 1
        elif decision_source == "synthetic_edit_plan_direct":
            execution_source_counts["synthetic_plan"] += 1
        else:
            execution_source_counts["deterministic_or_other"] += 1
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
    verifier_obligations = _episode_verifier_obligations(episode)
    changed_paths = _episode_changed_paths(episode)
    recovery_traces = _episode_recovery_traces(episode, verifier_obligations=verifier_obligations)
    edit_patches = _episode_edit_patches(episode)
    return {
        "task_id": episode.task_id,
        "success": episode.success,
        "termination_reason": episode.termination_reason,
        "step_count": len(episode.steps),
        "executed_command_count": len(executed_commands),
        "executed_commands": executed_commands,
        "execution_source_summary": {
            **execution_source_counts,
            "total_executed_commands": len(executed_commands),
        },
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
        "verifier_obligation_count": len(verifier_obligations),
        "verifier_obligations": verifier_obligations,
        "changed_paths": changed_paths,
        "edit_patch_count": len(edit_patches),
        "edit_patch_paths": [str(item.get("path", "")).strip() for item in edit_patches if str(item.get("path", "")).strip()],
        "edit_patches": [
            {
                "path": str(item.get("path", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "patch_summary": str(item.get("patch_summary", "")).strip(),
            }
            for item in edit_patches[:4]
            if str(item.get("path", "")).strip()
        ],
        "recovery_trace_count": len(recovery_traces),
        "recovery_traces": recovery_traces[:4],
    }


def build_episode_fragments(
    episode: EpisodeRecord,
    failure_types: list[str] | None = None,
) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    normalized_failure_types = failure_types or []
    verifier_obligations = _episode_verifier_obligations(episode)
    recovery_traces = _episode_recovery_traces(episode, verifier_obligations=verifier_obligations)
    for step in episode.steps:
        command_outcome_fragment: dict[str, object] | None = None
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
            command_outcome_fragment = {
                "kind": "command_outcome",
                "task_id": episode.task_id,
                "step_index": step.index,
                "command": step.content,
                "command_pattern": _command_pattern(step.content),
                "passed": bool(step.verification.get("passed", False)),
                "exit_code": _step_exit_code(step),
                "progress_delta": float(step.state_progress_delta),
                "verifier_delta": _state_transition_verifier_delta(step.state_transition),
                "changed_paths": _step_changed_paths(step),
                "failure_signals": list(step.failure_signals),
                "verification_reasons": [
                    str(reason).strip()
                    for reason in step.verification.get("reasons", [])
                    if str(reason).strip() and str(reason).strip().lower() != "verification passed"
                ][:4],
                "test_command": _is_test_like_command(step.content),
            }
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
            for patch in step.state_transition.get("edit_patches", []):
                if not isinstance(patch, dict):
                    continue
                path = str(patch.get("path", "")).strip()
                patch_text = str(patch.get("patch", "")).strip()
                if not path or not patch_text:
                    continue
                fragments.append(
                    {
                        "kind": "edit_patch",
                        "task_id": episode.task_id,
                        "step_index": step.index,
                        "path": path,
                        "status": str(patch.get("status", "")).strip(),
                        "patch": patch_text,
                        "patch_summary": str(patch.get("patch_summary", "")).strip(),
                        "before_excerpt": str(patch.get("before_excerpt", "")).strip(),
                        "after_excerpt": str(patch.get("after_excerpt", "")).strip(),
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
        if command_outcome_fragment is not None:
            fragments.append(command_outcome_fragment)
    for obligation in verifier_obligations:
        fragments.append(
            {
                "kind": "verifier_obligation",
                "task_id": episode.task_id,
                "obligation_kind": str(obligation.get("kind", "")).strip(),
                "target": str(obligation.get("target", "")).strip(),
                "text": str(obligation.get("text", "")).strip(),
            }
        )
    for trace in recovery_traces:
        fragments.append({"kind": "recovery_trace", "task_id": episode.task_id, **dict(trace)})
    return fragments


def _episode_verifier_obligations(episode: EpisodeRecord) -> list[dict[str, str]]:
    contract = dict(episode.task_contract) if isinstance(episode.task_contract, dict) else {}
    metadata = dict(contract.get("metadata", {})) if isinstance(contract.get("metadata", {}), dict) else {}
    task_metadata = dict(episode.task_metadata) if isinstance(episode.task_metadata, dict) else {}
    verifier = task_metadata.get("semantic_verifier", metadata.get("semantic_verifier", {}))
    verifier = dict(verifier) if isinstance(verifier, dict) else {}
    obligations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def append(kind: str, target: str, text: str) -> None:
        normalized_kind = str(kind).strip()
        normalized_target = str(target).strip()
        normalized_text = str(text).strip()
        if not normalized_text:
            return
        key = (normalized_kind, normalized_target, normalized_text)
        if key in seen:
            return
        seen.add(key)
        obligations.append(
            {
                "kind": normalized_kind,
                "target": normalized_target,
                "text": normalized_text,
            }
        )

    for path in contract.get("expected_files", []):
        normalized = str(path).strip()
        if normalized:
            append("expected_artifact", normalized, f"materialize expected artifact {normalized}")
    for path in dict(contract.get("expected_file_contents", {})).keys():
        normalized = str(path).strip()
        if normalized:
            append("expected_content", normalized, f"materialize expected content for {normalized}")
    for path in contract.get("forbidden_files", []):
        normalized = str(path).strip()
        if normalized:
            append("forbidden_artifact", normalized, f"remove forbidden artifact {normalized}")
    for path in verifier.get("expected_changed_paths", []):
        normalized = str(path).strip()
        if normalized:
            append("workflow_path", normalized, f"update workflow path {normalized}")
    for path in verifier.get("generated_paths", []):
        normalized = str(path).strip()
        if normalized:
            append("generated_artifact", normalized, f"regenerate generated artifact {normalized}")
    for path in verifier.get("preserved_paths", []):
        normalized = str(path).strip()
        if normalized:
            append("preserved_artifact", normalized, f"preserve required artifact {normalized}")
    for branch in verifier.get("required_merged_branches", []):
        normalized = str(branch).strip()
        if normalized:
            append("required_merge", normalized, f"accept required branch {normalized}")
    for path in verifier.get("resolved_conflict_paths", []):
        normalized = str(path).strip()
        if normalized:
            append("resolved_conflict", normalized, f"resolve merge conflict {normalized}")
    for rule in verifier.get("report_rules", []):
        if not isinstance(rule, dict):
            continue
        path = str(rule.get("path", "")).strip()
        must_mention = [str(value).strip() for value in rule.get("must_mention", []) if str(value).strip()]
        covers = [str(value).strip() for value in rule.get("covers", []) if str(value).strip()]
        if not path:
            continue
        detail = []
        if must_mention:
            detail.append("mention " + ", ".join(must_mention[:4]))
        if covers:
            detail.append("cover " + ", ".join(covers[:4]))
        description = f"write workflow report {path}"
        if detail:
            description = f"{description} and " + " and ".join(detail)
        append("report_rule", path, description)
    for rule in verifier.get("test_commands", []):
        if not isinstance(rule, dict):
            continue
        argv = [str(value).strip() for value in rule.get("argv", []) if str(value).strip()]
        if not argv:
            continue
        label = str(rule.get("label", "")).strip() or "workflow test"
        append("test_command", argv[0], f"run {label}: {' '.join(argv)}")
    return obligations


def _episode_changed_paths(episode: EpisodeRecord) -> list[str]:
    changed: list[str] = []
    for step in episode.steps:
        for path in _step_changed_paths(step):
            if path not in changed:
                changed.append(path)
    return changed


def _step_changed_paths(step) -> list[str]:
    transition = step.state_transition if isinstance(step.state_transition, dict) else {}
    changed: list[str] = []
    for key in (
        "newly_materialized_expected_artifacts",
        "newly_satisfied_expected_contents",
        "cleared_forbidden_artifacts",
        "newly_changed_preserved_artifacts",
        "newly_updated_workflow_paths",
        "newly_updated_generated_paths",
        "newly_updated_report_paths",
        "regressions",
    ):
        for value in transition.get(key, []):
            normalized = str(value).strip()
            if normalized and normalized not in changed:
                changed.append(normalized)
    return changed


def _state_transition_verifier_delta(transition: dict[str, object] | None) -> int:
    payload = dict(transition or {})
    explicit = payload.get("state_change_score")
    try:
        return int(explicit)
    except (TypeError, ValueError):
        pass
    positive = 0
    for key in (
        "newly_materialized_expected_artifacts",
        "newly_satisfied_expected_contents",
        "cleared_forbidden_artifacts",
        "newly_updated_workflow_paths",
        "newly_updated_generated_paths",
        "newly_updated_report_paths",
    ):
        positive += len([value for value in payload.get(key, []) if str(value).strip()])
    negative = len([value for value in payload.get("regressions", []) if str(value).strip()])
    return positive - negative


def _episode_recovery_traces(
    episode: EpisodeRecord,
    *,
    verifier_obligations: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    obligations = verifier_obligations or _episode_verifier_obligations(episode)
    obligation_texts = [str(item.get("text", "")).strip() for item in obligations if str(item.get("text", "")).strip()]
    traces: list[dict[str, object]] = []
    steps = list(episode.steps)
    for index, step in enumerate(steps[:-1]):
        if bool(step.verification.get("passed", False)):
            continue
        failure_reasons = [
            str(reason).strip()
            for reason in step.verification.get("reasons", [])
            if str(reason).strip() and str(reason).strip().lower() != "verification passed"
        ]
        recovery_step = next(
            (
                candidate
                for candidate in steps[index + 1 :]
                if str(candidate.action).strip() == "code_execute"
                and str(candidate.content).strip()
                and bool(candidate.verification.get("passed", False))
            ),
            None,
        )
        if recovery_step is None:
            continue
        traces.append(
            {
                "failed_step_index": int(step.index),
                "failed_command": str(step.content).strip(),
                "failure_signals": [str(signal).strip() for signal in step.failure_signals if str(signal).strip()],
                "failure_reasons": failure_reasons[:4],
                "recovery_step_index": int(recovery_step.index),
                "recovery_command": str(recovery_step.content).strip(),
                "recovered_changed_paths": _step_changed_paths(recovery_step),
                "outstanding_obligations": obligation_texts[:4],
            }
        )
    return traces[:4]


def _episode_edit_patches(episode: EpisodeRecord) -> list[dict[str, object]]:
    patches: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for step in episode.steps:
        transition = step.state_transition if isinstance(step.state_transition, dict) else {}
        for patch in transition.get("edit_patches", []):
            if not isinstance(patch, dict):
                continue
            path = str(patch.get("path", "")).strip()
            patch_text = str(patch.get("patch", "")).strip()
            if not path or not patch_text:
                continue
            key = (int(step.index), path)
            if key in seen:
                continue
            seen.add(key)
            patches.append(
                {
                    "step_index": int(step.index),
                    "path": path,
                    "status": str(patch.get("status", "")).strip(),
                    "patch": patch_text,
                    "patch_summary": str(patch.get("patch_summary", "")).strip(),
                    "before_excerpt": str(patch.get("before_excerpt", "")).strip(),
                    "after_excerpt": str(patch.get("after_excerpt", "")).strip(),
                }
            )
    return patches[:8]


def _command_pattern(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    if normalized.startswith("pytest"):
        return "pytest"
    if normalized.startswith("python -m pytest"):
        return "python -m pytest"
    if normalized.startswith("git "):
        parts = normalized.split()
        if len(parts) >= 2:
            return f"git {parts[1]}"
        return "git"
    match = re.match(r"([A-Za-z0-9_.-]+)", normalized)
    return match.group(1) if match else normalized


def _step_exit_code(step) -> int | None:
    payload = step.command_result if isinstance(step.command_result, dict) else {}
    try:
        return int(payload.get("exit_code"))
    except (TypeError, ValueError):
        return None


def _is_test_like_command(command: str) -> bool:
    normalized = str(command).strip()
    return bool(
        normalized.startswith("pytest")
        or normalized.startswith("python -m pytest")
        or normalized.startswith("test ")
        or "/test" in normalized
        or " verify" in normalized
    )


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


def _task_bank_for_extractors() -> TaskBank:
    manifest_paths = current_external_task_manifests_paths()
    try:
        return TaskBank(
            config=KernelConfig(),
            external_task_manifests=manifest_paths if manifest_paths else None,
        )
    except TypeError:
        return TaskBank()


def _success_skill_candidates_from_learning_artifacts(episodes_root: Path) -> list[dict[str, object]]:
    path = _learning_artifacts_path_for_episodes_root(episodes_root)
    bank = _task_bank_for_extractors()
    skills: list[dict[str, object]] = []
    for candidate in load_learning_candidates(path):
        artifact_kind = str(candidate.get("artifact_kind", "")).strip()
        if artifact_kind not in {"success_skill_candidate", "recovery_case"}:
            continue
        source_task_id, task_contract = _learning_candidate_source(candidate, bank=bank)
        if not source_task_id:
            continue
        if artifact_kind == "recovery_case":
            if not bool(candidate.get("success", False)):
                continue
            commands = [str(command).strip() for command in candidate.get("recovery_commands", []) if str(command).strip()]
        else:
            procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
            commands = [str(command).strip() for command in procedure.get("commands", []) if str(command).strip()]
        if not commands:
            continue
        try:
            source_task = bank.get(source_task_id)
        except KeyError:
            source_task = None
        if source_task is not None and not _procedure_matches_source_task(source_task, commands):
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
                    for value in candidate.get("known_failure_types", candidate.get("failure_types", []))
                    if str(value).strip()
                ],
                "task_contract": task_contract,
                "reuse_scope": "task_specific",
                "skill_signature": build_skill_signature(commands),
                "quality": _quality_with_retrieval_bonus(
                    float(candidate.get("quality", 0.78 if artifact_kind == "recovery_case" else 0.8) or 0.8),
                    candidate,
                ),
                "benchmark_family": str(candidate.get("benchmark_family", "bounded")).strip() or "bounded",
                **_retrieval_provenance_fields(candidate, commands),
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
                    **_retrieval_provenance_fields(data, commands),
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
    bank = _task_bank_for_extractors()
    candidates: list[dict[str, object]] = _tool_candidates_from_learning_artifacts(
        episodes_root,
        replay_hardening=replay_hardening,
    )
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
                if step.get("action") == "code_execute" and str(step.get("content", "")).strip()
            ]
        if not commands:
            continue
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            continue
        try:
            task = bank.get(task_id)
        except KeyError:
            continue
        benchmark_family = str(data.get("task_metadata", {}).get("benchmark_family", "bounded"))
        task_contract = _enriched_task_contract(
            dict(data.get("task_contract", {})) if isinstance(data.get("task_contract", {}), dict) else {},
            fallback_task=task,
        )
        candidate = _tool_candidate_payload(
            source_task_id=task_id,
            benchmark_family=benchmark_family,
            commands=commands,
            task_contract=task_contract,
            termination_reason=str(data.get("termination_reason", "")).strip(),
            quality=score_tool_candidate(data, commands, task_contract=task_contract),
            replay_hardening=replay_hardening,
            provenance=_retrieval_provenance_fields(data, commands),
        )
        if _procedure_has_transfer_potential(candidate):
            candidates.append(candidate)
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


def _enriched_task_contract(task_contract: dict[str, object], *, fallback_task: object | None) -> dict[str, object]:
    enriched = dict(task_contract)
    if fallback_task is None:
        return enriched
    fallback_metadata = dict(getattr(fallback_task, "metadata", {}))
    metadata = dict(enriched.get("metadata", {})) if isinstance(enriched.get("metadata", {}), dict) else {}
    merged_metadata = dict(fallback_metadata)
    merged_metadata.update(metadata)
    if merged_metadata:
        enriched["metadata"] = merged_metadata
    defaults = {
        "prompt": str(getattr(fallback_task, "prompt", "")),
        "workspace_subdir": str(getattr(fallback_task, "workspace_subdir", "")),
        "success_command": str(getattr(fallback_task, "success_command", "")),
        "setup_commands": list(getattr(fallback_task, "setup_commands", [])),
        "suggested_commands": list(getattr(fallback_task, "suggested_commands", [])),
        "expected_files": list(getattr(fallback_task, "expected_files", [])),
        "expected_output_substrings": list(getattr(fallback_task, "expected_output_substrings", [])),
        "forbidden_files": list(getattr(fallback_task, "forbidden_files", [])),
        "forbidden_output_substrings": list(getattr(fallback_task, "forbidden_output_substrings", [])),
        "expected_file_contents": dict(getattr(fallback_task, "expected_file_contents", {})),
        "max_steps": int(getattr(fallback_task, "max_steps", 5) or 5),
    }
    for key, value in defaults.items():
        current = enriched.get(key)
        if current in (None, "", [], {}):
            enriched[key] = value
    return enriched


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

    candidate_retrieval_score = _retrieval_preference_score(candidate)
    incumbent_retrieval_score = _retrieval_preference_score(incumbent)
    if candidate_retrieval_score != incumbent_retrieval_score:
        return candidate_retrieval_score > incumbent_retrieval_score

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
    score += _retrieval_quality_bonus(data)
    return round(min(score, 1.0), 2)


def score_tool_candidate(
    data: dict[str, object],
    commands: list[str],
    *,
    task_contract: dict[str, object] | None = None,
) -> float:
    score = score_skill_quality(data, commands)
    bundle = _shared_repo_bundle_metadata(
        task_contract
        if isinstance(task_contract, dict)
        else dict(data.get("task_contract", {})) if isinstance(data.get("task_contract", {}), dict) else {},
        commands,
    )
    if bundle:
        role = str(bundle.get("role", "")).strip()
        if role == "worker":
            score += 0.05
        elif not bool(bundle.get("bundle_complete", False)):
            score -= 0.45
        else:
            score += 0.05
    if len(commands) >= 2:
        score += 0.05
    if len(commands) <= 4:
        score += 0.05
    return round(max(0.0, min(score, 1.0)), 2)


def dedupe_tool_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_signature: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        normalized = _normalized_tool_candidate(candidate)
        commands = list(normalized.get("procedure", {}).get("commands", []))
        signature = build_skill_signature([_stable_command_signature(command) for command in commands])
        incumbent = best_by_signature.get(signature)
        if incumbent is None or _prefer_skill(normalized, incumbent):
            best_by_signature[signature] = normalized
    return sorted(
        best_by_signature.values(),
        key=lambda item: (-float(item.get("quality", 0.0)), str(item.get("tool_id", ""))),
    )


def _skill_has_transfer_potential(skill: dict[str, object]) -> bool:
    return _procedure_has_transfer_potential(skill)


def _procedure_has_transfer_potential(record: dict[str, object]) -> bool:
    bundle = record.get("shared_repo_bundle", {})
    if isinstance(bundle, dict) and str(bundle.get("role", "")).strip() == "integrator":
        if not bool(bundle.get("bundle_complete", False)):
            return False
    commands = list(record.get("procedure", {}).get("commands", []))
    benchmark_family = str(record.get("benchmark_family", "bounded"))
    return len(commands) >= 2 or benchmark_family in {
        "workflow",
        "tooling",
        "integration",
        "project",
        "repository",
    }


def _tool_candidate_payload(
    *,
    source_task_id: str,
    benchmark_family: str,
    commands: list[str],
    task_contract: dict[str, object],
    termination_reason: str,
    quality: float,
    replay_hardening: bool,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    compact_commands = _dedupe_procedure_commands(commands)
    script_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if replay_hardening:
        script_lines.extend(["IFS=$'\n\t'", "trap 'exit 1' ERR"])
    prompt = str(task_contract.get("prompt", "")).strip()
    shared_repo_bundle = _shared_repo_bundle_metadata(task_contract, compact_commands)
    payload = {
        "spec_version": "asi_v1",
        "tool_id": f"tool:{source_task_id}:primary",
        "kind": "local_shell_procedure",
        "lifecycle_state": "candidate",
        "promotion_stage": "candidate_procedure",
        "source_task_id": source_task_id,
        "benchmark_family": benchmark_family,
        "quality": quality,
        "name": f"{source_task_id}_procedure",
        "title": source_task_id.replace("_", " "),
        "summary": prompt or f"Reusable shell procedure derived from {source_task_id}.",
        "command": compact_commands[0] if len(compact_commands) == 1 else "",
        "script_name": f"{source_task_id}_tool.sh",
        "script_body": "\n".join([*script_lines, *compact_commands, ""]),
        "procedure": {"commands": compact_commands},
        "task_contract": dict(task_contract),
        "verifier": {"termination_reason": termination_reason},
        "shared_repo_bundle": shared_repo_bundle,
    }
    if provenance:
        payload.update(provenance)
    return payload


def _shared_repo_bundle_metadata(task_contract: dict[str, object], commands: list[str]) -> dict[str, object]:
    metadata = dict(task_contract.get("metadata", {})) if isinstance(task_contract.get("metadata", {}), dict) else {}
    workflow_guard = dict(metadata.get("workflow_guard", {})) if isinstance(metadata.get("workflow_guard", {}), dict) else {}
    verifier = dict(metadata.get("semantic_verifier", {})) if isinstance(metadata.get("semantic_verifier", {}), dict) else {}
    repo_id = str(workflow_guard.get("shared_repo_id", "")).strip()
    worker_branch = str(workflow_guard.get("worker_branch", "")).strip()
    try:
        shared_repo_order = int(metadata.get("shared_repo_order", 0) or 0)
    except (TypeError, ValueError):
        shared_repo_order = 0
    required_merged_branches = [
        str(value).strip()
        for value in verifier.get("required_merged_branches", [])
        if str(value).strip()
    ]
    if not repo_id and not worker_branch and shared_repo_order <= 0 and not required_merged_branches:
        return {}
    observed_merged_branches = [
        branch for branch in required_merged_branches if any(branch in str(command) for command in commands)
    ]
    role = "integrator" if shared_repo_order > 0 or required_merged_branches else "worker"
    bundle_complete = True if role != "integrator" else len(observed_merged_branches) >= len(required_merged_branches)
    return {
        "shared_repo_id": repo_id,
        "role": role,
        "shared_repo_order": shared_repo_order,
        "worker_branch": worker_branch,
        "required_merged_branches": required_merged_branches,
        "observed_merged_branches": observed_merged_branches,
        "bundle_complete": bundle_complete,
    }


def _dedupe_procedure_commands(commands: list[str]) -> list[str]:
    best_by_signature: dict[str, str] = {}
    order: list[str] = []
    for raw_command in commands:
        command = str(raw_command).strip()
        if not command:
            continue
        signature = _stable_command_signature(command)
        incumbent = best_by_signature.get(signature)
        if incumbent is None:
            order.append(signature)
            best_by_signature[signature] = command
            continue
        if _prefer_command_variant(command, incumbent):
            best_by_signature[signature] = command
    return [best_by_signature[signature] for signature in order]


def _stable_command_signature(command: str) -> str:
    normalized = str(command).replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = normalized.replace("\n", "\\n")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _prefer_command_variant(candidate: str, incumbent: str) -> bool:
    candidate_score = (candidate.count("\n"), len(candidate))
    incumbent_score = (incumbent.count("\n"), len(incumbent))
    if candidate_score != incumbent_score:
        return candidate_score < incumbent_score
    return candidate < incumbent


def _normalized_tool_candidate(candidate: dict[str, object]) -> dict[str, object]:
    normalized = dict(candidate)
    procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
    commands = _dedupe_procedure_commands(
        [str(value) for value in procedure.get("commands", []) if str(value).strip()]
    )
    normalized["procedure"] = {"commands": commands}
    script_lines = [line for line in str(candidate.get("script_body", "")).splitlines() if line.strip()]
    header_lines: list[str] = []
    body_started = False
    for line in script_lines:
        if not body_started and line.startswith("#!"):
            header_lines.append(line)
            continue
        if not body_started and (line.startswith("set -") or line.startswith("IFS=") or line.startswith("trap ")):
            header_lines.append(line)
            continue
        body_started = True
    if header_lines:
        normalized["script_body"] = "\n".join([*header_lines, *commands, ""])
    if len(commands) == 1:
        normalized["command"] = commands[0]
    elif "command" not in normalized:
        normalized["command"] = ""
    return normalized


def _retrieval_provenance_fields(record: dict[str, object], commands: list[str]) -> dict[str, object]:
    selected_span_ids = [
        str(value).strip()
        for value in record.get("selected_retrieval_span_ids", [])
        if str(value).strip()
    ]
    retrieval_backed_commands = [
        str(value).strip()
        for value in record.get("retrieval_backed_commands", [])
        if str(value).strip()
    ]
    retrieval_selected_steps = _safe_int(record.get("retrieval_selected_steps", 0))
    retrieval_influenced_steps = _safe_int(record.get("retrieval_influenced_steps", 0))
    trusted_retrieval_steps = _safe_int(record.get("trusted_retrieval_steps", 0))
    for step in record.get("steps", []) if isinstance(record.get("steps", []), list) else []:
        if not isinstance(step, dict):
            continue
        command = str(step.get("content", "")).strip()
        selected_span_id = str(step.get("selected_retrieval_span_id", "")).strip()
        retrieval_influenced = bool(step.get("retrieval_influenced", False))
        trusted_retrieval = bool(step.get("trust_retrieval", False))
        passed = bool(dict(step.get("verification", {})).get("passed", False))
        if selected_span_id and selected_span_id not in selected_span_ids:
            selected_span_ids.append(selected_span_id)
            retrieval_selected_steps += 1
        if retrieval_influenced:
            retrieval_influenced_steps += 1
        if trusted_retrieval:
            trusted_retrieval_steps += 1
        if passed and (selected_span_id or retrieval_influenced or trusted_retrieval) and command and command in commands:
            if command not in retrieval_backed_commands:
                retrieval_backed_commands.append(command)
    retrieval_backed = bool(record.get("retrieval_backed", False)) or bool(
        retrieval_backed_commands or retrieval_influenced_steps or trusted_retrieval_steps
    )
    return {
        "retrieval_backed": retrieval_backed,
        "retrieval_selected_steps": retrieval_selected_steps,
        "retrieval_influenced_steps": retrieval_influenced_steps,
        "trusted_retrieval_steps": trusted_retrieval_steps,
        "selected_retrieval_span_ids": selected_span_ids,
        "retrieval_backed_commands": retrieval_backed_commands,
    }


def _retrieval_quality_bonus(record: dict[str, object]) -> float:
    if not bool(record.get("retrieval_backed", False)):
        if (
            _safe_int(record.get("retrieval_selected_steps", 0)) <= 0
            and _safe_int(record.get("retrieval_influenced_steps", 0)) <= 0
            and _safe_int(record.get("trusted_retrieval_steps", 0)) <= 0
        ):
            return 0.0
    bonus = 0.05
    if _safe_int(record.get("trusted_retrieval_steps", 0)) > 0:
        bonus += 0.03
    return bonus


def _quality_with_retrieval_bonus(base_quality: float, record: dict[str, object]) -> float:
    return round(min(1.0, max(0.0, float(base_quality) + _retrieval_quality_bonus(record))), 2)


def _retrieval_preference_score(record: dict[str, object]) -> int:
    if not bool(record.get("retrieval_backed", False)):
        return 0
    score = 1
    if _safe_int(record.get("retrieval_influenced_steps", 0)) > 0:
        score += 1
    if _safe_int(record.get("trusted_retrieval_steps", 0)) > 0:
        score += 2
    if any(str(value).strip() for value in record.get("retrieval_backed_commands", [])):
        score += 1
    return score


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tool_candidates_from_learning_artifacts(
    episodes_root: Path,
    *,
    replay_hardening: bool = False,
) -> list[dict[str, object]]:
    path = _learning_artifacts_path_for_episodes_root(episodes_root)
    bank = _task_bank_for_extractors()
    tools: list[dict[str, object]] = []
    for candidate in load_learning_candidates(path):
        artifact_kind = str(candidate.get("artifact_kind", "")).strip()
        if artifact_kind not in {"success_skill_candidate", "recovery_case"}:
            continue
        source_task_id, task_contract = _tool_candidate_learning_source(candidate, bank=bank)
        if not source_task_id:
            continue
        procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
        if artifact_kind == "recovery_case":
            if not bool(candidate.get("success", False)):
                continue
            commands = [str(command).strip() for command in candidate.get("recovery_commands", []) if str(command).strip()]
        else:
            commands = [str(command).strip() for command in procedure.get("commands", []) if str(command).strip()]
        if not commands:
            continue
        try:
            fallback_task = bank.get(source_task_id)
        except KeyError:
            fallback_task = None
        if fallback_task is not None and not _procedure_matches_source_task(fallback_task, commands):
            continue
        task_contract = _enriched_task_contract(task_contract, fallback_task=fallback_task)
        tool = _tool_candidate_payload(
            source_task_id=source_task_id,
            benchmark_family=str(candidate.get("benchmark_family", "bounded")).strip() or "bounded",
            commands=commands,
            task_contract=task_contract,
            termination_reason=str(candidate.get("termination_reason", "")).strip(),
            quality=_quality_with_retrieval_bonus(
                float(candidate.get("quality", 0.78 if artifact_kind == "recovery_case" else 0.8) or 0.8),
                candidate,
            ),
            replay_hardening=replay_hardening,
            provenance=_retrieval_provenance_fields(candidate, commands),
        )
        if _procedure_has_transfer_potential(tool):
            tools.append(tool)
    return tools


def _tool_candidate_learning_source(
    candidate: dict[str, object],
    *,
    bank: TaskBank,
) -> tuple[str, dict[str, object]]:
    return _learning_candidate_source(candidate, bank=bank)


def _learning_candidate_source(
    candidate: dict[str, object],
    *,
    bank: TaskBank,
) -> tuple[str, dict[str, object]]:
    candidate_contract = (
        dict(candidate.get("task_contract", {}))
        if isinstance(candidate.get("task_contract", {}), dict)
        else {}
    )
    task_metadata = dict(candidate.get("task_metadata", {})) if isinstance(candidate.get("task_metadata", {}), dict) else {}
    contract_metadata = (
        dict(candidate_contract.get("metadata", {}))
        if isinstance(candidate_contract.get("metadata", {}), dict)
        else {}
    )
    aliases: list[str] = []
    for value in (
        candidate.get("source_task_id", ""),
        candidate.get("parent_task", ""),
        task_metadata.get("source_task", ""),
        task_metadata.get("parent_task", ""),
        contract_metadata.get("source_task", ""),
    ):
        token = str(value).strip()
        if token and token not in aliases:
            aliases.append(token)
    for alias in aliases:
        try:
            task = bank.get(alias)
        except KeyError:
            continue
        if candidate_contract and alias == str(candidate.get("source_task_id", "")).strip():
            return alias, candidate_contract
        if candidate_contract:
            rebased_contract = task.to_dict()
            rebased_metadata = (
                dict(rebased_contract.get("metadata", {}))
                if isinstance(rebased_contract.get("metadata", {}), dict)
                else {}
            )
            rebased_metadata.update(contract_metadata)
            if rebased_metadata:
                rebased_contract["metadata"] = rebased_metadata
            return alias, rebased_contract
        return alias, task.to_dict()
    return "", candidate_contract


def _procedure_matches_source_task(source_task: object, commands: list[str]) -> bool:
    normalized_commands = " ".join(str(command) for command in commands if str(command).strip())
    if not normalized_commands:
        return False
    expected_paths = [
        str(path)
        for path in [
            *getattr(source_task, "expected_files", []),
            *getattr(source_task, "expected_file_contents", {}).keys(),
        ]
        if str(path).strip()
    ]
    if any(path in normalized_commands for path in expected_paths):
        return True
    if expected_paths:
        return False
    success_command = str(getattr(source_task, "success_command", "")).strip().lower()
    if not success_command:
        return False
    return any(token in normalized_commands.lower() for token in _significant_tokens(success_command))


def _significant_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in (
        str(text)
        .lower()
        .replace("/", " ")
        .replace(".", " ")
        .replace("'", " ")
        .replace('"', " ")
        .split()
    ):
        normalized = token.strip()
        if len(normalized) >= 4 and normalized not in tokens:
            tokens.append(normalized)
    return tokens
