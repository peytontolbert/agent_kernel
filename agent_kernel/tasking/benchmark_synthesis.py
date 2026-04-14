from __future__ import annotations

import json
from pathlib import Path

from ..config import KernelConfig
from ..learning_compiler import load_learning_candidates
from ..memory import EpisodeMemory


def synthesize_benchmark_candidates(
    memory_root: Path,
    output_path: Path,
    *,
    limit: int = 20,
    focus: str | None = None,
) -> Path:
    config: KernelConfig | None = None
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and memory_root.resolve() == default_config.trajectories_root.resolve():
            config = default_config
    except OSError:
        config = None
    memory = EpisodeMemory(memory_root, config=config)
    proposals: list[dict[str, object]] = []
    for document in memory.list_documents():
        summary = document.get("summary", {})
        task_id = str(document.get("task_id", "")).strip()
        if not task_id:
            continue
        failure_types = [str(value) for value in summary.get("failure_types", [])]
        transition_failures = [str(value) for value in summary.get("transition_failures", [])]
        benchmark_family = str(document.get("task_metadata", {}).get("benchmark_family", "bounded"))
        if failure_types:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{task_id}:failure_cluster",
                    "source_task_id": task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "failure_cluster",
                    "failure_types": failure_types,
                    "required_properties": [
                        "deterministic verifier",
                        "explicit expected artifacts",
                        "localized failure mode pressure",
                    ],
                    "verifier_pressure": {
                        "target_failure_types": sorted(set(failure_types)),
                        "require_negative_case": True,
                    },
                    "prompt": (
                        f"Create a harder task derived from {task_id} that specifically discriminates against "
                        f"{', '.join(sorted(set(failure_types)))}"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
        if transition_failures:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{task_id}:transition_failure",
                    "source_task_id": task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "transition_failure",
                    "transition_failures": transition_failures,
                    "required_properties": [
                        "explicit intermediate-state regression pressure",
                        "verifier-visible recovery path",
                        "state-transition discrimination",
                    ],
                    "transition_pressure": {
                        "target_transition_failures": sorted(set(transition_failures)),
                        "require_observable_progress": True,
                    },
                    "prompt": (
                        f"Create a harder task derived from {task_id} that specifically discriminates against bad state "
                        f"transitions such as {', '.join(sorted(set(transition_failures)))}"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
        executed_commands = [str(value) for value in summary.get("executed_commands", [])]
        if len(executed_commands) >= 2:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{task_id}:environment_pattern",
                    "source_task_id": task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "environment_pattern",
                    "command_count": len(executed_commands),
                    "required_properties": [
                        "multi-step environment interaction",
                        "deterministic checks",
                        "artifact preservation constraints",
                    ],
                    "environment_contract": {
                        "min_command_count": len(executed_commands),
                        "require_setup_and_verification": True,
                    },
                    "prompt": (
                        f"Create a broader environment task derived from the successful multi-command pattern in {task_id}"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
        if (failure_types or transition_failures) and len(executed_commands) >= 2:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{task_id}:recovery_path",
                    "source_task_id": task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "recovery_path",
                    "failure_types": failure_types,
                    "transition_failures": transition_failures,
                    "command_count": len(executed_commands),
                    "required_properties": [
                        "recovery from a plausible bad intermediate state",
                        "verifier-preserving repair path",
                        "independently checkable completion",
                    ],
                    "prompt": (
                        f"Create a recovery-oriented task derived from {task_id} where the agent must repair a localized "
                        f"failure involving {', '.join(sorted(set(failure_types or transition_failures)))} while preserving the verifier contract"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
    learning_artifacts_path = (
        config.learning_artifacts_path
        if config is not None
        else memory_root.parent / "learning" / "run_learning_artifacts.json"
    )
    for candidate in load_learning_candidates(learning_artifacts_path, config=config):
        if str(candidate.get("artifact_kind", "")).strip() != "benchmark_gap":
            continue
        source_task_id = str(candidate.get("source_task_id", "")).strip()
        if not source_task_id:
            continue
        failure_types = [
            str(value)
            for value in candidate.get("failure_types", [])
            if str(value).strip()
        ]
        transition_failures = [
            str(value)
            for value in candidate.get("transition_failures", [])
            if str(value).strip()
        ]
        benchmark_family = str(candidate.get("benchmark_family", "bounded")).strip() or "bounded"
        gap_kind = str(candidate.get("gap_kind", "")).strip() or "failure_cluster"
        if gap_kind == "transition_pressure" and transition_failures:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{source_task_id}:learning_transition_gap",
                    "source_task_id": source_task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "transition_failure",
                    "transition_failures": transition_failures,
                    "required_properties": [
                        "explicit intermediate-state regression pressure",
                        "verifier-visible recovery path",
                        "state-transition discrimination",
                    ],
                    "transition_pressure": {
                        "target_transition_failures": sorted(set(transition_failures)),
                        "require_observable_progress": True,
                    },
                    "prompt": (
                        f"Create a harder task derived from {source_task_id} that specifically discriminates against bad state "
                        f"transitions such as {', '.join(sorted(set(transition_failures)))}"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
        elif failure_types:
            proposals.append(
                {
                    "proposal_id": f"benchmark:{source_task_id}:learning_failure_gap",
                    "source_task_id": source_task_id,
                    "benchmark_family": benchmark_family,
                    "kind": "failure_cluster",
                    "failure_types": failure_types,
                    "required_properties": [
                        "deterministic verifier",
                        "explicit expected artifacts",
                        "localized failure mode pressure",
                    ],
                    "verifier_pressure": {
                        "target_failure_types": sorted(set(failure_types)),
                        "require_negative_case": True,
                    },
                    "prompt": (
                        f"Create a harder task derived from {source_task_id} that specifically discriminates against "
                        f"{', '.join(sorted(set(failure_types)))}"
                        f"{_benchmark_focus_suffix(focus)}."
                    ),
                }
            )
    proposals = _sort_benchmark_proposals(proposals[:limit], focus=focus)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "benchmark_candidate_set",
        "lifecycle_state": "proposed",
        "retention_gate": {
            "min_pass_rate_delta_abs": 0.02,
            "require_family_discrimination": True,
            "max_false_failure_rate": 0.02,
            "max_regressed_families": 0,
            "required_confirmation_runs": 2,
        },
        "generation_focus": focus or "balanced",
        "proposals": proposals,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _benchmark_focus_suffix(focus: str | None) -> str:
    if focus == "confidence":
        return " under low-confidence retrieval conditions"
    if focus == "breadth":
        return " with broader environment branching and distractor structure"
    return ""


def _sort_benchmark_proposals(
    proposals: list[dict[str, object]],
    *,
    focus: str | None,
) -> list[dict[str, object]]:
    if focus == "confidence":
        priority = {"failure_cluster": 0, "recovery_path": 1, "environment_pattern": 2}
        return sorted(proposals, key=lambda proposal: (priority.get(str(proposal.get("kind", "")), 9), str(proposal.get("proposal_id", ""))))
    if focus == "breadth":
        priority = {"environment_pattern": 0, "recovery_path": 1, "failure_cluster": 2}
        return sorted(proposals, key=lambda proposal: (priority.get(str(proposal.get("kind", "")), 9), str(proposal.get("proposal_id", ""))))
    return proposals
