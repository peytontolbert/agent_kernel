from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any

from ..state import AgentState


def reusable_context_packet(policy: Any, state: AgentState):
    packet = state.context_packet
    if packet is None or not context_packet_reuse_enabled(state):
        return None
    control = packet.control if isinstance(packet.control, dict) else {}
    expected = context_reuse_signature(state)
    if str(control.get("context_reuse_signature", "")).strip() != expected:
        return None
    return packet


def stamp_context_reuse_signature(policy: Any, state: AgentState) -> None:
    del policy
    packet = state.context_packet
    if packet is None or not context_packet_reuse_enabled(state):
        return
    control = dict(packet.control) if isinstance(packet.control, dict) else {}
    control["context_reuse_signature"] = context_reuse_signature(state)
    packet.control = control


def context_packet_reuse_enabled(state: AgentState) -> bool:
    if not state.history:
        return False
    horizon = str(
        state.world_model_summary.get(
            "horizon",
            state.task.metadata.get("difficulty", state.task.metadata.get("horizon", "")),
        )
    ).strip()
    return horizon == "long_horizon"


def context_reuse_signature(state: AgentState) -> str:
    history_window = [
        {
            "index": int(step.index),
            "action": str(step.action),
            "content": str(step.content),
            "selected_retrieval_span_id": str(step.selected_retrieval_span_id or ""),
            "retrieval_influenced": bool(step.retrieval_influenced),
            "trust_retrieval": bool(step.trust_retrieval),
            "retrieval_command_match": bool(step.retrieval_command_match),
            "verification": dict(step.verification or {}),
            "command_result": dict(step.command_result or {}) if step.command_result else None,
        }
        for step in state.history[-3:]
    ]
    payload = {
        "task": asdict(state.task),
        "recent_workspace_summary": str(state.recent_workspace_summary),
        "world_model_summary": dict(state.world_model_summary or {}),
        "history_window": history_window,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def skill_is_safe_for_task(policy: Any, state: AgentState, skill: dict[str, object] | None) -> bool:
    if skill is None:
        return False
    curriculum_kind = str(state.task.metadata.get("curriculum_kind", "")).strip()
    if curriculum_kind != "failure_recovery":
        return True
    source_task_id = str(skill.get("source_task_id", skill.get("task_id", ""))).strip()
    source_task = str(state.task.metadata.get("source_task", "")).strip()
    if source_task_id and source_task_id in {state.task.task_id, source_task}:
        return True
    skill_family = str(skill.get("benchmark_family", "")).strip()
    task_family = str(state.task.metadata.get("benchmark_family", "")).strip()
    if skill_family and task_family and skill_family != task_family:
        return False
    commands = policy.skill_library._commands_for_skill(skill)
    if not commands:
        return False
    first_command = commands[0]
    anchors = task_command_anchors(state)
    if not anchors:
        return False
    return any(anchor in first_command for anchor in anchors)


def task_command_anchors(state: AgentState) -> list[str]:
    anchors: list[str] = []
    for path in [*state.task.expected_files, *state.task.expected_file_contents.keys()]:
        normalized = str(path).strip()
        if not normalized:
            continue
        for part in Path(normalized).parts:
            if part not in {"", "."} and part not in anchors:
                anchors.append(part)
    return anchors


def ranked_skill_summaries(policy: Any, state: AgentState, top_skill: dict[str, object] | None) -> list[dict[str, object]]:
    task_matches = policy.skill_library.matching_skills(state.task.task_id)
    if not task_matches:
        source_task_id = str(state.task.metadata.get("source_task", "")).strip()
        task_matches = policy.skill_library.matching_skills(
            state.task.task_id,
            source_task_id=source_task_id,
        )
    summaries = [policy.skill_library._to_summary(skill, state.task.task_id) for skill in task_matches]
    if top_skill is None:
        return summaries[:3]
    top_skill_id = str(top_skill.get("skill_id", ""))
    summaries.sort(key=lambda skill: (0 if str(skill.get("skill_id", "")) == top_skill_id else 1))
    return summaries[:3]


def preferred_task_ids(state: AgentState) -> list[str]:
    if state.context_packet is None:
        return []
    task_ids: list[str] = []
    for item in (
        state.context_packet.retrieval.get("branch_scoped", [])
        + state.context_packet.retrieval.get("fallback_scoped", [])
        + state.context_packet.retrieval.get("global", [])
    ):
        metadata = item.get("metadata") or {}
        task_id = str(metadata.get("task_id", "")).strip()
        if task_id and task_id not in task_ids:
            task_ids.append(task_id)
    return task_ids


def retrieval_guidance(state: AgentState) -> dict[str, list[str]]:
    if state.context_packet is None:
        return {
            "recommended_commands": [],
            "recommended_command_spans": [],
            "avoidance_notes": [],
            "evidence": [],
        }
    return state.context_packet.control.get("retrieval_guidance", {})


def blocked_commands(state: AgentState, *, avoidance_note_command_fn) -> list[str]:
    blocked: list[str] = []
    if state.context_packet is not None:
        for note in state.context_packet.control.get("retrieval_guidance", {}).get("avoidance_notes", []):
            command = avoidance_note_command_fn(note)
            if command:
                blocked.append(command)
    blocked.extend(sorted(state.all_failed_command_signatures()))
    return blocked


def avoidance_note_command(note: object) -> str | None:
    text = str(note).strip()
    prefix = "avoid repeating "
    if not text.startswith(prefix):
        return None
    literal = text[len(prefix) :].split(" when ", 1)[0].strip()
    if not literal:
        return None
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        match = re.match(r"""(['"])(.*)\1$""", literal)
        if match is None:
            return None
        value = match.group(2)
    return str(value).strip() or None


def has_retrieval_signal(state: AgentState) -> bool:
    if state.context_packet is None:
        return False
    guidance = state.context_packet.control.get("retrieval_guidance", {})
    if guidance.get("recommended_command_spans") or guidance.get("recommended_commands"):
        return True
    if guidance.get("avoidance_notes") or guidance.get("evidence"):
        return True
    for bucket in ("branch_scoped", "fallback_scoped", "global"):
        if state.context_packet.retrieval.get(bucket):
            return True
    return False


def compact_plan(policy: Any, plan: list[str]) -> list[str]:
    max_items = max(1, policy.config.llm_plan_max_items)
    return [policy._truncate_text(item) for item in plan[:max_items] if str(item).strip()]


def retrieval_plan(state: AgentState) -> dict[str, object]:
    if state.context_packet is None:
        return {
            "strategy": "synthesize",
            "trust_retrieval": False,
            "path_confidence": 0.0,
            "level_focus": "",
        }
    control = state.context_packet.control
    trust_retrieval = bool(control.get("trust_retrieval", False))
    path_confidence = float(control.get("path_confidence", 0.0))
    if trust_retrieval:
        strategy = "minimal_adaptation"
    elif path_confidence > 0.0:
        strategy = "guided_synthesis"
    else:
        strategy = "synthesize"
    return {
        "strategy": strategy,
        "trust_retrieval": trust_retrieval,
        "path_confidence": path_confidence,
        "level_focus": str(control.get("level_focus", "")),
        "acting_role": str(state.current_role or "executor"),
    }


def llm_context_packet(state: AgentState) -> dict[str, object] | None:
    if state.context_packet is None:
        return None
    control = state.context_packet.control
    retrieval = state.context_packet.retrieval
    max_chunks = int(control.get("context_chunk_budget", {}).get("max_chunks", 8) or 8)
    compact_control = {
        "mode": control.get("mode"),
        "level_focus": control.get("level_focus"),
        "path_confidence": control.get("path_confidence"),
        "trust_retrieval": control.get("trust_retrieval"),
        "selected_branch_level": control.get("selected_branch_level"),
        "branch_candidates": control.get("branch_candidates", [])[:3],
        "context_chunk_budget": control.get("context_chunk_budget", {}),
        "selected_context_chunks": control.get("selected_context_chunks", [])[:max_chunks],
        "retrieval_guidance": control.get("retrieval_guidance", {}),
    }
    compact_retrieval = {
        bucket: [
            {
                "span_id": str(item.get("span_id", "")),
                "span_type": str(item.get("span_type", "")),
                "score": float(item.get("score", 0.0)),
                "metadata": dict(item.get("metadata", {})),
            }
            for item in retrieval.get(bucket, [])[:3]
        ]
        for bucket in ("branch_scoped", "fallback_scoped", "global")
    }
    return {
        "task": dict(state.context_packet.task),
        "control": compact_control,
        "tolbert": dict(state.context_packet.tolbert),
        "retrieval": compact_retrieval,
        "verifier_contract": dict(state.context_packet.verifier_contract),
    }
