from __future__ import annotations

import json
import re
from typing import Any


def learned_world_progress_payload(latent_state_summary: dict[str, object]) -> dict[str, object]:
    learned = latent_state_summary.get("learned_world_state", {})
    if not isinstance(learned, dict):
        return {}
    controller = learned.get("controller_belief", {})
    if not isinstance(controller, dict):
        controller = {}
    top_states = learned.get("controller_expected_world_top_states", [])
    if not isinstance(top_states, list):
        top_states = []
    top_state_probs = learned.get("controller_expected_world_top_state_probs", [])
    if not isinstance(top_state_probs, list):
        top_state_probs = []
    return {
        "learned_world_source": str(learned.get("source", "")).strip(),
        "learned_world_controller_mode": str(learned.get("controller_mode", "")).strip(),
        "learned_world_controller_mode_probability": float(
            learned.get("controller_mode_probability", 0.0) or 0.0
        ),
        "learned_world_controller_belief": {
            "recover": float(controller.get("recover", 0.0) or 0.0),
            "continue": float(controller.get("continue", 0.0) or 0.0),
            "stop": float(controller.get("stop", 0.0) or 0.0),
        },
        "learned_world_expected_top_state": int(top_states[0]) if top_states else -1,
        "learned_world_expected_top_state_probability": float(top_state_probs[0] or 0.0)
        if top_state_probs
        else 0.0,
    }


def attach_critic_subgoal_diagnoses(
    kernel: Any,
    state,
    *,
    step_index: int,
    step_active_subgoal: str,
    failure_signals: list[str],
    failure_origin: str,
    command_result,
) -> None:
    ordered_signals = kernel._ordered_failure_signals(
        failure_signals=failure_signals,
        failure_origin=failure_origin,
        command_result=command_result,
    )
    if not ordered_signals:
        return
    for goal in kernel._diagnosis_candidate_subgoals(state, step_active_subgoal=step_active_subgoal):
        diagnosis = kernel._build_subgoal_failure_diagnosis(
            state,
            goal=goal,
            step_index=step_index,
            ordered_signals=ordered_signals,
            command_result=command_result,
            step_active_subgoal=step_active_subgoal,
        )
        if not diagnosis:
            continue
        existing = state.diagnosis_for_subgoal(goal)
        merged_signals: list[str] = []
        for signal in [*existing.get("signals", []), *diagnosis.get("signals", [])]:
            normalized = str(signal).strip()
            if normalized and normalized not in merged_signals:
                merged_signals.append(normalized)
        diagnosis["signals"] = merged_signals
        state.subgoal_diagnoses[str(goal).strip()] = diagnosis


def attach_verifier_subgoal_diagnoses(
    kernel: Any,
    state,
    *,
    step_index: int,
    verification_reasons: list[object],
) -> None:
    for entry in kernel._verifier_failure_entries(verification_reasons):
        goal = str(entry.get("subgoal", "")).strip()
        if not goal:
            continue
        existing = state.diagnosis_for_subgoal(goal)
        summary_parts: list[str] = []
        for candidate in [existing.get("summary", ""), entry.get("summary", "")]:
            normalized = str(candidate).strip()
            if normalized and normalized not in summary_parts:
                summary_parts.append(normalized)
        signals: list[str] = []
        for signal in [*existing.get("signals", []), "verifier_failure"]:
            normalized = str(signal).strip()
            if normalized and normalized not in signals:
                signals.append(normalized)
        path = str(existing.get("path", "")).strip() or str(entry.get("path", "")).strip() or kernel._subgoal_path(goal)
        expected_content = _expected_content_for_path(state, path)
        repair_instruction = str(entry.get("repair_instruction", "")).strip()
        if path == "patch.diff" and repair_instruction:
            candidate_brief = _swe_candidate_path_repair_brief(state)
            if candidate_brief and candidate_brief not in repair_instruction:
                repair_instruction = f"{repair_instruction}; {candidate_brief}"
        if repair_instruction and repair_instruction not in summary_parts:
            summary_parts.append(repair_instruction)
        if expected_content is not None:
            expected_summary = _expected_content_repair_summary(path, expected_content)
            if expected_summary and expected_summary not in summary_parts:
                summary_parts.append(expected_summary)
        diagnosis: dict[str, object] = {
            "summary": "; ".join(summary_parts[:3]),
            "signals": signals,
            "path": path,
            "source_role": "verifier",
            "updated_step_index": step_index,
        }
        if expected_content is not None:
            diagnosis["expected_content"] = expected_content
            diagnosis["expected_content_preview"] = _expected_content_preview(expected_content)
        if repair_instruction:
            diagnosis["repair_instruction"] = repair_instruction
        state.subgoal_diagnoses[goal] = {
            **diagnosis,
        }


def _expected_content_for_path(state, path: str) -> str | None:
    normalized_path = str(path).strip()
    if not normalized_path:
        return None
    task = getattr(state, "task", None)
    expected_contents = getattr(task, "expected_file_contents", {}) if task is not None else {}
    if not isinstance(expected_contents, dict) or normalized_path not in expected_contents:
        return None
    return str(expected_contents.get(normalized_path, ""))


def _swe_candidate_path_repair_brief(state) -> str:
    task = getattr(state, "task", None)
    metadata = getattr(task, "metadata", {}) if task is not None else {}
    if not isinstance(metadata, dict):
        return ""
    candidate_files = [
        str(path).strip()
        for path in metadata.get("swe_candidate_files", [])
        if str(path).strip()
    ]
    if not candidate_files:
        verifier = metadata.get("semantic_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, dict) else {}
        candidate_files = [
            str(path).strip()
            for path in verifier.get("expected_changed_paths", [])
            if str(path).strip()
        ]
    if not candidate_files:
        return ""
    anchor_preview = _swe_candidate_line_anchor_preview(metadata, candidate_files)
    brief = (
        "allowed patch paths are "
        f"{', '.join(candidate_files)}; inspect source_lines/<path>.lines for exact line anchors "
        "and write hunks only against those paths"
    )
    if anchor_preview:
        brief = f"{brief}; candidate line anchors: {anchor_preview}"
    return brief


def _swe_candidate_line_anchor_preview(metadata: dict[str, object], candidate_files: list[str]) -> str:
    setup_files = metadata.get("setup_file_contents", {})
    if not isinstance(setup_files, dict):
        return ""
    chunks: list[str] = []
    for path in candidate_files[:2]:
        line_path = f"source_lines/{path}.lines"
        content = str(setup_files.get(line_path, "")).strip()
        if not content:
            continue
        lines = [line for line in content.splitlines()[:18] if line.strip()]
        if not lines:
            continue
        preview = " | ".join(lines[:12])
        chunks.append(f"{path}: {preview}")
    return " || ".join(chunks)


def _expected_content_preview(expected_content: str, *, limit: int = 800) -> str:
    encoded = json.dumps(str(expected_content))
    if len(encoded) <= limit:
        return encoded
    return encoded[: max(0, limit - 3)].rstrip() + "..."


def _expected_content_repair_summary(path: str, expected_content: str) -> str:
    normalized_path = str(path).strip()
    if not normalized_path:
        return ""
    preview = _expected_content_preview(expected_content)
    return f"rewrite {normalized_path} to exact expected content {preview}"


def diagnosis_candidate_subgoals(kernel: Any, state, *, step_active_subgoal: str) -> list[str]:
    candidates: list[str] = []
    for goal in [step_active_subgoal, state.active_subgoal, *kernel._learned_world_hotspot_subgoals(state), *state.plan]:
        normalized = str(goal).strip()
        if not normalized or normalized in candidates:
            continue
        candidates.append(normalized)
    return candidates[:6]


def build_subgoal_failure_diagnosis(
    kernel: Any,
    state,
    *,
    goal: str,
    step_index: int,
    ordered_signals: list[str],
    command_result,
    step_active_subgoal: str,
) -> dict[str, object]:
    normalized_goal = str(goal).strip()
    if not normalized_goal:
        return {}
    path = kernel._subgoal_path(normalized_goal)
    regressed_paths = {
        str(item).strip() for item in state.latest_state_transition.get("regressions", []) if str(item).strip()
    }
    goal_signals: list[str] = []
    for signal in ordered_signals:
        normalized_signal = str(signal).strip()
        if not normalized_signal:
            continue
        if normalized_signal in {"command_failure", "command_timeout", "inference_failure", "retrieval_failure"}:
            if normalized_goal != str(step_active_subgoal).strip():
                continue
        if normalized_signal == "state_regression":
            if regressed_paths and path and path not in regressed_paths:
                continue
        if normalized_signal not in goal_signals:
            goal_signals.append(normalized_signal)
    evidence: list[str] = []
    if path:
        if normalized_goal.startswith("remove forbidden artifact "):
            present = {str(item).strip() for item in state.world_model_summary.get("present_forbidden_artifacts", [])}
            if path in present:
                evidence.append(f"{path} is still present")
        elif normalized_goal.startswith("materialize expected artifact "):
            missing = {str(item).strip() for item in state.world_model_summary.get("missing_expected_artifacts", [])}
            unsatisfied = {
                str(item).strip() for item in state.world_model_summary.get("unsatisfied_expected_contents", [])
            }
            if path in missing:
                evidence.append(f"{path} is still missing")
            if path in unsatisfied:
                evidence.append(f"{path} content is still unsatisfied")
        elif normalized_goal.startswith("preserve required artifact "):
            changed = {str(item).strip() for item in state.world_model_summary.get("changed_preserved_artifacts", [])}
            if path in changed:
                evidence.append(f"{path} changed unexpectedly")
    if not evidence and path:
        evidence.append(f"{path} remains on the critical path")
    summary = kernel._subgoal_diagnosis_summary(goal_signals, command_result=command_result)
    if evidence and summary:
        summary = f"{'; '.join(evidence)}; {summary}"
    elif evidence:
        summary = "; ".join(evidence)
    return {
        "summary": summary,
        "signals": list(goal_signals),
        "path": path,
        "source_role": "critic",
        "updated_step_index": step_index,
    }


def verifier_hotspot_subgoals(kernel: Any, verification_reasons: list[object]) -> list[str]:
    hotspots: list[str] = []
    for entry in kernel._verifier_failure_entries(verification_reasons):
        goal = str(entry.get("subgoal", "")).strip()
        if goal and goal not in hotspots:
            hotspots.append(goal)
    return hotspots


def verifier_failure_entries(verification_reasons: list[object]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for reason in verification_reasons:
        text = str(reason).strip()
        if not text or text == "verification passed":
            continue
        subgoal = ""
        path = ""
        if text.startswith("missing expected file: "):
            path = text.removeprefix("missing expected file: ").strip()
            subgoal = f"materialize expected artifact {path}"
        elif text.startswith("missing expected file content target: "):
            path = text.removeprefix("missing expected file content target: ").strip()
            subgoal = f"materialize expected artifact {path}"
        elif text.startswith("unexpected file content: "):
            path = text.removeprefix("unexpected file content: ").strip()
            subgoal = f"materialize expected artifact {path}"
        elif text.startswith("SWE patch apply check failed: "):
            path = "patch.diff"
            subgoal = "repair SWE patch.diff until it applies to the base commit"
        elif text.startswith("SWE patch verifier missing patch file: "):
            path = text.removeprefix("SWE patch verifier missing patch file: ").strip() or "patch.diff"
            subgoal = f"materialize expected artifact {path}"
        elif text.startswith("SWE patch diff includes unexpected path: "):
            path = "patch.diff"
            subgoal = "rewrite SWE patch.diff using only listed likely relevant files"
        elif text == "SWE patch diff contains placeholder/template content":
            path = "patch.diff"
            subgoal = "replace template SWE patch.diff with a source-grounded unified diff"
        elif text == "SWE patch diff has no meaningful content change":
            path = "patch.diff"
            subgoal = "replace no-op SWE patch.diff with a behavior-changing unified diff"
        elif text == "SWE patch diff has no changed file paths":
            path = "patch.diff"
            subgoal = "rewrite SWE patch.diff with real changed file paths"
        elif text.startswith("forbidden file present: "):
            path = text.removeprefix("forbidden file present: ").strip()
            subgoal = f"remove forbidden artifact {path}"
        elif text.startswith("git diff missing expected path: "):
            path = text.removeprefix("git diff missing expected path: ").strip()
            subgoal = f"update workflow path {path}"
        elif text.startswith("git diff unexpectedly changed preserved path: "):
            path = text.removeprefix("git diff unexpectedly changed preserved path: ").strip()
            subgoal = f"preserve required artifact {path}"
        elif text.startswith("semantic report missing: "):
            path = text.removeprefix("semantic report missing: ").strip()
            subgoal = f"write workflow report {path}"
        elif text.startswith("semantic report missing phrase "):
            path = text.rsplit(": ", 1)[-1].strip()
            subgoal = f"write workflow report {path}"
        elif text.startswith("semantic report does not cover "):
            path = text.rsplit(": ", 1)[-1].strip()
            subgoal = f"write workflow report {path}"
        elif text.startswith("generated artifact missing: "):
            path = text.removeprefix("generated artifact missing: ").strip()
            subgoal = f"regenerate generated artifact {path}"
        elif text.startswith("generated artifact not recorded in git diff: "):
            path = text.removeprefix("generated artifact not recorded in git diff: ").strip()
            subgoal = f"regenerate generated artifact {path}"
        elif text.startswith("git conflict remains unresolved: "):
            path = text.removeprefix("git conflict remains unresolved: ").strip()
            subgoal = f"update workflow path {path}"
        elif text.startswith("conflict markers still present after merge resolution: "):
            path = text.removeprefix("conflict markers still present after merge resolution: ").strip()
            subgoal = f"update workflow path {path}"
        elif text.startswith("required worker branch not accepted into "):
            branch = text.rsplit(": ", 1)[-1].strip()
            path = branch
            subgoal = f"accept required branch {branch}"
        else:
            match = re.fullmatch(r"(.+?) exited with code \d+", text)
            if match is not None:
                label = str(match.group(1)).strip()
                if label:
                    path = label
                    subgoal = f"run workflow test {label}"
        if subgoal:
            entry = {"subgoal": subgoal, "summary": text, "path": path}
            if text.startswith("missing expected file: ") and path == "patch.diff":
                entry["repair_instruction"] = (
                    "write patch.diff now as a source-grounded unified diff; "
                    "do not repeat source inspection commands, do not run ls/git/find, and do not write template content"
                )
            elif text.startswith("SWE patch verifier missing patch file: "):
                entry["repair_instruction"] = (
                    "write patch.diff now as a source-grounded unified diff using the provided candidate source files; "
                    "do not repeat cat/ls discovery and do not invent placeholder hunks"
                )
            elif text.startswith("SWE patch apply check failed: "):
                entry["repair_instruction"] = (
                    "rewrite patch.diff as a real unified diff that applies cleanly to the base commit; "
                    "use exact file paths and context from source excerpts; do not invent files or placeholder hunks"
                )
            elif text.startswith("SWE patch diff includes unexpected path: "):
                entry["repair_instruction"] = (
                    "rewrite patch.diff to change only the listed likely relevant files; "
                    "inspect source_context paths and do not invent alternate repository paths"
                )
            elif text.startswith("SWE patch diff has no meaningful content change"):
                entry["repair_instruction"] = (
                    "replace patch.diff with a behavior-changing unified diff that addresses the issue; "
                    "do not remove and re-add identical lines, and use exact context from source_lines"
                )
            elif text.startswith("SWE patch diff contains placeholder/template content") or text.startswith(
                "SWE patch diff has no changed file paths"
            ):
                entry["repair_instruction"] = (
                    "replace patch.diff with a source-grounded unified diff using exact context from source_context; "
                    "remove template comments, fake imports, placeholder functions, and invented hunks"
                )
            entries.append(entry)
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry["subgoal"], entry["summary"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def ordered_failure_signals(
    *,
    failure_signals: list[str],
    failure_origin: str,
    command_result,
) -> list[str]:
    ordered: list[str] = []

    def append(signal: object) -> None:
        normalized = str(signal).strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    for signal in failure_signals:
        append(signal)
    append(failure_origin)
    if command_result is not None:
        if bool(getattr(command_result, "timed_out", False)):
            append("command_timeout")
        elif int(getattr(command_result, "exit_code", 0)) != 0:
            append("command_failure")
    return ordered


def subgoal_diagnosis_summary(ordered_signals: list[str], *, command_result) -> str:
    parts: list[str] = []
    for signal in ordered_signals:
        if signal == "state_regression":
            parts.append("recent step regressed workspace state")
        elif signal == "no_state_progress":
            parts.append("recent step produced no state progress")
        elif signal == "command_timeout":
            parts.append("recent command timed out")
        elif signal == "command_failure":
            exit_code = int(getattr(command_result, "exit_code", 1)) if command_result is not None else 1
            parts.append(f"recent command exited {exit_code}")
        elif signal == "inference_failure":
            parts.append("recent policy inference failed")
        elif signal == "retrieval_failure":
            parts.append("recent retrieval guidance failed")
        elif signal:
            parts.append(signal.replace("_", " "))
    deduped: list[str] = []
    for part in parts:
        normalized = str(part).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return "; ".join(deduped[:3])


def subgoal_diagnosis_priority(diagnosis: dict[str, object]) -> int:
    signals = diagnosis.get("signals", []) if isinstance(diagnosis, dict) else []
    weights = {
        "state_regression": 5,
        "verifier_failure": 4,
        "command_timeout": 4,
        "command_failure": 3,
        "inference_failure": 3,
        "retrieval_failure": 3,
        "no_state_progress": 2,
    }
    return max((weights.get(str(signal).strip(), 1) for signal in signals), default=0)
