from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .improvement_catalog import catalog_string_set
from .improvement_common import (
    build_standard_proposal_artifact,
    ensure_proposals,
    normalized_generation_focus,
    retained_mapping_section,
    retained_sequence_section,
    retention_gate_preset,
)


_TRANSITION_SIGNALS = catalog_string_set("transition_model", "signals")
_TRANSITION_MODEL_GENERATION_FOCI = catalog_string_set("transition_model", "generation_foci")
_SHELL_TOKEN_RE = re.compile(r"""'[^']*'|"[^"]*"|&&|\|\||>>|<<|[|;<>]|[^\s]+""")
_PATH_LIKE_SUFFIXES = catalog_string_set("transition_model", "path_like_suffixes")
_PATH_TARGET_COMMANDS = catalog_string_set("transition_model", "path_target_commands")


def transition_model_controls(
    summary: dict[str, object],
    *,
    focus: str | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    signal_counts = summary.get("signal_counts", {}) if isinstance(summary.get("signal_counts", {}), dict) else {}
    regression_evidence = int(signal_counts.get("state_regression", 0)) > 0 or int(summary.get("regression_path_count", 0) or 0) > 0
    difficulty_counts = summary.get("difficulty_counts", {}) if isinstance(summary.get("difficulty_counts", {}), dict) else {}
    long_horizon_evidence = (
        int(difficulty_counts.get("long_horizon", 0) or 0) > 0
        or int(summary.get("long_horizon_signature_count", 0) or 0) > 0
    )
    controls: dict[str, object] = {
        "repeat_command_penalty": 4,
        "regressed_path_command_penalty": 3,
        "recovery_command_bonus": 2,
        "progress_command_bonus": 2,
        "long_horizon_repeat_command_penalty": 1,
        "long_horizon_progress_command_bonus": 1,
        "max_signatures": 12,
    }
    if isinstance(baseline, dict):
        controls.update(deepcopy(baseline))
    if int(signal_counts.get("no_state_progress", 0)) > 0:
        controls["repeat_command_penalty"] = max(int(controls["repeat_command_penalty"]), 5)
        controls["progress_command_bonus"] = max(int(controls["progress_command_bonus"]), 3)
    if regression_evidence:
        controls["regressed_path_command_penalty"] = max(int(controls["regressed_path_command_penalty"]), 4)
        controls["recovery_command_bonus"] = max(int(controls["recovery_command_bonus"]), 3)
    if focus == "repeat_avoidance":
        controls["repeat_command_penalty"] = max(int(controls["repeat_command_penalty"]), 6)
    elif focus == "regression_guard" and regression_evidence:
        controls["regressed_path_command_penalty"] = max(int(controls["regressed_path_command_penalty"]), 5)
    elif focus == "recovery_bias":
        controls["recovery_command_bonus"] = max(int(controls["recovery_command_bonus"]), 4)
        controls["progress_command_bonus"] = max(int(controls["progress_command_bonus"]), 4)
    if long_horizon_evidence:
        controls["long_horizon_repeat_command_penalty"] = max(int(controls["long_horizon_repeat_command_penalty"]), 3)
        controls["long_horizon_progress_command_bonus"] = max(int(controls["long_horizon_progress_command_bonus"]), 2)
    try:
        controls["max_signatures"] = max(1, int(controls.get("max_signatures", 12)))
    except (TypeError, ValueError):
        controls["max_signatures"] = 12
    return controls


def build_transition_model_proposal_artifact(
    memory_root: Path,
    *,
    focus: str | None = None,
    current_payload: object | None = None,
) -> dict[str, object]:
    generation_focus = normalized_generation_focus(focus)
    baseline_controls = runtime_transition_model_controls(current_payload)
    summary = transition_model_summary(memory_root)
    controls = transition_model_controls(
        summary,
        focus=None if generation_focus == "balanced" else generation_focus,
        baseline=baseline_controls,
    )
    signatures = _merged_transition_signatures(
        runtime_transition_model_signatures(current_payload),
        transition_failure_signatures(
            memory_root,
            max_signatures=int(controls.get("max_signatures", 12)),
        ),
        max_signatures=int(controls.get("max_signatures", 12)),
    )
    return build_standard_proposal_artifact(
        artifact_kind="transition_model_policy_set",
        generation_focus=generation_focus,
        control_schema="transition_model_controls_v1",
        retention_gate=retention_gate_preset("transition_model"),
        controls=controls,
        proposals=_proposals(summary, generation_focus),
        extra_sections={
            "transition_summary": summary,
            "signatures": signatures,
        },
    )


def retained_transition_model_controls(payload: object) -> dict[str, object]:
    controls = retained_mapping_section(payload, artifact_kind="transition_model_policy_set", section="controls")
    return _normalized_transition_model_controls(controls)


def runtime_transition_model_controls(payload: object) -> dict[str, object]:
    effective = _runtime_transition_model_payload(payload)
    if effective is None:
        return {}
    return _normalized_transition_model_controls(effective.get("controls", {}))


def _normalized_transition_model_controls(controls: object) -> dict[str, object]:
    if not isinstance(controls, dict):
        return {}
    normalized: dict[str, object] = {}
    for key in (
        "repeat_command_penalty",
        "regressed_path_command_penalty",
        "recovery_command_bonus",
        "progress_command_bonus",
        "long_horizon_repeat_command_penalty",
        "long_horizon_progress_command_bonus",
        "max_signatures",
    ):
        if key not in controls:
            continue
        try:
            normalized[key] = max(1, int(controls[key]))
        except (TypeError, ValueError):
            continue
    return normalized


def retained_transition_model_signatures(payload: object) -> list[dict[str, object]]:
    return _normalize_signatures(
        retained_sequence_section(payload, artifact_kind="transition_model_policy_set", section="signatures")
    )


def runtime_transition_model_signatures(payload: object) -> list[dict[str, object]]:
    effective = _runtime_transition_model_payload(payload)
    if effective is None:
        return []
    return _normalize_signatures(effective.get("signatures", []))


def _runtime_transition_model_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    if str(payload.get("artifact_kind", "")).strip() != "transition_model_policy_set":
        return None
    retention_decision = payload.get("retention_decision", {})
    if isinstance(retention_decision, dict) and str(retention_decision.get("state", "")).strip() == "reject":
        return None
    if str(payload.get("lifecycle_state", "")).strip() == "rejected":
        return None
    return payload


def transition_model_summary(memory_root: Path) -> dict[str, object]:
    documents = _list_documents(memory_root)
    signal_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    unique_commands: set[str] = set()
    regression_paths: set[str] = set()
    long_horizon_signature_count = 0
    for document in documents:
        difficulty = _document_difficulty(document)
        if difficulty:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        summary = document.get("summary", {})
        if isinstance(summary, dict):
            for signal in summary.get("transition_failures", []):
                normalized = str(signal).strip()
                if normalized in _TRANSITION_SIGNALS:
                    signal_counts[normalized] = signal_counts.get(normalized, 0) + 1
        for signature in _document_transition_signatures(document):
            signal = str(signature.get("signal", "")).strip()
            if signal:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            if str(signature.get("difficulty", "")).strip() == "long_horizon":
                long_horizon_signature_count += 1
            command = str(signature.get("command", "")).strip()
            if command:
                unique_commands.add(command)
            for path in signature.get("regressions", []):
                normalized_path = str(path).strip()
                if normalized_path:
                    regression_paths.add(normalized_path)
    return {
        "document_count": len(documents),
        "signal_counts": signal_counts,
        "difficulty_counts": difficulty_counts,
        "signature_command_count": len(unique_commands),
        "regression_path_count": len(regression_paths),
        "long_horizon_signature_count": long_horizon_signature_count,
    }


def transition_failure_signatures(memory_root: Path, *, max_signatures: int = 12) -> list[dict[str, object]]:
    aggregated: dict[tuple[str, str, str], dict[str, object]] = {}
    for document in _list_documents(memory_root):
        for signature in _document_transition_signatures(document):
            signal = str(signature.get("signal", "")).strip()
            command = _canonicalize_command(str(signature.get("command", "")))
            if signal not in _TRANSITION_SIGNALS or not command:
                continue
            command_pattern = transition_model_command_pattern(command) or command
            benchmark_family = str(signature.get("benchmark_family", "")).strip()
            difficulty = str(signature.get("difficulty", "")).strip()
            key = (signal, command_pattern, benchmark_family, difficulty)
            entry = aggregated.setdefault(
                key,
                {
                    "signature_id": _signature_id(signal, command_pattern),
                    "signal": signal,
                    "command": command,
                    "command_pattern": command_pattern,
                    "benchmark_family": benchmark_family,
                    "difficulty": difficulty,
                    "support": 0,
                    "regressions": [],
                    "touched_paths": [],
                    "patch_summaries": [],
                    "_representative_support": 0,
                },
            )
            signature_support = max(1, int(signature.get("support", 1)))
            entry["support"] = int(entry.get("support", 0)) + signature_support
            representative_command = str(entry.get("command", "")).strip()
            representative_support = int(entry.get("_representative_support", 0) or 0)
            if (
                not representative_command
                or signature_support > representative_support
                or (signature_support == representative_support and command < representative_command)
            ):
                entry["command"] = command
                entry["_representative_support"] = signature_support
            known_regressions = {str(path).strip() for path in entry.get("regressions", []) if str(path).strip()}
            for path in signature.get("regressions", []):
                normalized = str(path).strip()
                if normalized:
                    known_regressions.add(normalized)
            entry["regressions"] = sorted(known_regressions)
            known_touched_paths = {str(path).strip() for path in entry.get("touched_paths", []) if str(path).strip()}
            for path in signature.get("touched_paths", []):
                normalized = str(path).strip()
                if normalized:
                    known_touched_paths.add(normalized)
            entry["touched_paths"] = sorted(known_touched_paths)
            known_patch_summaries = {
                str(value).strip() for value in entry.get("patch_summaries", []) if str(value).strip()
            }
            for value in signature.get("patch_summaries", []):
                normalized = str(value).strip()
                if normalized:
                    known_patch_summaries.add(normalized)
            entry["patch_summaries"] = sorted(known_patch_summaries)[:4]
    signatures = []
    for entry in aggregated.values():
        entry.pop("_representative_support", None)
        signatures.append(entry)
    signatures = sorted(
        signatures,
        key=lambda item: (-int(item.get("support", 0)), str(item.get("signal", "")), str(item.get("command", ""))),
    )
    return signatures[: max(1, int(max_signatures))]


def _document_transition_signatures(document: dict[str, object]) -> list[dict[str, object]]:
    fragments = document.get("fragments", [])
    signatures: list[dict[str, object]] = []
    benchmark_family = _document_benchmark_family(document)
    difficulty = _document_difficulty(document)
    if isinstance(fragments, list) and fragments:
        commands_by_step: dict[int, str] = {}
        regressions_by_step: dict[int, list[str]] = {}
        touched_paths_by_step: dict[int, list[str]] = {}
        patch_summaries_by_step: dict[int, list[str]] = {}
        signals_by_step: dict[int, set[str]] = {}
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            try:
                step_index = int(fragment.get("step_index", 0))
            except (TypeError, ValueError):
                step_index = 0
            if step_index <= 0:
                continue
            kind = str(fragment.get("kind", "")).strip()
            if kind == "command":
                command = _canonicalize_command(str(fragment.get("command", "")))
                if command:
                    commands_by_step[step_index] = command
            elif kind == "command_outcome":
                command = _canonicalize_command(str(fragment.get("command", "")))
                if command:
                    commands_by_step[step_index] = command
                progress_delta = float(fragment.get("progress_delta", 0.0) or 0.0)
                if progress_delta <= 0.0:
                    failure_signals = {
                        str(signal).strip()
                        for signal in fragment.get("failure_signals", [])
                        if str(signal).strip() in _TRANSITION_SIGNALS
                    }
                    if failure_signals:
                        signals_by_step.setdefault(step_index, set()).update(failure_signals)
                    else:
                        signals_by_step.setdefault(step_index, set()).add("no_state_progress")
            elif kind == "failure":
                for signal in fragment.get("failure_signals", []):
                    normalized = str(signal).strip()
                    if normalized in _TRANSITION_SIGNALS:
                        signals_by_step.setdefault(step_index, set()).add(normalized)
            elif kind == "state_transition":
                regressions = [
                    str(path).strip()
                    for path in fragment.get("regressions", [])
                    if str(path).strip()
                ]
                if regressions:
                    regressions_by_step[step_index] = regressions
                progress_delta = float(fragment.get("progress_delta", 0.0) or 0.0)
                if progress_delta <= 0.0 and not regressions:
                    signals_by_step.setdefault(step_index, set()).add("no_state_progress")
                if regressions:
                    signals_by_step.setdefault(step_index, set()).add("state_regression")
            elif kind == "edit_patch":
                path = str(fragment.get("path", "")).strip()
                if path:
                    touched = touched_paths_by_step.setdefault(step_index, [])
                    if path not in touched:
                        touched.append(path)
                patch_summary = str(fragment.get("patch_summary", "")).strip() or _patch_excerpt(
                    str(fragment.get("patch", "")).strip()
                )
                if patch_summary:
                    summaries = patch_summaries_by_step.setdefault(step_index, [])
                    if patch_summary not in summaries:
                        summaries.append(patch_summary)
            elif kind == "recovery_trace":
                failed_command = _canonicalize_command(str(fragment.get("failed_command", "")))
                failed_step_index = fragment.get("failed_step_index")
                try:
                    failed_step = int(failed_step_index)
                except (TypeError, ValueError):
                    failed_step = 0
                if failed_step > 0 and failed_command:
                    commands_by_step.setdefault(failed_step, failed_command)
                    recovery_signals = {
                        str(signal).strip()
                        for signal in fragment.get("failure_signals", [])
                        if str(signal).strip() in _TRANSITION_SIGNALS
                    }
                    if recovery_signals:
                        signals_by_step.setdefault(failed_step, set()).update(recovery_signals)
        for step_index, command in commands_by_step.items():
            for signal in sorted(signals_by_step.get(step_index, set())):
                signatures.append(
                    {
                        "signal": signal,
                        "command": command,
                        "benchmark_family": benchmark_family,
                        "difficulty": difficulty,
                        "regressions": regressions_by_step.get(step_index, []),
                        "touched_paths": touched_paths_by_step.get(step_index, []),
                        "patch_summaries": patch_summaries_by_step.get(step_index, []),
                        "support": 1,
                    }
                )
    if signatures:
        return signatures
    summary = document.get("summary", {})
    if not isinstance(summary, dict):
        return []
    executed_commands = [
        _canonicalize_command(str(command))
        for command in summary.get("executed_commands", [])
        if _canonicalize_command(str(command))
    ]
    fallback_command = executed_commands[-1] if executed_commands else ""
    fallback_signatures: list[dict[str, object]] = []
    for signal in summary.get("transition_failures", []):
        normalized = str(signal).strip()
        if normalized not in _TRANSITION_SIGNALS or not fallback_command:
            continue
        fallback_signatures.append(
            {
                "signal": normalized,
                "command": fallback_command,
                "benchmark_family": benchmark_family,
                "difficulty": difficulty,
                "regressions": [],
                "support": 1,
            }
        )
    return fallback_signatures


def _list_documents(memory_root: Path) -> list[dict[str, object]]:
    if not memory_root.exists():
        return []
    documents: list[dict[str, object]] = []
    for path in sorted(memory_root.glob("*.json")):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return documents


def _normalize_signatures(signatures: object) -> list[dict[str, object]]:
    if not isinstance(signatures, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in signatures:
        if not isinstance(item, dict):
            continue
        signal = str(item.get("signal", "")).strip()
        command = _canonicalize_command(str(item.get("command", "")))
        if signal not in _TRANSITION_SIGNALS or not command:
            continue
        raw_command_pattern = _canonicalize_command(str(item.get("command_pattern", "")))
        command_pattern = raw_command_pattern or transition_model_command_pattern(command) or command
        regressions = sorted({str(path).strip() for path in item.get("regressions", []) if str(path).strip()})
        touched_paths = sorted({str(path).strip() for path in item.get("touched_paths", []) if str(path).strip()})
        patch_summaries = sorted(
            {str(value).strip() for value in item.get("patch_summaries", []) if str(value).strip()}
        )
        try:
            support = max(1, int(item.get("support", 1)))
        except (TypeError, ValueError):
            support = 1
        normalized.append(
            {
                "signature_id": str(item.get("signature_id", _signature_id(signal, command))).strip(),
                "signal": signal,
                "command": command,
                "command_pattern": command_pattern,
                "benchmark_family": str(item.get("benchmark_family", "")).strip(),
                "difficulty": str(item.get("difficulty", "")).strip(),
                "support": support,
                "regressions": regressions,
                "touched_paths": touched_paths,
                "patch_summaries": patch_summaries[:4],
            }
        )
    return normalized


def _merged_transition_signatures(
    baseline: list[dict[str, object]],
    extracted: list[dict[str, object]],
    *,
    max_signatures: int,
) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for source in (baseline, extracted):
        for signature in _normalize_signatures(source):
            key = (
                str(signature.get("signal", "")),
                str(signature.get("command_pattern", "")) or str(signature.get("command", "")),
                str(signature.get("benchmark_family", "")).strip(),
                str(signature.get("difficulty", "")).strip(),
            )
            entry = merged.setdefault(
                key,
                {
                    "signature_id": str(signature.get("signature_id", "")),
                    "signal": key[0],
                    "command": str(signature.get("command", "")),
                    "command_pattern": key[1],
                    "benchmark_family": key[2],
                    "difficulty": key[3],
                    "support": 0,
                    "regressions": [],
                    "touched_paths": [],
                    "patch_summaries": [],
                    "_representative_support": 0,
                },
            )
            signature_support = int(signature.get("support", 1))
            entry["support"] = int(entry.get("support", 0)) + signature_support
            regressions = {str(path).strip() for path in entry.get("regressions", []) if str(path).strip()}
            regressions.update(str(path).strip() for path in signature.get("regressions", []) if str(path).strip())
            entry["regressions"] = sorted(regressions)
            touched_paths = {str(path).strip() for path in entry.get("touched_paths", []) if str(path).strip()}
            touched_paths.update(str(path).strip() for path in signature.get("touched_paths", []) if str(path).strip())
            entry["touched_paths"] = sorted(touched_paths)
            patch_summaries = {
                str(value).strip() for value in entry.get("patch_summaries", []) if str(value).strip()
            }
            patch_summaries.update(
                str(value).strip() for value in signature.get("patch_summaries", []) if str(value).strip()
            )
            entry["patch_summaries"] = sorted(patch_summaries)[:4]
            representative_command = str(entry.get("command", "")).strip()
            representative_support = int(entry.get("_representative_support", 0) or 0)
            signature_command = str(signature.get("command", "")).strip()
            if (
                not representative_command
                or signature_support > representative_support
                or (signature_support == representative_support and signature_command < representative_command)
            ):
                entry["command"] = signature_command
                entry["_representative_support"] = signature_support
            if not str(entry.get("signature_id", "")).strip():
                entry["signature_id"] = str(signature.get("signature_id", "")).strip()
            if not str(entry.get("benchmark_family", "")).strip():
                entry["benchmark_family"] = str(signature.get("benchmark_family", "")).strip()
            if not str(entry.get("difficulty", "")).strip():
                entry["difficulty"] = str(signature.get("difficulty", "")).strip()
    signatures = []
    for entry in merged.values():
        entry.pop("_representative_support", None)
        signatures.append(entry)
    return sorted(
        signatures,
        key=lambda item: (-int(item.get("support", 0)), str(item.get("signal", "")), str(item.get("command", ""))),
    )[: max(1, int(max_signatures))]


def _proposals(summary: dict[str, object], focus: str) -> list[dict[str, object]]:
    signal_counts = summary.get("signal_counts", {}) if isinstance(summary.get("signal_counts", {}), dict) else {}
    regression_evidence = int(signal_counts.get("state_regression", 0)) > 0 or int(summary.get("regression_path_count", 0) or 0) > 0
    proposals: list[dict[str, object]] = []
    if int(signal_counts.get("no_state_progress", 0)) > 0 or focus == "repeat_avoidance":
        proposals.append(
            {
                "area": "repeat_avoidance",
                "priority": 5,
                "reason": "episodes are repeating commands that previously produced no state progress",
                "suggestion": "Penalize repeated stalled commands and bias action selection toward materially different recovery actions.",
            }
        )
    if regression_evidence:
        proposals.append(
            {
                "area": "regression_guard",
                "priority": 6,
                "reason": "recorded transitions repeatedly regress preserved or previously healthy paths",
                "suggestion": "Penalize commands that match retained regression signatures and prefer recovery actions over repeated regressors.",
            }
        )
    if int(signal_counts.get("no_state_progress", 0)) > 0 or int(signal_counts.get("state_regression", 0)) > 0 or focus == "recovery_bias":
        proposals.append(
            {
                "area": "recovery_bias",
                "priority": 4,
                "reason": "the runtime should respond to recent bad transitions with recovery-biased command scoring",
                "suggestion": "Raise the score of cleanup and progress-restoring commands when the latest state transition regressed or stalled.",
            }
        )
    difficulty_counts = summary.get("difficulty_counts", {}) if isinstance(summary.get("difficulty_counts", {}), dict) else {}
    if int(difficulty_counts.get("long_horizon", 0) or 0) > 0:
        proposals.append(
            {
                "area": "long_horizon_alignment",
                "priority": 5,
                "reason": "long-horizon episodes are present but transition signatures do not yet sufficiently distinguish durable progress from repeated stalls",
                "suggestion": "Bind retained bad-transition signatures to long-horizon tasks and increase the advantage of progress-restoring commands over repeated local retries.",
            }
        )
    return ensure_proposals(
        proposals,
        fallback={
            "area": "repeat_avoidance",
            "priority": 3,
            "reason": "transition-model policy should remain explicit even when recent regressions are rare",
            "suggestion": "Preserve retained bad-transition signatures as a first-class runtime policy surface.",
        },
    )


def _canonicalize_command(command: str) -> str:
    normalized = str(command).strip()
    if not normalized:
        return ""
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n").replace("\t", "\\t")
    return " ".join(normalized.split())


def transition_model_command_pattern(command: str) -> str:
    normalized = _canonicalize_command(command)
    if not normalized:
        return ""
    tokens = _SHELL_TOKEN_RE.findall(normalized)
    if not tokens:
        return normalized
    patterned: list[str] = []
    command_head = ""
    previous_token = ""
    for token in tokens:
        if token in {"&&", "||", ";", "|"}:
            patterned.append(token)
            command_head = ""
            previous_token = token
            continue
        if token in {">", ">>", "<", "<<"}:
            patterned.append(token)
            previous_token = token
            continue
        if not command_head and not token.startswith("-"):
            command_head = token
        placeholder = token
        if _looks_like_env_assignment(token):
            placeholder = "<env>"
        elif _looks_like_quoted_literal(token):
            placeholder = "<str>"
        elif _looks_like_number(token):
            placeholder = "<num>"
        elif previous_token in {">", ">>", "<", "<<"}:
            placeholder = "<path>"
        elif command_head in _PATH_TARGET_COMMANDS and previous_token not in {"", "&&", "||", ";", "|"} and not token.startswith("-"):
            placeholder = "<path>"
        elif _looks_like_path(token):
            placeholder = "<path>"
        patterned.append(placeholder)
        previous_token = token
    return " ".join(patterned)


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token or token.startswith(("=", "./", "../", "/")):
        return False
    name, _, value = token.partition("=")
    return bool(name) and name.replace("_", "a").isalnum() and bool(value)


def _looks_like_quoted_literal(token: str) -> bool:
    return len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}


def _looks_like_number(token: str) -> bool:
    return token.isdigit()


def _patch_excerpt(patch_text: str) -> str:
    for line in str(patch_text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("---", "+++", "@@")):
            continue
        if stripped.startswith(("+", "-")):
            return stripped[:120]
    return ""


def _looks_like_path(token: str) -> bool:
    if token in {".", "..", "~"}:
        return True
    if "/" in token:
        return True
    return any(token.endswith(suffix) for suffix in _PATH_LIKE_SUFFIXES)


def _signature_id(signal: str, command: str) -> str:
    digest = hashlib.sha1(f"{signal}\n{command}".encode("utf-8")).hexdigest()[:12]
    return f"transition:{signal}:{digest}"


def _document_benchmark_family(document: dict[str, object]) -> str:
    task_metadata = document.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        family = str(task_metadata.get("benchmark_family", "")).strip()
        if family:
            return family
    task_contract = document.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            family = str(metadata.get("benchmark_family", "")).strip()
            if family:
                return family
    summary = document.get("summary", {})
    if isinstance(summary, dict):
        family = str(summary.get("benchmark_family", "")).strip()
        if family:
            return family
    return ""


def _document_difficulty(document: dict[str, object]) -> str:
    task_metadata = document.get("task_metadata", {})
    if isinstance(task_metadata, dict):
        difficulty = str(task_metadata.get("difficulty", task_metadata.get("task_difficulty", ""))).strip()
        if difficulty:
            return difficulty
    task_contract = document.get("task_contract", {})
    if isinstance(task_contract, dict):
        metadata = task_contract.get("metadata", {})
        if isinstance(metadata, dict):
            difficulty = str(metadata.get("difficulty", metadata.get("task_difficulty", ""))).strip()
            if difficulty:
                return difficulty
    summary = document.get("summary", {})
    if isinstance(summary, dict):
        difficulty = str(summary.get("difficulty", summary.get("task_difficulty", ""))).strip()
        if difficulty:
            return difficulty
    return ""
