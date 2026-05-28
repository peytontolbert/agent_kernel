#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_kernel.neural_controller import (
    FULL_KERNEL_CONTROL_TOKENS,
    artifact_command_target_from_encoder,
    artifact_slot_target_from_encoder,
    build_neural_controller_encoder_text,
    localized_edit_candidate_pointer_token,
    localized_edit_candidates_from_encoder,
)
from agent_kernel.ops.episode_store import iter_episode_documents


def _compact(value: object, *, limit: int = 900) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()[:limit].rstrip()


def _line_content(value: object, *, limit: int = 1200) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\n")
    return text.strip()[:limit].rstrip()


def _canonical_text(value: object) -> str:
    text = str(value or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return " ".join(text.split())


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _verification_passed(step: dict[str, Any]) -> bool | None:
    verification = step.get("verification", {})
    if isinstance(verification, dict) and "passed" in verification:
        return bool(verification.get("passed", False))
    result = step.get("command_result", {})
    if isinstance(result, dict) and "exit_code" in result:
        return int(result.get("exit_code", 1) or 1) == 0
    return None


def _failure_mode(step: dict[str, Any]) -> str:
    metadata = step.get("proposal_metadata", {})
    if isinstance(metadata, dict):
        for key in (
            "artifact_failure_mode",
            "artifact_repair_mode",
            "artifact_contract_failure_mode",
        ):
            value = str(metadata.get(key, "")).strip()
            if value:
                return value
    verification = step.get("verification", {})
    if isinstance(verification, dict):
        codes = verification.get("failure_codes", [])
        if isinstance(codes, list) and codes:
            return str(codes[0]).strip()
        reasons = verification.get("reasons", [])
        if isinstance(reasons, list):
            for reason in reasons:
                normalized = str(reason).strip()
                if normalized and normalized.lower() != "verification passed":
                    return "verification_failed"
    return "artifact_contract_success" if _verification_passed(step) is True else "unknown"


def _is_source_inspect_command(content: str) -> bool:
    stripped = str(content or "").strip()
    return stripped.startswith(("cat ", "head ", "tail ", "grep ", "sed -n "))


def _retrieval_tokens(step: dict[str, Any], action: str, content: str) -> tuple[str, float, float]:
    lowered = content.lower()
    retrieval = bool(step.get("retrieval_influenced", False)) or bool(step.get("trust_retrieval", False))
    source_inspect = action == "code_execute" and _is_source_inspect_command(content)
    if retrieval or source_inspect or "source_lines/" in lowered or "source_context/" in lowered:
        return "<AK_RETRIEVE> <AK_RET_CODE> <AK_RET_EXACT>", 1.0, 0.75
    return "<AK_NO_RETRIEVAL>", 0.0, 0.0


def _execution_intent_token(action: str, content: str) -> str:
    if action != "code_execute":
        return ""
    normalized = " ".join(str(content or "").strip().split())
    lowered = normalized.lower()
    if not normalized:
        return ""
    if lowered.startswith(("cat ", "head ", "tail ", "grep ")) or lowered.startswith("sed -n "):
        return "<AK_EXEC_KIND_INSPECT_SOURCE>"
    if lowered.startswith(("test ! -f ", "test ! -e ", "[ ! -f ", "[ ! -e ")):
        return "<AK_EXEC_KIND_VERIFY_ABSENT>"
    if lowered.startswith(("test -f ", "test -e ", "[ -f ", "[ -e ")):
        return "<AK_EXEC_KIND_VERIFY_PRESENT>" if "&&" not in lowered else "<AK_EXEC_KIND_RUN_CHECK>"
    if lowered.startswith(("sed -i ", "perl -0pi ", "python - <<", "python3 - <<")):
        return "<AK_EXEC_KIND_LOCALIZED_EDIT>"
    if ">" in normalized and any(marker in lowered for marker in ("printf ", "cat <<", "echo ", "mkdir -p ")):
        return "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"
    if lowered.startswith(("pytest ", "python -m pytest", "python3 -m pytest", "python ", "python3 ", "bash ", "sh ")):
        return "<AK_EXEC_KIND_RUN_CHECK>"
    return "<AK_EXEC_KIND_RUN_CHECK>" if lowered.startswith("test ") else ""


def _command_argument_slots(action: str, content: str) -> dict[str, str]:
    if action != "code_execute":
        return {}
    text = str(content or "").strip()
    intent = _execution_intent_token(action, text)
    slots: dict[str, str] = {}
    if intent == "<AK_EXEC_KIND_VERIFY_PRESENT>":
        path = _first_path_after_patterns(text, (r"\btest\s+-[fe]\s+([^;&|]+)", r"\[\s+-[fe]\s+([^\]]+)\]"))
        if path:
            slots["target_path"] = path
            slots["verify_polarity"] = "present"
    elif intent == "<AK_EXEC_KIND_VERIFY_ABSENT>":
        path = _first_path_after_patterns(text, (r"\btest\s+!\s+-[fe]\s+([^;&|]+)", r"\[\s+!\s+-[fe]\s+([^\]]+)\]"))
        if path:
            slots["target_path"] = path
            slots["verify_polarity"] = "absent"
    elif intent == "<AK_EXEC_KIND_INSPECT_SOURCE>":
        path = _inspect_path(text)
        if path:
            slots["target_path"] = path
    elif intent == "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>":
        path = _redirect_path(text)
        value = _materialized_content(text)
        if path:
            slots["target_path"] = path
        if value:
            slots["target_content"] = value
    elif intent == "<AK_EXEC_KIND_LOCALIZED_EDIT>":
        old, new, path = _sed_edit_slots(text)
        if path:
            slots["target_path"] = path
        if old:
            slots["edit_old"] = old
        if new:
            slots["edit_new"] = new
    return {key: _line_content(value, limit=600) for key, value in slots.items() if str(value).strip()}


def _first_path_after_patterns(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        return _clean_shell_path(match.group(1))
    return ""


def _inspect_path(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("sed -n "):
        return _clean_shell_path(stripped.split()[-1])
    parts = stripped.split()
    return _clean_shell_path(parts[-1]) if len(parts) >= 2 else ""


def _redirect_path(text: str) -> str:
    match = re.search(r">\s*([^;&|]+)", text)
    return _clean_shell_path(match.group(1)) if match else ""


def _materialized_content(text: str) -> str:
    printf_match = re.search(r"printf\s+(?:%s\s+)?(['\"])(.*?)\1\s*>", text)
    if printf_match:
        return printf_match.group(2)
    echo_match = re.search(r"echo\s+(['\"])(.*?)\1\s*>", text)
    if echo_match:
        return echo_match.group(2)
    return ""


def _sed_edit_slots(text: str) -> tuple[str, str, str]:
    path = _clean_shell_path(text.split()[-1]) if text.split() else ""
    match = re.search(r"sed\s+-i\s+(['\"])(?:\d+(?:,\d+)?)?s(.)(.*?)\2(.*?)\2", text)
    if match:
        return match.group(3), match.group(4), path
    change_match = re.search(r"sed\s+-i\s+(['\"])(?:\d+(?:,\d+)?)c\\\\n(.*?)\1", text)
    if change_match:
        return "", change_match.group(2), path
    append_match = re.search(r"sed\s+-i\s+(['\"])\$a\\\\n(.*?)\1", text)
    if append_match:
        return "", append_match.group(2), path
    return "", "", path


def _clean_shell_path(value: object) -> str:
    text = str(value or "").strip().strip("'\"")
    return text.rstrip(";|&").strip()


def _control_tokens(
    step: dict[str, Any],
    action: str,
    content: str,
    failure_mode: str,
    *,
    terminal_step: bool = False,
) -> str:
    decision_source = str(step.get("decision_source", "")).strip()
    tokens: list[str] = ["<AK_DECIDE>"]
    if action == "respond":
        tokens.append("<AK_ACTION_SPACE_RESPOND>")
    elif _is_source_inspect_command(content):
        tokens.append("<AK_ACTION_SPACE_RETRIEVAL>")
    elif "artifact" in decision_source or "patch_builder" in content or "swe_patch_builder" in content:
        tokens.append("<AK_ACTION_SPACE_ARTIFACT>")
    else:
        tokens.append("<AK_ACTION_SPACE_CODE>")
    if _is_source_inspect_command(content):
        tokens.append("<AK_SOURCE_INSPECT>")
    if "artifact" in decision_source or "patch_builder" in content or "swe_patch_builder" in content:
        tokens.append("<AK_ARTIFACT_REPAIR>")
    if "patch_builder" in content or "swe_patch_builder" in content:
        tokens.append("<AK_PATCH_BUILD>")
    retrieval_tokens, _, _ = _retrieval_tokens(step, action, content)
    tokens.extend(part for part in retrieval_tokens.split() if part.startswith("<AK_"))
    intent_token = _execution_intent_token(action, content)
    if intent_token == "<AK_EXEC_KIND_VERIFY_PRESENT>":
        tokens.append("<AK_VALIDATE_PRESENT>")
    elif intent_token == "<AK_EXEC_KIND_VERIFY_ABSENT>":
        tokens.append("<AK_VALIDATE_ABSENT>")
    elif intent_token == "<AK_EXEC_KIND_INSPECT_SOURCE>":
        tokens.append("<AK_READ_SOURCE>")
    if intent_token:
        tokens.append(intent_token)
    if action == "respond":
        tokens.append("<AK_RESPOND>")
        tokens.append("<AK_CLOSEOUT>")
    else:
        tokens.append("<AK_EXECUTE>")
    passed = _verification_passed(step)
    tokens.append("<AK_VERIFY>")
    tokens.append("<AK_WORLD_UPDATE>")
    tokens.append("<AK_MEMORY_WRITE>")
    if passed is True:
        tokens.append("<AK_CONF_HIGH>")
    elif passed is False:
        tokens.append("<AK_CONF_LOW>")
    else:
        tokens.append("<AK_CONF_MEDIUM>")
    repairable_failure = failure_mode and failure_mode not in {
        "artifact_contract_success",
        "not_artifact_contract",
        "unknown",
    }
    if repairable_failure:
        tokens.append("<AK_OOD>")
        if "<AK_ARTIFACT_REPAIR>" not in tokens and action != "respond":
            tokens.append("<AK_ARTIFACT_REPAIR>")
    if terminal_step and passed is False and action != "respond":
        tokens.append("<AK_SAFE_STOP>")
    return " ".join(dict.fromkeys(tokens))


def _task_payload(document: dict[str, Any]) -> dict[str, Any]:
    contract = document.get("task_contract", {})
    contract = dict(contract) if isinstance(contract, dict) else {}
    task_metadata = document.get("task_metadata", {})
    task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    return {
        "task_id": str(document.get("task_id", contract.get("task_id", ""))).strip(),
        "prompt": str(document.get("prompt", contract.get("prompt", ""))).strip(),
        "workspace_subdir": str(document.get("workspace", contract.get("workspace_subdir", ""))).strip(),
        "suggested_commands": list(contract.get("suggested_commands", []))
        if isinstance(contract.get("suggested_commands", []), list)
        else [],
        "success_command": str(contract.get("success_command", "")).strip(),
        "expected_files": list(contract.get("expected_files", []))
        if isinstance(contract.get("expected_files", []), list)
        else [],
        "expected_file_contents": dict(contract.get("expected_file_contents", {}))
        if isinstance(contract.get("expected_file_contents", {}), dict)
        else {},
        "metadata": task_metadata,
    }


def _history_prefix(steps: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for prior in steps[max(0, index - 6) : index]:
        history.append(
            {
                "action": str(prior.get("action", "")).strip(),
                "content": _line_content(prior.get("content", ""), limit=260),
                "decision_source": str(prior.get("decision_source", "")).strip(),
            }
        )
    return history


def _row_from_step(
    *,
    document: dict[str, Any],
    source_root: Path,
    step: dict[str, Any],
    steps: list[dict[str, Any]],
    step_index: int,
    eval_fraction: float,
) -> dict[str, Any] | None:
    action = str(step.get("action", "")).strip()
    content = str(step.get("content", "")).strip()
    if not action or not content:
        return None
    task = _task_payload(document)
    failure_mode = _failure_mode(step)
    retrieval_token_text, retrieval_weight, retrieval_coverage = _retrieval_tokens(step, action, content)
    state_payload = {
        "task": task,
        "history": _history_prefix(steps, step_index),
        "retrieval_plan": {"mode": retrieval_token_text} if retrieval_weight else {},
        "world_model_summary": document.get("world_model_summary", {})
        if isinstance(document.get("world_model_summary", {}), dict)
        else {},
        "plan": document.get("plan", []) if isinstance(document.get("plan", []), list) else [],
        "active_subgoal": str(step.get("active_subgoal", "") or document.get("termination_reason", "")).strip(),
        "trajectory_step_index": step_index,
        "trajectory_step_count": len(steps),
    }
    encoder_text = build_neural_controller_encoder_text(state_payload=state_payload)
    terminal_step = step_index == len(steps) - 1 and document.get("success") is False
    control_tokens = _control_tokens(
        step,
        action,
        content,
        failure_mode,
        terminal_step=terminal_step,
    )
    decoder_text = "\n".join(
        [
            control_tokens,
            f"Action: {action}",
            f"Artifact-Failure-Mode: {failure_mode}",
            *_slot_lines(_command_argument_slots(action, content)),
            f"Content: {_line_content(content)}",
        ]
    )
    artifact_target = artifact_command_target_from_encoder(encoder_text)
    uses_artifact_pointer = bool(
        action == "code_execute"
        and artifact_target
        and _canonical_text(content) == _canonical_text(artifact_target)
    )
    if uses_artifact_pointer:
        token_parts = control_tokens.split()
        if "<AK_COPY_ARTIFACT_TARGET>" not in token_parts:
            insert_at = token_parts.index("<AK_EXECUTE>") if "<AK_EXECUTE>" in token_parts else len(token_parts)
            token_parts.insert(insert_at, "<AK_COPY_ARTIFACT_TARGET>")
        artifact_path, artifact_content = artifact_slot_target_from_encoder(encoder_text)
        artifact_slots = {
            "target_path": "<AK_COPY_ARTIFACT_PATH>" if artifact_path else "",
            "target_content": "<AK_COPY_ARTIFACT_CONTENT>" if artifact_content else "",
        }
        decoder_text = "\n".join(
            [
                " ".join(token_parts),
                f"Action: {action}",
                f"Artifact-Failure-Mode: {failure_mode}",
                *_slot_lines({key: value for key, value in artifact_slots.items() if value}),
                "Content: <AK_COPY_ARTIFACT_TARGET>",
            ]
        )
    localized_edit_candidates = localized_edit_candidates_from_encoder(encoder_text)
    localized_pointer_index = 0
    if (
        action == "code_execute"
        and _execution_intent_token(action, content) == "<AK_EXEC_KIND_LOCALIZED_EDIT>"
    ):
        for candidate_index, candidate in enumerate(localized_edit_candidates, start=1):
            if _canonical_text(candidate) == _canonical_text(content):
                localized_pointer_index = candidate_index
                break
    if localized_pointer_index:
        pointer_token = localized_edit_candidate_pointer_token(localized_pointer_index)
        token_parts = control_tokens.split()
        if pointer_token and pointer_token not in token_parts:
            insert_at = token_parts.index("<AK_EXECUTE>") if "<AK_EXECUTE>" in token_parts else len(token_parts)
            token_parts.insert(insert_at, pointer_token)
        decoder_text = "\n".join(
            [
                " ".join(token_parts),
                f"Action: {action}",
                f"Artifact-Failure-Mode: {failure_mode}",
                *_slot_lines(_command_argument_slots(action, content)),
                f"Content: {pointer_token}",
            ]
        )
    source_id = (
        f"{source_root}:{task.get('task_id', '')}:"
        f"{document.get('termination_reason', '')}:{step.get('index', step_index + 1)}"
    )
    passed = _verification_passed(step)
    confidence = 0.9 if passed is True else 0.25 if passed is False else 0.55
    return {
        "action": action,
        "answer_confidence_target": confidence,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24],
        "needs_verification_target": 0.9 if action == "code_execute" else 0.4,
        "ood_evidence_target": 0.1 if passed is True else 0.65 if passed is False else 0.35,
        "ood_query_target": 0.1 if retrieval_weight else 0.25,
        "paper_action_validity_target": 0.95 if action else 0.0,
        "query_confidence_target": confidence,
        "retrieval_coverage_target": retrieval_coverage,
        "retrieval_doc_text": _compact(document.get("prompt", ""), limit=900) if retrieval_weight else "",
        "retrieval_loss_weight": retrieval_weight,
        "retrieval_query_text": _compact(content, limit=260) if retrieval_weight else "",
        "source_id": source_id,
        "source_type": "agentkernel_long_horizon_episode_trace",
        "split": _hash_split(source_id, eval_fraction),
        "task_type": "controller_long_horizon_policy",
        "uses_artifact_command_pointer": uses_artifact_pointer,
        "uses_localized_edit_candidate_pointer": bool(localized_pointer_index),
        "localized_edit_candidate_pointer_index": localized_pointer_index,
        "trajectory_step_index": step_index,
        "trajectory_step_count": len(steps),
        "weight": 1.35 if len(steps) >= 3 else 1.1,
    }


def _slot_lines(slots: dict[str, str]) -> list[str]:
    labels = {
        "target_path": "Target-Path",
        "target_content": "Target-Content",
        "edit_old": "Edit-Old",
        "edit_new": "Edit-New",
        "verify_polarity": "Verify-Polarity",
    }
    return [f"{label}: {slots[key]}" for key, label in labels.items() if key in slots]


def _iter_episode_rows(
    *,
    roots: list[Path],
    min_steps: int,
    max_examples: int,
    eval_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        documents = iter_episode_documents(root)
        for document in documents:
            steps_raw = document.get("steps", [])
            if not isinstance(steps_raw, list) or len(steps_raw) < min_steps:
                continue
            steps = [step for step in steps_raw if isinstance(step, dict)]
            if len(steps) < min_steps:
                continue
            for index, step in enumerate(steps):
                row = _row_from_step(
                    document=document,
                    source_root=root,
                    step=step,
                    steps=steps,
                    step_index=index,
                    eval_fraction=eval_fraction,
                )
                if row is None:
                    continue
                key = str(row["example_id"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if max_examples > 0 and len(rows) >= max_examples:
                    return rows
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    roots = [Path(value).expanduser().resolve() for value in args.episodes_root]
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = _iter_episode_rows(
        roots=roots,
        min_steps=max(2, int(args.min_steps)),
        max_examples=max(0, int(args.max_examples)),
        eval_fraction=float(args.eval_fraction),
    )
    if not rows:
        raise SystemExit("no long-horizon episode trace rows generated")
    train_rows = [row for row in rows if row.get("split") != "eval"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    action_counts: dict[str, int] = {}
    step_count_histogram: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action", ""))
        action_counts[action] = action_counts.get(action, 0) + 1
        step_count = str(row.get("trajectory_step_count", ""))
        step_count_histogram[step_count] = step_count_histogram.get(step_count, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_controller_long_horizon_dataset",
        "objective": "agentkernel_controller_long_horizon_trajectory_policy",
        "dataset_format": "jsonl",
        "decoder_format": "line",
        "agentkernel_special_tokens": list(FULL_KERNEL_CONTROL_TOKENS),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"agentkernel_long_horizon_episode_trace": len(rows)},
        "action_counts": dict(sorted(action_counts.items())),
        "trajectory_step_count_histogram": dict(sorted(step_count_histogram.items())),
        "episodes_roots": [str(root) for root in roots],
        "schema": {
            "encoder_text": "full Agent Kernel runtime encoder with prior-step history from multi-step episodes",
            "decoder_text": "control tokens plus line-protocol action, artifact mode, and escaped content",
            "trajectory_fields": "trajectory_step_index and trajectory_step_count preserve long-horizon position",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", action="append", default=[])
    parser.add_argument("--output-dir", default="artifacts/agentkernel_controller/long_horizon_trajectory_v1")
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not args.episodes_root:
        args.episodes_root = ["trajectories"]
    print(json.dumps(build_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
