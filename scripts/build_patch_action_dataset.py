#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "agentkernel_patch_action_dataset_v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not str(path).strip() or not path.exists():
        return records
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _safe_id(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe.strip("._") or _sha256_text(text)[:16]


def _load_payload(path: str) -> dict[str, Any]:
    if not str(path).strip():
        return {}
    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _prediction_records(path: str) -> dict[str, dict[str, Any]]:
    if not str(path).strip():
        return {}
    rows = _read_jsonl(Path(path))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = _text(row.get("instance_id"))
        if instance_id:
            out[instance_id] = row
    return out


def _skill_cards_by_id(path: str) -> dict[str, dict[str, Any]]:
    if not str(path).strip():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(Path(path)):
        payload = row
        if row.get("event") == "skill_card":
            payload = {k: v for k, v in row.items() if k != "event"}
        skill_id = _text(payload.get("id") or payload.get("skill_id"))
        if skill_id:
            out[skill_id] = payload
    return out


def _changed_paths_from_diff(diff_text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_line in diff_text.splitlines():
        line = raw_line.strip()
        candidate = ""
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ "):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                candidate = parts[1]
        if not candidate or candidate == "/dev/null":
            continue
        if candidate.startswith("a/") or candidate.startswith("b/"):
            candidate = candidate[2:]
        if candidate and candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)
    return paths


def _added_removed_lines(diff_text: str) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return (
        "/test" in p
        or p.startswith("test")
        or "/tests/" in p
        or p.endswith("_test.py")
        or p.endswith("test.py")
        or ".spec." in p
    )


def _path_kind(path: str) -> str:
    p = path.lower()
    if _is_test_path(path):
        return "test"
    if p.endswith((".md", ".rst", ".txt")):
        return "docs"
    if p.endswith((".yml", ".yaml", ".toml", ".ini", ".cfg", ".json", "requirements.txt", "package.json")):
        return "config"
    if p.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp", ".h")):
        return "source"
    return "other"


def analyze_patch_diff(diff_text: str) -> dict[str, Any]:
    changed_paths = _changed_paths_from_diff(diff_text)
    added, removed = _added_removed_lines(diff_text)
    added_text = "\n".join(added)
    removed_text = "\n".join(removed)
    combined = f"{added_text}\n{removed_text}".lower()

    intents: list[dict[str, Any]] = []

    def add_intent(key: str, confidence: float, evidence: str = "") -> None:
        if key not in {item["key"] for item in intents}:
            intents.append({"key": key, "confidence": round(float(confidence), 3), "evidence": evidence[:240]})

    if not added and not removed:
        add_intent("patch.no_meaningful_change", 1.0)
    if len(changed_paths) > 1:
        add_intent("patch.multi_file", 0.9, ",".join(changed_paths[:8]))
    elif changed_paths:
        add_intent("patch.single_file", 0.8, changed_paths[0])

    test_paths = [p for p in changed_paths if _is_test_path(p)]
    source_paths = [p for p in changed_paths if _path_kind(p) == "source" and p not in test_paths]
    config_paths = [p for p in changed_paths if _path_kind(p) == "config"]
    docs_paths = [p for p in changed_paths if _path_kind(p) == "docs"]

    if test_paths:
        add_intent("test.add_or_update", 0.95, ",".join(test_paths[:8]))
    if config_paths:
        add_intent("edit.config_change", 0.82, ",".join(config_paths[:8]))
    if docs_paths and len(docs_paths) == len(changed_paths):
        add_intent("edit.docs_only", 0.9, ",".join(docs_paths[:8]))

    if re.search(r"(?m)^\s*(try|except|finally|raise)\b", added_text):
        add_intent("edit.exception_handling", 0.92)
    if any(token in combined for token in ("timeout", "timed_out", "timeouterror", "timeoutexpired")):
        add_intent("behavior.timeout_handling", 0.92)
    if any(token in combined for token in ("retry", "backoff", "retriable", "retryable")):
        add_intent("behavior.retry_recovery", 0.88)
    if re.search(r"(?m)^\s*(import|from)\s+", added_text):
        add_intent("edit.import_added", 0.78)
    if any(token in combined for token in ("logger.", "logging.", "console.log", "print(")):
        add_intent("observability.logging_or_print", 0.72)
    if any(token in combined for token in ("async ", "await ", "thread", "lock", "queue", "concurrent", "multiprocessing")):
        add_intent("behavior.concurrency", 0.76)
    if any(token in combined for token in ("subprocess", "os.system", "popen", "shell", "bash", "npm ", "pytest", "go test")):
        add_intent("runtime.command_execution", 0.84)
    if any(token in combined for token in ("http://", "https://", "requests.", "fetch(", "axios", "socket.")):
        add_intent("runtime.network_io", 0.82)
    if any(token in combined for token in ("open(", "read_text", "write_text", "write_bytes", "mkdir", "unlink")):
        add_intent("runtime.filesystem_io", 0.8)
    if any(token in combined for token in ("def ", "class ", "function ", "=>", "interface ", "type ")):
        add_intent("edit.api_or_symbol_change", 0.74)

    kind_counts: dict[str, int] = {}
    for path in changed_paths:
        kind_counts[_path_kind(path)] = kind_counts.get(_path_kind(path), 0) + 1

    dominant = "general_edit"
    for key in (
        "behavior.timeout_handling",
        "behavior.retry_recovery",
        "edit.exception_handling",
        "runtime.command_execution",
        "test.add_or_update",
        "edit.config_change",
    ):
        if any(item["key"] == key for item in intents):
            dominant = key
            break

    return {
        "diff_sha256": _sha256_text(diff_text),
        "changed_paths": changed_paths,
        "test_paths": test_paths,
        "source_paths": source_paths,
        "path_kind_counts": kind_counts,
        "added_line_count": len(added),
        "removed_line_count": len(removed),
        "intents": intents,
        "patch_operator": {
            "dominant_intent": dominant,
            "changed_path_count": len(changed_paths),
            "has_tests": bool(test_paths),
            "has_source_changes": bool(source_paths),
            "edit_shape": "multi_file" if len(changed_paths) > 1 else "single_file" if changed_paths else "empty",
        },
    }


def _prediction_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("prediction_manifest")
    if isinstance(nested, dict):
        return nested
    return payload


def _queue_by_instance(queue_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in queue_manifest.get("tasks", []):
        if not isinstance(task, dict):
            continue
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        instance_id = _text(metadata.get("swe_instance_id") or metadata.get("instance_id"))
        if instance_id:
            out[instance_id] = task
    return out


def _patch_for_prediction(
    prediction: dict[str, Any],
    *,
    prediction_records: dict[str, dict[str, Any]],
    base_dir: Path,
) -> str:
    instance_id = _text(prediction.get("instance_id"))
    record = prediction_records.get(instance_id, {})
    patch = _text(record.get("model_patch") or record.get("patch") or prediction.get("model_patch") or prediction.get("patch"))
    if patch:
        return patch
    patch_path = _text(prediction.get("patch_path"))
    if not patch_path:
        return ""
    path = Path(patch_path)
    if not path.is_absolute():
        path = base_dir / path
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _retrieval_labels(task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    decisions = metadata.get("action_decisions") or metadata.get("steps") or []
    selected_skill_ids: list[str] = []
    selected_span_ids: list[str] = []
    retrieval_influenced = False
    retrieval_ranked_skill = False
    if isinstance(decisions, list):
        for item in decisions:
            if not isinstance(item, dict):
                continue
            sid = _text(item.get("selected_skill_id"))
            rid = _text(item.get("selected_retrieval_span_id"))
            if sid and sid not in selected_skill_ids:
                selected_skill_ids.append(sid)
            if rid and rid not in selected_span_ids:
                selected_span_ids.append(rid)
            retrieval_influenced = retrieval_influenced or bool(item.get("retrieval_influenced", False))
            retrieval_ranked_skill = retrieval_ranked_skill or bool(item.get("retrieval_ranked_skill", False))
    for field in ("selected_skill_id", "skill_id"):
        sid = _text(metadata.get(field))
        if sid and sid not in selected_skill_ids:
            selected_skill_ids.append(sid)
    for field in ("selected_retrieval_span_id", "retrieval_span_id"):
        rid = _text(metadata.get(field))
        if rid and rid not in selected_span_ids:
            selected_span_ids.append(rid)
    retrieval_influenced = retrieval_influenced or bool(metadata.get("retrieval_influenced", False))
    retrieval_ranked_skill = retrieval_ranked_skill or bool(metadata.get("retrieval_ranked_skill", False))
    return {
        "selected_skill_ids": selected_skill_ids,
        "selected_retrieval_span_ids": selected_span_ids,
        "retrieval_influenced": retrieval_influenced,
        "retrieval_ranked_skill": retrieval_ranked_skill,
    }


def _verification_label(instance_id: str, verification: dict[str, Any]) -> dict[str, Any]:
    successful = {str(x).strip() for x in verification.get("successful_instance_ids", []) if str(x).strip()}
    failed = {str(x).strip() for x in verification.get("failed_instance_ids", []) if str(x).strip()}
    abstained = {str(x).strip() for x in verification.get("abstained_instance_ids", []) if str(x).strip()}
    if instance_id in successful:
        return {"passed": True, "outcome": "success"}
    if instance_id in abstained:
        return {"passed": False, "outcome": "abstained"}
    if instance_id in failed:
        return {"passed": False, "outcome": "failed"}
    return {"passed": None, "outcome": "unknown"}


def build_patch_action_dataset(
    *,
    prediction_task_manifest: dict[str, Any],
    queue_manifest: dict[str, Any],
    predictions_jsonl: str = "",
    skill_cards_jsonl: str = "",
    patch_job_verification: dict[str, Any] | None = None,
    include_diff: bool = True,
) -> list[dict[str, Any]]:
    pred_manifest = _prediction_manifest(prediction_task_manifest)
    base_dir = Path(_text(pred_manifest.get("base_dir")) or ".")
    prediction_records = _prediction_records(predictions_jsonl)
    queue = _queue_by_instance(queue_manifest)
    skill_cards = _skill_cards_by_id(skill_cards_jsonl)
    verification = patch_job_verification or {}

    examples: list[dict[str, Any]] = []
    for prediction in pred_manifest.get("predictions", []):
        if not isinstance(prediction, dict):
            continue
        instance_id = _text(prediction.get("instance_id"))
        if not instance_id:
            continue
        task = queue.get(instance_id, {})
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        patch = _patch_for_prediction(prediction, prediction_records=prediction_records, base_dir=base_dir)
        analysis = analyze_patch_diff(patch)
        retrieval = _retrieval_labels(task)
        selected_cards = [
            skill_cards[sid]
            for sid in retrieval["selected_skill_ids"]
            if sid in skill_cards
        ]
        repo = _text(metadata.get("repo") or prediction.get("repo"))
        base_commit = _text(metadata.get("base_commit") or prediction.get("base_commit"))
        prompt = _text(task.get("prompt") or metadata.get("problem_statement") or prediction.get("problem_statement"))
        example_id = _sha256_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "task_id": _text(task.get("task_id")),
                    "diff": analysis["diff_sha256"],
                    "repo": repo,
                    "base_commit": base_commit,
                },
                sort_keys=True,
            )
        )[:24]
        intents = [item["key"] for item in analysis["intents"]]
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "example_id": example_id,
            "instance_id": instance_id,
            "task": {
                "task_id": _text(task.get("task_id")),
                "prompt": prompt,
                "workspace_subdir": _text(task.get("workspace_subdir")),
                "repo": repo,
                "base_commit": base_commit,
                "success_command": _text(task.get("success_command")),
                "suggested_commands": _list_text(task.get("suggested_commands")),
                "expected_files": _list_text(task.get("expected_files")),
                "metadata": metadata,
            },
            "patch": {
                **analysis,
                "diff": patch if include_diff else "",
            },
            "retrieval": {
                **retrieval,
                "selected_skill_cards": selected_cards,
            },
            "verification": _verification_label(instance_id, verification),
            "model_targets": {
                "family": "patch_action",
                "path_labels": intents,
                "dominant_intent": analysis["patch_operator"]["dominant_intent"],
                "action_labels": [
                    analysis["patch_operator"]["edit_shape"],
                    "with_tests" if analysis["patch_operator"]["has_tests"] else "without_tests",
                    "retrieval_influenced" if retrieval["retrieval_influenced"] else "retrieval_unknown",
                ],
                "decoder_target": _text(prediction.get("patch_path")) or f"{_safe_id(instance_id)}.diff",
                "scalar_features": {
                    "changed_path_count": analysis["patch_operator"]["changed_path_count"],
                    "added_line_count": analysis["added_line_count"],
                    "removed_line_count": analysis["removed_line_count"],
                    "selected_skill_count": len(retrieval["selected_skill_ids"]),
                    "selected_retrieval_span_count": len(retrieval["selected_retrieval_span_ids"]),
                },
            },
        }
        examples.append(record)
    return examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("writing Parquet datasets requires pyarrow") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records) if records else pa.table({})
    try:
        pq.write_table(table, path, compression="zstd")
    except Exception:
        pq.write_table(table, path, compression="snappy")


def write_dataset(path: Path, records: list[dict[str, Any]]) -> str:
    if path.suffix.lower() == ".parquet":
        write_parquet(path, records)
        return "parquet"
    write_jsonl(path, records)
    return "jsonl"


def _load_hybrid_config(path: str) -> Any:
    from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig

    if str(path).strip():
        payload = _read_json(Path(path))
        if not isinstance(payload, dict):
            raise ValueError(f"expected config JSON object at {path}")
        return HybridTolbertSSMConfig.from_dict(payload)
    return HybridTolbertSSMConfig()


def _pad_scalars(values: list[float], size: int) -> list[float]:
    out = [float(x) for x in values[: max(0, int(size))]]
    while len(out) < int(size):
        out.append(0.0)
    return out


def _hybrid_world_target(example: dict[str, Any], config: Any) -> list[float]:
    size = int(config.world_state_dim)
    target = [0.0] * size
    outcome = str(example.get("verification", {}).get("outcome", "unknown"))
    dominant = str(example.get("model_targets", {}).get("dominant_intent", ""))
    if outcome == "success":
        idx = 0
    elif outcome == "failed":
        idx = 1
    elif outcome == "abstained":
        idx = 2
    elif "timeout" in dominant:
        idx = 3
    elif "test" in dominant:
        idx = 4
    elif "config" in dominant:
        idx = 5
    else:
        idx = min(size - 1, 6)
    target[idx % max(1, size)] = 1.0
    return target


def materialize_hybrid_examples(
    examples: list[dict[str, Any]],
    *,
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from agent_kernel.modeling.tolbert.tokenization import (
        build_decoder_vocabulary,
        encode_command_tokens,
        encode_decoder_sequence,
        hashed_id,
    )

    decoder_texts: list[str] = []
    for example in examples:
        targets = example.get("model_targets", {}) if isinstance(example.get("model_targets"), dict) else {}
        patch = example.get("patch", {}) if isinstance(example.get("patch"), dict) else {}
        decoder_texts.append(
            " ".join(
                [
                    "patch_action",
                    str(targets.get("dominant_intent", "")),
                    " ".join([str(x) for x in patch.get("changed_paths", [])][:4]),
                ]
            ).strip()
        )
    decoder_vocab = build_decoder_vocabulary(decoder_texts, config)

    out: list[dict[str, Any]] = []
    for example, decoder_text in zip(examples, decoder_texts):
        task = example.get("task", {}) if isinstance(example.get("task"), dict) else {}
        patch = example.get("patch", {}) if isinstance(example.get("patch"), dict) else {}
        retrieval = example.get("retrieval", {}) if isinstance(example.get("retrieval"), dict) else {}
        verification = example.get("verification", {}) if isinstance(example.get("verification"), dict) else {}
        targets = example.get("model_targets", {}) if isinstance(example.get("model_targets"), dict) else {}
        scalar = targets.get("scalar_features", {}) if isinstance(targets.get("scalar_features"), dict) else {}
        labels = [str(x) for x in targets.get("path_labels", []) if str(x).strip()]
        dominant = str(targets.get("dominant_intent", "")).strip() or "general_edit"
        repo = str(task.get("repo", "")).strip()
        instance_id = str(example.get("instance_id", "")).strip()

        path_values = [dominant, *(labels[: max(0, int(config.max_path_levels) - 2)]), repo or instance_id]
        path_level_ids = [hashed_id(value, int(config.path_vocab_size)) for value in path_values[: int(config.max_path_levels)]]
        while len(path_level_ids) < int(config.max_path_levels):
            path_level_ids.append(0)

        frames = [
            str(task.get("prompt", "")),
            " ".join([str(x) for x in retrieval.get("selected_skill_ids", [])]),
            " ".join([str(x) for x in patch.get("changed_paths", [])]),
            dominant,
            " ".join(labels),
        ]
        token_rows = [encode_command_tokens(frame, config) for frame in frames[-int(config.sequence_length) :]]
        while len(token_rows) < int(config.sequence_length):
            token_rows.insert(0, [0] * int(config.max_command_tokens))

        base_scalars = _pad_scalars(
            [
                float(scalar.get("changed_path_count", 0) or 0),
                float(scalar.get("added_line_count", 0) or 0),
                float(scalar.get("removed_line_count", 0) or 0),
                float(scalar.get("selected_skill_count", 0) or 0),
                float(scalar.get("selected_retrieval_span_count", 0) or 0),
                1.0 if bool(patch.get("patch_operator", {}).get("has_tests", False)) else 0.0,
                1.0 if bool(retrieval.get("retrieval_influenced", False)) else 0.0,
                1.0 if bool(retrieval.get("retrieval_ranked_skill", False)) else 0.0,
                float(len(labels)),
            ],
            int(config.scalar_feature_dim),
        )
        scalar_rows = [list(base_scalars) for _ in range(int(config.sequence_length))]

        passed = verification.get("passed")
        is_success = bool(passed is True)
        is_failed = bool(passed is False)
        has_noop = "patch.no_meaningful_change" in labels
        has_tests = bool(patch.get("patch_operator", {}).get("has_tests", False))
        retrieval_used = bool(retrieval.get("retrieval_influenced", False) or retrieval.get("retrieval_ranked_skill", False))
        policy_target = min(1.0, (0.65 if is_success else 0.15 if not is_failed else 0.0) + (0.15 if retrieval_used else 0.0) + (0.10 if has_tests else 0.0))
        risk_target = min(1.0, (0.15 if is_success else 0.75 if is_failed else 0.35) + (0.2 if has_noop else 0.0) + (0.1 if not has_tests else 0.0))
        value_target = min(1.0, (0.75 if is_success else 0.2 if not is_failed else 0.0) + (0.1 if retrieval_used else 0.0))
        stop_target = 1.0 if is_success else 0.0
        score_target = max(0.0, min(1.0, 0.4 * policy_target + 0.35 * value_target + 0.25 * (1.0 - risk_target)))
        decoder_input_ids, decoder_target_ids = encode_decoder_sequence(decoder_text, config, decoder_vocab=decoder_vocab)
        out.append(
            {
                "family_id": hashed_id("patch_action", int(config.family_vocab_size)),
                "path_level_ids": path_level_ids,
                "command_token_ids": token_rows,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids,
                "scalar_features": scalar_rows,
                "score_target": score_target,
                "policy_target": policy_target,
                "value_target": value_target,
                "stop_target": stop_target,
                "risk_target": risk_target,
                "transition_target": [
                    1.0 if is_success else 0.0,
                    float(len(labels)),
                ],
                "task_difficulty": str(task.get("metadata", {}).get("difficulty", "patch_action"))
                if isinstance(task.get("metadata", {}), dict)
                else "patch_action",
                "example_weight": 1.25 if retrieval_used and is_success else 1.0,
                "world_target": _hybrid_world_target(example, config),
                "source_example_id": str(example.get("example_id", "")),
            }
        )
    return out, decoder_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patch-action training records from agentkernel patch artifacts.")
    parser.add_argument("--prediction-task-manifest", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--skill-cards-jsonl", default="")
    parser.add_argument("--patch-job-verification-json", default="")
    parser.add_argument("--output", "--output-jsonl", dest="output_path", required=True, help="Output dataset path; use .parquet for compressed Parquet or .jsonl for compatibility.")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--no-include-diff", action="store_true")
    parser.add_argument("--hybrid-output", "--hybrid-output-jsonl", dest="hybrid_output_path", default="", help="Optional trainer-ready HybridTolBERTSSM dataset path; use .parquet or .jsonl.")
    parser.add_argument("--hybrid-config-json", default="", help="Optional HybridTolBERTSSMConfig JSON.")
    args = parser.parse_args()

    examples = build_patch_action_dataset(
        prediction_task_manifest=_load_payload(args.prediction_task_manifest),
        queue_manifest=_load_payload(args.queue_manifest),
        predictions_jsonl=str(args.predictions_jsonl or ""),
        skill_cards_jsonl=str(args.skill_cards_jsonl or ""),
        patch_job_verification=(_load_payload(args.patch_job_verification_json) if str(args.patch_job_verification_json).strip() else None),
        include_diff=not bool(args.no_include_diff),
    )
    output = Path(args.output_path)
    output_format = write_dataset(output, examples)
    hybrid_path = Path(args.hybrid_output_path) if str(args.hybrid_output_path).strip() else None
    hybrid_count = 0
    decoder_vocab_path = ""
    hybrid_format = ""
    if hybrid_path is not None:
        config = _load_hybrid_config(str(args.hybrid_config_json or ""))
        hybrid_examples, decoder_vocab = materialize_hybrid_examples(examples, config=config)
        hybrid_format = write_dataset(hybrid_path, hybrid_examples)
        hybrid_count = len(hybrid_examples)
        vocab_path = hybrid_path.with_suffix(".decoder_vocab.json")
        vocab_path.write_text(json.dumps(decoder_vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        decoder_vocab_path = str(vocab_path)
        hybrid_path.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "artifact_kind": "tolbert_hybrid_training_dataset",
                    "source_artifact_kind": "agentkernel_patch_action_dataset",
                    "dataset_path": str(hybrid_path),
                    "dataset_format": hybrid_format,
                    "example_count": hybrid_count,
                    "decoder_vocab_path": decoder_vocab_path,
                    "decoder_vocab_entry_count": len(decoder_vocab),
                    "config": config.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = {
        "artifact_kind": "agentkernel_patch_action_dataset",
        "schema_version": SCHEMA_VERSION,
        "example_count": len(examples),
        "output_path": str(output),
        "output_format": output_format,
        "hybrid_output_jsonl": str(hybrid_path) if hybrid_path is not None else "",
        "hybrid_output_path": str(hybrid_path) if hybrid_path is not None else "",
        "hybrid_output_format": hybrid_format,
        "hybrid_example_count": hybrid_count,
        "hybrid_decoder_vocab_path": decoder_vocab_path,
        "intent_counts": {},
    }
    counts: dict[str, int] = {}
    for example in examples:
        for label in example.get("model_targets", {}).get("path_labels", []):
            counts[str(label)] = counts.get(str(label), 0) + 1
    manifest["intent_counts"] = dict(sorted(counts.items()))
    manifest_path = Path(args.manifest_out) if str(args.manifest_out).strip() else output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
