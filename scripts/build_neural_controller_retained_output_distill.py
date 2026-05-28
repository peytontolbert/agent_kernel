from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import (  # noqa: E402
    FULL_KERNEL_CONTROL_TOKENS,
    localized_edit_candidates_from_encoder,
    materialization_candidates_from_encoder,
    source_inspection_candidates_from_encoder,
    validation_command_candidates_from_encoder,
)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _shadow_by_example(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for document in report.get("documents", []):
        if not isinstance(document, dict):
            continue
        steps = document.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        step = steps[0] if isinstance(steps[0], dict) else {}
        metadata = step.get("proposal_metadata", {}) if isinstance(step, dict) else {}
        shadow = metadata.get("neural_controller_shadow", {}) if isinstance(metadata, dict) else {}
        if not isinstance(shadow, dict):
            continue
        example_id = str(shadow.get("example_id", "")).strip() or str(document.get("task_id", "")).strip()
        if example_id:
            out[example_id] = shadow
    return out


def _line_value(value: Any) -> str:
    return str(value or "").replace("\n", "\\n").strip()


def _pointerized_content(*, content: str, encoder_text: str) -> tuple[str, str]:
    text = str(content or "").strip()
    if not text:
        return "", ""
    materialization_candidates = materialization_candidates_from_encoder(encoder_text)
    for index, candidate in enumerate(materialization_candidates[:24], start=1):
        if text == candidate:
            return f"<AK_COPY_MATERIALIZE_CANDIDATE_{index}>", f"<AK_COPY_MATERIALIZE_CANDIDATE_{index}>"
    source_candidates = source_inspection_candidates_from_encoder(encoder_text)
    for index, candidate in enumerate(source_candidates[:24], start=1):
        if text == candidate:
            return f"<AK_COPY_SOURCE_INSPECT_CANDIDATE_{index}>", f"<AK_COPY_SOURCE_INSPECT_CANDIDATE_{index}>"
    for polarity, prefix in (("present", "PRESENT"), ("absent", "ABSENT")):
        candidates = validation_command_candidates_from_encoder(encoder_text, polarity=polarity)
        for index, candidate in enumerate(candidates[:24], start=1):
            if text == candidate:
                return f"<AK_COPY_VALIDATE_{prefix}_CANDIDATE_{index}>", f"<AK_COPY_VALIDATE_{prefix}_CANDIDATE_{index}>"
    localized_candidates = localized_edit_candidates_from_encoder(encoder_text)
    for index, candidate in enumerate(localized_candidates[:24], start=1):
        if text == candidate:
            return f"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_{index}>", f"<AK_COPY_LOCALIZED_EDIT_CANDIDATE_{index}>"
    return text, ""


def _decoder_text_from_shadow(
    shadow: dict[str, Any],
    *,
    encoder_text: str,
    allow_preview_fallback: bool,
    pointerize_candidates: bool,
) -> tuple[str, str]:
    tokens = [
        str(token).strip()
        for token in shadow.get("target_control_tokens", [])
        if str(token).strip().startswith("<AK_")
    ]
    if not tokens:
        tokens = [
            str(token).strip()
            for token in shadow.get("control_tokens", [])
            if str(token).strip().startswith("<AK_")
        ]
    selected_action = str(shadow.get("selected_action") or shadow.get("predicted_action") or "code_execute").strip()
    selected_content = str(shadow.get("selected_content", "")).strip()
    if not selected_content and allow_preview_fallback:
        selected_content = str(shadow.get("selected_content_preview", "")).strip()
    if not selected_content:
        raise ValueError("retained shadow row is missing selected_content")
    pointer_token = ""
    if pointerize_candidates:
        selected_content, pointer_token = _pointerized_content(content=selected_content, encoder_text=encoder_text)
        if pointer_token and pointer_token not in tokens:
            tokens.append(pointer_token)
    lines: list[str] = []
    if tokens:
        lines.append(" ".join(tokens))
    lines.append(f"Action: {_line_value(selected_action)}")
    failure_mode = str(shadow.get("artifact_failure_mode", "")).strip()
    if failure_mode:
        lines.append(f"Artifact-Failure-Mode: {_line_value(failure_mode)}")
    target_path = str(shadow.get("target_target_path") or shadow.get("predicted_target_path") or "").strip()
    if target_path:
        lines.append(f"Target-Path: {_line_value(target_path)}")
    target_content = str(shadow.get("target_target_content") or shadow.get("predicted_target_content") or "").strip()
    if target_content:
        lines.append(f"Target-Content: {_line_value(target_content)}")
    verify_polarity = str(shadow.get("target_verify_polarity") or shadow.get("predicted_verify_polarity") or "").strip()
    if verify_polarity:
        lines.append(f"Verify-Polarity: {_line_value(verify_polarity)}")
    edit_old = str(shadow.get("target_edit_old") or shadow.get("predicted_edit_old") or "").strip()
    if edit_old:
        lines.append(f"Edit-Old: {_line_value(edit_old)}")
    edit_new = str(shadow.get("target_edit_new") or shadow.get("predicted_edit_new") or "").strip()
    if edit_new:
        lines.append(f"Edit-New: {_line_value(edit_new)}")
    lines.append(f"Content: {_line_value(selected_content)}")
    return "\n".join(lines), pointer_token


def _pointer_token_matches_family(pointer_token: str, family: str) -> bool:
    token = str(pointer_token or "").strip()
    family = str(family or "").strip().lower()
    if not family:
        return True
    prefixes = {
        "materialize": "<AK_COPY_MATERIALIZE_CANDIDATE_",
        "source": "<AK_COPY_SOURCE_INSPECT_CANDIDATE_",
        "validate_present": "<AK_COPY_VALIDATE_PRESENT_CANDIDATE_",
        "validate_absent": "<AK_COPY_VALIDATE_ABSENT_CANDIDATE_",
        "localized_edit": "<AK_COPY_LOCALIZED_EDIT_CANDIDATE_",
    }
    prefix = prefixes.get(family, "")
    return bool(prefix and token.startswith(prefix))


def _pointer_tokens_from_decoder_text(decoder_text: str) -> list[str]:
    return re.findall(
        r"<AK_COPY_(?:MATERIALIZE|SOURCE_INSPECT|VALIDATE_PRESENT|VALIDATE_ABSENT|LOCALIZED_EDIT)_CANDIDATE_\d+>",
        str(decoder_text or ""),
    )


def _pointer_token_expands(pointer_token: str, encoder_text: str) -> bool:
    token = str(pointer_token or "").strip()
    match = re.fullmatch(r"<AK_COPY_(?:MATERIALIZE|SOURCE_INSPECT|VALIDATE_PRESENT|VALIDATE_ABSENT|LOCALIZED_EDIT)_CANDIDATE_(\d+)>", token)
    if not match:
        return False
    index = int(match.group(1))
    if "_MATERIALIZE_" in token:
        return index <= len(materialization_candidates_from_encoder(encoder_text))
    if "_SOURCE_INSPECT_" in token:
        return index <= len(source_inspection_candidates_from_encoder(encoder_text))
    if "_VALIDATE_PRESENT_" in token:
        return index <= len(validation_command_candidates_from_encoder(encoder_text, polarity="present"))
    if "_VALIDATE_ABSENT_" in token:
        return index <= len(validation_command_candidates_from_encoder(encoder_text, polarity="absent"))
    if "_LOCALIZED_EDIT_" in token:
        return index <= len(localized_edit_candidates_from_encoder(encoder_text))
    return False


def _pointer_grounding_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pointer_rows = 0
    pointer_tokens = 0
    invalid_pointer_tokens = 0
    invalid_examples: list[dict[str, str]] = []
    family_counts: dict[str, int] = {}
    for row in rows:
        tokens = _pointer_tokens_from_decoder_text(str(row.get("decoder_text", "")))
        if not tokens:
            continue
        pointer_rows += 1
        encoder_text = str(row.get("encoder_text", ""))
        for token in tokens:
            pointer_tokens += 1
            family = token.rsplit("_CANDIDATE_", 1)[0].replace("<AK_COPY_", "").lower()
            family_counts[family] = family_counts.get(family, 0) + 1
            if _pointer_token_expands(token, encoder_text):
                continue
            invalid_pointer_tokens += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(
                    {
                        "example_id": str(row.get("example_id", "")),
                        "pointer_token": token,
                    }
                )
    return {
        "pointer_rows": pointer_rows,
        "pointer_tokens": pointer_tokens,
        "invalid_pointer_tokens": invalid_pointer_tokens,
        "pointer_family_counts": family_counts,
        "invalid_examples": invalid_examples,
    }


def _copy_with_distill_target(
    row: dict[str, Any],
    shadow: dict[str, Any],
    *,
    repeat_index: int,
    repeat_count: int,
    distill_loss_weight: float,
    allow_preview_fallback: bool,
    pointerize_candidates: bool,
    only_pointer_family: str = "",
) -> dict[str, Any]:
    copied = dict(row)
    base_id = str(copied.get("example_id") or copied.get("source_id") or "retained_output")
    copied["example_id"] = f"{base_id}:retained_output:{repeat_index:02d}"
    copied["source_type"] = "agentkernel_retained_output_distill"
    copied["source_id"] = base_id
    copied["task_type"] = "controller_retained_output_distill"
    decoder_text, pointer_token = _decoder_text_from_shadow(
        shadow,
        encoder_text=str(row.get("encoder_text", "")),
        allow_preview_fallback=allow_preview_fallback,
        pointerize_candidates=pointerize_candidates,
    )
    if only_pointer_family and not _pointer_token_matches_family(pointer_token, only_pointer_family):
        return {}
    copied["decoder_text"] = decoder_text
    copied["retained_output_repeat_index"] = int(repeat_index)
    copied["retained_output_repeat_count"] = int(repeat_count)
    copied["distill_loss_weight"] = max(float(distill_loss_weight), 0.0)
    try:
        weight = float(copied.get("weight") or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    copied["weight"] = max(weight, min(float(repeat_count), 8.0))
    return copied


def build_retained_output_distill(args: argparse.Namespace) -> dict[str, Any]:
    retained_report_path = Path(args.retained_report).expanduser().resolve()
    source_eval_dataset = Path(args.eval_dataset).expanduser().resolve()
    source_manifest_path = Path(args.source_manifest).expanduser().resolve() if args.source_manifest else None
    retained_report = _read_json_object(retained_report_path)
    retained_by_id = _shadow_by_example(retained_report)
    eval_rows = _iter_jsonl(source_eval_dataset)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_eval: list[dict[str, Any]] = []
    selected_train: list[dict[str, Any]] = []
    missing_content = 0
    repeat = max(1, int(args.repeat))
    only_pointer_family = str(getattr(args, "only_pointer_family", "") or "")
    for row in eval_rows:
        example_id = str(row.get("example_id", "")).strip()
        shadow = retained_by_id.get(example_id)
        if not shadow:
            continue
        if args.only_content_wins and not bool(shadow.get("content_exact_agreement", False)):
            continue
        try:
            eval_row = _copy_with_distill_target(
                row,
                shadow,
                repeat_index=0,
                repeat_count=1,
                distill_loss_weight=float(args.distill_loss_weight),
                allow_preview_fallback=bool(args.allow_preview_fallback),
                pointerize_candidates=bool(args.pointerize_candidates),
                only_pointer_family=only_pointer_family,
            )
        except ValueError:
            missing_content += 1
            continue
        if not eval_row:
            continue
        selected_eval.append(eval_row)
        for repeat_index in range(repeat):
            train_row = _copy_with_distill_target(
                row,
                shadow,
                repeat_index=repeat_index,
                repeat_count=repeat,
                distill_loss_weight=float(args.distill_loss_weight),
                allow_preview_fallback=bool(args.allow_preview_fallback),
                pointerize_candidates=bool(args.pointerize_candidates),
                only_pointer_family=only_pointer_family,
            )
            if train_row:
                selected_train.append(train_row)
    if not selected_train:
        raise SystemExit("no retained-output distillation rows selected")

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    for path, rows in ((train_path, selected_train), (eval_path, selected_eval)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest: dict[str, Any] = {
        "artifact_kind": "agentkernel_retained_output_distill_dataset",
        "objective": str(args.objective),
        "dataset_format": "jsonl",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "retained_report": str(retained_report_path),
        "source_eval_dataset": str(source_eval_dataset),
        "total_examples": len(selected_train) + len(selected_eval),
        "train_examples": len(selected_train),
        "eval_examples": len(selected_eval),
        "source_counts": {"agentkernel_retained_output_distill": len(selected_train) + len(selected_eval)},
        "retained_output_distill": {
            "allow_preview_fallback": bool(args.allow_preview_fallback),
            "distill_loss_weight": float(args.distill_loss_weight),
            "missing_selected_content_rows": int(missing_content),
            "only_content_wins": bool(args.only_content_wins),
            "only_pointer_family": only_pointer_family,
            "pointer_grounding_audit": _pointer_grounding_audit(selected_train + selected_eval),
            "pointerize_candidates": bool(args.pointerize_candidates),
            "repeat": repeat,
        },
    }
    if source_manifest_path:
        source_manifest = _read_json_object(source_manifest_path)
        tokens = source_manifest.get("agentkernel_special_tokens", [])
        if isinstance(tokens, list):
            manifest["agentkernel_special_tokens"] = list(
                dict.fromkeys([str(token) for token in tokens] + list(FULL_KERNEL_CONTROL_TOKENS))
            )
    else:
        manifest["agentkernel_special_tokens"] = list(FULL_KERNEL_CONTROL_TOKENS)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-report", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="agentkernel_retained_output_distill")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--allow-preview-fallback", action="store_true")
    parser.add_argument("--only-content-wins", action="store_true")
    parser.add_argument("--pointerize-candidates", action="store_true")
    parser.add_argument(
        "--only-pointer-family",
        choices=("", "materialize", "source", "validate_present", "validate_absent", "localized_edit"),
        default="",
    )
    args = parser.parse_args()
    print(json.dumps(build_retained_output_distill(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
