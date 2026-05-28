from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.neural_controller import (
    EXEC_KIND_FAMILY,
    compact_neural_controller_shadow,
    parse_neural_controller_line_protocol,
    repair_line_protocol_with_command_copy_target,
    summarize_neural_controller_shadow_documents,
    neural_controller_shadow_promotion_readiness,
)
from agent_kernel.modeling.neural_controller_runtime import generate_neural_controller_text


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iter_jsonl(path: Path, *, limit: int = 0, task_type: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if task_type and str(payload.get("task_type", "")) != task_type:
                continue
            rows.append(payload)
    return rows


def _target_line_protocol(row: dict[str, Any]) -> dict[str, Any]:
    target = parse_neural_controller_line_protocol(str(row.get("decoder_text", "")))
    target_content = str(target.get("content", "")).strip()
    if (
        target_content in {"<AK_COPY_COMMAND_TARGET>", "<AK_COPY_ARTIFACT_TARGET>"}
        or target_content.startswith("<AK_COPY_LOCALIZED_EDIT_CANDIDATE_")
    ):
        target, _warnings = repair_line_protocol_with_command_copy_target(
            target,
            encoder_text=str(row.get("encoder_text", "")),
        )
    return target


def _prediction_shadow(
    *,
    manifest_path: Path,
    row: dict[str, Any],
    repo_root: Path,
    device: str,
    max_new_tokens: int,
    max_encoder_tokens: int,
) -> dict[str, Any]:
    generated = generate_neural_controller_text(
        manifest_path=manifest_path,
        encoder_text=str(row.get("encoder_text", "")),
        repo_root=repo_root,
        device=device,
        max_new_tokens=max_new_tokens,
        max_encoder_tokens=max_encoder_tokens,
    )
    line_protocol = parse_neural_controller_line_protocol(str(generated.get("generated_text", "")))
    line_protocol, warnings = repair_line_protocol_with_command_copy_target(
        line_protocol,
        encoder_text=str(row.get("encoder_text", "")),
    )
    target = _target_line_protocol(row)
    compact = compact_neural_controller_shadow(
        {
            "ready": bool(line_protocol.get("action")),
            "manifest_path": str(manifest_path.resolve()),
            "generated_token_count": int(generated.get("generated_token_count") or 0),
            "line_protocol": line_protocol,
            "policy_heads": generated.get("policy_heads", {}),
            "scalar_control": generated.get("scalar_control", {}),
            "warnings": warnings,
        },
        selected_action=str(target.get("action", "")),
        selected_content=str(target.get("content", "")),
    )
    compact["example_id"] = str(row.get("example_id", ""))
    compact["task_type"] = str(row.get("task_type", ""))
    compact["target_content_preview"] = compact.get("selected_content_preview", "")
    predicted_tokens = set(compact.get("control_tokens", []))
    target_tokens = [
        str(token).strip()
        for token in target.get("tokens", [])
        if str(token).strip().startswith("<AK_")
    ]
    target_token_set = set(target_tokens)
    compact["target_control_tokens"] = target_tokens[:16]
    exec_kind_tokens = {
        "<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>",
        "<AK_EXEC_KIND_VERIFY_PRESENT>",
        "<AK_EXEC_KIND_VERIFY_ABSENT>",
        "<AK_EXEC_KIND_INSPECT_SOURCE>",
        "<AK_EXEC_KIND_LOCALIZED_EDIT>",
        "<AK_EXEC_KIND_RUN_CHECK>",
    }
    predicted_exec_kind = sorted(predicted_tokens & exec_kind_tokens)
    target_exec_kind = sorted(target_token_set & exec_kind_tokens)
    if predicted_exec_kind or target_exec_kind:
        compact["predicted_exec_kind"] = predicted_exec_kind[0] if predicted_exec_kind else ""
        compact["target_exec_kind"] = target_exec_kind[0] if target_exec_kind else ""
        compact["exec_kind_agreement"] = bool(predicted_exec_kind and predicted_exec_kind == target_exec_kind)
    compact["control_token_subset_agreement"] = bool(target_token_set and target_token_set.issubset(predicted_tokens))
    slot_keys = ("target_path", "target_content", "edit_old", "edit_new", "verify_polarity")
    slot_agreements: dict[str, bool] = {}
    for key in slot_keys:
        predicted_value = str(line_protocol.get(key, "")).strip()
        target_value = str(target.get(key, "")).strip()
        if predicted_value or target_value:
            compact[f"predicted_{key}"] = predicted_value[:240]
            compact[f"target_{key}"] = target_value[:240]
            slot_agreements[key] = predicted_value == target_value and bool(target_value)
    if slot_agreements:
        compact["slot_agreements"] = slot_agreements
        compact["slot_agreement_rate"] = round(
            sum(1 for value in slot_agreements.values() if value) / len(slot_agreements),
            6,
        )
    return compact


def summarize_family_metrics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    slot_keys = ("target_path", "target_content", "edit_old", "edit_new", "verify_polarity")

    def family_row(name: str) -> dict[str, Any]:
        return families.setdefault(
            name,
            {
                "total": 0,
                "content_exact": 0,
                "contract_content": 0,
                "exec_kind_agreement": 0,
                "slot_total": {key: 0 for key in slot_keys},
                "slot_exact": {key: 0 for key in slot_keys},
            },
        )

    for document in documents:
        steps = document.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            metadata = step.get("proposal_metadata", {})
            if not isinstance(metadata, dict):
                continue
            shadow = metadata.get("neural_controller_shadow", {})
            if not isinstance(shadow, dict):
                continue
            target_exec_kind = str(shadow.get("target_exec_kind", "")).strip()
            family = EXEC_KIND_FAMILY.get(target_exec_kind, "unknown")
            row = family_row(family)
            row["total"] += 1
            if bool(shadow.get("content_exact_agreement", False)):
                row["content_exact"] += 1
            if bool(shadow.get("content_exact_agreement", False)) or (
                str(shadow.get("artifact_failure_mode", "")).strip() == "artifact_contract_success"
            ):
                row["contract_content"] += 1
            if bool(shadow.get("exec_kind_agreement", False)):
                row["exec_kind_agreement"] += 1
            slot_agreements = shadow.get("slot_agreements", {})
            if not isinstance(slot_agreements, dict):
                slot_agreements = {}
            for key in slot_keys:
                target_value = str(shadow.get(f"target_{key}", "")).strip()
                if not target_value:
                    continue
                row["slot_total"][key] += 1
                if bool(slot_agreements.get(key, False)):
                    row["slot_exact"][key] += 1

    normalized: dict[str, Any] = {}
    for family, row in sorted(families.items()):
        total = int(row["total"])
        slot_rates = {}
        for key in slot_keys:
            denominator = int(row["slot_total"].get(key, 0))
            slot_rates[key] = round(int(row["slot_exact"].get(key, 0)) / denominator, 6) if denominator else None
        normalized[family] = {
            "total": total,
            "content_exact": int(row["content_exact"]),
            "content_exact_rate": round(int(row["content_exact"]) / total, 6) if total else 0.0,
            "contract_content": int(row["contract_content"]),
            "contract_content_rate": round(int(row["contract_content"]) / total, 6) if total else 0.0,
            "exec_kind_agreement": int(row["exec_kind_agreement"]),
            "exec_kind_agreement_rate": round(int(row["exec_kind_agreement"]) / total, 6) if total else 0.0,
            "slot_exact": row["slot_exact"],
            "slot_total": row["slot_total"],
            "slot_rates": slot_rates,
        }
    if normalized:
        normalized["_macro"] = _macro_family_metrics(normalized)
    return normalized


def _macro_family_metrics(families: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for family, row in families.items()
        if not family.startswith("_") and int(row.get("total", 0)) > 0
    ]
    if not rows:
        return {}
    content_rates = [float(row.get("content_exact_rate", 0.0)) for row in rows]
    contract_rates = [float(row.get("contract_content_rate", row.get("content_exact_rate", 0.0))) for row in rows]
    exec_rates = [float(row.get("exec_kind_agreement_rate", 0.0)) for row in rows]
    return {
        "family_count": len(rows),
        "macro_content_exact_rate": round(sum(content_rates) / len(content_rates), 6),
        "macro_contract_content_rate": round(sum(contract_rates) / len(contract_rates), 6),
        "macro_exec_kind_agreement_rate": round(sum(exec_rates) / len(exec_rates), 6),
        "min_content_exact_rate": round(min(content_rates), 6),
        "min_contract_content_rate": round(min(contract_rates), 6),
        "min_exec_kind_agreement_rate": round(min(exec_rates), 6),
    }


def evaluate_dataset(
    *,
    manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    repo_root: Path,
    device: str,
    limit: int,
    task_type: str,
    max_new_tokens: int,
    max_encoder_tokens: int,
    progress_every: int = 0,
    resume_partial: bool = False,
) -> dict[str, Any]:
    rows = _iter_jsonl(dataset_path, limit=limit, task_type=task_type)
    documents: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_index = 0
    if resume_partial:
        documents = _load_partial_documents(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            output_path=output_path,
            total_rows=len(rows),
        )
        start_index = min(len(documents), len(rows))
        if start_index:
            print(
                "neural_controller_shadow_dataset_eval_resume "
                f"rows={start_index}/{len(rows)} "
                f"partial={output_path.with_suffix(output_path.suffix + '.partial')}",
                flush=True,
            )
    for index, row in enumerate(rows[start_index:], start=start_index):
        shadow = _prediction_shadow(
            manifest_path=manifest_path,
            row=row,
            repo_root=repo_root,
            device=device,
            max_new_tokens=max_new_tokens,
            max_encoder_tokens=max_encoder_tokens,
        )
        documents.append(
            {
                "task_id": str(row.get("example_id", f"row_{index}")),
                "steps": [
                    {
                        "proposal_metadata": {"neural_controller_shadow": shadow},
                        "verification": {"passed": True},
                    }
                ],
            }
        )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            _write_partial_report(
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                output_path=output_path,
                limit=limit,
                task_type=task_type,
                documents=documents,
                completed_rows=index + 1,
                total_rows=len(rows),
            )
            print(
                "neural_controller_shadow_dataset_eval_progress "
                f"rows={index + 1}/{len(rows)} "
                f"output={output_path}",
                flush=True,
            )
    summary = summarize_neural_controller_shadow_documents(documents)
    family_metrics = summarize_family_metrics(documents)
    report = {
        "report_kind": "neural_controller_shadow_dataset_eval",
        "manifest_path": str(manifest_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "limit": int(limit),
        "task_type_filter": task_type,
        "documents": documents,
        "summary": summary,
        "family_metrics": family_metrics,
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _write_partial_report(
    *,
    manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int,
    task_type: str,
    documents: list[dict[str, Any]],
    completed_rows: int,
    total_rows: int,
) -> None:
    summary = summarize_neural_controller_shadow_documents(documents)
    family_metrics = summarize_family_metrics(documents)
    report = {
        "report_kind": "neural_controller_shadow_dataset_eval_partial",
        "manifest_path": str(manifest_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "limit": int(limit),
        "task_type_filter": task_type,
        "completed_rows": int(completed_rows),
        "total_rows": int(total_rows),
        "documents": documents,
        "summary": summary,
        "family_metrics": family_metrics,
        "promotion_readiness": neural_controller_shadow_promotion_readiness(summary),
    }
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    partial_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_partial_documents(
    *,
    manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    total_rows: int,
) -> list[dict[str, Any]]:
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if not partial_path.exists():
        return []
    try:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    if str(payload.get("manifest_path", "")) != str(manifest_path.resolve()):
        return []
    if str(payload.get("dataset_path", "")) != str(dataset_path.resolve()):
        return []
    if int(payload.get("total_rows", 0) or 0) != int(total_rows):
        return []
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return []
    return [document for document in documents if isinstance(document, dict)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--task-type", default="")
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--max-encoder-tokens", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(
        str(_read_json_object(manifest_path).get("training_summary", {}).get("eval_dataset_path", ""))
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"eval dataset not found: {dataset_path}")
    report = evaluate_dataset(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        output_path=Path(args.output),
        repo_root=Path(args.repo_root),
        device=args.device,
        limit=args.limit,
        task_type=args.task_type,
        max_new_tokens=args.max_new_tokens,
        max_encoder_tokens=args.max_encoder_tokens,
        progress_every=args.progress_every,
        resume_partial=bool(args.resume_partial),
    )
    summary = report["summary"]
    readiness = report["promotion_readiness"]
    print(
        "neural_controller_shadow_dataset_eval "
        f"rows={len(report['documents'])} "
        f"ready_steps={summary.get('ready_steps', 0)} "
        f"content_comparison_steps={summary.get('content_comparison_steps', 0)} "
        f"content_exact_agreement_rate={summary.get('content_exact_agreement_rate', 0.0):.3f} "
        f"warning_rate={summary.get('warning_rate', 0.0):.3f} "
        f"shadow_compare_ready={str(readiness.get('shadow_compare_ready', False)).lower()} "
        f"content_authority_ready={str(readiness.get('content_authority_ready', False)).lower()}"
    )


if __name__ == "__main__":
    main()
