from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.neural_controller import EXEC_KIND_FAMILY


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
    documents = report.get("documents", [])
    if not isinstance(documents, list):
        return out
    for document in documents:
        if not isinstance(document, dict):
            continue
        steps = document.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        metadata = steps[0].get("proposal_metadata", {}) if isinstance(steps[0], dict) else {}
        if not isinstance(metadata, dict):
            continue
        shadow = metadata.get("neural_controller_shadow", {})
        if not isinstance(shadow, dict):
            continue
        example_id = str(shadow.get("example_id", "")).strip()
        if example_id:
            out[example_id] = shadow
    return out


def _family(shadow: dict[str, Any]) -> str:
    return EXEC_KIND_FAMILY.get(str(shadow.get("target_exec_kind", "")).strip(), "unknown")


def _wins(shadow: dict[str, Any], metric: str) -> bool:
    if metric == "content":
        return bool(shadow.get("content_exact_agreement", False))
    if metric == "exec_kind":
        return bool(shadow.get("exec_kind_agreement", False))
    if metric == "either":
        return bool(shadow.get("content_exact_agreement", False)) or bool(shadow.get("exec_kind_agreement", False))
    raise ValueError(f"unknown preservation metric: {metric}")


def _with_replay_metadata(
    row: dict[str, Any],
    *,
    family: str,
    repeat_index: int,
    repeat_count: int,
    metric: str,
    distill_loss_weight: float,
) -> dict[str, Any]:
    copied = dict(row)
    base_id = str(copied.get("example_id") or copied.get("source_id") or "preservation_row")
    copied["example_id"] = f"{base_id}:preserve_{family}:{repeat_index:02d}"
    copied["source_type"] = "agentkernel_controller_preservation_replay"
    copied["source_id"] = base_id
    copied["task_type"] = "controller_long_horizon_preservation_replay"
    copied["preservation_family"] = family
    copied["preservation_metric"] = metric
    copied["preservation_repeat_index"] = int(repeat_index)
    copied["preservation_repeat_count"] = int(repeat_count)
    copied["distill_loss_weight"] = max(float(distill_loss_weight), 0.0)
    try:
        weight = float(copied.get("weight") or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    copied["weight"] = max(weight, min(float(repeat_count), 8.0))
    return copied


def build_preservation_replay(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _read_json_object(Path(args.baseline_report))
    candidate = _read_json_object(Path(args.candidate_report))
    eval_rows = _iter_jsonl(Path(args.eval_dataset))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = _shadow_by_example(baseline)
    candidate_rows = _shadow_by_example(candidate)
    eval_by_id = {str(row.get("example_id", "")).strip(): row for row in eval_rows}
    allowed_families = {
        item.strip()
        for item in str(args.family_include).split(",")
        if item.strip()
    }
    selected: list[dict[str, Any]] = []
    selected_eval: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    distill_loss_weight = float(getattr(args, "distill_loss_weight", 1.0) or 0.0)
    for example_id, baseline_shadow in sorted(baseline_rows.items()):
        candidate_shadow = candidate_rows.get(example_id)
        source_row = eval_by_id.get(example_id)
        if candidate_shadow is None or source_row is None:
            continue
        family = _family(baseline_shadow)
        if allowed_families and family not in allowed_families:
            continue
        if not _wins(baseline_shadow, str(args.metric)):
            continue
        if _wins(candidate_shadow, str(args.metric)):
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        selected_eval.append(
            _with_replay_metadata(
                source_row,
                family=family,
                repeat_index=0,
                repeat_count=1,
                metric=str(args.metric),
                distill_loss_weight=distill_loss_weight,
            )
        )
        for repeat_index in range(max(1, int(args.repeat))):
            selected.append(
                _with_replay_metadata(
                    source_row,
                    family=family,
                    repeat_index=repeat_index,
                    repeat_count=max(1, int(args.repeat)),
                    metric=str(args.metric),
                    distill_loss_weight=distill_loss_weight,
                )
            )
    if not selected:
        raise SystemExit("no preservation replay rows selected")

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    for path, rows in ((train_path, selected), (eval_path, selected_eval)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_controller_preservation_replay_dataset",
        "objective": str(args.objective),
        "dataset_format": "jsonl",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "baseline_report": str(args.baseline_report),
        "candidate_report": str(args.candidate_report),
        "source_eval_dataset": str(args.eval_dataset),
        "total_examples": len(selected) + len(selected_eval),
        "train_examples": len(selected),
        "eval_examples": len(selected_eval),
        "source_counts": {"agentkernel_preservation_replay": len(selected) + len(selected_eval)},
        "preservation_replay": {
            "family_counts": dict(sorted(family_counts.items())),
            "family_include": sorted(allowed_families),
            "metric": str(args.metric),
            "repeat": int(args.repeat),
            "distill_loss_weight": distill_loss_weight,
        },
    }
    source_manifest = str(args.source_manifest or "").strip()
    if source_manifest:
        source_payload = _read_json_object(Path(source_manifest))
        tokens = source_payload.get("agentkernel_special_tokens", [])
        if isinstance(tokens, list):
            manifest["agentkernel_special_tokens"] = [str(token) for token in tokens]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="agentkernel_controller_preservation_replay")
    parser.add_argument("--family-include", default="")
    parser.add_argument("--metric", choices=("content", "exec_kind", "either"), default="either")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--distill-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build_preservation_replay(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
