from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_neural_controller_shadow_dataset import evaluate_dataset
from scripts.select_neural_controller_candidate import select_candidate


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def checkpoint_label(path: Path) -> str:
    parent = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    stem = path.stem
    if stem.startswith("step_"):
        return f"{parent}_{stem}"
    return f"{parent}_{stem}".strip("_")


def _copy_tokenizer(template_tokenizer_dir: Path, target_tokenizer_dir: Path) -> None:
    target_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for source in template_tokenizer_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, target_tokenizer_dir / source.name)


def export_checkpoint_bundle(
    *,
    checkpoint_path: Path,
    template_manifest_path: Path,
    output_dir: Path,
    repo_root: Path,
    device: str,
) -> Path:
    import torch

    model_stack = repo_root / "other_repos" / "model-stack"
    for path in (repo_root, model_stack):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from runtime.checkpoint import load_config, save_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    from agent_kernel.modeling.neural_controller_runtime import _materialize_lazy_modules

    template = _read_json_object(template_manifest_path)
    template_model_dir = Path(str(template.get("model_dir", "")))
    template_tokenizer_dir = Path(str(template.get("tokenizer_dir", "")))
    if not template_model_dir.exists():
        raise FileNotFoundError(f"template model_dir not found: {template_model_dir}")
    if not template_tokenizer_dir.exists():
        raise FileNotFoundError(f"template tokenizer_dir not found: {template_tokenizer_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    tokenizer_dir = output_dir / "tokenizer"
    config = load_config(str(template_model_dir))
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    payload = torch.load(checkpoint_path, map_location=torch.device(device))
    state = payload.get("model_state_dict", {})
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint missing model_state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in missing
        if key != "enc_pos_embed.weight"
        and not key.startswith("retrieval_query_head.")
        and not key.startswith("retrieval_doc_head.")
        and not key.startswith("agent_policy_heads.")
        and not key.startswith("encoder_scalar_control.")
        and not key.startswith("decoder_scalar_control.")
    ]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    save_pretrained(model.eval().cpu(), config, str(model_dir))
    _copy_tokenizer(template_tokenizer_dir, tokenizer_dir)

    manifest = dict(template)
    manifest["manifest_path"] = str(output_dir / "agentkernel_controller_manifest.json")
    manifest["model_dir"] = str(model_dir)
    manifest["tokenizer_dir"] = str(tokenizer_dir)
    manifest["selected_checkpoint_path"] = str(checkpoint_path)
    manifest["selected_checkpoint_step"] = int(payload.get("step", 0) or 0)
    manifest["artifact_kind"] = str(manifest.get("artifact_kind") or "agentkernel_controller_seq2seq_bundle")
    manifest["model_family"] = str(manifest.get("model_family") or "agentkernel_controller_seq2seq_v1")
    manifest["full_agent_kernel_controller"] = True
    manifest.setdefault("training_summary", {})
    if isinstance(manifest["training_summary"], dict):
        manifest["training_summary"]["selected_checkpoint_path"] = str(checkpoint_path)
        manifest["training_summary"]["selected_checkpoint_step"] = int(payload.get("step", 0) or 0)
        manifest["training_summary"]["primary_authority_allowed"] = False
    manifest_path = output_dir / "agentkernel_controller_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "agentkernel_lite_encdec_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def select_checkpoints(
    *,
    baseline_report_path: Path,
    checkpoint_paths: list[Path],
    template_manifest_path: Path,
    dataset_path: Path,
    output_dir: Path,
    repo_root: Path,
    device: str,
    limit: int,
    max_new_tokens: int,
    max_encoder_tokens: int,
    min_family_total: int,
    tolerance: float,
    progress_every: int,
    resume_partial: bool,
) -> dict[str, Any]:
    reports: list[Path] = []
    checkpoint_entries: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        label = checkpoint_label(checkpoint_path)
        bundle_dir = output_dir / "bundles" / label
        report_path = output_dir / "reports" / f"{label}_shadow_report.json"
        manifest_path = export_checkpoint_bundle(
            checkpoint_path=checkpoint_path,
            template_manifest_path=template_manifest_path,
            output_dir=bundle_dir,
            repo_root=repo_root,
            device=device,
        )
        report = evaluate_dataset(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            output_path=report_path,
            repo_root=repo_root,
            device=device,
            limit=limit,
            task_type="",
            max_new_tokens=max_new_tokens,
            max_encoder_tokens=max_encoder_tokens,
            progress_every=progress_every,
            resume_partial=resume_partial,
        )
        reports.append(report_path)
        checkpoint_entries.append(
            {
                "label": label,
                "checkpoint_path": str(checkpoint_path),
                "manifest_path": str(manifest_path),
                "report_path": str(report_path),
                "summary": report.get("summary", {}),
                "family_metrics": report.get("family_metrics", {}),
            }
        )

    selection = select_candidate(
        baseline_report_path=baseline_report_path,
        candidate_report_paths=reports,
        baseline_label=Path(baseline_report_path).stem.replace("_slot_eval132_shadow_report", ""),
        min_family_total=min_family_total,
        tolerance=tolerance,
    )
    selection["report_kind"] = "neural_controller_checkpoint_selection"
    selection["template_manifest_path"] = str(template_manifest_path)
    selection["dataset_path"] = str(dataset_path)
    selection["checkpoint_entries"] = checkpoint_entries
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--template-manifest", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=132)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--min-family-total", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint:
        raise ValueError("at least one --checkpoint is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection = select_checkpoints(
        baseline_report_path=Path(args.baseline_report).expanduser().resolve(),
        checkpoint_paths=[Path(path).expanduser().resolve() for path in args.checkpoint],
        template_manifest_path=Path(args.template_manifest).expanduser().resolve(),
        dataset_path=Path(args.dataset_path).expanduser().resolve(),
        output_dir=output_dir,
        repo_root=Path(args.repo_root).expanduser().resolve(),
        device=str(args.device),
        limit=int(args.limit),
        max_new_tokens=int(args.max_new_tokens),
        max_encoder_tokens=int(args.max_encoder_tokens),
        min_family_total=int(args.min_family_total),
        tolerance=float(args.tolerance),
        progress_every=int(args.progress_every),
        resume_partial=bool(args.resume_partial),
    )
    output_path = output_dir / "checkpoint_selection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "neural_controller_checkpoint_selection "
        f"strict_recommendation={selection['strict_recommendation']} "
        f"accepted={selection['accepted_candidate_label'] or 'none'} "
        f"best_diagnostic={selection['best_diagnostic_candidate_label'] or 'none'} "
        f"checkpoints={len(selection['checkpoint_entries'])}"
    )


if __name__ == "__main__":
    main()
