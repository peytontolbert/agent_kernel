from __future__ import annotations

from pathlib import Path
import json

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.evaluation.universal_decoder_eval import evaluate_universal_decoder_against_seed


def test_evaluate_universal_decoder_against_seed_aggregates_metrics(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    eval_path = dataset_dir / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"prompt": "alpha", "target": "done", "source_type": "docs_markdown"}) + "\n"
        + json.dumps({"prompt": "beta", "target": "next", "source_type": "external_corpus"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_universal_decoder_dataset",
                "eval_dataset_path": str(eval_path),
            }
        ),
        encoding="utf-8",
    )

    seen_prompts: list[str] = []

    def _fake_hybrid(**kwargs):
        seen_prompts.append(str(kwargs["prompt"]))
        prompt = str(kwargs["prompt"])
        return {"generated_text": "done" if prompt == "alpha" else "wrong"}

    def _fake_seed(**kwargs):
        return "done" if kwargs["prompt"] == "alpha" else "next"

    monkeypatch.setattr(
        "agent_kernel.modeling.evaluation.universal_decoder_eval.generate_hybrid_decoder_completion",
        _fake_hybrid,
    )
    monkeypatch.setattr(
        "agent_kernel.modeling.evaluation.universal_decoder_eval._generate_seed_completion",
        _fake_seed,
    )

    report = evaluate_universal_decoder_against_seed(
        hybrid_bundle_manifest_path=tmp_path / "bundle.json",
        dataset_manifest_path=manifest_path,
        config=KernelConfig(),
        max_examples=2,
    )

    assert report.example_count == 2
    assert report.hybrid_exact_match_rate == 0.5
    assert report.baseline_exact_match_rate == 1.0
    assert report.slices["docs_markdown"]["hybrid_exact_match_rate"] == 1.0
    assert report.slices["external_corpus"]["hybrid_exact_match_rate"] == 0.0
    assert report.slices["external_corpus"]["baseline_exact_match_rate"] == 1.0
    assert seen_prompts == ["alpha", "beta"]
