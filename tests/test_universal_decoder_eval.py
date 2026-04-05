from __future__ import annotations

from pathlib import Path
import json

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.evaluation.universal_decoder_eval import (
    _generate_seed_completion,
    evaluate_universal_decoder_against_seed,
)


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
    assert report.total_dataset_examples == 2
    assert report.selection_strategy == "full_dataset"
    assert report.sample_coverage_rate == 1.0
    assert report.sampled_source_type_count == 2
    assert report.total_source_type_count == 2
    assert report.source_type_coverage_rate == 1.0
    assert "high_model_disagreement" in report.warnings
    assert report.hybrid_exact_match_rate == 0.5
    assert report.baseline_exact_match_rate == 1.0
    assert report.hybrid_win_rate == 0.0
    assert report.baseline_win_rate == 0.5
    assert report.disagreement_rate == 0.5
    assert report.slices["docs_markdown"]["hybrid_exact_match_rate"] == 1.0
    assert report.slices["external_corpus"]["hybrid_exact_match_rate"] == 0.0
    assert report.slices["external_corpus"]["baseline_exact_match_rate"] == 1.0
    assert report.slices["target_length:short"]["example_count"] == 2
    assert seen_prompts == ["alpha", "beta"]


def test_evaluate_universal_decoder_against_seed_uses_stratified_sampling_for_source_coverage(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    eval_path = dataset_dir / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "a1", "target": "done", "source_type": "docs_markdown"}),
                json.dumps({"prompt": "a2", "target": "done", "source_type": "docs_markdown"}),
                json.dumps({"prompt": "b1", "target": "done", "source_type": "external_corpus"}),
                json.dumps({"prompt": "b2", "target": "done", "source_type": "external_corpus"}),
                json.dumps({"prompt": "c1", "target": "done", "source_type": "repo_notes"}),
                json.dumps({"prompt": "c2", "target": "done", "source_type": "repo_notes"}),
            ]
        )
        + "\n",
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

    monkeypatch.setattr(
        "agent_kernel.modeling.evaluation.universal_decoder_eval.generate_hybrid_decoder_completion",
        lambda **kwargs: seen_prompts.append(str(kwargs["prompt"])) or {"generated_text": "done"},
    )
    monkeypatch.setattr(
        "agent_kernel.modeling.evaluation.universal_decoder_eval._generate_seed_completion",
        lambda **kwargs: "done",
    )

    report = evaluate_universal_decoder_against_seed(
        hybrid_bundle_manifest_path=tmp_path / "bundle.json",
        dataset_manifest_path=manifest_path,
        config=KernelConfig(),
        max_examples=3,
    )

    assert report.example_count == 3
    assert report.total_dataset_examples == 6
    assert report.selection_strategy == "stratified_source_type_even_spread"
    assert report.sample_coverage_rate == 0.5
    assert report.sampled_source_type_count == 3
    assert report.total_source_type_count == 3
    assert report.source_type_coverage_rate == 1.0
    assert "small_sample" in report.warnings
    assert "partial_dataset_coverage" in report.warnings
    assert seen_prompts == ["a2", "b2", "c2"]


def test_generate_seed_completion_uses_llm_timeout_seconds(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    def _fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("agent_kernel.modeling.evaluation.universal_decoder_eval.request.urlopen", _fake_urlopen)

    config = KernelConfig(vllm_host="http://127.0.0.1:8000", llm_timeout_seconds=37)
    result = _generate_seed_completion(config=config, prompt="hello", max_tokens=4)

    assert result == "ok"
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["timeout"] == 37


def test_generate_seed_completion_falls_back_to_reasoning_when_content_is_null(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": "fallback text",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "agent_kernel.modeling.evaluation.universal_decoder_eval.request.urlopen",
        lambda req, timeout: _Response(),
    )

    config = KernelConfig(vllm_host="http://127.0.0.1:8000", llm_timeout_seconds=20)
    result = _generate_seed_completion(config=config, prompt="hello", max_tokens=4)

    assert result == "fallback text"
