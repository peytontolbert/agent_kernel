from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any
from urllib import request

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.tolbert import generate_hybrid_decoder_completion


@dataclass(slots=True)
class UniversalDecoderEvalReport:
    hybrid_exact_match_rate: float
    baseline_exact_match_rate: float
    hybrid_token_f1: float
    baseline_token_f1: float
    example_count: int
    slices: dict[str, dict[str, float | int | str]]
    artifact_kind: str = "tolbert_universal_decoder_eval"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_universal_decoder_against_seed(
    *,
    hybrid_bundle_manifest_path: Path,
    dataset_manifest_path: Path,
    config: KernelConfig,
    device: str = "cpu",
    max_examples: int = 64,
) -> UniversalDecoderEvalReport:
    manifest = _load_json(dataset_manifest_path)
    eval_dataset_path = Path(str(manifest.get("eval_dataset_path", "")).strip())
    examples = _load_jsonl(eval_dataset_path)[: max(1, max_examples)]
    hybrid_exact = 0.0
    baseline_exact = 0.0
    hybrid_f1_total = 0.0
    baseline_f1_total = 0.0
    slice_stats: dict[str, dict[str, float | int]] = {}
    for example in examples:
        prompt = str(example.get("prompt", ""))
        target = str(example.get("target", ""))
        source_type = str(example.get("source_type", "unknown")).strip() or "unknown"
        hybrid = generate_hybrid_decoder_completion(
            prompt=prompt,
            bundle_manifest_path=hybrid_bundle_manifest_path,
            device=device,
            max_new_tokens=len(target.split()) + 2,
        )
        baseline = _generate_seed_completion(config=config, prompt=prompt, max_tokens=len(target.split()) + 2)
        hybrid_text = str(hybrid.get("generated_text", "")).strip()
        baseline_text = str(baseline).strip()
        hybrid_exact_hit = 1.0 if _normalize_text(hybrid_text) == _normalize_text(target) else 0.0
        baseline_exact_hit = 1.0 if _normalize_text(baseline_text) == _normalize_text(target) else 0.0
        if hybrid_exact_hit:
            hybrid_exact += 1.0
        if baseline_exact_hit:
            baseline_exact += 1.0
        hybrid_f1 = _token_f1(hybrid_text, target)
        baseline_f1 = _token_f1(baseline_text, target)
        hybrid_f1_total += hybrid_f1
        baseline_f1_total += baseline_f1
        bucket = slice_stats.setdefault(
            source_type,
            {
                "example_count": 0,
                "hybrid_exact_total": 0.0,
                "baseline_exact_total": 0.0,
                "hybrid_f1_total": 0.0,
                "baseline_f1_total": 0.0,
            },
        )
        bucket["example_count"] = int(bucket["example_count"]) + 1
        bucket["hybrid_exact_total"] = float(bucket["hybrid_exact_total"]) + hybrid_exact_hit
        bucket["baseline_exact_total"] = float(bucket["baseline_exact_total"]) + baseline_exact_hit
        bucket["hybrid_f1_total"] = float(bucket["hybrid_f1_total"]) + hybrid_f1
        bucket["baseline_f1_total"] = float(bucket["baseline_f1_total"]) + baseline_f1
    total = max(1, len(examples))
    return UniversalDecoderEvalReport(
        hybrid_exact_match_rate=hybrid_exact / total,
        baseline_exact_match_rate=baseline_exact / total,
        hybrid_token_f1=hybrid_f1_total / total,
        baseline_token_f1=baseline_f1_total / total,
        example_count=len(examples),
        slices=_finalize_slices(slice_stats),
    )


def _generate_seed_completion(*, config: KernelConfig, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max(1, int(max_tokens)),
    }
    headers = {"Content-Type": "application/json"}
    if config.vllm_api_key.strip():
        headers["Authorization"] = f"Bearer {config.vllm_api_key.strip()}"
    req = request.Request(
        url=f"{config.vllm_host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=config.provider_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _token_f1(predicted: str, target: str) -> float:
    pred_tokens = _normalize_text(predicted).split()
    target_tokens = _normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), target_counts.get(token, 0)) for token in set(pred_counts) | set(target_counts))
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(target_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _finalize_slices(raw: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int | str]]:
    finalized: dict[str, dict[str, float | int | str]] = {}
    for source_type, bucket in sorted(raw.items()):
        count = max(1, int(bucket.get("example_count", 0) or 0))
        finalized[source_type] = {
            "source_type": source_type,
            "example_count": count,
            "hybrid_exact_match_rate": float(bucket.get("hybrid_exact_total", 0.0)) / count,
            "baseline_exact_match_rate": float(bucket.get("baseline_exact_total", 0.0)) / count,
            "hybrid_token_f1": float(bucket.get("hybrid_f1_total", 0.0)) / count,
            "baseline_token_f1": float(bucket.get("baseline_f1_total", 0.0)) / count,
        }
    return finalized
