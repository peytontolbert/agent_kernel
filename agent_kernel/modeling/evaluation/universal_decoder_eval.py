from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    total_dataset_examples: int = 0
    sample_coverage_rate: float = 0.0
    sampled_source_type_count: int = 0
    total_source_type_count: int = 0
    source_type_coverage_rate: float = 0.0
    disagreement_rate: float = 0.0
    hybrid_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    selection_strategy: str = "full_dataset"
    warnings: list[str] = field(default_factory=list)
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
    dataset_examples = _load_jsonl(eval_dataset_path)
    examples, selection_strategy = _select_eval_examples(dataset_examples, max_examples=max_examples)
    hybrid_exact = 0.0
    baseline_exact = 0.0
    hybrid_f1_total = 0.0
    baseline_f1_total = 0.0
    hybrid_wins = 0.0
    baseline_wins = 0.0
    disagreements = 0.0
    slice_stats: dict[str, dict[str, float | int]] = {}
    for example in examples:
        prompt = str(example.get("prompt", ""))
        target = str(example.get("target", ""))
        source_type = _source_type_of(example)
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
        if hybrid_exact_hit != baseline_exact_hit or abs(hybrid_f1 - baseline_f1) > 1e-9:
            disagreements += 1.0
        hybrid_score = (hybrid_exact_hit, hybrid_f1)
        baseline_score = (baseline_exact_hit, baseline_f1)
        if hybrid_score > baseline_score:
            hybrid_wins += 1.0
        elif baseline_score > hybrid_score:
            baseline_wins += 1.0
        _update_slice_bucket(slice_stats, source_type, hybrid_exact_hit, baseline_exact_hit, hybrid_f1, baseline_f1)
        _update_slice_bucket(
            slice_stats,
            f"target_length:{_target_length_bucket(target)}",
            hybrid_exact_hit,
            baseline_exact_hit,
            hybrid_f1,
            baseline_f1,
        )
    total = max(1, len(examples))
    total_dataset_examples = len(dataset_examples)
    sampled_source_types = {_source_type_of(example) for example in examples}
    total_source_types = {_source_type_of(example) for example in dataset_examples}
    total_source_type_count = len(total_source_types)
    sampled_source_type_count = len(sampled_source_types)
    sample_coverage_rate = len(examples) / max(1, total_dataset_examples)
    source_type_coverage_rate = sampled_source_type_count / max(1, total_source_type_count)
    return UniversalDecoderEvalReport(
        hybrid_exact_match_rate=hybrid_exact / total,
        baseline_exact_match_rate=baseline_exact / total,
        hybrid_token_f1=hybrid_f1_total / total,
        baseline_token_f1=baseline_f1_total / total,
        example_count=len(examples),
        slices=_finalize_slices(slice_stats),
        total_dataset_examples=total_dataset_examples,
        sample_coverage_rate=sample_coverage_rate,
        sampled_source_type_count=sampled_source_type_count,
        total_source_type_count=total_source_type_count,
        source_type_coverage_rate=source_type_coverage_rate,
        disagreement_rate=disagreements / total,
        hybrid_win_rate=hybrid_wins / total,
        baseline_win_rate=baseline_wins / total,
        selection_strategy=selection_strategy,
        warnings=_evaluation_warnings(
            example_count=len(examples),
            total_dataset_examples=total_dataset_examples,
            sampled_source_type_count=sampled_source_type_count,
            total_source_type_count=total_source_type_count,
            disagreement_rate=disagreements / total,
        ),
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
    with request.urlopen(req, timeout=config.llm_timeout_seconds) as response:
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
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    normalized = str(content or "").strip()
    if normalized:
        return normalized
    return str(message.get("reasoning", "") or "").strip()


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


def _source_type_of(example: dict[str, Any]) -> str:
    return str(example.get("source_type", "unknown")).strip() or "unknown"


def _target_length_bucket(target: str) -> str:
    token_count = len(_normalize_text(target).split())
    if token_count <= 2:
        return "short"
    if token_count <= 6:
        return "medium"
    return "long"


def _select_eval_examples(
    examples: list[dict[str, Any]],
    *,
    max_examples: int,
) -> tuple[list[dict[str, Any]], str]:
    limit = max(1, int(max_examples))
    if len(examples) <= limit:
        return list(examples), "full_dataset"
    buckets: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        buckets.setdefault(_source_type_of(example), []).append(index)
    selected_indices: list[int] = []
    bucket_names = sorted(buckets)
    quota_by_bucket = {name: 0 for name in bucket_names}
    if limit >= len(bucket_names):
        for name in bucket_names:
            quota_by_bucket[name] = 1
        remaining = limit - len(bucket_names)
    else:
        ranked = sorted(bucket_names, key=lambda name: (-len(buckets[name]), name))
        for name in ranked[:limit]:
            quota_by_bucket[name] = 1
        remaining = 0
    if remaining > 0:
        total = max(1, len(examples))
        exact_by_bucket = {
            name: remaining * (len(buckets[name]) / total)
            for name in bucket_names
        }
        floor_by_bucket = {
            name: min(len(buckets[name]) - quota_by_bucket[name], int(exact_by_bucket[name]))
            for name in bucket_names
        }
        for name, floor in floor_by_bucket.items():
            quota_by_bucket[name] += max(0, floor)
        remaining_after_floor = remaining - sum(max(0, floor) for floor in floor_by_bucket.values())
        ranked_remainders = sorted(
            bucket_names,
            key=lambda name: (-(exact_by_bucket[name] - int(exact_by_bucket[name])), -len(buckets[name]), name),
        )
        for name in ranked_remainders:
            if remaining_after_floor <= 0:
                break
            capacity = len(buckets[name]) - quota_by_bucket[name]
            if capacity <= 0:
                continue
            quota_by_bucket[name] += 1
            remaining_after_floor -= 1
    for name in bucket_names:
        selected_indices.extend(
            buckets[name][index]
            for index in _evenly_spaced_relative_indices(len(buckets[name]), quota_by_bucket[name])
        )
    selected_index_set = set(selected_indices)
    if len(selected_indices) < limit:
        for index in _evenly_spaced_relative_indices(len(examples), limit):
            if index in selected_index_set:
                continue
            selected_indices.append(index)
            selected_index_set.add(index)
            if len(selected_indices) >= limit:
                break
    selected_indices = sorted(selected_indices[:limit])
    return [examples[index] for index in selected_indices], "stratified_source_type_even_spread"


def _evenly_spaced_relative_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    used: set[int] = set()
    selected: list[int] = []
    for slot in range(count):
        candidate = min(length - 1, max(0, int(((slot + 0.5) * length) / count)))
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
            continue
        for delta in range(1, length):
            lower = candidate - delta
            upper = candidate + delta
            if lower >= 0 and lower not in used:
                selected.append(lower)
                used.add(lower)
                break
            if upper < length and upper not in used:
                selected.append(upper)
                used.add(upper)
                break
    return sorted(selected)


def _update_slice_bucket(
    raw: dict[str, dict[str, float | int]],
    key: str,
    hybrid_exact_hit: float,
    baseline_exact_hit: float,
    hybrid_f1: float,
    baseline_f1: float,
) -> None:
    bucket = raw.setdefault(
        key,
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


def _evaluation_warnings(
    *,
    example_count: int,
    total_dataset_examples: int,
    sampled_source_type_count: int,
    total_source_type_count: int,
    disagreement_rate: float,
) -> list[str]:
    warnings: list[str] = []
    if example_count < min(32, total_dataset_examples):
        warnings.append("small_sample")
    if example_count < total_dataset_examples:
        warnings.append("partial_dataset_coverage")
    if sampled_source_type_count < total_source_type_count:
        warnings.append("partial_source_type_coverage")
    if disagreement_rate >= 0.25:
        warnings.append("high_model_disagreement")
    return warnings


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
