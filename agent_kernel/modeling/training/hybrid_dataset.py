from __future__ import annotations

from pathlib import Path
import json
import math
import os
import shutil
from typing import Any

from agent_kernel.config import KernelConfig

from ..tolbert.config import HybridTolbertSSMConfig
from ..tolbert.tokenization import (
    build_decoder_vocabulary,
    encode_command_tokens,
    encode_decoder_sequence,
    hashed_id,
)


def _should_write_dataset_shards() -> bool:
    return os.getenv("AGENT_KERNEL_STORAGE_WRITE_DATASET_SHARDS", "0") == "1"


def _sync_jsonl_shards(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> list[Path]:
    if _should_write_dataset_shards():
        return _write_jsonl_shards(output_dir, stem, rows)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    return []


def build_hybrid_training_examples(
    *,
    trajectories_root: Path,
    config: HybridTolbertSSMConfig,
    decoder_vocab: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for payload in _trajectory_payloads(trajectories_root):
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            continue
        task_metadata = payload.get("task_metadata", {})
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        task_difficulty = _task_difficulty(task_metadata)
        is_long_horizon = task_difficulty == "long_horizon"
        family_id = hashed_id(str(task_metadata.get("benchmark_family", "bounded")), config.family_vocab_size)
        path_level_ids = [
            hashed_id(str(value), config.path_vocab_size)
            for value in [
                str(task_metadata.get("benchmark_family", "bounded")),
                str(task_metadata.get("capability", "generic")),
                str(task_metadata.get("difficulty", "seed")),
                str(payload.get("task_id", "")),
            ][: config.max_path_levels]
        ]
        while len(path_level_ids) < config.max_path_levels:
            path_level_ids.append(0)
        success = bool(payload.get("success", False))
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            world_feedback = _world_feedback_summary(step)
            window = steps[max(0, index - config.sequence_length + 1) : index + 1]
            token_rows = []
            scalar_rows = []
            for frame in window:
                if not isinstance(frame, dict):
                    continue
                token_rows.append(encode_command_tokens(str(frame.get("content", "")), config))
                scalar_rows.append(_step_scalar_features(frame, config))
            while len(token_rows) < config.sequence_length:
                token_rows.insert(0, [0] * config.max_command_tokens)
                scalar_rows.insert(0, [0.0] * config.scalar_feature_dim)
            verification = step.get("verification", {})
            if not isinstance(verification, dict):
                verification = {}
            transition = step.get("state_transition", {})
            if not isinstance(transition, dict):
                transition = {}
            future_steps = [item for item in steps[index:] if isinstance(item, dict)]
            future_progress = sum(float(item.get("state_progress_delta", 0.0) or 0.0) for item in future_steps)
            future_regressions = sum(_regression_count(item) for item in future_steps)
            future_passes = sum(
                1.0
                for item in future_steps
                if bool((item.get("verification", {}) if isinstance(item.get("verification", {}), dict) else {}).get("passed", False))
            )
            remaining = max(1, len(future_steps))
            remaining_pass_rate = future_passes / remaining
            example_weight = _example_weight(
                is_long_horizon=is_long_horizon,
                remaining_steps=remaining,
                future_progress=future_progress,
                remaining_pass_rate=remaining_pass_rate,
                trusted_retrieval_alignment=world_feedback["trusted_retrieval_alignment"],
                graph_environment_alignment=world_feedback["graph_environment_alignment"],
                transfer_novelty=world_feedback["transfer_novelty"],
            )
            safe_transfer_bonus = max(0.0, world_feedback["graph_environment_alignment"])
            unsafe_transfer_penalty = max(0.0, -world_feedback["graph_environment_alignment"])
            local_failure = (
                1.0 if list(step.get("failure_signals", [])) else 0.0
            ) + (1.0 if not bool(verification.get("passed", False)) else 0.0)
            policy_target = _clamp01(
                (0.65 if bool(verification.get("passed", False)) else 0.0)
                + (0.20 * remaining_pass_rate)
                + (0.15 * _sigmoid_unit(float(step.get("state_progress_delta", 0.0) or 0.0)))
                + (0.10 * world_feedback["trusted_retrieval_alignment"])
                + (0.05 * safe_transfer_bonus)
                - (0.10 * unsafe_transfer_penalty)
            )
            risk_target = _clamp01(
                0.45 * min(1.0, local_failure)
                + 0.35 * min(1.0, future_regressions / max(1, remaining))
                + 0.20 * (1.0 if bool(step.get("no_progress", False)) else 0.0)
                + (0.20 * unsafe_transfer_penalty)
                + (0.10 * world_feedback["transfer_novelty"] * max(0.0, 1.0 - safe_transfer_bonus))
            )
            value_target = _clamp01(
                (0.40 if success else 0.0)
                + (0.35 * remaining_pass_rate)
                + (0.20 * _sigmoid_unit(future_progress))
                - (0.20 * min(1.0, future_regressions / max(1, remaining)))
                + (0.05 * world_feedback["progress_signal"])
                - (0.05 * world_feedback["risk_signal"])
                + (0.05 * world_feedback["trusted_retrieval_alignment"])
                + (0.05 * safe_transfer_bonus)
                - (0.05 * unsafe_transfer_penalty)
            )
            stop_target = 1.0 if index == len(steps) - 1 and success and bool(verification.get("passed", False)) else 0.0
            score_target = _clamp01(
                (0.35 * policy_target)
                + (0.30 * value_target)
                + (0.20 * (1.0 - risk_target))
                + (0.10 * _sigmoid_unit(future_progress))
                + (0.05 * stop_target)
                + (0.05 * _sigmoid_unit(world_feedback["hybrid_total_score"]))
                + (0.05 * world_feedback["trusted_retrieval_alignment"])
                + (0.05 * safe_transfer_bonus)
                - (0.10 * unsafe_transfer_penalty)
            )
            world_target = _world_target_distribution(
                step=step,
                success=success,
                future_progress=future_progress,
                future_regressions=future_regressions,
                remaining_pass_rate=remaining_pass_rate,
                config=config,
            )
            decoder_input_ids, decoder_target_ids = encode_decoder_sequence(
                str(step.get("content", "")),
                config,
                decoder_vocab=decoder_vocab,
            )
            examples.append(
                {
                    "family_id": family_id,
                    "path_level_ids": path_level_ids,
                    "command_token_ids": token_rows[-config.sequence_length :],
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_target_ids": decoder_target_ids,
                    "scalar_features": scalar_rows[-config.sequence_length :],
                    "score_target": score_target,
                    "policy_target": policy_target,
                    "value_target": value_target,
                    "stop_target": stop_target,
                    "risk_target": risk_target,
                    "transition_target": [
                        float(future_progress),
                        float(future_regressions),
                    ],
                    "task_difficulty": task_difficulty,
                    "example_weight": example_weight,
                    "world_target": world_target,
                    "world_feedback": world_feedback,
                }
            )
    return examples


def materialize_hybrid_training_dataset(
    *,
    trajectories_root: Path,
    output_path: Path,
    config: HybridTolbertSSMConfig,
) -> dict[str, Any]:
    texts = _trajectory_texts(trajectories_root)
    decoder_vocab = build_decoder_vocabulary(texts, config)
    decoder_vocab_path = output_path.with_suffix(".decoder_vocab.json")
    decoder_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_vocab_path.write_text(json.dumps(decoder_vocab, indent=2, sort_keys=True), encoding="utf-8")
    examples = build_hybrid_training_examples(
        trajectories_root=trajectories_root,
        config=config,
        decoder_vocab=decoder_vocab,
    )
    difficulty_counts: dict[str, int] = {}
    long_horizon_example_count = 0
    weighted_example_total = 0.0
    long_horizon_weighted_total = 0.0
    trusted_retrieval_aligned_example_count = 0
    transfer_novelty_example_count = 0
    environment_safe_example_count = 0
    for example in examples:
        difficulty = str(example.get("task_difficulty", "")).strip() or "seed"
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        weight = float(example.get("example_weight", 1.0) or 1.0)
        world_feedback = dict(example.get("world_feedback", {})) if isinstance(example.get("world_feedback", {}), dict) else {}
        weighted_example_total += weight
        if difficulty == "long_horizon":
            long_horizon_example_count += 1
            long_horizon_weighted_total += weight
        if float(world_feedback.get("trusted_retrieval_alignment", 0.0) or 0.0) > 0.0:
            trusted_retrieval_aligned_example_count += 1
        if float(world_feedback.get("transfer_novelty", 0.0) or 0.0) > 0.0:
            transfer_novelty_example_count += 1
        if float(world_feedback.get("graph_environment_alignment", 0.0) or 0.0) > 0.0:
            environment_safe_example_count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(example) + "\n" for example in examples), encoding="utf-8")
    shard_paths = _sync_jsonl_shards(output_path.parent / f"{output_path.stem}_shards", output_path.stem, examples)
    manifest = {
        "artifact_kind": "tolbert_hybrid_training_dataset",
        "example_count": len(examples),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "long_horizon_example_count": long_horizon_example_count,
        "average_example_weight": round(weighted_example_total / max(1, len(examples)), 4),
        "long_horizon_weighted_example_share": round(long_horizon_weighted_total / max(1.0, weighted_example_total), 4),
        "trusted_retrieval_aligned_example_count": trusted_retrieval_aligned_example_count,
        "transfer_novelty_example_count": transfer_novelty_example_count,
        "environment_safe_example_count": environment_safe_example_count,
        "dataset_path": str(output_path),
        "shard_paths": [str(path) for path in shard_paths],
        "decoder_vocab_path": str(decoder_vocab_path),
        "decoder_vocab_entry_count": len(decoder_vocab),
        "config": config.to_dict(),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and trajectories_root.resolve() == default_config.trajectories_root.resolve():
            default_config.sqlite_store().record_export_manifest(
                export_key=f"dataset:hybrid:{output_path.resolve()}",
                export_kind="hybrid_training_dataset",
                payload=manifest,
            )
    except OSError:
        pass
    return manifest


def _step_scalar_features(step: dict[str, Any], config: HybridTolbertSSMConfig) -> list[float]:
    verification = step.get("verification", {})
    if not isinstance(verification, dict):
        verification = {}
    world_feedback = _world_feedback_summary(step)
    transition = step.get("state_transition", {})
    if not isinstance(transition, dict):
        transition = {}
    no_progress = bool(step.get("no_progress", transition.get("no_progress", False)))
    features = [
        float(step.get("path_confidence", 0.0) or 0.0),
        1.0 if bool(step.get("trust_retrieval", False)) else 0.0,
        float(step.get("state_progress_delta", 0.0) or 0.0),
        float(step.get("state_regression_count", 0.0) or 0.0),
        1.0 if bool(step.get("retrieval_influenced", False)) else 0.0,
        1.0 if bool(step.get("retrieval_ranked_skill", False)) else 0.0,
        world_feedback["trusted_retrieval_alignment"],
        max(-1.0, min(1.0, world_feedback["graph_environment_alignment"])),
        world_feedback["transfer_novelty"],
        1.0 if str(step.get("action", "")) == "code_execute" else 0.0,
        1.0 if str(step.get("action", "")) == "respond" else 0.0,
        1.0 if bool(verification.get("passed", False)) else 0.0,
        float(step.get("latent_command_bias", 0.0) or 0.0),
        1.0 if no_progress else 0.0,
        world_feedback["progress_signal"],
        world_feedback["risk_signal"],
    ]
    return (features + [0.0] * config.scalar_feature_dim)[: config.scalar_feature_dim]


def _regression_count(step: dict[str, Any]) -> float:
    transition = step.get("state_transition", {})
    if not isinstance(transition, dict):
        transition = {}
    try:
        return float(step.get("state_regression_count", transition.get("regression_count", 0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _task_difficulty(task_metadata: dict[str, Any]) -> str:
    return str(task_metadata.get("difficulty", task_metadata.get("task_difficulty", "seed"))).strip() or "seed"


def _example_weight(
    *,
    is_long_horizon: bool,
    remaining_steps: int,
    future_progress: float,
    remaining_pass_rate: float,
    trusted_retrieval_alignment: float,
    graph_environment_alignment: float,
    transfer_novelty: float,
) -> float:
    if not is_long_horizon:
        return 1.0
    bonus = min(
        2.0,
        (0.3 * math.log1p(max(1, remaining_steps)))
        + (0.35 * min(1.0, max(0.0, future_progress)))
        + (0.25 * max(0.0, min(1.0, remaining_pass_rate))),
    )
    bonus += 0.20 * max(0.0, trusted_retrieval_alignment)
    bonus += 0.15 * max(0.0, graph_environment_alignment)
    if transfer_novelty > 0.0:
        bonus += 0.15 * (0.5 + max(0.0, graph_environment_alignment))
    return round(1.0 + bonus, 4)


def _world_target_distribution(
    *,
    step: dict[str, Any],
    success: bool,
    future_progress: float,
    future_regressions: float,
    remaining_pass_rate: float,
    config: HybridTolbertSSMConfig,
) -> list[float]:
    latent = step.get("latent_state_summary", {})
    if not isinstance(latent, dict):
        latent = {}
    world_feedback = _world_feedback_summary(step)
    progress_band = str(latent.get("progress_band", "flat")).strip()
    risk_band = str(latent.get("risk_band", "stable")).strip()
    trusted = bool(step.get("trust_retrieval", False))
    transition = step.get("state_transition", {})
    if not isinstance(transition, dict):
        transition = {}
    no_progress = bool(step.get("no_progress", transition.get("no_progress", False)))
    scores = [0.05] * max(1, config.world_state_dim)
    scores[0] += max(0.0, _sigmoid_unit(future_progress) if future_progress > 0 else 0.0)
    if progress_band in {"advancing", "improving"}:
        scores[0] += 0.5
    scores[0] += 0.35 * world_feedback["progress_signal"]
    if len(scores) > 1:
        scores[1] += min(1.0, future_regressions / 3.0)
        if risk_band == "regressive":
            scores[1] += 0.5
        scores[1] += 0.25 * world_feedback["risk_signal"]
    if len(scores) > 2:
        scores[2] += 1.0 if no_progress or risk_band == "stalled" else 0.0
        scores[2] += 0.25 * max(0.0, world_feedback["transition_regression_score"] - world_feedback["transition_progress_score"])
    if len(scores) > 3:
        scores[3] += 1.0 if risk_band == "blocked" else 0.0
    if len(scores) > 4:
        scores[4] += (0.5 if success else 0.0) + (0.5 * remaining_pass_rate)
        if str(step.get("action", "")).strip() == "respond":
            scores[4] += 0.15 * world_feedback["decoder_world_progress_score"]
    if len(scores) > 5:
        scores[5] += 1.0 if trusted else 0.0
        scores[5] += 0.50 * world_feedback["trusted_retrieval_alignment"]
    if len(scores) > 6:
        scores[6] += 1.0 if (future_regressions > 0 or no_progress or risk_band in {"blocked", "regressive"}) else 0.0
        scores[6] += 0.20 * world_feedback["decoder_world_risk_score"]
        scores[6] += 0.35 * max(0.0, -world_feedback["graph_environment_alignment"])
        scores[6] += 0.15 * world_feedback["transfer_novelty"]
    if len(scores) > 7:
        scores[7] += 1.0 if risk_band == "stable" and progress_band not in {"advancing", "improving"} else 0.0
        scores[7] += 0.10 * max(0.0, 1.0 - world_feedback["decoder_world_entropy_mean"])
        scores[7] += 0.25 * max(0.0, world_feedback["graph_environment_alignment"])
    total = sum(max(0.0, value) for value in scores)
    if total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [max(0.0, value) / total for value in scores]


def _world_feedback_summary(step: dict[str, Any]) -> dict[str, Any]:
    proposal = step.get("proposal_metadata", {})
    if not isinstance(proposal, dict):
        proposal = {}
    latent = step.get("latent_state_summary", {})
    if not isinstance(latent, dict):
        latent = {}
    learned = latent.get("learned_world_state", {})
    if not isinstance(learned, dict):
        learned = {}
    progress_signal = max(
        _float_value(learned.get("progress_signal"), 0.0),
        _float_value(learned.get("world_progress_score"), 0.0),
        _float_value(learned.get("decoder_world_progress_score"), 0.0),
        _float_value(proposal.get("hybrid_world_progress_score"), 0.0),
        _float_value(proposal.get("hybrid_decoder_world_progress_score"), 0.0),
    )
    risk_signal = max(
        _float_value(learned.get("risk_signal"), 0.0),
        _float_value(learned.get("world_risk_score"), 0.0),
        _float_value(learned.get("decoder_world_risk_score"), 0.0),
        _float_value(proposal.get("hybrid_world_risk_score"), 0.0),
        _float_value(proposal.get("hybrid_decoder_world_risk_score"), 0.0),
    )
    decoder_world_entropy_mean = _float_value(
        proposal.get(
            "hybrid_decoder_world_entropy_mean",
            learned.get("decoder_world_entropy_mean", 0.0),
        ),
        0.0,
    )
    trusted_retrieval_alignment = max(
        _float_value(learned.get("trusted_retrieval_alignment"), 0.0),
        _float_value(proposal.get("hybrid_trusted_retrieval_alignment"), 0.0),
    )
    graph_environment_alignment = _float_value(
        proposal.get(
            "hybrid_graph_environment_alignment",
            learned.get("graph_environment_alignment", 0.0),
        ),
        0.0,
    )
    transfer_novelty = max(
        _float_value(learned.get("transfer_novelty"), 0.0),
        _float_value(proposal.get("hybrid_transfer_novelty"), 0.0),
    )
    return {
        "progress_signal": progress_signal,
        "risk_signal": risk_signal,
        "world_progress_score": max(
            _float_value(learned.get("world_progress_score"), 0.0),
            _float_value(proposal.get("hybrid_world_progress_score"), 0.0),
        ),
        "world_risk_score": max(
            _float_value(learned.get("world_risk_score"), 0.0),
            _float_value(proposal.get("hybrid_world_risk_score"), 0.0),
        ),
        "decoder_world_progress_score": max(
            _float_value(learned.get("decoder_world_progress_score"), 0.0),
            _float_value(proposal.get("hybrid_decoder_world_progress_score"), 0.0),
        ),
        "decoder_world_risk_score": max(
            _float_value(learned.get("decoder_world_risk_score"), 0.0),
            _float_value(proposal.get("hybrid_decoder_world_risk_score"), 0.0),
        ),
        "decoder_world_entropy_mean": decoder_world_entropy_mean,
        "transition_progress_score": max(
            _float_value(learned.get("transition_progress_score"), 0.0),
            _float_value(proposal.get("hybrid_transition_progress"), 0.0),
        ),
        "transition_regression_score": max(
            _float_value(learned.get("transition_regression_score"), 0.0),
            _float_value(proposal.get("hybrid_transition_regression"), 0.0),
        ),
        "hybrid_total_score": _float_value(proposal.get("hybrid_total_score"), 0.0),
        "trusted_retrieval_alignment": max(0.0, min(1.0, trusted_retrieval_alignment)),
        "graph_environment_alignment": max(-1.0, min(1.0, graph_environment_alignment)),
        "transfer_novelty": max(0.0, min(1.0, transfer_novelty)),
        "source": str(learned.get("source", "")).strip(),
        "model_family": str(
            proposal.get("hybrid_model_family", learned.get("model_family", ""))
        ).strip(),
        "world_transition_family": str(
            proposal.get("hybrid_world_transition_family", learned.get("world_transition_family", ""))
        ).strip(),
        "top_probe_reason": str(learned.get("top_probe_reason", "")).strip(),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid_unit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trajectory_texts(trajectories_root: Path) -> list[str]:
    texts: list[str] = []
    for payload in _trajectory_payloads(trajectories_root):
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            texts.append(str(step.get("content", "")))
    return texts


def _trajectory_payloads(trajectories_root: Path) -> list[dict[str, Any]]:
    try:
        default_config = KernelConfig()
        if default_config.uses_sqlite_storage() and trajectories_root.resolve() == default_config.trajectories_root.resolve():
            return default_config.sqlite_store().iter_trajectory_payloads()
    except OSError:
        pass
    payloads: list[dict[str, Any]] = []
    for path in sorted(trajectories_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _write_jsonl_shards(output_dir: Path, stem: str, rows: list[dict[str, Any]], *, shard_size: int = 5000) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        shard_path = output_dir / f"{stem}_0000.jsonl"
        shard_path.write_text("", encoding="utf-8")
        return [shard_path]
    shard_paths: list[Path] = []
    normalized_size = max(1, int(shard_size))
    for shard_index, start in enumerate(range(0, len(rows), normalized_size)):
        shard_path = output_dir / f"{stem}_{shard_index:04d}.jsonl"
        shard_rows = rows[start : start + normalized_size]
        shard_path.write_text("".join(json.dumps(row) + "\n" for row in shard_rows), encoding="utf-8")
        shard_paths.append(shard_path)
    return shard_paths
