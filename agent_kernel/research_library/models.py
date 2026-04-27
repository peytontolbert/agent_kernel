from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


DEFAULT_RESEARCH_LIBRARY_STATUS = Path("var/research_library/status.json")

_CODING_GROUP_HINTS = {
    "coder_qlora.humaneval",
    "coder_rl.humaneval",
    "coder_sft",
    "coder_sft.humaneval",
    "humaneval_qlora",
}
_VERIFIER_GROUP_HINTS = {"verifier_qlora", "verifier_calib"}
_RESEARCH_GROUP_HINTS = {"A2", "A3", "M1", "P1"}
_ACTION_GROUP_HINTS = {"actuator_il_t5", "actuator_il_t5_base", "actuator_dom_t5", "actuator_dom_t5_large"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_research_library_status(path: Path | str = DEFAULT_RESEARCH_LIBRARY_STATUS) -> dict[str, Any]:
    status_path = Path(path)
    if not status_path.is_absolute():
        status_path = _repo_root() / status_path
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {status_path}")
    return payload


def iter_trained_model_assets(
    status: dict[str, Any] | None = None,
    *,
    status_path: Path | str = DEFAULT_RESEARCH_LIBRARY_STATUS,
    source_id: str | None = None,
    asset_type: str | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    payload = status or load_research_library_status(status_path)
    assets: list[dict[str, Any]] = []
    for source in payload.get("sources", []):
        if not isinstance(source, dict):
            continue
        if source_id is not None and source.get("id") != source_id:
            continue
        source_assets = source.get("model_assets", [])
        if not isinstance(source_assets, list):
            continue
        for asset in source_assets:
            if not isinstance(asset, dict):
                continue
            if asset_type is not None and asset.get("asset_type") != asset_type:
                continue
            if group is not None and asset.get("group") != group:
                continue
            enriched = dict(asset)
            enriched["source_id"] = source.get("id", "")
            enriched["source_label"] = source.get("label", "")
            enriched["source_role"] = source.get("role", "")
            assets.append(enriched)
    return assets


def trained_model_asset_catalog(
    status: dict[str, Any] | None = None,
    *,
    status_path: Path | str = DEFAULT_RESEARCH_LIBRARY_STATUS,
) -> dict[str, Any]:
    assets = iter_trained_model_assets(status, status_path=status_path)
    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for asset in assets:
        source = str(asset.get("source_id", ""))
        asset_type = str(asset.get("asset_type", ""))
        group = str(asset.get("group", ""))
        by_source[source] = by_source.get(source, 0) + 1
        by_type[asset_type] = by_type.get(asset_type, 0) + 1
        by_group[group] = by_group.get(group, 0) + 1
    return {
        "summary": {
            "asset_count": len(assets),
            "by_source": dict(sorted(by_source.items())),
            "by_type": dict(sorted(by_type.items())),
            "by_group": dict(sorted(by_group.items())),
        },
        "assets": assets,
    }


def _text_tokens(*values: object) -> set[str]:
    text = " ".join(str(value).lower().replace("_", " ") for value in values if value is not None)
    token = ""
    tokens: set[str] = set()
    for char in text:
        if char.isalnum():
            token += char
        else:
            if len(token) >= 3:
                tokens.add(token)
            token = ""
    if len(token) >= 3:
        tokens.add(token)
    return tokens


def _asset_checkpoint_step(asset: dict[str, Any]) -> int:
    value = asset.get("checkpoint_step")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    match = re.search(r"epoch(\d+)", str(asset.get("group", "")))
    if match is not None:
        return int(match.group(1))
    return 0


def _model_asset_score(asset: dict[str, Any], *, prompt_tokens: set[str], metadata: dict[str, Any], role: str) -> int:
    group = str(asset.get("group", "")).strip()
    asset_type = str(asset.get("asset_type", "")).strip()
    source_id = str(asset.get("source_id", "")).strip()
    task_family = str(metadata.get("benchmark_family", "")).strip().lower()
    score = 0
    if asset_type == "retrieval_cache":
        score += 80
    elif asset_type == "tolbert_checkpoint":
        score += 76
    elif asset_type == "faiss_index":
        score += 72
    elif asset_type == "peft_adapter":
        score += 60
    elif asset_type == "hf_model":
        score += 45
    elif asset_type == "lightweight_model":
        score += 42

    if group in _RESEARCH_GROUP_HINTS:
        score += 18
    if group in _CODING_GROUP_HINTS:
        score += 16
    if group in _VERIFIER_GROUP_HINTS or "verify" in group:
        score += 14
    if group in _ACTION_GROUP_HINTS:
        score += 8
    if source_id == "tolbert_checkpoints":
        score += 12
    if source_id == "repository_models":
        score += 10
    if source_id == "digital_world_model_checkpoints":
        score += 8

    coding_tokens = {"code", "coding", "patch", "bug", "test", "swe", "repo", "repository", "python"}
    research_tokens = {"paper", "research", "model", "retrieval", "embedding", "tolbert", "algorithm"}
    algorithm_tokens = {"algorithm", "graph", "dynamic", "shortest", "tree", "dp", "codeforces"}
    verifier_tokens = {"verify", "test", "risk", "failure", "regression"}
    dom_tokens = {"browser", "dom", "selector", "web"}

    if prompt_tokens & coding_tokens or task_family in {"swe_bench", "swe_bench_lite", "swe_bench_verified"}:
        if group in _CODING_GROUP_HINTS or source_id in {"repository_models", "tolbert_checkpoints"}:
            score += 18
    if prompt_tokens & research_tokens:
        if group in _RESEARCH_GROUP_HINTS or source_id == "tolbert_checkpoints":
            score += 18
    if prompt_tokens & algorithm_tokens or task_family in {"codeforces", "codecontests"}:
        if group in _RESEARCH_GROUP_HINTS or asset_type in {"faiss_index", "tolbert_checkpoint", "retrieval_cache"}:
            score += 10
    if prompt_tokens & verifier_tokens or role in {"critic", "verifier"}:
        if group in _VERIFIER_GROUP_HINTS or "verifier" in group:
            score += 12
    if prompt_tokens & dom_tokens:
        if "dom" in group or "actuator" in group:
            score += 14
    if "planner" in group and role == "planner":
        score += 10
    if "memory" in group:
        score += 12
    checkpoint_step = _asset_checkpoint_step(asset)
    score += min(10, checkpoint_step if asset_type == "tolbert_checkpoint" else checkpoint_step // 500)
    return score


def select_trained_model_assets(
    status: dict[str, Any] | None = None,
    *,
    status_path: Path | str = DEFAULT_RESEARCH_LIBRARY_STATUS,
    prompt: str = "",
    metadata: dict[str, Any] | None = None,
    role: str = "executor",
    limit: int = 8,
) -> list[dict[str, Any]]:
    task_metadata = metadata if isinstance(metadata, dict) else {}
    prompt_tokens = _text_tokens(prompt, task_metadata.get("benchmark_family", ""), task_metadata.get("repo", ""))
    assets = iter_trained_model_assets(status, status_path=status_path)
    ranked = sorted(
        assets,
        key=lambda asset: (
            -_model_asset_score(asset, prompt_tokens=prompt_tokens, metadata=task_metadata, role=role),
            str(asset.get("source_id", "")),
            str(asset.get("group", "")),
            -_asset_checkpoint_step(asset),
            str(asset.get("relative_path", "")),
        ),
    )
    max_assets = max(0, int(limit))
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_groups: set[str] = set()

    def asset_key(asset: dict[str, Any]) -> str:
        return f"{asset.get('source_id', '')}:{asset.get('relative_path', asset.get('path', ''))}"

    def asset_group_key(asset: dict[str, Any]) -> str:
        source_id = str(asset.get("source_id", "")).strip()
        group = str(asset.get("group", "")).strip()
        if source_id == "tolbert_checkpoints" and group.startswith("tolbert_epoch"):
            return "tolbert_epoch"
        return f"{source_id}:{group}"

    for asset in ranked:
        group = asset_group_key(asset)
        key = asset_key(asset)
        if key in selected_keys or group in selected_groups:
            continue
        selected.append(asset)
        selected_keys.add(key)
        selected_groups.add(group)
        if len(selected) >= max_assets:
            return selected
    for asset in ranked:
        key = asset_key(asset)
        if key in selected_keys:
            continue
        selected.append(asset)
        selected_keys.add(key)
        if len(selected) >= max_assets:
            break
    return selected


__all__ = [
    "DEFAULT_RESEARCH_LIBRARY_STATUS",
    "iter_trained_model_assets",
    "load_research_library_status",
    "select_trained_model_assets",
    "trained_model_asset_catalog",
]
