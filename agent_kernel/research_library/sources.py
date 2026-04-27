from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_RESEARCH_LIBRARY_CONFIG = Path("config/research_library_sources.json")

_MODEL_MARKER_FILES = {
    "adapter_model.safetensors": "peft_adapter",
    "model.safetensors": "hf_model",
    "pytorch_model.bin": "hf_model",
    "model.safetensors.index.json": "hf_model",
    "index.faiss": "faiss_index",
    "model.json": "lightweight_model",
}
_MODEL_METADATA_FILES = (
    "metadata.json",
    "adapter_config.json",
    "config.json",
    "metrics.json",
    "model.json",
    "meta.json",
    "trainer_state.json",
)
_MODEL_METADATA_KEYS = {
    "metadata.json": {
        "trained_at",
        "base_model",
        "mode",
        "epochs",
        "batch_size",
        "max_inp",
        "max_out",
        "val_ratio",
        "domain_feats",
        "ewc_lambda",
    },
    "adapter_config.json": {
        "base_model_name_or_path",
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
    },
    "config.json": {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "d_model",
        "num_hidden_layers",
        "num_attention_heads",
        "torch_dtype",
    },
    "metrics.json": None,
    "model.json": None,
    "meta.json": None,
    "trainer_state.json": {
        "best_metric",
        "best_model_checkpoint",
        "epoch",
        "global_step",
        "max_steps",
        "total_flos",
        "train_batch_size",
    },
}
_NON_MODEL_TORCH_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _json_object_or_empty(path: Path) -> dict[str, Any]:
    try:
        return _read_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _iter_matching_files(base: Path, pattern: str | None) -> Iterable[Path]:
    if not base.exists():
        return []
    if pattern:
        return (item for item in base.glob(pattern) if item.is_file())
    return (item for item in base.iterdir() if item.is_file())


def _file_stats(base: Path, pattern: str | None) -> dict[str, int]:
    count = 0
    size_bytes = 0
    for item in _iter_matching_files(base, pattern):
        count += 1
        try:
            size_bytes += item.stat().st_size
        except OSError:
            pass
    return {"file_count": count, "size_bytes": size_bytes}


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    size_bytes = 0
    try:
        items = path.rglob("*")
    except OSError:
        return 0
    for item in items:
        if not item.is_file():
            continue
        try:
            size_bytes += item.stat().st_size
        except OSError:
            pass
    return size_bytes


def _filtered_metadata(path: Path) -> dict[str, Any]:
    payload = _json_object_or_empty(path)
    if not payload:
        return {}
    allowed = _MODEL_METADATA_KEYS.get(path.name)
    if allowed is None:
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return {key: payload[key] for key in sorted(allowed) if key in payload}


def _checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith("checkpoint-"):
        return None
    return _as_int(name.removeprefix("checkpoint-"))


def _model_asset_type(files: set[str], *, path: Path | None = None) -> str:
    if path is not None and path.suffix == ".pt":
        if "cache" in str(path).lower():
            return "retrieval_cache"
        if path.name.startswith("tolbert_"):
            return "tolbert_checkpoint"
        return "torch_checkpoint"
    if "index.faiss" in files:
        return "faiss_index"
    if "adapter_model.safetensors" in files:
        return "peft_adapter"
    if (
        "model.safetensors" in files
        or "pytorch_model.bin" in files
        or "model.safetensors.index.json" in files
        or any(file_name.startswith("model-") and file_name.endswith(".safetensors") for file_name in files)
    ):
        return "hf_model"
    if "model.json" in files:
        return "lightweight_model"
    return "model_asset"


def _model_group(path: Path, base: Path) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError:
        return path.parent.name or path.name
    if not relative.parts:
        return base.name
    return relative.parts[0]


def _model_asset_record(path: Path, base: Path, *, root: Path) -> dict[str, Any]:
    if path.is_file():
        files = {path.name}
        metadata: dict[str, Any] = {}
        asset_type = _model_asset_type(files, path=path)
    else:
        files = {item.name for item in path.iterdir() if item.is_file()}
        metadata = {}
        for file_name in _MODEL_METADATA_FILES:
            candidate = path / file_name
            if candidate.exists():
                metadata[file_name] = _filtered_metadata(candidate)
        asset_type = _model_asset_type(files)
    return {
        "id": _safe_relative(path, base).replace("/", ":"),
        "path": str(path),
        "relative_path": _safe_relative(path, root),
        "group": _model_group(path, base),
        "asset_type": asset_type,
        "checkpoint_step": _checkpoint_step(path),
        "size_bytes": _path_size(path),
        "files": sorted(files),
        "metadata": metadata,
    }


def _discover_model_assets(base: Path, *, root: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    if not base.exists():
        return assets
    try:
        directories = [base, *[item for item in base.rglob("*") if item.is_dir()]]
    except OSError:
        directories = [base]
    for directory in directories:
        if ".no_exist" in directory.parts:
            continue
        try:
            files = {item.name for item in directory.iterdir() if item.is_file()}
        except OSError:
            continue
        has_marker = any(marker in files for marker in _MODEL_MARKER_FILES)
        has_shard = any(file_name.startswith("model-") and file_name.endswith(".safetensors") for file_name in files)
        if has_marker or has_shard:
            assets.append(_model_asset_record(directory, base, root=root))
    for checkpoint in sorted(base.rglob("*.pt")):
        if ".no_exist" in checkpoint.parts or checkpoint.name in _NON_MODEL_TORCH_STATE_FILES:
            continue
        assets.append(_model_asset_record(checkpoint, base, root=root))
    assets.sort(key=lambda item: (str(item.get("group", "")), str(item.get("relative_path", ""))))
    return assets


def load_research_library_config(path: Path | str = DEFAULT_RESEARCH_LIBRARY_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _repo_root() / config_path
    return _read_json_object(config_path)


def resolve_source_path(raw_path: str, *, root: Path | None = None) -> Path:
    root = root or _repo_root()
    path = Path(raw_path)
    if path.is_absolute():
        if root == _repo_root():
            return path
        try:
            rooted = root / path.relative_to("/")
        except ValueError:
            return path
        return rooted if rooted.exists() or not path.exists() else path
    return root / path


def _source_json(source: dict[str, Any], base: Path, field: str) -> dict[str, Any]:
    relative = str(source.get(field, "")).strip()
    if not relative:
        return {}
    return _json_object_or_empty(base / relative)


def _base_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    return {
        "id": str(source.get("id", "")).strip(),
        "label": str(source.get("label", "")).strip(),
        "kind": str(source.get("kind", "")).strip(),
        "role": str(source.get("role", "")).strip(),
        "path": str(path),
        "relative_path": _safe_relative(path, root),
        "exists": path.exists(),
        "status": "available" if path.exists() else "missing",
        "rows": None,
        "file_count": 0,
        "size_bytes": 0,
        "metrics": {},
        "missing_files": [],
        "notes": str(source.get("notes", "")).strip(),
    }


def _apply_row_metric(
    status: dict[str, Any],
    source: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    metric = str(source.get("row_metric", "")).strip()
    if not metric:
        return
    rows = _as_int(_nested_get(payload, metric))
    if rows is not None:
        status["rows"] = rows


def _apply_expected_rows(status: dict[str, Any], source: dict[str, Any]) -> None:
    expected = _as_int(source.get("expected_rows"))
    rows = _as_int(status.get("rows"))
    if expected is None or rows is None:
        return
    status["metrics"]["expected_rows"] = expected
    status["metrics"]["row_coverage"] = rows / expected if expected else None
    if rows < expected and status["status"] == "available":
        status["status"] = "partial"


def _parquet_dataset_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    stats = _source_json(source, path, "stats_path")
    status["metrics"].update(stats)
    _apply_row_metric(status, source, stats)
    files = _file_stats(path, str(source.get("file_glob", "")).strip() or None)
    status.update(files)
    if status["file_count"] == 0:
        status["status"] = "partial"
    _apply_expected_rows(status, source)
    return status


def _paper_graph_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    manifest = _source_json(source, path, "manifest_path")
    status["metrics"].update(manifest)
    _apply_row_metric(status, source, manifest)
    files = _file_stats(path, str(source.get("file_glob", "")).strip() or None)
    status.update(files)
    if not manifest:
        status["status"] = "partial"
    _apply_expected_rows(status, source)
    return status


def _repository_exports_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    manifest = _source_json(source, path, "manifest_path")
    repositories = manifest.get("repos", manifest.get("repositories", {}))
    if not isinstance(repositories, dict):
        repositories = {}
    repo_metrics: dict[str, Any] = {
        "repository_count": len(repositories),
        "manifest_schema": manifest.get("export_schema_version", manifest.get("schema_version")),
        "manifest_version": manifest.get("manifest_version"),
        "generated_at": manifest.get("generated_at"),
    }
    qa_index_count = 0
    mined_skill_count = 0
    languages: set[str] = set()
    for repo in repositories.values():
        if not isinstance(repo, dict):
            continue
        language = str(repo.get("language", "")).strip()
        if language:
            languages.add(language)
        raw_languages = repo.get("languages", [])
        if isinstance(raw_languages, list):
            languages.update(str(item).strip() for item in raw_languages if str(item).strip())
        indices = repo.get("indices", {})
        if isinstance(indices, dict) and indices.get("qa"):
            qa_index_count += 1
        artifacts = repo.get("artifacts", {})
        if isinstance(artifacts, dict) and artifacts.get("qa_index"):
            qa_index_count += 1
        mined = repo.get("mined_skills")
        mined_skill_count += _as_int(mined) or 0
        extensions = repo.get("extensions", {})
        if isinstance(extensions, dict):
            miner = extensions.get("repo_skills_miner", {})
            if isinstance(miner, dict):
                counts = miner.get("counts", {})
                if isinstance(counts, dict):
                    mined_skill_count += _as_int(counts.get("skills")) or 0
    repo_metrics["qa_index_count"] = qa_index_count
    repo_metrics["mined_skill_count"] = mined_skill_count
    repo_metrics["language_count"] = len(languages)
    repo_metrics["languages"] = sorted(languages)
    status["metrics"].update(repo_metrics)
    status["rows"] = len(repositories)
    status["file_count"] = len([item for item in path.iterdir() if item.is_dir()])
    if not manifest:
        status["status"] = "partial"
    return status


def _tolbert_projection_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    required_files = [str(item) for item in source.get("required_files", []) if str(item).strip()]
    missing_files: list[str] = []
    file_sizes: dict[str, int] = {}
    for relative in required_files:
        candidate = path / relative
        if candidate.exists():
            try:
                file_sizes[relative] = candidate.stat().st_size
            except OSError:
                file_sizes[relative] = 0
        else:
            missing_files.append(relative)
    level_sizes = _json_object_or_empty(path / "level_sizes_train_joint_v2.json")
    status["metrics"].update(
        {
            "required_file_sizes": file_sizes,
            "train_level_sizes": level_sizes,
        }
    )
    status["missing_files"] = missing_files
    status["file_count"] = len(file_sizes)
    status["size_bytes"] = sum(file_sizes.values())
    if missing_files:
        status["status"] = "partial"
    return status


def _jsonl_catalog_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    missing_files: list[str] = []
    jsonl_counts = source.get("jsonl_counts", {})
    if isinstance(jsonl_counts, dict):
        for key, relative in jsonl_counts.items():
            candidate = path / str(relative)
            if candidate.exists():
                counts[str(key)] = _count_lines(candidate)
                try:
                    sizes[str(key)] = candidate.stat().st_size
                except OSError:
                    sizes[str(key)] = 0
            else:
                missing_files.append(str(relative))
    status["metrics"].update({"line_counts": counts, "file_sizes": sizes})
    status["rows"] = sum(counts.values()) if counts else None
    status["file_count"] = len(counts)
    status["size_bytes"] = sum(sizes.values())
    status["missing_files"] = missing_files
    if missing_files:
        status["status"] = "partial"
    return status


def _tree_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _base_status(source, path, root)
    if not path.exists():
        return status
    files = _file_stats(path, str(source.get("file_glob", "")).strip() or None)
    status.update(files)
    status["rows"] = files["file_count"]
    return status


def _checkpoint_tree_status(source: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    status = _tree_status(source, path, root)
    if not path.exists():
        return status
    assets = _discover_model_assets(path, root=root)
    asset_type_counts: dict[str, int] = {}
    groups: dict[str, int] = {}
    for asset in assets:
        asset_type = str(asset.get("asset_type", "model_asset"))
        group = str(asset.get("group", ""))
        asset_type_counts[asset_type] = asset_type_counts.get(asset_type, 0) + 1
        groups[group] = groups.get(group, 0) + 1
    status["rows"] = len(assets)
    status["metrics"].update(
        {
            "model_asset_count": len(assets),
            "asset_type_counts": dict(sorted(asset_type_counts.items())),
            "groups": dict(sorted(groups.items())),
            "peft_adapter_count": asset_type_counts.get("peft_adapter", 0),
            "hf_model_count": asset_type_counts.get("hf_model", 0),
            "faiss_index_count": asset_type_counts.get("faiss_index", 0),
            "tolbert_checkpoint_count": asset_type_counts.get("tolbert_checkpoint", 0),
            "retrieval_cache_count": asset_type_counts.get("retrieval_cache", 0),
        }
    )
    status["model_assets"] = assets
    if not assets and status["status"] == "available":
        status["status"] = "partial"
    return status


def build_source_status(source: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    root = root or _repo_root()
    path = resolve_source_path(str(source.get("path", "")).strip(), root=root)
    kind = str(source.get("kind", "")).strip()
    if kind in {"paper_parquet_dataset", "paper_chunk_parquet_dataset"}:
        return _parquet_dataset_status(source, path, root)
    if kind == "paper_graph":
        return _paper_graph_status(source, path, root)
    if kind == "repo_graph_exports":
        return _repository_exports_status(source, path, root)
    if kind == "tolbert_projection":
        return _tolbert_projection_status(source, path, root)
    if kind == "jsonl_catalog":
        return _jsonl_catalog_status(source, path, root)
    if kind == "checkpoint_tree":
        return _checkpoint_tree_status(source, path, root)
    if kind == "source_tree":
        return _tree_status(source, path, root)
    return _base_status(source, path, root)


def _summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {source["id"]: source for source in sources}

    def rows(source_id: str) -> int | None:
        value = by_id.get(source_id, {}).get("rows")
        return _as_int(value)

    def metric(source_id: str, key: str) -> Any:
        metrics = by_id.get(source_id, {}).get("metrics", {})
        if not isinstance(metrics, dict):
            return None
        return _nested_get(metrics, key)

    def metric_int(source_id: str, key: str) -> int:
        return _as_int(metric(source_id, key)) or 0

    model_source_ids = [
        source["id"]
        for source in sources
        if str(source.get("kind", "")).strip() == "checkpoint_tree"
    ]

    statuses: dict[str, int] = {}
    for source in sources:
        status = str(source.get("status", "")).strip() or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "source_count": len(sources),
        "available_count": statuses.get("available", 0),
        "partial_count": statuses.get("partial", 0),
        "missing_count": statuses.get("missing", 0),
        "status_counts": dict(sorted(statuses.items())),
        "paper_rows": rows("paper_text_1m"),
        "paper_chunk_examples": rows("paper_chunks_p1"),
        "paper_universe_rows": rows("paper_universe"),
        "paper_knn_edges": metric("paper_universe", "paper_knn.edge_count"),
        "paper_topic_edges": metric("paper_universe", "paper_topic_edges"),
        "repository_count": rows("repository_exports"),
        "repository_qa_index_count": metric("repository_exports", "qa_index_count"),
        "repository_mined_skill_count": metric("repository_exports", "mined_skill_count"),
        "algorithm_catalog_rows": rows("algorithms"),
        "algorithm_implementation_files": rows("algorithms_library"),
        "trained_model_assets": sum(metric_int(source_id, "model_asset_count") for source_id in model_source_ids),
        "trained_adapter_assets": sum(metric_int(source_id, "peft_adapter_count") for source_id in model_source_ids),
        "full_model_assets": sum(metric_int(source_id, "hf_model_count") for source_id in model_source_ids),
        "faiss_index_assets": sum(metric_int(source_id, "faiss_index_count") for source_id in model_source_ids),
        "tolbert_checkpoint_assets": sum(
            metric_int(source_id, "tolbert_checkpoint_count") for source_id in model_source_ids
        ),
        "repository_model_assets": metric_int("repository_models", "model_asset_count"),
        "digital_world_model_assets": metric_int("digital_world_model_checkpoints", "model_asset_count"),
        "tolbert_model_assets": metric_int("tolbert_checkpoints", "model_asset_count"),
    }


def build_research_library_status(
    *,
    config_path: Path | str = DEFAULT_RESEARCH_LIBRARY_CONFIG,
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = root or _repo_root()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _read_json_object(config_path)
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError(f"expected sources list at {config_path}")
    sources = [build_source_status(source, root=root) for source in raw_sources if isinstance(source, dict)]
    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now(),
        "config_path": _safe_relative(config_path, root),
        "root": str(root),
        "summary": _summary(sources),
        "sources": sources,
    }


def write_research_library_status(
    output_path: Path | str,
    *,
    config_path: Path | str = DEFAULT_RESEARCH_LIBRARY_CONFIG,
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_research_library_status(
        config_path=config_path,
        root=root,
        generated_at=generated_at,
    )
    output = Path(output_path)
    if not output.is_absolute():
        output = (root or _repo_root()) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_RESEARCH_LIBRARY_CONFIG",
    "build_research_library_status",
    "build_source_status",
    "load_research_library_config",
    "resolve_source_path",
    "write_research_library_status",
]
