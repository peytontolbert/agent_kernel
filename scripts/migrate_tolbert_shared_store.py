from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.tolbert_model_improvement import (
    _compact_tolbert_model_candidate_output,
    _materialize_tolbert_model_shared_store,
)


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _artifact_paths(config: KernelConfig) -> list[Path]:
    paths: list[Path] = []
    if config.tolbert_model_artifact_path.exists():
        paths.append(config.tolbert_model_artifact_path)
    candidates_root = config.candidate_artifacts_root / "tolbert_model"
    if candidates_root.exists():
        paths.extend(sorted(candidates_root.glob("*/*.json")))
    return paths


def _collect_existing_paths(value: object) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, dict):
        for item in value.values():
            paths.update(_collect_existing_paths(item))
        return paths
    if isinstance(value, list):
        for item in value:
            paths.update(_collect_existing_paths(item))
        return paths
    if not isinstance(value, str):
        return paths
    raw = value.strip()
    if not raw or "/" not in raw:
        return paths
    path = Path(raw)
    if not path.exists():
        return paths
    try:
        paths.add(path.resolve())
    except OSError:
        paths.add(path)
    return paths


def _legacy_cycle_roots(config: KernelConfig) -> list[Path]:
    root = config.candidate_artifacts_root / "tolbert_model"
    if not root.exists():
        return []
    return sorted(
        path.parent
        for path in root.glob("*/tolbert_model_artifact")
        if path.is_dir()
    )


def _migrate_inline_artifact(
    *,
    config: KernelConfig,
    artifact_path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    output_dir_value = str(payload.get("output_dir", "")).strip()
    if not output_dir_value:
        return {"artifact_path": str(artifact_path), "migrated": False, "reason": "no output_dir"}
    output_dir = Path(output_dir_value)
    if not output_dir.exists():
        return {"artifact_path": str(artifact_path), "migrated": False, "reason": "output_dir_missing"}
    if isinstance(payload.get("shared_store", {}), dict) and payload.get("shared_store", {}):
        return {"artifact_path": str(artifact_path), "migrated": False, "reason": "already_manifest_first"}

    retained_checkpoint = None
    runtime_paths = payload.get("runtime_paths", {})
    if isinstance(runtime_paths, dict):
        checkpoint_value = str(runtime_paths.get("checkpoint_path", "")).strip()
        if checkpoint_value:
            retained_checkpoint = Path(checkpoint_value)
    storage_compaction = _compact_tolbert_model_candidate_output(
        output_dir,
        retained_checkpoint_path=retained_checkpoint,
    )
    shared_store = _materialize_tolbert_model_shared_store(
        config=config,
        output_dir=output_dir,
        payload=payload,
    )
    payload["storage_compaction"] = storage_compaction
    payload["shared_store"] = shared_store
    payload["materialization_mode"] = str(shared_store.get("mode", "inline_bundle"))
    _write_json(artifact_path, payload)
    return {
        "artifact_path": str(artifact_path),
        "migrated": True,
        "shared_store": shared_store,
    }


def _cleanup_unreferenced_shared_store(config: KernelConfig, *, referenced_paths: set[Path]) -> list[str]:
    store_root = config.tolbert_model_artifact_path.parent / "store"
    if not store_root.exists():
        return []
    removed: list[str] = []
    for group_dir in sorted(path for path in store_root.iterdir() if path.is_dir()):
        for digest_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            try:
                resolved = digest_dir.resolve()
            except OSError:
                resolved = digest_dir
            if resolved in referenced_paths:
                continue
            shutil.rmtree(digest_dir, ignore_errors=True)
            if not digest_dir.exists():
                removed.append(str(digest_dir))
        try:
            next(group_dir.iterdir())
        except StopIteration:
            try:
                group_dir.rmdir()
            except OSError:
                pass
    try:
        next(store_root.iterdir())
    except StopIteration:
        try:
            store_root.rmdir()
        except OSError:
            pass
    return removed


def _remove_empty_tolbert_candidate_dirs(config: KernelConfig) -> list[str]:
    root = config.candidate_artifacts_root / "tolbert_model"
    if not root.exists():
        return []
    removed: list[str] = []
    for path in sorted(candidate for candidate in root.iterdir() if candidate.is_dir()):
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                continue
            removed.append(str(path))
    return removed


def run_migration(*, config: KernelConfig, apply: bool) -> dict[str, object]:
    migrated_artifacts: list[dict[str, object]] = []
    if apply:
        for artifact_path in _artifact_paths(config):
            payload = _load_json(artifact_path)
            if not isinstance(payload, dict):
                continue
            output_dir_value = str(payload.get("output_dir", "")).strip()
            if not output_dir_value:
                continue
            output_dir = Path(output_dir_value)
            if output_dir.exists():
                migrated_artifacts.append(
                    _migrate_inline_artifact(config=config, artifact_path=artifact_path, payload=payload)
                )

    referenced_paths: set[Path] = set()
    for artifact_path in _artifact_paths(config):
        payload = _load_json(artifact_path)
        if not isinstance(payload, dict):
            continue
        referenced_paths.update(_collect_existing_paths(payload))

    legacy_cycle_roots = _legacy_cycle_roots(config)
    preserved_legacy_roots: list[str] = []
    removable_legacy_roots: list[str] = []
    for cycle_root in legacy_cycle_roots:
        try:
            resolved_root = cycle_root.resolve()
        except OSError:
            resolved_root = cycle_root
        if any(path == resolved_root or resolved_root in path.parents for path in referenced_paths):
            preserved_legacy_roots.append(str(cycle_root))
            continue
        removable_legacy_roots.append(str(cycle_root))
        if apply:
            shutil.rmtree(cycle_root, ignore_errors=True)

    removed_shared_store = _cleanup_unreferenced_shared_store(config, referenced_paths=referenced_paths) if apply else []
    removed_empty_candidate_dirs = _remove_empty_tolbert_candidate_dirs(config) if apply else []
    return {
        "report_kind": "tolbert_shared_store_migration_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apply": bool(apply),
        "tolbert_model_artifact_path": str(config.tolbert_model_artifact_path),
        "migrated_artifacts": migrated_artifacts,
        "preserved_legacy_cycle_roots": preserved_legacy_roots,
        "removed_legacy_cycle_roots": removable_legacy_roots,
        "removed_shared_store_paths": removed_shared_store,
        "removed_empty_candidate_dirs": removed_empty_candidate_dirs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    report = run_migration(config=config, apply=bool(args.apply))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        Path(args.report_path)
        if args.report_path
        else config.improvement_reports_dir / f"tolbert_shared_store_migration_{timestamp}.json"
    )
    _write_json(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
