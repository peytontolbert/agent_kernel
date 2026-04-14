from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

from ..config import KernelConfig


_PINNED_IMPROVEMENT_REPORT_FILENAMES = {
    "parallel_supervised_preview_history.jsonl",
    "supervised_parallel_frontier.json",
    "supervised_frontier_promotion_plan.json",
    "supervised_frontier_promotion_pass.json",
    "supervisor_baseline_bootstrap_queue.json",
    "supervisor_loop_history.jsonl",
    "supervisor_loop_report.json",
    "supervisor_loop_status.json",
    "supervisor_protected_review_queue.json",
    "supervisor_trust_streak_recovery_queue.json",
}

_RUN_REPORT_PRUNABLE_PREFIXES = (
    "task_report_",
    "generated_success_",
    "generated_failure_",
)


def _resolved_existing_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    try:
        return path.resolve()
    except OSError:
        return path


def _path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _path_sort_mtime(path: Path) -> float:
    try:
        latest = path.stat().st_mtime
    except OSError:
        return 0.0
    if not path.is_dir():
        return latest
    for root, _, files in os.walk(path):
        try:
            latest = max(latest, Path(root).stat().st_mtime)
        except OSError:
            pass
        for name in files:
            try:
                latest = max(latest, (Path(root) / name).stat().st_mtime)
            except OSError:
                continue
    return latest


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _prune_paths(paths: list[Path], *, keep: int, preserve: set[Path]) -> list[str]:
    ordered = sorted(
        paths,
        key=_path_sort_mtime,
        reverse=True,
    )
    removed: list[str] = []
    kept = 0
    for path in ordered:
        resolved = _resolved_existing_path(path)
        if kept < max(0, int(keep)):
            kept += 1
            continue
        if resolved is not None and resolved in preserve:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except OSError:
            continue
        if not path.exists():
            removed.append(str(path))
    return removed


def _compact_jsonl_tail(path: Path, *, keep_records: int) -> bool:
    if keep_records <= 0 or not path.exists() or not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    if len(lines) <= keep_records:
        return False
    _atomic_write_text(path, "\n".join(lines[-keep_records:]) + "\n", encoding="utf-8")
    return True


def _preserved_paths(paths: tuple[Path, ...]) -> set[Path]:
    preserved: set[Path] = set()
    for path in paths:
        resolved = _resolved_existing_path(path)
        if resolved is not None:
            preserved.add(resolved)
            preserved.update(parent for parent in resolved.parents)
    return preserved


def _scope_candidate_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    scope_dirs: list[Path] = []
    namespace_dirs: list[Path] = []
    if not root.exists() or not root.is_dir():
        return scope_dirs, namespace_dirs
    for child in sorted(item for item in root.iterdir() if item.is_dir()):
        if child.name == "tolbert_model":
            namespace_dirs.append(child)
            continue
        nested_dirs = [item for item in child.iterdir() if item.is_dir()]
        if nested_dirs and any(item.name.startswith("cycle_") for item in nested_dirs):
            namespace_dirs.append(child)
            continue
        scope_dirs.append(child)
    return scope_dirs, namespace_dirs


def _run_report_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.iterdir():
        if not path.is_file() or path.name.startswith('.'):
            continue
        if path.suffix not in {".json", ".jsonl"}:
            continue
        files.append(path)
    return files


def _checkpoint_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*.json")
        if path.is_file() and not any(part.startswith('.') for part in path.relative_to(root).parts)
    ]


def _prunable_run_reports(config: KernelConfig) -> list[Path]:
    pinned = {
        config.unattended_trust_ledger_path.name,
    }
    reports = []
    for path in _run_report_files(config.run_reports_dir):
        if path.name in pinned:
            continue
        if path.name.startswith("job_report_"):
            continue
        if path.name.endswith(".status.json") or path.name.endswith(".alerts.json") or path.name.endswith(".events.jsonl"):
            continue
        if path.name.startswith(_RUN_REPORT_PRUNABLE_PREFIXES):
            reports.append(path)
    return reports


def _prunable_run_checkpoints(root: Path) -> list[Path]:
    files = []
    for path in _checkpoint_files(root):
        if path.name.startswith("job_"):
            continue
        files.append(path)
    return files


def govern_improvement_export_storage(
    config: KernelConfig,
    *,
    preserve_paths: tuple[Path, ...] = (),
    include_cycle_exports: bool = True,
    include_report_exports: bool = True,
    include_candidate_exports: bool = True,
    include_run_reports: bool = True,
    include_run_checkpoints: bool = True,
) -> dict[str, object]:
    preserved = _preserved_paths(preserve_paths)

    cycle_root = config.improvement_cycles_path.parent
    cycle_files = sorted(cycle_root.glob("cycles*.jsonl")) if include_cycle_exports and cycle_root.exists() else []
    compacted_cycles = (
        [
            str(path)
            for path in cycle_files
            if _compact_jsonl_tail(path, keep_records=max(0, int(config.storage_max_cycle_export_records)))
        ]
        if include_cycle_exports
        else []
    )
    removed_cycle_exports = (
        _prune_paths(
            cycle_files,
            keep=max(0, int(config.storage_keep_cycle_export_files)),
            preserve=preserved,
        )
        if include_cycle_exports
        else []
    )

    reports_root = config.improvement_reports_dir
    report_files = _run_report_files(reports_root) if include_report_exports else []
    compacted_report_histories = (
        [
            str(path)
            for path in report_files
            if path.suffix == ".jsonl"
            and _compact_jsonl_tail(path, keep_records=max(0, int(config.storage_max_report_history_records)))
        ]
        if include_report_exports
        else []
    )
    prunable_reports = [path for path in report_files if path.name not in _PINNED_IMPROVEMENT_REPORT_FILENAMES]
    removed_report_exports = (
        _prune_paths(
            prunable_reports,
            keep=max(0, int(config.storage_keep_report_export_files)),
            preserve=preserved,
        )
        if include_report_exports
        else []
    )

    candidates_root = config.candidate_artifacts_root
    scope_dirs, namespace_dirs = _scope_candidate_dirs(candidates_root) if include_candidate_exports else ([], [])
    removed_candidate_scope_dirs = (
        _prune_paths(
            scope_dirs,
            keep=max(0, int(config.storage_keep_candidate_export_dirs)),
            preserve=preserved,
        )
        if include_candidate_exports
        else []
    )

    removed_namespace_candidate_dirs: dict[str, list[str]] = {}
    if include_candidate_exports:
        for namespace_dir in namespace_dirs:
            if namespace_dir.name == "tolbert_model":
                continue
            child_dirs = [item for item in namespace_dir.iterdir() if item.is_dir()]
            removed = _prune_paths(
                child_dirs,
                keep=max(0, int(config.storage_keep_namespace_candidate_dirs)),
                preserve=preserved,
            )
            if removed:
                removed_namespace_candidate_dirs[str(namespace_dir)] = removed

    run_reports_root = config.run_reports_dir
    removed_run_reports = (
        _prune_paths(
            _prunable_run_reports(config),
            keep=max(0, int(config.storage_keep_run_report_files)),
            preserve=preserved,
        )
        if include_run_reports
        else []
    )

    checkpoints_root = config.run_checkpoints_dir
    removed_run_checkpoints = (
        _prune_paths(
            _prunable_run_checkpoints(checkpoints_root),
            keep=max(0, int(config.storage_keep_run_checkpoint_files)),
            preserve=preserved,
        )
        if include_run_checkpoints
        else []
    )

    from ..extensions.improvement.tolbert_model_improvement import cleanup_tolbert_model_candidate_storage

    tolbert_preserve = tuple(
        path
        for path in preserve_paths
        if candidates_root in path.parents or path == candidates_root
    )
    tolbert_storage = (
        cleanup_tolbert_model_candidate_storage(
            config=config,
            preserve_paths=tolbert_preserve,
        )
        if include_candidate_exports
        else {}
    )

    return {
        "compacted_cycle_exports": compacted_cycles,
        "removed_cycle_exports": removed_cycle_exports,
        "compacted_report_histories": compacted_report_histories,
        "removed_report_exports": removed_report_exports,
        "removed_candidate_scope_dirs": removed_candidate_scope_dirs,
        "removed_namespace_candidate_dirs": removed_namespace_candidate_dirs,
        "removed_run_reports": removed_run_reports,
        "removed_run_checkpoints": removed_run_checkpoints,
        "tolbert_storage": tolbert_storage,
        "candidate_root_bytes": _path_bytes(candidates_root) if include_candidate_exports else 0,
        "report_root_bytes": _path_bytes(reports_root) if include_report_exports else 0,
        "cycle_root_bytes": _path_bytes(cycle_root) if include_cycle_exports else 0,
        "run_report_root_bytes": _path_bytes(run_reports_root) if include_run_reports else 0,
        "run_checkpoint_root_bytes": _path_bytes(checkpoints_root) if include_run_checkpoints else 0,
    }
