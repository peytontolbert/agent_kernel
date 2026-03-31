from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig


def _prune_paths(paths: list[Path], *, keep: int) -> list[str]:
    if keep < 0:
        keep = 0
    ordered = sorted(paths, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    removed: list[str] = []
    for path in ordered[keep:]:
        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            continue
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-cycle-exports", type=int, default=-1)
    parser.add_argument("--keep-report-exports", type=int, default=-1)
    parser.add_argument("--delete-episode-exports", action="store_true")
    parser.add_argument("--delete-learning-export", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    summary: dict[str, object] = {
        "removed_episode_exports": [],
        "removed_cycle_exports": [],
        "removed_report_exports": [],
        "removed_learning_export": "",
    }

    if args.delete_episode_exports:
        removed: list[str] = []
        for path in sorted(config.trajectories_root.rglob("*.json")):
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                continue
        summary["removed_episode_exports"] = removed

    if args.delete_learning_export and config.learning_artifacts_path.exists():
        try:
            config.learning_artifacts_path.unlink()
            summary["removed_learning_export"] = str(config.learning_artifacts_path)
        except OSError:
            pass

    keep_cycle = config.storage_keep_cycle_export_files if args.keep_cycle_exports < 0 else int(args.keep_cycle_exports)
    cycle_exports = list(config.improvement_cycles_path.parent.glob("cycles*.jsonl"))
    summary["removed_cycle_exports"] = _prune_paths(cycle_exports, keep=keep_cycle)

    keep_reports = config.storage_keep_report_export_files if args.keep_report_exports < 0 else int(args.keep_report_exports)
    report_exports = list(config.improvement_reports_dir.glob("*.json")) + list(config.improvement_reports_dir.glob("*.jsonl"))
    summary["removed_report_exports"] = _prune_paths(report_exports, keep=keep_reports)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
