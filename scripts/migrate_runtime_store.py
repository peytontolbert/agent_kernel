from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.ops.episode_store import episode_storage_metadata
from agent_kernel.storage import store_for_config


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, object]] = []
    for line in lines:
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_learning_candidates(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates", [])
    return [candidate for candidate in candidates if isinstance(candidate, dict)] if isinstance(candidates, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-glob", default="cycles*.jsonl")
    parser.add_argument("--skip-episodes", action="store_true")
    parser.add_argument("--skip-learning", action="store_true")
    parser.add_argument("--skip-cycles", action="store_true")
    parser.add_argument("--skip-jobs", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    store = store_for_config(config)

    imported = {
        "episodes": 0,
        "learning_candidates": 0,
        "cycle_records": 0,
        "job_rows": 0,
        "runtime_states": 0,
    }

    if not args.skip_episodes:
        for path in sorted(config.trajectories_root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            task_id = str(payload.get("task_id", "")).strip()
            if not task_id:
                continue
            store.upsert_episode_document(
                payload=payload,
                storage=episode_storage_metadata(config.trajectories_root, path),
            )
            imported["episodes"] += 1

    if not args.skip_learning and config.learning_artifacts_path.exists():
        candidates = _load_learning_candidates(config.learning_artifacts_path)
        store.upsert_learning_candidates(candidates)
        imported["learning_candidates"] = len(candidates)

    if not args.skip_cycles:
        for path in sorted(config.improvement_cycles_path.parent.glob(str(args.cycles_glob))):
            records = _load_jsonl_records(path)
            if not records:
                continue
            store.replace_cycle_records(output_path=path, records=records)
            imported["cycle_records"] += len(records)

    if not args.skip_jobs and config.delegated_job_queue_path.exists():
        try:
            queue_payload = json.loads(config.delegated_job_queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            queue_payload = {}
        jobs = queue_payload.get("jobs", []) if isinstance(queue_payload, dict) else []
        normalized_jobs = [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []
        store.replace_job_queue(queue_path=config.delegated_job_queue_path, jobs=normalized_jobs)
        imported["job_rows"] = len(normalized_jobs)

    if not args.skip_jobs and config.delegated_job_runtime_state_path.exists():
        try:
            runtime_payload = json.loads(config.delegated_job_runtime_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runtime_payload = {}
        if isinstance(runtime_payload, dict):
            store.replace_runtime_state(runtime_path=config.delegated_job_runtime_state_path, payload=runtime_payload)
            imported["runtime_states"] = 1

    print(json.dumps(imported, indent=2))


if __name__ == "__main__":
    main()
