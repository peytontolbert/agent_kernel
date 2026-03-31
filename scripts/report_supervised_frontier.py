from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

from agent_kernel.config import KernelConfig


def _load_cycle_records(path: Path, *, config: KernelConfig | None = None) -> list[dict[str, object]]:
    if config is not None and config.uses_sqlite_storage():
        records = config.sqlite_store().load_cycle_records(output_path=path)
        if records:
            return records
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, object]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _record_for_state(records: list[dict[str, object]], state: str) -> dict[str, object]:
    for record in records:
        if str(record.get("state", "")).strip() == state:
            return record
    return {}


def _record_scope_id(record: dict[str, object]) -> str:
    metrics = _metrics_summary(record)
    return str(metrics.get("scope_id", "")).strip()


def _latest_cycle_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    latest_cycle_id = ""
    for record in reversed(records):
        cycle_id = str(record.get("cycle_id", "")).strip()
        if cycle_id:
            latest_cycle_id = cycle_id
            break
    if not latest_cycle_id:
        return []
    return [record for record in records if str(record.get("cycle_id", "")).strip() == latest_cycle_id]


def _metrics_summary(record: dict[str, object]) -> dict[str, object]:
    metrics = record.get("metrics_summary", {})
    return metrics if isinstance(metrics, dict) else {}


def _record_variant_id(record: dict[str, object]) -> str:
    metrics = _metrics_summary(record)
    direct = str(metrics.get("selected_variant_id", "")).strip()
    if direct:
        return direct
    selected_variant = metrics.get("selected_variant", {})
    if isinstance(selected_variant, dict):
        return str(selected_variant.get("variant_id", "")).strip()
    return ""


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_sha256(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return sha256(data).hexdigest()


def _state_rank(state: str) -> int:
    order = {
        "retain": 5,
        "reject": 4,
        "generate": 3,
        "select": 2,
        "observe": 1,
    }
    return order.get(str(state).strip(), 0)


def _scope_summary(*, cycles_path: Path, config: KernelConfig) -> dict[str, object] | None:
    records = _load_cycle_records(cycles_path, config=config)
    expected_scope_id = cycles_path.stem.removeprefix("cycles_")
    scoped_records = [
        record
        for record in records
        if _record_scope_id(record) == expected_scope_id
    ]
    if scoped_records:
        records = scoped_records
    cycle_records = _latest_cycle_records(records)
    if not cycle_records:
        return None
    latest = cycle_records[-1]
    observe_record = _record_for_state(cycle_records, "observe")
    select_record = _record_for_state(cycle_records, "select")
    generate_record = _record_for_state(cycle_records, "generate")
    observe_metrics = _metrics_summary(observe_record)
    select_metrics = _metrics_summary(select_record)
    latest_metrics = _metrics_summary(latest)

    scope_id = (
        str(observe_metrics.get("scope_id", "")).strip()
        or str(select_metrics.get("scope_id", "")).strip()
        or str(latest_metrics.get("scope_id", "")).strip()
        or expected_scope_id
    )
    selected_subsystem = (
        str(select_record.get("subsystem", "")).strip()
        or str(generate_record.get("subsystem", "")).strip()
        or str(latest.get("subsystem", "")).strip()
    )
    selected_variant_id = (
        _record_variant_id(select_record)
        or _record_variant_id(generate_record)
        or _record_variant_id(latest)
    )
    candidate_artifact_path = str(
        generate_record.get("candidate_artifact_path")
        or generate_record.get("artifact_path")
        or latest.get("candidate_artifact_path")
        or ""
    ).strip()
    candidate_path = Path(candidate_artifact_path) if candidate_artifact_path else None
    candidate_exists = bool(candidate_path and candidate_path.exists())
    candidate_digest = _candidate_sha256(candidate_path) if candidate_path is not None and candidate_exists else ""
    observation_timed_out = bool(
        observe_metrics.get("observation_timed_out", False) or observe_metrics.get("observation_budget_exceeded", False)
    )
    return {
        "scope_id": scope_id,
        "cycles_path": str(cycles_path),
        "cycle_id": str(latest.get("cycle_id", "")).strip(),
        "protocol": str(
            observe_metrics.get("protocol", "")
            or select_metrics.get("protocol", "")
            or latest_metrics.get("protocol", "")
        ).strip(),
        "scoped_run": bool(
            observe_metrics.get("scoped_run", False)
            or select_metrics.get("scoped_run", False)
            or latest_metrics.get("scoped_run", False)
        ),
        "selected_subsystem": selected_subsystem,
        "selected_variant_id": selected_variant_id,
        "last_state": str(latest.get("state", "")).strip(),
        "generated_candidate": bool(generate_record) and bool(candidate_artifact_path),
        "candidate_artifact_path": candidate_artifact_path,
        "candidate_exists": candidate_exists,
        "candidate_sha256": candidate_digest,
        "candidate_artifact_kind": str(
            generate_record.get("artifact_kind", "") or latest.get("artifact_kind", "")
        ).strip(),
        "observation_timed_out": observation_timed_out,
        "observation_warning": str(observe_metrics.get("observation_warning", "")).strip(),
        "observation_elapsed_seconds": _safe_float(observe_metrics.get("observation_elapsed_seconds", 0.0)),
        "observation_timeout_budget_source": str(
            observe_metrics.get("observation_current_task_timeout_budget_source", "")
        ).strip(),
        "observation_timeout_budget_seconds": _safe_float(
            observe_metrics.get("observation_current_task_timeout_budget_seconds", 0.0)
        ),
    }


def _frontier_sort_key(summary: dict[str, object]) -> tuple[object, ...]:
    return (
        0 if bool(summary.get("generated_candidate", False)) else 1,
        0 if bool(summary.get("candidate_exists", False)) else 1,
        0 if not bool(summary.get("observation_timed_out", False)) else 1,
        -_state_rank(str(summary.get("last_state", "")).strip()),
        _safe_float(summary.get("observation_elapsed_seconds", 0.0)),
        str(summary.get("scope_id", "")).strip(),
    )


def _dedupe_frontier(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        digest = str(run.get("candidate_sha256", "")).strip()
        candidate_path = str(run.get("candidate_artifact_path", "")).strip()
        fingerprint = digest or candidate_path or str(run.get("scope_id", "")).strip()
        key = (
            str(run.get("selected_subsystem", "")).strip(),
            str(run.get("selected_variant_id", "")).strip(),
            fingerprint,
        )
        groups[key].append(run)
    frontier: list[dict[str, object]] = []
    for siblings in groups.values():
        ordered = sorted(siblings, key=_frontier_sort_key)
        winner = dict(ordered[0])
        winner["duplicate_scope_ids"] = [
            str(item.get("scope_id", "")).strip()
            for item in ordered[1:]
            if str(item.get("scope_id", "")).strip()
        ]
        winner["duplicate_count"] = len(ordered) - 1
        frontier.append(winner)
    frontier.sort(key=_frontier_sort_key)
    return frontier


def _summary_payload(runs: list[dict[str, object]], frontier: list[dict[str, object]]) -> dict[str, object]:
    subsystem_counter = Counter(
        str(run.get("selected_subsystem", "")).strip()
        for run in runs
        if str(run.get("selected_subsystem", "")).strip()
    )
    timeout_source_counter = Counter(
        str(run.get("observation_timeout_budget_source", "")).strip()
        for run in runs
        if str(run.get("observation_timeout_budget_source", "")).strip()
    )
    return {
        "scoped_run_count": len(runs),
        "frontier_candidate_count": len(frontier),
        "generated_candidate_runs": sum(1 for run in runs if bool(run.get("generated_candidate", False))),
        "timed_out_runs": sum(1 for run in runs if bool(run.get("observation_timed_out", False))),
        "deduped_runs": sum(int(item.get("duplicate_count", 0) or 0) for item in frontier),
        "subsystems": dict(sorted(subsystem_counter.items())),
        "timeout_budget_sources": dict(sorted(timeout_source_counter.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycles-glob",
        default="cycles_*.jsonl",
        help="Glob relative to trajectories/improvement/ for scoped cycle files to ingest.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional explicit output path. Defaults to improvement_reports_dir/supervised_parallel_frontier.json",
    )
    parser.add_argument(
        "--include-unscoped",
        action="store_true",
        help="Include cycle files whose latest cycle does not report scoped_run=true.",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    search_root = config.improvement_cycles_path.parent
    output_path = (
        Path(args.output_path)
        if str(args.output_path).strip()
        else config.improvement_reports_dir / "supervised_parallel_frontier.json"
    )

    runs: list[dict[str, object]] = []
    cycle_paths = (
        sorted(config.sqlite_store().list_cycle_paths(parent=search_root, pattern=str(args.cycles_glob)))
        if config.uses_sqlite_storage()
        else sorted(search_root.glob(str(args.cycles_glob)))
    )
    for cycles_path in cycle_paths:
        summary = _scope_summary(cycles_path=cycles_path, config=config)
        if summary is None:
            continue
        if not args.include_unscoped and not bool(summary.get("scoped_run", False)):
            continue
        runs.append(summary)

    frontier = _dedupe_frontier(runs)
    payload = {
        "report_kind": "supervised_parallel_frontier_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_root": str(search_root),
        "cycles_glob": str(args.cycles_glob),
        "summary": _summary_payload(runs, frontier),
        "frontier_candidates": frontier,
        "scoped_runs": sorted(runs, key=_frontier_sort_key),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
