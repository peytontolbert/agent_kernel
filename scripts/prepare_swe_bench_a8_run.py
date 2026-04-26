from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import sys
from typing import Any


DEFAULT_DATASETS = {
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified",
    "swe_rebench": "",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_swe_bench_command(
    *,
    python_bin: str,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str,
    report_dir: str,
    instance_ids: list[str] | None = None,
) -> list[str]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if cache_level not in {"none", "base", "env", "instance"}:
        raise ValueError("cache_level must be one of none, base, env, instance")
    command = [
        python_bin,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--cache_level",
        cache_level,
        "--run_id",
        run_id,
        "--namespace",
        namespace,
        "--report_dir",
        report_dir,
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def _count_from_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list | tuple | set):
            return len(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value.strip()))
            except ValueError:
                continue
    return None


def summarize_swe_bench_results(results: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
    resolved = _count_from_keys(
        results,
        (
            "resolved_count",
            "instances_resolved",
            "num_resolved",
            "resolved",
            "resolved_ids",
            "resolved_instances",
        ),
    )
    total = _count_from_keys(
        results,
        (
            "total",
            "total_instances",
            "instance_count",
            "instances_submitted",
            "submitted_instances",
            "submitted_ids",
            "all_instances",
        ),
    )
    if total is None:
        unresolved = _count_from_keys(results, ("unresolved", "unresolved_ids", "unresolved_instances"))
        errors = _count_from_keys(results, ("errors", "error_ids", "error_instances"))
        if resolved is not None and unresolved is not None:
            total = resolved + unresolved + int(errors or 0)
    if resolved is None:
        raise ValueError("SWE-bench results must include resolved count or resolved ID list")
    if total is None or total <= 0:
        raise ValueError("SWE-bench results must include positive total/submitted count")
    if resolved < 0 or resolved > total:
        raise ValueError("resolved count must be between zero and total")
    return {
        "report_kind": "official_swe_bench_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "resolved_count": int(resolved),
        "task_count": int(total),
        "resolve_rate": round(float(resolved) / float(total), 6),
    }


def _command_mode(args: argparse.Namespace) -> None:
    dataset_name = str(args.dataset_name or DEFAULT_DATASETS[args.benchmark]).strip()
    if not dataset_name:
        raise SystemExit("--dataset-name is required for this benchmark")
    instance_ids = [value.strip() for value in args.instance_ids if value.strip()] if args.instance_ids else None
    command = build_swe_bench_command(
        python_bin=str(args.python_bin),
        dataset_name=dataset_name,
        split=str(args.split),
        predictions_path=str(args.predictions_path),
        run_id=str(args.run_id),
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=str(args.cache_level),
        namespace=str(args.namespace),
        report_dir=str(args.report_dir),
        instance_ids=instance_ids,
    )
    _write_json(Path(args.output_command_json), command)
    if args.output_spec_json:
        _write_json(
            Path(args.output_spec_json),
            {
                "spec_version": "asi_v1",
                "report_kind": "a8_swe_bench_runner_spec",
                "benchmark": args.benchmark,
                "dataset_name": dataset_name,
                "split": args.split,
                "predictions_path": args.predictions_path,
                "run_id": args.run_id,
                "runner_cwd": args.swe_bench_root,
                "summary_json": args.summary_json,
                "output_packet_json": args.output_packet_json,
                "runner_command_json": args.output_command_json,
            },
        )
    print(f"benchmark={args.benchmark} command_json={args.output_command_json}")


def _summarize_mode(args: argparse.Namespace) -> None:
    results_path = Path(args.results_json)
    summary = summarize_swe_bench_results(_read_json(results_path), source_path=str(results_path))
    _write_json(Path(args.output_summary_json), summary)
    print(
        f"resolved_count={summary['resolved_count']} "
        f"task_count={summary['task_count']} "
        f"resolve_rate={summary['resolve_rate']} "
        f"summary_json={args.output_summary_json}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    command_parser = subparsers.add_parser("command")
    command_parser.add_argument("--benchmark", choices=tuple(DEFAULT_DATASETS), required=True)
    command_parser.add_argument("--dataset-name", default="")
    command_parser.add_argument("--split", default="test")
    command_parser.add_argument("--predictions-path", required=True)
    command_parser.add_argument("--run-id", required=True)
    command_parser.add_argument("--max-workers", type=int, default=4)
    command_parser.add_argument("--timeout", type=int, default=1800)
    command_parser.add_argument("--cache-level", default="env")
    command_parser.add_argument("--namespace", default="swebench")
    command_parser.add_argument("--report-dir", default=".")
    command_parser.add_argument("--python-bin", default=sys.executable)
    command_parser.add_argument("--swe-bench-root", default="/data/agiattempt/agi_dw/third_party/swe-bench")
    command_parser.add_argument("--summary-json", default="")
    command_parser.add_argument("--output-packet-json", default="")
    command_parser.add_argument("--output-command-json", required=True)
    command_parser.add_argument("--output-spec-json", default="")
    command_parser.add_argument("--instance-ids", nargs="*")
    command_parser.set_defaults(func=_command_mode)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results-json", required=True)
    summarize_parser.add_argument("--output-summary-json", required=True)
    summarize_parser.set_defaults(func=_summarize_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
