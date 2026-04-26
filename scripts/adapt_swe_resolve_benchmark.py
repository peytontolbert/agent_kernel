from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from typing import Any


SUPPORTED_BENCHMARKS = ("swe_bench_verified", "swe_rebench")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _nested_get(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _first_number(payload: dict[str, Any], field_paths: tuple[str, ...]) -> float | None:
    for field_path in field_paths:
        value = _nested_get(payload, field_path)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def build_swe_resolve_benchmark_result(
    summary: dict[str, Any],
    *,
    benchmark: str,
    source_path: str = "",
    conservative_comparison_report: bool = False,
) -> dict[str, Any]:
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"benchmark must be one of {SUPPORTED_BENCHMARKS!r}")
    resolved = _first_number(
        summary,
        (
            "resolved",
            "resolved_count",
            "passed",
            "success_count",
            "metrics.resolved",
            "metrics.resolved_count",
            "metrics.passed",
            "metrics.success_count",
        ),
    )
    total = _first_number(
        summary,
        (
            "total",
            "task_count",
            "instance_count",
            "total_count",
            "metrics.total",
            "metrics.task_count",
            "metrics.instance_count",
            "metrics.total_count",
        ),
    )
    explicit_rate = _first_number(
        summary,
        (
            "resolve_rate",
            "pass_rate",
            "success_rate",
            "metrics.resolve_rate",
            "metrics.pass_rate",
            "metrics.success_rate",
        ),
    )
    if total is None or total <= 0:
        raise ValueError("summary must include positive total/task_count")
    if resolved is None and explicit_rate is None:
        raise ValueError("summary must include resolved count or explicit resolve_rate")
    if resolved is None:
        resolved = explicit_rate * total if explicit_rate is not None else 0.0
    if resolved < 0 or resolved > total:
        raise ValueError("resolved count must be between zero and total")
    resolve_rate = explicit_rate if explicit_rate is not None else resolved / total
    if resolve_rate < 0.0 or resolve_rate > 1.0:
        raise ValueError("resolve_rate must be between zero and one")
    lower_bound = _first_number(
        summary,
        (
            "resolve_rate_lower_bound",
            "pass_rate_lower_bound",
            "metrics.resolve_rate_lower_bound",
            "metrics.pass_rate_lower_bound",
        ),
    )
    metrics: dict[str, Any] = {
        "resolve_rate": round(resolve_rate, 6),
        "resolved_count": int(resolved),
        "task_count": int(total),
        "conservative_comparison_report": bool(conservative_comparison_report),
    }
    if lower_bound is not None:
        metrics["resolve_rate_lower_bound"] = round(lower_bound, 6)
    return {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_result",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": benchmark,
        "metrics": metrics,
        "source": {
            "source_path": source_path,
            "summary_report_kind": str(summary.get("report_kind", "")).strip(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--conservative-comparison-report", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    packet = build_swe_resolve_benchmark_result(
        _read_json(summary_path),
        benchmark=args.benchmark,
        source_path=str(summary_path),
        conservative_comparison_report=bool(args.conservative_comparison_report),
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"benchmark={packet['benchmark']} "
        f"resolve_rate={packet['metrics']['resolve_rate']} "
        f"task_count={packet['metrics']['task_count']} "
        f"output_json={output_path}"
    )


if __name__ == "__main__":
    main()
