from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from typing import Any


BENCHMARK_FIELD_MAP = {
    "codeforces": {
        "rating_equivalent": (
            "rating_equivalent",
            "rating",
            "elo",
            "codeforces_rating",
            "metrics.rating_equivalent",
            "metrics.rating",
            "metrics.elo",
            "metrics.codeforces_rating",
        )
    },
    "mle_bench": {
        "gold_medal_rate": (
            "gold_medal_rate",
            "gold_rate",
            "medal_gold_rate",
            "metrics.gold_medal_rate",
            "metrics.gold_rate",
            "metrics.medal_gold_rate",
        )
    },
    "re_bench": {
        "human_expert_win_rate": (
            "human_expert_win_rate",
            "expert_win_rate",
            "win_rate",
            "metrics.human_expert_win_rate",
            "metrics.expert_win_rate",
            "metrics.win_rate",
        )
    },
    "sustained_coding_window": {
        "window_count": ("window_count", "windows", "metrics.window_count", "metrics.windows"),
        "task_count": ("task_count", "tasks", "metrics.task_count", "metrics.tasks"),
        "strong_human_baseline_win_rate": (
            "strong_human_baseline_win_rate",
            "baseline_win_rate",
            "metrics.strong_human_baseline_win_rate",
            "metrics.baseline_win_rate",
        ),
        "strong_human_baseline_win_rate_lower_bound": (
            "strong_human_baseline_win_rate_lower_bound",
            "baseline_win_rate_lower_bound",
            "metrics.strong_human_baseline_win_rate_lower_bound",
            "metrics.baseline_win_rate_lower_bound",
        ),
        "unfamiliar_domain_slice_count": (
            "unfamiliar_domain_slice_count",
            "unfamiliar_slices",
            "metrics.unfamiliar_domain_slice_count",
            "metrics.unfamiliar_slices",
        ),
        "long_horizon_transfer_slice_count": (
            "long_horizon_transfer_slice_count",
            "long_horizon_slices",
            "metrics.long_horizon_transfer_slice_count",
            "metrics.long_horizon_slices",
        ),
        "strong_baseline_comparison_slice_count": (
            "strong_baseline_comparison_slice_count",
            "baseline_comparison_slices",
            "metrics.strong_baseline_comparison_slice_count",
            "metrics.baseline_comparison_slices",
        ),
        "regression_rate": ("regression_rate", "metrics.regression_rate"),
    },
    "recursive_compounding": {
        "retained_gain_runs": (
            "retained_gain_runs",
            "retained_runs",
            "metrics.retained_gain_runs",
            "metrics.retained_runs",
        ),
        "window_count": ("window_count", "windows", "metrics.window_count", "metrics.windows"),
    },
}


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


def _first_bool(payload: dict[str, Any], field_paths: tuple[str, ...]) -> bool | None:
    for field_path in field_paths:
        value = _nested_get(payload, field_path)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
    return None


def build_a8_benchmark_result(
    summary: dict[str, Any],
    *,
    benchmark: str,
    source_path: str = "",
    conservative_comparison_report: bool = False,
) -> dict[str, Any]:
    if benchmark not in BENCHMARK_FIELD_MAP:
        raise ValueError(f"benchmark must be one of {tuple(BENCHMARK_FIELD_MAP)!r}")
    metrics: dict[str, Any] = {}
    for metric, field_paths in BENCHMARK_FIELD_MAP[benchmark].items():
        value = _first_number(summary, field_paths)
        if value is None:
            raise ValueError(f"summary must include {metric} for benchmark {benchmark}")
        metrics[metric] = int(value) if metric.endswith("_count") or metric.endswith("_runs") else round(value, 6)
    if benchmark == "codeforces":
        metrics["rating_equivalent"] = int(metrics["rating_equivalent"])
    if benchmark == "recursive_compounding":
        verified = _first_bool(
            summary,
            (
                "verified_recursive_compounding",
                "metrics.verified_recursive_compounding",
                "recursive_compounding_verified",
                "metrics.recursive_compounding_verified",
            ),
        )
        metrics["verified_recursive_compounding"] = bool(verified)
    metrics["conservative_comparison_report"] = bool(conservative_comparison_report)
    _validate_metrics(benchmark, metrics)
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


def _validate_metrics(benchmark: str, metrics: dict[str, Any]) -> None:
    rate_fields = {
        "gold_medal_rate",
        "human_expert_win_rate",
        "strong_human_baseline_win_rate",
        "strong_human_baseline_win_rate_lower_bound",
        "regression_rate",
    }
    for field in rate_fields:
        if field not in metrics:
            continue
        value = float(metrics[field])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{field} must be between zero and one")
    for field in (
        "window_count",
        "task_count",
        "unfamiliar_domain_slice_count",
        "long_horizon_transfer_slice_count",
        "strong_baseline_comparison_slice_count",
        "retained_gain_runs",
        "rating_equivalent",
    ):
        if field not in metrics:
            continue
        if int(metrics[field]) < 0:
            raise ValueError(f"{field} must be non-negative")
    if benchmark == "recursive_compounding" and metrics.get("verified_recursive_compounding") is not True:
        raise ValueError("recursive_compounding summary must mark verified_recursive_compounding true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_FIELD_MAP), required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--conservative-comparison-report", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    packet = build_a8_benchmark_result(
        _read_json(summary_path),
        benchmark=args.benchmark,
        source_path=str(summary_path),
        conservative_comparison_report=bool(args.conservative_comparison_report),
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metric_preview = " ".join(
        f"{key}={value}"
        for key, value in sorted(packet["metrics"].items())
        if key != "conservative_comparison_report"
    )
    print(f"benchmark={packet['benchmark']} {metric_preview} output_json={output_path}")


if __name__ == "__main__":
    main()
