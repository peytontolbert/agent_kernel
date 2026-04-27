from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from typing import Any


MetricSpec = dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


DEFAULT_ADAPTER_SPECS: dict[str, dict[str, Any]] = {
    "codeforces": {
        "metrics": {
            "rating_equivalent": {
                "type": "integer",
                "aliases": [
                    "rating_equivalent",
                    "rating",
                    "elo",
                    "codeforces_rating",
                    "metrics.rating_equivalent",
                    "metrics.rating",
                    "metrics.elo",
                    "metrics.codeforces_rating",
                ],
                "required": True,
                "minimum": 0,
            }
        }
    },
    "mle_bench": {
        "metrics": {
            "gold_medal_rate": {
                "type": "rate",
                "aliases": [
                    "gold_medal_rate",
                    "gold_rate",
                    "medal_gold_rate",
                    "metrics.gold_medal_rate",
                    "metrics.gold_rate",
                    "metrics.medal_gold_rate",
                ],
                "required": True,
            }
        }
    },
    "re_bench": {
        "metrics": {
            "human_expert_win_rate": {
                "type": "rate",
                "aliases": [
                    "human_expert_win_rate",
                    "expert_win_rate",
                    "win_rate",
                    "metrics.human_expert_win_rate",
                    "metrics.expert_win_rate",
                    "metrics.win_rate",
                ],
                "required": True,
            }
        }
    },
    "swe_bench_verified": {
        "metrics": {
            "resolved_count": {
                "type": "integer",
                "aliases": [
                    "resolved",
                    "resolved_count",
                    "passed",
                    "success_count",
                    "metrics.resolved",
                    "metrics.resolved_count",
                    "metrics.passed",
                    "metrics.success_count",
                ],
                "minimum": 0,
            },
            "task_count": {
                "type": "integer",
                "aliases": [
                    "total",
                    "task_count",
                    "instance_count",
                    "total_count",
                    "metrics.total",
                    "metrics.task_count",
                    "metrics.instance_count",
                    "metrics.total_count",
                ],
                "required": True,
                "minimum": 1,
            },
            "resolve_rate": {
                "type": "rate",
                "aliases": [
                    "resolve_rate",
                    "pass_rate",
                    "success_rate",
                    "metrics.resolve_rate",
                    "metrics.pass_rate",
                    "metrics.success_rate",
                ],
                "fallback": {
                    "op": "divide",
                    "numerator": "resolved_count",
                    "denominator": "task_count",
                },
                "required": True,
            },
            "resolve_rate_lower_bound": {
                "type": "rate",
                "aliases": [
                    "resolve_rate_lower_bound",
                    "pass_rate_lower_bound",
                    "metrics.resolve_rate_lower_bound",
                    "metrics.pass_rate_lower_bound",
                ],
            },
        }
    },
    "swe_rebench": {
        "inherits": "swe_bench_verified",
    },
    "sustained_coding_window": {
        "metrics": {
            "window_count": {
                "type": "integer",
                "aliases": ["window_count", "windows", "metrics.window_count", "metrics.windows"],
                "required": True,
                "minimum": 0,
            },
            "task_count": {
                "type": "integer",
                "aliases": ["task_count", "tasks", "metrics.task_count", "metrics.tasks"],
                "required": True,
                "minimum": 0,
            },
            "strong_human_baseline_win_rate": {
                "type": "rate",
                "aliases": [
                    "strong_human_baseline_win_rate",
                    "baseline_win_rate",
                    "metrics.strong_human_baseline_win_rate",
                    "metrics.baseline_win_rate",
                ],
                "required": True,
            },
            "strong_human_baseline_win_rate_lower_bound": {
                "type": "rate",
                "aliases": [
                    "strong_human_baseline_win_rate_lower_bound",
                    "baseline_win_rate_lower_bound",
                    "metrics.strong_human_baseline_win_rate_lower_bound",
                    "metrics.baseline_win_rate_lower_bound",
                ],
                "required": True,
            },
            "unfamiliar_domain_slice_count": {
                "type": "integer",
                "aliases": [
                    "unfamiliar_domain_slice_count",
                    "unfamiliar_slices",
                    "metrics.unfamiliar_domain_slice_count",
                    "metrics.unfamiliar_slices",
                ],
                "required": True,
                "minimum": 0,
            },
            "long_horizon_transfer_slice_count": {
                "type": "integer",
                "aliases": [
                    "long_horizon_transfer_slice_count",
                    "long_horizon_slices",
                    "metrics.long_horizon_transfer_slice_count",
                    "metrics.long_horizon_slices",
                ],
                "required": True,
                "minimum": 0,
            },
            "strong_baseline_comparison_slice_count": {
                "type": "integer",
                "aliases": [
                    "strong_baseline_comparison_slice_count",
                    "baseline_comparison_slices",
                    "metrics.strong_baseline_comparison_slice_count",
                    "metrics.baseline_comparison_slices",
                ],
                "required": True,
                "minimum": 0,
            },
            "regression_rate": {
                "type": "rate",
                "aliases": ["regression_rate", "metrics.regression_rate"],
                "required": True,
            },
        }
    },
    "recursive_compounding": {
        "metrics": {
            "retained_gain_runs": {
                "type": "integer",
                "aliases": [
                    "retained_gain_runs",
                    "retained_runs",
                    "metrics.retained_gain_runs",
                    "metrics.retained_runs",
                ],
                "required": True,
                "minimum": 0,
            },
            "window_count": {
                "type": "integer",
                "aliases": ["window_count", "windows", "metrics.window_count", "metrics.windows"],
                "required": True,
                "minimum": 0,
            },
            "verified_recursive_compounding": {
                "type": "bool",
                "aliases": [
                    "verified_recursive_compounding",
                    "metrics.verified_recursive_compounding",
                    "recursive_compounding_verified",
                    "metrics.recursive_compounding_verified",
                ],
                "required": True,
                "must_be_true": True,
            },
        }
    },
}


def _resolve_default_payload(benchmark: str) -> dict[str, Any]:
    if benchmark not in DEFAULT_ADAPTER_SPECS:
        raise ValueError(f"no default A8 adapter spec for benchmark {benchmark!r}")
    payload = dict(DEFAULT_ADAPTER_SPECS[benchmark])
    inherited = str(payload.pop("inherits", "")).strip()
    if inherited:
        inherited_payload = _resolve_default_payload(inherited)
        inherited_payload["benchmark"] = benchmark
        return inherited_payload
    payload["benchmark"] = benchmark
    return payload


def build_default_adapter_spec(benchmark: str) -> dict[str, Any]:
    payload = _resolve_default_payload(benchmark)
    return {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_adapter_spec",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": benchmark,
        "metrics": payload.get("metrics", {}),
        "open_limits": [
            "This adapter spec maps an already-produced benchmark summary into an A8 result packet.",
            "It does not execute the benchmark and does not make summary values trustworthy by itself.",
        ],
    }


def _nested_get(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _number_from_aliases(summary: dict[str, Any], aliases: list[str]) -> float | None:
    for alias in aliases:
        value = _nested_get(summary, alias)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _bool_from_aliases(summary: dict[str, Any], aliases: list[str]) -> bool | None:
    for alias in aliases:
        value = _nested_get(summary, alias)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
    return None


def _coerce_number(value: float, metric_type: str) -> int | float:
    if metric_type == "integer":
        return int(value)
    return round(float(value), 6)


def _metric_fallback_value(metric_spec: MetricSpec, metrics: dict[str, Any]) -> float | None:
    fallback = metric_spec.get("fallback") if isinstance(metric_spec.get("fallback"), dict) else {}
    if fallback.get("op") != "divide":
        return None
    numerator = metrics.get(str(fallback.get("numerator", "")))
    denominator = metrics.get(str(fallback.get("denominator", "")))
    if not isinstance(numerator, int | float) or not isinstance(denominator, int | float):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def validate_adapter_spec(adapter_spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if adapter_spec.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if adapter_spec.get("report_kind") != "a8_benchmark_adapter_spec":
        failures.append("report_kind must be a8_benchmark_adapter_spec")
    if not str(adapter_spec.get("benchmark", "")).strip():
        failures.append("benchmark is required")
    metrics = adapter_spec.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        failures.append("metrics must be a non-empty object")
        return failures
    for metric, raw_metric_spec in metrics.items():
        if not isinstance(metric, str) or not metric.strip():
            failures.append("metric names must be non-empty strings")
            continue
        if not isinstance(raw_metric_spec, dict):
            failures.append(f"metrics.{metric} must be an object")
            continue
        metric_type = str(raw_metric_spec.get("type", "")).strip()
        if metric_type not in {"integer", "number", "rate", "bool"}:
            failures.append(f"metrics.{metric}.type must be integer, number, rate, or bool")
        aliases = raw_metric_spec.get("aliases", [])
        if aliases is not None and (
            not isinstance(aliases, list)
            or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        ):
            failures.append(f"metrics.{metric}.aliases must be a string list")
        fallback = raw_metric_spec.get("fallback")
        if fallback is not None:
            if not isinstance(fallback, dict):
                failures.append(f"metrics.{metric}.fallback must be an object")
            elif fallback.get("op") != "divide":
                failures.append(f"metrics.{metric}.fallback.op must be divide")
    return failures


def build_result_from_adapter_spec(
    summary: dict[str, Any],
    *,
    adapter_spec: dict[str, Any],
    source_path: str = "",
    adapter_spec_path: str = "",
    conservative_comparison_report: bool = False,
) -> dict[str, Any]:
    failures = validate_adapter_spec(adapter_spec)
    if failures:
        raise ValueError("invalid A8 benchmark adapter spec: " + "; ".join(failures))
    benchmark = str(adapter_spec["benchmark"]).strip()
    metric_specs = adapter_spec["metrics"]
    metrics: dict[str, Any] = {}
    pending: list[tuple[str, MetricSpec]] = []
    for metric, raw_metric_spec in metric_specs.items():
        metric_spec = dict(raw_metric_spec)
        metric_type = str(metric_spec.get("type", "")).strip()
        aliases = [str(alias) for alias in metric_spec.get("aliases", [])]
        if metric_type == "bool":
            value = _bool_from_aliases(summary, aliases)
            if value is not None:
                metrics[metric] = value
            continue
        value = _number_from_aliases(summary, aliases)
        if value is None:
            pending.append((metric, metric_spec))
            continue
        metrics[metric] = _coerce_number(value, metric_type)
    for metric, metric_spec in pending:
        metric_type = str(metric_spec.get("type", "")).strip()
        fallback_value = _metric_fallback_value(metric_spec, metrics)
        if fallback_value is not None:
            metrics[metric] = _coerce_number(fallback_value, metric_type)
    for metric, raw_metric_spec in metric_specs.items():
        metric_spec = dict(raw_metric_spec)
        if metric_spec.get("required") is True and metric not in metrics:
            raise ValueError(f"summary must include {metric} for benchmark {benchmark}")
        if metric_spec.get("must_be_true") is True and metrics.get(metric) is not True:
            raise ValueError(f"summary must mark {metric} true for benchmark {benchmark}")
        if metric not in metrics:
            continue
        value = metrics[metric]
        if metric_spec.get("type") == "rate" and not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{metric} must be between zero and one")
        if isinstance(value, int | float):
            minimum = metric_spec.get("minimum")
            if isinstance(minimum, int | float) and float(value) < float(minimum):
                raise ValueError(f"{metric} must be at least {minimum}")
    metrics["conservative_comparison_report"] = bool(conservative_comparison_report)
    return {
        "spec_version": "asi_v1",
        "report_kind": "a8_benchmark_result",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": benchmark,
        "metrics": metrics,
        "source": {
            "source_path": source_path,
            "summary_report_kind": str(summary.get("report_kind", "")).strip(),
            "adapter_spec_path": adapter_spec_path,
            "adapter_spec_report_kind": str(adapter_spec.get("report_kind", "")).strip(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--adapter-spec-json", required=True)
    validate_parser.add_argument("--benchmark", default="")

    args = parser.parse_args()
    if args.mode == "validate":
        adapter_spec_path = Path(args.adapter_spec_json)
        adapter_spec = _read_json(adapter_spec_path)
        failures = validate_adapter_spec(adapter_spec)
        expected_benchmark = str(args.benchmark).strip()
        if expected_benchmark and str(adapter_spec.get("benchmark", "")).strip() != expected_benchmark:
            failures.append(
                "adapter spec benchmark "
                f"{str(adapter_spec.get('benchmark', '')).strip()!r} does not match {expected_benchmark!r}"
            )
        if failures:
            raise SystemExit("A8 benchmark adapter spec validation failed: " + "; ".join(failures))
        print(
            "verified_a8_benchmark_adapter_spec="
            f"{adapter_spec_path} benchmark={str(adapter_spec.get('benchmark', '')).strip()}"
        )


if __name__ == "__main__":
    main()
