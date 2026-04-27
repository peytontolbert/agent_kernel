from __future__ import annotations

from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
from typing import Any


DEFAULT_OUTPUT = Path("web/benchmark_browser/benchmark_index.json")
DEFAULT_TARGET_PACKET = Path("docs/evidence/a8_coding_superhuman_target_packet_20260426.json")
DEFAULT_SOURCE_MANIFEST = Path("config/a8_benchmark_dataset_sources.json")
DEFAULT_SOURCE_STATUS = Path("benchmarks/a8_dataset_sources/status.json")


A8_BENCHMARK_GATES = [
    {
        "benchmark": "codeforces",
        "label": "Codeforces",
        "metric": "rating_equivalent",
        "threshold_key": "codeforces_rating_equivalent",
        "target": "rating equivalent >= 3000",
        "kind": "score_at_least",
    },
    {
        "benchmark": "mle_bench",
        "label": "MLE-Bench",
        "metric": "gold_medal_rate",
        "threshold_key": "mle_bench_gold_medal_rate",
        "target": "gold medal rate >= 20%",
        "kind": "rate_at_least",
    },
    {
        "benchmark": "swe_bench_verified",
        "label": "SWE-Bench Verified",
        "metric": "resolve_rate",
        "threshold_key": "swe_bench_verified_resolve_rate",
        "target": "resolve rate >= 80%",
        "kind": "rate_at_least",
        "dataset_name": "SWE-Bench Verified",
    },
    {
        "benchmark": "swe_rebench",
        "label": "SWE-ReBench",
        "metric": "resolve_rate",
        "threshold_key": "swe_rebench_resolve_rate",
        "target": "resolve rate >= 60%",
        "kind": "rate_at_least",
        "dataset_name": "SWE-ReBench",
    },
    {
        "benchmark": "re_bench",
        "label": "RE-Bench",
        "metric": "human_expert_win_rate",
        "threshold_key": "re_bench_human_expert_win_rate",
        "target": "human expert win rate >= 50%",
        "kind": "rate_at_least",
    },
]

A8_SUPPORT_GATES = [
    {
        "benchmark": "sustained_coding_window",
        "label": "Sustained Coding Window",
        "metric": "task_count",
        "threshold_key": "superhuman_coding_task_count",
        "secondary_threshold_key": "superhuman_coding_window_count",
        "target": ">= 100 tasks across >= 3 windows",
        "kind": "count_at_least",
    },
    {
        "benchmark": "recursive_compounding",
        "label": "Recursive Compounding",
        "metric": "retained_gain_runs",
        "threshold_key": "recursive_compounding_retained_gain_runs",
        "secondary_threshold_key": "recursive_compounding_window_count",
        "target": ">= 5 retained-gain runs across >= 3 windows",
        "kind": "count_at_least",
    },
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("instances", "data", "rows", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"expected dataset list or object with instances/data/rows/tasks at {path}")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return []


def _shorten(value: str, limit: int = 7000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + f"\n...[truncated {len(value) - limit} chars]"


def _repo_from_instance(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    owner, rest = instance_id.split("__", 1)
    repo = rest.split("-", 1)[0]
    return f"{owner}/{repo}" if owner and repo else ""


def _dataset_name(path: Path) -> str:
    parts = path.parts
    if "swe_rebench" in parts:
        return "SWE-ReBench"
    if "swe_bench" in parts:
        return "SWE-Bench Full"
    if "swe_bench_lite_probe" in parts:
        return "SWE-Bench Lite"
    if "swe_bench_verified" in parts and "selected_lite_overlap" in parts:
        return "SWE-Bench Verified Lite Overlap"
    if "swe_bench_verified" in parts:
        return "SWE-Bench Verified"
    return path.stem


def _result_name(path: Path) -> str:
    parent = path.parent.name
    if parent == "evaluation_results":
        return path.parent.parent.name
    return parent.removeprefix("evaluation_results_") or path.stem


def _summary_name(path: Path) -> str:
    if path.name == "summary.json":
        return path.parent.name
    return path.stem.removeprefix("summary_")


def _prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append({"line": line_number, "error": str(exc)})
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _build_dataset(path: Path, root: Path) -> dict[str, Any]:
    records = _read_dataset(path)
    repos: dict[str, int] = {}
    years: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    for item in records:
        instance_id = _text(item.get("instance_id") or item.get("id"))
        repo = _text(item.get("repo")) or _repo_from_instance(instance_id)
        repos[repo or "unknown"] = repos.get(repo or "unknown", 0) + 1
        created_at = _text(item.get("created_at"))
        if len(created_at) >= 4 and created_at[:4].isdigit():
            years[created_at[:4]] = years.get(created_at[:4], 0) + 1
        instances.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "version": _text(item.get("version")),
                "created_at": created_at,
                "base_commit": _text(item.get("base_commit")),
                "problem_statement": _shorten(_text(item.get("problem_statement"))),
                "hints_text": _shorten(_text(item.get("hints_text")), 3000),
                "fail_to_pass": _json_list(item.get("FAIL_TO_PASS")),
                "pass_to_pass_count": len(_json_list(item.get("PASS_TO_PASS"))),
                "has_reference_patch": bool(_text(item.get("patch"))),
                "has_test_patch": bool(_text(item.get("test_patch"))),
            }
        )
    return {
        "name": _dataset_name(path),
        "path": str(path.relative_to(root)),
        "task_count": len(instances),
        "repo_counts": dict(sorted(repos.items())),
        "year_counts": dict(sorted(years.items())),
        "instances": instances,
    }


def _build_result(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    total = int(payload.get("total_instances") or payload.get("task_count") or 0)
    resolved_ids = [str(item) for item in payload.get("resolved_ids", []) if isinstance(item, str)]
    completed_ids = [str(item) for item in payload.get("completed_ids", []) if isinstance(item, str)]
    unresolved_ids = [str(item) for item in payload.get("unresolved_ids", []) if isinstance(item, str)]
    error_ids = [str(item) for item in payload.get("error_ids", []) if isinstance(item, str)]
    incomplete_ids = [str(item) for item in payload.get("incomplete_ids", []) if isinstance(item, str)]
    return {
        "name": _result_name(path),
        "path": str(path.relative_to(root)),
        "total": total,
        "completed": len(completed_ids),
        "resolved": len(resolved_ids),
        "unresolved": len(unresolved_ids),
        "errors": len(error_ids),
        "incomplete": len(incomplete_ids),
        "resolve_rate": (len(resolved_ids) / total) if total else 0.0,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "incomplete_ids": incomplete_ids,
    }


def _build_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return {
        "name": _summary_name(path),
        "path": str(path.relative_to(root)),
        "report_kind": _text(payload.get("report_kind")),
        "created_at": _text(payload.get("created_at")),
        "metrics": metrics,
        "task_count": metrics.get("task_count"),
        "resolved_count": metrics.get("resolved_count"),
        "resolve_rate": metrics.get("resolve_rate"),
        "benchmark": _text(payload.get("benchmark")),
        "repo_slices": payload.get("repo_slices", []),
    }


def _build_prediction_file(path: Path, root: Path) -> dict[str, Any]:
    records = _prediction_records(path)
    repos: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    for record in records:
        instance_id = _text(record.get("instance_id"))
        repo = _repo_from_instance(instance_id)
        repos[repo or "unknown"] = repos.get(repo or "unknown", 0) + 1
        patch = _text(record.get("model_patch") or record.get("patch"))
        instances.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "model_name_or_path": _text(record.get("model_name_or_path")),
                "patch_chars": len(patch),
                "has_patch": bool(patch.strip()),
            }
        )
    return {
        "name": path.stem,
        "path": str(path.relative_to(root)),
        "prediction_count": len(records),
        "repo_counts": dict(sorted(repos.items())),
        "instances": instances,
    }


def _build_run_spec(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    runner = payload.get("runner") if isinstance(payload.get("runner"), dict) else {}
    adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    return {
        "name": path.stem,
        "path": str(path.relative_to(root)),
        "benchmark": _text(payload.get("benchmark")),
        "ready_to_run": bool(payload.get("ready_to_run")),
        "runner_kind": _text(runner.get("kind")),
        "dataset_name": _text(runner.get("dataset_name")),
        "predictions_path": _text(runner.get("predictions_path")),
        "results_json": _text(runner.get("results_json")),
        "summary_json": _text(adapter.get("summary_json")),
        "open_limits": [str(item) for item in payload.get("open_limits", [])],
    }


def _build_targets(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = _read_json_object(path)
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return {
        "path": str(path),
        "thresholds": target.get("thresholds", {}),
        "benchmark_sources": target.get("benchmark_sources", {}),
        "acceptance_policy": target.get("acceptance_policy", {}),
        "current_status": payload.get("current_status", {}),
    }


def _build_dataset_sources(root: Path, source_manifest: Path | None, source_status: Path | None) -> dict[str, Any]:
    manifest_payload: dict[str, Any] = {}
    status_payload: dict[str, Any] = {}
    if source_manifest and source_manifest.exists():
        manifest_payload = _read_json_object(source_manifest)
    if source_status and source_status.exists():
        status_payload = _read_json_object(source_status)
    status_by_benchmark = {
        str(item.get("benchmark", "")): item
        for item in status_payload.get("sources", [])
        if isinstance(item, dict)
    }
    sources: list[dict[str, Any]] = []
    for item in manifest_payload.get("sources", []):
        if not isinstance(item, dict):
            continue
        benchmark = str(item.get("benchmark", ""))
        status = status_by_benchmark.get(benchmark, {})
        local_path = Path(str(item.get("local_path", "")).strip())
        target = local_path if local_path.is_absolute() else root / local_path
        exists = bool(status.get("exists", target.exists()))
        sources.append(
            {
                **item,
                "local_path": str(target.relative_to(root)) if target.is_relative_to(root) else str(target),
                "status": status.get("status", "available" if exists else "missing"),
                "exists": exists,
                "rows": status.get("rows"),
                "size_bytes": status.get("size_bytes", target.stat().st_size if target.exists() and target.is_file() else 0),
                "last_action": status.get("last_action", {}),
                "error": status.get("error", ""),
            }
        )
    return {
        "source_manifest": str(source_manifest.relative_to(root)) if source_manifest and source_manifest.exists() and source_manifest.is_relative_to(root) else str(source_manifest or ""),
        "status_path": str(source_status.relative_to(root)) if source_status and source_status.exists() and source_status.is_relative_to(root) else str(source_status or ""),
        "generated_at": status_payload.get("generated_at", ""),
        "sources": sources,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _summary_metric(summary: dict[str, Any], metric: str) -> float | None:
    value = _number(summary.get(metric))
    if value is not None:
        return value
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    value = _number(metrics.get(metric))
    if value is not None:
        return value
    if metric == "resolve_rate":
        resolved = _number(summary.get("resolved_count"))
        total = _number(summary.get("task_count"))
        if resolved is not None and total:
            return resolved / total
    return None


def _best_summary_for_benchmark(summaries: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    candidates = [
        summary
        for summary in summaries
        if summary.get("benchmark") == benchmark or benchmark in str(summary.get("path", ""))
    ]
    if benchmark == "swe_bench_verified":
        candidates.extend(
            summary
            for summary in summaries
            if "swe_bench_verified" in str(summary.get("path", "")) or summary.get("name") == "selected_lite_overlap"
        )
    if not candidates:
        return None
    if benchmark == "swe_bench_verified":
        return sorted(
            candidates,
            key=lambda item: (
                int(_number(item.get("task_count")) or 0),
                str(item.get("created_at") or item.get("path") or ""),
            ),
            reverse=True,
        )[0]
    return sorted(candidates, key=lambda item: str(item.get("created_at") or item.get("path") or ""), reverse=True)[0]


def _dataset_count(datasets: list[dict[str, Any]], dataset_name: str | None) -> int | None:
    if not dataset_name:
        return None
    for dataset in datasets:
        if dataset.get("name") == dataset_name:
            return int(dataset.get("task_count") or 0)
    return None


def _run_spec_for_benchmark(run_specs: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    candidates = [spec for spec in run_specs if spec.get("benchmark") == benchmark]
    config_candidates = [spec for spec in candidates if str(spec.get("path", "")).startswith("config/a8_benchmark_run_specs/")]
    for spec in config_candidates or candidates:
        if spec.get("benchmark") == benchmark:
            return spec
    return None


def _required_count(threshold: float | None, dataset_total: int | None) -> int | None:
    if threshold is None or dataset_total is None:
        return None
    return int(-(-threshold * dataset_total // 1))


def _gate_status(current: float | None, threshold: float | None, evidence: dict[str, Any] | None) -> str:
    if current is None:
        return "no_evidence"
    if threshold is None:
        return "tracked"
    if current >= threshold:
        return "met"
    return "partial" if evidence else "no_evidence"


def _count_aware_gate_status(
    *,
    current: float | None,
    threshold: float | None,
    evidence: dict[str, Any] | None,
    numerator: float | None,
    required: int | None,
) -> str:
    status = _gate_status(current, threshold, evidence)
    if status != "met" or required is None:
        return status
    if numerator is None or numerator < required:
        return "partial"
    return status


def _build_gate_progress(
    *,
    gate: dict[str, Any],
    thresholds: dict[str, Any],
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    support_gate: bool = False,
) -> dict[str, Any]:
    benchmark = str(gate["benchmark"])
    summary = _best_summary_for_benchmark(summaries, benchmark)
    threshold = _number(thresholds.get(str(gate["threshold_key"])))
    dataset_total = _dataset_count(datasets, gate.get("dataset_name"))
    required = _required_count(threshold, dataset_total)
    current = _summary_metric(summary, str(gate["metric"])) if summary else None
    numerator = _number(summary.get("resolved_count")) if summary else None
    denominator = _number(summary.get("task_count")) if summary else None
    if gate["kind"] == "count_at_least" and numerator is None:
        numerator = current
    if gate["kind"] == "score_at_least":
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    elif gate["kind"] == "count_at_least":
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    else:
        progress_to_gate = (current / threshold) if current is not None and threshold else 0.0
    run_spec = _run_spec_for_benchmark(run_specs, benchmark)
    caveats: list[str] = []
    if benchmark == "swe_bench_verified" and summary and denominator and dataset_total and denominator < dataset_total:
        caveats.append(
            f"Current evidence covers a {int(denominator)} task slice, not the full {dataset_total} task benchmark."
        )
    if not summary:
        caveats.append("No local evidence summary has been produced for this gate.")
    if run_spec and not run_spec.get("ready_to_run"):
        caveats.extend(str(item) for item in run_spec.get("open_limits", []))
    return {
        "benchmark": benchmark,
        "label": gate["label"],
        "metric": gate["metric"],
        "target": gate["target"],
        "threshold": threshold,
        "secondary_threshold": _number(thresholds.get(str(gate.get("secondary_threshold_key")))),
        "dataset_total": dataset_total,
        "required_count": required,
        "current_value": current,
        "current_numerator": numerator,
        "current_denominator": denominator,
        "progress_to_gate": min(max(progress_to_gate, 0.0), 1.0),
        "status": _count_aware_gate_status(
            current=current,
            threshold=threshold,
            evidence=summary,
            numerator=numerator,
            required=required,
        ),
        "support_gate": support_gate,
        "evidence_path": summary.get("path") if summary else "",
        "run_spec_path": run_spec.get("path") if run_spec else "",
        "ready_to_run": bool(run_spec.get("ready_to_run")) if run_spec else False,
        "caveats": caveats,
    }


def _build_a8_progress(
    targets: dict[str, Any],
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = targets.get("thresholds") if isinstance(targets.get("thresholds"), dict) else {}
    benchmark_gates = [
        _build_gate_progress(
            gate=gate,
            thresholds=thresholds,
            summaries=summaries,
            datasets=datasets,
            run_specs=run_specs,
        )
        for gate in A8_BENCHMARK_GATES
    ]
    support_gates = [
        _build_gate_progress(
            gate=gate,
            thresholds=thresholds,
            summaries=summaries,
            datasets=datasets,
            run_specs=run_specs,
            support_gate=True,
        )
        for gate in A8_SUPPORT_GATES
    ]
    all_gates = [*benchmark_gates, *support_gates]
    met_count = sum(1 for gate in all_gates if gate["status"] == "met")
    return {
        "level": "A8",
        "domain": "coding",
        "claim_ready": all(gate["status"] == "met" for gate in all_gates),
        "met_gate_count": met_count,
        "gate_count": len(all_gates),
        "benchmark_gates": benchmark_gates,
        "support_gates": support_gates,
        "acceptance_policy": targets.get("acceptance_policy", {}),
        "current_status": targets.get("current_status", {}),
    }


def build_benchmark_browser_index(
    root: Path,
    target_packet: Path | None = DEFAULT_TARGET_PACKET,
    source_manifest: Path | None = DEFAULT_SOURCE_MANIFEST,
    source_status: Path | None = DEFAULT_SOURCE_STATUS,
) -> dict[str, Any]:
    datasets = [
        _build_dataset(path, root)
        for path in sorted(
            [
                *root.glob("benchmarks/**/swe_bench*_test_dataset.json"),
                *root.glob("benchmarks/**/swe_rebench*_dataset.json"),
            ]
        )
        if path.is_file()
    ]
    selected = root / "benchmarks/swe_bench_verified/selected_lite_overlap/dataset_selected_lite_overlap.json"
    if selected.exists():
        datasets.append(_build_dataset(selected, root))

    results = [
        _build_result(path, root)
        for path in sorted(root.glob("benchmarks/**/results.json"))
        if path.is_file()
    ]
    summaries = [
        _build_summary(path, root)
        for path in sorted(root.glob("benchmarks/**/summary*.json"))
        if path.is_file()
    ]
    predictions = [
        _build_prediction_file(path, root)
        for path in sorted(root.glob("benchmarks/**/*.jsonl"))
        if "repo_cache" not in path.parts
    ]
    run_specs = [
        _build_run_spec(path, root)
        for path in sorted([*root.glob("benchmarks/**/*run_spec.json"), *root.glob("config/a8_benchmark_run_specs/*.json")])
        if path.is_file()
    ]
    targets = _build_targets(target_packet if target_packet and target_packet.is_absolute() else root / target_packet if target_packet else None)
    resolved_source_manifest = (
        source_manifest if source_manifest and source_manifest.is_absolute() else root / source_manifest if source_manifest else None
    )
    resolved_source_status = source_status if source_status and source_status.is_absolute() else root / source_status if source_status else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "targets": targets,
        "a8_progress": _build_a8_progress(targets, summaries, datasets, run_specs),
        "dataset_sources": _build_dataset_sources(root, resolved_source_manifest, resolved_source_status),
        "datasets": datasets,
        "results": results,
        "summaries": summaries,
        "predictions": predictions,
        "run_specs": run_specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-packet", default=str(DEFAULT_TARGET_PACKET))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--source-status", default=str(DEFAULT_SOURCE_STATUS))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    target_packet = Path(args.target_packet)
    index = build_benchmark_browser_index(root, target_packet, Path(args.source_manifest), Path(args.source_status))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"datasets={len(index['datasets'])} results={len(index['results'])} "
        f"predictions={len(index['predictions'])} run_specs={len(index['run_specs'])} output={output}"
    )


if __name__ == "__main__":
    main()
