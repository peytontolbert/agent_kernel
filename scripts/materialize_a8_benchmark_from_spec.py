from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

from prepare_swe_bench_a8_run import summarize_swe_bench_results
from run_a8_benchmark_adapter import build_result_for_benchmark
from validate_a8_benchmark_specs import validate_a8_benchmark_spec
from export_autonomy_evidence import verify_a8_benchmark_result_packet


SWE_BENCHMARKS = {"swe_bench_verified", "swe_rebench", "swe_bench_live"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adapter_payload(spec: dict[str, Any]) -> dict[str, Any]:
    adapter = spec.get("adapter") if isinstance(spec.get("adapter"), dict) else {}
    return adapter


def _runner_payload(spec: dict[str, Any]) -> dict[str, Any]:
    runner = spec.get("runner") if isinstance(spec.get("runner"), dict) else {}
    return runner


def materialize_a8_benchmark_from_spec(
    spec: dict[str, Any],
    *,
    spec_path: str = "",
    allow_not_ready: bool = False,
) -> dict[str, Any]:
    failures = validate_a8_benchmark_spec(spec, spec_path=spec_path)
    if failures:
        raise ValueError("invalid A8 benchmark spec: " + "; ".join(failures))
    if spec.get("ready_to_run") is not True and not allow_not_ready:
        raise ValueError("spec is not ready_to_run; pass allow_not_ready only for existing summary materialization")

    benchmark = str(spec["benchmark"]).strip()
    adapter = _adapter_payload(spec)
    runner = _runner_payload(spec)
    summary_path = Path(str(adapter["summary_json"]))
    if benchmark in SWE_BENCHMARKS:
        results_json = str(runner.get("results_json", "")).strip()
        if results_json:
            summary = summarize_swe_bench_results(_read_json(Path(results_json)), source_path=results_json)
            _write_json(summary_path, summary)
        elif not summary_path.exists():
            raise ValueError(
                "SWE benchmark materialization requires runner.results_json or an existing adapter.summary_json"
            )
    elif not summary_path.exists():
        raise ValueError(f"adapter.summary_json does not exist: {summary_path}")

    adapter_spec_path = str(adapter.get("adapter_spec_json", "")).strip()
    packet = build_result_for_benchmark(
        _read_json(summary_path),
        benchmark=benchmark,
        source_path=str(summary_path),
        conservative_comparison_report=bool(adapter.get("conservative_comparison_report", False)),
        adapter_spec=_read_json(Path(adapter_spec_path)) if adapter_spec_path else None,
        adapter_spec_path=adapter_spec_path,
    )
    packet["source"]["benchmark_run_spec_path"] = spec_path
    failures = verify_a8_benchmark_result_packet(packet)
    if failures:
        raise ValueError("materialized A8 benchmark result failed verification: " + "; ".join(failures))
    output_path = Path(str(adapter["output_packet_json"]))
    packet["source"]["output_packet_json"] = str(output_path)
    _write_json(output_path, packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args()

    spec_path = Path(args.spec_json)
    try:
        packet = materialize_a8_benchmark_from_spec(
            _read_json(spec_path),
            spec_path=str(spec_path),
            allow_not_ready=bool(args.allow_not_ready),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"benchmark={packet['benchmark']} "
        f"status=verified "
        f"output_json={packet['source'].get('output_packet_json', '')}"
    )


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
