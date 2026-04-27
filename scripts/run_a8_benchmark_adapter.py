from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
from typing import Any

from adapt_a8_benchmark_result import BENCHMARK_FIELD_MAP, build_a8_benchmark_result
from adapt_swe_resolve_benchmark import SUPPORTED_BENCHMARKS as SWE_BENCHMARKS
from adapt_swe_resolve_benchmark import build_swe_resolve_benchmark_result
from a8_benchmark_adapter_specs import build_result_from_adapter_spec
from export_autonomy_evidence import verify_a8_benchmark_result_packet


SUPPORTED_BENCHMARKS = tuple(sorted(set(BENCHMARK_FIELD_MAP) | set(SWE_BENCHMARKS)))


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict | list):
        raise SystemExit(f"expected JSON object or array at {path}")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _read_command(path: Path) -> list[str]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"runner command must be a non-empty JSON array at {path}")
    command = [str(part) for part in payload]
    if not command[0].strip():
        raise SystemExit("runner command executable is empty")
    return command


def run_benchmark_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd) if cwd is not None else "",
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_result_for_benchmark(
    summary: dict[str, Any],
    *,
    benchmark: str,
    source_path: str,
    conservative_comparison_report: bool,
    adapter_spec: dict[str, Any] | None = None,
    adapter_spec_path: str = "",
) -> dict[str, Any]:
    if adapter_spec is not None:
        spec_benchmark = str(adapter_spec.get("benchmark", "")).strip()
        if spec_benchmark != benchmark:
            raise ValueError(f"adapter spec benchmark {spec_benchmark!r} does not match {benchmark!r}")
        return build_result_from_adapter_spec(
            summary,
            adapter_spec=adapter_spec,
            source_path=source_path,
            adapter_spec_path=adapter_spec_path,
            conservative_comparison_report=conservative_comparison_report,
        )
    if benchmark in SWE_BENCHMARKS:
        return build_swe_resolve_benchmark_result(
            summary,
            benchmark=benchmark,
            source_path=source_path,
            conservative_comparison_report=conservative_comparison_report,
        )
    return build_a8_benchmark_result(
        summary,
        benchmark=benchmark,
        source_path=source_path,
        conservative_comparison_report=conservative_comparison_report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--adapter-spec-json", default=None)
    parser.add_argument("--runner-command-json", default=None)
    parser.add_argument("--runner-cwd", default=None)
    parser.add_argument("--runner-log-json", default=None)
    parser.add_argument("--conservative-comparison-report", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and not str(args.output_json).strip():
        raise SystemExit("--output-json is required unless --validate-only is set")

    runner_result: dict[str, Any] | None = None
    if args.runner_command_json:
        runner_cwd = Path(args.runner_cwd).resolve() if args.runner_cwd else None
        runner_result = run_benchmark_command(_read_command(Path(args.runner_command_json)), cwd=runner_cwd)
        if args.runner_log_json:
            log_path = Path(args.runner_log_json)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(runner_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if int(runner_result["returncode"]) != 0:
            raise SystemExit(
                "benchmark runner failed "
                f"returncode={runner_result['returncode']} "
                f"stderr={runner_result['stderr']}"
            )

    summary_path = Path(args.summary_json)
    adapter_spec_path = Path(args.adapter_spec_json) if args.adapter_spec_json else None
    packet = build_result_for_benchmark(
        _read_json_object(summary_path),
        benchmark=args.benchmark,
        source_path=str(summary_path),
        conservative_comparison_report=bool(args.conservative_comparison_report),
        adapter_spec=_read_json_object(adapter_spec_path) if adapter_spec_path is not None else None,
        adapter_spec_path=str(adapter_spec_path) if adapter_spec_path is not None else "",
    )
    if runner_result is not None:
        packet["source"]["runner"] = {
            "command": runner_result["command"],
            "cwd": runner_result["cwd"],
            "returncode": runner_result["returncode"],
            "runner_log_json": str(args.runner_log_json or ""),
        }
    failures = verify_a8_benchmark_result_packet(packet)
    if failures:
        raise SystemExit("adapted A8 benchmark result failed verification: " + "; ".join(failures))
    if args.validate_only:
        print(
            f"benchmark={packet['benchmark']} "
            f"status=validated "
            f"summary_json={summary_path}"
        )
        return
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"benchmark={packet['benchmark']} "
        f"status=verified "
        f"output_json={output_path}"
    )


if __name__ == "__main__":
    main()
