from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_swe_bench_predictions import _read_jsonl, validate_swe_predictions


SUPPORTED_BENCHMARKS = {
    "codeforces",
    "mle_bench",
    "swe_bench_verified",
    "swe_rebench",
    "re_bench",
    "sustained_coding_window",
    "recursive_compounding",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def validate_a8_benchmark_spec(spec: dict[str, Any], *, spec_path: str = "") -> list[str]:
    failures: list[str] = []
    if spec.get("spec_version") != "asi_v1":
        failures.append("spec_version must be asi_v1")
    if spec.get("report_kind") != "a8_benchmark_run_spec":
        failures.append("report_kind must be a8_benchmark_run_spec")
    benchmark = str(spec.get("benchmark", "")).strip()
    if benchmark not in SUPPORTED_BENCHMARKS:
        failures.append("benchmark must be one of " + ",".join(sorted(SUPPORTED_BENCHMARKS)))
    adapter = spec.get("adapter") if isinstance(spec.get("adapter"), dict) else {}
    if not str(adapter.get("script", "")).strip():
        failures.append("adapter.script is required")
    if not str(adapter.get("summary_json", "")).strip():
        failures.append("adapter.summary_json is required")
    if not str(adapter.get("output_packet_json", "")).strip():
        failures.append("adapter.output_packet_json is required")
    if adapter.get("conservative_comparison_report") is not True:
        failures.append("adapter.conservative_comparison_report must be true for A8 evidence")
    ready_to_run = spec.get("ready_to_run")
    if not isinstance(ready_to_run, bool):
        failures.append("ready_to_run must be boolean")
        ready_to_run = False
    prerequisites = spec.get("prerequisites", [])
    if prerequisites is not None and not isinstance(prerequisites, list):
        failures.append("prerequisites must be a list when provided")
        prerequisites = []
    if benchmark == "codeforces":
        account_prerequisites = [
            item for item in prerequisites if isinstance(item, dict) and item.get("kind") == "account"
        ]
        if not account_prerequisites:
            failures.append("codeforces specs must declare an account prerequisite")
        for item in account_prerequisites:
            required_env = item.get("required_env", [])
            proof_path = str(item.get("proof_path", "")).strip()
            if not isinstance(required_env, list) or any(not isinstance(value, str) or not value for value in required_env):
                failures.append("codeforces account prerequisite required_env must be a string list")
                required_env = []
            if not required_env and not proof_path:
                failures.append("codeforces account prerequisite must declare required_env or proof_path")
            if ready_to_run:
                env_ok = bool(required_env) and all(str(os.environ.get(value, "")).strip() for value in required_env)
                proof_ok = bool(proof_path) and Path(proof_path).exists()
                if not env_ok and not proof_ok:
                    failures.append("codeforces ready_to_run requires CODEFORCES_HANDLE or account proof_path")

    runner = spec.get("runner") if isinstance(spec.get("runner"), dict) else {}
    if benchmark in {"swe_bench_verified", "swe_rebench"}:
        if str(runner.get("kind", "")).strip() != "swebench_harness":
            failures.append("runner.kind must be swebench_harness for SWE benchmark specs")
        dataset_name = str(runner.get("dataset_name", "")).strip()
        if not dataset_name and (ready_to_run or benchmark != "swe_rebench"):
            failures.append("runner.dataset_name is required for SWE benchmark specs")
        if benchmark == "swe_rebench" and dataset_name.lower() in {"swe-rebench", "swe_rebench", "template"}:
            failures.append("runner.dataset_name must be the confirmed official SWE-ReBench dataset identifier")
        harness_root = str(runner.get("harness_root", "")).strip()
        if not harness_root:
            failures.append("runner.harness_root is required for SWE benchmark specs")
        elif not Path(harness_root).exists():
            failures.append(f"runner.harness_root does not exist: {harness_root}")
        predictions_path = str(runner.get("predictions_path", "")).strip()
        if not predictions_path:
            failures.append("runner.predictions_path is required for SWE benchmark specs")
        elif ready_to_run and not Path(predictions_path).exists():
            failures.append(f"runner.predictions_path does not exist: {predictions_path}")
        elif ready_to_run:
            prediction_failures = validate_swe_predictions(_read_jsonl(Path(predictions_path)))
            failures.extend(f"runner.predictions_path invalid: {failure}" for failure in prediction_failures)
        if not str(runner.get("run_id", "")).strip():
            failures.append("runner.run_id is required for SWE benchmark specs")
    else:
        if str(runner.get("kind", "")).strip() not in {"summary_only", "external_harness"}:
            failures.append("runner.kind must be summary_only or external_harness")
        summary_source = str(runner.get("summary_source", "")).strip()
        if ready_to_run and summary_source and not Path(summary_source).exists():
            failures.append(f"runner.summary_source does not exist: {summary_source}")
        if ready_to_run and not summary_source and str(runner.get("kind", "")).strip() == "summary_only":
            failures.append("runner.summary_source is required when ready_to_run=true")

    expected_open_limits = spec.get("open_limits")
    if not isinstance(expected_open_limits, list) or not expected_open_limits:
        failures.append("open_limits must be a non-empty list")
    if spec_path and not str(spec.get("spec_path", spec_path)).strip():
        failures.append("spec_path must be non-empty")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args()
    all_failures: list[str] = []
    for raw_path in args.specs:
        path = Path(raw_path)
        failures = validate_a8_benchmark_spec(_read_json(path), spec_path=str(path))
        if failures:
            all_failures.extend(f"{path}: {failure}" for failure in failures)
        else:
            print(f"verified_a8_benchmark_spec={path}")
    if all_failures:
        raise SystemExit("A8 benchmark spec validation failed: " + "; ".join(all_failures))


if __name__ == "__main__":
    main()
