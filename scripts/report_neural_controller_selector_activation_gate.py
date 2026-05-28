from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_neural_controller_rowwise_frontier import _read_json_object


def report_selector_activation_gate(
    *,
    retained_candidate_packet_path: Path,
    runtime_contract_metrics_path: Path,
    output_path: Path,
    min_runtime_contract_steps: int = 1,
    min_runtime_contract_tasks: int = 1,
    production_min_runtime_contract_steps: int = 25,
    production_min_runtime_contract_tasks: int = 5,
) -> dict[str, Any]:
    packet = _read_json_object(retained_candidate_packet_path)
    metrics_report = _read_json_object(runtime_contract_metrics_path)
    summary = metrics_report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    runtime_contract_steps = int(summary.get("runtime_contract_steps", 0) or 0)
    runtime_contract_tasks = int(summary.get("runtime_contract_task_count", 0) or 0)
    guarded_comparison_steps = int(summary.get("guarded_comparison_steps", 0) or 0)
    guarded_baseline_observed_steps = int(summary.get("guarded_baseline_observed_steps", 0) or 0)
    guarded_candidate_observed_steps = int(summary.get("guarded_candidate_observed_steps", 0) or 0)
    guarded_candidate_dry_run_attempts = int(summary.get("guarded_candidate_dry_run_attempts", 0) or 0)
    guarded_candidate_dry_run_successes = int(summary.get("guarded_candidate_dry_run_successes", 0) or 0)
    guarded_selected_dry_run_attempts = int(summary.get("guarded_selected_dry_run_attempts", 0) or 0)
    guarded_dry_run_switches_applied = int(summary.get("guarded_dry_run_switches_applied", 0) or 0)
    selector_signal_ready = bool(summary.get("selector_signal_ready", False))
    retained_candidate_ready = bool(packet.get("retained_candidate_ready", False))
    blockers: list[str] = []
    if not retained_candidate_ready:
        blockers.append("retained_selector_candidate_not_ready")
    if not selector_signal_ready:
        blockers.append("runtime_selector_signal_not_ready")
    if runtime_contract_steps < min_runtime_contract_steps:
        blockers.append("runtime_contract_steps_below_gate")
    if runtime_contract_tasks < min_runtime_contract_tasks:
        blockers.append("runtime_contract_tasks_below_gate")
    activation_ready = not blockers
    production_blockers: list[str] = []
    if not activation_ready:
        production_blockers.extend(blockers)
    if runtime_contract_steps < production_min_runtime_contract_steps:
        production_blockers.append("runtime_contract_steps_below_production_gate")
    if runtime_contract_tasks < production_min_runtime_contract_tasks:
        production_blockers.append("runtime_contract_tasks_below_production_gate")
    production_activation_ready = not production_blockers
    report = {
        "report_kind": "neural_controller_selector_activation_gate",
        "retained_candidate_packet_path": str(retained_candidate_packet_path),
        "runtime_contract_metrics_path": str(runtime_contract_metrics_path),
        "min_runtime_contract_steps": int(min_runtime_contract_steps),
        "min_runtime_contract_tasks": int(min_runtime_contract_tasks),
        "production_min_runtime_contract_steps": int(production_min_runtime_contract_steps),
        "production_min_runtime_contract_tasks": int(production_min_runtime_contract_tasks),
        "guarded_selector_activation_ready": activation_ready,
        "production_guarded_selector_activation_ready": production_activation_ready,
        "primary_authority_ready": False,
        "recommended_runtime_mode": "guarded" if activation_ready else "shadow",
        "blockers": blockers,
        "production_blockers": production_blockers,
        "selector_policy": str(packet.get("selector_policy", "")).strip(),
        "retained_candidate_ready": retained_candidate_ready,
        "runtime_contract_steps": runtime_contract_steps,
        "runtime_contract_task_count": runtime_contract_tasks,
        "runtime_contract_success_steps": int(summary.get("runtime_contract_success_steps", 0) or 0),
        "runtime_contract_success_rate": float(summary.get("runtime_contract_success_rate", 0.0) or 0.0),
        "guarded_comparison_steps": guarded_comparison_steps,
        "guarded_baseline_observed_steps": guarded_baseline_observed_steps,
        "guarded_candidate_observed_steps": guarded_candidate_observed_steps,
        "guarded_candidate_dry_run_attempts": guarded_candidate_dry_run_attempts,
        "guarded_candidate_dry_run_successes": guarded_candidate_dry_run_successes,
        "guarded_candidate_dry_run_success_rate": float(
            summary.get("guarded_candidate_dry_run_success_rate", 0.0) or 0.0
        ),
        "guarded_selected_dry_run_attempts": guarded_selected_dry_run_attempts,
        "guarded_selected_dry_run_success_rate": float(
            summary.get("guarded_selected_dry_run_success_rate", 0.0) or 0.0
        ),
        "guarded_dry_run_switches_applied": guarded_dry_run_switches_applied,
        "selector_signal_ready": selector_signal_ready,
        "authority_note": (
            "This gate authorizes guarded selector activation only. Primary authority remains blocked "
            "until a retained promotion gate explicitly approves primary mode."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-candidate-packet", required=True)
    parser.add_argument("--runtime-contract-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-runtime-contract-steps", type=int, default=1)
    parser.add_argument("--min-runtime-contract-tasks", type=int, default=1)
    parser.add_argument("--production-min-runtime-contract-steps", type=int, default=25)
    parser.add_argument("--production-min-runtime-contract-tasks", type=int, default=5)
    args = parser.parse_args()
    report = report_selector_activation_gate(
        retained_candidate_packet_path=Path(args.retained_candidate_packet),
        runtime_contract_metrics_path=Path(args.runtime_contract_metrics),
        output_path=Path(args.output),
        min_runtime_contract_steps=int(args.min_runtime_contract_steps),
        min_runtime_contract_tasks=int(args.min_runtime_contract_tasks),
        production_min_runtime_contract_steps=int(args.production_min_runtime_contract_steps),
        production_min_runtime_contract_tasks=int(args.production_min_runtime_contract_tasks),
    )
    print(
        "neural_controller_selector_activation_gate "
        f"guarded_ready={str(report['guarded_selector_activation_ready']).lower()} "
        f"production_guarded_ready={str(report['production_guarded_selector_activation_ready']).lower()} "
        f"mode={report['recommended_runtime_mode']} "
        f"runtime_contract_steps={report['runtime_contract_steps']} "
        f"runtime_contract_tasks={report['runtime_contract_task_count']} "
        f"primary_authority_ready={str(report['primary_authority_ready']).lower()} "
        f"blockers={','.join(report['blockers']) or 'none'}"
    )


if __name__ == "__main__":
    main()
