from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.config import KernelConfig
from agent_kernel.ops.episode_store import iter_episode_documents
from scripts.report_neural_controller_shadow_metrics import _document_shadow_steps, _iter_report_documents


def summarize_runtime_contract_metrics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    shadow_steps = 0
    runtime_contract_steps = 0
    runtime_contract_success_steps = 0
    runtime_contract_task_ids: set[str] = set()
    guarded_comparison_steps = 0
    guarded_candidate_observed_steps = 0
    guarded_baseline_observed_steps = 0
    guarded_candidate_dry_run_attempts = 0
    guarded_candidate_dry_run_successes = 0
    guarded_candidate_dry_run_skips = 0
    guarded_candidate_dry_run_mode_counts: Counter[str] = Counter()
    guarded_selected_dry_run_attempts = 0
    guarded_selected_dry_run_successes = 0
    guarded_dry_run_switches_applied = 0
    mode_counts: Counter[str] = Counter()
    selector_source_counts: Counter[str] = Counter()
    selector_policy_counts: Counter[str] = Counter()
    for document in documents:
        task_id = str(document.get("task_id", "")).strip()
        for step in _document_shadow_steps(document):
            metadata = step.get("proposal_metadata", {})
            if not isinstance(metadata, dict):
                continue
            shadow = metadata.get("neural_controller_shadow", {})
            if not isinstance(shadow, dict) or not shadow:
                continue
            shadow_steps += 1
            if isinstance(shadow.get("guarded_candidate_prediction", {}), dict) and shadow.get(
                "guarded_candidate_prediction"
            ):
                guarded_candidate_observed_steps += 1
            if isinstance(shadow.get("guarded_baseline_prediction", {}), dict) and shadow.get(
                "guarded_baseline_prediction"
            ):
                guarded_baseline_observed_steps += 1
            if (
                isinstance(shadow.get("guarded_candidate_prediction", {}), dict)
                and shadow.get("guarded_candidate_prediction")
                and isinstance(shadow.get("guarded_baseline_prediction", {}), dict)
                and shadow.get("guarded_baseline_prediction")
            ):
                guarded_comparison_steps += 1
            dry_run = metadata.get("neural_controller_guarded_candidate_dry_run", {})
            if isinstance(dry_run, dict) and dry_run:
                if bool(dry_run.get("skipped", False)):
                    guarded_candidate_dry_run_skips += 1
                if bool(dry_run.get("attempted", False)):
                    guarded_candidate_dry_run_attempts += 1
                    if bool(dry_run.get("candidate_verification_passed", False)):
                        guarded_candidate_dry_run_successes += 1
                    mode = str(dry_run.get("candidate_artifact_failure_mode", "")).strip()
                    guarded_candidate_dry_run_mode_counts[mode or "unknown"] += 1
            selected_dry_run = metadata.get("neural_controller_guarded_selected_dry_run", {})
            if isinstance(selected_dry_run, dict) and selected_dry_run:
                if bool(selected_dry_run.get("attempted", False)):
                    guarded_selected_dry_run_attempts += 1
                    if bool(selected_dry_run.get("selected_verification_passed", False)):
                        guarded_selected_dry_run_successes += 1
            switch = metadata.get("neural_controller_guarded_dry_run_switch", {})
            if isinstance(switch, dict) and bool(switch.get("applied", False)):
                guarded_dry_run_switches_applied += 1
            mode = str(shadow.get("runtime_artifact_failure_mode", "")).strip()
            if "runtime_contract_success" not in shadow and not mode:
                continue
            runtime_contract_steps += 1
            if task_id:
                runtime_contract_task_ids.add(task_id)
            if bool(shadow.get("runtime_contract_success", False)):
                runtime_contract_success_steps += 1
            mode_counts[mode or "unknown"] += 1
            source = str(shadow.get("rowwise_selector_source", "")).strip()
            if source:
                selector_source_counts[source] += 1
            policy = str(shadow.get("rowwise_selector_policy", "")).strip()
            if policy:
                selector_policy_counts[policy] += 1
    coverage_rate = runtime_contract_steps / shadow_steps if shadow_steps else 0.0
    success_rate = runtime_contract_success_steps / runtime_contract_steps if runtime_contract_steps else 0.0
    dry_run_success_rate = (
        guarded_candidate_dry_run_successes / guarded_candidate_dry_run_attempts
        if guarded_candidate_dry_run_attempts
        else 0.0
    )
    selected_dry_run_success_rate = (
        guarded_selected_dry_run_successes / guarded_selected_dry_run_attempts
        if guarded_selected_dry_run_attempts
        else 0.0
    )
    return {
        "shadow_steps": shadow_steps,
        "runtime_contract_steps": runtime_contract_steps,
        "runtime_contract_task_count": len(runtime_contract_task_ids),
        "runtime_contract_task_ids": sorted(runtime_contract_task_ids)[:200],
        "runtime_contract_success_steps": runtime_contract_success_steps,
        "runtime_contract_coverage_rate": round(coverage_rate, 6),
        "runtime_contract_success_rate": round(success_rate, 6),
        "guarded_comparison_steps": guarded_comparison_steps,
        "guarded_candidate_observed_steps": guarded_candidate_observed_steps,
        "guarded_baseline_observed_steps": guarded_baseline_observed_steps,
        "guarded_candidate_dry_run_attempts": guarded_candidate_dry_run_attempts,
        "guarded_candidate_dry_run_successes": guarded_candidate_dry_run_successes,
        "guarded_candidate_dry_run_skips": guarded_candidate_dry_run_skips,
        "guarded_candidate_dry_run_success_rate": round(dry_run_success_rate, 6),
        "guarded_candidate_dry_run_mode_counts": dict(sorted(guarded_candidate_dry_run_mode_counts.items())),
        "guarded_selected_dry_run_attempts": guarded_selected_dry_run_attempts,
        "guarded_selected_dry_run_successes": guarded_selected_dry_run_successes,
        "guarded_selected_dry_run_success_rate": round(selected_dry_run_success_rate, 6),
        "guarded_dry_run_switches_applied": guarded_dry_run_switches_applied,
        "runtime_artifact_failure_mode_counts": dict(sorted(mode_counts.items())),
        "rowwise_selector_source_counts": dict(sorted(selector_source_counts.items())),
        "rowwise_selector_policy_counts": dict(sorted(selector_policy_counts.items())),
        "selector_signal_ready": runtime_contract_steps > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-run-reports", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = KernelConfig()
    episodes_root = Path(args.episodes_root) if args.episodes_root else config.trajectories_root
    documents = list(iter_episode_documents(episodes_root, config=None if args.episodes_root else config))
    if args.include_run_reports or not args.episodes_root:
        documents.extend(_iter_report_documents(config.run_reports_dir))
    summary = summarize_runtime_contract_metrics(documents)
    report = {
        "report_kind": "neural_controller_runtime_contract_metrics",
        "episodes_root": str(episodes_root),
        "summary": summary,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "neural_controller_runtime_contract_metrics "
        f"shadow_steps={summary['shadow_steps']} "
        f"runtime_contract_steps={summary['runtime_contract_steps']} "
        f"runtime_contract_tasks={summary['runtime_contract_task_count']} "
        f"coverage={summary['runtime_contract_coverage_rate']:.3f} "
        f"success={summary['runtime_contract_success_rate']:.3f} "
        f"selector_signal_ready={str(summary['selector_signal_ready']).lower()}"
    )


if __name__ == "__main__":
    main()
