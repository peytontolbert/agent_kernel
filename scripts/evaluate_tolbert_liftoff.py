from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import json

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import retained_tolbert_liftoff_gate
from agent_kernel.modeling.evaluation.drift import run_takeover_drift_eval
from agent_kernel.modeling.evaluation.liftoff import build_liftoff_gate_report
from agent_kernel.trust import build_unattended_trust_ledger, write_unattended_trust_ledger
from evals.harness import run_eval


def _emit_summary(report) -> None:
    promoted = ",".join(report.primary_takeover_families) or "-"
    shadow = ",".join(report.shadow_only_families) or "-"
    demoted = ",".join(report.insufficient_proposal_families) or "-"
    print(
        f"[liftoff] state={report.state} promoted={promoted} shadow={shadow} insufficient_proposal={demoted}",
        file=sys.stderr,
        flush=True,
    )
    for family in report.insufficient_proposal_families:
        reason = str(report.proposal_gate_failure_reasons_by_benchmark_family.get(family, "")).strip()
        if reason:
            print(
                f"[liftoff] demotion family={family} reason={reason}",
                file=sys.stderr,
                flush=True,
            )


def _family_takeover_summary(report) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    evidence = getattr(report, "family_takeover_evidence", {}) or {}
    if not isinstance(evidence, dict):
        return summary
    for family, row in evidence.items():
        if not isinstance(row, dict):
            continue
        summary[str(family)] = {
            "decision": str(row.get("decision", "")),
            "failure_reason": str(row.get("failure_reason", "")),
            "pass_rate_delta": float(row.get("pass_rate_delta", 0.0) or 0.0),
            "proposal_selected_steps_delta": int(row.get("proposal_selected_steps_delta", 0) or 0),
            "novel_valid_command_rate_delta": float(row.get("novel_valid_command_rate_delta", 0.0) or 0.0),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--apply-routing", action="store_true")
    parser.add_argument("--include-discovered-tasks", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-benchmark-candidates", action="store_true")
    parser.add_argument("--include-verifier-candidates", action="store_true")
    parser.add_argument("--takeover-drift-step-budget", type=int, default=0)
    parser.add_argument("--takeover-drift-wave-task-limit", type=int, default=0)
    parser.add_argument("--takeover-drift-max-waves", type=int, default=0)
    args = parser.parse_args()

    config = KernelConfig()
    artifact_path = Path(args.artifact_path) if args.artifact_path else config.tolbert_model_artifact_path
    report_path = Path(args.report_path) if args.report_path else config.tolbert_liftoff_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_config = _scoped_liftoff_config(
        config,
        "tolbert_liftoff_baseline",
        use_tolbert_model_artifacts=False,
    )
    candidate_config = _scoped_liftoff_config(
        config,
        "tolbert_liftoff_candidate",
        tolbert_model_artifact_path=artifact_path,
        use_tolbert_model_artifacts=True,
    )
    _reset_reports_dir(baseline_config.run_reports_dir)
    _reset_reports_dir(candidate_config.run_reports_dir)

    baseline_metrics = run_eval(
        config=baseline_config,
        include_discovered_tasks=args.include_discovered_tasks,
        include_generated=args.include_curriculum,
        include_failure_generated=args.include_failure_curriculum,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_benchmark_candidates=args.include_benchmark_candidates,
        include_verifier_candidates=args.include_verifier_candidates,
        progress_label="tolbert_liftoff_baseline",
    )
    candidate_metrics = run_eval(
        config=candidate_config,
        include_discovered_tasks=args.include_discovered_tasks,
        include_generated=args.include_curriculum,
        include_failure_generated=args.include_failure_curriculum,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_benchmark_candidates=args.include_benchmark_candidates,
        include_verifier_candidates=args.include_verifier_candidates,
        progress_label="tolbert_liftoff_candidate",
    )
    payload = {}
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    gate = retained_tolbert_liftoff_gate(payload)
    baseline_trust_ledger = build_unattended_trust_ledger(baseline_config)
    candidate_trust_ledger = build_unattended_trust_ledger(candidate_config)
    write_unattended_trust_ledger(baseline_config)
    write_unattended_trust_ledger(candidate_config)
    drift_step_budget = max(0, args.takeover_drift_step_budget) or int(gate.get("takeover_drift_step_budget", 0))
    drift_wave_task_limit = max(0, args.takeover_drift_wave_task_limit) or int(gate.get("takeover_drift_wave_task_limit", 0))
    drift_max_waves = max(0, args.takeover_drift_max_waves) or int(gate.get("takeover_drift_max_waves", 0))
    takeover_drift_report = None
    if bool(gate.get("require_takeover_drift_eval", True)) and drift_step_budget > 0:
        takeover_drift_report = run_takeover_drift_eval(
            config=config,
            artifact_path=artifact_path,
            step_budget=drift_step_budget,
            wave_task_limit=drift_wave_task_limit,
            max_waves=drift_max_waves,
            eval_kwargs=_eval_kwargs_from_args(args),
        ).to_dict()
    report = build_liftoff_gate_report(
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        artifact_payload=payload,
        candidate_trust_ledger=candidate_trust_ledger,
        baseline_trust_ledger=baseline_trust_ledger,
        takeover_drift_report=takeover_drift_report,
    )
    output = {
        "spec_version": "asi_v1",
        "artifact_kind": "liftoff_gate_report",
        "artifact_path": str(artifact_path),
        "report": report.to_dict(),
        "summary": {
            "state": report.state,
            "reason": report.reason,
            "primary_takeover_families": list(report.primary_takeover_families),
            "shadow_only_families": list(report.shadow_only_families),
            "insufficient_proposal_families": list(report.insufficient_proposal_families),
            "proposal_gate_failure_reasons_by_benchmark_family": dict(
                report.proposal_gate_failure_reasons_by_benchmark_family
            ),
            "family_takeover": _family_takeover_summary(report),
        },
        "baseline": {
            "passed": baseline_metrics.passed,
            "total": baseline_metrics.total,
            "pass_rate": baseline_metrics.pass_rate,
            "average_steps": baseline_metrics.average_steps,
        },
        "candidate": {
            "passed": candidate_metrics.passed,
            "total": candidate_metrics.total,
            "pass_rate": candidate_metrics.pass_rate,
            "average_steps": candidate_metrics.average_steps,
            "proposal_metrics_by_benchmark_family": dict(candidate_metrics.proposal_metrics_by_benchmark_family),
        },
        "baseline_trust_ledger_path": str(baseline_config.unattended_trust_ledger_path),
        "candidate_trust_ledger_path": str(candidate_config.unattended_trust_ledger_path),
        "baseline_trust": baseline_trust_ledger,
        "candidate_trust": candidate_trust_ledger,
        "takeover_drift": takeover_drift_report or {},
    }
    if args.apply_routing and isinstance(payload, dict):
        runtime_policy = payload.get("runtime_policy", {})
        if not isinstance(runtime_policy, dict):
            runtime_policy = {}
        hybrid_runtime = payload.get("hybrid_runtime", {})
        if not isinstance(hybrid_runtime, dict):
            hybrid_runtime = {}
        if report.state == "retain":
            runtime_policy["primary_benchmark_families"] = list(report.primary_takeover_families)
            runtime_policy["shadow_benchmark_families"] = list(report.shadow_only_families)
            hybrid_runtime["primary_enabled"] = bool(report.primary_takeover_families)
            hybrid_runtime["shadow_enabled"] = bool(report.shadow_only_families)
        else:
            runtime_policy["shadow_benchmark_families"] = sorted(
                {
                    *[str(value) for value in runtime_policy.get("shadow_benchmark_families", [])],
                    *report.shadow_only_families,
                    *report.primary_takeover_families,
                }
            )
            hybrid_runtime["primary_enabled"] = False
            hybrid_runtime["shadow_enabled"] = bool(runtime_policy.get("shadow_benchmark_families", []))
        payload["runtime_policy"] = runtime_policy
        payload["hybrid_runtime"] = hybrid_runtime
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        output["routing_applied"] = True
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _emit_summary(report)
    print(report_path)

def _scoped_liftoff_config(base: KernelConfig, scope: str, **overrides) -> KernelConfig:
    scoped_reports_dir = base.run_reports_dir / scope
    scoped_ledger_path = base.unattended_trust_ledger_path.with_name(
        f"{scope}_{base.unattended_trust_ledger_path.name}"
    )
    config = replace(
        base,
        run_reports_dir=scoped_reports_dir,
        unattended_trust_ledger_path=scoped_ledger_path,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    config.run_reports_dir.mkdir(parents=True, exist_ok=True)
    config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    return config


def _reset_reports_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _eval_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "include_discovered_tasks": args.include_discovered_tasks,
        "include_generated": args.include_curriculum,
        "include_failure_generated": args.include_failure_curriculum,
        "include_episode_memory": args.include_episode_memory,
        "include_skill_memory": args.include_skill_memory,
        "include_skill_transfer": args.include_skill_transfer,
        "include_operator_memory": args.include_operator_memory,
        "include_tool_memory": args.include_tool_memory,
        "include_verifier_memory": args.include_verifier_memory,
        "include_benchmark_candidates": args.include_benchmark_candidates,
        "include_verifier_candidates": args.include_verifier_candidates,
    }


if __name__ == "__main__":
    main()
