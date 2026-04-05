from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import re

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import retained_tolbert_liftoff_gate
from agent_kernel.modeling.evaluation.drift import run_takeover_drift_eval
from agent_kernel.modeling.evaluation.liftoff import build_liftoff_gate_report
from agent_kernel.trust import build_unattended_trust_ledger, write_unattended_trust_ledger
from evals.harness import run_eval, scoped_eval_config


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
    parser.add_argument("--preserve-report-history", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument(
        "--priority-benchmark-family",
        action="append",
        default=[],
    )
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
    parser.add_argument("--takeover-drift-step-budget", type=int, default=None)
    parser.add_argument("--takeover-drift-wave-task-limit", type=int, default=None)
    parser.add_argument("--takeover-drift-max-waves", type=int, default=None)
    args = parser.parse_args()

    config = KernelConfig()
    operator_policy_overrides = _liftoff_operator_policy_overrides(
        priority_benchmark_families=args.priority_benchmark_family,
        config=config,
    )
    artifact_path = Path(args.artifact_path) if args.artifact_path else config.tolbert_model_artifact_path
    report_path = Path(args.report_path) if args.report_path else config.tolbert_liftoff_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

    baseline_config = _scoped_liftoff_config(
        config,
        _liftoff_scope_name("tolbert_liftoff_baseline", report_path=report_path),
        use_tolbert_model_artifacts=False,
        **operator_policy_overrides,
    )
    candidate_config = _scoped_liftoff_config(
        config,
        _liftoff_scope_name("tolbert_liftoff_candidate", report_path=report_path),
        tolbert_model_artifact_path=artifact_path,
        use_tolbert_model_artifacts=True,
        **operator_policy_overrides,
    )
    if args.preserve_report_history:
        baseline_config.run_reports_dir.mkdir(parents=True, exist_ok=True)
        candidate_config.run_reports_dir.mkdir(parents=True, exist_ok=True)
    else:
        _reset_reports_dir(baseline_config.run_reports_dir)
        _reset_reports_dir(candidate_config.run_reports_dir)
    candidate_config.tolbert_model_artifact_path = _materialize_liftoff_candidate_artifact(
        artifact_path=artifact_path,
        payload=payload,
        scoped_root=candidate_config.run_reports_dir,
        shadow_benchmark_families=_liftoff_shadow_enrollment_families(
            payload=payload,
            priority_benchmark_families=args.priority_benchmark_family,
            config=config,
        ),
    )
    eval_priority_families = _liftoff_eval_priority_families(
        payload=payload,
        priority_benchmark_families=args.priority_benchmark_family,
        config=config,
        preserve_report_history=args.preserve_report_history,
    )
    eval_priority_family_weights = _liftoff_eval_priority_family_weights(
        explicit_priority_families=args.priority_benchmark_family,
        eval_priority_families=eval_priority_families,
    )

    baseline_metrics = run_eval(
        config=baseline_config,
        task_limit=max(0, int(args.task_limit)) or None,
        priority_benchmark_families=eval_priority_families,
        priority_benchmark_family_weights=eval_priority_family_weights or None,
        prefer_low_cost_tasks=True,
        write_unattended_reports=True,
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
        surface_shared_repo_bundles=False,
        progress_label="tolbert_liftoff_baseline",
    )
    candidate_metrics = run_eval(
        config=candidate_config,
        task_limit=max(0, int(args.task_limit)) or None,
        priority_benchmark_families=eval_priority_families,
        priority_benchmark_family_weights=eval_priority_family_weights or None,
        prefer_low_cost_tasks=True,
        write_unattended_reports=True,
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
        surface_shared_repo_bundles=False,
        progress_label="tolbert_liftoff_candidate",
    )
    gate = retained_tolbert_liftoff_gate(payload)
    baseline_trust_config = _liftoff_trust_config(
        baseline_config,
        preserve_report_history=args.preserve_report_history,
    )
    candidate_trust_config = _liftoff_trust_config(
        candidate_config,
        preserve_report_history=args.preserve_report_history,
    )
    baseline_trust_ledger = build_unattended_trust_ledger(baseline_trust_config)
    candidate_trust_ledger = build_unattended_trust_ledger(candidate_trust_config)
    write_unattended_trust_ledger(baseline_trust_config)
    write_unattended_trust_ledger(candidate_trust_config)
    drift_step_budget = (
        max(0, int(args.takeover_drift_step_budget))
        if args.takeover_drift_step_budget is not None
        else int(gate.get("takeover_drift_step_budget", 0))
    )
    drift_wave_task_limit = (
        max(0, int(args.takeover_drift_wave_task_limit))
        if args.takeover_drift_wave_task_limit is not None
        else int(gate.get("takeover_drift_wave_task_limit", 0))
    )
    drift_max_waves = (
        max(0, int(args.takeover_drift_max_waves))
        if args.takeover_drift_max_waves is not None
        else int(gate.get("takeover_drift_max_waves", 0))
    )
    takeover_drift_report = None
    if bool(gate.get("require_takeover_drift_eval", True)) and drift_step_budget > 0:
        drift_config = replace(config, **operator_policy_overrides) if operator_policy_overrides else config
        takeover_drift_report = run_takeover_drift_eval(
            config=drift_config,
            artifact_path=candidate_config.tolbert_model_artifact_path,
            step_budget=drift_step_budget,
            wave_task_limit=drift_wave_task_limit,
            max_waves=drift_max_waves,
            eval_kwargs=_eval_kwargs_from_args(
                args,
                priority_benchmark_families=eval_priority_families,
                priority_benchmark_family_weights=eval_priority_family_weights or None,
            ),
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
        "preserve_report_history": bool(args.preserve_report_history),
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
    config = scoped_eval_config(base, scope)
    for key, value in overrides.items():
        setattr(config, key, value)
    config.run_reports_dir.mkdir(parents=True, exist_ok=True)
    config.unattended_trust_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    return config


def _liftoff_scope_name(base_scope: str, *, report_path: Path) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", report_path.stem.lower()).strip("_")
    if not token:
        return base_scope
    return f"{base_scope}_{token}"


def _reset_reports_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _liftoff_trust_config(
    base: KernelConfig,
    *,
    preserve_report_history: bool,
) -> KernelConfig:
    if not preserve_report_history:
        return base
    return replace(
        base,
        unattended_trust_recent_report_limit=0,
        run_reports_dir=_preserved_liftoff_reports_dir(base.run_reports_dir),
    )


def _preserved_liftoff_reports_dir(run_reports_dir: Path) -> Path:
    if _contains_unattended_reports(run_reports_dir):
        return run_reports_dir
    name = run_reports_dir.name
    for legacy_name in ("tolbert_liftoff_candidate", "tolbert_liftoff_baseline"):
        if name == legacy_name:
            return run_reports_dir
        if not name.startswith(f"{legacy_name}_"):
            continue
        legacy_root = run_reports_dir.with_name(legacy_name)
        if _contains_unattended_reports(legacy_root):
            return legacy_root
    return run_reports_dir


def _contains_unattended_reports(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("task_report_*.json"))


def _liftoff_operator_policy_overrides(
    *,
    priority_benchmark_families: list[str],
    config: KernelConfig,
) -> dict[str, object]:
    targeted_families = {
        str(value).strip()
        for value in [*priority_benchmark_families, *config.unattended_trust_required_benchmark_families]
        if str(value).strip()
    }
    if "repo_sandbox" not in targeted_families:
        return {}
    return {
        "unattended_allow_git_commands": True,
        "unattended_allow_generated_path_mutations": True,
    }


def _liftoff_shadow_enrollment_families(
    *,
    payload: object,
    priority_benchmark_families: list[str],
    config: KernelConfig,
) -> list[str]:
    candidates: list[str] = []
    if isinstance(payload, dict):
        runtime_policy = payload.get("runtime_policy", {})
        dataset_manifest = payload.get("dataset_manifest", {})
        if isinstance(runtime_policy, dict):
            candidates.extend(runtime_policy.get("shadow_benchmark_families", []))
            candidates.extend(runtime_policy.get("primary_benchmark_families", []))
        if isinstance(dataset_manifest, dict):
            candidates.extend(dataset_manifest.get("benchmark_families", []))
    candidates.extend(priority_benchmark_families)
    candidates.extend(config.unattended_trust_required_benchmark_families)
    families: list[str] = []
    for value in candidates:
        family = str(value).strip()
        if family and family not in families:
            families.append(family)
    return families


def _liftoff_eval_priority_families(
    *,
    payload: object,
    priority_benchmark_families: list[str],
    config: KernelConfig,
    preserve_report_history: bool,
) -> list[str]:
    explicit = [
        str(value).strip() for value in priority_benchmark_families if str(value).strip()
    ]
    if not preserve_report_history:
        return explicit
    merged: list[str] = []
    for value in [
        *explicit,
        *config.unattended_trust_required_benchmark_families,
        *_liftoff_shadow_enrollment_families(
            payload=payload,
            priority_benchmark_families=priority_benchmark_families,
            config=config,
        ),
    ]:
        family = str(value).strip()
        if family and family not in merged:
            merged.append(family)
    return merged


def _liftoff_eval_priority_family_weights(
    *,
    explicit_priority_families: list[str],
    eval_priority_families: list[str],
) -> dict[str, float]:
    explicit = {
        str(value).strip()
        for value in explicit_priority_families
        if str(value).strip()
    }
    if not explicit:
        return {}
    weights: dict[str, float] = {}
    mixed_breadth = any(family not in explicit for family in eval_priority_families)
    for family in eval_priority_families:
        if family in explicit:
            if mixed_breadth and family == "repo_sandbox":
                weights[family] = 2.0
            else:
                weights[family] = 4.0
        else:
            weights[family] = 1.0
    return weights


def _materialize_liftoff_candidate_artifact(
    *,
    artifact_path: Path,
    payload: object,
    scoped_root: Path,
    shadow_benchmark_families: list[str],
) -> Path:
    if not isinstance(payload, dict):
        return artifact_path
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return artifact_path
    if not shadow_benchmark_families:
        return artifact_path
    scoped_payload = deepcopy(payload)
    runtime_policy = scoped_payload.get("runtime_policy", {})
    if not isinstance(runtime_policy, dict):
        runtime_policy = {}
    current_shadow = [
        str(value).strip()
        for value in runtime_policy.get("shadow_benchmark_families", [])
        if str(value).strip()
    ]
    merged_shadow = sorted({*current_shadow, *shadow_benchmark_families})
    if merged_shadow == current_shadow:
        return artifact_path
    runtime_policy["shadow_benchmark_families"] = merged_shadow
    scoped_payload["runtime_policy"] = runtime_policy
    target = scoped_root / f"{artifact_path.stem}.liftoff_shadow.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scoped_payload, indent=2), encoding="utf-8")
    return target


def _eval_kwargs_from_args(
    args: argparse.Namespace,
    *,
    priority_benchmark_families: list[str] | None = None,
    priority_benchmark_family_weights: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "priority_benchmark_families": (
            list(priority_benchmark_families)
            if priority_benchmark_families is not None
            else [str(value).strip() for value in args.priority_benchmark_family if str(value).strip()]
        ),
        "priority_benchmark_family_weights": dict(priority_benchmark_family_weights or {}),
        "prefer_low_cost_tasks": True,
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
        "surface_shared_repo_bundles": False,
    }


if __name__ == "__main__":
    main()
