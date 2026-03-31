from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import selectors
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import json

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import retained_tolbert_liftoff_gate
from agent_kernel.modeling.evaluation.drift import run_takeover_drift_eval
from agent_kernel.modeling.evaluation.liftoff import build_liftoff_gate_report
from agent_kernel.modeling.evaluation.universal_decoder_eval import evaluate_universal_decoder_against_seed
from agent_kernel.runtime_supervision import (
    atomic_write_json,
    atomic_write_text,
    install_termination_handlers,
    spawn_process_group,
    terminate_process_tree,
)
from agent_kernel.trust import build_unattended_trust_ledger
from agent_kernel.tolbert_model_improvement import build_tolbert_model_candidate_artifact
from evals.harness import run_eval


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit_liftoff_summary(report) -> None:
    promoted = ",".join(report.primary_takeover_families) or "-"
    shadow = ",".join(report.shadow_only_families) or "-"
    demoted = ",".join(report.insufficient_proposal_families) or "-"
    _progress(
        f"[liftoff] summary state={report.state} promoted={promoted} shadow={shadow} insufficient_proposal={demoted}"
    )
    for family in report.insufficient_proposal_families:
        reason = str(report.proposal_gate_failure_reasons_by_benchmark_family.get(family, "")).strip()
        if reason:
            _progress(f"[liftoff] demotion family={family} reason={reason}")


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


def _run_and_stream(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    progress_label: str | None = None,
    heartbeat_interval_seconds: float = 60.0,
    max_silence_seconds: float = 0.0,
    max_runtime_seconds: float = 0.0,
    max_progress_stall_seconds: float = 0.0,
) -> dict[str, object]:
    completed_output: list[str] = []
    process = spawn_process_group(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        bufsize=1,
    )
    process_pid = int(getattr(process, "pid", 0) or 0)
    assert process.stdout is not None
    if not hasattr(process.stdout, "fileno"):
        for line in process.stdout:
            completed_output.append(line)
            print(line, end="", file=sys.stderr, flush=True)
        returncode = process.wait()
        return {
            "returncode": returncode,
            "stdout": "".join(completed_output).strip(),
            "timed_out": False,
        }
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_output_at = time.monotonic()
    last_progress_at = last_output_at
    last_heartbeat_at = last_output_at
    started_at = last_output_at
    heartbeat_interval = max(0.0, float(heartbeat_interval_seconds))
    max_silence = max(0.0, float(max_silence_seconds))
    max_runtime = max(0.0, float(max_runtime_seconds))
    max_progress_stall = max(0.0, float(max_progress_stall_seconds))
    try:
        while True:
            events = selector.select(timeout=1.0)
            if events:
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        break
                    completed_output.append(line)
                    now = time.monotonic()
                    last_output_at = now
                    if "[eval:" in line or "[cycle:" in line or "[repeated]" in line or "[liftoff]" in line or "finalize phase=" in line:
                        last_progress_at = now
                    print(line, end="", file=sys.stderr, flush=True)
            elif process.poll() is not None:
                break
            now = time.monotonic()
            silence = now - last_output_at
            progress_stall = now - last_progress_at
            runtime_elapsed = now - started_at
            if heartbeat_interval > 0.0 and (now - last_heartbeat_at) >= heartbeat_interval and silence >= heartbeat_interval:
                _progress(f"[liftoff] child={progress_label or 'repeated_improvement'} still_running silence={int(silence)}s")
                last_heartbeat_at = now
            if max_runtime > 0.0 and runtime_elapsed >= max_runtime:
                terminate_process_tree(process)
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max runtime of {int(max_runtime)} seconds",
                }
            if max_silence > 0.0 and silence >= max_silence:
                terminate_process_tree(process)
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max silence of {int(max_silence)} seconds",
                }
            if max_progress_stall > 0.0 and progress_stall >= max_progress_stall:
                terminate_process_tree(process)
                return {
                    "returncode": -9,
                    "stdout": "".join(completed_output).strip(),
                    "timed_out": True,
                    "timeout_reason": f"child exceeded max progress stall of {int(max_progress_stall)} seconds",
                }
        returncode = process.wait()
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        selector.close()
    return {
        "returncode": returncode,
        "stdout": "".join(completed_output).strip(),
        "timed_out": False,
    }


def _current_payload(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metrics_from_summary(payload: object) -> object | None:
    if not isinstance(payload, dict) or not payload:
        return None
    from evals.metrics import EvalMetrics

    return EvalMetrics(
        total=int(payload.get("total", 0) or 0),
        passed=int(payload.get("passed", 0) or 0),
        average_steps=float(payload.get("average_steps", 0.0) or 0.0),
        generated_total=int(payload.get("generated_total", 0) or 0),
        generated_passed=int(payload.get("generated_passed", 0) or 0),
        low_confidence_episodes=int(payload.get("low_confidence_episodes", 0) or 0),
        proposal_selected_steps=int(payload.get("proposal_selected_steps", 0) or 0),
        novel_command_steps=int(payload.get("novel_command_steps", 0) or 0),
        novel_valid_command_steps=int(payload.get("novel_valid_command_steps", 0) or 0),
        tolbert_shadow_episodes=int(payload.get("tolbert_shadow_episodes", 0) or 0),
        tolbert_primary_episodes=int(payload.get("tolbert_primary_episodes", 0) or 0),
        total_by_benchmark_family=dict(payload.get("total_by_benchmark_family", {}))
        if isinstance(payload.get("total_by_benchmark_family", {}), dict)
        else {},
        passed_by_benchmark_family=dict(payload.get("passed_by_benchmark_family", {}))
        if isinstance(payload.get("passed_by_benchmark_family", {}), dict)
        else {},
        tolbert_shadow_episodes_by_benchmark_family=dict(payload.get("tolbert_shadow_episodes_by_benchmark_family", {}))
        if isinstance(payload.get("tolbert_shadow_episodes_by_benchmark_family", {}), dict)
        else {},
        tolbert_primary_episodes_by_benchmark_family=dict(payload.get("tolbert_primary_episodes_by_benchmark_family", {}))
        if isinstance(payload.get("tolbert_primary_episodes_by_benchmark_family", {}), dict)
        else {},
        proposal_metrics_by_benchmark_family=dict(payload.get("proposal_metrics_by_benchmark_family", {}))
        if isinstance(payload.get("proposal_metrics_by_benchmark_family", {}), dict)
        else {},
        world_feedback_summary=dict(payload.get("world_feedback_summary", {}))
        if isinstance(payload.get("world_feedback_summary", {}), dict)
        else {},
        world_feedback_by_benchmark_family=dict(payload.get("world_feedback_by_benchmark_family", {}))
        if isinstance(payload.get("world_feedback_by_benchmark_family", {}), dict)
        else {},
    )


def _last_report_path(stdout: str) -> Path | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = Path(lines[-1])
    return candidate if candidate.exists() else None


def _eval_kwargs() -> dict[str, object]:
    return {
        "include_discovered_tasks": True,
        "include_generated": True,
        "include_failure_generated": True,
        "include_episode_memory": True,
        "include_skill_memory": True,
        "include_skill_transfer": True,
        "include_operator_memory": True,
        "include_tool_memory": True,
        "include_verifier_memory": True,
        "include_benchmark_candidates": True,
        "include_verifier_candidates": True,
    }


def _run_universal_decoder_eval(
    *,
    config: KernelConfig,
    candidate_artifact: dict[str, object],
) -> dict[str, object] | None:
    runtime_paths = candidate_artifact.get("runtime_paths", {})
    if not isinstance(runtime_paths, dict):
        return None
    dataset_manifest = candidate_artifact.get("universal_dataset_manifest", {})
    if not isinstance(dataset_manifest, dict):
        return None
    bundle_manifest_path = str(runtime_paths.get("universal_bundle_manifest_path", "")).strip()
    dataset_manifest_path = str(dataset_manifest.get("manifest_path", "")).strip()
    if not bundle_manifest_path or not dataset_manifest_path:
        return None
    bundle_path = Path(bundle_manifest_path)
    manifest_path = Path(dataset_manifest_path)
    if not bundle_path.exists() or not manifest_path.exists():
        return {
            "available": False,
            "reason": "universal bundle or dataset manifest is missing",
            "bundle_manifest_path": str(bundle_path),
            "dataset_manifest_path": str(manifest_path),
        }
    try:
        report = evaluate_universal_decoder_against_seed(
            hybrid_bundle_manifest_path=bundle_path,
            dataset_manifest_path=manifest_path,
            config=config,
            device="cpu" if not str(config.tolbert_device).startswith("cuda") else config.tolbert_device,
            max_examples=32,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "bundle_manifest_path": str(bundle_path),
            "dataset_manifest_path": str(manifest_path),
        }
    payload = report.to_dict()
    payload["available"] = True
    return payload


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    atomic_write_json(report_path, payload)


def _metric_summary(metrics) -> dict[str, object]:
    return {
        "total": metrics.total,
        "passed": metrics.passed,
        "pass_rate": metrics.pass_rate,
        "average_steps": metrics.average_steps,
        "generated_total": metrics.generated_total,
        "generated_passed": metrics.generated_passed,
        "low_confidence_episodes": metrics.low_confidence_episodes,
        "proposal_selected_steps": metrics.proposal_selected_steps,
        "novel_command_steps": metrics.novel_command_steps,
        "novel_valid_command_steps": metrics.novel_valid_command_steps,
        "novel_valid_command_rate": metrics.novel_valid_command_rate,
        "tolbert_shadow_episodes": metrics.tolbert_shadow_episodes,
        "tolbert_primary_episodes": metrics.tolbert_primary_episodes,
        "total_by_benchmark_family": dict(metrics.total_by_benchmark_family),
        "passed_by_benchmark_family": dict(metrics.passed_by_benchmark_family),
        "tolbert_shadow_episodes_by_benchmark_family": dict(metrics.tolbert_shadow_episodes_by_benchmark_family),
        "tolbert_primary_episodes_by_benchmark_family": dict(metrics.tolbert_primary_episodes_by_benchmark_family),
        "proposal_metrics_by_benchmark_family": dict(metrics.proposal_metrics_by_benchmark_family),
        "world_feedback_summary": dict(metrics.world_feedback_summary),
        "world_feedback_by_benchmark_family": dict(metrics.world_feedback_by_benchmark_family),
    }


def _world_feedback_calibration_delta(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_progress_mae = float(baseline.get("progress_calibration_mae", 0.0) or 0.0)
    candidate_progress_mae = float(candidate.get("progress_calibration_mae", 0.0) or 0.0)
    baseline_risk_mae = float(baseline.get("risk_calibration_mae", 0.0) or 0.0)
    candidate_risk_mae = float(candidate.get("risk_calibration_mae", 0.0) or 0.0)
    baseline_decoder_progress_mae = float(baseline.get("decoder_progress_calibration_mae", 0.0) or 0.0)
    candidate_decoder_progress_mae = float(candidate.get("decoder_progress_calibration_mae", 0.0) or 0.0)
    baseline_decoder_risk_mae = float(baseline.get("decoder_risk_calibration_mae", 0.0) or 0.0)
    candidate_decoder_risk_mae = float(candidate.get("decoder_risk_calibration_mae", 0.0) or 0.0)
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "progress_calibration_mae_gain": round(baseline_progress_mae - candidate_progress_mae, 4),
        "risk_calibration_mae_gain": round(baseline_risk_mae - candidate_risk_mae, 4),
        "decoder_progress_calibration_mae_gain": round(
            baseline_decoder_progress_mae - candidate_decoder_progress_mae,
            4,
        ),
        "decoder_risk_calibration_mae_gain": round(
            baseline_decoder_risk_mae - candidate_decoder_risk_mae,
            4,
        ),
    }


def _world_feedback_calibration_delta_by_benchmark_family(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(candidate, dict):
        candidate = {}
    families = sorted(set(baseline) | set(candidate))
    summary: dict[str, dict[str, object]] = {}
    for family in families:
        summary[family] = _world_feedback_calibration_delta(
            dict(baseline.get(family, {})) if isinstance(baseline.get(family, {}), dict) else {},
            dict(candidate.get(family, {})) if isinstance(candidate.get(family, {}), dict) else {},
        )
    return summary


def _proposal_metrics_delta_by_benchmark_family(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(candidate, dict):
        candidate = {}
    families = sorted(set(baseline) | set(candidate))
    summary: dict[str, dict[str, object]] = {}
    for family in families:
        baseline_row = baseline.get(family, {})
        candidate_row = candidate.get(family, {})
        if not isinstance(baseline_row, dict):
            baseline_row = {}
        if not isinstance(candidate_row, dict):
            candidate_row = {}
        baseline_task_count = int(baseline_row.get("task_count", 0) or 0)
        candidate_task_count = int(candidate_row.get("task_count", 0) or 0)
        if baseline_task_count == 0 and candidate_task_count == 0:
            continue
        baseline_proposal_steps = int(baseline_row.get("proposal_selected_steps", 0) or 0)
        candidate_proposal_steps = int(candidate_row.get("proposal_selected_steps", 0) or 0)
        baseline_valid_steps = int(baseline_row.get("novel_valid_command_steps", 0) or 0)
        candidate_valid_steps = int(candidate_row.get("novel_valid_command_steps", 0) or 0)
        baseline_valid_rate = float(baseline_row.get("novel_valid_command_rate", 0.0) or 0.0)
        candidate_valid_rate = float(candidate_row.get("novel_valid_command_rate", 0.0) or 0.0)
        summary[family] = {
            "baseline_task_count": baseline_task_count,
            "candidate_task_count": candidate_task_count,
            "baseline_proposal_selected_steps": baseline_proposal_steps,
            "candidate_proposal_selected_steps": candidate_proposal_steps,
            "proposal_selected_steps_delta": candidate_proposal_steps - baseline_proposal_steps,
            "baseline_novel_valid_command_steps": baseline_valid_steps,
            "candidate_novel_valid_command_steps": candidate_valid_steps,
            "novel_valid_command_steps_delta": candidate_valid_steps - baseline_valid_steps,
            "baseline_novel_valid_command_rate": round(baseline_valid_rate, 4),
            "candidate_novel_valid_command_rate": round(candidate_valid_rate, 4),
            "novel_valid_command_rate_delta": round(candidate_valid_rate - baseline_valid_rate, 4),
        }
    return summary


def _apply_routing_to_artifact(artifact_path: Path, artifact: dict[str, object], report) -> None:
    runtime_policy = artifact.get("runtime_policy", {})
    if not isinstance(runtime_policy, dict):
        runtime_policy = {}
    hybrid_runtime = artifact.get("hybrid_runtime", {})
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
                *[str(value) for value in runtime_policy.get("shadow_benchmark_families", []) if str(value).strip()],
                *report.shadow_only_families,
                *report.primary_takeover_families,
            }
        )
        hybrid_runtime["primary_enabled"] = False
        hybrid_runtime["shadow_enabled"] = bool(runtime_policy.get("shadow_benchmark_families", []))
    artifact["runtime_policy"] = runtime_policy
    artifact["hybrid_runtime"] = hybrid_runtime
    atomic_write_json(artifact_path, artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--campaign-width", type=int, default=2)
    parser.add_argument("--variant-width", type=int, default=1)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--skip-improvement", action="store_true")
    parser.add_argument(
        "--focus",
        choices=("balanced", "recovery_alignment", "discovered_task_adaptation"),
        default="balanced",
    )
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--candidate-artifact-path", default=None)
    parser.add_argument("--apply-routing", action="store_true")
    parser.add_argument("--promote-on-retain", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--baseline-task-limit", type=int, default=0)
    parser.add_argument("--candidate-task-limit", type=int, default=0)
    parser.add_argument("--priority-benchmark-family", action="append", default=[])
    parser.add_argument("--takeover-drift-step-budget", type=int, default=0)
    parser.add_argument("--takeover-drift-wave-task-limit", type=int, default=0)
    parser.add_argument("--takeover-drift-max-waves", type=int, default=0)
    parser.add_argument("--child-heartbeat-seconds", type=float, default=120.0)
    parser.add_argument("--max-child-silence-seconds", type=float, default=1800.0)
    parser.add_argument("--max-child-runtime-seconds", type=float, default=14400.0)
    parser.add_argument("--max-child-progress-stall-seconds", type=float, default=1800.0)
    parser.add_argument("--exclude-subsystem", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    received_signal = {"value": 0}

    def _handle_termination(signum: int) -> None:
        received_signal["value"] = int(signum)
        raise KeyboardInterrupt(f"received signal {signal.Signals(signum).name}")

    restore_signal_handlers = install_termination_handlers(_handle_termination)

    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device
    config.ensure_directories()

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        Path(args.report_path)
        if args.report_path
        else config.improvement_reports_dir / f"tolbert_liftoff_loop_{run_id}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resume_payload = _current_payload(report_path) if args.resume and report_path.exists() else {}
    resume_candidate_artifact_path = (
        Path(str(resume_payload.get("candidate_artifact_path", "")).strip())
        if isinstance(resume_payload, dict) and str(resume_payload.get("candidate_artifact_path", "")).strip()
        else None
    )
    candidate_output_dir = config.candidate_artifacts_root / f"tolbert_liftoff_loop_{run_id}"
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifact_path = (
        Path(args.candidate_artifact_path)
        if args.candidate_artifact_path
        else resume_candidate_artifact_path
        if resume_candidate_artifact_path is not None
        else candidate_output_dir / "tolbert_model_artifact.json"
    )
    baseline_metrics = None
    candidate_metrics = None
    liftoff_report = None
    takeover_drift_report = None
    universal_decoder_eval = None
    baseline_trust_ledger = None
    candidate_trust_ledger = None
    candidate_artifact = _current_payload(candidate_artifact_path) if args.resume else {}
    if not isinstance(candidate_artifact, dict):
        candidate_artifact = {}
    phase = "init"
    base_report = {
        "spec_version": "asi_v1",
        "report_kind": "tolbert_liftoff_loop_report",
        "created_at": created_at.isoformat(),
        "status": "running",
        "phase": phase,
        "focus": args.focus,
        "evaluation_scope": {
            "task_limit": max(0, args.task_limit),
            "baseline_task_limit": max(0, args.baseline_task_limit),
            "candidate_task_limit": max(0, args.candidate_task_limit),
            "priority_benchmark_families": [
                str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
            ],
        },
        "candidate_artifact_path": str(candidate_artifact_path),
        "retained_artifact_path": str(config.tolbert_model_artifact_path),
    }
    _write_report(report_path, base_report)

    campaign_report_path = (
        Path(str(resume_payload.get("campaign", {}).get("campaign_report_path", "")).strip())
        if isinstance(resume_payload, dict)
        and isinstance(resume_payload.get("campaign", {}), dict)
        and str(resume_payload.get("campaign", {}).get("campaign_report_path", "")).strip()
        else None
    )
    campaign_completed = bool(
        isinstance(resume_payload, dict)
        and isinstance(resume_payload.get("campaign", {}), dict)
        and resume_payload.get("campaign", {}).get("completed", False)
    )
    baseline_metrics = _metrics_from_summary(resume_payload.get("baseline_metrics", {})) if isinstance(resume_payload, dict) else None
    candidate_metrics = _metrics_from_summary(resume_payload.get("candidate_metrics", {})) if isinstance(resume_payload, dict) else None
    try:
        if not campaign_completed and not args.skip_improvement and max(0, args.cycles) > 0:
            phase = "campaign"
            base_report["phase"] = phase
            _write_report(report_path, base_report)
            _progress(
                f"[liftoff] phase=campaign cycles={max(1, args.cycles)} task_limit={max(0, args.task_limit)} "
                f"tolbert_device={config.tolbert_device}"
            )
            repeated_script = repo_root / "scripts" / "run_repeated_improvement_cycles.py"
            cmd = [
                sys.executable,
                str(repeated_script),
                "--cycles",
                str(max(1, args.cycles)),
                "--campaign-width",
                str(max(1, args.campaign_width)),
                "--variant-width",
                str(max(1, args.variant_width)),
                "--campaign-label",
                f"tolbert-liftoff-{run_id}",
                "--include-episode-memory",
                "--include-skill-memory",
                "--include-skill-transfer",
                "--include-operator-memory",
                "--include-tool-memory",
                "--include-verifier-memory",
                "--include-curriculum",
                "--include-failure-curriculum",
            ]
            if args.adaptive_search:
                cmd.append("--adaptive-search")
            if max(0, args.task_limit) > 0:
                cmd.extend(["--task-limit", str(max(0, args.task_limit))])
            for family in args.priority_benchmark_family:
                token = str(family).strip()
                if token:
                    cmd.extend(["--priority-benchmark-family", token])
            if args.provider:
                cmd.extend(["--provider", args.provider])
            if args.model:
                cmd.extend(["--model", args.model])
            if args.tolbert_device:
                cmd.extend(["--tolbert-device", args.tolbert_device])
            for excluded_subsystem in args.exclude_subsystem:
                token = str(excluded_subsystem).strip()
                if token:
                    cmd.extend(["--exclude-subsystem", token])
            env = dict(os.environ)
            env.update(config.to_env())
            completed = _run_and_stream(
                cmd,
                cwd=repo_root,
                env=env,
                progress_label="repeated_improvement",
                heartbeat_interval_seconds=float(args.child_heartbeat_seconds),
                max_silence_seconds=float(args.max_child_silence_seconds),
                max_runtime_seconds=float(args.max_child_runtime_seconds),
                max_progress_stall_seconds=float(args.max_child_progress_stall_seconds),
            )
            if int(completed["returncode"]) != 0:
                raise SystemExit(int(completed["returncode"]))
            campaign_report_path = _last_report_path(str(completed["stdout"]))
            campaign_completed = True

        if baseline_metrics is None:
            phase = "baseline_eval"
            base_report["phase"] = phase
            _write_report(report_path, base_report)
            _progress(
                f"[liftoff] phase=baseline_eval task_limit="
                f"{max(0, args.baseline_task_limit) or max(0, args.task_limit) or 0}"
            )
            baseline_eval_config = replace(config, use_tolbert_model_artifacts=False)
            baseline_metrics = run_eval(
                config=baseline_eval_config,
                progress_label="tolbert_liftoff_baseline",
                task_limit=max(0, args.baseline_task_limit) or max(0, args.task_limit) or None,
                priority_benchmark_families=[
                    str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
                ],
                **_eval_kwargs(),
            )
            baseline_trust_ledger = build_unattended_trust_ledger(baseline_eval_config)
        phase = "candidate_build"
        base_report["phase"] = phase
        base_report["baseline_metrics"] = _metric_summary(baseline_metrics)
        _write_report(report_path, base_report)
        _progress("[liftoff] phase=candidate_build")
        if not candidate_artifact:
            candidate_artifact = build_tolbert_model_candidate_artifact(
                config=config,
                repo_root=repo_root,
                output_dir=candidate_output_dir,
                metrics=baseline_metrics,
                focus=None if args.focus == "balanced" else args.focus,
                current_payload=_current_payload(config.tolbert_model_artifact_path),
            )
            candidate_artifact_path.write_text(json.dumps(candidate_artifact, indent=2), encoding="utf-8")

        if candidate_metrics is None:
            phase = "candidate_eval"
            base_report["phase"] = phase
            base_report["artifact_dataset_manifest"] = dict(candidate_artifact.get("dataset_manifest", {}))
            base_report["artifact_build_policy"] = dict(candidate_artifact.get("build_policy", {}))
            _write_report(report_path, base_report)
            _progress(
                f"[liftoff] phase=candidate_eval task_limit="
                f"{max(0, args.candidate_task_limit) or max(0, args.task_limit) or 0}"
            )
            candidate_eval_config = replace(
                config,
                tolbert_model_artifact_path=candidate_artifact_path,
                use_tolbert_model_artifacts=True,
            )
            candidate_metrics = run_eval(
                config=candidate_eval_config,
                progress_label="tolbert_liftoff_candidate",
                task_limit=max(0, args.candidate_task_limit) or max(0, args.task_limit) or None,
                priority_benchmark_families=[
                    str(value).strip() for value in args.priority_benchmark_family if str(value).strip()
                ],
                **_eval_kwargs(),
            )
            candidate_trust_ledger = build_unattended_trust_ledger(candidate_eval_config)
        phase = "liftoff_gate"
        base_report["phase"] = phase
        base_report["candidate_metrics"] = _metric_summary(candidate_metrics)
        _write_report(report_path, base_report)
        _progress("[liftoff] phase=liftoff_gate")
        gate = retained_tolbert_liftoff_gate(candidate_artifact)
        drift_step_budget = max(0, args.takeover_drift_step_budget) or int(gate.get("takeover_drift_step_budget", 0))
        drift_wave_task_limit = max(0, args.takeover_drift_wave_task_limit) or int(
            gate.get("takeover_drift_wave_task_limit", 0)
        )
        drift_max_waves = max(0, args.takeover_drift_max_waves) or int(gate.get("takeover_drift_max_waves", 0))
        if bool(gate.get("require_takeover_drift_eval", True)) and drift_step_budget > 0:
            takeover_drift_report = run_takeover_drift_eval(
                config=config,
                artifact_path=candidate_artifact_path,
                step_budget=drift_step_budget,
                wave_task_limit=drift_wave_task_limit,
                max_waves=drift_max_waves,
                eval_kwargs=_eval_kwargs(),
            ).to_dict()
        liftoff_report = build_liftoff_gate_report(
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            artifact_payload=candidate_artifact,
            candidate_trust_ledger=candidate_trust_ledger,
            baseline_trust_ledger=baseline_trust_ledger,
            takeover_drift_report=takeover_drift_report,
        )
        _emit_liftoff_summary(liftoff_report)
        if args.apply_routing:
            _apply_routing_to_artifact(candidate_artifact_path, candidate_artifact, liftoff_report)
        if args.promote_on_retain and liftoff_report.state == "retain":
            config.tolbert_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                config.tolbert_model_artifact_path,
                candidate_artifact_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        universal_decoder_eval = _run_universal_decoder_eval(
            config=config,
            candidate_artifact=candidate_artifact,
        )
    except KeyboardInterrupt:
        base_report.update(
            {
                "status": "interrupted",
                "phase": phase,
                "campaign": {
                    "requested_cycles": 0 if args.skip_improvement else max(0, args.cycles),
                    "completed": campaign_completed,
                    "campaign_report_path": str(campaign_report_path) if campaign_report_path is not None else "",
                    "adaptive_search": bool(args.adaptive_search),
                    "campaign_width": max(1, args.campaign_width),
                    "variant_width": max(1, args.variant_width),
                },
            }
        )
        if baseline_metrics is not None:
            base_report["baseline_metrics"] = _metric_summary(baseline_metrics)
        if candidate_metrics is not None:
            base_report["candidate_metrics"] = _metric_summary(candidate_metrics)
        if candidate_artifact:
            base_report["artifact_dataset_manifest"] = dict(candidate_artifact.get("dataset_manifest", {}))
            base_report["artifact_build_policy"] = dict(candidate_artifact.get("build_policy", {}))
            base_report["artifact_universal_dataset_manifest"] = dict(candidate_artifact.get("universal_dataset_manifest", {}))
        if takeover_drift_report is not None:
            base_report["takeover_drift"] = dict(takeover_drift_report)
        if universal_decoder_eval is not None:
            base_report["universal_decoder_eval"] = dict(universal_decoder_eval)
        _write_report(report_path, base_report)
        print(report_path)
        raise SystemExit(130)
    finally:
        restore_signal_handlers()

    dataset_manifest = candidate_artifact.get("dataset_manifest", {})
    build_policy = candidate_artifact.get("build_policy", {})
    runtime_paths = candidate_artifact.get("runtime_paths", {})
    checkpoint_path = Path(str(runtime_paths.get("checkpoint_path", "")).strip()) if runtime_paths.get("checkpoint_path") else None
    cache_paths = [
        Path(str(value))
        for value in runtime_paths.get("cache_paths", [])
        if str(value).strip()
    ] if isinstance(runtime_paths.get("cache_paths", []), list) else []
    report = {
        "spec_version": "asi_v1",
        "report_kind": "tolbert_liftoff_loop_report",
        "created_at": created_at.isoformat(),
        "status": "completed",
        "phase": "completed",
        "campaign": {
            "requested_cycles": 0 if args.skip_improvement else max(0, args.cycles),
            "completed": campaign_completed,
            "campaign_report_path": str(campaign_report_path) if campaign_report_path is not None else "",
            "adaptive_search": bool(args.adaptive_search),
            "campaign_width": max(1, args.campaign_width),
            "variant_width": max(1, args.variant_width),
        },
        "evaluation_scope": {
            "task_limit": max(0, args.task_limit),
            "baseline_task_limit": max(0, args.baseline_task_limit),
            "candidate_task_limit": max(0, args.candidate_task_limit),
        },
        "candidate_artifact_path": str(candidate_artifact_path),
        "retained_artifact_path": str(config.tolbert_model_artifact_path),
        "focus": args.focus,
        "dataset_readiness": {
            "total_examples": int(dataset_manifest.get("total_examples", 0) or 0),
            "trajectory_examples": int(dataset_manifest.get("trajectory_examples", 0) or 0),
            "synthetic_trajectory_examples": int(dataset_manifest.get("synthetic_trajectory_examples", 0) or 0),
            "transition_failure_examples": int(dataset_manifest.get("transition_failure_examples", 0) or 0),
            "discovered_task_examples": int(dataset_manifest.get("discovered_task_examples", 0) or 0),
            "verifier_label_examples": int(dataset_manifest.get("verifier_label_examples", 0) or 0),
            "benchmark_families": list(dataset_manifest.get("benchmark_families", []))
            if isinstance(dataset_manifest.get("benchmark_families", []), list)
            else [],
            "allow_kernel_autobuild": bool(build_policy.get("allow_kernel_autobuild", False)),
            "allow_kernel_rebuild": bool(build_policy.get("allow_kernel_rebuild", False)),
            "ready_total_examples": int(build_policy.get("ready_total_examples", 0) or 0),
            "ready_synthetic_examples": int(build_policy.get("ready_synthetic_examples", 0) or 0),
            "min_total_examples": int(build_policy.get("min_total_examples", 0) or 0),
            "min_synthetic_examples": int(build_policy.get("min_synthetic_examples", 0) or 0),
        },
        "training": {
            "job_records": list(candidate_artifact.get("job_records", []))
            if isinstance(candidate_artifact.get("job_records", []), list)
            else [],
            "external_training_backends": list(candidate_artifact.get("external_training_backends", []))
            if isinstance(candidate_artifact.get("external_training_backends", []), list)
            else [],
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
            "cache_exists": any(path.exists() for path in cache_paths),
            "runtime_paths": dict(runtime_paths) if isinstance(runtime_paths, dict) else {},
        },
        "baseline_metrics": _metric_summary(baseline_metrics),
        "candidate_metrics": _metric_summary(candidate_metrics),
        "comparison": {
            "pass_rate_delta": candidate_metrics.pass_rate - baseline_metrics.pass_rate,
            "average_steps_delta": candidate_metrics.average_steps - baseline_metrics.average_steps,
            "generated_pass_rate_delta": candidate_metrics.generated_pass_rate - baseline_metrics.generated_pass_rate,
            "proposal_selected_steps_delta": (
                candidate_metrics.proposal_selected_steps - baseline_metrics.proposal_selected_steps
            ),
            "novel_valid_command_rate_delta": (
                candidate_metrics.novel_valid_command_rate - baseline_metrics.novel_valid_command_rate
            ),
            "proposal_metrics_by_benchmark_family": _proposal_metrics_delta_by_benchmark_family(
                baseline_metrics.proposal_metrics_by_benchmark_family,
                candidate_metrics.proposal_metrics_by_benchmark_family,
            ),
            "world_feedback_calibration": _world_feedback_calibration_delta(
                baseline_metrics.world_feedback_summary,
                candidate_metrics.world_feedback_summary,
            ),
            "world_feedback_by_benchmark_family": _world_feedback_calibration_delta_by_benchmark_family(
                baseline_metrics.world_feedback_by_benchmark_family,
                candidate_metrics.world_feedback_by_benchmark_family,
            ),
            "family_takeover_summary": _family_takeover_summary(liftoff_report),
            "proposal_gate_failure_reasons_by_benchmark_family": dict(
                liftoff_report.proposal_gate_failure_reasons_by_benchmark_family
            ),
            "tolbert_primary_episodes_delta": (
                candidate_metrics.tolbert_primary_episodes - baseline_metrics.tolbert_primary_episodes
            ),
            "low_confidence_episode_pass_rate_delta": (
                candidate_metrics.low_confidence_episode_pass_rate
                - baseline_metrics.low_confidence_episode_pass_rate
            ),
        },
        "liftoff_report": liftoff_report.to_dict(),
        "takeover_drift": dict(takeover_drift_report or {}),
        "universal_decoder_eval": dict(universal_decoder_eval or {}),
        "baseline_trust": dict(baseline_trust_ledger or {}),
        "candidate_trust": dict(candidate_trust_ledger or {}),
        "artifact_build_policy": build_policy,
        "artifact_dataset_manifest": dataset_manifest,
        "artifact_universal_dataset_manifest": dict(candidate_artifact.get("universal_dataset_manifest", {})),
    }
    _write_report(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
