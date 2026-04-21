from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_token() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return _utc_now().isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _normalized_families(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    families: list[str] = []
    for item in value:
        token = str(item).strip()
        if token and token not in families:
            families.append(token)
    return families


def _normalized_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw in value.items():
        token = str(key).strip()
        if not token:
            continue
        counts[token] = _safe_int(raw)
    return counts


def _status_timestamp(status: dict[str, Any]) -> str:
    for key in ("created_at", "updated_at", "last_updated_at"):
        token = str(status.get(key, "")).strip()
        if token:
            return token
    return ""


def _a8_lane_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    active = status.get("active_cycle_progress", {})
    if not isinstance(active, dict):
        active = {}
    trust = status.get("trust_breadth_summary", {})
    if not isinstance(trust, dict):
        trust = {}
    tolbert = status.get("tolbert_runtime_summary", {})
    if not isinstance(tolbert, dict):
        tolbert = {}
    tolbert_active = tolbert.get("active", {})
    if not isinstance(tolbert_active, dict):
        tolbert_active = {}
    decision = status.get("decision_state_summary", {})
    if not isinstance(decision, dict):
        decision = {}
    run_decisions = decision.get("run_decisions", {})
    if not isinstance(run_decisions, dict):
        run_decisions = {}
    closeout_modes = decision.get("run_closeout_modes", {})
    if not isinstance(closeout_modes, dict):
        closeout_modes = {}

    required_families = _normalized_families(
        trust.get("required_families")
        or status.get("priority_benchmark_families")
        or ["integration", "project", "repository", "repo_chore"]
    )
    families_with_reports = set(_normalized_families(trust.get("required_families_with_reports", [])))
    clean_task_root_counts = _normalized_counts(trust.get("required_family_clean_task_root_counts", {}))
    missing_clean_breadth = _normalized_families(trust.get("missing_required_family_clean_task_root_breadth", []))
    missing_required = _normalized_families(trust.get("missing_required_families", []))
    external_report_count = _safe_int(trust.get("external_report_count"))
    candidate_generated = bool(active.get("candidate_generated", False))
    generated_success_started = bool(active.get("generated_success_started", False))
    generated_failure_started = bool(active.get("generated_failure_started", False))
    productive_partial = bool(active.get("productive_partial", False))
    retained_gain_runs = _safe_int(status.get("retained_gain_runs"))
    runtime_managed_decisions = _safe_int(status.get("runtime_managed_decisions"))
    child_native_decisions = _safe_int(run_decisions.get("child_native"))
    natural_closeouts = _safe_int(closeout_modes.get("natural"))
    used_tolbert_successfully = bool(tolbert_active.get("used_tolbert_successfully", False))
    successful_tolbert_stages = _normalized_families(tolbert_active.get("successful_tolbert_stages", []))

    retained_conversion_state = "open"
    if retained_gain_runs > 0:
        retained_conversion_state = "retained_gain"
    elif runtime_managed_decisions > 0 and child_native_decisions > 0 and natural_closeouts > 0:
        retained_conversion_state = "decision_closure_without_retention"
    elif candidate_generated and generated_success_started and generated_failure_started and productive_partial:
        retained_conversion_state = "productive_candidate_preview"

    trust_breadth_state = "coverage_gap"
    if not missing_required and not missing_clean_breadth:
        trust_breadth_state = "counted_required_family_breadth"
    elif not missing_required:
        trust_breadth_state = "family_reports_present_breadth_gap"

    retrieval_carryover_state = "unexercised"
    if used_tolbert_successfully and "generated_failure" in successful_tolbert_stages:
        retrieval_carryover_state = "recovery_carryover_live"
    elif used_tolbert_successfully:
        retrieval_carryover_state = "tolbert_live_without_recovery_credit"

    long_horizon_state = "not_started"
    if candidate_generated and generated_success_started and generated_failure_started:
        long_horizon_state = "full_generated_path_live"
    elif candidate_generated and generated_success_started:
        long_horizon_state = "generated_success_live"
    elif candidate_generated:
        long_horizon_state = "candidate_generated"

    decision_closure_state = "open"
    if retained_gain_runs > 0:
        decision_closure_state = "retained"
    elif runtime_managed_decisions > 0 and child_native_decisions > 0 and natural_closeouts > 0:
        decision_closure_state = "child_native_natural"
    elif runtime_managed_decisions > 0:
        decision_closure_state = "runtime_managed"

    return {
        "retained_conversion": {
            "state": retained_conversion_state,
            "runtime_managed_decisions": runtime_managed_decisions,
            "retained_gain_runs": retained_gain_runs,
            "child_native_decisions": child_native_decisions,
            "natural_closeouts": natural_closeouts,
        },
        "counted_trust_breadth": {
            "state": trust_breadth_state,
            "required_families": required_families,
            "required_families_with_reports": sorted(families_with_reports),
            "missing_required_families": missing_required,
            "required_family_clean_task_root_counts": clean_task_root_counts,
            "missing_required_family_clean_task_root_breadth": missing_clean_breadth,
            "external_report_count": external_report_count,
        },
        "retrieval_carryover": {
            "state": retrieval_carryover_state,
            "used_tolbert_successfully": used_tolbert_successfully,
            "successful_tolbert_stages": successful_tolbert_stages,
        },
        "long_horizon_execution": {
            "state": long_horizon_state,
            "candidate_generated": candidate_generated,
            "generated_success_started": generated_success_started,
            "generated_failure_started": generated_failure_started,
            "productive_partial": productive_partial,
        },
        "decision_closure": {
            "state": decision_closure_state,
            "runtime_managed_decisions": runtime_managed_decisions,
            "child_native_decisions": child_native_decisions,
            "natural_closeouts": natural_closeouts,
        },
    }


def _checkpoint_summary(
    *,
    status: dict[str, Any],
    observed_at: str,
    last_progress_line: str,
    unchanged_intervals: int,
    restart_count: int,
    alive: bool,
    monitor_duration_seconds: int,
) -> dict[str, Any]:
    verification = status.get("verification_outcome_summary", {})
    if not isinstance(verification, dict):
        verification = {}
    current_task = status.get("current_task", {})
    if not isinstance(current_task, dict):
        current_task = {}
    families_sampled = _normalized_families(status.get("families_sampled", []))
    risks: list[str] = []
    if unchanged_intervals >= 2:
        risks.append("progress_stalled_for_two_intervals")
    if not alive:
        risks.append("runner_not_alive")
    if _safe_int(status.get("runtime_managed_decisions")) <= 0:
        risks.append("no_runtime_managed_decision_yet")
    if _safe_int(status.get("retained_gain_runs")) <= 0:
        risks.append("no_retained_gain_yet")
    trust = status.get("trust_breadth_summary", {})
    if not isinstance(trust, dict):
        trust = {}
    if _safe_int(trust.get("external_report_count")) <= 0:
        risks.append("external_trust_breadth_still_zero")

    return {
        "report_kind": "repeated_improvement_monitor_checkpoint",
        "observed_at": observed_at,
        "status_timestamp": _status_timestamp(status),
        "campaign_label": str(status.get("campaign_label", "")).strip(),
        "status": str(status.get("status", "")).strip() or str(status.get("state", "")).strip(),
        "finalize_phase": str(status.get("finalize_phase", "")).strip(),
        "selected_subsystem": str(status.get("selected_subsystem", "")).strip(),
        "current_task": current_task,
        "families_sampled": families_sampled,
        "verification": {
            "verified_task_count": _safe_int(verification.get("verified_task_count")),
            "failed_task_count": _safe_int(verification.get("failed_task_count")),
            "successful_task_count": _safe_int(verification.get("successful_task_count")),
        },
        "a8_lane_snapshot": _a8_lane_snapshot(status),
        "last_progress_line": last_progress_line,
        "unchanged_intervals": unchanged_intervals,
        "restart_count": restart_count,
        "alive": alive,
        "monitor_duration_seconds": monitor_duration_seconds,
        "risks": risks,
    }


def _process_lines(match_token: str) -> list[str]:
    if not match_token.strip():
        return []
    result = subprocess.run(
        ["pgrep", "-af", match_token],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _runner_alive(match_token: str) -> bool:
    return bool(_process_lines(match_token))


def _apply_campaign_label(command: list[str], label_prefix: str) -> list[str]:
    updated = list(command)
    generated_label = f"{label_prefix}_{_timestamp_token()}"
    for index, token in enumerate(updated[:-1]):
        if token == "--campaign-label":
            updated[index + 1] = generated_label
            return updated
    updated.extend(["--campaign-label", generated_label])
    return updated


def _spawn_relaunch(
    *,
    command: list[str],
    cwd: Path,
    stdout_log_path: Path,
) -> int:
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(process.pid)


def _write_checkpoint(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_token()
    (output_dir / f"checkpoint_{stamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "latest_checkpoint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_event(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "monitor_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-seconds", type=int, default=7200)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--process-match-token", default="run_repeated_improvement_cycles.py")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--campaign-label-prefix", default="")
    parser.add_argument("--restart-on-exit", action="store_true")
    parser.add_argument("campaign_command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    status_path = Path(args.status_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    cwd = Path(args.cwd).resolve()
    campaign_command = list(args.campaign_command)
    if campaign_command and campaign_command[0] == "--":
        campaign_command = campaign_command[1:]
    if args.restart_on_exit and not campaign_command:
        raise SystemExit("--restart-on-exit requires campaign_command")

    started_at = time.monotonic()
    last_progress_line = ""
    unchanged_intervals = 0
    restart_count = 0

    _append_event(
        output_dir,
        {
            "event": "monitor_started",
            "observed_at": _iso_now(),
            "status_path": str(status_path),
            "duration_seconds": int(args.duration_seconds),
            "interval_seconds": int(args.interval_seconds),
            "restart_on_exit": bool(args.restart_on_exit),
            "process_match_token": str(args.process_match_token),
            "campaign_command": campaign_command,
        },
    )

    while time.monotonic() - started_at < max(1, int(args.duration_seconds)):
        alive = _runner_alive(str(args.process_match_token))
        if not alive and args.restart_on_exit:
            relaunched_command = (
                _apply_campaign_label(campaign_command, str(args.campaign_label_prefix).strip())
                if str(args.campaign_label_prefix).strip()
                else list(campaign_command)
            )
            log_path = output_dir / f"relaunch_{_timestamp_token()}.log"
            pid = _spawn_relaunch(command=relaunched_command, cwd=cwd, stdout_log_path=log_path)
            restart_count += 1
            alive = True
            _append_event(
                output_dir,
                {
                    "event": "runner_relaunched",
                    "observed_at": _iso_now(),
                    "pid": pid,
                    "command": relaunched_command,
                    "stdout_log_path": str(log_path),
                    "restart_count": restart_count,
                },
            )
            time.sleep(2.0)

        status = _load_json(status_path)
        current_progress = str(status.get("last_progress_line", "")).strip()
        if current_progress and current_progress == last_progress_line:
            unchanged_intervals += 1
        elif current_progress:
            unchanged_intervals = 0
            last_progress_line = current_progress

        observed_at = _iso_now()
        payload = _checkpoint_summary(
            status=status,
            observed_at=observed_at,
            last_progress_line=current_progress,
            unchanged_intervals=unchanged_intervals,
            restart_count=restart_count,
            alive=alive,
            monitor_duration_seconds=int(time.monotonic() - started_at),
        )
        _write_checkpoint(output_dir, payload)
        _append_event(
            output_dir,
            {
                "event": "checkpoint_written",
                "observed_at": observed_at,
                "campaign_label": payload.get("campaign_label", ""),
                "finalize_phase": payload.get("finalize_phase", ""),
                "current_task": payload.get("current_task", {}),
                "restart_count": restart_count,
                "alive": alive,
            },
        )

        remaining = max(0.0, started_at + max(1, int(args.duration_seconds)) - time.monotonic())
        if remaining <= 0.0:
            break
        time.sleep(min(max(1, int(args.interval_seconds)), remaining))

    _append_event(
        output_dir,
        {
            "event": "monitor_finished",
            "observed_at": _iso_now(),
            "restart_count": restart_count,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
