from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.improvement.artifacts import assess_artifact_compatibility
from agent_kernel.extensions.improvement.artifact_support_evidence import (
    artifact_retrieval_reuse_evidence,
    tool_shared_repo_bundle_evidence,
)
from agent_kernel.ops.runtime_supervision import atomic_write_json


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _load_frontier(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read frontier report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"frontier report is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"frontier report is not a JSON object: {path}")
    return payload


def _load_promotion_pass(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_trust_breadth_gate(config: KernelConfig) -> dict[str, object]:
    path = Path(config.unattended_trust_ledger_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "required_families_missing_clean_task_root_breadth": [],
            "required_family_clean_task_root_counts": {},
            "family_breadth_min_distinct_task_roots": 0,
            "finalize_gate_reason": "",
        }
    if not isinstance(payload, dict):
        return {
            "required_families_missing_clean_task_root_breadth": [],
            "required_family_clean_task_root_counts": {},
            "family_breadth_min_distinct_task_roots": 0,
            "finalize_gate_reason": "",
        }
    coverage_summary = payload.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    missing_families = [
        str(value).strip()
        for value in coverage_summary.get("required_families_missing_clean_task_root_breadth", [])
        if str(value).strip()
    ]
    counts = coverage_summary.get("required_family_clean_task_root_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    threshold = _safe_int(coverage_summary.get("family_breadth_min_distinct_task_roots", 0))
    details = ", ".join(
        f"{family}:{_safe_int(counts.get(family, 0))}/{threshold}"
        for family in missing_families
    )
    return {
        "required_families_missing_clean_task_root_breadth": missing_families,
        "required_family_clean_task_root_counts": {
            family: _safe_int(counts.get(family, 0))
            for family in missing_families
        },
        "family_breadth_min_distinct_task_roots": threshold,
        "finalize_gate_reason": (
            f"bootstrap finalize still gated by required clean task-root breadth ({details})"
            if missing_families and details
            else ""
        ),
    }


def _warning_profile(candidate: dict[str, object]) -> dict[str, object]:
    warning = str(candidate.get("observation_warning", "")).strip()
    timed_out = bool(candidate.get("observation_timed_out", False))
    budget_exceeded = bool(candidate.get("observation_budget_exceeded", False))
    if timed_out:
        return {"category": "timed_out", "penalty": 1.5}
    if not warning:
        return {"category": "clean", "penalty": 0.0}
    if warning.startswith("supplemental curriculum follow-up warning:"):
        return {
            "category": "supplemental_followup_budget" if budget_exceeded else "supplemental_followup",
            "penalty": 0.35 if budget_exceeded else 0.2,
        }
    if budget_exceeded:
        return {"category": "budget_exceeded", "penalty": 0.75}
    return {"category": "runtime_warning", "penalty": 1.0}


def _long_horizon_score_components(candidate: dict[str, object]) -> dict[str, object]:
    summary = candidate.get("long_horizon_summary", {})
    if not isinstance(summary, dict):
        return {"bonus": 0.0, "task_count": 0, "world_feedback_bonus": 0.0}
    task_count = _safe_int(summary.get("task_count", 0))
    if task_count <= 0:
        return {"bonus": 0.0, "task_count": 0, "world_feedback_bonus": 0.0}
    pass_rate = _safe_float(summary.get("pass_rate", 0.0))
    novel_valid_rate = _safe_float(summary.get("novel_valid_command_rate", 0.0))
    proposal_selected_steps = _safe_int(summary.get("proposal_selected_steps", 0))
    world_feedback = summary.get("world_feedback", {})
    if not isinstance(world_feedback, dict):
        world_feedback = {}
    world_feedback_step_count = _safe_int(
        summary.get("world_feedback_step_count", world_feedback.get("step_count", 0))
    )
    progress_calibration_mae = _safe_float(world_feedback.get("progress_calibration_mae", 1.0))
    pass_bonus = min(0.75, max(0.0, pass_rate) * 0.75)
    novel_bonus = min(0.45, max(0.0, novel_valid_rate) * 0.35 + min(4, proposal_selected_steps) * 0.025)
    world_feedback_bonus = (
        max(0.0, min(0.2, 0.3 - progress_calibration_mae))
        if world_feedback_step_count > 0 and "progress_calibration_mae" in world_feedback
        else 0.0
    )
    return {
        "bonus": round(pass_bonus + novel_bonus + world_feedback_bonus, 4),
        "task_count": task_count,
        "world_feedback_bonus": round(world_feedback_bonus, 4),
    }


def _shared_repo_bundle_score_components(candidate: dict[str, object]) -> dict[str, object]:
    summary = candidate.get("shared_repo_bundle_summary", {})
    if not isinstance(summary, dict):
        return {"bonus": 0.0, "penalty": 0.0, "candidate_count": 0}

    def _count(primary: str, fallback: str) -> int:
        return _safe_int(summary.get(primary, summary.get(fallback, 0)))

    candidate_count = _count("shared_repo_candidate_count", "candidate_shared_repo_candidate_count")
    if candidate_count <= 0:
        return {"bonus": 0.0, "penalty": 0.0, "candidate_count": 0}
    worker_count = _count("shared_repo_worker_candidate_count", "candidate_shared_repo_worker_candidate_count")
    complete_candidate_count = _count("shared_repo_complete_candidate_count", "candidate_shared_repo_complete_candidate_count")
    complete_integrator_count = _count(
        "shared_repo_complete_integrator_candidate_count",
        "candidate_shared_repo_complete_integrator_candidate_count",
    )
    incomplete_integrator_count = _count(
        "shared_repo_incomplete_integrator_candidate_count",
        "candidate_shared_repo_incomplete_integrator_candidate_count",
    )
    bundle_bonus = min(0.8, complete_candidate_count * 0.15 + worker_count * 0.1 + complete_integrator_count * 0.15)
    bundle_penalty = min(0.9, incomplete_integrator_count * 0.3)
    return {
        "bonus": round(bundle_bonus, 4),
        "penalty": round(bundle_penalty, 4),
        "candidate_count": candidate_count,
    }


def _validation_family_score_components(candidate: dict[str, object]) -> dict[str, object]:
    summary = candidate.get("validation_family_summary", {})
    if not isinstance(summary, dict):
        return {"bonus": 0.0, "task_count": 0, "generated_task_count": 0, "world_feedback_bonus": 0.0}
    primary_task_count = _safe_int(summary.get("primary_task_count", 0))
    generated_task_count = _safe_int(summary.get("generated_task_count", 0))
    task_count = primary_task_count + generated_task_count
    if task_count <= 0:
        return {"bonus": 0.0, "task_count": 0, "generated_task_count": 0, "world_feedback_bonus": 0.0}
    primary_pass_rate = _safe_float(summary.get("primary_pass_rate", 0.0))
    generated_pass_rate = _safe_float(summary.get("generated_pass_rate", 0.0))
    novel_valid_rate = _safe_float(summary.get("novel_valid_command_rate", 0.0))
    proposal_selected_steps = _safe_int(summary.get("proposal_selected_steps", 0))
    world_feedback = summary.get("world_feedback", {})
    if not isinstance(world_feedback, dict):
        world_feedback = {}
    world_feedback_step_count = _safe_int(
        summary.get("world_feedback_step_count", world_feedback.get("step_count", 0))
    )
    progress_calibration_mae = _safe_float(world_feedback.get("progress_calibration_mae", 1.0))
    pass_bonus = min(
        0.85,
        max(0.0, generated_pass_rate) * 0.5
        + max(0.0, primary_pass_rate) * 0.2
        + min(4, generated_task_count) * 0.05,
    )
    novel_bonus = min(0.35, max(0.0, novel_valid_rate) * 0.25 + min(4, proposal_selected_steps) * 0.025)
    world_feedback_bonus = (
        max(0.0, min(0.15, 0.25 - progress_calibration_mae))
        if world_feedback_step_count > 0 and "progress_calibration_mae" in world_feedback
        else 0.0
    )
    return {
        "bonus": round(pass_bonus + novel_bonus + world_feedback_bonus, 4),
        "task_count": task_count,
        "generated_task_count": generated_task_count,
        "world_feedback_bonus": round(world_feedback_bonus, 4),
    }


def _retrieval_reuse_score_components(candidate: dict[str, object]) -> dict[str, object]:
    summary = candidate.get("retrieval_reuse_summary", {})
    if not isinstance(summary, dict):
        return {
            "bonus": 0.0,
            "procedure_count": 0,
            "trusted_procedure_count": 0,
            "verified_command_count": 0,
        }
    procedure_count = _safe_int(summary.get("procedure_count", 0))
    retrieval_backed_procedure_count = _safe_int(summary.get("retrieval_backed_procedure_count", 0))
    trusted_retrieval_procedure_count = _safe_int(summary.get("trusted_retrieval_procedure_count", 0))
    verified_retrieval_command_count = _safe_int(summary.get("verified_retrieval_command_count", 0))
    selected_retrieval_span_count = _safe_int(summary.get("selected_retrieval_span_count", 0))
    if procedure_count <= 0 or (
        retrieval_backed_procedure_count <= 0
        and trusted_retrieval_procedure_count <= 0
        and verified_retrieval_command_count <= 0
    ):
        return {
            "bonus": 0.0,
            "procedure_count": procedure_count,
            "trusted_procedure_count": trusted_retrieval_procedure_count,
            "verified_command_count": verified_retrieval_command_count,
        }
    retrieval_bonus = min(0.4, retrieval_backed_procedure_count * 0.12)
    trusted_bonus = min(0.3, trusted_retrieval_procedure_count * 0.18)
    verified_command_bonus = min(0.2, verified_retrieval_command_count * 0.05)
    span_bonus = min(0.1, selected_retrieval_span_count * 0.025)
    return {
        "bonus": round(retrieval_bonus + trusted_bonus + verified_command_bonus + span_bonus, 4),
        "procedure_count": procedure_count,
        "trusted_procedure_count": trusted_retrieval_procedure_count,
        "verified_command_count": verified_retrieval_command_count,
    }


def _candidate_score_components(candidate: dict[str, object]) -> dict[str, object]:
    score = 0.0
    if bool(candidate.get("generated_candidate", False)):
        score += 4.0
    if bool(candidate.get("candidate_exists", False)):
        score += 2.0
    if not bool(candidate.get("observation_timed_out", False)):
        score += 1.5
    primary_pass_rate = _safe_float(candidate.get("primary_pass_rate", 0.0))
    if primary_pass_rate <= 0.0:
        primary_pass_rate = _safe_float(candidate.get("observation_pass_rate", 0.0))
    score += min(2.5, max(0.0, primary_pass_rate) * 2.5)
    score += min(1.0, float(candidate.get("duplicate_count", 0) or 0) * 0.25)

    generated_success_rate = _safe_float(candidate.get("generated_success_pass_rate", 0.0))
    generated_success_passed = _safe_int(candidate.get("generated_success_passed", 0))
    generated_success_bonus = min(1.2, max(0.0, generated_success_rate) * 0.8 + min(4, generated_success_passed) * 0.1)
    score += generated_success_bonus

    health_bonus = 0.35 if bool(candidate.get("healthy_run", False)) else 0.0
    score += health_bonus

    long_horizon = _long_horizon_score_components(candidate)
    long_horizon_bonus = _safe_float(long_horizon.get("bonus", 0.0))
    score += long_horizon_bonus

    validation_family = _validation_family_score_components(candidate)
    validation_family_bonus = _safe_float(validation_family.get("bonus", 0.0))
    score += validation_family_bonus

    retrieval_reuse = _retrieval_reuse_score_components(candidate)
    retrieval_reuse_bonus = _safe_float(retrieval_reuse.get("bonus", 0.0))
    score += retrieval_reuse_bonus

    shared_repo_bundle = _shared_repo_bundle_score_components(candidate)
    shared_repo_bundle_bonus = _safe_float(shared_repo_bundle.get("bonus", 0.0))
    shared_repo_bundle_penalty = _safe_float(shared_repo_bundle.get("penalty", 0.0))
    score += shared_repo_bundle_bonus
    score -= shared_repo_bundle_penalty

    warning_profile = _warning_profile(candidate)
    warning_penalty = _safe_float(warning_profile.get("penalty", 0.0))
    score -= warning_penalty

    elapsed = _safe_float(candidate.get("observation_elapsed_seconds", 0.0))
    score -= min(2.0, elapsed / 30.0)
    timeout_source = str(candidate.get("observation_timeout_budget_source", "")).strip()
    timeout_source_penalty = 0.5 if timeout_source else 0.0
    score -= timeout_source_penalty

    return {
        "score": round(score, 4),
        "primary_pass_rate": primary_pass_rate,
        "generated_success_bonus": round(generated_success_bonus, 4),
        "health_bonus": round(health_bonus, 4),
        "long_horizon_bonus": round(long_horizon_bonus, 4),
        "validation_family_bonus": round(validation_family_bonus, 4),
        "retrieval_reuse_bonus": round(retrieval_reuse_bonus, 4),
        "shared_repo_bundle_bonus": round(shared_repo_bundle_bonus, 4),
        "shared_repo_bundle_penalty": round(shared_repo_bundle_penalty, 4),
        "warning_category": str(warning_profile.get("category", "clean")).strip() or "clean",
        "warning_penalty": round(warning_penalty, 4),
        "timeout_source_penalty": round(timeout_source_penalty, 4),
    }


def _candidate_score(candidate: dict[str, object]) -> float:
    return _safe_float(_candidate_score_components(candidate).get("score", 0.0))


def _validation_family_compare_guard_reasons(candidate: dict[str, object]) -> list[str]:
    payload = candidate.get("validation_family_compare_guard_reasons", [])
    if isinstance(payload, list) and payload:
        return [str(reason).strip() for reason in payload if str(reason).strip()]
    summary = candidate.get("validation_family_summary", {})
    if not isinstance(summary, dict):
        return []
    reasons: list[str] = []
    primary_task_count = _safe_int(summary.get("primary_task_count", 0))
    generated_task_count = _safe_int(summary.get("generated_task_count", 0))
    world_feedback_step_count = _safe_int(summary.get("world_feedback_step_count", 0))
    if primary_task_count > 0:
        reasons.append("validation_family_pass_rate_regressed")
    if generated_task_count > 0:
        reasons.append("validation_family_generated_pass_rate_regressed")
    if primary_task_count + generated_task_count > 0:
        reasons.append("validation_family_novel_command_rate_regressed")
    if world_feedback_step_count > 0:
        reasons.append("validation_family_world_feedback_regressed")
    return reasons


def _candidate_generated_evidence_count(candidate: dict[str, object]) -> int:
    validation_summary = candidate.get("validation_family_summary", {})
    if not isinstance(validation_summary, dict):
        validation_summary = {}
    return (
        max(0, _safe_int(candidate.get("generated_success_total", 0)))
        + max(0, _safe_int(candidate.get("generated_failure_total", 0)))
        + max(0, _safe_int(validation_summary.get("generated_task_count", 0)))
    )


def _bootstrap_review_guard_reasons(candidate: dict[str, object]) -> list[str]:
    payload = candidate.get("bootstrap_review_guard_reasons", [])
    if isinstance(payload, list) and payload:
        return [str(reason).strip() for reason in payload if str(reason).strip()]
    subsystem = str(candidate.get("selected_subsystem", "")).strip()
    artifact_kind = str(candidate.get("candidate_artifact_kind", "")).strip()
    if subsystem != "policy" and artifact_kind != "prompt_proposal_set":
        return []
    if _candidate_generated_evidence_count(candidate) > 0:
        return []
    return ["policy_bootstrap_generated_evidence_missing"]


def _candidate_history_key(candidate: dict[str, object]) -> tuple[str, str]:
    return (
        str(candidate.get("selected_subsystem", "")).strip(),
        str(candidate.get("selected_variant_id", "")).strip(),
    )


def _candidate_recency_key(candidate: dict[str, object]) -> tuple[str, str, str]:
    cycle_id = str(candidate.get("cycle_id", "")).strip()
    timestamp = ""
    if cycle_id:
        parts = cycle_id.split(":")
        if len(parts) >= 4:
            timestamp = parts[2]
    return (
        timestamp,
        cycle_id,
        str(candidate.get("scope_id", "")).strip(),
    )


def _artifact_compatibility(candidate: dict[str, object], *, repo_root: Path) -> dict[str, object]:
    subsystem = str(candidate.get("selected_subsystem", "")).strip()
    artifact_path_text = str(candidate.get("candidate_artifact_path", "")).strip()
    if not subsystem:
        return {"compatible": False, "violations": ["candidate is missing selected_subsystem"]}
    if not artifact_path_text:
        return {"compatible": False, "violations": ["candidate is missing candidate_artifact_path"]}
    artifact_path = Path(artifact_path_text)
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    if not artifact_path.exists():
        return {"compatible": True, "checked_rules": [], "violations": []}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {"compatible": False, "violations": [f"failed to read candidate artifact: {exc}"]}
    except json.JSONDecodeError:
        return {"compatible": False, "violations": ["candidate artifact is not valid JSON"]}
    compatibility = assess_artifact_compatibility(subsystem=subsystem, payload=payload)
    if not isinstance(compatibility, dict):
        return {"compatible": False, "violations": ["candidate compatibility check returned an invalid result"]}
    violations = compatibility.get("violations", [])
    if not isinstance(violations, list):
        violations = [str(violations).strip()] if str(violations).strip() else []
    return {
        "compatible": bool(compatibility.get("compatible", False)),
        "checked_rules": list(compatibility.get("checked_rules", []))
        if isinstance(compatibility.get("checked_rules", []), list)
        else [],
        "violations": [str(value).strip() for value in violations if str(value).strip()],
        "retrieval_reuse_summary": (
            artifact_retrieval_reuse_evidence(payload, subsystem=subsystem)
            if subsystem in {"skills", "tooling"}
            else {}
        ),
        "shared_repo_bundle_summary": (
            tool_shared_repo_bundle_evidence(payload) if subsystem == "tooling" else {}
        ),
    }


def _promotion_history_penalties(pass_payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    results = pass_payload.get("results", [])
    if not isinstance(results, list):
        return {}
    penalties: dict[tuple[str, str], dict[str, object]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        key = (
            str(result.get("selected_subsystem", "")).strip(),
            str(result.get("selected_variant_id", "")).strip(),
        )
        if not key[0]:
            continue
        penalty = 0.0
        reasons: list[str] = []
        compare_status = str(result.get("compare_status", "")).strip()
        finalize_state = str(result.get("finalize_state", "")).strip()
        finalize_skip_reason = str(result.get("finalize_skip_reason", "")).strip()
        if compare_status == "compare_failed":
            penalty += 1.5
            reasons.append("recent_compare_failed")
        elif compare_status == "bootstrap_first_retain":
            penalty += 0.6
            reasons.append("recent_bootstrap_first_retain")
        if finalize_skip_reason == "bootstrap_requires_review":
            penalty += 0.5
            reasons.append("recent_bootstrap_requires_review")
        if finalize_state == "reject":
            penalty += 1.25
            reasons.append("recent_finalize_reject")
        elif finalize_state == "retain":
            penalty -= 0.4
            reasons.append("recent_finalize_retain")
        existing = penalties.get(key, {"penalty": 0.0, "reasons": []})
        existing["penalty"] = round(float(existing.get("penalty", 0.0) or 0.0) + penalty, 4)
        existing_reasons = list(existing.get("reasons", []))
        existing_reasons.extend(reasons)
        existing["reasons"] = existing_reasons
        penalties[key] = existing
    return penalties


def _superseded_variant_penalties(candidates: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    penalties: dict[tuple[str, str, str], dict[str, object]] = {}
    latest_by_variant: dict[tuple[str, str], dict[str, object]] = {}
    for candidate in candidates:
        history_key = _candidate_history_key(candidate)
        if not history_key[0]:
            continue
        recency_key = _candidate_recency_key(candidate)
        if not recency_key[0]:
            continue
        existing = latest_by_variant.get(history_key)
        if existing is None or recency_key > _candidate_recency_key(existing):
            latest_by_variant[history_key] = candidate
    for candidate in candidates:
        history_key = _candidate_history_key(candidate)
        if not history_key[0]:
            continue
        recency_key = _candidate_recency_key(candidate)
        if not recency_key[0]:
            continue
        latest = latest_by_variant.get(history_key)
        if latest is None:
            continue
        latest_recency_key = _candidate_recency_key(latest)
        if recency_key == latest_recency_key:
            continue
        latest_score = _safe_float(latest.get("promotion_base_score", 0.0))
        current_score = _safe_float(candidate.get("promotion_base_score", 0.0))
        if latest_score + 0.05 < current_score:
            continue
        penalties[(history_key[0], history_key[1], str(candidate.get("scope_id", "")).strip())] = {
            "penalty": 1.5,
            "reasons": ["superseded_by_newer_same_variant_candidate"],
        }
    return penalties


def _compare_command(candidate: dict[str, object]) -> str:
    subsystem = str(candidate.get("selected_subsystem", "")).strip()
    artifact_path = str(candidate.get("candidate_artifact_path", "")).strip()
    cycle_id = str(candidate.get("cycle_id", "")).strip()
    return (
        "python scripts/compare_retained_baseline.py"
        f" --subsystem {subsystem}"
        f" --artifact-path {artifact_path}"
        f" --before-cycle-id {cycle_id}"
    )


def _finalize_command(frontier_path: Path, candidate: dict[str, object], *, candidate_index: int = 0) -> str:
    subsystem = str(candidate.get("selected_subsystem", "")).strip()
    variant_id = str(candidate.get("selected_variant_id", "")).strip()
    scope_id = str(candidate.get("scope_id", "")).strip()
    return (
        "python scripts/finalize_latest_candidate_from_cycles.py"
        f" --frontier-report {frontier_path}"
        f" --subsystem {subsystem}"
        + (f" --variant-id {variant_id}" if variant_id else "")
        + (f" --scope-id {scope_id}" if scope_id else "")
        + f" --candidate-index {candidate_index}"
        + " --dry-run"
    )


def _filtered_candidates(
    frontier: dict[str, object],
    *,
    subsystem: str | None,
    variant_id: str | None,
    include_timed_out: bool,
) -> list[dict[str, object]]:
    candidates = frontier.get("frontier_candidates", [])
    if not isinstance(candidates, list):
        return []
    requested_subsystem = str(subsystem or "").strip()
    requested_variant_id = str(variant_id or "").strip()
    filtered: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if requested_subsystem and str(candidate.get("selected_subsystem", "")).strip() != requested_subsystem:
            continue
        if requested_variant_id and str(candidate.get("selected_variant_id", "")).strip() != requested_variant_id:
            continue
        if not include_timed_out and bool(candidate.get("observation_timed_out", False)):
            continue
        if not bool(candidate.get("generated_candidate", False)):
            continue
        if not bool(candidate.get("candidate_exists", False)):
            continue
        filtered.append(dict(candidate))
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-report",
        default="",
        help="Frontier report from scripts/report_supervised_frontier.py. Defaults to the shared report path.",
    )
    parser.add_argument("--subsystem", default="", help="Optional subsystem filter")
    parser.add_argument("--variant-id", default="", help="Optional variant filter")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-subsystem", type=int, default=2)
    parser.add_argument("--include-timed-out", action="store_true")
    parser.add_argument(
        "--promotion-pass-report",
        default="",
        help="Optional promotion pass report used to penalize recently failed candidates.",
    )
    parser.add_argument("--output-path", default="", help="Optional explicit output path")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    frontier_path = (
        Path(str(args.frontier_report).strip())
        if str(args.frontier_report).strip()
        else config.improvement_reports_dir / "supervised_parallel_frontier.json"
    )
    promotion_pass_path = (
        Path(str(args.promotion_pass_report).strip())
        if str(args.promotion_pass_report).strip()
        else config.improvement_reports_dir / "supervised_frontier_promotion_pass.json"
    )
    frontier = _load_frontier(frontier_path)
    promotion_history = _promotion_history_penalties(_load_promotion_pass(promotion_pass_path))
    trust_breadth_gate = _load_trust_breadth_gate(config)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = _filtered_candidates(
        frontier,
        subsystem=str(args.subsystem or "").strip() or None,
        variant_id=str(args.variant_id or "").strip() or None,
        include_timed_out=bool(args.include_timed_out),
    )
    incompatible_candidates = 0
    compatibility_filtered: list[dict[str, object]] = []
    for candidate in candidates:
        compatibility = _artifact_compatibility(candidate, repo_root=repo_root)
        candidate["candidate_compatible"] = bool(compatibility.get("compatible", False))
        candidate["candidate_compatibility_checked_rules"] = list(compatibility.get("checked_rules", []))
        candidate["candidate_compatibility_violations"] = list(compatibility.get("violations", []))
        if not candidate.get("shared_repo_bundle_summary") and isinstance(
            compatibility.get("shared_repo_bundle_summary", {}),
            dict,
        ):
            candidate["shared_repo_bundle_summary"] = dict(compatibility.get("shared_repo_bundle_summary", {}))
        if not candidate.get("retrieval_reuse_summary") and isinstance(
            compatibility.get("retrieval_reuse_summary", {}),
            dict,
        ):
            candidate["retrieval_reuse_summary"] = dict(compatibility.get("retrieval_reuse_summary", {}))
        if not candidate["candidate_compatible"]:
            incompatible_candidates += 1
            continue
        score_components = _candidate_score_components(candidate)
        base_score = _safe_float(score_components.get("score", 0.0))
        history = promotion_history.get(_candidate_history_key(candidate), {})
        history_penalty = _safe_float(history.get("penalty", 0.0))
        candidate["promotion_base_score"] = base_score
        candidate["promotion_generated_success_bonus"] = _safe_float(score_components.get("generated_success_bonus", 0.0))
        candidate["promotion_health_bonus"] = _safe_float(score_components.get("health_bonus", 0.0))
        candidate["promotion_long_horizon_bonus"] = _safe_float(score_components.get("long_horizon_bonus", 0.0))
        candidate["promotion_validation_family_bonus"] = _safe_float(
            score_components.get("validation_family_bonus", 0.0)
        )
        candidate["promotion_retrieval_reuse_bonus"] = _safe_float(
            score_components.get("retrieval_reuse_bonus", 0.0)
        )
        candidate["promotion_shared_repo_bundle_bonus"] = _safe_float(
            score_components.get("shared_repo_bundle_bonus", 0.0)
        )
        candidate["promotion_shared_repo_bundle_penalty"] = _safe_float(
            score_components.get("shared_repo_bundle_penalty", 0.0)
        )
        candidate["promotion_warning_category"] = str(score_components.get("warning_category", "clean")).strip() or "clean"
        candidate["promotion_warning_penalty"] = _safe_float(score_components.get("warning_penalty", 0.0))
        candidate["promotion_timeout_source_penalty"] = _safe_float(score_components.get("timeout_source_penalty", 0.0))
        candidate["validation_family_compare_guard_reasons"] = _validation_family_compare_guard_reasons(candidate)
        candidate["bootstrap_review_guard_reasons"] = _bootstrap_review_guard_reasons(candidate)
        candidate["promotion_history_penalty"] = history_penalty
        candidate["promotion_history_reasons"] = list(history.get("reasons", []))
        candidate["required_families_missing_clean_task_root_breadth"] = list(
            trust_breadth_gate.get("required_families_missing_clean_task_root_breadth", [])
        )
        candidate["required_family_clean_task_root_counts"] = dict(
            trust_breadth_gate.get("required_family_clean_task_root_counts", {})
        )
        candidate["family_breadth_min_distinct_task_roots"] = _safe_int(
            trust_breadth_gate.get("family_breadth_min_distinct_task_roots", 0)
        )
        candidate["bootstrap_finalize_trust_breadth_gate_reason"] = str(
            trust_breadth_gate.get("finalize_gate_reason", "")
        ).strip()
        candidate["bootstrap_finalize_trust_breadth_gated"] = bool(
            candidate["bootstrap_finalize_trust_breadth_gate_reason"]
        )
        candidate["promotion_score"] = round(base_score - history_penalty, 4)
        compatibility_filtered.append(candidate)
    candidates = compatibility_filtered
    superseded_penalties = _superseded_variant_penalties(candidates)
    for candidate in candidates:
        superseded = superseded_penalties.get(
            (
                str(candidate.get("selected_subsystem", "")).strip(),
                str(candidate.get("selected_variant_id", "")).strip(),
                str(candidate.get("scope_id", "")).strip(),
            ),
            {},
        )
        superseded_penalty = _safe_float(superseded.get("penalty", 0.0))
        candidate["promotion_superseded_penalty"] = superseded_penalty
        superseded_reasons = list(superseded.get("reasons", []))
        candidate["promotion_superseded_reasons"] = superseded_reasons
        if superseded_penalty > 0.0:
            candidate["promotion_history_penalty"] = round(
                _safe_float(candidate.get("promotion_history_penalty", 0.0)) + superseded_penalty,
                4,
            )
            history_reasons = list(candidate.get("promotion_history_reasons", []))
            history_reasons.extend(superseded_reasons)
            candidate["promotion_history_reasons"] = history_reasons
            candidate["promotion_score"] = round(
                _safe_float(candidate.get("promotion_base_score", 0.0))
                - _safe_float(candidate.get("promotion_history_penalty", 0.0)),
                4,
            )
    candidates.sort(
        key=lambda item: (
            -_safe_float(item.get("promotion_score", 0.0)),
            _safe_float(item.get("observation_elapsed_seconds", 0.0)),
            str(item.get("scope_id", "")).strip(),
        )
    )

    selected: list[dict[str, object]] = []
    per_subsystem: dict[str, int] = {}
    for candidate in candidates:
        subsystem = str(candidate.get("selected_subsystem", "")).strip()
        if int(args.max_per_subsystem) > 0 and per_subsystem.get(subsystem, 0) >= int(args.max_per_subsystem):
            continue
        candidate["compare_command"] = _compare_command(candidate)
        candidate["finalize_command"] = _finalize_command(frontier_path, candidate, candidate_index=0)
        selected.append(candidate)
        per_subsystem[subsystem] = per_subsystem.get(subsystem, 0) + 1
        if int(args.top_k) > 0 and len(selected) >= int(args.top_k):
            break

    validation_compare_guard_counter: dict[str, int] = {}
    bootstrap_review_guard_counter: dict[str, int] = {}
    for candidate in selected:
        for reason in _validation_family_compare_guard_reasons(candidate):
            validation_compare_guard_counter[reason] = validation_compare_guard_counter.get(reason, 0) + 1
        for reason in _bootstrap_review_guard_reasons(candidate):
            bootstrap_review_guard_counter[reason] = bootstrap_review_guard_counter.get(reason, 0) + 1

    payload = {
        "report_kind": "supervised_frontier_promotion_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frontier_report_path": str(frontier_path),
        "promotion_pass_report_path": str(promotion_pass_path),
        "summary": {
            "eligible_candidates": len(candidates),
            "selected_candidates": len(selected),
            "subsystem_filter": str(args.subsystem or "").strip(),
            "variant_filter": str(args.variant_id or "").strip(),
            "top_k": int(args.top_k),
            "max_per_subsystem": int(args.max_per_subsystem),
            "history_penalized_candidates": sum(
                1 for candidate in candidates if _safe_float(candidate.get("promotion_history_penalty", 0.0)) > 0.0
            ),
            "incompatible_candidates": incompatible_candidates,
            "validation_ranked_candidates": sum(
                1
                for candidate in candidates
                if _safe_float(candidate.get("promotion_validation_family_bonus", 0.0)) > 0.0
            ),
            "retrieval_reuse_ranked_candidates": sum(
                1
                for candidate in candidates
                if _safe_float(candidate.get("promotion_retrieval_reuse_bonus", 0.0)) > 0.0
            ),
            "validation_compare_guard_reason_counts": dict(sorted(validation_compare_guard_counter.items())),
            "bootstrap_review_guard_reason_counts": dict(sorted(bootstrap_review_guard_counter.items())),
            "required_families_missing_clean_task_root_breadth": list(
                trust_breadth_gate.get("required_families_missing_clean_task_root_breadth", [])
            ),
            "required_family_clean_task_root_counts": dict(
                trust_breadth_gate.get("required_family_clean_task_root_counts", {})
            ),
            "family_breadth_min_distinct_task_roots": _safe_int(
                trust_breadth_gate.get("family_breadth_min_distinct_task_roots", 0)
            ),
            "bootstrap_finalize_trust_breadth_gate_reason": str(
                trust_breadth_gate.get("finalize_gate_reason", "")
            ).strip(),
        },
        "promotion_candidates": selected,
    }
    output_path = (
        Path(str(args.output_path).strip())
        if str(args.output_path).strip()
        else config.improvement_reports_dir / "supervised_frontier_promotion_plan.json"
    )
    atomic_write_json(output_path, payload, config=config)
    print(output_path)


if __name__ == "__main__":
    main()
