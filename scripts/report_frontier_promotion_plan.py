from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

from agent_kernel.config import KernelConfig


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def _candidate_score(candidate: dict[str, object]) -> float:
    score = 0.0
    if bool(candidate.get("generated_candidate", False)):
        score += 4.0
    if bool(candidate.get("candidate_exists", False)):
        score += 2.0
    if not bool(candidate.get("observation_timed_out", False)):
        score += 1.5
    score += min(1.0, float(candidate.get("duplicate_count", 0) or 0) * 0.25)
    elapsed = _safe_float(candidate.get("observation_elapsed_seconds", 0.0))
    score -= min(2.0, elapsed / 30.0)
    timeout_source = str(candidate.get("observation_timeout_budget_source", "")).strip()
    if timeout_source:
        score -= 0.5
    return round(score, 4)


def _candidate_history_key(candidate: dict[str, object]) -> tuple[str, str]:
    return (
        str(candidate.get("selected_subsystem", "")).strip(),
        str(candidate.get("selected_variant_id", "")).strip(),
    )


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
    candidates = _filtered_candidates(
        frontier,
        subsystem=str(args.subsystem or "").strip() or None,
        variant_id=str(args.variant_id or "").strip() or None,
        include_timed_out=bool(args.include_timed_out),
    )
    for candidate in candidates:
        base_score = _candidate_score(candidate)
        history = promotion_history.get(_candidate_history_key(candidate), {})
        history_penalty = _safe_float(history.get("penalty", 0.0))
        candidate["promotion_base_score"] = base_score
        candidate["promotion_history_penalty"] = history_penalty
        candidate["promotion_history_reasons"] = list(history.get("reasons", []))
        candidate["promotion_score"] = round(base_score - history_penalty, 4)
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
        },
        "promotion_candidates": selected,
    }
    output_path = (
        Path(str(args.output_path).strip())
        if str(args.output_path).strip()
        else config.improvement_reports_dir / "supervised_frontier_promotion_plan.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
