#!/usr/bin/env python3
"""Backtest benchmark failure diagnosis against existing AgentKernel reports.

This script does not re-run benchmark tasks. It audits completed failure reports
and asks whether the supervision runbook can classify the failure and point to a
concrete improvement path using the local AgentKernel and OpenClaw/Hermes skill
datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/agentkernel")
DEFAULT_REPORTS_ROOT = ROOT / "benchmarks/swe_bench_live/autonomous_harness_runs"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/benchmark_failure_backtests"
DEFAULT_AGENTKERNEL_SKILLS = Path(
    "/data/repo_skills_miner/artifacts/hf_agentkernel_source_skills/data/train.parquet"
)
DEFAULT_HARNESS_SKILLS = Path(
    "/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet"
)


ROOT_CAUSE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "patch_materialization_failed",
        (
            "missing expected file",
            "missing patch.diff",
            "swe patch verifier missing patch file",
            "artifact_missing_after_response",
        ),
    ),
    (
        "policy_loop_failure",
        (
            "policy_terminated",
            "policy terminated",
            "no_state_progress",
            "repeated_failed_action",
            "diagnostic context budget",
            "artifact_guard_backoff",
            "max_steps_reached",
        ),
    ),
    (
        "invalid_patch_syntax",
        (
            "invalid_python",
            "invalid python",
            "malformed",
            "does not apply",
            "apply check",
            "definition_header_removal",
            "definition header",
        ),
    ),
    (
        "semantic_patch_wrong",
        (
            "no meaningful content change",
            "placeholder",
            "fake",
            "constant",
            "docstring-only",
            "comment-only",
            "statement_kind_replacement",
        ),
    ),
    (
        "localization_wrong",
        (
            "off_target",
            "wrong span",
            "unexpected_change_files",
            "unexpected file",
            "outside the fail-to-pass function",
            "different span",
        ),
    ),
    (
        "sandbox_tool_failure",
        (
            "exit code was 126",
            "blocked unsupported shell operator",
            "timed_out",
            "timeout",
            "command_failure",
            "runtime_or_command_failure",
        ),
    ),
    (
        "retrieval_wrong",
        (
            "retrieval_influenced",
            "trusted_retrieval",
            "selected_retrieval_span",
        ),
    ),
]


RECOMMENDATIONS: dict[str, str] = {
    "patch_materialization_failed": (
        "Add or strengthen artifact materialization fallback. Force one real "
        "builder command before diagnostic-context budget is exhausted."
    ),
    "policy_loop_failure": (
        "Add a regression test for the loop/guard transition and require a "
        "state-changing action after repeated virtual context reads."
    ),
    "invalid_patch_syntax": (
        "Harden patch builder/syntax guard with complete statement ranges and "
        "reject def/class header mutation unless explicitly required."
    ),
    "semantic_patch_wrong": (
        "Add semantic patch sanity checks: reject placeholder constants, no-op "
        "edits, comment-only/docstring-only edits, and off-contract behavior."
    ),
    "localization_wrong": (
        "Constrain repair candidates to fail-to-pass functions/files before "
        "re-anchoring to broader source landmarks."
    ),
    "sandbox_tool_failure": (
        "Normalize allowed shell forms and avoid virtual-context heredocs where "
        "the command governance layer blocks redirection/operators."
    ),
    "retrieval_absent": (
        "Build a failure retrieval row and add hard negatives so ToLBERT can "
        "activate trusted retrieval for this task shape."
    ),
    "retrieval_wrong": (
        "Inspect selected skill spans and add hard negatives for semantically "
        "similar but operationally wrong skills."
    ),
    "unknown": "Inspect checkpoint history manually and add a new root-cause rule.",
}

_TERM_COUNT_CACHE: dict[tuple[str, str], int] = {}


@dataclass
class BacktestRow:
    task_id: str
    report_path: str
    generated_at: str
    success: bool
    outcome: str
    termination_reason: str
    failure_reason: str
    primary_root_cause: str
    contributing_root_causes: list[str]
    trusted_retrieval_steps: int
    retrieval_influenced_steps: int
    selected_retrieval_span_count: int
    patch_materialized: bool
    hidden_side_effect_risk: bool
    agentkernel_skill_hits: int
    harness_skill_hits: int
    likely_helpful: bool
    recommendation: str
    evidence: list[str]


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _text_blob(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _report_generated_at(obj: dict[str, Any], path: Path) -> str:
    raw = str(obj.get("generated_at") or "").strip()
    if raw:
        return raw
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _is_unattended_report(obj: dict[str, Any]) -> bool:
    if obj.get("report_kind") == "unattended_task_report":
        return True
    return "task_id" in obj and "summary" in obj and "outcome" in obj


def _iter_reports(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/reports/*.json*")
        if path.is_file() and "job_report" in path.name
    )


def _skill_index(path: Path, *, include_source_excerpt: bool = False) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return "", 0
    try:
        columns = ["source_path", "qualname", "primitive_type"]
        if include_source_excerpt:
            columns.append("source_excerpt")
        df = pd.read_parquet(path, columns=[c for c in columns])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return "", 0
    candidates = ["source_path", "qualname", "primitive_type", "llm_summary", "labels"]
    if include_source_excerpt:
        candidates.append("source_excerpt")
    usable = [c for c in candidates if c in df]
    if not usable:
        return "", len(df)
    return "\n".join(
        df[usable]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    ), len(df)


def _query_terms(obj: dict[str, Any], roots: list[str]) -> list[str]:
    text = _text_blob(
        {
            "prompt": obj.get("prompt") or obj.get("task_contract", {}).get("prompt", ""),
            "failure": obj.get("failure_reason"),
            "artifact": obj.get("artifact_contract_failure"),
            "outcome": obj.get("outcome"),
        }
    ).lower()
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", text)
    stop = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "must",
        "patch",
        "diff",
        "file",
        "task",
        "repo",
        "source",
        "line",
        "format",
        "expected",
    }
    counts = Counter(t for t in tokens if t not in stop)
    terms = [term for term, _ in counts.most_common(10)]
    terms.extend(roots)
    return sorted(set(terms))


def _hit_count(index: str, terms: list[str], *, index_name: str, limit: int = 20) -> int:
    if not index or not terms:
        return 0
    needles = [t.lower() for t in terms if len(t) >= 4][:limit]
    total = 0
    for needle in needles:
        key = (index_name, needle)
        if key not in _TERM_COUNT_CACHE:
            _TERM_COUNT_CACHE[key] = min(index.count(needle), 25)
        total += _TERM_COUNT_CACHE[key]
    return total


def _classify(obj: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    text = _text_blob(
        {
            "outcome": obj.get("outcome"),
            "termination_reason": obj.get("termination_reason"),
            "failure_reason": obj.get("failure_reason"),
            "artifact_contract_failure": obj.get("artifact_contract_failure"),
            "outcome_reasons": obj.get("outcome_reasons"),
            "acceptance_packet": obj.get("acceptance_packet"),
            "policy_trace": obj.get("policy_trace"),
            "summary": summary,
        }
    ).lower()
    labels: list[str] = []
    evidence: list[str] = []
    for label, needles in ROOT_CAUSE_RULES:
        matched = [needle for needle in needles if needle in text]
        if matched:
            labels.append(label)
            evidence.extend(f"{label}:{needle}" for needle in matched[:3])
    trusted = _as_int(summary.get("trusted_retrieval_steps"))
    selected = obj.get("selected_retrieval_span_ids")
    selected_count = len(selected) if isinstance(selected, list) else 0
    if trusted == 0 and selected_count == 0:
        labels.append("retrieval_absent")
        evidence.append("retrieval_absent:trusted_retrieval_steps=0")
    if not labels:
        labels.append("unknown")
    order = [
        "patch_materialization_failed",
        "invalid_patch_syntax",
        "semantic_patch_wrong",
        "localization_wrong",
        "sandbox_tool_failure",
        "policy_loop_failure",
        "retrieval_absent",
        "retrieval_wrong",
        "unknown",
    ]
    labels = sorted(set(labels), key=lambda item: order.index(item) if item in order else 999)
    return labels[0], labels[1:], evidence[:16]


def _patch_materialized(obj: dict[str, Any]) -> bool:
    packet = obj.get("acceptance_packet") if isinstance(obj.get("acceptance_packet"), dict) else {}
    selected = packet.get("selected_edits")
    if isinstance(selected, list) and selected:
        return True
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    if _as_int(summary.get("created_files")) > 0 and "missing expected file" not in _text_blob(packet).lower():
        return True
    return False


def backtest(
    reports_root: Path,
    output_dir: Path,
    agentkernel_skills: Path,
    harness_skills: Path,
    include_source_excerpt: bool = False,
) -> tuple[list[BacktestRow], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agentkernel_index, agentkernel_row_count = _skill_index(
        agentkernel_skills,
        include_source_excerpt=include_source_excerpt,
    )
    harness_index, harness_row_count = _skill_index(
        harness_skills,
        include_source_excerpt=include_source_excerpt,
    )

    rows: list[BacktestRow] = []
    for path in _iter_reports(reports_root):
        obj = _json_load(path)
        if not obj or not _is_unattended_report(obj):
            continue
        if bool(obj.get("success")):
            continue
        summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
        primary, contributing, evidence = _classify(obj)
        terms = _query_terms(obj, [primary, *contributing])
        ak_hits = _hit_count(agentkernel_index, terms, index_name="agentkernel")
        harness_hits = _hit_count(harness_index, terms, index_name="harness")
        trusted = _as_int(summary.get("trusted_retrieval_steps"))
        influenced = _as_int(summary.get("retrieval_influenced_steps"))
        selected = obj.get("selected_retrieval_span_ids")
        selected_count = len(selected) if isinstance(selected, list) else 0
        likely_helpful = (
            primary != "unknown"
            and (
                primary in {
                    "patch_materialization_failed",
                    "policy_loop_failure",
                    "invalid_patch_syntax",
                    "semantic_patch_wrong",
                    "localization_wrong",
                }
                or ak_hits > 0
                or harness_hits > 0
            )
        )
        rows.append(
            BacktestRow(
                task_id=str(obj.get("task_id") or ""),
                report_path=str(path),
                generated_at=_report_generated_at(obj, path),
                success=False,
                outcome=str(obj.get("outcome") or ""),
                termination_reason=str(obj.get("termination_reason") or ""),
                failure_reason=str(obj.get("failure_reason") or ""),
                primary_root_cause=primary,
                contributing_root_causes=contributing,
                trusted_retrieval_steps=trusted,
                retrieval_influenced_steps=influenced,
                selected_retrieval_span_count=selected_count,
                patch_materialized=_patch_materialized(obj),
                hidden_side_effect_risk=bool(summary.get("hidden_side_effect_risk")),
                agentkernel_skill_hits=ak_hits,
                harness_skill_hits=harness_hits,
                likely_helpful=likely_helpful,
                recommendation=RECOMMENDATIONS.get(primary, RECOMMENDATIONS["unknown"]),
                evidence=evidence,
            )
        )

    by_task_latest: dict[str, BacktestRow] = {}
    for row in rows:
        if row.task_id not in by_task_latest or row.generated_at > by_task_latest[row.task_id].generated_at:
            by_task_latest[row.task_id] = row
    latest_rows = list(by_task_latest.values())

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_root": str(reports_root),
        "total_failed_reports": len(rows),
        "unique_failed_tasks": len(latest_rows),
        "agentkernel_skill_rows_loaded": agentkernel_row_count,
        "harness_skill_rows_loaded": harness_row_count,
        "skill_coverage_mode": "metadata+source_excerpt" if include_source_excerpt else "metadata_only",
        "root_cause_counts_all_reports": Counter(row.primary_root_cause for row in rows),
        "root_cause_counts_latest_by_task": Counter(row.primary_root_cause for row in latest_rows),
        "likely_helpful_all_reports": sum(1 for row in rows if row.likely_helpful),
        "likely_helpful_latest_by_task": sum(1 for row in latest_rows if row.likely_helpful),
        "retrieval_absent_all_reports": sum(
            1 for row in rows if row.primary_root_cause == "retrieval_absent" or "retrieval_absent" in row.contributing_root_causes
        ),
        "zero_trusted_retrieval_all_reports": sum(1 for row in rows if row.trusted_retrieval_steps == 0),
    }
    summary["root_cause_counts_all_reports"] = dict(summary["root_cause_counts_all_reports"])
    summary["root_cause_counts_latest_by_task"] = dict(summary["root_cause_counts_latest_by_task"])

    def write_outputs(name: str, data: list[BacktestRow]) -> None:
        records = [asdict(row) for row in data]
        (output_dir / f"{name}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(records).to_parquet(output_dir / f"{name}.parquet", index=False)
        except Exception:
            pass

    write_outputs("all_failed_report_backtest", rows)
    write_outputs("latest_failed_task_backtest", latest_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown_summary(summary, latest_rows), encoding="utf-8")
    return rows, summary


def _markdown_summary(summary: dict[str, Any], latest_rows: list[BacktestRow]) -> str:
    lines = [
        "# Benchmark Failure Backtest",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Reports root: `{summary['reports_root']}`",
        "",
        "## Coverage",
        "",
        f"- Failed report artifacts analyzed: {summary['total_failed_reports']}",
        f"- Unique failed tasks: {summary['unique_failed_tasks']}",
        f"- AgentKernel skill rows loaded: {summary['agentkernel_skill_rows_loaded']}",
        f"- OpenClaw/Hermes skill rows loaded: {summary['harness_skill_rows_loaded']}",
        f"- Reports with zero trusted retrieval: {summary['zero_trusted_retrieval_all_reports']}",
        f"- Reports where the runbook likely helps: {summary['likely_helpful_all_reports']}",
        "",
        "## Root Cause Counts",
        "",
    ]
    for key, value in sorted(summary["root_cause_counts_all_reports"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Latest Failed Task Rows", ""])
    for row in sorted(latest_rows, key=lambda item: item.task_id)[:100]:
        lines.extend(
            [
                f"### {row.task_id}",
                "",
                f"- Primary root cause: `{row.primary_root_cause}`",
                f"- Contributing: `{', '.join(row.contributing_root_causes) or 'none'}`",
                f"- Trusted retrieval steps: {row.trusted_retrieval_steps}",
                f"- AgentKernel skill hits: {row.agentkernel_skill_hits}",
                f"- OpenClaw/Hermes skill hits: {row.harness_skill_hits}",
                f"- Likely helpful: {row.likely_helpful}",
                f"- Recommendation: {row.recommendation}",
                f"- Report: `{row.report_path}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--agentkernel-skills", type=Path, default=DEFAULT_AGENTKERNEL_SKILLS)
    parser.add_argument("--harness-skills", type=Path, default=DEFAULT_HARNESS_SKILLS)
    parser.add_argument(
        "--include-source-excerpt",
        action="store_true",
        help="Include source_excerpt text in skill coverage checks. Slower, useful for offline deep backtests.",
    )
    args = parser.parse_args()

    rows, summary = backtest(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        agentkernel_skills=args.agentkernel_skills,
        harness_skills=args.harness_skills,
        include_source_excerpt=args.include_source_excerpt,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(rows)} failed-report rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
