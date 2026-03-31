from __future__ import annotations

import argparse
from pathlib import Path
import sys


MONITORED_TOLBERT_FAMILIES = ("workflow", "project", "repository", "tooling")


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _find_prefixed_fields(lines: list[str], prefix: str) -> list[dict[str, str]]:
    return [_parse_fields(line) for line in lines if line.startswith(prefix)]


def _expect_single(lines: list[str], prefix: str) -> dict[str, str]:
    matches = _find_prefixed_fields(lines, prefix)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one line starting with '{prefix}', found {len(matches)}")
    return matches[0]


def _float(fields: dict[str, str], key: str) -> float:
    return float(fields[key])


def _expect_summary(lines: list[str], prefix: str) -> dict[str, str]:
    matches = [
        fields
        for fields in _find_prefixed_fields(lines, prefix)
        if "benchmark_family" not in fields and "capability" not in fields and "mode" not in fields
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one summary line starting with '{prefix}', found {len(matches)}")
    return matches[0]


def _validate_full_eval(lines: list[str]) -> list[str]:
    failures: list[str] = []
    generated = _expect_single(lines, "generated_passed=")
    failure_recovery_total = 0
    failure_recovery_rate = 0.0
    for fields in _find_prefixed_fields(lines, "generated_kind="):
        if fields.get("generated_kind") == "failure_recovery":
            failure_recovery_total = int(fields.get("total", "0"))
            failure_recovery_rate = _float(fields, "pass_rate")
            break
    if failure_recovery_total == 0:
        failures.append("full lane did not exercise any failure_recovery generated tasks")
    elif failure_recovery_rate <= 0.0:
        failures.append("full lane failure_recovery generation collapsed to zero pass rate")
    if int(generated.get("generated_total", "0")) == 0:
        failures.append("full lane did not run generated tasks")
    return failures


def _validate_tolbert_compare(lines: list[str]) -> list[str]:
    failures: list[str] = []
    summary = _expect_summary(lines, "tolbert_compare ")
    pass_rate_delta = _float(summary, "pass_rate_delta")
    if pass_rate_delta < 0.0:
        failures.append(
            f"compare-tolbert regressed overall fixed-bank pass rate ({pass_rate_delta:.2f})"
        )
    family_deltas: dict[str, float] = {}
    for fields in _find_prefixed_fields(lines, "tolbert_compare "):
        family = fields.get("benchmark_family")
        if family:
            family_deltas[family] = _float(fields, "pass_rate_delta")
    negative_families = [
        family
        for family in MONITORED_TOLBERT_FAMILIES
        if family in family_deltas and family_deltas[family] < 0.0
    ]
    if negative_families:
        failures.append(
            "compare-tolbert showed sustained negative family deltas: "
            + ", ".join(f"{family}={family_deltas[family]:.2f}" for family in negative_families)
        )
    return failures


def _validate_tolbert_features(lines: list[str]) -> list[str]:
    failures: list[str] = []
    mode_metrics: dict[str, dict[str, float]] = {}
    for fields in _find_prefixed_fields(lines, "tolbert_mode "):
        mode = fields.get("mode")
        if mode:
            mode_metrics[mode] = {
                "pass_rate": _float(fields, "pass_rate"),
                "average_steps": _float(fields, "average_steps"),
                "retrieval_influenced_steps": _float(fields, "retrieval_influenced_steps"),
                "retrieval_ranked_skill_steps": _float(fields, "retrieval_ranked_skill_steps"),
                "retrieval_selected_steps": _float(fields, "retrieval_selected_steps"),
            }
    if "full" not in mode_metrics:
        failures.append("compare-tolbert-features did not report full mode")
        return failures
    full_rate = mode_metrics["full"]["pass_rate"]
    stronger_modes = [
        mode
        for mode, metrics in mode_metrics.items()
        if mode != "full" and metrics["pass_rate"] > full_rate
    ]
    if stronger_modes:
        failures.append(
            "full Tolbert mode underperformed isolated feature modes: "
            + ", ".join(
                f"{mode}={mode_metrics[mode]['pass_rate']:.2f}" for mode in sorted(stronger_modes)
            )
            + f" vs full={full_rate:.2f}"
        )
    signatures = {
        (
            round(metrics["pass_rate"], 4),
            round(metrics["average_steps"], 4),
            int(metrics["retrieval_influenced_steps"]),
            int(metrics["retrieval_ranked_skill_steps"]),
            int(metrics["retrieval_selected_steps"]),
        )
        for metrics in mode_metrics.values()
    }
    if len(signatures) == 1:
        failures.append("compare-tolbert-features produced identical metrics for every Tolbert mode")
    if all(metrics["retrieval_influenced_steps"] <= 0 for metrics in mode_metrics.values()):
        failures.append("compare-tolbert-features showed no retrieval influence in any Tolbert mode")
    if all(
        metrics["retrieval_ranked_skill_steps"] <= 0 and metrics["retrieval_selected_steps"] <= 0
        for metrics in mode_metrics.values()
    ):
        failures.append(
            "compare-tolbert-features showed no skill-ranking or retrieval-selection activity in any Tolbert mode"
        )
    return failures


def _validate_skill_compare(lines: list[str]) -> list[str]:
    summary = _expect_summary(lines, "skill_compare ")
    pass_rate_delta = _float(summary, "pass_rate_delta")
    if pass_rate_delta < 0.0:
        return [f"compare-skills regressed overall pass rate ({pass_rate_delta:.2f})"]
    return []


def _validate_baseline(lines: list[str]) -> list[str]:
    _expect_single(lines, "passed=")
    return []


def _validate_tolbert_first_steps(lines: list[str]) -> list[str]:
    failures: list[str] = []
    rows = _find_prefixed_fields(lines, "family=")
    if not rows:
        return ["Tolbert first-step diagnostic produced no rows"]
    confidences = [float(fields.get("path_confidence", "0.0")) for fields in rows if "path_confidence" in fields]
    trusted = [
        fields.get("trust_retrieval", "false") == "true"
        or fields.get("retrieval_ranked_skill", "false") == "true"
        or bool(fields.get("selected_skill_id"))
        or bool(fields.get("selected_retrieval_span_id"))
        for fields in rows
    ]
    if confidences and max(confidences) <= 0.0:
        failures.append("Tolbert first-step diagnostic showed zero path confidence across all reported rows")
    if trusted and not any(trusted):
        failures.append("Tolbert first-step diagnostic showed no trusted retrieval or retrieval-driven first-step selections")
    return failures


def _validate_failure_recovery(lines: list[str]) -> list[str]:
    failures: list[str] = []
    try:
        summary = _expect_summary(lines, "failure_recovery ")
    except ValueError:
        return ["failure-recovery diagnostic produced no summary"]
    total = int(summary.get("total", "0"))
    pass_rate = float(summary.get("pass_rate", "0.0"))
    if total <= 0:
        failures.append("failure-recovery diagnostic produced no recovery tasks")
    elif pass_rate <= 0.0:
        failures.append("failure-recovery diagnostic showed zero recovery pass rate")
    family_rows = _find_prefixed_fields(lines, "failure_recovery_family ")
    if not family_rows:
        failures.append("failure-recovery diagnostic produced no family summaries")
    else:
        family_pass_rates = {
            str(fields.get("benchmark_family", "")).strip(): float(fields.get("pass_rate", "0.0"))
            for fields in family_rows
            if str(fields.get("benchmark_family", "")).strip()
        }
        negative_families = [
            family
            for family in MONITORED_TOLBERT_FAMILIES
            if family in family_pass_rates and family_pass_rates[family] <= 0.0
        ]
        if negative_families:
            failures.append(
                "failure-recovery diagnostic showed zero pass rate in monitored families: "
                + ", ".join(f"{family}={family_pass_rates[family]:.2f}" for family in negative_families)
            )
    rows = _find_prefixed_fields(lines, "family=")
    if not rows:
        failures.append("failure-recovery diagnostic produced no episode rows")
        return failures
    has_guided_first_step = any(
        bool(fields.get("first_action"))
        or bool(fields.get("selected_skill_id"))
        or bool(fields.get("selected_retrieval_span_id"))
        for fields in rows
    )
    if not has_guided_first_step:
        failures.append("failure-recovery diagnostic showed no first-step guidance")
    has_reference_signal = any(
        fields.get("reference_commands", "[]") != "[]"
        or fields.get("suggested_commands", "[]") != "[]"
        or fields.get("executed_commands", "[]") != "[]"
        for fields in rows
    )
    if not has_reference_signal:
        failures.append("failure-recovery diagnostic showed no reference or suggested command signal")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-eval", type=Path, required=True)
    parser.add_argument("--tolbert-compare", type=Path, required=True)
    parser.add_argument("--tolbert-features", type=Path, required=True)
    parser.add_argument("--skill-compare", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--tolbert-first-steps", type=Path, required=True)
    parser.add_argument("--failure-recovery", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    failures.extend(_validate_full_eval(_read_lines(args.full_eval)))
    failures.extend(_validate_tolbert_compare(_read_lines(args.tolbert_compare)))
    failures.extend(_validate_tolbert_features(_read_lines(args.tolbert_features)))
    failures.extend(_validate_skill_compare(_read_lines(args.skill_compare)))
    failures.extend(_validate_baseline(_read_lines(args.baseline_eval)))
    failures.extend(_validate_tolbert_first_steps(_read_lines(args.tolbert_first_steps)))
    failures.extend(_validate_failure_recovery(_read_lines(args.failure_recovery)))

    if failures:
        print("verify_impl empirical gate failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("verify_impl empirical gates passed")


if __name__ == "__main__":
    main()
