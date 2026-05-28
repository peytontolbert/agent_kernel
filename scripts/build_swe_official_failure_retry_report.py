from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.extensions.artifact_repair_contracts import classify_artifact_contract_failure_report


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _tail(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _failed_tests_from_status(path: Path, *, limit: int = 30) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    failed: list[str] = []
    for key, value in payload.items():
        if str(value).strip().lower() == "fail":
            failed.append(str(key).strip())
    test_like = [
        item
        for item in failed
        if "::" in item or "/test" in item or item.startswith("tests/") or "/tests/" in item
    ]
    selected = test_like if test_like else failed
    return selected[:limit]


def _added_patch_lines(patch_text: str, *, limit: int = 50) -> list[str]:
    added: list[str] = []
    for line in str(patch_text or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        value = line[1:].strip()
        if value:
            added.append(value)
        if len(added) >= limit:
            break
    return added


def _looks_like_shallow_constant_patch(patch_text: str) -> bool:
    added = _added_patch_lines(patch_text)
    if not added:
        return False
    if len(added) > 3:
        return False
    constant_patterns = (
        r"=\s*-?\d+(?:\.\d+)?\s*$",
        r"=\s*['\"][^'\"]{0,80}['\"]\s*$",
        r"=\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:[+\-*/]|//|%)\s*-?\d+(?:\.\d+)?\s*$",
        r"=\s*-?\d+(?:\.\d+)?\s*(?:[+\-*/]|//|%)\s*[A-Za-z_][A-Za-z0-9_]*\s*$",
        r"return\s+-?\d+(?:\.\d+)?\s*$",
        r"return\s+['\"][^'\"]{0,80}['\"]\s*$",
    )
    for line in added:
        if any(re.search(pattern, line) for pattern in constant_patterns):
            return True
    return False


def _official_failure_mode(
    *,
    fail_to_pass_failures: list[str],
    pass_to_pass_failure_count: int,
    prior_patch: str,
    post_patch_log_tail: str = "",
) -> tuple[str, str]:
    if (
        not fail_to_pass_failures
        and pass_to_pass_failure_count
        and "E2533" in str(post_patch_log_tail)
        and "Runtime " in str(post_patch_log_tail)
        and "was deprecated" in str(post_patch_log_tail)
    ):
        return (
            "official_environment_pass_to_pass_drift",
            (
                "The official fail-to-pass target passed, but pass-to-pass failed on date-sensitive runtime "
                "deprecation checks. Do not keep mutating the task patch as if the issue fix failed; classify this "
                "as evaluator/environment drift unless baseline comparison proves the patch introduced the failures."
            ),
        )
    if pass_to_pass_failure_count:
        return (
            "official_pass_to_pass_regression",
            (
                "The prior patch introduced official pass-to-pass regressions. Change strategy: preserve existing "
                "behavior first, read the failing regression context, and prefer a narrower conditional/root-cause "
                "patch over broad rewrites."
            ),
        )
    if fail_to_pass_failures and _looks_like_shallow_constant_patch(prior_patch):
        return (
            "official_shallow_constant_patch_failed",
            (
                "The prior patch was syntactically valid but appears to be a shallow constant or direct expression "
                "replacement, and official fail-to-pass tests still failed. Change strategy: do not repeat magic "
                "constants or one-line guesses; inspect the failing test intent and patch the underlying algorithm "
                "or data-flow root cause."
            ),
        )
    if fail_to_pass_failures and prior_patch:
        return (
            "official_fail_to_pass_still_failing",
            (
                "The prior patch applied but official fail-to-pass tests still failed. Change strategy: treat the "
                "official failures as the verifier target, compare the rejected patch against the failing test "
                "intent, and produce a materially different root-cause patch."
            ),
        )
    return ("official_swe_evaluator_unresolved", "")


def _artifact_repair_directive(mode: str) -> str:
    normalized = str(mode or "").strip()
    directives = {
        "artifact_repeated_official_failed_patch": (
            "The local guard rejected a patch that repeated an official-failed change. Change strategy: do not "
            "resubmit the same diff or a shallow variant. Re-read the failing test/source evidence, state a "
            "root-cause hypothesis in the executable patch choice, and produce a materially different source-grounded "
            "change."
        ),
        "artifact_invalid_python_replacement": (
            "The local guard rejected an invalid Python replacement. Change strategy: stop editing a fragment in "
            "isolation. Patch the smallest complete syntactic statement or enclosing branch/function body using exact "
            "source anchors, then verify the replacement parses before submitting."
        ),
        "artifact_policy_terminated": (
            "The local policy terminated without a valid artifact. Change strategy: do not spend the next attempt on "
            "more source inspection or prose. Use the already provided source/test evidence to emit one bounded "
            "patch-writing action that changes executable behavior."
        ),
        "artifact_inference_failure": (
            "The local model failed to produce a usable action. Change strategy: switch from open-ended reasoning to "
            "a constrained repair action: identify the failing test expectation, identify the production code path, "
            "and write one minimal patch touching that path."
        ),
        "artifact_missing_after_response": (
            "The local model responded without materializing the required artifact. Change strategy: the next action "
            "must create patch.diff directly with an allowed patch builder command or a valid unified diff; no prose "
            "or diagnostic-only response is acceptable."
        ),
        "artifact_literal_constant_assignment_guess": (
            "The local guard rejected a literal-constant assignment guess. Change strategy: do not patch by magic "
            "constant substitution. Patch the condition, data-flow, conversion, or algorithmic path that explains the "
            "failed test behavior."
        ),
        "artifact_escaped_newline_replacement": (
            "The local guard rejected a replacement containing escaped newline text. Change strategy: do not embed "
            "\\n escapes inside a single replacement argument. For multiline edits, use one complete syntactic "
            "statement range and one separate --with argument per output source line, preserving indentation."
        ),
        "artifact_shallow_one_line_patch": (
            "The local guard rejected an isolated shallow one-line edit. Change strategy: patch the surrounding "
            "control flow, data flow, or smallest complete statement range that explains the failure; do not submit "
            "single-line guesses, magic constants, or expression swaps without executable context."
        ),
        "artifact_isolated_timeout_requeue_limit": (
            "The local runner exhausted isolated timeout retries before producing a valid artifact. Change strategy: "
            "do not spend the next attempt on broad inspection or repeated failed commands. Use the existing source "
            "and failure memory to emit one bounded patch-writing action, or terminalize cleanly if no executable "
            "root-cause edit can be grounded."
        ),
    }
    return directives.get(normalized, "")


def _test_failures_from_report_section(report: dict[str, Any], section: str, *, limit: int = 30) -> list[str]:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return []
    failures = payload.get("failure", payload.get("failed", []))
    if not isinstance(failures, list):
        return []
    selected: list[str] = []
    for item in failures:
        test_id = str(item).strip()
        if test_id and test_id not in selected:
            selected.append(test_id)
        if len(selected) >= limit:
            break
    return selected


def _instance_ids_from_results(results: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    submitted = [str(value).strip() for value in results.get("submitted_ids", []) if str(value).strip()]
    failures = [str(value).strip() for value in results.get("failure_ids", []) if str(value).strip()]
    errors = [str(value).strip() for value in results.get("error_ids", []) if str(value).strip()]
    successes = {str(value).strip() for value in results.get("success_ids", []) if str(value).strip()}
    unresolved = [instance_id for instance_id in submitted if instance_id and instance_id not in successes]
    retry_ids: list[str] = []
    for instance_id in [*failures, *errors, *unresolved]:
        if instance_id and instance_id not in retry_ids:
            retry_ids.append(instance_id)
    return retry_ids, failures, errors


def _instance_ids_from_predictions(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    payload = _read_json(path)
    if isinstance(payload, dict):
        return [str(key).strip() for key in payload.keys() if str(key).strip()]
    if isinstance(payload, list):
        ids: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                instance_id = str(item.get("instance_id", "")).strip()
                if instance_id:
                    ids.append(instance_id)
        return ids
    return []


def _predictions_by_instance(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    payload = _read_json(path)
    if isinstance(payload, dict):
        predictions: dict[str, dict[str, Any]] = {}
        for instance_id, value in payload.items():
            if isinstance(value, dict):
                predictions[str(instance_id).strip()] = value
        return predictions
    if isinstance(payload, list):
        predictions = {}
        for item in payload:
            if isinstance(item, dict):
                instance_id = str(item.get("instance_id", "")).strip()
                if instance_id:
                    predictions[instance_id] = item
        return predictions
    return {}


def _instance_id_from_task_id(task_id: str) -> str:
    normalized = str(task_id or "").strip()
    if normalized.startswith("swe_patch_"):
        return normalized[len("swe_patch_") :]
    return normalized


def _terminal_queue_failures(queue_json: Path | None, known_instance_ids: set[str]) -> list[dict[str, Any]]:
    failures_by_instance = _terminal_queue_failures_by_instance(queue_json)
    failures: list[dict[str, Any]] = []
    for instance_id, failure in failures_by_instance.items():
        if instance_id in known_instance_ids:
            continue
        failures.append(failure)
        known_instance_ids.add(instance_id)
    return failures


def _terminal_queue_failures_by_instance(queue_json: Path | None) -> dict[str, dict[str, Any]]:
    if queue_json is None or not queue_json.exists() or not queue_json.is_file():
        return {}
    try:
        payload = _read_json(queue_json)
    except json.JSONDecodeError:
        return {}
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    failures: dict[str, dict[str, Any]] = {}
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict):
            continue
        state = _text(job.get("state"))
        outcome = _text(job.get("outcome"))
        if state not in {"safe_stop", "failed", "cancelled", "expired"} and outcome not in {"safe_stop", "failed"}:
            continue
        instance_id = _instance_id_from_task_id(_text(job.get("task_id")))
        if not instance_id:
            continue
        report: dict[str, Any] = {}
        report_path = Path(_text(job.get("report_path")))
        if report_path.exists() and report_path.is_file():
            try:
                report = _read_json_object(report_path)
            except (json.JSONDecodeError, SystemExit):
                report = {}
        artifact_failure = report.get("artifact_contract_failure", {}) if isinstance(report, dict) else {}
        if not isinstance(artifact_failure, dict):
            artifact_failure = {}
        if report:
            refreshed_artifact_failure = classify_artifact_contract_failure_report(report)
            if refreshed_artifact_failure.get("mode") != "not_artifact_contract":
                artifact_failure = refreshed_artifact_failure
        reason = _text(artifact_failure.get("mode")) or "; ".join(
            _text(value) for value in job.get("outcome_reasons", []) if _text(value)
        )
        artifact_directive = _artifact_repair_directive(reason)
        failures[instance_id] = {
            "instance_id": instance_id,
            "task_id": _text(job.get("task_id")) or f"swe_patch_{instance_id}",
            "state": state or outcome,
            "outcome": outcome or state,
            "reason": reason or "terminal_queue_failure",
            "artifact_repair_directive": artifact_directive,
            "resolved": False,
            "failed_tests": [],
            "fail_to_pass_failures": [],
            "pass_to_pass_failures": [],
            "pass_to_pass_failure_count": 0,
            "post_patch_log_tail": "",
            "prior_model_patch_tail": "",
            "artifact_contract_failure": artifact_failure,
            "local_report_path": str(report_path) if str(report_path) else "",
        }
    return failures


def _queue_official_feedback_by_instance(queue_json: Path | None) -> dict[str, dict[str, Any]]:
    if queue_json is None or not queue_json.exists() or not queue_json.is_file():
        return {}
    try:
        payload = _read_json(queue_json)
    except json.JSONDecodeError:
        return {}
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    feedback: dict[str, dict[str, Any]] = {}
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict):
            continue
        instance_id = _instance_id_from_task_id(_text(job.get("task_id")))
        metadata = (
            job.get("runtime_overrides", {})
            if isinstance(job.get("runtime_overrides", {}), dict)
            else {}
        ).get("task_payload", {})
        if not isinstance(metadata, dict):
            continue
        task_metadata = metadata.get("metadata", {})
        if not isinstance(task_metadata, dict):
            continue
        official_feedback = task_metadata.get("swe_official_feedback", {})
        if instance_id and isinstance(official_feedback, dict):
            feedback[instance_id] = official_feedback
    return feedback


def _rejected_patch_tails(*values: Any, limit: int = 8) -> list[str]:
    rejected: list[str] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text or text in rejected:
                continue
            rejected.append(text[-4000:])
            if len(rejected) >= limit:
                return rejected
    return rejected


def _partial_instance_ids(evaluation_results_dir: Path) -> tuple[list[str], list[str]]:
    retry_ids: list[str] = []
    success_ids: list[str] = []
    for report_path in sorted(evaluation_results_dir.glob("*/report.json")):
        try:
            report = _read_json_object(report_path)
        except (json.JSONDecodeError, SystemExit):
            continue
        instance_id = _text(report.get("instance_id")) or report_path.parent.name
        if not instance_id:
            continue
        if report.get("resolved") is True:
            success_ids.append(instance_id)
        else:
            retry_ids.append(instance_id)
    return retry_ids, success_ids


def build_official_failure_retry_report(
    *,
    results_json: Path | None = None,
    evaluation_results_dir: Path,
    output_json: Path,
    predictions_json: Path | None = None,
    queue_json: Path | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    report_mode = "final_results"
    if results_json is not None and results_json.exists():
        results = _read_json_object(results_json)
        retry_ids, failure_ids, error_ids = _instance_ids_from_results(results)
        successful_instance_ids = [
            str(value).strip()
            for value in results.get("success_ids", [])
            if str(value).strip()
        ]
        task_count = int(results.get("submitted") or len(results.get("submitted_ids", [])) or 0)
    else:
        report_mode = "partial_reports"
        retry_ids, successful_instance_ids = _partial_instance_ids(evaluation_results_dir)
        failure_ids = list(retry_ids)
        error_ids = []
        predicted_ids = _instance_ids_from_predictions(predictions_json) if predictions_json is not None else []
        task_count = len(predicted_ids) if predicted_ids else len(retry_ids) + len(successful_instance_ids)
    failed_jobs: list[dict[str, Any]] = []
    predictions_by_instance = _predictions_by_instance(predictions_json)
    terminal_queue_failures = _terminal_queue_failures_by_instance(queue_json)
    prior_feedback_by_instance = _queue_official_feedback_by_instance(queue_json)
    for instance_id in retry_ids:
        report_dir = evaluation_results_dir / instance_id
        report = {}
        report_path = report_dir / "report.json"
        if report_path.exists():
            try:
                report = _read_json_object(report_path)
            except (json.JSONDecodeError, SystemExit):
                report = {}
        fail_to_pass_failures = _test_failures_from_report_section(report, "FAIL_TO_PASS")
        pass_to_pass_failures = _test_failures_from_report_section(report, "PASS_TO_PASS", limit=60)
        pass_to_pass_failure_count = len(
            report.get("PASS_TO_PASS", {}).get("failure", [])
            if isinstance(report.get("PASS_TO_PASS"), dict)
            and isinstance(report.get("PASS_TO_PASS", {}).get("failure", []), list)
            else []
        )
        failed_tests = fail_to_pass_failures or _failed_tests_from_status(report_dir / "status.json")
        prior_prediction = predictions_by_instance.get(instance_id, {})
        prior_patch = str(prior_prediction.get("model_patch", "")).strip() if isinstance(prior_prediction, dict) else ""
        previous_feedback = prior_feedback_by_instance.get(instance_id, {})
        rejected_patch_tails = _rejected_patch_tails(
            previous_feedback.get("rejected_patch_tails", []) if isinstance(previous_feedback, dict) else [],
            previous_feedback.get("prior_model_patch_tail", "") if isinstance(previous_feedback, dict) else "",
            prior_patch,
        )
        post_patch_log_tail = _tail(report_dir / "post_patch_log.txt", max_chars=20000)
        post_patch_log_scan = _tail(report_dir / "post_patch_log.txt", max_chars=500000)
        official_mode, official_directive = _official_failure_mode(
            fail_to_pass_failures=fail_to_pass_failures,
            pass_to_pass_failure_count=pass_to_pass_failure_count,
            prior_patch=prior_patch,
            post_patch_log_tail=post_patch_log_scan or post_patch_log_tail,
        )
        failed_job = {
                "instance_id": instance_id,
                "task_id": f"swe_patch_{instance_id}",
                "state": "completed",
                "outcome": "official_unresolved",
                "reason": official_mode,
                "official_failure_mode": official_mode,
                "official_repair_directive": official_directive,
                "resolved": bool(report.get("resolved")) if "resolved" in report else False,
                "failed_tests": failed_tests,
                "fail_to_pass_failures": fail_to_pass_failures,
                "pass_to_pass_failures": pass_to_pass_failures,
                "pass_to_pass_failure_count": pass_to_pass_failure_count,
                "post_patch_log_tail": post_patch_log_tail,
                "prior_model_patch_tail": prior_patch[-4000:] if prior_patch else "",
                "rejected_patch_tails": rejected_patch_tails,
            }
        terminal_failure = terminal_queue_failures.get(instance_id, {})
        if terminal_failure:
            failed_job["local_terminal_failure"] = terminal_failure
            artifact_failure = terminal_failure.get("artifact_contract_failure", {})
            if isinstance(artifact_failure, dict) and artifact_failure:
                failed_job["artifact_contract_failure"] = artifact_failure
                artifact_mode = str(artifact_failure.get("mode", "")).strip()
                artifact_directive = _artifact_repair_directive(artifact_mode)
                if artifact_directive:
                    failed_job["artifact_repair_directive"] = artifact_directive
                if not fail_to_pass_failures and not prior_patch:
                    failed_job["reason"] = str(artifact_failure.get("mode", "")).strip() or failed_job["reason"]
        failed_jobs.append(failed_job)
    queue_failures = _terminal_queue_failures(queue_json, set(retry_ids) | set(successful_instance_ids))
    for failure in queue_failures:
        instance_id = str(failure.get("instance_id", "")).strip()
        if instance_id and instance_id not in retry_ids:
            retry_ids.append(instance_id)
        failed_jobs.append(failure)
    return {
        "report_kind": "swe_official_failure_retry_report",
        "report_mode": report_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "results_json": str(results_json or ""),
        "predictions_json": str(predictions_json or ""),
        "queue_json": str(queue_json or ""),
        "evaluation_results_dir": str(evaluation_results_dir),
        "task_count": task_count,
        "scored_instance_count": len(retry_ids) + len(successful_instance_ids),
        "successful_instance_ids": successful_instance_ids,
        "retry_instance_ids": retry_ids,
        "failed_instance_ids": failure_ids,
        "error_instance_ids": error_ids,
        "failed_patch_count": len(retry_ids),
        "failed_jobs": failed_jobs,
        "success": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", default="")
    parser.add_argument("--predictions-json", default="")
    parser.add_argument("--queue-json", default="")
    parser.add_argument("--evaluation-results-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    results_json = Path(args.results_json) if str(args.results_json).strip() else None
    predictions_json = Path(args.predictions_json) if str(args.predictions_json).strip() else None
    queue_json = Path(args.queue_json) if str(args.queue_json).strip() else None
    if results_json is None and predictions_json is None:
        raise SystemExit("provide --results-json for final reports or --predictions-json for partial reports")
    report = build_official_failure_retry_report(
        results_json=results_json,
        evaluation_results_dir=Path(args.evaluation_results_dir),
        output_json=Path(args.output_json),
        predictions_json=predictions_json,
        queue_json=queue_json,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"retry_count={len(report['retry_instance_ids'])} "
        f"success_count={len(report['successful_instance_ids'])} "
        f"output_json={output_path}"
    )


if __name__ == "__main__":
    main()
