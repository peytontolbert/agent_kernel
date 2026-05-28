#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_VERSION = "agentkernel_swe_live_failure_retrieval_dataset_v1"


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _operator_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "agentkernel.swe_live.stale_verification_freshness_guard",
            "failure_family": "stale_queue_verification",
            "query": (
                "SWE-live run emitted empty model_patch predictions because collect_predictions consumed an old "
                "patch_jobs_verification.json before later jobs completed successfully. Need queue snapshot freshness, "
                "verified patch export, checkpoint reconciliation, and safe retry after stale queued/in_progress state."
            ),
            "doc": (
                "skill_id: agentkernel.swe_live.stale_verification_freshness_guard\n"
                "name: Guard SWE patch collection against stale verification snapshots\n"
                "trigger_conditions: patch_jobs_verification is older than queue.json, job reports, or checkpoint files; "
                "collect_predictions copied zero patches while later reports contain verification_passed; apply check "
                "returns empty_patch_noop for instances with completed patch jobs.\n"
                "patch_operator: before collecting predictions, compare verification timestamp with queue/job report mtimes "
                "and current queue state; rerun verify_swe_bench_patch_jobs when newer successful jobs exist; refuse to "
                "emit empty no-op predictions from stale verification unless explicitly requested.\n"
                "verification_pattern: create stale verification with queued jobs, then update queue to completed success "
                "with patch.diff; assert collection reruns verification or fails closed instead of writing empty patches.\n"
                "anti_patterns: trusting old patch_jobs_verification.json blindly; treating empty_patch_noop as success; "
                "collecting predictions while jobs are in_progress."
            ),
        },
        {
            "source_id": "agentkernel.swe_live.success_continuation_collect",
            "failure_family": "late_patch_success_collection",
            "query": (
                "SWE-live jobs produced patch.diff after the initial scoring attempt, but predictions were not recollected. "
                "Need a success-continuation collector that exports newly verified patches without overwriting original run outputs."
            ),
            "doc": (
                "skill_id: agentkernel.swe_live.success_continuation_collect\n"
                "name: Collect late successful SWE-live patches into a fresh predictions file\n"
                "trigger_conditions: a run has empty predictions, later job reports show verification_passed, or fresh "
                "verify_swe_bench_patch_jobs finds successful_instance_ids absent from the submitted predictions file.\n"
                "patch_operator: write a new predictions JSONL path under artifacts or a retry label; include verified "
                "patches, preserve abstentions for semantic failures, and leave original submitted predictions immutable.\n"
                "verification_pattern: fresh verification finds two successful patches and one abstention; collect emits "
                "non-empty model_patch for only successful instances and empty patches for abstentions.\n"
                "anti_patterns: mutating historical leaderboard submission files; copying patches for failed/in_progress jobs."
            ),
        },
        {
            "source_id": "agentkernel.swe_live.official_harness_python_preflight",
            "failure_family": "official_harness_environment",
            "query": (
                "SWE-bench-Live official_harness failed before scoring with ModuleNotFoundError No module named docker. "
                "The generated live retry harness used the wrong Python interpreter. Need dependency preflight and configured python_bin preservation."
            ),
            "doc": (
                "skill_id: agentkernel.swe_live.official_harness_python_preflight\n"
                "name: Preserve configured Python and preflight Docker dependency for SWE-bench-Live\n"
                "trigger_conditions: official_harness imports launch.core.runtime SetupRuntime and fails importing docker; "
                "source run_config python_bin points at an environment that imports docker but retry generation replaces it "
                "with sys.executable or a base conda interpreter.\n"
                "patch_operator: keep source_harness.run_config.python_bin for live retry specs; add preflight_argv "
                "`python -c 'import docker'` to official_harness or harness status; surface a blocking environment failure "
                "before scoring starts.\n"
                "verification_pattern: build a live retry harness with python_bin=/envs/ai/bin/python and assert official "
                "phase argv starts with that path; simulate missing docker preflight failure.\n"
                "anti_patterns: forcing sys.executable for live retries; discovering missing docker only inside official scoring."
            ),
        },
        {
            "source_id": "agentkernel.swe_live.patchdiff_source_path_repair",
            "failure_family": "artifact_materialization_loop",
            "query": (
                "python-babel remains in artifact_materialization_guard with missing patch.diff. The rejected retry command "
                "uses swe_patch_builder --path patch.diff as the source path. Need repair to use candidate source path from source-lines."
            ),
            "doc": (
                "skill_id: agentkernel.swe_live.patchdiff_source_path_repair\n"
                "name: Repair SWE patch-builder plans that use patch.diff as source path\n"
                "trigger_conditions: artifact_missing_after_response; retry_command contains `--path patch.diff`; verifier "
                "reports missing expected file patch.diff; policy repeatedly records virtual artifact context without command execution.\n"
                "patch_operator: reject builder commands with patch.diff as source path before sandbox execution; infer the "
                "real source file from --source-lines or candidate source context, then regenerate `swe_patch_builder --path <source-file> ... > patch.diff`.\n"
                "verification_pattern: feed a failed Babel-style retry command and assert the repaired command targets "
                "babel/dates.py, not patch.diff, and produces a non-empty unified diff.\n"
                "anti_patterns: allowing --path patch.diff; repeatedly adding no-op virtual context instead of executing a corrected builder command."
            ),
        },
        {
            "source_id": "agentkernel.swe_live.semantic_artifact_abstention",
            "failure_family": "weak_semantic_patch",
            "query": (
                "A SWE-live patch applies and patch.diff exists, but semantic artifact verification rejects it as an isolated "
                "one-line production replacement without enough repair structure. Need abstention, retry labeling, and stronger patch strategy."
            ),
            "doc": (
                "skill_id: agentkernel.swe_live.semantic_artifact_abstention\n"
                "name: Treat weak semantic patch verifier failures as retryable abstentions\n"
                "trigger_conditions: verify_swe_bench_patch_jobs reports semantic_artifact_failure after a completed success; "
                "verification reasons mention isolated one-line production replacement or insufficient repair structure.\n"
                "patch_operator: keep the patch out of official predictions, classify as abstained/retryable, and feed the "
                "failure reason into a retry task that asks for broader behavioral repair and relevant tests.\n"
                "verification_pattern: completed job with patch.diff and semantic failure becomes abstained_instance_ids, "
                "not successful_instance_ids, and collect_predictions emits empty model_patch only for that abstention.\n"
                "anti_patterns: submitting semantically rejected patches; flattening semantic failures into generic empty no-op."
            ),
        },
    ]


def _row_from_operator(operator: dict[str, Any], *, repeat_index: int, source_run: str) -> dict[str, Any]:
    query = (
        "<AK_USER> Diagnose an AgentKernel SWE-live harness failure and retrieve the smallest repair operator.\n"
        f"failure family: {operator['failure_family']}\n"
        f"{operator['query']}\n"
        "<AK_RETRIEVE> <AK_RET_SKILLS> <AK_RET_SEMANTIC>"
    )
    negatives = [item["doc"] for item in _operator_rows() if item["source_id"] != operator["source_id"]]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_run": source_run,
        "failure_family": operator["failure_family"],
        "operator_id": operator["source_id"],
        "repeat_index": repeat_index,
    }
    return {
        "source_id": f"{operator['source_id']}:{repeat_index:05d}",
        "encoder_text": query,
        "decoder_text": "<AK_GATHER_CONTEXT> <AK_RETRIEVE> <AK_RET_SKILLS> <AK_CONF_HIGH>",
        "action": "gather_context",
        "task_type": "swe_live_failure_retrieval",
        "weight": 0.0,
        "distill_loss_weight": 0.0,
        "retrieval_query_text": query,
        "retrieval_doc_text": operator["doc"],
        "retrieval_negative_doc_texts": _json(negatives),
        "retrieval_loss_weight": 1.0,
        "query_confidence_target": 0.95,
        "retrieval_coverage_target": 0.95,
        "ood_query_target": 0.05,
        "ood_evidence_target": 0.05,
        "answer_confidence_target": 0.9,
        "needs_verification_target": 0.8,
        "paper_action_validity_target": 1.0,
        "metadata": _json(metadata),
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("writing SWE-live failure retrieval Parquet requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    repeat = max(1, int(args.repeat))
    source_run = str(Path(args.source_run).expanduser().resolve()) if _text(args.source_run) else ""
    operators = _operator_rows()
    rows = [
        _row_from_operator(operator, repeat_index=index, source_run=source_run)
        for operator in operators
        for index in range(repeat)
    ]
    eval_rows = []
    train_rows = []
    for row in rows:
        repeat_index = json.loads(row["metadata"])["repeat_index"]
        if repeat_index == 0:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows:
        train_rows = list(rows)
        eval_rows = list(rows)
    train_path = output_dir / "train" / "part-00000.parquet"
    eval_path = output_dir / "eval" / "part-00000.parquet"
    _write_parquet(train_path, train_rows)
    _write_parquet(eval_path, eval_rows)
    counts = Counter(row["task_type"] for row in rows)
    manifest_path = output_dir / "agentkernel_swe_live_failure_retrieval_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "schema_version": SCHEMA_VERSION,
        "objective": "swe_live_failure_retrieval",
        "dataset_format": "parquet",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path.parent),
        "eval_dataset_path": str(eval_path.parent),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "retrieval_pair_count": len(rows),
        "operator_count": len(operators),
        "repeat": repeat,
        "source_run": source_run,
        "task_type_counts": dict(counts),
        "action_counts": {"gather_context": len(rows)},
        "created_at": datetime.now(UTC).isoformat(),
        "agentkernel_special_tokens": [
            "<AK_USER>",
            "<AK_RETRIEVE>",
            "<AK_RET_SKILLS>",
            "<AK_RET_SEMANTIC>",
            "<AK_GATHER_CONTEXT>",
            "<AK_CONF_HIGH>",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        default="benchmarks/swe_bench_live/autonomous_harness_runs/official_score_feedback_r42_20260514",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/agentkernel_lite_encdec/swe_live_failure_retrieval_dataset",
    )
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
