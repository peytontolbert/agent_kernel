from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.ops.episode_store import iter_episode_documents


def _load_documents(root: Path) -> list[dict[str, object]]:
    return iter_episode_documents(root)


def _first_step(document: dict[str, object]) -> dict[str, object] | None:
    steps = document.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return None
    step = steps[0]
    return step if isinstance(step, dict) else None


def _metadata(document: dict[str, object]) -> dict[str, object]:
    metadata = document.get("task_metadata", {})
    if isinstance(metadata, dict):
        return metadata
    return {}


def _contract_metadata(document: dict[str, object]) -> dict[str, object]:
    contract = document.get("task_contract", {})
    if not isinstance(contract, dict):
        return {}
    metadata = contract.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata
    return {}


def _recovery_rows(root: Path, *, failures_only: bool, families: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for document in _load_documents(root):
        task_id = str(document.get("task_id", "")).strip()
        if not task_id:
            continue
        meta = _metadata(document)
        contract_meta = _contract_metadata(document)
        curriculum_kind = str(meta.get("curriculum_kind") or contract_meta.get("curriculum_kind") or "").strip()
        if curriculum_kind != "failure_recovery":
            continue
        family = str(meta.get("benchmark_family") or contract_meta.get("benchmark_family") or "unknown")
        if families and family not in families:
            continue
        success = bool(document.get("success", False))
        if failures_only and success:
            continue
        step = _first_step(document)
        task_contract = document.get("task_contract", {})
        suggested_commands = []
        if isinstance(task_contract, dict):
            suggested_commands = [
                str(command)
                for command in task_contract.get("suggested_commands", [])
                if str(command).strip()
            ]
        executed_commands = []
        summary = document.get("summary", {})
        if isinstance(summary, dict):
            executed_commands = [
                str(command)
                for command in summary.get("executed_commands", [])
                if str(command).strip()
            ]
        rows.append(
            {
                "task_id": task_id,
                "family": family,
                "success": success,
                "termination_reason": str(document.get("termination_reason", "")),
                "source_task": str(meta.get("parent_task") or contract_meta.get("parent_task") or ""),
                "failed_command": str(meta.get("failed_command") or contract_meta.get("failed_command") or ""),
                "failure_types": [str(value) for value in (meta.get("failure_types") or contract_meta.get("failure_types") or [])],
                "reference_task_ids": [str(value) for value in (meta.get("reference_task_ids") or contract_meta.get("reference_task_ids") or [])],
                "reference_commands": [str(value) for value in (meta.get("reference_commands") or contract_meta.get("reference_commands") or [])],
                "selected_skill_id": "" if step is None else str(step.get("selected_skill_id", "") or ""),
                "selected_retrieval_span_id": "" if step is None else str(step.get("selected_retrieval_span_id", "") or ""),
                "first_action": "" if step is None else str(step.get("action", "") or ""),
                "first_content": "" if step is None else str(step.get("content", "") or ""),
                "suggested_commands": suggested_commands,
                "executed_commands": executed_commands,
            }
        )
    rows.sort(key=lambda row: (row["family"], row["success"], row["task_id"]))
    return rows


def _summary_rows(rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    total = len(rows)
    passed = sum(1 for row in rows if bool(row["success"]))
    summary = {
        "total": total,
        "passed": passed,
        "failed": max(0, total - passed),
        "pass_rate": 0.0 if total <= 0 else passed / total,
        "families": len({str(row["family"]) for row in rows}),
    }
    family_rows: list[dict[str, object]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        matching = [row for row in rows if str(row["family"]) == family]
        family_total = len(matching)
        family_passed = sum(1 for row in matching if bool(row["success"]))
        family_rows.append(
            {
                "benchmark_family": family,
                "total": family_total,
                "passed": family_passed,
                "failed": max(0, family_total - family_passed),
                "pass_rate": 0.0 if family_total <= 0 else family_passed / family_total,
            }
        )
    return summary, family_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-root", default=None)
    parser.add_argument("--benchmark-family", action="append", default=None)
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    config = KernelConfig()
    root = Path(args.trajectories_root) if args.trajectories_root else config.trajectories_root
    families = set(args.benchmark_family or [])
    rows = _recovery_rows(root, failures_only=args.failures_only, families=families)
    summary, family_rows = _summary_rows(rows)

    print(
        f"failure_recovery total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"pass_rate={summary['pass_rate']:.2f} families={summary['families']}"
    )
    for family in family_rows:
        print(
            f"failure_recovery_family benchmark_family={family['benchmark_family']} total={family['total']} "
            f"passed={family['passed']} failed={family['failed']} pass_rate={family['pass_rate']:.2f}"
        )

    for row in rows[: max(1, args.limit)]:
        print(
            f"family={row['family']} task_id={row['task_id']} success={str(row['success']).lower()} "
            f"termination_reason={row['termination_reason']} source_task={row['source_task']} "
            f"failed_command={json.dumps(row['failed_command'])} selected_skill_id={row['selected_skill_id']} "
            f"selected_retrieval_span_id={row['selected_retrieval_span_id']} first_action={row['first_action']} "
            f"first_content={json.dumps(row['first_content'])} "
            f"reference_task_ids={json.dumps(row['reference_task_ids'])} "
            f"reference_commands={json.dumps(row['reference_commands'])} "
            f"suggested_commands={json.dumps(row['suggested_commands'])} "
            f"executed_commands={json.dumps(row['executed_commands'])}"
        )


if __name__ == "__main__":
    main()
