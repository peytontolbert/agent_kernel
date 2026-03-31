from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig


def _load_documents(root: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return documents


def _first_step(document: dict[str, object]) -> dict[str, object] | None:
    steps = document.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return None
    step = steps[0]
    return step if isinstance(step, dict) else None


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
    rows: list[dict[str, object]] = []
    for document in _load_documents(root):
        metadata = document.get("task_metadata", {})
        family = str(metadata.get("benchmark_family", "unknown"))
        if families and family not in families:
            continue
        success = bool(document.get("success", False))
        if args.failures_only and success:
            continue
        step = _first_step(document)
        if step is None:
            continue
        rows.append(
            {
                "task_id": str(document.get("task_id", "")),
                "family": family,
                "success": success,
                "termination_reason": str(document.get("termination_reason", "")),
                "path_confidence": float(step.get("path_confidence", 0.0) or 0.0),
                "trust_retrieval": bool(step.get("trust_retrieval", False)),
                "selected_skill_id": str(step.get("selected_skill_id", "") or ""),
                "selected_retrieval_span_id": str(step.get("selected_retrieval_span_id", "") or ""),
                "retrieval_ranked_skill": bool(step.get("retrieval_ranked_skill", False)),
                "action": str(step.get("action", "")),
                "content": str(step.get("content", "")),
            }
        )

    rows.sort(
        key=lambda row: (
            row["family"],
            row["success"],
            row["path_confidence"],
            row["task_id"],
        )
    )
    for row in rows[: max(1, args.limit)]:
        print(
            f"family={row['family']} task_id={row['task_id']} success={str(row['success']).lower()} "
            f"termination_reason={row['termination_reason']} path_confidence={row['path_confidence']:.2f} "
            f"trust_retrieval={str(row['trust_retrieval']).lower()} selected_skill_id={row['selected_skill_id']} "
            f"selected_retrieval_span_id={row['selected_retrieval_span_id']} "
            f"retrieval_ranked_skill={str(row['retrieval_ranked_skill']).lower()} "
            f"action={row['action']} content={json.dumps(row['content'])}"
        )


if __name__ == "__main__":
    main()
