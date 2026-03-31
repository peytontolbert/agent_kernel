from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.cycle_runner import finalize_cycle


def _load_last_generate_record(cycles_path: Path, *, subsystem: str | None) -> dict[str, object]:
    last: dict[str, object] | None = None
    requested = str(subsystem or "").strip()
    for raw in cycles_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("state") != "generate":
            continue
        if requested and str(record.get("subsystem", "")).strip() != requested:
            continue
        last = record
    if last is None:
        hint = f" for subsystem={requested!r}" if requested else ""
        raise SystemExit(f"no generate records found in {cycles_path}{hint}")
    return last


def _load_frontier_candidate(
    frontier_path: Path,
    *,
    subsystem: str | None,
    scope_id: str | None,
    variant_id: str | None,
    candidate_index: int = 0,
) -> dict[str, object]:
    try:
        payload = json.loads(frontier_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read frontier report {frontier_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"frontier report is not valid JSON: {frontier_path}") from exc
    candidates = payload.get("frontier_candidates", [])
    if not isinstance(candidates, list):
        raise SystemExit(f"frontier report is missing frontier_candidates list: {frontier_path}")
    filtered: list[dict[str, object]] = []
    requested_subsystem = str(subsystem or "").strip()
    requested_scope_id = str(scope_id or "").strip()
    requested_variant_id = str(variant_id or "").strip()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if requested_subsystem and str(candidate.get("selected_subsystem", "")).strip() != requested_subsystem:
            continue
        if requested_scope_id and str(candidate.get("scope_id", "")).strip() != requested_scope_id:
            continue
        if requested_variant_id and str(candidate.get("selected_variant_id", "")).strip() != requested_variant_id:
            continue
        filtered.append(candidate)
    if not filtered:
        raise SystemExit(
            "no frontier candidates matched"
            f" subsystem={requested_subsystem!r} scope_id={requested_scope_id!r} variant_id={requested_variant_id!r}"
        )
    if candidate_index < 0 or candidate_index >= len(filtered):
        raise SystemExit(
            f"--candidate-index {candidate_index} is out of range for {len(filtered)} matching frontier candidates"
        )
    candidate = filtered[candidate_index]
    cycle_id = str(candidate.get("cycle_id", "")).strip()
    selected_subsystem = str(candidate.get("selected_subsystem", "")).strip()
    artifact_path = str(candidate.get("candidate_artifact_path", "")).strip()
    if not cycle_id or not selected_subsystem or not artifact_path:
        raise SystemExit(
            "frontier candidate is missing required fields: "
            f"cycle_id={cycle_id!r} subsystem={selected_subsystem!r} candidate_artifact_path={artifact_path!r}"
        )
    return {
        "cycle_id": cycle_id,
        "subsystem": selected_subsystem,
        "candidate_artifact_path": artifact_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-path", default="", help="Path to a cycles_*.jsonl file")
    parser.add_argument(
        "--frontier-report",
        default="",
        help="Optional shared frontier report from scripts/report_supervised_frontier.py",
    )
    parser.add_argument("--subsystem", default="", help="Optional filter (e.g. benchmark, retrieval, policy)")
    parser.add_argument("--scope-id", default="", help="Optional frontier filter")
    parser.add_argument("--variant-id", default="", help="Optional frontier filter")
    parser.add_argument("--candidate-index", type=int, default=0, help="Optional index into filtered frontier candidates")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only print the implied finalize command")
    parser.add_argument("--comparison-task-limit", type=int, default=0)
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    args = parser.parse_args()

    frontier_report = Path(str(args.frontier_report).strip()) if str(args.frontier_report).strip() else None
    cycles_path = Path(str(args.cycles_path).strip()) if str(args.cycles_path).strip() else None
    if frontier_report is not None:
        if not frontier_report.exists():
            raise SystemExit(f"frontier report does not exist: {frontier_report}")
        selected = _load_frontier_candidate(
            frontier_report,
            subsystem=str(args.subsystem or "").strip() or None,
            scope_id=str(args.scope_id or "").strip() or None,
            variant_id=str(args.variant_id or "").strip() or None,
            candidate_index=int(args.candidate_index),
        )
    else:
        if cycles_path is None:
            raise SystemExit("either --cycles-path or --frontier-report is required")
        if not cycles_path.exists():
            raise SystemExit(f"cycles file does not exist: {cycles_path}")
        record = _load_last_generate_record(cycles_path, subsystem=str(args.subsystem or "").strip() or None)
        selected = {
            "cycle_id": str(record.get("cycle_id", "")).strip(),
            "subsystem": str(record.get("subsystem", "")).strip(),
            "candidate_artifact_path": str(record.get("candidate_artifact_path", "")).strip(),
        }

    cycle_id = str(selected.get("cycle_id", "")).strip()
    subsystem = str(selected.get("subsystem", "")).strip()
    candidate_artifact_path = str(selected.get("candidate_artifact_path", "")).strip()

    implied_cmd = (
        f"python scripts/finalize_improvement_cycle.py"
        f" --subsystem {subsystem}"
        f" --cycle-id {cycle_id}"
        f" --artifact-path {candidate_artifact_path}"
    )
    if args.provider:
        implied_cmd += f" --provider {args.provider}"
    if args.model:
        implied_cmd += f" --model {args.model}"
    if args.include_episode_memory:
        implied_cmd += " --include-episode-memory"
    if args.include_skill_memory:
        implied_cmd += " --include-skill-memory"
    if args.include_skill_transfer:
        implied_cmd += " --include-skill-transfer"
    if args.include_operator_memory:
        implied_cmd += " --include-operator-memory"
    if args.include_tool_memory:
        implied_cmd += " --include-tool-memory"
    if args.include_verifier_memory:
        implied_cmd += " --include-verifier-memory"
    if args.include_curriculum:
        implied_cmd += " --include-curriculum"
    if args.include_failure_curriculum:
        implied_cmd += " --include-failure-curriculum"
    comparison_task_limit_arg = max(0, int(getattr(args, "comparison_task_limit", 0) or 0))
    if comparison_task_limit_arg > 0:
        implied_cmd += f" --comparison-task-limit {comparison_task_limit_arg}"

    if args.dry_run:
        print(implied_cmd)
        return

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    comparison_task_limit = comparison_task_limit_arg if comparison_task_limit_arg > 0 else None
    if comparison_task_limit is None:
        config_limit = max(0, int(getattr(config, "compare_feature_max_tasks", 0) or 0))
        if config_limit > 0:
            comparison_task_limit = config_limit
            print(
                f"comparison_task_limit={comparison_task_limit} (from config.compare_feature_max_tasks)"
            )
    state, reason = finalize_cycle(
        config=config,
        subsystem=subsystem,
        cycle_id=cycle_id,
        artifact_path=Path(candidate_artifact_path),
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_curriculum=args.include_curriculum,
        include_failure_curriculum=args.include_failure_curriculum,
        comparison_task_limit=comparison_task_limit,
    )
    print(f"cycle_id={cycle_id} subsystem={subsystem} state={state} reason={reason}")


if __name__ == "__main__":
    main()
