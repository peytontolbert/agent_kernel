from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.curriculum_improvement import build_curriculum_proposal_artifact
from evals.harness import run_eval


def _current_payload(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--focus", choices=("balanced", "failure_recovery", "benchmark_family"), default="balanced")
    parser.add_argument("--family", default=None)
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    metrics = run_eval(
        config=config,
        include_discovered_tasks=True,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_generated=args.include_curriculum,
        include_failure_generated=args.include_failure_curriculum,
    )
    proposals = build_curriculum_proposal_artifact(
        metrics,
        focus=None if args.focus == "balanced" else args.focus,
        family=args.family,
        current_payload=_current_payload(config.curriculum_proposals_path),
        cycles_path=config.improvement_cycles_path,
    )
    config.curriculum_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.curriculum_proposals_path)


if __name__ == "__main__":
    main()
