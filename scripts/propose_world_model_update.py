from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner
from agent_kernel.extensions.improvement.world_model_improvement import build_world_model_proposal_artifact
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
    parser.add_argument(
        "--focus",
        choices=("balanced", "workflow_alignment", "preservation_bias", "conflict_avoidance"),
        default="balanced",
    )
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
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    proposals = build_world_model_proposal_artifact(
        metrics,
        planner.failure_counts(),
        focus=None if args.focus == "balanced" else args.focus,
        current_payload=_current_payload(config.world_model_proposals_path),
    )
    config.world_model_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.world_model_proposals_path)


if __name__ == "__main__":
    main()
