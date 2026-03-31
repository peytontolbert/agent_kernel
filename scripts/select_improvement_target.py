from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner
from evals.harness import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model

    metrics = run_eval(
        config=config,
        include_discovered_tasks=True,
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_generated=True,
        include_failure_generated=True,
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
    if args.all_candidates:
        for experiment in planner.rank_experiments(metrics):
            history = experiment.evidence.get("history", {})
            print(
                "subsystem="
                f"{experiment.subsystem} "
                f"priority={experiment.priority} "
                f"expected_gain={experiment.expected_gain:.4f} "
                f"estimated_cost={experiment.estimated_cost} "
                f"score={experiment.score:.4f} "
                f"history_retention_rate={float(history.get('retention_rate', 0.0)):.4f} "
                f"history_retained_pass_rate_delta={float(history.get('average_retained_pass_rate_delta', 0.0)):.4f} "
                f"reason={experiment.reason}"
            )
        return
    experiment = planner.choose_experiment(metrics)
    history = experiment.evidence.get("history", {})
    print(
        "subsystem="
        f"{experiment.subsystem} "
        f"priority={experiment.priority} "
        f"expected_gain={experiment.expected_gain:.4f} "
        f"estimated_cost={experiment.estimated_cost} "
        f"score={experiment.score:.4f} "
        f"history_retention_rate={float(history.get('retention_rate', 0.0)):.4f} "
        f"history_retained_pass_rate_delta={float(history.get('average_retained_pass_rate_delta', 0.0)):.4f} "
        f"reason={experiment.reason}"
    )


if __name__ == "__main__":
    main()
