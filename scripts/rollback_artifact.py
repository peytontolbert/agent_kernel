from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--cycles-path", default=None)
    args = parser.parse_args()

    config = KernelConfig()
    cycles_path = Path(args.cycles_path) if args.cycles_path else config.improvement_cycles_path
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    restored = planner.rollback_artifact(cycles_path, Path(args.artifact_path))
    print(f"restored_artifact={restored}")


if __name__ == "__main__":
    main()
