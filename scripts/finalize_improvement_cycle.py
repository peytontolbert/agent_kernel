from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import agent_kernel.cycle_runner as cycle_runner
from agent_kernel.config import KernelConfig


def _baseline_candidate_flags(subsystem: str) -> tuple[dict[str, bool], dict[str, bool]]:
    return cycle_runner.baseline_candidate_flags(subsystem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--artifact-path", required=True)
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
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    state, reason = cycle_runner.finalize_cycle(
        config=config,
        subsystem=args.subsystem,
        cycle_id=args.cycle_id,
        artifact_path=Path(args.artifact_path),
        include_episode_memory=args.include_episode_memory,
        include_skill_memory=args.include_skill_memory,
        include_skill_transfer=args.include_skill_transfer,
        include_operator_memory=args.include_operator_memory,
        include_tool_memory=args.include_tool_memory,
        include_verifier_memory=args.include_verifier_memory,
        include_curriculum=True,
        include_failure_curriculum=True,
    )
    print(f"cycle_id={args.cycle_id} state={state} reason={reason}")


if __name__ == "__main__":
    main()
