from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.improvement.delegation_improvement import build_delegation_proposal_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus",
        choices=("balanced", "throughput_balance", "queue_elasticity", "worker_depth"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    proposals = build_delegation_proposal_artifact(
        config,
        focus=None if args.focus == "balanced" else args.focus,
    )
    config.delegation_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.delegation_proposals_path)


if __name__ == "__main__":
    main()
