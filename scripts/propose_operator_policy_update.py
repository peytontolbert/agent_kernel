from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.operator_policy_improvement import build_operator_policy_proposal_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus",
        choices=("balanced", "family_breadth", "git_http_scope", "generated_path_scope"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    proposals = build_operator_policy_proposal_artifact(
        config,
        focus=None if args.focus == "balanced" else args.focus,
    )
    config.operator_policy_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.operator_policy_proposals_path)


if __name__ == "__main__":
    main()
