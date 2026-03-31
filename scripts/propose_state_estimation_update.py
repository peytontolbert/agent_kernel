from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner
from agent_kernel.state_estimation_improvement import build_state_estimation_proposal_artifact
from evals.metrics import EvalMetrics


def _current_payload(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus",
        choices=("balanced", "transition_normalization", "risk_sensitivity", "recovery_bias"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    planner = ImprovementPlanner(memory_root=config.trajectories_root, runtime_config=config)
    proposals = build_state_estimation_proposal_artifact(
        EvalMetrics(total=0, passed=0),
        planner.transition_summary(),
        focus=None if args.focus == "balanced" else args.focus,
        current_payload=_current_payload(config.state_estimation_proposals_path),
    )
    config.state_estimation_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.state_estimation_proposals_path)


if __name__ == "__main__":
    main()
