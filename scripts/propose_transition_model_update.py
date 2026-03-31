from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.transition_model_improvement import build_transition_model_proposal_artifact


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
        choices=("balanced", "repeat_avoidance", "regression_guard", "recovery_bias"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    proposals = build_transition_model_proposal_artifact(
        config.trajectories_root,
        focus=None if args.focus == "balanced" else args.focus,
        current_payload=_current_payload(config.transition_model_proposals_path),
    )
    config.transition_model_proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(config.transition_model_proposals_path)


if __name__ == "__main__":
    main()
