from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extractors import extract_tool_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--replay-hardening", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    output = extract_tool_candidates(
        config.trajectories_root,
        config.tool_candidates_path,
        min_quality=args.min_quality,
        replay_hardening=args.replay_hardening,
    )
    print(output)


if __name__ == "__main__":
    main()
