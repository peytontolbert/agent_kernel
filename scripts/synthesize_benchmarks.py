from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.benchmark_synthesis import synthesize_benchmark_candidates
from agent_kernel.config import KernelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus", choices=("balanced", "confidence", "breadth"), default="balanced")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    output = synthesize_benchmark_candidates(
        config.trajectories_root,
        config.benchmark_candidates_path,
        focus=None if args.focus == "balanced" else args.focus,
    )
    print(output)


if __name__ == "__main__":
    main()
