from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.extractors import extract_successful_command_skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--transfer-only", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    output = extract_successful_command_skills(
        config.trajectories_root,
        config.skills_path,
        min_quality=args.min_quality,
        transfer_only=args.transfer_only,
    )
    print(output)


if __name__ == "__main__":
    main()
