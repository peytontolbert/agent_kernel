from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.extractors import extract_operator_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cross-family-only", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    output = extract_operator_classes(
        config.trajectories_root,
        config.operator_classes_path,
        min_support=args.min_support,
        cross_family_only=args.cross_family_only,
    )
    print(output)


if __name__ == "__main__":
    main()
