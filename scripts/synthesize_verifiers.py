from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.verifier_improvement import synthesize_verifier_contracts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("balanced", "false_failure_guard", "strict_contract_growth"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    output = synthesize_verifier_contracts(
        config.trajectories_root,
        config.verifier_contracts_path,
        strategy=args.strategy,
    )
    print(output)


if __name__ == "__main__":
    main()
