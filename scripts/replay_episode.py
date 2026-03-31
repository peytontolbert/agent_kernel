from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.memory import EpisodeMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    memory = EpisodeMemory(KernelConfig().trajectories_root)
    print(json.dumps(memory.load(args.task_id), indent=2))


if __name__ == "__main__":
    main()
