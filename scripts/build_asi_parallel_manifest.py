from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.ops.roadmap_parallel import write_asi_parallel_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asi-path", default="asi.md")
    parser.add_argument("--output-path", default="config/asi_parallel_development_manifest.json")
    parser.add_argument("--worker-count", type=int, default=5)
    args = parser.parse_args()

    manifest = write_asi_parallel_manifest(
        Path(args.asi_path),
        output_path=Path(args.output_path),
        worker_count=max(1, int(args.worker_count)),
    )
    print(Path(args.output_path))
    print(f"worker_count={manifest.get('worker_count', 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
