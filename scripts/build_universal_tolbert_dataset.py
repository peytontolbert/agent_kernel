from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig
from agent_kernel.modeling.training.universal_dataset import materialize_universal_decoder_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()

    config = KernelConfig()
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else config.candidate_artifacts_root / "tolbert_universal_dataset_manual"
    )
    manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        hybrid_config=HybridTolbertSSMConfig(),
        eval_fraction=float(args.eval_fraction),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
