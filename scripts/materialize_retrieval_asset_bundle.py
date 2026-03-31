from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.tolbert_assets import materialize_retained_retrieval_asset_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional Tolbert asset output directory. Defaults next to the bundle manifest.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Optional retrieval asset bundle manifest path. Defaults to config.retrieval_asset_bundle_path.",
    )
    parser.add_argument(
        "--base-model-name",
        default="bert-base-uncased",
        help="Base encoder name to embed into the generated training config.",
    )
    parser.add_argument(
        "--cycle-id",
        default="",
        help="Optional improvement cycle id to stamp into the bundle manifest.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig()
    config.ensure_directories()
    manifest_path = materialize_retained_retrieval_asset_bundle(
        repo_root=repo_root,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        base_model_name=args.base_model_name,
        cycle_id=args.cycle_id or None,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
