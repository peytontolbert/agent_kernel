from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.extensions.tolbert_assets import build_agentkernel_tolbert_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="var/tolbert/agentkernel",
        help="Directory where nodes, spans, label map, and config will be written.",
    )
    parser.add_argument(
        "--base-model-name",
        default="bert-base-uncased",
        help="Base encoder name to embed into the generated training config.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=output_dir,
        base_model_name=args.base_model_name,
    )
    print(f"nodes={paths.nodes_path}")
    print(f"source_spans={paths.source_spans_path}")
    print(f"model_spans={paths.model_spans_path}")
    print(f"label_map={paths.label_map_path}")
    print(f"config={paths.config_path}")
    print(f"level_sizes={paths.level_sizes_path}")


if __name__ == "__main__":
    main()
