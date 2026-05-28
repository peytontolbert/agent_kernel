#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_agentkernel_lite_controller_trace_dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="benchmarks")
    parser.add_argument("--report-root", action="append", default=[])
    parser.add_argument("--output-dir", default="artifacts/agentkernel_controller/controller_trace_dataset")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    args = parser.parse_args()
    if not args.report_root:
        args.report_root = ["benchmarks"]

    output_dir = Path(args.output_dir)
    manifest = build_dataset(args)
    train_path = output_dir / "agentkernel_controller_train.jsonl"
    eval_path = output_dir / "agentkernel_controller_eval.jsonl"
    shutil.copyfile(output_dir / "agentkernel_lite_encdec_train.jsonl", train_path)
    shutil.copyfile(output_dir / "agentkernel_lite_encdec_eval.jsonl", eval_path)

    manifest_path = output_dir / "agentkernel_controller_trace_dataset_manifest.json"
    full_manifest = dict(manifest)
    full_manifest.update(
        {
            "artifact_kind": "agentkernel_controller_trace_dataset",
            "objective": "agentkernel_controller_trace_policy",
            "source_builder": "scripts/build_agentkernel_lite_controller_trace_dataset.py",
            "source_builder_scope": "shared_trace_extraction_scaffold_only",
            "manifest_path": str(manifest_path),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "runtime_target": "full_agent_kernel_neural_controller",
        }
    )
    manifest_path.write_text(json.dumps(full_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(full_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
