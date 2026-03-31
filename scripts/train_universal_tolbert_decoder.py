from __future__ import annotations

from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.modeling.model_python import maybe_reexec_under_model_python
from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig
from agent_kernel.modeling.tolbert.runtime_status import hybrid_runtime_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-checkpoint-path", default=None)
    args = parser.parse_args()

    maybe_reexec_under_model_python(require_full_torch=True)

    status = hybrid_runtime_status()
    if not status.available:
        print(
            json.dumps(
                {
                    "artifact_kind": "tolbert_universal_decoder_runtime_error",
                    "available": False,
                    "reason": status.reason,
                    "runtime_status": status.to_dict(),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    from agent_kernel.modeling.training.universal_trainer import train_hybrid_decoder_from_universal_dataset

    if args.config_path:
        payload = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
        hybrid_config = HybridTolbertSSMConfig.from_dict(payload if isinstance(payload, dict) else {})
    else:
        hybrid_config = HybridTolbertSSMConfig()
    manifest = train_hybrid_decoder_from_universal_dataset(
        dataset_manifest_path=Path(args.dataset_manifest_path),
        output_dir=Path(args.output_dir),
        config=hybrid_config,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        lr=float(args.lr),
        device=args.device,
        parent_checkpoint_path=(
            Path(args.parent_checkpoint_path)
            if args.parent_checkpoint_path
            else Path(str(os.getenv("AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH", "")).strip())
            if str(os.getenv("AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH", "")).strip()
            else None
        ),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
