from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.model_python import maybe_reexec_under_model_python


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest-path", required=True)
    parser.add_argument("--dataset-manifest-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=64)
    args = parser.parse_args()

    maybe_reexec_under_model_python(require_full_torch=True)

    from agent_kernel.modeling.evaluation.universal_decoder_eval import evaluate_universal_decoder_against_seed

    config = KernelConfig()
    report = evaluate_universal_decoder_against_seed(
        hybrid_bundle_manifest_path=Path(args.bundle_manifest_path),
        dataset_manifest_path=Path(args.dataset_manifest_path),
        config=config,
        device=args.device,
        max_examples=max(1, args.max_examples),
    )
    payload = report.to_dict()
    report_path = Path(args.report_path) if args.report_path else config.improvement_reports_dir / "universal_decoder_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
