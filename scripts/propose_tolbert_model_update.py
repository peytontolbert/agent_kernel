from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.tolbert_model_improvement import build_tolbert_model_candidate_artifact
from evals.harness import run_eval


def _current_payload(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--focus",
        choices=("balanced", "recovery_alignment", "discovered_task_adaptation"),
        default="balanced",
    )
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    metrics = run_eval(
        config=config,
        include_discovered_tasks=True,
        include_generated=True,
        include_failure_generated=True,
        include_benchmark_candidates=True,
    )
    artifact = build_tolbert_model_candidate_artifact(
        config=config,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=config.candidate_artifacts_root / "tolbert_model_manual",
        metrics=metrics,
        focus=None if args.focus == "balanced" else args.focus,
        current_payload=_current_payload(config.tolbert_model_artifact_path),
    )
    config.tolbert_model_artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(config.tolbert_model_artifact_path)


if __name__ == "__main__":
    main()
