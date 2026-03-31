from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementPlanner, artifact_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--cycles-path", default=None)
    args = parser.parse_args()

    config = KernelConfig()
    cycles_path = Path(args.cycles_path) if args.cycles_path else config.improvement_cycles_path
    artifact_path = Path(args.artifact_path)
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    metadata = planner.artifact_rollback_metadata(cycles_path, artifact_path)
    snapshot_path = Path(str(metadata.get("rollback_artifact_path", "")).strip())
    if not snapshot_path.exists():
        raise SystemExit(f"rollback snapshot does not exist: {snapshot_path}")
    live_sha256 = artifact_sha256(artifact_path)
    snapshot_sha256 = artifact_sha256(snapshot_path)
    if live_sha256 != snapshot_sha256:
        raise SystemExit(
            "rollback validation failed: "
            f"artifact_path={artifact_path} live_sha256={live_sha256} snapshot_sha256={snapshot_sha256}"
        )
    print(
        "validation_state=passed "
        f"artifact_path={artifact_path} "
        f"snapshot_path={snapshot_path} "
        f"live_sha256={live_sha256}"
    )


if __name__ == "__main__":
    main()
