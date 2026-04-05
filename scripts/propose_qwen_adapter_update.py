from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.qwen import build_qwen_adapter_candidate_artifact


def _current_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="")
    parser.add_argument("--focus", default="")
    parser.add_argument("--artifact-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig()
    artifact_path = Path(args.artifact_path) if args.artifact_path else config.qwen_adapter_artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_qwen_adapter_candidate_artifact(
        config=config,
        repo_root=repo_root,
        output_dir=artifact_path.parent,
        base_model_name=args.base_model,
        focus=args.focus,
        current_payload=_current_payload(artifact_path),
    )
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(artifact_path)


if __name__ == "__main__":
    main()
