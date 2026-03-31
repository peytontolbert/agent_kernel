from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.modeling.model_python import preferred_model_python_path
from agent_kernel.modeling.training_backends import (
    build_training_backend_launch,
    discover_training_backends,
    resolve_training_backend,
    tolbert_artifact_training_env,
)


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"invalid env override: {raw}")
        key, value = raw.split("=", 1)
        env_key = key.strip()
        if not env_key:
            raise ValueError(f"invalid env override: {raw}")
        overrides[env_key] = value
    return overrides


def _normalized_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        current = argv[index]
        if current == "--arg" and index + 1 < len(argv):
            normalized.append(f"--arg={argv[index + 1]}")
            index += 2
            continue
        normalized.append(current)
        index += 1
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="state_space_causal_machine")
    parser.add_argument("--python", default=str(preferred_model_python_path() or Path(sys.executable)))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--tolbert-artifact", default="")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--arg", action="append", default=[])
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(_normalized_argv(sys.argv[1:]))

    repo_root = Path(args.repo_root).resolve()
    manifests = discover_training_backends(repo_root)
    if args.list:
        payload = {
            "repo_root": str(repo_root),
            "backends": [
                {
                    "backend_id": manifest.get("backend_id", ""),
                    "label": manifest.get("label", ""),
                    "trainer_exists": bool(manifest.get("trainer_exists", False)),
                    "features": manifest.get("features", {}),
                }
                for manifest in manifests
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    manifest = resolve_training_backend(repo_root, args.backend)
    if not manifest:
        raise SystemExit(f"unknown training backend: {args.backend}")
    if not bool(manifest.get("trainer_exists", False)):
        raise SystemExit(f"training backend entrypoint is missing: {manifest.get('trainer_path', '')}")
    env_overrides = _parse_env_overrides(args.env)
    artifact_path = Path(str(args.tolbert_artifact).strip()) if str(args.tolbert_artifact).strip() else None
    if artifact_path is not None:
        try:
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact_payload = {}
        env_overrides.update(tolbert_artifact_training_env(artifact_payload))
    launch = build_training_backend_launch(
        manifest,
        python_bin=args.python,
        env_overrides=env_overrides,
        extra_args=args.arg,
    )
    payload = {
        "backend_id": launch["backend_id"],
        "label": launch["label"],
        "backend_root": launch["backend_root"],
        "trainer_cwd": launch["trainer_cwd"],
        "trainer_path": launch["trainer_path"],
        "command": launch["command"],
        "env": launch["env"],
        "artifacts": launch["artifacts"],
        "features": launch["features"],
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    completed = subprocess.run(
        launch["command"],
        cwd=launch["trainer_cwd"],
        env=launch["resolved_env"],
        check=True,
    )
    payload["dry_run"] = False
    payload["returncode"] = completed.returncode
    payload["cwd_contents"] = sorted(os.listdir(launch["trainer_cwd"]))[:32]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
