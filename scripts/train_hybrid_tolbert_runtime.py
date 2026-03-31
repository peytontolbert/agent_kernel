from __future__ import annotations

from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.model_python import maybe_reexec_under_model_python
from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig
from agent_kernel.modeling.tolbert.runtime_status import hybrid_runtime_status


def train_hybrid_runtime_from_dataset(**kwargs):
    from agent_kernel.modeling.training.hybrid_trainer import train_hybrid_runtime_from_dataset as _train

    return _train(**kwargs)


def train_hybrid_runtime_from_trajectories(**kwargs):
    from agent_kernel.modeling.training.hybrid_trainer import train_hybrid_runtime_from_trajectories as _train

    return _train(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-root", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    maybe_reexec_under_model_python(require_full_torch=True)

    status = hybrid_runtime_status()
    if not status.available:
        print(
            json.dumps(
                {
                    "artifact_kind": "tolbert_hybrid_runtime_error",
                    "available": False,
                    "reason": status.reason,
                    "runtime_status": status.to_dict(),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    config = KernelConfig()
    dataset_manifest = _load_optional_json(
        _env_or_cli_path(
            cli_value=args.dataset_path,
            direct_env_key="AGENTKERNEL_TOLBERT_DATASET_PATH",
            manifest_env_key="AGENTKERNEL_TOLBERT_DATASET_MANIFEST_PATH",
        )
    )
    training_inputs_manifest = _load_optional_json(
        _env_or_cli_path(
            cli_value=None,
            direct_env_key="AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH",
            manifest_env_key="AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH",
        )
    )
    dataset_path = _resolve_dataset_path(
        cli_value=args.dataset_path,
        dataset_manifest=dataset_manifest,
        training_inputs_manifest=training_inputs_manifest,
    )
    hybrid_config = _resolve_hybrid_config(
        cli_config_path=args.config_path,
        dataset_manifest=dataset_manifest,
    )
    runtime_paths = _resolve_runtime_paths(
        cli_output_dir=args.output_dir,
        config=config,
    )
    output_dir = runtime_paths["output_dir"]
    if dataset_path is not None:
        manifest = train_hybrid_runtime_from_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=hybrid_config,
            epochs=max(1, args.epochs),
            batch_size=max(1, args.batch_size),
            lr=float(args.lr),
            device=args.device,
            runtime_paths=runtime_paths,
        )
    else:
        trajectories_root = Path(args.trajectories_root) if args.trajectories_root else config.trajectories_root
        manifest = train_hybrid_runtime_from_trajectories(
            trajectories_root=trajectories_root,
            output_dir=output_dir,
            config=hybrid_config,
            epochs=max(1, args.epochs),
            batch_size=max(1, args.batch_size),
            lr=float(args.lr),
            device=args.device,
            runtime_paths=runtime_paths,
        )
    print(json.dumps(manifest, indent=2))

def _load_optional_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _env_or_cli_path(*, cli_value: str | None, direct_env_key: str, manifest_env_key: str) -> Path | None:
    raw = str(cli_value or "").strip()
    if raw:
        return Path(raw)
    direct = str(os.getenv(direct_env_key, "")).strip()
    if direct:
        return Path(direct)
    manifest = str(os.getenv(manifest_env_key, "")).strip()
    if manifest:
        return Path(manifest)
    return None


def _resolve_dataset_path(
    *,
    cli_value: str | None,
    dataset_manifest: dict[str, object],
    training_inputs_manifest: dict[str, object],
) -> Path | None:
    raw = str(cli_value or "").strip()
    if raw:
        candidate = Path(raw)
        if candidate.suffix == ".json":
            payload = _load_optional_json(candidate)
            dataset_path = str(payload.get("dataset_path", "")).strip()
            return Path(dataset_path) if dataset_path else None
        return candidate
    dataset_path = str(dataset_manifest.get("dataset_path", "")).strip()
    if dataset_path:
        return Path(dataset_path)
    dataset_manifest_path = str(training_inputs_manifest.get("dataset_manifest_path", "")).strip()
    if dataset_manifest_path:
        nested_manifest = _load_optional_json(Path(dataset_manifest_path))
        nested_dataset_path = str(nested_manifest.get("dataset_path", "")).strip()
        if nested_dataset_path:
            return Path(nested_dataset_path)
    return None


def _resolve_hybrid_config(
    *,
    cli_config_path: str | None,
    dataset_manifest: dict[str, object],
) -> HybridTolbertSSMConfig:
    if cli_config_path:
        payload = _load_optional_json(Path(cli_config_path))
        if payload:
            return HybridTolbertSSMConfig.from_dict(payload)
    env_config_path = str(os.getenv("AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH", "")).strip()
    if env_config_path:
        payload = _load_optional_json(Path(env_config_path))
        if payload:
            return HybridTolbertSSMConfig.from_dict(payload)
    manifest_config = dataset_manifest.get("config", {})
    if isinstance(manifest_config, dict) and manifest_config:
        return HybridTolbertSSMConfig.from_dict(manifest_config)
    return HybridTolbertSSMConfig()


def _resolve_runtime_paths(*, cli_output_dir: str | None, config: KernelConfig) -> dict[str, str | Path]:
    output_dir = (
        Path(cli_output_dir)
        if cli_output_dir
        else _runtime_output_dir_from_env() or (config.candidate_artifacts_root / "tolbert_hybrid_runtime_manual")
    )
    checkpoint_path = str(os.getenv("AGENTKERNEL_TOLBERT_CHECKPOINT_OUTPUT_PATH", "")).strip()
    parent_checkpoint_path = str(os.getenv("AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH", "")).strip()
    checkpoint_delta_path = str(os.getenv("AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH", "")).strip()
    config_path = str(os.getenv("AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH", "")).strip()
    manifest_path = str(os.getenv("AGENTKERNEL_TOLBERT_BUNDLE_MANIFEST_PATH", "")).strip()
    return {
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path or str(output_dir / "hybrid_checkpoint.pt"),
        "parent_checkpoint_path": parent_checkpoint_path,
        "checkpoint_delta_path": checkpoint_delta_path or str(output_dir / "hybrid_checkpoint__delta.pt"),
        "config_path": config_path or str(output_dir / "hybrid_config.json"),
        "bundle_manifest_path": manifest_path or str(output_dir / "hybrid_bundle_manifest.json"),
    }


def _runtime_output_dir_from_env() -> Path | None:
    for env_name in (
        "AGENTKERNEL_TOLBERT_BUNDLE_MANIFEST_PATH",
        "AGENTKERNEL_TOLBERT_CHECKPOINT_OUTPUT_PATH",
        "AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH",
        "AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH",
        "AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH",
    ):
        raw = str(os.getenv(env_name, "")).strip()
        if raw:
            return Path(raw).parent
    return None


if __name__ == "__main__":
    main()
