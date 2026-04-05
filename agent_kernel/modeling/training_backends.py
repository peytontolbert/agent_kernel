from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def discover_training_backends(repo_root: Path) -> list[dict[str, Any]]:
    other_repos = repo_root / "other_repos"
    backends: list[dict[str, Any]] = []
    if not other_repos.exists():
        return backends
    for child in sorted(other_repos.iterdir()):
        if not child.is_dir():
            continue
        manifest = load_training_backend_manifest(child)
        if manifest:
            backends.append(manifest)
    return backends


def resolve_training_backend(repo_root: Path, backend_id: str) -> dict[str, Any]:
    normalized_backend_id = str(backend_id).strip()
    if not normalized_backend_id:
        return {}
    for manifest in discover_training_backends(repo_root):
        if str(manifest.get("backend_id", "")).strip() == normalized_backend_id:
            return manifest
    return {}


def load_training_backend_manifest(backend_root: Path) -> dict[str, Any]:
    manifest_path = backend_root / "agentkernel_backend.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    backend_id = str(payload.get("backend_id", "")).strip()
    trainer_entrypoint = str(payload.get("trainer_entrypoint", "")).strip()
    if not backend_id or not trainer_entrypoint:
        return {}
    trainer_path = backend_root / trainer_entrypoint
    normalized = {
        "backend_id": backend_id,
        "label": str(payload.get("label", backend_id)).strip() or backend_id,
        "backend_kind": str(payload.get("backend_kind", "external_model_trainer")).strip() or "external_model_trainer",
        "backend_root": str(backend_root.resolve()),
        "trainer_entrypoint": trainer_entrypoint,
        "trainer_path": str(trainer_path.resolve()),
        "trainer_exists": trainer_path.exists(),
        "trainer_cwd": str((backend_root / str(payload.get("trainer_cwd", ".")).strip()).resolve()),
        "default_launch": (
            dict(payload.get("default_launch", {}))
            if isinstance(payload.get("default_launch", {}), dict)
            else {}
        ),
        "features": dict(payload.get("features", {})) if isinstance(payload.get("features", {}), dict) else {},
        "artifacts": dict(payload.get("artifacts", {})) if isinstance(payload.get("artifacts", {}), dict) else {},
        "notes": [
            str(value).strip()
            for value in payload.get("notes", [])
            if str(value).strip()
        ] if isinstance(payload.get("notes", []), list) else [],
        "manifest_path": str(manifest_path.resolve()),
    }
    return normalized


def build_training_backend_launch(
    manifest: dict[str, Any],
    *,
    python_bin: str,
    env_overrides: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    backend_root = Path(str(manifest.get("backend_root", "")).strip())
    trainer_path = Path(str(manifest.get("trainer_path", "")).strip())
    trainer_cwd = Path(str(manifest.get("trainer_cwd", "")).strip())
    default_launch = dict(manifest.get("default_launch", {})) if isinstance(manifest.get("default_launch", {}), dict) else {}
    launcher = str(default_launch.get("launcher", "")).strip()
    launcher_args = [
        str(value)
        for value in default_launch.get("launcher_args", [])
        if str(value).strip()
    ] if isinstance(default_launch.get("launcher_args", []), list) else []
    launch_env = {
        key: str(value)
        for key, value in default_launch.get("env", {}).items()
        if str(key).strip()
    } if isinstance(default_launch.get("env", {}), dict) else {}
    if env_overrides:
        for key, value in env_overrides.items():
            env_key = str(key).strip()
            if env_key:
                launch_env[env_key] = str(value)
    command: list[str]
    if launcher:
        command = [launcher, *launcher_args, str(trainer_path)]
    else:
        command = [python_bin, str(trainer_path)]
    if extra_args:
        command.extend(str(value) for value in extra_args if str(value).strip())
    env = os.environ.copy()
    env.update(launch_env)
    return {
        "backend_id": str(manifest.get("backend_id", "")).strip(),
        "label": str(manifest.get("label", "")).strip(),
        "backend_root": str(backend_root),
        "trainer_cwd": str(trainer_cwd),
        "trainer_path": str(trainer_path),
        "command": command,
        "env": launch_env,
        "resolved_env": env,
        "artifacts": dict(manifest.get("artifacts", {})) if isinstance(manifest.get("artifacts", {}), dict) else {},
        "features": dict(manifest.get("features", {})) if isinstance(manifest.get("features", {}), dict) else {},
    }


def tolbert_artifact_training_env(payload: dict[str, Any]) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    if str(payload.get("artifact_kind", "")).strip() != "tolbert_model_bundle":
        return {}
    training_inputs = payload.get("training_inputs", {})
    if not isinstance(training_inputs, dict):
        training_inputs = {}
    env: dict[str, str] = {}
    for key, env_name in (
        ("training_inputs_manifest_path", "AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH"),
        ("dataset_manifest_path", "AGENTKERNEL_TOLBERT_DATASET_MANIFEST_PATH"),
        ("supervised_examples_path", "AGENTKERNEL_TOLBERT_SUPERVISED_EXAMPLES_PATH"),
        ("policy_examples_path", "AGENTKERNEL_TOLBERT_POLICY_EXAMPLES_PATH"),
        ("transition_examples_path", "AGENTKERNEL_TOLBERT_TRANSITION_EXAMPLES_PATH"),
        ("value_examples_path", "AGENTKERNEL_TOLBERT_VALUE_EXAMPLES_PATH"),
        ("stop_examples_path", "AGENTKERNEL_TOLBERT_STOP_EXAMPLES_PATH"),
    ):
        value = str(training_inputs.get(key, "")).strip()
        if value:
            env[env_name] = value
    runtime_paths = payload.get("runtime_paths", {})
    if isinstance(runtime_paths, dict):
        for key, env_name in (
            ("config_path", "AGENTKERNEL_TOLBERT_TRAIN_CONFIG_PATH"),
            ("checkpoint_path", "AGENTKERNEL_TOLBERT_CHECKPOINT_OUTPUT_PATH"),
            ("parent_checkpoint_path", "AGENTKERNEL_TOLBERT_PARENT_CHECKPOINT_PATH"),
            ("checkpoint_delta_path", "AGENTKERNEL_TOLBERT_CHECKPOINT_DELTA_OUTPUT_PATH"),
            ("training_inputs_manifest_path", "AGENTKERNEL_TOLBERT_TRAINING_INPUTS_MANIFEST_PATH"),
        ):
            value = str(runtime_paths.get(key, "")).strip()
            if value:
                env[env_name] = value
    enabled_heads = training_inputs.get("enabled_heads", [])
    if isinstance(enabled_heads, list):
        env["AGENTKERNEL_TOLBERT_ENABLED_HEADS"] = ",".join(
            str(value).strip()
            for value in enabled_heads
            if str(value).strip()
        )
    return env


def qwen_artifact_training_env(payload: dict[str, Any]) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    if str(payload.get("artifact_kind", "")).strip() != "qwen_adapter_bundle":
        return {}
    env: dict[str, str] = {}
    base_model_name = str(payload.get("base_model_name", "")).strip()
    if base_model_name:
        env["AGENTKERNEL_QWEN_BASE_MODEL_NAME"] = base_model_name
    training_objective = str(payload.get("training_objective", "")).strip()
    if training_objective:
        env["AGENTKERNEL_QWEN_TRAINING_OBJECTIVE"] = training_objective
    dataset_manifest = payload.get("training_dataset_manifest", {})
    if isinstance(dataset_manifest, dict):
        for key, env_name in (
            ("manifest_path", "AGENTKERNEL_QWEN_DATASET_MANIFEST_PATH"),
            ("train_dataset_path", "AGENTKERNEL_QWEN_TRAIN_DATASET_PATH"),
            ("eval_dataset_path", "AGENTKERNEL_QWEN_EVAL_DATASET_PATH"),
        ):
            value = str(dataset_manifest.get(key, "")).strip()
            if value:
                env[env_name] = value
    runtime_paths = payload.get("runtime_paths", {})
    if isinstance(runtime_paths, dict):
        for key, env_name in (
            ("adapter_output_dir", "AGENTKERNEL_QWEN_ADAPTER_OUTPUT_DIR"),
            ("merged_output_dir", "AGENTKERNEL_QWEN_MERGED_OUTPUT_DIR"),
            ("adapter_manifest_path", "AGENTKERNEL_QWEN_ADAPTER_MANIFEST_PATH"),
        ):
            value = str(runtime_paths.get(key, "")).strip()
            if value:
                env[env_name] = value
    supported_families = payload.get("supported_benchmark_families", [])
    if isinstance(supported_families, list):
        normalized = [
            str(value).strip()
            for value in supported_families
            if str(value).strip()
        ]
        if normalized:
            env["AGENTKERNEL_QWEN_SUPPORTED_BENCHMARK_FAMILIES"] = ",".join(normalized)
    return env


def artifact_training_env(payload: dict[str, Any]) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    kind = str(payload.get("artifact_kind", "")).strip()
    if kind == "tolbert_model_bundle":
        return tolbert_artifact_training_env(payload)
    if kind == "qwen_adapter_bundle":
        return qwen_artifact_training_env(payload)
    return {}
