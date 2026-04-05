"""Modeling package for TOLBERT-family checkpoints, training, and evaluation.

This package is intentionally separate from the task-execution runtime so model
code can evolve without tangling with the core agent loop, sandbox, verifier,
and unattended-control paths.
"""

from .training_backends import (
    artifact_training_env,
    build_training_backend_launch,
    discover_training_backends,
    qwen_artifact_training_env,
    resolve_training_backend,
    tolbert_artifact_training_env,
)

__all__ = [
    "artifact_training_env",
    "build_training_backend_launch",
    "discover_training_backends",
    "qwen_artifact_training_env",
    "resolve_training_backend",
    "tolbert_artifact_training_env",
]
