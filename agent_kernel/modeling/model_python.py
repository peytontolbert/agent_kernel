from __future__ import annotations

from pathlib import Path
from typing import Mapping
import os
import sys


_BOOTSTRAP_ENV_KEY = "AGENTKERNEL_MODEL_PYTHON_BOOTSTRAPPED"
_MODEL_PYTHON_ENV_KEYS = (
    "AGENTKERNEL_MODEL_PYTHON",
    "AGENTKERNEL_PREFERRED_MODEL_PYTHON",
)
_DEFAULT_AI_PYTHON = Path("/home/peyton/miniconda3/envs/ai/bin/python")


def preferred_model_python_path(env: Mapping[str, str] | None = None) -> Path | None:
    resolved_env = os.environ if env is None else env
    for key in _MODEL_PYTHON_ENV_KEYS:
        raw = str(resolved_env.get(key, "")).strip()
        if raw:
            candidate = Path(raw)
            if candidate.exists():
                return candidate
    if _DEFAULT_AI_PYTHON.exists():
        return _DEFAULT_AI_PYTHON
    return None


def should_reexec_under_model_python(
    *,
    current_python: str | Path | None = None,
    preferred_python: Path | None = None,
    env: Mapping[str, str] | None = None,
    require_full_torch: bool = False,
    runtime_available: bool | None = None,
) -> bool:
    resolved_env = os.environ if env is None else env
    if str(resolved_env.get(_BOOTSTRAP_ENV_KEY, "")).strip() == "1":
        return False
    preferred = preferred_python if preferred_python is not None else preferred_model_python_path(resolved_env)
    if preferred is None or not preferred.exists():
        return False
    current = Path(current_python or sys.executable)
    try:
        current = current.resolve()
    except OSError:
        current = Path(current)
    try:
        preferred = preferred.resolve()
    except OSError:
        pass
    if current == preferred:
        return False
    if not require_full_torch:
        return True
    if runtime_available is None:
        from .tolbert.runtime_status import hybrid_runtime_status

        runtime_available = hybrid_runtime_status().available
    return not bool(runtime_available)


def maybe_reexec_under_model_python(
    *,
    argv: list[str] | None = None,
    require_full_torch: bool = False,
    runtime_available: bool | None = None,
) -> bool:
    preferred = preferred_model_python_path()
    if not should_reexec_under_model_python(
        current_python=sys.executable,
        preferred_python=preferred,
        require_full_torch=require_full_torch,
        runtime_available=runtime_available,
    ):
        return False
    env = dict(os.environ)
    env[_BOOTSTRAP_ENV_KEY] = "1"
    command_argv = [str(preferred), *(sys.argv if argv is None else argv)]
    os.execve(str(preferred), command_argv, env)
    return True
