from __future__ import annotations

from pathlib import Path
import os

from agent_kernel.modeling.model_python import (
    maybe_reexec_under_model_python,
    preferred_model_python_path,
    should_reexec_under_model_python,
)


def test_preferred_model_python_path_uses_env_override(tmp_path: Path) -> None:
    python_path = tmp_path / "envs" / "ai" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")

    resolved = preferred_model_python_path({"AGENTKERNEL_MODEL_PYTHON": str(python_path)})

    assert resolved == python_path


def test_should_reexec_under_model_python_prefers_ai_interpreter_when_current_differs(tmp_path: Path) -> None:
    preferred = tmp_path / "envs" / "ai" / "bin" / "python"
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_text("", encoding="utf-8")
    current = tmp_path / "base" / "bin" / "python"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("", encoding="utf-8")

    assert should_reexec_under_model_python(
        current_python=current,
        preferred_python=preferred,
        env={},
        require_full_torch=False,
    )


def test_should_reexec_under_model_python_respects_bootstrap_guard(tmp_path: Path) -> None:
    preferred = tmp_path / "envs" / "ai" / "bin" / "python"
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_text("", encoding="utf-8")
    current = tmp_path / "base" / "bin" / "python"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("", encoding="utf-8")

    assert not should_reexec_under_model_python(
        current_python=current,
        preferred_python=preferred,
        env={"AGENTKERNEL_MODEL_PYTHON_BOOTSTRAPPED": "1"},
        require_full_torch=False,
    )


def test_maybe_reexec_under_model_python_execves_preferred_interpreter(monkeypatch, tmp_path: Path) -> None:
    preferred = tmp_path / "envs" / "ai" / "bin" / "python"
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_text("", encoding="utf-8")
    current = tmp_path / "base" / "bin" / "python"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    monkeypatch.setenv("AGENTKERNEL_MODEL_PYTHON", str(preferred))
    monkeypatch.setattr("sys.executable", str(current))
    monkeypatch.setattr("sys.argv", ["train_hybrid_tolbert_runtime.py", "--device", "cpu"])

    def _fake_execve(path, argv, env):
        seen["path"] = path
        seen["argv"] = list(argv)
        seen["env"] = dict(env)
        raise SystemExit(0)

    monkeypatch.setattr(os, "execve", _fake_execve)

    try:
        maybe_reexec_under_model_python(require_full_torch=False)
    except SystemExit as exc:
        assert int(exc.code) == 0
    else:  # pragma: no cover
        raise AssertionError("expected execve path to terminate test control flow")

    assert seen["path"] == str(preferred)
    assert seen["argv"] == [str(preferred), "train_hybrid_tolbert_runtime.py", "--device", "cpu"]
    assert seen["env"]["AGENTKERNEL_MODEL_PYTHON_BOOTSTRAPPED"] == "1"
