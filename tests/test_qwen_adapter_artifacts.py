from __future__ import annotations

from io import StringIO
from pathlib import Path
import importlib.util
import json
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.artifacts import (
    retained_qwen_adapter_retention_gate,
    retained_qwen_adapter_runtime_policy,
)
from agent_kernel.modeling.qwen import build_qwen_adapter_candidate_artifact
from agent_kernel.modeling.training.qwen_dataset import materialize_qwen_sft_dataset


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialize_qwen_sft_dataset_creates_train_eval_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "run_learning_artifacts.json",
    )
    config.ensure_directories()
    trajectory = {
        "task_id": "repo_fix_task",
        "prompt": "Fix the repository validation summary.",
        "success": True,
        "task_metadata": {"benchmark_family": "repository", "difficulty": "long_horizon"},
        "steps": [
            {"action": "code_execute", "content": "git status", "verification": {"passed": True}},
            {"action": "code_execute", "content": "pytest -q", "verification": {"passed": True}},
        ],
    }
    (config.trajectories_root / "repo_fix_task.json").write_text(json.dumps(trajectory), encoding="utf-8")

    manifest = materialize_qwen_sft_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=tmp_path / "qwen_dataset",
        eval_fraction=0.5,
    )

    assert manifest["artifact_kind"] == "qwen_sft_dataset"
    assert Path(manifest["train_dataset_path"]).exists()
    assert Path(manifest["eval_dataset_path"]).exists()
    assert manifest["total_examples"] >= 1
    assert manifest["long_horizon_example_count"] >= 1


def test_build_qwen_adapter_candidate_artifact_uses_support_runtime_role(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig(
        trajectories_root=tmp_path / "episodes",
        run_reports_dir=tmp_path / "reports",
        learning_artifacts_path=tmp_path / "learning" / "run_learning_artifacts.json",
        qwen_adapter_artifact_path=tmp_path / "retained" / "qwen_adapter_artifact.json",
    )
    config.ensure_directories()
    (config.trajectories_root / "repo_fix_task.json").write_text(
        json.dumps(
            {
                "task_id": "repo_fix_task",
                "prompt": "Repair the repository changelog.",
                "success": True,
                "task_metadata": {"benchmark_family": "repository", "difficulty": "long_horizon"},
                "steps": [
                    {"action": "code_execute", "content": "git status", "verification": {"passed": True}},
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_qwen_adapter_candidate_artifact(
        config=config,
        repo_root=repo_root,
        output_dir=tmp_path / "candidate",
    )

    assert artifact["artifact_kind"] == "qwen_adapter_bundle"
    assert artifact["runtime_role"] == "support_runtime"
    assert artifact["training_objective"] == "qlora_sft"
    assert artifact["retention_gate"]["disallow_liftoff_authority"] is True
    assert artifact["runtime_policy"]["allow_teacher_generation"] is True
    assert artifact["runtime_paths"]["served_model_name"] == artifact["base_model_name"]
    assert Path(artifact["runtime_paths"]["dataset_manifest_path"]).exists()


def test_retained_qwen_adapter_helpers_normalize_defaults() -> None:
    runtime = retained_qwen_adapter_runtime_policy({})
    gate = retained_qwen_adapter_retention_gate({})

    assert runtime["allow_primary_routing"] is False
    assert runtime["allow_teacher_generation"] is True
    assert gate["require_non_regression"] is True
    assert gate["disallow_liftoff_authority"] is True


def test_propose_qwen_adapter_update_script_writes_candidate_artifact(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("propose_qwen_adapter_update.py")
    artifact_path = tmp_path / "qwen_adapter" / "artifact.json"
    trajectories_root = tmp_path / "episodes"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    (trajectories_root / "repo_fix_task.json").write_text(
        json.dumps(
            {
                "task_id": "repo_fix_task",
                "prompt": "Repair the repository changelog.",
                "success": True,
                "task_metadata": {"benchmark_family": "repository", "difficulty": "long_horizon"},
                "steps": [
                    {"action": "code_execute", "content": "git status", "verification": {"passed": True}},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "propose_qwen_adapter_update.py",
            "--base-model",
            "Qwen/Qwen3.5-9B",
            "--focus",
            "teacher_shadow",
            "--artifact-path",
            str(artifact_path),
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "qwen_adapter_bundle"
    assert payload["base_model_name"] == "Qwen/Qwen3.5-9B"
    assert payload["generation_focus"] == "teacher_shadow"
    assert stream.getvalue().strip().endswith("artifact.json")
