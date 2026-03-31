from pathlib import Path
import importlib.util
from io import StringIO
import sys

from agent_kernel.config import KernelConfig


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selfplay_config(tmp_path: Path) -> KernelConfig:
    return KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        skills_path=tmp_path / "skills" / "command_skills.json",
    )


def test_run_selfplay_chain_progresses_beyond_one_hop(tmp_path: Path):
    module = _load_script("run_selfplay.py")
    seed, followups = module.run_selfplay_chain(config=_selfplay_config(tmp_path), max_followup_hops=3)

    assert seed.task_id == "hello_task"
    assert seed.success is True
    assert [task.task_id for task, _ in followups] == [
        "hello_task_followup",
        "hello_task_followup_adjacent",
        "hello_task_followup_adjacent_adjacent",
    ]
    assert [task.metadata["parent_task"] for task, _ in followups] == [
        "hello_task",
        "hello_task_followup",
        "hello_task_followup_adjacent",
    ]
    assert all(result.success for _, result in followups)


def test_run_selfplay_chain_failure_seed_recovers_and_continues(tmp_path: Path):
    module = _load_script("run_selfplay.py")
    seed, followups = module.run_selfplay_chain(
        config=_selfplay_config(tmp_path),
        seed_mode="failure",
        max_followup_hops=2,
    )

    assert seed.task_id == "hello_task"
    assert seed.success is False
    assert len(followups) == 2
    assert followups[0][0].metadata["curriculum_kind"] == "failure_recovery"
    assert followups[0][1].success is True
    assert followups[1][0].metadata["curriculum_kind"] == "adjacent_success"
    assert followups[1][0].metadata["parent_task"] == followups[0][1].task_id


def test_run_selfplay_main_reports_multi_hop_progression(tmp_path: Path, monkeypatch):
    module = _load_script("run_selfplay.py")
    monkeypatch.setattr(module, "KernelConfig", lambda: _selfplay_config(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_selfplay.py", "--provider", "mock", "--max-followup-hops", "2"],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    assert (
        stream.getvalue().strip()
        == "seed=hello_task:True "
        "hop1=hello_task_followup:True:adjacent_success "
        "hop2=hello_task_followup_adjacent:True:adjacent_success "
        "hops=2"
    )
