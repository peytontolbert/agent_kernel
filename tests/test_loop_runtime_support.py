from pathlib import Path

import pytest

from agent_kernel.ops.loop_runtime_support import materialize_setup_file_contents
from agent_kernel.schemas import TaskSpec


def test_materialize_setup_file_contents_writes_metadata_fixtures(tmp_path: Path):
    task = TaskSpec(
        task_id="fixture_task",
        prompt="read source context",
        workspace_subdir="fixture_task",
        metadata={
            "setup_file_contents": {
                "source_context/pkg/module.py": "def value():\n    return 1\n",
            }
        },
    )

    materialize_setup_file_contents(task, tmp_path)

    assert (tmp_path / "source_context/pkg/module.py").read_text(encoding="utf-8") == "def value():\n    return 1\n"


def test_materialize_setup_file_contents_rejects_escape_paths(tmp_path: Path):
    task = TaskSpec(
        task_id="fixture_task",
        prompt="read source context",
        workspace_subdir="fixture_task",
        metadata={"setup_file_contents": {"../escape.py": "bad\n"}},
    )

    with pytest.raises(ValueError, match="unsafe setup file path"):
        materialize_setup_file_contents(task, tmp_path)
