from pathlib import Path
import importlib.util
import json
import sys


def _load_queue_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_queue_manifest.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_queue_manifest", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_task_manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "report_kind": "swe_bench_prediction_task_manifest",
        "tasks": [
            {
                "instance_id": "django__django-1",
                "model_name_or_path": "agentkernel",
                "repo": "django/django",
                "base_commit": "abc123",
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": str(tmp_path / "patches" / "django__django-1.diff"),
                "problem_statement": "Fix timezone parsing regression.",
                "hints_text": "Look at parser edge cases.",
                "candidate_files": ["django/utils/dateparse.py", "tests/test_time.py"],
                "source_context": [
                    {
                        "path": "django/utils/dateparse.py",
                        "content": "def parse_datetime(value):\n    return value\n",
                        "truncated": False,
                    }
                ],
                "fail_to_pass": ["tests/test_time.py::test_parse"],
                "pass_to_pass": ["tests/test_time.py::test_existing"],
            }
        ],
    }


def test_build_swe_queue_manifest_creates_external_tasks(tmp_path):
    module = _load_queue_module()

    manifest = module.build_swe_queue_manifest(_prediction_task_manifest(tmp_path))

    assert manifest["manifest_kind"] == "swe_bench_patch_generation_queue_manifest"
    assert len(manifest["tasks"]) == 1
    task = manifest["tasks"][0]
    assert task["task_id"] == "swe_patch_django__django-1"
    assert task["workspace_subdir"] == "swe_bench_predictions/swe_patch_django__django-1"
    assert task["expected_files"] == ["patch.diff"]
    assert task["setup_commands"] == []
    assert "patch.diff" in task["success_command"]
    assert "django/utils/dateparse\\.py" in task["success_command"]
    assert "placeholder" in task["success_command"]
    assert "django/utils/dateparse.py" in task["prompt"]
    assert "source_context/django/utils/dateparse.py" in task["prompt"]
    assert "source_lines/django/utils/dateparse.py.lines" in task["prompt"]
    assert "line-numbered files to choose exact hunk anchors" in task["prompt"]
    assert "Do not use git show" in task["prompt"]
    assert "sed -n '1,120p' astropy/<path>" in task["prompt"]
    assert "do not run git" in task["prompt"]
    assert "fake imports" in task["prompt"]
    assert "applies cleanly" in task["prompt"]
    assert "do not invent files" in task["prompt"]
    assert "def parse_datetime(value):" in task["prompt"]
    assert task["metadata"]["swe_instance_id"] == "django__django-1"
    assert task["metadata"]["swe_candidate_files"] == ["django/utils/dateparse.py", "tests/test_time.py"]
    assert task["metadata"]["setup_file_contents"] == {
        "django/utils/dateparse.py": "def parse_datetime(value):\n    return value\n",
        "source_context/django/utils/dateparse.py": "def parse_datetime(value):\n    return value\n",
        "source_lines/django/utils/dateparse.py.lines": "   1: def parse_datetime(value):\n   2:     return value\n   3: \n",
    }
    assert task["metadata"]["semantic_verifier"]["kind"] == "swe_patch_apply_check"
    assert task["metadata"]["semantic_verifier"]["repo_cache_root"] == str(tmp_path / "repos")
    assert task["metadata"]["semantic_verifier"]["expected_changed_paths"] == [
        "django/utils/dateparse.py",
        "tests/test_time.py",
    ]
    assert task["metadata"]["swe_patch_output_path"].endswith("django__django-1.diff")
    assert task["metadata"]["workflow_guard"]["managed_paths"] == [
        "patch.diff",
        "source_context/django/utils/dateparse.py",
    ]


def test_prepare_swe_bench_queue_manifest_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    module = _load_queue_module()
    input_path = tmp_path / "prediction_tasks.json"
    output_path = tmp_path / "queue_manifest.json"
    input_path.write_text(json.dumps(_prediction_task_manifest(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_queue_manifest.py",
            "--prediction-task-manifest",
            str(input_path),
            "--output-manifest-json",
            str(output_path),
        ],
    )

    module.main()

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(manifest["tasks"]) == 1
    assert f"output_manifest_json={output_path}" in capsys.readouterr().out


def test_build_swe_queue_manifest_accepts_workspace_prefix(tmp_path):
    module = _load_queue_module()

    manifest = module.build_swe_queue_manifest(
        _prediction_task_manifest(tmp_path),
        workspace_prefix="swe_source_probe",
    )

    assert manifest["tasks"][0]["workspace_subdir"] == "swe_source_probe/swe_patch_django__django-1"


def test_candidate_file_success_command_rejects_unrelated_diff():
    module = _load_queue_module()

    command = module._candidate_file_success_command(["django/utils/dateparse.py"])

    assert "grep -Eq 'django/utils/dateparse\\.py' patch.diff" in command
    assert "! grep -Eiq" in command
