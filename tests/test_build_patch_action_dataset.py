from pathlib import Path
import importlib.util
import json
import sys

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "build_patch_action_dataset.py"
    spec = importlib.util.spec_from_file_location("build_patch_action_dataset", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch() -> str:
    return (
        "diff --git a/pkg/runtime.py b/pkg/runtime.py\n"
        "--- a/pkg/runtime.py\n"
        "+++ b/pkg/runtime.py\n"
        "@@ -1,3 +1,8 @@\n"
        "+import subprocess\n"
        " def run(cmd):\n"
        "-    return subprocess.run(cmd)\n"
        "+    try:\n"
        "+        return subprocess.run(cmd, timeout=5)\n"
        "+    except subprocess.TimeoutExpired:\n"
        "+        return {'error': 'timeout', 'retryable': True}\n"
        "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n"
        "--- a/tests/test_runtime.py\n"
        "+++ b/tests/test_runtime.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_timeout():\n"
        "+    assert True\n"
    )


def test_analyze_patch_diff_labels_timeout_runtime_and_tests():
    module = _load_module()
    analysis = module.analyze_patch_diff(_patch())

    labels = {item["key"] for item in analysis["intents"]}
    assert "behavior.timeout_handling" in labels
    assert "edit.exception_handling" in labels
    assert "runtime.command_execution" in labels
    assert "test.add_or_update" in labels
    assert analysis["patch_operator"]["edit_shape"] == "multi_file"
    assert analysis["patch_operator"]["has_tests"] is True
    assert analysis["source_paths"] == ["pkg/runtime.py"]
    assert analysis["test_paths"] == ["tests/test_runtime.py"]


def test_build_patch_action_dataset_joins_queue_predictions_and_skill_cards(tmp_path):
    module = _load_module()
    patch_path = tmp_path / "patches" / "repo__pkg-1.diff"
    patch_path.parent.mkdir()
    patch_path.write_text(_patch(), encoding="utf-8")
    prediction_task_manifest = {
        "prediction_manifest": {
            "base_dir": str(patch_path.parent),
            "predictions": [
                {
                    "instance_id": "repo__pkg-1",
                    "patch_path": patch_path.name,
                }
            ],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_repo__pkg-1",
                "prompt": "Fix timeout recovery.",
                "workspace_subdir": "work/repo__pkg-1",
                "metadata": {
                    "swe_instance_id": "repo__pkg-1",
                    "repo": "repo/pkg",
                    "base_commit": "abc123",
                    "action_decisions": [
                        {
                            "selected_skill_id": "skill-timeout",
                            "selected_retrieval_span_id": "span-timeout",
                            "retrieval_influenced": True,
                            "retrieval_ranked_skill": True,
                        }
                    ],
                },
            }
        ]
    }
    cards_jsonl = tmp_path / "cards.jsonl"
    cards_jsonl.write_text(
        json.dumps(
            {
                "event": "skill_card",
                "id": "skill-timeout",
                "metadata": {"primitive_type": "code_symbol"},
                "required_permissions": ["run_shell"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = module.build_patch_action_dataset(
        prediction_task_manifest=prediction_task_manifest,
        queue_manifest=queue_manifest,
        skill_cards_jsonl=str(cards_jsonl),
        patch_job_verification={"successful_instance_ids": ["repo__pkg-1"]},
    )

    assert len(examples) == 1
    example = examples[0]
    assert example["schema_version"] == module.SCHEMA_VERSION
    assert example["task"]["repo"] == "repo/pkg"
    assert example["task"]["base_commit"] == "abc123"
    assert example["verification"]["passed"] is True
    assert example["retrieval"]["selected_skill_ids"] == ["skill-timeout"]
    assert example["retrieval"]["selected_retrieval_span_ids"] == ["span-timeout"]
    assert example["retrieval"]["retrieval_influenced"] is True
    assert example["retrieval"]["selected_skill_cards"][0]["id"] == "skill-timeout"
    assert "behavior.timeout_handling" in example["model_targets"]["path_labels"]
    assert example["model_targets"]["scalar_features"]["selected_skill_count"] == 1


def test_materialize_hybrid_examples_matches_tolbert_trainer_shape(tmp_path):
    module = _load_module()
    from agent_kernel.modeling.tolbert.config import HybridTolbertSSMConfig

    patch_path = tmp_path / "patches" / "repo__pkg-1.diff"
    patch_path.parent.mkdir()
    patch_path.write_text(_patch(), encoding="utf-8")
    prediction_task_manifest = {
        "prediction_manifest": {
            "base_dir": str(patch_path.parent),
            "predictions": [{"instance_id": "repo__pkg-1", "patch_path": patch_path.name}],
        }
    }
    queue_manifest = {
        "tasks": [
            {
                "task_id": "swe_patch_repo__pkg-1",
                "prompt": "Fix timeout recovery.",
                "metadata": {
                    "swe_instance_id": "repo__pkg-1",
                    "repo": "repo/pkg",
                    "base_commit": "abc123",
                    "action_decisions": [
                        {
                            "selected_skill_id": "skill-timeout",
                            "retrieval_influenced": True,
                            "retrieval_ranked_skill": True,
                        }
                    ],
                },
            }
        ]
    }
    examples = module.build_patch_action_dataset(
        prediction_task_manifest=prediction_task_manifest,
        queue_manifest=queue_manifest,
        patch_job_verification={"successful_instance_ids": ["repo__pkg-1"]},
    )
    config = HybridTolbertSSMConfig(sequence_length=4, max_command_tokens=8, max_path_levels=5, scalar_feature_dim=12)

    hybrid_examples, decoder_vocab = module.materialize_hybrid_examples(examples, config=config)

    assert decoder_vocab
    assert len(hybrid_examples) == 1
    hybrid = hybrid_examples[0]
    assert len(hybrid["path_level_ids"]) == config.max_path_levels
    assert len(hybrid["command_token_ids"]) == config.sequence_length
    assert all(len(row) == config.max_command_tokens for row in hybrid["command_token_ids"])
    assert len(hybrid["scalar_features"]) == config.sequence_length
    assert all(len(row) == config.scalar_feature_dim for row in hybrid["scalar_features"])
    assert len(hybrid["world_target"]) == config.world_state_dim
    assert len(hybrid["transition_target"]) == 2
    assert hybrid["stop_target"] == 1.0
    assert hybrid["example_weight"] > 1.0
    assert hybrid["source_example_id"] == examples[0]["example_id"]


def test_write_dataset_supports_compressed_parquet(tmp_path):
    module = _load_module()
    pq = pytest.importorskip("pyarrow.parquet")

    records = [
        {
            "schema_version": module.SCHEMA_VERSION,
            "example_id": "ex-1",
            "instance_id": "repo__pkg-1",
            "task": {"repo": "repo/pkg", "prompt": "Fix timeout recovery."},
            "patch": {"changed_paths": ["pkg/runtime.py"], "intents": [{"key": "behavior.timeout_handling"}]},
            "retrieval": {"selected_skill_ids": ["skill-timeout"]},
            "verification": {"passed": True, "outcome": "success"},
            "model_targets": {"path_labels": ["behavior.timeout_handling"]},
        }
    ]
    output = tmp_path / "patch_action_examples.parquet"

    dataset_format = module.write_dataset(output, records)

    assert dataset_format == "parquet"
    table = pq.read_table(output)
    assert table.num_rows == 1
    assert table.column("example_id").to_pylist() == ["ex-1"]
