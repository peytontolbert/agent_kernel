from __future__ import annotations

from pathlib import Path
import json

from agent_kernel.config import KernelConfig
from agent_kernel.modeling.training.corpus_acquisition import external_corpus_examples, materialize_external_corpus
from agent_kernel.modeling.training.universal_dataset import materialize_universal_decoder_dataset


def test_materialize_universal_decoder_dataset_collects_local_sources(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "agent_kernel").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "docs" / "guide.md").write_text(
        "This autonomous kernel keeps verifier trust and recovery aligned with controlled model takeover.\n\n"
        "It builds retained checkpoints from local signals without requiring a human curator.",
        encoding="utf-8",
    )
    (repo_root / "agent_kernel" / "sample.py").write_text(
        "def alpha():\n    return 'alpha'\n\ndef beta():\n    return 'beta'\n",
        encoding="utf-8",
    )
    trajectories = tmp_path / "episodes"
    trajectories.mkdir()
    (trajectories / "episode.json").write_text(
        json.dumps(
            {
                "task_id": "hello",
                "prompt": "create file",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "steps": [
                    {
                        "action": "code_execute",
                        "content": "printf 'hello\\n' > hello.txt",
                        "verification": {"passed": True},
                    },
                    {"action": "respond", "content": "done"},
                ],
            }
        ),
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "report.json").write_text(
        json.dumps(
            {
                "task_id": "hello",
                "task_metadata": {"benchmark_family": "workflow"},
                "acceptance_packet": {
                    "selected_edits": [{"path": "hello.txt", "kind": "rewrite"}],
                    "verifier_result": {"passed": True},
                    "capability_usage": {"workspace_exec": 1},
                    "tests": ["pytest -q"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(trajectories_root=trajectories, run_reports_dir=reports)

    manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["artifact_kind"] == "tolbert_universal_decoder_dataset"
    assert Path(manifest["train_dataset_path"]).exists()
    assert Path(manifest["eval_dataset_path"]).exists()
    assert Path(manifest["decoder_vocab_path"]).exists()
    assert Path(manifest["decoder_tokenizer_manifest_path"]).exists()
    assert manifest["decoder_tokenizer_stats"]["tokenizer_kind"] == "regex_v1"
    assert manifest["total_examples"] >= 3
    assert "trajectory_step" in manifest["source_counts"]
    assert "trajectory_success_command" in manifest["source_counts"]


def test_materialize_universal_decoder_dataset_removes_stale_shards_by_default(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "agent_kernel").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "docs" / "guide.md").write_text("dataset compaction guidance", encoding="utf-8")
    trajectories = tmp_path / "episodes"
    trajectories.mkdir()
    (trajectories / "episode.json").write_text(
        json.dumps(
            {
                "task_id": "cleanup",
                "prompt": "clean stale shards",
                "success": True,
                "task_metadata": {"benchmark_family": "workflow"},
                "steps": [{"action": "code_execute", "content": "printf hi", "verification": {"passed": True}}],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    stale_shards = output_dir / "shards"
    stale_shards.mkdir(parents=True, exist_ok=True)
    (stale_shards / "stale.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    config = KernelConfig(trajectories_root=trajectories, run_reports_dir=tmp_path / "reports")

    manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
    )

    assert manifest["train_shard_paths"] == []
    assert manifest["eval_shard_paths"] == []
    assert not stale_shards.exists()


def test_materialize_universal_decoder_dataset_tracks_long_horizon_examples(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "agent_kernel").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "docs" / "guide.md").write_text("Long-horizon guidance.", encoding="utf-8")
    trajectories = tmp_path / "episodes"
    trajectories.mkdir()
    (trajectories / "episode.json").write_text(
        json.dumps(
            {
                "task_id": "project_handoff",
                "prompt": "advance the project handoff",
                "success": True,
                "task_metadata": {
                    "benchmark_family": "project",
                    "difficulty": "long_horizon",
                    "synthetic_edit_plan": [{"path": "handoff.txt", "edit_kind": "line_replace"}],
                },
                "world_model_summary": {
                    "missing_expected_artifacts": ["handoff.txt"],
                    "workflow_report_paths": ["reports/handoff_review.txt"],
                    "workflow_required_tests": ["handoff smoke"],
                },
                "steps": [
                    {
                        "action": "code_execute",
                        "active_subgoal": "materialize expected artifact handoff.txt",
                        "state_progress_delta": 0.0,
                        "state_transition": {"no_progress": True, "progress_delta": 0.0},
                        "verification": {"passed": False},
                        "content": "python scripts/fix_handoff.py --path handoff.txt",
                    },
                    {
                        "action": "code_execute",
                        "active_subgoal": "write workflow report reports/handoff_review.txt",
                        "content": "printf 'ready\\n' > handoff.txt",
                        "verification": {"passed": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(trajectories_root=trajectories, run_reports_dir=tmp_path / "reports")

    manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=tmp_path / "out",
    )

    combined = (
        Path(manifest["train_dataset_path"]).read_text(encoding="utf-8")
        + Path(manifest["eval_dataset_path"]).read_text(encoding="utf-8")
    )

    assert manifest["difficulty_counts"]["long_horizon"] >= 1
    assert manifest["long_horizon_example_count"] >= 1
    assert "difficulty: long_horizon" in combined
    assert "trajectory_software_work_command" in manifest["source_counts"]
    assert "software_work_agenda:" in combined
    assert "software_work_phase_state:" in combined
    assert "suggested_phase:" in combined
    assert "software_work_recent_outcomes:" in combined
    assert "stalled" in combined


def test_materialize_external_corpus_fetches_enabled_module_sources(tmp_path: Path) -> None:
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "generic_http_docs",
                        "adapter_kind": "generic_http",
                        "enabled": True,
                        "capabilities": ["http_service_read"],
                        "settings": {
                            "http_allowed_hosts": ["example.com"],
                            "corpus_seed_urls": ["https://example.com/corpus"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        capability_modules_path=modules_path,
        unattended_allow_http_requests=True,
        unattended_http_allowed_hosts=("example.com",),
    )

    class _Response:
        status = 200
        headers = {"Content-Type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, n: int = -1):
            del n
            return b"Autonomous corpus collection expands the retained decoder without manual curation."

    manifest = materialize_external_corpus(
        config=config,
        output_dir=tmp_path / "corpus",
        opener=lambda req, timeout=0: _Response(),
    )
    examples = external_corpus_examples(manifest)

    assert manifest["artifact_kind"] == "tolbert_external_corpus_manifest"
    assert manifest["fetched_count"] == 1
    assert manifest["quality"]["usable_record_count"] == 1
    assert manifest["quality"]["average_tokens_per_usable_record"] > 5
    assert Path(manifest["records_path"]).exists()
    assert examples
    assert examples[0]["source_type"] == "external_corpus"


def test_external_corpus_examples_decode_github_readme_json(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "module_id": "github",
                "adapter_kind": "github",
                "url": "https://api.github.com/repos/openai/agentkernel/readme",
                "status": "ok",
                "content_type": "application/json",
                "body_text": json.dumps(
                        {
                            "encoding": "base64",
                            "content": "IyBBZ2VudEtlcm5lbAoKQXV0b25vbW91cyBrZXJuZWwgd2l0aCBjb250cm9sbGVkIGxpZnRvZmYgYW5kIGJyb2FkZXIgZGF0YSBhY3F1aXNpdGlvbi4K",
                        }
                    ),
                }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = external_corpus_examples({"records_path": str(records_path)})

    assert examples
    assert "AgentKernel Autonomous kernel with controlled liftoff and broader data acquisition." in (
        examples[0]["prompt"] + " " + examples[0]["target"]
    )


def test_materialize_universal_decoder_dataset_includes_external_corpus(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "docs" / "guide.md").write_text(
        "Local docs are present but the external corpus should also be merged into the dataset.",
        encoding="utf-8",
    )
    trajectories = tmp_path / "episodes"
    trajectories.mkdir()
    config = KernelConfig(trajectories_root=trajectories, run_reports_dir=tmp_path / "reports")
    monkeypatch.setattr(
        "agent_kernel.modeling.training.universal_dataset.materialize_external_corpus",
        lambda **kwargs: {
            "artifact_kind": "tolbert_external_corpus_manifest",
            "manifest_path": str(tmp_path / "external" / "manifest.json"),
            "records_path": str(tmp_path / "external" / "records.jsonl"),
            "fetched_count": 1,
        },
    )
    monkeypatch.setattr(
        "agent_kernel.modeling.training.universal_dataset.external_corpus_examples",
        lambda manifest: [
            {
                "source_type": "external_corpus",
                "source_id": "generic:http",
                "prompt": "external prompt section",
                "target": "external target continuation for training",
            }
        ],
    )

    manifest = materialize_universal_decoder_dataset(
        config=config,
        repo_root=repo_root,
        output_dir=tmp_path / "out",
    )

    combined_lines = (
        Path(manifest["train_dataset_path"]).read_text(encoding="utf-8")
        + Path(manifest["eval_dataset_path"]).read_text(encoding="utf-8")
    )
    assert manifest["external_corpus_fetched_count"] == 1
    assert manifest["external_corpus_example_count"] == 1
    assert manifest["external_corpus_quality"] == {}
    assert manifest["decoder_vocab_size"] > 0
    assert "external_corpus" in manifest["source_counts"]
    assert "external target continuation for training" in combined_lines
