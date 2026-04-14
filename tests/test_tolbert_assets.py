from __future__ import annotations

import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.memory import EpisodeMemory
from agent_kernel.extensions.tolbert_assets import (
    build_agentkernel_tolbert_assets,
    materialize_retained_retrieval_asset_bundle,
    retained_tolbert_runtime_paths,
)
from agent_kernel.extensions.improvement.tolbert_model_improvement import build_tolbert_supervised_dataset_manifest
from agent_kernel.schemas import EpisodeRecord, StepRecord


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_agentkernel_tolbert_assets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skills_path = tmp_path / "skills" / "command_skills.json"
    tool_candidates_path = tmp_path / "tools" / "tool_candidates.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    tool_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [
                    {
                        "skill_id": "skill:math_task:primary",
                        "source_task_id": "math_task",
                        "procedure": {"commands": ["printf '42\\n' > result.txt"]},
                        "quality": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tool_candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "replay_verified",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "benchmark_family": "integration",
                        "quality": 0.9,
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "replay_verified",
                        "procedure": {
                            "commands": [
                                "mkdir -p gateway services reports",
                                "printf 'routes synced\\n' > gateway/routes.txt",
                            ]
                        },
                        "verifier": {"termination_reason": "success"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    kernel_config = KernelConfig(
        skills_path=skills_path,
        tool_candidates_path=tool_candidates_path,
    )
    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=tmp_path,
        base_model_name="bert-base-uncased",
        config=kernel_config,
    )

    nodes = _load_jsonl(paths.nodes_path)
    source_spans = _load_jsonl(paths.source_spans_path)
    model_spans = _load_jsonl(paths.model_spans_path)
    label_map = json.loads(paths.label_map_path.read_text(encoding="utf-8"))
    config = json.loads(paths.config_path.read_text(encoding="utf-8"))

    assert nodes[0]["node_id"] == 0
    assert nodes[0]["level"] == 0
    assert any(node["name"] == "Tasks" for node in nodes)
    assert any(node["name"] == "Docs" for node in nodes)
    assert any(node["name"] == "Capabilities" for node in nodes)
    assert any(node["name"] == "Failures" for node in nodes)
    assert any(node["name"] == "Procedures" for node in nodes)
    assert any(node["name"] == "Tools" for node in nodes)
    assert any(node["name"] == "Artifacts" for node in nodes)

    span_ids = {span["span_id"] for span in source_spans}
    assert "task:hello_task:overview" in span_ids
    assert "skill:math_task:sequence" in span_ids
    assert any(span["span_id"].startswith("capability:file_write:task:") for span in source_spans)
    assert any(span["span_id"].startswith("tool:text_write:task:") for span in source_spans)
    assert any(span["span_id"].startswith("artifact:text_file:task:") for span in source_spans)
    assert "tool:service_mesh_task:candidate" in span_ids
    assert any(span["meta"]["span_type"] == "agent:procedure_span" for span in source_spans)
    assert any(span["meta"]["span_type"] == "agent:episode_step" for span in source_spans)
    assert any(span["meta"]["span_type"] == "agent:failure_pattern" for span in source_spans)
    assert all(span["node_path"][0] == 0 for span in source_spans)
    assert all(len(span["node_path"]) == 4 for span in source_spans)
    assert all(span["node_path"][0] == 0 for span in model_spans)
    assert all(source["span_id"] == model["span_id"] for source, model in zip(source_spans, model_spans))
    assert any(source["node_path"] != model["node_path"] for source, model in zip(source_spans, model_spans))

    assert set(label_map) == {"1", "2", "3"}
    assert config["base_model_name"] == "bert-base-uncased"
    assert config["spans_files"] == [str(paths.model_spans_path)]
    assert "retrieval_asset_controls" in config
    assert config["level_sizes"] == {str(level): len(mapping) for level, mapping in label_map.items()}


def test_build_agentkernel_tolbert_assets_includes_edit_patch_spans(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    trajectories_root = tmp_path / "episodes"
    config = KernelConfig(trajectories_root=trajectories_root)
    memory = EpisodeMemory(trajectories_root, config=config)
    memory.save(
        EpisodeRecord(
            task_id="patch_episode_task",
            prompt="Repair the release review.",
            workspace=str(tmp_path / "workspace" / "patch_episode_task"),
            success=True,
            task_metadata={"benchmark_family": "repository"},
            task_contract={"prompt": "Repair the release review.", "workspace_subdir": "patch_episode_task"},
            termination_reason="success",
            steps=[
                StepRecord(
                    index=1,
                    thought="repair review",
                    action="code_execute",
                    content="printf 'READY\\n' > reports/release_review.txt",
                    selected_skill_id=None,
                    command_result={"command": "printf 'READY\\n' > reports/release_review.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                    verification={"passed": True, "reasons": ["verification passed"]},
                    state_transition={
                        "progress_delta": 0.5,
                        "edit_patches": [
                            {
                                "path": "reports/release_review.txt",
                                "status": "modified",
                                "patch": "--- a/reports/release_review.txt\n+++ b/reports/release_review.txt\n@@ -1 +1 @@\n-BROKEN\n+READY\n",
                                "patch_summary": "modified reports/release_review.txt (+1 -1)",
                            }
                        ],
                    },
                )
            ],
        )
    )

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=tmp_path / "assets",
        base_model_name="bert-base-uncased",
        config=config,
    )
    source_spans = _load_jsonl(paths.source_spans_path)

    assert any(span["meta"]["span_type"] == "agent:edit_patch" for span in source_spans)


def test_build_agentkernel_tolbert_assets_filters_unretained_memory_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skills_path = tmp_path / "skills" / "command_skills.json"
    tool_candidates_path = tmp_path / "tools" / "tool_candidates.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    tool_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "rejected",
                "retention_decision": {"state": "reject"},
                "skills": [
                    {
                        "skill_id": "skill:math_task:primary",
                        "source_task_id": "math_task",
                        "procedure": {"commands": ["printf '42\\n' > result.txt"]},
                        "quality": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tool_candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "benchmark_family": "integration",
                        "quality": 0.9,
                        "promotion_stage": "candidate_procedure",
                        "lifecycle_state": "candidate",
                        "procedure": {"commands": ["printf 'routes synced\\n' > gateway/routes.txt"]},
                    },
                    {
                        "tool_id": "tool:rejected:primary",
                        "source_task_id": "service_mesh_task",
                        "benchmark_family": "integration",
                        "quality": 0.9,
                        "promotion_stage": "replay_verified",
                        "lifecycle_state": "rejected",
                        "retention_decision": {"state": "reject"},
                        "procedure": {"commands": ["printf 'bad\\n' > gateway/routes.txt"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    kernel_config = KernelConfig(
        skills_path=skills_path,
        tool_candidates_path=tool_candidates_path,
    )

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=tmp_path,
        config=kernel_config,
    )

    source_spans = _load_jsonl(paths.source_spans_path)
    span_ids = {span["span_id"] for span in source_spans}

    assert "skill:math_task:sequence" not in span_ids
    assert "tool:service_mesh_task:candidate" not in span_ids


def test_build_agentkernel_tolbert_assets_applies_retained_retrieval_asset_controls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "asset_controls": {
                    "include_tool_candidate_spans": False,
                    "include_episode_step_spans": False,
                    "include_negative_episode_spans": True,
                },
            }
        ),
        encoding="utf-8",
    )
    tool_candidates_path = tmp_path / "tools" / "tool_candidates.json"
    tool_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    tool_candidates_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tool_candidate_set",
                "lifecycle_state": "retained",
                "candidates": [
                    {
                        "tool_id": "tool:service_mesh_task:primary",
                        "source_task_id": "service_mesh_task",
                        "benchmark_family": "integration",
                        "quality": 0.9,
                        "promotion_stage": "promoted_tool",
                        "procedure": {"commands": ["printf 'routes synced\\n' > gateway/routes.txt"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    kernel_config = KernelConfig(
        retrieval_proposals_path=retrieval_path,
        tool_candidates_path=tool_candidates_path,
        use_retrieval_proposals=True,
    )

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=tmp_path,
        config=kernel_config,
    )

    source_spans = _load_jsonl(paths.source_spans_path)
    span_ids = {span["span_id"] for span in source_spans}
    config_payload = json.loads(paths.config_path.read_text(encoding="utf-8"))

    assert "tool:service_mesh_task:candidate" not in span_ids
    assert not any(span["meta"]["span_type"] == "agent:episode_step" for span in source_spans)
    assert config_payload["retrieval_asset_controls"]["include_tool_candidate_spans"] is False


def test_build_agentkernel_tolbert_assets_uses_configured_trajectory_root_and_failure_alignment(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    episodes_root = tmp_path / "episodes"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "asset_controls": {
                    "include_episode_step_spans": True,
                    "include_negative_episode_spans": True,
                    "prefer_failure_alignment": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (episodes_root / "custom_success.json").write_text(
        json.dumps(
            {
                "task_id": "custom_success",
                "success": True,
                "steps": [
                    {
                        "index": 1,
                        "thought": "write success",
                        "action": "code_execute",
                        "content": "printf 'ok\\n' > ok.txt",
                        "verification": {"passed": True, "reasons": []},
                        "command_result": {"exit_code": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (episodes_root / "custom_failure.json").write_text(
        json.dumps(
            {
                "task_id": "custom_failure",
                "success": False,
                "steps": [
                    {
                        "index": 1,
                        "thought": "write failure",
                        "action": "code_execute",
                        "content": "false",
                        "verification": {"passed": False, "reasons": ["exit code was 1"]},
                        "command_result": {"exit_code": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    kernel_config = KernelConfig(
        retrieval_proposals_path=retrieval_path,
        trajectories_root=episodes_root,
        use_retrieval_proposals=True,
    )

    paths = build_agentkernel_tolbert_assets(
        repo_root=repo_root,
        output_dir=tmp_path,
        config=kernel_config,
    )

    source_spans = _load_jsonl(paths.source_spans_path)
    span_ids = {span["span_id"] for span in source_spans}


def test_retained_tolbert_runtime_paths_prefers_retained_tolbert_model_artifact(tmp_path: Path) -> None:
    retrieval_bundle = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    tolbert_model_artifact = tmp_path / "tolbert_model" / "tolbert_model_artifact.json"
    retrieval_bundle.parent.mkdir(parents=True, exist_ok=True)
    tolbert_model_artifact.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        tmp_path / "baseline.json",
        tmp_path / "baseline.pt",
        tmp_path / "baseline_nodes.jsonl",
        tmp_path / "baseline_label_map.json",
        tmp_path / "baseline_spans.jsonl",
        tmp_path / "baseline_cache.pt",
        tmp_path / "candidate.json",
        tmp_path / "candidate.pt",
        tmp_path / "candidate_nodes.jsonl",
        tmp_path / "candidate_label_map.json",
        tmp_path / "candidate_spans.jsonl",
        tmp_path / "candidate_cache.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    retrieval_bundle.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(tmp_path / "baseline.json"),
                    "checkpoint_path": str(tmp_path / "baseline.pt"),
                    "nodes_path": str(tmp_path / "baseline_nodes.jsonl"),
                    "label_map_path": str(tmp_path / "baseline_label_map.json"),
                    "source_spans_paths": [str(tmp_path / "baseline_spans.jsonl")],
                    "cache_paths": [str(tmp_path / "baseline_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    tolbert_model_artifact.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(tmp_path / "candidate.json"),
                    "checkpoint_path": str(tmp_path / "candidate.pt"),
                    "nodes_path": str(tmp_path / "candidate_nodes.jsonl"),
                    "label_map_path": str(tmp_path / "candidate_label_map.json"),
                    "source_spans_paths": [str(tmp_path / "candidate_spans.jsonl")],
                    "cache_paths": [str(tmp_path / "candidate_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        retrieval_asset_bundle_path=retrieval_bundle,
        tolbert_model_artifact_path=tolbert_model_artifact,
        use_tolbert_model_artifacts=True,
    )

    runtime_paths = retained_tolbert_runtime_paths(config, repo_root=tmp_path)

    assert runtime_paths["tolbert_checkpoint_path"] == str(tmp_path / "candidate.pt")
    assert runtime_paths["tolbert_cache_paths"] == (str(tmp_path / "candidate_cache.pt"),)


def test_retained_tolbert_runtime_paths_ignores_conflicting_rejected_retained_model_artifact(tmp_path: Path) -> None:
    retrieval_bundle = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    tolbert_model_artifact = tmp_path / "tolbert_model" / "tolbert_model_artifact.json"
    retrieval_bundle.parent.mkdir(parents=True, exist_ok=True)
    tolbert_model_artifact.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        tmp_path / "baseline.json",
        tmp_path / "baseline.pt",
        tmp_path / "baseline_nodes.jsonl",
        tmp_path / "baseline_label_map.json",
        tmp_path / "baseline_spans.jsonl",
        tmp_path / "baseline_cache.pt",
        tmp_path / "candidate.json",
        tmp_path / "candidate.pt",
        tmp_path / "candidate_nodes.jsonl",
        tmp_path / "candidate_label_map.json",
        tmp_path / "candidate_spans.jsonl",
        tmp_path / "candidate_cache.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    retrieval_bundle.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(tmp_path / "baseline.json"),
                    "checkpoint_path": str(tmp_path / "baseline.pt"),
                    "nodes_path": str(tmp_path / "baseline_nodes.jsonl"),
                    "label_map_path": str(tmp_path / "baseline_label_map.json"),
                    "source_spans_paths": [str(tmp_path / "baseline_spans.jsonl")],
                    "cache_paths": [str(tmp_path / "baseline_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    tolbert_model_artifact.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "reject"},
                "runtime_paths": {
                    "config_path": str(tmp_path / "candidate.json"),
                    "checkpoint_path": str(tmp_path / "candidate.pt"),
                    "nodes_path": str(tmp_path / "candidate_nodes.jsonl"),
                    "label_map_path": str(tmp_path / "candidate_label_map.json"),
                    "source_spans_paths": [str(tmp_path / "candidate_spans.jsonl")],
                    "cache_paths": [str(tmp_path / "candidate_cache.pt")],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        retrieval_asset_bundle_path=retrieval_bundle,
        tolbert_model_artifact_path=tolbert_model_artifact,
        use_tolbert_model_artifacts=True,
    )

    runtime_paths = retained_tolbert_runtime_paths(config, repo_root=tmp_path)

    assert runtime_paths["tolbert_checkpoint_path"] == str(tmp_path / "baseline.pt")
    assert runtime_paths["tolbert_cache_paths"] == (str(tmp_path / "baseline_cache.pt"),)


def test_materialize_retained_retrieval_asset_bundle_writes_manifest_and_assets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    checkpoint_path = tmp_path / "tolbert" / "checkpoint.pt"
    cache_path = tmp_path / "tolbert" / "cache.pt"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "asset_strategy": "balanced_rebuild",
                "asset_controls": {
                    "include_tool_candidate_spans": False,
                    "include_episode_step_spans": True,
                },
                "asset_rebuild_plan": {
                    "rebuild_required": True,
                    "strategy": "balanced_rebuild",
                },
                "overrides": {"tolbert_branch_results": 2},
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        retrieval_proposals_path=retrieval_path,
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_cache_paths=(str(cache_path),),
    )

    manifest_path = materialize_retained_retrieval_asset_bundle(
        repo_root=repo_root,
        config=config,
        cycle_id="cycle:retrieval:test",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == bundle_manifest_path
    assert manifest["artifact_kind"] == "tolbert_retrieval_asset_bundle"
    assert manifest["cycle_id"] == "cycle:retrieval:test"
    assert Path(manifest["materialized_paths"]["nodes_path"]).exists()
    assert Path(manifest["materialized_paths"]["source_spans_path"]).exists()
    assert manifest["runtime_paths"]["checkpoint_path"] == str(checkpoint_path)
    assert manifest["runtime_paths"]["cache_paths"] == [str(cache_path)]


def test_retained_tolbert_runtime_paths_prefer_bundle_manifest(tmp_path: Path) -> None:
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / "config.json"
    checkpoint_path = runtime_dir / "checkpoint.pt"
    nodes_path = runtime_dir / "nodes.jsonl"
    label_map_path = runtime_dir / "label_map.json"
    source_spans_path = runtime_dir / "source_spans.jsonl"
    cache_path = runtime_dir / "cache.pt"
    for path in (config_path, nodes_path, label_map_path, source_spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    bundle_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "nodes_path": str(nodes_path),
                    "label_map_path": str(label_map_path),
                    "source_spans_paths": [str(source_spans_path)],
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_config_path="missing/config.json",
        tolbert_checkpoint_path="missing/checkpoint.pt",
        tolbert_nodes_path="missing/nodes.jsonl",
        tolbert_label_map_path="missing/label_map.json",
        tolbert_source_spans_paths=("missing/source_spans.jsonl",),
        tolbert_cache_paths=("missing/cache.pt",),
    )

    runtime_paths = retained_tolbert_runtime_paths(config, repo_root=tmp_path)

    assert runtime_paths["tolbert_config_path"] == str(config_path)
    assert runtime_paths["tolbert_checkpoint_path"] == str(checkpoint_path)
    assert runtime_paths["tolbert_nodes_path"] == str(nodes_path)
    assert runtime_paths["tolbert_label_map_path"] == str(label_map_path)
    assert runtime_paths["tolbert_source_spans_paths"] == (str(source_spans_path),)
    assert runtime_paths["tolbert_cache_paths"] == (str(cache_path),)


def test_retained_tolbert_runtime_paths_materializes_delta_checkpoint(monkeypatch, tmp_path: Path) -> None:
    tolbert_model_artifact = tmp_path / "tolbert_model" / "tolbert_model_artifact.json"
    tolbert_model_artifact.parent.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = tmp_path / "parent.pt"
    delta_checkpoint = tmp_path / "delta.pt"
    parent_checkpoint.write_bytes(b"parent")
    delta_checkpoint.write_bytes(b"delta")
    materialized_checkpoint = tolbert_model_artifact.parent / ".materialized_checkpoints" / "candidate.pt"
    tolbert_model_artifact.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "checkpoint_path": "",
                    "parent_checkpoint_path": str(parent_checkpoint),
                    "checkpoint_delta_path": str(delta_checkpoint),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent_kernel.extensions.tolbert_assets.resolve_tolbert_runtime_checkpoint_path",
        lambda runtime_paths, *, artifact_path=None: str(materialized_checkpoint),
    )
    config = KernelConfig(
        tolbert_model_artifact_path=tolbert_model_artifact,
        use_tolbert_model_artifacts=True,
    )

    runtime_paths = retained_tolbert_runtime_paths(config, repo_root=tmp_path)

    assert runtime_paths["tolbert_checkpoint_path"] == str(materialized_checkpoint)


def test_build_tolbert_supervised_dataset_manifest_captures_required_sources(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    verifier_path = tmp_path / "verifiers" / "verifier_contracts.json"
    output_dir = tmp_path / "dataset"
    episodes.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    (episodes / "hello_task.json").write_text(
        json.dumps(
            {
                "task_id": "hello_task",
                "success": False,
                "summary": {
                    "failure_types": ["missing_expected_file"],
                    "transition_failures": ["no_state_progress"],
                },
                "task_contract": {
                    "prompt": "Create hello.txt containing hello agent kernel.",
                    "workspace_subdir": "hello_task",
                    "setup_commands": [],
                    "success_command": "test -f hello.txt",
                    "suggested_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "expected_files": ["hello.txt"],
                    "expected_output_substrings": [],
                    "forbidden_files": [],
                    "forbidden_output_substrings": [],
                    "expected_file_contents": {"hello.txt": "hello agent kernel\\n"},
                    "max_steps": 5,
                    "metadata": {"benchmark_family": "micro", "capability": "file_write", "difficulty": "seed"},
                },
                "world_model_summary": {
                    "completion_ratio": 0.0,
                    "missing_expected_artifacts": ["hello.txt"],
                    "present_forbidden_artifacts": [],
                    "changed_preserved_artifacts": [],
                },
                "steps": [
                    {
                        "index": 1,
                        "action": "code_execute",
                        "content": "printf 'hello agent kernel\\n' > hello.txt",
                        "verification": {
                            "passed": False,
                            "reasons": ["missing expected file: hello.txt"],
                        },
                        "state_progress_delta": 0.4,
                        "state_regression_count": 0,
                        "state_transition": {
                            "progress_delta": 0.4,
                            "regressions": [],
                            "cleared_forbidden_artifacts": [],
                            "newly_materialized_expected_artifacts": ["hello.txt"],
                            "no_progress": False,
                        },
                        "failure_signals": [],
                        "decision_source": "llm",
                        "tolbert_route_mode": "shadow",
                        "path_confidence": 0.75,
                        "trust_retrieval": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verifier_path.write_text(
        json.dumps(
            {
                "artifact_kind": "verifier_candidate_set",
                "proposals": [
                    {
                        "proposal_id": "verifier:hello:strict",
                        "source_task_id": "hello_task",
                        "contract": {
                            "prompt": "Create hello.txt containing hello agent kernel.",
                            "success_command": "test -f hello.txt",
                            "expected_files": ["hello.txt"],
                            "forbidden_files": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        trajectories_root=episodes,
        verifier_contracts_path=verifier_path,
    )

    manifest = build_tolbert_supervised_dataset_manifest(
        config=config,
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert manifest["trajectory_examples"] == 1
    assert manifest["transition_failure_examples"] == 1
    assert manifest["verifier_label_examples"] == 1
    assert manifest["discovered_task_examples"] == 1
    assert manifest["policy_examples"] == 1
    assert manifest["transition_examples"] == 1
    assert manifest["value_examples"] == 1
    assert manifest["stop_examples"] == 1
    assert manifest["action_generation_examples"] == 1
    assert manifest["action_generation_positive_examples"] == 0
    assert manifest["action_generation_summary"]["template_preferences"]["micro"][0]["template_kind"] == "expected_file_content"
    assert manifest["target_heads"] == {
        "policy": True,
        "transition": True,
        "value": True,
        "stop": True,
    }
    assert Path(manifest["supervised_examples_path"]).exists()
    assert Path(manifest["policy_examples_path"]).exists()
    assert Path(manifest["transition_examples_path"]).exists()
    assert Path(manifest["value_examples_path"]).exists()
    assert Path(manifest["stop_examples_path"]).exists()
