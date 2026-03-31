from pathlib import Path

from agent_kernel.roadmap_parallel import build_asi_parallel_manifest, write_asi_parallel_manifest


def test_build_asi_parallel_manifest_from_repo_asi_document(tmp_path):
    manifest = build_asi_parallel_manifest(Path("asi.md"), worker_count=5)

    assert manifest["task_manifest_kind"] == "asi_parallel_development_manifest"
    assert manifest["worker_count"] == 5
    tasks = manifest["tasks"]
    assert len(tasks) == 6
    worker_ids = [task["task_id"] for task in tasks[:-1]]
    assert worker_ids == [
        "asi_parallel_lane__open_world_task_acquisition",
        "asi_parallel_lane__fixed_model_independence",
        "asi_parallel_lane__latent_world_model",
        "asi_parallel_lane__evidence_hygiene",
        "asi_parallel_lane__unattended_operations",
    ]
    integrator = tasks[-1]
    assert integrator["task_id"] == "asi_parallel_roadmap_integrator"
    assert len(integrator["metadata"]["semantic_verifier"]["required_merged_branches"]) == 5
    evidence_lane = next(task for task in tasks if task["task_id"] == "asi_parallel_lane__evidence_hygiene")
    unattended_lane = next(task for task in tasks if task["task_id"] == "asi_parallel_lane__unattended_operations")
    assert "agent_kernel/job_queue.py" in evidence_lane["metadata"]["workflow_guard"]["claimed_paths"]
    assert "scripts/run_unattended_campaign.py" in unattended_lane["metadata"]["workflow_guard"]["claimed_paths"]


def test_write_asi_parallel_manifest_emits_json_file(tmp_path):
    output_path = tmp_path / "asi_parallel_manifest.json"

    manifest = write_asi_parallel_manifest(Path("asi.md"), output_path=output_path, worker_count=5)

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"task_manifest_kind": "asi_parallel_development_manifest"' in text
    assert manifest["worker_count"] == 5
