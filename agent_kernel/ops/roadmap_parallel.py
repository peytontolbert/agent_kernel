from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re


@dataclass(frozen=True, slots=True)
class ParallelLaneSpec:
    lane_id: str
    title: str
    objective: str
    keywords: tuple[str, ...]


_LANE_SPECS: tuple[ParallelLaneSpec, ...] = (
    ParallelLaneSpec(
        lane_id="open_world_task_acquisition",
        title="Open-World Task Acquisition",
        objective="Broaden discovered-task growth beyond replay-derived task ecology.",
        keywords=("open-world", "environment", "benchmark", "verifier", "discovered-task", "task acquisition"),
    ),
    ParallelLaneSpec(
        lane_id="fixed_model_independence",
        title="Fixed-Model Independence",
        objective="Reduce runtime dependence on a fixed external base model outside the Tolbert takeover lane.",
        keywords=("fixed-model", "tolbert", "takeover", "runtime", "external base model", "model dependence"),
    ),
    ParallelLaneSpec(
        lane_id="latent_world_model",
        title="Latent World Model",
        objective="Move from symbolic continuity scaffolding toward a learned open-world latent controller.",
        keywords=("world model", "latent", "universe", "symbolic", "predictive model", "controller"),
    ),
    ParallelLaneSpec(
        lane_id="evidence_hygiene",
        title="Evidence Hygiene",
        objective="Deepen delegated-job diagnostics and durable evidence capture for unattended improvement.",
        keywords=("postmortem", "diagnosis", "checkpoint", "evidence", "stdout", "stderr", "delegated-job"),
    ),
    ParallelLaneSpec(
        lane_id="unattended_operations",
        title="Unattended Operations",
        objective="Close unattended restart, handoff, and operational self-recovery gaps.",
        keywords=("unattended", "restart", "handoff", "operational closure", "lock-safe", "self-recovery"),
    ),
)


def build_asi_parallel_manifest(
    asi_path: Path,
    *,
    worker_count: int = 5,
) -> dict[str, object]:
    bullets = _known_gap_bullets(asi_path)
    lane_specs = list(_LANE_SPECS[: max(1, min(len(_LANE_SPECS), int(worker_count)))])
    lane_assignments: list[dict[str, object]] = []
    unassigned = list(bullets)
    for lane in lane_specs:
        matched = [bullet for bullet in unassigned if _lane_match_score(lane, bullet) > 0]
        matched.sort(key=lambda bullet: (_lane_match_score(lane, bullet), bullet["summary"]), reverse=True)
        selected = matched[:2]
        if not selected and unassigned:
            selected = [unassigned[0]]
        for bullet in selected:
            if bullet in unassigned:
                unassigned.remove(bullet)
        lane_assignments.append(_lane_task_payload(lane, selected))
    for bullet in unassigned:
        best_index = max(
            range(len(lane_assignments)),
            key=lambda index: (
                _lane_match_score(lane_specs[index], bullet),
                -len(lane_assignments[index]["bullet_summaries"]),
            ),
        )
        lane_assignments[best_index]["bullet_summaries"].append(str(bullet["summary"]))
        lane_assignments[best_index]["source_refs"].extend(
            ref for ref in bullet["source_refs"] if ref not in lane_assignments[best_index]["source_refs"]
        )
    worker_tasks = [_worker_task_from_lane(payload) for payload in lane_assignments]
    integrator_task = _integrator_task(worker_tasks)
    return {
        "spec_version": "asi_v1",
        "task_manifest_kind": "asi_parallel_development_manifest",
        "source_document": str(asi_path),
        "worker_count": len(worker_tasks),
        "tasks": [*worker_tasks, integrator_task],
    }


def write_asi_parallel_manifest(
    asi_path: Path,
    *,
    output_path: Path,
    worker_count: int = 5,
) -> dict[str, object]:
    manifest = build_asi_parallel_manifest(asi_path, worker_count=worker_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _known_gap_bullets(asi_path: Path) -> list[dict[str, object]]:
    text = asi_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_section = False
    bullets: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "## Known Evidence And Capability Gaps":
            in_section = True
            current = []
            continue
        if in_section and stripped.startswith("## "):
            break
        if not in_section:
            continue
        if line.startswith("- "):
            if current:
                bullets.append(current)
            current = [stripped[2:].strip()]
            continue
        if current and (line.startswith("  - ") or line.startswith("    ") or stripped):
            current.append(stripped)
    if current:
        bullets.append(current)
    payloads: list[dict[str, object]] = []
    for bullet_lines in bullets:
        summary = bullet_lines[0]
        detail = " ".join(part for part in bullet_lines[1:] if part)
        source_refs = _markdown_file_refs(" ".join(bullet_lines))
        payloads.append(
            {
                "summary": summary,
                "detail": detail,
                "source_refs": source_refs,
            }
        )
    return payloads


def _markdown_file_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"\[[^\]]+\]\((/data/agentkernel/[^)]+)\)", text):
        ref = str(match.group(1)).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _lane_match_score(lane: ParallelLaneSpec, bullet: dict[str, object]) -> int:
    haystack = " ".join(
        [
            str(bullet.get("summary", "")),
            str(bullet.get("detail", "")),
            " ".join(str(ref) for ref in bullet.get("source_refs", [])),
        ]
    ).lower()
    return sum(1 for keyword in lane.keywords if keyword.lower() in haystack)


def _lane_task_payload(lane: ParallelLaneSpec, bullets: list[dict[str, object]]) -> dict[str, object]:
    summaries = [str(bullet.get("summary", "")).strip() for bullet in bullets if str(bullet.get("summary", "")).strip()]
    source_refs: list[str] = []
    for bullet in bullets:
        for ref in bullet.get("source_refs", []):
            normalized = str(ref).strip()
            if normalized and normalized not in source_refs:
                source_refs.append(normalized)
    claimed_paths = [
        str(Path(ref).relative_to(Path("/data/agentkernel")))
        for ref in source_refs
        if ref.startswith("/data/agentkernel/")
    ]
    if not claimed_paths:
        claimed_paths = ["asi.md"]
    return {
        "lane": lane,
        "bullet_summaries": summaries,
        "source_refs": source_refs,
        "claimed_paths": claimed_paths,
    }


def _worker_task_from_lane(payload: dict[str, object]) -> dict[str, object]:
    lane = payload["lane"]
    assert isinstance(lane, ParallelLaneSpec)
    worker_branch = f"worker/{lane.lane_id.replace('_', '-')}"
    claimed_paths = list(payload.get("claimed_paths", []))
    bullet_summaries = [str(value).strip() for value in payload.get("bullet_summaries", []) if str(value).strip()]
    report_path = f"reports/{lane.lane_id}_lane_report.txt"
    return {
        "task_id": f"asi_parallel_lane__{lane.lane_id}",
        "prompt": (
            f"Advance the ASI roadmap lane '{lane.title}'. Objective: {lane.objective} "
            f"Relevant gap bullets: {'; '.join(bullet_summaries) if bullet_summaries else 'see asi.md known gaps section'}."
        ),
        "workspace_subdir": f"asi_parallel_lane__{lane.lane_id}",
        "expected_files": [*claimed_paths, report_path],
        "metadata": {
            "benchmark_family": "repository",
            "capability": "repo_environment",
            "shared_repo_order": 0,
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "asi-roadmap",
                "target_branch": "main",
                "worker_branch": worker_branch,
                "claimed_paths": claimed_paths,
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": worker_branch,
                "diff_base_ref": "origin/main",
                "expected_changed_paths": claimed_paths,
                "clean_worktree": True,
                "report_rules": [
                    {
                        "path": report_path,
                        "must_mention": ["asi", lane.lane_id, "lane"],
                        "covers": claimed_paths,
                    }
                ],
            },
            "asi_parallel_lane": {
                "lane_id": lane.lane_id,
                "title": lane.title,
                "objective": lane.objective,
                "gap_bullets": bullet_summaries,
                "source_refs": list(payload.get("source_refs", [])),
            },
        },
    }


def _integrator_task(worker_tasks: list[dict[str, object]]) -> dict[str, object]:
    required_worker_branches = [
        str(task.get("metadata", {}).get("workflow_guard", {}).get("worker_branch", "")).strip()
        for task in worker_tasks
        if str(task.get("metadata", {}).get("workflow_guard", {}).get("worker_branch", "")).strip()
    ]
    expected_changed_paths: list[str] = []
    for task in worker_tasks:
        for path in task.get("metadata", {}).get("workflow_guard", {}).get("claimed_paths", []):
            normalized = str(path).strip()
            if normalized and normalized not in expected_changed_paths:
                expected_changed_paths.append(normalized)
    return {
        "task_id": "asi_parallel_roadmap_integrator",
        "prompt": (
            "Integrate the ASI roadmap worker branches into one reviewed plan-of-record change set, "
            "preserving worker-owned boundaries while producing a coherent roadmap increment."
        ),
        "workspace_subdir": "asi_parallel_roadmap_integrator",
        "metadata": {
            "benchmark_family": "repository",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "parallel_workers": [
                {
                    "worker_branch": str(task.get("metadata", {}).get("workflow_guard", {}).get("worker_branch", "")).strip(),
                    "prompt": str(task.get("prompt", "")).strip(),
                    "expected_changed_paths": list(
                        task.get("metadata", {}).get("workflow_guard", {}).get("claimed_paths", [])
                    ),
                    "claimed_paths": list(task.get("metadata", {}).get("workflow_guard", {}).get("claimed_paths", [])),
                }
                for task in worker_tasks
            ],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "asi-roadmap",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": required_worker_branches,
                "expected_changed_paths": expected_changed_paths,
                "clean_worktree": True,
                "report_rules": [
                    {
                        "path": "reports/asi_parallel_integration_report.txt",
                        "must_mention": ["asi", "integration", "roadmap"],
                        "covers": expected_changed_paths,
                    }
                ],
            },
        },
    }
