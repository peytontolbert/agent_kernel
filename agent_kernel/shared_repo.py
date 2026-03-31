from __future__ import annotations

from copy import deepcopy
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .sandbox import Sandbox
from .schemas import TaskSpec


def shared_repo_claim(
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
) -> dict[str, object]:
    metadata = getattr(task, "metadata", {})
    workflow_guard = metadata.get("workflow_guard", {}) if isinstance(metadata, dict) else {}
    guard = dict(workflow_guard) if isinstance(workflow_guard, dict) else {}
    overrides = dict(runtime_overrides or {})
    shared_repo_id = str(overrides.get("shared_repo_id") or guard.get("shared_repo_id") or "").strip()
    target_branch = str(overrides.get("target_branch") or guard.get("target_branch") or "").strip()
    worker_branch = str(overrides.get("worker_branch") or guard.get("worker_branch") or "").strip()
    claimed_paths = _normalize_paths(
        overrides.get("claimed_paths")
        or guard.get("claimed_paths")
        or _derived_claimed_paths(task, workflow_guard=guard)
    )
    clone_label = worker_branch or target_branch or job_id or getattr(task, "task_id", "task")
    return {
        "shared_repo_id": shared_repo_id,
        "target_branch": target_branch,
        "worker_branch": worker_branch,
        "claimed_paths": claimed_paths,
        "claim_source": (
            "runtime_override"
            if overrides.get("claimed_paths")
            else "workflow_guard"
            if guard.get("claimed_paths")
            else "derived"
            if claimed_paths
            else ""
        ),
        "clone_label": clone_label,
    }


def uses_shared_repo(
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None = None,
) -> bool:
    claim = shared_repo_claim(task, runtime_overrides=runtime_overrides)
    return bool(claim["shared_repo_id"])


def prepare_runtime_task(
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
) -> TaskSpec:
    prepared = deepcopy(task)
    claim = shared_repo_claim(prepared, runtime_overrides=runtime_overrides, job_id=job_id)
    overrides = dict(runtime_overrides or {})
    verifier_metadata = prepared.metadata.get("semantic_verifier", {})
    semantic_verifier = dict(verifier_metadata) if isinstance(verifier_metadata, dict) else {}
    required_worker_branches = _normalize_paths(overrides.get("required_worker_branches"))
    if required_worker_branches:
        semantic_verifier["required_merged_branches"] = list(required_worker_branches)
        prepared.metadata["semantic_verifier"] = semantic_verifier
    if not claim["shared_repo_id"]:
        return prepared
    workflow_guard = dict(prepared.metadata.get("workflow_guard", {}))
    workflow_guard["shared_repo_id"] = str(claim["shared_repo_id"])
    workflow_guard["target_branch"] = str(claim["target_branch"])
    if str(claim["worker_branch"]).strip():
        workflow_guard["worker_branch"] = str(claim["worker_branch"])
    if claim["claimed_paths"]:
        workflow_guard["claimed_paths"] = list(claim["claimed_paths"])
        workflow_guard["claimed_paths_source"] = str(claim.get("claim_source", "")).strip()
    prepared.metadata["workflow_guard"] = workflow_guard
    prepared.workspace_subdir = shared_repo_clone_subdir(
        prepared,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
    )
    # Shared-repo bootstrap is handled externally against the shared origin, not per-clone setup.
    prepared.setup_commands = [
        command
        for command in prepared.setup_commands
        if command.strip() and not bool(prepared.metadata.get("shared_repo_bootstrap_commands"))
    ]
    if prepared.metadata.get("shared_repo_bootstrap_commands"):
        prepared.setup_commands = []
    return prepared


def shared_repo_clone_subdir(
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
) -> str:
    claim = shared_repo_claim(task, runtime_overrides=runtime_overrides, job_id=job_id)
    if not claim["shared_repo_id"]:
        return task.workspace_subdir
    return str(
        Path("_shared_repo_runtime")
        / _safe_name(str(claim["shared_repo_id"]))
        / "clones"
        / _safe_name(str(claim["clone_label"]))
    )


def shared_repo_workspace_path(
    workspace_root: Path,
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
) -> Path:
    return (workspace_root / shared_repo_clone_subdir(task, runtime_overrides=runtime_overrides, job_id=job_id)).resolve()


def materialize_shared_repo_workspace(
    task: TaskSpec,
    *,
    config: KernelConfig,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
    resume: bool = False,
) -> Path:
    claim = shared_repo_claim(task, runtime_overrides=runtime_overrides, job_id=job_id)
    workspace = shared_repo_workspace_path(
        config.workspace_root,
        task,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
    )
    if not claim["shared_repo_id"]:
        return workspace
    if resume and workspace.exists():
        return workspace

    shared_root = _shared_repo_root(config.workspace_root, str(claim["shared_repo_id"]))
    origin_path = shared_root / "origin.git"
    _ensure_shared_repo_origin(task, config=config, origin_path=origin_path)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", str(origin_path), str(workspace)], cwd=shared_root)
    _run_git(["config", "user.email", "agent@example.com"], cwd=workspace)
    _run_git(["config", "user.name", "Agent Kernel"], cwd=workspace)
    _checkout_clone_branch(workspace, claim)
    _materialize_local_tracking_branches(workspace)
    return workspace


def publish_shared_repo_branch(
    task: TaskSpec,
    *,
    config: KernelConfig,
    runtime_overrides: dict[str, object] | None = None,
    job_id: str = "",
) -> None:
    claim = shared_repo_claim(task, runtime_overrides=runtime_overrides, job_id=job_id)
    if not claim["shared_repo_id"]:
        return
    workspace = shared_repo_workspace_path(
        config.workspace_root,
        task,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
    )
    branch = _git_output(workspace, "branch", "--show-current")
    if not branch:
        return
    _run_git(["push", "-u", "origin", f"{branch}:{branch}"], cwd=workspace)


def _ensure_shared_repo_origin(task: TaskSpec, *, config: KernelConfig, origin_path: Path) -> None:
    if origin_path.exists():
        return
    metadata = dict(task.metadata)
    bootstrap_commands = [
        str(command).strip()
        for command in metadata.get("shared_repo_bootstrap_commands", [])
        if str(command).strip()
    ]
    if not bootstrap_commands:
        bootstrap_commands = list(task.setup_commands)
    if not bootstrap_commands:
        raise ValueError(f"shared repo task {task.task_id} does not define bootstrap commands")
    bootstrap_managed_paths = [
        str(path).strip()
        for path in metadata.get("shared_repo_bootstrap_managed_paths", [])
        if str(path).strip()
    ]
    claim = shared_repo_claim(task)
    shared_root = origin_path.parent
    seed = shared_root / "seed"
    if seed.exists():
        shutil.rmtree(seed)
    seed.mkdir(parents=True, exist_ok=True)
    bootstrap_task = TaskSpec(
        task_id=f"{task.task_id}_bootstrap",
        prompt=f"Bootstrap shared repo for {task.task_id}",
        workspace_subdir=str(seed.relative_to(config.workspace_root))
        if config.workspace_root in seed.parents
        else seed.name,
        metadata={
            "benchmark_family": str(task.metadata.get("benchmark_family", "repo_sandbox")),
            "workflow_guard": {
                "requires_git": True,
                "touches_generated_paths": bool(
                    isinstance(task.metadata.get("workflow_guard"), dict)
                    and task.metadata.get("workflow_guard", {}).get("touches_generated_paths")
                ),
                "shared_repo_id": str(claim["shared_repo_id"]),
                "managed_paths": bootstrap_managed_paths,
            },
        },
        expected_files=list(bootstrap_managed_paths),
    )
    sandbox = Sandbox(config.command_timeout_seconds, config=config)
    for command in bootstrap_commands:
        result = sandbox.run(command, seed, task=bootstrap_task)
        if result.exit_code != 0:
            raise RuntimeError(
                f"shared repo bootstrap failed for {task.task_id}: {command}: {result.stderr or result.stdout}"
            )
    if not (seed / ".git").exists():
        raise RuntimeError(f"shared repo bootstrap did not initialize git for {task.task_id}")
    origin_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--bare", str(seed), str(origin_path)], cwd=shared_root)


def _checkout_clone_branch(workspace: Path, claim: dict[str, object]) -> None:
    target_branch = str(claim["target_branch"]).strip()
    worker_branch = str(claim["worker_branch"]).strip()
    base_branch = target_branch or "main"
    if target_branch:
        _checkout_or_create_branch(workspace, target_branch, start_ref=f"origin/{target_branch}")
    if worker_branch:
        start_ref = f"origin/{base_branch}"
        _checkout_or_create_branch(workspace, worker_branch, start_ref=start_ref)


def _checkout_or_create_branch(workspace: Path, branch: str, *, start_ref: str) -> None:
    if _git_returncode(workspace, "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}") == 0:
        _run_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=workspace)
        return
    if _git_returncode(workspace, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}") == 0:
        _run_git(["checkout", branch], cwd=workspace)
        return
    _run_git(["checkout", "-B", branch, start_ref], cwd=workspace)


def _materialize_local_tracking_branches(workspace: Path) -> None:
    output = _git_output(
        workspace,
        "for-each-ref",
        "--format=%(refname:strip=3)",
        "refs/remotes/origin",
    )
    for branch in [line.strip() for line in output.splitlines() if line.strip()]:
        if branch == "HEAD":
            continue
        if _git_returncode(workspace, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}") == 0:
            continue
        _run_git(["branch", branch, f"origin/{branch}"], cwd=workspace)


def _run_git(argv: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        ["git", *argv],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git command failed in {cwd}: {' '.join(argv)}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def _git_output(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _git_returncode(workspace: Path, *args: str) -> int:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    return completed.returncode


def _shared_repo_root(workspace_root: Path, shared_repo_id: str) -> Path:
    return workspace_root / "_shared_repo_runtime" / _safe_name(shared_repo_id)


def _derived_claimed_paths(task: TaskSpec, *, workflow_guard: dict[str, object]) -> tuple[str, ...]:
    metadata = dict(getattr(task, "metadata", {}))
    verifier = metadata.get("semantic_verifier", {})
    contract = dict(verifier) if isinstance(verifier, dict) else {}
    derived: list[str] = []
    if contract:
        derived.extend(_string_values(contract.get("expected_changed_paths", [])))
        derived.extend(_string_values(contract.get("generated_paths", [])))
        derived.extend(_string_values(contract.get("resolved_conflict_paths", [])))
        for rule in contract.get("report_rules", []):
            if not isinstance(rule, dict):
                continue
            path = str(rule.get("path", "")).strip()
            if path:
                derived.append(path)
    for step in metadata.get("synthetic_edit_plan", []):
        if not isinstance(step, dict):
            continue
        path = str(step.get("path", "")).strip()
        if path:
            derived.append(path)
    for candidate_set in metadata.get("synthetic_edit_candidates", []):
        if not isinstance(candidate_set, dict):
            continue
        path = str(candidate_set.get("path", "")).strip()
        if path:
            derived.append(path)
        selected = candidate_set.get("selected", {})
        if isinstance(selected, dict):
            selected_path = str(selected.get("path", "")).strip()
            if selected_path:
                derived.append(selected_path)
        for candidate in candidate_set.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_path = str(candidate.get("path", "")).strip()
            if candidate_path:
                derived.append(candidate_path)
    derived.extend(_string_values(getattr(task, "expected_file_contents", {})))
    if not derived:
        derived.extend(_string_values(getattr(task, "expected_files", [])))
        derived.extend(_string_values(getattr(task, "forbidden_files", [])))
        derived.extend(_string_values(workflow_guard.get("delete_paths", [])))
        derived.extend(_string_values(workflow_guard.get("managed_paths", [])))
    return _normalize_paths(derived)


def _normalize_paths(values: object) -> tuple[str, ...]:
    if isinstance(values, (list, tuple, set)):
        raw = [str(value) for value in values]
    elif isinstance(values, str):
        raw = [values]
    else:
        raw = []
    normalized = {
        value.strip().strip("/")
        for value in raw
        if value.strip()
    }
    return tuple(sorted(normalized))


def _string_values(values: object) -> list[str]:
    if isinstance(values, dict):
        values = values.keys()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
