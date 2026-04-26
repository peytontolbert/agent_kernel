from __future__ import annotations

from copy import deepcopy
import shutil
from pathlib import Path
from typing import Any, Callable

from ..config import KernelConfig
from ..schemas import EpisodeRecord, TaskSpec
from .vllm_runtime import ensure_vllm_runtime


def build_default_policy(
    *,
    config: KernelConfig,
    repo_root: Path,
    skill_library_cls,
    llm_decision_policy_cls,
    context_provider_factory: Callable[..., object | None],
    ollama_client_cls,
    vllm_client_cls,
    model_stack_client_cls,
    mock_client_factory: Callable[[], object],
    hybrid_client_factory: Callable[..., object],
):
    provider = config.normalized_provider()
    context_provider = context_provider_factory(config=config, repo_root=repo_root)
    skill_library = (
        skill_library_cls.from_path(
            repo_root / config.skills_path,
            min_quality=config.min_skill_quality,
        )
        if config.use_skills
        else skill_library_cls([])
    )

    if provider == "ollama":
        client = ollama_client_cls(
            host=config.ollama_host,
            model_name=config.model_name,
            timeout_seconds=config.llm_timeout_seconds,
            retry_attempts=config.llm_retry_attempts,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
        )
    elif provider == "vllm":
        status = ensure_vllm_runtime(config)
        if not status.ready:
            raise RuntimeError(status.detail)
        client = vllm_client_cls(
            host=config.vllm_host,
            model_name=config.model_name,
            timeout_seconds=config.llm_timeout_seconds,
            retry_attempts=config.llm_retry_attempts,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
            api_key=config.vllm_api_key,
        )
    elif provider == "model_stack":
        model_stack_repo_path = Path(config.model_stack_repo_path)
        if not model_stack_repo_path.is_absolute():
            model_stack_repo_path = repo_root / model_stack_repo_path
        model_stack_model_dir = Path(config.model_stack_model_dir) if config.model_stack_model_dir else Path()
        if config.model_stack_model_dir and not model_stack_model_dir.is_absolute():
            model_stack_model_dir = repo_root / model_stack_model_dir
        model_stack_tokenizer_path = (
            Path(config.model_stack_tokenizer_path) if config.model_stack_tokenizer_path else Path()
        )
        if config.model_stack_tokenizer_path and not model_stack_tokenizer_path.is_absolute():
            model_stack_tokenizer_path = repo_root / model_stack_tokenizer_path
        client = model_stack_client_cls(
            host=config.model_stack_host,
            model_name=config.model_name,
            timeout_seconds=config.llm_timeout_seconds,
            retry_attempts=config.llm_retry_attempts,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
            model_dir=str(model_stack_model_dir) if config.model_stack_model_dir else "",
            tokenizer_path=str(model_stack_tokenizer_path) if config.model_stack_tokenizer_path else "",
            repo_path=str(model_stack_repo_path),
            api_key=config.model_stack_api_key,
        )
    elif provider == "mock":
        client = mock_client_factory()
    elif provider == "hybrid":
        client = hybrid_client_factory(config=config, repo_root=repo_root)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

    return llm_decision_policy_cls(
        client,
        context_provider=context_provider,
        skill_library=skill_library,
        config=config,
    )


def prepare_task_for_run(
    task: TaskSpec,
    *,
    runtime_overrides: dict[str, object] | None,
    job_id: str,
    uses_shared_repo_fn: Callable[..., bool],
    prepare_runtime_task_fn: Callable[..., TaskSpec],
) -> TaskSpec:
    if uses_shared_repo_fn(task, runtime_overrides=runtime_overrides):
        return prepare_runtime_task_fn(task, runtime_overrides=runtime_overrides, job_id=job_id)
    return deepcopy(task)


def _safe_setup_file_path(raw_path: object) -> str:
    relative_path = str(raw_path).strip().strip("/")
    if not relative_path or relative_path == "." or relative_path.startswith("../") or "/../" in relative_path:
        raise ValueError(f"unsafe setup file path: {raw_path!r}")
    return relative_path


def materialize_setup_file_contents(task: TaskSpec, workspace: Path) -> None:
    setup_files = task.metadata.get("setup_file_contents", {})
    if not isinstance(setup_files, dict):
        return
    for raw_path, raw_content in setup_files.items():
        relative_path = _safe_setup_file_path(raw_path)
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(raw_content), encoding="utf-8")


def materialize_workspace_for_new_run(
    task: TaskSpec,
    *,
    config: KernelConfig,
    runtime_overrides: dict[str, object] | None,
    job_id: str,
    resume: bool,
    checkpoint_path: Path | None,
    clean_workspace: bool,
    uses_shared_repo_fn: Callable[..., bool],
    materialize_shared_repo_workspace_fn: Callable[..., Path],
    ensure_parallel_worker_branches_fn: Callable[..., None],
) -> tuple[Path, bool]:
    workspace = config.workspace_root / task.workspace_subdir
    if uses_shared_repo_fn(task, runtime_overrides=runtime_overrides):
        workspace = materialize_shared_repo_workspace_fn(
            task,
            config=config,
            runtime_overrides=runtime_overrides,
            job_id=job_id,
            resume=resume and checkpoint_path is not None and checkpoint_path.exists(),
        )
        ensure_parallel_worker_branches_fn(
            task,
            workspace=workspace,
            runtime_overrides=runtime_overrides,
            job_id=job_id,
        )
        clean_workspace = False
    if clean_workspace and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    materialize_setup_file_contents(task, workspace)
    return workspace, clean_workspace


def persist_episode_outputs(
    *,
    task: TaskSpec,
    episode: EpisodeRecord,
    config: KernelConfig,
    memory,
    should_persist_learning_candidates_fn: Callable[..., bool],
    persist_episode_learning_candidates_fn: Callable[..., None],
    episode_storage_metadata_fn: Callable[..., dict[str, object]],
) -> Path | None:
    episode_path = None
    if config.persist_episode_memory:
        episode_path = memory.save(episode)
    if should_persist_learning_candidates_fn(task, config=config):
        persist_episode_learning_candidates_fn(
            episode,
            config=config,
            episode_storage=(
                episode_storage_metadata_fn(config.trajectories_root, episode_path)
                if episode_path is not None
                else None
            ),
        )
    return episode_path


def maybe_publish_shared_repo_branch(
    *,
    episode: EpisodeRecord,
    task: TaskSpec,
    config: KernelConfig,
    runtime_overrides: dict[str, object] | None,
    job_id: str,
    uses_shared_repo_fn: Callable[..., bool],
    publish_shared_repo_branch_fn: Callable[..., None],
) -> None:
    if not episode.success:
        return
    if not uses_shared_repo_fn(task, runtime_overrides=runtime_overrides):
        return
    publish_shared_repo_branch_fn(
        task,
        config=config,
        runtime_overrides=runtime_overrides,
        job_id=job_id,
    )


__all__ = [
    "build_default_policy",
    "materialize_setup_file_contents",
    "materialize_workspace_for_new_run",
    "maybe_publish_shared_repo_branch",
    "persist_episode_outputs",
    "prepare_task_for_run",
]
