from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
from agent_kernel.tasking.task_bank import TaskBank


PROFILE_NAME = "qwen_tolbert_reference_implementation"


def _build_task_bank(config: KernelConfig) -> TaskBank:
    try:
        return TaskBank(config=config)
    except TypeError:
        return TaskBank()


def _state_root_overrides(root: Path) -> dict[str, object]:
    root = root.resolve()
    trajectories_root = root / "trajectories"
    runtime_root = root / "var" / "runtime"
    return {
        "workspace_root": root / "workspace",
        "trajectories_root": trajectories_root / "episodes",
        "skills_path": trajectories_root / "skills" / "command_skills.json",
        "operator_classes_path": trajectories_root / "operators" / "operator_classes.json",
        "tool_candidates_path": trajectories_root / "tools" / "tool_candidates.json",
        "benchmark_candidates_path": trajectories_root / "benchmarks" / "benchmark_candidates.json",
        "retrieval_proposals_path": trajectories_root / "retrieval" / "retrieval_proposals.json",
        "retrieval_asset_bundle_path": trajectories_root / "retrieval" / "retrieval_asset_bundle.json",
        "tolbert_model_artifact_path": trajectories_root / "tolbert_model" / "tolbert_model_artifact.json",
        "tolbert_supervised_datasets_dir": trajectories_root / "tolbert_model" / "datasets",
        "qwen_adapter_artifact_path": trajectories_root / "qwen_adapter" / "qwen_adapter_artifact.json",
        "qwen_supervised_datasets_dir": trajectories_root / "qwen_adapter" / "datasets",
        "tolbert_liftoff_report_path": trajectories_root / "tolbert_model" / "liftoff_gate_report.json",
        "verifier_contracts_path": trajectories_root / "verifiers" / "verifier_contracts.json",
        "prompt_proposals_path": trajectories_root / "prompts" / "prompt_proposals.json",
        "universe_constitution_path": trajectories_root / "universe" / "universe_constitution.json",
        "operating_envelope_path": trajectories_root / "universe" / "operating_envelope.json",
        "universe_contract_path": trajectories_root / "universe" / "universe_contract.json",
        "world_model_proposals_path": trajectories_root / "world_model" / "world_model_proposals.json",
        "state_estimation_proposals_path": trajectories_root / "state_estimation" / "state_estimation_proposals.json",
        "trust_proposals_path": trajectories_root / "trust" / "trust_proposals.json",
        "recovery_proposals_path": trajectories_root / "recovery" / "recovery_proposals.json",
        "delegation_proposals_path": trajectories_root / "delegation" / "delegation_proposals.json",
        "operator_policy_proposals_path": trajectories_root / "operator_policy" / "operator_policy_proposals.json",
        "transition_model_proposals_path": trajectories_root / "transition_model" / "transition_model_proposals.json",
        "curriculum_proposals_path": trajectories_root / "curriculum" / "curriculum_proposals.json",
        "improvement_cycles_path": trajectories_root / "improvement" / "cycles.jsonl",
        "candidate_artifacts_root": trajectories_root / "improvement" / "candidates",
        "improvement_reports_dir": trajectories_root / "improvement" / "reports",
        "strategy_memory_nodes_path": trajectories_root / "improvement" / "strategy_memory" / "nodes.jsonl",
        "strategy_memory_snapshots_path": trajectories_root / "improvement" / "strategy_memory" / "snapshots.json",
        "semantic_hub_root": trajectories_root / "semantic_hub",
        "run_reports_dir": trajectories_root / "reports",
        "learning_artifacts_path": trajectories_root / "learning" / "run_learning_artifacts.json",
        "capability_modules_path": root / "config" / "capabilities.json",
        "run_checkpoints_dir": trajectories_root / "checkpoints",
        "delegated_job_queue_path": trajectories_root / "jobs" / "queue.json",
        "delegated_job_runtime_state_path": trajectories_root / "jobs" / "runtime_state.json",
        "runtime_database_path": runtime_root / "agentkernel.sqlite3",
        "unattended_workspace_snapshot_root": trajectories_root / "recovery" / "workspaces",
        "unattended_trust_ledger_path": trajectories_root / "reports" / "unattended_trust_ledger.json",
        "vllm_runtime_log_path": runtime_root / "vllm_runtime.log",
        "vllm_runtime_state_path": runtime_root / "vllm_runtime.json",
        "vllm_runtime_lock_path": runtime_root / "vllm_runtime.lock",
    }


def build_reference_config(
    *,
    state_root: Path | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    use_tolbert_context: bool | None = None,
    use_graph_memory: bool | None = None,
    use_world_model: bool | None = None,
    use_universe_model: bool | None = None,
    persist_episode_memory: bool | None = None,
    persist_learning_candidates: bool | None = None,
    max_steps: int | None = None,
    tolbert_python_bin: str | None = None,
    tolbert_config_path: str | None = None,
    tolbert_checkpoint_path: str | None = None,
    tolbert_nodes_path: str | None = None,
    tolbert_label_map_path: str | None = None,
    tolbert_cache_paths: tuple[str, ...] | None = None,
    tolbert_source_spans_paths: tuple[str, ...] | None = None,
    tolbert_device: str | None = None,
    external_task_manifests_paths: tuple[str, ...] | None = None,
) -> KernelConfig:
    overrides: dict[str, object] = {}
    if state_root is not None:
        overrides.update(_state_root_overrides(state_root))
    if provider is not None:
        overrides["provider"] = provider
    if model_name is not None:
        overrides["model_name"] = model_name
    if use_tolbert_context is not None:
        overrides["use_tolbert_context"] = use_tolbert_context
    if use_graph_memory is not None:
        overrides["use_graph_memory"] = use_graph_memory
    if use_world_model is not None:
        overrides["use_world_model"] = use_world_model
    if use_universe_model is not None:
        overrides["use_universe_model"] = use_universe_model
    if persist_episode_memory is not None:
        overrides["persist_episode_memory"] = persist_episode_memory
    if persist_learning_candidates is not None:
        overrides["persist_learning_candidates"] = persist_learning_candidates
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    if tolbert_python_bin is not None:
        overrides["tolbert_python_bin"] = tolbert_python_bin
    if tolbert_config_path is not None:
        overrides["tolbert_config_path"] = tolbert_config_path
    if tolbert_checkpoint_path is not None:
        overrides["tolbert_checkpoint_path"] = tolbert_checkpoint_path
    if tolbert_nodes_path is not None:
        overrides["tolbert_nodes_path"] = tolbert_nodes_path
    if tolbert_label_map_path is not None:
        overrides["tolbert_label_map_path"] = tolbert_label_map_path
    if tolbert_cache_paths is not None:
        overrides["tolbert_cache_paths"] = tolbert_cache_paths
    if tolbert_source_spans_paths is not None:
        overrides["tolbert_source_spans_paths"] = tolbert_source_spans_paths
    if tolbert_device is not None:
        overrides["tolbert_device"] = tolbert_device
    if external_task_manifests_paths is not None:
        overrides["external_task_manifests_paths"] = external_task_manifests_paths
    return KernelConfig.qwen_tolbert_reference_implementation(**overrides)


def _profile_summary(config: KernelConfig, *, state_root: Path | None = None) -> dict[str, object]:
    decoder_runtime = config.decoder_runtime_posture()
    return {
        "profile": PROFILE_NAME,
        "provider": config.provider,
        "model_name": config.model_name,
        "kernel_claimed_runtime_shape": config.claimed_runtime_shape(),
        "decoder_authority": decoder_runtime.get("authority", ""),
        "use_tolbert_context": config.use_tolbert_context,
        "use_graph_memory": config.use_graph_memory,
        "use_world_model": config.use_world_model,
        "use_universe_model": config.use_universe_model,
        "persist_episode_memory": config.persist_episode_memory,
        "persist_learning_candidates": config.persist_learning_candidates,
        "state_root": str(state_root.resolve()) if state_root is not None else "",
    }


def _safe_task_id(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="hello_task")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--use-tolbert-context", choices=("0", "1"), default=None)
    parser.add_argument("--use-graph-memory", choices=("0", "1"), default=None)
    parser.add_argument("--use-world-model", choices=("0", "1"), default=None)
    parser.add_argument("--use-universe-model", choices=("0", "1"), default=None)
    parser.add_argument("--persist-episode-memory", choices=("0", "1"), default=None)
    parser.add_argument("--persist-learning-candidates", choices=("0", "1"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tolbert-python", default=None)
    parser.add_argument("--tolbert-config", default=None)
    parser.add_argument("--tolbert-checkpoint", default=None)
    parser.add_argument("--tolbert-nodes", default=None)
    parser.add_argument("--tolbert-label-map", default=None)
    parser.add_argument("--tolbert-source-spans", action="append", default=None)
    parser.add_argument("--tolbert-cache", action="append", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--external-task-manifest", action="append", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-profile", action="store_true")
    args = parser.parse_args()

    state_root = Path(args.state_root) if args.state_root else None
    config = build_reference_config(
        state_root=state_root,
        provider=args.provider,
        model_name=args.model,
        use_tolbert_context=None if args.use_tolbert_context is None else args.use_tolbert_context == "1",
        use_graph_memory=None if args.use_graph_memory is None else args.use_graph_memory == "1",
        use_world_model=None if args.use_world_model is None else args.use_world_model == "1",
        use_universe_model=None if args.use_universe_model is None else args.use_universe_model == "1",
        persist_episode_memory=None
        if args.persist_episode_memory is None
        else args.persist_episode_memory == "1",
        persist_learning_candidates=None
        if args.persist_learning_candidates is None
        else args.persist_learning_candidates == "1",
        max_steps=args.max_steps,
        tolbert_python_bin=args.tolbert_python,
        tolbert_config_path=args.tolbert_config,
        tolbert_checkpoint_path=args.tolbert_checkpoint,
        tolbert_nodes_path=args.tolbert_nodes,
        tolbert_label_map_path=args.tolbert_label_map,
        tolbert_cache_paths=tuple(args.tolbert_cache) if args.tolbert_cache else None,
        tolbert_source_spans_paths=tuple(args.tolbert_source_spans) if args.tolbert_source_spans else None,
        tolbert_device=args.tolbert_device,
        external_task_manifests_paths=tuple(args.external_task_manifest) if args.external_task_manifest else None,
    )
    if args.print_profile:
        print(json.dumps(_profile_summary(config, state_root=state_root), sort_keys=True))

    bank = _build_task_bank(config)
    task = bank.get(args.task_id)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else config.run_checkpoints_dir / f"{_safe_task_id(task.task_id)}.json"
    )

    kernel = AgentKernel(config=config)
    try:
        episode = kernel.run_task(
            task,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )
    finally:
        kernel.close()

    print(
        f"profile={PROFILE_NAME} runtime_shape={config.claimed_runtime_shape()} "
        f"decoder_authority={config.decoder_runtime_posture().get('authority', '')} "
        f"task={episode.task_id} success={episode.success} steps={len(episode.steps)}"
    )


if __name__ == "__main__":
    main()
