from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from agent_kernel.config import KernelConfig
from agent_kernel.tasking.curriculum import CurriculumEngine
from agent_kernel.loop import AgentKernel
from agent_kernel.policy import Policy
from agent_kernel.schemas import ActionDecision, EpisodeRecord, TaskSpec
from agent_kernel.tasking.task_bank import TaskBank


class ForcedFailurePolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="force a deterministic failure for curriculum generation",
            action="code_execute",
            content="false",
            done=False,
        )


def _task_bank(config: KernelConfig) -> TaskBank:
    try:
        return TaskBank(config=config)
    except TypeError:
        return TaskBank()


def run_selfplay_chain(
    *,
    seed_task_id: str = "hello_task",
    seed_mode: str = "success",
    max_followup_hops: int = 3,
    config: KernelConfig | None = None,
) -> tuple[EpisodeRecord, list[tuple[TaskSpec, EpisodeRecord]]]:
    if max_followup_hops < 0:
        raise ValueError("max_followup_hops must be non-negative")
    runtime_config = config or KernelConfig()
    bank = _task_bank(runtime_config)
    normal_kernel = AgentKernel(config=runtime_config)
    engine = CurriculumEngine(memory_root=runtime_config.trajectories_root, config=runtime_config)
    seed_task = bank.get(seed_task_id)
    if seed_mode == "failure":
        seed = AgentKernel(config=runtime_config, policy=ForcedFailurePolicy()).run_task(seed_task)
    else:
        seed = normal_kernel.run_task(seed_task)
    followups: list[tuple[TaskSpec, EpisodeRecord]] = []
    current = seed
    seen_task_ids = {seed.task_id}
    for _ in range(max_followup_hops):
        followup = engine.generate_followup_task(current)
        if followup.task_id in seen_task_ids:
            break
        seen_task_ids.add(followup.task_id)
        result = normal_kernel.run_task(followup)
        followups.append((followup, result))
        current = result
    return seed, followups


def format_selfplay_progression(seed: EpisodeRecord, followups: list[tuple[TaskSpec, EpisodeRecord]]) -> str:
    parts = [f"seed={seed.task_id}:{seed.success}"]
    for index, (task, result) in enumerate(followups, start=1):
        parts.append(
            f"hop{index}={task.task_id}:{result.success}:{task.metadata.get('curriculum_kind', 'unknown')}"
        )
    parts.append(f"hops={len(followups)}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-task-id", default="hello_task")
    parser.add_argument("--seed-mode", choices=("success", "failure"), default="success")
    parser.add_argument("--max-followup-hops", type=int, default=3)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tolbert-python", default=None)
    parser.add_argument("--tolbert-config", default=None)
    parser.add_argument("--tolbert-checkpoint", default=None)
    parser.add_argument("--tolbert-nodes", default=None)
    parser.add_argument("--tolbert-cache", action="append", default=None)
    parser.add_argument("--tolbert-label-map", default=None)
    parser.add_argument("--tolbert-device", default=None)
    args = parser.parse_args()
    if args.max_followup_hops < 0:
        parser.error("--max-followup-hops must be non-negative")

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.tolbert_python:
        config.tolbert_python_bin = args.tolbert_python
    if args.tolbert_config:
        config.tolbert_config_path = args.tolbert_config
    if args.tolbert_checkpoint:
        config.tolbert_checkpoint_path = args.tolbert_checkpoint
    if args.tolbert_nodes:
        config.tolbert_nodes_path = args.tolbert_nodes
    if args.tolbert_cache:
        config.tolbert_cache_paths = tuple(args.tolbert_cache)
    if args.tolbert_label_map:
        config.tolbert_label_map_path = args.tolbert_label_map
    if args.tolbert_device:
        config.tolbert_device = args.tolbert_device
    seed, followups = run_selfplay_chain(
        seed_task_id=args.seed_task_id,
        seed_mode=args.seed_mode,
        max_followup_hops=args.max_followup_hops,
        config=config,
    )
    print(format_selfplay_progression(seed, followups))


if __name__ == "__main__":
    main()
