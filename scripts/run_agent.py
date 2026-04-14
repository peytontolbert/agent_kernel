from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
from agent_kernel.ops.preflight import (
    capture_workspace_snapshot,
    classify_run_outcome,
    run_unattended_preflight,
    write_unattended_task_report,
)
from agent_kernel.tasking.task_bank import TaskBank
from agent_kernel.extensions.trust import write_unattended_trust_ledger
from agent_kernel.ops.workspace_recovery import (
    annotate_task_report_recovery,
    recovery_annotation,
    restore_workspace_tree,
    should_restore_on_outcome,
    should_restore_on_runner_exception,
    should_snapshot_workspace,
    snapshot_workspace_tree,
)


def _build_task_bank(config: KernelConfig) -> TaskBank:
    try:
        return TaskBank(config=config)
    except TypeError:
        return TaskBank()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="hello_task")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-tolbert-context", choices=("0", "1"), default=None)
    parser.add_argument("--use-skills", choices=("0", "1"), default=None)
    parser.add_argument("--use-graph-memory", choices=("0", "1"), default=None)
    parser.add_argument("--use-world-model", choices=("0", "1"), default=None)
    parser.add_argument("--use-planner", choices=("0", "1"), default=None)
    parser.add_argument("--use-role-specialization", choices=("0", "1"), default=None)
    parser.add_argument("--use-prompt-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-curriculum-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-retrieval-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-state-estimation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-trust-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-recovery-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-delegation-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-operator-policy-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--use-transition-model-proposals", choices=("0", "1"), default=None)
    parser.add_argument("--allow-git-commands", choices=("0", "1"), default=None)
    parser.add_argument("--allow-http-requests", choices=("0", "1"), default=None)
    parser.add_argument("--allow-generated-path-mutations", choices=("0", "1"), default=None)
    parser.add_argument("--tolbert-python", default=None)
    parser.add_argument("--tolbert-config", default=None)
    parser.add_argument("--tolbert-checkpoint", default=None)
    parser.add_argument("--tolbert-nodes", default=None)
    parser.add_argument("--tolbert-cache", action="append", default=None)
    parser.add_argument("--tolbert-label-map", default=None)
    parser.add_argument("--tolbert-device", default=None)
    parser.add_argument("--external-task-manifest", action="append", default=None)
    parser.add_argument("--enforce-preflight", choices=("0", "1"), default="1")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    if args.use_tolbert_context is not None:
        config.use_tolbert_context = args.use_tolbert_context == "1"
    if args.use_skills is not None:
        config.use_skills = args.use_skills == "1"
    if args.use_graph_memory is not None:
        config.use_graph_memory = args.use_graph_memory == "1"
    if args.use_world_model is not None:
        config.use_world_model = args.use_world_model == "1"
    if args.use_planner is not None:
        config.use_planner = args.use_planner == "1"
    if args.use_role_specialization is not None:
        config.use_role_specialization = args.use_role_specialization == "1"
    if args.use_prompt_proposals is not None:
        config.use_prompt_proposals = args.use_prompt_proposals == "1"
    if args.use_curriculum_proposals is not None:
        config.use_curriculum_proposals = args.use_curriculum_proposals == "1"
    if args.use_retrieval_proposals is not None:
        config.use_retrieval_proposals = args.use_retrieval_proposals == "1"
    if args.use_state_estimation_proposals is not None:
        config.use_state_estimation_proposals = args.use_state_estimation_proposals == "1"
    if args.use_trust_proposals is not None:
        config.use_trust_proposals = args.use_trust_proposals == "1"
    if args.use_recovery_proposals is not None:
        config.use_recovery_proposals = args.use_recovery_proposals == "1"
    if args.use_delegation_proposals is not None:
        config.use_delegation_proposals = args.use_delegation_proposals == "1"
    if args.use_operator_policy_proposals is not None:
        config.use_operator_policy_proposals = args.use_operator_policy_proposals == "1"
    if args.use_transition_model_proposals is not None:
        config.use_transition_model_proposals = args.use_transition_model_proposals == "1"
    if args.allow_git_commands is not None:
        config.unattended_allow_git_commands = args.allow_git_commands == "1"
    if args.allow_http_requests is not None:
        config.unattended_allow_http_requests = args.allow_http_requests == "1"
    if args.allow_generated_path_mutations is not None:
        config.unattended_allow_generated_path_mutations = args.allow_generated_path_mutations == "1"
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
    if args.external_task_manifest:
        config.external_task_manifests_paths = tuple(args.external_task_manifest)
    bank = _build_task_bank(config)
    task = bank.get(args.task_id)
    safe_task_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task.task_id)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else config.run_checkpoints_dir / f"{safe_task_id}.json"
    )
    workspace_path = config.workspace_root / task.workspace_subdir
    before_workspace_snapshot = capture_workspace_snapshot(workspace_path)
    workspace_snapshot_path = (
        snapshot_workspace_tree(
            workspace_path,
            config.unattended_workspace_snapshot_root,
            task_id=task.task_id,
            run_label=safe_task_id,
        )
        if should_snapshot_workspace(config)
        else None
    )
    preflight = None
    if args.enforce_preflight == "1":
        preflight = run_unattended_preflight(config, task, repo_root=repo_root)
        if not preflight.passed:
            report_path = write_unattended_task_report(
                task=task,
                config=config,
                episode=None,
                preflight=preflight,
                report_path=Path(args.report_path) if args.report_path else None,
                before_workspace_snapshot=before_workspace_snapshot,
            )
            annotate_task_report_recovery(
                report_path,
                recovery_annotation(
                    config=config,
                    workspace_snapshot_path=workspace_snapshot_path,
                    rollback_performed=False,
                    rollback_reason="preflight_failed_before_execution",
                    workspace_path=workspace_path,
                ),
            )
            if report_path.exists():
                write_unattended_trust_ledger(config)
            print(f"outcome=safe_stop report={report_path}", file=sys.stderr)
            raise SystemExit(2)

    try:
        episode = AgentKernel(config=config).run_task(
            task,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )
    except Exception as exc:
        report_path = write_unattended_task_report(
            task=task,
            config=config,
            episode=None,
            preflight=preflight,
            report_path=Path(args.report_path) if args.report_path else None,
            before_workspace_snapshot=before_workspace_snapshot,
            outcome_override="unsafe_ambiguous",
            outcome_reasons_override=["runner_exception"],
            termination_reason_override=f"{exc.__class__.__name__}: {exc}",
            extra_uncertainties=["runner raised an exception before producing a complete episode record"],
        )
        rollback_performed = False
        if workspace_snapshot_path is not None and should_restore_on_runner_exception(config):
            restore_workspace_tree(workspace_snapshot_path, workspace_path)
            rollback_performed = True
        annotate_task_report_recovery(
            report_path,
            recovery_annotation(
                config=config,
                workspace_snapshot_path=workspace_snapshot_path,
                rollback_performed=rollback_performed,
                rollback_reason="runner_exception",
                workspace_path=workspace_path,
            ),
        )
        if report_path.exists():
            write_unattended_trust_ledger(config)
        print(f"outcome=unsafe_ambiguous report={report_path}", file=sys.stderr)
        raise SystemExit(1) from exc
    report_path = write_unattended_task_report(
        task=task,
        config=config,
        episode=episode,
        preflight=preflight,
        report_path=Path(args.report_path) if args.report_path else None,
        before_workspace_snapshot=before_workspace_snapshot,
    )
    print(f"task={episode.task_id} success={episode.success} steps={len(episode.steps)}")
    outcome, _ = classify_run_outcome(episode=episode, preflight=preflight)
    rollback_performed = False
    if workspace_snapshot_path is not None and should_restore_on_outcome(config, outcome):
        restore_workspace_tree(workspace_snapshot_path, workspace_path)
        rollback_performed = True
    annotate_task_report_recovery(
        report_path,
        recovery_annotation(
            config=config,
            workspace_snapshot_path=workspace_snapshot_path,
            rollback_performed=rollback_performed,
            rollback_reason="" if not rollback_performed else f"non_success_outcome:{outcome}",
            workspace_path=workspace_path,
        ),
    )
    if report_path.exists():
        write_unattended_trust_ledger(config)
    print(f"outcome={outcome} report={report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
