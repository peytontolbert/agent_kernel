from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.cycle_runner import autonomous_runtime_eval_flags, comparison_flags
from agent_kernel.improvement import ImprovementPlanner
from agent_kernel.extensions.improvement.artifacts import payload_with_active_artifact_context, retention_gate_for_payload
from agent_kernel.improvement_retention import proposal_gate_failure_reasons_by_benchmark_family, retention_evidence
from agent_kernel.extensions.strategy.subsystems import active_artifact_path_for_subsystem, base_subsystem_for, comparison_config_for_subsystem_artifact
from evals.harness import compare_abstraction_transfer_modes, run_eval


def _artifact_path_for_subsystem(config: KernelConfig, subsystem: str) -> Path:
    return active_artifact_path_for_subsystem(config, subsystem)


def _comparison_flags(subsystem: str, *, config: KernelConfig | None = None) -> dict[str, bool]:
    return comparison_flags(subsystem, config=config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem", required=True)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--cycles-path", default=None)
    parser.add_argument("--before-cycle-id", default=None)
    args = parser.parse_args()

    config = KernelConfig()
    cycles_path = Path(args.cycles_path) if args.cycles_path else config.improvement_cycles_path
    artifact_path = Path(args.artifact_path) if args.artifact_path else active_artifact_path_for_subsystem(config, args.subsystem)
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    prior_record = planner.prior_retained_artifact_record(
        cycles_path,
        args.subsystem,
        before_cycle_id=args.before_cycle_id,
    )
    if prior_record is None:
        raise SystemExit(f"no prior retained baseline exists for subsystem={args.subsystem}")
    snapshot_path = Path(str(prior_record.get("artifact_snapshot_path", "")))
    if not snapshot_path.exists():
        raise SystemExit(f"retained artifact snapshot does not exist: {snapshot_path}")

    baseline_config = comparison_config_for_subsystem_artifact(config, args.subsystem, snapshot_path)
    current_config = comparison_config_for_subsystem_artifact(config, args.subsystem, artifact_path)
    flags = autonomous_runtime_eval_flags(config, _comparison_flags(args.subsystem, config=config))
    if base_subsystem_for(args.subsystem, config.capability_modules_path) == "operators":
        baseline_metrics = compare_abstraction_transfer_modes(
            config=baseline_config,
            include_episode_memory=flags["include_episode_memory"],
            include_verifier_memory=flags["include_verifier_memory"],
            include_benchmark_candidates=flags["include_benchmark_candidates"],
            include_verifier_candidates=flags["include_verifier_candidates"],
            include_generated=flags["include_generated"],
            include_failure_generated=flags["include_failure_generated"],
        ).operator_metrics
        current_metrics = compare_abstraction_transfer_modes(
            config=current_config,
            include_episode_memory=flags["include_episode_memory"],
            include_verifier_memory=flags["include_verifier_memory"],
            include_benchmark_candidates=flags["include_benchmark_candidates"],
            include_verifier_candidates=flags["include_verifier_candidates"],
            include_generated=flags["include_generated"],
            include_failure_generated=flags["include_failure_generated"],
        ).operator_metrics
    else:
        baseline_metrics = run_eval(config=baseline_config, **flags)
        current_metrics = run_eval(config=current_config, **flags)
    current_payload: dict[str, object] | None = None
    baseline_payload: dict[str, object] | None = None
    try:
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current_payload = loaded
    except (OSError, json.JSONDecodeError):
        current_payload = None
    try:
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            baseline_payload = loaded
    except (OSError, json.JSONDecodeError):
        baseline_payload = None
    comparison_payload = payload_with_active_artifact_context(
        current_payload,
        active_artifact_path=snapshot_path,
        active_artifact_payload=baseline_payload,
    )
    evidence = retention_evidence(
        args.subsystem,
        baseline_metrics,
        current_metrics,
        payload=comparison_payload,
        capability_modules_path=config.capability_modules_path,
    )
    retention_gate = retention_gate_for_payload(
        args.subsystem,
        current_payload,
        capability_modules_path=config.capability_modules_path,
    )
    proposal_gate_failures = proposal_gate_failure_reasons_by_benchmark_family(
        retention_gate,
        evidence,
        subject="current artifact",
    )

    print(
        f"subsystem={args.subsystem} "
        f"current_artifact_path={artifact_path} "
        f"baseline_snapshot_path={snapshot_path} "
        f"baseline_cycle_id={prior_record.get('cycle_id', '')} "
        f"baseline_pass_rate={baseline_metrics.pass_rate:.2f} "
        f"current_pass_rate={current_metrics.pass_rate:.2f} "
        f"pass_rate_delta={current_metrics.pass_rate - baseline_metrics.pass_rate:.2f} "
        f"baseline_average_steps={baseline_metrics.average_steps:.2f} "
        f"current_average_steps={current_metrics.average_steps:.2f} "
        f"average_steps_delta={current_metrics.average_steps - baseline_metrics.average_steps:.2f}"
    )
    print(
        "proposal_metrics "
        f"baseline_proposal_selected_steps={baseline_metrics.proposal_selected_steps} "
        f"current_proposal_selected_steps={current_metrics.proposal_selected_steps} "
        f"proposal_selected_steps_delta={current_metrics.proposal_selected_steps - baseline_metrics.proposal_selected_steps} "
        f"baseline_novel_valid_command_rate={baseline_metrics.novel_valid_command_rate:.2f} "
        f"current_novel_valid_command_rate={current_metrics.novel_valid_command_rate:.2f} "
        f"novel_valid_command_rate_delta={current_metrics.novel_valid_command_rate - baseline_metrics.novel_valid_command_rate:.2f}"
    )
    print(
        "tolbert_takeover "
        f"baseline_tolbert_primary_episodes={baseline_metrics.tolbert_primary_episodes} "
        f"current_tolbert_primary_episodes={current_metrics.tolbert_primary_episodes} "
        f"tolbert_primary_episodes_delta={current_metrics.tolbert_primary_episodes - baseline_metrics.tolbert_primary_episodes}"
    )
    family_pass_rate_delta = evidence.get("family_pass_rate_delta", {})
    if isinstance(family_pass_rate_delta, dict):
        for family in sorted(family_pass_rate_delta):
            print(
                f"family_delta benchmark_family={family} "
                f"pass_rate_delta={float(family_pass_rate_delta[family]):.2f}"
            )
    generated_family_pass_rate_delta = evidence.get("generated_family_pass_rate_delta", {})
    if isinstance(generated_family_pass_rate_delta, dict):
        for family in sorted(generated_family_pass_rate_delta):
            print(
                f"generated_family_delta benchmark_family={family} "
                f"pass_rate_delta={float(generated_family_pass_rate_delta[family]):.2f}"
            )
    if "failure_recovery_pass_rate_delta" in evidence:
        print(
            "generated_kind_delta generated_kind=failure_recovery "
            f"pass_rate_delta={float(evidence['failure_recovery_pass_rate_delta']):.2f}"
        )
    long_horizon_summary = evidence.get("long_horizon_summary", {})
    if isinstance(long_horizon_summary, dict):
        long_horizon_task_count = int(long_horizon_summary.get("baseline_task_count", 0) or 0) + int(
            long_horizon_summary.get("candidate_task_count", 0) or 0
        )
        if long_horizon_task_count > 0:
            world_feedback = long_horizon_summary.get("world_feedback", {})
            if not isinstance(world_feedback, dict):
                world_feedback = {}
            print(
                "long_horizon_delta "
                f"pass_rate_delta={float(long_horizon_summary.get('pass_rate_delta', 0.0) or 0.0):.2f} "
                f"novel_valid_command_rate_delta={float(long_horizon_summary.get('novel_valid_command_rate_delta', 0.0) or 0.0):.2f} "
                f"progress_calibration_mae_gain={float(world_feedback.get('progress_calibration_mae_gain', 0.0) or 0.0):.2f}"
            )
    validation_family_summary = evidence.get("validation_family_summary", {})
    if isinstance(validation_family_summary, dict):
        validation_task_count = int(validation_family_summary.get("baseline_primary_task_count", 0) or 0) + int(
            validation_family_summary.get("candidate_primary_task_count", 0) or 0
        )
        validation_generated_count = int(validation_family_summary.get("baseline_generated_task_count", 0) or 0) + int(
            validation_family_summary.get("candidate_generated_task_count", 0) or 0
        )
        validation_feedback_steps = int(
            validation_family_summary.get("baseline_world_feedback_step_count", 0) or 0
        ) + int(validation_family_summary.get("candidate_world_feedback_step_count", 0) or 0)
        if validation_task_count > 0 or validation_generated_count > 0 or validation_feedback_steps > 0:
            world_feedback = validation_family_summary.get("world_feedback", {})
            if not isinstance(world_feedback, dict):
                world_feedback = {}
            print(
                "validation_family_delta "
                f"primary_pass_rate_delta={float(validation_family_summary.get('primary_pass_rate_delta', 0.0) or 0.0):.2f} "
                f"generated_pass_rate_delta={float(validation_family_summary.get('generated_pass_rate_delta', 0.0) or 0.0):.2f} "
                f"novel_valid_command_rate_delta={float(validation_family_summary.get('novel_valid_command_rate_delta', 0.0) or 0.0):.2f} "
                f"progress_calibration_mae_gain={float(world_feedback.get('progress_calibration_mae_gain', 0.0) or 0.0):.2f}"
            )
    shared_repo_bundle_summary = evidence.get("shared_repo_bundle_summary", {})
    if isinstance(shared_repo_bundle_summary, dict):
        shared_repo_candidate_count = int(
            shared_repo_bundle_summary.get("baseline_shared_repo_candidate_count", 0) or 0
        ) + int(shared_repo_bundle_summary.get("candidate_shared_repo_candidate_count", 0) or 0)
        if shared_repo_candidate_count > 0:
            print(
                "shared_repo_bundle_delta "
                f"baseline_complete_candidate_count={int(shared_repo_bundle_summary.get('baseline_shared_repo_complete_candidate_count', 0) or 0)} "
                f"current_complete_candidate_count={int(shared_repo_bundle_summary.get('candidate_shared_repo_complete_candidate_count', 0) or 0)} "
                f"complete_candidate_delta={int(shared_repo_bundle_summary.get('shared_repo_complete_candidate_count_delta', 0) or 0)} "
                f"baseline_worker_candidate_count={int(shared_repo_bundle_summary.get('baseline_shared_repo_worker_candidate_count', 0) or 0)} "
                f"current_worker_candidate_count={int(shared_repo_bundle_summary.get('candidate_shared_repo_worker_candidate_count', 0) or 0)} "
                f"worker_candidate_delta={int(shared_repo_bundle_summary.get('shared_repo_worker_candidate_count_delta', 0) or 0)} "
                f"baseline_complete_integrator_candidate_count={int(shared_repo_bundle_summary.get('baseline_shared_repo_complete_integrator_candidate_count', 0) or 0)} "
                f"current_complete_integrator_candidate_count={int(shared_repo_bundle_summary.get('candidate_shared_repo_complete_integrator_candidate_count', 0) or 0)} "
                f"complete_integrator_candidate_delta={int(shared_repo_bundle_summary.get('shared_repo_complete_integrator_candidate_count_delta', 0) or 0)} "
                f"baseline_incomplete_integrator_candidate_count={int(shared_repo_bundle_summary.get('baseline_shared_repo_incomplete_integrator_candidate_count', 0) or 0)} "
                f"current_incomplete_integrator_candidate_count={int(shared_repo_bundle_summary.get('candidate_shared_repo_incomplete_integrator_candidate_count', 0) or 0)} "
                f"incomplete_integrator_candidate_delta={int(shared_repo_bundle_summary.get('shared_repo_incomplete_integrator_candidate_count_delta', 0) or 0)} "
                f"bundle_coherence_delta={int(shared_repo_bundle_summary.get('candidate_bundle_coherence_delta', 0) or 0)}"
            )
    retrieval_reuse_summary = evidence.get("retrieval_reuse_summary", {})
    if isinstance(retrieval_reuse_summary, dict):
        retrieval_procedure_count = int(retrieval_reuse_summary.get("baseline_procedure_count", 0) or 0) + int(
            retrieval_reuse_summary.get("candidate_procedure_count", 0) or 0
        )
        if retrieval_procedure_count > 0:
            print(
                "retrieval_reuse_delta "
                f"baseline_retrieval_backed_procedure_count={int(retrieval_reuse_summary.get('baseline_retrieval_backed_procedure_count', 0) or 0)} "
                f"current_retrieval_backed_procedure_count={int(retrieval_reuse_summary.get('candidate_retrieval_backed_procedure_count', 0) or 0)} "
                f"retrieval_backed_procedure_delta={int(retrieval_reuse_summary.get('retrieval_backed_procedure_count_delta', 0) or 0)} "
                f"baseline_trusted_retrieval_procedure_count={int(retrieval_reuse_summary.get('baseline_trusted_retrieval_procedure_count', 0) or 0)} "
                f"current_trusted_retrieval_procedure_count={int(retrieval_reuse_summary.get('candidate_trusted_retrieval_procedure_count', 0) or 0)} "
                f"trusted_retrieval_procedure_delta={int(retrieval_reuse_summary.get('trusted_retrieval_procedure_count_delta', 0) or 0)} "
                f"baseline_verified_retrieval_command_count={int(retrieval_reuse_summary.get('baseline_verified_retrieval_command_count', 0) or 0)} "
                f"current_verified_retrieval_command_count={int(retrieval_reuse_summary.get('candidate_verified_retrieval_command_count', 0) or 0)} "
                f"verified_retrieval_command_delta={int(retrieval_reuse_summary.get('verified_retrieval_command_count_delta', 0) or 0)} "
                f"baseline_selected_retrieval_span_count={int(retrieval_reuse_summary.get('baseline_selected_retrieval_span_count', 0) or 0)} "
                f"current_selected_retrieval_span_count={int(retrieval_reuse_summary.get('candidate_selected_retrieval_span_count', 0) or 0)} "
                f"selected_retrieval_span_delta={int(retrieval_reuse_summary.get('selected_retrieval_span_count_delta', 0) or 0)} "
                f"retrieval_reuse_delta={int(retrieval_reuse_summary.get('retrieval_reuse_delta', 0) or 0)}"
            )
    for family in sorted(proposal_gate_failures):
        print(
            f"family_proposal_gate_failure benchmark_family={family} "
            f"reason={proposal_gate_failures[family]}"
        )


if __name__ == "__main__":
    main()
