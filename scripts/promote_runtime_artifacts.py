from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.delegation_improvement import build_delegation_proposal_artifact
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner, retention_gate_for_payload
from agent_kernel.operator_policy_improvement import build_operator_policy_proposal_artifact
from agent_kernel.recovery_improvement import build_recovery_proposal_artifact
from agent_kernel.state_estimation_improvement import build_state_estimation_proposal_artifact
from agent_kernel.transition_model_improvement import build_transition_model_proposal_artifact
from agent_kernel.trust_improvement import build_trust_proposal_artifact
from agent_kernel.universe_improvement import (
    build_operating_envelope_artifact,
    build_universe_constitution_artifact,
    compose_universe_bundle_payloads,
    materialize_operating_envelope_payload,
    materialize_universe_constitution_payload,
    write_universe_bundle_files,
)
from agent_kernel.world_model_improvement import build_world_model_proposal_artifact
from evals.metrics import EvalMetrics


def _normalize_tool_candidates(payload: object) -> dict[str, object]:
    if isinstance(payload, list):
        candidates = []
        for record in payload:
            if not isinstance(record, dict):
                continue
            candidate = dict(record)
            candidate.setdefault("promotion_stage", "candidate_procedure")
            candidate.setdefault("lifecycle_state", "candidate")
            candidates.append(candidate)
        return {
            "spec_version": "asi_v1",
            "artifact_kind": "tool_candidate_set",
            "lifecycle_state": "candidate",
            "retention_gate": retention_gate_for_payload("tooling", None),
            "candidates": candidates,
        }
    if isinstance(payload, dict):
        normalized = dict(payload)
        normalized.setdefault("spec_version", "asi_v1")
        normalized.setdefault("artifact_kind", "tool_candidate_set")
        normalized.setdefault("lifecycle_state", "candidate")
        normalized.setdefault("retention_gate", retention_gate_for_payload("tooling", normalized))
        candidates = normalized.get("candidates", [])
        if isinstance(candidates, list):
            rewritten = []
            for record in candidates:
                if not isinstance(record, dict):
                    continue
                candidate = dict(record)
                candidate.setdefault("promotion_stage", "candidate_procedure")
                candidate.setdefault("lifecycle_state", "candidate")
                rewritten.append(candidate)
            normalized["candidates"] = rewritten
        return normalized
    return {
        "spec_version": "asi_v1",
        "artifact_kind": "tool_candidate_set",
        "lifecycle_state": "candidate",
        "retention_gate": retention_gate_for_payload("tooling", None),
        "candidates": [],
    }


def _normalize_proposal_artifact(
    payload: object,
    *,
    artifact_kind: str,
    subsystem: str,
) -> dict[str, object]:
    defaults: dict[str, object] = {}
    if subsystem == "world_model":
        defaults = build_world_model_proposal_artifact(EvalMetrics(total=0, passed=0), {})
    elif subsystem == "state_estimation":
        defaults = build_state_estimation_proposal_artifact(EvalMetrics(total=0, passed=0), {})
    elif subsystem == "trust":
        defaults = build_trust_proposal_artifact(EvalMetrics(total=0, passed=0), {})
    elif subsystem == "recovery":
        defaults = build_recovery_proposal_artifact(EvalMetrics(total=0, passed=0), {})
    elif subsystem == "delegation":
        defaults = build_delegation_proposal_artifact(KernelConfig())
    elif subsystem == "operator_policy":
        defaults = build_operator_policy_proposal_artifact(KernelConfig())
    elif subsystem == "transition_model":
        defaults = build_transition_model_proposal_artifact(KernelConfig().trajectories_root)
    normalized = dict(defaults)
    if isinstance(payload, dict):
        normalized.update(payload)
    normalized.setdefault("spec_version", "asi_v1")
    normalized.setdefault("artifact_kind", artifact_kind)
    normalized.setdefault("lifecycle_state", "proposed")
    normalized.setdefault("retention_gate", retention_gate_for_payload(subsystem, normalized))
    for key in ("controls", "planning_controls", "latent_controls", "policy_controls", "transition_summary", "ledger_summary", "retention_gate"):
        default_value = defaults.get(key)
        current_value = normalized.get(key)
        if isinstance(default_value, dict):
            merged = dict(default_value)
            if isinstance(current_value, dict):
                merged.update(current_value)
            normalized[key] = merged
    proposals = normalized.get("proposals", [])
    if not isinstance(proposals, list):
        normalized["proposals"] = []
    context = normalized.get("generation_context", {})
    if not isinstance(context, dict):
        context = {}
    context.setdefault("migrated_from_legacy_runtime_artifact", True)
    normalized["generation_context"] = context
    return normalized


def _normalize_capability_modules(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        normalized = dict(payload)
    elif isinstance(payload, list):
        normalized = {"modules": list(payload)}
    else:
        normalized = {}
    normalized.setdefault("spec_version", "asi_v1")
    normalized.setdefault("artifact_kind", "capability_module_set")
    normalized.setdefault("lifecycle_state", "proposed")
    normalized.setdefault("retention_gate", retention_gate_for_payload("capabilities", normalized))
    modules = normalized.get("modules", [])
    if not isinstance(modules, list):
        normalized["modules"] = []
    context = normalized.get("generation_context", {})
    if not isinstance(context, dict):
        context = {}
    context.setdefault("migrated_from_legacy_runtime_artifact", True)
    normalized["generation_context"] = context
    return normalized


def _normalize_runtime_artifact(path: Path, subsystem: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if subsystem == "tooling":
        return _normalize_tool_candidates(payload)
    if subsystem == "capabilities":
        return _normalize_capability_modules(payload)
    if subsystem == "tolbert_model":
        normalized = dict(payload) if isinstance(payload, dict) else {}
        normalized.setdefault("spec_version", "asi_v1")
        normalized.setdefault("artifact_kind", "tolbert_model_bundle")
        normalized.setdefault("lifecycle_state", "candidate")
        normalized.setdefault("retention_gate", retention_gate_for_payload("tolbert_model", normalized))
        normalized.setdefault("training_controls", {"base_model_name": "bert-base-uncased", "num_epochs": 1, "lr": 5.0e-5, "batch_size": 8})
        normalized.setdefault("dataset_manifest", {"total_examples": 0})
        normalized.setdefault(
            "model_surfaces",
            {
                "retrieval_surface": True,
                "policy_head": True,
                "value_head": True,
                "transition_head": True,
                "latent_state": True,
            },
        )
        normalized.setdefault(
            "runtime_policy",
            {
                "shadow_benchmark_families": [],
                "primary_benchmark_families": [],
                "min_path_confidence": 0.75,
                "require_trusted_retrieval": True,
                "fallback_to_vllm_on_low_confidence": True,
                "allow_direct_command_primary": True,
                "allow_skill_primary": True,
                "primary_min_command_score": 2,
                "use_value_head": True,
                "use_transition_head": True,
                "use_policy_head": True,
                "use_latent_state": True,
            },
        )
        normalized.setdefault(
            "decoder_policy",
            {
                "allow_retrieval_guidance": True,
                "allow_skill_commands": True,
                "allow_task_suggestions": True,
                "allow_stop_decision": True,
                "min_stop_completion_ratio": 0.95,
                "max_task_suggestions": 3,
            },
        )
        normalized.setdefault(
            "rollout_policy",
            {
                "predicted_progress_gain_weight": 3.0,
                "predicted_conflict_penalty_weight": 4.0,
                "predicted_preserved_bonus_weight": 1.0,
                "predicted_workflow_bonus_weight": 1.5,
                "latent_progress_bonus_weight": 1.0,
                "latent_risk_penalty_weight": 2.0,
                "recover_from_stall_bonus_weight": 1.5,
                "stop_completion_weight": 8.0,
                "stop_missing_expected_penalty_weight": 6.0,
                "stop_forbidden_penalty_weight": 6.0,
                "stop_preserved_penalty_weight": 4.0,
                "stable_stop_bonus_weight": 1.5,
            },
        )
        normalized.setdefault(
            "liftoff_gate",
            {
                "min_pass_rate_delta": 0.0,
                "max_step_regression": 0.0,
                "max_regressed_families": 0,
                "require_generated_lane_non_regression": True,
                "require_failure_recovery_non_regression": True,
                "require_shadow_signal": True,
                "min_shadow_episodes_per_promoted_family": 1,
            },
        )
        normalized.setdefault("runtime_paths", {})
        normalized.setdefault("proposals", [{"area": "balanced", "priority": 3, "reason": "migrated legacy Tolbert model artifact"}])
        return normalized
    if subsystem == "qwen_adapter":
        normalized = dict(payload) if isinstance(payload, dict) else {}
        normalized.setdefault("spec_version", "asi_v1")
        normalized.setdefault("artifact_kind", "qwen_adapter_bundle")
        normalized.setdefault("lifecycle_state", "candidate")
        normalized.setdefault("retention_gate", retention_gate_for_payload("qwen_adapter", normalized))
        normalized.setdefault("runtime_role", "support_runtime")
        normalized.setdefault("training_objective", "qlora_sft")
        normalized.setdefault("base_model_name", KernelConfig().model_name)
        normalized.setdefault(
            "runtime_policy",
            {
                "allow_primary_routing": False,
                "allow_shadow_routing": True,
                "allow_teacher_generation": True,
                "allow_post_liftoff_fallback": True,
                "require_retained_promotion_for_runtime_use": True,
            },
        )
        normalized.setdefault("training_dataset_manifest", {"total_examples": 0})
        normalized.setdefault("supported_benchmark_families", ["repository", "project", "workflow"])
        normalized.setdefault("runtime_paths", {"adapter_output_dir": "", "merged_output_dir": "", "adapter_manifest_path": ""})
        return normalized
    artifact_kind = {
        "benchmark": "benchmark_candidate_set",
        "retrieval": "retrieval_policy_set",
        "tolbert_model": "tolbert_model_bundle",
        "qwen_adapter": "qwen_adapter_bundle",
        "verifier": "verifier_candidate_set",
        "policy": "prompt_proposal_set",
        "world_model": "world_model_policy_set",
        "state_estimation": "state_estimation_policy_set",
        "trust": "trust_policy_set",
        "recovery": "recovery_policy_set",
        "delegation": "delegated_runtime_policy_set",
        "operator_policy": "operator_policy_set",
        "transition_model": "transition_model_policy_set",
        "curriculum": "curriculum_proposal_set",
    }[subsystem]
    return _normalize_proposal_artifact(payload, artifact_kind=artifact_kind, subsystem=subsystem)


def _normalize_universe_bundle(config: KernelConfig) -> dict[str, dict[str, object]]:
    zero_metrics = EvalMetrics(total=0, passed=0)
    combined_source = _load_json_dict(config.universe_contract_path)
    constitution_source = _load_json_dict(config.universe_constitution_path) or combined_source
    envelope_source = _load_json_dict(config.operating_envelope_path) or combined_source

    constitution_defaults = build_universe_constitution_artifact(
        zero_metrics,
        {},
        current_payload=constitution_source or combined_source,
    )
    constitution_payload = dict(constitution_defaults)
    if constitution_source:
        constitution_payload.update(
            materialize_universe_constitution_payload(constitution_source, lifecycle_state="retained")
        )
    constitution_payload["spec_version"] = "asi_v1"
    constitution_payload["artifact_kind"] = "universe_constitution"
    constitution_payload["control_schema"] = "universe_constitution_v1"
    constitution_payload["lifecycle_state"] = "retained"
    constitution_payload["retention_gate"] = retention_gate_for_payload("universe_constitution", constitution_payload)
    constitution_payload["retention_decision"] = {"state": "retain"}
    constitution_payload["generation_context"] = _promoted_runtime_generation_context(constitution_payload)

    envelope_defaults = build_operating_envelope_artifact(
        zero_metrics,
        {},
        current_payload=envelope_source or combined_source,
    )
    operating_envelope_payload = dict(envelope_defaults)
    if envelope_source:
        operating_envelope_payload.update(
            materialize_operating_envelope_payload(envelope_source, lifecycle_state="retained")
        )
    operating_envelope_payload["spec_version"] = "asi_v1"
    operating_envelope_payload["artifact_kind"] = "operating_envelope"
    operating_envelope_payload["control_schema"] = "operating_envelope_v1"
    operating_envelope_payload["lifecycle_state"] = "retained"
    operating_envelope_payload["retention_gate"] = retention_gate_for_payload(
        "operating_envelope",
        operating_envelope_payload,
    )
    operating_envelope_payload["retention_decision"] = {"state": "retain"}
    operating_envelope_payload["generation_context"] = _promoted_runtime_generation_context(operating_envelope_payload)

    combined_payload = compose_universe_bundle_payloads(
        constitution_payload=constitution_payload,
        operating_envelope_payload=operating_envelope_payload,
        baseline_payload=combined_source,
        lifecycle_state="retained",
    )["universe"]
    combined_payload["retention_gate"] = retention_gate_for_payload("universe", combined_payload)
    combined_payload["retention_decision"] = {"state": "retain"}
    combined_payload["generation_context"] = _promoted_runtime_generation_context(
        combined_payload,
        synchronized_from_split_bundle=True,
    )

    return {
        "universe_constitution": constitution_payload,
        "operating_envelope": operating_envelope_payload,
        "universe": combined_payload,
    }


def _load_json_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _promoted_runtime_generation_context(
    payload: dict[str, object],
    *,
    synchronized_from_split_bundle: bool = False,
) -> dict[str, object]:
    context = payload.get("generation_context", {})
    normalized = dict(context) if isinstance(context, dict) else {}
    normalized.setdefault("migrated_from_legacy_runtime_artifact", True)
    if synchronized_from_split_bundle:
        normalized["synchronized_from_split_universe_bundle"] = True
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsystem",
        action="append",
        choices=(
            "benchmark",
            "retrieval",
            "tolbert_model",
            "qwen_adapter",
            "tooling",
            "verifier",
            "policy",
            "universe",
            "universe_constitution",
            "operating_envelope",
            "world_model",
            "state_estimation",
            "trust",
            "recovery",
            "delegation",
            "operator_policy",
            "transition_model",
            "curriculum",
            "capabilities",
        ),
        help="limit normalization to one or more specific runtime-managed artifact families",
    )
    args = parser.parse_args()

    config = KernelConfig()
    config.ensure_directories()
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    subsystems = args.subsystem or [
        "benchmark",
        "retrieval",
        "tolbert_model",
        "qwen_adapter",
        "tooling",
        "verifier",
        "policy",
        "universe_constitution",
        "operating_envelope",
        "world_model",
        "state_estimation",
        "trust",
        "recovery",
        "delegation",
        "operator_policy",
        "transition_model",
        "curriculum",
        "capabilities",
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    path_map = {
        "benchmark": config.benchmark_candidates_path,
        "retrieval": config.retrieval_proposals_path,
        "tolbert_model": config.tolbert_model_artifact_path,
        "qwen_adapter": config.qwen_adapter_artifact_path,
        "tooling": config.tool_candidates_path,
        "verifier": config.verifier_contracts_path,
        "policy": config.prompt_proposals_path,
        "universe": config.universe_contract_path,
        "universe_constitution": config.universe_constitution_path,
        "operating_envelope": config.operating_envelope_path,
        "world_model": config.world_model_proposals_path,
        "state_estimation": config.state_estimation_proposals_path,
        "trust": config.trust_proposals_path,
        "recovery": config.recovery_proposals_path,
        "delegation": config.delegation_proposals_path,
        "operator_policy": config.operator_policy_proposals_path,
        "transition_model": config.transition_model_proposals_path,
        "curriculum": config.curriculum_proposals_path,
        "capabilities": config.capability_modules_path,
    }

    promoted_paths: list[str] = []
    universe_promoted = False
    for subsystem in subsystems:
        if subsystem in {"universe", "universe_constitution", "operating_envelope"}:
            if universe_promoted:
                continue
            universe_promoted = True
            universe_payloads = _normalize_universe_bundle(config)
            canonical_subsystems = ("universe_constitution", "operating_envelope")
            for canonical_subsystem in canonical_subsystems:
                artifact_path = path_map[canonical_subsystem]
                normalized = universe_payloads[canonical_subsystem]
                planner.append_cycle_record(
                    config.improvement_cycles_path,
                    ImprovementCycleRecord(
                        cycle_id=f"bootstrap:{canonical_subsystem}:{timestamp}",
                        state="record",
                        subsystem=canonical_subsystem,
                        action="promote_runtime_artifact",
                        artifact_path=str(artifact_path),
                        artifact_kind=str(normalized.get("artifact_kind", "")),
                        reason="normalize live universe runtime artifact into a managed canonical split form",
                        metrics_summary={
                            "migration": True,
                            "lifecycle_state": str(normalized.get("lifecycle_state", "")),
                            "managed_proposals": len(normalized.get("proposals", [])),
                            "synchronized_legacy_contract_path": str(config.universe_contract_path),
                        },
                        artifact_lifecycle_state=str(normalized.get("lifecycle_state", "")),
                    ),
                )
            promoted_paths.extend(
                write_universe_bundle_files(
                    contract_path=path_map["universe"],
                    constitution_path=path_map["universe_constitution"],
                    operating_envelope_path=path_map["operating_envelope"],
                    bundle=universe_payloads,
                ).values()
            )
            continue
        artifact_path = path_map[subsystem]
        normalized = _normalize_runtime_artifact(artifact_path, subsystem)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        planner.append_cycle_record(
            config.improvement_cycles_path,
            ImprovementCycleRecord(
                cycle_id=f"bootstrap:{subsystem}:{timestamp}",
                state="record",
                subsystem=subsystem,
                action="promote_runtime_artifact",
                artifact_path=str(artifact_path),
                artifact_kind=str(normalized.get("artifact_kind", "")),
                reason="normalize live runtime artifact into a managed spec-compliant form",
                metrics_summary={
                    "migration": True,
                    "lifecycle_state": str(normalized.get("lifecycle_state", "")),
                    "managed_proposals": len(normalized.get("proposals", [])) if isinstance(normalized, dict) else 0,
                    "managed_candidates": len(normalized.get("candidates", [])) if isinstance(normalized, dict) else 0,
                },
                artifact_lifecycle_state=str(normalized.get("lifecycle_state", "")),
            ),
        )
        promoted_paths.append(str(artifact_path))

    for path in promoted_paths:
        print(path)


if __name__ == "__main__":
    main()
