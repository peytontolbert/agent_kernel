from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import json
from pathlib import Path

from .benchmark_synthesis import synthesize_benchmark_candidates
from .capabilities import load_capability_modules
from .capability_improvement import build_capability_module_artifact
from .config import KernelConfig
from .modeling.qwen import build_qwen_adapter_candidate_artifact
from .modeling.tolbert.delta import resolve_tolbert_runtime_checkpoint_path
from .curriculum_improvement import build_curriculum_proposal_artifact
from .delegation_improvement import build_delegation_proposal_artifact
from .export_governance import govern_improvement_export_storage
from .extractors import extract_operator_classes, extract_successful_command_skills, extract_tool_candidates
from .operator_policy_improvement import build_operator_policy_proposal_artifact
from .policy_improvement import build_policy_proposal_artifact
from .recovery_improvement import build_recovery_proposal_artifact
from .retrieval_improvement import build_retrieval_proposal_artifact
from .state_estimation_improvement import build_state_estimation_proposal_artifact
from .tolbert_model_improvement import build_tolbert_model_candidate_artifact, cleanup_tolbert_model_candidate_storage
from .transition_model_improvement import build_transition_model_proposal_artifact
from .trust_improvement import build_trust_proposal_artifact
from .kernel_catalog import kernel_catalog_list, kernel_catalog_mapping, kernel_catalog_record_list
from .universe_improvement import (
    build_operating_envelope_artifact,
    build_universe_constitution_artifact,
    build_universe_contract_artifact,
    comparison_shadow_universe_bundle_paths,
    compose_universe_bundle_payloads,
    universe_bundle_lifecycle_state,
    write_universe_bundle_files,
)
from .verifier_improvement import synthesize_verifier_contracts
from .world_model_improvement import build_world_model_proposal_artifact


@dataclass(frozen=True, slots=True)
class SubsystemSpec:
    subsystem: str
    base_subsystem: str
    artifact_path_attr: str
    proposal_toggle_attr: str | None = None
    generator_kind: str = "default"
    artifact_kind: str = ""
    action: str = "observe"
    baseline_flag_updates: dict[str, bool] = field(default_factory=dict)
    candidate_flag_updates: dict[str, bool] = field(default_factory=dict)
    capability_tags: tuple[str, ...] = ()
    strategy_hooks: dict[str, str] = field(default_factory=dict)


_BASE_FLAGS = {
    str(key): bool(value)
    for key, value in kernel_catalog_mapping("subsystems", "base_flags").items()
}


def _load_builtin_specs() -> dict[str, SubsystemSpec]:
    specs: dict[str, SubsystemSpec] = {}
    for item in kernel_catalog_record_list("subsystems", "builtin_specs"):
        subsystem_id = str(item.get("subsystem", "")).strip()
        if not subsystem_id:
            continue
        specs[subsystem_id] = SubsystemSpec(
            subsystem=subsystem_id,
            base_subsystem=str(item.get("base_subsystem", "")).strip() or subsystem_id,
            artifact_path_attr=str(item.get("artifact_path_attr", "")).strip(),
            proposal_toggle_attr=str(item.get("proposal_toggle_attr", "")).strip() or None,
            generator_kind=str(item.get("generator_kind", "default")).strip() or "default",
            artifact_kind=str(item.get("artifact_kind", "")).strip(),
            action=str(item.get("action", "observe")).strip() or "observe",
            baseline_flag_updates={
                str(key): bool(value)
                for key, value in dict(item.get("baseline_flag_updates", {})).items()
                if str(key).strip()
            },
            candidate_flag_updates={
                str(key): bool(value)
                for key, value in dict(item.get("candidate_flag_updates", {})).items()
                if str(key).strip()
            },
            capability_tags=tuple(
                token
                for token in (str(value).strip() for value in item.get("capability_tags", []))
                if token
            ),
            strategy_hooks={
                str(key): str(value).strip()
                for key, value in dict(item.get("strategy_hooks", {})).items()
                if str(key).strip() and str(value).strip()
            },
        )
    return specs


_BUILTIN_SPECS: dict[str, SubsystemSpec] = _load_builtin_specs()
_CONFIG_FIELD_NAMES = {entry.name for entry in fields(KernelConfig)}


def subsystem_spec(subsystem: str) -> SubsystemSpec:
    normalized = str(subsystem).strip()
    if normalized not in _BUILTIN_SPECS:
        raise ValueError(f"unsupported subsystem: {subsystem}")
    return _BUILTIN_SPECS[normalized]


def resolved_subsystem_spec(subsystem: str, capability_modules_path: Path | None = None) -> SubsystemSpec:
    normalized = str(subsystem).strip()
    if normalized in _BUILTIN_SPECS:
        return _BUILTIN_SPECS[normalized]
    external = external_subsystem_specs(capability_modules_path)
    if normalized not in external:
        raise ValueError(f"unsupported subsystem: {subsystem}")
    return external[normalized]


def base_subsystem_for(subsystem: str, capability_modules_path: Path | None = None) -> str:
    return resolved_subsystem_spec(subsystem, capability_modules_path).base_subsystem


def active_artifact_path_for_subsystem(config: KernelConfig, subsystem: str) -> Path:
    spec = resolved_subsystem_spec(subsystem, config.capability_modules_path)
    return getattr(config, spec.artifact_path_attr)


def config_with_subsystem_artifact_path(
    config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
) -> KernelConfig:
    spec = resolved_subsystem_spec(subsystem, config.capability_modules_path)
    return replace(config, **{spec.artifact_path_attr: artifact_path})


def baseline_candidate_flags(
    subsystem: str,
    capability_modules_path: Path | None = None,
) -> tuple[dict[str, bool], dict[str, bool]]:
    spec = resolved_subsystem_spec(subsystem, capability_modules_path)
    baseline = dict(_BASE_FLAGS)
    candidate = dict(_BASE_FLAGS)
    baseline.update(spec.baseline_flag_updates)
    candidate.update(spec.candidate_flag_updates)
    return baseline, candidate


def comparison_config_for_subsystem_artifact(
    config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
) -> KernelConfig:
    spec = resolved_subsystem_spec(subsystem, config.capability_modules_path)
    if spec.base_subsystem == "universe":
        return _comparison_universe_config(config, spec.subsystem, artifact_path)
    candidate = config_with_subsystem_artifact_path(config, subsystem, artifact_path)
    if spec.base_subsystem == "tolbert_model" and artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        runtime_paths = payload.get("runtime_paths", {}) if isinstance(payload, dict) else {}
        if isinstance(runtime_paths, dict):
            config_path = runtime_paths.get("config_path", candidate.tolbert_config_path)
            checkpoint_path = resolve_tolbert_runtime_checkpoint_path(runtime_paths, artifact_path=artifact_path) or runtime_paths.get(
                "checkpoint_path",
                candidate.tolbert_checkpoint_path,
            )
            nodes_path = runtime_paths.get("nodes_path", candidate.tolbert_nodes_path)
            label_map_path = runtime_paths.get("label_map_path", candidate.tolbert_label_map_path)
            candidate = replace(
                candidate,
                tolbert_config_path=None if config_path is None else str(config_path),
                tolbert_checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
                tolbert_nodes_path=None if nodes_path is None else str(nodes_path),
                tolbert_label_map_path=None if label_map_path is None else str(label_map_path),
                tolbert_source_spans_paths=tuple(
                    str(item)
                    for item in runtime_paths.get("source_spans_paths", candidate.tolbert_source_spans_paths)
                ),
                tolbert_cache_paths=tuple(
                    str(item)
                    for item in runtime_paths.get("cache_paths", candidate.tolbert_cache_paths)
                ),
            )
    if spec.base_subsystem == "qwen_adapter" and artifact_path.exists():
        payload = _load_artifact_payload(artifact_path)
        runtime_paths = payload.get("runtime_paths", {}) if isinstance(payload, dict) else {}
        if isinstance(runtime_paths, dict):
            configured_model_name = str(
                runtime_paths.get("served_model_name")
                or payload.get("base_model_name")
                or runtime_paths.get("merged_output_dir")
                or runtime_paths.get("adapter_output_dir")
                or candidate.model_name
            ).strip()
            configured_vllm_host = str(runtime_paths.get("vllm_host", candidate.vllm_host)).strip() or candidate.vllm_host
            configured_provider = str(runtime_paths.get("provider", candidate.provider)).strip() or candidate.provider
            candidate = replace(
                candidate,
                model_name=configured_model_name or candidate.model_name,
                vllm_host=configured_vllm_host,
                provider=configured_provider,
                qwen_adapter_artifact_path=artifact_path,
            )
    if spec.proposal_toggle_attr:
        return replace(candidate, **{spec.proposal_toggle_attr: True})
    return candidate


def _comparison_universe_config(
    config: KernelConfig,
    subsystem: str,
    artifact_path: Path,
) -> KernelConfig:
    if subsystem not in {"universe", "universe_constitution", "operating_envelope"}:
        return config_with_subsystem_artifact_path(config, subsystem, artifact_path)
    payload = _load_artifact_payload(artifact_path)
    if subsystem == "universe" or not payload:
        return _comparison_config_from_universe_contract(config, artifact_path, payload)
    if subsystem == "universe_constitution":
        envelope_payload = _comparison_companion_universe_payload(
            config,
            subsystem="operating_envelope",
            artifact_path=artifact_path,
        )
        bundle = compose_universe_bundle_payloads(
            constitution_payload=payload,
            operating_envelope_payload=envelope_payload,
            baseline_payload=envelope_payload or payload,
            lifecycle_state=universe_bundle_lifecycle_state(payload, envelope_payload),
        )
        return _write_universe_comparison_bundle(
            config,
            artifact_path=artifact_path,
            bundle=bundle,
        )
    constitution_payload = _comparison_companion_universe_payload(
        config,
        subsystem="universe_constitution",
        artifact_path=artifact_path,
    )
    bundle = compose_universe_bundle_payloads(
        constitution_payload=constitution_payload,
        operating_envelope_payload=payload,
        baseline_payload=constitution_payload or payload,
        lifecycle_state=universe_bundle_lifecycle_state(payload, constitution_payload),
    )
    return _write_universe_comparison_bundle(config, artifact_path=artifact_path, bundle=bundle)


def _comparison_config_from_universe_contract(
    config: KernelConfig,
    artifact_path: Path,
    payload: dict[str, object],
) -> KernelConfig:
    if not payload:
        shadow_root = artifact_path.parent / ".comparison_artifacts"
        return replace(
            config,
            universe_contract_path=artifact_path,
            universe_constitution_path=shadow_root / f"{artifact_path.stem}.universe_constitution.json",
            operating_envelope_path=shadow_root / f"{artifact_path.stem}.operating_envelope.json",
        )
    bundle = compose_universe_bundle_payloads(
        constitution_payload=payload,
        operating_envelope_payload=payload,
        baseline_payload=payload,
        lifecycle_state=universe_bundle_lifecycle_state(payload),
    )
    return _write_universe_comparison_bundle(config, artifact_path=artifact_path, bundle=bundle)


def _comparison_companion_universe_payload(
    config: KernelConfig,
    *,
    subsystem: str,
    artifact_path: Path,
) -> dict[str, object]:
    sibling = artifact_path.parent / active_artifact_path_for_subsystem(config, subsystem).name
    for path in (
        sibling,
        active_artifact_path_for_subsystem(config, subsystem),
        config.universe_contract_path,
    ):
        payload = _load_artifact_payload(path)
        if payload:
            return payload
    return {}


def _load_artifact_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_universe_comparison_bundle(
    config: KernelConfig,
    *,
    artifact_path: Path,
    bundle: dict[str, dict[str, object]],
) -> KernelConfig:
    shadow_paths = comparison_shadow_universe_bundle_paths(artifact_path)
    write_universe_bundle_files(
        contract_path=shadow_paths["universe"],
        constitution_path=shadow_paths["universe_constitution"],
        operating_envelope_path=shadow_paths["operating_envelope"],
        bundle=bundle,
    )
    return replace(
        config,
        universe_contract_path=shadow_paths["universe"],
        universe_constitution_path=shadow_paths["universe_constitution"],
        operating_envelope_path=shadow_paths["operating_envelope"],
    )


def default_variant_definitions(
    subsystem: str,
    experiment,
    metrics,
    *,
    capability_modules_path: Path | None = None,
) -> list[dict[str, object]]:
    resolved = resolved_subsystem_spec(subsystem, capability_modules_path)
    resolved_subsystem = resolved.subsystem
    subsystem = resolved.base_subsystem
    if subsystem == "benchmark":
        return [
            _variant_definition("failure_cluster_growth", "expand benchmark coverage from clustered failures", 0.03, 3, {"focus": "confidence"}),
            _variant_definition("environment_pattern_growth", "expand environment-pattern benchmark breadth", 0.025, 3, {"focus": "breadth"}),
        ]
    if subsystem == "retrieval":
        return [
            _variant_definition("confidence_gating", "tighten trust and confidence thresholds", 0.03, 2, {"focus": "confidence"}),
            _variant_definition("breadth_rebalance", "rebalance branch/global retrieval breadth", 0.025, 3, {"focus": "breadth"}),
        ]
    if subsystem == "tolbert_model":
        return [
            _variant_definition("recovery_alignment", "fine-tune Tolbert against low-confidence and failure-aligned traces", 0.03, 4, {"focus": "recovery_alignment"}),
            _variant_definition("discovered_task_adaptation", "fine-tune Tolbert against discovered tasks and verifier-backed contracts", 0.028, 4, {"focus": "discovered_task_adaptation"}),
        ]
    if subsystem == "qwen_adapter":
        return [
            _variant_definition("coding_lane_sft", "adapt Qwen on retained kernel coding traces without changing liftoff ownership", 0.025, 4, {"focus": "coding_lane_sft"}),
            _variant_definition("teacher_shadow", "adapt Qwen for teacher and shadow-runtime roles on approved coding families", 0.022, 4, {"focus": "teacher_shadow"}),
        ]
    if subsystem == "verifier":
        return [
            _variant_definition("false_failure_guard", "tighten false-failure controls from failed traces", 0.03, 3, {"focus": "false_failure"}),
            _variant_definition("strict_contract_growth", "increase verifier strictness with successful traces", 0.025, 2, {"focus": "strictness"}),
        ]
    if subsystem == "tooling":
        command_failures = int(experiment.evidence.get("command_failure_count", 0))
        return [
            _variant_definition("procedure_promotion", "promote compact reusable procedures from failures", 0.02, 2, {"min_failures": command_failures}),
            _variant_definition("script_hardening", "emit stricter replay-verified shell procedures", 0.018, 3, {"focus": "replay"}),
        ]
    if subsystem == "skills":
        return [
            _variant_definition("task_specific_replay", "improve task-specific replay skill coverage", 0.02, 2, {"scope": "task_specific"}),
            _variant_definition("cross_task_transfer", "favor skills with stronger transfer potential", 0.018, 3, {"scope": "transfer"}),
        ]
    if subsystem == "operators":
        return [
            _variant_definition("single_family_operator", "induce compact operator classes within a family", 0.03, 3, {"min_support": 2}),
            _variant_definition("cross_family_operator", "induce broader operator classes across related tasks", 0.02, 4, {"min_support": 3}),
        ]
    if subsystem == "curriculum":
        weakest_family = ""
        if metrics.generated_by_benchmark_family:
            weakest_family = min(
                metrics.generated_by_benchmark_family,
                key=lambda family: 0.0
                if metrics.generated_by_benchmark_family[family] == 0
                else metrics.generated_passed_by_benchmark_family.get(family, 0)
                / metrics.generated_by_benchmark_family[family],
            )
        return [
            _variant_definition("failure_recovery_specificity", "raise failure-recovery task specificity", 0.03, 3, {"focus": "failure_recovery"}),
            _variant_definition("weakest_family_expansion", "expand the weakest generated family", 0.02, 2, {"family": weakest_family}),
        ]
    if subsystem == "policy":
        return [
            _variant_definition("retrieval_caution", "make low-confidence retrieval less binding", 0.015, 2, {"focus": "retrieval_caution"}),
            _variant_definition("verifier_alignment", "bias decisions toward verifier-compatible artifact checks", 0.012, 2, {"focus": "verifier_alignment"}),
            _variant_definition(
                "long_horizon_orientation",
                "bias coding decisions toward durable progress, preservation, and validated stops",
                0.014,
                3,
                {"focus": "long_horizon_success"},
            ),
        ]
    if resolved_subsystem == "universe_constitution":
        return [
            _variant_definition("governance_hardening", "tighten destructive-command and bounded-action constitutional rules", 0.018, 2, {"focus": "governance"}),
            _variant_definition("verification_bias", "raise constitutional preference for verifier-aligned execution", 0.016, 2, {"focus": "verification"}),
        ]
    if resolved_subsystem == "operating_envelope":
        return [
            _variant_definition("environment_envelope", "calibrate retained operating-envelope assumptions to repeated runtime evidence", 0.017, 2, {"focus": "environment_envelope"}),
            _variant_definition("operator_scope", "refine operating-envelope scope boundaries above task-local heuristics", 0.014, 2, {"focus": "operator_scope"}),
        ]
    if subsystem == "universe":
        return [
            _variant_definition("governance_hardening", "tighten destructive-command and bounded-action governance", 0.018, 2, {"focus": "governance"}),
            _variant_definition("verification_bias", "raise stable preference for verification-aligned commands", 0.016, 2, {"focus": "verification"}),
            _variant_definition("operator_scope", "refine stable command-scope boundaries above task-local heuristics", 0.014, 2, {"focus": "operator_scope"}),
            _variant_definition("environment_envelope", "calibrate retained environment-envelope assumptions to repeated runtime evidence", 0.017, 2, {"focus": "environment_envelope"}),
        ]
    if subsystem == "world_model":
        return [
            _variant_definition("workflow_alignment", "bias world-model scoring toward workflow paths and reports", 0.02, 2, {"focus": "workflow_alignment"}),
            _variant_definition("preservation_bias", "front-load preserved-path protection in planning and scoring", 0.018, 2, {"focus": "preservation_bias"}),
        ]
    if subsystem == "state_estimation":
        return [
            _variant_definition("transition_normalization", "normalize transition summaries to capture subtle stalls and regressions", 0.02, 2, {"focus": "transition_normalization"}),
            _variant_definition("risk_sensitivity", "tighten latent-state risk bands around regressive and blocked states", 0.02, 2, {"focus": "risk_sensitivity"}),
            _variant_definition("recovery_bias", "bias policy scoring more strongly toward remediation after regressive transitions", 0.018, 2, {"focus": "recovery_bias"}),
        ]
    if subsystem == "trust":
        return [
            _variant_definition("safety_tightening", "tighten unattended trust thresholds around ambiguous or hidden-risk outcomes", 0.02, 2, {"focus": "safety"}),
            _variant_definition("breadth_guard", "expand unattended trust breadth requirements across benchmark families", 0.018, 2, {"focus": "breadth"}),
        ]
    if subsystem == "recovery":
        return [
            _variant_definition("rollback_safety", "expand rollback coverage and verify clean restores after non-success outcomes", 0.02, 2, {"focus": "rollback_safety"}),
            _variant_definition("snapshot_coverage", "preserve stronger snapshot and restore discipline for unattended runs", 0.018, 2, {"focus": "snapshot_coverage"}),
        ]
    if subsystem == "delegation":
        return [
            _variant_definition("throughput_balance", "increase delegated concurrency and active campaign breadth", 0.02, 2, {"focus": "throughput_balance"}),
            _variant_definition("queue_elasticity", "expand delegated queue and artifact budgets for parallel campaigns", 0.018, 2, {"focus": "queue_elasticity"}),
            _variant_definition("worker_depth", "increase delegated step, subprocess, and timeout depth", 0.018, 2, {"focus": "worker_depth"}),
        ]
    if subsystem == "operator_policy":
        return [
            _variant_definition("family_breadth", "expand unattended benchmark-family coverage", 0.02, 2, {"focus": "family_breadth"}),
            _variant_definition("git_http_scope", "enable bounded unattended git and http execution", 0.018, 2, {"focus": "git_http_scope"}),
            _variant_definition("generated_path_scope", "enable bounded generated-path mutation policy", 0.018, 2, {"focus": "generated_path_scope"}),
        ]
    if subsystem == "transition_model":
        return [
            _variant_definition("repeat_avoidance", "penalize repeated stalled commands more aggressively", 0.02, 2, {"focus": "repeat_avoidance"}),
            _variant_definition("regression_guard", "penalize commands that match retained regression signatures", 0.02, 2, {"focus": "regression_guard"}),
            _variant_definition("recovery_bias", "reward recovery actions after stalled or regressive transitions", 0.018, 2, {"focus": "recovery_bias"}),
        ]
    if subsystem == "capabilities":
        return [
            _variant_definition("policy_surface", "bootstrap policy-level improvement surfaces for enabled modules", 0.02, 2, {"focus": "policy_surface"}),
            _variant_definition("tooling_surface", "bootstrap tooling-level improvement surfaces for enabled modules", 0.02, 2, {"focus": "tooling_surface"}),
        ]
    return [_variant_definition("default", "default bounded experiment", 0.01, 1, {"default": True})]


def generate_candidate_artifact(
    *,
    config: KernelConfig,
    planner,
    subsystem: str,
    metrics,
    generation_kwargs: dict[str, object],
    candidate_artifact_path: Path,
    progress=None,
) -> tuple[str, str, str]:
    spec = resolved_subsystem_spec(subsystem, config.capability_modules_path)
    artifact = ""
    current_payload = _active_artifact_payload(config, subsystem)
    if spec.generator_kind == "skills":
        artifact = str(extract_successful_command_skills(config.trajectories_root, candidate_artifact_path, **generation_kwargs))
    elif spec.generator_kind == "tooling":
        artifact = str(extract_tool_candidates(config.trajectories_root, candidate_artifact_path, **generation_kwargs))
    elif spec.generator_kind == "operators":
        artifact = str(extract_operator_classes(config.trajectories_root, candidate_artifact_path, **generation_kwargs))
    elif spec.generator_kind == "verifier":
        artifact = str(synthesize_verifier_contracts(config.trajectories_root, candidate_artifact_path, **generation_kwargs))
    elif spec.generator_kind == "benchmark":
        artifact = str(synthesize_benchmark_candidates(config.trajectories_root, candidate_artifact_path, **generation_kwargs))
    elif spec.generator_kind == "retrieval":
        proposals = build_retrieval_proposal_artifact(metrics, current_payload=current_payload, **generation_kwargs)
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "tolbert_model":
        candidate_payload = build_tolbert_model_candidate_artifact(
            config=config,
            repo_root=Path(__file__).resolve().parents[1],
            output_dir=candidate_artifact_path.parent / candidate_artifact_path.stem,
            metrics=metrics,
            current_payload=current_payload,
            progress=progress,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
        candidate_payload["storage_governance"] = cleanup_tolbert_model_candidate_storage(
            config=config,
            preserve_paths=(candidate_artifact_path.parent,),
        )
        candidate_artifact_path.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "qwen_adapter":
        candidate_payload = build_qwen_adapter_candidate_artifact(
            config=config,
            repo_root=Path(__file__).resolve().parents[1],
            output_dir=candidate_artifact_path.parent / candidate_artifact_path.stem,
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "policy":
        proposals = build_policy_proposal_artifact(
            metrics,
            planner.failure_counts(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "universe":
        proposals = build_universe_contract_artifact(
            metrics,
            planner.failure_counts(),
            environment_violation_summary=planner.environment_violation_summary(),
            cycle_feedback_summary=planner.universe_cycle_feedback_summary(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "universe_constitution":
        proposals = build_universe_constitution_artifact(
            metrics,
            planner.failure_counts(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "operating_envelope":
        proposals = build_operating_envelope_artifact(
            metrics,
            planner.failure_counts(),
            environment_violation_summary=planner.environment_violation_summary(),
            cycle_feedback_summary=planner.universe_cycle_feedback_summary(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "curriculum":
        proposals = build_curriculum_proposal_artifact(
            metrics,
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "world_model":
        proposals = build_world_model_proposal_artifact(
            metrics,
            planner.failure_counts(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "state_estimation":
        proposals = build_state_estimation_proposal_artifact(
            metrics,
            planner.transition_summary(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "trust":
        proposals = build_trust_proposal_artifact(
            metrics,
            planner.trust_ledger_payload(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "recovery":
        proposals = build_recovery_proposal_artifact(
            metrics,
            planner.trust_ledger_payload(),
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "delegation":
        proposals = build_delegation_proposal_artifact(config, **generation_kwargs)
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "operator_policy":
        proposals = build_operator_policy_proposal_artifact(config, **generation_kwargs)
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "transition_model":
        proposals = build_transition_model_proposal_artifact(
            config.trajectories_root,
            current_payload=current_payload,
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    elif spec.generator_kind == "capabilities":
        proposals = build_capability_module_artifact(
            config,
            metrics,
            planner.failure_counts(),
            **generation_kwargs,
        )
        candidate_artifact_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        artifact = str(candidate_artifact_path)
    if artifact:
        govern_improvement_export_storage(
            config,
            preserve_paths=(Path(artifact),),
            include_cycle_exports=False,
            include_report_exports=False,
            include_run_reports=False,
            include_run_checkpoints=False,
        )
    return artifact, spec.action, spec.artifact_kind


def _active_artifact_payload(config: KernelConfig, subsystem: str) -> object | None:
    if subsystem == "universe":
        return _active_universe_payload(config)
    path = active_artifact_path_for_subsystem(config, subsystem)
    if not path.exists():
        if subsystem in {"universe_constitution", "operating_envelope"} and config.universe_contract_path.exists():
            return _load_artifact_payload(config.universe_contract_path) or None
        return None
    return _load_artifact_payload(path) or None


def _active_universe_payload(config: KernelConfig) -> dict[str, object] | None:
    contract_payload = _load_artifact_payload(config.universe_contract_path)
    constitution_payload = _load_artifact_payload(config.universe_constitution_path)
    operating_envelope_payload = _load_artifact_payload(config.operating_envelope_path)
    if constitution_payload or operating_envelope_payload:
        return compose_universe_bundle_payloads(
            constitution_payload=constitution_payload or contract_payload,
            operating_envelope_payload=operating_envelope_payload or contract_payload,
            baseline_payload=contract_payload or constitution_payload or operating_envelope_payload,
            lifecycle_state=universe_bundle_lifecycle_state(
                constitution_payload,
                operating_envelope_payload,
                contract_payload,
            ),
        )["universe"]
    if contract_payload:
        return contract_payload
    return None


def external_subsystem_specs(capability_modules_path: Path | None) -> dict[str, SubsystemSpec]:
    specs: dict[str, SubsystemSpec] = {}
    for module in load_capability_modules(capability_modules_path) if capability_modules_path is not None else []:
        if not bool(module.get("enabled", False)) or not bool(module.get("valid", False)):
            continue
        module_settings = module.get("settings", {})
        if not isinstance(module_settings, dict):
            continue
        subsystem_defs = module_settings.get("improvement_subsystems", [])
        if not isinstance(subsystem_defs, list):
            continue
        for item in subsystem_defs:
            if not isinstance(item, dict):
                continue
            subsystem_id = str(item.get("subsystem_id", "")).strip()
            base_subsystem = str(item.get("base_subsystem", "")).strip()
            if not subsystem_id or not base_subsystem or subsystem_id in _BUILTIN_SPECS:
                continue
            base_spec = specs.get(base_subsystem) or _BUILTIN_SPECS.get(base_subsystem)
            if base_spec is None:
                continue
            baseline_updates = dict(base_spec.baseline_flag_updates)
            if isinstance(item.get("baseline_flag_updates"), dict):
                baseline_updates.update(
                    {str(key): bool(value) for key, value in item["baseline_flag_updates"].items()}
                )
            candidate_updates = dict(base_spec.candidate_flag_updates)
            if isinstance(item.get("candidate_flag_updates"), dict):
                candidate_updates.update(
                    {str(key): bool(value) for key, value in item["candidate_flag_updates"].items()}
                )
            artifact_path_attr = str(item.get("artifact_path_attr", base_spec.artifact_path_attr)).strip()
            if artifact_path_attr not in _CONFIG_FIELD_NAMES:
                artifact_path_attr = base_spec.artifact_path_attr
            proposal_toggle_attr = str(item.get("proposal_toggle_attr", "")).strip()
            if proposal_toggle_attr and proposal_toggle_attr not in _CONFIG_FIELD_NAMES:
                proposal_toggle_attr = ""
            capability_tags: list[str] = list(base_spec.capability_tags)
            seen_capability_tags = set(capability_tags)
            for raw_tag in item.get("capability_tags", []):
                tag = str(raw_tag).strip()
                if tag and tag not in seen_capability_tags:
                    seen_capability_tags.add(tag)
                    capability_tags.append(tag)
            strategy_hooks = dict(base_spec.strategy_hooks)
            if isinstance(item.get("strategy_hooks"), dict):
                strategy_hooks.update(
                    {
                        str(key): str(value).strip()
                        for key, value in item["strategy_hooks"].items()
                        if str(key).strip() and str(value).strip()
                    }
                )
            specs[subsystem_id] = SubsystemSpec(
                subsystem=subsystem_id,
                base_subsystem=base_subsystem,
                artifact_path_attr=artifact_path_attr or base_spec.artifact_path_attr,
                proposal_toggle_attr=proposal_toggle_attr or base_spec.proposal_toggle_attr,
                generator_kind=base_spec.generator_kind,
                artifact_kind=base_spec.artifact_kind,
                action=str(item.get("action", base_spec.action)).strip() or base_spec.action,
                baseline_flag_updates=baseline_updates,
                candidate_flag_updates=candidate_updates,
                capability_tags=tuple(capability_tags),
                strategy_hooks=strategy_hooks,
            )
    return specs


def external_planner_experiments(capability_modules_path: Path | None) -> list[dict[str, object]]:
    specs = external_subsystem_specs(capability_modules_path)
    experiments: list[dict[str, object]] = []
    for module in load_capability_modules(capability_modules_path) if capability_modules_path is not None else []:
        if not bool(module.get("enabled", False)) or not bool(module.get("valid", False)):
            continue
        module_id = str(module.get("module_id", "")).strip()
        module_settings = module.get("settings", {})
        if not isinstance(module_settings, dict):
            continue
        for item in module_settings.get("improvement_subsystems", []):
            if not isinstance(item, dict):
                continue
            subsystem_id = str(item.get("subsystem_id", "")).strip()
            if not subsystem_id or subsystem_id not in specs:
                continue
            base_subsystem = specs[subsystem_id].base_subsystem
            experiments.append(
                {
                    "subsystem": subsystem_id,
                    "reason": str(item.get("reason", "")).strip()
                    or f"enabled module {module_id or subsystem_id} exposes an autonomous improvement surface",
                    "priority": int(item.get("priority", 3)),
                    "expected_gain": float(item.get("expected_gain", 0.01)),
                    "estimated_cost": int(item.get("estimated_cost", 2)),
                    "evidence": {
                        "external_subsystem": True,
                        "module_id": module_id,
                        "base_subsystem": base_subsystem,
                    },
                }
            )
    return experiments


def _variant_definition(
    variant_id: str,
    description: str,
    expected_gain: float,
    estimated_cost: int,
    controls: dict[str, object],
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "description": description,
        "expected_gain": expected_gain,
        "estimated_cost": estimated_cost,
        "controls": controls,
    }
