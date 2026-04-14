from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from evals.metrics import EvalMetrics

from ...config import KernelConfig
from ...modeling.evaluation.liftoff import build_liftoff_gate_report
from ...modeling.tolbert.delta import materialize_tolbert_checkpoint_from_delta, resolve_tolbert_runtime_checkpoint_path
from ...ops.runtime_supervision import atomic_write_json
from ...strategy_memory import load_strategy_nodes, summarize_strategy_priors
from ..strategy.subsystems import (
    active_artifact_path_for_subsystem,
    base_subsystem_for,
    default_variant_definitions,
    external_planner_experiments,
)
from ..tolbert_assets import materialize_retained_retrieval_asset_bundle
from .universe_improvement import (
    compose_universe_bundle_payloads,
    retained_universe_action_risk_controls,
    retained_universe_environment_assumptions,
    retained_universe_forbidden_command_patterns,
    retained_universe_governance,
    retained_universe_invariants,
    retained_universe_preferred_command_prefixes,
    sibling_universe_bundle_paths,
    universe_bundle_contains_path,
    universe_bundle_paths,
    write_universe_bundle_files,
)

from ...improvement_engine import ImprovementExperiment, RetentionApplyContext, empty_strategy_memory_summary


def _resolve_cycle_runner_override(name: str, fallback: Any) -> Any:
    cycle_runner_module = sys.modules.get("agent_kernel.cycle_runner")
    if cycle_runner_module is None:
        return fallback
    return getattr(cycle_runner_module, name, fallback)


@dataclass(slots=True)
class ImprovementRetentionPlugin:
    evidence_builder: Any = None
    evaluator: Any = None
    post_apply_hook: Any = None


class StrategyPriorStore:
    def summarize(
        self,
        *,
        runtime_config: KernelConfig | None,
        candidate_subsystem: str,
        strategy_candidate: dict[str, object] | None,
    ) -> dict[str, object]:
        if runtime_config is None:
            return empty_strategy_memory_summary()
        strategy = strategy_candidate if isinstance(strategy_candidate, dict) else {}
        nodes = load_strategy_nodes(runtime_config)
        return summarize_strategy_priors(
            nodes,
            subsystem=str(candidate_subsystem).strip(),
            strategy_candidate_id=str(strategy.get("strategy_candidate_id", "")).strip(),
            semantic_hypotheses=[
                str(value).strip()
                for value in list(strategy.get("semantic_hypotheses", []) or [])
                if str(value).strip()
            ],
            context_terms=[
                str(strategy.get("strategy_candidate_kind", "")).strip(),
                str(strategy.get("strategy_label", "")).strip(),
                str(strategy.get("rationale", "")).strip(),
                str(strategy.get("target_subsystem", "")).strip(),
                dict(strategy.get("generation_basis", {}))
                if isinstance(strategy.get("generation_basis", {}), dict)
                else {},
                dict(strategy.get("target_conditions", {}))
                if isinstance(strategy.get("target_conditions", {}), dict)
                else {},
                list(strategy.get("expected_signals", []) or []),
            ],
        )


class ImprovementPluginLayer:
    def __init__(self) -> None:
        self._retention_plugins: dict[str, ImprovementRetentionPlugin] = {}

    def base_subsystem(self, subsystem: str, capability_modules_path: Path | None = None) -> str:
        return base_subsystem_for(subsystem, capability_modules_path)

    def active_artifact_path(self, config: KernelConfig, subsystem: str) -> Path:
        return active_artifact_path_for_subsystem(config, subsystem)

    def default_variants(
        self,
        subsystem: str,
        experiment: ImprovementExperiment,
        metrics: EvalMetrics,
        *,
        capability_modules_path: Path | None = None,
    ) -> list[dict[str, object]]:
        return default_variant_definitions(
            subsystem,
            experiment,
            metrics,
            capability_modules_path=capability_modules_path,
        )

    def external_experiments(self, capability_modules_path: Path | None = None) -> list[dict[str, object]]:
        return external_planner_experiments(capability_modules_path)

    def register_retention_plugin(
        self,
        subsystem: str,
        *,
        evidence_builder: Any = None,
        evaluator: Any = None,
        post_apply_hook: Any = None,
    ) -> None:
        normalized = str(subsystem).strip()
        if not normalized:
            return
        existing = self._retention_plugins.get(normalized)
        if existing is None:
            existing = ImprovementRetentionPlugin()
        if evidence_builder is not None:
            existing.evidence_builder = evidence_builder
        if evaluator is not None:
            existing.evaluator = evaluator
        if post_apply_hook is not None:
            existing.post_apply_hook = post_apply_hook
        self._retention_plugins[normalized] = existing

    def retention_plugin(
        self,
        subsystem: str,
        *,
        capability_modules_path: Path | None = None,
    ) -> ImprovementRetentionPlugin:
        normalized = self.base_subsystem(subsystem, capability_modules_path)
        return self._retention_plugins.get(normalized, ImprovementRetentionPlugin())

    def materialize_tolbert_checkpoint_from_delta(
        self,
        *,
        parent_checkpoint_path: Path,
        delta_checkpoint_path: Path,
        output_checkpoint_path: Path,
    ) -> Path:
        return materialize_tolbert_checkpoint_from_delta(
            parent_checkpoint_path=parent_checkpoint_path,
            delta_checkpoint_path=delta_checkpoint_path,
            output_checkpoint_path=output_checkpoint_path,
        )

    def resolve_tolbert_runtime_checkpoint_path(
        self,
        runtime_paths: dict[str, Any],
        *,
        artifact_path: Path | None = None,
    ) -> str | None:
        return resolve_tolbert_runtime_checkpoint_path(runtime_paths, artifact_path=artifact_path)

    def compose_universe_bundle_payloads(
        self,
        *,
        constitution_payload: dict[str, object] | None,
        operating_envelope_payload: dict[str, object] | None,
        baseline_payload: dict[str, object] | None,
        lifecycle_state: str,
    ) -> dict[str, object]:
        return compose_universe_bundle_payloads(
            constitution_payload=constitution_payload,
            operating_envelope_payload=operating_envelope_payload,
            baseline_payload=baseline_payload,
            lifecycle_state=lifecycle_state,
        )

    def retained_universe_governance(self, payload: dict[str, object] | None) -> dict[str, object]:
        return retained_universe_governance(payload)

    def retained_universe_invariants(self, payload: dict[str, object] | None) -> list[object]:
        return retained_universe_invariants(payload)

    def retained_universe_forbidden_command_patterns(
        self,
        payload: dict[str, object] | None,
    ) -> list[object]:
        return retained_universe_forbidden_command_patterns(payload)

    def retained_universe_preferred_command_prefixes(
        self,
        payload: dict[str, object] | None,
    ) -> list[object]:
        return retained_universe_preferred_command_prefixes(payload)

    def retained_universe_action_risk_controls(
        self,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        return retained_universe_action_risk_controls(payload)

    def retained_universe_environment_assumptions(
        self,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        return retained_universe_environment_assumptions(payload)

    def sibling_universe_bundle_paths(self, live_artifact_path: Path) -> dict[str, Path]:
        return sibling_universe_bundle_paths(live_artifact_path)

    def universe_bundle_paths(
        self,
        *,
        universe_contract_path: Path,
        universe_constitution_path: Path,
        operating_envelope_path: Path,
    ) -> dict[str, Path]:
        return universe_bundle_paths(
            universe_contract_path=universe_contract_path,
            universe_constitution_path=universe_constitution_path,
            operating_envelope_path=operating_envelope_path,
        )

    def universe_bundle_contains_path(self, bundle_paths: dict[str, Path], path: Path) -> bool:
        return universe_bundle_contains_path(bundle_paths, path)

    def write_universe_bundle_files(
        self,
        *,
        contract_path: Path,
        constitution_path: Path,
        operating_envelope_path: Path,
        bundle: dict[str, object],
    ) -> dict[str, str]:
        return write_universe_bundle_files(
            contract_path=contract_path,
            constitution_path=constitution_path,
            operating_envelope_path=operating_envelope_path,
            bundle=bundle,
        )


def _retrieval_post_apply_hook(context: RetentionApplyContext) -> list[dict[str, object]]:
    if context.decision_state != "retain" or context.runtime_config is None or context.repo_root is None:
        return []
    materialize_bundle = _resolve_cycle_runner_override(
        "materialize_retained_retrieval_asset_bundle",
        materialize_retained_retrieval_asset_bundle,
    )
    bundle_manifest_path = materialize_bundle(
        repo_root=context.repo_root,
        config=context.runtime_config,
        cycle_id=context.cycle_id,
    )
    return [
        {
            "action": "materialize_retrieval_asset_bundle",
            "artifact_path": str(bundle_manifest_path),
            "artifact_kind": "tolbert_retrieval_asset_bundle",
            "reason": "materialized retained retrieval controls into a Tolbert runtime bundle",
            "metrics_summary": {
                "baseline_pass_rate": context.baseline_metrics.pass_rate,
                "candidate_pass_rate": context.candidate_metrics.pass_rate,
                "decision_pass_rate_delta": context.candidate_metrics.pass_rate - context.baseline_metrics.pass_rate,
            },
        }
    ]


def _tolbert_model_post_apply_hook(context: RetentionApplyContext) -> list[dict[str, object]]:
    if context.decision_state != "retain" or context.runtime_config is None:
        return []
    payload = {}
    if context.active_artifact_path.exists():
        try:
            payload = json.loads(context.active_artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    build_report = _resolve_cycle_runner_override(
        "build_liftoff_gate_report",
        build_liftoff_gate_report,
    )
    liftoff_report = build_report(
        candidate_metrics=context.candidate_metrics,
        baseline_metrics=context.baseline_metrics,
        artifact_payload=payload,
    )
    atomic_write_json(
        context.runtime_config.tolbert_liftoff_report_path,
        {
            "spec_version": "asi_v1",
            "artifact_kind": "liftoff_gate_report",
            "cycle_id": context.cycle_id,
            "subsystem": context.subsystem,
            "report": liftoff_report.to_dict(),
        },
        config=context.runtime_config,
    )
    if isinstance(payload, dict):
        runtime_policy = payload.get("runtime_policy", {})
        if not isinstance(runtime_policy, dict):
            runtime_policy = {}
        if liftoff_report.state == "retain":
            runtime_policy["primary_benchmark_families"] = list(liftoff_report.primary_takeover_families)
            runtime_policy["shadow_benchmark_families"] = list(liftoff_report.shadow_only_families)
        else:
            runtime_policy["shadow_benchmark_families"] = sorted(
                {
                    *[
                        str(value).strip()
                        for value in runtime_policy.get("shadow_benchmark_families", [])
                        if str(value).strip()
                    ],
                    *liftoff_report.primary_takeover_families,
                    *liftoff_report.shadow_only_families,
                }
            )
        payload["runtime_policy"] = runtime_policy
        atomic_write_json(
            context.active_artifact_path,
            payload,
            config=context.runtime_config,
        )
    return [
        {
            "action": "write_tolbert_liftoff_gate_report",
            "artifact_path": str(context.runtime_config.tolbert_liftoff_report_path),
            "artifact_kind": "liftoff_gate_report",
            "reason": liftoff_report.reason,
            "metrics_summary": {
                **liftoff_report.to_dict(),
            },
        }
    ]


DEFAULT_STRATEGY_PRIOR_STORE = StrategyPriorStore()
DEFAULT_IMPROVEMENT_PLUGIN_LAYER = ImprovementPluginLayer()
DEFAULT_IMPROVEMENT_PLUGIN_LAYER.register_retention_plugin(
    "retrieval",
    post_apply_hook=_retrieval_post_apply_hook,
)
DEFAULT_IMPROVEMENT_PLUGIN_LAYER.register_retention_plugin(
    "tolbert_model",
    post_apply_hook=_tolbert_model_post_apply_hook,
)


__all__ = [
    "DEFAULT_IMPROVEMENT_PLUGIN_LAYER",
    "DEFAULT_STRATEGY_PRIOR_STORE",
    "ImprovementPluginLayer",
    "ImprovementRetentionPlugin",
    "StrategyPriorStore",
]
