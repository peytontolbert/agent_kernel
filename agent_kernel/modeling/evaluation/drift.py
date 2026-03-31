from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Callable

from agent_kernel.config import KernelConfig
from agent_kernel.trust import build_unattended_trust_ledger
from evals.harness import run_eval, scoped_eval_config
from evals.metrics import EvalMetrics


@dataclass(slots=True)
class TakeoverDriftWave:
    wave_index: int
    baseline_estimated_steps: float
    candidate_estimated_steps: float
    cumulative_baseline_estimated_steps: float
    cumulative_candidate_estimated_steps: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    unsafe_ambiguous_rate_delta: float
    hidden_side_effect_rate_delta: float
    trust_success_rate_delta: float
    trust_unsafe_ambiguous_rate_delta: float
    baseline_trust_status: str
    candidate_trust_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TakeoverDriftReport:
    scope_label: str
    step_budget: int
    wave_task_limit: int
    max_waves: int
    waves_completed: int
    baseline_estimated_steps: float
    candidate_estimated_steps: float
    budget_reached: bool
    stalled: bool
    worst_pass_rate_delta: float = 0.0
    worst_unsafe_ambiguous_rate_delta: float = 0.0
    worst_hidden_side_effect_rate_delta: float = 0.0
    worst_trust_success_rate_delta: float = 0.0
    worst_trust_unsafe_ambiguous_rate_delta: float = 0.0
    final_pass_rate_delta: float = 0.0
    final_unsafe_ambiguous_rate_delta: float = 0.0
    final_hidden_side_effect_rate_delta: float = 0.0
    final_trust_success_rate_delta: float = 0.0
    final_trust_unsafe_ambiguous_rate_delta: float = 0.0
    baseline_final_trust_status: str = ""
    candidate_final_trust_status: str = ""
    waves: list[TakeoverDriftWave] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["waves"] = [wave.to_dict() for wave in self.waves]
        return payload


def run_takeover_drift_eval(
    *,
    config: KernelConfig,
    artifact_path: Path,
    step_budget: int,
    wave_task_limit: int = 64,
    max_waves: int = 16,
    eval_kwargs: dict[str, object] | None = None,
    run_eval_fn: Callable[..., EvalMetrics] = run_eval,
    build_trust_ledger_fn: Callable[[KernelConfig], dict[str, object]] = build_unattended_trust_ledger,
    scope_label: str | None = None,
) -> TakeoverDriftReport:
    normalized_budget = max(0, int(step_budget))
    normalized_wave_task_limit = max(0, int(wave_task_limit))
    normalized_max_waves = max(1, int(max_waves))
    normalized_scope = _normalized_scope_label(
        artifact_path=artifact_path,
        step_budget=normalized_budget,
        wave_task_limit=normalized_wave_task_limit,
        max_waves=normalized_max_waves,
        scope_label=scope_label,
    )
    baseline_config = scoped_eval_config(
        config,
        f"{normalized_scope}_baseline",
        use_tolbert_model_artifacts=False,
    )
    candidate_config = scoped_eval_config(
        config,
        f"{normalized_scope}_candidate",
        tolbert_model_artifact_path=artifact_path,
        use_tolbert_model_artifacts=True,
    )
    for scoped in (baseline_config, candidate_config):
        _reset_dir(scoped.workspace_root)
        _reset_dir(scoped.trajectories_root)
        _reset_dir(scoped.run_reports_dir)
        _reset_dir(scoped.unattended_workspace_snapshot_root)
    drift_kwargs = dict(eval_kwargs or {})
    if normalized_wave_task_limit > 0:
        drift_kwargs["task_limit"] = normalized_wave_task_limit
    elif "task_limit" not in drift_kwargs:
        drift_kwargs["task_limit"] = None
    waves: list[TakeoverDriftWave] = []
    baseline_total_steps = 0.0
    candidate_total_steps = 0.0
    worst_pass_rate_delta = float("inf")
    worst_unsafe_delta = float("-inf")
    worst_hidden_delta = float("-inf")
    worst_trust_success_delta = float("inf")
    worst_trust_unsafe_delta = float("-inf")
    baseline_final_status = ""
    candidate_final_status = ""
    final_pass_rate_delta = 0.0
    final_unsafe_delta = 0.0
    final_hidden_delta = 0.0
    final_trust_success_delta = 0.0
    final_trust_unsafe_delta = 0.0
    stalled = False
    for wave_index in range(1, normalized_max_waves + 1):
        baseline_metrics = run_eval_fn(
            config=baseline_config,
            progress_label=f"tolbert_takeover_drift_baseline_{wave_index}",
            **drift_kwargs,
        )
        candidate_metrics = run_eval_fn(
            config=candidate_config,
            progress_label=f"tolbert_takeover_drift_candidate_{wave_index}",
            **drift_kwargs,
        )
        baseline_ledger = build_trust_ledger_fn(baseline_config)
        candidate_ledger = build_trust_ledger_fn(candidate_config)
        baseline_wave_steps = max(0.0, baseline_metrics.average_steps * baseline_metrics.total)
        candidate_wave_steps = max(0.0, candidate_metrics.average_steps * candidate_metrics.total)
        baseline_total_steps += baseline_wave_steps
        candidate_total_steps += candidate_wave_steps
        baseline_trust_summary = _trust_summary(baseline_ledger)
        candidate_trust_summary = _trust_summary(candidate_ledger)
        baseline_final_status = _trust_status(baseline_ledger)
        candidate_final_status = _trust_status(candidate_ledger)
        wave = TakeoverDriftWave(
            wave_index=wave_index,
            baseline_estimated_steps=baseline_wave_steps,
            candidate_estimated_steps=candidate_wave_steps,
            cumulative_baseline_estimated_steps=baseline_total_steps,
            cumulative_candidate_estimated_steps=candidate_total_steps,
            baseline_pass_rate=baseline_metrics.pass_rate,
            candidate_pass_rate=candidate_metrics.pass_rate,
            pass_rate_delta=candidate_metrics.pass_rate - baseline_metrics.pass_rate,
            unsafe_ambiguous_rate_delta=candidate_metrics.unsafe_ambiguous_rate - baseline_metrics.unsafe_ambiguous_rate,
            hidden_side_effect_rate_delta=(
                candidate_metrics.hidden_side_effect_risk_rate - baseline_metrics.hidden_side_effect_risk_rate
            ),
            trust_success_rate_delta=(
                float(candidate_trust_summary.get("success_rate", 0.0))
                - float(baseline_trust_summary.get("success_rate", 0.0))
            ),
            trust_unsafe_ambiguous_rate_delta=(
                float(candidate_trust_summary.get("unsafe_ambiguous_rate", 0.0))
                - float(baseline_trust_summary.get("unsafe_ambiguous_rate", 0.0))
            ),
            baseline_trust_status=baseline_final_status,
            candidate_trust_status=candidate_final_status,
        )
        waves.append(wave)
        worst_pass_rate_delta = min(worst_pass_rate_delta, wave.pass_rate_delta)
        worst_unsafe_delta = max(worst_unsafe_delta, wave.unsafe_ambiguous_rate_delta)
        worst_hidden_delta = max(worst_hidden_delta, wave.hidden_side_effect_rate_delta)
        worst_trust_success_delta = min(worst_trust_success_delta, wave.trust_success_rate_delta)
        worst_trust_unsafe_delta = max(worst_trust_unsafe_delta, wave.trust_unsafe_ambiguous_rate_delta)
        final_pass_rate_delta = wave.pass_rate_delta
        final_unsafe_delta = wave.unsafe_ambiguous_rate_delta
        final_hidden_delta = wave.hidden_side_effect_rate_delta
        final_trust_success_delta = wave.trust_success_rate_delta
        final_trust_unsafe_delta = wave.trust_unsafe_ambiguous_rate_delta
        if baseline_wave_steps <= 0.0 and candidate_wave_steps <= 0.0:
            stalled = True
            break
        if normalized_budget > 0 and min(baseline_total_steps, candidate_total_steps) >= normalized_budget:
            break
    budget_reached = normalized_budget <= 0 or min(baseline_total_steps, candidate_total_steps) >= normalized_budget
    if not waves:
        worst_pass_rate_delta = 0.0
        worst_unsafe_delta = 0.0
        worst_hidden_delta = 0.0
        worst_trust_success_delta = 0.0
        worst_trust_unsafe_delta = 0.0
    return TakeoverDriftReport(
        scope_label=normalized_scope,
        step_budget=normalized_budget,
        wave_task_limit=normalized_wave_task_limit,
        max_waves=normalized_max_waves,
        waves_completed=len(waves),
        baseline_estimated_steps=baseline_total_steps,
        candidate_estimated_steps=candidate_total_steps,
        budget_reached=budget_reached,
        stalled=stalled,
        worst_pass_rate_delta=worst_pass_rate_delta,
        worst_unsafe_ambiguous_rate_delta=worst_unsafe_delta,
        worst_hidden_side_effect_rate_delta=worst_hidden_delta,
        worst_trust_success_rate_delta=worst_trust_success_delta,
        worst_trust_unsafe_ambiguous_rate_delta=worst_trust_unsafe_delta,
        final_pass_rate_delta=final_pass_rate_delta,
        final_unsafe_ambiguous_rate_delta=final_unsafe_delta,
        final_hidden_side_effect_rate_delta=final_hidden_delta,
        final_trust_success_rate_delta=final_trust_success_delta,
        final_trust_unsafe_ambiguous_rate_delta=final_trust_unsafe_delta,
        baseline_final_trust_status=baseline_final_status,
        candidate_final_trust_status=candidate_final_status,
        waves=waves,
    )


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _trust_status(ledger: dict[str, object]) -> str:
    assessment = ledger.get("overall_assessment", {}) if isinstance(ledger, dict) else {}
    if not isinstance(assessment, dict):
        return ""
    return str(assessment.get("status", "")).strip()


def _trust_summary(ledger: dict[str, object]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        return {}
    summary = ledger.get("gated_summary", {})
    return summary if isinstance(summary, dict) else {}


def _normalized_scope_label(
    *,
    artifact_path: Path,
    step_budget: int,
    wave_task_limit: int,
    max_waves: int,
    scope_label: str | None,
) -> str:
    if isinstance(scope_label, str) and scope_label.strip():
        return scope_label.strip()
    digest = hashlib.sha1(
        f"{artifact_path.resolve()}:{step_budget}:{wave_task_limit}:{max_waves}:{os.getpid()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"tolbert_takeover_drift_{digest}"
