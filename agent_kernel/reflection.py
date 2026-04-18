from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from evals.metrics import EvalMetrics

from .resource_registry import ResourceRegistry
from .resource_types import RESOURCE_KIND_ARTIFACT, subsystem_resource_id


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_string_list(values: list[object] | tuple[object, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def _top_count_labels(counts: object, *, limit: int = 3) -> tuple[str, ...]:
    if not isinstance(counts, dict):
        return ()
    ranked: list[tuple[float, str]] = []
    for label, value in counts.items():
        normalized = str(label).strip()
        if not normalized:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score <= 0.0:
            continue
        ranked.append((score, normalized))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(label for _, label in ranked[:limit])


def _strategy_basis_labels(strategy_candidate: dict[str, object]) -> tuple[str, ...]:
    basis = strategy_candidate.get("generation_basis", {})
    if not isinstance(basis, dict):
        return ()
    labels: list[object] = []
    for field in ("repeated_failure_motifs", "transfer_gaps", "repo_signals"):
        labels.extend(list(basis.get(field, []) or []))
    return _normalized_string_list(labels)


def _matched_observe_hypothesis(
    subsystem: str,
    observe_hypothesis: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(observe_hypothesis, dict):
        return None
    hypotheses = observe_hypothesis.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        return None
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        if str(item.get("subsystem", "")).strip() == subsystem:
            return item
    return None


def _metrics_digest(metrics: EvalMetrics) -> dict[str, object]:
    return {
        "total": int(metrics.total),
        "passed": int(metrics.passed),
        "pass_rate": round(float(metrics.pass_rate), 4),
        "generated_total": int(metrics.generated_total),
        "generated_passed": int(metrics.generated_passed),
        "generated_pass_rate": round(float(metrics.generated_pass_rate), 4),
        "low_confidence_episodes": int(metrics.low_confidence_episodes),
        "trusted_retrieval_steps": int(metrics.trusted_retrieval_steps),
    }


@dataclass(frozen=True, slots=True)
class CausalHypothesis:
    subsystem: str
    confidence: float
    rationale: str
    source: str
    failure_modes: tuple[str, ...] = ()
    evidence_labels: tuple[str, ...] = ()
    expected_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "subsystem": self.subsystem,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "source": self.source,
            "failure_modes": list(self.failure_modes),
            "evidence_labels": list(self.evidence_labels),
            "expected_signals": list(self.expected_signals),
        }


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    reflection_id: str
    cycle_id: str
    subsystem: str
    resource_id: str
    resource_kind: str
    summary: str
    hypothesis: CausalHypothesis
    metrics_digest: dict[str, object] = field(default_factory=dict)
    observe_status: str = ""
    observe_provider: str = ""
    active_version_id: str = ""
    active_resource_path: str = ""
    strategy_candidate_id: str = ""
    strategy_candidate_kind: str = ""
    strategy_origin: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "reflection_id": self.reflection_id,
            "cycle_id": self.cycle_id,
            "subsystem": self.subsystem,
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "summary": self.summary,
            "hypothesis": self.hypothesis.to_dict(),
            "metrics_digest": dict(self.metrics_digest),
            "observe_status": self.observe_status,
            "observe_provider": self.observe_provider,
            "active_version_id": self.active_version_id,
            "active_resource_path": self.active_resource_path,
            "strategy_candidate_id": self.strategy_candidate_id,
            "strategy_candidate_kind": self.strategy_candidate_kind,
            "strategy_origin": self.strategy_origin,
            "generated_at": self.generated_at,
        }


def build_reflection_record(
    *,
    cycle_id: str,
    target_subsystem: str,
    reason: str,
    metrics: EvalMetrics,
    evidence: dict[str, object] | None = None,
    observe_hypothesis: dict[str, object] | None = None,
    strategy_candidate: dict[str, object] | None = None,
    resource_registry: ResourceRegistry | None = None,
    generated_at: str | None = None,
) -> ReflectionRecord:
    subsystem = str(target_subsystem).strip()
    resource_id = subsystem_resource_id(subsystem)
    strategy = dict(strategy_candidate or {})
    evidence_payload = dict(evidence or {})
    active_version = None
    if resource_registry is not None and resource_registry.has(resource_id):
        active_version = resource_registry.resolve_active_version(resource_id)

    matched_hypothesis = _matched_observe_hypothesis(subsystem, observe_hypothesis)
    observe_payload = dict(observe_hypothesis or {})
    summary = str(observe_payload.get("summary", "")).strip() or str(reason).strip()
    rationale = str(reason).strip()
    confidence = 0.0
    source = "target_reason"
    if matched_hypothesis is not None:
        rationale = str(matched_hypothesis.get("rationale", "")).strip() or rationale
        try:
            confidence = max(0.0, min(1.0, float(matched_hypothesis.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        source = "observe_hypothesis"
    elif str(strategy.get("rationale", "")).strip():
        rationale = str(strategy.get("rationale", "")).strip()
        source = "strategy_candidate"

    failure_modes = _normalized_string_list(
        list(_top_count_labels(evidence_payload.get("transition_failure_counts", {})))
        + list(_top_count_labels(evidence_payload.get("failure_counts", {})))
    )
    evidence_labels = _normalized_string_list(
        list(_strategy_basis_labels(strategy))
        + list(dict(evidence_payload.get("portfolio", {})).get("reasons", []) or [])
    )
    expected_signals = _normalized_string_list(list(strategy.get("expected_signals", []) or []))

    return ReflectionRecord(
        reflection_id=f"reflect:{cycle_id}",
        cycle_id=cycle_id,
        subsystem=subsystem,
        resource_id=resource_id,
        resource_kind=RESOURCE_KIND_ARTIFACT,
        summary=summary,
        hypothesis=CausalHypothesis(
            subsystem=subsystem,
            confidence=confidence,
            rationale=rationale,
            source=source,
            failure_modes=failure_modes,
            evidence_labels=evidence_labels,
            expected_signals=expected_signals,
        ),
        metrics_digest=_metrics_digest(metrics),
        observe_status=str(observe_payload.get("status", "")).strip(),
        observe_provider=str(observe_payload.get("provider", "")).strip(),
        active_version_id="" if active_version is None else str(active_version.version_id).strip(),
        active_resource_path="" if active_version is None else str(active_version.path),
        strategy_candidate_id=str(strategy.get("strategy_candidate_id", strategy.get("strategy_id", ""))).strip(),
        strategy_candidate_kind=str(
            strategy.get("strategy_candidate_kind", strategy.get("strategy_kind", ""))
        ).strip(),
        strategy_origin=str(strategy.get("origin", strategy.get("strategy_origin", ""))).strip(),
        generated_at=str(generated_at).strip() if generated_at is not None else _utcnow_iso(),
    )


__all__ = [
    "CausalHypothesis",
    "ReflectionRecord",
    "build_reflection_record",
]
