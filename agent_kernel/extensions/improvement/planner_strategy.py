from __future__ import annotations

import hashlib
import json
from typing import Any

from evals.metrics import EvalMetrics


def strategy_candidate_options(
    planner: Any,
    candidate: Any,
    *,
    metrics: EvalMetrics,
    recent_activity: dict[str, object] | None = None,
    broad_observe_signal: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    authored = planner._authored_strategy_candidate(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
    synthesized = planner._synthesized_strategy_candidate(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
    options: list[dict[str, object]] = [authored]
    synthesized_id = str(synthesized.get("strategy_candidate_id", synthesized.get("strategy_id", ""))).strip()
    authored_id = str(authored.get("strategy_candidate_id", authored.get("strategy_id", ""))).strip()
    if synthesized_id and synthesized_id != authored_id:
        options.append(synthesized)
    return options


def strategy_signal_basis(
    planner: Any,
    candidate: Any,
    *,
    metrics: EvalMetrics,
    recent_activity: dict[str, object] | None = None,
    broad_observe_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    from ... import improvement as core

    evidence = dict(candidate.evidence) if isinstance(candidate.evidence, dict) else {}
    recent = recent_activity if isinstance(recent_activity, dict) else {}
    broad_signal = broad_observe_signal if isinstance(broad_observe_signal, dict) else {}
    transition_failure_counts = (
        dict(evidence.get("transition_failure_counts", {}))
        if isinstance(evidence.get("transition_failure_counts", {}), dict)
        else {}
    )
    failure_counts = (
        dict(evidence.get("failure_counts", {}))
        if isinstance(evidence.get("failure_counts", {}), dict)
        else {}
    )
    repeated_failure_motifs = core.normalized_control_mapping(
        {
            "items": [
                core._dominant_count_label(transition_failure_counts),
                core._dominant_count_label(failure_counts),
                "recent_no_yield" if int(recent.get("no_yield_cycles", 0) or 0) > 0 else "",
                "recent_reject" if int(recent.get("rejected_cycles", 0) or 0) > 0 else "",
                "recent_incomplete" if int(recent.get("recent_incomplete_cycles", 0) or 0) > 0 else "",
                "recent_regression" if int(recent.get("recent_regression_cycles", 0) or 0) > 0 else "",
            ]
        },
        list_fields=("items",),
    ).get("items", [])
    transfer_gaps = core.normalized_control_mapping(
        {
            "items": [
                "retrieval_confidence_gap"
                if int(getattr(metrics, "low_confidence_episodes", 0) or 0) > 0
                else "",
                "retrieval_carryover_gap"
                if int(getattr(metrics, "trusted_retrieval_steps", 0) or 0)
                < max(1, int(getattr(metrics, "total", 0) or 0) // 2)
                else "",
                "generated_task_transfer_gap"
                if int(getattr(metrics, "generated_total", 0) or 0) > 0
                and float(getattr(metrics, "generated_pass_rate", 0.0) or 0.0)
                < float(getattr(metrics, "pass_rate", 0.0) or 0.0)
                else "",
                "skill_transfer_gap"
                if int(metrics.total_by_memory_source.get("skill_transfer", 0) or 0) <= 0
                else "",
                "operator_transfer_gap"
                if int(metrics.total_by_memory_source.get("operator", 0) or 0) <= 0
                and int(metrics.total_by_memory_source.get("skill_transfer", 0) or 0) <= 0
                else "",
            ]
        },
        list_fields=("items",),
    ).get("items", [])
    repo_signals = [
        f"family:{family}"
        for family in ("repository", "project", "integration", "repo_chore")
        if int(metrics.total_by_benchmark_family.get(family, 0) or 0) > 0
    ]
    if bool(broad_signal.get("active", False)):
        repo_signals.append("broad_observe_ready")
    elif int(broad_signal.get("observed_family_count", 0) or 0) >= 2:
        repo_signals.append("multi_family_repo_activity")
    if int(evidence.get("repo_world_model_total", 0) or 0) > 0:
        repo_signals.append("repo_workflow_failures")
    return {
        "repeated_failure_motifs": core.normalized_control_mapping(
            {"items": repeated_failure_motifs},
            list_fields=("items",),
        ).get("items", []),
        "transfer_gaps": core.normalized_control_mapping(
            {"items": transfer_gaps},
            list_fields=("items",),
        ).get("items", []),
        "repo_signals": core.normalized_control_mapping(
            {"items": repo_signals},
            list_fields=("items",),
        ).get("items", []),
    }


def build_strategy_candidate(
    planner: Any,
    *,
    strategy_candidate_id: str,
    strategy_candidate_kind: str,
    origin: str,
    strategy_label: str,
    target_subsystem: str,
    rationale: str,
    generation_basis: dict[str, object] | None = None,
    target_conditions: dict[str, object] | None = None,
    controls: dict[str, object] | None = None,
    expected_signals: list[str] | None = None,
    selection_bonus: float = 0.0,
    portfolio_reasons: list[str] | None = None,
) -> dict[str, object]:
    from ... import improvement as core

    strategy_id = str(strategy_candidate_id).strip()
    basis = dict(generation_basis or {})
    basis.setdefault("repeated_failure_motifs", [])
    basis.setdefault("transfer_gaps", [])
    basis.setdefault("repo_signals", [])
    semantic_hypotheses = planner._strategy_semantic_hypotheses(
        strategy_candidate_kind=strategy_candidate_kind,
        target_subsystem=target_subsystem,
        rationale=rationale,
        generation_basis=basis,
        target_conditions=target_conditions,
        expected_signals=expected_signals,
    )
    return {
        "strategy_id": strategy_id,
        "strategy_candidate_id": strategy_id,
        "strategy_kind": str(strategy_candidate_kind).strip(),
        "strategy_candidate_kind": str(strategy_candidate_kind).strip(),
        "origin": str(origin).strip() or "authored_strategy",
        "strategy_label": str(strategy_label).strip() or strategy_id.rsplit(":", 1)[-1],
        "target_subsystem": str(target_subsystem).strip(),
        "rationale": str(rationale).strip(),
        "generation_basis": basis,
        "target_conditions": dict(target_conditions or {}),
        "controls": dict(controls or {}),
        "expected_signals": core.normalized_control_mapping(
            {"items": list(expected_signals or [])},
            list_fields=("items",),
        ).get("items", []),
        "semantic_hypotheses": semantic_hypotheses,
        "parent_strategy_node_ids": [],
        "parent_control_surface": {},
        "retention_inputs": {},
        "selection_bonus": round(float(selection_bonus), 4),
        "portfolio_reasons": core.normalized_control_mapping(
            {"items": list(portfolio_reasons or [])},
            list_fields=("items",),
        ).get("items", []),
        "source": "runtime_synthesized",
    }


def strategy_semantic_hypotheses(
    *,
    strategy_candidate_kind: str,
    target_subsystem: str,
    rationale: str,
    generation_basis: dict[str, object] | None,
    target_conditions: dict[str, object] | None,
    expected_signals: list[str] | None,
) -> list[str]:
    from ... import improvement as core

    basis = generation_basis if isinstance(generation_basis, dict) else {}
    conditions = target_conditions if isinstance(target_conditions, dict) else {}
    items: list[str] = [
        str(strategy_candidate_kind).strip(),
        str(target_subsystem).strip(),
        str(rationale).strip(),
    ]
    for field in ("repeated_failure_motifs", "transfer_gaps", "repo_signals"):
        items.extend(str(value).strip() for value in list(basis.get(field, []) or []) if str(value).strip())
    for key, value in conditions.items():
        if isinstance(value, (list, tuple, set)):
            items.extend(str(item).strip() for item in value if str(item).strip())
        elif str(value).strip():
            items.append(f"{key}:{value}")
    items.extend(str(value).strip() for value in list(expected_signals or []) if str(value).strip())
    return core.normalized_control_mapping({"items": items}, list_fields=("items",)).get("items", [])[:12]


def authored_strategy_candidate(
    planner: Any,
    candidate: Any,
    *,
    metrics: EvalMetrics,
    recent_activity: dict[str, object] | None = None,
    broad_observe_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    base_subsystem = planner._base_subsystem(candidate.subsystem)
    generation_basis = planner._strategy_signal_basis(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
    if base_subsystem == "retrieval":
        return planner._build_strategy_candidate(
            strategy_candidate_id="strategy:retrieval_direct_iteration",
            strategy_candidate_kind="retrieval_direct_iteration",
            origin="authored_strategy",
            strategy_label="retrieval_direct_iteration",
            target_subsystem=candidate.subsystem,
            rationale="retrieval selected directly without a broader composite intervention",
            generation_basis={**generation_basis, "selection_basis": "direct_portfolio"},
            target_conditions={
                "target_subsystem": candidate.subsystem,
                "base_subsystem": base_subsystem,
            },
            controls={"selection_mode": "direct_portfolio"},
            expected_signals=["retrieval_signal", "preview_outcome"],
        )
    return planner._build_strategy_candidate(
        strategy_candidate_id=f"strategy:subsystem:{base_subsystem or candidate.subsystem}",
        strategy_candidate_kind="subsystem_direct",
        origin="authored_strategy",
        strategy_label=f"{base_subsystem or candidate.subsystem}_direct",
        target_subsystem=candidate.subsystem,
        rationale="candidate selected directly from planner-scored subsystem portfolio",
        generation_basis={**generation_basis, "selection_basis": "direct_portfolio"},
        target_conditions={
            "target_subsystem": candidate.subsystem,
            "base_subsystem": base_subsystem,
        },
        controls={"selection_mode": "direct_portfolio"},
        expected_signals=["decision_credit", "retained_gain"],
    )


def stable_strategy_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def normalized_strategy_candidate(
    planner: Any,
    *,
    candidate_subsystem: str,
    strategy_candidate: dict[str, object] | None,
) -> dict[str, object]:
    from ... import improvement as core

    candidate = dict(strategy_candidate or {})
    base_subsystem = planner._base_subsystem(candidate_subsystem)
    strategy_candidate_id = str(
        candidate.get("strategy_candidate_id", "") or candidate.get("strategy_id", "")
    ).strip()
    if not strategy_candidate_id:
        strategy_candidate_id = f"strategy:subsystem:{base_subsystem or candidate_subsystem}"
    strategy_candidate_kind = str(
        candidate.get("strategy_candidate_kind", "") or candidate.get("strategy_kind", "")
    ).strip()
    if not strategy_candidate_kind:
        strategy_candidate_kind = "subsystem_direct"
    origin = str(candidate.get("origin", "")).strip()
    if not origin:
        origin = "discovered_strategy" if strategy_candidate_kind not in {"subsystem_direct", "retrieval_direct_iteration"} else "authored_strategy"
    generation_basis = dict(candidate.get("generation_basis", {})) if isinstance(candidate.get("generation_basis", {}), dict) else {}
    generation_basis.setdefault("repeated_failure_motifs", [])
    generation_basis.setdefault("transfer_gaps", [])
    generation_basis.setdefault("repo_signals", [])
    target_conditions = (
        dict(candidate.get("target_conditions", {}))
        if isinstance(candidate.get("target_conditions", {}), dict)
        else {}
    )
    expected_signals = core.normalized_control_mapping(
        {"items": list(candidate.get("expected_signals", []))},
        list_fields=("items",),
    ).get("items", [])
    semantic_hypotheses = core.normalized_control_mapping(
        {"items": list(candidate.get("semantic_hypotheses", []) or [])},
        list_fields=("items",),
    ).get("items", [])
    parent_control_surface = (
        dict(candidate.get("parent_control_surface", {}))
        if isinstance(candidate.get("parent_control_surface", {}), dict)
        else {}
    )
    if not semantic_hypotheses:
        semantic_hypotheses = planner._strategy_semantic_hypotheses(
            strategy_candidate_kind=strategy_candidate_kind,
            target_subsystem=str(candidate.get("target_subsystem", "")).strip() or candidate_subsystem,
            rationale=str(candidate.get("rationale", "")).strip()
            or "candidate selected directly from planner-scored subsystem portfolio",
            generation_basis=generation_basis,
            target_conditions=target_conditions,
            expected_signals=expected_signals,
        )
    return {
        "strategy_id": strategy_candidate_id,
        "strategy_candidate_id": strategy_candidate_id,
        "strategy_kind": strategy_candidate_kind,
        "strategy_candidate_kind": strategy_candidate_kind,
        "origin": origin,
        "strategy_label": str(candidate.get("strategy_label", "")).strip()
        or strategy_candidate_id.rsplit(":", 1)[-1],
        "target_subsystem": str(candidate.get("target_subsystem", "")).strip() or candidate_subsystem,
        "rationale": str(candidate.get("rationale", "")).strip()
        or "candidate selected directly from planner-scored subsystem portfolio",
        "generation_basis": generation_basis,
        "target_conditions": target_conditions,
        "controls": dict(candidate.get("controls", {}))
        if isinstance(candidate.get("controls", {}), dict)
        else {},
        "expected_signals": expected_signals,
        "semantic_hypotheses": semantic_hypotheses,
        "parent_strategy_node_ids": list(candidate.get("parent_strategy_node_ids", []) or []),
        "parent_control_surface": parent_control_surface,
        "retention_inputs": dict(candidate.get("retention_inputs", {}))
        if isinstance(candidate.get("retention_inputs", {}), dict)
        else {},
        "selection_bonus": round(float(candidate.get("selection_bonus", 0.0) or 0.0), 4),
        "portfolio_reasons": core.normalized_control_mapping(
            {"items": list(candidate.get("portfolio_reasons", []))},
            list_fields=("items",),
        ).get("items", []),
        "source": str(candidate.get("source", "runtime_synthesized")).strip() or "runtime_synthesized",
    }


def synthesized_strategy_candidate(
    planner: Any,
    candidate: Any,
    *,
    metrics: EvalMetrics,
    recent_activity: dict[str, object] | None = None,
    broad_observe_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    base_subsystem = planner._base_subsystem(candidate.subsystem)
    recent = recent_activity if isinstance(recent_activity, dict) else {}
    broad_signal = broad_observe_signal if isinstance(broad_observe_signal, dict) else {}
    signal_basis = planner._strategy_signal_basis(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
    if (
        bool(broad_signal.get("active", False))
        and not bool(broad_signal.get("retrieval_emergency", False))
        and base_subsystem not in {"retrieval", "tolbert_model", "qwen_adapter"}
    ):
        return planner._build_strategy_candidate(
            strategy_candidate_id="strategy:broad_observe_diversification",
            strategy_candidate_kind="broad_observe_diversification",
            origin="discovered_strategy",
            strategy_label="broad_observe_diversification",
            target_subsystem=candidate.subsystem,
            rationale="clean broad observe preferred a non-retrieval intervention lane",
            generation_basis={**signal_basis, "selection_basis": "repo_signal_diversification"},
            target_conditions={
                "target_subsystem": candidate.subsystem,
                "base_subsystem": base_subsystem,
                "required_repo_signals": list(signal_basis.get("repo_signals", [])),
                "disallowed_base_subsystems": ["retrieval", "tolbert_model", "qwen_adapter"],
            },
            controls={
                "selection_mode": "prefer_non_retrieval_first",
                "avoid_base_subsystems": ["retrieval", "tolbert_model", "qwen_adapter"],
            },
            expected_signals=["required_family_breadth", "decision_credit", "non_retrieval_first_pick"],
            selection_bonus=0.02,
            portfolio_reasons=["strategy_origin=discovered_strategy", "strategy_basis=repo_signals"],
        )
    if base_subsystem == "retrieval":
        if int(recent.get("no_yield_cycles", 0) or 0) > 0 or int(recent.get("rejected_cycles", 0) or 0) > 0:
            low_yield_total = int(recent.get("no_yield_cycles", 0) or 0) + int(recent.get("rejected_cycles", 0) or 0)
            return planner._build_strategy_candidate(
                strategy_candidate_id="strategy:retrieval_support_rebalance",
                strategy_candidate_kind="retrieval_support_rebalance",
                origin="discovered_strategy",
                strategy_label="retrieval_support_rebalance",
                target_subsystem=candidate.subsystem,
                rationale="retrieval remains active but recent history shows low-yield or reject-heavy behavior",
                generation_basis={**signal_basis, "selection_basis": "transfer_gap_rebalance"},
                target_conditions={
                    "target_subsystem": candidate.subsystem,
                    "base_subsystem": base_subsystem,
                    "required_transfer_gaps": list(signal_basis.get("transfer_gaps", [])),
                    "required_failure_motifs": list(signal_basis.get("repeated_failure_motifs", [])),
                },
                controls={
                    "selection_mode": "rebalanced_iteration",
                    "prefer_preview_yield": True,
                },
                expected_signals=["decision_credit", "retained_gain", "preview_yield"],
                selection_bonus=min(0.03, 0.01 * max(1, low_yield_total)),
                portfolio_reasons=["strategy_origin=discovered_strategy", "strategy_basis=transfer_gaps"],
            )
    low_yield_candidate = (
        int(recent.get("no_yield_cycles", 0) or 0) > 0
        or int(recent.get("rejected_cycles", 0) or 0) > 0
        or int(recent.get("recent_incomplete_cycles", 0) or 0) > 0
    )
    signal_count = sum(len(list(signal_basis.get(key, []))) for key in ("repeated_failure_motifs", "transfer_gaps", "repo_signals"))
    if low_yield_candidate and signal_count > 0:
        fingerprint = planner._stable_strategy_fingerprint(
            {
                "base_subsystem": base_subsystem,
                "generation_basis": signal_basis,
                "target_subsystem": candidate.subsystem,
            }
        )
        return planner._build_strategy_candidate(
            strategy_candidate_id=f"strategy:adaptive_countermeasure:{base_subsystem}:{fingerprint}",
            strategy_candidate_kind="adaptive_countermeasure",
            origin="discovered_strategy",
            strategy_label=f"{base_subsystem}_adaptive_countermeasure",
            target_subsystem=candidate.subsystem,
            rationale="repeated low-yield history plus runtime signals justify a discovered composite intervention",
            generation_basis={**signal_basis, "selection_basis": "failure_motif_countermeasure"},
            target_conditions={
                "target_subsystem": candidate.subsystem,
                "base_subsystem": base_subsystem,
                "required_failure_motifs": list(signal_basis.get("repeated_failure_motifs", [])),
                "required_transfer_gaps": list(signal_basis.get("transfer_gaps", [])),
                "required_repo_signals": list(signal_basis.get("repo_signals", [])),
            },
            controls={
                "selection_mode": "adaptive_countermeasure",
                "prefer_composite_intervention": True,
            },
            expected_signals=["decision_credit", "retained_gain", "failure_motif_clearance"],
            selection_bonus=min(0.03, signal_count * 0.005),
            portfolio_reasons=[
                "strategy_origin=discovered_strategy",
                "strategy_basis=repeated_failure_motifs",
            ],
        )
    return planner._authored_strategy_candidate(
        candidate,
        metrics=metrics,
        recent_activity=recent_activity,
        broad_observe_signal=broad_observe_signal,
    )
