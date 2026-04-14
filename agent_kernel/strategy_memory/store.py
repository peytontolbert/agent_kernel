from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re

from ..config import KernelConfig
from ..ops.runtime_supervision import atomic_write_json
from ..extensions.strategy.semantic_hub import record_semantic_attempt, record_semantic_note, record_semantic_skill
from .lesson import synthesize_strategy_lesson
from .node import StrategyNode
from .snapshot import build_strategy_snapshots


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_node_id(cycle_id: str, selected_variant_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cycle_id).strip()).strip("._") or "cycle"
    variant = re.sub(r"[^A-Za-z0-9._-]+", "_", str(selected_variant_id).strip()).strip("._")
    return f"strategy_node:{base}:{variant}" if variant else f"strategy_node:{base}"


def _strategy_id_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("strategy_id", payload.get("strategy_candidate_id", ""))).strip()


def _parent_node_ids_from_payload(payload: object, *, continuation_parent_node_id: str = "") -> list[str]:
    if not isinstance(payload, dict):
        return [continuation_parent_node_id] if continuation_parent_node_id else []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in list(payload.get("parent_strategy_node_ids", []) or []):
        parent_id = str(value).strip()
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            normalized.append(parent_id)
    if continuation_parent_node_id and continuation_parent_node_id not in seen:
        normalized.append(continuation_parent_node_id)
    return normalized


def _summarize_actor_summary(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    result = payload.get("result", {})
    if not isinstance(result, dict):
        result = {}
    summary = {
        "actor_type": str(payload.get("actor_type", "")).strip(),
        "actor_id": str(payload.get("actor_id", result.get("actor_id", ""))).strip(),
        "mode": str(payload.get("mode", result.get("mode", ""))).strip(),
        "status": str(payload.get("status", "")).strip(),
        "decision_ready": bool(payload.get("decision_ready", False)),
        "retained_gain_hint": bool(payload.get("retained_gain_hint", False)),
        "decision_credit_hint": bool(payload.get("decision_credit_hint", False)),
        "outcome": str(result.get("outcome", "")).strip(),
        "family": str(result.get("family", "")).strip(),
        "verifier_passed": bool(result.get("verifier_passed", False)),
        "steps_executed": int(result.get("steps_executed", 0) or 0),
        "commands_executed": int(result.get("commands_executed", 0) or 0),
        "runtime_minutes": float(result.get("runtime_minutes", 0.0) or 0.0),
        "files_touched": [str(value).strip() for value in result.get("files_touched", []) if str(value).strip()],
        "tests_run": [str(value).strip() for value in result.get("tests_run", []) if str(value).strip()],
        "notes": [str(value).strip() for value in result.get("notes", []) if str(value).strip()],
    }
    return {key: value for key, value in summary.items() if value not in ("", [], {})}


def _summarize_execution_evidence(report: dict[str, object], *, family_coverage: dict[str, object]) -> dict[str, object]:
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    phase_gate = report.get("phase_gate_report", {})
    if not isinstance(phase_gate, dict):
        phase_gate = {}
    improved_families = sorted(
        str(family).strip()
        for family, value in family_coverage.items()
        if str(family).strip() and float(value or 0.0) > 0.0
    )
    regressed_families = sorted(
        str(family).strip()
        for family, value in family_coverage.items()
        if str(family).strip() and float(value or 0.0) < 0.0
    )
    closeout_mode = str(report.get("closeout_mode", "")).strip()
    phase_gate_failures = [str(value).strip() for value in list(phase_gate.get("failures", []) or []) if str(value).strip()]
    payload = {
        "pass_rate_delta": float(evidence.get("pass_rate_delta", 0.0) or 0.0),
        "average_step_delta": float(evidence.get("average_step_delta", 0.0) or 0.0),
        "trusted_carryover_repair_rate_delta": float(
            evidence.get("trusted_carryover_repair_rate_delta", 0.0) or 0.0
        ),
        "false_failure_rate": float(evidence.get("false_failure_rate", 0.0) or 0.0),
        "improved_families": improved_families,
        "regressed_families": regressed_families,
        "family_positive_count": len(improved_families),
        "family_regression_count": len(regressed_families),
        "family_breadth_gain": len(improved_families),
        "closeout_mode": closeout_mode,
        "closeout_ready": bool(
            str(report.get("final_state", "")).strip() == "retain"
            and bool(phase_gate.get("passed", False))
            and closeout_mode in {"natural", "child_native_before_partial_timeout"}
        ),
        "phase_gate_passed": bool(phase_gate.get("passed", False)),
        "phase_gate_failures": phase_gate_failures,
        "phase_gate_failure_count": len(phase_gate_failures),
        "generated_lane_included": bool(phase_gate.get("generated_lane_included", False)),
        "failure_recovery_lane_included": bool(phase_gate.get("failure_recovery_lane_included", False)),
    }
    return {key: value for key, value in payload.items() if value not in ("", [], {})}


def load_strategy_nodes(config: KernelConfig) -> list[StrategyNode]:
    path = config.strategy_memory_nodes_path
    if not path.exists():
        return []
    nodes: list[StrategyNode] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        node = StrategyNode.from_dict(payload)
        if node is not None:
            nodes.append(node)
    return nodes


def _write_nodes(path: Path, nodes: list[StrategyNode]) -> None:
    payload = "\n".join(json.dumps(node.to_dict(), sort_keys=True) for node in nodes)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def upsert_strategy_node(config: KernelConfig, node: StrategyNode) -> StrategyNode:
    nodes = load_strategy_nodes(config)
    replaced = False
    for index, existing in enumerate(nodes):
        if existing.strategy_node_id == node.strategy_node_id:
            nodes[index] = node
            replaced = True
            break
    if not replaced:
        nodes.append(node)
    parent_node_ids = []
    for parent_id in [*node.parent_strategy_node_ids, node.continuation_parent_node_id]:
        normalized = str(parent_id).strip()
        if normalized and normalized not in parent_node_ids:
            parent_node_ids.append(normalized)
    if parent_node_ids:
        for index, existing in enumerate(nodes):
            if existing.strategy_node_id not in parent_node_ids:
                continue
            descendants = list(existing.descendant_node_ids)
            if node.strategy_node_id not in descendants:
                descendants.append(node.strategy_node_id)
            nodes[index] = StrategyNode(
                **{
                    **existing.to_dict(),
                    "descendant_node_ids": descendants,
                }
            )
    nodes.sort(key=lambda item: (item.created_at, item.strategy_node_id))
    _write_nodes(config.strategy_memory_nodes_path, nodes)
    atomic_write_json(config.strategy_memory_snapshots_path, build_strategy_snapshots(nodes), config=config)
    return node


def record_pending_strategy_node(
    config: KernelConfig,
    *,
    cycle_id: str,
    subsystem: str,
    selected_variant_id: str,
    strategy_candidate: dict[str, object],
    motivation: str,
    controls: dict[str, object],
    score: float,
    family_coverage: dict[str, object],
    artifact_paths: dict[str, str],
) -> StrategyNode:
    created_at = _now_iso()
    continuation_parent_node_id = str(
        strategy_candidate.get("continuation_parent_node_id")
        or strategy_candidate.get("best_retained_strategy_node_id")
        or next(iter(list(strategy_candidate.get("parent_strategy_node_ids", []) or [])), "")
    ).strip()
    parent_strategy_node_ids = _parent_node_ids_from_payload(
        strategy_candidate,
        continuation_parent_node_id=continuation_parent_node_id,
    )
    node = StrategyNode(
        strategy_node_id=_safe_node_id(cycle_id, selected_variant_id),
        created_at=created_at,
        updated_at=created_at,
        parent_strategy_node_ids=parent_strategy_node_ids,
        cycle_id=str(cycle_id).strip(),
        subsystem=str(subsystem).strip(),
        selected_variant_id=str(selected_variant_id).strip(),
        strategy_id=_strategy_id_from_payload(strategy_candidate),
        strategy_candidate_id=str(strategy_candidate.get("strategy_candidate_id", "")).strip(),
        strategy_candidate_kind=str(strategy_candidate.get("strategy_candidate_kind", "")).strip(),
        strategy_origin=str(strategy_candidate.get("origin", strategy_candidate.get("strategy_origin", ""))).strip(),
        motivation=str(motivation).strip(),
        controls=dict(controls),
        actor_summary=_summarize_actor_summary(strategy_candidate.get("actor_summary", {})),
        results_summary={"status": "pending"},
        retention_state="pending",
        retained_gain=0.0,
        continuation_parent_node_id=continuation_parent_node_id,
        continuation_workspace_ref=str(strategy_candidate.get("continuation_workspace_ref", "")).strip(),
        continuation_artifact_path=str(strategy_candidate.get("continuation_artifact_path", "")).strip()
        or str(artifact_paths.get("active_artifact_path", "")).strip(),
        continuation_branch=str(strategy_candidate.get("continuation_branch", "")).strip(),
        semantic_hypotheses=[
            str(value).strip()
            for value in list(strategy_candidate.get("semantic_hypotheses", []) or [])
            if str(value).strip()
        ],
        stagnation_count=max(0, int(strategy_candidate.get("stagnation_count", 0) or 0)),
        score=float(score or 0.0),
        visit_count=1,
        family_coverage=dict(family_coverage),
        artifact_paths={str(key): str(value) for key, value in artifact_paths.items() if str(value).strip()},
    )
    node = upsert_strategy_node(config, node)
    attempt_path = record_semantic_attempt(
        config,
        attempt_id=node.strategy_node_id,
        payload={
            "created_at": created_at,
            "cycle_id": node.cycle_id,
            "subsystem": node.subsystem,
            "status": "pending",
            "motivation": node.motivation,
            "selected_variant_id": node.selected_variant_id,
            "strategy_id": node.strategy_id or node.strategy_candidate_id,
            "strategy_candidate_id": node.strategy_candidate_id,
            "strategy_candidate_kind": node.strategy_candidate_kind,
            "strategy_origin": node.strategy_origin,
            "parent_strategy_node_ids": list(node.parent_strategy_node_ids),
            "continuation_parent_node_id": node.continuation_parent_node_id,
            "continuation_workspace_ref": node.continuation_workspace_ref,
            "continuation_artifact_path": node.continuation_artifact_path,
            "continuation_branch": node.continuation_branch,
            "semantic_hypotheses": list(node.semantic_hypotheses),
            "artifact_paths": dict(node.artifact_paths),
            "family_coverage": dict(node.family_coverage),
            "score": float(node.score),
        },
    )
    node.artifact_paths["semantic_attempt_path"] = str(attempt_path)
    return upsert_strategy_node(config, node)


def finalize_strategy_node(config: KernelConfig, report: dict[str, object]) -> StrategyNode:
    cycle_id = str(report.get("cycle_id", "")).strip()
    strategy_candidate = report.get("strategy_candidate", {})
    if not isinstance(strategy_candidate, dict):
        strategy_candidate = {}
    selected_variant_id = str(strategy_candidate.get("selected_variant_id", "")).strip()
    nodes = load_strategy_nodes(config)
    node = next((item for item in nodes if item.cycle_id == cycle_id), None)
    if node is None:
        created_at = _now_iso()
        node = StrategyNode(
            strategy_node_id=_safe_node_id(cycle_id, selected_variant_id),
            created_at=created_at,
            updated_at=created_at,
            parent_strategy_node_ids=_parent_node_ids_from_payload(
                strategy_candidate,
                continuation_parent_node_id=str(
                    strategy_candidate.get("continuation_parent_node_id")
                    or strategy_candidate.get("best_retained_strategy_node_id", "")
                ).strip(),
            ),
            cycle_id=cycle_id,
            subsystem=str(report.get("subsystem", "")).strip(),
            selected_variant_id=selected_variant_id,
            strategy_id=_strategy_id_from_payload(strategy_candidate) or str(report.get("strategy_id", "")).strip(),
            strategy_candidate_id=str(report.get("strategy_candidate_id", "")).strip(),
            strategy_candidate_kind=str(report.get("strategy_candidate_kind", "")).strip(),
            strategy_origin=str(
                strategy_candidate.get("origin")
                or strategy_candidate.get("strategy_origin")
                or report.get("strategy_origin", "")
            ).strip(),
            continuation_parent_node_id=str(
                strategy_candidate.get("continuation_parent_node_id")
                or strategy_candidate.get("best_retained_strategy_node_id", "")
            ).strip(),
            continuation_workspace_ref=str(strategy_candidate.get("continuation_workspace_ref", "")).strip(),
            continuation_artifact_path=str(strategy_candidate.get("continuation_artifact_path", "")).strip(),
            continuation_branch=str(strategy_candidate.get("continuation_branch", "")).strip(),
            semantic_hypotheses=[
                str(value).strip()
                for value in list(strategy_candidate.get("semantic_hypotheses", []) or report.get("semantic_hypotheses", []) or [])
                if str(value).strip()
            ],
        )
    if not node.strategy_id:
        node.strategy_id = _strategy_id_from_payload(strategy_candidate) or str(report.get("strategy_id", "")).strip()
    if not node.strategy_candidate_id:
        node.strategy_candidate_id = node.strategy_id
    if not node.strategy_candidate_kind:
        node.strategy_candidate_kind = str(
            strategy_candidate.get("strategy_candidate_kind")
            or strategy_candidate.get("strategy_kind")
            or report.get("strategy_candidate_kind", "")
        ).strip()
    if not node.strategy_origin:
        node.strategy_origin = str(
            strategy_candidate.get("origin")
            or strategy_candidate.get("strategy_origin")
            or report.get("strategy_origin", "")
        ).strip()
    if not node.parent_strategy_node_ids:
        node.parent_strategy_node_ids = _parent_node_ids_from_payload(
            strategy_candidate,
            continuation_parent_node_id=node.continuation_parent_node_id,
        )
    actor_summary = _summarize_actor_summary(report.get("actor_summary", {}))
    if not actor_summary:
        actor_summary = _summarize_actor_summary(strategy_candidate.get("actor_summary", {}))
    if actor_summary:
        node.actor_summary = actor_summary
    if not node.semantic_hypotheses:
        node.semantic_hypotheses = [
            str(value).strip()
            for value in list(strategy_candidate.get("semantic_hypotheses", []) or report.get("semantic_hypotheses", []) or [])
            if str(value).strip()
        ]
    lesson = synthesize_strategy_lesson(report)
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    family_coverage = evidence.get("family_pass_rate_delta", {})
    if not isinstance(family_coverage, dict):
        family_coverage = {}
    node.updated_at = _now_iso()
    node.retention_state = str(report.get("final_state", "")).strip() or node.retention_state
    node.retained_gain = float(lesson.get("retained_gain", 0.0) or 0.0)
    node.analysis_lesson = str(lesson.get("analysis_lesson", "")).strip()
    node.reuse_conditions = [str(value).strip() for value in lesson.get("reuse_conditions", []) if str(value).strip()]
    node.avoid_conditions = [str(value).strip() for value in lesson.get("avoid_conditions", []) if str(value).strip()]
    if node.retention_state == "reject":
        node.stagnation_count = max(1, int(node.stagnation_count or 0) + 1)
    else:
        node.stagnation_count = 0
    node.score = round(float(node.retained_gain) + max(0.0, -float(evidence.get("average_step_delta", 0.0) or 0.0)) * 0.05, 4)
    node.visit_count = max(1, int(node.visit_count or 0))
    node.family_coverage = {str(key): value for key, value in family_coverage.items() if str(key).strip()}
    node.execution_evidence = _summarize_execution_evidence(report, family_coverage=node.family_coverage)
    node.results_summary = {
        "status": node.retention_state,
        "final_reason": str(report.get("final_reason", "")).strip(),
        "selected_variant_id": node.selected_variant_id,
        "strategy_id": node.strategy_id or node.strategy_candidate_id,
        "strategy_candidate_kind": node.strategy_candidate_kind,
        "strategy_origin": node.strategy_origin,
        "pass_rate_delta": float(evidence.get("pass_rate_delta", 0.0) or 0.0),
        "average_step_delta": float(evidence.get("average_step_delta", 0.0) or 0.0),
        "trusted_carryover_repair_rate_delta": float(evidence.get("trusted_carryover_repair_rate_delta", 0.0) or 0.0),
        "false_failure_rate": float(evidence.get("false_failure_rate", 0.0) or 0.0),
        "family_pass_rate_delta": node.family_coverage,
        "artifact_path": str(report.get("artifact_path", "")).strip(),
        "candidate_artifact_path": str(report.get("candidate_artifact_path", "")).strip(),
        "artifact_snapshot_path": str(report.get("artifact_snapshot_path", "")).strip(),
    }
    node.artifact_paths.update(
        {
            "report_path": str(report.get("report_path", "")).strip(),
            "artifact_path": str(report.get("artifact_path", "")).strip(),
            "candidate_artifact_path": str(report.get("candidate_artifact_path", "")).strip(),
            "artifact_snapshot_path": str(report.get("artifact_snapshot_path", "")).strip(),
        }
    )
    if not node.continuation_artifact_path:
        node.continuation_artifact_path = (
            str(report.get("candidate_artifact_path", "")).strip()
            or str(report.get("artifact_path", "")).strip()
            or node.continuation_artifact_path
        )
    note_path = record_semantic_note(
        config,
        note_id=f"{node.strategy_node_id}.reflection",
        payload={
            "created_at": _now_iso(),
            "strategy_node_id": node.strategy_node_id,
            "strategy_id": node.strategy_id or node.strategy_candidate_id,
            "strategy_origin": node.strategy_origin,
            "subsystem": node.subsystem,
            "retention_state": node.retention_state,
            "analysis_lesson": node.analysis_lesson,
            "reuse_conditions": list(node.reuse_conditions),
            "avoid_conditions": list(node.avoid_conditions),
            "semantic_hypotheses": list(node.semantic_hypotheses),
            "stagnation_count": int(node.stagnation_count),
            "family_coverage": dict(node.family_coverage),
            "execution_evidence": dict(node.execution_evidence),
        },
    )
    node.transfer_artifact_ids = sorted(
        {
            *node.transfer_artifact_ids,
            f"note:{note_path.stem}",
        }
    )
    if node.analysis_lesson:
        skill_path = record_semantic_skill(
            config,
            skill_id=f"{node.strategy_node_id}.skill",
            payload={
                "created_at": _now_iso(),
                "strategy_node_id": node.strategy_node_id,
                "strategy_id": node.strategy_id or node.strategy_candidate_id,
                "strategy_origin": node.strategy_origin,
                "subsystem": node.subsystem,
                "retention_state": node.retention_state,
                "analysis_lesson": node.analysis_lesson,
                "reuse_conditions": list(node.reuse_conditions),
                "avoid_conditions": list(node.avoid_conditions),
                "semantic_hypotheses": list(node.semantic_hypotheses),
                "continuation_artifact_path": node.continuation_artifact_path,
                "execution_evidence": dict(node.execution_evidence),
            },
        )
        node.transfer_artifact_ids = sorted(
            {
                *node.transfer_artifact_ids,
                f"skill:{skill_path.stem}",
            }
        )
    node = upsert_strategy_node(config, node)
    attempt_path = record_semantic_attempt(
        config,
        attempt_id=node.strategy_node_id,
        payload={
            "cycle_id": node.cycle_id,
            "subsystem": node.subsystem,
            "status": node.retention_state,
            "strategy_id": node.strategy_id or node.strategy_candidate_id,
            "strategy_origin": node.strategy_origin,
            "parent_strategy_node_ids": list(node.parent_strategy_node_ids),
            "actor_summary": dict(node.actor_summary),
            "results_summary": dict(node.results_summary),
            "analysis_lesson": node.analysis_lesson,
            "reuse_conditions": list(node.reuse_conditions),
            "avoid_conditions": list(node.avoid_conditions),
            "continuation_parent_node_id": node.continuation_parent_node_id,
            "continuation_workspace_ref": node.continuation_workspace_ref,
            "continuation_artifact_path": node.continuation_artifact_path,
            "continuation_branch": node.continuation_branch,
            "semantic_hypotheses": list(node.semantic_hypotheses),
            "descendant_node_ids": list(node.descendant_node_ids),
            "transfer_artifact_ids": list(node.transfer_artifact_ids),
            "artifact_paths": dict(node.artifact_paths),
            "family_coverage": dict(node.family_coverage),
            "execution_evidence": dict(node.execution_evidence),
            "retained_gain": float(node.retained_gain),
            "stagnation_count": int(node.stagnation_count),
        },
    )
    node.artifact_paths["semantic_attempt_path"] = str(attempt_path)
    return upsert_strategy_node(config, node)
