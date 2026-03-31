from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner


def _protocol_index(records: list[dict[str, object]]) -> dict[str, str]:
    protocols: dict[str, str] = {}
    for record in records:
        cycle_id = str(record.get("cycle_id", "")).strip()
        metrics_summary = record.get("metrics_summary", {})
        if not cycle_id or not isinstance(metrics_summary, dict):
            continue
        protocol = str(metrics_summary.get("protocol", "")).strip()
        if protocol and cycle_id not in protocols:
            protocols[cycle_id] = protocol
    return protocols


def _protocol_match_index(records: list[dict[str, object]]) -> dict[str, str]:
    matches: dict[str, str] = {}
    for record in records:
        cycle_id = str(record.get("cycle_id", "")).strip()
        metrics_summary = record.get("metrics_summary", {})
        if not cycle_id or not isinstance(metrics_summary, dict):
            continue
        protocol_match_id = str(metrics_summary.get("protocol_match_id", "")).strip()
        if protocol_match_id and cycle_id not in matches:
            matches[cycle_id] = protocol_match_id
    return matches


def _average_delta(rows: list[dict[str, object]], *, baseline_key: str, candidate_key: str) -> float:
    deltas: list[float] = []
    for row in rows:
        metrics_summary = row.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            continue
        try:
            baseline = float(metrics_summary.get(baseline_key, 0.0))
            candidate = float(metrics_summary.get(candidate_key, 0.0))
        except (TypeError, ValueError):
            continue
        deltas.append(candidate - baseline)
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _protocol_summary_for(decisions: list[dict[str, object]], protocol_by_cycle: dict[str, str]) -> dict[str, dict[str, object]]:
    protocol_summary: dict[str, dict[str, object]] = {}
    for protocol in sorted(set(protocol_by_cycle.values()) | {"autonomous", "human_guided"}):
        protocol_rows = [
            record for record in decisions if protocol_by_cycle.get(str(record.get("cycle_id", ""))) == protocol
        ]
        retained = [record for record in protocol_rows if str(record.get("state", "")) == "retain"]
        rejected = [record for record in protocol_rows if str(record.get("state", "")) == "reject"]
        protocol_summary[protocol] = {
            "decision_count": len(protocol_rows),
            "retained_cycles": len(retained),
            "rejected_cycles": len(rejected),
            "average_pass_rate_delta": _average_delta(
                protocol_rows,
                baseline_key="baseline_pass_rate",
                candidate_key="candidate_pass_rate",
            ),
            "average_step_delta": _average_delta(
                protocol_rows,
                baseline_key="baseline_average_steps",
                candidate_key="candidate_average_steps",
            ),
        }
    return protocol_summary


def _winner_for_protocol_summary(protocol_summary: dict[str, dict[str, object]]) -> dict[str, object]:
    autonomous = dict(protocol_summary.get("autonomous", {}))
    human_guided = dict(protocol_summary.get("human_guided", {}))

    def _score(summary: dict[str, object]) -> tuple[int, float, float]:
        return (
            int(summary.get("retained_cycles", 0)) - int(summary.get("rejected_cycles", 0)),
            float(summary.get("average_pass_rate_delta", 0.0)),
            -float(summary.get("average_step_delta", 0.0)),
        )

    autonomous_score = _score(autonomous)
    guided_score = _score(human_guided)
    if autonomous_score > guided_score:
        return {
            "winner": "autonomous",
            "autonomous_beats_human_guided": True,
            "reason": "autonomous retained more or stronger measured gains than the human-guided protocol",
        }
    if guided_score > autonomous_score:
        return {
            "winner": "human_guided",
            "autonomous_beats_human_guided": False,
            "reason": "human-guided retained more or stronger measured gains than the autonomous protocol",
        }
    return {
        "winner": "tie",
        "autonomous_beats_human_guided": False,
        "reason": "protocol summaries were tied on retained/rejected outcomes and measured deltas",
    }


def _match_outcome(protocol_rows: dict[str, dict[str, object]]) -> dict[str, object]:
    autonomous = protocol_rows.get("autonomous")
    human_guided = protocol_rows.get("human_guided")
    if autonomous is None or human_guided is None:
        return {
            "winner": "incomplete",
            "reason": "one protocol is missing a final decision for the match",
        }

    def _score(record: dict[str, object]) -> tuple[int, float, float]:
        metrics_summary = record.get("metrics_summary", {})
        if not isinstance(metrics_summary, dict):
            metrics_summary = {}
        return (
            1 if str(record.get("state", "")) == "retain" else 0,
            float(metrics_summary.get("candidate_pass_rate", 0.0)) - float(metrics_summary.get("baseline_pass_rate", 0.0)),
            float(metrics_summary.get("baseline_average_steps", 0.0)) - float(metrics_summary.get("candidate_average_steps", 0.0)),
        )

    autonomous_score = _score(autonomous)
    guided_score = _score(human_guided)
    if autonomous_score > guided_score:
        return {
            "winner": "autonomous",
            "reason": "autonomous retained a stronger measured result on the matched work item",
        }
    if guided_score > autonomous_score:
        return {
            "winner": "human_guided",
            "reason": "human-guided retained a stronger measured result on the matched work item",
        }
    return {
        "winner": "tie",
        "reason": "both protocols produced the same measured decision outcome on the matched work item",
    }


def _matched_results(
    decisions: list[dict[str, object]],
    *,
    protocol_by_cycle: dict[str, str],
    protocol_match_by_cycle: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    matched_protocol_rows: dict[str, dict[str, dict[str, object]]] = {}
    for record in decisions:
        cycle_id = str(record.get("cycle_id", "")).strip()
        protocol_match_id = protocol_match_by_cycle.get(cycle_id, "")
        protocol = protocol_by_cycle.get(cycle_id, "")
        if not protocol_match_id or protocol not in {"autonomous", "human_guided"}:
            continue
        matched_protocol_rows.setdefault(protocol_match_id, {})[protocol] = record

    results: list[dict[str, object]] = []
    for protocol_match_id in sorted(matched_protocol_rows):
        rows = matched_protocol_rows[protocol_match_id]
        outcome = _match_outcome(rows)
        results.append(
            {
                "protocol_match_id": protocol_match_id,
                "autonomous_cycle_id": str(rows.get("autonomous", {}).get("cycle_id", "")),
                "human_guided_cycle_id": str(rows.get("human_guided", {}).get("cycle_id", "")),
                "winner": outcome["winner"],
                "reason": outcome["reason"],
            }
        )

    summary = {
        "matched_pairs": len(results),
        "autonomous_wins": sum(1 for result in results if result["winner"] == "autonomous"),
        "human_guided_wins": sum(1 for result in results if result["winner"] == "human_guided"),
        "ties": sum(1 for result in results if result["winner"] == "tie"),
        "incomplete_pairs": sum(1 for result in results if result["winner"] == "incomplete"),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-path", default=None)
    args = parser.parse_args()

    config = KernelConfig()
    cycles_path = Path(args.cycles_path) if args.cycles_path else config.improvement_cycles_path
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    records = planner.load_cycle_records(cycles_path)
    protocol_by_cycle = _protocol_index(records)
    protocol_match_by_cycle = _protocol_match_index(records)
    decisions = [record for record in records if str(record.get("state", "")) in {"retain", "reject"}]
    protocol_summary = _protocol_summary_for(decisions, protocol_by_cycle)
    winner_summary = _winner_for_protocol_summary(protocol_summary)
    matched_results, head_to_head_summary = _matched_results(
        decisions,
        protocol_by_cycle=protocol_by_cycle,
        protocol_match_by_cycle=protocol_match_by_cycle,
    )

    report = {
        "spec_version": "asi_v1",
        "report_kind": "improvement_protocol_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cycles_path": str(cycles_path),
        "protocol_summary": protocol_summary,
        "cycle_protocol_index": protocol_by_cycle,
        "cycle_protocol_match_index": protocol_match_by_cycle,
        "winner_summary": winner_summary,
        "head_to_head_summary": head_to_head_summary,
        "matched_results": matched_results,
    }
    report_path = config.improvement_reports_dir / (
        f"protocol_comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=f"protocol:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            state="record",
            subsystem="protocol_comparison",
            action="summarize_protocols",
            artifact_path=str(report_path),
            artifact_kind="improvement_protocol_comparison",
            reason="summarize autonomous and human-guided improvement outcomes from cycle history",
            metrics_summary={
                "protocols_present": sorted(
                    protocol for protocol, summary in protocol_summary.items() if summary["decision_count"] > 0
                ),
                "winner": winner_summary["winner"],
                "autonomous_beats_human_guided": winner_summary["autonomous_beats_human_guided"],
                "matched_pairs": head_to_head_summary["matched_pairs"],
            },
        ),
    )
    print(report_path)


if __name__ == "__main__":
    main()
