from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner


def _match_id(index: int) -> str:
    return f"match:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}:{index}"


def _protocol_root(config: KernelConfig, *, protocol_match_id: str, protocol: str) -> Path:
    safe_match_id = protocol_match_id.replace(":", "_")
    return config.improvement_reports_dir / ".protocol_runtime" / safe_match_id / protocol


def _runtime_feature_env(config: KernelConfig) -> dict[str, str]:
    return config.to_env()


def _protocol_env(config: KernelConfig, protocol_root: Path) -> dict[str, str]:
    trajectories_root = protocol_root / "trajectories"
    env = config.to_env()
    env.update(
        {
        "AGENT_KERNEL_WORKSPACE_ROOT": str(protocol_root / "workspace"),
        "AGENT_KERNEL_TRAJECTORIES_ROOT": str(trajectories_root / "episodes"),
        "AGENT_KERNEL_SKILLS_PATH": str(trajectories_root / "skills" / "command_skills.json"),
        "AGENT_KERNEL_OPERATOR_CLASSES_PATH": str(trajectories_root / "operators" / "operator_classes.json"),
        "AGENT_KERNEL_TOOL_CANDIDATES_PATH": str(trajectories_root / "tools" / "tool_candidates.json"),
        "AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH": str(trajectories_root / "benchmarks" / "benchmark_candidates.json"),
        "AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH": str(trajectories_root / "retrieval" / "retrieval_proposals.json"),
        "AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH": str(trajectories_root / "retrieval" / "retrieval_asset_bundle.json"),
        "AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH": str(trajectories_root / "tolbert_model" / "tolbert_model_artifact.json"),
        "AGENT_KERNEL_TOLBERT_SUPERVISED_DATASETS_DIR": str(trajectories_root / "tolbert_model" / "datasets"),
        "AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH": str(trajectories_root / "tolbert_model" / "liftoff_gate_report.json"),
        "AGENT_KERNEL_VERIFIER_CONTRACTS_PATH": str(trajectories_root / "verifiers" / "verifier_contracts.json"),
        "AGENT_KERNEL_PROMPT_PROPOSALS_PATH": str(trajectories_root / "prompts" / "prompt_proposals.json"),
        "AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH": str(trajectories_root / "world_model" / "world_model_proposals.json"),
        "AGENT_KERNEL_STATE_ESTIMATION_PROPOSALS_PATH": str(trajectories_root / "state_estimation" / "state_estimation_proposals.json"),
        "AGENT_KERNEL_TRUST_PROPOSALS_PATH": str(trajectories_root / "trust" / "trust_proposals.json"),
        "AGENT_KERNEL_RECOVERY_PROPOSALS_PATH": str(trajectories_root / "recovery" / "recovery_proposals.json"),
        "AGENT_KERNEL_DELEGATION_PROPOSALS_PATH": str(trajectories_root / "delegation" / "delegation_proposals.json"),
        "AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH": str(trajectories_root / "operator_policy" / "operator_policy_proposals.json"),
        "AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH": str(trajectories_root / "transition_model" / "transition_model_proposals.json"),
        "AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH": str(trajectories_root / "curriculum" / "curriculum_proposals.json"),
        "AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH": str(trajectories_root / "improvement" / "cycles.jsonl"),
        "AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT": str(trajectories_root / "improvement" / "candidates"),
        "AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR": str(trajectories_root / "improvement" / "reports"),
        "AGENT_KERNEL_RUN_REPORTS_DIR": str(trajectories_root / "reports"),
        "AGENT_KERNEL_CAPABILITY_MODULES_PATH": str(protocol_root / "config" / "capabilities.json"),
        "AGENT_KERNEL_RUN_CHECKPOINTS_DIR": str(trajectories_root / "checkpoints"),
        "AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH": str(trajectories_root / "jobs" / "queue.json"),
        "AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH": str(trajectories_root / "jobs" / "runtime_state.json"),
        "AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT": str(trajectories_root / "recovery" / "workspaces"),
        "AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH": str(trajectories_root / "reports" / "unattended_trust_ledger.json"),
        }
    )
    return env


def _copy_file(src: Path | str, dst: Path | str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def _copy_tree(src: Path, dst: Path, *, ignore=None) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_file, ignore=ignore)


def _seed_episode_documents(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_root.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(src_root)
        except ValueError:
            continue
        _copy_file(path, dst_root / relative)


def _seed_protocol_runtime(config: KernelConfig, env: dict[str, str]) -> None:
    _seed_episode_documents(config.trajectories_root, Path(env["AGENT_KERNEL_TRAJECTORIES_ROOT"]))
    _copy_tree(config.run_checkpoints_dir, Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]))
    _copy_tree(config.unattended_workspace_snapshot_root, Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]))
    _copy_file(config.skills_path, Path(env["AGENT_KERNEL_SKILLS_PATH"]))
    _copy_file(config.operator_classes_path, Path(env["AGENT_KERNEL_OPERATOR_CLASSES_PATH"]))
    _copy_file(config.tool_candidates_path, Path(env["AGENT_KERNEL_TOOL_CANDIDATES_PATH"]))
    _copy_file(config.benchmark_candidates_path, Path(env["AGENT_KERNEL_BENCHMARK_CANDIDATES_PATH"]))
    _copy_file(config.retrieval_proposals_path, Path(env["AGENT_KERNEL_RETRIEVAL_PROPOSALS_PATH"]))
    _copy_file(config.retrieval_asset_bundle_path, Path(env["AGENT_KERNEL_RETRIEVAL_ASSET_BUNDLE_PATH"]))
    _copy_file(config.tolbert_model_artifact_path, Path(env["AGENT_KERNEL_TOLBERT_MODEL_ARTIFACT_PATH"]))
    _copy_file(config.verifier_contracts_path, Path(env["AGENT_KERNEL_VERIFIER_CONTRACTS_PATH"]))
    _copy_file(config.prompt_proposals_path, Path(env["AGENT_KERNEL_PROMPT_PROPOSALS_PATH"]))
    _copy_file(config.world_model_proposals_path, Path(env["AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH"]))
    _copy_file(config.state_estimation_proposals_path, Path(env["AGENT_KERNEL_STATE_ESTIMATION_PROPOSALS_PATH"]))
    _copy_file(config.trust_proposals_path, Path(env["AGENT_KERNEL_TRUST_PROPOSALS_PATH"]))
    _copy_file(config.recovery_proposals_path, Path(env["AGENT_KERNEL_RECOVERY_PROPOSALS_PATH"]))
    _copy_file(config.delegation_proposals_path, Path(env["AGENT_KERNEL_DELEGATION_PROPOSALS_PATH"]))
    _copy_file(config.operator_policy_proposals_path, Path(env["AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH"]))
    _copy_file(config.transition_model_proposals_path, Path(env["AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH"]))
    _copy_file(config.curriculum_proposals_path, Path(env["AGENT_KERNEL_CURRICULUM_PROPOSALS_PATH"]))
    _copy_file(config.improvement_cycles_path, Path(env["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"]))
    _copy_file(config.capability_modules_path, Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]))
    _copy_file(config.delegated_job_queue_path, Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]))
    _copy_file(config.delegated_job_runtime_state_path, Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]))
    _copy_file(config.unattended_trust_ledger_path, Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]))
    Path(env["AGENT_KERNEL_WORKSPACE_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_CANDIDATE_ARTIFACTS_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_IMPROVEMENT_REPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_RUN_REPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_TOLBERT_SUPERVISED_DATASETS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_RUN_CHECKPOINTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_DELEGATED_JOB_QUEUE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_DELEGATED_JOB_RUNTIME_STATE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]).parent.mkdir(parents=True, exist_ok=True)


def _protocol_command(
    *,
    protocol: str,
    repo_root: Path,
    args: argparse.Namespace,
    protocol_match_id: str,
) -> list[str]:
    if protocol == "autonomous":
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_improvement_cycle.py"),
            "--campaign-width",
            "1",
            "--variant-width",
            str(max(1, args.variant_width)),
            "--protocol-match-id",
            protocol_match_id,
            "--progress-label",
            protocol_match_id.replace(":", "_"),
        ]
    else:
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_human_guided_improvement_cycle.py"),
            "--protocol-match-id",
            protocol_match_id,
        ]
        if str(args.guided_subsystem).strip():
            cmd.extend(["--subsystem", str(args.guided_subsystem).strip()])
        if str(args.guided_variant_id).strip():
            cmd.extend(["--variant-id", str(args.guided_variant_id).strip()])
        if str(args.guided_notes).strip():
            cmd.extend(["--notes", str(args.guided_notes).strip()])
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.model:
        cmd.extend(["--model", args.model])
    for flag, enabled in (
        ("--include-episode-memory", args.include_episode_memory),
        ("--include-skill-memory", args.include_skill_memory),
        ("--include-skill-transfer", args.include_skill_transfer),
        ("--include-operator-memory", args.include_operator_memory),
        ("--include-tool-memory", args.include_tool_memory),
        ("--include-verifier-memory", args.include_verifier_memory),
        ("--include-curriculum", args.include_curriculum),
        ("--include-failure-curriculum", args.include_failure_curriculum),
    ):
        if enabled:
            cmd.append(flag)
    return cmd


def _run_protocol(
    *,
    protocol: str,
    repo_root: Path,
    config: KernelConfig,
    args: argparse.Namespace,
    protocol_match_id: str,
) -> dict[str, object]:
    root = _protocol_root(config, protocol_match_id=protocol_match_id, protocol=protocol)
    env_overrides = _protocol_env(config, root)
    _seed_protocol_runtime(config, env_overrides)
    env = dict(os.environ)
    env.update(env_overrides)
    completed = subprocess.run(
        _protocol_command(
            protocol=protocol,
            repo_root=repo_root,
            args=args,
            protocol_match_id=protocol_match_id,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    planner = ImprovementPlanner(
        memory_root=Path(env_overrides["AGENT_KERNEL_TRAJECTORIES_ROOT"]),
        cycles_path=Path(env_overrides["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"]),
        prompt_proposals_path=Path(env_overrides["AGENT_KERNEL_PROMPT_PROPOSALS_PATH"]),
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=Path(env_overrides["AGENT_KERNEL_CAPABILITY_MODULES_PATH"]),
        trust_ledger_path=Path(env_overrides["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"]),
        runtime_config=config,
    )
    records = planner.load_cycle_records(Path(env_overrides["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"]))
    decision = next(
        (
            record
            for record in reversed(records)
            if str(record.get("state", "")) in {"retain", "reject"}
        ),
        {},
    )
    cycle_id = str(decision.get("cycle_id", ""))
    select_record = next(
        (
            record
            for record in records
            if str(record.get("cycle_id", "")) == cycle_id and str(record.get("state", "")) == "select"
        ),
        {},
    )
    select_summary = select_record.get("metrics_summary", {})
    if not isinstance(select_summary, dict):
        select_summary = {}
    selected_variant = select_summary.get("selected_variant", {})
    if not isinstance(selected_variant, dict):
        selected_variant = {}
    metrics_summary = decision.get("metrics_summary", {})
    if not isinstance(metrics_summary, dict):
        metrics_summary = {}
    return {
        "protocol": protocol,
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout).strip(),
        "stderr": str(completed.stderr).strip(),
        "runtime_root": str(root),
        "cycles_path": env_overrides["AGENT_KERNEL_IMPROVEMENT_CYCLES_PATH"],
        "cycle_id": cycle_id,
        "state": str(decision.get("state", "")),
        "subsystem": str(decision.get("subsystem", "")),
        "variant_id": str(selected_variant.get("variant_id", "")),
        "baseline_pass_rate": float(metrics_summary.get("baseline_pass_rate", 0.0)),
        "candidate_pass_rate": float(metrics_summary.get("candidate_pass_rate", 0.0)),
        "baseline_average_steps": float(metrics_summary.get("baseline_average_steps", 0.0)),
        "candidate_average_steps": float(metrics_summary.get("candidate_average_steps", 0.0)),
    }


def _protocol_score(result: dict[str, object]) -> tuple[int, float, float]:
    return (
        1 if str(result.get("state", "")) == "retain" else 0,
        float(result.get("candidate_pass_rate", 0.0)) - float(result.get("baseline_pass_rate", 0.0)),
        float(result.get("baseline_average_steps", 0.0)) - float(result.get("candidate_average_steps", 0.0)),
    )


def _winner_for_match(autonomous: dict[str, object], human_guided: dict[str, object]) -> dict[str, object]:
    if int(autonomous.get("returncode", 1)) != 0 or int(human_guided.get("returncode", 1)) != 0:
        return {
            "winner": "incomplete",
            "reason": "one or both protocol runs failed before a comparable decision was produced",
        }
    autonomous_score = _protocol_score(autonomous)
    guided_score = _protocol_score(human_guided)
    if autonomous_score > guided_score:
        return {
            "winner": "autonomous",
            "reason": "autonomous retained a stronger measured result on the isolated matched run",
        }
    if guided_score > autonomous_score:
        return {
            "winner": "human_guided",
            "reason": "human-guided retained a stronger measured result on the isolated matched run",
        }
    return {
        "winner": "tie",
        "reason": "both isolated runs produced the same measured decision outcome",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--variant-width", type=int, default=1)
    parser.add_argument("--guided-subsystem", default="")
    parser.add_argument("--guided-variant-id", default="")
    parser.add_argument("--guided-notes", default="")
    parser.add_argument("--include-episode-memory", action="store_true")
    parser.add_argument("--include-skill-memory", action="store_true")
    parser.add_argument("--include-skill-transfer", action="store_true")
    parser.add_argument("--include-operator-memory", action="store_true")
    parser.add_argument("--include-tool-memory", action="store_true")
    parser.add_argument("--include-verifier-memory", action="store_true")
    parser.add_argument("--include-curriculum", action="store_true")
    parser.add_argument("--include-failure-curriculum", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model_name = args.model
    config.ensure_directories()

    repo_root = Path(__file__).resolve().parents[1]
    matches: list[dict[str, object]] = []
    for index in range(1, max(1, args.matches) + 1):
        protocol_match_id = _match_id(index)
        autonomous = _run_protocol(
            protocol="autonomous",
            repo_root=repo_root,
            config=config,
            args=args,
            protocol_match_id=protocol_match_id,
        )
        human_guided = _run_protocol(
            protocol="human_guided",
            repo_root=repo_root,
            config=config,
            args=args,
            protocol_match_id=protocol_match_id,
        )
        winner = _winner_for_match(autonomous, human_guided)
        matches.append(
            {
                "protocol_match_id": protocol_match_id,
                "autonomous": autonomous,
                "human_guided": human_guided,
                "winner": winner["winner"],
                "reason": winner["reason"],
            }
        )

    summary = {
        "matched_pairs": len(matches),
        "autonomous_wins": sum(1 for match in matches if match["winner"] == "autonomous"),
        "human_guided_wins": sum(1 for match in matches if match["winner"] == "human_guided"),
        "ties": sum(1 for match in matches if match["winner"] == "tie"),
        "incomplete_pairs": sum(1 for match in matches if match["winner"] == "incomplete"),
    }
    if summary["autonomous_wins"] > summary["human_guided_wins"]:
        winner = "autonomous"
        autonomous_beats_human_guided = True
    elif summary["human_guided_wins"] > summary["autonomous_wins"]:
        winner = "human_guided"
        autonomous_beats_human_guided = False
    else:
        winner = "tie"
        autonomous_beats_human_guided = False

    report = {
        "spec_version": "asi_v1",
        "report_kind": "protocol_head_to_head_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matches_requested": max(1, args.matches),
        "summary": {
            **summary,
            "winner": winner,
            "autonomous_beats_human_guided": autonomous_beats_human_guided,
        },
        "matches": matches,
    }
    report_path = config.improvement_reports_dir / (
        f"protocol_head_to_head_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    planner = ImprovementPlanner(
        memory_root=config.trajectories_root,
        cycles_path=config.improvement_cycles_path,
        prompt_proposals_path=config.prompt_proposals_path,
        use_prompt_proposals=config.use_prompt_proposals,
        capability_modules_path=config.capability_modules_path,
        trust_ledger_path=config.unattended_trust_ledger_path,
        runtime_config=config,
    )
    planner.append_cycle_record(
        config.improvement_cycles_path,
        ImprovementCycleRecord(
            cycle_id=f"protocol_match:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            state="record",
            subsystem="protocol_match",
            action="run_head_to_head_protocols",
            artifact_path=str(report_path),
            artifact_kind="protocol_head_to_head_report",
            reason="run isolated autonomous and human-guided improvement protocols on matched work",
            metrics_summary={
                "matched_pairs": summary["matched_pairs"],
                "autonomous_wins": summary["autonomous_wins"],
                "human_guided_wins": summary["human_guided_wins"],
                "ties": summary["ties"],
                "incomplete_pairs": summary["incomplete_pairs"],
                "winner": winner,
                "autonomous_beats_human_guided": autonomous_beats_human_guided,
            },
        ),
    )
    print(report_path)


if __name__ == "__main__":
    main()
