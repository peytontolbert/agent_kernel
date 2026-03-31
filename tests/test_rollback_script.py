from pathlib import Path
import json
import subprocess
import sys

from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner


def test_rollback_artifact_script_restores_snapshot(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "rollback_artifact.py"
    cycles_path = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "skills.json"
    snapshot_path = tmp_path / ".artifact_history" / "skills.cycle_skills_1.pre.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [{"skill_id": "skill:hello_task:primary"}],
            }
        ),
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "skills": [{"skill_id": "skill:hello_task:primary"}],
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="reject",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="skill_set",
            reason="no gain",
            metrics_summary={},
            rollback_artifact_path=str(snapshot_path),
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-path",
            str(artifact_path),
            "--cycles-path",
            str(cycles_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    restored_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert "restored_artifact=" in completed.stdout
    assert restored_payload["lifecycle_state"] == "retained"


def test_rollback_artifact_script_resynchronizes_universe_bundle(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "rollback_artifact.py"
    cycles_path = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    constitution_path = tmp_path / "universe" / "universe_constitution.json"
    envelope_path = tmp_path / "universe" / "operating_envelope.json"
    snapshot_path = tmp_path / ".artifact_history" / "universe_contract.cycle_universe_1.pre.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_contract",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_contract_v1",
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["rollback invariant"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "action_risk_controls": {"verification_bonus": 4},
                "environment_assumptions": {"network_access_mode": "blocked"},
                "allowed_http_hosts": ["api.github.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["python", "pytest"],
                "learned_calibration_priors": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_contract",
                "lifecycle_state": "proposed",
                "control_schema": "universe_contract_v1",
                "governance": {"require_verification": False, "require_bounded_steps": False},
                "invariants": ["candidate invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "action_risk_controls": {},
                "environment_assumptions": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    constitution_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "universe_constitution",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_constitution_v1",
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["stale live invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "operating_envelope",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "operating_envelope_v1",
                "action_risk_controls": {},
                "environment_assumptions": {"network_access_mode": "open"},
                "allowed_http_hosts": [],
                "writable_path_prefixes": [],
                "toolchain_requirements": [],
                "learned_calibration_priors": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:universe:1",
            state="reject",
            subsystem="universe",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="universe_contract",
            reason="no gain",
            metrics_summary={},
            rollback_artifact_path=str(snapshot_path),
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-path",
            str(artifact_path),
            "--cycles-path",
            str(cycles_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    restored_contract = json.loads(artifact_path.read_text(encoding="utf-8"))
    restored_constitution = json.loads(constitution_path.read_text(encoding="utf-8"))
    restored_envelope = json.loads(envelope_path.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert restored_contract["invariants"] == ["rollback invariant"]
    assert restored_constitution["artifact_kind"] == "universe_constitution"
    assert restored_constitution["invariants"] == ["rollback invariant"]
    assert restored_envelope["artifact_kind"] == "operating_envelope"
    assert restored_envelope["allowed_http_hosts"] == ["api.github.com"]
    assert restored_envelope["environment_assumptions"]["network_access_mode"] == "blocked"


def test_validate_rollback_artifact_script_passes_when_live_artifact_matches_snapshot(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "validate_rollback_artifact.py"
    cycles_path = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "skills.json"
    snapshot_path = tmp_path / ".artifact_history" / "skills.cycle_skills_1.pre.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "asi_v1",
        "artifact_kind": "skill_set",
        "lifecycle_state": "retained",
        "skills": [{"skill_id": "skill:hello_task:primary"}],
    }
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="reject",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="skill_set",
            reason="no gain",
            metrics_summary={},
            rollback_artifact_path=str(snapshot_path),
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-path",
            str(artifact_path),
            "--cycles-path",
            str(cycles_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validation_state=passed" in completed.stdout


def test_validate_rollback_artifact_script_fails_when_live_artifact_drifted(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "validate_rollback_artifact.py"
    cycles_path = tmp_path / "cycles.jsonl"
    artifact_path = tmp_path / "skills.json"
    snapshot_path = tmp_path / ".artifact_history" / "skills.cycle_skills_1.pre.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "skills": [{"skill_id": "skill:hello_task:primary"}],
            }
        ),
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "skill_set",
                "lifecycle_state": "proposed",
                "skills": [{"skill_id": "skill:hello_task:secondary"}],
            }
        ),
        encoding="utf-8",
    )
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="reject",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="skill_set",
            reason="no gain",
            metrics_summary={},
            rollback_artifact_path=str(snapshot_path),
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-path",
            str(artifact_path),
            "--cycles-path",
            str(cycles_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "rollback validation failed:" in completed.stderr
