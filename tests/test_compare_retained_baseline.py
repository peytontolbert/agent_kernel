from pathlib import Path
import importlib.util
import json
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementCycleRecord, ImprovementPlanner
from agent_kernel.task_bank import TaskBank
from agent_kernel.universe_model import UniverseModel
from evals.metrics import AbstractionComparison, EvalMetrics


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "compare_retained_baseline.py"
    spec = importlib.util.spec_from_file_location("compare_retained_baseline", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_retained_baseline_restores_prior_snapshot_and_reports_delta(tmp_path, monkeypatch, capsys):
    module = _load_compare_module()
    artifact_path = tmp_path / "skills" / "command_skills.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps({"artifact_kind": "skill_set", "lifecycle_state": "retained", "skills": [{"skill_id": "current"}]}),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / ".artifact_history" / "command_skills.cycle_skills_1.post_retain.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps({"artifact_kind": "skill_set", "lifecycle_state": "retained", "skills": [{"skill_id": "prior"}]}),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:skills:1",
            state="retain",
            subsystem="skills",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="skill_set",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    observed_skill_ids: list[str] = []
    run_results = [
        EvalMetrics(
            total=10,
            passed=7,
            average_steps=1.4,
            proposal_selected_steps=1,
            novel_command_steps=1,
            novel_valid_command_steps=0,
            tolbert_primary_episodes=0,
            total_by_origin_benchmark_family={"workflow": 5, "project": 5},
            passed_by_origin_benchmark_family={"workflow": 4, "project": 3},
            generated_total=4,
            generated_passed=2,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 1},
            generated_by_benchmark_family={"workflow": 2, "project": 2},
            generated_passed_by_benchmark_family={"workflow": 1, "project": 1},
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.1,
            proposal_selected_steps=3,
            novel_command_steps=3,
            novel_valid_command_steps=2,
            tolbert_primary_episodes=4,
            total_by_origin_benchmark_family={"workflow": 5, "project": 5},
            passed_by_origin_benchmark_family={"workflow": 5, "project": 4},
            generated_total=4,
            generated_passed=3,
            generated_by_kind={"failure_recovery": 2},
            generated_passed_by_kind={"failure_recovery": 2},
            generated_by_benchmark_family={"workflow": 2, "project": 2},
            generated_passed_by_benchmark_family={"workflow": 2, "project": 1},
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        payload = json.loads(config.skills_path.read_text(encoding="utf-8"))
        observed_skill_ids.append(payload["skills"][0]["skill_id"])
        return run_results.pop(0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            skills_path=artifact_path,
            improvement_cycles_path=cycles_path,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_retained_baseline.py",
            "--subsystem",
            "skills",
            "--cycles-path",
            str(cycles_path),
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    restored_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert observed_skill_ids == ["prior", "current"]
    assert f"current_artifact_path={artifact_path}" in output
    assert f"baseline_snapshot_path={snapshot_path}" in output
    assert "baseline_cycle_id=cycle:skills:1" in output
    assert "pass_rate_delta=0.20" in output
    assert "proposal_selected_steps_delta=2" in output
    assert "novel_valid_command_rate_delta=0.67" in output
    assert "tolbert_primary_episodes_delta=4" in output
    assert "family_delta benchmark_family=project pass_rate_delta=0.20" in output
    assert "family_delta benchmark_family=workflow pass_rate_delta=0.20" in output
    assert "generated_family_delta benchmark_family=workflow pass_rate_delta=0.50" in output
    assert "generated_kind_delta generated_kind=failure_recovery pass_rate_delta=0.50" in output
    assert restored_payload["skills"][0]["skill_id"] == "current"


def test_compare_retained_baseline_reports_family_proposal_gate_failure_reason(tmp_path, monkeypatch, capsys):
    module = _load_compare_module()
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "tolbert_model" / "checkpoints" / "candidate.pt"
    cache_path = tmp_path / "tolbert_model" / "cache" / "candidate.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("pt", encoding="utf-8")
    cache_path.write_text("cache", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "retention_gate": {
                    "proposal_gate_by_benchmark_family": {
                        "project": {
                            "require_novel_command_signal": True,
                            "min_proposal_selected_steps_delta": 1,
                            "min_novel_valid_command_steps": 1,
                            "min_novel_valid_command_rate_delta": 0.1,
                        }
                    }
                },
                "runtime_paths": {
                    "checkpoint_path": str(checkpoint_path),
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / ".artifact_history" / "tolbert_model.cycle_tolbert_1.post_retain.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_paths": {
                    "checkpoint_path": str(checkpoint_path),
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:tolbert_model:1",
            state="retain",
            subsystem="tolbert_model",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="tolbert_model_bundle",
            reason="retained prior Tolbert baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    run_results = [
        EvalMetrics(
            total=10,
            passed=8,
            average_steps=1.4,
            proposal_selected_steps=0,
            novel_command_steps=0,
            novel_valid_command_steps=0,
            total_by_benchmark_family={"project": 5, "workflow": 5},
            passed_by_benchmark_family={"project": 4, "workflow": 4},
            proposal_metrics_by_benchmark_family={
                "project": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
                "workflow": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
            },
        ),
        EvalMetrics(
            total=10,
            passed=9,
            average_steps=1.1,
            proposal_selected_steps=2,
            novel_command_steps=2,
            novel_valid_command_steps=2,
            total_by_benchmark_family={"project": 5, "workflow": 5},
            passed_by_benchmark_family={"project": 4, "workflow": 5},
            proposal_metrics_by_benchmark_family={
                "project": {
                    "task_count": 5,
                    "proposal_selected_steps": 0,
                    "novel_valid_command_steps": 0,
                    "novel_valid_command_rate": 0.0,
                },
                "workflow": {
                    "task_count": 5,
                    "proposal_selected_steps": 2,
                    "novel_valid_command_steps": 2,
                    "novel_valid_command_rate": 1.0,
                },
            },
        ),
    ]

    def fake_run_eval(*, config, **kwargs):
        del config, kwargs
        return run_results.pop(0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            tolbert_model_artifact_path=artifact_path,
            improvement_cycles_path=cycles_path,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_retained_baseline.py",
            "--subsystem",
            "tolbert_model",
            "--cycles-path",
            str(cycles_path),
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out

    assert f"current_artifact_path={artifact_path}" in output
    assert f"baseline_snapshot_path={snapshot_path}" in output
    assert "family_proposal_gate_failure benchmark_family=project" in output
    assert "reason=current artifact produced no proposal-selected commands on project tasks" in output


def test_compare_retained_baseline_supports_operator_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        operator_classes_path=Path("trajectories/operators/operator_classes.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "operators")

    assert resolved == config.operator_classes_path


def test_compare_retained_baseline_supports_retrieval_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        retrieval_proposals_path=Path("trajectories/retrieval/retrieval_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "retrieval")

    assert resolved == config.retrieval_proposals_path


def test_compare_retained_baseline_supports_world_model_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        world_model_proposals_path=Path("trajectories/world_model/world_model_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "world_model")

    assert resolved == config.world_model_proposals_path


def test_compare_retained_baseline_supports_state_estimation_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        state_estimation_proposals_path=Path("trajectories/state_estimation/state_estimation_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "state_estimation")

    assert resolved == config.state_estimation_proposals_path


def test_compare_retained_baseline_supports_trust_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        trust_proposals_path=Path("trajectories/trust/trust_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "trust")

    assert resolved == config.trust_proposals_path


def test_compare_retained_baseline_supports_recovery_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        recovery_proposals_path=Path("trajectories/recovery/recovery_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "recovery")

    assert resolved == config.recovery_proposals_path


def test_compare_retained_baseline_supports_delegation_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        delegation_proposals_path=Path("trajectories/delegation/delegation_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "delegation")

    assert resolved == config.delegation_proposals_path


def test_compare_retained_baseline_supports_operator_policy_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        operator_policy_proposals_path=Path("trajectories/operator_policy/operator_policy_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "operator_policy")

    assert resolved == config.operator_policy_proposals_path


def test_compare_retained_baseline_supports_transition_model_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        transition_model_proposals_path=Path("trajectories/transition_model/transition_model_proposals.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "transition_model")

    assert resolved == config.transition_model_proposals_path


def test_compare_retained_baseline_supports_benchmark_artifact_path():
    module = _load_compare_module()
    config = KernelConfig(
        benchmark_candidates_path=Path("trajectories/benchmarks/benchmark_candidates.json"),
    )

    resolved = module._artifact_path_for_subsystem(config, "benchmark")

    assert resolved == config.benchmark_candidates_path


def test_compare_retained_baseline_uses_shadowed_universe_snapshot_not_live_split(tmp_path, monkeypatch, capsys):
    module = _load_compare_module()
    current_contract = tmp_path / "universe" / "universe_contract.json"
    current_constitution = tmp_path / "universe" / "universe_constitution.json"
    current_envelope = tmp_path / "universe" / "operating_envelope.json"
    snapshot_path = tmp_path / ".artifact_history" / "universe_contract.cycle_universe_1.post_retain.json"
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    current_contract.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    current_contract.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_contract_v1",
                "retention_gate": {"min_pass_rate_delta_abs": 0.0},
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["current combined invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "action_risk_controls": {},
                "environment_assumptions": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_contract_v1",
                "retention_gate": {"min_pass_rate_delta_abs": 0.0},
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["baseline combined invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "action_risk_controls": {},
                "environment_assumptions": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    current_constitution.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_constitution",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_constitution_v1",
                "retention_gate": {"min_pass_rate_delta_abs": 0.0},
                "governance": {"require_verification": True, "require_bounded_steps": True},
                "invariants": ["live split invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    current_envelope.write_text(
        json.dumps(
            {
                "artifact_kind": "operating_envelope",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "operating_envelope_v1",
                "retention_gate": {"min_pass_rate_delta_abs": 0.0},
                "action_risk_controls": {},
                "environment_assumptions": {},
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
            state="retain",
            subsystem="universe",
            action="finalize_cycle",
            artifact_path=str(current_contract),
            artifact_kind="universe_contract",
            reason="retained prior baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    observed_invariants: list[list[str]] = []

    def fake_run_eval(*, config, **kwargs):
        del kwargs
        summary = UniverseModel(config=config).summarize(TaskBank().get("hello_task"))
        observed_invariants.append(list(summary["constitution"]["invariants"]))
        return EvalMetrics(total=1, passed=1, average_steps=1.0)

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            universe_contract_path=current_contract,
            universe_constitution_path=current_constitution,
            operating_envelope_path=current_envelope,
            improvement_cycles_path=cycles_path,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_retained_baseline.py",
            "--subsystem",
            "universe",
            "--cycles-path",
            str(cycles_path),
            "--artifact-path",
            str(current_contract),
        ],
    )

    module.main()

    assert "baseline combined invariant" in observed_invariants[0]
    assert "current combined invariant" in observed_invariants[1]
    assert "live split invariant" not in observed_invariants[0]
    assert "live split invariant" not in observed_invariants[1]
    assert f"baseline_snapshot_path={snapshot_path}" in capsys.readouterr().out


def test_compare_retained_baseline_uses_abstraction_lane_for_operators(tmp_path, monkeypatch, capsys):
    module = _load_compare_module()
    artifact_path = tmp_path / "operators" / "operator_classes.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_class_set",
                "lifecycle_state": "retained",
                "operators": [{"operator_id": "current"}],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / ".artifact_history" / "operator_classes.cycle_operators_1.post_retain.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operator_class_set",
                "lifecycle_state": "retained",
                "operators": [{"operator_id": "prior"}],
            }
        ),
        encoding="utf-8",
    )
    cycles_path = tmp_path / "improvement" / "cycles.jsonl"
    planner = ImprovementPlanner(memory_root=tmp_path / "episodes")
    planner.append_cycle_record(
        cycles_path,
        ImprovementCycleRecord(
            cycle_id="cycle:operators:1",
            state="retain",
            subsystem="operators",
            action="finalize_cycle",
            artifact_path=str(artifact_path),
            artifact_kind="operator_class_set",
            reason="retained prior operator baseline",
            metrics_summary={},
            artifact_snapshot_path=str(snapshot_path),
        ),
    )

    observed_operator_ids: list[str] = []
    comparison_results = [
        AbstractionComparison(
            operator_metrics=EvalMetrics(
                total=10,
                passed=6,
                average_steps=1.6,
                total_by_memory_source={"operator": 10},
                passed_by_memory_source={"operator": 6},
            ),
            raw_skill_metrics=EvalMetrics(total=10, passed=5, average_steps=1.8),
            pass_rate_delta=0.1,
            average_steps_delta=-0.2,
            transfer_pass_rate_delta=0.1,
        ),
        AbstractionComparison(
            operator_metrics=EvalMetrics(
                total=10,
                passed=8,
                average_steps=1.2,
                total_by_memory_source={"operator": 10},
                passed_by_memory_source={"operator": 8},
            ),
            raw_skill_metrics=EvalMetrics(total=10, passed=5, average_steps=1.8),
            pass_rate_delta=0.3,
            average_steps_delta=-0.6,
            transfer_pass_rate_delta=0.3,
        ),
    ]

    def fake_compare_abstractions(*, config, **kwargs):
        del kwargs
        payload = json.loads(config.operator_classes_path.read_text(encoding="utf-8"))
        observed_operator_ids.append(payload["operators"][0]["operator_id"])
        return comparison_results.pop(0)

    def fail_run_eval(**kwargs):
        raise AssertionError("operator baseline comparison should use compare_abstraction_transfer_modes")

    monkeypatch.setattr(module, "compare_abstraction_transfer_modes", fake_compare_abstractions)
    monkeypatch.setattr(module, "run_eval", fail_run_eval)
    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            operator_classes_path=artifact_path,
            improvement_cycles_path=cycles_path,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_retained_baseline.py",
            "--subsystem",
            "operators",
            "--cycles-path",
            str(cycles_path),
            "--artifact-path",
            str(artifact_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    restored_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert observed_operator_ids == ["prior", "current"]
    assert "baseline_cycle_id=cycle:operators:1" in output
    assert "pass_rate_delta=0.20" in output
    assert restored_payload["operators"][0]["operator_id"] == "current"
