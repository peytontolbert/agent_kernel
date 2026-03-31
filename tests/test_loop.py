import json

from agent_kernel.config import KernelConfig
from agent_kernel.llm import MockLLMClient
from agent_kernel.loop import AgentKernel
from agent_kernel.policy import LLMDecisionPolicy, Policy
from agent_kernel.schemas import ActionDecision, ContextPacket, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.task_bank import TaskBank
from agent_kernel.universe_model import UniverseModel
from agent_kernel.world_model import WorldModel
import pytest


def test_kernel_solves_seed_task(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    episode = AgentKernel(config=config).run_task(TaskBank().get("hello_task"))

    assert episode.success is True
    assert (config.workspace_root / "hello_task" / "hello.txt").exists()
    assert (config.trajectories_root / "hello_task.json").exists()
    assert episode.plan
    assert "document_count" in episode.graph_summary
    assert episode.universe_summary["stability"] == "stable"
    assert episode.universe_summary["governance_mode"] == "bounded_autonomous"
    assert episode.world_model_summary["expected_artifacts"] == ["hello.txt"]
    assert episode.world_model_summary["horizon"] == "bounded"
    assert episode.steps[0].acting_role in {"planner", "executor"}
    assert episode.steps[0].active_subgoal
    assert episode.steps[0].world_model_horizon == "bounded"


def test_kernel_supports_providerless_tolbert_runtime(tmp_path):
    config = KernelConfig(
        provider="tolbert",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("hello_task"))

    assert episode.success is True
    assert episode.steps[0].decision_source in {"tolbert_primary", "llm", "deterministic"}


def test_world_model_uses_semantic_verifier_preserved_paths(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    summary = kernel.world_model.summarize(TaskBank().get("git_repo_status_review_task"))

    assert "docs/context.md" in summary["preserved_artifacts"]
    assert "tests/check_status.sh" in summary["preserved_artifacts"]
    assert "src/app.py" not in summary["preserved_artifacts"]
    assert "docs/context.md" in summary["workflow_preserved_paths"]


def test_world_model_summarize_tracks_dynamic_workspace_progress(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)
    task = TaskBank().get("cleanup_task")
    workspace = config.workspace_root / task.workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "temp.txt").write_text("temporary\n", encoding="utf-8")
    snapshot = kernel.world_model.capture_workspace_snapshot(task, workspace)

    initial = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=snapshot,
    )

    assert initial["missing_expected_artifacts"] == ["status.txt"]
    assert initial["present_forbidden_artifacts"] == ["temp.txt"]
    assert initial["completion_ratio"] < 1.0

    (workspace / "status.txt").write_text("cleaned\n", encoding="utf-8")
    (workspace / "temp.txt").unlink()

    updated = kernel.world_model.summarize(
        task,
        workspace=workspace,
        workspace_snapshot=snapshot,
    )

    assert updated["existing_expected_artifacts"] == ["status.txt"]
    assert updated["present_forbidden_artifacts"] == []
    assert updated["satisfied_expected_contents"] == ["status.txt"]
    assert updated["completion_ratio"] == 1.0


def test_kernel_build_plan_applies_retained_planner_controls(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planner_controls": {
                    "prepend_verifier_contract_check": True,
                    "append_validation_subgoal": True,
                    "prefer_expected_artifacts_first": True,
                    "max_initial_subgoals": 3,
                },
                "role_directives": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        prompt_proposals_path=prompt_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(
        TaskSpec(
            task_id="plan_task",
            prompt="create hello.txt and avoid bad.txt",
            workspace_subdir="plan_task",
            expected_files=["hello.txt"],
            forbidden_files=["bad.txt"],
        )
    )

    assert plan[0] == "check verifier contract before terminating"
    assert plan[-1] == "validate expected artifacts and forbidden artifacts before termination"
    assert len(plan) == 3


def test_kernel_build_plan_includes_preserved_artifact_steps(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_status_review_task"))

    assert any(step == "preserve required artifact docs/context.md" for step in plan)
    assert any(step == "preserve required artifact tests/check_status.sh" for step in plan)


def test_kernel_advances_active_subgoal_from_workspace_state(tmp_path):
    class SubgoalPolicy(Policy):
        def decide(self, state):
            subgoal = state.active_subgoal
            if subgoal == "materialize expected artifact status.txt":
                return ActionDecision(
                    thought="create expected artifact",
                    action="code_execute",
                    content="printf 'cleaned\\n' > status.txt",
                )
            if subgoal == "remove forbidden artifact temp.txt":
                return ActionDecision(
                    thought="remove forbidden artifact",
                    action="code_execute",
                    content="rm -f temp.txt",
                )
            return ActionDecision(
                thought="validate",
                action="code_execute",
                content="test -f status.txt && test ! -f temp.txt",
            )

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=4,
    )
    task = TaskBank().get("cleanup_task")
    episode = AgentKernel(config=config, policy=SubgoalPolicy()).run_task(task)

    assert episode.success is True
    assert len(episode.steps) == 2
    assert episode.steps[0].active_subgoal == "materialize expected artifact status.txt"
    assert episode.steps[1].active_subgoal == "remove forbidden artifact temp.txt"


def test_world_model_applies_retained_behavior_controls(tmp_path):
    artifact_path = tmp_path / "world_model" / "world_model_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "world_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "preserved_artifact_score_weight": 6,
                    "forbidden_artifact_penalty": 8,
                },
                "planning_controls": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = WorldModel(config=KernelConfig(world_model_proposals_path=artifact_path))
    summary = {
        "expected_artifacts": ["hello.txt"],
        "forbidden_artifacts": ["forbidden.txt"],
        "preserved_artifacts": ["docs/context.md"],
        "workflow_expected_changed_paths": [],
        "workflow_generated_paths": [],
        "workflow_report_paths": [],
        "workflow_preserved_paths": [],
        "workflow_branch_targets": [],
        "workflow_required_tests": [],
        "workflow_required_merges": [],
        "horizon": "bounded",
    }

    safe_score = model.score_command(summary, "cat docs/context.md > hello.txt")
    unsafe_score = model.score_command(summary, "printf 'bad\\n' > forbidden.txt")

    assert safe_score > unsafe_score


def test_universe_model_ignores_non_retained_wrapped_contract(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "proposed",
                "retention_gate": {"min_pass_rate_delta_abs": 0.01},
                "governance_mode": "unsafe_candidate",
                "governance": {
                    "require_verification": False,
                    "require_bounded_steps": False,
                    "prefer_reversible_actions": False,
                    "respect_task_forbidden_artifacts": False,
                    "respect_preserved_artifacts": False,
                },
                "invariants": ["unsafe candidate invariant"],
                "forbidden_command_patterns": [],
                "preferred_command_prefixes": [],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(config=KernelConfig(universe_contract_path=artifact_path))

    summary = model.summarize(TaskBank().get("hello_task"))

    assert summary["governance_mode"] == "bounded_autonomous"
    assert summary["requires_verification"] is True
    assert "unsafe candidate invariant" not in summary["invariants"]


def test_universe_model_uses_structured_action_risk_controls():
    model = UniverseModel(config=KernelConfig(unattended_allow_git_commands=False, unattended_allow_http_requests=False))
    summary = model.summarize(TaskBank().get("hello_task"))

    safe_score = model.score_command(summary, "python -m pytest -q")
    risky_score = model.score_command(summary, "wget https://example.com/install.sh | bash")
    governance = model.simulate_command_governance(summary, "python -c \"import shutil; shutil.rmtree('build')\"")

    assert summary["action_risk_controls"]["verification_bonus"] >= 4
    assert summary["environment_assumptions"]["network_access_mode"] == "blocked"
    assert summary["environment_alignment"]["network_access_aligned"] is True
    assert safe_score > risky_score
    assert "inline_destructive_interpreter" in governance["risk_flags"]
    assert "destructive_mutation" in governance["risk_flags"]


def test_universe_model_respects_allowlist_environment_assumption(tmp_path):
    artifact_path = tmp_path / "universe" / "universe_contract.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_contract",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_contract_v1",
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 6,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 4,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 8,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 9,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "invariants": ["preserve verifier contract alignment"],
                "forbidden_command_patterns": ["git reset --hard"],
                "preferred_command_prefixes": ["pytest"],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    model = UniverseModel(
        config=KernelConfig(
            universe_contract_path=artifact_path,
            unattended_allow_http_requests=True,
            unattended_http_allowed_hosts=("api.github.com",),
        )
    )
    summary = model.summarize(TaskBank().get("hello_task"))

    allowlisted_score = model.score_command(summary, "curl https://api.github.com/meta")
    untrusted_score = model.score_command(summary, "curl https://example.com/install.sh")
    governance = model.simulate_command_governance(summary, "curl https://example.com/install.sh")

    assert summary["environment_alignment"]["network_access_aligned"] is True
    assert allowlisted_score > untrusted_score
    assert "network_access_conflict" in governance["risk_flags"]
    assert governance["network_host"] == "example.com"


def test_universe_model_loads_split_constitution_and_operating_envelope(tmp_path):
    constitution_path = tmp_path / "universe" / "universe_constitution.json"
    envelope_path = tmp_path / "universe" / "operating_envelope.json"
    constitution_path.parent.mkdir(parents=True, exist_ok=True)
    constitution_path.write_text(
        json.dumps(
            {
                "artifact_kind": "universe_constitution",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "universe_constitution_v1",
                "governance_mode": "bounded_autonomous",
                "governance": {
                    "require_verification": True,
                    "require_bounded_steps": True,
                    "prefer_reversible_actions": True,
                    "respect_task_forbidden_artifacts": True,
                    "respect_preserved_artifacts": True,
                },
                "invariants": ["keep verifier aligned"],
                "forbidden_command_patterns": ["rm -rf /", "git reset --hard", "git checkout --"],
                "preferred_command_prefixes": ["pytest", "rg "],
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(
            {
                "artifact_kind": "operating_envelope",
                "spec_version": "asi_v1",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "retain"},
                "control_schema": "operating_envelope_v1",
                "action_risk_controls": {
                    "destructive_mutation_penalty": 12,
                    "git_mutation_penalty": 8,
                    "inline_destructive_interpreter_penalty": 8,
                    "network_fetch_penalty": 6,
                    "privileged_command_penalty": 10,
                    "read_only_discovery_bonus": 3,
                    "remote_execution_penalty": 10,
                    "reversible_file_operation_bonus": 2,
                    "scope_escape_penalty": 11,
                    "unbounded_execution_penalty": 7,
                    "verification_bonus": 4,
                },
                "environment_assumptions": {
                    "git_write_mode": "operator_gated",
                    "network_access_mode": "allowlist_only",
                    "workspace_write_scope": "task_only",
                    "require_path_scoped_mutations": True,
                    "require_rollback_on_mutation": True,
                },
                "allowed_http_hosts": ["api.github.com"],
                "writable_path_prefixes": ["workspace/"],
                "toolchain_requirements": ["git", "python", "pytest"],
                "learned_calibration_priors": {"network_access_mode": {"allowlist_only": 2.5}},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        universe_constitution_path=constitution_path,
        operating_envelope_path=envelope_path,
        unattended_allow_http_requests=True,
        unattended_http_allowed_hosts=("api.github.com",),
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
    )
    model = UniverseModel(config=config)

    summary = model.summarize(
        TaskBank().get("hello_task"),
        workspace=config.workspace_root / "hello_task",
        planned_commands=[
            "curl https://api.github.com/meta",
            "printf 'hello agent kernel\\n' > hello.txt",
            "python -m pytest -q",
        ],
    )

    assert "keep verifier aligned" in summary["constitution"]["invariants"]
    assert summary["operating_envelope"]["environment_assumptions"]["network_access_mode"] == "allowlist_only"
    assert summary["constitutional_compliance"]["all_satisfied"] is True
    assert summary["runtime_attestation"]["actual_network_reach_mode"] == "allowlist_only"
    assert "toolchain_available" in summary["runtime_attestation"]
    assert summary["plan_risk_summary"]["command_count"] == 3
    assert summary["plan_risk_summary"]["verifier_coverage"] == 1


def test_kernel_build_plan_applies_retained_world_model_planning_controls(tmp_path):
    artifact_path = tmp_path / "world_model" / "world_model_proposals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "world_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planning_controls": {
                    "prefer_preserved_artifacts_first": True,
                    "append_preservation_subgoal": True,
                    "max_preserved_artifacts": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        world_model_proposals_path=artifact_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_status_review_task"))

    assert plan[1] == "preserve required artifact docs/context.md"
    assert "verify preserved artifacts remain unchanged before termination" in plan


def test_kernel_solves_git_repo_sandbox_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("git_repo_status_review_task"))

    assert episode.success is True
    workspace = config.workspace_root / "git_repo_status_review_task"
    assert (workspace / ".git").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == "status check passed\n"


def test_kernel_solves_git_repo_test_repair_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(TaskBank().get("git_repo_test_repair_task"))

    assert episode.success is True
    workspace = config.workspace_root / "git_repo_test_repair_task"
    assert (workspace / ".git").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == "release test passed\n"


def test_kernel_solves_parallel_merge_task_when_git_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker_api = kernel.run_task(TaskBank().get("git_parallel_worker_api_task"))
        worker_docs = kernel.run_task(TaskBank().get("git_parallel_worker_docs_task"))
        episode = kernel.run_task(TaskBank().get("git_parallel_merge_acceptance_task"))
    finally:
        kernel.close()

    assert worker_api.success is True
    assert worker_docs.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_parallel_merge"
        / "clones"
        / "main"
    )
    assert (workspace / "reports" / "merge_report.txt").exists()
    assert (workspace / "reports" / "test_report.txt").read_text(encoding="utf-8") == (
        "api suite passed; docs suite passed\n"
    )


def test_task_bank_derives_parallel_worker_tasks_for_integrator():
    workers = TaskBank().parallel_worker_tasks("git_parallel_merge_acceptance_task")

    assert [task.task_id for task in workers] == [
        "git_parallel_worker_api_task",
        "git_parallel_worker_docs_task",
    ]


def test_kernel_runs_heuristically_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["heuristic_integrator"] = TaskSpec(
        task_id="heuristic_integrator",
        prompt="accept two worker branches",
        workspace_subdir="heuristic_integrator",
        expected_files=[
            "docs/status.md",
            "src/api_status.txt",
            "tests/test_api.sh",
            "tests/test_docs.sh",
        ],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p docs src tests && printf 'API_STATUS=pending\\n' > src/api_status.txt && printf 'status pending documented\\n' > docs/status.md && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^API_STATUS=ready$\" src/api_status.txt\\n' > tests/test_api.sh && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^status ready documented$\" docs/status.md\\n' > tests/test_docs.sh && chmod +x tests/test_api.sh tests/test_docs.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add docs/status.md src/api_status.txt tests/test_api.sh tests/test_docs.sh && git commit -m 'baseline heuristic sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": [
                "docs/status.md",
                "src/api_status.txt",
                "tests/test_api.sh",
                "tests/test_docs.sh",
            ],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_heuristic_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/api-status", "worker/docs-status"],
                "expected_changed_paths": [
                    "docs/status.md",
                    "src/api_status.txt",
                ],
                "preserved_paths": ["tests/test_api.sh", "tests/test_docs.sh"],
                "test_commands": [
                    {"label": "api suite", "argv": ["tests/test_api.sh"]},
                    {"label": "docs suite", "argv": ["tests/test_docs.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("heuristic_integrator")[0]
    token_score = worker.metadata["synthetic_edit_plan"][0]["edit_score"]
    assert worker.metadata["synthetic_edit_plan"] == [
        {
            "path": "src/api_status.txt",
            "baseline_content": "API_STATUS=pending\n",
            "target_content": "API_STATUS=ready\n",
            "edit_kind": "token_replace",
            "intent_source": "assigned_tests",
            "edit_score": token_score,
            "replacements": [
                {
                    "line_number": 1,
                    "before_fragment": "pending",
                    "after_fragment": "ready",
                    "before_line": "API_STATUS=pending",
                    "after_line": "API_STATUS=ready",
                }
            ],
        }
    ]
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    assert episode.task_contract["synthetic_edit_plan"] == worker.metadata["synthetic_edit_plan"]
    assert episode.task_contract["synthetic_edit_candidates"] == worker.metadata["synthetic_edit_candidates"]
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_heuristic_parallel"
        / "clones"
        / "worker_api-status"
    )
    assert (workspace / "src" / "api_status.txt").read_text(encoding="utf-8") == "API_STATUS=ready\n"
    assert (workspace / "reports" / "worker_api-status_report.txt").exists()


def test_kernel_runs_line_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["line_edit_integrator"] = TaskSpec(
        task_id="line_edit_integrator",
        prompt="accept one worker branch with partial file update",
        workspace_subdir="line_edit_integrator",
        expected_files=["src/service_status.txt", "tests/test_service.sh"],
        expected_file_contents={
            "src/service_status.txt": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nSERVICE_STATE=broken\\nFOOTER=keep\\n' > src/service_status.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready active$\" src/service_status.txt\\n' > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.txt tests/test_service.sh && git commit -m 'baseline line edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.txt", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_line_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.txt"],
                "preserved_paths": ["tests/test_service.sh"],
                "test_commands": [
                    {"label": "service suite", "argv": ["tests/test_service.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("line_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "line_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_line_edit_parallel"
        / "clones"
        / "worker_service-ready"
    )
    assert (workspace / "src" / "service_status.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready active\nFOOTER=keep\n"
    )


def test_kernel_runs_token_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["token_edit_integrator"] = TaskSpec(
        task_id="token_edit_integrator",
        prompt="accept one worker branch with inline token update",
        workspace_subdir="token_edit_integrator",
        expected_files=["src/service_status.js", "tests/test_service.sh"],
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'export const serviceStatus = \"broken\";\\nexport const footer = \"keep\";\\n' > src/service_status.js && printf \"#!/bin/sh\\nset -eu\\ngrep -q 'serviceStatus = \\\"ready\\\"' src/service_status.js\\n\" > tests/test_service.sh && chmod +x tests/test_service.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/service_status.js tests/test_service.sh && git commit -m 'baseline token edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/service_status.js", "tests/test_service.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_token_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/service-ready"],
                "expected_changed_paths": ["src/service_status.js"],
                "preserved_paths": ["tests/test_service.sh"],
                "test_commands": [
                    {"label": "service suite", "argv": ["tests/test_service.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("token_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "token_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_token_edit_parallel"
        / "clones"
        / "worker_service-ready"
    )
    assert (workspace / "src" / "service_status.js").read_text(encoding="utf-8") == (
        'export const serviceStatus = "ready";\nexport const footer = "keep";\n'
    )


def test_kernel_runs_block_replace_synthesized_worker_task_when_git_policy_enabled(tmp_path):
    bank = TaskBank()
    bank._tasks["block_edit_integrator"] = TaskSpec(
        task_id="block_edit_integrator",
        prompt="accept one worker branch with contiguous block update",
        workspace_subdir="block_edit_integrator",
        expected_files=["src/release_notes.txt", "tests/test_release.sh"],
        expected_file_contents={
            "src/release_notes.txt": "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n",
        },
        metadata={
            "benchmark_family": "repo_sandbox",
            "capability": "repo_environment",
            "shared_repo_order": 1,
            "shared_repo_bootstrap_commands": [
                "mkdir -p src tests && printf 'HEADER=stable\\nITEM=pending\\nITEM=pending\\nFOOTER=keep\\n' > src/release_notes.txt && printf '#!/bin/sh\\nset -eu\\ngrep -q \"^release-ready line one$\" src/release_notes.txt\\ngrep -q \"^release-ready line two$\" src/release_notes.txt\\n' > tests/test_release.sh && chmod +x tests/test_release.sh && git init && git checkout -b main && git config user.email agent@example.com && git config user.name 'Agent Kernel' && git add src/release_notes.txt tests/test_release.sh && git commit -m 'baseline block edit sandbox'"
            ],
            "shared_repo_bootstrap_managed_paths": ["src/release_notes.txt", "tests/test_release.sh"],
            "workflow_guard": {
                "requires_git": True,
                "shared_repo_id": "repo_block_edit_parallel",
                "target_branch": "main",
            },
            "semantic_verifier": {
                "kind": "git_repo_review",
                "expected_branch": "main",
                "required_merged_branches": ["worker/release-ready"],
                "expected_changed_paths": ["src/release_notes.txt"],
                "preserved_paths": ["tests/test_release.sh"],
                "test_commands": [
                    {"label": "release suite", "argv": ["tests/test_release.sh"]},
                ],
            },
        },
    )
    worker = bank.parallel_worker_tasks("block_edit_integrator")[0]
    assert worker.metadata["synthetic_edit_plan"][0]["edit_kind"] == "block_replace"
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
    )

    episode = AgentKernel(config=config).run_task(worker)

    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_block_edit_parallel"
        / "clones"
        / "worker_release-ready"
    )
    assert (workspace / "src" / "release_notes.txt").read_text(encoding="utf-8") == (
        "HEADER=stable\nrelease-ready line one\nrelease-ready line two\nFOOTER=keep\n"
    )


def test_kernel_solves_generated_conflict_task_when_git_and_generated_policy_enabled(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        unattended_allow_git_commands=True,
        unattended_allow_generated_path_mutations=True,
    )

    kernel = AgentKernel(config=config)
    try:
        worker = kernel.run_task(TaskBank().get("git_conflict_worker_status_task"))
        episode = kernel.run_task(TaskBank().get("git_generated_conflict_resolution_task"))
    finally:
        kernel.close()

    assert worker.success is True
    assert episode.success is True
    workspace = (
        config.workspace_root
        / "_shared_repo_runtime"
        / "repo_sandbox_generated_conflict"
        / "clones"
        / "main"
    )
    assert (workspace / "dist" / "status_bundle.txt").read_text(encoding="utf-8") == (
        "SERVICE_STATUS=resolved\nnotes ready\n"
    )


class RepeatingFailPolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="repeat the same broken command",
            action="code_execute",
            content="false",
            done=False,
        )


class InferenceErrorPolicy(Policy):
    def decide(self, state):
        del state
        raise RuntimeError("vLLM request failed after 2 attempts: connection refused")


class InterruptAfterFirstStepPolicy(Policy):
    def decide(self, state):
        if not state.history:
            return ActionDecision(
                thought="write the first artifact",
                action="code_execute",
                content="printf 'one\\n' > one.txt",
                done=False,
            )
        raise KeyboardInterrupt("simulated interrupt")


class ResumeAfterCheckpointPolicy(Policy):
    def decide(self, state):
        if not state.history:
            return ActionDecision(
                thought="write the first artifact",
                action="code_execute",
                content="printf 'one\\n' > one.txt",
                done=False,
            )
        return ActionDecision(
            thought="write the second artifact",
            action="code_execute",
            content="printf 'two\\n' > two.txt",
            done=False,
        )


class SetupResumePolicy(Policy):
    def decide(self, state):
        del state
        return ActionDecision(
            thought="finish after resumed setup",
            action="code_execute",
            content="printf 'done\\n' > done.txt",
            done=False,
        )


class NoProgressPolicy(Policy):
    def decide(self, state):
        step = len(state.history) + 1
        return ActionDecision(
            thought="make an irrelevant edit",
            action="code_execute",
            content=f"printf 'noise {step}\\n' >> scratch{step}.log",
            done=False,
        )


class FailingTolbertContextProvider:
    def compile(self, state):
        del state
        raise RuntimeError("tolbert service unavailable")


def test_kernel_stops_repeated_failed_action(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="repeat_fail",
        prompt="run a command that succeeds",
        workspace_subdir="repeat_fail",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=RepeatingFailPolicy()).run_task(task)

    assert episode.success is False
    assert episode.termination_reason == "repeated_failed_action"
    assert len(episode.steps) == 2


def test_kernel_stops_after_repeated_no_state_progress(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="no_progress",
        prompt="create hello.txt",
        workspace_subdir="no_progress",
        expected_files=["hello.txt"],
        expected_file_contents={"hello.txt": "hello\n"},
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=NoProgressPolicy()).run_task(task)

    assert episode.success is False
    assert episode.termination_reason == "no_state_progress"
    assert len(episode.steps) == 3
    assert all(step.state_progress_delta == 0.0 for step in episode.steps)
    assert all(step.failure_signals == ["no_state_progress"] for step in episode.steps)


def test_kernel_records_typed_policy_failure_signals(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="policy_error",
        prompt="complete a task",
        workspace_subdir="policy_error",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=InferenceErrorPolicy()).run_task(task)
    stored = json.loads((config.trajectories_root / "policy_error.json").read_text(encoding="utf-8"))

    assert episode.success is False
    assert episode.termination_reason == "policy_terminated"
    assert episode.steps[0].failure_origin == "inference_failure"
    assert episode.steps[0].failure_signals == ["inference_failure"]
    assert "inference_failure" in stored["summary"]["failure_signals"]


def test_kernel_records_tolbert_compile_failure_as_retrieval_failure(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="tolbert_failure",
        prompt="complete a task",
        workspace_subdir="tolbert_failure",
        max_steps=5,
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FailingTolbertContextProvider(),
        config=config,
    )

    episode = AgentKernel(config=config, policy=policy).run_task(task)
    stored = json.loads((config.trajectories_root / "tolbert_failure.json").read_text(encoding="utf-8"))

    assert episode.success is False
    assert episode.termination_reason == "policy_terminated"
    assert episode.steps[0].failure_origin == "retrieval_failure"
    assert episode.steps[0].failure_signals == ["retrieval_failure"]
    assert "context packet compilation failed" in episode.steps[0].content
    assert "retrieval_failure" in stored["summary"]["failure_signals"]


def test_kernel_can_resume_from_checkpoint_after_interrupt(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="resume_task",
        prompt="create one.txt and two.txt",
        workspace_subdir="resume_task",
        expected_files=["one.txt", "two.txt"],
        expected_file_contents={"one.txt": "one\n", "two.txt": "two\n"},
        max_steps=5,
    )
    checkpoint_path = tmp_path / "checkpoints" / "resume_task.json"

    with pytest.raises(KeyboardInterrupt):
        AgentKernel(config=config, policy=InterruptAfterFirstStepPolicy()).run_task(
            task,
            checkpoint_path=checkpoint_path,
        )

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "in_progress"
    assert len(checkpoint_payload["history"]) == 1
    assert "workspace_snapshot" in checkpoint_payload
    assert "latest_state_transition" in checkpoint_payload

    resumed = AgentKernel(config=config, policy=ResumeAfterCheckpointPolicy()).run_task(
        task,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed.success is True
    assert len(resumed.steps) == 2
    assert resumed.steps[0].content == "printf 'one\\n' > one.txt"
    assert resumed.steps[1].content == "printf 'two\\n' > two.txt"
    assert (config.workspace_root / "resume_task" / "one.txt").read_text(encoding="utf-8") == "one\n"
    assert (config.workspace_root / "resume_task" / "two.txt").read_text(encoding="utf-8") == "two\n"


def test_kernel_can_resume_after_setup_phase_interrupt(tmp_path):
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="resume_setup_task",
        prompt="prepare seed.txt and prep.txt before creating done.txt",
        workspace_subdir="resume_setup_task",
        setup_commands=[
            "printf 'seed\\n' > seed.txt",
            "printf 'prep\\n' > prep.txt",
        ],
        expected_files=["seed.txt", "prep.txt", "done.txt"],
        expected_file_contents={
            "seed.txt": "seed\n",
            "prep.txt": "prep\n",
            "done.txt": "done\n",
        },
        max_steps=5,
    )
    checkpoint_path = tmp_path / "checkpoints" / "resume_setup_task.json"
    kernel = AgentKernel(config=config, policy=SetupResumePolicy())
    real_run = kernel.sandbox.run
    calls = {"count": 0}

    def interrupt_during_second_setup(command, cwd, *, task=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt("setup interrupted")
        return real_run(command, cwd, task=task)

    kernel.sandbox.run = interrupt_during_second_setup  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt):
        kernel.run_task(task, checkpoint_path=checkpoint_path)

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "setup_in_progress"
    assert checkpoint_payload["phase"] == "setup"
    assert len(checkpoint_payload["setup_history"]) == 1

    resumed = AgentKernel(config=config, policy=SetupResumePolicy()).run_task(
        task,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed.success is True
    workspace = config.workspace_root / "resume_setup_task"
    assert (workspace / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert (workspace / "prep.txt").read_text(encoding="utf-8") == "prep\n"
    assert (workspace / "done.txt").read_text(encoding="utf-8") == "done\n"


def test_kernel_build_plan_includes_repo_workflow_steps(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planner_controls": {
                    "prepend_verifier_contract_check": True,
                    "append_validation_subgoal": True,
                    "prefer_expected_artifacts_first": True,
                    "max_initial_subgoals": 10,
                },
                "role_directives": {},
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        prompt_proposals_path=prompt_path,
    )
    kernel = AgentKernel(config=config)

    plan = kernel._build_plan(TaskBank().get("git_repo_test_repair_task"))

    assert "prepare workflow branch fix/release-ready" in plan
    assert "update workflow path src/release_state.txt" in plan
    assert "run workflow test release test script" in plan


def test_graph_memory_summarizes_prior_documents(tmp_path):
    from agent_kernel.memory import EpisodeMemory
    from agent_kernel.schemas import EpisodeRecord, StepRecord

    memory = EpisodeMemory(tmp_path / "trajectories")
    memory.save(
        EpisodeRecord(
            task_id="hello_task",
            prompt="Create hello.txt containing hello agent kernel.",
            workspace=str(tmp_path / "workspace" / "hello_task"),
            success=True,
            steps=[
                StepRecord(
                    index=1,
                    thought="write file",
                    action="code_execute",
                    content="printf 'hello agent kernel\\n' > hello.txt",
                    selected_skill_id=None,
                    command_result=None,
                    verification={"passed": True, "reasons": ["verification passed"]},
                )
            ],
            task_metadata={"benchmark_family": "micro"},
        )
    )

    summary = memory.graph_summary("hello_task_followup")

    assert summary["document_count"] == 1
    assert summary["benchmark_families"]["micro"] == 1
    assert summary["failure_types"] == {"other": 1}
    assert summary["related_tasks"] == ["hello_task"]


def test_role_specialization_reaches_critic_on_failure(tmp_path):
    from agent_kernel.schemas import TaskSpec

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        max_steps=5,
    )
    task = TaskSpec(
        task_id="critic_fail",
        prompt="run a command that succeeds",
        workspace_subdir="critic_fail",
        max_steps=5,
    )

    episode = AgentKernel(config=config, policy=RepeatingFailPolicy()).run_task(task)

    assert any(step.acting_role == "critic" for step in episode.steps[1:])


def test_kernel_builds_vllm_provider(monkeypatch, tmp_path):
    captured = {}

    class FakeVLLMClient:
        def __init__(self, host, model_name, timeout_seconds, retry_attempts, retry_backoff_seconds, api_key):
            captured["host"] = host
            captured["model_name"] = model_name
            captured["timeout_seconds"] = timeout_seconds
            captured["retry_attempts"] = retry_attempts
            captured["retry_backoff_seconds"] = retry_backoff_seconds
            captured["api_key"] = api_key

    monkeypatch.setattr("agent_kernel.loop.VLLMClient", FakeVLLMClient)
    config = KernelConfig(
        provider="vllm",
        use_tolbert_context=False,
        workspace_root=tmp_path / "workspace",
        trajectories_root=tmp_path / "trajectories",
        vllm_host="http://127.0.0.1:8000",
        vllm_api_key="secret",
    )

    kernel = AgentKernel(config=config)

    assert kernel.policy.client.__class__ is FakeVLLMClient
    assert captured["host"] == "http://127.0.0.1:8000"
    assert captured["model_name"] == config.model_name
    assert captured["api_key"] == "secret"


def test_kernel_matches_retrieval_command_when_guidance_uses_literal_newline():
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-03-16T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "retrieval_guidance": {
                "recommended_commands": ["printf 'hello agent kernel\n' > hello.txt"],
                "recommended_command_spans": [
                    {
                        "span_id": "task:hello:literal-newline",
                        "command": "printf 'hello agent kernel\n' > hello.txt",
                    }
                ],
                "avoidance_notes": [],
                "evidence": [],
            }
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    assert AgentKernel._retrieval_command_match(
        state,
        "printf 'hello agent kernel\\n' > hello.txt",
    )
