from pathlib import Path
import importlib.util
from io import StringIO
import json
import os
import sys

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementVariant
from agent_kernel.ops.preflight import PreflightCheck, PreflightReport
from agent_kernel.schemas import TaskSpec
from evals.metrics import EvalMetrics
import pytest


def test_kernel_config_allows_runtime_toggles():
    config = KernelConfig(
        use_tolbert_context=False,
        use_skills=False,
        use_graph_memory=False,
        use_world_model=False,
        use_universe_model=False,
        use_planner=False,
        use_role_specialization=False,
        persist_learning_candidates=False,
    )

    assert config.use_tolbert_context is False
    assert config.use_skills is False
    assert config.use_graph_memory is False
    assert config.use_world_model is False
    assert config.use_universe_model is False
    assert config.use_planner is False
    assert config.use_role_specialization is False
    assert config.persist_learning_candidates is False
    assert str(config.skills_path).endswith("command_skills.json")
    assert str(config.operator_classes_path).endswith("operator_classes.json")
    assert str(config.tool_candidates_path).endswith("tool_candidates.json")
    assert str(config.benchmark_candidates_path).endswith("benchmark_candidates.json")
    assert str(config.retrieval_proposals_path).endswith("retrieval_proposals.json")
    assert str(config.retrieval_asset_bundle_path).endswith("retrieval_asset_bundle.json")
    assert str(config.tolbert_liftoff_report_path).endswith("liftoff_gate_report.json")
    assert str(config.verifier_contracts_path).endswith("verifier_contracts.json")
    assert str(config.prompt_proposals_path).endswith("prompt_proposals.json")
    assert str(config.world_model_proposals_path).endswith("world_model_proposals.json")
    assert str(config.trust_proposals_path).endswith("trust_proposals.json")
    assert str(config.recovery_proposals_path).endswith("recovery_proposals.json")
    assert str(config.delegation_proposals_path).endswith("delegation_proposals.json")
    assert str(config.operator_policy_proposals_path).endswith("operator_policy_proposals.json")
    assert str(config.transition_model_proposals_path).endswith("transition_model_proposals.json")
    assert str(config.curriculum_proposals_path).endswith("curriculum_proposals.json")
    assert str(config.improvement_cycles_path).endswith("cycles.jsonl")
    assert str(config.run_reports_dir).endswith("reports")
    assert str(config.capability_modules_path).endswith("config/capabilities.json")
    assert str(config.run_checkpoints_dir).endswith("checkpoints")
    assert str(config.delegated_job_queue_path).endswith("jobs/queue.json")
    assert str(config.delegated_job_runtime_state_path).endswith("jobs/runtime_state.json")
    assert config.max_steps == 12
    assert config.delegated_job_max_concurrency == 3
    assert config.delegated_job_max_active_per_budget_group == 2
    assert config.delegated_job_max_queued_per_budget_group == 8
    assert config.delegated_job_max_artifact_bytes == 8 * 1024 * 1024
    assert config.delegated_job_max_subprocesses_per_job == 2
    assert config.delegated_job_max_consecutive_selections_per_budget_group == 0
    assert "repo_chore" in config.unattended_allowed_benchmark_families
    assert "repo_sandbox" in config.unattended_allowed_benchmark_families
    assert config.unattended_allow_git_commands is False
    assert config.unattended_allow_http_requests is False
    assert config.unattended_http_allowed_hosts == ()
    assert config.unattended_http_timeout_seconds == 10
    assert config.unattended_http_max_body_bytes == 64 * 1024
    assert config.unattended_allow_generated_path_mutations is False
    assert "build" in config.unattended_generated_path_prefixes
    assert str(config.unattended_workspace_snapshot_root).endswith("recovery/workspaces")
    assert config.unattended_rollback_on_failure is True
    assert config.unattended_trust_enforce is True
    assert str(config.unattended_trust_ledger_path).endswith("reports/unattended_trust_ledger.json")
    assert config.unattended_trust_recent_report_limit == 50
    assert "repo_sandbox" in config.unattended_trust_required_benchmark_families
    assert config.unattended_trust_bootstrap_min_reports == 5
    assert config.unattended_trust_breadth_min_reports == 10
    assert config.unattended_trust_min_distinct_families == 2
    assert config.unattended_trust_min_success_rate == 0.7


def test_kernel_config_executable_floor_preset():
    config = KernelConfig.executable_floor()

    assert config.use_tolbert_context is False
    assert config.use_skills is False
    assert config.use_graph_memory is False
    assert config.use_world_model is False
    assert config.use_universe_model is False
    assert config.use_planner is False
    assert config.use_role_specialization is False
    assert config.persist_learning_candidates is False
    assert config.persist_episode_memory is True


def test_kernel_config_qwen_tolbert_reference_implementation_preset():
    config = KernelConfig.qwen_tolbert_reference_implementation()

    assert config.provider == "vllm"
    assert config.model_name == "Qwen/Qwen3.5-9B"
    assert config.use_tolbert_context is True
    assert config.use_paper_research_context is False
    assert config.use_skills is False
    assert config.use_graph_memory is True
    assert config.use_world_model is True
    assert config.use_universe_model is True
    assert config.use_planner is False
    assert config.use_role_specialization is False
    assert config.use_prompt_proposals is False
    assert config.use_curriculum_proposals is False
    assert config.use_retrieval_proposals is False
    assert config.use_state_estimation_proposals is False
    assert config.use_trust_proposals is False
    assert config.use_recovery_proposals is False
    assert config.use_delegation_proposals is False
    assert config.use_operator_policy_proposals is False
    assert config.use_transition_model_proposals is False
    assert config.use_tolbert_model_artifacts is False
    assert config.persist_episode_memory is True
    assert config.persist_learning_candidates is True
    assert config.asi_coding_require_live_llm is True


def test_kernel_config_defaults_to_vllm_runtime(monkeypatch):
    for name in ("AGENT_KERNEL_PROVIDER", "AGENT_KERNEL_MODEL", "AGENT_KERNEL_VLLM_HOST"):
        monkeypatch.delenv(name, raising=False)

    config = KernelConfig()

    assert config.provider == "vllm"
    assert config.model_name == "Qwen/Qwen3.5-9B"
    assert config.vllm_host == "http://127.0.0.1:8000"


def test_kernel_config_to_env_exports_full_runtime_surface(tmp_path):
    config = KernelConfig(
        model_name="Qwen/test",
        tolbert_confidence_threshold=0.42,
        tolbert_source_spans_paths=(str(tmp_path / "one.jsonl"), str(tmp_path / "two.jsonl")),
        external_task_manifests_paths=(str(tmp_path / "tasks.json"),),
        unattended_http_allowed_hosts=("api.github.com", "example.com"),
        capability_modules_path=tmp_path / "config" / "capabilities.json",
        world_model_proposals_path=tmp_path / "world_model" / "world_model_proposals.json",
        trust_proposals_path=tmp_path / "trust" / "trust_proposals.json",
        recovery_proposals_path=tmp_path / "recovery" / "recovery_proposals.json",
        delegation_proposals_path=tmp_path / "delegation" / "delegation_proposals.json",
        operator_policy_proposals_path=tmp_path / "operator_policy" / "operator_policy_proposals.json",
        transition_model_proposals_path=tmp_path / "transition_model" / "transition_model_proposals.json",
        unattended_workspace_snapshot_root=tmp_path / "recovery" / "workspaces",
        unattended_trust_ledger_path=tmp_path / "reports" / "trust.json",
    )

    env = config.to_env()

    assert env["AGENT_KERNEL_MODEL"] == "Qwen/test"
    assert env["AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD"] == "0.42"
    assert env["AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS"] == os.pathsep.join(
        [str(tmp_path / "one.jsonl"), str(tmp_path / "two.jsonl")]
    )
    assert env["AGENT_KERNEL_EXTERNAL_TASK_MANIFESTS_PATHS"] == str(tmp_path / "tasks.json")
    assert env["AGENT_KERNEL_UNATTENDED_HTTP_ALLOWED_HOSTS"] == "api.github.com,example.com"
    assert env["AGENT_KERNEL_CAPABILITY_MODULES_PATH"].endswith("config/capabilities.json")
    assert env["AGENT_KERNEL_WORLD_MODEL_PROPOSALS_PATH"].endswith("world_model/world_model_proposals.json")
    assert env["AGENT_KERNEL_TRUST_PROPOSALS_PATH"].endswith("trust/trust_proposals.json")
    assert env["AGENT_KERNEL_RECOVERY_PROPOSALS_PATH"].endswith("recovery/recovery_proposals.json")
    assert env["AGENT_KERNEL_TOLBERT_LIFTOFF_REPORT_PATH"].endswith("tolbert_model/liftoff_gate_report.json")
    assert env["AGENT_KERNEL_DELEGATION_PROPOSALS_PATH"].endswith("delegation/delegation_proposals.json")
    assert env["AGENT_KERNEL_OPERATOR_POLICY_PROPOSALS_PATH"].endswith(
        "operator_policy/operator_policy_proposals.json"
    )
    assert env["AGENT_KERNEL_TRANSITION_MODEL_PROPOSALS_PATH"].endswith(
        "transition_model/transition_model_proposals.json"
    )
    assert env["AGENT_KERNEL_UNATTENDED_WORKSPACE_SNAPSHOT_ROOT"].endswith("recovery/workspaces")
    assert env["AGENT_KERNEL_UNATTENDED_TRUST_LEDGER_PATH"].endswith("reports/trust.json")
    assert env["AGENT_KERNEL_DELEGATED_JOB_MAX_CONSECUTIVE_SELECTIONS_PER_BUDGET_GROUP"] == "0"


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_agent_cli_applies_proposal_toggles(monkeypatch, tmp_path):
    module = _load_script_module("run_agent.py")
    seen = {}

    class FakeTaskBank:
        def get(self, task_id):
            return TaskSpec(
                task_id=task_id,
                prompt="create ok.txt",
                workspace_subdir=task_id,
                expected_files=["ok.txt"],
            )

    class FakeKernel:
        def __init__(self, config):
            seen["config"] = config

        def run_task(self, task, **kwargs):
            seen["task"] = task
            seen["run_kwargs"] = kwargs
            return type("Episode", (), {"task_id": "hello_task", "success": True, "steps": [object()]})()

    monkeypatch.setattr(module, "TaskBank", FakeTaskBank)
    monkeypatch.setattr(module, "AgentKernel", FakeKernel)
    monkeypatch.setattr(
        module,
        "run_unattended_preflight",
        lambda config, task, repo_root: PreflightReport(
            passed=True,
            checked_at="2026-03-24T00:00:00+00:00",
            checks=[PreflightCheck(name="provider_health", passed=True, detail="ok")],
        ),
    )
    monkeypatch.setattr(
        module,
        "write_unattended_task_report",
        lambda **kwargs: tmp_path / "task_report.json",
    )
    monkeypatch.setattr(
        module,
        "classify_run_outcome",
        lambda **kwargs: ("success", ["verification_passed"]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent.py",
            "--task-id",
            "hello_task",
            "--use-prompt-proposals",
            "0",
            "--use-curriculum-proposals",
            "0",
            "--use-retrieval-proposals",
            "0",
            "--use-transition-model-proposals",
            "0",
            "--allow-http-requests",
            "1",
        ],
    )
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    module.main()

    config = seen["config"]
    assert config.use_prompt_proposals is False
    assert config.use_curriculum_proposals is False
    assert config.use_retrieval_proposals is False
    assert config.use_transition_model_proposals is False
    assert config.unattended_allow_http_requests is True
    assert seen["run_kwargs"]["resume"] is False
    assert str(seen["run_kwargs"]["checkpoint_path"]).endswith("hello_task.json")
    assert stream.getvalue().strip() == "task=hello_task success=True steps=1"
    assert "outcome=success report=" in err_stream.getvalue()


def test_run_agent_cli_safe_stops_on_failed_preflight(monkeypatch, tmp_path):
    module = _load_script_module("run_agent.py")
    seen = {"kernel_called": False}

    class FakeTaskBank:
        def get(self, task_id):
            return TaskSpec(
                task_id=task_id,
                prompt="create ok.txt",
                workspace_subdir=task_id,
                expected_files=["ok.txt"],
            )

    class FakeKernel:
        def __init__(self, config):
            del config

        def run_task(self, task, **kwargs):
            del task
            del kwargs
            seen["kernel_called"] = True
            raise AssertionError("kernel should not run when preflight fails")

    monkeypatch.setattr(module, "TaskBank", FakeTaskBank)
    monkeypatch.setattr(module, "AgentKernel", FakeKernel)
    monkeypatch.setattr(
        module,
        "run_unattended_preflight",
        lambda config, task, repo_root: PreflightReport(
            passed=False,
            checked_at="2026-03-24T00:00:00+00:00",
            checks=[PreflightCheck(name="provider_health", passed=False, detail="probe failed")],
        ),
    )
    monkeypatch.setattr(
        module,
        "write_unattended_task_report",
        lambda **kwargs: tmp_path / "task_report.json",
    )
    monkeypatch.setattr(sys, "argv", ["run_agent.py", "--task-id", "hello_task"])
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 2
    assert seen["kernel_called"] is False
    assert stream.getvalue() == ""
    assert "outcome=safe_stop report=" in err_stream.getvalue()


def test_run_minimal_asi_cli_applies_reference_profile(monkeypatch, tmp_path):
    module = _load_script_module("run_minimal_asi.py")
    seen = {"closed": False}

    class FakeTaskBank:
        def __init__(self, config=None):
            seen["task_bank_config"] = config

        def get(self, task_id):
            return TaskSpec(
                task_id=task_id,
                prompt="create ok.txt",
                workspace_subdir=task_id,
                expected_files=["ok.txt"],
            )

    class FakeKernel:
        def __init__(self, config):
            seen["config"] = config

        def run_task(self, task, **kwargs):
            seen["task"] = task
            seen["run_kwargs"] = kwargs
            return type("Episode", (), {"task_id": task.task_id, "success": True, "steps": [object(), object()]})()

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(module, "TaskBank", FakeTaskBank)
    monkeypatch.setattr(module, "AgentKernel", FakeKernel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_minimal_asi.py",
            "--task-id",
            "hello_task",
            "--state-root",
            str(tmp_path / "fresh_state"),
            "--use-graph-memory",
            "0",
            "--persist-learning-candidates",
            "0",
            "--print-profile",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    profile = json.loads(lines[0])
    config = seen["config"]
    state_root = (tmp_path / "fresh_state").resolve()

    assert profile["profile"] == "qwen_tolbert_reference_implementation"
    assert profile["provider"] == "vllm"
    assert profile["decoder_authority"] == "external_decoder"
    assert profile["state_root"] == str(state_root)
    assert config.use_tolbert_context is True
    assert config.use_graph_memory is False
    assert config.use_world_model is True
    assert config.use_universe_model is True
    assert config.persist_learning_candidates is False
    assert config.persist_episode_memory is True
    assert config.workspace_root == state_root / "workspace"
    assert config.capability_modules_path == state_root / "config" / "capabilities.json"
    assert config.runtime_database_path == state_root / "var" / "runtime" / "agentkernel.sqlite3"
    assert seen["task_bank_config"] is config
    assert seen["run_kwargs"]["resume"] is False
    assert seen["run_kwargs"]["checkpoint_path"] == state_root / "trajectories" / "checkpoints" / "hello_task.json"
    assert seen["closed"] is True
    assert (
        lines[1]
        == "profile=qwen_tolbert_reference_implementation runtime_shape=executable_floor "
        "decoder_authority=external_decoder task=hello_task success=True steps=2"
    )


def test_run_eval_cli_applies_extended_runtime_toggles(monkeypatch):
    module = _load_script_module("run_eval.py")
    seen = {}

    monkeypatch.setattr(
        module,
        "run_eval",
        lambda *, config, **kwargs: seen.update({"config": config, "kwargs": kwargs}) or EvalMetrics(total=1, passed=1),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--use-trust-proposals",
            "0",
            "--use-recovery-proposals",
            "0",
            "--use-delegation-proposals",
            "0",
            "--use-operator-policy-proposals",
            "0",
            "--use-transition-model-proposals",
            "0",
            "--allow-git-commands",
            "1",
            "--allow-http-requests",
            "1",
            "--allow-generated-path-mutations",
            "1",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    config = seen["config"]
    assert config.use_trust_proposals is False
    assert config.use_recovery_proposals is False
    assert config.use_delegation_proposals is False
    assert config.use_operator_policy_proposals is False
    assert config.use_transition_model_proposals is False
    assert config.unattended_allow_git_commands is True
    assert config.unattended_allow_http_requests is True
    assert config.unattended_allow_generated_path_mutations is True
    assert "pass_rate=" in stream.getvalue()


def test_run_agent_cli_writes_report_on_runner_exception(monkeypatch, tmp_path):
    module = _load_script_module("run_agent.py")

    class FakeTaskBank:
        def get(self, task_id):
            return TaskSpec(
                task_id=task_id,
                prompt="create ok.txt",
                workspace_subdir=task_id,
                expected_files=["ok.txt"],
                metadata={"benchmark_family": "repo_chore"},
            )

    class FailingKernel:
        def __init__(self, config):
            del config

        def run_task(self, task, **kwargs):
            del task
            del kwargs
            raise RuntimeError("boom")

    def fake_write_report(**kwargs):
        del kwargs
        path = tmp_path / "task_report.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        module,
        "KernelConfig",
        lambda: KernelConfig(
            provider="mock",
            use_tolbert_context=False,
            workspace_root=tmp_path / "workspace",
            trajectories_root=tmp_path / "trajectories" / "episodes",
            run_reports_dir=tmp_path / "trajectories" / "reports",
            run_checkpoints_dir=tmp_path / "trajectories" / "checkpoints",
            unattended_trust_ledger_path=tmp_path / "trajectories" / "reports" / "trust.json",
        ),
    )
    monkeypatch.setattr(module, "TaskBank", FakeTaskBank)
    monkeypatch.setattr(module, "AgentKernel", FailingKernel)
    monkeypatch.setattr(
        module,
        "run_unattended_preflight",
        lambda config, task, repo_root: PreflightReport(
            passed=True,
            checked_at="2026-03-24T00:00:00+00:00",
            checks=[PreflightCheck(name="provider_health", passed=True, detail="ok")],
        ),
    )
    monkeypatch.setattr(module, "write_unattended_task_report", fake_write_report)
    monkeypatch.setattr(sys, "argv", ["run_agent.py", "--task-id", "hello_task"])
    stream = StringIO()
    err_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err_stream)

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 1
    assert stream.getvalue() == ""
    assert "outcome=unsafe_ambiguous report=" in err_stream.getvalue()


def test_run_eval_cli_applies_proposal_toggles(monkeypatch):
    module = _load_script_module("run_eval.py")
    seen = {}

    def fake_run_eval(*, config, **kwargs):
        seen["config"] = config
        seen["kwargs"] = kwargs
        return type(
            "Metrics",
            (),
            {
                "passed": 1,
                "total": 1,
                "pass_rate": 1.0,
                "average_steps": 1.0,
                "average_success_steps": 1.0,
                "skill_selected_steps": 0,
                "episodes_with_skill_use": 0,
                "skill_use_rate": 0.0,
                "available_but_unused_skill_steps": 0,
                "excess_available_skill_slots": 0,
                "average_available_skills": 0.0,
                "average_retrieval_candidates": 0.0,
                "average_retrieval_evidence": 0.0,
                "average_retrieval_direct_candidates": 0.0,
                "retrieval_guided_steps": 0,
                "retrieval_selected_steps": 0,
                "retrieval_influenced_steps": 0,
                "retrieval_ranked_skill_steps": 0,
                "trusted_retrieval_steps": 0,
                "low_confidence_steps": 0,
                "first_step_successes": 0,
                "average_first_step_path_confidence": 0.0,
                "success_first_step_path_confidence": 0.0,
                "failed_first_step_path_confidence": 0.0,
                "low_confidence_episodes": 0,
                "low_confidence_episode_pass_rate": 0.0,
                "memory_documents": 0,
                "reusable_skills": 0,
                "total_by_capability": {},
                "passed_by_capability": {},
                "total_by_difficulty": {},
                "passed_by_difficulty": {},
                "total_by_benchmark_family": {},
                "passed_by_benchmark_family": {},
                "total_by_memory_source": {},
                "passed_by_memory_source": {},
                "total_by_origin_benchmark_family": {},
                "passed_by_origin_benchmark_family": {},
                "generated_total": 0,
                "generated_passed": 0,
                "generated_pass_rate": 0.0,
                "generated_by_kind": {},
                "generated_passed_by_kind": {},
                "generated_by_benchmark_family": {},
                "generated_passed_by_benchmark_family": {},
                "termination_reasons": {},
            },
        )()

    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--use-prompt-proposals",
            "0",
            "--use-curriculum-proposals",
            "0",
            "--use-retrieval-proposals",
            "0",
            "--include-discovered-tasks",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    module.main()

    config = seen["config"]
    assert config.use_prompt_proposals is False
    assert config.use_curriculum_proposals is False
    assert config.use_retrieval_proposals is False
    assert seen["kwargs"]["include_discovered_tasks"] is True
    output = stream.getvalue()
    assert "passed=1 total=1 pass_rate=1.00" in output


def test_run_repeated_improvement_cycles_streams_child_output(monkeypatch, tmp_path):
    module = _load_script_module("run_repeated_improvement_cycles.py")
    seen = {}

    class FakeStdout:
        def __iter__(self):
            return iter(["[cycle:cycle-1] task 1/1 hello_task\n", "subsystem=policy final_state=retain\n"])

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, bufsize, env=None):
            seen["cmd"] = cmd
            seen["cwd"] = cwd
            seen["env"] = env
            self.stdout = FakeStdout()

        def wait(self):
            return 0

    class FakePlanner:
        def __init__(
            self,
            memory_root=None,
            cycles_path=None,
            prompt_proposals_path=None,
            use_prompt_proposals=True,
            capability_modules_path=None,
            trust_ledger_path=None,
            runtime_config=None,
        ):
            self.memory_root = memory_root
            self.cycles_path = cycles_path
            self.prompt_proposals_path = prompt_proposals_path
            self.use_prompt_proposals = use_prompt_proposals
            self.capability_modules_path = capability_modules_path
            self.trust_ledger_path = trust_ledger_path
            self.runtime_config = runtime_config

        def incomplete_cycle_summaries(self, output_path, protocol=None):
            del output_path, protocol
            return []

        def retained_gain_summary(self, output_path):
            return type(
                "Summary",
                (),
                {
                    "retained_cycles": 0,
                    "rejected_cycles": 0,
                    "total_decisions": 0,
                    "retained_by_subsystem": {},
                    "rejected_by_subsystem": {},
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 0.0,
                    "average_rejected_pass_rate_delta": 0.0,
                    "average_rejected_step_delta": 0.0,
                },
            )()

        def load_cycle_records(self, output_path):
            return []

        def append_cycle_record(self, output_path, record, govern_exports=True):
            del govern_exports
            seen["record"] = record
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            return output_path

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(module, "ImprovementPlanner", FakePlanner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--task-limit",
            "64",
            "--tolbert-device",
            "cuda:2",
            "--provider",
            "vllm",
            "--model",
            "Qwen/Qwen3.5-9B",
        ],
    )
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.chdir(tmp_path)

    module.main()

    assert "--progress-label" in seen["cmd"]
    assert "cycle-1" in seen["cmd"]
    assert "--task-limit" in seen["cmd"]
    assert "64" in seen["cmd"]
    assert "--tolbert-device" in seen["cmd"]
    assert "cuda:2" in seen["cmd"]
    assert seen["env"]["AGENT_KERNEL_USE_PROMPT_PROPOSALS"] == "1"
    assert seen["env"]["AGENT_KERNEL_USE_CURRICULUM_PROPOSALS"] == "1"
    assert seen["env"]["AGENT_KERNEL_USE_RETRIEVAL_PROPOSALS"] == "1"
    assert seen["env"]["AGENT_KERNEL_TOLBERT_DEVICE"] == "cuda:2"
    assert "AGENT_KERNEL_TOLBERT_CONFIDENCE_THRESHOLD" in seen["env"]
    assert "AGENT_KERNEL_CAPABILITY_MODULES_PATH" in seen["env"]
    output = stream.getvalue()
    assert "[cycle:cycle-1] task 1/1 hello_task" in output
    assert "subsystem=policy final_state=retain" in output


def test_run_repeated_improvement_cycles_semantic_only_runtime_disables_tolbert(monkeypatch, tmp_path):
    module = _load_script_module("run_repeated_improvement_cycles.py")
    seen = {}

    class FakeStdout:
        def __iter__(self):
            return iter(["subsystem=retrieval final_state=reject\n"])

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, bufsize, env=None):
            seen["cmd"] = cmd
            seen["cwd"] = cwd
            seen["env"] = env
            self.stdout = FakeStdout()

        def wait(self):
            return 0

    class FakePlanner:
        def __init__(
            self,
            memory_root=None,
            cycles_path=None,
            prompt_proposals_path=None,
            use_prompt_proposals=True,
            capability_modules_path=None,
            trust_ledger_path=None,
            runtime_config=None,
        ):
            del memory_root, cycles_path, prompt_proposals_path, use_prompt_proposals
            del capability_modules_path, trust_ledger_path, runtime_config

        def incomplete_cycle_summaries(self, output_path, protocol=None):
            del output_path, protocol
            return []

        def retained_gain_summary(self, output_path):
            del output_path
            return type(
                "Summary",
                (),
                {
                    "retained_cycles": 0,
                    "rejected_cycles": 1,
                    "total_decisions": 1,
                    "retained_by_subsystem": {},
                    "rejected_by_subsystem": {"retrieval": 1},
                    "average_retained_pass_rate_delta": 0.0,
                    "average_retained_step_delta": 0.0,
                    "average_rejected_pass_rate_delta": 0.0,
                    "average_rejected_step_delta": 0.0,
                },
            )()

        def load_cycle_records(self, output_path):
            del output_path
            return []

        def append_cycle_record(self, output_path, record, govern_exports=True):
            del record, govern_exports
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            return output_path

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(module, "ImprovementPlanner", FakePlanner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_repeated_improvement_cycles.py",
            "--cycles",
            "1",
            "--semantic-only-runtime",
        ],
    )
    monkeypatch.chdir(tmp_path)

    module.main()

    exclude_indexes = [index for index, token in enumerate(seen["cmd"]) if token == "--exclude-subsystem"]
    assert exclude_indexes
    assert any(seen["cmd"][index + 1] == "tolbert_model" for index in exclude_indexes)
    assert seen["env"]["AGENT_KERNEL_USE_TOLBERT_CONTEXT"] == "0"
    assert seen["env"]["AGENT_KERNEL_USE_TOLBERT_MODEL_ARTIFACTS"] == "0"


def test_run_repeated_improvement_cycles_emits_explicit_timeout_reason(monkeypatch, tmp_path):
    module = _load_script_module("run_repeated_improvement_cycles.py")
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    completed = module._run_and_stream(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('child-start', flush=True); time.sleep(10)",
        ],
        cwd=tmp_path,
        progress_label="cycle-timeout-test",
        heartbeat_interval_seconds=0.3,
        max_silence_seconds=1.0,
    )

    output = stream.getvalue()
    assert completed["returncode"] == -9
    assert completed["timed_out"] is True
    assert completed["timeout_reason"] == "child exceeded max silence of 1 seconds"
    assert "[repeated] child=cycle-timeout-test timeout reason=child exceeded max silence of 1 seconds" in output


def test_run_improvement_cycle_variant_generation_kwargs_use_base_subsystem(tmp_path):
    module = _load_script_module("run_improvement_cycle.py")
    modules_path = tmp_path / "config" / "capabilities.json"
    modules_path.parent.mkdir(parents=True, exist_ok=True)
    modules_path.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "module_id": "github",
                        "enabled": True,
                        "settings": {
                            "improvement_subsystems": [
                                {
                                    "subsystem_id": "github_policy",
                                    "base_subsystem": "policy",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    kwargs = module._variant_generation_kwargs(
        ImprovementVariant("github_policy", "retrieval_caution", "d", 0.01, 2, 0.01, {"focus": "retrieval_caution"}),
        capability_modules_path=modules_path,
    )

    assert kwargs == {"focus": "retrieval_caution"}
