import importlib.util
from pathlib import Path
from types import SimpleNamespace

from agent_kernel.config import KernelConfig
from agent_kernel.improvement import ImprovementExperiment
from evals.metrics import EvalMetrics


def _load_run_improvement_cycle_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location("run_improvement_cycle", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_eval_kwargs_enable_unattended_reports():
    module = _load_run_improvement_cycle_module()
    config = KernelConfig()
    args = module.argparse.Namespace(
        include_episode_memory=False,
        include_skill_memory=False,
        include_skill_transfer=False,
        include_operator_memory=False,
        include_tool_memory=False,
        include_verifier_memory=False,
        include_curriculum=False,
        include_failure_curriculum=False,
        task_limit=0,
        max_observation_seconds=0.0,
        priority_benchmark_family=[],
        priority_benchmark_family_weight=[],
    )

    flags = module._observation_eval_kwargs(config, args)

    assert flags["write_unattended_reports"] is True


def test_generate_observe_hypothesis_passes_max_tokens(monkeypatch):
    module = _load_run_improvement_cycle_module()
    config = KernelConfig(provider="vllm", llm_timeout_seconds=30)
    metrics = EvalMetrics(total=10, passed=7)
    ranked_experiments = [ImprovementExperiment("policy", "policy candidate", 1, 0.1, 1, 0.1, {})]
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs

        def _chat_completion(self, **kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            return {"choices": [{"message": {"content": "{\"hypotheses\": []}"}}]}

    monkeypatch.setattr(module, "ensure_vllm_runtime", lambda config: type("Status", (), {"ready": True})())
    monkeypatch.setattr(module, "VLLMClient", FakeClient)

    payload = module._generate_observe_hypothesis(
        config=config,
        metrics=metrics,
        ranked_experiments=ranked_experiments,
        progress_label=None,
    )

    assert payload["status"] == "generated"
    assert seen["max_tokens"] == 256


def test_priority_family_allocation_summary_tolerates_sparse_weight_map():
    module = _load_run_improvement_cycle_module()
    metrics = SimpleNamespace(
        total_by_benchmark_family={"integration": 1, "repository": 0},
        passed_by_benchmark_family={"integration": 1, "repository": 0},
    )

    summary = module._priority_family_allocation_summary(
        metrics,
        {
            "priority_benchmark_families": ["integration", "repository"],
            "priority_benchmark_family_weights": {"integration": 1.0},
            "task_limit": 2,
        },
    )

    assert summary["priority_benchmark_family_weights"] == {
        "integration": 1.0,
        "repository": 0.0,
    }
    assert summary["planned_weight_shares"] == {
        "integration": 1.0,
        "repository": 0.0,
    }
    assert summary["top_planned_family"] == "integration"
