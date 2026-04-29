from __future__ import annotations

from agent_kernel.integrations.terminal_bench_harbor import (
    AgentKernelTerminalBenchAgent,
    build_terminal_bench_config,
    build_terminal_bench_task,
)


def test_terminal_bench_task_wraps_instruction() -> None:
    task = build_terminal_bench_task("Create /tmp/example.txt", max_steps=7)

    assert task.task_id == "terminal_bench_task"
    assert task.max_steps == 7
    assert "Terminal-Bench" in task.prompt
    assert "Create /tmp/example.txt" in task.prompt
    assert task.metadata["benchmark_family"] == "terminal-bench"


def test_terminal_bench_config_uses_harbor_model_name(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_PROVIDER", "vllm")
    monkeypatch.setenv("AGENT_KERNEL_TBENCH_MAX_STEPS", "13")

    config = build_terminal_bench_config("Qwen/Qwen3-Coder-480B-A35B-Instruct")

    assert config.provider == "vllm"
    assert config.model_name == "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    assert config.max_steps == 13
    assert not config.use_tolbert_context
    assert not config.vllm_autostart


def test_terminal_bench_agent_imports_without_harbor(tmp_path) -> None:
    agent = AgentKernelTerminalBenchAgent(logs_dir=tmp_path, model_name="qwen")

    assert agent.name() == "agent_kernel"
    assert agent.model_name == "qwen"
