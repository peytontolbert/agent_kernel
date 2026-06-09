# Terminal-Bench 2.0 Leaderboard Run

AgentKernel exposes a Harbor custom agent at:

```text
agent_kernel.integrations.terminal_bench_harbor:AgentKernelHarborAgent
```

The adapter runs AgentKernel's policy loop as an external Harbor agent and sends
commands to the Terminal-Bench task container through Harbor's `environment.exec`
API. Harbor remains responsible for task setup, teardown, resource limits,
timeouts, verification, and result logs.

## Prerequisites

- Docker running locally, or a Harbor-supported remote environment.
- Harbor installed: `uv tool install harbor` or `pip install harbor`.
- A Qwen-compatible model endpoint reachable from the host process. For the
  default AgentKernel path, expose an OpenAI-compatible vLLM server and set:

```bash
export AGENT_KERNEL_PROVIDER=vllm
export AGENT_KERNEL_VLLM_HOST=http://127.0.0.1:8000
export AGENT_KERNEL_VLLM_AUTOSTART=0
```

For Ollama instead:

```bash
export AGENT_KERNEL_PROVIDER=ollama
export AGENT_KERNEL_OLLAMA_HOST=http://127.0.0.1:11434
```

## Smoke Test

Run one task first:

```bash
harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path agent_kernel.integrations.terminal_bench_harbor:AgentKernelHarborAgent \
  -m Qwen/Qwen3-Coder-480B-A35B-Instruct \
  -t headless-terminal \
  -k 1
```

## Leaderboard-Candidate Run

Do not pass Harbor resource or timeout overrides for leaderboard submission.

```bash
harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path agent_kernel.integrations.terminal_bench_harbor:AgentKernelHarborAgent \
  -m Qwen/Qwen3-Coder-480B-A35B-Instruct \
  -k 5
```

Useful AgentKernel-only knobs:

```bash
export AGENT_KERNEL_TBENCH_MAX_STEPS=80
export AGENT_KERNEL_TBENCH_COMMAND_TIMEOUT_SECONDS=120
export AGENT_KERNEL_TBENCH_LLM_TIMEOUT_SECONDS=120
```

The adapter writes `agent_kernel_terminal_bench_transcript.json` under each
trial's Harbor agent log directory.

## Submission

Terminal-Bench 2.0 submissions are run through Harbor and submitted by opening a
pull request with the generated leaderboard logs to the Hugging Face repository
linked from the official Terminal-Bench docs. Use agent name `agent_kernel` and
the Qwen model name exactly as passed to `-m`.
