# Tasks

## Task model

Tasks are defined by `TaskSpec` in [`agent_kernel/schemas.py`](/data/agentkernel/agent_kernel/schemas.py).

Key fields:

- `task_id`
- `prompt`
- `workspace_subdir`
- `setup_commands`
- `success_command`
- `suggested_commands`
- `expected_files`
- `expected_output_substrings`
- `forbidden_files`
- `forbidden_output_substrings`
- `expected_file_contents`
- `max_steps`
- `metadata`

The verifier uses the contract, not the model’s self-report, to decide success.

## Built-in task bank

[`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py) currently defines more than sixty built-in tasks.

They include:

- bounded seed tasks such as `hello_task` and `math_task`
- filesystem mutation tasks such as `rename_task`, `rewrite_task`, and `cleanup_task`
- workflow tasks and paired retrieval tasks
- multiple project, repository, tooling, and integration tasks with paired retrieval variants
- replay tasks synthesized from episodes, promoted skills, operator abstractions, tools, benchmark candidates, and stricter verifier contracts

Common metadata dimensions:

- `capability`
- `difficulty`
- `benchmark_family`
- `requires_retrieval`
- `source_task`
- `memory_source`

## Verification model

The verifier checks:

- command exit status and timeout
- expected file existence
- forbidden file absence
- expected output substrings
- forbidden output substrings
- exact expected file contents when present

Verifier-memory tasks can also be synthesized into stricter variants from previously successful tasks.

## Replay tasks

The task bank can generate replay tasks from artifacts:

- episode replay via `load_episode_replay_tasks(...)`
- skill replay via `load_skill_replay_tasks(...)`
- skill transfer via `load_skill_transfer_tasks(...)`
- operator replay via `load_operator_replay_tasks(...)`
- tool replay via `load_tool_replay_tasks(...)`
- benchmark candidates via `load_benchmark_candidate_tasks(...)`
- verifier replay via `load_verifier_replay_tasks(...)`
- verifier candidates via `load_verifier_candidate_tasks(...)`

These tasks are tagged with memory-oriented benchmark families such as:

- `episode_memory`
- `skill_memory`
- `skill_transfer`
- `operator_memory`
- `tool_memory`
- `verifier_memory`

## Workspace behavior

Each task gets an isolated directory under the configured workspace root. By default that is `workspace/<task_id>/`, and the runtime cleans it before each run unless told otherwise in code.

## Curriculum behavior

[`agent_kernel/curriculum.py`](/data/agentkernel/agent_kernel/curriculum.py) now supports two paths:

- success-adjacent followups
- failure-recovery followups

Generated tasks vary by benchmark family. For example:

- workflow tasks produce audit-style followups
- project/repository/tooling/integration tasks produce family-specific handoff tasks
- certain failure patterns produce targeted recovery tasks

Generated task metadata includes fields such as:

- `parent_task`
- `curriculum_kind`
- `failure_types`
- `failure_pattern`

## Adding a task

The simplest way to add a new built-in task is to extend [`agent_kernel/task_bank.py`](/data/agentkernel/agent_kernel/task_bank.py) with another `TaskSpec`.

Good tasks for this harness are:

- bounded local shell tasks
- deterministic verifier contracts
- no network dependency
- small workspace footprint
- explicit expected artifacts
- prompts that can benefit from retrieval or replay without requiring hidden state
