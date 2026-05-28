from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
from agent_kernel.policy import Policy
from agent_kernel.schemas import ActionDecision, TaskSpec


class RuntimeContractSmokePolicy(Policy):
    def __init__(
        self,
        *,
        relative_path: str,
        content: str,
        source: str,
        policy: str,
        include_guarded_comparison: bool,
        selected_command_mode: str,
    ) -> None:
        self.relative_path = relative_path
        self.content = content
        self.source = source
        self.policy = policy
        self.include_guarded_comparison = include_guarded_comparison
        self.selected_command_mode = selected_command_mode

    def decide(self, state):  # noqa: ANN001 - Policy protocol intentionally accepts AgentState.
        del state
        candidate_command = f"mkdir -p {Path(self.relative_path).parent} && printf %s {self.content!r} > {self.relative_path}"
        command = "true" if self.selected_command_mode == "noop" else candidate_command
        shadow = {
            "ready": True,
            "rowwise_selector_source": self.source,
            "rowwise_selector_policy": self.policy,
            "manifest_path": "runtime_contract_smoke",
        }
        if self.include_guarded_comparison:
            shadow.update(
                {
                    "guarded_selected_source": self.source,
                    "guarded_selector_policy": self.policy,
                    "guarded_baseline_prediction": {
                        "action": "code_execute",
                        "content_preview": command,
                        "content": command,
                        "control_tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
                        "exec_kind_family": "materialize_artifact",
                    },
                    "guarded_candidate_prediction": {
                        "action": "code_execute",
                        "content_preview": candidate_command,
                        "content": candidate_command,
                        "control_tokens": ["<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>"],
                        "exec_kind_family": "materialize_artifact",
                    },
                }
            )
        return ActionDecision(
            thought="runtime contract smoke",
            action="code_execute",
            content=command,
            proposal_metadata={"neural_controller_shadow": shadow},
        )


def _task_specs(*, count: int, run_id: str) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    for index in range(max(1, count)):
        relative_path = f"out/result_{index}.txt"
        content = f"ok-{index}"
        task_id = f"neural_runtime_contract_smoke_{run_id}_{index:02d}"
        specs.append(
            TaskSpec(
                task_id=task_id,
                prompt=f"write expected artifact {index}",
                workspace_subdir=task_id,
                expected_file_contents={relative_path: content},
                max_steps=1,
            )
        )
    return specs


def _config(*, dry_run_compare: bool, dry_run_switch: bool) -> KernelConfig:
    return KernelConfig(
        provider="mock",
        storage_backend="json",
        use_world_model=False,
        use_universe_model=False,
        use_planner=False,
        use_graph_memory=False,
        use_tolbert_context=False,
        use_retrieval_proposals=False,
        use_role_specialization=False,
        neural_controller_guarded_dry_run_compare=bool(dry_run_compare),
        neural_controller_guarded_dry_run_switch=bool(dry_run_switch),
        max_steps=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source", default="v64_guarded")
    parser.add_argument("--selector-policy", default="candidate_contract_improves")
    parser.add_argument("--include-guarded-comparison", action="store_true")
    parser.add_argument("--enable-guarded-dry-run-compare", action="store_true")
    parser.add_argument("--enable-guarded-dry-run-switch", action="store_true")
    parser.add_argument("--selected-command-mode", choices=("candidate", "noop"), default="candidate")
    args = parser.parse_args()
    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    successes = 0
    for task in _task_specs(count=int(args.count), run_id=run_id):
        relative_path, content = next(iter(task.expected_file_contents.items()))
        kernel = AgentKernel(
            config=_config(
                dry_run_compare=bool(args.enable_guarded_dry_run_compare),
                dry_run_switch=bool(args.enable_guarded_dry_run_switch),
            ),
            policy=RuntimeContractSmokePolicy(
                relative_path=relative_path,
                content=content,
                source=str(args.source),
                policy=str(args.selector_policy),
                include_guarded_comparison=bool(args.include_guarded_comparison),
                selected_command_mode=str(args.selected_command_mode),
            ),
        )
        episode = kernel.run_task(task, clean_workspace=True)
        successes += 1 if episode.success else 0
    print(
        "neural_controller_runtime_contract_smoke "
        f"run_id={run_id} tasks={int(args.count)} successes={successes}"
    )


if __name__ == "__main__":
    main()
