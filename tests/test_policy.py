import hashlib
import json

from agent_kernel.llm import MockLLMClient
from agent_kernel.config import KernelConfig
from agent_kernel.modeling.policy import decoder as decoder_module
from agent_kernel.modeling.policy.decoder import (
    _best_non_overlapping_window_subset,
    decode_bounded_action_candidates,
    decode_action_generation_candidates,
    _overlap_block_replace_rank_key,
    _structured_edit_window_bonus,
    _workspace_preview_window_proposals,
)
from agent_kernel.modeling.world.latent_state import build_latent_state_summary
from agent_kernel.modeling.world.rollout import rollout_action_value
from agent_kernel.policy import LLMDecisionPolicy, SkillLibrary
from agent_kernel.schemas import ActionDecision, ContextPacket, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.task_bank import TaskBank


def test_active_prompt_adjustments_dedupe_before_top3_truncation(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "proposals": [
                    {
                        "area": "decision",
                        "priority": 6,
                        "reason": "selected variant targets low-confidence retrieval caution",
                        "suggestion": "When retrieval confidence is weak, prefer additional search or verifier checks before committing to a command path.",
                    },
                    {
                        "area": "decision",
                        "priority": 5,
                        "reason": "low-confidence retrieval remains common",
                        "suggestion": "Tell the model to widen search and avoid overcommitting when retrieval confidence is low.",
                    },
                    {
                        "area": "decision",
                        "priority": 4,
                        "reason": "missing expected files recur in memory",
                        "suggestion": "Require explicit confirmation that expected artifacts were created before terminating.",
                    },
                    {
                        "area": "reflection",
                        "priority": 3,
                        "reason": "generated-task performance trails base task performance",
                        "suggestion": "Bias reflection prompts toward adaptation from prior traces instead of one-shot synthesis.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig(prompt_proposals_path=prompt_path))

    adjustments = policy._active_prompt_adjustments()

    assert len(adjustments) == 3
    assert [int(item.get("priority", 0) or 0) for item in adjustments] == [6, 4, 3]
    assert sum(1 for item in adjustments if "retrieval" in str(item.get("suggestion", "")).lower()) == 1


def test_llm_policy_returns_allowed_action():
    policy = LLMDecisionPolicy(MockLLMClient())
    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    assert "hello.txt" in decision.content


class CapturingClient:
    def __init__(self) -> None:
        self.last_payload = None
        self.last_system_prompt = None
        self.last_decision_prompt = None

    def create_decision(self, *, system_prompt, decision_prompt, state_payload):
        self.last_system_prompt = system_prompt
        self.last_decision_prompt = decision_prompt
        self.last_payload = state_payload
        return {
            "thought": "use deterministic command",
            "action": "code_execute",
            "content": "printf 'hello agent kernel\\n' > hello.txt",
            "done": False,
        }


class FakeContextProvider:
    def compile(self, state):
        del state
        return ContextPacket(
            request_id="req-1",
            created_at="2026-03-16T00:00:00+00:00",
            task={"goal": "g", "completion_criteria": "c"},
            control={
                "mode": "verify",
                "path_confidence": 0.95,
                "trust_retrieval": True,
                "retrieval_guidance": {
                    "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                    "recommended_command_spans": [
                        {
                            "span_id": "task:hello:suggested:1",
                            "command": "printf 'hello agent kernel\\n' > hello.txt",
                        }
                    ],
                    "avoidance_notes": [],
                    "evidence": ["task:hello:suggested:1: template command"],
                },
            },
            tolbert={"path_prediction": {"tree_version": "tol_v1"}},
            retrieval={
                "branch_scoped": [
                    {"metadata": {"task_id": "hello_task"}},
                ],
                "fallback_scoped": [],
                "global": [],
            },
            verifier_contract={"success_command": "true"},
        )


class NonDeterministicContextProvider:
    def compile(self, state):
        del state
        return ContextPacket(
            request_id="req-1",
            created_at="2026-03-16T00:00:00+00:00",
            task={"goal": "g", "completion_criteria": "c"},
            control={
                "mode": "verify",
                "path_confidence": 0.2,
                "trust_retrieval": False,
                "retrieval_guidance": {
                    "recommended_commands": [],
                    "recommended_command_spans": [],
                    "avoidance_notes": [],
                    "evidence": [],
                },
            },
            tolbert={"path_prediction": {"tree_version": "tol_v1"}},
            retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
            verifier_contract={"success_command": "true"},
        )


def test_llm_policy_includes_tolbert_context_packet():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    assert client.last_payload is not None
    assert client.last_payload["context_packet"] is not None
    assert client.last_payload["context_packet"]["tolbert"]["path_prediction"]["tree_version"] == "tol_v1"


def test_llm_policy_emits_decision_progress_stages_for_llm_path():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    observed = []

    policy.set_decision_progress_callback(lambda payload: observed.append(dict(payload)))

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    assert [payload["step_stage"] for payload in observed] == [
        "context_compile",
        "context_ready",
        "universe_summary",
        "memory_retrieved",
        "plan_candidates",
        "transition_simulated",
        "payload_build",
        "llm_request",
        "llm_response",
    ]


def test_llm_policy_reuses_context_packet_for_unchanged_long_horizon_continuation():
    class CountingContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            return ContextPacket(
                request_id=f"req-{self.compile_calls}",
                created_at=f"2026-04-03T00:00:0{self.compile_calls}+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.4,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": f"tol_v{self.compile_calls}"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    context_provider = CountingContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="context_reuse_task",
            prompt="Inspect the long-horizon state and respond.",
            workspace_subdir="context_reuse_task",
            suggested_commands=[],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="prior attempt",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'done\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["still need review"]},
        )
    ]
    state.recent_workspace_summary = "step 1: command='printf done' exit_code=0 timed_out=False outcome=ok"
    state.world_model_summary = {"horizon": "long_horizon", "completion_ratio": 0.5}

    first = policy.decide(state)
    first_request_id = state.context_packet.request_id
    first_tree_version = state.context_packet.tolbert["path_prediction"]["tree_version"]
    state.current_role = "planner"
    second = policy.decide(state)

    assert first.action == "code_execute"
    assert second.action == "code_execute"
    assert context_provider.compile_calls == 1
    assert state.context_packet.request_id == first_request_id
    assert state.context_packet.tolbert["path_prediction"]["tree_version"] == first_tree_version


def test_llm_policy_invalidates_context_packet_reuse_after_world_state_change():
    class CountingContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            return ContextPacket(
                request_id=f"req-{self.compile_calls}",
                created_at=f"2026-04-03T00:00:0{self.compile_calls}+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.4,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": f"tol_v{self.compile_calls}"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    context_provider = CountingContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="context_reuse_invalidation_task",
            prompt="Inspect the long-horizon state and respond.",
            workspace_subdir="context_reuse_invalidation_task",
            suggested_commands=[],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="prior attempt",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'done\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["still need review"]},
        )
    ]
    state.recent_workspace_summary = "step 1: command='printf done' exit_code=0 timed_out=False outcome=ok"
    state.world_model_summary = {"horizon": "long_horizon", "completion_ratio": 0.5}

    policy.decide(state)
    first_request_id = state.context_packet.request_id
    state.world_model_summary = {"horizon": "long_horizon", "completion_ratio": 0.75}
    policy.decide(state)

    assert context_provider.compile_calls == 2
    assert state.context_packet.request_id != first_request_id


def test_llm_policy_preserves_context_compile_subphases_from_provider():
    class ReportingContextProvider:
        def __init__(self) -> None:
            self._progress_callback = None

        def set_progress_callback(self, callback) -> None:
            self._progress_callback = callback

        def compile(self, state):
            del state
            assert callable(self._progress_callback)
            self._progress_callback(
                {
                    "step_stage": "context_compile",
                    "step_subphase": "query_build",
                    "step_elapsed_seconds": 0.01,
                }
            )
            self._progress_callback(
                {
                    "step_stage": "context_compile",
                    "step_subphase": "chunk_select",
                    "step_elapsed_seconds": 0.02,
                }
            )
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.2,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=ReportingContextProvider())
    observed = []

    policy.set_decision_progress_callback(lambda payload: observed.append(dict(payload)))

    decision = policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    compile_events = [payload for payload in observed if payload["step_stage"] == "context_compile"]
    assert [payload.get("step_subphase", "") for payload in compile_events] == ["", "query_build", "chunk_select"]


def test_llm_policy_uses_adjacent_success_direct_command_before_llm():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    decision = policy.decide(
        AgentState(
            task=TaskSpec(
                task_id="repo_adjacent",
                prompt="adjacent",
                workspace_subdir="repo_adjacent",
                suggested_commands=[
                    "mkdir -p repo && printf 'repo ready\\n' > repo/summary.txt && printf 'repo verified\\n' > repo/status.txt",
                ],
                expected_files=["repo/summary.txt", "repo/status.txt"],
                expected_file_contents={
                    "repo/summary.txt": "repo ready\n",
                    "repo/status.txt": "repo verified\n",
                },
                metadata={"curriculum_kind": "adjacent_success", "benchmark_family": "repository"},
            )
        )
    )

    assert decision.action == "code_execute"
    assert decision.decision_source == "adjacent_success_direct"
    assert client.last_payload is None


def test_llm_policy_adjacent_success_skips_context_compile_when_first_step_is_structurally_safe():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError('adjacent_success direct path should skip context compilation')

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)

    decision = policy.decide(
        AgentState(
            task=TaskSpec(
                task_id="repo_adjacent",
                prompt="adjacent",
                workspace_subdir="repo_adjacent",
                suggested_commands=[
                    "mkdir -p repo && printf 'repo ready\n' > repo/summary.txt && printf 'repo verified\n' > repo/status.txt",
                ],
                expected_files=["repo/summary.txt", "repo/status.txt"],
                expected_file_contents={
                    "repo/summary.txt": "repo ready\n",
                    "repo/status.txt": "repo verified\n",
                },
                metadata={"curriculum_kind": "adjacent_success", "benchmark_family": "repository"},
            )
        )
    )

    assert decision.action == "code_execute"
    assert decision.decision_source == "adjacent_success_direct"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_continues_shared_repo_integrator_sequential_segments_despite_soft_penalty():
    policy = LLMDecisionPolicy(MockLLMClient())
    original_score = policy._command_control_score

    def patched_score(state, command):
        normalized = str(command).strip()
        if normalized.startswith("tests/test_api.sh") or "reports/merge_report.txt" in normalized:
            return -6
        return original_score(state, command)

    policy._command_control_score = patched_score  # type: ignore[method-assign]
    task = TaskBank().get("git_parallel_merge_acceptance_task")
    state = AgentState(
        task=task,
        history=[
            StepRecord(
                index=1,
                thought="merge api",
                action="code_execute",
                content="git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": []},
                decision_source="shared_repo_integrator_segment_direct",
            ),
            StepRecord(
                index=2,
                thought="merge docs",
                action="code_execute",
                content="git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": []},
                decision_source="shared_repo_integrator_segment_direct",
            ),
        ],
        world_model_summary={"horizon": "long_horizon"},
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "tests/test_api.sh && tests/test_docs.sh"


def test_llm_policy_critic_keeps_shared_repo_integrator_segment_path_active():
    policy = LLMDecisionPolicy(MockLLMClient())
    task = TaskBank().get("git_parallel_merge_acceptance_task")
    state = AgentState(
        task=task,
        history=[
            StepRecord(
                index=1,
                thought="merge api",
                action="code_execute",
                content="git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["still missing worker/docs-status"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
        ],
        current_role="critic",
        consecutive_failures=1,
        latest_state_transition={"regressed": False},
        world_model_summary={"horizon": "long_horizon"},
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'"


def test_llm_policy_shared_repo_integrator_skips_context_compile_for_first_merge_segment():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("shared-repo integrator direct path should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)

    decision = policy.decide(AgentState(task=TaskBank().get("git_parallel_merge_acceptance_task")))

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "git merge --no-ff worker/api-status -m 'merge worker/api-status'"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_shared_repo_integrator_generated_conflict_skips_context_compile_for_premerge_segment():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("shared-repo generated-conflict integrator should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)

    decision = policy.decide(AgentState(task=TaskBank().get("git_generated_conflict_resolution_task")))

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == (
        "printf 'SERVICE_STATUS=mainline-ready\n' > src/shared_status.txt && "
        "git add src/shared_status.txt && git commit -m 'mainline status change'"
    )
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_shared_repo_integrator_continuation_skips_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("shared-repo integrator continuation should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskBank().get("git_parallel_merge_acceptance_task"),
        history=[
            StepRecord(
                index=1,
                thought="merge api",
                action="code_execute",
                content="git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["still missing worker/docs-status"]},
                decision_source="shared_repo_integrator_segment_direct",
            )
        ],
        current_role="critic",
        consecutive_failures=1,
        world_model_summary={"horizon": "long_horizon"},
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_shared_repo_integrator_retries_failed_merge_segment_before_advancing():
    policy = LLMDecisionPolicy(MockLLMClient())
    task = TaskBank().get("git_parallel_merge_acceptance_task")
    state = AgentState(
        task=task,
        history=[
            StepRecord(
                index=1,
                thought="merge api",
                action="code_execute",
                content="git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                selected_skill_id=None,
                command_result={
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "merge: worker/api-status - not something we can merge",
                    "timed_out": False,
                },
                verification={"passed": False, "reasons": ["missing required merged branch worker/api-status"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
            StepRecord(
                index=2,
                thought="merge docs",
                action="code_execute",
                content="git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["missing required merged branch worker/api-status"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
        ],
        world_model_summary={"horizon": "long_horizon"},
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "git merge --no-ff worker/api-status -m 'merge worker/api-status'"


def test_llm_policy_shared_repo_integrator_retries_failed_test_segment_before_reporting():
    policy = LLMDecisionPolicy(MockLLMClient())
    task = TaskBank().get("git_parallel_merge_acceptance_task")
    state = AgentState(
        task=task,
        history=[
            StepRecord(
                index=1,
                thought="merge api",
                action="code_execute",
                content="git merge --no-ff worker/api-status -m 'merge worker/api-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["missing required merged branch worker/docs-status"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
            StepRecord(
                index=2,
                thought="merge docs",
                action="code_execute",
                content="git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                selected_skill_id=None,
                command_result={"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
                verification={"passed": False, "reasons": ["missing workflow report reports/test_report.txt"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
            StepRecord(
                index=3,
                thought="run tests",
                action="code_execute",
                content="tests/test_api.sh && tests/test_docs.sh",
                selected_skill_id=None,
                command_result={"exit_code": 1, "stdout": "", "stderr": "api failed", "timed_out": False},
                verification={"passed": False, "reasons": ["workflow test api suite failed"]},
                decision_source="shared_repo_integrator_segment_direct",
            ),
        ],
        world_model_summary={"horizon": "long_horizon"},
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "shared_repo_integrator_segment_direct"
    assert decision.content == "tests/test_api.sh && tests/test_docs.sh"


def test_llm_policy_uses_git_repo_review_direct_command_before_llm():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)

    decision = policy.decide(AgentState(task=TaskBank().get("git_repo_test_repair_task")))

    assert decision.action == "code_execute"
    assert decision.decision_source == "git_repo_review_direct"
    assert decision.content.startswith("git checkout -b fix/release-ready &&")
    assert "tests/test_release.sh" in decision.content
    assert client.last_payload is None


def test_llm_policy_git_repo_review_direct_command_preserves_tolbert_shadow_route(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["repository", "project", "workflow"],
                    "primary_benchmark_families": [],
                },
            }
        ),
        encoding="utf-8",
    )
    client = CapturingClient()
    policy = LLMDecisionPolicy(
        client,
        config=KernelConfig(
            tolbert_model_artifact_path=artifact_path,
            use_tolbert_model_artifacts=True,
        ),
    )
    task = TaskBank().get("git_repo_status_review_task")
    task.metadata["benchmark_family"] = "repository"

    decision = policy.decide(AgentState(task=task))

    assert decision.action == "code_execute"
    assert decision.decision_source == "git_repo_review_direct"
    assert decision.tolbert_route_mode == "shadow"
    assert client.last_payload is None


def test_llm_policy_uses_git_repo_review_direct_command_for_shared_repo_worker_before_llm():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)

    decision = policy.decide(AgentState(task=TaskBank().get("git_parallel_worker_api_task")))

    assert decision.action == "code_execute"
    assert decision.decision_source == "git_repo_review_direct"
    assert decision.content.startswith("printf 'API_STATUS=ready")
    assert "tests/test_api.sh" in decision.content
    assert client.last_payload is None


def test_llm_policy_git_repo_review_skips_context_compile_when_first_step_is_structurally_safe():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("git repo review direct path should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)

    decision = policy.decide(AgentState(task=TaskBank().get("git_repo_status_review_task")))

    assert decision.action == "code_execute"
    assert decision.decision_source == "git_repo_review_direct"
    assert decision.content.startswith("git checkout -b review/status-ready &&")
    assert "tests/check_status.sh" in decision.content
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_git_repo_review_requires_first_command_to_cover_repo_workflow_contract():
    class CountingContextProvider(NonDeterministicContextProvider):
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            self.compile_calls += 1
            return super().compile(state)

    client = CapturingClient()
    context_provider = CountingContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    decision = policy.decide(
        AgentState(
            task=TaskSpec(
                task_id="incomplete_git_review",
                prompt="repair the repo workflow",
                workspace_subdir="incomplete_git_review",
                suggested_commands=[
                    "git checkout -b review/status-ready && printf 'STATUS=ready\\n' > src/app.py",
                ],
                expected_files=["src/app.py", "reports/test_report.txt"],
                expected_file_contents={
                    "src/app.py": "STATUS=ready\n",
                    "reports/test_report.txt": "status check passed\n",
                },
                metadata={
                    "benchmark_family": "repo_sandbox",
                    "requires_git": True,
                    "workflow_guard": {"requires_git": True},
                    "semantic_verifier": {
                        "kind": "git_repo_review",
                        "expected_branch": "review/status-ready",
                        "expected_changed_paths": [
                            "reports/test_report.txt",
                            "src/app.py",
                        ],
                        "test_commands": [
                            {
                                "label": "status check script",
                                "argv": ["tests/check_status.sh"],
                            }
                        ],
                    },
                },
            )
        )
    )

    assert context_provider.compile_calls == 1
    assert decision.decision_source != "git_repo_review_direct"
    assert client.last_payload is not None


def test_llm_policy_sends_compact_chunked_context_packet():
    class ChunkedContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.55,
                    "trust_retrieval": False,
                    "selected_context_chunks": [
                        {"span_id": "doc:1", "bucket": "branch_scoped", "text": "first chunk"},
                        {"span_id": "doc:2", "bucket": "global", "text": "second chunk"},
                    ],
                    "context_chunk_budget": {"max_chunks": 8, "char_budget": 2400},
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={
                    "branch_scoped": [{"span_id": "a", "span_type": "agent:task", "score": 1.0, "metadata": {}}],
                    "fallback_scoped": [{"span_id": "b", "span_type": "doc:readme_chunk", "score": 0.8, "metadata": {}}],
                    "global": [{"span_id": "c", "span_type": "agent:prompt", "score": 0.7, "metadata": {}}],
                },
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=ChunkedContextProvider())

    policy.decide(AgentState(task=TaskBank().get("hello_task")))

    packet = client.last_payload["context_packet"]
    assert packet["control"]["selected_context_chunks"][0]["span_id"] == "doc:1"
    assert packet["control"]["context_chunk_budget"]["max_chunks"] == 8
    assert len(packet["retrieval"]["branch_scoped"]) == 1
    assert packet["retrieval"]["branch_scoped"][0]["span_id"] == "a"


def test_llm_policy_compacts_graph_world_and_plan_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.graph_summary = {
        "document_count": 12,
        "benchmark_families": {f"family{i}": i for i in range(8)},
        "failure_types": {f"failure{i}": i for i in range(8)},
        "related_tasks": [f"task_{i}" for i in range(8)],
    }
    state.world_model_summary = {
        "benchmark_family": "project",
        "horizon": "long_horizon",
        "expected_artifacts": [f"path_{i}.txt" for i in range(8)],
        "forbidden_artifacts": [f"bad_{i}.txt" for i in range(8)],
        "preserved_artifacts": [f"keep_{i}.txt" for i in range(8)],
    }
    state.plan = [f"step {i}" for i in range(8)]
    state.active_subgoal = "x" * 1200

    policy.decide(state)

    assert client.last_payload["graph_summary"]["document_count"] == 12
    assert len(client.last_payload["graph_summary"]["benchmark_families"]) == 6
    assert len(client.last_payload["world_model_summary"]["expected_artifacts"]) == 6
    assert len(client.last_payload["plan"]) == 4
    assert len(client.last_payload["active_subgoal"]) < 1200
    assert len(client.last_payload["state_context_chunks"]) <= 8


def test_llm_policy_limits_history_payload_and_preserves_archive_summary():
    client = CapturingClient()
    config = KernelConfig(provider="mock", use_tolbert_context=False, payload_history_step_window=2)
    policy = LLMDecisionPolicy(client, config=config)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.history_archive = {
        "archived_step_count": 3,
        "recent_archived_summaries": [
            "1:code_execute:ok:printf 'one\\n' > one.txt",
            "2:code_execute:ok:printf 'two\\n' > two.txt",
            "3:code_execute:ok:printf 'three\\n' > three.txt",
        ],
        "failed_command_signatures": ["false"],
    }
    state.history = [
        StepRecord(
            index=4,
            thought="write four",
            action="code_execute",
            content="printf 'four\\n' > four.txt",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": []},
        ),
        StepRecord(
            index=5,
            thought="write five",
            action="code_execute",
            content="printf 'five\\n' > five.txt",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": []},
        ),
        StepRecord(
            index=6,
            thought="write six",
            action="code_execute",
            content="printf 'six\\n' > six.txt",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": []},
        ),
    ]

    policy.decide(state)

    assert len(client.last_payload["history"]) == 2
    assert client.last_payload["history"][0]["index"] == 5
    assert client.last_payload["history"][1]["index"] == 6
    assert client.last_payload["history_archive"]["archived_step_count"] == 3
    assert client.last_payload["history_archive"]["recent_archived_summaries"]


def test_llm_policy_includes_dynamic_world_state_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.world_model_summary = {
        "benchmark_family": "micro",
        "horizon": "bounded",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
        "preserved_artifacts": [],
        "missing_expected_artifacts": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "changed_preserved_artifacts": [],
        "updated_workflow_paths": [],
        "completion_ratio": 0.25,
    }
    state.active_subgoal = "remove forbidden artifact temp.txt"

    policy.decide(state)

    assert client.last_payload["world_model_summary"]["missing_expected_artifacts"] == ["status.txt"]
    assert client.last_payload["world_model_summary"]["present_forbidden_artifacts"] == ["temp.txt"]
    assert client.last_payload["world_model_summary"]["completion_ratio"] == 0.25
    assert client.last_payload["transition_preview"]["present_forbidden_artifacts"] == ["temp.txt"]


def test_llm_policy_includes_active_subgoal_diagnosis_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.current_role = "planner"
    state.active_subgoal = "remove forbidden artifact temp.txt"
    state.subgoal_diagnoses = {
        "remove forbidden artifact temp.txt": {
            "summary": "temp.txt is still present; recent step regressed workspace state",
            "signals": ["state_regression"],
            "source_role": "critic",
            "updated_step_index": 4,
        }
    }

    policy.decide(state)

    assert client.last_payload["active_subgoal_diagnosis"]["source_role"] == "critic"
    assert "temp.txt is still present" in client.last_payload["active_subgoal_diagnosis"]["summary"]
    assert any(chunk["source"] == "subgoal_diagnosis" for chunk in client.last_payload["state_context_chunks"])


def test_llm_policy_includes_universe_summary_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)

    policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert client.last_payload["universe_summary"]["stability"] == "stable"
    assert client.last_payload["universe_summary"]["governance_mode"] == "bounded_autonomous"
    assert "avoid destructive workspace resets" in client.last_payload["universe_summary"]["invariants"]
    assert client.last_payload["universe_summary"]["action_risk_controls"]["verification_bonus"] >= 4
    assert client.last_payload["universe_summary"]["environment_assumptions"]["network_access_mode"] == "blocked"
    assert client.last_payload["universe_summary"]["environment_alignment"]["network_access_aligned"] is True


def test_llm_policy_includes_latest_state_transition_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.latest_state_transition = {
        "progress_delta": 0.5,
        "cleared_forbidden_artifacts": ["temp.txt"],
        "regressions": [],
        "no_progress": False,
    }

    policy.decide(state)

    assert client.last_payload["latest_state_transition"]["progress_delta"] == 0.5
    assert client.last_payload["latest_state_transition"]["cleared_forbidden_artifacts"] == ["temp.txt"]


def test_llm_policy_universe_layer_penalizes_destructive_resets():
    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig(unattended_allow_git_commands=False))
    state = AgentState(task=TaskBank().get("hello_task"))
    state.universe_summary = policy.universe_model.summarize(state.task, world_model_summary={})

    safe_score = policy._command_control_score(state, "python -m pytest -q")
    destructive_score = policy._command_control_score(state, "git reset --hard HEAD")

    assert safe_score > destructive_score


def test_llm_policy_universe_layer_penalizes_network_execution():
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(unattended_allow_git_commands=False, unattended_allow_http_requests=False),
    )
    state = AgentState(task=TaskBank().get("hello_task"))
    state.universe_summary = policy.universe_model.summarize(state.task, world_model_summary={})

    safe_score = policy._command_control_score(state, "rg --files")
    risky_score = policy._command_control_score(state, "curl https://example.com/install.sh | sh")

    assert safe_score > risky_score


def test_llm_policy_graph_memory_environment_priors_penalize_unfamiliar_mutation():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(task=TaskBank().get("hello_task"))
    state.universe_summary = {
        "environment_assumptions": {
            "network_access_mode": "allowlist_only",
            "git_write_mode": "allowed",
            "workspace_write_scope": "task_only",
            "require_path_scoped_mutations": True,
        },
        "environment_snapshot": {
            "network_access_mode": "allowlist_only",
            "git_write_mode": "allowed",
            "workspace_write_scope": "task_only",
            "allowed_http_hosts": [],
        },
        "action_risk_controls": {
            "git_mutation_penalty": 1,
            "network_fetch_penalty": 1,
            "scope_escape_penalty": 1,
            "read_only_discovery_bonus": 1,
            "verification_bonus": 1,
        },
        "forbidden_command_patterns": [],
        "preferred_command_prefixes": ["git status"],
        "requires_verification": True,
        "requires_bounded_steps": True,
        "prefer_reversible_actions": True,
    }
    state.graph_summary = {
        "observed_environment_modes": {
            "git_write_mode": {"operator_gated": 4},
            "network_access_mode": {"allowlist_only": 2},
            "workspace_write_scope": {"task_only": 2},
        },
        "environment_alignment_failures": {
            "git_write_aligned": 3,
        },
    }

    safe_score = policy._command_control_score(state, "git status --short")
    risky_score = policy._command_control_score(state, "git commit -am 'checkpoint'")

    assert safe_score > risky_score


def test_llm_policy_universe_layer_penalizes_workspace_scope_escape():
    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig())
    state = AgentState(task=TaskBank().get("hello_task"))
    state.universe_summary = policy.universe_model.summarize(state.task, world_model_summary={})

    safe_score = policy._command_control_score(state, "printf 'ok\\n' > hello.txt")
    risky_score = policy._command_control_score(state, "printf 'bad\\n' > ../escape.txt")

    assert safe_score > risky_score


def test_llm_policy_uses_retained_transition_model_to_penalize_bad_transition_commands(tmp_path):
    proposal_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "artifact_kind": "transition_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "repeat_command_penalty": 6,
                    "regressed_path_command_penalty": 4,
                    "recovery_command_bonus": 3,
                    "progress_command_bonus": 2,
                    "max_signatures": 12,
                },
                "signatures": [
                    {
                        "signal": "no_state_progress",
                        "command": "false",
                        "support": 2,
                        "regressions": [],
                    },
                    {
                        "signal": "state_regression",
                        "command": "printf 'x\\n' > keep.txt",
                        "support": 1,
                        "regressions": ["keep.txt"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(transition_model_proposals_path=proposal_path),
    )
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.world_model_summary = {
        "present_forbidden_artifacts": ["temp.txt"],
        "missing_expected_artifacts": [],
        "forbidden_artifacts": ["temp.txt"],
        "preserved_artifacts": ["keep.txt"],
    }
    state.history.append(
        StepRecord(
            index=1,
            thought="stalled",
            action="code_execute",
            content="false",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["exit code was 1"]},
        )
    )
    state.latest_state_transition = {
        "no_progress": True,
        "regressions": ["temp.txt", "keep.txt"],
    }

    repeat_score = policy._command_control_score(state, "false")
    recover_score = policy._command_control_score(state, "rm -f temp.txt")

    assert recover_score > repeat_score


def test_llm_policy_treats_same_pattern_stalled_command_as_repeat(tmp_path):
    proposal_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "artifact_kind": "transition_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "repeat_command_penalty": 6,
                    "regressed_path_command_penalty": 4,
                    "recovery_command_bonus": 3,
                    "progress_command_bonus": 2,
                    "max_signatures": 12,
                },
                "signatures": [
                    {
                        "signal": "no_state_progress",
                        "command": "echo 'ENV=base' > config/base.env",
                        "command_pattern": "echo <str> > <path>",
                        "support": 2,
                        "regressions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(transition_model_proposals_path=proposal_path),
    )
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.history.append(
        StepRecord(
            index=1,
            thought="stalled",
            action="code_execute",
            content="echo 'ENV=base' > config/base.env",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["no progress"]},
        )
    )
    state.latest_state_transition = {"no_progress": True, "regressions": []}

    same_pattern_score = policy._command_control_score(state, "echo 'ENV=prod' > config/prod.env")
    new_pattern_score = policy._command_control_score(state, "mkdir -p config && touch ready.txt")

    assert new_pattern_score > same_pattern_score


def test_llm_policy_ignores_family_scoped_transition_signature_for_other_families(tmp_path):
    proposal_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "artifact_kind": "transition_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "repeat_command_penalty": 6,
                    "regressed_path_command_penalty": 4,
                    "recovery_command_bonus": 3,
                    "progress_command_bonus": 2,
                    "max_signatures": 12,
                },
                "signatures": [
                    {
                        "signal": "no_state_progress",
                        "command": "echo 'ENV=base' > config/base.env",
                        "command_pattern": "echo <str> > <path>",
                        "benchmark_family": "repository",
                        "support": 2,
                        "regressions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(transition_model_proposals_path=proposal_path),
    )
    baseline_policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig())
    state = AgentState(
        task=TaskSpec(
            task_id="workflow_task",
            prompt="write a workflow config",
            success_command="true",
            workspace_subdir="workflow_task",
            metadata={"benchmark_family": "workflow"},
        )
    )
    state.history.append(
        StepRecord(
            index=1,
            thought="stalled",
            action="code_execute",
            content="echo 'ENV=base' > config/base.env",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["no progress"]},
        )
    )
    state.latest_state_transition = {"no_progress": False, "regressions": []}

    same_pattern_score = policy._command_control_score(state, "echo 'ENV=prod' > config/prod.env")
    baseline_score = baseline_policy._command_control_score(state, "echo 'ENV=prod' > config/prod.env")

    assert same_pattern_score == baseline_score


def test_llm_policy_prefers_long_horizon_transition_signatures_only_for_long_horizon_tasks(tmp_path):
    proposal_path = tmp_path / "transition_model" / "transition_model_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "artifact_kind": "transition_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "repeat_command_penalty": 6,
                    "regressed_path_command_penalty": 4,
                    "recovery_command_bonus": 3,
                    "progress_command_bonus": 2,
                    "long_horizon_repeat_command_penalty": 3,
                    "long_horizon_progress_command_bonus": 2,
                    "max_signatures": 12,
                },
                "signatures": [
                    {
                        "signal": "no_state_progress",
                        "command": "python scripts/generate_report.py",
                        "command_pattern": "python <path>",
                        "benchmark_family": "workflow",
                        "difficulty": "long_horizon",
                        "support": 2,
                        "regressions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(transition_model_proposals_path=proposal_path),
    )
    baseline_policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig())
    long_state = AgentState(
        task=TaskSpec(
            task_id="workflow_long",
            prompt="generate a report through a multi-step workflow",
            success_command="true",
            workspace_subdir="workflow_long",
            metadata={"benchmark_family": "workflow", "difficulty": "long_horizon"},
        )
    )
    long_state.world_model_summary = {"horizon": "long_horizon"}
    long_state.history.append(
        StepRecord(
            index=1,
            thought="stalled",
            action="code_execute",
            content="python scripts/generate_report.py",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["no progress"]},
        )
    )
    long_state.latest_state_transition = {"no_progress": True, "regressions": []}

    bounded_state = AgentState(
        task=TaskSpec(
            task_id="workflow_bounded",
            prompt="generate a report",
            success_command="true",
            workspace_subdir="workflow_bounded",
            metadata={"benchmark_family": "workflow", "difficulty": "seed"},
        )
    )
    bounded_state.world_model_summary = {"horizon": "bounded"}
    bounded_state.history = list(long_state.history)
    bounded_state.latest_state_transition = dict(long_state.latest_state_transition)

    long_score = policy._command_control_score(long_state, "python scripts/generate_report.py")
    bounded_score = policy._command_control_score(bounded_state, "python scripts/generate_report.py")
    baseline_long_score = baseline_policy._command_control_score(long_state, "python scripts/generate_report.py")

    assert long_score < bounded_score
    assert long_score < baseline_long_score


def test_llm_policy_uses_retained_state_estimation_to_bias_recovery_commands(tmp_path):
    proposal_path = tmp_path / "state_estimation" / "state_estimation_proposals.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "state_estimation_policy_set",
                "lifecycle_state": "retained",
                "generation_focus": "recovery_bias",
                "control_schema": "state_estimation_controls_v1",
                "controls": {
                    "no_progress_progress_epsilon": 0.02,
                    "min_state_change_score_for_progress": 2,
                    "regression_path_budget": 8,
                    "regression_severity_weight": 1.5,
                    "progress_recovery_credit": 1.5,
                },
                "latent_controls": {
                    "advancing_completion_ratio": 0.8,
                    "advancing_progress_delta": 0.2,
                    "improving_progress_delta": 0.0,
                    "regressing_progress_delta": -0.05,
                    "regressive_regression_count": 1,
                    "blocked_forbidden_count": 1,
                    "active_path_budget": 8,
                },
                "policy_controls": {
                    "regressive_path_match_bonus": 4,
                    "regressive_cleanup_bonus": 3,
                    "blocked_command_bonus": 2,
                    "advancing_path_match_bonus": 1,
                    "trusted_retrieval_path_bonus": 2,
                },
                "transition_summary": {"state_regression_steps": 2, "state_progress_gain_steps": 0},
                "proposals": [
                    {
                        "area": "recovery_bias",
                        "priority": 4,
                        "reason": "recovery should be favored after regressive transitions",
                        "suggestion": "bias cleanup commands over repeated risky edits",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(state_estimation_proposals_path=proposal_path),
    )
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.world_model_summary = {
        "present_forbidden_artifacts": ["temp.txt"],
        "missing_expected_artifacts": [],
        "forbidden_artifacts": ["temp.txt"],
    }
    state.latent_state_summary = {
        "risk_band": "regressive",
        "progress_band": "flat",
        "retrieval_mode": "trusted",
        "active_paths": ["temp.txt"],
    }

    cleanup_score = policy._command_control_score(state, "rm -f temp.txt")
    risky_score = policy._command_control_score(state, "printf 'x\\n' > temp.txt")

    assert cleanup_score > risky_score


def test_build_latent_state_summary_uses_state_estimation_controls():
    summary = build_latent_state_summary(
        world_model_summary={
            "completion_ratio": 0.2,
            "missing_expected_artifacts": ["report.txt"],
            "present_forbidden_artifacts": [],
        },
        latest_transition={
            "progress_delta": 0.12,
            "regressions": [],
            "no_progress": False,
        },
        task_metadata={"benchmark_family": "repo_sandbox"},
        recent_history=[],
        context_control={"path_confidence": 0.0, "trust_retrieval": False},
        latent_controls={
            "advancing_completion_ratio": 0.1,
            "advancing_progress_delta": 0.1,
            "improving_progress_delta": 0.05,
            "regressing_progress_delta": -0.05,
            "regressive_regression_count": 1,
            "blocked_forbidden_count": 1,
            "active_path_budget": 2,
        },
    )

    assert summary["progress_band"] == "advancing"


def test_build_latent_state_summary_blends_learned_world_signal():
    summary = build_latent_state_summary(
        world_model_summary={
            "completion_ratio": 0.1,
            "missing_expected_artifacts": ["report.txt"],
            "present_forbidden_artifacts": [],
        },
        latest_transition={
            "progress_delta": 0.0,
            "regressions": [],
            "no_progress": True,
        },
        task_metadata={"benchmark_family": "repo_sandbox"},
        learned_world_signal={
            "source": "tolbert_hybrid_runtime",
            "model_family": "tolbert_ssm_v1",
            "progress_signal": 0.82,
            "risk_signal": 0.77,
            "world_progress_score": 0.8,
            "world_risk_score": 0.76,
            "decoder_world_progress_score": 0.74,
            "decoder_world_risk_score": 0.41,
            "decoder_world_entropy_mean": 0.29,
            "top_probe_reason": "respond candidate",
            "probe_count": 3,
            "world_prior_backend": "profile_conditioned",
            "world_prior_top_state": 1,
            "world_prior_top_probability": 0.73,
            "world_profile_horizons": [1, 2],
            "world_signature_token_count": 12,
            "world_transition_family": "banded",
            "world_transition_bandwidth": 2,
            "world_transition_gate": 0.61,
            "world_final_entropy_mean": 0.44,
            "ssm_last_state_norm_mean": 0.18,
            "ssm_pooled_state_norm_mean": 0.23,
        },
        latent_controls={
            "learned_world_progress_threshold": 0.55,
            "learned_world_risk_threshold": 0.55,
            "learned_world_blend_weight": 0.7,
        },
    )

    assert summary["progress_band"] == "improving"
    assert summary["risk_band"] == "blocked"
    assert summary["learned_world_state"]["source"] == "tolbert_hybrid_runtime"
    assert summary["learned_world_state"]["world_prior_backend"] == "profile_conditioned"
    assert summary["learned_world_state"]["world_profile_horizons"] == [1, 2]
    assert summary["learned_world_state"]["decoder_world_progress_score"] == 0.74
    assert summary["learned_world_state"]["world_transition_family"] == "banded"
    assert summary["learned_world_state"]["probe_count"] == 3


def test_command_control_score_uses_learned_world_signal():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.world_model_summary = {
        "present_forbidden_artifacts": ["temp.txt"],
        "missing_expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
    }
    state.latent_state_summary = {
        "risk_band": "stable",
        "progress_band": "flat",
        "retrieval_mode": "blind",
        "active_paths": ["status.txt", "temp.txt"],
        "learned_world_state": {
            "progress_signal": 0.9,
            "risk_signal": 0.9,
        },
    }

    progress_score = policy._command_control_score(state, "printf 'cleaned\\n' > status.txt")
    risky_score = policy._command_control_score(state, "rm -rf .")

    assert progress_score > risky_score


def test_command_control_score_prefers_active_path_recovery_when_learned_world_risk_is_high():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.world_model_summary = {
        "present_forbidden_artifacts": ["temp.txt"],
        "missing_expected_artifacts": [],
        "forbidden_artifacts": ["temp.txt"],
        "horizon": "long_horizon",
    }
    state.latent_state_summary = {
        "risk_band": "stable",
        "progress_band": "flat",
        "retrieval_mode": "blind",
        "active_paths": ["temp.txt"],
        "learned_world_state": {
            "progress_signal": 0.2,
            "risk_signal": 0.91,
            "world_risk_score": 0.88,
        },
    }

    recovery_score = policy._command_control_score(state, "rm -f temp.txt")
    unrelated_score = policy._command_control_score(state, "printf 'note\\n' > notes.txt")

    assert recovery_score > unrelated_score


def test_command_control_score_prefers_diagnosed_hotspot_repair_for_generic_subgoal():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.current_role = "critic"
    state.active_subgoal = "satisfy verifier contract"
    state.subgoal_diagnoses = {
        "satisfy verifier contract": {
            "path": "temp.txt",
            "signals": ["state_regression", "command_failure"],
            "summary": "temp.txt is still present; prior command failed",
            "source_role": "critic",
        }
    }
    state.world_model_summary = {
        "present_forbidden_artifacts": ["temp.txt"],
        "missing_expected_artifacts": [],
        "forbidden_artifacts": ["temp.txt"],
        "horizon": "long_horizon",
    }

    repair_score = policy._command_control_score(state, "rm -f temp.txt")
    unrelated_score = policy._command_control_score(state, "printf 'done\\n' > status.txt")

    assert repair_score > unrelated_score
    assert policy._subgoal_alignment_score(state, "rm -f temp.txt") > 0


def test_llm_policy_uses_retained_tolbert_primary_family_routing(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["workflow"],
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 0,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    task = TaskBank().get("hello_task")
    task.metadata["benchmark_family"] = "workflow"

    decision = policy.decide(AgentState(task=task))

    assert decision.action == "code_execute"
    assert "hello.txt" in decision.content
    assert decision.decision_source == "tolbert_decoder"
    assert decision.tolbert_route_mode == "primary"


def test_llm_policy_hybrid_primary_uses_world_feedback_to_choose_respond(tmp_path, monkeypatch):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 0,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "hybrid_runtime": {
                    "model_family": "tolbert_ssm_v1",
                    "primary_enabled": True,
                    "shadow_enabled": False,
                    "bundle_manifest_path": str(tmp_path / "fake_bundle.json"),
                    "preferred_device": "cpu",
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fake_bundle.json").write_text("{}", encoding="utf-8")

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )

    def _prefer_respond(state, candidates):
        del state
        respond = next(candidate for candidate in candidates if str(candidate.get("action")) == "respond")
        execute = next(candidate for candidate in candidates if str(candidate.get("action")) == "code_execute")
        return [
            {
                **respond,
                "hybrid_total_score": 9.0,
                "hybrid_decoder_world_progress_score": 0.95,
                "hybrid_decoder_world_risk_score": 0.05,
                "hybrid_decoder_world_entropy_mean": 0.21,
                "hybrid_world_progress_score": 0.8,
                "hybrid_world_risk_score": 0.2,
                "hybrid_world_transition_family": "banded",
                "hybrid_world_transition_bandwidth": 1,
                "hybrid_world_transition_gate": 0.67,
                "hybrid_model_family": "tolbert_ssm_v1",
                "reason": "stop_decision",
            },
            {
                **execute,
                "hybrid_total_score": 1.0,
                "hybrid_decoder_world_progress_score": 0.1,
                "hybrid_decoder_world_risk_score": 0.9,
                "hybrid_model_family": "tolbert_ssm_v1",
            },
        ]

    monkeypatch.setattr(policy, "_hybrid_scored_candidates", _prefer_respond)
    task = TaskBank().get("release_bundle_task")
    task.metadata["benchmark_family"] = "workflow"
    state = AgentState(task=task)
    state.world_model_summary = {
        "completion_ratio": 1.0,
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latent_state_summary = {"risk_band": "stable", "progress_band": "advancing"}

    decision = policy.decide(state)

    assert decision.action == "respond"
    assert decision.done is True
    assert decision.decision_source == "tolbert_hybrid_decoder"
    assert decision.tolbert_route_mode == "primary"
    assert decision.proposal_metadata["hybrid_decoder_world_progress_score"] == 0.95
    assert decision.proposal_metadata["hybrid_world_transition_family"] == "banded"
    assert decision.proposal_metadata["hybrid_reason"] == "stop_decision"


def test_llm_policy_prefers_novel_tolbert_action_generation_candidate(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 0,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {
                        "workflow": [
                            {
                                "template_kind": "expected_file_content",
                                "support": 3,
                                "success_count": 3,
                                "pass_rate": 1.0,
                                "provenance": ["hello_task"],
                            }
                        ]
                    },
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    task = TaskBank().get("hello_task")
    task.metadata["benchmark_family"] = "workflow"
    state = AgentState(task=task)
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "missing_expected_artifacts": ["hello.txt"],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["hello.txt"],
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert "cat <<'__AK_EOF__'" in decision.content
    assert "hello.txt" in decision.content
    assert decision.proposal_source == "expected_file_content"
    assert decision.proposal_novel is True
    assert decision.decision_source == "tolbert_proposal_decoder"
    assert decision.tolbert_route_mode == "primary"


def test_llm_policy_records_retained_tolbert_shadow_decision(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["workflow"],
                    "primary_benchmark_families": [],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )
    client = CapturingClient()
    policy = LLMDecisionPolicy(
        client,
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    task = TaskBank().get("hello_task")
    task.metadata["benchmark_family"] = "workflow"

    decision = policy.decide(AgentState(task=task))

    assert decision.decision_source in {"tolbert_decoder", "llm"}
    assert decision.tolbert_route_mode in {"shadow", "primary"}
    assert decision.shadow_decision["command"]


def test_llm_policy_retained_tolbert_primary_uses_workspace_preview_structured_edit(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="preview_edit_task",
            prompt="Update note.txt from todo to done.",
            workspace_subdir="preview_edit_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "done\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["note.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "todo\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '1s#^todo$#done#' note.txt"
    assert decision.proposal_source == "structured_edit:line_replace"
    assert decision.proposal_novel is True
    assert decision.proposal_metadata["edit_kind"] == "line_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"
    assert decision.decision_source == "tolbert_proposal_decoder"
    assert decision.tolbert_route_mode == "primary"


def test_llm_policy_retained_tolbert_primary_uses_workspace_preview_line_insert(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="preview_insert_task",
            prompt="Insert beta into note.txt.",
            workspace_subdir="preview_insert_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "alpha\nbeta\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["note.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "alpha\nomega\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2i\\\nbeta' note.txt"
    assert decision.proposal_source == "structured_edit:line_insert"
    assert decision.proposal_novel is True
    assert decision.proposal_metadata["edit_kind"] == "line_insert"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"
    assert decision.decision_source == "tolbert_proposal_decoder"
    assert decision.tolbert_route_mode == "primary"


def test_llm_policy_retained_tolbert_primary_uses_workspace_preview_line_delete(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="preview_delete_task",
            prompt="Delete beta from note.txt.",
            workspace_subdir="preview_delete_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "alpha\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["note.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "alpha\nbeta\nomega\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2,2d' note.txt"
    assert decision.proposal_source == "structured_edit:line_delete"
    assert decision.proposal_novel is True
    assert decision.proposal_metadata["edit_kind"] == "line_delete"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"
    assert decision.decision_source == "tolbert_proposal_decoder"
    assert decision.tolbert_route_mode == "primary"


def test_llm_policy_workspace_preview_structured_edit_anchors_duplicate_lines_by_line_number(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="duplicate_line_preview_edit_task",
            prompt="Update the final repeated line only.",
            workspace_subdir="duplicate_line_preview_edit_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "shared shared\nkeep\nshared fixed\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["note.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "shared shared\nkeep\nshared shared\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '3s#^shared\\ shared$#shared fixed#' note.txt"
    assert decision.proposal_source == "structured_edit:line_replace"
    assert decision.proposal_metadata["edit_kind"] == "line_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"


def test_llm_policy_workspace_preview_structured_edit_anchors_duplicate_tokens_by_line_number(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="duplicate_token_preview_edit_task",
            prompt="Update the final version token only.",
            workspace_subdir="duplicate_token_preview_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "version=1\nother=1\nversion=2\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "version=1\nother=1\nversion=1\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '3s#1#2#' config.txt"
    assert decision.proposal_source == "structured_edit:token_replace"
    assert decision.proposal_metadata["edit_kind"] == "token_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"


def test_llm_policy_workspace_preview_uses_token_insert_for_inline_addition(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="token_insert_preview_edit_task",
            prompt="Add environ to the import line.",
            workspace_subdir="token_insert_preview_edit_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "from os import path, environ\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["note.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "from os import path\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '1s#^from\\ os\\ import\\ path$#from os import path, environ#' note.txt"
    assert decision.proposal_source == "structured_edit:token_insert"
    assert decision.proposal_metadata["edit_kind"] == "token_insert"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"


def test_llm_policy_workspace_preview_uses_token_delete_for_inline_removal(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="token_delete_preview_edit_task",
            prompt="Remove the maxfail flag from the test command.",
            workspace_subdir="token_delete_preview_edit_task",
            suggested_commands=[],
            expected_files=["script.sh"],
            expected_file_contents={"script.sh": "pytest -q tests/test_api.py\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["script.sh"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["script.sh"],
        "workspace_file_previews": {
            "script.sh": {
                "content": "pytest -q --maxfail=1 tests/test_api.py\n",
                "truncated": False,
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '1s#^pytest\\ \\-q\\ \\-\\-maxfail=1\\ tests/test_api\\.py$#pytest -q tests/test_api.py#' script.sh"
    assert decision.proposal_source == "structured_edit:token_delete"
    assert decision.proposal_metadata["edit_kind"] == "token_delete"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview"


def test_llm_policy_workspace_preview_range_uses_structured_edit_for_truncated_prefix(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    expected_target = "alpha=1\nbeta=2\ngamma=1\ndelta=1\n"
    state = AgentState(
        task=TaskSpec(
            task_id="truncated_preview_edit_task",
            prompt="Repair the visible setting without rewriting the large file.",
            workspace_subdir="truncated_preview_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "alpha=1\nbeta=1\ngamma=1\ndel",
                "truncated": True,
                "edit_content": "alpha=1\nbeta=1\ngamma=1\n",
                "line_start": 1,
                "line_end": 3,
                "omitted_sha1": hashlib.sha1("delta=1\n".encode("utf-8")).hexdigest(),
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2s#1#2#' config.txt"
    assert decision.proposal_source == "structured_edit:token_replace"
    assert decision.proposal_metadata["edit_kind"] == "token_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_llm_policy_workspace_preview_range_rejects_truncated_preview_when_suffix_mismatches(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    expected_target = "alpha=1\nbeta=2\ngamma=1\ndelta=1\n"
    state = AgentState(
        task=TaskSpec(
            task_id="truncated_preview_mismatch_task",
            prompt="Fall back when the unseen suffix is not known-good.",
            workspace_subdir="truncated_preview_mismatch_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "alpha=1\nbeta=1\ngamma=1\ndel",
                "truncated": True,
                "edit_content": "alpha=1\nbeta=1\ngamma=1\n",
                "line_start": 1,
                "line_end": 3,
                "omitted_sha1": hashlib.sha1("delta=9\n".encode("utf-8")).hexdigest(),
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.proposal_source == "expected_file_content"
    assert decision.content == "cat <<'__AK_EOF__' > 'config.txt'\nalpha=1\nbeta=2\ngamma=1\ndelta=1\n\n__AK_EOF__"


def test_llm_policy_workspace_preview_range_uses_mid_file_window_line_offset(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=1\n",
        "beta=1\n",
        "gamma=1\n",
        "keep=1\n",
        "mode=old\n",
        "tail=1\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[4] = "mode=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="mid_file_window_preview_edit_task",
            prompt="Repair the visible mid-file setting without rewriting the whole file.",
            workspace_subdir="mid_file_window_preview_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "".join(current_lines[3:6]),
                "truncated": True,
                "edit_content": "".join(current_lines[3:6]),
                "line_start": 4,
                "line_end": 6,
                "omitted_prefix_sha1": hashlib.sha1("".join(current_lines[:3]).encode("utf-8")).hexdigest(),
                "omitted_suffix_sha1": hashlib.sha1("".join(current_lines[6:]).encode("utf-8")).hexdigest(),
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '5s#old#new#' config.txt"
    assert decision.proposal_source == "structured_edit:token_replace"
    assert decision.proposal_metadata["edit_kind"] == "token_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_llm_policy_workspace_preview_range_rejects_mid_file_window_when_prefix_mismatches(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=1\n",
        "beta=1\n",
        "gamma=1\n",
        "keep=1\n",
        "mode=old\n",
        "tail=1\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[4] = "mode=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="mid_file_window_preview_mismatch_task",
            prompt="Fall back when the mid-file prefix hash does not align.",
            workspace_subdir="mid_file_window_preview_mismatch_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "".join(current_lines[3:6]),
                "truncated": True,
                "edit_content": "".join(current_lines[3:6]),
                "line_start": 4,
                "line_end": 6,
                "omitted_prefix_sha1": hashlib.sha1("alpha=9\nbeta=9\ngamma=9\n".encode("utf-8")).hexdigest(),
                "omitted_suffix_sha1": hashlib.sha1("".join(current_lines[6:]).encode("utf-8")).hexdigest(),
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.proposal_source == "expected_file_content"
    assert decision.content == "cat <<'__AK_EOF__' > 'config.txt'\nalpha=1\nbeta=1\ngamma=1\nkeep=1\nmode=new\ntail=1\nomega=1\n\n__AK_EOF__"


def test_llm_policy_workspace_preview_range_can_use_later_multi_window_candidate(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=1\n",
        "beta=1\n",
        "gamma=1\n",
        "keep=1\n",
        "mode=old\n",
        "tail=1\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[1] = "beta=2\n"
    target_lines[4] = "mode=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="multi_window_preview_edit_task",
            prompt="Use the later valid retained window when the first one is unusable.",
            workspace_subdir="multi_window_preview_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "beta=1\n",
                "truncated": True,
                "edit_content": "beta=1\n",
                "target_edit_content": "beta=2\n",
                "line_start": 2,
                "line_end": 2,
                "edit_windows": [
                    {
                        "content": "beta=1\n",
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                    },
                    {
                        "content": "".join(current_lines[3:6]),
                        "truncated": True,
                        "edit_content": "".join(current_lines[3:6]),
                        "target_edit_content": "".join(target_lines[3:6]),
                        "line_start": 4,
                        "line_end": 6,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '5s#old#new#' config.txt"
    assert decision.proposal_source == "structured_edit:token_replace"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"
    assert decision.proposal_metadata["window_index"] == 1


def test_llm_policy_multi_window_workspace_preview_can_emit_combined_structured_edit(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=1\n",
        "mode=old\n",
        "middle=1\n",
        "carry=1\n",
        "tail=old\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[1] = "mode=new\n"
    target_lines[4] = "tail=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="multi_window_preview_combined_edit_task",
            prompt="Use both retained preview windows when two distant localized edits are required.",
            workspace_subdir="multi_window_preview_combined_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": current_lines[1],
                "truncated": True,
                "edit_content": current_lines[1],
                "target_edit_content": target_lines[1],
                "line_start": 2,
                "line_end": 2,
                "edit_windows": [
                    {
                        "content": current_lines[1],
                        "truncated": True,
                        "edit_content": current_lines[1],
                        "target_edit_content": target_lines[1],
                        "line_start": 2,
                        "line_end": 2,
                    },
                    {
                        "content": current_lines[4],
                        "truncated": True,
                        "edit_content": current_lines[4],
                        "target_edit_content": target_lines[4],
                        "line_start": 5,
                        "line_end": 5,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '5s#old#new#' config.txt && sed -i '2s#old#new#' config.txt"
    assert decision.proposal_source == "structured_edit:multi_edit"
    assert decision.proposal_metadata["edit_kind"] == "multi_edit"
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"
    assert decision.proposal_metadata["edit_window_count"] == 2
    assert decision.proposal_metadata["window_indices"] == [0, 1]
    assert decision.proposal_metadata["component_edit_kinds"] == ["token_replace", "token_replace"]


def test_llm_policy_multi_window_workspace_preview_can_emit_three_window_combined_structured_edit(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=1\n",
        "beta=old\n",
        "gamma=1\n",
        "middle=old\n",
        "delta=1\n",
        "tail=old\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[1] = "beta=new\n"
    target_lines[3] = "middle=new\n"
    target_lines[5] = "tail=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="three_window_preview_combined_edit_task",
            prompt="Bundle three distant localized edits from retained preview windows.",
            workspace_subdir="three_window_preview_combined_edit_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": current_lines[1],
                "truncated": True,
                "edit_content": current_lines[1],
                "target_edit_content": target_lines[1],
                "line_start": 2,
                "line_end": 2,
                "edit_windows": [
                    {
                        "content": current_lines[1],
                        "truncated": True,
                        "edit_content": current_lines[1],
                        "target_edit_content": target_lines[1],
                        "line_start": 2,
                        "line_end": 2,
                    },
                    {
                        "content": current_lines[3],
                        "truncated": True,
                        "edit_content": current_lines[3],
                        "target_edit_content": target_lines[3],
                        "line_start": 4,
                        "line_end": 4,
                    },
                    {
                        "content": current_lines[5],
                        "truncated": True,
                        "edit_content": current_lines[5],
                        "target_edit_content": target_lines[5],
                        "line_start": 6,
                        "line_end": 6,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == (
        "sed -i '6s#old#new#' config.txt && "
        "sed -i '4s#old#new#' config.txt && "
        "sed -i '2s#old#new#' config.txt"
    )
    assert decision.proposal_source == "structured_edit:multi_edit"
    assert decision.proposal_metadata["edit_kind"] == "multi_edit"
    assert decision.proposal_metadata["edit_window_count"] == 3
    assert decision.proposal_metadata["available_window_count"] == 3
    assert decision.proposal_metadata["window_indices"] == [0, 1, 2]
    assert decision.proposal_metadata["component_edit_kinds"] == [
        "token_replace",
        "token_replace",
        "token_replace",
    ]


def test_llm_policy_prefers_expected_file_write_when_retained_preview_coverage_is_partial(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    current_lines = [
        "alpha=old\n",
        "beta=1\n",
        "gamma=old\n",
        "delta=1\n",
        "epsilon=old\n",
        "zeta=1\n",
        "eta=old\n",
        "omega=1\n",
    ]
    target_lines = list(current_lines)
    target_lines[0] = "alpha=new\n"
    target_lines[2] = "gamma=new\n"
    target_lines[4] = "epsilon=new\n"
    target_lines[6] = "eta=new\n"
    expected_target = "".join(target_lines)
    state = AgentState(
        task=TaskSpec(
            task_id="partial_preview_coverage_rank_task",
            prompt="Prefer the bounded full write when retained preview windows only cover part of the file-wide change set.",
            workspace_subdir="partial_preview_coverage_rank_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": current_lines[0],
                "truncated": True,
                "edit_content": current_lines[0],
                "target_edit_content": target_lines[0],
                "line_start": 1,
                "line_end": 1,
                "retained_edit_window_count": 3,
                "total_edit_window_count": 4,
                "partial_window_coverage": True,
                "edit_windows": [
                    {
                        "content": current_lines[0],
                        "truncated": True,
                        "edit_content": current_lines[0],
                        "target_edit_content": target_lines[0],
                        "line_start": 1,
                        "line_end": 1,
                    },
                    {
                        "content": current_lines[2],
                        "truncated": True,
                        "edit_content": current_lines[2],
                        "target_edit_content": target_lines[2],
                        "line_start": 3,
                        "line_end": 3,
                    },
                    {
                        "content": current_lines[4],
                        "truncated": True,
                        "edit_content": current_lines[4],
                        "target_edit_content": target_lines[4],
                        "line_start": 5,
                        "line_end": 5,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.proposal_source == "expected_file_content"
    assert decision.content == (
        "cat <<'__AK_EOF__' > 'config.txt'\n"
        "alpha=new\nbeta=1\ngamma=new\ndelta=1\nepsilon=new\nzeta=1\neta=new\nomega=1\n\n__AK_EOF__"
    )


def test_llm_policy_prefers_expected_file_write_when_retained_preview_overlap_is_ambiguous(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    expected_target = "header\nbeta=2\ngamma=2\ntail=2\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_retained_overlap_rank_task",
            prompt="Prefer the exact bounded write when retained preview windows overlap ambiguously and no structured subset covers them all.",
            workspace_subdir="partial_retained_overlap_rank_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "beta=1\ngamma=1\n",
                "truncated": True,
                "edit_content": "beta=1\ngamma=1\n",
                "target_edit_content": "beta=2\ngamma=2\n",
                "line_start": 2,
                "line_end": 3,
                "target_line_start": 2,
                "target_line_end": 3,
                "edit_windows": [
                    {
                        "content": "beta=1\ngamma=1\n",
                        "truncated": True,
                        "edit_content": "beta=1\ngamma=1\n",
                        "target_edit_content": "beta=2\ngamma=2\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                    },
                    {
                        "content": "gamma=1\ntail=1\n",
                        "truncated": True,
                        "edit_content": "gamma=1\ntail=1\n",
                        "target_edit_content": "gamma=3\ntail=2\n",
                        "line_start": 3,
                        "line_end": 4,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.proposal_source == "expected_file_content"
    assert decision.content == (
        "cat <<'__AK_EOF__' > 'config.txt'\n"
        "header\nbeta=2\ngamma=2\ntail=2\nomega\n\n__AK_EOF__"
    )


def test_workspace_preview_window_proposals_can_emit_overlap_recovery_block_replace():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta=1\ngamma=1\n",
                "target_content": "beta=2\ngamma=2\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 0,
                "retained_window_count": 2,
                "total_window_count": 2,
            },
            {
                "baseline_content": "gamma=1\ntail=1\n",
                "target_content": "gamma=2\ntail=2\n",
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "exact_target_span": False,
                "window_index": 1,
                "retained_window_count": 2,
                "total_window_count": 2,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["covered_window_indices"] == [0, 1]
    assert block_replace["covered_window_count"] == 2
    assert block_replace["exact_window_count"] == 1
    assert block_replace["inexact_window_count"] == 1
    assert block_replace["edit_source"] == "workspace_preview_range"
    assert block_replace["command"] == "sed -i '2,4c\\\nbeta=2\\\ngamma=2\\\ntail=2' config.txt"



def test_workspace_preview_window_proposals_emit_multiple_overlap_block_replace_alternatives():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta=1\ngamma=1\n",
                "target_content": "beta=2\ngamma=2\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "gamma=1\ntail=1\n",
                "target_content": "gamma=2\ntail=2\n",
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "tail=1\nomega=1\n",
                "target_content": "tail=2\nomega=2\n",
                "line_offset": 3,
                "target_line_offset": 3,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    block_replaces = [
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices")
    ]

    assert [proposal["window_indices"] for proposal in block_replaces] == [[0, 1, 2], [0, 1], [1, 2]]
    assert [proposal["span_line_count"] for proposal in block_replaces] == [4, 3, 3]


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha=1\n",
                "target_content": "alpha=2\n",
                "line_offset": 0,
                "target_line_offset": 0,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 0,
                "retained_window_count": 2,
                "total_window_count": 2,
            },
            {
                "baseline_content": "beta=1\n",
                "target_content": "beta=2\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 1,
                "retained_window_count": 2,
                "total_window_count": 2,
            },
        ],
    )

    region_block = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert region_block["command"] == "sed -i '1,2c\\\nalpha=2\\\nbeta=2' config.txt"
    assert region_block["exact_contiguous_region_block"] is True
    assert region_block["covered_window_indices"] == [0, 1]
    assert region_block["covered_window_count"] == 2
    assert region_block["exact_window_count"] == 2


def test_workspace_preview_window_proposals_can_emit_exact_hidden_gap_region_block_replace():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="exact_hidden_gap_region_block_probe",
            prompt="Emit one exact hidden-gap bounded block when the task target supplies the missing lines.",
            workspace_subdir="exact_hidden_gap_region_block_probe",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "alpha=2\nkeep=1\ngamma=2\nomega=1\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    region_block = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("exact_hidden_gap_region_block", False)
    )

    assert region_block.content == "sed -i '1,3c\\\nalpha=2\\\nkeep=1\\\ngamma=2' config.txt"
    assert region_block.proposal_metadata["window_indices"] == [0, 1]
    assert region_block.proposal_metadata["hidden_gap_current_line_count"] == 1
    assert region_block.proposal_metadata["hidden_gap_target_line_count"] == 1


def test_action_generation_candidates_include_syntax_motor_metadata_for_synthetic_python_edit():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="python_structured_edit_task",
            prompt="Update the service function.",
            workspace_subdir="python_structured_edit_task",
            expected_files=["service.py"],
            expected_file_contents={
                "service.py": "import os\n\ndef apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n"
            },
            metadata={
                "benchmark_family": "repository",
                "synthetic_edit_plan": [
                    {
                        "path": "service.py",
                        "edit_kind": "line_replace",
                        "baseline_content": "import os\n\ndef apply_status(value):\n    return normalize(value)\n",
                        "target_content": "import os\n\ndef apply_status(value, *, strict=False):\n    persist(value)\n    return normalize(value)\n",
                        "replacements": [
                            {
                                "line_number": 3,
                                "before_line": "def apply_status(value):",
                                "after_line": "def apply_status(value, *, strict=False):",
                            }
                        ],
                        "edit_score": 10,
                    }
                ],
            },
        )
    )
    state.world_model_summary = {
        "missing_expected_artifacts": ["service.py"],
        "unsatisfied_expected_contents": ["service.py"],
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    candidate = next(item for item in candidates if item.proposal_source == "structured_edit:line_replace")

    assert candidate.proposal_metadata["target_symbol_fqn"].endswith("apply_status")
    assert candidate.proposal_metadata["signature_change_risk"] is True
    assert candidate.proposal_metadata["call_targets"] == ["persist", "normalize"]
    assert candidate.proposal_metadata["syntax_motor"]["edited_symbol_fqn"].endswith("apply_status")


def test_action_generation_candidates_prefer_bounded_multi_edit_over_ambiguous_hidden_gap_frontier(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep=1\ngamma=2\nomega=1\n"
    ambiguous_preview_proposals = [
        {
            "command": "sed -i '1,3c\\\nalpha=2\\\nkeep=1\\\ngamma=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "allow_expected_write_fallback": True,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "hidden_gap_region_ambiguity_count": 2,
            "hidden_gap_region_frontier_count": 2,
            "hidden_gap_region_frontier_gap": 0.0,
            "hidden_gap_bounded_alternative_count": 1,
            "hidden_gap_bounded_alternative_gap": 0.0,
            "hidden_gap_region_ambiguous": True,
            "span_line_count": 3,
            "exact_hidden_gap_region_block": True,
            "step": {"edit_kind": "block_replace", "edit_score": 60},
        },
        {
            "command": "sed -i '1,3c\\\nalpha=2\\\nbridge=2\\\ngamma=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "allow_expected_write_fallback": True,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "hidden_gap_region_ambiguity_count": 2,
            "hidden_gap_region_frontier_count": 2,
            "hidden_gap_region_frontier_gap": 0.0,
            "hidden_gap_bounded_alternative_count": 1,
            "hidden_gap_bounded_alternative_gap": 0.0,
            "hidden_gap_region_ambiguous": True,
            "span_line_count": 3,
            "exact_hidden_gap_region_block": True,
            "step": {"edit_kind": "block_replace", "edit_score": 60},
        },
        {
            "command": "python - <<'EOF'\\nfrom pathlib import Path\\npath = Path('config.txt')\\ntext = path.read_text()\\ntext = text.replace('alpha=1', 'alpha=2').replace('gamma=1', 'gamma=2')\\npath.write_text(text)\\nEOF",
            "proposal_source": "structured_edit:multi_edit",
            "edit_kind": "multi_edit",
            "edit_source": "workspace_preview",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "allow_expected_write_fallback": True,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 0,
            "hidden_gap_target_line_count": 0,
            "hidden_gap_region_ambiguity_count": 2,
            "hidden_gap_region_frontier_count": 2,
            "hidden_gap_region_frontier_gap": 1.0,
            "hidden_gap_bounded_alternative_count": 0,
            "hidden_gap_bounded_alternative_gap": 0.0,
            "hidden_gap_region_ambiguous": True,
            "span_line_count": 2,
            "step": {"edit_kind": "multi_edit", "edit_score": 60},
        },
    ]

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(proposal) for proposal in ambiguous_preview_proposals],
    )

    state = AgentState(
        task=TaskSpec(
            task_id="hidden_gap_bounded_alternative_task",
            prompt="Prefer the bounded mixed repair over speculative hidden-gap frontier ties.",
            workspace_subdir="hidden_gap_bounded_alternative_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
    )
    block_candidates = [
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    ]

    assert block_candidates
    assert all(multi_edit.score > candidate.score for candidate in block_candidates)


def test_action_generation_candidates_prefer_shared_anchor_delete_block_replace_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_delete_block_task",
            prompt="Prefer one bounded delete block over two adjacent delete edits.",
            workspace_subdir="shared_anchor_delete_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1]
    )

    assert block_replace.content == "sed -i 2,3d config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_action_generation_candidates_prefer_longer_shared_anchor_delete_chain_block_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_delete_chain_block_task",
            prompt="Prefer one bounded delete block over a fragmented delete chain.",
            workspace_subdir="shared_anchor_delete_chain_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\n",
                        "target_edit_content": "",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i 2,4d config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_workspace_preview_window_proposals_allow_expected_write_fallback_when_current_hidden_gap_requires_proof(monkeypatch):
    proofless_block_proposals = [
        {
            "command": "sed -i '1,4c\\\nalpha=2\\\nkeep=1\\\nbridge=2\\\ngamma=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 2,
            "hidden_gap_target_line_count": 0,
            "current_span_line_count": 4,
            "target_span_line_count": 2,
            "span_line_count": 4,
            "step": {"edit_kind": "block_replace", "edit_score": 60},
        }
    ]

    monkeypatch.setattr(decoder_module, "_preview_window_candidate_steps", lambda **kwargs: [])
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [dict(proposal) for proposal in proofless_block_proposals],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha=1\n",
                "target_content": "alpha=2\n",
                "line_offset": 0,
                "target_line_offset": 0,
                "truncated": True,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "gamma=1\n",
                "target_content": "gamma=2\n",
                "line_offset": 3,
                "target_line_offset": 2,
                "truncated": True,
                "exact_target_span": False,
                "window_index": 1,
            },
        ],
    )

    assert len(proposals) == 1
    assert proposals[0]["hidden_gap_current_ambiguity_count"] == 0
    assert proposals[0]["hidden_gap_current_frontier_count"] == 1
    assert proposals[0]["hidden_gap_current_proof_required_count"] == 1
    assert proposals[0]["hidden_gap_current_proof_required"] is True
    assert proposals[0]["allow_expected_write_fallback"] is True


def test_annotate_hidden_gap_current_target_equivalent_ties_marks_broader_current_candidates():
    proposals = [
        {
            "command": "sed -i '1,3c\\nalpha=2\\nbridge=2\\ngamma=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "exact_target_span": True,
            "exact_window_count": 2,
            "inexact_window_count": 0,
            "covered_window_count": 2,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 0,
            "current_span_line_count": 3,
            "target_span_line_count": 3,
            "span_line_count": 3,
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "target_start_line": 1,
                "target_end_line": 3,
                "replacement": {
                    "start_line": 1,
                    "end_line": 3,
                    "content": "alpha=2\nbridge=2\ngamma=2\n",
                },
            },
        },
        {
            "command": "sed -i '1,4c\\nalpha=2\\nbridge=2\\ngamma=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "exact_target_span": True,
            "exact_window_count": 2,
            "inexact_window_count": 0,
            "covered_window_count": 2,
            "hidden_gap_current_line_count": 2,
            "hidden_gap_target_line_count": 0,
            "current_span_line_count": 4,
            "target_span_line_count": 3,
            "span_line_count": 4,
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "target_start_line": 1,
                "target_end_line": 3,
                "replacement": {
                    "start_line": 1,
                    "end_line": 4,
                    "content": "alpha=2\nbridge=2\ngamma=2\n",
                },
            },
        },
    ]

    tie_count = decoder_module._annotate_hidden_gap_current_target_equivalent_ties(proposals)

    assert tie_count == 2
    assert proposals[0]["hidden_gap_current_target_equivalent_count"] == 2
    assert proposals[1]["hidden_gap_current_target_equivalent_count"] == 2
    assert proposals[0]["hidden_gap_current_target_equivalent_ambiguous"] is True
    assert proposals[1]["hidden_gap_current_target_equivalent_ambiguous"] is True
    assert proposals[0]["hidden_gap_current_target_equivalent_gap"] == 0.0
    assert proposals[1]["hidden_gap_current_target_equivalent_gap"] > 0.0


def test_action_generation_candidates_prefer_expected_write_when_no_proof_broader_current_target_frontier_is_tied(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="current_hidden_gap_target_equivalent_tie_task",
            prompt="Prefer rewrite fallback when multiple broader-current edits reach the same target span without current-state proof.",
            workspace_subdir="current_hidden_gap_target_equivalent_tie_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "alpha=2\nbridge=2\ngamma=2\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [
            {
                "command": "sed -i '1,3c\\nalpha=2\\nbridge=2\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "hidden_gap_region_ambiguity_count": 0,
                "hidden_gap_region_frontier_count": 1,
                "hidden_gap_region_frontier_gap": 0.0,
                "hidden_gap_region_ambiguous": False,
                "hidden_gap_current_ambiguity_count": 0,
                "hidden_gap_current_frontier_count": 1,
                "hidden_gap_current_frontier_gap": 0.0,
                "hidden_gap_current_ambiguous": False,
                "hidden_gap_current_target_equivalent_count": 2,
                "hidden_gap_current_target_equivalent_gap": 0.0,
                "hidden_gap_current_target_equivalent_ambiguous": True,
                "hidden_gap_current_proof_required_count": 0,
                "hidden_gap_current_proof_required": False,
                "hidden_gap_current_line_count": 1,
                "hidden_gap_target_line_count": 0,
                "current_span_line_count": 3,
                "target_span_line_count": 3,
                "span_line_count": 3,
                "step": {
                    "path": "config.txt",
                    "edit_kind": "block_replace",
                    "target_start_line": 1,
                    "target_end_line": 3,
                    "replacement": {
                        "start_line": 1,
                        "end_line": 3,
                        "content": "alpha=2\nbridge=2\ngamma=2\n",
                    },
                },
            },
            {
                "command": "sed -i '1,4c\\nalpha=2\\nbridge=2\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "hidden_gap_region_ambiguity_count": 0,
                "hidden_gap_region_frontier_count": 1,
                "hidden_gap_region_frontier_gap": 0.0,
                "hidden_gap_region_ambiguous": False,
                "hidden_gap_current_ambiguity_count": 0,
                "hidden_gap_current_frontier_count": 1,
                "hidden_gap_current_frontier_gap": 0.0,
                "hidden_gap_current_ambiguous": False,
                "hidden_gap_current_target_equivalent_count": 2,
                "hidden_gap_current_target_equivalent_gap": 2.0,
                "hidden_gap_current_target_equivalent_ambiguous": True,
                "hidden_gap_current_proof_required_count": 0,
                "hidden_gap_current_proof_required": False,
                "hidden_gap_current_line_count": 2,
                "hidden_gap_target_line_count": 0,
                "current_span_line_count": 4,
                "target_span_line_count": 3,
                "span_line_count": 4,
                "step": {
                    "path": "config.txt",
                    "edit_kind": "block_replace",
                    "target_start_line": 1,
                    "target_end_line": 3,
                    "replacement": {
                        "start_line": 1,
                        "end_line": 4,
                        "content": "alpha=2\nbridge=2\ngamma=2\n",
                    },
                },
            },
        ],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates[0].proposal_source == "expected_file_content"
    assert (
        candidates[0].proposal_metadata["workspace_preview_hidden_gap_current_target_equivalent_count"]
        == 2
    )
    block_candidates = [
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    ]
    assert len(block_candidates) == 2
    assert all(
        candidate.proposal_metadata["hidden_gap_current_target_equivalent_count"] == 2
        for candidate in block_candidates
    )
    assert block_candidates[0].proposal_metadata["hidden_gap_current_target_equivalent_gap"] == 0.0
    assert block_candidates[1].proposal_metadata["hidden_gap_current_target_equivalent_gap"] > 0.0


def test_action_generation_candidates_prefer_expected_write_when_current_hidden_gap_frontier_is_ambiguous(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="current_hidden_gap_frontier_ambiguity_task",
            prompt="Prefer a bounded rewrite when unseen current-state lines keep broader local blocks underdetermined.",
            workspace_subdir="current_hidden_gap_frontier_ambiguity_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "alpha=2\nbridge=2\ngamma=2\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [
            {
                "command": "sed -i '1,3c\\\nalpha=2\\\nbridge=2\\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "hidden_gap_region_ambiguity_count": 0,
                "hidden_gap_region_frontier_count": 1,
                "hidden_gap_region_frontier_gap": 0.0,
                "hidden_gap_region_ambiguous": False,
                "hidden_gap_current_ambiguity_count": 2,
                "hidden_gap_current_frontier_count": 2,
                "hidden_gap_current_frontier_gap": 0.0,
                "hidden_gap_current_ambiguous": True,
                "hidden_gap_current_line_count": 1,
                "hidden_gap_target_line_count": 0,
                "current_span_line_count": 3,
                "target_span_line_count": 2,
                "span_line_count": 3,
            },
            {
                "command": "sed -i '1,4c\\\nalpha=2\\\nkeep=1\\\nbridge=2\\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "hidden_gap_region_ambiguity_count": 0,
                "hidden_gap_region_frontier_count": 1,
                "hidden_gap_region_frontier_gap": 0.0,
                "hidden_gap_region_ambiguous": False,
                "hidden_gap_current_ambiguity_count": 2,
                "hidden_gap_current_frontier_count": 2,
                "hidden_gap_current_frontier_gap": 0.0,
                "hidden_gap_current_ambiguous": True,
                "hidden_gap_current_line_count": 2,
                "hidden_gap_target_line_count": 0,
                "current_span_line_count": 4,
                "target_span_line_count": 2,
                "span_line_count": 4,
            },
        ],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates[0].proposal_source == "expected_file_content"
    assert (
        candidates[0].proposal_metadata["workspace_preview_hidden_gap_current_ambiguity_count"]
        == 2
    )
    block_candidates = [
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    ]
    assert len(block_candidates) == 2
    assert all(
        candidate.proposal_metadata["hidden_gap_current_ambiguity_count"] == 2
        for candidate in block_candidates
    )
    assert all(
        candidate.proposal_metadata["hidden_gap_region_ambiguity_count"] == 0
        for candidate in block_candidates
    )


def test_action_generation_candidates_prefer_expected_write_when_current_hidden_gap_requires_proof(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="current_hidden_gap_proof_required_task",
            prompt="Prefer fallback when a broader local block still needs unseen current-state proof.",
            workspace_subdir="current_hidden_gap_proof_required_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "alpha=2\nbridge=2\ngamma=2\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [
            {
                "command": "sed -i '1,4c\\\nalpha=2\\\nkeep=1\\\nbridge=2\\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 1,
                "inexact_window_count": 1,
                "hidden_gap_region_ambiguity_count": 0,
                "hidden_gap_region_frontier_count": 1,
                "hidden_gap_region_frontier_gap": 0.0,
                "hidden_gap_region_ambiguous": False,
                "hidden_gap_current_ambiguity_count": 0,
                "hidden_gap_current_frontier_count": 1,
                "hidden_gap_current_frontier_gap": 0.0,
                "hidden_gap_current_ambiguous": False,
                "hidden_gap_current_proof_required_count": 1,
                "hidden_gap_current_proof_required": True,
                "hidden_gap_current_line_count": 2,
                "hidden_gap_target_line_count": 0,
                "current_span_line_count": 4,
                "target_span_line_count": 2,
                "span_line_count": 4,
            }
        ],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates[0].proposal_source == "expected_file_content"
    assert (
        candidates[0].proposal_metadata["workspace_preview_hidden_gap_current_proof_required_count"]
        == 1
    )
    block_candidate = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    assert block_candidate.proposal_metadata["hidden_gap_current_proof_required_count"] == 1
    assert block_candidate.proposal_metadata["hidden_gap_current_proof_required"] is True


def test_action_generation_candidates_prefer_exact_localized_block_replace_over_alias_token_edit():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nbeta=2\ngamma=2\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="localized_alias_block_replace_task",
            prompt="Prefer an exact localized block replace over alias-prone token edits.",
            workspace_subdir="localized_alias_block_replace_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta=1\ngamma=1\n",
                        "target_edit_content": "beta=2\ngamma=2\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\ngamma=1\n",
                        "target_edit_content": "beta=2\ngamma=2\n",
                        "line_start": 2,
                        "line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.content == "sed -i '2,3c\\\nbeta=2\\\ngamma=2' config.txt"
    )
    token_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:token_replace"
        and candidate.proposal_metadata["window_index"] == 0
    )

    assert block_replace.score > token_replace.score
    assert block_replace.proposal_metadata["block_alternative_count"] >= 3
    assert block_replace.proposal_metadata["same_window_exact_block_alternative"] is True
    assert token_replace.proposal_metadata["same_window_exact_block_alternative"] is True


def test_llm_policy_prefers_localized_block_replace_over_partial_bundle_for_overlap(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    expected_target = "header\nbeta=2\ngamma=2\ntail=2\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="overlap_recovery_block_replace_task",
            prompt="Prefer a localized block replace when overlapping retained windows can be stitched safely.",
            workspace_subdir="overlap_recovery_block_replace_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "beta=1\ngamma=1\n",
                "truncated": True,
                "edit_content": "beta=1\ngamma=1\n",
                "target_edit_content": "beta=2\ngamma=2\n",
                "line_start": 2,
                "line_end": 3,
                "target_line_start": 2,
                "target_line_end": 3,
                "edit_windows": [
                    {
                        "content": "beta=1\ngamma=1\n",
                        "truncated": True,
                        "edit_content": "beta=1\ngamma=1\n",
                        "target_edit_content": "beta=2\ngamma=2\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                    },
                    {
                        "content": "gamma=1\ntail=1\n",
                        "truncated": True,
                        "edit_content": "gamma=1\ntail=1\n",
                        "target_edit_content": "gamma=2\ntail=2\n",
                        "line_start": 3,
                        "line_end": 4,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.proposal_source == "structured_edit:block_replace"
    assert decision.content == "sed -i '2,4c\\\nbeta=2\\\ngamma=2\\\ntail=2' config.txt"
    assert decision.proposal_metadata["covered_window_indices"] == [0, 1]
    assert decision.proposal_metadata["covered_window_count"] == 2
    assert decision.proposal_metadata["window_indices"] == [0, 1]
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_workspace_preview_window_proposals_can_bundle_best_non_overlapping_subset():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta=1\ngamma=1\n",
                "target_content": "beta=2\ngamma=2\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "window_index": 0,
            },
            {
                "baseline_content": "gamma=1\n",
                "target_content": "gamma=2\n",
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "window_index": 1,
            },
            {
                "baseline_content": "tail=1\n",
                "target_content": "tail=2\n",
                "line_offset": 5,
                "target_line_offset": 5,
                "truncated": True,
                "window_index": 2,
            },
        ],
    )

    composite = next(
        proposal for proposal in proposals if proposal["proposal_source"] == "structured_edit:multi_edit"
    )

    assert composite["edit_window_count"] == 2
    assert composite["available_window_count"] == 3
    assert composite["window_indices"] == [0, 2]
    assert composite["covered_window_indices"] == [0, 1, 2]
    assert composite["covered_window_count"] == 3
    assert len(composite["component_edit_kinds"]) == 2


def test_best_non_overlapping_window_subset_prefers_higher_total_edit_score():
    subset = _best_non_overlapping_window_subset(
        [
            {
                "proposal": {"step": {"edit_score": 50}},
                "current_start": 1,
                "current_end": 2,
                "target_start": 1,
                "target_end": 2,
                "window_index": 0,
            },
            {
                "proposal": {"step": {"edit_score": 1}},
                "current_start": 2,
                "current_end": 3,
                "target_start": 2,
                "target_end": 3,
                "window_index": 1,
            },
            {
                "proposal": {"step": {"edit_score": 10}},
                "current_start": 6,
                "current_end": 6,
                "target_start": 6,
                "target_end": 6,
                "window_index": 2,
            },
        ]
    )

    assert [item["window_index"] for item in subset] == [0, 2]


def test_best_non_overlapping_window_subset_prefers_exact_subset_over_inexact_superset():
    subset = _best_non_overlapping_window_subset(
        [
            {
                "proposal": {"step": {"edit_score": 10}, "exact_target_span": True, "precision_penalty": 0},
                "current_start": 1,
                "current_end": 1,
                "target_start": 1,
                "target_end": 1,
                "window_index": 0,
            },
            {
                "proposal": {"step": {"edit_score": 10}, "exact_target_span": True, "precision_penalty": 0},
                "current_start": 4,
                "current_end": 4,
                "target_start": 4,
                "target_end": 4,
                "window_index": 1,
            },
            {
                "proposal": {"step": {"edit_score": 20}, "exact_target_span": False, "precision_penalty": 3},
                "current_start": 7,
                "current_end": 8,
                "target_start": 7,
                "target_end": 8,
                "window_index": 2,
            },
        ]
    )

    assert [item["window_index"] for item in subset] == [0, 1]


def test_best_non_overlapping_window_subset_prefers_frontier_localized_block_candidate_when_hybrid_bundles_tie():
    subset = _best_non_overlapping_window_subset(
        [
            {
                "proposal": {
                    "step": {"edit_score": 20},
                    "window_indices": [0, 1],
                    "covered_window_indices": [0, 1],
                    "exact_window_count": 1,
                    "inexact_window_count": 1,
                    "safe_inexact_window_count": 1,
                    "overlap_alias_pair_count": 1,
                    "precision_penalty": 0,
                    "span_line_count": 5,
                    "overlap_component_frontier_gap": 1.0,
                },
                "current_start": 1,
                "current_end": 5,
                "target_start": 1,
                "target_end": 5,
                "window_index": 0,
                "window_indices": [0, 1],
            },
            {
                "proposal": {
                    "step": {"edit_score": 20},
                    "window_indices": [1, 2],
                    "covered_window_indices": [1, 2],
                    "exact_window_count": 1,
                    "inexact_window_count": 1,
                    "safe_inexact_window_count": 1,
                    "overlap_alias_pair_count": 1,
                    "precision_penalty": 0,
                    "span_line_count": 3,
                    "overlap_component_frontier_gap": 0.0,
                },
                "current_start": 2,
                "current_end": 4,
                "target_start": 2,
                "target_end": 4,
                "window_index": 1,
                "window_indices": [1, 2],
            },
            {
                "proposal": {
                    "step": {"edit_score": 5},
                    "window_indices": [3],
                    "covered_window_indices": [3],
                    "exact_target_span": True,
                    "exact_window_count": 1,
                    "inexact_window_count": 0,
                    "precision_penalty": 0,
                },
                "current_start": 10,
                "current_end": 10,
                "target_start": 10,
                "target_end": 10,
                "window_index": 3,
                "window_indices": [3],
            },
        ]
    )

    assert [item.get("window_indices") for item in subset] == [[1, 2], [3]]


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_same_anchor_insert_and_replace():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": "header\nalpha\nbeta2\nomega\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "beta2\n",
                "full_target_content": "header\nalpha\nbeta2\nomega\n",
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["command"] == "sed -i '2,2c\\\nalpha\\\nbeta2' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_same_anchor_insert_and_delete():
    expected_target = "header\nalpha\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["command"] == "sed -i '2,2c\\\nalpha' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_adjacent_delete_and_replace():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": "header\nbeta=new\nomega\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta=old\n",
                "target_content": "beta=new\n",
                "full_target_content": "header\nbeta=new\nomega\n",
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["command"] == "sed -i '2,3c\\\nbeta=new' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_adjacent_delete_and_insert():
    expected_target = "header\ngamma\nbeta\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "gamma\nbeta\n",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["command"] == "sed -i '2,3c\\\ngamma\\\nbeta' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True
    assert block_replace["shared_anchor_pair_kinds"] == ["adjacent_delete_insert"]


def test_workspace_preview_window_proposals_can_extend_same_anchor_insert_replace_block_with_exact_neighbor():
    expected_target = "header\nalpha\nbeta2\ngamma=new\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "beta2\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "gamma=old\n",
                "target_content": "gamma=new\n",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 3,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
    )

    assert block_replace["command"] == "sed -i '2,3c\\\nalpha\\\nbeta2\\\ngamma=new' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_three_same_anchor_windows():
    expected_target = "header\nalpha\nbeta2\ngamma\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "beta2\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "beta2\ngamma\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
    )

    assert block_replace["command"] == "sed -i '2,2c\\\nalpha\\\nbeta2\\\ngamma' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_extend_adjacent_delete_replace_block_with_exact_neighbor():
    expected_target = "header\nbeta=new\ngamma=new\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta=old\n",
                "target_content": "beta=new\n",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "gamma=old\n",
                "target_content": "gamma=new\n",
                "full_target_content": expected_target,
                "line_offset": 3,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
    )

    assert block_replace["command"] == "sed -i '2,4c\\\nbeta=new\\\ngamma=new' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_delete_chain_ending_in_replace():
    expected_target = "header\ngamma2\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "gamma\n",
                "target_content": "gamma2\n",
                "full_target_content": expected_target,
                "line_offset": 3,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    shared_anchor_blocks = [
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
        and proposal.get("shared_anchor_exact_region_block") is True
    ]

    assert len(shared_anchor_blocks) == 1
    assert shared_anchor_blocks[0]["command"] == "sed -i '2,4c\\\ngamma2' config.txt"
    assert shared_anchor_blocks[0]["exact_contiguous_region_block"] is True


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_delete_chain_ending_in_insert():
    expected_target = "header\ndelta\ngamma\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "gamma\n",
                "target_content": "delta\ngamma\n",
                "full_target_content": expected_target,
                "line_offset": 3,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    shared_anchor_blocks = [
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
        and proposal.get("shared_anchor_exact_region_block") is True
    ]

    assert len(shared_anchor_blocks) == 1
    assert shared_anchor_blocks[0]["command"] == "sed -i '2,4c\\\ndelta\\\ngamma' config.txt"
    assert shared_anchor_blocks[0]["exact_contiguous_region_block"] is True


def test_workspace_preview_window_proposals_track_absorbed_exact_neighbors_for_longer_mixed_chain():
    expected_target = "header\nbefore2\ndelta\ngamma\nafter2\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "before\n",
                "target_content": "before2\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 3,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 2,
            },
            {
                "baseline_content": "gamma\n",
                "target_content": "delta\ngamma\n",
                "full_target_content": expected_target,
                "line_offset": 4,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 3,
            },
            {
                "baseline_content": "after\n",
                "target_content": "after2\n",
                "full_target_content": expected_target,
                "line_offset": 5,
                "target_line_offset": 4,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 4,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2, 3, 4]
    )

    assert block_replace["command"] == "sed -i '2,6c\\\nbefore2\\\ndelta\\\ngamma\\\nafter2' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True
    assert block_replace["shared_anchor_pair_kinds"] == [
        "adjacent_delete_delete",
        "adjacent_delete_insert",
    ]
    assert block_replace["shared_anchor_exact_neighbor_count"] == 2


def test_workspace_preview_window_proposals_preserve_shared_anchor_metadata_when_frontier_extends_mixed_core_with_multiple_exact_neighbors(
    monkeypatch,
):
    def _shared_anchor_core(*, path: str, preview_windows: list[dict[str, object]], available_window_count: int):
        del preview_windows
        return [
            {
                "command": "sed -i '3,3c\\\nalpha' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [1, 2],
                "edit_window_count": 2,
                "covered_window_indices": [1, 2],
                "covered_window_count": 2,
                "available_window_count": available_window_count,
                "retained_window_count": available_window_count,
                "total_window_count": available_window_count,
                "partial_window_coverage": False,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "current_span_line_count": 1,
                "target_span_line_count": 1,
                "span_line_count": 1,
                "exact_contiguous_region_block": True,
                "shared_anchor_exact_region_block": True,
                "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
                "shared_anchor_exact_neighbor_count": 0,
                "shared_anchor_mixed_insert_delete": True,
                "step": {
                    "path": path,
                    "edit_kind": "block_replace",
                    "replacement": {
                        "start_line": 3,
                        "end_line": 3,
                        "before_lines": ["beta"],
                        "after_lines": ["alpha"],
                    },
                    "current_start_line": 3,
                    "current_end_line": 3,
                    "target_start_line": 3,
                    "target_end_line": 3,
                    "line_delta": 0,
                    "edit_score": 8,
                },
            }
        ]

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        _shared_anchor_core,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "before\n",
                "target_content": "before2\n",
                "full_target_content": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n",
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n",
                "line_offset": 2,
                "target_line_offset": 3,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 2,
            },
            {
                "baseline_content": "gamma\n",
                "target_content": "gamma2\n",
                "full_target_content": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n",
                "line_offset": 3,
                "target_line_offset": 3,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 3,
            },
            {
                "baseline_content": "after\n",
                "target_content": "after2\n",
                "full_target_content": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n",
                "line_offset": 4,
                "target_line_offset": 4,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 4,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2, 3, 4]
    )

    assert block_replace["command"] == "sed -i '2,5c\\\nbefore2\\\nalpha\\\ngamma2\\\nafter2' config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True
    assert block_replace["shared_anchor_pair_kinds"] == ["same_anchor_insert_delete"]
    assert block_replace["shared_anchor_exact_neighbor_count"] == 3
    assert block_replace["shared_anchor_mixed_insert_delete"] is True
    assert block_replace["synthetic_single_line_block_replace_count"] == 3


def test_workspace_preview_window_proposals_can_extend_same_anchor_insert_delete_block_with_exact_neighbor():
    expected_target = "header\nalpha\ngamma2\nomega\n"
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "beta\n",
                "target_content": "alpha\nbeta\n",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": 1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "full_target_content": expected_target,
                "line_offset": 1,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
            {
                "baseline_content": "gamma\n",
                "target_content": "gamma2\n",
                "full_target_content": expected_target,
                "line_offset": 2,
                "target_line_offset": 2,
                "truncated": True,
                "line_delta": 0,
                "exact_target_span": True,
                "window_index": 2,
            },
        ],
    )

    shared_anchor_blocks = [
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1, 2]
        and proposal.get("shared_anchor_exact_region_block") is True
    ]

    assert len(shared_anchor_blocks) == 1
    assert shared_anchor_blocks[0]["command"] == "sed -i '2,3c\\\nalpha\\\ngamma2' config.txt"
    assert shared_anchor_blocks[0]["exact_contiguous_region_block"] is True


def test_decode_action_generation_candidates_prefer_same_anchor_insert_replace_block_with_exact_neighbor_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_insert_replace_neighbor_block_task",
            prompt="Prefer one bounded block when a same-anchor insert/replace pair has an adjacent exact neighbor.",
            workspace_subdir="shared_anchor_insert_replace_neighbor_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nbeta2\ngamma=new\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 3,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "beta2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=old\n",
                        "target_edit_content": "gamma=new\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 4,
                        "target_line_end": 4,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i '2,3c\\\nalpha\\\nbeta2\\\ngamma=new' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_decode_action_generation_candidates_prefer_same_anchor_insert_delete_block_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_insert_delete_block_task",
            prompt="Prefer one bounded block when a same-anchor insert is followed by a delete on the same anchor.",
            workspace_subdir="shared_anchor_insert_delete_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 2,
                        "line_delta": -1,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1]
    )
    fragmented_candidates = [
        candidate
        for candidate in candidates
        if candidate.proposal_source.startswith("structured_edit:")
        and candidate.proposal_source != "structured_edit:block_replace"
    ]

    assert block_replace.content == "sed -i '2,2c\\\nalpha' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_pair_kinds"] == [
        "same_anchor_insert_delete"
    ]
    assert block_replace.proposal_metadata["shared_anchor_mixed_insert_delete"] is True
    assert fragmented_candidates
    assert block_replace.score > max(candidate.score for candidate in fragmented_candidates)


def test_decode_action_generation_candidates_prefer_adjacent_delete_insert_block_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="adjacent_delete_insert_block_task",
            prompt="Prefer one bounded block when an adjacent delete is followed by an insert on the shifted anchor.",
            workspace_subdir="adjacent_delete_insert_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\ngamma\nbeta\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "gamma\nbeta\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1]
    )

    assert block_replace.content == "sed -i '2,3c\\\ngamma\\\nbeta' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_pair_kinds"] == [
        "adjacent_delete_insert"
    ]
    assert block_replace.proposal_metadata["shared_anchor_mixed_insert_delete"] is True
    assert block_replace.score > multi_edit.score


def test_decode_action_generation_candidates_prefer_delete_chain_replace_block_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="delete_chain_replace_block_task",
            prompt="Prefer one bounded block when a delete chain terminates in a replace.",
            workspace_subdir="delete_chain_replace_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\ngamma2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\n",
                        "target_edit_content": "gamma2\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i '2,4c\\\ngamma2' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_decode_action_generation_candidates_prefer_delete_chain_insert_block_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="delete_chain_insert_block_task",
            prompt="Prefer one bounded block when a delete chain terminates in an insert.",
            workspace_subdir="delete_chain_insert_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\ndelta\ngamma\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\n",
                        "target_edit_content": "delta\ngamma\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i '2,4c\\\ndelta\\\ngamma' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_decode_action_generation_candidates_reward_longer_mixed_chain_with_more_exact_neighbors():
    policy = LLMDecisionPolicy(MockLLMClient())

    def _block_replace_for_state(expected_target: str, edit_windows: list[dict[str, object]], window_indices: list[int]):
        state = AgentState(
            task=TaskSpec(
                task_id="longer_mixed_chain_neighbor_block_task",
                prompt="Prefer the broader bounded block when a mixed shared-anchor chain absorbs more exact neighbors.",
                workspace_subdir="longer_mixed_chain_neighbor_block_task",
                suggested_commands=[],
                expected_files=["config.txt"],
                expected_file_contents={"config.txt": expected_target},
                metadata={"benchmark_family": "bounded"},
            )
        )
        state.world_model_summary = {
            "completion_ratio": 0.0,
            "existing_expected_artifacts": ["config.txt"],
            "missing_expected_artifacts": [],
            "present_forbidden_artifacts": [],
            "changed_preserved_artifacts": [],
            "unsatisfied_expected_contents": ["config.txt"],
            "workspace_file_previews": {
                "config.txt": {
                    "edit_windows": edit_windows,
                }
            },
        }

        candidates = decode_action_generation_candidates(
            state=state,
            world_model=policy.world_model,
            skill_library=policy.skill_library,
            top_skill=None,
            retrieval_guidance={"recommended_command_spans": []},
            blocked_commands=[],
            proposal_policy={
                "enabled": True,
                "max_candidates": 20,
                "proposal_score_bias": 0.0,
                "novel_command_bonus": 0.0,
                "verifier_alignment_bonus": 0.0,
                "expected_file_template_bonus": 0.0,
                "cleanup_template_bonus": 0.0,
                "template_preferences": {},
            },
            rollout_policy={},
            command_score_fn=lambda state, command: 0.0,
            normalize_command_fn=lambda command, workspace_subdir: command,
            canonicalize_command_fn=lambda command: command,
        )

        return next(
            candidate
            for candidate in candidates
            if candidate.proposal_source == "structured_edit:block_replace"
            and candidate.proposal_metadata.get("window_indices") == window_indices
        )

    four_window_block = _block_replace_for_state(
        "header\nbefore2\ndelta\ngamma\nomega\n",
        [
            {
                "truncated": True,
                "edit_content": "before\n",
                "target_edit_content": "before2\n",
                "line_start": 2,
                "line_end": 2,
                "target_line_start": 2,
                "target_line_end": 2,
                "line_delta": 0,
            },
            {
                "truncated": True,
                "edit_content": "alpha\n",
                "target_edit_content": "",
                "line_start": 3,
                "line_end": 3,
                "target_line_start": 3,
                "target_line_end": 2,
                "line_delta": -1,
            },
            {
                "truncated": True,
                "edit_content": "beta\n",
                "target_edit_content": "",
                "line_start": 4,
                "line_end": 4,
                "target_line_start": 3,
                "target_line_end": 2,
                "line_delta": -1,
            },
            {
                "truncated": True,
                "edit_content": "gamma\n",
                "target_edit_content": "delta\ngamma\n",
                "line_start": 5,
                "line_end": 5,
                "target_line_start": 3,
                "target_line_end": 4,
                "line_delta": 1,
            },
        ],
        [0, 1, 2, 3],
    )
    five_window_block = _block_replace_for_state(
        "header\nbefore2\ndelta\ngamma\nafter2\nomega\n",
        [
            {
                "truncated": True,
                "edit_content": "before\n",
                "target_edit_content": "before2\n",
                "line_start": 2,
                "line_end": 2,
                "target_line_start": 2,
                "target_line_end": 2,
                "line_delta": 0,
            },
            {
                "truncated": True,
                "edit_content": "alpha\n",
                "target_edit_content": "",
                "line_start": 3,
                "line_end": 3,
                "target_line_start": 3,
                "target_line_end": 2,
                "line_delta": -1,
            },
            {
                "truncated": True,
                "edit_content": "beta\n",
                "target_edit_content": "",
                "line_start": 4,
                "line_end": 4,
                "target_line_start": 3,
                "target_line_end": 2,
                "line_delta": -1,
            },
            {
                "truncated": True,
                "edit_content": "gamma\n",
                "target_edit_content": "delta\ngamma\n",
                "line_start": 5,
                "line_end": 5,
                "target_line_start": 3,
                "target_line_end": 4,
                "line_delta": 1,
            },
            {
                "truncated": True,
                "edit_content": "after\n",
                "target_edit_content": "after2\n",
                "line_start": 6,
                "line_end": 6,
                "target_line_start": 5,
                "target_line_end": 5,
                "line_delta": 0,
            },
        ],
        [0, 1, 2, 3, 4],
    )

    assert four_window_block.proposal_metadata["shared_anchor_exact_neighbor_count"] == 1
    assert five_window_block.proposal_metadata["shared_anchor_exact_neighbor_count"] == 2
    assert five_window_block.score > four_window_block.score


def test_decode_action_generation_candidates_preserve_frontier_shared_anchor_metadata_with_multiple_exact_neighbors(
    monkeypatch,
):
    def _shared_anchor_core(*, path: str, preview_windows: list[dict[str, object]], available_window_count: int):
        del preview_windows
        return [
            {
                "command": "sed -i '3,3c\\\nalpha' config.txt",
                "proposal_source": "structured_edit:block_replace",
                "edit_kind": "block_replace",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [1, 2],
                "edit_window_count": 2,
                "covered_window_indices": [1, 2],
                "covered_window_count": 2,
                "available_window_count": available_window_count,
                "retained_window_count": available_window_count,
                "total_window_count": available_window_count,
                "partial_window_coverage": False,
                "exact_target_span": True,
                "precision_penalty": 0,
                "exact_window_count": 2,
                "inexact_window_count": 0,
                "current_span_line_count": 1,
                "target_span_line_count": 1,
                "span_line_count": 1,
                "exact_contiguous_region_block": True,
                "shared_anchor_exact_region_block": True,
                "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
                "shared_anchor_exact_neighbor_count": 0,
                "shared_anchor_mixed_insert_delete": True,
                "step": {
                    "path": path,
                    "edit_kind": "block_replace",
                    "replacement": {
                        "start_line": 3,
                        "end_line": 3,
                        "before_lines": ["beta"],
                        "after_lines": ["alpha"],
                    },
                    "current_start_line": 3,
                    "current_end_line": 3,
                    "target_start_line": 3,
                    "target_end_line": 3,
                    "line_delta": 0,
                    "edit_score": 8,
                },
            }
        ]

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        _shared_anchor_core,
    )

    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="frontier_shared_anchor_metadata_fidelity_task",
            prompt="Prefer the broader bounded block when frontier extension preserves shared-anchor metadata across multiple exact neighbors.",
            workspace_subdir="frontier_shared_anchor_metadata_fidelity_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nbefore2\nalpha\ngamma2\nafter2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "before\n",
                        "target_edit_content": "before2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 4,
                        "line_delta": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 4,
                        "target_line_end": 3,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\n",
                        "target_edit_content": "gamma2\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 4,
                        "target_line_end": 4,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "after\n",
                        "target_edit_content": "after2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2, 3, 4]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2, 3, 4]
    )

    assert block_replace.content == "sed -i '2,5c\\\nbefore2\\\nalpha\\\ngamma2\\\nafter2' config.txt"
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_pair_kinds"] == [
        "same_anchor_insert_delete"
    ]
    assert block_replace.proposal_metadata["shared_anchor_exact_neighbor_count"] == 3
    assert block_replace.proposal_metadata["shared_anchor_mixed_insert_delete"] is True
    assert block_replace.proposal_metadata["synthetic_single_line_block_replace_count"] == 3
    assert block_replace.score > multi_edit.score


def test_frontier_bridge_hybrid_preserves_shared_anchor_metadata():
    path = "config.txt"
    shared_anchor = {
        "command": "sed -i '3,3c\\\nalpha' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [1, 2],
        "edit_window_count": 2,
        "covered_window_indices": [1, 2],
        "covered_window_count": 2,
        "available_window_count": 4,
        "retained_window_count": 4,
        "total_window_count": 4,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 3,
                "end_line": 3,
                "before_lines": ["beta"],
                "after_lines": ["alpha"],
            },
            "current_start_line": 3,
            "current_end_line": 3,
            "target_start_line": 3,
            "target_end_line": 3,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    bridged_hidden_gap = {
        "command": "sed -i '4,5c\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [3, 4],
        "edit_window_count": 2,
        "covered_window_indices": [3, 4],
        "covered_window_count": 2,
        "available_window_count": 4,
        "retained_window_count": 4,
        "total_window_count": 4,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 2,
        "target_span_line_count": 2,
        "span_line_count": 2,
        "bridged_hidden_gap_region_block": True,
        "hidden_gap_bridge_count": 1,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "explicit_hidden_gap_current_proof": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 4,
                "end_line": 5,
                "before_lines": ["gamma", "delta"],
                "after_lines": ["gamma2", "delta2"],
            },
            "current_start_line": 4,
            "current_end_line": 5,
            "target_start_line": 4,
            "target_end_line": 5,
            "line_delta": 0,
            "edit_score": 9,
        },
    }

    candidates = decoder_module._compose_workspace_preview_frontier_block_replace_proposals(
        path=path,
        proposals=[shared_anchor, bridged_hidden_gap],
        available_window_count=4,
    )

    assert candidates
    hybrid = candidates[0]
    assert hybrid["window_indices"] == [1, 2, 3, 4]
    assert hybrid["bridged_hidden_gap_region_block"] is True
    assert hybrid["shared_anchor_exact_region_block"] is True
    assert hybrid["shared_anchor_pair_kinds"] == ["same_anchor_insert_delete"]
    assert hybrid["shared_anchor_core_count"] == 1
    assert hybrid["shared_anchor_hybrid_component_count"] == 1
    assert hybrid["shared_anchor_mixed_insert_delete"] is True


def test_frontier_exact_hidden_gap_hybrid_preserves_shared_anchor_proof_metadata():
    path = "config.txt"
    shared_anchor = {
        "command": "sed -i '3,3c\\\nalpha' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [1, 2],
        "edit_window_count": 2,
        "covered_window_indices": [1, 2],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 3,
                "end_line": 3,
                "before_lines": ["beta"],
                "after_lines": ["alpha"],
            },
            "current_start_line": 3,
            "current_end_line": 3,
            "target_start_line": 3,
            "target_end_line": 3,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    exact_hidden_gap = {
        "command": "sed -i '4,5c\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [3, 4],
        "edit_window_count": 2,
        "covered_window_indices": [3, 4],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 2,
        "target_span_line_count": 2,
        "span_line_count": 2,
        "exact_hidden_gap_region_block": True,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "hidden_gap_target_from_expected_content": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 4,
                "end_line": 5,
                "before_lines": ["gamma", "delta"],
                "after_lines": ["gamma2", "delta2"],
            },
            "current_start_line": 4,
            "current_end_line": 5,
            "target_start_line": 4,
            "target_end_line": 5,
            "line_delta": 0,
            "edit_score": 9,
        },
    }

    candidates = decoder_module._compose_workspace_preview_frontier_block_replace_proposals(
        path=path,
        proposals=[shared_anchor, exact_hidden_gap],
        available_window_count=5,
    )

    assert candidates
    hybrid = candidates[0]
    assert hybrid["window_indices"] == [1, 2, 3, 4]
    assert hybrid["exact_hidden_gap_region_block"] is True
    assert hybrid["shared_anchor_exact_region_block"] is True
    assert hybrid["shared_anchor_pair_kinds"] == ["same_anchor_insert_delete"]
    assert hybrid["shared_anchor_core_count"] == 1
    assert hybrid["shared_anchor_hybrid_component_count"] == 1
    assert hybrid["hidden_gap_target_from_expected_content"] is True
    assert hybrid["shared_anchor_mixed_insert_delete"] is True


def test_frontier_current_proof_hybrid_preserves_multi_hybrid_shared_anchor_metadata():
    path = "config.txt"
    overlap_hybrid = {
        "command": "sed -i '3,3c\\\nalpha' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [1, 2],
        "edit_window_count": 2,
        "covered_window_indices": [1, 2],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 0,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 1,
        "overlap_component_frontier_gap": 0.0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 3,
                "end_line": 3,
                "before_lines": ["beta"],
                "after_lines": ["alpha"],
            },
            "current_start_line": 3,
            "current_end_line": 3,
            "target_start_line": 3,
            "target_end_line": 3,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    current_proof_hybrid = {
        "command": "sed -i '4,6c\\\ngamma2\\\nkeep_b\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [3, 4],
        "edit_window_count": 2,
        "covered_window_indices": [3, 4],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 1,
        "inexact_window_count": 1,
        "safe_inexact_window_count": 1,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 1,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 3,
        "target_span_line_count": 3,
        "span_line_count": 3,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 4,
                "end_line": 6,
                "before_lines": ["gamma", "keep_b", "delta"],
                "after_lines": ["gamma2", "keep_b", "delta2"],
            },
            "current_start_line": 4,
            "current_end_line": 6,
            "target_start_line": 4,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 9,
        },
    }

    candidates = decoder_module._compose_workspace_preview_frontier_block_replace_proposals(
        path=path,
        proposals=[overlap_hybrid, current_proof_hybrid],
        available_window_count=5,
    )

    assert candidates
    hybrid = candidates[0]
    assert hybrid["window_indices"] == [1, 2, 3, 4]
    assert hybrid["shared_anchor_exact_region_block"] is True
    assert hybrid["shared_anchor_pair_kinds"] == [
        "same_anchor_insert_delete",
        "adjacent_delete_insert",
    ]
    assert hybrid["shared_anchor_core_count"] == 2
    assert hybrid["shared_anchor_hybrid_component_count"] == 2
    assert hybrid["recovered_conflicting_alias"] is True
    assert hybrid["current_proof_region_block"] is True
    assert hybrid["current_proof_span_count"] == 2
    assert hybrid["hidden_gap_current_from_line_span_proof"] is True
    assert hybrid["hidden_gap_target_from_expected_content"] is True


def test_frontier_preserves_multi_hybrid_shared_anchor_metadata_across_multiple_cores():
    path = "config.txt"
    overlap_hybrid = {
        "command": "sed -i '3,3c\\\nalpha' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [1, 2],
        "edit_window_count": 2,
        "covered_window_indices": [1, 2],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 0,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 1,
        "overlap_component_frontier_gap": 0.0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 3,
                "end_line": 3,
                "before_lines": ["beta"],
                "after_lines": ["alpha"],
            },
            "current_start_line": 3,
            "current_end_line": 3,
            "target_start_line": 3,
            "target_end_line": 3,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    exact_hidden_gap_hybrid = {
        "command": "sed -i '4,5c\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [3, 4],
        "edit_window_count": 2,
        "covered_window_indices": [3, 4],
        "covered_window_count": 2,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 2,
        "target_span_line_count": 2,
        "span_line_count": 2,
        "exact_hidden_gap_region_block": True,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "hidden_gap_target_from_expected_content": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 4,
                "end_line": 5,
                "before_lines": ["gamma", "delta"],
                "after_lines": ["gamma2", "delta2"],
            },
            "current_start_line": 4,
            "current_end_line": 5,
            "target_start_line": 4,
            "target_end_line": 5,
            "line_delta": 0,
            "edit_score": 9,
        },
    }

    candidates = decoder_module._compose_workspace_preview_frontier_block_replace_proposals(
        path=path,
        proposals=[overlap_hybrid, exact_hidden_gap_hybrid],
        available_window_count=5,
    )

    assert candidates
    hybrid = candidates[0]
    assert hybrid["window_indices"] == [1, 2, 3, 4]
    assert hybrid["shared_anchor_exact_region_block"] is True
    assert hybrid["shared_anchor_pair_kinds"] == [
        "same_anchor_insert_delete",
        "adjacent_delete_insert",
    ]
    assert hybrid["shared_anchor_core_count"] == 2
    assert hybrid["shared_anchor_hybrid_component_count"] == 2
    assert hybrid["recovered_conflicting_alias"] is True
    assert hybrid["exact_hidden_gap_region_block"] is True
    assert hybrid["hidden_gap_target_from_expected_content"] is True
    assert hybrid["shared_anchor_mixed_insert_delete"] is True


def test_frontier_preserves_multiple_shared_anchor_cores():
    path = "config.txt"
    shared_anchor_left = {
        "command": "sed -i '3,3c\\\nalpha' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [1, 2],
        "edit_window_count": 2,
        "covered_window_indices": [1, 2],
        "covered_window_count": 2,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 3,
                "end_line": 3,
                "before_lines": ["beta"],
                "after_lines": ["alpha"],
            },
            "current_start_line": 3,
            "current_end_line": 3,
            "target_start_line": 3,
            "target_end_line": 3,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    exact_neighbor = {
        "command": "sed -i '4,4c\\\ngamma2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [3],
        "edit_window_count": 1,
        "covered_window_indices": [3],
        "covered_window_count": 1,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 1,
        "inexact_window_count": 0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": False,
        "shared_anchor_pair_kinds": [],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 0,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": False,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 4,
                "end_line": 4,
                "before_lines": ["gamma"],
                "after_lines": ["gamma2"],
            },
            "current_start_line": 4,
            "current_end_line": 4,
            "target_start_line": 4,
            "target_end_line": 4,
            "line_delta": 0,
            "edit_score": 8,
        },
    }
    shared_anchor_right = {
        "command": "sed -i '5,5c\\\ntheta' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [4, 5],
        "edit_window_count": 2,
        "covered_window_indices": [4, 5],
        "covered_window_count": 2,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "current_span_line_count": 1,
        "target_span_line_count": 1,
        "span_line_count": 1,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": path,
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 5,
                "end_line": 5,
                "before_lines": ["delta"],
                "after_lines": ["theta"],
            },
            "current_start_line": 5,
            "current_end_line": 5,
            "target_start_line": 5,
            "target_end_line": 5,
            "line_delta": 0,
            "edit_score": 8,
        },
    }

    candidates = decoder_module._compose_workspace_preview_frontier_block_replace_proposals(
        path=path,
        proposals=[shared_anchor_left, exact_neighbor, shared_anchor_right],
        available_window_count=6,
    )

    assert candidates
    block = candidates[0]
    assert block["window_indices"] == [1, 2, 3, 4, 5]
    assert block["shared_anchor_exact_region_block"] is True
    assert block["shared_anchor_pair_kinds"] == [
        "same_anchor_insert_delete",
        "adjacent_delete_insert",
    ]
    assert block["shared_anchor_exact_neighbor_count"] == 1
    assert block["shared_anchor_core_count"] == 2
    assert block["shared_anchor_hybrid_component_count"] == 0


def test_decode_action_generation_candidates_prefer_same_anchor_insert_delete_block_with_exact_neighbor_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_insert_delete_neighbor_block_task",
            prompt="Prefer one bounded block when a same-anchor insert/delete pair has an adjacent exact neighbor.",
            workspace_subdir="shared_anchor_insert_delete_neighbor_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\ngamma2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 2,
                        "line_delta": -1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\n",
                        "target_edit_content": "gamma2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i '2,3c\\\nalpha\\\ngamma2' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_decode_action_generation_candidates_prefer_shifted_delete_replace_block_with_exact_neighbor_over_multi_edit():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shifted_delete_replace_neighbor_block_task",
            prompt="Prefer one bounded block when a shifted delete/replace pair has an adjacent exact neighbor.",
            workspace_subdir="shifted_delete_replace_neighbor_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nbeta=new\ngamma=new\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=old\n",
                        "target_edit_content": "beta=new\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 2,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=old\n",
                        "target_edit_content": "gamma=new\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 20,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
        and candidate.proposal_metadata.get("window_indices") == [0, 1, 2]
    )

    assert block_replace.content == "sed -i '2,4c\\\nbeta=new\\\ngamma=new' config.txt"
    assert block_replace.proposal_metadata["exact_contiguous_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.score > multi_edit.score


def test_workspace_preview_window_proposals_can_emit_exact_contiguous_region_block_replace_for_adjacent_shared_anchor_deletes():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    block_replace = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"] == "structured_edit:block_replace"
        and proposal.get("window_indices") == [0, 1]
    )

    assert block_replace["command"] == "sed -i 2,3d config.txt"
    assert block_replace["exact_contiguous_region_block"] is True
    assert block_replace["shared_anchor_exact_region_block"] is True


def test_workspace_preview_window_proposals_allow_adjacent_deletes_with_shared_target_anchor():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    composite = next(
        proposal for proposal in proposals if proposal["proposal_source"] == "structured_edit:multi_edit"
    )

    assert composite["command"] == "sed -i '3,3d' config.txt && sed -i '2,2d' config.txt"
    assert composite["window_indices"] == [0, 1]
    assert composite["component_edit_kinds"] == ["line_delete", "line_delete"]


def test_structured_edit_window_bonus_scales_for_three_plus_windows():
    assert _structured_edit_window_bonus({"edit_window_count": 1, "available_window_count": 3}) == 0.0
    assert _structured_edit_window_bonus({"edit_window_count": 3, "available_window_count": 3}) > _structured_edit_window_bonus(
        {"edit_window_count": 2, "available_window_count": 2}
    )
    assert _structured_edit_window_bonus({"edit_window_count": 4, "available_window_count": 4}) > _structured_edit_window_bonus(
        {"edit_window_count": 3, "available_window_count": 3}
    )


def test_action_generation_candidates_keep_expected_write_fallback_for_mixed_precision_multi_window_preview():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nkeep=1\nbeta=new\ngamma=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="mixed_precision_preview_fallback_task",
            prompt="Choose between a mixed-precision bundle and a bounded rewrite fallback.",
            workspace_subdir="mixed_precision_preview_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\n",
                        "target_edit_content": "alpha=new\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=old\ngamma=old\n",
                        "target_edit_content": "beta=new\ngamma=new\n",
                        "line_start": 4,
                        "line_end": 5,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert any(candidate.proposal_source == "structured_edit:multi_edit" for candidate in candidates)
    assert any(candidate.content.startswith("cat <<'__AK_EOF__' > 'config.txt'") for candidate in candidates)


def test_action_generation_candidates_prefer_world_model_exact_edit_proof_over_expected_write():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nbeta=2\nkeep_b\ngamma=2\nkeep_c\ndelta=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="world_model_exact_edit_proof_task",
            prompt="Prefer explicit world-model current-span proof when retained previews alone are partial.",
            workspace_subdir="world_model_exact_edit_proof_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 2,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "exact_edit_window_proofs": [
                    {
                        "window_index": 3,
                        "explicit_current_span_proof": True,
                        "line_start": 7,
                        "line_end": 7,
                        "target_line_start": 7,
                        "target_line_end": 7,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source.startswith("structured_edit:")
    assert "delta=2" in candidates[0].content
    assert candidates[0].proposal_metadata.get("window_indices") == [0, 1, 2, 3]
    assert candidates[0].proposal_metadata.get("covered_window_count") == 4
    assert all(candidate.proposal_source != "expected_file_content" for candidate in candidates[:3])


def test_action_generation_candidates_prefer_proof_only_current_bridge_over_expected_write():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    target_gap_lines = [f"new_gap_{index:03d}=y\n" for index in range(1, 41)]
    expected_target = "".join(
        [
            "prefix=ok\n",
            "alpha=2\n",
            *target_gap_lines,
            "omega=2\n",
            "suffix=ok\n",
        ]
    )
    state = AgentState(
        task=TaskSpec(
            task_id="proof_only_current_bridge_task",
            prompt="Prefer the bounded bridge when the world model provides exact hidden-gap span proof without inline current payload.",
            workspace_subdir="proof_only_current_bridge_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "omega=1\n",
                        "target_edit_content": "omega=2\n",
                        "line_start": 43,
                        "line_end": 43,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_windows": [
                    {
                        "bridge_window_indices": [0, 1],
                        "truncated": True,
                        "line_start": 2,
                        "line_end": 43,
                        "target_line_start": 2,
                        "target_line_end": 43,
                        "hidden_gap_current_line_start": 3,
                        "hidden_gap_current_line_end": 42,
                        "hidden_gap_target_line_start": 3,
                        "hidden_gap_target_line_end": 42,
                        "hidden_gap_current_content": "",
                        "hidden_gap_target_content": "",
                        "hidden_gap_current_from_line_span_proof": True,
                        "hidden_gap_target_from_expected_content": True,
                        "hidden_gap_current_line_count": 40,
                        "hidden_gap_target_line_count": 40,
                        "explicit_hidden_gap_current_proof": True,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["hidden_gap_current_from_line_span_proof"] is True
    assert candidates[0].proposal_metadata["hidden_gap_target_from_expected_content"] is True


def test_action_generation_candidates_prefer_expected_write_when_current_proof_region_is_only_partial():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid=2\nkeep_c\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_region_task",
            prompt="Prefer expected-content fallback when the local hidden-current region is only partially proven.",
            workspace_subdir="partial_current_proof_region_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 5,
                        "target_line_start": 1,
                        "target_line_end": 5,
                        "current_proof_span_count": 2,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "",
                                "target_content": "keep_c\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": True,
                            },
                        ],
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["fallback_from_workspace_preview"] is True
    assert all(
        candidate.proposal_metadata.get("current_proof_region_block") is not True
        for candidate in candidates
    )
    structured_candidate = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source.startswith("structured_edit:")
    )
    assert structured_candidate.proposal_metadata["hidden_gap_current_partial_proof_count"] == 1
    assert (
        structured_candidate.proposal_metadata["hidden_gap_current_partial_proof_missing_line_count"]
        == 1
    )
    assert structured_candidate.proposal_metadata[
        "hidden_gap_current_partial_proof_opaque_spans"
    ] == []
    assert structured_candidate.proposal_metadata["allow_expected_write_fallback"] is True


def test_action_generation_candidates_prefer_localized_block_when_partial_current_proof_opaque_span_is_internal():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid=2\nkeep_b\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_internal_opaque_task",
            prompt="Prefer a localized bounded rewrite when the only opaque current-proof gap is strictly internal to an otherwise exact region.",
            workspace_subdir="partial_current_proof_internal_opaque_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 5,
                        "target_line_start": 1,
                        "target_line_end": 5,
                        "current_proof_span_count": 2,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 2,
                        "current_proof_missing_line_count": 1,
                        "current_proof_missing_span_count": 1,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 3,
                                "target_line_start": 3,
                                "target_line_end": 3,
                                "reason": "no_adjacent_pair_bridge",
                            }
                        ],
                        "current_proof_opaque_span_count": 1,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_partial_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_partial_coverage"] is True
    assert candidates[0].proposal_metadata["current_proof_opaque_span_count"] == 1
    assert candidates[0].proposal_metadata["current_proof_opaque_boundary_touch_count"] == 0
    assert candidates[0].proposal_metadata["allow_expected_write_fallback"] is True
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_keep_expected_write_when_partial_current_proof_opaque_span_touches_boundary():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid=2\nkeep_b\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_boundary_opaque_task",
            prompt="Keep whole-file fallback preferred when the opaque current-proof gap touches the bounded region boundary.",
            workspace_subdir="partial_current_proof_boundary_opaque_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 5,
                        "target_line_start": 1,
                        "target_line_end": 5,
                        "current_proof_span_count": 2,
                        "current_proof_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 3,
                                "target_line_start": 3,
                                "target_line_end": 3,
                                "current_content": "",
                                "target_content": "mid=2\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 2,
                        "current_proof_missing_line_count": 1,
                        "current_proof_missing_span_count": 1,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 1,
                                "current_line_end": 1,
                                "target_line_start": 1,
                                "target_line_end": 1,
                                "reason": "missing_current_proof",
                            }
                        ],
                        "current_proof_opaque_span_count": 1,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["fallback_from_workspace_preview"] is True
    assert all(
        candidate.proposal_metadata.get("current_proof_region_block") is not True
        for candidate in candidates
    )


def test_action_generation_candidates_prefer_localized_block_when_partial_current_proof_has_sparse_internal_opaque_islands():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid_a=2\nkeep_b\nmid_b=2\nkeep_c\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_sparse_multi_opaque_task",
            prompt="Prefer a localized bounded rewrite when several opaque current-proof islands are all internal and sparse.",
            workspace_subdir="partial_current_proof_sparse_multi_opaque_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 7,
                        "line_end": 7,
                        "target_line_start": 7,
                        "target_line_end": 7,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 7,
                        "target_line_start": 1,
                        "target_line_end": 7,
                        "current_proof_span_count": 3,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 6,
                                "current_line_end": 6,
                                "target_line_start": 6,
                                "target_line_end": 6,
                                "current_content": "",
                                "target_content": "keep_c\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 3,
                        "current_proof_missing_line_count": 2,
                        "current_proof_missing_span_count": 2,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 3,
                                "target_line_start": 3,
                                "target_line_end": 3,
                                "reason": "no_adjacent_pair_bridge",
                            },
                            {
                                "current_line_start": 5,
                                "current_line_end": 5,
                                "target_line_start": 5,
                                "target_line_end": 5,
                                "reason": "no_adjacent_pair_bridge",
                            },
                        ],
                        "current_proof_opaque_span_count": 2,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_partial_region_block"] is True
    assert (
        candidates[0].proposal_metadata["current_proof_partial_region_topology"]
        == "sparse_internal_multi_opaque"
    )
    assert candidates[0].proposal_metadata["current_proof_opaque_span_count"] == 2
    assert candidates[0].proposal_metadata["current_proof_opaque_boundary_touch_count"] == 0
    assert candidates[0].proposal_metadata["current_proof_opaque_max_span_line_count"] == 1
    assert candidates[0].proposal_metadata["allow_expected_write_fallback"] is True
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_expected_write_when_partial_current_proof_has_dense_internal_opaque_islands():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid_a=2\nmid_b=2\nkeep_b\nmid_c=2\nkeep_c\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_dense_multi_opaque_task",
            prompt="Prefer whole-file fallback when multiple opaque current-proof islands are too coarse to justify a localized bounded rewrite.",
            workspace_subdir="partial_current_proof_dense_multi_opaque_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 8,
                        "line_end": 8,
                        "target_line_start": 8,
                        "target_line_end": 8,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 8,
                        "target_line_start": 1,
                        "target_line_end": 8,
                        "current_proof_span_count": 3,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 5,
                                "current_line_end": 5,
                                "target_line_start": 5,
                                "target_line_end": 5,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 7,
                                "current_line_end": 7,
                                "target_line_start": 7,
                                "target_line_end": 7,
                                "current_content": "",
                                "target_content": "keep_c\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 3,
                        "current_proof_missing_line_count": 3,
                        "current_proof_missing_span_count": 2,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 4,
                                "target_line_start": 3,
                                "target_line_end": 4,
                                "reason": "no_adjacent_pair_bridge",
                            },
                            {
                                "current_line_start": 6,
                                "current_line_end": 6,
                                "target_line_start": 6,
                                "target_line_end": 6,
                                "reason": "no_adjacent_pair_bridge",
                            },
                        ],
                        "current_proof_opaque_span_count": 2,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_partial_region_block"] is True
    assert (
        candidates[0].proposal_metadata["current_proof_partial_region_topology"]
        == "single_internal_opaque"
    )
    structured_candidate = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    assert structured_candidate.proposal_metadata[
        "hidden_gap_current_partial_proof_topologies"
    ] == ["mixed_internal_opaque"]
    assert (
        structured_candidate.proposal_metadata[
            "hidden_gap_current_partial_proof_admissible_opaque_span_count"
        ]
        == 1
    )
    assert (
        structured_candidate.proposal_metadata[
            "hidden_gap_current_partial_proof_blocking_opaque_span_count"
        ]
        == 1
    )
    assert (
        structured_candidate.proposal_metadata[
            "hidden_gap_current_partial_proof_coarse_opaque_span_count"
        ]
        == 1
    )
    assert (
        structured_candidate.proposal_metadata[
            "hidden_gap_current_partial_proof_mixed_opaque_topology"
        ]
        is True
    )
    assert structured_candidate.proposal_metadata["current_proof_factorized_subregion"] is True
    assert (
        structured_candidate.proposal_metadata["current_proof_parent_partial_region_topology"]
        == "mixed_internal_opaque"
    )
    assert structured_candidate.proposal_metadata["allow_expected_write_fallback"] is True
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_keep_expected_write_when_partial_current_proof_is_only_coarse_internal_opaque():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid_a=2\nmid_b=2\nkeep_b\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_coarse_internal_only_task",
            prompt="Keep whole-file fallback preferred when partial current-proof only exposes coarse internal opaque spans and no safe local subregion.",
            workspace_subdir="partial_current_proof_coarse_internal_only_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 6,
                        "line_end": 6,
                        "target_line_start": 6,
                        "target_line_end": 6,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 6,
                        "target_line_start": 1,
                        "target_line_end": 6,
                        "current_proof_span_count": 2,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 5,
                                "current_line_end": 5,
                                "target_line_start": 5,
                                "target_line_end": 5,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 2,
                        "current_proof_missing_line_count": 2,
                        "current_proof_missing_span_count": 1,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 4,
                                "target_line_start": 3,
                                "target_line_end": 4,
                                "reason": "coarse_missing_current_proof",
                            }
                        ],
                        "current_proof_opaque_span_count": 1,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["fallback_from_workspace_preview"] is True
    assert all(
        candidate.proposal_metadata.get("current_proof_region_block") is not True
        for candidate in candidates
    )


def test_action_generation_candidates_factorize_safe_subregion_from_mixed_partial_current_proof():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nmid_a=2\nkeep_b\nmid_b=2\nmid_c=2\nkeep_c\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_current_proof_factorized_subregion_task",
            prompt="Preserve a localized bounded rewrite for the safe subregion when a larger mixed partial-proof region also contains a blocking opaque island.",
            workspace_subdir="partial_current_proof_factorized_subregion_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 2,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "full_target_content": expected_target,
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 1,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "full_target_content": expected_target,
                        "line_start": 8,
                        "line_end": 8,
                        "target_line_start": 8,
                        "target_line_end": 8,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "proof_region_index": 0,
                        "window_indices": [0, 1],
                        "line_start": 1,
                        "line_end": 8,
                        "target_line_start": 1,
                        "target_line_end": 8,
                        "current_proof_span_count": 3,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                            {
                                "current_line_start": 7,
                                "current_line_end": 7,
                                "target_line_start": 7,
                                "target_line_end": 7,
                                "current_content": "",
                                "target_content": "keep_c\n",
                                "current_from_line_span_proof": True,
                                "target_from_expected_content": True,
                            },
                        ],
                        "current_proof_complete": False,
                        "current_proof_partial_coverage": True,
                        "current_proof_covered_line_count": 3,
                        "current_proof_missing_line_count": 3,
                        "current_proof_missing_span_count": 2,
                        "current_proof_opaque_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 3,
                                "target_line_start": 3,
                                "target_line_end": 3,
                                "reason": "no_adjacent_pair_bridge",
                            },
                            {
                                "current_line_start": 5,
                                "current_line_end": 6,
                                "target_line_start": 5,
                                "target_line_end": 6,
                                "reason": "coarse_missing_current_proof",
                            },
                        ],
                        "current_proof_opaque_span_count": 2,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_partial_region_block"] is True
    assert (
        candidates[0].proposal_metadata["current_proof_partial_region_topology"]
        == "single_internal_opaque"
    )
    assert candidates[0].proposal_metadata["current_proof_factorized_subregion"] is True
    assert (
        candidates[0].proposal_metadata["current_proof_parent_partial_region_topology"]
        == "mixed_internal_opaque"
    )
    assert (
        candidates[0].proposal_metadata["hidden_gap_current_partial_proof_mixed_opaque_topology"]
        is True
    )
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_extend_bridge_run_across_omitted_exact_proof_window():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nbeta=2\nkeep_b\ngamma=2\nkeep_c\ndelta=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="bridge_run_with_omitted_proof_task",
            prompt="Prefer one bridge-backed bounded block across retained windows plus an omitted exact-proof tail.",
            workspace_subdir="bridge_run_with_omitted_proof_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 3,
                "total_edit_window_count": 4,
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 2,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "exact_edit_window_proofs": [
                    {
                        "window_index": 3,
                        "explicit_current_span_proof": True,
                        "line_start": 7,
                        "line_end": 7,
                        "target_line_start": 7,
                        "target_line_end": 7,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    }
                ],
                "bridged_edit_window_runs": [
                    {
                        "bridge_window_indices": [0, 1, 2, 3],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 7,
                        "target_line_start": 1,
                        "target_line_end": 7,
                        "hidden_gap_current_line_count": 3,
                        "hidden_gap_target_line_count": 3,
                        "line_delta": 0,
                        "explicit_hidden_gap_current_proof": True,
                        "bridge_segments": [
                            {
                                "bridge_window_indices": [0, 1],
                                "left_window_index": 0,
                                "right_window_index": 1,
                                "line_start": 1,
                                "line_end": 3,
                                "target_line_start": 1,
                                "target_line_end": 3,
                                "hidden_gap_current_line_start": 2,
                                "hidden_gap_current_line_end": 2,
                                "hidden_gap_target_line_start": 2,
                                "hidden_gap_target_line_end": 2,
                                "hidden_gap_current_content": "keep_a\n",
                                "hidden_gap_target_content": "keep_a\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 1,
                                "line_delta": 0,
                                "explicit_hidden_gap_current_proof": True,
                            },
                            {
                                "bridge_window_indices": [1, 2],
                                "left_window_index": 1,
                                "right_window_index": 2,
                                "line_start": 3,
                                "line_end": 5,
                                "target_line_start": 3,
                                "target_line_end": 5,
                                "hidden_gap_current_line_start": 4,
                                "hidden_gap_current_line_end": 4,
                                "hidden_gap_target_line_start": 4,
                                "hidden_gap_target_line_end": 4,
                                "hidden_gap_current_content": "keep_b\n",
                                "hidden_gap_target_content": "keep_b\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 1,
                                "line_delta": 0,
                                "explicit_hidden_gap_current_proof": True,
                            },
                            {
                                "bridge_window_indices": [2, 3],
                                "left_window_index": 2,
                                "right_window_index": 3,
                                "line_start": 5,
                                "line_end": 7,
                                "target_line_start": 5,
                                "target_line_end": 7,
                                "hidden_gap_current_line_start": 6,
                                "hidden_gap_current_line_end": 6,
                                "hidden_gap_target_line_start": 6,
                                "hidden_gap_target_line_end": 6,
                                "hidden_gap_current_content": "keep_c\n",
                                "hidden_gap_target_content": "keep_c\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 1,
                                "line_delta": 0,
                                "explicit_hidden_gap_current_proof": True,
                            },
                        ],
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,7c\\\nalpha=2\\\nkeep_a\\\nbeta=2\\\nkeep_b\\\ngamma=2\\\nkeep_c\\\ndelta=2' config.txt"
    )
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2, 3]
    assert candidates[0].proposal_metadata["covered_window_count"] == 4
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_multi_span_current_proof_region_without_bridge_chain():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep_a\nbeta=2\nkeep_b\ngamma=2\nkeep_c\ndelta=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="multi_span_current_proof_region_task",
            prompt="Prefer the bounded region when the world model provides multiple disjoint hidden-current proof spans without a bridge chain payload.",
            workspace_subdir="multi_span_current_proof_region_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 4,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 2,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "exact_edit_window_proofs": [
                    {
                        "window_index": 1,
                        "explicit_current_span_proof": True,
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 3,
                        "explicit_current_span_proof": True,
                        "line_start": 7,
                        "line_end": 7,
                        "target_line_start": 7,
                        "target_line_end": 7,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "window_indices": [0, 1, 2, 3],
                        "line_start": 1,
                        "line_end": 7,
                        "target_line_start": 1,
                        "target_line_end": 7,
                        "current_proof_span_count": 3,
                        "current_proof_spans": [
                            {
                                "current_line_start": 2,
                                "current_line_end": 2,
                                "target_line_start": 2,
                                "target_line_end": 2,
                                "current_content": "keep_a\n",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": False,
                            },
                            {
                                "current_line_start": 4,
                                "current_line_end": 4,
                                "target_line_start": 4,
                                "target_line_end": 4,
                                "current_content": "keep_b\n",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": False,
                            },
                            {
                                "current_line_start": 6,
                                "current_line_end": 6,
                                "target_line_start": 6,
                                "target_line_end": 6,
                                "current_content": "keep_c\n",
                                "target_content": "keep_c\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": False,
                            },
                        ],
                        "truncated": True,
                        "explicit_hidden_gap_current_proof": True,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,7c\\\nalpha=2\\\nkeep_a\\\nbeta=2\\\nkeep_b\\\ngamma=2\\\nkeep_c\\\ndelta=2' config.txt"
    )
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["current_proof_span_count"] == 3
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2, 3]
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_sparse_current_proof_region_across_nonbridge_joins():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nbeta=2\nkeep_a\ngamma=2\ndelta=2\nkeep_b\nepsilon=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="sparse_current_proof_region_task",
            prompt="Prefer the bounded region when current-proof spans are separated by zero-gap joins instead of one consecutive bridge chain.",
            workspace_subdir="sparse_current_proof_region_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 3,
                "total_edit_window_count": 5,
                "edit_windows": [
                    {
                        "window_index": 0,
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 2,
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 4,
                        "target_line_end": 4,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 4,
                        "truncated": True,
                        "edit_content": "epsilon=1\n",
                        "target_edit_content": "epsilon=2\n",
                        "line_start": 7,
                        "line_end": 7,
                        "target_line_start": 7,
                        "target_line_end": 7,
                        "line_delta": 0,
                    },
                ],
                "exact_edit_window_proofs": [
                    {
                        "window_index": 1,
                        "explicit_current_span_proof": True,
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    },
                    {
                        "window_index": 3,
                        "explicit_current_span_proof": True,
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "current_line_count": 1,
                        "target_line_count": 1,
                        "line_delta": 0,
                    },
                ],
                "hidden_gap_current_proof_regions": [
                    {
                        "window_indices": [0, 1, 2, 3, 4],
                        "line_start": 1,
                        "line_end": 7,
                        "target_line_start": 1,
                        "target_line_end": 7,
                        "current_proof_span_count": 2,
                        "current_proof_spans": [
                            {
                                "current_line_start": 3,
                                "current_line_end": 3,
                                "target_line_start": 3,
                                "target_line_end": 3,
                                "current_content": "keep_a\n",
                                "target_content": "keep_a\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": False,
                            },
                            {
                                "current_line_start": 6,
                                "current_line_end": 6,
                                "target_line_start": 6,
                                "target_line_end": 6,
                                "current_content": "keep_b\n",
                                "target_content": "keep_b\n",
                                "current_from_line_span_proof": False,
                                "target_from_expected_content": False,
                            },
                        ],
                        "truncated": True,
                        "explicit_hidden_gap_current_proof": True,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,7c\\\nalpha=2\\\nbeta=2\\\nkeep_a\\\ngamma=2\\\ndelta=2\\\nkeep_b\\\nepsilon=2' config.txt"
    )
    assert candidates[0].proposal_metadata["current_proof_region_block"] is True
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2, 3, 4]
    assert candidates[0].proposal_metadata["covered_window_count"] == 5
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_bridged_hidden_gap_region_block_over_expected_write():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nkeep=1\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="bridged_hidden_gap_region_block_task",
            prompt="Prefer the exact bounded bridge when the world model provides explicit hidden-gap current proof.",
            workspace_subdir="bridged_hidden_gap_region_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_windows": [
                    {
                        "bridge_window_indices": [0, 1],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 3,
                        "target_line_start": 1,
                        "target_line_end": 3,
                        "hidden_gap_current_line_start": 2,
                        "hidden_gap_current_line_end": 2,
                        "hidden_gap_target_line_start": 2,
                        "hidden_gap_target_line_end": 2,
                        "hidden_gap_current_content": "keep=1\n",
                        "hidden_gap_target_content": "keep=1\n",
                        "hidden_gap_current_line_count": 1,
                        "hidden_gap_target_line_count": 1,
                        "explicit_hidden_gap_current_proof": True,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == "sed -i '1,3c\\\nalpha=2\\\nkeep=1\\\ngamma=2' config.txt"
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["explicit_hidden_gap_current_proof"] is True
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_current_proof_bridge_with_expected_target_reconstruction():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nadd_1\nadd_2\nbeta=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="bridged_hidden_gap_expected_target_reconstruction_task",
            prompt="Prefer the bounded bridge when the world model supplies current proof and target span coordinates.",
            workspace_subdir="bridged_hidden_gap_expected_target_reconstruction_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "line_delta": 1,
                    },
                ],
                "bridged_edit_windows": [
                    {
                        "bridge_window_indices": [0, 1],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 3,
                        "target_line_start": 1,
                        "target_line_end": 4,
                        "hidden_gap_current_line_start": 2,
                        "hidden_gap_current_line_end": 2,
                        "hidden_gap_target_line_start": 2,
                        "hidden_gap_target_line_end": 3,
                        "hidden_gap_current_content": "old_gap\n",
                        "hidden_gap_target_content": "",
                        "hidden_gap_target_from_expected_content": True,
                        "hidden_gap_current_line_count": 1,
                        "hidden_gap_target_line_count": 2,
                        "explicit_hidden_gap_current_proof": True,
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,3c\\\nalpha=2\\\nadd_1\\\nadd_2\\\nbeta=2' config.txt"
    )
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["explicit_hidden_gap_current_proof"] is True
    assert candidates[0].proposal_metadata["hidden_gap_target_from_expected_content"] is True
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_chained_bridged_hidden_gap_region_block_over_expected_write():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nadd_1\nadd_2\nbeta=2\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="chained_bridged_hidden_gap_region_block_task",
            prompt="Prefer one exact bounded block when consecutive explicit bridges cover the full hidden-gap region.",
            workspace_subdir="chained_bridged_hidden_gap_region_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_windows": [
                    {
                        "bridge_window_indices": [0, 1],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 3,
                        "target_line_start": 1,
                        "target_line_end": 4,
                        "hidden_gap_current_line_start": 2,
                        "hidden_gap_current_line_end": 2,
                        "hidden_gap_target_line_start": 2,
                        "hidden_gap_target_line_end": 3,
                        "hidden_gap_current_content": "old_gap\n",
                        "hidden_gap_target_content": "add_1\nadd_2\n",
                        "hidden_gap_current_line_count": 1,
                        "hidden_gap_target_line_count": 2,
                        "explicit_hidden_gap_current_proof": True,
                    },
                    {
                        "bridge_window_indices": [1, 2],
                        "truncated": True,
                        "line_start": 3,
                        "line_end": 5,
                        "target_line_start": 4,
                        "target_line_end": 5,
                        "hidden_gap_current_line_start": 4,
                        "hidden_gap_current_line_end": 4,
                        "hidden_gap_target_line_start": 5,
                        "hidden_gap_target_line_end": 4,
                        "hidden_gap_current_content": "drop_me\n",
                        "hidden_gap_target_content": "",
                        "hidden_gap_current_line_count": 1,
                        "hidden_gap_target_line_count": 0,
                        "explicit_hidden_gap_current_proof": True,
                    },
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,5c\\\nalpha=2\\\nadd_1\\\nadd_2\\\nbeta=2\\\ngamma=2' config.txt"
    )
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2]
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_bridge_run_payload_over_expected_write():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nadd_1\nadd_2\nbeta=2\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="bridged_hidden_gap_run_task",
            prompt="Prefer the world-model bridge run payload over broader fallback when the hidden gaps are fully explicit.",
            workspace_subdir="bridged_hidden_gap_run_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_window_runs": [
                    {
                        "bridge_window_indices": [0, 1, 2],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 5,
                        "target_line_start": 1,
                        "target_line_end": 5,
                        "hidden_gap_current_line_count": 2,
                        "hidden_gap_target_line_count": 2,
                        "line_delta": 0,
                        "explicit_hidden_gap_current_proof": True,
                        "bridge_segments": [
                            {
                                "bridge_window_indices": [0, 1],
                                "left_window_index": 0,
                                "right_window_index": 1,
                                "line_start": 1,
                                "line_end": 3,
                                "target_line_start": 1,
                                "target_line_end": 4,
                                "hidden_gap_current_line_start": 2,
                                "hidden_gap_current_line_end": 2,
                                "hidden_gap_target_line_start": 2,
                                "hidden_gap_target_line_end": 3,
                                "hidden_gap_current_content": "old_gap\n",
                                "hidden_gap_target_content": "add_1\nadd_2\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 2,
                                "line_delta": 1,
                                "explicit_hidden_gap_current_proof": True,
                            },
                            {
                                "bridge_window_indices": [1, 2],
                                "left_window_index": 1,
                                "right_window_index": 2,
                                "line_start": 3,
                                "line_end": 5,
                                "target_line_start": 4,
                                "target_line_end": 5,
                                "hidden_gap_current_line_start": 4,
                                "hidden_gap_current_line_end": 4,
                                "hidden_gap_target_line_start": 5,
                                "hidden_gap_target_line_end": 4,
                                "hidden_gap_current_content": "drop_me\n",
                                "hidden_gap_target_content": "",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 0,
                                "line_delta": -1,
                                "explicit_hidden_gap_current_proof": True,
                            },
                        ],
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '1,5c\\\nalpha=2\\\nadd_1\\\nadd_2\\\nbeta=2\\\ngamma=2' config.txt"
    )
    assert candidates[0].proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2]
    assert candidates[0].proposal_metadata["hidden_gap_bridge_count"] == 2
    assert any(candidate.proposal_source == "expected_file_content" for candidate in candidates[1:])


def test_action_generation_candidates_prefer_maximal_bridge_run_frontier_over_shorter_bridge_run():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nadd_1\nadd_2\nbeta=2\ngamma=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="bridged_hidden_gap_run_frontier_task",
            prompt="Prefer the maximal bridge-run frontier over a shorter bridge-run alternative.",
            workspace_subdir="bridged_hidden_gap_run_frontier_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=1\n",
                        "target_edit_content": "beta=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 5,
                        "line_end": 5,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_window_runs": [
                    {
                        "bridge_window_indices": [0, 1],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 3,
                        "target_line_start": 1,
                        "target_line_end": 4,
                        "hidden_gap_current_line_count": 1,
                        "hidden_gap_target_line_count": 2,
                        "line_delta": 1,
                        "explicit_hidden_gap_current_proof": True,
                        "bridge_segments": [
                            {
                                "bridge_window_indices": [0, 1],
                                "left_window_index": 0,
                                "right_window_index": 1,
                                "line_start": 1,
                                "line_end": 3,
                                "target_line_start": 1,
                                "target_line_end": 4,
                                "hidden_gap_current_line_start": 2,
                                "hidden_gap_current_line_end": 2,
                                "hidden_gap_target_line_start": 2,
                                "hidden_gap_target_line_end": 3,
                                "hidden_gap_current_content": "old_gap\n",
                                "hidden_gap_target_content": "add_1\nadd_2\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 2,
                                "line_delta": 1,
                                "explicit_hidden_gap_current_proof": True,
                            }
                        ],
                    },
                    {
                        "bridge_window_indices": [0, 1, 2],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 5,
                        "target_line_start": 1,
                        "target_line_end": 5,
                        "hidden_gap_current_line_count": 2,
                        "hidden_gap_target_line_count": 2,
                        "line_delta": 0,
                        "explicit_hidden_gap_current_proof": True,
                        "bridge_segments": [
                            {
                                "bridge_window_indices": [0, 1],
                                "left_window_index": 0,
                                "right_window_index": 1,
                                "line_start": 1,
                                "line_end": 3,
                                "target_line_start": 1,
                                "target_line_end": 4,
                                "hidden_gap_current_line_start": 2,
                                "hidden_gap_current_line_end": 2,
                                "hidden_gap_target_line_start": 2,
                                "hidden_gap_target_line_end": 3,
                                "hidden_gap_current_content": "old_gap\n",
                                "hidden_gap_target_content": "add_1\nadd_2\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 2,
                                "line_delta": 1,
                                "explicit_hidden_gap_current_proof": True,
                            },
                            {
                                "bridge_window_indices": [1, 2],
                                "left_window_index": 1,
                                "right_window_index": 2,
                                "line_start": 3,
                                "line_end": 5,
                                "target_line_start": 4,
                                "target_line_end": 5,
                                "hidden_gap_current_line_start": 4,
                                "hidden_gap_current_line_end": 4,
                                "hidden_gap_target_line_start": 5,
                                "hidden_gap_target_line_end": 4,
                                "hidden_gap_current_content": "drop_me\n",
                                "hidden_gap_target_content": "",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 0,
                                "line_delta": -1,
                                "explicit_hidden_gap_current_proof": True,
                            },
                        ],
                    },
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    maximal_bridge_run = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata["window_indices"] == [0, 1, 2]
    )
    shorter_bridge_run = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.proposal_metadata["window_indices"] == [0, 1]
    )

    assert candidates[0].content == (
        "sed -i '1,5c\\\nalpha=2\\\nadd_1\\\nadd_2\\\nbeta=2\\\ngamma=2' config.txt"
    )
    assert maximal_bridge_run.score > shorter_bridge_run.score
    assert maximal_bridge_run.proposal_metadata["bridge_run_frontier_gap"] == 0.0
    assert shorter_bridge_run.proposal_metadata["bridge_run_frontier_gap"] > 0.0


def test_action_generation_candidates_prefer_exact_localized_block_over_partial_bridge_run():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nbeta=2\nkeep_a\ngamma=2\nkeep_b\ndelta=2\nhidden=2\n"
    state = AgentState(
        task=TaskSpec(
            task_id="partial_bridge_run_vs_localized_exact_task",
            prompt="Prefer the smaller exact localized block over a partial maximal bridge run.",
            workspace_subdir="partial_bridge_run_vs_localized_exact_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 3,
                "total_edit_window_count": 4,
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\nbeta=1\n",
                        "target_edit_content": "alpha=2\nbeta=2\n",
                        "line_start": 1,
                        "line_end": 2,
                        "target_line_start": 1,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 4,
                        "line_end": 4,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "delta=1\n",
                        "target_edit_content": "delta=2\n",
                        "line_start": 6,
                        "line_end": 6,
                        "line_delta": 0,
                    },
                ],
                "bridged_edit_window_runs": [
                    {
                        "bridge_window_indices": [0, 1, 2],
                        "truncated": True,
                        "line_start": 1,
                        "line_end": 6,
                        "target_line_start": 1,
                        "target_line_end": 6,
                        "hidden_gap_current_line_count": 2,
                        "hidden_gap_target_line_count": 2,
                        "line_delta": 0,
                        "explicit_hidden_gap_current_proof": True,
                        "bridge_segments": [
                            {
                                "bridge_window_indices": [0, 1],
                                "left_window_index": 0,
                                "right_window_index": 1,
                                "line_start": 1,
                                "line_end": 4,
                                "target_line_start": 1,
                                "target_line_end": 4,
                                "hidden_gap_current_line_start": 3,
                                "hidden_gap_current_line_end": 3,
                                "hidden_gap_target_line_start": 3,
                                "hidden_gap_target_line_end": 3,
                                "hidden_gap_current_content": "keep_a\n",
                                "hidden_gap_target_content": "keep_a\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 1,
                                "line_delta": 0,
                                "explicit_hidden_gap_current_proof": True,
                            },
                            {
                                "bridge_window_indices": [1, 2],
                                "left_window_index": 1,
                                "right_window_index": 2,
                                "line_start": 4,
                                "line_end": 6,
                                "target_line_start": 4,
                                "target_line_end": 6,
                                "hidden_gap_current_line_start": 5,
                                "hidden_gap_current_line_end": 5,
                                "hidden_gap_target_line_start": 5,
                                "hidden_gap_target_line_end": 5,
                                "hidden_gap_current_content": "keep_b\n",
                                "hidden_gap_target_content": "keep_b\n",
                                "hidden_gap_current_line_count": 1,
                                "hidden_gap_target_line_count": 1,
                                "line_delta": 0,
                                "explicit_hidden_gap_current_proof": True,
                            },
                        ],
                    }
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    exact_localized_block = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.content == "sed -i '1,2c\\\nalpha=2\\\nbeta=2' config.txt"
    )
    partial_bridge_run = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
        and candidate.content == (
            "sed -i '1,6c\\\nalpha=2\\\nbeta=2\\\nkeep_a\\\ngamma=2\\\nkeep_b\\\ndelta=2' config.txt"
        )
    )

    assert exact_localized_block.score > partial_bridge_run.score
    assert partial_bridge_run.proposal_metadata["partial_window_coverage"] is True
    assert partial_bridge_run.proposal_metadata["bridge_run_exact_localized_alternative_count"] >= 1
    assert exact_localized_block.proposal_metadata["bridge_run_partial_alternative_count"] >= 1


def test_action_generation_candidates_prefer_alias_recovery_bundle_over_expected_write_fallback():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nbeta=new\ngamma=new\ndelta=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="alias_recovery_bundle_task",
            prompt="Prefer a localized overlap-recovery bundle over a broad rewrite fallback.",
            workspace_subdir="alias_recovery_bundle_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\nbeta=old\n",
                        "target_edit_content": "alpha=new\nbeta=new\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=old\ngamma=old\n",
                        "target_edit_content": "beta=new\ngamma=new\n",
                        "line_start": 3,
                        "line_end": 4,
                    },
                    {
                        "truncated": True,
                        "edit_content": "delta=old\n",
                        "target_edit_content": "delta=new\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:multi_edit"
    assert candidates[0].content == (
        "sed -i '5s#old#new#' config.txt && sed -i '2,4c\\\nalpha=new\\\nbeta=new\\\ngamma=new' config.txt"
    )
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2]
    assert candidates[0].proposal_metadata["overlap_alias_pair_count"] == 1
    assert candidates[0].proposal_metadata["safe_inexact_window_count"] == 1
    assert all(candidate.proposal_source != "expected_file_content" for candidate in candidates)


def test_action_generation_candidates_prefer_conflicting_alias_recovery_bundle_over_expected_write_fallback():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nbeta=new\ngamma=new\ndelta=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="conflicting_alias_recovery_bundle_task",
            prompt="Prefer a localized recovered overlap bundle over a broad rewrite fallback.",
            workspace_subdir="conflicting_alias_recovery_bundle_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\nbeta=old\n",
                        "target_edit_content": "alpha=new\nbeta=new\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=old\ngamma=old\n",
                        "target_edit_content": "beta=stale\ngamma=new\n",
                        "line_start": 3,
                        "line_end": 4,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "delta=old\n",
                        "target_edit_content": "delta=new\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "structured_edit:block_replace"
    assert candidates[0].content == (
        "sed -i '2,5c\\\nalpha=new\\\nbeta=new\\\ngamma=new\\\ndelta=new' config.txt"
    )
    assert candidates[0].proposal_metadata["window_indices"] == [0, 1, 2]
    assert candidates[0].proposal_metadata["covered_window_indices"] == [0, 1, 2]
    assert candidates[0].proposal_metadata["recovered_conflicting_alias"] is True
    assert all(candidate.proposal_source != "expected_file_content" for candidate in candidates)


def test_action_generation_candidates_prefer_shared_anchor_overlap_hybrid_block_over_multi_edit(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_overlap_hybrid_block_task",
            prompt="Prefer one bounded block when a shared-anchor core merges with a recovered overlap block.",
            workspace_subdir="shared_anchor_overlap_hybrid_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\ngamma2\ndelta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\ndelta\n",
                        "target_edit_content": "gamma2\ndelta2\n",
                        "line_start": 3,
                        "line_end": 4,
                        "target_line_start": 3,
                        "target_line_end": 4,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    hybrid_block = {
        "command": "sed -i '2,4c\\\nalpha\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3],
        "edit_window_count": 4,
        "covered_window_indices": [0, 1, 2, 3],
        "covered_window_count": 4,
        "available_window_count": 4,
        "retained_window_count": 4,
        "total_window_count": 4,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 3,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 0,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "current_span_line_count": 3,
        "target_span_line_count": 3,
        "span_line_count": 3,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 17,
        },
    }
    fragmented_multi_edit = dict(hybrid_block)
    fragmented_multi_edit.update(
        {
            "command": (
                "sed -i '2s#beta#alpha#' config.txt && sed -i '3,4c\\\n"
                "gamma2\\\ndelta2' config.txt"
            ),
            "proposal_source": "structured_edit:multi_edit",
            "edit_kind": "multi_edit",
            "exact_target_span": False,
            "component_edit_kinds": ["token_replace", "block_replace"],
            "step": {
                "edit_kind": "multi_edit",
                "edit_score": 14,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(hybrid_block), dict(fragmented_multi_edit)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
    )

    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_hybrid_component_count"] == 1
    assert block_replace.proposal_metadata["recovered_conflicting_alias"] is True
    assert block_replace.score > multi_edit.score


def test_action_generation_candidates_prefer_shared_anchor_exact_hidden_gap_hybrid_block_over_multi_edit(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_exact_hidden_gap_hybrid_block_task",
            prompt="Prefer one bounded block when a shared-anchor core merges with an exact hidden-gap proof block.",
            workspace_subdir="shared_anchor_exact_hidden_gap_hybrid_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\ngamma2\ndelta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma\ndelta\n",
                        "target_edit_content": "gamma2\ndelta2\n",
                        "line_start": 3,
                        "line_end": 4,
                        "target_line_start": 3,
                        "target_line_end": 4,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    hybrid_block = {
        "command": "sed -i '2,4c\\\nalpha\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3],
        "edit_window_count": 4,
        "covered_window_indices": [0, 1, 2, 3],
        "covered_window_count": 4,
        "available_window_count": 4,
        "retained_window_count": 4,
        "total_window_count": 4,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 0,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 3,
        "target_span_line_count": 3,
        "span_line_count": 3,
        "exact_hidden_gap_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 2,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 17,
        },
    }
    fragmented_multi_edit = dict(hybrid_block)
    fragmented_multi_edit.update(
        {
            "command": (
                "sed -i '2s#beta#alpha#' config.txt && sed -i '3,4c\\\n"
                "gamma2\\\ndelta2' config.txt"
            ),
            "proposal_source": "structured_edit:multi_edit",
            "edit_kind": "multi_edit",
            "exact_target_span": False,
            "component_edit_kinds": ["token_replace", "block_replace"],
            "step": {
                "edit_kind": "multi_edit",
                "edit_score": 14,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(hybrid_block), dict(fragmented_multi_edit)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
    )

    assert block_replace.proposal_metadata["exact_hidden_gap_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert block_replace.proposal_metadata["shared_anchor_hybrid_component_count"] == 1
    assert block_replace.proposal_metadata["hidden_gap_target_from_expected_content"] is True
    assert block_replace.score > multi_edit.score


def test_action_generation_candidates_prefer_multi_core_shared_anchor_block_over_single_core_variant(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="multi_core_shared_anchor_block_task",
            prompt="Prefer the bounded block that preserves both shared-anchor cores when two localized repairs compete.",
            workspace_subdir="multi_core_shared_anchor_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nmid2\ntheta\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    single_core_block = {
        "command": "sed -i '2,5c\\\nalpha\\\nmid2\\\ntheta\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4, 5],
        "edit_window_count": 6,
        "covered_window_indices": [0, 1, 2, 3, 4, 5],
        "covered_window_count": 6,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 6,
        "inexact_window_count": 0,
        "current_span_line_count": 4,
        "target_span_line_count": 4,
        "span_line_count": 4,
        "exact_contiguous_region_block": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 2,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 0,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 24,
        },
    }
    multi_core_block = dict(single_core_block)
    multi_core_block.update(
        {
            "command": "sed -i '3,6c\\\nalpha\\\nmid2\\\ntheta\\\nomega' config.txt",
            "shared_anchor_pair_kinds": [
                "same_anchor_insert_delete",
                "adjacent_delete_insert",
            ],
            "shared_anchor_core_count": 2,
            "step": {
                "edit_kind": "block_replace",
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(single_core_block), dict(multi_core_block)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    multi_core_candidate = next(
        candidate for candidate in candidates if candidate.content == multi_core_block["command"]
    )
    single_core_candidate = next(
        candidate for candidate in candidates if candidate.content == single_core_block["command"]
    )
    assert multi_core_candidate.proposal_metadata["shared_anchor_core_count"] == 2
    assert multi_core_candidate.score > single_core_candidate.score


def test_action_generation_candidates_prefer_multi_core_multi_hybrid_shared_anchor_block_over_single_core_variant(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="multi_core_multi_hybrid_shared_anchor_block_task",
            prompt="Prefer the bounded block that preserves both shared-anchor cores when both overlap recovery and hidden-gap proof are present.",
            workspace_subdir="multi_core_multi_hybrid_shared_anchor_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nmid2\ntheta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    single_core_multi_hybrid_block = {
        "command": "sed -i '2,5c\\\nalpha\\\nmid2\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4, 5],
        "edit_window_count": 6,
        "covered_window_indices": [0, 1, 2, 3, 4, 5],
        "covered_window_count": 6,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 6,
        "inexact_window_count": 0,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 0,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_span_line_count": 4,
        "target_span_line_count": 4,
        "span_line_count": 4,
        "exact_hidden_gap_region_block": True,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "hidden_gap_target_from_expected_content": True,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 2,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 24,
        },
    }
    multi_core_multi_hybrid_block = dict(single_core_multi_hybrid_block)
    multi_core_multi_hybrid_block.update(
        {
            "command": "sed -i '3,6c\\\nalpha\\\nmid2\\\ntheta2\\\nomega' config.txt",
            "shared_anchor_pair_kinds": [
                "same_anchor_insert_delete",
                "adjacent_delete_insert",
            ],
            "shared_anchor_core_count": 2,
            "step": {
                "edit_kind": "block_replace",
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [
            dict(single_core_multi_hybrid_block),
            dict(multi_core_multi_hybrid_block),
        ],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    multi_core_candidate = next(
        candidate
        for candidate in candidates
        if candidate.content == multi_core_multi_hybrid_block["command"]
    )
    single_core_candidate = next(
        candidate
        for candidate in candidates
        if candidate.content == single_core_multi_hybrid_block["command"]
    )
    assert multi_core_candidate.proposal_metadata["shared_anchor_core_count"] == 2
    assert multi_core_candidate.proposal_metadata["shared_anchor_hybrid_component_count"] == 2
    assert multi_core_candidate.proposal_metadata["recovered_conflicting_alias"] is True
    assert multi_core_candidate.proposal_metadata["exact_hidden_gap_region_block"] is True
    assert multi_core_candidate.score > single_core_candidate.score


def test_action_generation_candidates_prefer_multi_core_current_proof_multi_hybrid_block_over_single_core_variant(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="multi_core_current_proof_multi_hybrid_block_task",
            prompt="Prefer the bounded block that preserves both shared-anchor cores when overlap recovery and current-proof spans are both present.",
            workspace_subdir="multi_core_current_proof_multi_hybrid_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nmid2\nkeep_b\ntheta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    single_core_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4, 5],
        "edit_window_count": 6,
        "covered_window_indices": [0, 1, 2, 3, 4, 5],
        "covered_window_count": 6,
        "available_window_count": 6,
        "retained_window_count": 6,
        "total_window_count": 6,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 5,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 1,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 2,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 24,
        },
    }
    multi_core_block = dict(single_core_block)
    multi_core_block.update(
        {
            "command": "sed -i '3,7c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
            "shared_anchor_pair_kinds": [
                "same_anchor_insert_delete",
                "adjacent_delete_insert",
            ],
            "shared_anchor_core_count": 2,
            "step": {
                "edit_kind": "block_replace",
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(single_core_block), dict(multi_core_block)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    multi_core_candidate = max(
        (
            candidate
            for candidate in candidates
            if candidate.proposal_source == "structured_edit:block_replace"
            and candidate.proposal_metadata.get("shared_anchor_core_count") == 2
        ),
        key=lambda candidate: candidate.score,
    )
    single_core_candidate = max(
        (
            candidate
            for candidate in candidates
            if candidate.proposal_source == "structured_edit:block_replace"
            and candidate.proposal_metadata.get("shared_anchor_core_count") == 1
        ),
        key=lambda candidate: candidate.score,
    )
    assert multi_core_candidate.proposal_metadata["current_proof_region_block"] is True
    assert multi_core_candidate.proposal_metadata["shared_anchor_hybrid_component_count"] == 2
    assert multi_core_candidate.proposal_metadata["shared_anchor_core_count"] == 2
    assert multi_core_candidate.score > single_core_candidate.score


def test_workspace_preview_window_proposals_prune_weaker_same_span_bounded_variants_upstream(
    monkeypatch,
):
    proof_complete_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    lower_fidelity_block = dict(proof_complete_block)
    lower_fidelity_block.update(
        {
            "command": "sed -i '2,6c\\\nalpha\\\nmid3\\\nkeep_b\\\ntheta3\\\nomega' config.txt",
            "recovered_conflicting_alias": False,
            "overlap_alias_pair_count": 0,
            "overlap_alias_window_count": 0,
            "safe_inexact_window_count": 0,
            "explicit_hidden_gap_current_proof": False,
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
            "shared_anchor_exact_neighbor_count": 0,
            "shared_anchor_core_count": 1,
            "shared_anchor_hybrid_component_count": 1,
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "replacement": {
                    "start_line": 2,
                    "end_line": 6,
                    "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                    "after_lines": ["alpha", "mid3", "keep_b", "theta3", "omega"],
                },
                "current_start_line": 2,
                "current_end_line": 6,
                "target_start_line": 2,
                "target_end_line": 6,
                "line_delta": 0,
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [dict(lower_fidelity_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [dict(proof_complete_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    proof_complete_proposal = next(
        proposal
        for proposal in proposals
        if proposal.get("command") == proof_complete_block["command"]
    )
    assert all(
        proposal.get("command") != lower_fidelity_block["command"]
        for proposal in proposals
    )
    assert proof_complete_proposal["same_span_block_alternative_count"] == 1
    assert proof_complete_proposal["same_span_block_quality_gap"] == 0


def test_workspace_preview_window_proposals_prefer_recovered_alias_same_span_variant_when_quality_ties(
    monkeypatch,
):
    recovered_alias_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 0,
        "overlap_alias_window_count": 0,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 0,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 0,
        "overlap_component_candidate_count": 1,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    generic_alias_block = dict(recovered_alias_block)
    generic_alias_block.update(
        {
            "command": "sed -i '2,6c\\\nalpha\\\nmid2b\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
            "overlap_alias_pair_count": 1,
            "overlap_alias_window_count": 2,
            "recovered_conflicting_alias": False,
            "overlap_component_alias_pair_count": 1,
            "overlap_component_conflicting_alias_pair_count": 1,
            "overlap_component_candidate_count": 2,
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [dict(generic_alias_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [dict(recovered_alias_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    survivor = next(
        proposal
        for proposal in proposals
        if proposal.get("proposal_source") == "structured_edit:block_replace"
    )
    assert survivor["command"] == recovered_alias_block["command"]
    assert survivor["recovered_conflicting_alias"] is True
    assert all(
        proposal.get("command") != generic_alias_block["command"]
        for proposal in proposals
    )


def test_workspace_preview_window_proposals_collapse_equal_rank_same_span_duplicates_to_single_canonical_block(
    monkeypatch,
):
    first_duplicate = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    second_duplicate = dict(first_duplicate)
    second_duplicate.update(
        {
            "command": "sed -i '2,6c\\\nalpha\\\nmid2_alt\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [dict(first_duplicate), dict(second_duplicate)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    block_replace_proposals = [
        proposal
        for proposal in proposals
        if proposal.get("proposal_source") == "structured_edit:block_replace"
    ]
    assert len(block_replace_proposals) == 1
    assert block_replace_proposals[0]["command"] == first_duplicate["command"]


def test_workspace_preview_window_proposals_prune_same_span_multi_edit_reentry_after_composite_assembly(
    monkeypatch,
):
    canonical_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    composite_reentry = {
        "command": (
            "sed -i '2s#beta#alpha#' config.txt && "
            "sed -i '3,6c\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt"
        ),
        "proposal_source": "structured_edit:multi_edit",
        "edit_kind": "multi_edit",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "component_edit_kinds": ["token_replace", "block_replace"],
        "step": {
            "path": "config.txt",
            "edit_kind": "multi_edit",
            "steps": [
                {
                    "path": "config.txt",
                    "edit_kind": "token_replace",
                    "replacements": [
                        {
                            "line_number": 2,
                            "before_fragment": "beta",
                            "after_fragment": "alpha",
                        }
                    ],
                    "current_start_line": 2,
                    "current_end_line": 2,
                    "target_start_line": 2,
                    "target_end_line": 2,
                    "line_delta": 0,
                    "edit_score": 8,
                },
                {
                    "path": "config.txt",
                    "edit_kind": "block_replace",
                    "replacement": {
                        "start_line": 3,
                        "end_line": 6,
                        "before_lines": ["mid", "keep_b", "theta", "omega"],
                        "after_lines": ["mid2", "keep_b", "theta2", "omega"],
                    },
                    "current_start_line": 3,
                    "current_end_line": 6,
                    "target_start_line": 3,
                    "target_end_line": 6,
                    "line_delta": 0,
                    "edit_score": 16,
                },
            ],
            "edit_score": 24,
        },
    }

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [dict(canonical_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: dict(composite_reentry),
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    assert any(
        proposal.get("command") == canonical_block["command"]
        for proposal in proposals
    )
    assert all(
        proposal.get("command") != composite_reentry["command"]
        for proposal in proposals
    )


def test_workspace_preview_window_proposals_prune_same_command_multi_edit_reentry_after_composite_assembly(
    monkeypatch,
):
    canonical_block = {
        "command": "sed -i '2,4c\\\nalpha\\\nbeta2\\\ngamma2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2],
        "edit_window_count": 3,
        "covered_window_indices": [0, 1, 2],
        "covered_window_count": 3,
        "available_window_count": 3,
        "retained_window_count": 3,
        "total_window_count": 3,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 3,
        "inexact_window_count": 0,
        "current_span_line_count": 3,
        "target_span_line_count": 3,
        "span_line_count": 3,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 1,
        "shared_anchor_hybrid_component_count": 1,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 4,
                "before_lines": ["beta", "gamma", "delta"],
                "after_lines": ["alpha", "beta2", "gamma2"],
            },
            "current_start_line": 2,
            "current_end_line": 4,
            "target_start_line": 2,
            "target_end_line": 4,
            "line_delta": 0,
            "edit_score": 18,
        },
    }
    composite_reentry = {
        "command": canonical_block["command"],
        "proposal_source": "structured_edit:multi_edit",
        "edit_kind": "multi_edit",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2],
        "edit_window_count": 3,
        "covered_window_indices": [0, 1, 2],
        "covered_window_count": 3,
        "available_window_count": 3,
        "retained_window_count": 3,
        "total_window_count": 3,
        "partial_window_coverage": False,
        "component_edit_kinds": ["line_replace", "line_replace"],
        "step": {
            "path": "config.txt",
            "edit_kind": "multi_edit",
            "steps": [
                {
                    "path": "config.txt",
                    "edit_kind": "line_replace",
                    "replacements": [
                        {
                            "line_number": 10,
                            "before_line": "x=1",
                            "after_line": "x=2",
                        }
                    ],
                    "current_start_line": 10,
                    "current_end_line": 10,
                    "target_start_line": 10,
                    "target_end_line": 10,
                    "line_delta": 0,
                    "edit_score": 6,
                },
                {
                    "path": "config.txt",
                    "edit_kind": "line_replace",
                    "replacements": [
                        {
                            "line_number": 11,
                            "before_line": "y=1",
                            "after_line": "y=2",
                        }
                    ],
                    "current_start_line": 11,
                    "current_end_line": 11,
                    "target_start_line": 11,
                    "target_end_line": 11,
                    "line_delta": 0,
                    "edit_score": 6,
                },
            ],
            "edit_score": 12,
        },
    }

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [dict(canonical_block)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: dict(composite_reentry),
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    assert len(
        [
            proposal
            for proposal in proposals
            if proposal.get("command") == canonical_block["command"]
        ]
    ) == 1
    assert all(
        str(proposal.get("proposal_source", "")).strip() != "structured_edit:multi_edit"
        for proposal in proposals
    )


def test_workspace_preview_window_proposals_suppress_same_span_hidden_gap_region_duplicates_in_annotation_frontier(
    monkeypatch,
):
    first_duplicate = {
        "command": "sed -i '2,4c\\\nalpha\\\nbridge\\\ngamma' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1],
        "edit_window_count": 2,
        "covered_window_indices": [0, 1],
        "covered_window_count": 2,
        "available_window_count": 2,
        "retained_window_count": 2,
        "total_window_count": 2,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 1,
        "inexact_window_count": 1,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "span_line_count": 3,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 4,
                "before_lines": ["beta", "mid", "gamma"],
                "after_lines": ["alpha", "bridge", "gamma"],
            },
            "current_start_line": 2,
            "current_end_line": 4,
            "target_start_line": 2,
            "target_end_line": 4,
            "line_delta": 0,
            "edit_score": 12,
        },
    }
    second_duplicate = dict(first_duplicate)
    second_duplicate.update(
        {
            "command": "sed -i '2,4c\\\nalpha\\\nbridge_alt\\\ngamma' config.txt",
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "replacement": {
                    "start_line": 2,
                    "end_line": 4,
                    "before_lines": ["beta", "mid", "gamma"],
                    "after_lines": ["alpha", "bridge_alt", "gamma"],
                },
                "current_start_line": 2,
                "current_end_line": 4,
                "target_start_line": 2,
                "target_end_line": 4,
                "line_delta": 0,
                "edit_score": 12,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [dict(first_duplicate), dict(second_duplicate)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    assert len(
        [
            proposal
            for proposal in proposals
            if proposal.get("proposal_source") == "structured_edit:block_replace"
        ]
    ) == 2
    assert all(
        proposal["hidden_gap_region_ambiguity_count"] == 0
        for proposal in proposals
    )
    assert all(
        proposal["hidden_gap_region_frontier_count"] == 1
        for proposal in proposals
    )
    assert all(
        not proposal.get("allow_expected_write_fallback", False)
        for proposal in proposals
    )


def test_workspace_preview_window_proposals_suppress_same_span_hidden_gap_current_duplicates_in_annotation_frontier(
    monkeypatch,
):
    first_duplicate = {
        "command": "sed -i '2,5c\\\nalpha\\\nkeep\\\nbridge\\\ngamma' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1],
        "edit_window_count": 2,
        "covered_window_indices": [0, 1],
        "covered_window_count": 2,
        "available_window_count": 2,
        "retained_window_count": 2,
        "total_window_count": 2,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 2,
        "inexact_window_count": 0,
        "hidden_gap_current_line_count": 2,
        "hidden_gap_target_line_count": 0,
        "current_span_line_count": 4,
        "target_span_line_count": 3,
        "span_line_count": 4,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 5,
                "before_lines": ["beta", "keep", "mid", "gamma"],
                "after_lines": ["alpha", "keep", "bridge", "gamma"],
            },
            "current_start_line": 2,
            "current_end_line": 5,
            "target_start_line": 2,
            "target_end_line": 4,
            "line_delta": -1,
            "edit_score": 14,
        },
    }
    second_duplicate = dict(first_duplicate)
    second_duplicate.update(
        {
            "command": "sed -i '2,5c\\\nalpha\\\nkeep\\\nbridge_alt\\\ngamma' config.txt",
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "replacement": {
                    "start_line": 2,
                    "end_line": 5,
                    "before_lines": ["beta", "keep", "mid", "gamma"],
                    "after_lines": ["alpha", "keep", "bridge_alt", "gamma"],
                },
                "current_start_line": 2,
                "current_end_line": 5,
                "target_start_line": 2,
                "target_end_line": 4,
                "line_delta": -1,
                "edit_score": 14,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_proof_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_overlap_block_replace_proposals",
        lambda **kwargs: [dict(first_duplicate), dict(second_duplicate)],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_exact_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_bridged_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_current_proof_region_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_frontier_block_replace_proposals",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        decoder_module,
        "_compose_workspace_preview_window_proposal",
        lambda **kwargs: None,
    )

    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[],
    )

    assert len(
        [
            proposal
            for proposal in proposals
            if proposal.get("proposal_source") == "structured_edit:block_replace"
        ]
    ) == 2
    assert all(
        proposal["hidden_gap_current_ambiguity_count"] == 0
        for proposal in proposals
    )
    assert all(
        proposal["hidden_gap_current_frontier_count"] == 1
        for proposal in proposals
    )
    assert all(
        proposal["hidden_gap_current_proof_required_count"] == 0
        for proposal in proposals
    )
    assert all(
        not proposal.get("allow_expected_write_fallback", False)
        for proposal in proposals
    )


def test_action_generation_candidates_prefer_same_span_proof_complete_multi_hybrid_block_over_partial_current_proof_variant(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="same_span_proof_complete_multi_hybrid_block_task",
            prompt="Prefer the same-span bounded repair with complete current-proof and full multi-hybrid provenance over a weaker partial-proof variant.",
            workspace_subdir="same_span_proof_complete_multi_hybrid_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nmid2\nkeep_b\ntheta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\nmid\nkeep_b\ntheta\nomega\n",
                        "target_edit_content": "alpha\nmid2\nkeep_b\ntheta2\nomega\n",
                        "line_start": 2,
                        "line_end": 6,
                        "target_line_start": 2,
                        "target_line_end": 6,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    proof_complete_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    partial_current_proof_block = dict(proof_complete_block)
    partial_current_proof_block.update(
        {
            "command": "sed -i '2,6c\\\nalpha\\\nmidX\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
            "partial_window_coverage": True,
            "recovered_conflicting_alias": False,
            "overlap_alias_pair_count": 0,
            "overlap_alias_window_count": 0,
            "safe_inexact_window_count": 0,
            "current_proof_complete": False,
            "current_proof_partial_coverage": True,
            "current_proof_covered_line_count": 1,
            "current_proof_missing_line_count": 2,
            "current_proof_missing_span_count": 1,
            "explicit_hidden_gap_current_proof": False,
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
            "shared_anchor_exact_neighbor_count": 0,
            "shared_anchor_core_count": 1,
            "shared_anchor_hybrid_component_count": 1,
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "replacement": {
                    "start_line": 2,
                    "end_line": 6,
                    "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                    "after_lines": ["alpha", "midX", "keep_b", "theta2", "omega"],
                },
                "current_start_line": 2,
                "current_end_line": 6,
                "target_start_line": 2,
                "target_end_line": 6,
                "line_delta": 0,
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(proof_complete_block), dict(partial_current_proof_block)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    proof_complete_candidate = next(
        candidate for candidate in candidates if candidate.content == proof_complete_block["command"]
    )
    partial_candidate = next(
        candidate for candidate in candidates if candidate.content == partial_current_proof_block["command"]
    )
    assert proof_complete_candidate.proposal_metadata["same_span_block_alternative_count"] == 2
    assert proof_complete_candidate.proposal_metadata["same_span_block_proof_frontier"] is True
    assert partial_candidate.proposal_metadata["same_span_block_quality_gap"] > 0
    assert proof_complete_candidate.score > partial_candidate.score


def test_action_generation_candidates_prefer_same_span_proof_complete_multi_hybrid_block_over_lower_fidelity_bounded_variant(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="same_span_proof_complete_vs_lower_fidelity_block_task",
            prompt="Prefer the same-span bounded repair with fuller proof and provenance over a lower-fidelity bounded competitor.",
            workspace_subdir="same_span_proof_complete_vs_lower_fidelity_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nmid2\nkeep_b\ntheta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\nmid\nkeep_b\ntheta\nomega\n",
                        "target_edit_content": "alpha\nmid2\nkeep_b\ntheta2\nomega\n",
                        "line_start": 2,
                        "line_end": 6,
                        "target_line_start": 2,
                        "target_line_end": 6,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    proof_complete_multi_hybrid_block = {
        "command": "sed -i '2,6c\\\nalpha\\\nmid2\\\nkeep_b\\\ntheta2\\\nomega' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3, 4],
        "edit_window_count": 5,
        "covered_window_indices": [0, 1, 2, 3, 4],
        "covered_window_count": 5,
        "available_window_count": 5,
        "retained_window_count": 5,
        "total_window_count": 5,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 4,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "current_proof_region_block": True,
        "current_proof_span_count": 2,
        "current_proof_complete": True,
        "current_proof_partial_coverage": False,
        "current_proof_covered_line_count": 2,
        "current_proof_missing_line_count": 0,
        "current_proof_missing_span_count": 0,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 5,
        "target_span_line_count": 5,
        "span_line_count": 5,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "path": "config.txt",
            "edit_kind": "block_replace",
            "replacement": {
                "start_line": 2,
                "end_line": 6,
                "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                "after_lines": ["alpha", "mid2", "keep_b", "theta2", "omega"],
            },
            "current_start_line": 2,
            "current_end_line": 6,
            "target_start_line": 2,
            "target_end_line": 6,
            "line_delta": 0,
            "edit_score": 24,
        },
    }
    lower_fidelity_block = dict(proof_complete_multi_hybrid_block)
    lower_fidelity_block.update(
        {
            "command": "sed -i '2,6c\\\nalpha\\\nmid3\\\nkeep_b\\\ntheta3\\\nomega' config.txt",
            "recovered_conflicting_alias": False,
            "overlap_alias_pair_count": 0,
            "overlap_alias_window_count": 0,
            "safe_inexact_window_count": 0,
            "explicit_hidden_gap_current_proof": False,
            "hidden_gap_current_from_line_span_proof": False,
            "hidden_gap_target_from_expected_content": False,
            "shared_anchor_pair_kinds": ["same_anchor_insert_delete"],
            "shared_anchor_exact_neighbor_count": 0,
            "shared_anchor_core_count": 1,
            "shared_anchor_hybrid_component_count": 1,
            "step": {
                "path": "config.txt",
                "edit_kind": "block_replace",
                "replacement": {
                    "start_line": 2,
                    "end_line": 6,
                    "before_lines": ["beta", "mid", "keep_b", "theta", "omega"],
                    "after_lines": ["alpha", "mid3", "keep_b", "theta3", "omega"],
                },
                "current_start_line": 2,
                "current_end_line": 6,
                "target_start_line": 2,
                "target_end_line": 6,
                "line_delta": 0,
                "edit_score": 24,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(proof_complete_multi_hybrid_block), dict(lower_fidelity_block)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    proof_complete_candidate = next(
        candidate
        for candidate in candidates
        if candidate.content == proof_complete_multi_hybrid_block["command"]
    )
    lower_fidelity_candidate = next(
        candidate for candidate in candidates if candidate.content == lower_fidelity_block["command"]
    )
    assert proof_complete_candidate.proposal_metadata["same_span_block_proof_frontier"] is True
    assert lower_fidelity_candidate.proposal_metadata["same_span_block_quality_gap"] > 0
    assert proof_complete_candidate.score > lower_fidelity_candidate.score


def test_action_generation_candidates_prefer_shared_anchor_bridged_multi_hybrid_block_over_multi_edit(
    monkeypatch,
):
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_anchor_bridged_multi_hybrid_block_task",
            prompt="Prefer one bounded block when overlap recovery and bridged current-proof evidence land on the same localized repair.",
            workspace_subdir="shared_anchor_bridged_multi_hybrid_block_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nkeep_b\ngamma2\ndelta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    }
                ]
            }
        },
    }

    hybrid_block = {
        "command": "sed -i '2,5c\\\nalpha\\\nkeep_b\\\ngamma2\\\ndelta2' config.txt",
        "proposal_source": "structured_edit:block_replace",
        "edit_kind": "block_replace",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1, 2, 3],
        "edit_window_count": 4,
        "covered_window_indices": [0, 1, 2, 3],
        "covered_window_count": 4,
        "available_window_count": 4,
        "retained_window_count": 4,
        "total_window_count": 4,
        "partial_window_coverage": False,
        "exact_target_span": True,
        "precision_penalty": 0,
        "exact_window_count": 3,
        "inexact_window_count": 1,
        "overlap_alias_pair_count": 1,
        "overlap_alias_window_count": 2,
        "safe_inexact_window_count": 1,
        "recovered_conflicting_alias": True,
        "overlap_component_alias_pair_count": 1,
        "overlap_component_unresolved_alias_pair_count": 0,
        "overlap_component_conflicting_alias_pair_count": 1,
        "overlap_component_candidate_count": 2,
        "overlap_component_frontier_gap": 0.0,
        "bridged_hidden_gap_region_block": True,
        "hidden_gap_bridge_count": 1,
        "hidden_gap_current_line_count": 1,
        "hidden_gap_target_line_count": 1,
        "explicit_hidden_gap_current_proof": True,
        "hidden_gap_current_from_line_span_proof": True,
        "hidden_gap_target_from_expected_content": True,
        "current_span_line_count": 4,
        "target_span_line_count": 4,
        "span_line_count": 4,
        "shared_anchor_exact_region_block": True,
        "shared_anchor_pair_kinds": ["same_anchor_insert_delete", "adjacent_delete_insert"],
        "shared_anchor_exact_neighbor_count": 1,
        "shared_anchor_core_count": 2,
        "shared_anchor_hybrid_component_count": 2,
        "shared_anchor_mixed_insert_delete": True,
        "step": {
            "edit_kind": "block_replace",
            "edit_score": 18,
        },
    }
    fragmented_multi_edit = dict(hybrid_block)
    fragmented_multi_edit.update(
        {
            "command": (
                "sed -i '2s#beta#alpha#' config.txt && sed -i '3,5c\\\n"
                "keep_b\\\ngamma2\\\ndelta2' config.txt"
            ),
            "proposal_source": "structured_edit:multi_edit",
            "edit_kind": "multi_edit",
            "exact_target_span": False,
            "component_edit_kinds": ["token_replace", "block_replace"],
            "step": {
                "edit_kind": "multi_edit",
                "edit_score": 15,
            },
        }
    )

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(hybrid_block), dict(fragmented_multi_edit)],
    )

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    block_replace = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    multi_edit = next(
        candidate
        for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
    )
    assert block_replace.proposal_metadata["bridged_hidden_gap_region_block"] is True
    assert block_replace.proposal_metadata["recovered_conflicting_alias"] is True
    assert block_replace.proposal_metadata["shared_anchor_core_count"] == 2
    assert block_replace.proposal_metadata["shared_anchor_hybrid_component_count"] == 2
    assert block_replace.score > multi_edit.score


def test_action_generation_candidates_prefer_expected_write_fallback_when_all_inexact_overlap_component_has_no_safe_recovery():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nbeta=new\ngamma=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="all_inexact_conflicting_alias_fallback_task",
            prompt="Prefer bounded rewrite fallback when all overlap anchors are inexact and conflicting.",
            workspace_subdir="all_inexact_conflicting_alias_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\nbeta=old\n",
                        "target_edit_content": "alpha=new\nbeta=left\n",
                        "line_start": 2,
                        "line_end": 3,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=old\ngamma=old\n",
                        "target_edit_content": "beta=right\ngamma=new\n",
                        "line_start": 3,
                        "line_end": 4,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["fallback_from_workspace_preview"] is True
    assert any(candidate.proposal_source == "structured_edit:block_replace" for candidate in candidates)


def test_action_generation_candidates_prefer_expected_write_when_exact_overlap_component_has_no_localized_recovery():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nbeta=new\ngamma=new\ndelta=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="exact_alias_overlap_fallback_task",
            prompt="Prefer bounded rewrite fallback when an exact/inexact alias component has no safe localized recovery.",
            workspace_subdir="exact_alias_overlap_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\nbeta=old\n",
                        "target_edit_content": "alpha=new\nbeta=new\n",
                        "line_start": 2,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "beta=other\ngamma=old\n",
                        "target_edit_content": "beta=stale\ngamma=new\n",
                        "line_start": 3,
                        "line_end": 4,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "delta=old\n",
                        "target_edit_content": "delta=new\n",
                        "line_start": 5,
                        "line_end": 5,
                        "target_line_start": 5,
                        "target_line_end": 5,
                        "line_delta": 0,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    structured = next(
        candidate for candidate in candidates
        if candidate.proposal_source.startswith("structured_edit:")
    )
    assert structured.proposal_metadata["overlap_component_unrecoverable_count"] == 1
    assert structured.proposal_metadata["overlap_component_unrecoverable_window_count"] == 2


def test_action_generation_candidates_prefer_expected_write_when_hidden_gap_requires_broader_region():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "header\nalpha=new\nbeta=new\ngamma=new\nomega\n"
    state = AgentState(
        task=TaskSpec(
            task_id="hidden_gap_broader_region_fallback_task",
            prompt="Prefer bounded rewrite fallback when the correct edit extends beyond every retained-window union.",
            workspace_subdir="hidden_gap_broader_region_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "retained_edit_window_count": 2,
                "total_edit_window_count": 3,
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=old\n",
                        "target_edit_content": "alpha=new\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=old\n",
                        "target_edit_content": "gamma=new\n",
                        "line_start": 4,
                        "line_end": 4,
                        "target_line_start": 4,
                        "target_line_end": 4,
                        "line_delta": 0,
                    },
                ],
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    structured = next(
        candidate for candidate in candidates
        if candidate.proposal_source == "structured_edit:multi_edit"
    )
    assert structured.proposal_metadata["hidden_gap_broader_region_required"] is True
    assert structured.proposal_metadata["hidden_gap_line_count"] == 1


def test_action_generation_candidates_prefer_expected_write_when_hidden_gap_frontier_is_ambiguous(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nbeta=2\ngamma=2\ndelta=2\n"
    ambiguous_preview_proposals = [
        {
            "command": "sed -i '1,4c\\\nalpha=2\\\nbeta=2\\\ngamma=2\\\ndelta=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "allow_expected_write_fallback": True,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "hidden_gap_region_ambiguity_count": 2,
            "hidden_gap_region_ambiguous": True,
            "span_line_count": 4,
            "step": {"edit_kind": "block_replace", "edit_score": 60},
        },
        {
            "command": "sed -i '1,4c\\\nalpha=2\\\nbridge=2\\\ngamma=2\\\ndelta=2' config.txt",
            "proposal_source": "structured_edit:block_replace",
            "edit_kind": "block_replace",
            "edit_source": "workspace_preview_range",
            "window_index": None,
            "window_indices": [0, 1],
            "edit_window_count": 2,
            "covered_window_indices": [0, 1],
            "covered_window_count": 2,
            "available_window_count": 2,
            "retained_window_count": 2,
            "total_window_count": 2,
            "partial_window_coverage": False,
            "allow_expected_write_fallback": True,
            "exact_target_span": True,
            "precision_penalty": 0,
            "exact_window_count": 1,
            "inexact_window_count": 1,
            "hidden_gap_current_line_count": 1,
            "hidden_gap_target_line_count": 1,
            "hidden_gap_region_ambiguity_count": 2,
            "hidden_gap_region_ambiguous": True,
            "span_line_count": 4,
            "step": {"edit_kind": "block_replace", "edit_score": 60},
        },
    ]

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(proposal) for proposal in ambiguous_preview_proposals],
    )

    state = AgentState(
        task=TaskSpec(
            task_id="hidden_gap_frontier_ambiguity_fallback_task",
            prompt="Prefer bounded rewrite fallback when multiple broader hidden-gap regions remain plausible.",
            workspace_subdir="hidden_gap_frontier_ambiguity_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["workspace_preview_hidden_gap_region_ambiguity_count"] == 2
    structured = next(
        candidate for candidate in candidates
        if candidate.proposal_source == "structured_edit:block_replace"
    )
    assert structured.proposal_metadata["hidden_gap_region_ambiguity_count"] == 2
    assert structured.proposal_metadata["hidden_gap_region_ambiguous"] is True


def test_workspace_preview_window_proposals_attach_path_fallback_summary_to_composite_repairs():
    proposals = _workspace_preview_window_proposals(
        path="config.txt",
        preview_windows=[
            {
                "baseline_content": "alpha\n",
                "target_content": "",
                "line_offset": 1,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 0,
            },
            {
                "baseline_content": "beta\n",
                "target_content": "",
                "line_offset": 2,
                "target_line_offset": 1,
                "truncated": True,
                "line_delta": -1,
                "exact_target_span": True,
                "window_index": 1,
            },
        ],
    )

    structured = next(
        proposal
        for proposal in proposals
        if proposal["proposal_source"].startswith("structured_edit:")
        and proposal.get("window_indices") == [0, 1]
    )

    assert "workspace_preview_hidden_gap_region_ambiguity_count" in structured
    assert structured["workspace_preview_hidden_gap_region_ambiguity_count"] == 0
    assert structured["workspace_preview_best_block_proof_quality"] > 0


def test_action_generation_candidates_prefer_expected_write_when_chained_preview_composite_carries_fallback_summary(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    expected_target = "alpha=2\nbeta=2\ngamma=2\ndelta=2\n"
    chained_preview_composite = {
        "command": (
            "sed -i '3s#gamma=1#gamma=2#' config.txt && "
            "sed -i '1,2c\\\nalpha=2\\\nbeta=2' config.txt"
        ),
        "proposal_source": "structured_edit:multi_edit",
        "edit_kind": "multi_edit",
        "edit_source": "workspace_preview_range",
        "window_index": None,
        "window_indices": [0, 1],
        "edit_window_count": 2,
        "covered_window_indices": [0, 1],
        "covered_window_count": 2,
        "available_window_count": 2,
        "retained_window_count": 2,
        "total_window_count": 2,
        "partial_window_coverage": False,
        "allow_expected_write_fallback": True,
        "exact_target_span": False,
        "precision_penalty": 0,
        "exact_window_count": 1,
        "inexact_window_count": 1,
        "workspace_preview_hidden_gap_current_ambiguity_count": 2,
        "workspace_preview_hidden_gap_current_target_equivalent_count": 2,
        "workspace_preview_hidden_gap_current_proof_required_count": 1,
        "workspace_preview_best_block_proof_quality": 4,
        "step": {"edit_kind": "multi_edit", "edit_score": 24},
    }

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(chained_preview_composite)],
    )

    state = AgentState(
        task=TaskSpec(
            task_id="chained_preview_composite_fallback_task",
            prompt="Prefer bounded rewrite fallback when a chained preview composite carries broader-current ambiguity and proof debt.",
            workspace_subdir="chained_preview_composite_fallback_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": expected_target},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    },
                    {
                        "truncated": True,
                        "edit_content": "gamma=1\n",
                        "target_edit_content": "gamma=2\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 3,
                        "target_line_end": 3,
                    },
                ]
            }
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    assert candidates
    assert candidates[0].proposal_source == "expected_file_content"
    assert candidates[0].proposal_metadata["workspace_preview_hidden_gap_current_ambiguity_count"] == 2
    assert (
        candidates[0].proposal_metadata[
            "workspace_preview_hidden_gap_current_target_equivalent_count"
        ]
        == 2
    )
    assert candidates[0].proposal_metadata["workspace_preview_hidden_gap_current_proof_required_count"] == 1
    structured = next(
        candidate for candidate in candidates if candidate.proposal_source == "structured_edit:multi_edit"
    )
    assert structured.proposal_metadata["workspace_preview_hidden_gap_current_ambiguity_count"] == 2
    assert candidates[0].score > structured.score


def test_action_generation_candidates_keep_carried_preview_fallback_summary_isolated_per_file(monkeypatch):
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    carried_preview_proposals = {
        "config.txt": [
            {
                "command": "sed -i '1,3c\\\nalpha=2\\\nbridge=2\\\ngamma=2' config.txt",
                "proposal_source": "structured_edit:multi_edit",
                "edit_kind": "multi_edit",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": False,
                "precision_penalty": 0,
                "exact_window_count": 1,
                "inexact_window_count": 1,
                "workspace_preview_hidden_gap_current_proof_required_count": 1,
                "workspace_preview_best_block_proof_quality": 4,
                "step": {"edit_kind": "multi_edit", "edit_score": 18},
            }
        ],
        "notes.txt": [
            {
                "command": "sed -i '1,4c\\\nintro\\\nbody2\\\nbridge\\\noutro' notes.txt",
                "proposal_source": "structured_edit:multi_edit",
                "edit_kind": "multi_edit",
                "edit_source": "workspace_preview_range",
                "window_index": None,
                "window_indices": [0, 1],
                "edit_window_count": 2,
                "covered_window_indices": [0, 1],
                "covered_window_count": 2,
                "available_window_count": 2,
                "retained_window_count": 2,
                "total_window_count": 2,
                "partial_window_coverage": False,
                "allow_expected_write_fallback": True,
                "exact_target_span": False,
                "precision_penalty": 0,
                "exact_window_count": 1,
                "inexact_window_count": 1,
                "workspace_preview_hidden_gap_region_ambiguity_count": 2,
                "workspace_preview_best_block_proof_quality": 4,
                "step": {"edit_kind": "multi_edit", "edit_score": 18},
            }
        ],
    }

    monkeypatch.setattr(
        decoder_module,
        "_workspace_preview_window_proposals",
        lambda **kwargs: [dict(proposal) for proposal in carried_preview_proposals[kwargs["path"]]],
    )

    state = AgentState(
        task=TaskSpec(
            task_id="multi_file_carried_preview_summary_task",
            prompt="Keep carried workspace-preview fallback summaries isolated per file.",
            workspace_subdir="multi_file_carried_preview_summary_task",
            suggested_commands=[],
            expected_files=["config.txt", "notes.txt"],
            expected_file_contents={
                "config.txt": "alpha=2\nbridge=2\ngamma=2\n",
                "notes.txt": "intro\nbody2\nbridge\noutro\n",
            },
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt", "notes.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt", "notes.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "alpha=1\n",
                        "target_edit_content": "alpha=2\n",
                        "line_start": 1,
                        "line_end": 1,
                        "target_line_start": 1,
                        "target_line_end": 1,
                    }
                ]
            },
            "notes.txt": {
                "edit_windows": [
                    {
                        "truncated": True,
                        "edit_content": "body1\n",
                        "target_edit_content": "body2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 2,
                    }
                ]
            },
        },
    }

    candidates = decode_action_generation_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={"recommended_command_spans": []},
        blocked_commands=[],
        proposal_policy={
            "enabled": True,
            "max_candidates": 10,
            "proposal_score_bias": 0.0,
            "novel_command_bonus": 0.0,
            "verifier_alignment_bonus": 0.0,
            "expected_file_template_bonus": 0.0,
            "cleanup_template_bonus": 0.0,
            "template_preferences": {},
        },
        rollout_policy={},
        command_score_fn=lambda state, command: 0.0,
        normalize_command_fn=lambda command, workspace_subdir: command,
        canonicalize_command_fn=lambda command: command,
    )

    expected_candidates = {
        candidate.proposal_metadata["target_path"]: candidate
        for candidate in candidates
        if candidate.proposal_source == "expected_file_content"
    }

    assert expected_candidates["config.txt"].proposal_metadata["workspace_preview_hidden_gap_current_proof_required_count"] == 1
    assert expected_candidates["config.txt"].proposal_metadata["workspace_preview_hidden_gap_region_ambiguity_count"] == 0
    assert expected_candidates["notes.txt"].proposal_metadata["workspace_preview_hidden_gap_region_ambiguity_count"] == 2
    assert expected_candidates["notes.txt"].proposal_metadata["workspace_preview_hidden_gap_current_proof_required_count"] == 0


def test_llm_policy_workspace_preview_range_can_batch_same_anchor_insert_and_replace(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="same_anchor_insert_replace_task",
            prompt="Batch an insert and replacement anchored to the same current line.",
            workspace_subdir="same_anchor_insert_replace_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nbeta2\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "beta\n",
                "truncated": True,
                "edit_content": "beta\n",
                "target_edit_content": "alpha\nbeta\n",
                "line_start": 2,
                "line_end": 2,
                "target_line_start": 2,
                "target_line_end": 3,
                "line_delta": 1,
                "edit_windows": [
                    {
                        "content": "beta\n",
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                    {
                        "content": "beta\n",
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "beta2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2,2c\\\nalpha\\\nbeta2' config.txt"
    assert decision.proposal_source == "structured_edit:block_replace"
    assert decision.proposal_metadata["edit_kind"] == "block_replace"
    assert decision.proposal_metadata["window_indices"] == [0, 1]
    assert decision.proposal_metadata["exact_contiguous_region_block"] is True
    assert decision.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_llm_policy_workspace_preview_range_can_batch_adjacent_delete_and_shifted_replace(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="adjacent_delete_shifted_replace_task",
            prompt="Delete the obsolete line and update the line that shifts up into its place.",
            workspace_subdir="adjacent_delete_shifted_replace_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nbeta=new\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "alpha\n",
                "truncated": True,
                "edit_content": "alpha\n",
                "target_edit_content": "",
                "line_start": 2,
                "line_end": 2,
                "target_line_start": 2,
                "target_line_end": 1,
                "line_delta": -1,
                "edit_windows": [
                    {
                        "content": "alpha\n",
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "target_edit_content": "",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 1,
                        "line_delta": -1,
                    },
                    {
                        "content": "beta=old\n",
                        "truncated": True,
                        "edit_content": "beta=old\n",
                        "target_edit_content": "beta=new\n",
                        "line_start": 3,
                        "line_end": 3,
                        "target_line_start": 2,
                        "target_line_end": 2,
                        "line_delta": 0,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2,3c\\\nbeta=new' config.txt"
    assert decision.proposal_source == "structured_edit:block_replace"
    assert decision.proposal_metadata["edit_kind"] == "block_replace"
    assert decision.proposal_metadata["window_indices"] == [0, 1]
    assert decision.proposal_metadata["exact_contiguous_region_block"] is True
    assert decision.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_llm_policy_workspace_preview_range_can_batch_three_same_anchor_exact_windows(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["bounded"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "action_generation_policy": {
                    "enabled": True,
                    "max_candidates": 3,
                    "proposal_score_bias": 2.0,
                    "novel_command_bonus": 2.0,
                    "verifier_alignment_bonus": 1.0,
                    "expected_file_template_bonus": 1.0,
                    "cleanup_template_bonus": 0.5,
                    "min_family_support": 1,
                    "template_preferences": {},
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert primary routing takes over")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="three_same_anchor_exact_windows_task",
            prompt="Insert alpha before beta, rename beta to beta2, and append gamma in the same localized region.",
            workspace_subdir="three_same_anchor_exact_windows_task",
            suggested_commands=[],
            expected_files=["config.txt"],
            expected_file_contents={"config.txt": "header\nalpha\nbeta2\ngamma\nomega\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )
    state.world_model_summary = {
        "completion_ratio": 0.0,
        "existing_expected_artifacts": ["config.txt"],
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": ["config.txt"],
        "workspace_file_previews": {
            "config.txt": {
                "content": "beta\n",
                "truncated": True,
                "edit_content": "beta\n",
                "target_edit_content": "alpha\nbeta\n",
                "line_start": 2,
                "line_end": 2,
                "target_line_start": 2,
                "target_line_end": 3,
                "line_delta": 1,
                "edit_windows": [
                    {
                        "content": "beta\n",
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "alpha\nbeta\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 2,
                        "target_line_end": 3,
                        "line_delta": 1,
                    },
                    {
                        "content": "beta\n",
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "beta2\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 3,
                        "line_delta": 0,
                    },
                    {
                        "content": "beta\n",
                        "truncated": True,
                        "edit_content": "beta\n",
                        "target_edit_content": "beta2\ngamma\n",
                        "line_start": 2,
                        "line_end": 2,
                        "target_line_start": 3,
                        "target_line_end": 4,
                        "line_delta": 1,
                    },
                ],
            }
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2,2c\\\nalpha\\\nbeta2\\\ngamma' config.txt"
    assert decision.proposal_source == "structured_edit:block_replace"
    assert decision.proposal_metadata["edit_kind"] == "block_replace"
    assert decision.proposal_metadata["window_indices"] == [0, 1, 2]
    assert decision.proposal_metadata["exact_contiguous_region_block"] is True
    assert decision.proposal_metadata["shared_anchor_exact_region_block"] is True
    assert decision.proposal_metadata["edit_source"] == "workspace_preview_range"


def test_llm_policy_retained_tolbert_decoder_can_stop_when_world_state_is_complete(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 1,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": True,
                    "allow_skill_commands": True,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 3,
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
                "liftoff_gate": {
                    "min_pass_rate_delta": 0.0,
                    "max_step_regression": 0.0,
                    "max_regressed_families": 0,
                    "require_generated_lane_non_regression": True,
                    "require_failure_recovery_non_regression": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class NeverCalledClient:
        def create_decision(self, **kwargs):
            raise AssertionError("LLM should not be called when Tolbert decoder chooses stop")

    policy = LLMDecisionPolicy(
        NeverCalledClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    task = TaskBank().get("release_bundle_task")
    task.metadata["benchmark_family"] = "workflow"
    state = AgentState(task=task)
    state.world_model_summary = {
        "completion_ratio": 1.0,
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latent_state_summary = {"risk_band": "stable", "progress_band": "advancing"}

    decision = policy.decide(state)

    assert decision.action == "respond"
    assert decision.done is True
    assert decision.decision_source == "tolbert_decoder"


def test_llm_policy_retained_tolbert_decoder_blocks_stop_when_learned_world_is_risky(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": [],
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": True,
                    "fallback_to_vllm_on_low_confidence": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 0,
                    "use_value_head": True,
                    "use_transition_head": True,
                    "use_policy_head": True,
                    "use_latent_state": True,
                },
                "decoder_policy": {
                    "allow_retrieval_guidance": False,
                    "allow_skill_commands": False,
                    "allow_task_suggestions": True,
                    "allow_stop_decision": True,
                    "min_stop_completion_ratio": 0.95,
                    "max_task_suggestions": 1,
                },
                "rollout_policy": {
                    "predicted_progress_gain_weight": 3.0,
                    "predicted_conflict_penalty_weight": 4.0,
                    "predicted_preserved_bonus_weight": 1.0,
                    "predicted_workflow_bonus_weight": 1.5,
                    "latent_progress_bonus_weight": 1.0,
                    "latent_risk_penalty_weight": 2.0,
                    "learned_world_progress_bonus_weight": 1.25,
                    "learned_world_recovery_bonus_weight": 1.5,
                    "learned_world_continue_penalty_weight": 1.25,
                    "recover_from_stall_bonus_weight": 1.5,
                    "stop_completion_weight": 8.0,
                    "stop_missing_expected_penalty_weight": 6.0,
                    "stop_forbidden_penalty_weight": 6.0,
                    "stop_preserved_penalty_weight": 4.0,
                    "stop_learned_progress_weight": 1.5,
                    "stop_learned_risk_penalty_weight": 4.0,
                    "long_horizon_stop_risk_gap_penalty_weight": 3.0,
                    "stable_stop_bonus_weight": 1.5,
                },
                "model_surfaces": {
                    "retrieval_surface": True,
                    "policy_head": True,
                    "value_head": True,
                    "transition_head": True,
                    "latent_state": True,
                },
            }
        ),
        encoding="utf-8",
    )

    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(tolbert_model_artifact_path=artifact_path, use_tolbert_model_artifacts=True),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_world_risk_task",
            prompt="Continue driving the workflow until the risky state is resolved.",
            workspace_subdir="long_horizon_world_risk_task",
            suggested_commands=["printf 'continue\\n' > status.txt"],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "continue\n"},
            metadata={"benchmark_family": "workflow", "difficulty": "long_horizon"},
        )
    )
    state.world_model_summary = {
        "horizon": "long_horizon",
        "completion_ratio": 1.0,
        "missing_expected_artifacts": [],
        "present_forbidden_artifacts": [],
        "changed_preserved_artifacts": [],
        "unsatisfied_expected_contents": [],
    }
    state.latent_state_summary = {
        "risk_band": "stable",
        "progress_band": "flat",
        "active_paths": ["status.txt"],
        "learned_world_state": {
            "source": "tolbert_hybrid_runtime",
            "progress_signal": 0.31,
            "risk_signal": 0.87,
            "world_risk_score": 0.82,
            "decoder_world_risk_score": 0.79,
        },
    }

    candidates = decode_bounded_action_candidates(
        state=state,
        world_model=policy.world_model,
        skill_library=policy.skill_library,
        top_skill=None,
        retrieval_guidance={
            "recommended_commands": [],
            "recommended_command_spans": [],
            "avoidance_notes": [],
            "evidence": [],
        },
        blocked_commands=[],
        decoder_policy=policy._tolbert_decoder_policy(),
        rollout_policy=policy._tolbert_rollout_policy(),
        command_score_fn=policy._command_control_score,
        match_span_fn=policy._matching_retrieval_span_id,
        normalize_command_fn=lambda command, workspace: command,
        canonicalize_command_fn=lambda command: str(command).strip(),
    )

    assert not any(candidate.action == "respond" for candidate in candidates)
    assert any(candidate.content == "printf 'continue\\n' > status.txt" for candidate in candidates)


def test_llm_policy_compacts_graph_summary_legacy_keys():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.graph_summary = {
        "document_count": 3,
        "benchmark_families": {"micro": 1},
        "failure_type_counts": {"command_failure": 2},
        "neighbors": ["hello_task_followup"],
        "retrieval_backed_successes": 2,
        "retrieval_influenced_successes": 1,
        "trusted_retrieval_successes": 1,
        "trusted_retrieval_command_counts": {
            "printf 'hello agent kernel\\n' > hello.txt": 2,
        },
    }

    policy.decide(state)

    assert client.last_payload["graph_summary"]["failure_types"] == {"command_failure": 2}
    assert client.last_payload["graph_summary"]["related_tasks"] == ["hello_task_followup"]
    assert client.last_payload["graph_summary"]["retrieval_backed_successes"] == 2
    assert client.last_payload["graph_summary"]["trusted_retrieval_successes"] == 1
    assert client.last_payload["graph_summary"]["trusted_retrieval_command_counts"] == {
        "printf 'hello agent kernel\\n' > hello.txt": 2,
    }


def test_llm_policy_compacts_graph_summary_environment_priors():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.graph_summary = {
        "observed_environment_modes": {
            "network_access_mode": {"allowlist_only": 4, "blocked": 1},
            "git_write_mode": {"operator_gated": 3},
            "workspace_write_scope": {"task_only": 5},
        },
        "environment_alignment_failures": {
            "network_access_aligned": 3,
            "git_write_aligned": 1,
        },
    }

    policy.decide(state)

    assert client.last_payload["graph_summary"]["observed_environment_modes"] == {
        "network_access_mode": "allowlist_only",
        "git_write_mode": "operator_gated",
        "workspace_write_scope": "task_only",
    }
    assert client.last_payload["graph_summary"]["environment_alignment_failures"] == {
        "network_access_aligned": 3,
        "git_write_aligned": 1,
    }


def test_llm_policy_compacts_graph_summary_trusted_retrieval_procedures():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.graph_summary = {
        "trusted_retrieval_procedures": [
            {
                "commands": [
                    "printf 'status ready\\n' > reports/status.txt",
                    "pytest -q tests/test_status.py",
                ],
                "count": 2,
            }
        ]
    }

    policy.decide(state)

    assert client.last_payload["graph_summary"]["trusted_retrieval_procedures"] == [
        {
            "commands": [
                "printf 'status ready\\n' > reports/status.txt",
                "pytest -q tests/test_status.py",
            ],
            "count": 2,
        }
    ]


def test_llm_policy_uses_trusted_retrieval_carryover_before_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("trusted retrieval carryover direct path should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_carryover_task",
            prompt="Recover the long-horizon workflow state.",
            workspace_subdir="trusted_retrieval_carryover_task",
            suggested_commands=[],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            forbidden_files=["temp.txt"],
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "planner"
    state.history = [
        StepRecord(
            index=1,
            thought="prior write did not remove temp",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'done\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["temp.txt still present"]},
        )
    ]
    state.active_subgoal = "remove forbidden artifact temp.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "temp.txt",
            "signals": ["state_regression", "no_state_progress"],
            "summary": "temp.txt is still present",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "completion_ratio": 0.75,
    }
    state.graph_summary = {
        "trusted_retrieval_successes": 2,
        "trusted_retrieval_command_counts": {
            "rm -f temp.txt": 2,
            "printf 'done\\n' > status.txt": 1,
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "rm -f temp.txt"
    assert decision.decision_source == "trusted_retrieval_carryover_direct"
    assert decision.retrieval_influenced is True
    assert decision.selected_retrieval_span_id.startswith("graph:trusted_retrieval:")
    assert context_provider.compile_calls == 0


def test_llm_policy_synthesizes_trusted_retrieval_write_for_unsatisfied_artifact_before_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("trusted retrieval write carryover should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_materialize_task",
            prompt="Recover the missing release status artifact.",
            workspace_subdir="trusted_retrieval_materialize_task",
            suggested_commands=[],
            expected_files=["reports/status.txt"],
            expected_file_contents={"reports/status.txt": "status ready\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "repository"},
        )
    )
    state.current_role = "planner"
    state.history = [
        StepRecord(
            index=1,
            thought="prior verifier run showed the report is still missing",
            action="code_execute",
            content="true",
            selected_skill_id=None,
            command_result={"command": "true", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["reports/status.txt missing"]},
        )
    ]
    state.active_subgoal = "materialize expected artifact reports/status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "reports/status.txt",
            "signals": ["missing_expected_file", "no_state_progress"],
            "summary": "reports/status.txt is still missing",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["reports/status.txt"],
        "missing_expected_artifacts": ["reports/status.txt"],
        "unsatisfied_expected_contents": ["reports/status.txt"],
        "completion_ratio": 0.6,
    }
    state.graph_summary = {
        "trusted_retrieval_successes": 2,
        "trusted_retrieval_command_counts": {
            "mkdir -p app && printf 'release ready\\n' > app/release.txt": 2,
        },
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "trusted_retrieval_carryover_direct"
    assert decision.retrieval_influenced is True
    assert decision.selected_retrieval_span_id.startswith("graph:trusted_retrieval:materialize:")
    assert decision.content == policy._trusted_retrieval_materialize_write_command(
        "reports/status.txt",
        "status ready\n",
    )
    assert context_provider.compile_calls == 0


def test_llm_policy_does_not_synthesize_trusted_retrieval_rewrite_when_preview_exists():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_preview_guard_task",
            prompt="Repair the release status artifact without clobbering preserved lines.",
            workspace_subdir="trusted_retrieval_preview_guard_task",
            suggested_commands=[],
            expected_files=["reports/status.txt"],
            expected_file_contents={"reports/status.txt": "HEADER=stable\nstatus ready\nFOOTER=keep\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "repository"},
        )
    )
    state.current_role = "planner"
    state.history = [
        StepRecord(
            index=1,
            thought="prior verifier run showed the report content is still wrong",
            action="code_execute",
            content="true",
            selected_skill_id=None,
            command_result={"command": "true", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["reports/status.txt content mismatch"]},
        )
    ]
    state.active_subgoal = "materialize expected artifact reports/status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "reports/status.txt",
            "signals": ["unexpected_file_content", "no_state_progress"],
            "summary": "reports/status.txt still needs an in-place repair",
            "source_role": "critic",
        }
    }
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["reports/status.txt"],
        "unsatisfied_expected_contents": ["reports/status.txt"],
        "workspace_file_previews": {
            "reports/status.txt": {
                "content": "HEADER=stable\nstatus stale\nFOOTER=keep\n",
                "truncated": False,
            }
        },
    }
    state.graph_summary = {
        "trusted_retrieval_command_counts": {
            "mkdir -p app && printf 'release ready\\n' > app/release.txt": 2,
        },
    }

    candidates = policy._trusted_retrieval_carryover_candidates(
        state,
        blocked_commands=set(),
    )

    assert candidates == []


def test_llm_policy_continues_trusted_retrieval_procedure_before_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("trusted retrieval procedure carryover should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="trusted_retrieval_sequence_task",
            prompt="Finish the long-horizon status recovery.",
            workspace_subdir="trusted_retrieval_sequence_task",
            suggested_commands=[],
            expected_files=["reports/status.txt"],
            expected_file_contents={"reports/status.txt": "status ready\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "repository"},
        )
    )
    state.current_role = "planner"
    state.history = [
        StepRecord(
            index=1,
            thought="replayed the trusted repair write",
            action="code_execute",
            content="printf 'status ready\\n' > reports/status.txt",
            selected_skill_id=None,
            command_result={
                "command": "printf 'status ready\\n' > reports/status.txt",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["tests not run yet"]},
        )
    ]
    state.active_subgoal = "materialize expected artifact reports/status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "reports/status.txt",
            "signals": ["missing_expected_file", "no_state_progress"],
            "summary": "reports/status.txt needs write-plus-verify completion",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["reports/status.txt"],
        "missing_expected_artifacts": ["reports/status.txt"],
        "completion_ratio": 0.65,
    }
    state.graph_summary = {
        "trusted_retrieval_procedures": [
            {
                "commands": [
                    "printf 'status ready\\n' > reports/status.txt",
                    "pytest -q tests/test_status.py",
                ],
                "count": 2,
            }
        ]
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "pytest -q tests/test_status.py"
    assert decision.decision_source == "trusted_retrieval_carryover_direct"
    assert decision.selected_retrieval_span_id.startswith("graph:trusted_retrieval:procedure:")
    assert context_provider.compile_calls == 0


def test_llm_policy_world_model_blocks_bad_skill_short_circuit():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:dangerous:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'bad\\n' > forbidden.txt"]},
                "known_failure_types": [],
                "quality": 0.95,
            }
        ]
    )
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, skill_library=skill_library)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.world_model_summary = {
        "expected_artifacts": ["hello.txt"],
        "forbidden_artifacts": ["forbidden.txt"],
        "preserved_artifacts": ["hello.txt"],
        "horizon": "bounded",
    }

    decision = policy.decide(state)

    assert decision.selected_skill_id is None
    assert decision.content == "printf 'hello agent kernel\\n' > hello.txt"


def test_llm_policy_planner_role_prefers_subgoal_aligned_direct_command():
    class PlannerContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.95,
                    "trust_retrieval": True,
                    "retrieval_guidance": {
                        "recommended_commands": [
                            "printf 'data\\n' > scratch.txt",
                            "printf 'hello agent kernel\\n' > hello.txt",
                        ],
                        "recommended_command_spans": [
                            {"span_id": "span:scratch", "command": "printf 'data\\n' > scratch.txt"},
                            {"span_id": "span:hello", "command": "printf 'hello agent kernel\\n' > hello.txt"},
                        ],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=PlannerContextProvider())
    state = AgentState(task=TaskBank().get("hello_task"))
    state.current_role = "planner"
    state.active_subgoal = "materialize expected artifact hello.txt"
    state.world_model_summary = {
        "expected_artifacts": ["hello.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": ["hello.txt"],
        "horizon": "bounded",
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "printf 'hello agent kernel\\n' > hello.txt"
    assert decision.selected_retrieval_span_id == "span:hello"


def test_llm_policy_uses_plan_progress_direct_command_for_mid_horizon_recovery():
    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="long_horizon_progress_task",
            prompt="Create status.txt and remove temp.txt.",
            workspace_subdir="long_horizon_progress_task",
            suggested_commands=[
                "printf 'done\\n' > status.txt",
                "rm -f temp.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            forbidden_files=["temp.txt"],
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="created status",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'done\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["temp.txt still present"]},
        )
    ]
    state.active_subgoal = "remove forbidden artifact temp.txt"
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
        "preserved_artifacts": [],
        "existing_expected_artifacts": ["status.txt"],
        "satisfied_expected_contents": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "completion_ratio": 0.75,
    }
    state.universe_summary = {
        "stability": "stable",
        "governance_mode": "bounded_autonomous",
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "rm -f temp.txt"
    assert decision.decision_source == "plan_progress_direct"


def test_llm_policy_critic_uses_diagnosed_plan_progress_direct_command():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="critic_repair_task",
            prompt="Recover the failing workspace state.",
            workspace_subdir="critic_repair_task",
            suggested_commands=[
                "printf 'done\\n' > status.txt",
                "rm -f temp.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            forbidden_files=["temp.txt"],
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "critic"
    state.history = [
        StepRecord(
            index=1,
            thought="tried the status write first",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={
                "command": "printf 'done\\n' > status.txt",
                "exit_code": 1,
                "stdout": "",
                "stderr": "status write failed",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["temp.txt still present"]},
        )
    ]
    state.active_subgoal = "satisfy verifier contract"
    state.subgoal_diagnoses = {
        "satisfy verifier contract": {
            "path": "temp.txt",
            "signals": ["state_regression", "command_failure"],
            "summary": "temp.txt is still present; the prior status write did not repair the state",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
        "preserved_artifacts": [],
        "existing_expected_artifacts": ["status.txt"],
        "satisfied_expected_contents": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "completion_ratio": 0.75,
    }
    state.universe_summary = {
        "stability": "stable",
        "governance_mode": "bounded_autonomous",
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "rm -f temp.txt"
    assert decision.decision_source == "plan_progress_direct"


def test_llm_policy_uses_synthetic_edit_plan_direct_command_before_llm():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_edit_worker_task",
            prompt="Update src/service_status.txt to ready.",
            workspace_subdir="synthetic_edit_worker_task",
            suggested_commands=[],
            expected_files=["src/service_status.txt"],
            expected_file_contents={"src/service_status.txt": "HEADER=stable\nrelease-ready active\nFOOTER=keep\n"},
            metadata={
                "benchmark_family": "repo_sandbox",
                "synthetic_worker": True,
                "synthetic_edit_plan": [
                    {
                        "path": "src/service_status.txt",
                        "edit_kind": "line_replace",
                        "replacements": [
                            {
                                "line_number": 2,
                                "before_line": "SERVICE_STATE=broken",
                                "after_line": "release-ready active",
                            }
                        ],
                        "edit_score": 12,
                    }
                ],
            },
        )
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2s#^SERVICE_STATE=broken$#release-ready active#' src/service_status.txt"
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert decision.proposal_source == "structured_edit:line_replace"
    assert decision.proposal_metadata["edit_kind"] == "line_replace"
    assert decision.proposal_metadata["edit_source"] == "synthetic_edit_plan"
    assert client.last_payload is None


def test_llm_policy_synthetic_edit_plan_skips_context_compile_when_first_step_is_structurally_safe():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("synthetic edit-plan direct path should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)

    decision = policy.decide(
        AgentState(
            task=TaskSpec(
                task_id="long_horizon_adjacent",
                prompt="Advance the staged project handoff.",
                workspace_subdir="long_horizon_adjacent",
                setup_commands=[
                    "mkdir -p project",
                    "printf 'source retained\n' > project/source.txt",
                    "printf 'handoff pending\n' > project/summary.txt",
                ],
                expected_files=["project/source.txt", "project/summary.txt"],
                expected_file_contents={
                    "project/source.txt": "source retained\n",
                    "project/summary.txt": "deployment_manifest_task project handoff ready\n",
                },
                metadata={
                    "curriculum_kind": "adjacent_success",
                    "benchmark_family": "project",
                    "difficulty": "long_horizon",
                    "synthetic_edit_plan": [
                        {
                            "path": "project/summary.txt",
                            "edit_kind": "line_replace",
                            "replacements": [
                                {
                                    "line_number": 1,
                                    "before_line": "handoff pending",
                                    "after_line": "deployment_manifest_task project handoff ready",
                                }
                            ],
                            "edit_score": 12,
                        }
                    ],
                },
            )
        )
    )

    assert decision.action == "code_execute"
    assert decision.content == r"sed -i '1s#^handoff\ pending$#deployment_manifest_task project handoff ready#' project/summary.txt"
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_advances_to_next_synthetic_edit_plan_step_after_history():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_edit_progress_task",
            prompt="Update note.txt in two steps.",
            workspace_subdir="synthetic_edit_progress_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "alpha\nbeta\nomega\n"},
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [
                    {
                        "path": "note.txt",
                        "edit_kind": "line_replace",
                        "replacements": [
                            {
                                "line_number": 1,
                                "before_line": "draft",
                                "after_line": "alpha",
                            }
                        ],
                        "edit_score": 7,
                    },
                    {
                        "path": "note.txt",
                        "edit_kind": "line_insert",
                        "insertion": {
                            "line_number": 2,
                            "after_line": "alpha",
                            "inserted_lines": ["beta"],
                        },
                        "edit_score": 6,
                    },
                ],
            },
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="apply first edit",
            action="code_execute",
            content="sed -i '1s#^draft$#alpha#' note.txt",
            selected_skill_id=None,
            command_result={"command": "sed -i '1s#^draft$#alpha#' note.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["beta line still missing"]},
        )
    ]

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2i\\\nbeta' note.txt"
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert decision.proposal_source == "structured_edit:line_insert"
    assert decision.proposal_metadata["edit_kind"] == "line_insert"
    assert client.last_payload is None


def test_llm_policy_synthetic_edit_plan_continuation_skips_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("synthetic edit-plan continuation should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_edit_progress_skip_context_task",
            prompt="Update note.txt in two steps.",
            workspace_subdir="synthetic_edit_progress_skip_context_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "alpha\nbeta\nomega\n"},
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [
                    {
                        "path": "note.txt",
                        "edit_kind": "line_replace",
                        "replacements": [
                            {
                                "line_number": 1,
                                "before_line": "draft",
                                "after_line": "alpha",
                            }
                        ],
                        "edit_score": 7,
                    },
                    {
                        "path": "note.txt",
                        "edit_kind": "line_insert",
                        "insertion": {
                            "line_number": 2,
                            "after_lines": ["beta"],
                        },
                        "edit_score": 6,
                    },
                ],
            },
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="apply first edit",
            action="code_execute",
            content="sed -i '1s#^draft$#alpha#' note.txt",
            selected_skill_id=None,
            command_result={"command": "sed -i '1s#^draft$#alpha#' note.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["beta line still missing"]},
        )
    ]

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content.splitlines() == ["sed -i '2i\\", "beta' note.txt"]
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_critic_diagnosed_repair_skips_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("critic diagnosed repair should skip context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="critic_repair_skip_context_task",
            prompt="Recover the failing workspace state.",
            workspace_subdir="critic_repair_skip_context_task",
            suggested_commands=[
                "printf 'done\\n' > status.txt",
                "rm -f temp.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            forbidden_files=["temp.txt"],
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "critic"
    state.history = [
        StepRecord(
            index=1,
            thought="tried the status write first",
            action="code_execute",
            content="printf 'done\\n' > status.txt",
            selected_skill_id=None,
            command_result={
                "command": "printf 'done\\n' > status.txt",
                "exit_code": 1,
                "stdout": "",
                "stderr": "status write failed",
                "timed_out": False,
            },
            verification={"passed": False, "reasons": ["temp.txt still present"]},
        )
    ]
    state.active_subgoal = "satisfy verifier contract"
    state.subgoal_diagnoses = {
        "satisfy verifier contract": {
            "path": "temp.txt",
            "signals": ["state_regression", "command_failure"],
            "summary": "temp.txt is still present; the prior status write did not repair the state",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 1
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": ["temp.txt"],
        "preserved_artifacts": [],
        "existing_expected_artifacts": ["status.txt"],
        "satisfied_expected_contents": ["status.txt"],
        "present_forbidden_artifacts": ["temp.txt"],
        "completion_ratio": 0.75,
    }
    state.universe_summary = {
        "stability": "stable",
        "governance_mode": "bounded_autonomous",
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "rm -f temp.txt"
    assert decision.decision_source == "plan_progress_direct"
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_synthetic_edit_plan_continuation_outranks_tolbert_primary():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_edit_progress_tolbert_task",
            prompt="Update note.txt in two steps.",
            workspace_subdir="synthetic_edit_progress_tolbert_task",
            suggested_commands=[],
            expected_files=["note.txt"],
            expected_file_contents={"note.txt": "alpha\nbeta\nomega\n"},
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [
                    {
                        "path": "note.txt",
                        "edit_kind": "line_replace",
                        "replacements": [
                            {
                                "line_number": 1,
                                "before_line": "draft",
                                "after_line": "alpha",
                            }
                        ],
                        "edit_score": 7,
                    },
                    {
                        "path": "note.txt",
                        "edit_kind": "line_insert",
                        "insertion": {
                            "line_number": 2,
                            "after_lines": ["beta"],
                        },
                        "edit_score": 6,
                    },
                ],
            },
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="apply first edit",
            action="code_execute",
            content="sed -i '1s#^draft$#alpha#' note.txt",
            selected_skill_id=None,
            command_result={"command": "sed -i '1s#^draft$#alpha#' note.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["beta line still missing"]},
        )
    ]

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '2i\\\nbeta' note.txt"
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert client.last_payload is None


def test_llm_policy_prefers_localized_syntax_safe_edit_over_import_drift_step():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_edit_syntax_priority_task",
            prompt="Apply the best first edit.",
            workspace_subdir="synthetic_edit_syntax_priority_task",
            suggested_commands=[],
            expected_files=["service.py"],
            expected_file_contents={
                "service.py": "import os\n\ndef apply_status(value):\n    return normalize(value)\n"
            },
            metadata={
                "benchmark_family": "repository",
                "synthetic_edit_plan": [
                    {
                        "path": "service.py",
                        "edit_kind": "rewrite",
                        "baseline_content": "import os\n\ndef apply_status(value):\n    return value.strip()\n",
                        "target_content": "import os\nimport json\n\ndef apply_status(value):\n    return json.dumps(value.strip())\n",
                        "edit_score": 11,
                    },
                    {
                        "path": "service.py",
                        "edit_kind": "line_replace",
                        "baseline_content": "import os\n\ndef apply_status(value):\n    return value.strip()\n",
                        "target_content": "import os\n\ndef apply_status(value):\n    return normalize(value)\n",
                        "replacements": [
                            {
                                "line_number": 4,
                                "before_line": "    return value.strip()",
                                "after_line": "    return normalize(value)",
                            }
                        ],
                        "edit_score": 9,
                    },
                ],
            },
        )
    )
    state.history = [
        StepRecord(
            index=1,
            thought="setup already advanced",
            action="code_execute",
            content="true",
            selected_skill_id=None,
            command_result={"command": "true", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["service.py still needs update"]},
        )
    ]

    decision = policy.decide(state)

    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert decision.proposal_metadata["edit_kind"] == "line_replace"
    assert decision.proposal_metadata["import_change_risk"] is False
    assert decision.proposal_metadata["target_symbol_fqn"].endswith("apply_status")
    assert "normalize" in decision.proposal_metadata["call_targets"]
    assert client.last_payload is None


def test_llm_policy_uses_synthetic_worker_changed_path_for_first_step_coverage():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="synthetic_worker_changed_path_task",
            prompt="Update only the worker-owned file in the shared repo.",
            workspace_subdir="synthetic_worker_changed_path_task",
            suggested_commands=[],
            expected_files=["docs/status.md", "src/api_status.txt", "tests/test_api.sh"],
            expected_file_contents={"src/api_status.txt": "API_STATUS=ready\n"},
            metadata={
                "benchmark_family": "project",
                "synthetic_worker": True,
                "workflow_guard": {
                    "shared_repo_id": "repo_sandbox_parallel_merge",
                    "worker_branch": "worker/api-status",
                    "claimed_paths": ["src/api_status.txt"],
                },
                "semantic_verifier": {
                    "expected_changed_paths": ["src/api_status.txt"],
                    "preserved_paths": ["docs/status.md", "tests/test_api.sh"],
                },
                "synthetic_edit_plan": [
                    {
                        "path": "src/api_status.txt",
                        "edit_kind": "line_replace",
                        "replacements": [
                            {
                                "line_number": 1,
                                "before_line": "API_STATUS=pending",
                                "after_line": "API_STATUS=ready",
                            }
                        ],
                        "edit_score": 17,
                    }
                ],
            },
        )
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "sed -i '1s#^API_STATUS=pending$#API_STATUS=ready#' src/api_status.txt"
    assert decision.decision_source == "synthetic_edit_plan_direct"
    assert client.last_payload is None


def test_llm_policy_uses_role_specific_prompts_and_budgeted_context():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(task=TaskBank().get("hello_task"))
    state.current_role = "critic"
    state.active_subgoal = "avoid repeated failure on hello.txt"
    state.graph_summary = {
        "document_count": 4,
        "benchmark_families": {"micro": 2, "workflow": 1},
        "failure_types": {"command_failure": 3},
        "related_tasks": ["hello_task_followup"],
    }
    state.world_model_summary = {
        "benchmark_family": "micro",
        "horizon": "bounded",
        "expected_artifacts": ["hello.txt"],
        "forbidden_artifacts": ["forbidden.txt"],
        "preserved_artifacts": ["hello.txt"],
    }
    state.plan = ["materialize expected artifact hello.txt", "avoid forbidden artifact forbidden.txt"]

    policy.decide(state)

    assert "Active role: critic" in client.last_system_prompt
    assert "Active role: critic" in client.last_decision_prompt
    assert client.last_payload["acting_role"] == "critic"
    assert client.last_payload["state_context_chunks"]
    assert len(client.last_payload["state_context_chunks"]) <= 8
    assert "graph_summary" in client.last_payload
    assert "world_model_summary" in client.last_payload


def test_llm_policy_includes_workspace_file_previews_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(task=TaskBank().get("rewrite_task"))
    state.world_model_summary = {
        "benchmark_family": "bounded",
        "horizon": "bounded",
        "expected_artifacts": ["note.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "todo\n",
                "truncated": False,
            }
        },
    }

    policy.decide(state)

    assert client.last_payload["world_model_summary"]["workspace_file_previews"]["note.txt"]["content"] == "todo\n"
    assert any(
        chunk["source"] == "world_preview" and chunk["key"] == "note.txt"
        for chunk in client.last_payload["state_context_chunks"]
    )


def test_llm_policy_includes_multi_window_workspace_file_previews_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(task=TaskBank().get("rewrite_task"))
    state.world_model_summary = {
        "benchmark_family": "bounded",
        "horizon": "bounded",
        "expected_artifacts": ["note.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "unsatisfied_expected_contents": ["note.txt"],
        "workspace_file_previews": {
            "note.txt": {
                "content": "alpha\n",
                "truncated": True,
                "edit_content": "alpha\n",
                "line_start": 2,
                "line_end": 2,
                "edit_windows": [
                    {
                        "content": "alpha\n",
                        "truncated": True,
                        "edit_content": "alpha\n",
                        "line_start": 2,
                        "line_end": 2,
                    },
                    {
                        "content": "omega\n",
                        "truncated": True,
                        "edit_content": "omega\n",
                        "line_start": 8,
                        "line_end": 8,
                    },
                ],
            }
        },
    }

    policy.decide(state)

    preview = client.last_payload["world_model_summary"]["workspace_file_previews"]["note.txt"]
    assert len(preview["edit_windows"]) == 2
    assert preview["edit_windows"][0]["line_start"] == 2
    assert preview["edit_windows"][1]["line_start"] == 8
    assert any(
        chunk["source"] == "world_preview" and chunk["key"] == "note.txt#1"
        for chunk in client.last_payload["state_context_chunks"]
    )
    assert any(
        chunk["source"] == "world_preview" and chunk["key"] == "note.txt#2"
        for chunk in client.last_payload["state_context_chunks"]
    )


def test_llm_policy_includes_active_prompt_adjustments(tmp_path):
    config = KernelConfig(
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.prompt_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.prompt_proposals_path.write_text(
        '{"artifact_kind":"prompt_proposal_set","lifecycle_state":"retained","proposals":[{"area":"decision","priority":5,"reason":"test","suggestion":"Widen search before committing."}]}',
        encoding="utf-8",
    )
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=config)

    policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert client.last_payload["prompt_adjustments"][0]["suggestion"] == "Widen search before committing."


def test_llm_policy_skips_conflicting_rejected_prompt_adjustments(tmp_path):
    config = KernelConfig(
        prompt_proposals_path=tmp_path / "prompts" / "prompt_proposals.json",
    )
    config.prompt_proposals_path.parent.mkdir(parents=True, exist_ok=True)
    config.prompt_proposals_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "reject"},
                "proposals": [
                    {
                        "area": "decision",
                        "priority": 5,
                        "reason": "test",
                        "suggestion": "Widen search before committing.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, config=config)

    policy.decide(AgentState(task=TaskBank().get("hello_task")))

    assert client.last_payload["prompt_adjustments"] == []


def test_skill_library_skips_rejected_managed_artifact(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "rejected",
                "retention_decision": {"state": "reject"},
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                        "quality": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    library = SkillLibrary.from_path(skills_path)

    assert library.skills == []


def test_skill_library_skips_conflicting_rejected_retained_artifact(tmp_path):
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "artifact_kind": "skill_set",
                "lifecycle_state": "retained",
                "retention_decision": {"state": "reject"},
                "skills": [
                    {
                        "skill_id": "skill:hello_task:primary",
                        "source_task_id": "hello_task",
                        "procedure": {"commands": ["printf 'hello\\n' > hello.txt"]},
                        "quality": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    library = SkillLibrary.from_path(skills_path)

    assert library.skills == []


def test_llm_policy_critic_branch_avoids_deterministic_repeat():
    policy = LLMDecisionPolicy(MockLLMClient(), context_provider=FakeContextProvider())
    state = AgentState(task=TaskBank().get("hello_task"))
    state.current_role = "critic"
    state.consecutive_failures = 1
    state.history.append(
        StepRecord(
            index=1,
            thought="failed prior command",
            action="code_execute",
            content="printf 'hello agent kernel\\n' > hello.txt",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["failed"]},
        )
    )

    decision = policy.decide(state)

    assert decision.action == "respond"
    assert decision.done is True


def test_llm_policy_plan_progress_recovery_skips_failed_command_for_planner():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_skip_repeat",
            prompt="repair the expected status artifact",
            workspace_subdir="planner_recovery_skip_repeat",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "mkdir -p reports && printf 'done\\n' > status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
        )
    )
    state.current_role = "planner"
    state.active_subgoal = "materialize expected artifact status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
        }
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 1
    state.last_action_signature = "code_execute:printf 'old\\n' > status.txt"
    state.repeated_action_count = 1
    state.history.append(
        StepRecord(
            index=1,
            thought="prior failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        )
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "plan_progress_direct"
    assert decision.content == "mkdir -p reports && printf 'done\\n' > status.txt"


def test_llm_policy_plan_progress_recovery_skips_failed_command_for_critic():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="critic_recovery_skip_repeat",
            prompt="repair the expected status artifact",
            workspace_subdir="critic_recovery_skip_repeat",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "cp template/status.txt status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
        )
    )
    state.current_role = "critic"
    state.active_subgoal = "materialize expected artifact status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
        }
    }
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.last_action_signature = "code_execute:printf 'old\\n' > status.txt"
    state.repeated_action_count = 2
    state.history.append(
        StepRecord(
            index=1,
            thought="prior failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        )
    )

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.decision_source == "plan_progress_direct"
    assert decision.content == "cp template/status.txt status.txt"


def test_llm_policy_critic_recovery_exhaustion_stops_before_context_compile():
    class UnexpectedContextProvider:
        def __init__(self) -> None:
            self.compile_calls = 0

        def compile(self, state):
            del state
            self.compile_calls += 1
            raise AssertionError("critic recovery exhaustion should stop before context compilation")

    client = CapturingClient()
    context_provider = UnexpectedContextProvider()
    policy = LLMDecisionPolicy(client, context_provider=context_provider)
    state = AgentState(
        task=TaskSpec(
            task_id="critic_recovery_exhaustion_task",
            prompt="repair the expected status artifact",
            workspace_subdir="critic_recovery_exhaustion_task",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "cp template/status.txt status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "critic"
    state.active_subgoal = "materialize expected artifact status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.last_action_signature = "code_execute:cp template/status.txt status.txt"
    state.history = [
        StepRecord(
            index=1,
            thought="prior failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        ),
        StepRecord(
            index=2,
            thought="prior failed copy",
            action="code_execute",
            content="cp template/status.txt status.txt",
            selected_skill_id=None,
            command_result={"command": "cp template/status.txt status.txt", "exit_code": 1, "stdout": "", "stderr": "missing template", "timed_out": False},
            verification={"passed": False, "reasons": ["copy failed"]},
        ),
    ]

    decision = policy.decide(state)

    assert decision.action == "respond"
    assert decision.done is True
    assert decision.content == "No safe deterministic recovery command remains."
    assert context_provider.compile_calls == 0
    assert client.last_payload is None


def test_llm_policy_planner_recovery_rewrites_after_critic_proves_exhaustion():
    class PlannerRewriteClient(CapturingClient):
        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            self.last_system_prompt = system_prompt
            self.last_decision_prompt = decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "rewrite the recovery contract around the verifier hotspot",
                "action": "code_execute",
                "content": "python tools/rewrite_recovery_contract.py --target status.txt",
                "done": False,
            }

    client = PlannerRewriteClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_rewrite_task",
            prompt="repair the expected status artifact",
            workspace_subdir="planner_recovery_rewrite_task",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "cp template/status.txt status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "planner"
    state.active_subgoal = "materialize expected artifact status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
            "summary": "status.txt remains incorrect after the bounded repair commands were attempted",
            "source_role": "critic",
        }
    }
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.last_action_signature = "code_execute:cp template/status.txt status.txt"
    state.history = [
        StepRecord(
            index=1,
            thought="prior failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        ),
        StepRecord(
            index=2,
            thought="prior failed copy",
            action="code_execute",
            content="cp template/status.txt status.txt",
            selected_skill_id=None,
            command_result={"command": "cp template/status.txt status.txt", "exit_code": 1, "stdout": "", "stderr": "missing template", "timed_out": False},
            verification={"passed": False, "reasons": ["copy failed"]},
        ),
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "unsatisfied_expected_contents": ["status.txt"],
        "completion_ratio": 0.5,
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "python tools/rewrite_recovery_contract.py --target status.txt"
    assert decision.decision_source == "llm"
    assert "planner_recovery_brief" in client.last_payload
    assert "Critic has exhausted the bounded recovery set" in client.last_payload["planner_recovery_brief"]
    assert "Synthesize a fresh verifier-relevant subgoal or rewrite the recovery contract" in client.last_payload["planner_recovery_brief"]
    assert "Recovery rewrite directive:" in client.last_decision_prompt


def test_llm_policy_planner_recovery_rewrite_requires_critic_proof():
    class PlannerRewriteClient(CapturingClient):
        def create_decision(self, *, system_prompt, decision_prompt, state_payload):
            self.last_system_prompt = system_prompt
            self.last_decision_prompt = decision_prompt
            self.last_payload = state_payload
            return {
                "thought": "try another planner command",
                "action": "code_execute",
                "content": "python tools/try_another_repair.py",
                "done": False,
            }

    client = PlannerRewriteClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="planner_recovery_rewrite_requires_proof_task",
            prompt="repair the expected status artifact",
            workspace_subdir="planner_recovery_rewrite_requires_proof_task",
            suggested_commands=[
                "printf 'old\\n' > status.txt",
                "cp template/status.txt status.txt",
            ],
            expected_files=["status.txt"],
            expected_file_contents={"status.txt": "done\n"},
            metadata={"difficulty": "long_horizon", "benchmark_family": "workflow"},
        )
    )
    state.current_role = "planner"
    state.active_subgoal = "materialize expected artifact status.txt"
    state.subgoal_diagnoses = {
        state.active_subgoal: {
            "path": "status.txt",
            "signals": ["command_failure", "no_state_progress"],
            "summary": "status.txt remains incorrect after the bounded repair commands were attempted",
            "source_role": "planner",
        }
    }
    state.consecutive_failures = 2
    state.consecutive_no_progress_steps = 2
    state.repeated_action_count = 2
    state.last_action_signature = "code_execute:cp template/status.txt status.txt"
    state.history = [
        StepRecord(
            index=1,
            thought="prior failed write",
            action="code_execute",
            content="printf 'old\\n' > status.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'old\\n' > status.txt", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["content mismatch"]},
        ),
        StepRecord(
            index=2,
            thought="prior failed copy",
            action="code_execute",
            content="cp template/status.txt status.txt",
            selected_skill_id=None,
            command_result={"command": "cp template/status.txt status.txt", "exit_code": 1, "stdout": "", "stderr": "missing template", "timed_out": False},
            verification={"passed": False, "reasons": ["copy failed"]},
        ),
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "expected_artifacts": ["status.txt"],
        "forbidden_artifacts": [],
        "preserved_artifacts": [],
        "unsatisfied_expected_contents": ["status.txt"],
        "completion_ratio": 0.5,
    }

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert decision.content == "python tools/try_another_repair.py"
    assert decision.decision_source == "llm"
    assert "planner_recovery_brief" not in client.last_payload
    assert "Recovery rewrite directive:" not in client.last_decision_prompt


def test_llm_policy_includes_planner_recovery_artifact_in_payload():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(task=TaskBank().get("cleanup_task"))
    state.current_role = "planner"
    state.plan = [
        "materialize expected artifact status.txt",
        "remove forbidden artifact temp.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.planner_recovery_artifact = {
        "kind": "planner_recovery_rewrite",
        "source_subgoal": state.active_subgoal,
        "rewritten_subgoal": "reframe verifier-visible recovery for expected artifact status.txt",
        "next_stage_objective": "write workflow report reports/status.txt",
        "staged_plan_update": [
            "write workflow report reports/status.txt",
            "run workflow test release status check",
        ],
        "summary": "status.txt remains incorrect after bounded repairs",
        "stale_commands": [
            "printf 'old\\n' > status.txt",
            "cp template/status.txt status.txt",
        ],
        "related_objectives": [
            "write workflow report reports/status.txt",
            "run workflow test release status check",
        ],
        "ranked_objectives": [
            {
                "objective": "write workflow report reports/status.txt",
                "score": 96,
                "status": "pending",
                "reason": "still blocks verifier-visible progress",
            },
            {
                "objective": "run workflow test release status check",
                "score": 68,
                "status": "pending",
                "reason": "still lacks matching execution evidence",
            },
        ],
        "contract_outline": [
            "inspect current repo/workspace state around status.txt",
            "define the next verifier-visible milestone for status.txt",
            "choose a new command path outside the exhausted task-contract repair set",
        ],
        "source_role": "critic",
    }

    policy.decide(state)

    assert client.last_payload["planner_recovery_artifact"]["rewritten_subgoal"] == (
        "reframe verifier-visible recovery for expected artifact status.txt"
    )
    assert client.last_payload["planner_recovery_plan_update"] == [
        "write workflow report reports/status.txt",
        "run workflow test release status check",
    ]
    assert client.last_payload["plan"][:2] == client.last_payload["planner_recovery_plan_update"]
    assert "planner_recovery_brief" in client.last_payload
    assert "New planner objective" in client.last_payload["planner_recovery_brief"]
    assert "Current staged recovery objective" in client.last_payload["planner_recovery_brief"]
    assert "Related verifier obligations" in client.last_payload["planner_recovery_brief"]
    assert any(
        chunk["source"] == "planner_recovery"
        for chunk in client.last_payload["state_context_chunks"]
    )


def test_llm_policy_surfaces_long_horizon_software_work_agenda_before_recovery():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_agenda_task",
            prompt="Complete the staged software release work.",
            workspace_subdir="software_work_agenda_task",
            suggested_commands=[],
            expected_files=["src/release_state.txt", "reports/release_review.txt"],
            metadata={
                "benchmark_family": "project",
                "difficulty": "long_horizon",
                "synthetic_edit_plan": [
                    {
                        "path": "src/release_state.txt",
                        "edit_kind": "line_replace",
                    }
                ],
            },
        )
    )
    state.current_role = "planner"
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_report_paths": ["reports/release_review.txt"],
        "workflow_required_tests": ["release smoke"],
        "updated_report_paths": [],
    }
    state.software_work_stage_state = {
        "current_objective": "materialize expected artifact src/release_state.txt",
        "last_status": "stalled",
        "objective_states": {
            "materialize expected artifact src/release_state.txt": "stalled",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {
            "materialize expected artifact src/release_state.txt": 2,
        },
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "stalled",
                "step_index": 1,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }

    policy.decide(state)

    assert client.last_payload["software_work_plan_update"][:3] == [
        "apply planned edit src/release_state.txt",
        "complete implementation for src/release_state.txt",
        "materialize expected artifact src/release_state.txt",
    ]
    assert client.last_payload["software_work_plan_update"].index("run workflow test release smoke") > 2
    assert client.last_payload["software_work_plan_update"].index(
        "write workflow report reports/release_review.txt"
    ) > 2
    assert client.last_payload["plan"][:2] == [
        "apply planned edit src/release_state.txt",
        "complete implementation for src/release_state.txt",
    ]
    assert client.last_payload["software_work_stage_state"]["last_status"] == "stalled"
    assert client.last_payload["software_work_phase_state"]["suggested_phase"] == "implementation"
    assert client.last_payload["software_work_phase_gate_state"]["gate_phase"] == "implementation"
    assert any(
        chunk["source"] in {"software_work", "campaign_contract"}
        for chunk in client.last_payload["state_context_chunks"]
    )
    assert any(chunk["source"] == "software_work_phase" for chunk in client.last_payload["state_context_chunks"])
    assert any(chunk["source"] == "software_work_gate" for chunk in client.last_payload["state_context_chunks"])


def test_llm_policy_surfaces_software_work_phase_gate_for_unresolved_branch_acceptance():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_phase_gate_payload_task",
            prompt="Complete the release-acceptance workflow.",
            workspace_subdir="software_work_phase_gate_payload_task",
            suggested_commands=[],
            metadata={"benchmark_family": "repo_sandbox", "difficulty": "long_horizon"},
        )
    )
    state.current_role = "planner"
    state.plan = [
        "accept required branch worker/api-release",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = state.plan[0]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["worker/api-release"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch worker/api-release",
        "last_status": "stalled",
        "objective_states": {
            "accept required branch worker/api-release": "stalled",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"accept required branch worker/api-release": 1},
        "recent_outcomes": [
            {
                "objective": "accept required branch worker/api-release",
                "status": "stalled",
                "step_index": 1,
                "command": "git merge worker/api-release",
                "progress_delta": 0.0,
                "regressed": False,
            }
        ],
    }

    policy.decide(state)

    assert client.last_payload["software_work_phase_gate_state"]["gate_kind"] == "merge_acceptance"
    assert client.last_payload["software_work_phase_gate_state"]["gate_phase"] == "migration"
    assert client.last_payload["software_work_phase_gate_state"]["gate_objectives"] == [
        "accept required branch worker/api-release"
    ]
    assert "software_work_phase_gate_brief" in client.last_payload
    assert "Resolve these obligations first" in client.last_payload["software_work_phase_gate_brief"]
    assert any(chunk["source"] == "software_work_gate" for chunk in client.last_payload["state_context_chunks"])


def test_llm_policy_surfaces_campaign_contract_and_scores_off_contract_mutation_lower():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client, context_provider=NonDeterministicContextProvider())
    state = AgentState(
        task=TaskSpec(
            task_id="campaign_contract_payload_task",
            prompt="Finish the release workflow without drifting off the main obligations.",
            workspace_subdir="campaign_contract_payload_task",
            suggested_commands=[],
            expected_files=["src/release_state.txt", "reports/release_review.txt"],
            metadata={"benchmark_family": "project", "difficulty": "long_horizon"},
        )
    )
    state.current_role = "planner"
    state.plan = [
        "materialize expected artifact src/release_state.txt",
        "run workflow test release smoke",
        "write workflow report reports/release_review.txt",
    ]
    state.active_subgoal = "materialize expected artifact src/release_state.txt"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "missing_expected_artifacts": ["src/release_state.txt"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/release_review.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "materialize expected artifact src/release_state.txt",
        "last_status": "regressed",
        "objective_states": {
            "materialize expected artifact src/release_state.txt": "regressed",
            "run workflow test release smoke": "pending",
            "write workflow report reports/release_review.txt": "pending",
        },
        "attempt_counts": {"materialize expected artifact src/release_state.txt": 2},
        "recent_outcomes": [
            {
                "objective": "materialize expected artifact src/release_state.txt",
                "status": "regressed",
                "step_index": 2,
                "command": "python scripts/fix_release.py --path src/release_state.txt",
                "progress_delta": 0.0,
                "regressed": True,
            }
        ],
    }
    state.consecutive_failures = 1
    state.consecutive_no_progress_steps = 1
    state.repeated_action_count = 2
    state.latest_state_transition = {"regressed": True, "regressions": ["src/release_state.txt"]}

    policy.decide(state)

    assert client.last_payload["campaign_contract_state"]["current_objective"] == (
        "materialize expected artifact src/release_state.txt"
    )
    assert client.last_payload["campaign_contract_state"]["anchor_objectives"][:2] == [
        "materialize expected artifact src/release_state.txt",
        "complete implementation for src/release_state.txt",
    ]
    assert "campaign_contract_brief" in client.last_payload
    assert "Current campaign objective" in client.last_payload["campaign_contract_brief"]
    assert any(chunk["source"] == "campaign_contract" for chunk in client.last_payload["state_context_chunks"])
    assert policy._command_control_score(
        state,
        "python scripts/fix_release.py --path src/release_state.txt",
    ) > policy._command_control_score(
        state,
        "printf 'scratch\\n' > notes/todo.txt",
    )


def test_llm_policy_plan_progress_direct_prioritizes_gate_aligned_merge_command():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="software_work_phase_gate_direct_task",
            prompt="Complete the release-acceptance workflow.",
            workspace_subdir="software_work_phase_gate_direct_task",
            suggested_commands=[
                "tests/test_release.sh && printf 'release suite passed\\n' > reports/test_report.txt",
                "git merge --no-ff worker/api-release -m 'merge worker/api-release' && tests/test_release.sh && printf 'release suite passed\\n' > reports/test_report.txt",
                "cat reports/test_report.txt",
            ],
            metadata={"benchmark_family": "repo_sandbox", "difficulty": "long_horizon"},
        )
    )
    state.current_role = "planner"
    state.active_subgoal = "accept required branch worker/api-release"
    state.history = [
        StepRecord(
            index=1,
            thought="previous failed attempt",
            action="code_execute",
            content="git status --short",
            selected_skill_id=None,
            command_result={"command": "git status --short", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["required worker branch not accepted into main: worker/api-release"]},
        )
    ]
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["worker/api-release"],
        "workflow_required_tests": ["release smoke"],
        "workflow_report_paths": ["reports/test_report.txt"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch worker/api-release",
        "last_status": "stalled",
        "objective_states": {
            "accept required branch worker/api-release": "stalled",
            "run workflow test release smoke": "pending",
            "write workflow report reports/test_report.txt": "pending",
        },
        "attempt_counts": {"accept required branch worker/api-release": 1},
        "recent_outcomes": [],
    }

    decision = policy._plan_progress_direct_decision(state, blocked_commands=[])

    assert decision is not None
    assert decision.decision_source == "plan_progress_direct"
    assert decision.content.startswith("git merge --no-ff worker/api-release")


def test_llm_policy_shared_repo_integrator_keeps_merge_segment_under_phase_gate():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="shared_repo_phase_gate_task",
            prompt="Accept required worker branches before later release steps.",
            workspace_subdir="shared_repo_phase_gate_task",
            suggested_commands=[
                "git merge --no-ff worker/api-release -m 'merge worker/api-release' && printf 'release runbook ready\\n' > docs/runbook.md && tests/test_release.sh"
            ],
            metadata={
                "benchmark_family": "repo_sandbox",
                "difficulty": "long_horizon",
                "shared_repo_order": 1,
                "semantic_verifier": {
                    "required_merged_branches": ["worker/api-release"],
                },
                "workflow_guard": {
                    "shared_repo_id": "repo_sandbox_release_train_conflict",
                    "claimed_paths": ["docs/runbook.md"],
                    "target_branch": "main",
                },
            },
        )
    )
    state.active_subgoal = "preserve required artifact docs/ops_notes.md"
    state.world_model_summary = {
        "horizon": "long_horizon",
        "workflow_required_merges": ["worker/api-release"],
        "workflow_required_tests": ["release smoke"],
    }
    state.software_work_stage_state = {
        "current_objective": "accept required branch worker/api-release",
        "last_status": "pending",
        "objective_states": {
            "accept required branch worker/api-release": "pending",
            "run workflow test release smoke": "pending",
        },
        "attempt_counts": {},
        "recent_outcomes": [],
    }

    decision = policy._pre_context_shared_repo_integrator_direct_decision(state)

    assert decision is not None
    assert decision.content == "git merge --no-ff worker/api-release -m 'merge worker/api-release'"


class InvalidActionClient:
    def create_decision(self, *, system_prompt, decision_prompt, state_payload):
        del system_prompt, decision_prompt, state_payload
        return {
            "thought": "do something unsupported",
            "action": "launch_rocket",
            "content": "boom",
            "done": False,
        }


class WorkspacePrefixedClient:
    def create_decision(self, *, system_prompt, decision_prompt, state_payload):
        del system_prompt, decision_prompt, state_payload
        return {
            "thought": "write file into task subdir",
            "action": "code_execute",
            "content": "printf 'done\\n' > rewrite_task/note.txt",
            "done": False,
        }


class FalseCommandClient:
    def create_decision(self, *, system_prompt, decision_prompt, state_payload):
        del system_prompt, decision_prompt, state_payload
        return {
            "thought": "run a null failing command",
            "action": "code_execute",
            "content": "false",
            "done": False,
        }


def test_llm_policy_falls_back_on_unsupported_action():
    decision = LLMDecisionPolicy(InvalidActionClient()).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "respond"
    assert decision.done is True
    assert "unsupported action" in decision.thought.lower()


def test_llm_policy_normalizes_workspace_prefixed_llm_commands():
    decision = LLMDecisionPolicy(WorkspacePrefixedClient()).decide(
        AgentState(task=TaskBank().get("rewrite_task"))
    )

    assert decision.action == "code_execute"
    assert decision.content == "printf 'done\\n' > note.txt"


def test_llm_policy_rejects_prohibited_null_failing_llm_commands():
    decision = LLMDecisionPolicy(FalseCommandClient(), config=KernelConfig(provider="mock", use_tolbert_context=False)).decide(
        AgentState(
            task=TaskSpec(
                task_id="null_guard_task",
                prompt="Create ready.txt.",
                workspace_subdir="null_guard_task",
                suggested_commands=[],
                expected_files=["ready.txt"],
                expected_file_contents={"ready.txt": "ready\n"},
                metadata={"benchmark_family": "bounded"},
            )
        )
    )

    assert decision.action == "respond"
    assert decision.done is True
    assert decision.decision_source == "llm_guardrail"


def test_llm_policy_allows_explicit_task_required_false_command():
    decision = LLMDecisionPolicy(FalseCommandClient(), config=KernelConfig(provider="mock", use_tolbert_context=False)).decide(
        AgentState(
            task=TaskSpec(
                task_id="explicit_false_task",
                prompt="Run the explicit failing command.",
                workspace_subdir="explicit_false_task",
                suggested_commands=["false"],
                metadata={"benchmark_family": "bounded"},
            )
        )
    )

    assert decision.action == "code_execute"
    assert decision.content == "false"


def test_apply_tolbert_shadow_rejects_prohibited_null_failing_direct_command():
    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(
        task=TaskSpec(
            task_id="null_guard_task",
            prompt="Create ready.txt.",
            workspace_subdir="null_guard_task",
            expected_files=["ready.txt"],
            expected_file_contents={"ready.txt": "ready\n"},
            metadata={"benchmark_family": "bounded"},
        )
    )

    decision = policy._apply_tolbert_shadow(
        ActionDecision(
            thought="execute direct command",
            action="code_execute",
            content="false",
            done=False,
            decision_source="tolbert_direct",
        ),
        {},
        "primary",
        state=state,
    )

    assert decision.action == "respond"
    assert decision.done is True
    assert decision.decision_source == "command_guardrail"


def test_skill_action_decision_uses_non_llm_source():
    policy = LLMDecisionPolicy(MockLLMClient(), config=KernelConfig(provider="mock", use_tolbert_context=False))
    state = AgentState(task=TaskBank().get("hello_task"))

    decision = policy._skill_action_decision(
        state,
        {
            "skill_id": "skill:hello_task:primary",
            "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
        },
        thought_prefix="Use skill",
    )

    assert decision.action == "code_execute"
    assert decision.decision_source == "skill_direct"


def test_llm_policy_uses_matching_skill_before_llm():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
            }
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:primary"
    assert decision.action == "code_execute"
    assert decision.content == "printf 'hello agent kernel\\n' > hello.txt"


def test_llm_policy_uses_tolbert_command_before_skill_short_circuit():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["echo 'hello' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id is None
    assert decision.selected_retrieval_span_id == "task:hello:suggested:1"
    assert decision.retrieval_influenced is True
    assert decision.action == "code_execute"
    assert decision.content == "printf 'hello agent kernel\\n' > hello.txt"


def test_llm_policy_defers_first_step_direct_retrieval_when_confidence_is_below_strict_gate():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class AlmostTrustedContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.82,
                    "trust_retrieval": True,
                    "retrieval_guidance": {
                        "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                        "recommended_command_spans": [
                            {
                                "span_id": "task:hello:suggested:1",
                                "command": "printf 'hello agent kernel\\n' > hello.txt",
                            }
                        ],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=AlmostTrustedContextProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:primary"
    assert decision.selected_retrieval_span_id == "task:hello:suggested:1"
    assert decision.retrieval_ranked_skill is True


def test_llm_policy_keeps_trusted_workflow_guidance_under_low_confidence_for_screened_tolbert_candidates():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="workflow_guarded_project_task",
            prompt="repair the guarded project outputs",
            workspace_subdir="workflow_guarded_project_task",
            suggested_commands=[],
            expected_files=["src/api_status.txt", "reports/test_report.txt"],
            expected_file_contents={
                "src/api_status.txt": "API_STATUS=ready\n",
                "reports/test_report.txt": "api suite passed\n",
            },
            metadata={
                "benchmark_family": "project",
                "workflow_guard": {
                    "claimed_paths": ["src/api_status.txt", "reports/test_report.txt"],
                },
            },
        )
    )
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "path_confidence": 0.4,
            "trust_retrieval": True,
            "retrieval_guidance": {
                "recommended_commands": [
                    "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt"
                ],
                "recommended_command_spans": [
                    {
                        "span_id": "learning:success_skill:workflow_guarded_project_task",
                        "command": "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt",
                    }
                ],
                "avoidance_notes": [],
                "evidence": ["learning:success_skill:workflow_guarded_project_task: learned project recovery"],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    screened = policy._screened_tolbert_retrieval_guidance(
        state,
        top_skill=None,
        retrieval_guidance=state.context_packet.control["retrieval_guidance"],
        blocked_commands=[],
    )

    assert screened["recommended_commands"] == [
        "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt"
    ]
    assert screened["recommended_command_spans"] == [
        {
            "span_id": "learning:success_skill:workflow_guarded_project_task",
            "command": "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt",
        }
    ]


def test_llm_policy_executes_trusted_workflow_guidance_on_low_confidence_first_step():
    class LowConfidenceWorkflowGuidanceProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-04-02T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.4,
                    "trust_retrieval": True,
                    "retrieval_guidance": {
                        "recommended_commands": [
                            "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt"
                        ],
                        "recommended_command_spans": [
                            {
                                "span_id": "learning:success_skill:workflow_guarded_project_task",
                                "command": "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && printf 'api suite passed\\n' > reports/test_report.txt",
                            }
                        ],
                        "avoidance_notes": [],
                        "evidence": ["learning:success_skill:workflow_guarded_project_task: learned project recovery"],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    policy = LLMDecisionPolicy(
        client,
        context_provider=LowConfidenceWorkflowGuidanceProvider(),
        config=KernelConfig(provider="mock", tolbert_mode="deterministic_command"),
    )
    decision = policy.decide(
        AgentState(
            task=TaskSpec(
                task_id="workflow_guarded_project_task",
                prompt="repair the guarded project outputs",
                workspace_subdir="workflow_guarded_project_task",
                suggested_commands=[],
                expected_files=["src/api_status.txt", "reports/test_report.txt"],
                expected_file_contents={
                    "src/api_status.txt": "API_STATUS=ready\n",
                    "reports/test_report.txt": "api suite passed\n",
                },
                metadata={
                    "benchmark_family": "project",
                    "workflow_guard": {
                        "claimed_paths": ["src/api_status.txt", "reports/test_report.txt"],
                    },
                },
            )
        )
    )

    assert decision.action == "code_execute"
    assert decision.selected_retrieval_span_id == "learning:success_skill:workflow_guarded_project_task"
    assert decision.retrieval_influenced is True
    assert decision.content == (
        "mkdir -p src reports && printf 'API_STATUS=ready\\n' > src/api_status.txt && "
        "printf 'api suite passed\\n' > reports/test_report.txt"
    )
    assert client.last_payload is None


def test_llm_policy_inference_failure_fallback_preserves_retrieval_guidance():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(
        task=TaskSpec(
            task_id="repository_release_recovery_task",
            prompt="repair the repository release outputs",
            workspace_subdir="repository_release_recovery_task",
            suggested_commands=[],
            expected_files=["config/schema_status.txt", "reports/release_status.txt"],
            expected_file_contents={
                "config/schema_status.txt": "SCHEMA_STATUS=ready\n",
                "reports/release_status.txt": "release verified\n",
            },
            metadata={"benchmark_family": "repository"},
        )
    )
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "path_confidence": 0.22,
            "trust_retrieval": True,
            "retrieval_guidance": {
                "recommended_commands": [
                    "mkdir -p config reports && printf 'SCHEMA_STATUS=ready\\n' > config/schema_status.txt && printf 'release verified\\n' > reports/release_status.txt"
                ],
                "recommended_command_spans": [
                    {
                        "span_id": "learning:success_skill:repository_release_recovery_task",
                        "command": "mkdir -p config reports && printf 'SCHEMA_STATUS=ready\\n' > config/schema_status.txt && printf 'release verified\\n' > reports/release_status.txt",
                    }
                ],
                "avoidance_notes": [],
                "evidence": [
                    "learning:success_skill:repository_release_recovery_task: learned repository recovery"
                ],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    decision = policy.fallback_decision(state, failure_origin="inference_failure")

    assert decision is not None
    assert decision.action == "code_execute"
    assert decision.decision_source == "deterministic_fallback"
    assert decision.selected_retrieval_span_id == "learning:success_skill:repository_release_recovery_task"
    assert decision.retrieval_influenced is True
    assert decision.content == (
        "mkdir -p config reports && printf 'SCHEMA_STATUS=ready\\n' > config/schema_status.txt && "
        "printf 'release verified\\n' > reports/release_status.txt"
    )


def test_llm_policy_inference_failure_fallback_preserves_tolbert_shadow_route(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "shadow_benchmark_families": ["repository", "project", "workflow"],
                    "primary_benchmark_families": [],
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        config=KernelConfig(
            tolbert_model_artifact_path=artifact_path,
            use_tolbert_model_artifacts=True,
        ),
    )
    state = AgentState(
        task=TaskSpec(
            task_id="deployment_manifest_fallback_shadow",
            prompt="complete the deployment manifest task",
            workspace_subdir="deployment_manifest_fallback_shadow",
            suggested_commands=[
                "mkdir -p deploy && printf 'deployment manifest ready\\n' > deploy/manifest.txt && printf 'deployment checklist complete\\n' > deploy/checklist.txt"
            ],
            expected_files=["deploy/manifest.txt", "deploy/checklist.txt"],
            expected_file_contents={
                "deploy/manifest.txt": "deployment manifest ready\n",
                "deploy/checklist.txt": "deployment checklist complete\n",
            },
            metadata={"benchmark_family": "project"},
        )
    )
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "path_confidence": 0.18,
            "trust_retrieval": True,
            "retrieval_guidance": {
                "recommended_commands": [],
                "recommended_command_spans": [],
                "avoidance_notes": [],
                "evidence": [],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    decision = policy.fallback_decision(state, failure_origin="inference_failure")

    assert decision is not None
    assert decision.action == "code_execute"
    assert decision.decision_source == "deterministic_fallback"
    assert decision.tolbert_route_mode == "shadow"
    assert decision.content.startswith("mkdir -p deploy &&")


def test_llm_policy_extracts_blocked_command_from_double_quoted_avoidance_note():
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-03-16T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "path_confidence": 0.2,
            "trust_retrieval": False,
            "retrieval_guidance": {
                "recommended_commands": [],
                "recommended_command_spans": [],
                "avoidance_notes": ['avoid repeating "printf \'hello agent kernel\\\\n\' > hello.txt" when it previously failed verification'],
                "evidence": [],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    blocked = LLMDecisionPolicy._blocked_commands(state)

    assert blocked == ["printf 'hello agent kernel\\n' > hello.txt"]


def test_llm_policy_requires_active_skill_ranking_for_first_step_when_retrieval_signal_is_low_confidence():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class LowConfidenceSourceTaskProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.2,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [{"metadata": {"task_id": "hello_task"}}], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=LowConfidenceSourceTaskProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id is None
    assert decision.retrieval_ranked_skill is False


def test_llm_policy_applies_retained_behavior_controls(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "critic_repeat_failure_bias": 2,
                    "planner_subgoal_command_bias": 3,
                    "required_artifact_first_step_bias": 1,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(prompt_proposals_path=prompt_path),
    )
    state = AgentState(task=TaskBank().get("hello_task"))
    state.current_role = "critic"
    state.consecutive_failures = 1
    state.history.append(
        StepRecord(
            index=1,
            thought="failed prior command",
            action="code_execute",
            content="printf 'hello agent kernel\\n' > hello.txt",
            selected_skill_id=None,
            command_result=None,
            verification={"passed": False, "reasons": ["failed"]},
        )
    )

    decision = policy.decide(state)

    assert decision.action == "respond"
    assert decision.done is True


def test_llm_policy_verifier_alignment_bias_prefers_expected_artifact_command(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "verifier_alignment_bias": 2,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(prompt_proposals_path=prompt_path),
    )
    state = AgentState(task=TaskBank().get("hello_task"))

    expected_score = policy._command_control_score(state, "printf 'hello agent kernel\\n' > hello.txt")
    unrelated_score = policy._command_control_score(state, "printf 'hello agent kernel\\n' > notes.txt")

    assert expected_score > unrelated_score


def test_llm_policy_applies_retained_role_directives(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "planner_controls": {},
                "role_directives": {
                    "planner": "Bias the next subgoal toward verifier-visible artifact checks.",
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(prompt_proposals_path=prompt_path),
    )

    system_prompt = policy._role_system_prompt("planner")

    assert "verifier-visible artifact checks" in system_prompt


def test_llm_policy_disables_prompt_adjustments_when_prompt_proposals_disabled(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {"skill_ranking_confidence_boost": 0.2},
                "planner_controls": {"prepend_verifier_contract_check": True},
                "role_directives": {"planner": "extra planner directive"},
                "proposals": [
                    {
                        "area": "decision",
                        "priority": 6,
                        "reason": "test",
                        "suggestion": "Prefer more caution on weak retrieval.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(prompt_proposals_path=prompt_path, use_prompt_proposals=False),
    )

    assert policy._active_prompt_adjustments() == []
    assert "extra planner directive" not in policy._role_system_prompt("planner")


def test_llm_policy_ignores_low_confidence_retrieval_task_bias_for_unrelated_skill():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            },
            {
                "skill_id": "skill:archive_command_seed_task:primary",
                "kind": "command_sequence",
                "source_task_id": "archive_command_seed_task",
                "applicable_tasks": ["archive_command_seed_task"],
                "procedure": {"commands": ["mkdir -p archive && printf 'archived\\n' > archive/out.txt"]},
                "known_failure_types": [],
                "quality": 0.95,
            },
        ]
    )

    class LowConfidenceSourceBiasProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.04,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={
                    "branch_scoped": [{"metadata": {"task_id": "archive_command_seed_task"}}],
                    "fallback_scoped": [],
                    "global": [],
                },
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=LowConfidenceSourceBiasProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("deployment_manifest_task")))

    assert decision.selected_skill_id is None


def test_llm_policy_blocks_unrelated_skill_short_circuit_for_failure_recovery_task():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.99,
                "benchmark_family": "micro",
            }
        ]
    )
    policy = LLMDecisionPolicy(MockLLMClient(), skill_library=skill_library)
    task = TaskSpec(
        task_id="deployment_manifest_task_project_recovery",
        prompt="Recover the project workspace.",
        workspace_subdir="deployment_manifest_task_project_recovery",
        suggested_commands=["mkdir -p project && printf 'project recovery complete\\n' > project/recovery.txt"],
        success_command="test -f project/recovery.txt",
        expected_files=["project/recovery.txt"],
        expected_file_contents={"project/recovery.txt": "project recovery complete\n"},
        metadata={
            "curriculum_kind": "failure_recovery",
            "source_task": "deployment_manifest_task",
            "benchmark_family": "project",
        },
    )

    decision = policy.decide(AgentState(task=task))

    assert decision.selected_skill_id is None


def test_llm_policy_penalizes_workspace_prefixed_retrieval_commands():
    class MixedContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.95,
                    "trust_retrieval": True,
                    "retrieval_guidance": {
                        "recommended_commands": [
                            "printf 'done\\n' > rewrite_task/note.txt",
                            "printf 'done\\n' > note.txt",
                        ],
                        "recommended_command_spans": [
                            {
                                "span_id": "bad",
                                "command": "printf 'done\\n' > rewrite_task/note.txt",
                            },
                            {
                                "span_id": "good",
                                "command": "printf 'done\\n' > note.txt",
                            },
                        ],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=MixedContextProvider(),
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("rewrite_task")))

    assert decision.selected_retrieval_span_id == "good"
    assert decision.content == "printf 'done\\n' > note.txt"


def test_llm_policy_uses_tolbert_ranked_source_task_skill():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:handshake_seed_task:primary",
                "kind": "command_sequence",
                "source_task_id": "handshake_seed_task",
                "applicable_tasks": ["handshake_seed_task"],
                "procedure": {"commands": ["printf 'kernel handshake ready\\n' > handshake.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class SourceTaskContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.8,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={
                    "branch_scoped": [{"metadata": {"task_id": "handshake_seed_task"}}],
                    "fallback_scoped": [],
                    "global": [],
                },
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=SourceTaskContextProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("handshake_retrieval_task")))

    assert decision.selected_skill_id == "skill:handshake_seed_task:primary"
    assert decision.retrieval_influenced is True
    assert decision.retrieval_ranked_skill is True
    assert decision.content == "printf 'kernel handshake ready\\n' > handshake.txt"


def test_llm_policy_records_retrieval_span_for_ranked_skill():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class RankedSkillContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.5,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                        "recommended_command_spans": [
                            {
                                "span_id": "task:hello:suggested:1",
                                "command": "printf 'hello agent kernel\\n' > hello.txt",
                            }
                        ],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={
                    "branch_scoped": [{"metadata": {"task_id": "hello_task"}}],
                    "fallback_scoped": [],
                    "global": [],
                },
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=RankedSkillContextProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:primary"
    assert decision.selected_retrieval_span_id == "task:hello:suggested:1"
    assert decision.retrieval_ranked_skill is True


def test_choose_tolbert_route_allows_trusted_primary_override_below_min_confidence():
    from agent_kernel.modeling.policy.runtime import choose_tolbert_route

    route = choose_tolbert_route(
        runtime_policy={
            "primary_benchmark_families": ["project"],
            "shadow_benchmark_families": ["project"],
            "min_path_confidence": 0.7,
            "allow_trusted_primary_without_min_confidence": True,
            "trusted_primary_min_confidence": 0.4,
            "require_trusted_retrieval": True,
            "use_latent_state": True,
        },
        benchmark_family="project",
        path_confidence=0.45,
        trust_retrieval=True,
    )

    assert route.mode == "primary"
    assert "override" in route.reason


def test_llm_policy_applies_retained_retrieval_threshold_overrides(tmp_path):
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "overrides": {
                    "tolbert_skill_ranking_min_confidence": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class LowButOverriddenConfidenceProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.25,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [{"metadata": {"task_id": "hello_task"}}], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=LowButOverriddenConfidenceProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock", retrieval_proposals_path=retrieval_path),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:primary"
    assert decision.retrieval_ranked_skill is True


def test_llm_policy_applies_retained_prompt_behavior_controls(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {
                    "skill_ranking_confidence_boost": 0.2,
                },
                "proposals": [
                    {
                        "area": "decision",
                        "priority": 6,
                        "reason": "test",
                        "suggestion": "Prefer more caution on weak retrieval.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:primary",
                "kind": "command_sequence",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "known_failure_types": [],
                "quality": 0.9,
            }
        ]
    )

    class LowConfidenceProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.4,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": [],
                        "recommended_command_spans": [],
                        "avoidance_notes": [],
                        "evidence": [],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [{"metadata": {"task_id": "hello_task"}}], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=LowConfidenceProvider(),
        skill_library=skill_library,
        config=KernelConfig(provider="mock", prompt_proposals_path=prompt_path),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id is None
    assert decision.retrieval_ranked_skill is False


def test_llm_policy_can_use_source_task_skill():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:handshake_seed_task:primary",
                "kind": "command_sequence",
                "source_task_id": "handshake_seed_task",
                "applicable_tasks": ["handshake_seed_task"],
                "procedure": {"commands": ["printf 'kernel handshake ready\\n' > handshake.txt"]},
                "known_failure_types": [],
            }
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("handshake_retrieval_task")))

    assert decision.selected_skill_id == "skill:handshake_seed_task:primary"
    assert decision.action == "code_execute"
    assert decision.content == "printf 'kernel handshake ready\\n' > handshake.txt"


def test_llm_policy_normalizes_workspace_prefixed_skill_commands():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:rewrite_task:primary",
                "kind": "command_sequence",
                "source_task_id": "rewrite_task",
                "applicable_tasks": ["rewrite_task"],
                "procedure": {
                    "commands": ["mkdir -p rewrite_task && printf 'done\\n' > rewrite_task/note.txt"]
                },
                "known_failure_types": [],
            }
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("rewrite_task")))

    assert decision.selected_skill_id == "skill:rewrite_task:primary"
    assert decision.content == "printf 'done\\n' > note.txt"


def test_skill_library_filters_low_quality_skills():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:weak",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["echo 'hello' > hello.txt"]},
                "quality": 0.4,
            },
            {
                "skill_id": "skill:hello_task:strong",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "quality": 0.9,
            },
        ],
        min_quality=0.75,
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:strong"


def test_skill_library_prefers_highest_quality_match():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:hello_task:mid",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["echo 'hello' > hello.txt"]},
                "quality": 0.8,
            },
            {
                "skill_id": "skill:hello_task:best",
                "source_task_id": "hello_task",
                "applicable_tasks": ["hello_task"],
                "procedure": {"commands": ["printf 'hello agent kernel\\n' > hello.txt"]},
                "quality": 0.95,
            },
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.selected_skill_id == "skill:hello_task:best"


def test_llm_policy_matches_retrieval_span_when_retrieved_command_contains_literal_newline():
    class NewlineSpanContextProvider:
        def compile(self, state):
            del state
            return ContextPacket(
                request_id="req-1",
                created_at="2026-03-16T00:00:00+00:00",
                task={"goal": "g", "completion_criteria": "c"},
                control={
                    "mode": "verify",
                    "path_confidence": 0.5,
                    "trust_retrieval": False,
                    "retrieval_guidance": {
                        "recommended_commands": ["printf 'hello agent kernel\n' > hello.txt"],
                        "recommended_command_spans": [
                            {
                                "span_id": "task:hello:literal-newline",
                                "command": "printf 'hello agent kernel\n' > hello.txt",
                            }
                        ],
                        "avoidance_notes": [],
                        "evidence": ["task:hello:literal-newline: template command"],
                    },
                },
                tolbert={"path_prediction": {"tree_version": "tol_v1"}},
                retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
                verifier_contract={"success_command": "true"},
            )

    client = CapturingClient()
    decision = LLMDecisionPolicy(
        client,
        context_provider=NewlineSpanContextProvider(),
        config=KernelConfig(provider="mock"),
    ).decide(AgentState(task=TaskBank().get("hello_task")))

    assert decision.action == "code_execute"
    assert decision.content == "printf 'hello agent kernel\\n' > hello.txt"


def test_llm_policy_applies_retained_tolbert_policy_overrides(tmp_path):
    prompt_path = tmp_path / "prompts" / "prompt_proposals.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(
            {
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "controls": {},
                "tolbert_runtime_policy_overrides": {
                    "min_path_confidence": 0.91,
                    "primary_min_command_score": 4,
                },
                "tolbert_decoder_policy_overrides": {
                    "max_task_suggestions": 5,
                },
                "tolbert_rollout_policy_overrides": {
                    "stop_completion_weight": 9.5,
                },
                "tolbert_hybrid_scoring_policy_overrides": {
                    "decoder_logprob_weight": 0.22,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(prompt_proposals_path=prompt_path),
    )

    assert policy._tolbert_runtime_policy()["min_path_confidence"] == 0.91
    assert policy._tolbert_runtime_policy()["primary_min_command_score"] == 4
    assert policy._tolbert_decoder_policy()["max_task_suggestions"] == 5
    assert policy._tolbert_rollout_policy()["stop_completion_weight"] == 9.5
    assert policy._tolbert_hybrid_runtime()["scoring_policy"]["decoder_logprob_weight"] == 0.22


def test_llm_policy_raises_primary_min_command_score_from_retrieval_overrides(tmp_path):
    artifact_path = tmp_path / "tolbert_model" / "artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "runtime_policy": {
                    "primary_benchmark_families": ["workflow"],
                    "min_path_confidence": 0.7,
                    "require_trusted_retrieval": False,
                    "fallback_to_vllm_on_low_confidence": False,
                    "primary_min_command_score": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "overrides": {
                    "tolbert_deterministic_command_confidence": 0.88,
                    "tolbert_direct_command_min_score": 4,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FakeContextProvider(),
        config=KernelConfig(
            retrieval_proposals_path=retrieval_path,
            tolbert_model_artifact_path=artifact_path,
            use_tolbert_model_artifacts=True,
        ),
    )

    assert policy._tolbert_runtime_policy()["min_path_confidence"] == 0.88
    assert policy._tolbert_runtime_policy()["require_trusted_retrieval"] is True
    assert policy._tolbert_runtime_policy()["fallback_to_vllm_on_low_confidence"] is True
    assert policy._tolbert_runtime_policy()["primary_min_command_score"] == 4


def test_llm_policy_screens_partial_first_step_retrieval_guidance_for_tolbert_candidates():
    policy = LLMDecisionPolicy(MockLLMClient())
    state = AgentState(task=TaskBank().get("api_contract_task"))
    state.context_packet = ContextPacket(
        request_id="req-1",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "path_confidence": 0.95,
            "trust_retrieval": True,
            "retrieval_guidance": {
                "recommended_commands": [
                    "mkdir -p api && printf '{\"route\": \"/health\", \"method\": \"GET\"}\\n' > api/request.json"
                ],
                "recommended_command_spans": [
                    {
                        "span_id": "task:api:partial",
                        "command": "mkdir -p api && printf '{\"route\": \"/health\", \"method\": \"GET\"}\\n' > api/request.json",
                    }
                ],
                "avoidance_notes": [],
                "evidence": ["task:api:partial: retrieved partial command"],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={"branch_scoped": [{"metadata": {"task_id": "api_contract_task"}}], "fallback_scoped": [], "global": []},
        verifier_contract={"success_command": "true"},
    )

    screened = policy._screened_tolbert_retrieval_guidance(
        state,
        top_skill=None,
        retrieval_guidance=state.context_packet.control["retrieval_guidance"],
        blocked_commands=[],
    )

    assert screened["recommended_commands"] == []
    assert screened["recommended_command_spans"] == []


def test_long_horizon_rollout_action_value_prefers_reversible_progress():
    class FakeWorldModel:
        def simulate_command_effect(self, world_model_summary, content):
            del world_model_summary, content
            return {
                "predicted_progress_gain": 1.0,
                "predicted_conflicts": [],
                "predicted_preserved": ["keep.txt"],
                "predicted_workflow_paths": [],
            }

    rollout_policy = {
        "predicted_progress_gain_weight": 3.0,
        "predicted_conflict_penalty_weight": 4.0,
        "predicted_preserved_bonus_weight": 1.0,
        "predicted_workflow_bonus_weight": 1.5,
        "latent_progress_bonus_weight": 1.0,
        "latent_risk_penalty_weight": 2.0,
        "recover_from_stall_bonus_weight": 1.5,
        "long_horizon_progress_bonus_weight": 2.0,
        "long_horizon_preserved_bonus_weight": 1.5,
        "long_horizon_risk_penalty_weight": 3.0,
    }

    bounded = rollout_action_value(
        world_model_summary={"horizon": "bounded"},
        latent_state_summary={"progress_band": "flat", "risk_band": "stable"},
        latest_transition={},
        action="code_execute",
        content="printf 'hello\\n' > hello.txt",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )
    long_horizon = rollout_action_value(
        world_model_summary={"horizon": "long_horizon"},
        latent_state_summary={"progress_band": "flat", "risk_band": "stable"},
        latest_transition={},
        action="code_execute",
        content="printf 'hello\\n' > hello.txt",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )

    assert long_horizon > bounded


def test_long_horizon_rollout_action_value_uses_learned_world_signal_for_recovery():
    class FakeWorldModel:
        def simulate_command_effect(self, world_model_summary, content):
            del world_model_summary
            if "recover" in content:
                return {
                    "predicted_progress_gain": 1.0,
                    "predicted_conflicts": [],
                    "predicted_preserved": ["keep.txt"],
                    "predicted_workflow_paths": [],
                }
            return {
                "predicted_progress_gain": 0.0,
                "predicted_conflicts": [],
                "predicted_preserved": [],
                "predicted_workflow_paths": [],
            }

    rollout_policy = {
        "predicted_progress_gain_weight": 3.0,
        "predicted_conflict_penalty_weight": 4.0,
        "predicted_preserved_bonus_weight": 1.0,
        "predicted_workflow_bonus_weight": 1.5,
        "latent_progress_bonus_weight": 1.0,
        "latent_risk_penalty_weight": 2.0,
        "learned_world_progress_bonus_weight": 1.25,
        "learned_world_recovery_bonus_weight": 1.5,
        "learned_world_continue_penalty_weight": 1.25,
        "recover_from_stall_bonus_weight": 1.5,
        "long_horizon_progress_bonus_weight": 2.0,
        "long_horizon_preserved_bonus_weight": 1.5,
        "long_horizon_risk_penalty_weight": 3.0,
    }
    latent_state_summary = {
        "progress_band": "flat",
        "risk_band": "stable",
        "learned_world_state": {
            "progress_signal": 0.82,
            "risk_signal": 0.9,
            "world_progress_score": 0.79,
            "world_risk_score": 0.88,
        },
    }
    latest_transition = {"no_progress": True}

    recover_value = rollout_action_value(
        world_model_summary={"horizon": "long_horizon"},
        latent_state_summary=latent_state_summary,
        latest_transition=latest_transition,
        action="code_execute",
        content="recover state",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )
    stall_value = rollout_action_value(
        world_model_summary={"horizon": "long_horizon"},
        latent_state_summary=latent_state_summary,
        latest_transition=latest_transition,
        action="code_execute",
        content="repeat stale command",
        rollout_policy=rollout_policy,
        world_model=FakeWorldModel(),
    )

    assert recover_value > stall_value


def test_llm_policy_blocks_first_step_partial_skill_for_multi_artifact_task():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:api_contract_task:primary",
                "kind": "command_sequence",
                "source_task_id": "api_contract_task",
                "applicable_tasks": ["api_contract_task"],
                "procedure": {
                    "commands": [
                        "mkdir -p api && printf '{\"route\": \"/health\", \"method\": \"GET\"}\\n' > api/request.json"
                    ]
                },
                "quality": 1.0,
            }
        ]
    )

    client = CapturingClient()
    decision = LLMDecisionPolicy(
        client,
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("api_contract_task")))

    assert decision.selected_skill_id is None
    assert decision.action == "code_execute"
    assert "hello.txt" in decision.content or "responses/expected.json" in decision.content or "api/request.json" in decision.content


def test_llm_policy_allows_first_step_skill_when_only_remaining_artifact_is_missing():
    skill_library = SkillLibrary(
        [
            {
                "skill_id": "skill:config_sync_task:primary",
                "kind": "command_sequence",
                "source_task_id": "config_sync_task",
                "applicable_tasks": ["config_sync_task"],
                "procedure": {
                    "commands": ["mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env"]
                },
                "quality": 1.0,
            }
        ]
    )

    decision = LLMDecisionPolicy(
        MockLLMClient(),
        skill_library=skill_library,
    ).decide(AgentState(task=TaskBank().get("config_sync_task")))

    assert decision.selected_skill_id == "skill:config_sync_task:primary"
    assert decision.content == "mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env"
