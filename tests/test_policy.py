from agent_kernel.llm import MockLLMClient
from agent_kernel.config import KernelConfig
from agent_kernel.modeling.world.latent_state import build_latent_state_summary
from agent_kernel.policy import LLMDecisionPolicy, SkillLibrary
from agent_kernel.schemas import ContextPacket, StepRecord, TaskSpec
from agent_kernel.state import AgentState
from agent_kernel.task_bank import TaskBank
import json


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
        "payload_build",
        "llm_request",
        "llm_response",
    ]


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


def test_llm_policy_compacts_graph_summary_legacy_keys():
    client = CapturingClient()
    policy = LLMDecisionPolicy(client)
    state = AgentState(task=TaskBank().get("hello_task"))
    state.graph_summary = {
        "document_count": 3,
        "benchmark_families": {"micro": 1},
        "failure_type_counts": {"command_failure": 2},
        "neighbors": ["hello_task_followup"],
    }

    policy.decide(state)

    assert client.last_payload["graph_summary"]["failure_types"] == {"command_failure": 2}
    assert client.last_payload["graph_summary"]["related_tasks"] == ["hello_task_followup"]


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
