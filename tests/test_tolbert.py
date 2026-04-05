from __future__ import annotations

import json
from pathlib import Path
import time
from types import MethodType

import pytest

from agent_kernel.config import KernelConfig
from agent_kernel.llm import MockLLMClient
from agent_kernel.policy import LLMDecisionPolicy
from agent_kernel.state import AgentState
from agent_kernel.task_bank import TaskBank
from agent_kernel.tolbert import (
    TolbertContextCompiler,
    TolbertServiceClient,
    _discover_v2_paper_research_runtime_paths,
    _paper_research_runtime_paths,
)
from agent_kernel.schemas import ContextPacket, StepRecord, TaskSpec


class FakeTolbertClient:
    def query(
        self,
        *,
        query_text,
        timeout_seconds=None,
        branch_results,
        global_results,
        confidence_threshold,
        top_branches,
        branch_confidence_margin,
        low_confidence_widen_threshold,
        ancestor_branch_levels,
        low_confidence_branch_multiplier,
        low_confidence_global_multiplier,
    ):
        del (
            query_text,
            timeout_seconds,
            branch_results,
            global_results,
            confidence_threshold,
            top_branches,
            branch_confidence_margin,
            low_confidence_widen_threshold,
            ancestor_branch_levels,
            low_confidence_branch_multiplier,
            low_confidence_global_multiplier,
        )
        return {
            "backend": "tolbert_brain_service",
            "index_shards": ["joint-paper-cache"],
            "level_focus": "repo",
            "selected_branch_level": 2,
            "branch_candidates": [
                {"level": 2, "local_id": 0, "confidence": 0.8, "label": "Tasks", "node_id": 20},
                {"level": 2, "local_id": 1, "confidence": 0.75, "label": "Docs", "node_id": 21},
            ],
            "path_prediction": {
                "tree_version": "tol_v1",
                "decode_mode": "greedy_hierarchical_decode",
                "levels": [1, 2],
                "predicted_level_ids": {"1": 0, "2": 0},
                "confidence_by_level": {"1": 0.9, "2": 0.8},
                "labels_by_level": {"1": "Tooling", "2": "Tasks"},
                "fallbacks": [{"level": "2", "local_id": "1", "label": "Docs", "confidence": "0.7500"}],
            },
            "retrieval": {
                "branch_scoped": [
                    {
                        "span_id": "task:hello:suggested:1",
                        "text": "printf 'hello agent kernel\\n' > hello.txt",
                        "source_id": "hello",
                        "span_type": "agent:command_template",
                        "score": 1.2,
                        "node_path": [0, 0, 0],
                        "metadata": {"span_type": "agent:command_template", "task_id": "hello_task"},
                    },
                    {
                        "span_id": "task:status:primary:suggested:1",
                        "text": "printf 'diagnostics green stable\\n' > phrase.txt",
                        "source_id": "status",
                        "span_type": "agent:command_template",
                        "score": 1.1,
                        "node_path": [0, 0, 0],
                        "metadata": {
                            "span_type": "agent:command_template",
                            "task_id": "status_phrase_seed_task",
                        },
                    },
                    {
                        "span_id": "task:status:fallback:suggested:1",
                        "text": "printf 'diagnostics amber fallback\\n' > phrase.txt",
                        "source_id": "status",
                        "span_type": "agent:command_template",
                        "score": 1.3,
                        "node_path": [0, 0, 0],
                        "metadata": {
                            "span_type": "agent:command_template",
                            "task_id": "status_phrase_fallback_seed_task",
                        },
                    },
                    {
                        "span_id": "task:hello",
                        "text": "hello agent kernel task span",
                        "source_id": "hello",
                        "span_type": "agent:task",
                        "score": 1.0,
                        "node_path": [0, 0, 0],
                        "metadata": {"span_type": "agent:task"},
                    }
                ],
                "fallback_scoped": [
                    {
                        "span_id": "doc:runtime:1",
                        "text": "runtime documentation span",
                        "source_id": "runtime",
                        "span_type": "doc:readme_chunk",
                        "score": 0.95,
                        "node_path": [0, 0, 1],
                        "metadata": {"path": "README.md"},
                    }
                ],
                "global": [
                    {
                        "span_id": "episode:hello_task:step:1",
                        "text": "\n".join(
                            [
                                "task_id: hello_task",
                                "step_index: 1",
                                "action: code_execute",
                                "content: bad command",
                                "verification_passed: False",
                                "verification_reasons: exit code was 1",
                            ]
                        ),
                        "source_id": "hello",
                        "span_type": "agent:episode_step",
                        "score": 0.9,
                        "node_path": [0, 0, 0],
                        "metadata": {"span_type": "agent:episode_step", "task_id": "hello_task"},
                    }
                ],
            },
        }


class CapturingTolbertClient(FakeTolbertClient):
    def __init__(self) -> None:
        self.last_request: dict[str, object] | None = None

    def query(self, **kwargs):
        self.last_request = dict(kwargs)
        return super().query(**kwargs)


class SparseTolbertClient:
    def __init__(self) -> None:
        self.last_request: dict[str, object] | None = None

    def query(
        self,
        *,
        query_text,
        timeout_seconds=None,
        branch_results,
        global_results,
        confidence_threshold,
        top_branches,
        branch_confidence_margin,
        low_confidence_widen_threshold,
        ancestor_branch_levels,
        low_confidence_branch_multiplier,
        low_confidence_global_multiplier,
    ):
        del (
            timeout_seconds,
            branch_results,
            global_results,
            confidence_threshold,
            top_branches,
            branch_confidence_margin,
            low_confidence_widen_threshold,
            ancestor_branch_levels,
            low_confidence_branch_multiplier,
            low_confidence_global_multiplier,
        )
        self.last_request = {"query_text": query_text}
        return {
            "backend": "tolbert_brain_service",
            "index_shards": ["joint-paper-cache"],
            "level_focus": "repo",
            "selected_branch_level": 2,
            "branch_candidates": [],
            "path_prediction": {
                "tree_version": "tol_v1",
                "decode_mode": "greedy_hierarchical_decode",
                "levels": [1, 2],
                "predicted_level_ids": {"1": 0, "2": 0},
                "confidence_by_level": {"1": 0.85, "2": 0.76},
                "labels_by_level": {"1": "Tooling", "2": "Tasks"},
                "fallbacks": [],
            },
            "retrieval": {
                "branch_scoped": [
                    {
                        "span_id": "task:unrelated:suggested:1",
                        "text": "printf 'unrelated\\n' > unrelated.txt",
                        "source_id": "unrelated",
                        "span_type": "agent:command_template",
                        "score": 0.8,
                        "node_path": [0, 0, 0],
                        "metadata": {"span_type": "agent:command_template", "task_id": "unrelated_task"},
                    }
                ],
                "fallback_scoped": [],
                "global": [],
            },
        }


class FakeResearchTolbertClient:
    def __init__(self) -> None:
        self.calls = 0

    def query(
        self,
        *,
        query_text,
        timeout_seconds=None,
        branch_results,
        global_results,
        confidence_threshold,
        top_branches,
        branch_confidence_margin,
        low_confidence_widen_threshold,
        ancestor_branch_levels,
        low_confidence_branch_multiplier,
        low_confidence_global_multiplier,
    ):
        del (
            query_text,
            timeout_seconds,
            branch_results,
            global_results,
            confidence_threshold,
            top_branches,
            branch_confidence_margin,
            low_confidence_widen_threshold,
            ancestor_branch_levels,
            low_confidence_branch_multiplier,
            low_confidence_global_multiplier,
        )
        self.calls += 1
        return {
            "backend": "tolbert_brain_service",
            "index_shards": ["paper_spans_joint_mapped__tolbert_epoch3.pt"],
            "level_focus": "repo",
            "selected_branch_level": 2,
            "branch_candidates": [],
            "path_prediction": {
                "tree_version": "tol_v1",
                "decode_mode": "greedy_hierarchical_decode",
                "levels": [1, 2],
                "predicted_level_ids": {"1": 1, "2": 0},
                "confidence_by_level": {"1": 0.91, "2": 0.78},
                "labels_by_level": {"1": "Papers", "2": "MachineLearningTheory"},
                "fallbacks": [],
            },
            "retrieval": {
                "branch_scoped": [],
                "fallback_scoped": [],
                "global": [
                    {
                        "span_id": "paper:1",
                        "text": "Transformer scaling laws suggest larger models improve benchmark quality.",
                        "source_id": "/arxiv/pdfs/2401/2401.00001.pdf",
                        "span_type": "doc:paper_paragraph",
                        "score": 0.72,
                        "node_path": [0, 1, 3, 9, 20],
                        "metadata": {
                            "title": "Scaling Laws for Useful Models",
                            "pdf_path": "/arxiv/pdfs/2401/2401.00001.pdf",
                            "retrieval_source": "paper_research",
                        },
                    }
                ],
            },
        }


class DelayedTolbertClient(FakeTolbertClient):
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request: dict[str, object] | None = None

    def query(self, **kwargs):
        self.last_request = dict(kwargs)
        time.sleep(self.delay_seconds)
        return super().query(**{key: value for key, value in kwargs.items() if key != "timeout_seconds"})


def _monotonic_sequence(values: list[float]):
    iterator = iter(values)
    last_value = values[-1]

    def _next() -> float:
        nonlocal last_value
        try:
            last_value = next(iterator)
        except StopIteration:
            pass
        return last_value

    return _next


def test_tolbert_compiler_uses_real_decode_contract(tmp_path: Path) -> None:
    nodes_path = tmp_path / "nodes.jsonl"
    spans_path = tmp_path / "spans.jsonl"
    label_map_path = tmp_path / "label_map.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"stub")

    nodes = [
        {"node_id": 0, "level": 0, "parent_id": None, "name": "root", "attributes": {}},
        {"node_id": 10, "level": 1, "parent_id": 0, "name": "Tooling", "attributes": {}},
        {"node_id": 20, "level": 2, "parent_id": 10, "name": "Tasks", "attributes": {}},
        {"node_id": 21, "level": 2, "parent_id": 10, "name": "Docs", "attributes": {}},
    ]
    nodes_path.write_text("".join(json.dumps(row) + "\n" for row in nodes), encoding="utf-8")
    label_map_path.write_text(json.dumps({"1": {"0": 10}, "2": {"0": 20, "1": 21}}), encoding="utf-8")

    spans = [
        {
            "span_id": "task:hello",
            "text": "hello agent kernel task span",
            "source_id": "hello",
            "node_path": [0, 0, 0],
            "meta": {"span_type": "agent:task"},
        },
        {
            "span_id": "doc:runtime",
            "text": "runtime documentation span",
            "source_id": "runtime",
            "node_path": [0, 0, 1],
            "meta": {"span_type": "doc:readme_chunk"},
        },
    ]
    spans_path.write_text("".join(json.dumps(row) + "\n" for row in spans), encoding="utf-8")

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_branch_results=1,
        tolbert_global_results=2,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert packet.tolbert["backend"] == "tolbert_brain_service"
    assert packet.tolbert["path_prediction"]["decode_mode"] == "greedy_hierarchical_decode"
    assert packet.tolbert["path_prediction"]["labels_by_level"]["2"] == "Tasks"
    assert packet.retrieval["branch_scoped"][0]["span_id"] == "task:hello:suggested:1"
    assert packet.retrieval["fallback_scoped"][0]["span_id"] == "doc:runtime:1"
    assert packet.control["selected_branch_level"] == 2
    assert packet.control["trust_retrieval"] is True
    assert packet.control["retrieval_guidance"]["recommended_commands"][0] == "printf 'hello agent kernel\\n' > hello.txt"
    assert packet.control["retrieval_guidance"]["recommended_command_spans"][0]["span_id"] == "task:hello:suggested:1"
    assert "avoid repeating 'bad command'" in packet.control["retrieval_guidance"]["avoidance_notes"][0]


def test_tolbert_compiler_penalizes_declared_distractor_tasks(tmp_path: Path) -> None:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=tmp_path / "learning" / "run_learning_artifacts.json",
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("status_phrase_retrieval_task")))

    assert packet.retrieval["branch_scoped"][0]["metadata"]["task_id"] == "status_phrase_seed_task"
    assert packet.control["retrieval_guidance"]["recommended_commands"][0] == "printf 'diagnostics green stable\\n' > phrase.txt"


def test_tolbert_compiler_enriches_guidance_with_learning_candidates(tmp_path: Path) -> None:
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:hello_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "hello_task",
                        "benchmark_family": "bounded",
                        "procedure": {"commands": ["printf 'hello learned\\n' > hello.txt"]},
                        "applicable_tasks": ["hello_task"],
                        "support_count": 2,
                    },
                    {
                        "candidate_id": "learning:negative_command:hello_task:false",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "hello_task",
                        "benchmark_family": "bounded",
                        "command": "false",
                        "verification_reasons": ["exit code was 1"],
                        "support_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert "printf 'hello learned\\n' > hello.txt" in packet.control["retrieval_guidance"]["recommended_commands"]
    assert any("avoid repeating 'false'" in note for note in packet.control["retrieval_guidance"]["avoidance_notes"])
    assert any("learned success_skill_candidate" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_tolbert_compiler_surfaces_memory_source_on_learning_guidance(tmp_path: Path) -> None:
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:episode-memory:seed",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_task",
                        "benchmark_family": "episode_memory",
                        "memory_source": "episode",
                        "memory_sources": ["episode"],
                        "procedure": {"commands": ["printf 'episode memory\\n' > memory.txt"]},
                        "support_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )
    task = TaskSpec(
        task_id="episode_followup_task",
        prompt="Reuse episode memory to create memory.txt.",
        workspace_subdir="episode_followup_task",
        success_command="test -f memory.txt",
        max_steps=3,
        metadata={"benchmark_family": "episode_memory", "memory_source": "episode"},
    )

    packet = compiler.compile(AgentState(task=task))

    assert "printf 'episode memory\\n' > memory.txt" in packet.control["retrieval_guidance"]["recommended_commands"]
    assert any("via episode memory" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_tolbert_compiler_prefers_trusted_retrieval_backed_learning_guidance(tmp_path: Path) -> None:
    class EmptyRetrievalClient(FakeTolbertClient):
        def query(self, **kwargs):
            payload = super().query(**kwargs)
            payload["retrieval"] = {"branch_scoped": [], "fallback_scoped": [], "global": []}
            return payload

    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:generic-family",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "other_workflow_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'generic memory\\n' > result.txt"]},
                        "support_count": 6,
                    },
                    {
                        "candidate_id": "learning:trusted-retrieval",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_workflow_task",
                        "benchmark_family": "workflow",
                        "procedure": {"commands": ["printf 'retrieval memory\\n' > result.txt"]},
                        "retrieval_backed": True,
                        "retrieval_selected_steps": 1,
                        "retrieval_influenced_steps": 1,
                        "trusted_retrieval_steps": 1,
                        "retrieval_backed_commands": ["printf 'retrieval memory\\n' > result.txt"],
                        "support_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        use_tolbert_context=True,
        tolbert_branch_results=0,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=EmptyRetrievalClient(),
    )
    task = TaskSpec(
        task_id="target_workflow_task",
        prompt="Use learned workflow memory to create result.txt.",
        workspace_subdir="target_workflow_task",
        success_command="test -f result.txt",
        max_steps=3,
        metadata={"benchmark_family": "workflow"},
    )

    packet = compiler.compile(AgentState(task=task))

    assert packet.control["retrieval_guidance"]["recommended_commands"][0] == "printf 'retrieval memory\\n' > result.txt"
    assert any(
        "trusted retrieval-backed success_skill_candidate" in item
        for item in packet.control["retrieval_guidance"]["evidence"]
    )


def test_tolbert_compiler_uses_applicable_transfer_learning_candidates(tmp_path: Path) -> None:
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:transfer-applicable",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "seed_task",
                        "benchmark_family": "workflow",
                        "applicable_tasks": ["target_task"],
                        "procedure": {"commands": ["printf 'transfer memory\\n' > target.txt"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )
    task = TaskSpec(
        task_id="target_task",
        prompt="Use learned transfer to create target.txt.",
        workspace_subdir="target_task",
        success_command="test -f target.txt",
        max_steps=3,
        metadata={"benchmark_family": "workflow"},
    )

    packet = compiler.compile(AgentState(task=task))

    assert "printf 'transfer memory\\n' > target.txt" in packet.control["retrieval_guidance"]["recommended_commands"]
    assert any("learning:transfer-applicable" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_tolbert_compiler_limits_spans_per_source(tmp_path: Path) -> None:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=3,
        tolbert_global_results=0,
        tolbert_max_spans_per_source=1,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    source_ids = [item["source_id"] for item in packet.retrieval["branch_scoped"]]
    assert len(source_ids) == len(set(source_ids))


def test_tolbert_compiler_applies_retained_retrieval_overrides(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval" / "retrieval_proposals.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "retained",
                "overrides": {
                    "tolbert_branch_results": 1,
                    "tolbert_global_results": 1,
                    "tolbert_confidence_threshold": 0.4,
                    "tolbert_context_max_chunks": 1,
                    "tolbert_max_spans_per_source": 1,
                    "tolbert_distractor_penalty": 7.0,
                },
            }
        ),
        encoding="utf-8",
    )
    client = CapturingTolbertClient()
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        retrieval_proposals_path=retrieval_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=client,
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert client.last_request is not None
    assert client.last_request["branch_results"] == 1
    assert client.last_request["global_results"] == 1
    assert client.last_request["confidence_threshold"] == 0.4
    assert len(packet.control["selected_context_chunks"]) == 1


def test_tolbert_compiler_carries_selected_retrieval_into_next_guidance(tmp_path: Path) -> None:
    compiler = TolbertContextCompiler(
        config=KernelConfig(provider="mock", use_tolbert_context=True),
        repo_root=tmp_path,
        client=SparseTolbertClient(),
    )
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-carry",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "retrieval_guidance": {
                "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                "recommended_command_spans": [
                    {
                        "span_id": "carry:hello:selected",
                        "command": "printf 'hello agent kernel\\n' > hello.txt",
                    }
                ],
                "avoidance_notes": [],
                "evidence": ["carry:hello:selected: template command"],
            },
        },
        tolbert={"path_prediction": {"tree_version": "tol_v1"}},
        retrieval={
            "branch_scoped": [
                {
                    "span_id": "carry:hello:selected",
                    "text": "printf 'hello agent kernel\\n' > hello.txt",
                    "source_id": "hello_task",
                    "span_type": "agent:command_template",
                    "score": 0.0,
                    "node_path": [0, 0, 0],
                    "metadata": {"task_id": "hello_task", "span_type": "agent:command_template"},
                }
            ],
            "fallback_scoped": [],
            "global": [],
        },
        verifier_contract={"success_command": "true"},
    )
    state.history.append(
        StepRecord(
            index=1,
            thought="look around",
            action="code_execute",
            content="grep -n hello README.md",
            selected_skill_id=None,
            command_result={"command": "grep -n hello README.md", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["missing expected file"]},
            retrieval_command_match=False,
            selected_retrieval_span_id="carry:hello:selected",
            retrieval_influenced=False,
            trust_retrieval=False,
        )
    )

    packet = compiler.compile(state)

    assert "printf 'hello agent kernel\\n' > hello.txt" in packet.control["retrieval_guidance"]["recommended_commands"]
    assert any("carried retrieval guidance" in item for item in packet.control["retrieval_guidance"]["evidence"])
    assert any(chunk["span_id"] == "carry:hello:selected" for chunk in packet.control["selected_context_chunks"])


def test_tolbert_compiler_adds_retrieval_carry_to_query_text(tmp_path: Path) -> None:
    client = SparseTolbertClient()
    compiler = TolbertContextCompiler(
        config=KernelConfig(provider="mock", use_tolbert_context=True),
        repo_root=tmp_path,
        client=client,
    )
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-carry",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "retrieval_guidance": {
                "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                "recommended_command_spans": [
                    {
                        "span_id": "carry:hello:selected",
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
    state.history.append(
        StepRecord(
            index=1,
            thought="read current state",
            action="code_execute",
            content="grep -n hello README.md",
            selected_skill_id=None,
            command_result={"command": "grep -n hello README.md", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["missing expected file"]},
            retrieval_command_match=False,
            selected_retrieval_span_id="carry:hello:selected",
        )
    )

    compiler.compile(state)

    assert client.last_request is not None
    assert "retrieval carry step=1 command=printf 'hello agent kernel\\n' > hello.txt" in client.last_request["query_text"]


def test_tolbert_compiler_does_not_carry_failed_matched_retrieval_command(tmp_path: Path) -> None:
    client = SparseTolbertClient()
    compiler = TolbertContextCompiler(
        config=KernelConfig(provider="mock", use_tolbert_context=True),
        repo_root=tmp_path,
        client=client,
    )
    state = AgentState(task=TaskBank().get("hello_task"))
    state.context_packet = ContextPacket(
        request_id="req-carry",
        created_at="2026-04-02T00:00:00+00:00",
        task={"goal": "g", "completion_criteria": "c"},
        control={
            "mode": "verify",
            "retrieval_guidance": {
                "recommended_commands": ["printf 'hello agent kernel\\n' > hello.txt"],
                "recommended_command_spans": [
                    {
                        "span_id": "carry:hello:selected",
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
    state.history.append(
        StepRecord(
            index=1,
            thought="try retrieved command",
            action="code_execute",
            content="printf 'hello agent kernel\\n' > hello.txt",
            selected_skill_id=None,
            command_result={"command": "printf 'hello agent kernel\\n' > hello.txt", "exit_code": 1, "stdout": "", "stderr": "", "timed_out": False},
            verification={"passed": False, "reasons": ["exit code was 1"]},
            retrieval_command_match=True,
            selected_retrieval_span_id="carry:hello:selected",
            retrieval_influenced=True,
            trust_retrieval=True,
        )
    )

    packet = compiler.compile(state)

    assert "retrieval carry step=1 command=printf 'hello agent kernel\\n' > hello.txt" not in client.last_request["query_text"]
    assert not any(
        entry.get("span_id") == "carry:hello:selected"
        for entry in packet.control["retrieval_guidance"]["recommended_command_spans"]
    )


def test_tolbert_compiler_applies_retained_world_model_controls_to_retrieval_ranking(tmp_path: Path) -> None:
    world_model_path = tmp_path / "world_model" / "world_model_proposals.json"
    world_model_path.parent.mkdir(parents=True, exist_ok=True)
    world_model_path.write_text(
        json.dumps(
            {
                "artifact_kind": "world_model_policy_set",
                "lifecycle_state": "retained",
                "controls": {
                    "retrieved_preserved_artifact_score_weight": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    class WorldModelRankingClient(FakeTolbertClient):
        def query(self, **kwargs):
            del kwargs
            return {
                "backend": "tolbert_brain_service",
                "index_shards": ["joint-paper-cache"],
                "level_focus": "skill",
                "selected_branch_level": 2,
                "branch_candidates": [],
                "path_prediction": {
                    "tree_version": "tol_v1",
                    "decode_mode": "greedy_hierarchical_decode",
                    "levels": [1, 2],
                    "predicted_level_ids": {"1": 0, "2": 0},
                    "confidence_by_level": {"1": 0.9, "2": 0.8},
                    "labels_by_level": {"1": "Tooling", "2": "Tasks"},
                    "fallbacks": [],
                },
                "retrieval": {
                    "branch_scoped": [],
                    "fallback_scoped": [],
                    "global": [
                        {
                            "span_id": "global:generic",
                            "text": "general guidance",
                            "source_id": "generic",
                            "span_type": "agent:task",
                            "score": 3.0,
                            "node_path": [0, 0, 0],
                            "metadata": {"task_id": "hello_task"},
                        },
                        {
                            "span_id": "global:preserved",
                            "text": "leave README.md untouched while updating hello.txt",
                            "source_id": "preserved",
                            "span_type": "agent:task",
                            "score": 0.0,
                            "node_path": [0, 0, 1],
                            "metadata": {"task_id": "hello_task"},
                        },
                    ],
                },
            }

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        world_model_proposals_path=world_model_path,
        tolbert_branch_results=0,
        tolbert_global_results=2,
        tolbert_context_max_chunks=1,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=WorldModelRankingClient(),
    )
    state = AgentState(
        task=TaskBank().get("hello_task"),
        world_model_summary={"preserved_artifacts": ["README.md"]},
    )

    packet = compiler.compile(state)

    assert packet.control["selected_context_chunks"][0]["span_id"] == "global:preserved"


def test_tolbert_service_client_prefers_retained_bundle_runtime_paths(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bundle" / "config.json"
    checkpoint_path = tmp_path / "bundle" / "checkpoint.pt"
    nodes_path = tmp_path / "bundle" / "nodes.jsonl"
    label_map_path = tmp_path / "bundle" / "label_map.json"
    spans_path = tmp_path / "bundle" / "spans.jsonl"
    cache_path = tmp_path / "bundle" / "cache.pt"
    bundle_manifest_path = tmp_path / "retrieval" / "retrieval_asset_bundle.json"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    cache_path.write_bytes(b"stub")
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_retrieval_asset_bundle",
                "lifecycle_state": "retained",
                "runtime_paths": {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "nodes_path": str(nodes_path),
                    "label_map_path": str(label_map_path),
                    "source_spans_paths": [str(spans_path)],
                    "cache_paths": [str(cache_path)],
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def poll(self):
            return None

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", _fake_popen)

    config = KernelConfig(
        provider="mock",
        retrieval_asset_bundle_path=bundle_manifest_path,
        tolbert_config_path="missing/config.json",
        tolbert_checkpoint_path="missing/checkpoint.pt",
        tolbert_nodes_path="missing/nodes.jsonl",
        tolbert_label_map_path="missing/label_map.json",
        tolbert_source_spans_paths=("missing/source_spans.jsonl",),
        tolbert_cache_paths=("missing/cache.pt",),
    )
    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client._ensure_process()

    args = captured["args"]
    assert str(config_path) in args
    assert str(checkpoint_path) in args
    assert str(nodes_path) in args
    assert str(label_map_path) in args
    assert str(spans_path) in args
    assert str(cache_path) in args


def test_tolbert_compiler_globally_ranks_context_chunks_across_buckets(tmp_path: Path) -> None:
    class GlobalBetterClient(FakeTolbertClient):
        def query(self, **kwargs):
            payload = super().query(**kwargs)
            payload["retrieval"] = {
                "branch_scoped": [
                    {
                        "span_id": "branch:weak",
                        "text": "general branch note",
                        "source_id": "branch",
                        "span_type": "agent:task",
                        "score": 0.2,
                        "node_path": [0, 0, 0],
                        "metadata": {"task_id": "other_task"},
                    }
                ],
                "fallback_scoped": [],
                "global": [
                    {
                        "span_id": "global:strong",
                        "text": "printf 'hello agent kernel\\n' > hello.txt",
                        "source_id": "global",
                        "span_type": "agent:command_template",
                        "score": 0.9,
                        "node_path": [0, 0, 1],
                        "metadata": {"task_id": "hello_task"},
                    }
                ],
            }
            return payload

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=1,
        tolbert_context_max_chunks=1,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=GlobalBetterClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert packet.control["selected_context_chunks"][0]["span_id"] == "global:strong"


def test_tolbert_compiler_promotes_procedure_spans_into_retrieval_guidance(tmp_path: Path) -> None:
    class ProcedureGuidanceClient(FakeTolbertClient):
        def query(self, **kwargs):
            payload = super().query(**kwargs)
            payload["retrieval"] = {
                "branch_scoped": [
                    {
                        "span_id": "tool:merge:procedure:1",
                        "text": "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                        "source_id": "git_parallel_merge_acceptance_task",
                        "span_type": "agent:procedure_span",
                        "score": 0.9,
                        "node_path": [0, 0, 0],
                        "metadata": {
                            "task_id": "git_parallel_merge_acceptance_task",
                            "benchmark_family": "repo_sandbox",
                            "capability": "repo_environment",
                            "touched_files": ["docs/status.md"],
                        },
                    }
                ],
                "fallback_scoped": [],
                "global": [
                    {
                        "span_id": "tool:merge:candidate",
                        "text": "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'",
                        "source_id": "git_parallel_merge_acceptance_task",
                        "span_type": "agent:tool_candidate",
                        "score": 0.8,
                        "node_path": [0, 0, 1],
                        "metadata": {"task_id": "git_parallel_merge_acceptance_task"},
                    }
                ],
            }
            return payload

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=1,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=ProcedureGuidanceClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("git_parallel_merge_acceptance_task")))

    guidance = packet.control["retrieval_guidance"]
    assert guidance["recommended_commands"][0] == "git merge --no-ff worker/docs-status -m 'merge worker/docs-status'"
    assert guidance["recommended_command_spans"][0]["span_id"] == "tool:merge:procedure:1"
    assert any("procedure guidance" in item for item in guidance["evidence"])
    assert any("reusable tool candidate" in item for item in guidance["evidence"])


def test_tolbert_compiler_ranks_artifact_aligned_procedure_spans_for_project_tasks(tmp_path: Path) -> None:
    class ProjectAlignedProcedureClient(FakeTolbertClient):
        def query(self, **kwargs):
            payload = super().query(**kwargs)
            payload["retrieval"] = {
                "branch_scoped": [
                    {
                        "span_id": "task:merge:overview",
                        "text": "merge worker branches and verify acceptance",
                        "source_id": "git_parallel_merge_acceptance_task",
                        "span_type": "agent:task",
                        "score": 3.0,
                        "node_path": [0, 0, 0],
                        "metadata": {"task_id": "git_parallel_merge_acceptance_task"},
                    },
                    {
                        "span_id": "tool:merge:procedure:accept",
                        "text": "printf 'api suite passed; docs suite passed\\n' > reports/test_report.txt",
                        "source_id": "git_parallel_merge_acceptance_task",
                        "span_type": "agent:procedure_span",
                        "score": 0.2,
                        "node_path": [0, 0, 1],
                        "metadata": {
                            "task_id": "git_parallel_merge_acceptance_task",
                            "benchmark_family": "repo_sandbox",
                            "capability": "repo_environment",
                            "touched_files": ["reports/test_report.txt"],
                        },
                    },
                ],
                "fallback_scoped": [],
                "global": [],
            }
            return payload

    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=2,
        tolbert_global_results=0,
        tolbert_context_max_chunks=1,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=ProjectAlignedProcedureClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("git_parallel_merge_acceptance_task")))

    assert packet.retrieval["branch_scoped"][0]["span_id"] == "tool:merge:procedure:accept"
    assert packet.control["selected_context_chunks"][0]["span_id"] == "tool:merge:procedure:accept"


def test_tolbert_compiler_trusts_workflow_guarded_learning_guidance_on_low_confidence_paths(
    tmp_path: Path,
) -> None:
    class LowConfidenceDocsClient(FakeTolbertClient):
        def query(self, **kwargs):
            payload = super().query(**kwargs)
            payload["path_prediction"]["confidence_by_level"] = {"1": 0.45, "2": 0.4}
            payload["retrieval"] = {
                "branch_scoped": [
                    {
                        "span_id": "doc:merge:guard",
                        "text": "merge workflow notes",
                        "source_id": "docs/runtime.md",
                        "span_type": "doc:readme_chunk",
                        "score": 0.4,
                        "node_path": [0, 0, 0],
                        "metadata": {"path": "docs/runtime.md"},
                    }
                ],
                "fallback_scoped": [],
                "global": [],
            }
            return payload

    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:success_skill:git_parallel_merge_acceptance_task",
                        "artifact_kind": "success_skill_candidate",
                        "source_task_id": "git_parallel_merge_acceptance_task",
                        "benchmark_family": "repo_sandbox",
                        "procedure": {
                            "commands": [
                                "git merge --no-ff worker/api-status -m 'merge worker/api-status' && git merge --no-ff worker/docs-status -m 'merge worker/docs-status' && tests/test_api.sh && tests/test_docs.sh && mkdir -p reports && printf 'accepted worker/api-status for src/api_status.txt and worker/docs-status for docs/status.md into main without collisions\\n' > reports/merge_report.txt && printf 'api suite passed; docs suite passed\\n' > reports/test_report.txt && git add reports/merge_report.txt reports/test_report.txt && git commit -m 'record merge acceptance reports'"
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=LowConfidenceDocsClient(),
    )

    packet = compiler.compile(AgentState(task=TaskBank().get("git_parallel_merge_acceptance_task")))

    assert packet.control["path_confidence"] == pytest.approx(0.4)
    assert packet.control["trust_retrieval"] is True
    assert any(
        span["span_id"].startswith("learning:success_skill:")
        for span in packet.control["retrieval_guidance"]["recommended_command_spans"]
    )


def test_tolbert_compiler_merges_auxiliary_paper_research_context(tmp_path: Path) -> None:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        paper_research_query_mode="always",
        tolbert_branch_results=1,
        tolbert_global_results=1,
        paper_research_global_results=2,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
        research_client=FakeResearchTolbertClient(),
    )
    task = TaskSpec(
        task_id="paper_research_task",
        prompt="Review papers on transformer scaling laws and summarize the best evidence.",
        workspace_subdir="workspace/paper_research_task",
    )

    packet = compiler.compile(AgentState(task=task))

    assert packet.control["paper_research_used"] is True
    assert any(item["span_id"] == "paper:1" for item in packet.retrieval["global"])
    assert any(
        item["span_id"] == "paper:1"
        for item in packet.control["selected_context_chunks"]
    )
    assert any("Scaling Laws for Useful Models" in item for item in packet.control["retrieval_guidance"]["evidence"])


def test_tolbert_compiler_emits_context_compile_subphases(tmp_path: Path) -> None:
    compile_budget_seconds = 10.0
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        tolbert_context_compile_budget_seconds=compile_budget_seconds,
    )
    client = CapturingTolbertClient()
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=client,
    )
    observed: list[dict[str, object]] = []
    compiler.set_progress_callback(lambda payload: observed.append(dict(payload)))

    packet = compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert packet.control["context_compile_elapsed_seconds"] >= 0.0
    assert float(client.last_request.get("timeout_seconds", 0.0)) == pytest.approx(0.8 * compile_budget_seconds)
    assert [payload.get("step_subphase", "") for payload in observed] == [
        "query_build",
        "tolbert_query",
        "retrieval_normalize",
        "skill_query",
        "tool_query",
        "skill_rank",
        "chunk_select",
        "guidance_build",
        "tool_plan",
        "complete",
    ]
    assert [float(payload.get("step_budget_seconds", 0.0)) for payload in observed] == [
        pytest.approx(0.5 * compile_budget_seconds),
        pytest.approx(0.8 * compile_budget_seconds),
        pytest.approx(0.6 * compile_budget_seconds),
        pytest.approx(0.6 * compile_budget_seconds),
        pytest.approx(0.5 * compile_budget_seconds),
        pytest.approx(0.4 * compile_budget_seconds),
        pytest.approx(0.5 * compile_budget_seconds),
        pytest.approx(0.5 * compile_budget_seconds),
        pytest.approx(0.4 * compile_budget_seconds),
        pytest.approx(0.5 * compile_budget_seconds),
    ]
    assert all(payload.get("step_stage") == "context_compile" for payload in observed)


def test_tolbert_compiler_uses_task_contract_guidance_for_adjacent_success(tmp_path: Path) -> None:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=tmp_path / "learning" / "run_learning_artifacts.json",
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(
        AgentState(
            task=TaskSpec(
                task_id="repo_adjacent",
                prompt="adjacent",
                workspace_subdir="repo_adjacent",
                suggested_commands=[
                    "mkdir -p repo && printf 'repo ready\\n' > repo/summary.txt",
                    "cat repo/summary.txt",
                ],
                metadata={
                    "benchmark_family": "repository",
                    "curriculum_kind": "adjacent_success",
                    "parent_task": "repo_sync_matrix_task",
                },
            )
        )
    )

    guidance = packet.control["retrieval_guidance"]
    assert guidance["recommended_commands"][:2] == [
        "mkdir -p repo && printf 'repo ready\\n' > repo/summary.txt",
        "cat repo/summary.txt",
    ]
    assert guidance["evidence"][:2] == [
        "task:repo_adjacent: adjacent success task contract",
        "task:repo_sync_matrix_task: successful parent episode",
    ]


def test_tolbert_compiler_preserves_adjacent_success_task_contract_while_merging_learning_guidance(
    tmp_path: Path,
) -> None:
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:recovery_case:repo_adjacent_parent",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "repo_adjacent_parent",
                        "benchmark_family": "repository",
                        "task_metadata": {"curriculum_kind": "failure_recovery"},
                        "recovery_commands": ["printf 'recovered\\n' > repo/recovery.txt"],
                        "applicable_tasks": ["repo_adjacent"],
                        "support_count": 2,
                    },
                    {
                        "candidate_id": "learning:negative_command:repo_adjacent:false",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "repo_adjacent_parent",
                        "benchmark_family": "repository",
                        "command": "rm -rf repo",
                        "verification_reasons": ["it deleted the expected repo state"],
                        "applicable_tasks": ["repo_adjacent"],
                        "support_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(
        AgentState(
            task=TaskSpec(
                task_id="repo_adjacent",
                prompt="adjacent",
                workspace_subdir="repo_adjacent",
                suggested_commands=[
                    "mkdir -p repo && printf 'repo ready\\n' > repo/summary.txt",
                    "cat repo/summary.txt",
                ],
                metadata={
                    "benchmark_family": "repository",
                    "curriculum_kind": "adjacent_success",
                    "parent_task": "repo_sync_matrix_task",
                },
            )
        )
    )

    guidance = packet.control["retrieval_guidance"]
    assert guidance["recommended_commands"][:2] == [
        "mkdir -p repo && printf 'repo ready\\n' > repo/summary.txt",
        "cat repo/summary.txt",
    ]
    assert "printf 'recovered\\n' > repo/recovery.txt" in guidance["recommended_commands"]
    assert any("avoid repeating 'rm -rf repo'" in note for note in guidance["avoidance_notes"])
    assert guidance["evidence"][:2] == [
        "task:repo_adjacent: adjacent success task contract",
        "task:repo_sync_matrix_task: successful parent episode",
    ]
    assert any("learned recovery_case" in item for item in guidance["evidence"])
    assert any("learned negative command pattern" in item for item in guidance["evidence"])


def test_tolbert_compiler_matches_learning_guidance_via_source_task_alias(tmp_path: Path) -> None:
    learning_path = tmp_path / "learning" / "run_learning_artifacts.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(
        json.dumps(
            {
                "artifact_kind": "run_learning_candidate_set",
                "lifecycle_state": "candidate",
                "candidates": [
                    {
                        "candidate_id": "learning:negative_command:config_sync_retrieval_task",
                        "artifact_kind": "negative_command_pattern",
                        "source_task_id": "config_sync_retrieval_task",
                        "benchmark_family": "workflow",
                        "command": "cp template.env config/app.env",
                        "verification_reasons": ["unexpected file content: config/app.env"],
                        "task_metadata": {
                            "source_task": "config_sync_task",
                        },
                    },
                    {
                        "candidate_id": "learning:recovery_case:config_sync_retrieval_task",
                        "artifact_kind": "recovery_case",
                        "source_task_id": "config_sync_retrieval_task",
                        "benchmark_family": "workflow",
                        "recovery_commands": ["mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env"],
                        "task_metadata": {
                            "curriculum_kind": "failure_recovery",
                            "source_task": "config_sync_task",
                        },
                        "success": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = KernelConfig(
        provider="mock",
        storage_backend="json",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        learning_artifacts_path=learning_path,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    packet = compiler.compile(
        AgentState(
            task=TaskSpec(
                task_id="fresh_config_repair_task",
                prompt="repair config",
                workspace_subdir="fresh_config_repair_task",
                suggested_commands=[
                    "mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env",
                ],
                metadata={
                    "benchmark_family": "workflow",
                    "curriculum_kind": "failure_recovery",
                    "source_task": "config_sync_task",
                },
            )
        )
    )

    guidance = packet.control["retrieval_guidance"]
    assert "mkdir -p config && printf 'MODE=prod\\nPORT=8080\\n' > config/app.env" in guidance["recommended_commands"]
    assert any("avoid repeating 'cp template.env config/app.env'" in note for note in guidance["avoidance_notes"])
    assert any("learned recovery_case" in item for item in guidance["evidence"])
    assert any("learned negative command pattern" in item for item in guidance["evidence"])


def test_tolbert_compiler_guidance_build_encompasses_retrieval_guidance(tmp_path: Path) -> None:
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=FakeTolbertClient(),
    )

    observed_payloads: list[dict[str, object]] = []
    guidance_phase: list[str] = []

    compiler.set_progress_callback(lambda payload: observed_payloads.append(dict(payload)))

    def fake_build_retrieval_guidance(self, retrieval: dict[str, list[dict[str, object]]], *, state):
        guidance_phase.append(str(observed_payloads[-1].get("step_subphase", "")))
        return {
            "recommended_commands": [],
            "recommended_command_spans": [],
            "avoidance_notes": [],
            "evidence": [],
        }

    compiler._build_retrieval_guidance = MethodType(fake_build_retrieval_guidance, compiler)  # type: ignore[attr-defined]
    compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert guidance_phase == ["guidance_build"]
    observed_subphases = [payload.get("step_subphase", "") for payload in observed_payloads]
    assert observed_subphases == [
        "query_build",
        "tolbert_query",
        "retrieval_normalize",
        "skill_query",
        "tool_query",
        "skill_rank",
        "chunk_select",
        "guidance_build",
        "tool_plan",
        "complete",
    ]


def test_tolbert_compiler_enforces_context_compile_budget(tmp_path: Path) -> None:
    client = DelayedTolbertClient(delay_seconds=0.02)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=True,
        tolbert_branch_results=1,
        tolbert_global_results=0,
        tolbert_context_compile_budget_seconds=0.001,
    )
    compiler = TolbertContextCompiler(
        config=config,
        repo_root=tmp_path,
        client=client,
    )

    with pytest.raises(RuntimeError, match="context compile budget .* exceeded during tolbert_query"):
        compiler.compile(AgentState(task=TaskBank().get("hello_task")))

    assert client.last_request is not None
    assert float(client.last_request["timeout_seconds"]) >= 0.0


def test_paper_research_runtime_paths_prefers_discovered_v2_bundle(monkeypatch, tmp_path: Path) -> None:
    config = KernelConfig(provider="mock")
    discovered = {
        "tolbert_config_path": "/data/TOLBERT_BRAIN/configs/tolbert_brain_joint_v2.yaml",
        "tolbert_checkpoint_path": "/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/tolbert_epoch1.pt",
        "tolbert_nodes_path": "/data/TOLBERT_BRAIN/data/joint_v2/nodes_joint_v2.jsonl",
        "tolbert_label_map_path": "/data/TOLBERT_BRAIN/data/joint_v2/label_map_joint_v2.json",
        "tolbert_source_spans_paths": (
            "/data/TOLBERT_BRAIN/data/joint_v2/code_spans_joint_v2_mapped.jsonl",
            "/data/TOLBERT_BRAIN/data/joint_v2/paper_spans_paragraphs_joint_v2_mapped.jsonl",
        ),
        "tolbert_cache_paths": (
            "/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2/retrieval_cache/cache.pt",
        ),
    }
    monkeypatch.setattr("agent_kernel.tolbert._discover_v2_paper_research_runtime_paths", lambda: discovered)

    runtime_paths = _paper_research_runtime_paths(config, repo_root=tmp_path)

    assert runtime_paths == discovered


def test_discover_v2_paper_research_runtime_paths_prefers_checkpoint_manifest(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "TOLBERT_BRAIN"
    ckpt_dir = base / "checkpoints" / "tolbert_brain_joint_v2"
    retrieval_dir = ckpt_dir / "retrieval_cache"
    data_dir = base / "data" / "joint_v2"
    config_path = base / "configs" / "tolbert_brain_joint_v2.yaml"
    for path in (
        config_path,
        data_dir / "nodes_joint_v2.jsonl",
        data_dir / "label_map_joint_v2.json",
        data_dir / "code_spans_joint_v2_mapped.jsonl",
        data_dir / "paper_spans_paragraphs_joint_v2_mapped.jsonl",
        ckpt_dir / "tolbert_epoch4.pt",
        retrieval_dir / "tolbert_epoch4.json",
        retrieval_dir / "tolbert_epoch4.shard00001.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    real_path = Path
    monkeypatch.setattr(
        "agent_kernel.tolbert.Path",
        lambda raw: base if str(raw) == "/data/TOLBERT_BRAIN" else real_path(raw),
    )

    runtime_paths = _discover_v2_paper_research_runtime_paths()

    assert runtime_paths is not None
    assert runtime_paths["tolbert_checkpoint_path"] == str(ckpt_dir / "tolbert_epoch4.pt")
    assert runtime_paths["tolbert_cache_paths"] == (str(retrieval_dir / "tolbert_epoch4.json"),)


def test_tolbert_service_client_times_out_and_resets(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    created_processes = []

    class FakeStdin:
        def write(self, text):
            self.last = text

        def flush(self):
            return None

    class FakeStdout:
        def readline(self):
            return ""

    class FakeStderr:
        def __init__(self):
            self._ready_sent = False

        def readline(self):
            if not self._ready_sent:
                self._ready_sent = True
                return json.dumps({"event": "startup_ready"}) + "\n"
            return ""

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = FakeStderr()
            self._poll = None
            created_processes.append(self)

        def poll(self):
            return self._poll

        def terminate(self):
            self._poll = 0

        def wait(self, timeout=None):
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        instance_count = 0

        def __init__(self):
            type(self).instance_count += 1
            self.instance_index = type(self).instance_count

        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            return [object()] if self.instance_index == 1 else []

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr("agent_kernel.tolbert.time.monotonic", _monotonic_sequence([0.0, 0.0, 1.1, 11.2, 21.3]))

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
        tolbert_service_timeout_seconds=1,
        tolbert_service_startup_attempts=1,
    )
    client = TolbertServiceClient(config=config, repo_root=tmp_path)

    with pytest.raises(RuntimeError, match="timed out|failed to become ready"):
        client.query(
            query_text="hello",
            branch_results=1,
            global_results=1,
            confidence_threshold=0.0,
            top_branches=1,
            branch_confidence_margin=0.1,
            low_confidence_widen_threshold=0.5,
            ancestor_branch_levels=1,
            low_confidence_branch_multiplier=1.0,
            low_confidence_global_multiplier=1.0,
        )

    assert len(created_processes) >= 2


def test_tolbert_service_client_waits_for_startup_ready(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    seen: list[str] = []

    class FakeStderr:
        def __init__(self) -> None:
            self._lines = [
                "transformers warning on stderr\n",
                json.dumps(
                    {
                        "event": "startup_ready",
                        "backend": "tolbert_brain_service",
                        "pid": 1234,
                        "cache_shard_count": 2,
                        "startup_elapsed_seconds": 0.25,
                    }
                )
                + "\n",
            ]

        def readline(self):
            line = self._lines.pop(0) if self._lines else ""
            if line:
                seen.append(line.strip())
            return line

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr()
            self._poll = None

        def poll(self):
            return self._poll

        def terminate(self):
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            return [object()]

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
    )
    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client._ensure_process()

    assert seen[0] == "transformers warning on stderr"
    payload = json.loads(seen[1])
    assert payload["event"] == "startup_ready"
    assert payload["backend"] == "tolbert_brain_service"
    assert "startup_elapsed_seconds" in payload


def test_tolbert_service_main_emits_structured_startup_telemetry(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    cache_path = tmp_path / "cache.pt"
    config_path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")
    nodes_path.write_text("", encoding="utf-8")
    label_map_path.write_text("{}", encoding="utf-8")
    spans_path.write_text("", encoding="utf-8")
    cache_path.write_bytes(b"stub")

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.shard_names = ["cache-a", "cache-b"]

    monkeypatch.setattr("scripts.tolbert_service.parse_args", lambda: type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "nodes": str(nodes_path),
            "label_map": str(label_map_path),
            "source_spans": [str(spans_path)],
            "cache_path": [str(cache_path)],
            "device": "cpu",
        },
    )())
    monkeypatch.setattr("scripts.tolbert_service.TolbertRuntime", FakeRuntime)
    monkeypatch.setattr("scripts.tolbert_service.sys.stdin", [])

    from scripts import tolbert_service as service_module

    service_module.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["event"] == "startup_ready"
    assert payload["device"] == "cpu"
    assert payload["cache_shard_count"] == 2
    assert "pid" in payload
    assert "startup_elapsed_seconds" in payload


def test_tolbert_service_client_raises_on_startup_error(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    class FakeStderr:
        def readline(self):
            return json.dumps({"startup_error": "CUDA requested but unavailable."}) + "\n"

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr()
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    process = FakeProcess()

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            return [object()]

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
    )

    with pytest.raises(RuntimeError, match="CUDA requested but unavailable."):
        TolbertServiceClient(config=config, repo_root=tmp_path)._ensure_process()

    assert process.terminated is True


def test_tolbert_service_client_retries_startup_ready_timeout_once(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    created_processes: list[object] = []

    class FakeStderr:
        def __init__(self, ready: bool) -> None:
            self._lines = (
                [json.dumps({"event": "startup_ready", "backend": "tolbert_brain_service"}) + "\n"]
                if ready
                else []
            )

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self, ready: bool):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr(ready)
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            instance = created_processes[-1]
            if instance.stderr._lines:
                return [object()]
            return []

        def close(self):
            return None

    def fake_popen(*args, **kwargs):
        del args, kwargs
        process = FakeProcess(ready=len(created_processes) > 0)
        created_processes.append(process)
        return process

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(
        "agent_kernel.tolbert.time.monotonic",
        _monotonic_sequence([0.0, 0.0, 1.1, 11.2, 11.3]),
    )

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
        tolbert_service_timeout_seconds=1,
        tolbert_service_startup_attempts=2,
    )

    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client._ensure_process()

    assert client.process is created_processes[-1]
    assert len(created_processes) == 2
    assert created_processes[0].terminated is True


def test_tolbert_service_client_reports_attempt_count_after_repeated_startup_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    created_processes: list[object] = []

    class FakeStderr:
        def readline(self):
            return ""

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr()
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            return []

        def close(self):
            return None

    def fake_popen(*args, **kwargs):
        del args, kwargs
        process = FakeProcess()
        created_processes.append(process)
        return process

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(
        "agent_kernel.tolbert.time.monotonic",
        _monotonic_sequence([0.0, 0.0, 1.1, 11.2, 11.3, 12.4, 23.5]),
    )

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
        tolbert_service_timeout_seconds=1,
        tolbert_service_startup_attempts=2,
    )

    with pytest.raises(RuntimeError, match="attempted startup 2 times"):
        TolbertServiceClient(config=config, repo_root=tmp_path)._ensure_process()

    assert len(created_processes) == 2
    assert all(process.terminated is True for process in created_processes)


def test_tolbert_service_client_uses_distinct_startup_timeout_budget(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    seen_timeouts: list[float | None] = []

    class FakeStderr:
        def readline(self):
            return ""

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr()
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            seen_timeouts.append(timeout)
            return []

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "9")
    monkeypatch.setattr("agent_kernel.tolbert.time.monotonic", _monotonic_sequence([0.0, 0.0, 9.1, 19.2]))

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
        tolbert_service_timeout_seconds=1,
        tolbert_service_startup_attempts=1,
    )

    with pytest.raises(RuntimeError, match="failed to become ready after 9.000 seconds"):
        TolbertServiceClient(config=config, repo_root=tmp_path)._ensure_process()

    assert seen_timeouts[0] == pytest.approx(9.0)
    assert seen_timeouts[1] == pytest.approx(10.0)


def test_tolbert_service_client_keeps_same_process_alive_during_startup_grace(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    seen_timeouts: list[float | None] = []

    class FakeStderr:
        def __init__(self) -> None:
            self._lines = [
                json.dumps({"event": "startup_ready", "backend": "tolbert_brain_service"}) + "\n"
            ]

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def read(self):
            return ""

    class FakeProcess:
        def __init__(self):
            self.stdin = None
            self.stdout = None
            self.stderr = FakeStderr()
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    class FakeSelector:
        def __init__(self) -> None:
            self.calls = 0

        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            seen_timeouts.append(timeout)
            self.calls += 1
            return [] if self.calls == 1 else [object()]

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr("agent_kernel.tolbert.time.monotonic", _monotonic_sequence([0.0, 0.0, 1.1]))

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
        tolbert_service_timeout_seconds=1,
        tolbert_service_startup_attempts=1,
    )

    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client._ensure_process()

    assert len(seen_timeouts) == 2
    assert seen_timeouts[0] == pytest.approx(1.0)
    assert seen_timeouts[1] == pytest.approx(10.0)
    assert client.process is not None


def test_tolbert_service_client_preserves_full_request_controls(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    captured: dict[str, str] = {}

    class FakeStdin:
        def write(self, text):
            captured["request"] = text

        def flush(self):
            return None

    class FakeStdout:
        def readline(self):
            return json.dumps(
                {
                    "backend": "tolbert_brain_service",
                    "index_shards": ["cache.pt"],
                    "level_focus": "repo",
                    "path_prediction": {"confidence_by_level": {"1": 0.9}},
                    "selected_branch_level": 1,
                    "branch_candidates": [],
                    "retrieval": {"branch_scoped": [], "fallback_scoped": [], "global": []},
                }
            )

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None

        def poll(self):
            return None

    class FakeSelector:
        def register(self, fileobj, events):
            del fileobj, events

        def select(self, timeout=None):
            del timeout
            return [object()]

        def close(self):
            return None

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("agent_kernel.tolbert.selectors.DefaultSelector", FakeSelector)

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
    )
    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client.query(
        query_text="hello",
        branch_results=3,
        global_results=2,
        confidence_threshold=0.1,
        top_branches=4,
        branch_confidence_margin=0.2,
        low_confidence_widen_threshold=0.5,
        ancestor_branch_levels=3,
        low_confidence_branch_multiplier=1.5,
        low_confidence_global_multiplier=2.5,
    )

    request = json.loads(captured["request"])
    assert request["ancestor_branch_levels"] == 3
    assert request["low_confidence_branch_multiplier"] == 1.5
    assert request["low_confidence_global_multiplier"] == 2.5


def test_tolbert_service_client_close_terminates_live_process(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    nodes_path = tmp_path / "nodes.jsonl"
    label_map_path = tmp_path / "label_map.json"
    spans_path = tmp_path / "spans.jsonl"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("tolbert_service.py").write_text("print('stub')\n", encoding="utf-8")
    for path in (config_path, nodes_path, label_map_path, spans_path):
        path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"stub")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self._poll = None
            self.terminated = False

        def poll(self):
            return self._poll

        def terminate(self):
            self.terminated = True
            self._poll = 0

        def wait(self, timeout=None):
            del timeout
            self._poll = 0
            return 0

        def kill(self):
            self._poll = -9

    created: list[FakeProcess] = []

    def _fake_popen(*args, **kwargs):
        del args, kwargs
        process = FakeProcess()
        created.append(process)
        return process

    monkeypatch.setattr("agent_kernel.tolbert.subprocess.Popen", _fake_popen)

    config = KernelConfig(
        provider="mock",
        tolbert_config_path=str(config_path),
        tolbert_checkpoint_path=str(checkpoint_path),
        tolbert_nodes_path=str(nodes_path),
        tolbert_label_map_path=str(label_map_path),
        tolbert_source_spans_paths=(str(spans_path),),
        tolbert_cache_paths=(str(checkpoint_path),),
    )
    client = TolbertServiceClient(config=config, repo_root=tmp_path)
    client._ensure_process()

    client.close()

    assert created[0].terminated is True


def test_tolbert_service_preserves_structured_text_for_command_and_episode_spans() -> None:
    runtime = object.__new__(__import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime)

    skill_result = runtime._meta_to_result(
        {
            "span_id": "skill:1",
            "text": "printf 'one\\n'\nprintf 'two\\n'\n",
            "source_id": "skill",
            "meta": {"span_type": "agent:skill_fragment"},
        },
        0.9,
    )
    episode_result = runtime._meta_to_result(
        {
            "span_id": "episode:1",
            "text": "task_id: hello_task\naction: code_execute\ncontent: bad command\n",
            "source_id": "episode",
            "meta": {"span_type": "agent:episode_step"},
        },
        0.8,
    )

    assert skill_result["text"].splitlines() == ["printf 'one\\n'", "printf 'two\\n'"]
    assert "content: bad command" in episode_result["text"].splitlines()


def test_tolbert_service_infers_paper_span_type_from_kind() -> None:
    runtime = object.__new__(__import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime)

    result = runtime._meta_to_result(
        {
            "span_id": "paper:1",
            "text": "Long paper paragraph text",
            "source_id": "/arxiv/pdfs/2401/2401.00001.pdf",
            "meta": {
                "kind": "paper",
                "paragraph_index": 0,
                "title": "Scaling Laws for Useful Models",
            },
        },
        0.7,
    )

    assert result["span_type"] == "doc:paper_paragraph"
    assert result["metadata"]["title"] == "Scaling Laws for Useful Models"


def test_tolbert_service_uses_hierarchy_faithful_level_focus_labels() -> None:
    runtime = object.__new__(__import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime)

    assert runtime._level_focus(1) == "pillar"
    assert runtime._level_focus(2) == "theme"
    assert runtime._level_focus(3) == "corpus"
    assert runtime._level_focus(4) == "artifact"


def test_tolbert_service_search_loads_shards_lazily_per_query() -> None:
    service_module = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"])
    TolbertRuntime = service_module.TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    runtime.cache_descriptors = [
        {"path": "shard_a.pt", "name": "shard_a.pt", "branch_presence": {}},
        {"path": "shard_b.pt", "name": "shard_b.pt", "branch_presence": {}},
    ]
    runtime.shard_names = [str(item["name"]) for item in runtime.cache_descriptors]
    runtime._loaded_shards = {}
    runtime._max_loaded_shards = 2
    loaded: list[str] = []

    def _fake_load_cache(path: Path):
        loaded.append(path.name)

        return {
            "name": path.name,
            "embs": [[1.0, 0.0]],
            "metas": [
                {
                    "span_id": path.stem,
                    "text": f"text from {path.name}",
                    "source_id": path.name,
                    "node_path": [0, 0, 0],
                    "meta": {"span_type": "agent:task"},
                }
            ],
        }

    runtime._load_cache = _fake_load_cache

    class FakeTensor:
        def __init__(self, values):
            self._values = list(values)

        def numel(self):
            return len(self._values)

        def tolist(self):
            return list(self._values)

    had_matmul = hasattr(service_module.torch, "matmul")
    had_topk = hasattr(service_module.torch, "topk")
    original_matmul = getattr(service_module.torch, "matmul", None)
    original_topk = getattr(service_module.torch, "topk", None)

    service_module.torch.matmul = lambda rows, query: FakeTensor(
        sum(float(row[idx]) * float(query[idx]) for idx in range(len(query)))
        for row in rows
    )
    service_module.torch.topk = lambda tensor, k: (
        FakeTensor(
            score for _, score in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
        FakeTensor(
            index for index, _ in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
    )
    try:
        results = runtime._search(
            [1.0, 0.0],
            limit=2,
        )
    finally:
        if had_matmul:
            service_module.torch.matmul = original_matmul
        else:
            delattr(service_module.torch, "matmul")
        if had_topk:
            service_module.torch.topk = original_topk
        else:
            delattr(service_module.torch, "topk")

    assert loaded == ["shard_a.pt", "shard_b.pt"]
    assert [item["span_id"] for item in results] == ["shard_a", "shard_b"]


def test_tolbert_service_expands_cache_manifest_paths(tmp_path: Path) -> None:
    TolbertRuntime = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    shard_a = tmp_path / "cache_a.pt"
    shard_b = tmp_path / "cache_b.pt"
    manifest = tmp_path / "cache_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_cache_manifest",
                "cache_paths": [str(shard_a), str(shard_b)],
            }
        ),
        encoding="utf-8",
    )

    expanded = runtime._load_cache_descriptors([manifest])

    assert expanded == [
        {"path": str(shard_a), "name": shard_a.name, "branch_presence": {}},
        {"path": str(shard_b), "name": shard_b.name, "branch_presence": {}},
    ]


def test_tolbert_service_uses_branch_presence_to_skip_irrelevant_shards() -> None:
    service_module = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"])
    TolbertRuntime = service_module.TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    runtime.cache_descriptors = [
        {"path": "shard_a.pt", "name": "shard_a.pt", "branch_presence": {"1": [1]}},
        {"path": "shard_b.pt", "name": "shard_b.pt", "branch_presence": {"1": [2]}},
    ]
    runtime._loaded_shards = {}
    runtime._max_loaded_shards = 2
    loaded: list[str] = []

    def _fake_load_cache(path: Path):
        loaded.append(path.name)
        return {
            "name": path.name,
            "embs": [[1.0, 0.0]],
            "metas": [
                {
                    "span_id": path.stem,
                    "text": f"text from {path.name}",
                    "source_id": path.name,
                    "node_path": [0, 2, 0],
                    "meta": {"span_type": "agent:task"},
                }
            ],
        }

    runtime._load_cache = _fake_load_cache

    class FakeTensor:
        def __init__(self, values):
            self._values = list(values)

        def numel(self):
            return len(self._values)

        def tolist(self):
            return list(self._values)

    had_matmul = hasattr(service_module.torch, "matmul")
    had_topk = hasattr(service_module.torch, "topk")
    original_matmul = getattr(service_module.torch, "matmul", None)
    original_topk = getattr(service_module.torch, "topk", None)
    service_module.torch.matmul = lambda rows, query: FakeTensor(
        sum(float(row[idx]) * float(query[idx]) for idx in range(len(query)))
        for row in rows
    )
    service_module.torch.topk = lambda tensor, k: (
        FakeTensor(
            score for _, score in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
        FakeTensor(
            index for index, _ in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
    )
    try:
        results = runtime._search(
            [1.0, 0.0],
            limit=1,
            branch_filter=(1, 2),
        )
    finally:
        if had_matmul:
            service_module.torch.matmul = original_matmul
        else:
            delattr(service_module.torch, "matmul")
        if had_topk:
            service_module.torch.topk = original_topk
        else:
            delattr(service_module.torch, "topk")

    assert loaded == ["shard_b.pt"]
    assert [item["span_id"] for item in results] == ["shard_b"]


def test_tolbert_service_uses_precomputed_branch_indices() -> None:
    service_module = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"])
    TolbertRuntime = service_module.TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    runtime.cache_descriptors = [
        {"path": "shard_a.pt", "name": "shard_a.pt", "branch_presence": {"1": [2]}},
    ]
    runtime._loaded_shards = {}
    runtime._max_loaded_shards = 2

    def _fake_load_cache(path: Path):
        del path
        return {
            "name": "shard_a.pt",
            "embs": [[1.0, 0.0], [0.0, 1.0]],
            "metas": [
                {
                    "span_id": "first",
                    "text": "first text",
                    "source_id": "first",
                    "meta": {"span_type": "agent:task"},
                },
                {
                    "span_id": "second",
                    "text": "second text",
                    "source_id": "second",
                    "meta": {"span_type": "agent:task"},
                },
            ],
            "branch_indices": {1: {2: [1]}},
        }

    runtime._load_cache = _fake_load_cache

    class FakeTensor:
        def __init__(self, values):
            self._values = list(values)

        def numel(self):
            return len(self._values)

        def tolist(self):
            return list(self._values)

    had_matmul = hasattr(service_module.torch, "matmul")
    had_topk = hasattr(service_module.torch, "topk")
    original_matmul = getattr(service_module.torch, "matmul", None)
    original_topk = getattr(service_module.torch, "topk", None)
    service_module.torch.matmul = lambda rows, query: FakeTensor(
        sum(float(row[idx]) * float(query[idx]) for idx in range(len(query)))
        for row in rows
    )
    service_module.torch.topk = lambda tensor, k: (
        FakeTensor(
            score for _, score in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
        FakeTensor(
            index for index, _ in sorted(enumerate(tensor.tolist()), key=lambda item: item[1], reverse=True)[:k]
        ),
    )
    try:
        results = runtime._search(
            [0.0, 1.0],
            limit=1,
            branch_filter=(1, 2),
        )
    finally:
        if had_matmul:
            service_module.torch.matmul = original_matmul
        else:
            delattr(service_module.torch, "matmul")
        if had_topk:
            service_module.torch.topk = original_topk
        else:
            delattr(service_module.torch, "topk")

    assert [item["span_id"] for item in results] == ["second"]


def test_build_tolbert_cache_writes_runtime_discoverable_shards(tmp_path: Path) -> None:
    cache_module = __import__("scripts.build_tolbert_cache", fromlist=["_write_sharded_cache"])
    spans_path = tmp_path / "spans.jsonl"
    checkpoint_path = tmp_path / "tolbert_epoch5.pt"
    out_path = tmp_path / "retrieval_cache" / "tolbert_epoch5.pt"
    spans_path.write_text(
        "".join(
            json.dumps({"span_id": f"s{index}", "text": f"text {index}"}) + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    checkpoint_path.write_bytes(b"stub")

    original_encode_batch = cache_module._encode_batch
    original_save_cache_payload = cache_module._save_cache_payload
    had_torch_cat = hasattr(cache_module.torch, "cat")
    original_torch_cat = getattr(cache_module.torch, "cat", None)

    def _fake_encode_batch(*, model, tokenizer, records, max_length, device):
        del model, tokenizer, max_length, device
        return [[f"emb:{record['span_id']}"] for record in records], list(records)

    def _fake_save_cache_payload(*, out_path, embs, metas, spans_mtime, ckpt_mtime):
        del embs, metas, spans_mtime, ckpt_mtime
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("stub", encoding="utf-8")

    cache_module._encode_batch = _fake_encode_batch
    cache_module._save_cache_payload = _fake_save_cache_payload
    cache_module.torch.cat = lambda tensors, dim: [item for tensor in tensors for item in tensor]
    try:
        shard_paths, manifest_path, total_spans = cache_module._write_sharded_cache(
            model=None,
            tokenizer=None,
            spans_path=spans_path,
            out_path=out_path,
            max_length=4,
            device=None,
            batch_size=2,
            shard_size=2,
            progress_every=0,
            checkpoint=checkpoint_path,
        )
    finally:
        cache_module._encode_batch = original_encode_batch
        cache_module._save_cache_payload = original_save_cache_payload
        if had_torch_cat:
            cache_module.torch.cat = original_torch_cat
        else:
            delattr(cache_module.torch, "cat")

    assert total_spans == 5
    assert [path.name for path in shard_paths] == [
        "tolbert_epoch5.shard00001.pt",
        "tolbert_epoch5.shard00002.pt",
        "tolbert_epoch5.shard00003.pt",
    ]
    assert all(path.parent == out_path.parent for path in shard_paths)
    assert all(path.exists() for path in shard_paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_shards"] == 3
    assert manifest["cache_paths"] == [str(path) for path in shard_paths]


def test_tolbert_service_uses_path_positions_for_label_heads(tmp_path: Path) -> None:
    TolbertRuntime = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    runtime.nodes_by_id = {
        0: {"node_id": 0, "level": 0, "name": "root"},
        10: {"node_id": 10, "level": 1, "name": "Code"},
        30: {"node_id": 30, "level": 3, "name": "ResearchPapers"},
    }
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(
        json.dumps({"1": {"0": 10}, "2": {"0": 30}}),
        encoding="utf-8",
    )

    label_maps, position_to_head_level, head_level_to_position = runtime._load_label_maps(label_map_path)

    assert sorted(label_maps) == [1, 2]
    assert position_to_head_level == {1: 1, 2: 2}
    assert head_level_to_position == {1: 1, 2: 2}


def test_tolbert_service_decode_restricts_to_mapped_local_ids(tmp_path: Path) -> None:
    TolbertRuntime = __import__("scripts.tolbert_service", fromlist=["TolbertRuntime"]).TolbertRuntime
    runtime = object.__new__(TolbertRuntime)
    runtime.nodes_by_id = {
        10: {"node_id": 10, "level": 1, "name": "Code"},
        11: {"node_id": 11, "level": 1, "name": "Papers"},
        20: {"node_id": 20, "level": 3, "name": "CodeRepositories"},
        21: {"node_id": 21, "level": 3, "name": "ResearchPapers"},
    }
    runtime.label_maps = {
        1: {0: 10, 1: 11},
        2: {0: 20, 1: 21},
    }
    runtime.parent_to_children = {
        2: {
            0: [0],
            1: [1],
        }
    }

    import torch

    prediction = runtime._decode_path(
        {
            1: torch.as_tensor([0.1, 0.9, 5.0], dtype=torch.float32),
            2: torch.as_tensor([0.2, 0.8, 4.0], dtype=torch.float32),
        },
        confidence_threshold=0.0,
    )
    candidates = runtime._branch_candidates(
        level_logits={
            1: torch.as_tensor([0.1, 0.9, 5.0], dtype=torch.float32),
            2: torch.as_tensor([0.2, 0.8, 4.0], dtype=torch.float32),
        },
        path_prediction=prediction,
        branch_level=1,
        top_branches=2,
        confidence_margin=10.0,
    )

    assert prediction["predicted_level_ids"]["1"] == 1
    assert prediction["predicted_level_ids"]["2"] == 1
    assert prediction["labels_by_level"]["2"] == "ResearchPapers"
    assert [item["local_id"] for item in candidates] == [1, 0]


def test_policy_falls_back_when_tolbert_compile_fails() -> None:
    class FailingContextProvider:
        def compile(self, state):
            del state
            raise RuntimeError("tolbert unavailable")

    policy = LLMDecisionPolicy(
        MockLLMClient(),
        context_provider=FailingContextProvider(),
        config=KernelConfig(provider="mock", use_tolbert_context=True),
    )
    state = AgentState(task=TaskBank().get("hello_task"))

    decision = policy.decide(state)

    assert decision.action == "code_execute"
    assert state.context_packet is None
