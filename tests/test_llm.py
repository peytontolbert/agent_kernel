import io
import time
from urllib import error
from pathlib import Path

import pytest

import json

from agent_kernel.config import KernelConfig
from agent_kernel.llm import (
    HybridDecoderClient,
    ModelStackClient,
    VLLMClient,
    OllamaClient,
    _extract_json_object,
    _post_json,
    coerce_action_decision,
)


def test_extract_json_object_reads_thinking_payload():
    text = """
    leading text
    {
      "thought": "x",
      "action": "respond",
      "content": "ok",
      "done": true
    }
    """

    parsed = _extract_json_object(text)

    assert parsed == {
        "thought": "x",
        "action": "respond",
        "content": "ok",
        "done": True,
    }


def test_coerce_action_decision_forces_respond_to_done():
    parsed = coerce_action_decision(
        {
            "thought": "stop here",
            "action": "respond",
            "content": "done",
            "done": False,
        }
    )

    assert parsed["action"] == "respond"
    assert parsed["done"] is True


def test_ollama_client_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"response":"{\\"thought\\":\\"ok\\",\\"action\\":\\"respond\\",\\"content\\":\\"done\\",\\"done\\":true}"}'

    def fake_urlopen(req, timeout):
        del req, timeout
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.URLError("temporary failure")
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        model_name="test",
        timeout_seconds=1,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert calls["count"] == 2


def test_ollama_client_raises_after_retries(monkeypatch):
    def fake_urlopen(req, timeout):
        del req, timeout
        raise error.URLError("still failing")

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        model_name="test",
        timeout_seconds=1,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="Ollama request failed after 2 attempts"):
        client.create_decision(
            system_prompt="system",
            decision_prompt="decision",
            state_payload={"task": {}, "history": []},
        )


def test_vllm_client_uses_structured_chat_completion(monkeypatch):
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"thought":"ok","action":"respond","content":"done","done":true}'
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=3,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        api_key="secret",
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["timeout"] == 3
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["payload"]["response_format"]["type"] == "json_schema"
    assert seen["payload"]["chat_template_kwargs"]["enable_thinking"] is False
    assert seen["payload"]["max_tokens"] == 384
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert seen["payload"]["messages"][1]["role"] == "user"


def test_vllm_client_falls_back_when_structured_output_request_fails(monkeypatch):
    calls = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"thought\\":\\"ok\\",\\"action\\":\\"respond\\",\\"content\\":\\"done\\",\\"done\\":true}"}}]}'

    def fake_urlopen(req, timeout):
        del timeout
        calls["count"] += 1
        payload = json.loads(req.data.decode("utf-8"))
        if calls["count"] == 1:
            assert "response_format" in payload
            raise error.URLError("structured output unsupported")
        assert "response_format" not in payload
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert calls["count"] == 2


def test_vllm_client_reduces_completion_budget_after_context_limit_error(monkeypatch):
    calls = {"count": 0, "max_tokens": []}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"thought\\":\\"ok\\",\\"action\\":\\"respond\\",\\"content\\":\\"done\\",\\"done\\":true}"}}]}'

    def fake_urlopen(req, timeout):
        del timeout
        payload = json.loads(req.data.decode("utf-8"))
        calls["count"] += 1
        calls["max_tokens"].append(int(payload["max_tokens"]))
        if int(payload["max_tokens"]) == 384:
            raise error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(
                    b'{"error":{"message":"maximum input length exceeded: input tokens too large"}}'
                ),
            )
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert calls["max_tokens"][:3] == [384, 384, 256]


def test_model_stack_client_uses_token_generation_endpoint(monkeypatch):
    seen = {}

    class FakeTokenizer:
        def encode(self, text):
            seen["prompt"] = text
            return [1, 2, 3]

        def decode(self, ids):
            seen["decoded_ids"] = list(ids)
            return '{"thought":"ok","action":"respond","content":"done","done":true}'

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_ids": [[1, 2, 3, 100, 101]]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    monkeypatch.setattr(ModelStackClient, "_load_tokenizer", lambda self: FakeTokenizer())
    client = ModelStackClient(
        host="http://127.0.0.1:8001",
        model_name="model-stack-test",
        timeout_seconds=3,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        model_dir="/tmp/model-stack-model",
        repo_path="/tmp/model-stack",
        api_key="secret",
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert decision["decision_source"] == "model_stack"
    assert seen["url"] == "http://127.0.0.1:8001/v1/generate"
    assert seen["timeout"] == 3
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["payload"]["input_ids"] == [[1, 2, 3]]
    assert seen["payload"]["do_sample"] is False
    assert seen["payload"]["temperature"] == 0.0
    assert seen["payload"]["max_new_tokens"] == 32
    assert seen["decoded_ids"] == [100, 101]


def test_model_stack_client_recovers_from_transient_generation_failure(monkeypatch):
    calls = {"count": 0}

    class FakeTokenizer:
        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return '{"thought":"ok","action":"respond","content":"done","done":true}'

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_ids": [[1, 2, 3, 100, 101]]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        del req, timeout
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("slow local model")
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    monkeypatch.setattr(ModelStackClient, "_load_tokenizer", lambda self: FakeTokenizer())
    client = ModelStackClient(
        host="http://127.0.0.1:8001",
        model_name="model-stack-test",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        model_dir="/tmp/model-stack-model",
        repo_path="/tmp/model-stack",
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["decision_source"] == "model_stack"
    assert calls["count"] == 2


def test_model_stack_client_accepts_decision_token_budget_override(monkeypatch):
    seen = {"max_new_tokens": []}

    class FakeTokenizer:
        def encode(self, text):
            del text
            return [1, 2, 3]

        def decode(self, ids):
            del ids
            return '{"thought":"ok","action":"respond","content":"done","done":true}'

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_ids": [[1, 2, 3, 100]]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        del timeout
        payload = json.loads(req.data.decode("utf-8"))
        seen["max_new_tokens"].append(int(payload["max_new_tokens"]))
        return _Response()

    monkeypatch.setenv("AGENT_KERNEL_MODEL_STACK_DECISION_MAX_TOKENS", "8,16")
    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    monkeypatch.setattr(ModelStackClient, "_load_tokenizer", lambda self: FakeTokenizer())
    client = ModelStackClient(
        host="http://127.0.0.1:8001",
        model_name="model-stack-test",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        model_dir="/tmp/model-stack-model",
        repo_path="/tmp/model-stack",
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["decision_source"] == "model_stack"
    assert seen["max_new_tokens"] == [8]


def test_post_json_enforces_wall_timeout(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            time.sleep(2)
            return b'{"ok":true}'

    def fake_urlopen(req, timeout):
        del req, timeout
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="wall timeout"):
        _post_json(
            url="http://127.0.0.1:8001/v1/generate",
            payload={"input_ids": [[1]], "max_new_tokens": 1},
            timeout_seconds=1,
            retry_attempts=1,
            retry_backoff_seconds=0.0,
            headers={"Content-Type": "application/json"},
            error_label="Model Stack request",
        )

    assert time.monotonic() - started < 1.8


def test_model_stack_client_requires_tokenizer_root():
    client = ModelStackClient(
        host="http://127.0.0.1:8001",
        model_name="not-a-path",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        repo_path="/tmp/model-stack",
    )

    with pytest.raises(RuntimeError, match="requires AGENT_KERNEL_MODEL_STACK_TOKENIZER_PATH"):
        client._tokenizer_root()


def test_vllm_client_compacts_large_prompt_payload(monkeypatch):
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"thought\\":\\"ok\\",\\"action\\":\\"respond\\",\\"content\\":\\"done\\",\\"done\\":true}"}}]}'

    def fake_urlopen(req, timeout):
        del timeout
        payload = json.loads(req.data.decode("utf-8"))
        seen["prompt"] = payload["messages"][1]["content"]
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    state_payload = {
        "task": {
            "task_id": "large",
            "prompt": "Investigate the integration path and produce a concrete repo-specific task.",
            "workspace_subdir": "semantic_open_world/large",
            "success_command": "printf 'ok\\n' > out.txt",
            "suggested_commands": ["echo a", "echo b", "echo c", "echo d"],
            "expected_files": ["out.txt", "notes.txt", "summary.txt", "plan.txt", "extra.txt"],
            "expected_output_substrings": ["done", "verified", "retained", "ready", "extra"],
        },
        "history": [
            {
                "index": 1,
                "action": "code_execute",
                "content": "printf 'first' > one.txt",
                "verification": {"passed": False, "reason": "needs more work", "details": "x" * 5000},
            }
        ],
        "graph_summary": {"nodes": [{"id": f"n{i}", "summary": "x" * 4000} for i in range(20)]},
        "world_model_summary": {"facts": ["y" * 4000 for _ in range(12)]},
        "state_context_chunks": [{"path": f"f{i}.py", "content": "z" * 2000} for i in range(10)],
        "allowed_actions": ["code_execute", "respond"],
    }

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload=state_payload,
    )

    assert decision["action"] == "respond"
    prompt = str(seen["prompt"])
    assert len(prompt) < 4000
    assert '"graph_summary":{"nodes":[{"id":"n0","summary":"x' not in prompt


def test_vllm_client_falls_back_to_lean_payload_after_prompt_context_limit_error(monkeypatch):
    calls = {"count": 0, "prompts": []}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"thought\\":\\"ok\\",\\"action\\":\\"respond\\",\\"content\\":\\"done\\",\\"done\\":true}"}}]}'

    def fake_urlopen(req, timeout):
        del timeout
        payload = json.loads(req.data.decode("utf-8"))
        prompt = payload["messages"][1]["content"]
        calls["count"] += 1
        calls["prompts"].append(prompt)
        if '"graph_summary"' in prompt or '"world_model_summary"' in prompt or '"state_context_chunks"' in prompt:
            raise error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(
                    b'{"error":{"message":"maximum input length exceeded: input tokens too large"}}'
                ),
            )
        return _Response()

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={
            "task": {
                "task_id": "large",
                "prompt": "Investigate the integration path and produce a concrete repo-specific task.",
                "workspace_subdir": "semantic_open_world/large",
                "success_command": "printf 'ok\\n' > out.txt",
                "suggested_commands": ["echo a", "echo b", "echo c"],
            },
            "history": [{"index": 1, "action": "respond", "content": "thinking", "verification": {}}],
            "graph_summary": {"nodes": [{"id": "n1", "summary": "x" * 5000}]},
            "world_model_summary": {"facts": ["y" * 5000]},
            "state_context_chunks": [{"path": "file.py", "content": "z" * 4000}],
            "allowed_actions": ["code_execute", "respond"],
        },
    )

    assert decision["action"] == "respond"
    assert calls["count"] >= 5
    assert '"graph_summary"' not in calls["prompts"][-1]
    assert '"world_model_summary"' not in calls["prompts"][-1]
    assert '"state_context_chunks"' not in calls["prompts"][-1]


def test_vllm_client_retries_without_schema_when_structured_response_is_unparseable(monkeypatch):
    calls = {"count": 0}

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(req, timeout):
        del timeout
        calls["count"] += 1
        payload = json.loads(req.data.decode("utf-8"))
        if calls["count"] == 1:
            assert "response_format" in payload
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": "thinking without a final json object",
                            }
                        }
                    ]
                }
            )
        assert "response_format" not in payload
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"thought":"ok","action":"respond","content":"done","done":true}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agent_kernel.llm.request.urlopen", fake_urlopen)
    client = VLLMClient(
        host="http://127.0.0.1:8000",
        model_name="test-model",
        timeout_seconds=1,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert decision["action"] == "respond"
    assert calls["count"] == 2


def test_vllm_client_extracts_json_from_reasoning_when_content_is_null():
    parsed = VLLMClient._extract_decision(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": 'prefix {"thought":"ok","action":"respond","content":"done","done":true}',
                    }
                }
            ]
        }
    )

    assert parsed == {
        "thought": "ok",
        "action": "respond",
        "content": "done",
        "done": True,
    }


def test_hybrid_decoder_client_uses_retained_bundle_and_marks_decoder_source(monkeypatch, tmp_path):
    bundle_path = tmp_path / "tolbert" / "hybrid_bundle_manifest.json"
    artifact_path = tmp_path / "tolbert" / "tolbert_model_artifact.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "primary_enabled": True,
                    "supports_decoder_surface": True,
                    "bundle_manifest_path": str(bundle_path),
                },
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_generate_hybrid_decoder_completion(**kwargs):
        seen.update(kwargs)
        return {
            "generated_text": json.dumps(
                {
                    "thought": "decode directly",
                    "action": "code_execute",
                    "content": "printf 'hybrid\\n' > hybrid.txt",
                    "done": False,
                }
            ),
            "model_family": "tolbert_ssm_v1",
            "avg_logprob": -0.25,
        }

    monkeypatch.setattr("agent_kernel.llm.generate_hybrid_decoder_completion", fake_generate_hybrid_decoder_completion)
    client = HybridDecoderClient(
        config=KernelConfig(provider="hybrid", tolbert_model_artifact_path=artifact_path),
        repo_root=Path(__file__).resolve().parents[1],
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert seen["bundle_manifest_path"] == bundle_path
    assert decision["action"] == "code_execute"
    assert decision["content"] == "printf 'hybrid\\n' > hybrid.txt"
    assert decision["decision_source"] == "hybrid_decoder"
    assert decision["proposal_metadata"]["decoder_bundle_manifest_path"] == str(bundle_path)


def test_hybrid_decoder_client_prefers_materialized_universal_decoder_runtime(monkeypatch, tmp_path):
    hybrid_bundle_path = tmp_path / "tolbert" / "hybrid_bundle_manifest.json"
    universal_bundle_path = tmp_path / "tolbert" / "universal_bundle_manifest.json"
    artifact_path = tmp_path / "tolbert" / "tolbert_model_artifact.json"
    hybrid_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
    universal_bundle_path.write_text(json.dumps({"artifact_kind": "tolbert_hybrid_runtime_bundle"}), encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "tolbert_model_bundle",
                "hybrid_runtime": {
                    "primary_enabled": True,
                    "supports_decoder_surface": True,
                    "bundle_manifest_path": str(hybrid_bundle_path),
                },
                "universal_decoder_runtime": {
                    "materialized": True,
                    "bundle_manifest_path": str(universal_bundle_path),
                    "training_objective": "universal_decoder_only",
                },
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_generate_hybrid_decoder_completion(**kwargs):
        seen.update(kwargs)
        return {
            "generated_text": json.dumps(
                {
                    "thought": "decode directly",
                    "action": "code_execute",
                    "content": "printf 'universal\\n' > universal.txt",
                    "done": False,
                }
            ),
            "model_family": "tolbert_ssm_v1",
            "avg_logprob": -0.1,
        }

    monkeypatch.setattr("agent_kernel.llm.generate_hybrid_decoder_completion", fake_generate_hybrid_decoder_completion)
    client = HybridDecoderClient(
        config=KernelConfig(provider="hybrid", tolbert_model_artifact_path=artifact_path),
        repo_root=Path(__file__).resolve().parents[1],
    )

    decision = client.create_decision(
        system_prompt="system",
        decision_prompt="decision",
        state_payload={"task": {}, "history": []},
    )

    assert seen["bundle_manifest_path"] == universal_bundle_path
    assert decision["content"] == "printf 'universal\\n' > universal.txt"
    assert decision["decision_source"] == "retained_decoder"
    assert decision["proposal_metadata"]["decoder_runtime_key"] == "universal_decoder_runtime"
    assert decision["proposal_metadata"]["decoder_training_objective"] == "universal_decoder_only"
