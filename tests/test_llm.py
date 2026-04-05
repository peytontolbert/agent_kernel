from urllib import error

import pytest

import json

from agent_kernel.llm import VLLMClient, OllamaClient, _extract_json_object, coerce_action_decision


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
