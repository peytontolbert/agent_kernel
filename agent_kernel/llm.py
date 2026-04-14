from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib import error, request

DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["respond", "code_execute"]},
        "content": {"type": "string"},
        "done": {"type": "boolean"},
    },
    "required": ["thought", "action", "content", "done"],
    "additionalProperties": False,
}


class LLMClient(Protocol):
    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class OllamaClient:
    def __init__(
        self,
        host: str,
        model_name: str,
        timeout_seconds: int,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = [
            _render_prompt(
                decision_prompt=decision_prompt,
                state_payload=_compact_state_payload(state_payload),
            ),
            _render_prompt(
                decision_prompt=decision_prompt,
                state_payload=_minimal_state_payload(state_payload),
            ),
        ]
        last_data: dict[str, Any] | None = None
        for index, prompt in enumerate(attempts):
            data = self._generate(system_prompt=system_prompt, prompt=prompt, compact=True)
            last_data = data
            for field in ("response", "thinking"):
                parsed = _extract_json_object(data.get(field, ""))
                if parsed is not None:
                    return parsed
            if data.get("done_reason") != "length":
                break
        raise ValueError(f"Ollama did not return a parseable JSON decision: {last_data}")

    def _generate(self, *, system_prompt: str, prompt: str, compact: bool) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024 if compact else 768,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (TimeoutError, error.URLError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self.retry_attempts:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(
            f"Ollama request failed after {self.retry_attempts} attempts: {last_error}"
        )


class VLLMClient:
    def __init__(
        self,
        host: str,
        model_name: str,
        timeout_seconds: int,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
        api_key: str = "",
    ) -> None:
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.api_key = api_key.strip()

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = [
            _render_prompt(
                decision_prompt=decision_prompt,
                state_payload=_compact_state_payload(state_payload),
            ),
            _render_prompt(
                decision_prompt=decision_prompt,
                state_payload=_minimal_state_payload(state_payload),
            ),
        ]
        last_data: dict[str, Any] | None = None
        last_error: Exception | None = None
        for prompt in attempts:
            parsed: dict[str, Any] | None = None
            try:
                data = self._chat_completion(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    use_json_schema=True,
                )
            except RuntimeError as exc:
                last_error = exc
                # Fall back to prompt-only JSON if the server rejects structured output.
                data = self._chat_completion(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    use_json_schema=False,
                )
                last_data = data
                parsed = self._extract_decision(data)
            else:
                last_data = data
                parsed = self._extract_decision(data)
                if parsed is None:
                    # Some Qwen/vLLM stacks spend the completion budget in reasoning
                    # and leave content empty even when the transport succeeds.
                    data = self._chat_completion(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        use_json_schema=False,
                    )
                    last_data = data
                    parsed = self._extract_decision(data)
            if parsed is not None:
                return parsed
        if last_error is not None:
            raise ValueError(f"vLLM did not return a parseable JSON decision: {last_data}") from last_error
        raise ValueError(f"vLLM did not return a parseable JSON decision: {last_data}")

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        use_json_schema: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_kernel_decision",
                    "schema": DECISION_JSON_SCHEMA,
                },
            }
        return _post_json(
            url=f"{self.host}/v1/chat/completions",
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry_attempts=self.retry_attempts,
            retry_backoff_seconds=self.retry_backoff_seconds,
            headers=_authorization_headers(self.api_key),
            error_label="vLLM request",
        )

    @staticmethod
    def _extract_decision(data: dict[str, Any]) -> dict[str, Any] | None:
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message", {})
        if not isinstance(message, dict):
            return None
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        parsed = _extract_json_object(str(content))
        if parsed is not None:
            return parsed
        return _extract_json_object(str(message.get("reasoning", "")))


class MockLLMClient:
    """Deterministic test double for the LLM client interface."""

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        del system_prompt, decision_prompt
        task = state_payload["task"]
        attempted = {
            step["content"]
            for step in state_payload["history"]
            if step["action"] == "code_execute"
        }
        candidates = [*task["suggested_commands"], task["success_command"]]
        for command in candidates:
            if command and command not in attempted:
                return {
                    "thought": "Try the next command suggested by the task.",
                    "action": "code_execute",
                    "content": command,
                    "done": False,
                }

        return {
            "thought": "No remaining commands to try.",
            "action": "respond",
            "content": "No remaining deterministic commands to try.",
            "done": True,
        }


class HybridFallbackClient:
    """Providerless fallback for the retained hybrid runtime path."""

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        del system_prompt, decision_prompt
        task = state_payload["task"]
        attempted = {
            step["content"]
            for step in state_payload["history"]
            if step["action"] == "code_execute"
        }
        candidates = [
            *task.get("suggested_commands", []),
            task.get("success_command", ""),
        ]
        for command in candidates:
            normalized = str(command).strip()
            if normalized and normalized not in attempted:
                return {
                    "thought": "Use the next deterministic task or verifier command.",
                    "action": "code_execute",
                    "content": normalized,
                    "done": False,
                }
        return {
            "thought": "No deterministic providerless fallback command remains.",
            "action": "respond",
            "content": "Providerless hybrid fallback exhausted deterministic commands.",
            "done": True,
        }


# Backward-compatible alias while callers migrate from the old overloaded name.
TolbertFallbackClient = HybridFallbackClient


def coerce_action_decision(raw: dict[str, Any]) -> dict[str, Any]:
    action = str(raw.get("action", "respond")).strip().lower() or "respond"
    thought = str(raw.get("thought", "")).strip() or "No thought provided."
    content = str(raw.get("content", "")).strip()
    done = bool(raw.get("done", False))
    if action == "respond":
        done = True
    return {
        "thought": thought,
        "action": action,
        "content": content,
        "done": done,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _render_prompt(*, decision_prompt: str, state_payload: dict[str, Any]) -> str:
    return (
        f"{decision_prompt}\n\n"
        "State payload:\n"
        f"{json.dumps(state_payload, indent=2)}\n\n"
        "Return a JSON object with keys thought, action, content, done. "
        "Keep thought to one short sentence."
    )


def _compact_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    task = state_payload["task"]
    compact_task = {
        "task_id": task.get("task_id"),
        "prompt": task.get("prompt"),
        "workspace_subdir": task.get("workspace_subdir"),
        "success_command": task.get("success_command"),
        "suggested_commands": task.get("suggested_commands", [])[:3],
        "expected_files": task.get("expected_files", []),
        "expected_output_substrings": task.get("expected_output_substrings", []),
    }
    compact_history = []
    for step in state_payload.get("history", [])[-2:]:
        compact_history.append(
            {
                "index": step.get("index"),
                "action": step.get("action"),
                "content": step.get("content"),
                "verification": step.get("verification", {}),
            }
        )

    compact_context = None
    context_packet = state_payload.get("context_packet")
    if context_packet:
        compact_context = {
            "task": context_packet.get("task"),
            "control": context_packet.get("control"),
            "tolbert": context_packet.get("tolbert"),
            "retrieval": {
                "branch_scoped": [
                    _compact_retrieved_span(span)
                    for span in context_packet.get("retrieval", {}).get("branch_scoped", [])[:3]
                ],
                "global": [
                    _compact_retrieved_span(span)
                    for span in context_packet.get("retrieval", {}).get("global", [])[:2]
                ],
            },
            "verifier_contract": context_packet.get("verifier_contract"),
        }

    return {
        "task": compact_task,
        "history": compact_history,
        "recent_workspace_summary": state_payload.get("recent_workspace_summary", ""),
        "context_packet": compact_context,
        "retrieval_plan": state_payload.get("retrieval_plan", {}),
        "available_skills": state_payload.get("available_skills", [])[:3],
        "graph_summary": state_payload.get("graph_summary", {}),
        "world_model_summary": state_payload.get("world_model_summary", {}),
        "latest_state_transition": state_payload.get("latest_state_transition", {}),
        "plan": state_payload.get("plan", [])[:4],
        "active_subgoal": state_payload.get("active_subgoal", ""),
        "active_subgoal_diagnosis": state_payload.get("active_subgoal_diagnosis", {}),
        "acting_role": state_payload.get("acting_role", ""),
        "state_context_chunks": state_payload.get("state_context_chunks", [])[:8],
        "allowed_actions": state_payload.get("allowed_actions", []),
    }


def _minimal_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_state_payload(state_payload)
    compact["context_packet"] = {
        "control": {
            "path_confidence": compact.get("context_packet", {}).get("control", {}).get("path_confidence", 0.0),
            "trust_retrieval": compact.get("context_packet", {}).get("control", {}).get("trust_retrieval", False),
            "retrieval_guidance": compact.get("context_packet", {}).get("control", {}).get("retrieval_guidance", {}),
            "selected_context_chunks": compact.get("context_packet", {}).get("control", {}).get("selected_context_chunks", [])[:4],
        },
        "verifier_contract": compact.get("context_packet", {}).get("verifier_contract", {}),
    } if compact.get("context_packet") else None
    return compact


def _compact_retrieved_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": span.get("span_id"),
        "source_id": span.get("source_id"),
        "span_type": span.get("span_type"),
        "score": span.get("score"),
        "node_path": span.get("node_path"),
        "text": str(span.get("text", ""))[:160],
    }


def _authorization_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
    headers: dict[str, str],
    error_label: str,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retry_attempts):
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 >= retry_attempts:
                break
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    raise RuntimeError(f"{error_label} failed after {retry_attempts} attempts: {last_error}")
