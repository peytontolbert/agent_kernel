from __future__ import annotations

import json
import time
from pathlib import Path
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
    _DECISION_MAX_TOKENS = (384, 256, 192, 128, 96, 64)

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
            _render_prompt(
                decision_prompt=decision_prompt,
                state_payload=_lean_state_payload(state_payload),
            ),
        ]
        last_data: dict[str, Any] | None = None
        last_error: Exception | None = None
        for prompt in attempts:
            for max_tokens in self._DECISION_MAX_TOKENS:
                parsed: dict[str, Any] | None = None
                try:
                    data = self._chat_completion(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        use_json_schema=True,
                        max_tokens=max_tokens,
                    )
                except RuntimeError as exc:
                    last_error = exc
                    try:
                        # Fall back to prompt-only JSON if the server rejects structured output.
                        data = self._chat_completion(
                            system_prompt=system_prompt,
                            prompt=prompt,
                            use_json_schema=False,
                            max_tokens=max_tokens,
                        )
                    except RuntimeError as fallback_exc:
                        last_error = fallback_exc
                        if self._is_context_limit_error(str(exc)) or self._is_context_limit_error(str(fallback_exc)):
                            continue
                        raise
                    last_data = data
                    parsed = self._extract_decision(data)
                else:
                    last_data = data
                    parsed = self._extract_decision(data)
                    if parsed is None:
                        # Some Qwen/vLLM stacks spend the completion budget in reasoning
                        # and leave content empty even when the transport succeeds.
                        try:
                            data = self._chat_completion(
                                system_prompt=system_prompt,
                                prompt=prompt,
                                use_json_schema=False,
                                max_tokens=max_tokens,
                            )
                        except RuntimeError as fallback_exc:
                            last_error = fallback_exc
                            if self._is_context_limit_error(str(fallback_exc)):
                                continue
                            raise
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
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(64, int(max_tokens)),
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
    def _is_context_limit_error(error_text: str) -> bool:
        normalized = str(error_text).strip().lower()
        if not normalized:
            return False
        return (
            "maximum input length" in normalized
            or "context length" in normalized
            or "input tokens" in normalized
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


def generate_hybrid_decoder_completion(
    *,
    prompt: str,
    bundle_manifest_path: Path,
    device: str = "cpu",
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    from .modeling.tolbert import generate_hybrid_decoder_completion as _generate_hybrid_decoder_completion

    return _generate_hybrid_decoder_completion(
        prompt=prompt,
        bundle_manifest_path=bundle_manifest_path,
        device=device,
        max_new_tokens=max_new_tokens,
    )


def coerce_decoder_text_decision(
    raw_text: str,
    *,
    default_command_thought: str,
    default_response_thought: str,
) -> dict[str, Any] | None:
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        return coerce_action_decision(parsed)
    normalized = " ".join(str(raw_text).strip().split())
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered.startswith(("respond:", "respond ", "stop:", "stop ", "done:", "done ")):
        response = normalized.split(":", 1)[-1].strip() if ":" in normalized else normalized
        return {
            "thought": default_response_thought,
            "action": "respond",
            "content": response or normalized,
            "done": True,
        }
    return {
        "thought": default_command_thought,
        "action": "code_execute",
        "content": normalized,
        "done": False,
    }


class HybridDecoderClient:
    """Retained Tolbert-family decoder client for provider='hybrid'."""

    def __init__(self, *, config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root

    def _bundle_manifest_path(self) -> Path:
        from .extensions.runtime_modeling_adapter import (
            load_model_artifact,
            retained_tolbert_active_decoder_runtime,
        )

        payload = load_model_artifact(self.config.tolbert_model_artifact_path)
        runtime = retained_tolbert_active_decoder_runtime(payload)
        manifest_raw = str(runtime.get("bundle_manifest_path", "")).strip()
        if not manifest_raw:
            raise RuntimeError("provider='hybrid' requires a retained Tolbert bundle_manifest_path")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = self.repo_root / manifest_path
        if not manifest_path.exists():
            raise RuntimeError(
                f"provider='hybrid' bundle manifest does not exist: {manifest_path}"
            )
        return manifest_path

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        from .extensions.runtime_modeling_adapter import (
            load_model_artifact,
            retained_tolbert_active_decoder_runtime,
        )

        payload = load_model_artifact(self.config.tolbert_model_artifact_path)
        runtime = retained_tolbert_active_decoder_runtime(payload)
        manifest_path = self._bundle_manifest_path()
        runtime_key = str(runtime.get("runtime_key", "hybrid_runtime")).strip() or "hybrid_runtime"
        attempts = [
            _render_prompt(
                decision_prompt=f"System prompt:\n{system_prompt}\n\n{decision_prompt}",
                state_payload=_compact_state_payload(state_payload),
            ),
            _render_prompt(
                decision_prompt=f"System prompt:\n{system_prompt}\n\n{decision_prompt}",
                state_payload=_minimal_state_payload(state_payload),
            ),
        ]
        last_output: dict[str, Any] | None = None
        for prompt in attempts:
            output = generate_hybrid_decoder_completion(
                prompt=prompt,
                bundle_manifest_path=manifest_path,
                device="cpu",
                max_new_tokens=128,
            )
            last_output = output
            decision = coerce_decoder_text_decision(
                str(output.get("generated_text", "")),
                default_command_thought="Execute the retained decoder command.",
                default_response_thought="Stop because the retained decoder emitted a terminal response.",
            )
            if decision is None:
                continue
            decision["decision_source"] = (
                "retained_decoder" if runtime_key == "universal_decoder_runtime" else "hybrid_decoder"
            )
            decision["proposal_metadata"] = {
                "decoder_model_family": str(output.get("model_family", "")).strip(),
                "decoder_avg_logprob": float(output.get("avg_logprob", 0.0) or 0.0),
                "decoder_bundle_manifest_path": str(manifest_path),
                "decoder_runtime_key": runtime_key,
                "decoder_training_objective": str(runtime.get("training_objective", "")).strip(),
            }
            return decision
        raise ValueError(
            "Retained decoder did not return a parseable or actionable decision: "
            f"{last_output}"
        )


HybridFallbackClient = HybridDecoderClient

# Backward-compatible alias while callers migrate from the old overloaded name.
TolbertFallbackClient = HybridDecoderClient


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
    serialized_state = json.dumps(state_payload, ensure_ascii=True, separators=(",", ":"))
    return (
        f"{decision_prompt}\n\n"
        "State payload JSON:\n"
        f"{serialized_state}\n\n"
        "Return a JSON object with keys thought, action, content, done. "
        "Keep thought to one short sentence."
    )


def _compact_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    task = state_payload["task"]
    compact_task = {
        "task_id": task.get("task_id"),
        "prompt": _truncate_text_value(task.get("prompt"), limit=320),
        "workspace_subdir": _truncate_text_value(task.get("workspace_subdir"), limit=120),
        "success_command": _truncate_text_value(task.get("success_command"), limit=200),
        "suggested_commands": _compact_string_list(
            task.get("suggested_commands", []),
            max_items=3,
            text_limit=160,
        ),
        "expected_files": _compact_string_list(
            task.get("expected_files", []),
            max_items=4,
            text_limit=80,
        ),
        "expected_output_substrings": _compact_string_list(
            task.get("expected_output_substrings", []),
            max_items=4,
            text_limit=80,
        ),
    }
    compact_history = []
    for step in state_payload.get("history", [])[-2:]:
        compact_history.append(
            {
                "index": step.get("index"),
                "action": step.get("action"),
                "content": _truncate_text_value(step.get("content"), limit=200),
                "verification": _compact_json_value(
                    step.get("verification", {}),
                    max_depth=1,
                    max_items=4,
                    text_limit=80,
                ),
            }
        )

    compact_context = None
    context_packet = state_payload.get("context_packet")
    if context_packet:
        compact_context = {
            "task": _compact_json_value(
                context_packet.get("task"),
                max_depth=1,
                max_items=4,
                text_limit=120,
            ),
            "control": _compact_json_value(
                context_packet.get("control"),
                max_depth=2,
                max_items=5,
                text_limit=100,
            ),
            "tolbert": _compact_json_value(
                context_packet.get("tolbert"),
                max_depth=1,
                max_items=4,
                text_limit=100,
            ),
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
            "verifier_contract": _compact_json_value(
                context_packet.get("verifier_contract"),
                max_depth=2,
                max_items=6,
                text_limit=100,
            ),
        }

    return {
        "task": compact_task,
        "history": compact_history,
        "recent_workspace_summary": _truncate_text_value(
            state_payload.get("recent_workspace_summary", ""),
            limit=240,
        ),
        "context_packet": compact_context,
        "retrieval_plan": _compact_json_value(
            state_payload.get("retrieval_plan", {}),
            max_depth=2,
            max_items=5,
            text_limit=100,
        ),
        "available_skills": _compact_string_list(
            state_payload.get("available_skills", []),
            max_items=3,
            text_limit=80,
        ),
        "graph_summary": _compact_json_value(
            state_payload.get("graph_summary", {}),
            max_depth=2,
            max_items=5,
            text_limit=100,
        ),
        "world_model_summary": _compact_json_value(
            state_payload.get("world_model_summary", {}),
            max_depth=2,
            max_items=5,
            text_limit=100,
        ),
        "latest_state_transition": _compact_json_value(
            state_payload.get("latest_state_transition", {}),
            max_depth=2,
            max_items=5,
            text_limit=100,
        ),
        "plan": _compact_string_list(
            state_payload.get("plan", []),
            max_items=4,
            text_limit=120,
        ),
        "active_subgoal": _truncate_text_value(state_payload.get("active_subgoal", ""), limit=160),
        "active_subgoal_diagnosis": _compact_json_value(
            state_payload.get("active_subgoal_diagnosis", {}),
            max_depth=2,
            max_items=5,
            text_limit=100,
        ),
        "acting_role": _truncate_text_value(state_payload.get("acting_role", ""), limit=60),
        "state_context_chunks": _compact_json_value(
            state_payload.get("state_context_chunks", [])[:6],
            max_depth=2,
            max_items=4,
            text_limit=100,
        ),
        "allowed_actions": _compact_string_list(
            state_payload.get("allowed_actions", []),
            max_items=4,
            text_limit=40,
        ),
        "context_compile_warning": _compact_json_value(
            state_payload.get("context_compile_warning"),
            max_depth=2,
            max_items=5,
            text_limit=120,
        ),
        "planner_recovery_brief": _truncate_text_value(
            state_payload.get("planner_recovery_brief", ""),
            limit=180,
        ),
        "software_work_phase_gate_brief": _truncate_text_value(
            state_payload.get("software_work_phase_gate_brief", ""),
            limit=180,
        ),
        "campaign_contract_brief": _truncate_text_value(
            state_payload.get("campaign_contract_brief", ""),
            limit=180,
        ),
        "planner_recovery_artifact": _compact_json_value(
            state_payload.get("planner_recovery_artifact"),
            max_depth=2,
            max_items=4,
            text_limit=100,
        ),
    }


def _minimal_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_state_payload(state_payload)
    compact["context_packet"] = (
        {
            "control": {
                "path_confidence": compact.get("context_packet", {}).get("control", {}).get("path_confidence", 0.0),
                "trust_retrieval": compact.get("context_packet", {}).get("control", {}).get("trust_retrieval", False),
                "retrieval_guidance": _compact_json_value(
                    compact.get("context_packet", {}).get("control", {}).get("retrieval_guidance", {}),
                    max_depth=1,
                    max_items=4,
                    text_limit=80,
                ),
                "selected_context_chunks": _compact_json_value(
                    compact.get("context_packet", {}).get("control", {}).get("selected_context_chunks", [])[:2],
                    max_depth=1,
                    max_items=2,
                    text_limit=80,
                ),
            },
            "verifier_contract": _compact_json_value(
                compact.get("context_packet", {}).get("verifier_contract", {}),
                max_depth=1,
                max_items=4,
                text_limit=80,
            ),
        }
        if compact.get("context_packet")
        else None
    )
    compact["graph_summary"] = None
    compact["world_model_summary"] = None
    compact["latest_state_transition"] = None
    compact["active_subgoal_diagnosis"] = None
    compact["state_context_chunks"] = None
    compact["planner_recovery_artifact"] = None
    return compact


def _lean_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    compact = _minimal_state_payload(state_payload)
    lean: dict[str, Any] = {
        "task": compact.get("task", {}),
        "history": [
            {
                "index": step.get("index"),
                "action": step.get("action"),
                "content": _truncate_text_value(step.get("content"), limit=160),
                "verification": _compact_json_value(
                    step.get("verification", {}),
                    max_depth=1,
                    max_items=2,
                    text_limit=60,
                ),
            }
            for step in compact.get("history", [])[-1:]
        ],
        "allowed_actions": compact.get("allowed_actions", []),
        "context_compile_warning": compact.get("context_compile_warning"),
        "software_work_phase_gate_brief": compact.get("software_work_phase_gate_brief"),
        "campaign_contract_brief": compact.get("campaign_contract_brief"),
    }
    context_packet = compact.get("context_packet")
    if context_packet:
        lean["context_packet"] = {
            "verifier_contract": context_packet.get("verifier_contract"),
        }
    active_subgoal = compact.get("active_subgoal")
    if active_subgoal:
        lean["active_subgoal"] = active_subgoal
    return lean


def _compact_retrieved_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": span.get("span_id"),
        "source_id": span.get("source_id"),
        "span_type": span.get("span_type"),
        "score": span.get("score"),
        "node_path": _compact_string_list(span.get("node_path", []), max_items=4, text_limit=80),
        "text": _truncate_text_value(span.get("text", ""), limit=120),
    }


def _truncate_text_value(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def _compact_string_list(values: object, *, max_items: int, text_limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    compacted = [_truncate_text_value(item, limit=text_limit) for item in values[:max_items]]
    remaining = len(values) - len(compacted)
    if remaining > 0:
        compacted.append(f"... ({remaining} more)")
    return compacted


def _compact_json_value(
    value: object,
    *,
    max_depth: int,
    max_items: int,
    text_limit: int,
) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text_value(value, limit=text_limit)
    if max_depth <= 0:
        if isinstance(value, dict):
            return {
                "__summary__": f"{len(value)} keys",
            }
        if isinstance(value, list):
            return {
                "__summary__": f"{len(value)} items",
            }
        return _truncate_text_value(repr(value), limit=text_limit)
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                compacted["__truncated_keys__"] = len(value) - max_items
                break
            compacted[str(key)] = _compact_json_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                text_limit=text_limit,
            )
        return compacted
    if isinstance(value, list):
        compacted_list = [
            _compact_json_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                text_limit=text_limit,
            )
            for item in value[:max_items]
        ]
        remaining = len(value) - len(compacted_list)
        if remaining > 0:
            compacted_list.append(f"... ({remaining} more)")
        return compacted_list
    return _truncate_text_value(repr(value), limit=text_limit)


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
            if isinstance(exc, error.HTTPError):
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                if error_body.strip():
                    exc = RuntimeError(f"{exc} body={error_body.strip()}")
            last_error = exc
            if attempt + 1 >= retry_attempts:
                break
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    raise RuntimeError(f"{error_label} failed after {retry_attempts} attempts: {last_error}")
