from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
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
        decision_max_tokens = self._decision_max_tokens_for_payload(state_payload)
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
                    return self._with_decoding_telemetry(parsed, last_data)
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
    _DECISION_MAX_TOKENS = (384, 256, 192, 128, 96, 64, 48, 32)

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
        raw_total_timeout = os.getenv("AGENT_KERNEL_LLM_DECISION_TOTAL_TIMEOUT_SECONDS", "").strip()
        try:
            total_timeout = (
                int(raw_total_timeout)
                if raw_total_timeout
                else max(int(timeout_seconds) * 6, 60)
            )
        except ValueError:
            total_timeout = max(int(timeout_seconds) * 6, 60)
        self.decision_total_timeout_seconds = max(1, total_timeout)
        raw_min_request_timeout = os.getenv("AGENT_KERNEL_LLM_MIN_REQUEST_TIMEOUT_SECONDS", "").strip()
        try:
            min_request_timeout = int(raw_min_request_timeout) if raw_min_request_timeout else 3
        except ValueError:
            min_request_timeout = 3
        self.min_request_timeout_seconds = max(1, min_request_timeout)

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        decision_max_tokens = self._decision_max_tokens_for_payload(state_payload)
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
            _render_prompt(
                decision_prompt=_ultra_lean_decision_prompt(decision_prompt),
                state_payload=_ultra_lean_state_payload(state_payload),
            ),
        ]
        last_data: dict[str, Any] | None = None
        last_error: Exception | None = None
        deadline = time.monotonic() + float(self.decision_total_timeout_seconds)
        for prompt in attempts:
            for max_tokens in decision_max_tokens:
                remaining_timeout = self._remaining_decision_timeout_or_raise(deadline)
                parsed: dict[str, Any] | None = None
                try:
                    data = self._chat_completion(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        use_json_schema=True,
                        max_tokens=max_tokens,
                        timeout_seconds=remaining_timeout,
                    )
                except RuntimeError as exc:
                    last_error = exc
                    remaining_timeout = self._remaining_decision_timeout_or_raise(deadline, from_exc=exc)
                    try:
                        # Fall back to prompt-only JSON if the server rejects structured output.
                        data = self._chat_completion(
                            system_prompt=system_prompt,
                            prompt=prompt,
                            use_json_schema=False,
                            max_tokens=max_tokens,
                            timeout_seconds=remaining_timeout,
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
                                timeout_seconds=self._remaining_decision_timeout_or_raise(deadline),
                            )
                        except RuntimeError as fallback_exc:
                            last_error = fallback_exc
                            if self._is_context_limit_error(str(fallback_exc)):
                                continue
                            raise
                        last_data = data
                        parsed = self._extract_decision(data)
                if parsed is not None:
                    return self._with_decoding_telemetry(parsed, last_data)
        if last_error is not None:
            raise ValueError(f"vLLM did not return a parseable JSON decision: {last_data}") from last_error
        raise ValueError(f"vLLM did not return a parseable JSON decision: {last_data}")

    def _decision_max_tokens_for_payload(self, state_payload: dict[str, Any]) -> tuple[int, ...]:
        raw_budget = state_payload.get("decision_token_budget")
        try:
            budget = int(raw_budget) if raw_budget is not None else 0
        except (TypeError, ValueError):
            budget = 0
        if _payload_needs_full_artifact_decision_budget(state_payload):
            if budget > max(self._DECISION_MAX_TOKENS):
                expanded = (budget,) + tuple(
                    max_tokens for max_tokens in self._DECISION_MAX_TOKENS if max_tokens < budget
                )
                return expanded
            return self._DECISION_MAX_TOKENS
        if budget <= 0:
            return self._DECISION_MAX_TOKENS
        min_json_decision_tokens = 32
        bounded = tuple(
            max_tokens
            for max_tokens in self._DECISION_MAX_TOKENS
            if max_tokens <= budget and max_tokens >= min_json_decision_tokens
        )
        return bounded or (max(1, budget),)

    @staticmethod
    def _decision_logprob_telemetry_enabled() -> bool:
        raw = os.getenv("AGENT_KERNEL_VLLM_DECISION_LOGPROBS", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @classmethod
    def _with_decoding_telemetry(cls, decision: dict[str, Any], data: dict[str, Any] | None) -> dict[str, Any]:
        telemetry = cls._chat_logprob_telemetry(data or {})
        if not telemetry:
            return decision
        metadata = decision.get("proposal_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        decision["proposal_metadata"] = {
            **metadata,
            "vllm_decoding_telemetry": telemetry,
        }
        return decision

    @staticmethod
    def _chat_logprob_telemetry(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return {}
        logprobs = choices[0].get("logprobs", {})
        if not isinstance(logprobs, dict):
            return {}
        content = logprobs.get("content", [])
        if not isinstance(content, list) or not content:
            return {}
        token_logprobs: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        tokens: list[str] = []
        for item in content[:64]:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", ""))
            if token:
                tokens.append(token)
            try:
                token_logprobs.append(float(item.get("logprob")))
            except (TypeError, ValueError):
                pass
            top = item.get("top_logprobs", [])
            if not isinstance(top, list) or len(top) < 2:
                continue
            values: list[float] = []
            for entry in top[:5]:
                if not isinstance(entry, dict):
                    continue
                try:
                    values.append(float(entry.get("logprob")))
                except (TypeError, ValueError):
                    continue
            if len(values) < 2:
                continue
            ordered = sorted(values, reverse=True)
            margins.append(float(ordered[0] - ordered[1]))
            max_logprob = ordered[0]
            probs = [math.exp(value - max_logprob) for value in values]
            total = sum(probs)
            if total > 0:
                normalized = [value / total for value in probs]
                entropies.append(float(-sum(p * math.log(max(p, 1e-45)) for p in normalized)))

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "token_count": len(content),
            "sampled_token_count": len(token_logprobs),
            "mean_logprob": _mean(token_logprobs),
            "min_logprob": float(min(token_logprobs)) if token_logprobs else 0.0,
            "mean_top_margin": _mean(margins),
            "mean_top_entropy": _mean(entropies),
            "low_margin_token_count": sum(1 for value in margins if value < 0.25),
            "tokens_preview": tokens[:16],
        }

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        use_json_schema: bool,
        max_tokens: int,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        request_timeout = max(1, int(timeout_seconds if timeout_seconds is not None else self.timeout_seconds))
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(1, int(max_tokens)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._decision_logprob_telemetry_enabled():
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
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
            timeout_seconds=min(max(1, int(self.timeout_seconds)), request_timeout),
            retry_attempts=self.retry_attempts,
            retry_backoff_seconds=self.retry_backoff_seconds,
            headers=_authorization_headers(self.api_key),
            error_label="vLLM request",
        )

    @staticmethod
    def _remaining_decision_timeout(deadline: float) -> int:
        return int(math.ceil(max(0.0, deadline - time.monotonic())))

    def _remaining_decision_timeout_or_raise(self, deadline: float, *, from_exc: Exception | None = None) -> int:
        remaining = self._remaining_decision_timeout(deadline)
        if remaining < self.min_request_timeout_seconds:
            message = (
                "vLLM decision exceeded total timeout "
                f"{self.decision_total_timeout_seconds}s"
            )
            if from_exc is not None:
                raise RuntimeError(message) from from_exc
            raise RuntimeError(message)
        return remaining

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


class ModelStackClient:
    """Client for the local model-stack token generation server."""

    _DECISION_MAX_TOKENS = (32, 64, 96, 128, 192, 256)
    _DEFAULT_PROMPT_TOKEN_BUDGET = 1536

    def __init__(
        self,
        host: str,
        model_name: str,
        timeout_seconds: int,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
        *,
        model_dir: str = "",
        tokenizer_path: str = "",
        repo_path: str = "",
        api_key: str = "",
    ) -> None:
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.model_dir = str(model_dir or "").strip()
        self.tokenizer_path = str(tokenizer_path or "").strip()
        self.repo_path = str(repo_path or "").strip()
        self.api_key = api_key.strip()
        self._tokenizer: Any | None = None

    def create_decision(
        self,
        *,
        system_prompt: str,
        decision_prompt: str,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = [
            _render_prompt(
                decision_prompt=f"System prompt:\n{system_prompt}\n\n{decision_prompt}",
                state_payload=_compact_state_payload(state_payload),
            ),
            _render_prompt(
                decision_prompt=f"System prompt:\n{system_prompt}\n\n{decision_prompt}",
                state_payload=_minimal_state_payload(state_payload),
            ),
            _render_prompt(
                decision_prompt=f"System prompt:\n{system_prompt}\n\n{decision_prompt}",
                state_payload=_lean_state_payload(state_payload),
            ),
        ]
        last_data: dict[str, Any] | None = None
        last_text = ""
        last_error: Exception | None = None
        for prompt in attempts:
            for max_tokens in self._decision_max_tokens():
                try:
                    data = self._generate_text(prompt=prompt, max_new_tokens=max_tokens)
                except RuntimeError as exc:
                    last_error = exc
                    continue
                last_data = data
                last_text = str(data.get("generated_text", ""))
                parsed = _extract_json_object(last_text)
                if parsed is not None:
                    decision = coerce_action_decision(parsed)
                    decision["decision_source"] = "model_stack"
                    return decision
        error_detail = f" last_error={last_error}" if last_error is not None else ""
        raise ValueError(
            "Model Stack did not return a parseable JSON decision: "
            f"text={last_text!r} response={last_data}{error_detail}"
        )

    def _generate_text(self, *, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        tokenizer = self._load_tokenizer()
        input_ids = list(tokenizer.encode(prompt))
        if not input_ids:
            raise ValueError("Model Stack tokenizer produced no input tokens")
        raw_input_token_count = len(input_ids)
        input_ids = self._budget_prompt_tokens(input_ids)
        data = _post_json(
            url=f"{self.host}/v1/generate",
            payload={
                "input_ids": [input_ids],
                "max_new_tokens": max(1, int(max_new_tokens)),
                "do_sample": False,
                "temperature": 0.0,
            },
            timeout_seconds=self.timeout_seconds,
            retry_attempts=self.retry_attempts,
            retry_backoff_seconds=self.retry_backoff_seconds,
            headers=_authorization_headers(self.api_key),
            error_label="Model Stack request",
        )
        output_ids = data.get("output_ids")
        if not isinstance(output_ids, list) or not output_ids or not isinstance(output_ids[0], list):
            raise ValueError(f"Model Stack response missing output_ids: {data}")
        generated_ids = [int(token_id) for token_id in output_ids[0][len(input_ids) :]]
        generated_text = str(tokenizer.decode(generated_ids)) if generated_ids else ""
        return {
            "generated_text": generated_text,
            "input_token_count": len(input_ids),
            "raw_input_token_count": raw_input_token_count,
            "prompt_truncated": raw_input_token_count != len(input_ids),
            "generated_token_count": len(generated_ids),
            "raw_response": data,
        }

    def _budget_prompt_tokens(self, input_ids: list[int]) -> list[int]:
        raw_budget = os.getenv("AGENT_KERNEL_MODEL_STACK_PROMPT_TOKEN_BUDGET", "").strip()
        try:
            budget = int(raw_budget) if raw_budget else self._DEFAULT_PROMPT_TOKEN_BUDGET
        except ValueError:
            budget = self._DEFAULT_PROMPT_TOKEN_BUDGET
        if budget <= 0 or len(input_ids) <= budget:
            return input_ids
        prefix_count = max(1, min(256, budget // 4))
        suffix_count = max(1, budget - prefix_count)
        return [*input_ids[:prefix_count], *input_ids[-suffix_count:]]

    def _decision_max_tokens(self) -> tuple[int, ...]:
        raw = os.getenv("AGENT_KERNEL_MODEL_STACK_DECISION_MAX_TOKENS", "").strip()
        if not raw:
            return self._DECISION_MAX_TOKENS
        budgets: list[int] = []
        for token in raw.split(","):
            try:
                value = int(token.strip())
            except ValueError:
                continue
            if value > 0:
                budgets.append(value)
        return tuple(budgets) if budgets else self._DECISION_MAX_TOKENS

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        tokenizer_root = self._tokenizer_root()
        repo_path = Path(self.repo_path) if self.repo_path else Path()
        if not repo_path.is_absolute() and self.repo_path:
            repo_path = Path.cwd() / repo_path
        if not repo_path.exists():
            raise RuntimeError(f"Model Stack repo path does not exist: {repo_path}")
        repo_path_text = str(repo_path)
        inserted = False
        if repo_path_text not in sys.path:
            sys.path.insert(0, repo_path_text)
            inserted = True
        try:
            from data.tokenizer import get_tokenizer  # type: ignore

            self._tokenizer = get_tokenizer(str(tokenizer_root))
            return self._tokenizer
        finally:
            if inserted:
                try:
                    sys.path.remove(repo_path_text)
                except ValueError:
                    pass

    def _tokenizer_root(self) -> Path:
        raw = self.tokenizer_path or self.model_dir
        if not raw:
            model_name_path = Path(self.model_name)
            if model_name_path.exists():
                raw = str(model_name_path)
        if not raw:
            raise RuntimeError(
                "provider='model_stack' requires AGENT_KERNEL_MODEL_STACK_TOKENIZER_PATH "
                "or AGENT_KERNEL_MODEL_STACK_MODEL_DIR for text/token conversion"
            )
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise RuntimeError(f"Model Stack tokenizer path does not exist: {path}")
        return path


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


def _payload_needs_full_artifact_decision_budget(state_payload: dict[str, Any]) -> bool:
    artifact_keys = (
        "artifact_repair_context",
        "artifact_materialization_guard",
        "artifact_repair_continue_guard",
        "artifact_required_identifier_guard",
        "artifact_semantic_repair_guard",
        "artifact_anchor_repair_guard",
        "artifact_diagnostic_repair",
        "artifact_action_handoff",
        "artifact_action_failure_memory",
        "artifact_placeholder_candidate_guard",
        "artifact_placeholder_statement_range_guard",
    )
    for key in artifact_keys:
        value = state_payload.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    active_subgoal = str(state_payload.get("active_subgoal", "") or "").lower()
    if "artifact" in active_subgoal or "patch.diff" in active_subgoal:
        return True
    task = state_payload.get("task")
    if isinstance(task, dict):
        expected_files = task.get("expected_files", [])
        if isinstance(expected_files, list) and any(str(path).strip() == "patch.diff" for path in expected_files):
            return True
    return False


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
        compact_step = {
            "index": step.get("index"),
            "action": step.get("action"),
            "content": _truncate_text_value(step.get("content"), limit=200),
            "decision_source": _truncate_text_value(step.get("decision_source", ""), limit=80),
            "verification": _compact_json_value(
                step.get("verification", {}),
                max_depth=1,
                max_items=4,
                text_limit=80,
            ),
        }
        command_result = _compact_command_result(step)
        if command_result:
            compact_step["command_result"] = command_result
        compact_history.append(compact_step)

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

    compact_payload = {
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
    for key in (
        "artifact_repair_context",
        "artifact_materialization_guard",
        "artifact_repair_continue_guard",
        "artifact_required_identifier_guard",
        "artifact_semantic_repair_guard",
        "artifact_anchor_repair_guard",
        "decoder_uncertainty",
    ):
        if state_payload.get(key) is not None:
            compact_payload[key] = _compact_json_value(
                state_payload.get(key),
                max_depth=2,
                max_items=8,
                text_limit=500,
            )
    return compact_payload


def _minimal_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_state_payload(state_payload)
    compact["history"] = [_minimal_history_step(step) for step in compact.get("history", [])[-2:]]
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
    if isinstance(compact.get("artifact_repair_context"), dict):
        compact["artifact_repair_context"] = _minimal_artifact_repair_context(compact["artifact_repair_context"])
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


def _ultra_lean_decision_prompt(decision_prompt: str) -> str:
    return (
        _truncate_block_value(decision_prompt, limit=700)
        + "\nProvider recovery mode: return only a valid minified JSON decision. "
        "Use action code_execute when an expected artifact is still missing; do not include prose outside JSON."
    )


def _ultra_lean_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    task = state_payload.get("task", {}) if isinstance(state_payload.get("task"), dict) else {}
    history = state_payload.get("history", [])
    history_steps = history if isinstance(history, list) else []
    last_step = history_steps[-1] if history_steps and isinstance(history_steps[-1], dict) else {}
    verification = last_step.get("verification", {}) if isinstance(last_step.get("verification"), dict) else {}
    payload: dict[str, Any] = {
        "task": {
            "task_id": task.get("task_id"),
            "prompt": _truncate_text_value(task.get("prompt", ""), limit=180),
            "expected_files": _compact_string_list(task.get("expected_files", []), max_items=3, text_limit=80),
            "workspace_subdir": _truncate_text_value(task.get("workspace_subdir", ""), limit=100),
        },
        "latest_step": {
            "action": last_step.get("action"),
            "content": _truncate_text_value(last_step.get("content", ""), limit=180),
            "decision_source": _truncate_text_value(last_step.get("decision_source", ""), limit=80),
            "verification_reasons": _compact_string_list(
                list(verification.get("reasons", []) or []),
                max_items=3,
                text_limit=100,
            ),
        },
        "active_subgoal": _truncate_text_value(state_payload.get("active_subgoal", ""), limit=140),
        "allowed_actions": ["code_execute", "respond"],
    }
    for key in (
        "artifact_repair_context",
        "artifact_materialization_guard",
        "artifact_repair_continue_guard",
        "artifact_anchor_repair_guard",
        "decoder_uncertainty",
    ):
        value = state_payload.get(key)
        if isinstance(value, dict):
            if key == "artifact_anchor_repair_guard":
                payload[key] = _minimal_artifact_anchor_repair_guard(value)
            elif key == "decoder_uncertainty":
                payload[key] = _compact_json_value(value, max_depth=2, max_items=8, text_limit=180)
            else:
                payload[key] = _minimal_artifact_repair_context(value)
    return payload


def _compact_retrieved_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": span.get("span_id"),
        "source_id": span.get("source_id"),
        "span_type": span.get("span_type"),
        "score": span.get("score"),
        "node_path": _compact_string_list(span.get("node_path", []), max_items=4, text_limit=80),
        "text": _truncate_text_value(span.get("text", ""), limit=120),
    }


def _minimal_history_step(step: object) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    return {
        "index": step.get("index"),
        "action": step.get("action"),
        "content": _truncate_text_value(step.get("content"), limit=160),
        "decision_source": _truncate_text_value(step.get("decision_source", ""), limit=80),
        "verification": _compact_json_value(
            step.get("verification", {}),
            max_depth=1,
            max_items=3,
            text_limit=70,
        ),
    }


def _minimal_artifact_repair_context(context: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "artifact_path",
        "builder_command",
        "required_command_shape",
        "last_source_path",
        "source_lines_path",
    ):
        value = context.get(key)
        if value:
            compact[key] = _truncate_text_value(value, limit=180)
    allowed_source_paths = context.get("allowed_source_paths", [])
    if isinstance(allowed_source_paths, list):
        compact["allowed_source_paths"] = _compact_string_list(
            allowed_source_paths,
            max_items=4,
            text_limit=120,
        )
    return compact


def _minimal_artifact_anchor_repair_guard(context: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "fixed_path",
        "rejected_reason",
        "required_response_content",
        "previous_anchor_response",
    ):
        value = context.get(key)
        if value:
            compact[key] = _truncate_text_value(value, limit=220)
    valid_line_numbers = context.get("valid_line_numbers", [])
    if isinstance(valid_line_numbers, list):
        compact["valid_line_numbers"] = [
            int(value)
            for value in valid_line_numbers[:32]
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        ]
    source_lines_excerpt = str(context.get("source_lines_excerpt", "") or "").strip()
    if source_lines_excerpt:
        compact["source_lines_excerpt"] = _truncate_block_value(source_lines_excerpt, limit=700)
    edit_windows = str(context.get("edit_windows", "") or "").strip()
    if edit_windows:
        compact["edit_windows"] = _truncate_block_value(edit_windows, limit=500)
    return compact


def _compact_command_result(step: dict[str, Any]) -> dict[str, Any]:
    result = step.get("command_result")
    if not isinstance(result, dict):
        return {}
    compact: dict[str, Any] = {
        "exit_code": result.get("exit_code"),
        "timed_out": bool(result.get("timed_out", False)),
    }
    content = str(step.get("content", "") or "")
    stdout_limit = 1200 if _command_reads_source_context(content) else 600
    stderr_limit = 600
    stdout = str(result.get("stdout", "") or "")
    stderr = str(result.get("stderr", "") or "")
    if stdout:
        compact["stdout_preview"] = _truncate_block_value(stdout, limit=stdout_limit)
    if stderr:
        compact["stderr_preview"] = _truncate_block_value(stderr, limit=stderr_limit)
    return compact


def _command_reads_source_context(command: str) -> bool:
    normalized = str(command or "")
    return "source_lines/" in normalized or "source_context/" in normalized


def _truncate_block_value(value: object, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    if limit <= 32:
        return text[:limit]
    keep = max(8, (limit - 24) // 2)
    return f"{text[:keep].rstrip()}\n...<truncated>...\n{text[-keep:].lstrip()}"


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
    transport = os.getenv("AGENT_KERNEL_LLM_HTTP_TRANSPORT", "").strip().lower()
    if transport == "curl" or _needs_subprocess_wall_timeout():
        return _post_json_with_curl(
            url=url,
            payload=payload,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            headers=headers,
            error_label=error_label,
        )
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
            with _request_wall_timeout(timeout_seconds, error_label):
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


def _needs_subprocess_wall_timeout() -> bool:
    """urllib socket timeouts are not enough for worker-thread LLM calls."""
    return threading.current_thread() is not threading.main_thread()


def _post_json_with_curl(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
    headers: dict[str, str],
    error_label: str,
) -> dict[str, Any]:
    body = json.dumps(payload)
    timeout = max(1, int(timeout_seconds))
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--max-time",
        str(timeout),
        "--request",
        "POST",
    ]
    for name, value in headers.items():
        command.extend(["--header", f"{name}: {value}"])
    command.extend(["--data-binary", "@-", url])
    last_error: Exception | None = None
    for attempt in range(max(1, int(retry_attempts))):
        try:
            completed = subprocess.run(
                command,
                input=body,
                text=True,
                capture_output=True,
                timeout=timeout + 2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = exc
        else:
            if completed.returncode == 0:
                try:
                    return json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    last_error = exc
            else:
                detail = "\n".join(
                    part
                    for part in [
                        completed.stderr.strip(),
                        completed.stdout.strip(),
                    ]
                    if part
                )
                last_error = RuntimeError(
                    f"curl exited {completed.returncode}: {detail[:1000]}"
                )
        if attempt + 1 >= max(1, int(retry_attempts)):
            break
        if retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds * (attempt + 1))
    raise RuntimeError(f"{error_label} failed after {retry_attempts} attempts: {last_error}")


class _request_wall_timeout:
    def __init__(self, timeout_seconds: int, error_label: str) -> None:
        self.timeout_seconds = max(0, int(timeout_seconds))
        self.error_label = str(error_label).strip() or "request"
        self._active = False
        self._previous_handler: Any = None
        self._previous_timer: tuple[float, float] = (0.0, 0.0)

    def __enter__(self) -> "_request_wall_timeout":
        if (
            self.timeout_seconds <= 0
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
        ):
            return self
        self._active = True
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        self._previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._active:
            return False
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        delay, interval = self._previous_timer
        if delay > 0:
            signal.setitimer(signal.ITIMER_REAL, delay, interval)
        return False

    def _raise_timeout(self, signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(f"{self.error_label} exceeded wall timeout {self.timeout_seconds}s")
