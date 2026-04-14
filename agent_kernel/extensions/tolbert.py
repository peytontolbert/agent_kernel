"""Seed TOLBERT compiler runtime for the broader encoder-latent-decoder family.

This module keeps the original strict context-compilation service alive inside the
kernel. In the current repo, that service is the encoder/retrieval slice of the
larger TOLBERT universal family; later retained family members can add learned
latent dynamics and decoder control without replacing this seed interface first.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any, Protocol
from uuid import uuid4

from ..config import KernelConfig
from ..learning_compiler import matching_learning_candidates
from .improvement.retrieval_improvement import retained_retrieval_overrides
from ..schemas import ContextPacket
from ..state import AgentState
from .tolbert_assets import retained_tolbert_runtime_paths
from ..world_model import WorldModel


class TolbertQueryClient(Protocol):
    def query(
        self,
        *,
        query_text: str,
        branch_results: int,
        global_results: int,
        confidence_threshold: float,
        top_branches: int,
        branch_confidence_margin: float,
        low_confidence_widen_threshold: float,
        ancestor_branch_levels: int,
        low_confidence_branch_multiplier: float,
        low_confidence_global_multiplier: float,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        ...


class TolbertServiceClient:
    def __init__(
        self,
        *,
        config: KernelConfig,
        repo_root: Path,
        runtime_paths: dict[str, Any] | None = None,
        service_name: str = "tolbert",
    ) -> None:
        self.repo_root = repo_root
        self.service_timeout_seconds = config.tolbert_service_timeout_seconds
        self.service_startup_timeout_seconds = max(
            float(self.service_timeout_seconds),
            float(os.getenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_TIMEOUT_SECONDS", "45")),
        )
        self.service_startup_grace_seconds = max(
            0.0,
            float(os.getenv("AGENT_KERNEL_TOLBERT_SERVICE_STARTUP_GRACE_SECONDS", "10")),
        )
        self.service_startup_attempts = max(1, int(config.tolbert_service_startup_attempts))
        resolved_runtime_paths = runtime_paths or retained_tolbert_runtime_paths(config, repo_root=repo_root)
        labels = {
            "tolbert_config_path": (
                "AGENT_KERNEL_TOLBERT_CONFIG_PATH"
                if runtime_paths is None
                else f"{service_name}.config_path"
            ),
            "tolbert_checkpoint_path": (
                "AGENT_KERNEL_TOLBERT_CHECKPOINT_PATH"
                if runtime_paths is None
                else f"{service_name}.checkpoint_path"
            ),
            "tolbert_nodes_path": (
                "AGENT_KERNEL_TOLBERT_NODES_PATH"
                if runtime_paths is None
                else f"{service_name}.nodes_path"
            ),
            "tolbert_label_map_path": (
                "AGENT_KERNEL_TOLBERT_LABEL_MAP_PATH"
                if runtime_paths is None
                else f"{service_name}.label_map_path"
            ),
            "tolbert_source_spans_paths": (
                "AGENT_KERNEL_TOLBERT_SOURCE_SPANS_PATHS"
                if runtime_paths is None
                else f"{service_name}.source_spans_paths"
            ),
            "tolbert_cache_paths": (
                "AGENT_KERNEL_TOLBERT_CACHE_PATHS"
                if runtime_paths is None
                else f"{service_name}.cache_paths"
            ),
        }
        script_path = repo_root / "scripts" / "tolbert_service.py"
        if not script_path.exists():
            raise FileNotFoundError(f"TOLBERT service script not found: {script_path}")

        self.command = [
            config.tolbert_python_bin,
            "-u",
            str(script_path),
            "--repo-root",
            str(repo_root),
            "--config",
            self._require_path(
                resolved_runtime_paths["tolbert_config_path"],
                labels["tolbert_config_path"],
            ),
            "--checkpoint",
            self._require_path(
                resolved_runtime_paths["tolbert_checkpoint_path"],
                labels["tolbert_checkpoint_path"],
            ),
            "--nodes",
            self._require_path(
                resolved_runtime_paths["tolbert_nodes_path"],
                labels["tolbert_nodes_path"],
            ),
            "--label-map",
            self._require_path(
                resolved_runtime_paths["tolbert_label_map_path"],
                labels["tolbert_label_map_path"],
            ),
            "--device",
            config.tolbert_device,
        ]
        for source_path in resolved_runtime_paths["tolbert_source_spans_paths"]:
            self.command.extend(
                [
                    "--source-spans",
                    self._require_path(source_path, labels["tolbert_source_spans_paths"]),
                ]
            )
        for cache_path in resolved_runtime_paths["tolbert_cache_paths"]:
            self.command.extend(
                [
                    "--cache-path",
                    self._require_path(cache_path, labels["tolbert_cache_paths"]),
                ]
            )

        self.env = dict(os.environ)
        self.env["PYTHONNOUSERSITE"] = "1"
        self.process: subprocess.Popen[str] | None = None

    @staticmethod
    def _is_retryable_startup_failure(error_text: str) -> bool:
        normalized = str(error_text).strip().lower()
        if not normalized:
            return False
        return (
            "tolbert service failed to become ready" in normalized
            or "tolbert service exited before startup ready" in normalized
        )

    def close(self) -> None:
        self._terminate_process()

    def query(
        self,
        *,
        query_text: str,
        branch_results: int,
        global_results: int,
        confidence_threshold: float,
        top_branches: int,
        branch_confidence_margin: float,
        low_confidence_widen_threshold: float,
        ancestor_branch_levels: int,
        low_confidence_branch_multiplier: float,
        low_confidence_global_multiplier: float,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        process = self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("TOLBERT service pipes are unavailable.")

        request = {
            "query_text": query_text,
            "branch_results": branch_results,
            "global_results": global_results,
            "confidence_threshold": confidence_threshold,
            "top_branches": top_branches,
            "branch_confidence_margin": branch_confidence_margin,
            "low_confidence_widen_threshold": low_confidence_widen_threshold,
            "ancestor_branch_levels": ancestor_branch_levels,
            "low_confidence_branch_multiplier": low_confidence_branch_multiplier,
            "low_confidence_global_multiplier": low_confidence_global_multiplier,
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        timeout_window = self.service_timeout_seconds
        if timeout_seconds is not None:
            timeout_window = max(0.05, min(float(timeout_seconds), float(self.service_timeout_seconds)))
        events = selector.select(timeout_window)
        selector.close()
        if not events:
            self._reset_process()
            raise RuntimeError(
                f"TOLBERT service timed out after {timeout_window:.3f} seconds."
            )
        line = process.stdout.readline()
        if not line:
            stderr = ""
            if process.stderr is not None:
                try:
                    stderr = process.stderr.read()
                except Exception:
                    stderr = ""
            raise RuntimeError(
                f"TOLBERT service exited unexpectedly with code {process.poll()}. {stderr}".strip()
            )

        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None or process.poll() is not None:
            process = self._spawn_process()
            self.process = process
        return process

    def _spawn_process(self) -> subprocess.Popen[str]:
        last_error: RuntimeError | None = None
        for attempt_index in range(self.service_startup_attempts):
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.repo_root,
                env=self.env,
            )
            try:
                self._wait_for_process_ready(process)
                return process
            except RuntimeError as exc:
                last_error = exc
                retryable = self._is_retryable_startup_failure(str(exc))
                if not retryable or attempt_index + 1 >= self.service_startup_attempts:
                    break
        if last_error is None:
            raise RuntimeError("TOLBERT service failed to start for an unknown reason.")
        if self.service_startup_attempts <= 1:
            raise last_error
        raise RuntimeError(
            f"{last_error} (attempted startup {self.service_startup_attempts} times)"
        ) from last_error

    def _reset_process(self) -> None:
        self._terminate_process()
        self.process = self._spawn_process()

    def _terminate_process(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self.process = None

    def _wait_for_process_ready(self, process: subprocess.Popen[str]) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        selector = selectors.DefaultSelector()
        selector.register(stderr, selectors.EVENT_READ)
        startup_timeout_seconds = max(0.05, float(self.service_startup_timeout_seconds))
        startup_grace_seconds = max(0.0, float(self.service_startup_grace_seconds))
        deadline = time.monotonic() + startup_timeout_seconds
        grace_deadline = None
        buffered_stderr: list[str] = []
        try:
            while True:
                if process.poll() is not None:
                    remainder = ""
                    try:
                        remainder = stderr.read()
                    except Exception:
                        remainder = ""
                    details = "".join(buffered_stderr) + remainder
                    raise RuntimeError(
                        f"TOLBERT service exited before startup ready with code {process.poll()}. {details}".strip()
                    )
                now = time.monotonic()
                if grace_deadline is None and now >= deadline:
                    if startup_grace_seconds <= 0.0:
                        self._terminate_process_handle(process)
                        raise RuntimeError(
                            f"TOLBERT service failed to become ready after {startup_timeout_seconds:.3f} seconds."
                        )
                    grace_deadline = now + startup_grace_seconds
                if grace_deadline is not None and now >= grace_deadline:
                    self._terminate_process_handle(process)
                    raise RuntimeError(
                        "TOLBERT service failed to become ready after "
                        f"{startup_timeout_seconds:.3f} seconds "
                        f"(plus {startup_grace_seconds:.3f} seconds grace)."
                    )
                remaining = (grace_deadline if grace_deadline is not None else deadline) - now
                events = selector.select(remaining)
                if not events:
                    continue
                line = stderr.readline()
                if not line:
                    continue
                buffered_stderr.append(line)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "startup_ready":
                    return
                startup_error = str(payload.get("startup_error", "")).strip()
                if startup_error:
                    self._terminate_process_handle(process)
                    raise RuntimeError(startup_error)
        finally:
            selector.close()

    @staticmethod
    def _terminate_process_handle(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _require_path(self, raw_path: str | None, env_name: str) -> str:
        if not raw_path:
            raise RuntimeError(f"{env_name} is required when TOLBERT context is enabled.")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.repo_root / path
        if not path.exists():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        return str(path)


class TolbertContextCompiler:
    _CONTEXT_COMPILE_SUBPHASE_BUDGET_FACTORS: dict[str, float] = {
        "query_build": 0.5,
        "tolbert_query": 0.8,
        "skill_query": 0.6,
        "skill_rank": 0.4,
        "tool_query": 0.5,
        "tool_plan": 0.4,
        "verifier_query": 0.4,
        "retrieval_normalize": 0.6,
        "paper_research_query": 1.0,
        "paper_research_merge": 0.5,
        "chunk_select": 0.5,
        "guidance_build": 0.5,
        "complete": 0.5,
    }

    def __init__(
        self,
        *,
        config: KernelConfig,
        repo_root: Path,
        client: TolbertQueryClient | None = None,
        research_client: TolbertQueryClient | None = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.client = client or TolbertServiceClient(config=config, repo_root=repo_root)
        self.research_client = research_client
        self._research_client_error = ""
        self._research_runtime_paths = (
            None
            if client is not None or research_client is not None
            else _paper_research_runtime_paths(config, repo_root=repo_root)
        )
        self.world_model = WorldModel(config=self.config)
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def compile(self, state: AgentState) -> ContextPacket:
        compile_started_at = time.monotonic()
        retrieval_config = self._retrieval_config()
        self._emit_compile_progress(
            "query_build",
            compile_started_at=compile_started_at,
            subphase_started_at=compile_started_at,
        )
        query_text = self._build_query_text(state)
        self._enforce_compile_budget(
            "query_build",
            compile_started_at=compile_started_at,
            subphase_started_at=compile_started_at,
        )
        research_requested = self._should_query_paper_research(state, query_text=query_text)
        tolbert_query_started_at = time.monotonic()
        self._emit_compile_progress(
            "tolbert_query",
            compile_started_at=compile_started_at,
            subphase_started_at=tolbert_query_started_at,
            paper_research_requested=research_requested,
        )
        tolbert_query_kwargs = {
            "query_text": query_text,
            "branch_results": int(retrieval_config["tolbert_branch_results"]),
            "global_results": int(retrieval_config["tolbert_global_results"]),
            "confidence_threshold": float(retrieval_config["tolbert_confidence_threshold"]),
            "top_branches": int(retrieval_config["tolbert_top_branches"]),
            "branch_confidence_margin": float(retrieval_config["tolbert_branch_confidence_margin"]),
            "low_confidence_widen_threshold": float(retrieval_config["tolbert_low_confidence_widen_threshold"]),
            "ancestor_branch_levels": int(retrieval_config["tolbert_ancestor_branch_levels"]),
            "low_confidence_branch_multiplier": float(retrieval_config["tolbert_low_confidence_branch_multiplier"]),
            "low_confidence_global_multiplier": float(retrieval_config["tolbert_low_confidence_global_multiplier"]),
        }
        tolbert_query_budget = self._context_compile_subphase_budget_seconds("tolbert_query")
        if tolbert_query_budget is not None:
            tolbert_query_kwargs["timeout_seconds"] = tolbert_query_budget
        tolbert_result = self.client.query(
            **tolbert_query_kwargs,
        )
        self._enforce_compile_budget(
            "tolbert_query",
            compile_started_at=compile_started_at,
            subphase_started_at=tolbert_query_started_at,
        )

        recent_steps = [
            {
                "action": step.action,
                "content": step.content,
                "verification": step.verification,
            }
            for step in state.history[-3:]
        ]
        retrieval_normalize_started_at = time.monotonic()
        self._emit_compile_progress(
            "retrieval_normalize",
            compile_started_at=compile_started_at,
            subphase_started_at=retrieval_normalize_started_at,
        )
        retrieval = self._normalize_retrieval(
            self._merge_carried_retrieval(
                tolbert_result["retrieval"],
                state=state,
            ),
            state=state,
            level_focus=str(tolbert_result.get("level_focus", "skill")),
            retrieval_config=retrieval_config,
            compile_started_at=compile_started_at,
        )
        self._enforce_compile_budget(
            "retrieval_normalize",
            compile_started_at=compile_started_at,
            subphase_started_at=retrieval_normalize_started_at,
        )
        research_backend = ""
        research_index_shards: list[str] = []
        if research_requested:
            paper_research_query_started_at = time.monotonic()
            self._emit_compile_progress(
                "paper_research_query",
                compile_started_at=compile_started_at,
                subphase_started_at=paper_research_query_started_at,
            )
            research_result = self._query_paper_research(
                query_text=query_text,
                compile_started_at=compile_started_at,
            )
            self._enforce_compile_budget(
                "paper_research_query",
                compile_started_at=compile_started_at,
                subphase_started_at=paper_research_query_started_at,
            )
            if research_result is not None:
                research_backend = str(research_result.get("backend", ""))
                research_index_shards = [str(value) for value in research_result.get("index_shards", [])]
                paper_research_merge_started_at = time.monotonic()
                self._emit_compile_progress(
                    "paper_research_merge",
                    compile_started_at=compile_started_at,
                    subphase_started_at=paper_research_merge_started_at,
                    paper_research_backend=research_backend,
                )
                research_retrieval = self._normalize_retrieval(
                    research_result.get("retrieval", {}),
                    state=state,
                    level_focus=str(research_result.get("level_focus", "skill")),
                    retrieval_config=retrieval_config,
                    compile_started_at=compile_started_at,
                )
                retrieval = self._merge_research_retrieval(
                    retrieval,
                    research_retrieval,
                    state=state,
                    level_focus=str(research_result.get("level_focus", "skill")),
                    retrieval_config=retrieval_config,
                )
                self._enforce_compile_budget(
                    "paper_research_merge",
                    compile_started_at=compile_started_at,
                    subphase_started_at=paper_research_merge_started_at,
                )
        chunk_select_started_at = time.monotonic()
        self._emit_compile_progress(
            "chunk_select",
            compile_started_at=compile_started_at,
            subphase_started_at=chunk_select_started_at,
        )
        selected_context_chunks = self._select_context_chunks(
            retrieval,
            state=state,
            level_focus=str(tolbert_result.get("level_focus", "skill")),
            retrieval_config=retrieval_config,
        )
        self._enforce_compile_budget(
            "chunk_select",
            compile_started_at=compile_started_at,
            subphase_started_at=chunk_select_started_at,
        )
        guidance_build_started_at = time.monotonic()
        self._emit_compile_progress(
            "guidance_build",
            compile_started_at=compile_started_at,
            subphase_started_at=guidance_build_started_at,
        )
        retrieval_guidance = self._build_retrieval_guidance(retrieval, state=state)
        path_confidence = self._path_confidence(tolbert_result["path_prediction"])
        trust_retrieval = path_confidence >= float(retrieval_config["tolbert_deterministic_command_confidence"])
        if not trust_retrieval and self._workflow_guarded_guidance_is_trustworthy(state, retrieval_guidance):
            trust_retrieval = True
        self._enforce_compile_budget(
            "guidance_build",
            compile_started_at=compile_started_at,
            subphase_started_at=guidance_build_started_at,
        )
        tool_plan_started_at = time.monotonic()
        self._emit_compile_progress(
            "tool_plan",
            compile_started_at=compile_started_at,
            subphase_started_at=tool_plan_started_at,
        )
        self._enforce_compile_budget(
            "tool_plan",
            compile_started_at=compile_started_at,
            subphase_started_at=tool_plan_started_at,
        )
        complete_started_at = time.monotonic()
        self._emit_compile_progress(
            "complete",
            compile_started_at=compile_started_at,
            subphase_started_at=complete_started_at,
            paper_research_used=bool(research_backend),
        )
        packet = ContextPacket(
            request_id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            task={
                "goal": state.task.prompt,
                "completion_criteria": state.task.success_command,
                "workspace_summary": state.recent_workspace_summary,
                "recent_steps": json.dumps(recent_steps),
            },
            control={
                "mode": "verify",
                "level_focus": tolbert_result["level_focus"],
                "window_budget_tokens": 2048,
                "retrieval_k": int(retrieval_config["tolbert_branch_results"]),
                "confidence_threshold": float(retrieval_config["tolbert_confidence_threshold"]),
                "path_confidence": path_confidence,
                "trust_retrieval": trust_retrieval,
                "selected_branch_level": tolbert_result.get("selected_branch_level"),
                "branch_candidates": tolbert_result.get("branch_candidates", []),
                "selected_context_chunks": selected_context_chunks,
                "paper_research_requested": research_requested,
                "paper_research_used": bool(research_backend),
                "paper_research_backend": research_backend,
                "paper_research_error": self._research_client_error,
                "context_chunk_budget": {
                    "max_chunks": int(retrieval_config["tolbert_context_max_chunks"]),
                    "char_budget": self.config.tolbert_context_char_budget,
                },
                "context_compile_budget_seconds": max(
                    0.0,
                    float(self.config.tolbert_context_compile_budget_seconds or 0.0),
                ),
                "context_compile_elapsed_seconds": round(
                    max(0.0, time.monotonic() - compile_started_at),
                    4,
                ),
                "retrieval_guidance": retrieval_guidance,
            },
            tolbert={
                "path_prediction": tolbert_result["path_prediction"],
                "backend": tolbert_result["backend"],
                "device": tolbert_result.get("device", self.config.tolbert_device),
                "index_shards": tolbert_result["index_shards"],
                "auxiliary_backends": [
                    {
                        "backend": research_backend,
                        "index_shards": research_index_shards,
                    }
                ]
                if research_backend
                else [],
            },
            retrieval=retrieval,
            verifier_contract={
                "success_command": state.task.success_command,
                "expected_files": state.task.expected_files,
                "expected_output_substrings": state.task.expected_output_substrings,
                "forbidden_files": state.task.forbidden_files,
                "forbidden_output_substrings": state.task.forbidden_output_substrings,
                "expected_file_contents": state.task.expected_file_contents,
            },
        )
        self._enforce_compile_budget(
            "complete",
            compile_started_at=compile_started_at,
            subphase_started_at=complete_started_at,
        )
        return packet

    def close(self) -> None:
        for runtime_client in (self.client, self.research_client):
            close = getattr(runtime_client, "close", None)
            if callable(close):
                close()

    def _build_query_text(self, state: AgentState) -> str:
        parts = [state.task.prompt]
        if state.recent_workspace_summary:
            parts.append(state.recent_workspace_summary)
        if state.history:
            last_step = state.history[-1]
            parts.append(last_step.content)
            if last_step.command_result:
                parts.append(json.dumps(last_step.command_result))
            reasons = last_step.verification.get("reasons", [])
            if reasons:
                parts.append(" ".join(reasons))
        carry_summary = self._retrieval_carry_summary(state)
        if carry_summary:
            parts.append(carry_summary)
        return "\n".join(part for part in parts if part).strip()

    def _merge_carried_retrieval(
        self,
        retrieval: dict[str, list[dict[str, Any]]],
        *,
        state: AgentState,
    ) -> dict[str, list[dict[str, Any]]]:
        seed = self._retrieval_carry_seed(state)
        if not seed:
            return retrieval
        merged = {
            "branch_scoped": list(retrieval.get("branch_scoped", [])),
            "fallback_scoped": list(retrieval.get("fallback_scoped", [])),
            "global": list(retrieval.get("global", [])),
        }
        seen = {
            str(item.get("span_id", "")).strip()
            for bucket in merged.values()
            for item in bucket
            if isinstance(item, dict)
        }
        carry_items = [
            dict(item)
            for item in seed.get("items", [])
            if isinstance(item, dict)
        ]
        for item in reversed(carry_items):
            span_id = str(item.get("span_id", "")).strip()
            if not span_id or span_id in seen:
                continue
            merged["branch_scoped"].insert(0, item)
            seen.add(span_id)
        return merged

    def _normalize_retrieval(
        self,
        retrieval: dict[str, list[dict[str, Any]]],
        *,
        state: AgentState,
        level_focus: str,
        retrieval_config: dict[str, object],
        compile_started_at: float,
    ) -> dict[str, list[dict[str, Any]]]:
        skill_query_started_at = time.monotonic()
        self._emit_compile_progress(
            "skill_query",
            compile_started_at=compile_started_at,
            subphase_started_at=skill_query_started_at,
        )
        branch_scoped = self._dedupe_and_rank(
            retrieval.get("branch_scoped", []),
            state=state,
            level_focus=level_focus,
            retrieval_config=retrieval_config,
        )
        branch_scoped = self._apply_source_diversity(branch_scoped, retrieval_config=retrieval_config)
        seen = {item["span_id"] for item in branch_scoped}
        self._enforce_compile_budget(
            "skill_query",
            compile_started_at=compile_started_at,
            subphase_started_at=skill_query_started_at,
        )
        tool_query_started_at = time.monotonic()
        self._emit_compile_progress(
            "tool_query",
            compile_started_at=compile_started_at,
            subphase_started_at=tool_query_started_at,
        )
        fallback_scoped = [
            item
            for item in self._dedupe_and_rank(
                retrieval.get("fallback_scoped", []),
                state=state,
                level_focus=level_focus,
                retrieval_config=retrieval_config,
            )
            if item["span_id"] not in seen
        ]
        fallback_scoped = self._apply_source_diversity(fallback_scoped, retrieval_config=retrieval_config)
        seen |= {item["span_id"] for item in fallback_scoped}
        self._enforce_compile_budget(
            "tool_query",
            compile_started_at=compile_started_at,
            subphase_started_at=tool_query_started_at,
        )
        skill_rank_started_at = time.monotonic()
        self._emit_compile_progress(
            "skill_rank",
            compile_started_at=compile_started_at,
            subphase_started_at=skill_rank_started_at,
        )
        global_results = [
            item
            for item in self._dedupe_and_rank(
                retrieval.get("global", []),
                state=state,
                level_focus=level_focus,
                retrieval_config=retrieval_config,
            )
            if item["span_id"] not in seen
        ]
        global_results = self._apply_source_diversity(global_results, retrieval_config=retrieval_config)
        self._enforce_compile_budget(
            "skill_rank",
            compile_started_at=compile_started_at,
            subphase_started_at=skill_rank_started_at,
        )
        return {
            "branch_scoped": branch_scoped[: int(retrieval_config["tolbert_branch_results"])],
            "fallback_scoped": fallback_scoped[: int(retrieval_config["tolbert_branch_results"])],
            "global": global_results[: int(retrieval_config["tolbert_global_results"])],
        }

    def _dedupe_and_rank(
        self,
        results: list[dict[str, Any]],
        *,
        state: AgentState,
        level_focus: str,
        retrieval_config: dict[str, object],
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for item in results:
            span_id = str(item.get("span_id", ""))
            if not span_id:
                continue
            current = deduped.get(span_id)
            if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
                deduped[span_id] = item
        return sorted(
            deduped.values(),
            key=lambda item: (
                -self._retrieval_rank(item, state=state, level_focus=level_focus, retrieval_config=retrieval_config),
                -float(item.get("score", 0.0)),
                str(item.get("span_id", "")),
            ),
        )

    def _retrieval_rank(
        self,
        item: dict[str, Any],
        *,
        state: AgentState,
        level_focus: str,
        retrieval_config: dict[str, object],
    ) -> float:
        span_type = str(item.get("span_type", ""))
        metadata = item.get("metadata") or {}
        task = state.task
        task_id = str(task.task_id)
        source_task = str(task.metadata.get("source_task", ""))
        distractor_tasks = {str(value) for value in task.metadata.get("distractor_tasks", [])}
        item_task_id = str(metadata.get("task_id", ""))
        rank = float(item.get("score", 0.0)) * self.config.tolbert_semantic_score_weight
        if item_task_id == task_id:
            rank += self.config.tolbert_task_match_weight
        if source_task and item_task_id == source_task:
            rank += self.config.tolbert_source_task_weight
        if distractor_tasks and item_task_id in distractor_tasks:
            rank -= float(retrieval_config["tolbert_distractor_penalty"])
        if bool(metadata.get("carried_retrieval", False)):
            rank += 6.0
        broad_focus = level_focus in {"domain", "subdomain", "repo", "pillar", "theme", "corpus"}
        if span_type == "agent:command_template":
            rank += 2 if broad_focus else 6
        elif span_type in {"agent:tool_template", "agent:procedure_span"}:
            rank += 4 if broad_focus else 7
        elif span_type == "agent:skill_fragment":
            rank += 3 if broad_focus else 5
        elif span_type == "agent:procedure":
            rank += 3 if broad_focus else 6
        elif span_type == "agent:episode_step":
            rank += 2 if broad_focus else 4
        elif span_type == "agent:task":
            rank += 6 if broad_focus else 3
        elif span_type == "agent:prompt":
            rank += 4 if broad_focus else 2
        elif span_type.startswith("doc:"):
            rank += 4 if broad_focus else 1
        elif span_type.startswith("agent:"):
            rank += 2
        if self._is_research_request(state):
            if span_type.startswith("doc:paper"):
                rank += 6
            elif span_type.startswith("doc:"):
                rank += 2
        rank += self._artifact_alignment_bonus(state, item)
        rank += self.world_model.score_retrieved_span(state.world_model_summary, item)
        rank += self._role_retrieval_bonus(state, item)
        return rank

    @staticmethod
    def _artifact_alignment_bonus(state: AgentState, item: dict[str, Any]) -> float:
        metadata = item.get("metadata") or {}
        touched_files = {
            str(value).strip()
            for value in metadata.get("touched_files", [])
            if str(value).strip()
        }
        benchmark_family = str(state.task.metadata.get("benchmark_family", "")).strip()
        item_benchmark_family = str(metadata.get("benchmark_family", "")).strip()
        task_capability = str(state.task.metadata.get("capability", "")).strip()
        item_capability = str(metadata.get("capability", "")).strip()
        desired_paths = {
            str(path).strip()
            for path in (
                *state.task.expected_files,
                *state.task.expected_file_contents.keys(),
                *(state.task.metadata.get("workflow_guard", {}) or {}).get("claimed_paths", []),
            )
            if str(path).strip()
        }
        bonus = 0.0
        if touched_files and desired_paths and touched_files & desired_paths:
            bonus += 5.0
        elif touched_files and desired_paths:
            bonus += 1.0
        if benchmark_family and item_benchmark_family and benchmark_family == item_benchmark_family:
            bonus += 1.5
        if task_capability and item_capability and task_capability == item_capability:
            bonus += 1.5
        return bonus

    @staticmethod
    def _role_retrieval_bonus(state: AgentState, item: dict[str, Any]) -> int:
        role = str(state.current_role or "executor")
        span_type = str(item.get("span_type", ""))
        if role == "planner":
            if span_type in {"agent:task", "agent:prompt"} or span_type.startswith("doc:"):
                return 2
        elif role == "critic":
            if span_type == "agent:episode_step":
                return 3
        elif role == "executor":
            if span_type in {"agent:command_template", "agent:procedure", "agent:skill_fragment"}:
                return 2
        return 0

    def _apply_source_diversity(
        self,
        results: list[dict[str, Any]],
        *,
        retrieval_config: dict[str, object],
    ) -> list[dict[str, Any]]:
        max_spans_per_source = int(retrieval_config["tolbert_max_spans_per_source"])
        if max_spans_per_source <= 0:
            return results
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for item in results:
            source_id = str(item.get("source_id", item.get("span_id", "")))
            counts[source_id] = counts.get(source_id, 0)
            if counts[source_id] >= max_spans_per_source:
                continue
            counts[source_id] += 1
            kept.append(item)
        return kept

    def _retrieval_config(self) -> dict[str, object]:
        values: dict[str, object] = {
            "tolbert_branch_results": self.config.tolbert_branch_results,
            "tolbert_global_results": self.config.tolbert_global_results,
            "tolbert_top_branches": self.config.tolbert_top_branches,
            "tolbert_ancestor_branch_levels": self.config.tolbert_ancestor_branch_levels,
            "tolbert_max_spans_per_source": self.config.tolbert_max_spans_per_source,
            "tolbert_context_max_chunks": self.config.tolbert_context_max_chunks,
            "tolbert_confidence_threshold": self.config.tolbert_confidence_threshold,
            "tolbert_branch_confidence_margin": self.config.tolbert_branch_confidence_margin,
            "tolbert_low_confidence_widen_threshold": self.config.tolbert_low_confidence_widen_threshold,
            "tolbert_low_confidence_branch_multiplier": self.config.tolbert_low_confidence_branch_multiplier,
            "tolbert_low_confidence_global_multiplier": self.config.tolbert_low_confidence_global_multiplier,
            "tolbert_deterministic_command_confidence": self.config.tolbert_deterministic_command_confidence,
            "tolbert_distractor_penalty": self.config.tolbert_distractor_penalty,
        }
        if not self.config.use_retrieval_proposals:
            return values
        path = self.config.retrieval_proposals_path
        if not path.exists():
            return values
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in retained_retrieval_overrides(payload).items():
            if key in values:
                values[key] = value
        return values

    @staticmethod
    def _adjacent_success_shortcut_enabled(state: AgentState) -> bool:
        return str(state.task.metadata.get("curriculum_kind", "")).strip() == "adjacent_success"

    def _build_adjacent_success_guidance(self, state: AgentState) -> dict[str, Any]:
        recommended_commands = self._dedupe_strings(
            [str(command).strip() for command in state.task.suggested_commands if str(command).strip()]
        )[:3]
        recommended_command_spans = [
            {"span_id": f"task:{state.task.task_id}:suggested:{index + 1}", "command": command}
            for index, command in enumerate(recommended_commands)
        ]
        evidence = [f"task:{state.task.task_id}: adjacent success task contract"]
        parent_task = str(state.task.metadata.get("parent_task", "")).strip()
        if parent_task:
            evidence.append(f"task:{parent_task}: successful parent episode")
        return {
            "recommended_commands": recommended_commands,
            "recommended_command_spans": recommended_command_spans,
            "avoidance_notes": [],
            "evidence": evidence,
        }

    def _build_retrieval_guidance(
        self,
        retrieval: dict[str, list[dict[str, Any]]],
        *,
        state: AgentState,
    ) -> dict[str, Any]:
        adjacent_success = self._adjacent_success_shortcut_enabled(state)
        seeded_guidance = self._build_adjacent_success_guidance(state) if adjacent_success else {}
        carry_guidance = {} if adjacent_success else self._retrieval_carry_seed(state)
        recommended_commands = [
            *list(carry_guidance.get("recommended_commands", [])),
            *list(seeded_guidance.get("recommended_commands", [])),
        ]
        recommended_command_spans = [
            *list(carry_guidance.get("recommended_command_spans", [])),
            *list(seeded_guidance.get("recommended_command_spans", [])),
        ]
        avoidance_notes = list(seeded_guidance.get("avoidance_notes", []))
        learned_avoidance_notes: list[str] = []
        evidence = [
            *list(carry_guidance.get("evidence", [])),
            *list(seeded_guidance.get("evidence", [])),
        ]
        if not adjacent_success:
            for item in [
                *retrieval.get("branch_scoped", []),
                *retrieval.get("fallback_scoped", []),
                *retrieval.get("global", []),
            ]:
                span_type = str(item.get("span_type", ""))
                text = str(item.get("text", "")).strip()
                metadata = dict(item.get("metadata") or {})
                if not text:
                    continue
                if span_type in {"agent:command_template", "agent:setup_command"}:
                    recommended_commands.append(text)
                    recommended_command_spans.append({"span_id": str(item["span_id"]), "command": text})
                    evidence.append(f"{item['span_id']}: template command")
                elif span_type in {"agent:tool_template", "agent:procedure", "agent:procedure_span"}:
                    for command in self._procedure_commands(text):
                        recommended_commands.append(command)
                        recommended_command_spans.append({"span_id": str(item["span_id"]), "command": command})
                    evidence.append(f"{item['span_id']}: procedure guidance")
                elif span_type == "agent:skill_fragment":
                    for line in text.splitlines():
                        command = line.strip()
                        if command:
                            recommended_commands.append(command)
                            recommended_command_spans.append(
                                {"span_id": str(item["span_id"]), "command": command}
                            )
                    evidence.append(f"{item['span_id']}: skill fragment")
                elif span_type == "agent:episode_step":
                    parsed = self._parse_episode_step_span(text)
                    command = parsed.get("content", "")
                    if parsed.get("verification_passed") == "True" and command:
                        recommended_commands.append(command)
                        recommended_command_spans.append(
                            {"span_id": str(item["span_id"]), "command": command}
                        )
                        evidence.append(f"{item['span_id']}: successful episode step")
                    elif command:
                        reasons = parsed.get("verification_reasons", "")
                        avoidance_notes.append(f"avoid repeating {command!r} when {reasons}")
                elif span_type == "agent:task":
                    evidence.append(f"{item['span_id']}: task overview")
                elif span_type == "agent:tool_candidate":
                    evidence.append(f"{item['span_id']}: reusable tool candidate")
                elif span_type.startswith("doc:"):
                    title = str(metadata.get("title", "")).strip()
                    path = str(metadata.get("path", metadata.get("pdf_path", ""))).strip()
                    label = title or path or str(item.get("source_id", item["span_id"]))
                    evidence.append(f"{item['span_id']}: reference evidence from {label}")
        for candidate in matching_learning_candidates(
            self.config.learning_artifacts_path,
            config=self.config,
            task_id=state.task.task_id,
            source_task_id=str(state.task.metadata.get("source_task", "")).strip(),
            benchmark_family=str(state.task.metadata.get("benchmark_family", "")).strip(),
            curriculum_kind=str(state.task.metadata.get("curriculum_kind", "")).strip(),
            memory_source=str(state.task.metadata.get("memory_source", "")).strip(),
        )[:6]:
            artifact_kind = str(candidate.get("artifact_kind", "")).strip()
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            candidate_memory_source = str(candidate.get("memory_source", "")).strip()
            source_suffix = f" via {candidate_memory_source} memory" if candidate_memory_source else ""
            retrieval_prefix = self._learning_candidate_retrieval_prefix(candidate)
            if artifact_kind in {"success_skill_candidate", "recovery_case"}:
                procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
                commands = [
                    str(command).strip()
                    for command in procedure.get("commands", candidate.get("recovery_commands", []))
                    if str(command).strip()
                ]
                for command in commands[:2]:
                    recommended_commands.append(command)
                    recommended_command_spans.append({"span_id": candidate_id or artifact_kind, "command": command})
                evidence.append(
                    f"{candidate_id or artifact_kind}: learned {retrieval_prefix}{artifact_kind} from {str(candidate.get('source_task_id', '')).strip()}{source_suffix}"
                )
            elif artifact_kind == "negative_command_pattern":
                command = str(candidate.get("command", "")).strip()
                reasons = ", ".join(
                    str(reason).strip()
                    for reason in candidate.get("verification_reasons", [])
                    if str(reason).strip()
                )
                if command:
                    learned_avoidance_notes.append(
                        f"avoid repeating {command!r} when {reasons or 'it previously failed verification'}"
                    )
                evidence.append(
                    f"{candidate_id or artifact_kind}: learned negative command pattern from {str(candidate.get('source_task_id', '')).strip()}{source_suffix}"
                )
            elif artifact_kind in {"failure_case", "benchmark_gap"}:
                evidence.append(
                    f"{candidate_id or artifact_kind}: learned {artifact_kind} evidence from {str(candidate.get('source_task_id', '')).strip()}{source_suffix}"
                )
        deduped_command_spans: list[dict[str, str]] = []
        seen_commands: set[str] = set()
        for entry in recommended_command_spans:
            command = entry["command"].strip()
            if not command or command in seen_commands:
                continue
            seen_commands.add(command)
            deduped_command_spans.append({"span_id": entry["span_id"], "command": command})
        return {
            "recommended_commands": self._dedupe_strings(recommended_commands)[:5],
            "recommended_command_spans": deduped_command_spans[:5],
            "avoidance_notes": self._dedupe_strings([*learned_avoidance_notes, *avoidance_notes])[:3],
            "evidence": self._dedupe_strings(evidence)[:5],
        }

    @staticmethod
    def _learning_candidate_retrieval_prefix(candidate: dict[str, Any]) -> str:
        if not bool(candidate.get("retrieval_backed", False)):
            return ""
        if int(candidate.get("trusted_retrieval_steps", 0) or 0) > 0:
            return "trusted retrieval-backed "
        if int(candidate.get("retrieval_influenced_steps", 0) or 0) > 0:
            return "retrieval-backed "
        return ""

    def _select_context_chunks(
        self,
        retrieval: dict[str, list[dict[str, Any]]],
        *,
        state: AgentState,
        level_focus: str,
        retrieval_config: dict[str, object],
    ) -> list[dict[str, Any]]:
        budget = max(1, self.config.tolbert_context_char_budget)
        max_chunks = max(1, int(retrieval_config["tolbert_context_max_chunks"]))
        candidates: list[dict[str, Any]] = []
        bucket_bonus = {"branch_scoped": 3.0, "fallback_scoped": 1.5, "global": 0.5}
        seen: set[str] = set()
        for bucket in ("branch_scoped", "fallback_scoped", "global"):
            for item in retrieval.get(bucket, []):
                span_id = str(item.get("span_id", ""))
                if not span_id or span_id in seen:
                    continue
                seen.add(span_id)
                text = " ".join(str(item.get("text", "")).split())
                if not text:
                    continue
                metadata = dict(item.get("metadata") or {})
                summary = {
                    "span_id": span_id,
                    "source_id": str(item.get("source_id", item.get("span_id", ""))),
                    "span_type": str(item.get("span_type", "")),
                    "bucket": bucket,
                    "task_id": str(metadata.get("task_id", "")),
                    "score": float(item.get("score", 0.0)),
                    "rank": self._retrieval_rank(
                        item,
                        state=state,
                        level_focus=level_focus,
                        retrieval_config=retrieval_config,
                    )
                    + bucket_bonus[bucket],
                    "text": text[:280],
                }
                candidates.append(summary)
        selected: list[dict[str, Any]] = []
        used_chars = 0
        for summary in sorted(candidates, key=lambda item: (-float(item["rank"]), -float(item["score"]), str(item["span_id"]))):
            chunk_cost = len(summary["text"])
            if selected and used_chars + chunk_cost > budget:
                continue
            selected.append(summary)
            used_chars += chunk_cost
            if len(selected) >= max_chunks or used_chars >= budget:
                break
        return selected

    def _query_paper_research(
        self,
        *,
        query_text: str,
        compile_started_at: float,
    ) -> dict[str, Any] | None:
        client = self._ensure_research_client()
        if client is None:
            return None
        remaining_budget = self._remaining_compile_budget_seconds(compile_started_at)
        if remaining_budget is not None and remaining_budget <= 0.0:
            self._research_client_error = "context compile budget exhausted before paper research query"
            return None
        try:
            query_kwargs = {
                "query_text": query_text,
                "branch_results": max(1, self.config.paper_research_branch_results),
                "global_results": max(1, self.config.paper_research_global_results),
                "confidence_threshold": 0.0,
                "top_branches": max(1, self.config.tolbert_top_branches),
                "branch_confidence_margin": self.config.tolbert_branch_confidence_margin,
                "low_confidence_widen_threshold": self.config.tolbert_low_confidence_widen_threshold,
                "ancestor_branch_levels": self.config.tolbert_ancestor_branch_levels,
                "low_confidence_branch_multiplier": self.config.tolbert_low_confidence_branch_multiplier,
                "low_confidence_global_multiplier": self.config.tolbert_low_confidence_global_multiplier,
            }
            subphase_budget = self._context_compile_subphase_budget_seconds("paper_research_query")
            if subphase_budget is not None:
                query_kwargs["timeout_seconds"] = subphase_budget
            return client.query(
                **query_kwargs,
            )
        except Exception as exc:
            self._research_client_error = str(exc)
            return None

    def _emit_compile_progress(
        self,
        subphase: str,
        *,
        compile_started_at: float,
        subphase_started_at: float,
        **payload: object,
    ) -> None:
        if self._progress_callback is None:
            return
        normalized_subphase = str(subphase).strip()
        subphase_budget_seconds = self._context_compile_subphase_budget_seconds(normalized_subphase)
        event = {
            "step_stage": "context_compile",
            "step_subphase": normalized_subphase,
            "step_elapsed_seconds": round(max(0.0, time.monotonic() - compile_started_at), 4),
            "step_budget_seconds": max(0.0, float(subphase_budget_seconds or 0.0)),
        }
        event.update(payload)
        self._progress_callback(event)

    def _remaining_compile_budget_seconds(self, compile_started_at: float) -> float | None:
        budget_seconds = self._total_context_compile_budget_seconds()
        if budget_seconds <= 0.0:
            return None
        return max(0.0, budget_seconds - max(0.0, time.monotonic() - compile_started_at))

    def _context_compile_subphase_budget_seconds(self, subphase: str) -> float | None:
        budget_seconds = self._total_context_compile_budget_seconds()
        if budget_seconds <= 0.0:
            return None
        raw_factor = self._CONTEXT_COMPILE_SUBPHASE_BUDGET_FACTORS.get(str(subphase).strip(), 1.0)
        try:
            factor = max(0.0, min(1.0, float(raw_factor)))
        except (TypeError, ValueError):
            factor = 1.0
        return max(0.0, budget_seconds * factor)

    def _total_context_compile_budget_seconds(self) -> float:
        return max(0.0, float(self.config.tolbert_context_compile_budget_seconds or 0.0))

    def _enforce_compile_budget(self, subphase: str, *, compile_started_at: float, subphase_started_at: float) -> None:
        remaining_budget = self._remaining_compile_budget_seconds(compile_started_at)
        if remaining_budget is not None and remaining_budget <= 0.0:
            raise RuntimeError(
                "context compile budget "
                f"of {float(self.config.tolbert_context_compile_budget_seconds or 0.0):.3f} seconds "
                f"exceeded during {str(subphase).strip() or 'unknown'}"
            )

        normalized_subphase = str(subphase).strip()
        subphase_budget_seconds = self._context_compile_subphase_budget_seconds(normalized_subphase)
        if subphase_budget_seconds is None or subphase_budget_seconds <= 0.0:
            return
        subphase_elapsed = max(0.0, time.monotonic() - max(0.0, float(subphase_started_at)))
        if subphase_elapsed <= subphase_budget_seconds:
            return
        raise RuntimeError(
            "context compile budget "
            f"of {float(subphase_budget_seconds):.3f} seconds "
            "exceeded during "
            f"{normalized_subphase or 'unknown'}"
        )

    def _ensure_research_client(self) -> TolbertQueryClient | None:
        if self.research_client is not None:
            return self.research_client
        if not self._research_runtime_paths or self._research_client_error:
            return None
        try:
            self.research_client = TolbertServiceClient(
                config=self.config,
                repo_root=self.repo_root,
                runtime_paths=self._research_runtime_paths,
                service_name="paper_research",
            )
        except Exception as exc:
            self._research_client_error = str(exc)
            return None
        return self.research_client

    def _should_query_paper_research(self, state: AgentState, *, query_text: str) -> bool:
        if not self.config.use_paper_research_context:
            return False
        mode = self.config.paper_research_query_mode.strip().lower()
        if mode in {"0", "false", "never", "off"}:
            return False
        if mode in {"1", "always", "on", "true"}:
            return True
        return self._research_prompt_hint(query_text)

    def _is_research_request(self, state: AgentState) -> bool:
        text_parts = [state.task.prompt, state.recent_workspace_summary]
        if state.history:
            text_parts.append(state.history[-1].content)
        return self._research_prompt_hint(" ".join(part for part in text_parts if part))

    def _merge_research_retrieval(
        self,
        retrieval: dict[str, list[dict[str, Any]]],
        research_retrieval: dict[str, list[dict[str, Any]]],
        *,
        state: AgentState,
        level_focus: str,
        retrieval_config: dict[str, object],
    ) -> dict[str, list[dict[str, Any]]]:
        merged = {
            "branch_scoped": list(retrieval.get("branch_scoped", [])),
            "fallback_scoped": list(retrieval.get("fallback_scoped", [])),
            "global": list(retrieval.get("global", [])),
        }
        seen = {
            str(item.get("span_id", ""))
            for bucket in merged.values()
            for item in bucket
            if str(item.get("span_id", ""))
        }
        for bucket in ("branch_scoped", "fallback_scoped", "global"):
            for item in research_retrieval.get(bucket, []):
                span_id = str(item.get("span_id", ""))
                if not span_id or span_id in seen:
                    continue
                metadata = dict(item.get("metadata") or {})
                metadata.setdefault("retrieval_source", "paper_research")
                merged["global"].append({**item, "metadata": metadata})
                seen.add(span_id)
        combined_global_limit = int(retrieval_config["tolbert_global_results"]) + max(
            1,
            int(self.config.paper_research_global_results),
        )
        merged["global"] = self._dedupe_and_rank(
            merged["global"],
            state=state,
            level_focus=level_focus,
            retrieval_config=retrieval_config,
        )[:combined_global_limit]
        return merged

    @staticmethod
    def _research_prompt_hint(text: str) -> bool:
        lowered = text.lower()
        keywords = (
            "arxiv",
            "paper",
            "papers",
            "research",
            "literature",
            "citation",
            "citations",
            "related work",
            "baseline",
            "survey",
            "benchmark",
            "benchmarks",
            "method",
            "methods",
            "algorithm",
            "algorithms",
            "theorem",
            "proof",
            "ablation",
        )
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _path_confidence(path_prediction: dict[str, Any]) -> float:
        confidences = [
            float(value)
            for value in path_prediction.get("confidence_by_level", {}).values()
        ]
        if not confidences:
            return 0.0
        return min(confidences)

    @staticmethod
    def _parse_episode_step_span(text: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    @staticmethod
    def _procedure_commands(text: str) -> list[str]:
        stripped = str(text).strip()
        if not stripped:
            return []
        commands = [line.strip() for line in stripped.splitlines() if line.strip()]
        return commands or [stripped]

    def _workflow_guarded_guidance_is_trustworthy(
        self,
        state: AgentState,
        retrieval_guidance: dict[str, Any],
    ) -> bool:
        workflow_guard = state.task.metadata.get("workflow_guard", {}) or {}
        if not isinstance(workflow_guard, dict):
            return False
        claimed_paths = [
            str(path).strip()
            for path in workflow_guard.get("claimed_paths", [])
            if str(path).strip()
        ]
        if not claimed_paths:
            return False
        for entry in retrieval_guidance.get("recommended_command_spans", [])[:2]:
            command = str(entry.get("command", "")).strip()
            span_id = str(entry.get("span_id", "")).strip()
            if not command or not self._first_step_guarded_command_coverage(state, command, claimed_paths):
                continue
            if span_id.startswith("learning:success_skill:") or span_id.startswith("learning:recovery_case:"):
                return True
            if span_id.startswith("procedure:") or span_id.startswith("tool:"):
                return True
        return False

    @staticmethod
    def _first_step_guarded_command_coverage(
        state: AgentState,
        command: str,
        claimed_paths: list[str],
    ) -> bool:
        desired_paths = {
            str(path).strip()
            for path in (
                *claimed_paths,
                *state.task.expected_files,
                *state.task.expected_file_contents.keys(),
            )
            if str(path).strip()
        }
        covered_paths = {path for path in desired_paths if path in command}
        if len(covered_paths) >= 2:
            return True
        expected_outputs = {
            str(path).strip()
            for path in (
                *state.task.expected_files,
                *state.task.expected_file_contents.keys(),
            )
            if str(path).strip()
        }
        return bool(covered_paths and covered_paths & expected_outputs and "git " in command)

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _retrieval_carry_summary(self, state: AgentState) -> str:
        seed = self._retrieval_carry_seed(state)
        summaries = [
            str(value).strip()
            for value in seed.get("summaries", [])
            if str(value).strip()
        ]
        return "\n".join(summaries[:2])

    def _retrieval_carry_seed(self, state: AgentState) -> dict[str, Any]:
        packet = state.context_packet
        if packet is None or not state.history:
            return {}
        guidance = packet.control.get("retrieval_guidance", {})
        if not isinstance(guidance, dict):
            guidance = {}
        guidance_spans = guidance.get("recommended_command_spans", [])
        if not isinstance(guidance_spans, list):
            guidance_spans = []
        by_span_id: dict[str, str] = {}
        by_command: dict[str, str] = {}
        for entry in guidance_spans:
            if not isinstance(entry, dict):
                continue
            span_id = str(entry.get("span_id", "")).strip()
            command = str(entry.get("command", "")).strip()
            if not span_id or not command:
                continue
            by_span_id[span_id] = command
            by_command[command] = span_id
        retrieval_items: dict[str, dict[str, Any]] = {}
        for bucket in ("branch_scoped", "fallback_scoped", "global"):
            items = packet.retrieval.get(bucket, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                span_id = str(item.get("span_id", "")).strip()
                if span_id and span_id not in retrieval_items:
                    retrieval_items[span_id] = dict(item)

        recommended_commands: list[str] = []
        recommended_command_spans: list[dict[str, str]] = []
        evidence: list[str] = []
        summaries: list[str] = []
        carry_items: list[dict[str, Any]] = []
        seen_spans: set[str] = set()
        seen_commands: set[str] = set()

        def append_carry(*, span_id: str, command: str, step_index: int, reason: str) -> None:
            normalized_span_id = str(span_id).strip()
            normalized_command = str(command).strip()
            if not normalized_command:
                return
            carry_span_id = normalized_span_id or f"carry:{state.task.task_id}:{step_index}"
            if carry_span_id in seen_spans or normalized_command in seen_commands:
                return
            seen_spans.add(carry_span_id)
            seen_commands.add(normalized_command)
            recommended_commands.append(normalized_command)
            recommended_command_spans.append({"span_id": carry_span_id, "command": normalized_command})
            evidence.append(f"{carry_span_id}: carried retrieval guidance from step {step_index} ({reason})")
            summaries.append(f"retrieval carry step={step_index} command={normalized_command}")
            carried_item = retrieval_items.get(normalized_span_id, {})
            if carried_item:
                metadata = dict(carried_item.get("metadata") or {})
                metadata["carried_retrieval"] = True
                metadata["carry_step_index"] = step_index
                metadata["carry_reason"] = reason
                carry_items.append({**carried_item, "metadata": metadata})
                return
            carry_items.append(
                {
                    "span_id": carry_span_id,
                    "text": normalized_command,
                    "source_id": state.task.task_id,
                    "span_type": "agent:command_template",
                    "score": 0.0,
                    "node_path": [0, 0, 0],
                    "metadata": {
                        "span_type": "agent:command_template",
                        "task_id": state.task.task_id,
                        "carried_retrieval": True,
                        "carry_step_index": step_index,
                        "carry_reason": reason,
                    },
                }
            )

        for step in reversed(state.history[-3:]):
            selected_span_id = str(step.selected_retrieval_span_id or "").strip()
            step_verified = bool(step.verification.get("passed", False))
            if selected_span_id and (step_verified or not bool(step.retrieval_command_match)):
                append_carry(
                    span_id=selected_span_id,
                    command=by_span_id.get(selected_span_id, ""),
                    step_index=int(step.index),
                    reason="selected_span",
                )
            if step_verified and str(step.content).strip() and (step.retrieval_influenced or step.trust_retrieval):
                append_carry(
                    span_id=by_command.get(str(step.content).strip(), ""),
                    command=str(step.content).strip(),
                    step_index=int(step.index),
                    reason="trusted_or_influenced_command",
                )
            if recommended_commands:
                break
        if not recommended_commands:
            return {}
        return {
            "recommended_commands": recommended_commands,
            "recommended_command_spans": recommended_command_spans,
            "evidence": evidence,
            "summaries": summaries,
            "items": carry_items,
        }


class MockTolbertContextCompiler:
    def __init__(self, *, config: KernelConfig, repo_root: Path) -> None:
        del repo_root
        self.config = config

    def close(self) -> None:
        return None

    def compile(self, state: AgentState) -> ContextPacket:
        recommended_commands = list(state.task.suggested_commands[:3])
        avoidance_notes: list[str] = []
        evidence: list[str] = []
        for candidate in matching_learning_candidates(
            self.config.learning_artifacts_path,
            config=self.config,
            task_id=state.task.task_id,
            source_task_id=str(state.task.metadata.get("source_task", "")).strip(),
            benchmark_family=str(state.task.metadata.get("benchmark_family", "")).strip(),
            curriculum_kind=str(state.task.metadata.get("curriculum_kind", "")).strip(),
        )[:4]:
            artifact_kind = str(candidate.get("artifact_kind", "")).strip()
            if artifact_kind in {"success_skill_candidate", "recovery_case"}:
                procedure = dict(candidate.get("procedure", {})) if isinstance(candidate.get("procedure", {}), dict) else {}
                commands = procedure.get("commands", candidate.get("recovery_commands", []))
                for command in commands:
                    normalized = str(command).strip()
                    if normalized and normalized not in recommended_commands:
                        recommended_commands.append(normalized)
            elif artifact_kind == "negative_command_pattern":
                command = str(candidate.get("command", "")).strip()
                if command:
                    avoidance_notes.append(f"avoid repeating {command!r} when it previously failed verification")
            evidence.append(
                f"{str(candidate.get('candidate_id', artifact_kind)).strip()}: learned {artifact_kind}"
            )
        recommended_command_spans = [
            {"span_id": f"task:{state.task.task_id}:suggested:{index + 1}", "command": command}
            for index, command in enumerate(recommended_commands)
        ]
        branch_scoped = [
            {
                "span_id": entry["span_id"],
                "text": entry["command"],
                "source_id": state.task.task_id,
                "span_type": "agent:command_template",
                "score": 1.0,
                "node_path": [0, 0, 0],
                "metadata": {"task_id": state.task.task_id, "span_type": "agent:command_template"},
            }
            for entry in recommended_command_spans
        ]
        selected_chunks = [
            {
                "span_id": entry["span_id"],
                "source_id": state.task.task_id,
                "span_type": "agent:command_template",
                "bucket": "branch_scoped",
                "task_id": state.task.task_id,
                "score": 1.0,
                "text": entry["command"][:280],
            }
            for entry in recommended_command_spans[: self.config.tolbert_context_max_chunks]
        ]
        return ContextPacket(
            request_id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            task={
                "goal": state.task.prompt,
                "completion_criteria": state.task.success_command,
                "workspace_summary": state.recent_workspace_summary,
                "recent_steps": "[]",
            },
            control={
                "mode": "verify",
                "level_focus": "task",
                "window_budget_tokens": 1024,
                "retrieval_k": len(recommended_commands),
                "confidence_threshold": 0.0,
                "path_confidence": 1.0 if recommended_commands else 0.0,
                "trust_retrieval": bool(recommended_commands),
                "selected_branch_level": 0,
                "branch_candidates": [],
                "selected_context_chunks": selected_chunks,
                "context_chunk_budget": {
                    "max_chunks": self.config.tolbert_context_max_chunks,
                    "char_budget": self.config.tolbert_context_char_budget,
                },
                "retrieval_guidance": {
                    "recommended_commands": recommended_commands,
                    "recommended_command_spans": recommended_command_spans,
                    "avoidance_notes": avoidance_notes[:3],
                    "evidence": [
                        *[f"{entry['span_id']}: mock task suggestion" for entry in recommended_command_spans],
                        *evidence[:3],
                    ],
                },
            },
            tolbert={
                "path_prediction": {
                    "tree_version": "mock_tol_v1",
                    "decode_mode": "mock",
                    "confidence_by_level": {"1": 1.0 if recommended_commands else 0.0},
                },
                "backend": "mock_tolbert",
                "device": "cpu",
                "index_shards": ["mock"],
            },
            retrieval={"branch_scoped": branch_scoped, "fallback_scoped": [], "global": []},
            verifier_contract={
                "success_command": state.task.success_command,
                "expected_files": state.task.expected_files,
                "expected_output_substrings": state.task.expected_output_substrings,
                "forbidden_files": state.task.forbidden_files,
                "forbidden_output_substrings": state.task.forbidden_output_substrings,
                "expected_file_contents": state.task.expected_file_contents,
            },
        )


def _paper_research_runtime_paths(
    config: KernelConfig,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    if not _paper_research_env_overrides_present():
        discovered = _discover_v2_paper_research_runtime_paths()
        if discovered is not None:
            return discovered

    def _resolve(raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        return str(path)

    source_spans_paths = tuple(
        path for path in (_resolve(raw_path) for raw_path in config.paper_research_source_spans_paths) if path
    )
    cache_paths = tuple(
        path for path in (_resolve(raw_path) for raw_path in config.paper_research_cache_paths) if path
    )
    runtime_paths = {
        "tolbert_config_path": _resolve(config.paper_research_config_path),
        "tolbert_checkpoint_path": _resolve(config.paper_research_checkpoint_path),
        "tolbert_nodes_path": _resolve(config.paper_research_nodes_path),
        "tolbert_label_map_path": _resolve(config.paper_research_label_map_path),
        "tolbert_source_spans_paths": source_spans_paths,
        "tolbert_cache_paths": cache_paths,
    }
    required_paths = [
        runtime_paths["tolbert_config_path"],
        runtime_paths["tolbert_checkpoint_path"],
        runtime_paths["tolbert_nodes_path"],
        runtime_paths["tolbert_label_map_path"],
        *runtime_paths["tolbert_source_spans_paths"],
        *runtime_paths["tolbert_cache_paths"],
    ]
    if not required_paths or any(path is None or not Path(path).exists() for path in required_paths):
        return None
    return runtime_paths


def _paper_research_env_overrides_present() -> bool:
    override_names = (
        "AGENT_KERNEL_PAPER_RESEARCH_CONFIG_PATH",
        "AGENT_KERNEL_PAPER_RESEARCH_CHECKPOINT_PATH",
        "AGENT_KERNEL_PAPER_RESEARCH_NODES_PATH",
        "AGENT_KERNEL_PAPER_RESEARCH_LABEL_MAP_PATH",
        "AGENT_KERNEL_PAPER_RESEARCH_SOURCE_SPANS_PATHS",
        "AGENT_KERNEL_PAPER_RESEARCH_CACHE_PATHS",
    )
    return any(os.getenv(name) is not None for name in override_names)


def _discover_v2_paper_research_runtime_paths() -> dict[str, Any] | None:
    base = Path("/data/TOLBERT_BRAIN")
    ckpt_dir = base / "checkpoints" / "tolbert_brain_joint_v2"
    retrieval_dir = ckpt_dir / "retrieval_cache"
    checkpoint_paths = sorted(
        ckpt_dir.glob("tolbert_epoch*.pt"),
        key=_tolbert_epoch_sort_key,
    )
    latest_checkpoint = checkpoint_paths[-1] if checkpoint_paths else None
    cache_paths: list[Path] = []
    if latest_checkpoint is not None:
        manifest_path = retrieval_dir / f"{latest_checkpoint.stem}.json"
        if manifest_path.is_file():
            cache_paths = [manifest_path]
        else:
            cache_paths = sorted(
                path
                for path in retrieval_dir.glob(f"{latest_checkpoint.stem}.shard*.pt")
                if path.is_file()
            )
    if not cache_paths:
        cache_paths = sorted(path for path in retrieval_dir.glob("*.json") if path.is_file())
    if not cache_paths:
        cache_paths = sorted(path for path in retrieval_dir.glob("*.pt") if path.is_file())
    runtime_paths = {
        "tolbert_config_path": str(base / "configs" / "tolbert_brain_joint_v2.yaml"),
        "tolbert_checkpoint_path": str(checkpoint_paths[-1]) if checkpoint_paths else None,
        "tolbert_nodes_path": str(base / "data" / "joint_v2" / "nodes_joint_v2.jsonl"),
        "tolbert_label_map_path": str(base / "data" / "joint_v2" / "label_map_joint_v2.json"),
        "tolbert_source_spans_paths": (
            str(base / "data" / "joint_v2" / "code_spans_joint_v2_mapped.jsonl"),
            str(base / "data" / "joint_v2" / "paper_spans_paragraphs_joint_v2_mapped.jsonl"),
        ),
        "tolbert_cache_paths": tuple(str(path) for path in cache_paths),
    }
    required_paths = [
        runtime_paths["tolbert_config_path"],
        runtime_paths["tolbert_checkpoint_path"],
        runtime_paths["tolbert_nodes_path"],
        runtime_paths["tolbert_label_map_path"],
        *runtime_paths["tolbert_source_spans_paths"],
        *runtime_paths["tolbert_cache_paths"],
    ]
    if not required_paths or any(path is None or not Path(path).exists() for path in required_paths):
        return None
    return runtime_paths


def _tolbert_epoch_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if "epoch" not in stem:
        return (-1, stem)
    suffix = stem.rsplit("epoch", 1)[-1]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if not digits:
        return (-1, stem)
    return (int(digits), stem)
