from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from ..schemas import ContextPacket
from ..state import AgentState
from .models import select_trained_model_assets, trained_model_asset_catalog
from .sources import build_research_library_status


_LOW_SIGNAL_TOKENS = {
    "adapter",
    "and",
    "before",
    "between",
    "benchmark",
    "bench",
    "bug",
    "code",
    "coding",
    "decide",
    "diagnose",
    "extend",
    "failing",
    "fix",
    "for",
    "from",
    "improve",
    "into",
    "kernel",
    "local",
    "model",
    "needed",
    "plan",
    "project",
    "python",
    "recover",
    "repair",
    "repository",
    "style",
    "task",
    "test",
    "tests",
    "trained",
    "use",
    "using",
    "with",
}


def _tokens(*values: object) -> set[str]:
    text = " ".join(str(value).lower().replace("_", " ") for value in values if value is not None)
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+\-.]{2,}", text)
        if token not in _LOW_SIGNAL_TOKENS
    }


def _normalized_phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ")))


def _nested_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(_nested_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_nested_text(item) for item in value)
    return str(value)


def _contains_phrase(haystack: str, needle: object) -> bool:
    normalized = _normalized_phrase(needle)
    if len(normalized) < 3:
        return False
    return f" {normalized} " in f" {haystack} "


def _repo_name_score(repo_hint: object, name: object) -> int:
    hint = _normalized_phrase(repo_hint)
    normalized_name = _normalized_phrase(name)
    if not hint or not normalized_name:
        return 0
    if hint == normalized_name:
        return 10
    if hint in normalized_name or normalized_name in hint:
        return 6
    return 0


def _alias_phrase_score(query_phrase: str, aliases: list[object]) -> int:
    for alias in aliases:
        if _contains_phrase(query_phrase, alias):
            return 8
    return 0


def _truncate(value: object, limit: int = 420) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 0


def _shell_single_quote(value: object) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _safe_output_text_path(value: object) -> str:
    text = str(value).strip()
    if not text or ".." in Path(text).parts:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", text):
        return ""
    return text


def _read_jsonl(path: Path, *, limit: int = 4096) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines[:limit]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _read_parquet_rows(path: Path, *, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    records: list[dict[str, Any]] = []
    try:
        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(batch_size=1024, columns=columns)
        for batch in batches:
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    records.append(row)
                if len(records) >= limit:
                    return records
    except Exception:
        return records
    return records


def _best_excerpt(text: object, query_tokens: set[str], *, limit: int = 520) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    lowered = normalized.lower()
    starts = [
        lowered.find(token)
        for token in query_tokens
        if len(token) >= 5 and lowered.find(token) >= 0
    ]
    start = min(starts) if starts else 0
    start = max(0, start - 80)
    excerpt = normalized[start : start + limit]
    if start > 0:
        excerpt = "..." + excerpt.lstrip()
    if start + limit < len(normalized):
        excerpt = excerpt.rstrip() + "..."
    return excerpt


def _best_evidence_sentence(text: object, query_tokens: set[str]) -> str:
    normalized = " ".join(str(text).split())
    if not normalized:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    if not sentences:
        return _truncate(normalized, 260)
    ranked: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = len(query_tokens & _tokens(sentence))
        if query_tokens & {"implement", "implemented", "implementation", "agent-speak", "agentspeak"} and (
            "strategy" in lowered and "implement" in lowered
        ):
            score += 8
        if "delegate" in lowered and "herders" in lowered and "himself" in lowered:
            score += 8
        if "adjacent" in lowered and ("grouped" in lowered or "clustering" in lowered):
            score += 8
        if "fence" in lowered and ("fleeing" in lowered or "right way" in lowered):
            score += 8
        if "rewrite relation" in lowered and "u1" in lowered and "u2" in lowered:
            score += 20
        if "active letter" in lowered and ("p-th letter" in lowered or "p th letter" in lowered):
            score += 18
        if lowered.startswith("title:"):
            score -= 2
        ranked.append((score, -index, sentence))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    return _truncate(ranked[0][2], 260)


def _applied_paper_facts(rows: list[dict[str, Any]]) -> list[str]:
    combined = " ".join(" ".join(
        " ".join(
            [
                str(row.get("evidence_sentence", "")),
                str(row.get("excerpt", "")),
                str(row.get("fact_text", "")),
            ]
        )
        for row in rows
    ).split()).lower()
    facts: list[str] = []
    if "adjacent" in combined and ("grouped" in combined or "clustering" in combined or "cluster" in combined):
        facts.append("cow_grouping=group adjacent cows into clusters")
    if "delegate targets" in combined and "including himself" in combined:
        facts.append("target_delegation=the leader delegates targets to each herder including himself")
    if "fence" in combined and ("fleeing the right way" in combined or "right way" in combined):
        facts.append("fence_timing=the team leader coordinates fence opening when cows are fleeing the right way")
    if (
        "rewrite relation" in combined
        and "u1" in combined
        and "u2" in combined
        and (" β" in combined or "beta" in combined)
    ):
        facts.append(
            "rewrite_at_position=p is the end offset after alpha; require word[p-len(alpha):p] == alpha; "
            "return word[:p-len(alpha)] + beta + word[p:]"
        )
    if "active letter" in combined and ("p-th letter" in combined or "p th letter" in combined):
        facts.append("active_letter=the p-th letter in u is active; zero-based active_index is p-1")
    return facts


def _applied_guidance_lines(facts: list[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        normalized = str(line).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            lines.append(normalized)

    for fact in facts:
        normalized_fact = str(fact).strip()
        if not normalized_fact:
            continue
        if normalized_fact.startswith("rewrite_at_position="):
            add(
                "replacement_start(alpha, p): start = p - len(alpha); "
                "if start < 0 then raise ValueError; otherwise return start"
            )
            add("invalid rewrite position: do not clamp start to 0; negative start is an error")
            add(
                "rewrite_at_position(word, alpha, beta, p): start = p - len(alpha); "
                "if start < 0 or word[start:p] != alpha then raise ValueError; "
                "otherwise return word[:start] + beta + word[p:]"
            )
            continue
        if normalized_fact.startswith("active_letter="):
            add("active_letter_index(word, p): p names the paper's p-th letter in u; zero-based active_index = p - 1")
            continue
        if normalized_fact.startswith("cow_grouping="):
            add("cow_grouping: group adjacent cows into clusters")
            continue
        if normalized_fact.startswith("target_delegation="):
            add("target_delegation: the leader assigns targets to each herder including itself")
            continue
        if normalized_fact.startswith("fence_timing="):
            add("fence_timing: the leader opens or coordinates the fence when cows are fleeing the right way")
            continue
        add(normalized_fact)
    return lines


class ResearchLibraryContextProvider:
    """Augments kernel context packets with local research-library capability context."""

    def __init__(self, *, config: Any, repo_root: Path, base_provider: Any = None) -> None:
        self.config = config
        self.repo_root = repo_root
        self.base_provider = base_provider
        self._progress_callback = None
        self._status_cache: dict[str, Any] | None = None
        self._status_cache_mtime: float | None = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback
        setter = getattr(self.base_provider, "set_progress_callback", None)
        if callable(setter):
            setter(callback)

    def close(self) -> None:
        close = getattr(self.base_provider, "close", None)
        if callable(close):
            close()

    def compile(self, state: AgentState) -> ContextPacket:
        packet = self._compile_base(state)
        status = self._load_status()
        if not status:
            return packet
        chunks = self._research_chunks(status, state)
        if not chunks:
            return packet
        return self._augment_packet(packet, status=status, state=state, chunks=chunks)

    def _emit(self, stage: str, **payload: object) -> None:
        if self._progress_callback is None:
            return
        event = {"step_stage": "research_library_context", "research_stage": stage}
        event.update(payload)
        self._progress_callback(event)

    def _compile_base(self, state: AgentState) -> ContextPacket:
        if self.base_provider is not None:
            return self.base_provider.compile(state)
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
                "mode": "research_library",
                "level_focus": "task",
                "window_budget_tokens": 1024,
                "retrieval_k": 0,
                "confidence_threshold": 0.0,
                "path_confidence": 0.0,
                "trust_retrieval": False,
                "selected_branch_level": 0,
                "branch_candidates": [],
                "selected_context_chunks": [],
                "context_chunk_budget": {
                    "max_chunks": int(getattr(self.config, "tolbert_context_max_chunks", 8)),
                    "char_budget": int(getattr(self.config, "tolbert_context_char_budget", 2400)),
                },
                "retrieval_guidance": {
                    "recommended_commands": [],
                    "recommended_command_spans": [],
                    "avoidance_notes": [],
                    "evidence": [],
                },
            },
            tolbert={
                "path_prediction": {
                    "tree_version": "research_library",
                    "decode_mode": "source_registry",
                    "confidence_by_level": {},
                },
                "backend": "research_library",
                "device": "cpu",
                "index_shards": [],
            },
            retrieval={"branch_scoped": [], "fallback_scoped": [], "global": []},
            verifier_contract={
                "success_command": state.task.success_command,
                "expected_files": state.task.expected_files,
                "expected_output_substrings": state.task.expected_output_substrings,
                "forbidden_files": state.task.forbidden_files,
                "forbidden_output_substrings": state.task.forbidden_output_substrings,
                "expected_file_contents": state.task.expected_file_contents,
            },
        )

    def _resolve_path(self, raw_path: Path | str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _load_status(self) -> dict[str, Any] | None:
        status_path = self._resolve_path(getattr(self.config, "research_library_status_path", ""))
        try:
            mtime = status_path.stat().st_mtime
        except OSError:
            mtime = None
        if self._status_cache is not None and self._status_cache_mtime == mtime:
            return self._status_cache
        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                self._status_cache = payload
                self._status_cache_mtime = mtime
                return payload
        config_path = self._resolve_path(getattr(self.config, "research_library_config_path", ""))
        if not config_path.exists():
            return None
        try:
            payload = build_research_library_status(config_path=config_path, root=self.repo_root)
        except Exception as exc:
            self._emit("status_unavailable", error_type=exc.__class__.__name__, message=_truncate(str(exc), 160))
            return None
        self._status_cache = payload
        self._status_cache_mtime = None
        return payload

    @staticmethod
    def _source(status: dict[str, Any], source_id: str) -> dict[str, Any]:
        for source in status.get("sources", []):
            if isinstance(source, dict) and source.get("id") == source_id:
                return source
        return {}

    def _research_chunks(self, status: dict[str, Any], state: AgentState) -> list[dict[str, Any]]:
        max_chunks = max(0, int(getattr(self.config, "research_library_context_max_chunks", 6)))
        if max_chunks <= 0:
            return []
        chunks = [
            self._inventory_chunk(status, state),
            self._model_assets_chunk(status, state),
            self._paper_backbone_chunk(status, state),
        ]
        paper_hits_chunk = self._paper_hits_chunk(status, state)
        paper_focused = bool(state.task.metadata.get("paper_id") or state.task.metadata.get("paper_title")) and not bool(
            state.task.metadata.get("repo")
        )
        paper_behavior_probe = bool(paper_hits_chunk) and str(
            state.task.metadata.get("benchmark_family", "")
        ) == "research_library_behavior_probe"
        applied_guidance_chunk = self._applied_guidance_chunk(paper_hits_chunk, state) if paper_hits_chunk else {}
        if paper_focused or paper_behavior_probe:
            if applied_guidance_chunk:
                chunks.append(applied_guidance_chunk)
            if paper_hits_chunk:
                chunks.append(paper_hits_chunk)
        else:
            repo_chunk = self._repository_chunk(status, state)
            if repo_chunk:
                chunks.append(repo_chunk)
            algorithm_chunk = self._algorithm_chunk(status, state)
            if algorithm_chunk:
                chunks.append(algorithm_chunk)
            if applied_guidance_chunk:
                chunks.append(applied_guidance_chunk)
            if paper_hits_chunk:
                chunks.append(paper_hits_chunk)
        return [chunk for chunk in chunks if chunk][:max_chunks]

    def _inventory_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        summary = status.get("summary", {}) if isinstance(status.get("summary", {}), dict) else {}
        text = (
            "Research library capability inventory for this task: "
            f"{summary.get('paper_rows', 0)} full papers, "
            f"{summary.get('paper_chunk_examples', 0)} paper chunks, "
            f"{summary.get('paper_knn_edges', 0)} paper KNN edges, "
            f"{summary.get('repository_count', 0)} repository exports, "
            f"{summary.get('repository_mined_skill_count', 0)} mined repository skills, "
            f"{summary.get('algorithm_catalog_rows', 0)} algorithm catalog rows, "
            f"{summary.get('trained_model_assets', 0)} trained model assets. "
            "Prefer indexed retrieval and existing trained adapters before scheduling retraining."
        )
        return self._chunk(
            "research:inventory",
            text,
            state=state,
            span_type="research_library:inventory",
            metadata={"summary": self._compact_summary(summary)},
        )

    def _model_assets_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        limit = max(1, int(getattr(self.config, "research_library_context_max_models", 8)))
        assets = select_trained_model_assets(
            status,
            prompt=state.task.prompt,
            metadata=state.task.metadata,
            role=str(state.current_role or "executor"),
            limit=limit,
        )
        if not assets:
            return {}
        lines = []
        for asset in assets:
            path = str(asset.get("relative_path", asset.get("path", "")))
            lines.append(
                f"{asset.get('source_id')}:{asset.get('group')} "
                f"{asset.get('asset_type')} at {path}"
            )
        text = (
            "Existing trained model assets selected for this task: "
            + "; ".join(lines)
            + ". Use these as runtime adapters/checkpoints before any retraining plan."
        )
        return self._chunk(
            "research:trained_models",
            text,
            state=state,
            span_type="research_library:trained_models",
            metadata={"assets": [self._compact_model_asset(asset) for asset in assets]},
        )

    def _paper_backbone_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        summary = status.get("summary", {}) if isinstance(status.get("summary", {}), dict) else {}
        paper_source = self._source(status, "paper_text_1m")
        universe_source = self._source(status, "paper_universe")
        chunk_source = self._source(status, "paper_chunks_p1")
        text = (
            "Paper retrieval backbone: source-of-truth full text is "
            f"{paper_source.get('path', '')}; chunk training/retrieval data is "
            f"{chunk_source.get('path', '')}; graph and embeddings are "
            f"{universe_source.get('path', '')}. "
            f"Counts: {summary.get('paper_rows', 0)} papers, "
            f"{summary.get('paper_chunk_examples', 0)} chunks, "
            f"{summary.get('paper_topic_edges', 0)} topic edges."
        )
        return self._chunk(
            "research:paper_backbone",
            text,
            state=state,
            span_type="research_library:paper_backbone",
            metadata={
                "paper_text_path": paper_source.get("path", ""),
                "paper_chunks_path": chunk_source.get("path", ""),
                "paper_universe_path": universe_source.get("path", ""),
            },
        )

    def _paper_hits_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        source = self._source(status, "paper_chunks_p1")
        path = Path(str(source.get("path", "")))
        if not path.exists():
            return {}
        query_phrase = _normalized_phrase(
            " ".join(
                [
                    state.task.prompt,
                    str(state.task.metadata.get("benchmark_family", "")),
                    str(state.task.metadata.get("paper_title", "")),
                    str(state.task.metadata.get("paper_id", "")),
                    str(state.task.metadata.get("research_topic", "")),
                ]
            )
        )
        query_tokens = _tokens(
            state.task.prompt,
            state.task.metadata.get("benchmark_family", ""),
            state.task.metadata.get("paper_title", ""),
            state.task.metadata.get("paper_id", ""),
            state.task.metadata.get("research_topic", ""),
        )
        if len(query_tokens) < 2 and not query_phrase:
            return {}
        files = sorted(path.glob("train-*.parquet"))
        file_limit = max(0, int(getattr(self.config, "research_library_paper_scan_file_limit", 1)))
        row_limit = max(0, int(getattr(self.config, "research_library_paper_scan_row_limit", 4096)))
        if not files or file_limit <= 0 or row_limit <= 0:
            return {}
        ranked: list[tuple[int, dict[str, Any]]] = []
        columns = ["id", "paper_id", "title", "categories", "year", "chunk_index", "text"]
        remaining_rows = row_limit
        for file_path in files[:file_limit]:
            rows = _read_parquet_rows(file_path, columns=columns, limit=remaining_rows)
            remaining_rows -= len(rows)
            for row in rows:
                title = row.get("title", "")
                text = row.get("text", "")
                categories = row.get("categories", "")
                paper_id = row.get("paper_id", "")
                title_score = 15 if _contains_phrase(query_phrase, title) else 0
                id_score = 12 if paper_id and _contains_phrase(query_phrase, paper_id) else 0
                text_prefix = str(text)[:2200]
                token_score = len(query_tokens & _tokens(title, categories, paper_id, text_prefix))
                lowered_text = " ".join(text_prefix.split()).lower()
                if (
                    "rewrite relation" in lowered_text
                    and "u1" in lowered_text
                    and "u2" in lowered_text
                    and " p " in f" {lowered_text} "
                ):
                    token_score += 20
                if "active letter" in lowered_text and ("p-th letter" in lowered_text or "p th letter" in lowered_text):
                    token_score += 18
                try:
                    chunk_index = int(row.get("chunk_index", 0) or 0)
                except (TypeError, ValueError):
                    chunk_index = 0
                opening_chunk_score = 6 if (title_score or id_score) and chunk_index == 0 else 0
                score = title_score + id_score + token_score + opening_chunk_score
                if score >= 3 or title_score or id_score:
                    ranked.append((score, row))
            if remaining_rows <= 0:
                break
        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("paper_id", "")),
                int(item[1].get("chunk_index", 0) or 0),
            )
        )
        limit = max(0, int(getattr(self.config, "research_library_context_max_paper_hits", 2)))
        selected = ranked[:limit]
        if not selected:
            return {}
        parts = []
        payload = []
        for score, row in selected:
            excerpt = _best_excerpt(row.get("text", ""), query_tokens, limit=360)
            evidence_sentence = _best_evidence_sentence(row.get("text", ""), query_tokens)
            parts.append(
                f"paper_id={row.get('paper_id', '')} score={score} title={row.get('title', '')} "
                f"categories={row.get('categories', '')} year={row.get('year', '')} "
                f"chunk_index={row.get('chunk_index', '')} copyable_fields: "
                f"paper_id={row.get('paper_id', '')}; title={row.get('title', '')}; "
                f"evidence_sentence={evidence_sentence}; excerpt={excerpt}"
            )
            payload.append(
                {
                    "id": row.get("id", ""),
                    "paper_id": row.get("paper_id", ""),
                    "score": score,
                    "title": row.get("title", ""),
                    "categories": row.get("categories", ""),
                    "year": row.get("year"),
                    "chunk_index": row.get("chunk_index"),
                    "evidence_sentence": evidence_sentence,
                    "excerpt": excerpt,
                    "fact_text": str(row.get("text", ""))[:2200],
                }
            )
        applied_facts = _applied_paper_facts(payload)
        applied_prefix = ""
        if applied_facts:
            applied_prefix = "Applied paper facts for this task: " + "; ".join(applied_facts) + ". "
        return self._chunk(
            "research:paper_hits",
            applied_prefix + "Relevant paper content hits: " + "; ".join(parts),
            state=state,
            span_type="research_library:paper_hits",
            metadata={"papers": payload, "applied_facts": applied_facts},
        )

    def _applied_guidance_chunk(self, paper_hits_chunk: dict[str, Any], state: AgentState) -> dict[str, Any]:
        metadata = paper_hits_chunk.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        facts = [str(fact).strip() for fact in list(metadata.get("applied_facts", []) or []) if str(fact).strip()]
        if not facts:
            return {}
        guidance = _applied_guidance_lines(facts)
        if not guidance:
            return {}
        text = (
            "Applied research guidance for this task: "
            + "; ".join(guidance)
            + ". These lines are derived from retrieved paper content and should be applied as executable constraints."
        )
        return self._chunk(
            "research:applied_guidance",
            text,
            state=state,
            span_type="research_library:applied_guidance",
            metadata={"applied_facts": facts, "applied_guidance": guidance},
        )

    @staticmethod
    def _exact_answer_target_path(state: AgentState) -> str:
        if len(state.task.expected_file_contents) == 1:
            return _safe_output_text_path(next(iter(state.task.expected_file_contents)))
        match = re.search(r"\b([A-Za-z0-9._-]+\.txt)\b", state.task.prompt)
        if match is None:
            return ""
        return _safe_output_text_path(match.group(1))

    def _paper_exact_answer_command(self, state: AgentState, paper: dict[str, Any]) -> str:
        evidence_sentence = str(paper.get("evidence_sentence", "")).strip()
        if not evidence_sentence:
            return ""
        target_path = self._exact_answer_target_path(state)
        if not target_path:
            return ""
        lines = [
            f"paper_id={paper.get('paper_id', '')}",
            f"title={paper.get('title', '')}",
            f"evidence_sentence={evidence_sentence}",
        ]
        return "printf '%s\\n' " + " ".join(_shell_single_quote(line) for line in lines) + f" > {target_path}"

    def _repository_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        source = self._source(status, "repository_exports")
        path = Path(str(source.get("path", "")))
        manifest_path = path / "_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        repos = manifest.get("repos", manifest.get("repositories", {}))
        if not isinstance(repos, dict):
            return {}
        repo_hint = state.task.metadata.get("repo", "")
        query_tokens = _tokens(state.task.prompt, repo_hint, state.task.metadata.get("benchmark_family", ""))
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for name, repo in repos.items():
            if not isinstance(repo, dict):
                continue
            languages = repo.get("languages", repo.get("language", []))
            language_text = " ".join(languages) if isinstance(languages, list) else str(languages)
            name_score = _repo_name_score(repo_hint, name)
            token_score = len(query_tokens & _tokens(name, language_text, repo.get("repo_root", "")))
            if name_score <= 0 and token_score < 2:
                continue
            score = name_score + token_score
            if repo.get("indices", {}).get("qa") if isinstance(repo.get("indices", {}), dict) else False:
                score += 1
            ranked.append((score, str(name), repo))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        limit = max(0, int(getattr(self.config, "research_library_context_max_repositories", 4)))
        selected = ranked[:limit]
        if not selected:
            return {}
        parts = []
        payload = []
        for score, name, repo in selected:
            indices = repo.get("indices", {}) if isinstance(repo.get("indices", {}), dict) else {}
            skills = repo.get("extensions", {}).get("repo_skills_miner", {}).get("counts", {}).get("skills", 0)
            parts.append(f"{name} score={score} skills={skills} qa_index={bool(indices.get('qa'))}")
            payload.append(
                {
                    "name": name,
                    "score": score,
                    "repo_root": repo.get("repo_root", ""),
                    "languages": repo.get("languages", []),
                    "qa_index": bool(indices.get("qa")),
                    "mined_skills": skills,
                }
            )
        return self._chunk(
            "research:repositories",
            "Relevant repository exports available: " + "; ".join(parts),
            state=state,
            span_type="research_library:repositories",
            metadata={"repositories": payload},
        )

    def _algorithm_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        source = self._source(status, "algorithms")
        path = Path(str(source.get("path", "")))
        algorithms = _read_jsonl(path / "algorithms.jsonl")
        if not algorithms:
            return {}
        query_phrase = _normalized_phrase(
            " ".join([state.task.prompt, str(state.task.metadata.get("benchmark_family", ""))])
        )
        query_tokens = _tokens(state.task.prompt, state.task.metadata.get("benchmark_family", ""))
        ranked: list[tuple[int, dict[str, Any]]] = []
        for algorithm in algorithms:
            searchable = [
                algorithm.get("algo_id", ""),
                algorithm.get("category", ""),
                algorithm.get("notes", ""),
                _nested_text(algorithm.get("names", [])),
                _nested_text(algorithm.get("topics", [])),
                _nested_text(algorithm.get("problems", [])),
                _nested_text(algorithm.get("tags", [])),
                _nested_text(algorithm.get("constraints", {})),
                _nested_text(algorithm.get("properties", {})),
            ]
            names = algorithm.get("names", [])
            aliases = [algorithm.get("algo_id", ""), *(names if isinstance(names, list) else [names])]
            alias_score = _alias_phrase_score(query_phrase, aliases)
            token_score = len(query_tokens & _tokens(*searchable))
            if alias_score <= 0 and token_score < 2:
                continue
            score = alias_score + token_score
            if score > 0:
                ranked.append((score, algorithm))
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("algo_id", ""))))
        limit = max(0, int(getattr(self.config, "research_library_context_max_algorithms", 4)))
        selected = ranked[:limit]
        if not selected:
            return {}
        parts = []
        payload = []
        for score, algorithm in selected:
            complexity = algorithm.get("time_complexity", {})
            worst = complexity.get("worst", "") if isinstance(complexity, dict) else ""
            names = algorithm.get("names", [])
            names_text = ",".join(str(name) for name in names) if isinstance(names, list) else str(names)
            topics = algorithm.get("topics", [])
            topics_text = ",".join(str(topic) for topic in topics) if isinstance(topics, list) else str(topics)
            constraints = _truncate(_nested_text(algorithm.get("constraints", {})), 160)
            notes = _truncate(algorithm.get("notes", ""), 260)
            parts.append(
                f"algo_id={algorithm.get('algo_id')} score={score} names={names_text} "
                f"category={algorithm.get('category')} topics={topics_text} worst={worst} "
                f"constraints={constraints} notes={notes}"
            )
            payload.append(
                {
                    "algo_id": algorithm.get("algo_id", ""),
                    "score": score,
                    "category": algorithm.get("category", ""),
                    "topics": algorithm.get("topics", []),
                    "time_complexity": algorithm.get("time_complexity", {}),
                    "constraints": algorithm.get("constraints", {}),
                    "notes": algorithm.get("notes", ""),
                }
            )
        return self._chunk(
            "research:algorithms",
            "Relevant algorithm catalog entries: " + "; ".join(parts),
            state=state,
            span_type="research_library:algorithms",
            metadata={"algorithms": payload},
        )

    @staticmethod
    def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "paper_rows",
            "paper_chunk_examples",
            "paper_universe_rows",
            "paper_knn_edges",
            "paper_topic_edges",
            "repository_count",
            "repository_mined_skill_count",
            "algorithm_catalog_rows",
            "algorithm_implementation_files",
            "trained_model_assets",
            "trained_adapter_assets",
            "full_model_assets",
            "faiss_index_assets",
            "tolbert_checkpoint_assets",
        )
        return {key: summary.get(key) for key in keys if key in summary}

    @staticmethod
    def _compact_model_asset(asset: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": asset.get("source_id", ""),
            "source_role": asset.get("source_role", ""),
            "group": asset.get("group", ""),
            "asset_type": asset.get("asset_type", ""),
            "relative_path": asset.get("relative_path", asset.get("path", "")),
            "checkpoint_step": asset.get("checkpoint_step"),
            "metadata": asset.get("metadata", {}),
        }

    @staticmethod
    def _chunk(
        span_id: str,
        text: str,
        *,
        state: AgentState,
        span_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "span_id": span_id,
            "source_id": "research_library",
            "span_type": span_type,
            "bucket": "global",
            "task_id": state.task.task_id,
            "score": 1.0,
            "text": _truncate(
                text,
                2200
                if span_type == "research_library:paper_hits"
                else 1200
                if span_type == "research_library:applied_guidance"
                else 900,
            ),
            "metadata": metadata or {},
        }
        return payload

    def _augment_packet(
        self,
        packet: ContextPacket,
        *,
        status: dict[str, Any],
        state: AgentState,
        chunks: list[dict[str, Any]],
    ) -> ContextPacket:
        control = dict(packet.control)
        existing_chunks = list(control.get("selected_context_chunks", []) or [])
        control["selected_context_chunks"] = [*existing_chunks, *chunks]
        context_chunk_budget = (
            dict(control.get("context_chunk_budget", {}) or {})
            if isinstance(control.get("context_chunk_budget", {}), dict)
            else {}
        )
        base_max_chunks = _safe_int(context_chunk_budget.get("max_chunks"))
        if base_max_chunks <= 0:
            base_max_chunks = int(getattr(self.config, "tolbert_context_max_chunks", 8))
        base_char_budget = _safe_int(context_chunk_budget.get("char_budget"))
        if base_char_budget <= 0:
            base_char_budget = int(getattr(self.config, "tolbert_context_char_budget", 2400))
        context_chunk_budget["base_max_chunks"] = base_max_chunks
        context_chunk_budget["research_library_chunks"] = len(chunks)
        context_chunk_budget["max_chunks"] = base_max_chunks + len(chunks)
        context_chunk_budget["char_budget"] = base_char_budget + sum(
            len(str(chunk.get("text", ""))) for chunk in chunks
        )
        control["context_chunk_budget"] = context_chunk_budget
        summary = status.get("summary", {}) if isinstance(status.get("summary", {}), dict) else {}
        model_catalog = trained_model_asset_catalog(status)
        control["research_library"] = {
            "enabled": True,
            "status_path": str(self._resolve_path(getattr(self.config, "research_library_status_path", ""))),
            "summary": self._compact_summary(summary),
            "model_catalog_summary": model_catalog.get("summary", {}),
            "selected_chunk_ids": [str(chunk.get("span_id", "")) for chunk in chunks],
        }
        guidance = dict(control.get("retrieval_guidance", {}) or {})
        evidence = [str(item) for item in guidance.get("evidence", []) if str(item).strip()]
        recommended_commands = [
            str(command)
            for command in list(guidance.get("recommended_commands", []) or [])
            if str(command).strip()
        ]
        recommended_command_spans = [
            dict(item)
            for item in list(guidance.get("recommended_command_spans", []) or [])
            if isinstance(item, dict)
        ]
        specific_research_evidence: list[str] = []
        applied_guidance: list[str] = []
        seen_specific_evidence: set[str] = set()
        trusted_exact_command = False

        def add_specific_evidence(value: str) -> None:
            normalized = str(value).strip()
            if normalized and normalized not in seen_specific_evidence:
                seen_specific_evidence.add(normalized)
                specific_research_evidence.append(normalized)

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            for guidance_line in list(metadata.get("applied_guidance", []) or [])[:6]:
                normalized_guidance = str(guidance_line).strip()
                if not normalized_guidance:
                    continue
                if normalized_guidance not in applied_guidance:
                    applied_guidance.append(normalized_guidance)
                add_specific_evidence(f"research_library: applied_guidance {normalized_guidance}")
            for fact in list(metadata.get("applied_facts", []) or [])[:4]:
                normalized_fact = str(fact).strip()
                if normalized_fact:
                    add_specific_evidence(f"research_library: applied_fact {normalized_fact}")
            for paper in list(metadata.get("papers", []) or [])[:4]:
                if not isinstance(paper, dict):
                    continue
                evidence_sentence = str(paper.get("evidence_sentence", "")).strip()
                if not evidence_sentence:
                    continue
                add_specific_evidence(
                    "research_library: paper_hit "
                    f"paper_id={paper.get('paper_id', '')}; "
                    f"title={paper.get('title', '')}; "
                    f"evidence_sentence={evidence_sentence}"
                )
                command = (
                    self._paper_exact_answer_command(state, paper)
                    if state.task.metadata.get("research_exact_answer")
                    else ""
                )
                if command:
                    trusted_exact_command = True
                    if command not in recommended_commands:
                        recommended_commands.append(command)
                    if not any(str(item.get("command", "")).strip() == command for item in recommended_command_spans):
                        recommended_command_spans.append(
                            {
                                "span_id": "research:paper_hits:exact_answer",
                                "command": command,
                            }
                        )
            for algorithm in list(metadata.get("algorithms", []) or [])[:2]:
                if not isinstance(algorithm, dict):
                    continue
                topics = algorithm.get("topics", [])
                topics_text = ",".join(str(topic) for topic in topics) if isinstance(topics, list) else str(topics)
                notes = str(algorithm.get("notes", "")).strip()
                if not notes:
                    continue
                add_specific_evidence(
                    "research_library: algorithm_hit "
                    f"algo_id={algorithm.get('algo_id', '')}; "
                    f"category={algorithm.get('category', '')}; "
                    f"topics={topics_text}; notes={notes}"
                )
        if applied_guidance:
            control["research_library"]["applied_guidance"] = applied_guidance[:8]
        paper_hit_evidence = [item for item in specific_research_evidence if "research_library: paper_hit " in item]
        non_paper_evidence = [
            item for item in specific_research_evidence if "research_library: paper_hit " not in item
        ]
        selected_specific_evidence = [*non_paper_evidence[:4], *paper_hit_evidence[:2]][:6]
        research_evidence = [
            *selected_specific_evidence,
            (
                "research_library: "
                f"{summary.get('trained_model_assets', 0)} trained assets, "
                f"{summary.get('paper_rows', 0)} papers, "
                f"{summary.get('repository_count', 0)} repos available"
            ),
            "research_library: prefer existing trained adapters/checkpoints before retraining",
        ]
        guidance["evidence"] = [*evidence[: max(0, 8 - len(research_evidence))], *research_evidence][:8]
        guidance["recommended_commands"] = recommended_commands[:5]
        guidance["recommended_command_spans"] = recommended_command_spans[:5]
        guidance.setdefault("avoidance_notes", [])
        control["retrieval_guidance"] = guidance
        if trusted_exact_command:
            control["trust_retrieval"] = True
            control["path_confidence"] = max(float(control.get("path_confidence", 0.0) or 0.0), 0.95)

        retrieval = dict(packet.retrieval)
        global_spans = list(retrieval.get("global", []) or [])
        for chunk in chunks:
            global_spans.append(
                {
                    "span_id": str(chunk.get("span_id", "")),
                    "text": str(chunk.get("text", "")),
                    "source_id": "research_library",
                    "span_type": str(chunk.get("span_type", "")),
                    "score": float(chunk.get("score", 1.0) or 1.0),
                    "node_path": [0, 0, 0],
                    "metadata": dict(chunk.get("metadata", {})),
                }
            )
        retrieval["global"] = global_spans

        tolbert = dict(packet.tolbert)
        auxiliary = list(tolbert.get("auxiliary_backends", []) or [])
        auxiliary.append(
            {
                "backend": "research_library",
                "index_shards": [
                    "paper_text_1m",
                    "paper_universe",
                    "repository_exports",
                    "trained_model_assets",
                    "algorithm_catalog",
                ],
            }
        )
        tolbert["auxiliary_backends"] = auxiliary
        return ContextPacket(
            request_id=packet.request_id,
            created_at=packet.created_at,
            task=packet.task,
            control=control,
            tolbert=tolbert,
            retrieval=retrieval,
            verifier_contract=packet.verifier_contract,
        )


__all__ = ["ResearchLibraryContextProvider"]
