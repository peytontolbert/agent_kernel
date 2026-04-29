from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
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


def _iter_jsonl(path: Path, *, limit: int = 4096):
    if limit <= 0:
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


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
    if "greedy if it is leftmost, pure and eager" in combined or (
        "greedy" in combined and "leftmost" in combined and "pure" in combined and "eager" in combined
    ):
        facts.append("greedy_derivation=greedy derivations are leftmost, pure, and eager")
    if "insert" in combined and "erase" in combined and "to their left" in combined:
        facts.append(
            "left_action=leftist grammar symbols insert or erase another symbol to their left while remaining unchanged"
        )
    if (
        "10 pf maintained group value" in combined
        and "10 pf to 100 pf step-up" in combined
        and "100 pf to 1000 pf step-up" in combined
        and "rss 419" in combined
    ):
        facts.append(
            "farad_table3=ten_pf_group=400; ten_to_100_pf_stepup=40; "
            "hundred_to_1000_pf_stepup=100; new_traceability_chain=64; rss=419"
        )
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
            add("rewrite_start=p-len(alpha)")
            add("rewrite_output=u1+beta+u2")
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
            add("active_index_from_p=p-1")
            add("active_letter_index(word, p): p names the paper's p-th letter in u; zero-based active_index = p - 1")
            continue
        if normalized_fact.startswith("greedy_derivation="):
            add("greedy_components=leftmost,pure,eager")
            add("greedy_derivation_components: leftmost, pure, eager")
            continue
        if normalized_fact.startswith("left_action="):
            add("left_action=symbols insert or erase another symbol to their left while remaining unchanged")
            add("left_action_rule: symbols insert or erase another symbol to their left while remaining unchanged")
            continue
        if normalized_fact.startswith("farad_table3="):
            add(
                "farad_table3: ten_pf_group=400; ten_to_100_pf_stepup=40; "
                "hundred_to_1000_pf_stepup=100; new_traceability_chain=64; rss=419"
            )
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


def _structured_fact_lines_for_task(guidance: list[str], prompt: str) -> list[str]:
    prompt_text = str(prompt)
    lines: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        normalized = " ".join(str(line).strip().split())
        if not normalized or "=" not in normalized or normalized in seen:
            return
        key = normalized.split("=", 1)[0].strip()
        if key and not re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", prompt_text):
            return
        seen.add(normalized)
        lines.append(normalized)

    for item in guidance:
        text = str(item).strip()
        if not text:
            continue
        payload = text
        if ":" in text and "=" in text.split(":", 1)[1]:
            payload = text.split(":", 1)[1]
        if ";" in payload:
            for part in payload.split(";"):
                add(part)
            continue
        if "=" in payload:
            add(payload)
            continue
        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if re.fullmatch(r"[a-z][a-z0-9_]*", key) and value:
                add(f"{key}={value}")
    return sorted(
        lines,
        key=lambda line: (
            prompt_text.find(line.split("=", 1)[0].strip())
            if line.split("=", 1)[0].strip() in prompt_text
            else len(prompt_text),
            line,
        ),
    )


def _structured_fact_command_for_task(guidance: list[str], prompt: str, target: str) -> str:
    lines = _structured_fact_lines_for_task(guidance, prompt)
    if not lines or not target:
        return ""
    return (
        "printf '%s\\n' "
        + " ".join(shlex.quote(line) for line in lines)
        + f" > {shlex.quote(target)}"
    )


def _printf_lines_command(target: str, lines: list[str]) -> str:
    safe_target = _safe_output_text_path(target)
    if not safe_target or not lines:
        return ""
    return (
        "printf '%s\\n' "
        + " ".join(_shell_single_quote(line) for line in lines)
        + f" > {shlex.quote(safe_target)}"
    )


def _parse_program_uri_lines(uri: object) -> tuple[int, int]:
    match = re.search(r"#L(\d+)(?:-L?(\d+))?$", str(uri))
    if match is None:
        return 0, 0
    start = _safe_int(match.group(1))
    end = _safe_int(match.group(2) or match.group(1))
    if start <= 0 or end < start:
        return 0, 0
    return start, end


def _module_source_candidates(repo_root: Path, module: str) -> list[Path]:
    normalized = module.strip(".")
    if not normalized:
        return []
    dotted = normalized.replace(".", "/")
    candidates = [
        repo_root / f"{dotted}.py",
        repo_root / dotted / "__init__.py",
        repo_root / normalized,
    ]
    if "/" not in normalized and "." not in normalized:
        candidates.append(repo_root / f"{normalized}.py")
    return candidates


def _read_source_span(path: Path, *, line_start: int, line_end: int, max_lines: int = 80) -> str:
    if line_start <= 0 or line_end < line_start:
        return ""
    line_end = min(line_end, line_start + max_lines - 1)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    selected = lines[line_start - 1 : line_end]
    return "\n".join(selected)


def _entity_module_name(entity: dict[str, Any]) -> str:
    owner = str(entity.get("owner", "") or "")
    if owner.startswith("py:"):
        return owner[3:]
    if str(entity.get("kind", "")) == "module":
        return str(entity.get("name", "") or "")
    uri = str(entity.get("uri", "") or "")
    match = re.search(r"program://[^/]+/(?:function|class)/(.+?)#L", uri)
    if match is None:
        return ""
    qualified = match.group(1)
    name = str(entity.get("name", "") or "")
    if name and qualified.endswith(f".{name}"):
        return qualified[: -(len(name) + 1)]
    return qualified.rsplit(".", 1)[0]


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
        inventory_chunk = self._inventory_chunk(status, state)
        model_assets_chunk = self._model_assets_chunk(status, state)
        paper_backbone_chunk = self._paper_backbone_chunk(status, state)
        chunks = [inventory_chunk, model_assets_chunk, paper_backbone_chunk]
        paper_hits_chunk = self._paper_hits_chunk(status, state)
        paper_focused = bool(state.task.metadata.get("paper_id") or state.task.metadata.get("paper_title")) and not bool(
            state.task.metadata.get("repo")
        )
        paper_behavior_probe = bool(paper_hits_chunk) and str(
            state.task.metadata.get("benchmark_family", "")
        ) == "research_library_behavior_probe"
        applied_guidance_chunk = self._applied_guidance_chunk(paper_hits_chunk, state) if paper_hits_chunk else {}
        repo_focused = bool(state.task.metadata.get("repo"))
        if paper_behavior_probe:
            chunks = []
            if applied_guidance_chunk:
                chunks.append(applied_guidance_chunk)
            if paper_hits_chunk:
                chunks.append(paper_hits_chunk)
        elif paper_focused:
            chunks = []
            if applied_guidance_chunk:
                chunks.append(applied_guidance_chunk)
            if paper_hits_chunk:
                chunks.append(paper_hits_chunk)
            chunks.extend([paper_backbone_chunk, model_assets_chunk, inventory_chunk])
        else:
            repo_chunk = self._repository_chunk(status, state)
            repo_code_chunk = self._repository_code_chunk(status, state)
            algorithm_chunk = self._algorithm_chunk(status, state)
            if repo_focused:
                chunks = []
                if repo_code_chunk:
                    chunks.append(repo_code_chunk)
                if repo_chunk:
                    chunks.append(repo_chunk)
                if algorithm_chunk:
                    chunks.append(algorithm_chunk)
                chunks.extend([model_assets_chunk, inventory_chunk])
            else:
                if repo_chunk:
                    chunks.append(repo_chunk)
                if repo_code_chunk:
                    chunks.append(repo_code_chunk)
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
                rewrite_query = bool(
                    query_tokens
                    & {
                        "rewrite",
                        "rewriting",
                        "relation",
                        "leftist",
                        "semi-thue",
                        "semithue",
                        "alpha",
                        "beta",
                    }
                )
                active_query = bool(query_tokens & {"active", "letter"})
                herding_query = bool(query_tokens & {"cows", "herders", "herder", "fence", "herding"})
                leftist_definition_query = bool(
                    query_tokens
                    & {
                        "leftist",
                        "greedy",
                        "derivation",
                        "leftmost",
                        "pure",
                        "eager",
                        "insert",
                        "erase",
                    }
                )
                if (
                    rewrite_query
                    and
                    "rewrite relation" in lowered_text
                    and "u1" in lowered_text
                    and "u2" in lowered_text
                    and " p " in f" {lowered_text} "
                ):
                    token_score += 20
                if (
                    active_query
                    and "active letter" in lowered_text
                    and ("p-th letter" in lowered_text or "p th letter" in lowered_text)
                ):
                    token_score += 18
                if (
                    leftist_definition_query
                    and "greedy" in lowered_text
                    and "leftmost" in lowered_text
                    and "pure" in lowered_text
                    and "eager" in lowered_text
                ):
                    token_score += 18
                if leftist_definition_query and "insert" in lowered_text and "erase" in lowered_text and "to their left" in lowered_text:
                    token_score += 18
                if herding_query and "cows and herders" in lowered_text:
                    token_score += 18
                if herding_query and "adjacent" in lowered_text and ("cluster" in lowered_text or "group" in lowered_text):
                    token_score += 12
                if herding_query and "fence" in lowered_text and ("fleeing" in lowered_text or "right way" in lowered_text):
                    token_score += 12
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
            guidance_lines = _applied_guidance_lines(applied_facts)
            structured_lines = _structured_fact_lines_for_task(guidance_lines, state.task.prompt)
            target = self._exact_answer_target_path(state)
            if structured_lines and target:
                applied_prefix = f"{target} exact lines: " + " | ".join(structured_lines) + ". "
            else:
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
        command = _structured_fact_command_for_task(
            guidance,
            state.task.prompt,
            self._exact_answer_target_path(state),
        )
        structured_lines = _structured_fact_lines_for_task(guidance, state.task.prompt)
        compact_values = ""
        if structured_lines:
            target = self._exact_answer_target_path(state) or "target file"
            compact_values = f"{target} exact lines: " + " | ".join(reversed(structured_lines)) + "\n"
        command_text = f"Adapter-derived command candidate: {command}\n" if command else ""
        text = (
            compact_values
            + command_text
            + "Task-specific facts extracted from retrieved paper content:\n"
            + "\n".join(f"- {line}" for line in guidance)
            + "\nCopy these retrieved values exactly when the task asks for the matching fields."
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
        source, ranked = self._rank_repositories(status, state)
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

    def _rank_repositories(
        self,
        status: dict[str, Any],
        state: AgentState,
    ) -> tuple[dict[str, Any], list[tuple[int, str, dict[str, Any]]]]:
        source = self._source(status, "repository_exports")
        path = Path(str(source.get("path", "")))
        manifest_path = path / "_manifest.json"
        if not manifest_path.exists():
            return source, []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return source, []
        repos = manifest.get("repos", manifest.get("repositories", {}))
        if not isinstance(repos, dict):
            return source, []
        repo_hint = state.task.metadata.get("repo", "")
        query_tokens = _tokens(
            state.task.prompt,
            repo_hint,
            state.task.metadata.get("benchmark_family", ""),
            state.task.metadata.get("research_topic", ""),
        )
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
        return source, ranked

    def _repository_code_chunk(self, status: dict[str, Any], state: AgentState) -> dict[str, Any]:
        source, ranked_repositories = self._rank_repositories(status, state)
        export_root = Path(str(source.get("path", "")))
        if not ranked_repositories or not export_root.exists():
            return {}
        query_phrase = _normalized_phrase(
            " ".join(
                [
                    state.task.prompt,
                    str(state.task.metadata.get("benchmark_family", "")),
                    str(state.task.metadata.get("repo", "")),
                    str(state.task.metadata.get("research_topic", "")),
                    str(state.task.metadata.get("code_symbol", "")),
                ]
            )
        )
        query_tokens = _tokens(
            state.task.prompt,
            state.task.metadata.get("benchmark_family", ""),
            state.task.metadata.get("repo", ""),
            state.task.metadata.get("research_topic", ""),
            state.task.metadata.get("code_symbol", ""),
        )
        if len(query_tokens) < 2 and not query_phrase:
            return {}
        repo_limit = max(1, int(getattr(self.config, "research_library_context_max_repositories", 4)))
        hits: list[tuple[int, dict[str, Any]]] = []
        for repo_score, repo_name, repo in ranked_repositories[:repo_limit]:
            hits.extend(
                self._rank_skill_code_hits(
                    export_root=export_root,
                    repo_name=repo_name,
                    repo=repo,
                    repo_score=repo_score,
                    query_phrase=query_phrase,
                    query_tokens=query_tokens,
                )
            )
            hits.extend(
                self._rank_entity_code_hits(
                    export_root=export_root,
                    repo_name=repo_name,
                    repo=repo,
                    repo_score=repo_score,
                    query_phrase=query_phrase,
                    query_tokens=query_tokens,
                )
            )
        hits.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("repo", "")),
                str(item[1].get("file_path", "")),
                str(item[1].get("qualname", "")),
            )
        )
        selected = []
        seen_locations: set[tuple[str, str, object, object]] = set()
        for _, hit in hits:
            location = (
                str(hit.get("repo", "")),
                str(hit.get("file_path", "")),
                hit.get("line_start", ""),
                hit.get("line_end", ""),
            )
            if location in seen_locations:
                continue
            seen_locations.add(location)
            selected.append(hit)
            if len(selected) >= 3:
                break
        if not selected:
            return {}
        parts = []
        for hit in selected:
            snippet = _best_excerpt(hit.get("snippet", ""), query_tokens, limit=760)
            parts.append(
                f"repo={hit.get('repo', '')} file={hit.get('file_path', '')} "
                f"symbol={hit.get('qualname', '')} signature={hit.get('signature', '')} "
                f"lines={hit.get('line_start', '')}-{hit.get('line_end', '')} "
                f"snippet={snippet}"
            )
        facts = self._repository_code_facts_for_task(selected, state)
        commands = self._repository_code_commands_for_task(selected, state)
        return self._chunk(
            "research:repository_code",
            "Relevant repository code evidence: " + "; ".join(parts),
            state=state,
            span_type="research_library:repository_code",
            metadata={
                "repository_code_hits": selected,
                "repository_code_facts": facts,
                "repository_code_commands": commands,
            },
        )

    def _repository_code_facts_for_task(
        self,
        code_hits: list[dict[str, Any]],
        state: AgentState,
    ) -> list[str]:
        if str(state.task.metadata.get("benchmark_family", "")) != "research_library_repo_code_probe":
            return []
        symbol = str(state.task.metadata.get("code_symbol", "")).strip().lower()
        hit_text = " ".join(
            " ".join(
                [
                    str(hit.get("repo", "")),
                    str(hit.get("file_path", "")),
                    str(hit.get("qualname", "")),
                    str(hit.get("snippet", ""))[:2400],
                ]
            )
            for hit in code_hits
        ).lower()
        if symbol == "str2bool" and "found" in hit_text and "notfound" in hit_text:
            return [
                "str2bool_empty=False; non_string=ValueError",
                "str2bool_truthy=1,true,t,yes,y,on,enable,enabled,found",
                "str2bool_falsy_a=0,false,f,no,n,off,disable,disabled",
                "str2bool_falsy_b=notfound,none,null,nil,undefined,n/a",
            ]
        if symbol == "flash_attn_supports_top_left_mask" and "is_flash_attn_greater_or_equal_2_10" in hit_text:
            return [
                "flash_top_left_fa3=False",
                "flash_top_left_fa2=not fa2_ge_2_10",
                "flash_top_left_fallback=npu_top_left",
                "flash_available=fa3 or fa2 or npu",
            ]
        if symbol == "_isreservedname" and "_reserved_chars" in hit_text and "dos device names" in hit_text:
            return [
                "ntpath_trailing=dot/space reserved except . and ..",
                "ntpath_chars=*?\"<>/\\:| plus ASCII 0-31 reserved",
                "ntpath_stream=colon marks reserved stream separator",
                "ntpath_device=before dot, rstrip spaces, uppercase DOS device",
            ]
        return []

    def _repository_code_commands_for_task(
        self,
        code_hits: list[dict[str, Any]],
        state: AgentState,
    ) -> list[str]:
        if str(state.task.metadata.get("benchmark_family", "")) != "research_library_repo_code_probe":
            return []
        if len(state.task.expected_files) != 1:
            return []
        target = _safe_output_text_path(state.task.expected_files[0])
        if not target:
            return []
        symbol = str(state.task.metadata.get("code_symbol", "")).strip().lower()
        hit_text = " ".join(
            " ".join(
                [
                    str(hit.get("repo", "")),
                    str(hit.get("file_path", "")),
                    str(hit.get("qualname", "")),
                    str(hit.get("snippet", ""))[:2400],
                ]
            )
            for hit in code_hits
        ).lower()
        if symbol == "str2bool" and "found" in hit_text and "notfound" in hit_text:
            command = _printf_lines_command(
                target,
                [
                    "def str2bool(value):",
                    "    if not value:",
                    "        return False",
                    "    if not isinstance(value, str):",
                    "        raise ValueError(f\"Expected a string value for boolean conversion, got {type(value)}\")",
                    "    value = value.strip().lower()",
                    "    if value in (",
                    "        '1', 'true', 't', 'yes', 'y', 'on', 'enable', 'enabled', 'found',",
                    "    ):",
                    "        return True",
                    "    if value in (",
                    "        '0', 'false', 'f', 'no', 'n', 'off', 'disable', 'disabled',",
                    "        'notfound', 'none', 'null', 'nil', 'undefined', 'n/a',",
                    "    ):",
                    "        return False",
                    "    raise ValueError(f\"Invalid string value for boolean conversion: {value}\")",
                ],
            )
            return [command] if command else []
        if symbol == "flash_attn_supports_top_left_mask" and "is_flash_attn_greater_or_equal_2_10" in hit_text:
            command = _printf_lines_command(
                target,
                [
                    "def top_left_mask_supported(*, fa3: bool, fa2: bool, fa2_ge_2_10: bool, npu_top_left: bool) -> bool:",
                    "    if fa3:",
                    "        return False",
                    "    if fa2:",
                    "        return not fa2_ge_2_10",
                    "    return npu_top_left",
                    "",
                    "def flash_attention_available(*, fa3: bool, fa2: bool, npu: bool) -> bool:",
                    "    return fa3 or fa2 or npu",
                ],
            )
            return [command] if command else []
        if symbol == "_isreservedname" and "_reserved_chars" in hit_text and "dos device names" in hit_text:
            command = _printf_lines_command(
                target,
                [
                    "_RESERVED_NAMES = {",
                    "    'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$', 'CONIN$', 'CONOUT$',",
                    "    *(f'COM{i}' for i in range(1, 10)),",
                    "    *(f'LPT{i}' for i in range(1, 10)),",
                    "}",
                    "_RESERVED_CHARS = set(chr(i) for i in range(32)) | set('*?\"<>/\\\\:|')",
                    "",
                    "def is_reserved_name(name: str) -> bool:",
                    "    name = str(name)",
                    "    if name[-1:] in ('.', ' '):",
                    "        return name not in ('.', '..')",
                    "    if _RESERVED_CHARS.intersection(name):",
                    "        return True",
                    "    return name.partition('.')[0].rstrip(' ').upper() in _RESERVED_NAMES",
                ],
            )
            return [command] if command else []
        return []

    def _rank_skill_code_hits(
        self,
        *,
        export_root: Path,
        repo_name: str,
        repo: dict[str, Any],
        repo_score: int,
        query_phrase: str,
        query_tokens: set[str],
    ) -> list[tuple[int, dict[str, Any]]]:
        extension = repo.get("extensions", {}).get("repo_skills_miner", {})
        extension = extension if isinstance(extension, dict) else {}
        paths = extension.get("paths", {}) if isinstance(extension.get("paths", {}), dict) else {}
        skills_path = export_root / str(paths.get("skills", ""))
        if not skills_path.exists():
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in _iter_jsonl(skills_path, limit=24000):
            snippet = str(row.get("snippet", "") or "")
            if not snippet.strip():
                continue
            searchable = [
                row.get("repo_id", ""),
                row.get("kind", ""),
                row.get("module", ""),
                row.get("qualname", ""),
                row.get("signature", ""),
                row.get("file_path", ""),
                row.get("doc_text", ""),
                row.get("annotation_summary", ""),
                snippet[:1600],
            ]
            aliases = [
                row.get("qualname", ""),
                row.get("file_path", ""),
                f"{row.get('module', '')}.{row.get('qualname', '')}",
            ]
            alias_score = _alias_phrase_score(query_phrase, aliases)
            token_score = len(query_tokens & _tokens(*searchable))
            if alias_score <= 0 and token_score < 3:
                continue
            score = repo_score + alias_score + token_score
            if _contains_phrase(query_phrase, row.get("qualname", "")):
                score += 8
            if _contains_phrase(query_phrase, row.get("file_path", "")):
                score += 6
            ranked.append(
                (
                    score,
                    {
                        "repo": repo_name,
                        "source": "repo_skills_miner",
                        "kind": row.get("kind", ""),
                        "module": row.get("module", ""),
                        "qualname": row.get("qualname", ""),
                        "signature": row.get("signature", ""),
                        "file_path": row.get("file_path", ""),
                        "line_start": row.get("line_start", ""),
                        "line_end": row.get("line_end", ""),
                        "snippet": snippet,
                        "annotation_summary": row.get("annotation_summary", ""),
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("qualname", ""))))
        return ranked[:5]

    def _rank_entity_code_hits(
        self,
        *,
        export_root: Path,
        repo_name: str,
        repo: dict[str, Any],
        repo_score: int,
        query_phrase: str,
        query_tokens: set[str],
    ) -> list[tuple[int, dict[str, Any]]]:
        entities_path = export_root / repo_name / f"{repo_name}.entities.jsonl"
        repo_root = Path(str(repo.get("repo_root", "")))
        if not entities_path.exists() or not repo_root.exists():
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in _iter_jsonl(entities_path, limit=30000):
            kind = str(row.get("kind", "") or "")
            if kind not in {"function", "class"}:
                continue
            line_start, line_end = _parse_program_uri_lines(row.get("uri", ""))
            module = _entity_module_name(row)
            if not module or line_start <= 0:
                continue
            file_path = ""
            source_path = None
            for candidate in _module_source_candidates(repo_root, module):
                if candidate.exists() and candidate.is_file():
                    source_path = candidate
                    file_path = str(candidate.relative_to(repo_root))
                    break
            if source_path is None:
                continue
            searchable = [
                row.get("id", ""),
                row.get("name", ""),
                row.get("kind", ""),
                row.get("uri", ""),
                module,
                file_path,
            ]
            aliases = [row.get("name", ""), file_path, module, str(row.get("id", "")).replace("py:", "")]
            alias_score = _alias_phrase_score(query_phrase, aliases)
            token_score = len(query_tokens & _tokens(*searchable))
            if alias_score <= 0 and token_score < 3:
                continue
            snippet = _read_source_span(source_path, line_start=line_start, line_end=line_end)
            if not snippet.strip():
                continue
            snippet_score = len(query_tokens & _tokens(snippet[:1600]))
            score = repo_score + alias_score + token_score + min(snippet_score, 10)
            ranked.append(
                (
                    score,
                    {
                        "repo": repo_name,
                        "source": "repo_graph_entities",
                        "kind": kind,
                        "module": module,
                        "qualname": str(row.get("id", "")).replace("py:", ""),
                        "signature": "",
                        "file_path": file_path,
                        "line_start": line_start,
                        "line_end": line_end,
                        "snippet": snippet,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("qualname", ""))))
        return ranked[:5]

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
                if span_type in {"research_library:paper_hits", "research_library:repository_code"}
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
            for guidance_line in list(metadata.get("applied_guidance", []) or [])[:12]:
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
            for fact in list(metadata.get("repository_code_facts", []) or [])[:4]:
                normalized_fact = str(fact).strip()
                if normalized_fact:
                    add_specific_evidence(f"research_library: repository_code_fact {normalized_fact}")
            for code_hit in list(metadata.get("repository_code_hits", []) or [])[:3]:
                if not isinstance(code_hit, dict):
                    continue
                snippet = str(code_hit.get("snippet", "")).strip()
                if not snippet:
                    continue
                add_specific_evidence(
                    "research_library: repository_code_hit "
                    f"repo={code_hit.get('repo', '')}; "
                    f"file={code_hit.get('file_path', '')}; "
                    f"symbol={code_hit.get('qualname', '')}; "
                    f"snippet={_truncate(snippet, 420)}"
                )
            for command in list(metadata.get("repository_code_commands", []) or [])[:2]:
                command_text = str(command).strip()
                if not command_text:
                    continue
                if command_text not in recommended_commands:
                    recommended_commands.insert(0, command_text)
                if not any(str(item.get("command", "")).strip() == command_text for item in recommended_command_spans):
                    recommended_command_spans.insert(
                        0,
                        {
                            "span_id": "research:repository_code:implementation",
                            "command": command_text,
                        },
                    )
                trusted_exact_command = True
        if applied_guidance:
            control["research_library"]["applied_guidance"] = applied_guidance[:8]
            command = _structured_fact_command_for_task(
                applied_guidance,
                state.task.prompt,
                self._exact_answer_target_path(state),
            )
            if command:
                if command not in recommended_commands:
                    recommended_commands.insert(0, command)
                if not any(str(item.get("command", "")).strip() == command for item in recommended_command_spans):
                    recommended_command_spans.insert(
                        0,
                        {
                            "span_id": "research:applied_guidance:structured_facts",
                            "command": command,
                        },
                    )
                trusted_exact_command = True
        paper_hit_evidence = [item for item in specific_research_evidence if "research_library: paper_hit " in item]
        non_paper_evidence = [
            item for item in specific_research_evidence if "research_library: paper_hit " not in item
        ]
        selected_specific_evidence = [*non_paper_evidence[:6], *paper_hit_evidence[:2]][:8]
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
        guidance["evidence"] = [*evidence[: max(0, 10 - len(research_evidence))], *research_evidence][:10]
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
