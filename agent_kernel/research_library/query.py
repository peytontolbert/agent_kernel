from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .models import DEFAULT_RESEARCH_LIBRARY_STATUS, load_research_library_status


_LOW_SIGNAL_TOKENS = {
    "about",
    "after",
    "and",
    "are",
    "best",
    "build",
    "can",
    "code",
    "does",
    "find",
    "for",
    "from",
    "give",
    "how",
    "into",
    "library",
    "model",
    "need",
    "paper",
    "papers",
    "repo",
    "repositories",
    "repository",
    "retrieval",
    "rag",
    "show",
    "the",
    "this",
    "use",
    "using",
    "what",
    "with",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tokens(*values: object) -> set[str]:
    text = " ".join(str(value).lower().replace("_", " ") for value in values if value is not None)
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+\-.]{2,}", text)
        if token not in _LOW_SIGNAL_TOKENS
    }


def _normalized_phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ")))


def _contains_phrase(haystack: object, needle: object) -> bool:
    normalized_haystack = _normalized_phrase(haystack)
    normalized_needle = _normalized_phrase(needle)
    if len(normalized_haystack) < 3 or len(normalized_needle) < 3:
        return False
    return normalized_needle in normalized_haystack or normalized_haystack in normalized_needle


def _compact(value: object, *, limit: int = 800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _best_excerpt(text: object, query_tokens: set[str], *, limit: int = 700) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    lowered = normalized.lower()
    starts = [lowered.find(token) for token in query_tokens if len(token) >= 5 and lowered.find(token) >= 0]
    start = min(starts) if starts else 0
    start = max(0, start - 100)
    excerpt = normalized[start : start + limit]
    if start > 0:
        excerpt = "..." + excerpt.lstrip()
    if start + limit < len(normalized):
        excerpt = excerpt.rstrip() + "..."
    return excerpt


def _source(status: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in status.get("sources", []):
        if isinstance(source, dict) and source.get("id") == source_id:
            return source
    return {}


def _read_jsonl(path: Path, *, limit: int) -> Iterable[dict[str, Any]]:
    if limit <= 0:
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    return
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


def _parquet_rows(path: Path, *, columns: list[str], max_rows: int, max_files: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0:
        return
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    emitted = 0
    for file_path in paths:
        try:
            parquet_file = pq.ParquetFile(file_path)
            available = set(parquet_file.schema_arrow.names)
            read_columns = [column for column in columns if column in available]
            if not read_columns:
                continue
            for batch in parquet_file.iter_batches(batch_size=1024, columns=read_columns):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        payload = dict(row)
                        payload["_source_file"] = file_path.name
                        yield payload
                        emitted += 1
                        if emitted >= max_rows:
                            return
        except Exception:
            continue


@dataclass(frozen=True)
class ResearchLibraryQueryLimits:
    limit: int = 8
    paper_file_limit: int = 2
    paper_row_limit: int = 8192
    repository_limit: int = 6
    code_rows_per_repo: int = 24000
    excerpt_chars: int = 700


class ResearchLibraryQueryClient:
    """Bounded read-only retrieval API over the local research library."""

    def __init__(
        self,
        *,
        status: dict[str, Any] | None = None,
        status_path: Path | str = DEFAULT_RESEARCH_LIBRARY_STATUS,
    ) -> None:
        self.status = status or load_research_library_status(status_path)

    def query(
        self,
        query: str,
        *,
        kinds: Iterable[str] = ("papers", "repositories", "code"),
        limits: ResearchLibraryQueryLimits | None = None,
    ) -> dict[str, Any]:
        active_limits = limits or ResearchLibraryQueryLimits()
        requested = {kind.strip().lower() for kind in kinds if kind.strip()}
        results: dict[str, list[dict[str, Any]]] = {}
        if "papers" in requested or "paper" in requested:
            results["papers"] = self.search_papers(query, limits=active_limits)
        if "repositories" in requested or "repository" in requested or "repos" in requested:
            results["repositories"] = self.search_repositories(query, limits=active_limits)
        if "code" in requested or "repository_code" in requested:
            repositories = results.get("repositories") or self.search_repositories(query, limits=active_limits)
            results["code"] = self.search_repository_code(query, repositories=repositories, limits=active_limits)
        return {
            "query": query,
            "limits": {
                "limit": active_limits.limit,
                "paper_file_limit": active_limits.paper_file_limit,
                "paper_row_limit": active_limits.paper_row_limit,
                "repository_limit": active_limits.repository_limit,
                "code_rows_per_repo": active_limits.code_rows_per_repo,
            },
            "results": results,
        }

    def search_papers(self, query: str, *, limits: ResearchLibraryQueryLimits | None = None) -> list[dict[str, Any]]:
        active_limits = limits or ResearchLibraryQueryLimits()
        query_tokens = _tokens(query)
        if not query_tokens and not query.strip():
            return []
        rows: list[tuple[int, dict[str, Any]]] = []
        rows.extend(self._rank_paper_chunks(query, query_tokens=query_tokens, limits=active_limits))
        if len(rows) < active_limits.limit:
            rows.extend(self._rank_paper_abstracts(query, query_tokens=query_tokens, limits=active_limits))
        rows.sort(key=lambda item: (-item[0], str(item[1].get("paper_id", "")), str(item[1].get("title", ""))))
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for score, row in rows:
            key = (
                str(row.get("paper_id", "")),
                str(row.get("chunk_index", "")),
                str(row.get("source_file", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append({"kind": "paper", "score": score, **row})
            if len(selected) >= active_limits.limit:
                break
        return selected

    def _rank_paper_chunks(
        self,
        query: str,
        *,
        query_tokens: set[str],
        limits: ResearchLibraryQueryLimits,
    ) -> list[tuple[int, dict[str, Any]]]:
        source = _source(self.status, "paper_chunks_p1")
        path = Path(str(source.get("path", "")))
        if not path.exists():
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        columns = ["id", "paper_id", "title", "categories", "year", "chunk_index", "text"]
        for row in _parquet_rows(
            path,
            columns=columns,
            max_rows=limits.paper_row_limit,
            max_files=limits.paper_file_limit,
        ):
            text = str(row.get("text", "") or "")
            title = str(row.get("title", "") or "")
            paper_id = str(row.get("paper_id", "") or "")
            token_score = len(query_tokens & _tokens(title, row.get("categories", ""), paper_id, text[:2400]))
            title_score = 18 if title and _contains_phrase(title, query) else 0
            id_score = 16 if paper_id and _contains_phrase(query, paper_id) else 0
            score = token_score + title_score + id_score
            if score < 3 and not title_score and not id_score:
                continue
            ranked.append(
                (
                    score,
                    {
                        "source_id": "paper_chunks_p1",
                        "paper_id": paper_id,
                        "title": title,
                        "categories": row.get("categories", ""),
                        "year": row.get("year", ""),
                        "chunk_index": row.get("chunk_index", ""),
                        "source_file": row.get("_source_file", ""),
                        "excerpt": _best_excerpt(text, query_tokens, limit=limits.excerpt_chars),
                        "provenance": {
                            "path": str(path),
                            "source_file": row.get("_source_file", ""),
                            "row_id": row.get("id", ""),
                        },
                    },
                )
            )
        return ranked

    def _rank_paper_abstracts(
        self,
        query: str,
        *,
        query_tokens: set[str],
        limits: ResearchLibraryQueryLimits,
    ) -> list[tuple[int, dict[str, Any]]]:
        source = _source(self.status, "paper_text_1m")
        path = Path(str(source.get("path", "")))
        if not path.exists():
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        columns = [
            "paper_id",
            "canonical_paper_id",
            "arxiv_id",
            "id",
            "title",
            "abstract",
            "text",
            "categories",
            "primary_category",
            "year",
            "published_year",
        ]
        for row in _parquet_rows(
            path,
            columns=columns,
            max_rows=limits.paper_row_limit,
            max_files=limits.paper_file_limit,
        ):
            title = str(row.get("title", "") or "")
            body = str(row.get("abstract", row.get("text", "")) or "")
            paper_id = str(
                row.get("paper_id", row.get("canonical_paper_id", row.get("arxiv_id", row.get("id", "")))) or ""
            )
            categories = row.get("categories", row.get("primary_category", ""))
            year = row.get("year", row.get("published_year", ""))
            token_score = len(query_tokens & _tokens(title, categories, paper_id, body[:2400]))
            title_score = 18 if title and _contains_phrase(title, query) else 0
            id_score = 16 if paper_id and _contains_phrase(query, paper_id) else 0
            score = token_score + title_score + id_score
            if score < 3 and not title_score and not id_score:
                continue
            ranked.append(
                (
                    score,
                    {
                        "source_id": "paper_text_1m",
                        "paper_id": paper_id,
                        "title": title,
                        "categories": categories,
                        "year": year,
                        "source_file": row.get("_source_file", ""),
                        "excerpt": _best_excerpt(body, query_tokens, limit=limits.excerpt_chars),
                        "provenance": {
                            "path": str(path),
                            "source_file": row.get("_source_file", ""),
                        },
                    },
                )
            )
        return ranked

    def search_repositories(
        self,
        query: str,
        *,
        limits: ResearchLibraryQueryLimits | None = None,
    ) -> list[dict[str, Any]]:
        active_limits = limits or ResearchLibraryQueryLimits()
        source = _source(self.status, "repository_exports")
        export_root = Path(str(source.get("path", "")))
        manifest_path = export_root / "_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        repos = manifest.get("repos", manifest.get("repositories", {}))
        if not isinstance(repos, dict):
            return []
        query_tokens = _tokens(query)
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for name, repo in repos.items():
            if not isinstance(repo, dict):
                continue
            languages = repo.get("languages", repo.get("language", []))
            language_text = " ".join(languages) if isinstance(languages, list) else str(languages)
            token_score = len(query_tokens & _tokens(name, language_text, repo.get("repo_root", "")))
            name_score = 12 if _contains_phrase(name, query) else 0
            if token_score < 2 and name_score <= 0:
                continue
            indices = repo.get("indices", {}) if isinstance(repo.get("indices", {}), dict) else {}
            skills = repo.get("extensions", {}).get("repo_skills_miner", {}).get("counts", {}).get("skills", 0)
            score = token_score + name_score + (1 if indices.get("qa") else 0) + min(3, int(skills or 0) // 10000)
            ranked.append((score, str(name), repo))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for score, name, repo in ranked[: active_limits.repository_limit]:
            indices = repo.get("indices", {}) if isinstance(repo.get("indices", {}), dict) else {}
            skills = repo.get("extensions", {}).get("repo_skills_miner", {}).get("counts", {}).get("skills", 0)
            results.append(
                {
                    "kind": "repository",
                    "source_id": "repository_exports",
                    "score": score,
                    "name": name,
                    "repo_root": repo.get("repo_root", ""),
                    "languages": repo.get("languages", repo.get("language", [])),
                    "qa_index": bool(indices.get("qa")),
                    "mined_skills": skills,
                    "provenance": {"path": str(export_root / name), "manifest": str(manifest_path)},
                }
            )
        return results

    def search_repository_code(
        self,
        query: str,
        *,
        repositories: list[dict[str, Any]] | None = None,
        limits: ResearchLibraryQueryLimits | None = None,
    ) -> list[dict[str, Any]]:
        active_limits = limits or ResearchLibraryQueryLimits()
        source = _source(self.status, "repository_exports")
        export_root = Path(str(source.get("path", "")))
        manifest_path = export_root / "_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        manifest_repos = manifest.get("repos", manifest.get("repositories", {}))
        if not isinstance(manifest_repos, dict):
            return []
        selected_repositories = repositories or self.search_repositories(query, limits=active_limits)
        query_tokens = _tokens(query)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for repo_result in selected_repositories[: active_limits.repository_limit]:
            repo_name = str(repo_result.get("name", "")).strip()
            repo = manifest_repos.get(repo_name, {})
            if not isinstance(repo, dict):
                continue
            extension = repo.get("extensions", {}).get("repo_skills_miner", {})
            extension = extension if isinstance(extension, dict) else {}
            paths = extension.get("paths", {}) if isinstance(extension.get("paths", {}), dict) else {}
            skills_path = export_root / str(paths.get("skills", ""))
            if not skills_path.exists():
                continue
            repo_score = int(repo_result.get("score", 0) or 0)
            for row in _read_jsonl(skills_path, limit=active_limits.code_rows_per_repo):
                snippet = str(row.get("snippet", "") or "")
                if not snippet.strip():
                    continue
                searchable = [
                    repo_name,
                    row.get("kind", ""),
                    row.get("module", ""),
                    row.get("qualname", ""),
                    row.get("signature", ""),
                    row.get("file_path", ""),
                    row.get("doc_text", ""),
                    row.get("annotation_summary", ""),
                    snippet[:1800],
                ]
                token_score = len(query_tokens & _tokens(*searchable))
                alias_score = 10 if any(_contains_phrase(value, query) for value in searchable[:6]) else 0
                if token_score < 3 and alias_score <= 0:
                    continue
                score = repo_score + token_score + alias_score
                ranked.append(
                    (
                        score,
                        {
                            "kind": "code",
                            "source_id": "repository_exports",
                            "score": score,
                            "repo": repo_name,
                            "symbol_kind": row.get("kind", ""),
                            "module": row.get("module", ""),
                            "qualname": row.get("qualname", ""),
                            "signature": row.get("signature", ""),
                            "file_path": row.get("file_path", ""),
                            "line_start": row.get("line_start", ""),
                            "line_end": row.get("line_end", ""),
                            "summary": _compact(row.get("annotation_summary", ""), limit=300),
                            "excerpt": _best_excerpt(snippet, query_tokens, limit=active_limits.excerpt_chars),
                            "provenance": {
                                "skills_path": str(skills_path),
                                "repo_export": str(export_root / repo_name),
                            },
                        },
                    )
                )
        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("repo", "")),
                str(item[1].get("file_path", "")),
                str(item[1].get("qualname", "")),
            )
        )
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for _, hit in ranked:
            key = (str(hit.get("repo", "")), str(hit.get("file_path", "")), str(hit.get("qualname", "")))
            if key in seen:
                continue
            seen.add(key)
            results.append(hit)
            if len(results) >= active_limits.limit:
                break
        return results


__all__ = ["ResearchLibraryQueryClient", "ResearchLibraryQueryLimits"]
