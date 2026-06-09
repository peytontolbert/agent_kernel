#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.research_library import (  # noqa: E402
    DEFAULT_RESEARCH_LIBRARY_STATUS,
    ResearchLibraryQueryClient,
    ResearchLibraryQueryLimits,
)


def _kinds(values: list[str]) -> tuple[str, ...]:
    if not values:
        return ("papers", "repositories", "code")
    kinds: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                kinds.append(part)
    return tuple(kinds)


def _print_text(payload: dict[str, object]) -> None:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return
    for section in ("papers", "repositories", "code"):
        records = results.get(section, [])
        if not isinstance(records, list) or not records:
            continue
        print(f"\n## {section}")
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            score = record.get("score", "")
            if section == "papers":
                title = record.get("title", "")
                paper_id = record.get("paper_id", "")
                excerpt = record.get("excerpt", "")
                print(f"{index}. score={score} paper_id={paper_id} title={title}\n   {excerpt}")
            elif section == "repositories":
                name = record.get("name", "")
                root = record.get("repo_root", "")
                print(f"{index}. score={score} repo={name} root={root}")
            else:
                repo = record.get("repo", "")
                file_path = record.get("file_path", "")
                qualname = record.get("qualname", "")
                excerpt = record.get("excerpt", "")
                print(f"{index}. score={score} repo={repo} file={file_path} symbol={qualname}\n   {excerpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local paper and repository research-library sources.")
    parser.add_argument("query", nargs="+", help="Search query text.")
    parser.add_argument("--status-path", type=Path, default=DEFAULT_RESEARCH_LIBRARY_STATUS)
    parser.add_argument("--kind", action="append", default=[], help="papers, repositories, code, or comma-separated.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--paper-file-limit", type=int, default=2)
    parser.add_argument("--paper-row-limit", type=int, default=8192)
    parser.add_argument("--repository-limit", type=int, default=6)
    parser.add_argument("--code-rows-per-repo", type=int, default=24000)
    parser.add_argument("--excerpt-chars", type=int, default=700)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    limits = ResearchLibraryQueryLimits(
        limit=args.limit,
        paper_file_limit=args.paper_file_limit,
        paper_row_limit=args.paper_row_limit,
        repository_limit=args.repository_limit,
        code_rows_per_repo=args.code_rows_per_repo,
        excerpt_chars=args.excerpt_chars,
    )
    client = ResearchLibraryQueryClient(status_path=args.status_path)
    payload = client.query(" ".join(args.query), kinds=_kinds(args.kind), limits=limits)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "text":
        _print_text(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
