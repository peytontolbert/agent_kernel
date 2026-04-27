from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.runtime_modeling_adapter import build_context_provider
from agent_kernel.state import AgentState
from scripts.run_research_library_hard_probe import (
    DEFAULT_RESEARCH_LIBRARY_CONFIG,
    DEFAULT_STATUS,
    _expected_paper_answer,
    _hard_task,
    _load_status,
)


DEFAULT_OUTPUT = Path("var/research_library/llm_paper_output_probe.json")


def _post_chat(*, host: str, model: str, system: str, user: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 220,
    }
    request = Request(
        host.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return f"ERROR: {exc.__class__.__name__}: {exc}"
    choices = body.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip() + "\n"


def _paper_evidence_context(
    status: dict[str, Any],
    expected_answer: str,
    paper_row: dict[str, Any],
    *,
    status_path: Path,
    config_path: Path,
) -> str:
    task = _hard_task(expected_answer, probe="paper", paper_row=paper_row)
    config = KernelConfig(
        provider="mock",
        use_tolbert_context=False,
        use_research_library_context=True,
        research_library_standalone_context=True,
        research_library_status_path=status_path,
        research_library_config_path=config_path,
        research_library_context_max_chunks=7,
        research_library_context_max_paper_hits=1,
        research_library_paper_scan_file_limit=1,
        research_library_paper_scan_row_limit=512,
    )
    provider = build_context_provider(config=config, repo_root=Path.cwd())
    if provider is None:
        return ""
    packet = provider.compile(AgentState(task=task))
    evidence = packet.control.get("retrieval_guidance", {}).get("evidence", [])
    paper_evidence = [str(item) for item in evidence if "research_library: paper_hit" in str(item)]
    chunks = [
        str(chunk.get("text", ""))
        for chunk in packet.control.get("selected_context_chunks", [])
        if str(chunk.get("span_id", "")) == "research:paper_hits"
    ]
    return "\n".join([*paper_evidence, *chunks])


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe direct LLM output with and without compiled research context.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_RESEARCH_LIBRARY_CONFIG)
    parser.add_argument("--host", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    repo_root = Path.cwd()
    status_path = args.status if args.status.is_absolute() else repo_root / args.status
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    status = _load_status(repo_root=repo_root, status_path=status_path, config_path=config_path, refresh=False)
    expected_answer, paper_row = _expected_paper_answer(status)
    base_config = KernelConfig(provider="vllm")
    host = args.host or base_config.vllm_host
    model = args.model or base_config.model_name
    system = (
        "Return only the requested answer. Copy exact values from research context when it is provided. "
        "Do not summarize, abbreviate, add ellipsis, or add prose."
    )
    request = (
        "Produce exactly three newline-delimited fields for the paper titled "
        f"{paper_row.get('title', '')}: paper_id=<exact>, title=<exact>, "
        "evidence_sentence=<the exact sentence saying how the strategy is implemented>."
    )
    context = _paper_evidence_context(
        status,
        expected_answer,
        paper_row,
        status_path=status_path,
        config_path=config_path,
    )
    variants = {
        "without_research": _post_chat(
            host=host,
            model=model,
            system=system,
            user=request,
            timeout=args.timeout,
        ),
        "with_research": _post_chat(
            host=host,
            model=model,
            system=system,
            user=f"{request}\n\nResearch context:\n{context}",
            timeout=args.timeout,
        ),
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "host": host,
        "expected_answer": expected_answer,
        "context": context,
        "variants": {
            name: {
                "answer": answer,
                "success": answer == expected_answer,
            }
            for name, answer in variants.items()
        },
        "helps": variants["with_research"] == expected_answer and variants["without_research"] != expected_answer,
    }
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "research_llm_output_probe "
        f"without_success={str(report['variants']['without_research']['success']).lower()} "
        f"with_success={str(report['variants']['with_research']['success']).lower()} "
        f"helps={str(report['helps']).lower()} output={output_path}"
    )


if __name__ == "__main__":
    main()
