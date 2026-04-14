from __future__ import annotations

import base64
from pathlib import Path
import json
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from agent_kernel.extensions.capabilities import enabled_corpus_sources
from agent_kernel.config import KernelConfig


def materialize_external_corpus(
    *,
    config: KernelConfig,
    output_dir: Path,
    max_sources: int = 64,
    opener: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "external_corpus_records.jsonl"
    manifest_path = output_dir / "external_corpus_manifest.json"
    open_url = opener or url_request.urlopen
    sources = enabled_corpus_sources(config)[: max(0, max_sources) or 0]
    records: list[dict[str, Any]] = []
    timeout_seconds = max(1, int(config.unattended_http_timeout_seconds))
    max_body_bytes = max(1, int(config.unattended_http_max_body_bytes))
    for source in sources:
        url = str(source.get("url", "")).strip()
        if not url:
            continue
        record = {
            "module_id": str(source.get("module_id", "")).strip(),
            "adapter_kind": str(source.get("adapter_kind", "")).strip(),
            "url": url,
            "host": str(source.get("host", "")).strip(),
            "status": "error",
            "body_text": "",
        }
        req = url_request.Request(
            url=url,
            headers={
                "Accept": "application/json, text/plain;q=0.9, */*;q=0.1",
                "User-Agent": "agentkernel-universal-corpus/1.0",
            },
            method="GET",
        )
        try:
            with open_url(req, timeout=timeout_seconds) as response:
                raw = response.read(max_body_bytes + 1)
                if len(raw) > max_body_bytes:
                    record["status"] = "truncated"
                    record["body_text"] = raw[:max_body_bytes].decode("utf-8", errors="replace")
                else:
                    record["status"] = "ok"
                    record["body_text"] = raw.decode("utf-8", errors="replace")
                record["content_type"] = str(response.headers.get("Content-Type", "")).strip()
                record["http_status"] = int(getattr(response, "status", 200))
        except url_error.HTTPError as exc:
            record["status"] = "http_error"
            record["http_status"] = int(exc.code)
            record["body_text"] = exc.read(max_body_bytes).decode("utf-8", errors="replace")
        except (url_error.URLError, OSError, TimeoutError) as exc:
            record["status"] = "fetch_error"
            record["error"] = str(exc)
        records.append(record)
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    quality = _corpus_quality_summary(records)
    manifest = {
        "artifact_kind": "tolbert_external_corpus_manifest",
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "source_count": len(sources),
        "fetched_count": sum(1 for record in records if str(record.get("status", "")) in {"ok", "truncated"}),
        "status_counts": _status_counts(records),
        "quality": quality,
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def external_corpus_examples(manifest: dict[str, Any], *, max_records: int = 64) -> list[dict[str, Any]]:
    records_path = Path(str(manifest.get("records_path", "")).strip())
    if not records_path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for index, line in enumerate(records_path.read_text(encoding="utf-8").splitlines()):
        if index >= max_records:
            break
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        body_text = _normalized_record_text(record)
        if len(body_text.split()) < 8:
            continue
        prompt, target = _split_prompt_target(body_text)
        if not prompt or not target:
            continue
        examples.append(
            {
                "source_type": "external_corpus",
                "source_id": f"{record.get('module_id', '')}:{record.get('url', '')}",
                "prompt": f"url: {record.get('url', '')}\nmodule: {record.get('module_id', '')}\n{prompt}",
                "target": target,
            }
        )
    return examples


def _split_prompt_target(text: str) -> tuple[str, str]:
    tokens = str(text).split()
    split_index = max(4, min(len(tokens) - 2, int(len(tokens) * 0.6)))
    return " ".join(tokens[:split_index]).strip(), " ".join(tokens[split_index:]).strip()


def _normalized_record_text(record: dict[str, Any]) -> str:
    body_text = str(record.get("body_text", "")).strip()
    content_type = str(record.get("content_type", "")).lower()
    if "json" in content_type or body_text[:1] in {"{", "["}:
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            return _normalize_whitespace(body_text)
        github_readme = _decode_github_readme(payload)
        if github_readme:
            return _normalize_whitespace(github_readme)
        return _normalize_whitespace(_json_to_text(payload))
    return _normalize_whitespace(body_text)


def _decode_github_readme(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    if str(payload.get("encoding", "")).strip().lower() != "base64":
        return ""
    raw_content = str(payload.get("content", "")).strip()
    if not raw_content:
        return ""
    try:
        decoded = base64.b64decode(raw_content.encode("utf-8"), validate=False)
    except (ValueError, OSError):
        return ""
    return decoded.decode("utf-8", errors="replace")


def _json_to_text(payload: Any) -> str:
    parts: list[str] = []
    if isinstance(payload, dict):
        preferred = [
            "full_name",
            "name",
            "description",
            "bio",
            "company",
            "blog",
            "language",
            "topics",
            "content",
            "message",
        ]
        for key in preferred:
            if key in payload:
                parts.append(f"{key}: {_json_to_text(payload[key])}")
        for key, value in payload.items():
            if key in preferred:
                continue
            if len(parts) >= 32:
                break
            text = _json_to_text(value)
            if text:
                parts.append(f"{key}: {text}")
    elif isinstance(payload, list):
        for value in payload[:16]:
            text = _json_to_text(value)
            if text:
                parts.append(text)
    elif isinstance(payload, (str, int, float, bool)):
        return str(payload)
    return "\n".join(part for part in parts if part).strip()


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown")).strip() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _corpus_quality_summary(records: list[dict[str, Any]]) -> dict[str, object]:
    usable_records = 0
    total_chars = 0
    total_tokens = 0
    source_kind_counts: dict[str, int] = {}
    content_type_counts: dict[str, int] = {}
    for record in records:
        normalized = _normalized_record_text(record)
        tokens = normalized.split()
        if len(tokens) >= 8:
            usable_records += 1
            total_chars += len(normalized)
            total_tokens += len(tokens)
        adapter_kind = str(record.get("adapter_kind", "unknown")).strip() or "unknown"
        source_kind_counts[adapter_kind] = source_kind_counts.get(adapter_kind, 0) + 1
        content_type = str(record.get("content_type", "")).strip() or "unknown"
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
    return {
        "usable_record_count": usable_records,
        "normalized_char_count": total_chars,
        "normalized_token_count": total_tokens,
        "average_tokens_per_usable_record": (float(total_tokens) / usable_records) if usable_records else 0.0,
        "adapter_kind_counts": dict(sorted(source_kind_counts.items())),
        "content_type_counts": dict(sorted(content_type_counts.items())),
    }
