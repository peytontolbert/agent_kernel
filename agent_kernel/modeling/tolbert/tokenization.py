from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .config import HybridTolbertSSMConfig


_DECODER_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_./:-]*"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|->|=>|::"
    r"|[{}()\[\],.:;=+\-*/<>\"'`|]"
)


def hashed_id(value: str, vocab_size: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return 1 + (int.from_bytes(digest[:8], "big") % max(2, vocab_size - 1))


def encode_command_tokens(command: str, config: HybridTolbertSSMConfig) -> list[int]:
    pieces = [piece for piece in str(command).strip().split() if piece][: config.max_command_tokens]
    ids = [hashed_id(piece, config.token_vocab_size) for piece in pieces]
    while len(ids) < config.max_command_tokens:
        ids.append(0)
    return ids


def build_decoder_vocabulary(
    texts: Iterable[str],
    config: HybridTolbertSSMConfig,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in _decoder_tokens(text):
            counts[token] = counts.get(token, 0) + 1
    usable = max(0, config.decoder_vocab_size - _decoder_special_token_count(config))
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    vocab: dict[str, int] = {}
    next_id = _decoder_special_token_count(config)
    for token, _ in ordered[:usable]:
        vocab[token] = next_id
        next_id += 1
    return vocab


def decoder_tokenizer_stats(texts: Iterable[str]) -> dict[str, int]:
    text_count = 0
    token_count = 0
    unique: set[str] = set()
    for text in texts:
        text_count += 1
        tokens = _decoder_tokens(text)
        token_count += len(tokens)
        unique.update(tokens)
    return {
        "tokenizer_kind": "regex_v1",
        "text_count": text_count,
        "token_count": token_count,
        "unique_token_count": len(unique),
    }


def encode_decoder_sequence(
    text: str,
    config: HybridTolbertSSMConfig,
    decoder_vocab: dict[str, int] | None = None,
) -> tuple[list[int], list[int]]:
    decoder_vocab = decoder_vocab or {}
    pieces = _decoder_tokens(text)
    usable = max(0, config.max_command_tokens - 1)
    token_ids = [_decoder_lookup(token, config, decoder_vocab) for token in pieces[:usable]]
    input_ids = [config.decoder_bos_token_id] + token_ids
    target_ids = token_ids + [config.decoder_eos_token_id]
    input_ids = input_ids[: config.max_command_tokens]
    target_ids = target_ids[: config.max_command_tokens]
    while len(input_ids) < config.max_command_tokens:
        input_ids.append(config.decoder_pad_token_id)
    while len(target_ids) < config.max_command_tokens:
        target_ids.append(config.decoder_pad_token_id)
    return input_ids, target_ids


def decode_decoder_ids(
    token_ids: Iterable[int],
    config: HybridTolbertSSMConfig,
    decoder_vocab: dict[str, int] | None = None,
) -> str:
    decoder_vocab = decoder_vocab or {}
    reverse = {int(value): token for token, value in decoder_vocab.items()}
    tokens: list[str] = []
    for raw in token_ids:
        token_id = int(raw)
        if token_id in {config.decoder_pad_token_id, config.decoder_eos_token_id}:
            break
        if token_id == config.decoder_bos_token_id:
            continue
        if token_id == config.decoder_unk_token_id:
            tokens.append("<unk>")
            continue
        tokens.append(reverse.get(token_id, "<unk>"))
    return " ".join(token for token in tokens if token)


def _decoder_lookup(
    token: str,
    config: HybridTolbertSSMConfig,
    decoder_vocab: dict[str, int],
) -> int:
    value = decoder_vocab.get(token)
    if value is None:
        return config.decoder_unk_token_id
    return int(value)


def _decoder_special_token_count(config: HybridTolbertSSMConfig) -> int:
    return max(
        int(config.decoder_pad_token_id),
        int(config.decoder_bos_token_id),
        int(config.decoder_eos_token_id),
        int(config.decoder_unk_token_id),
    ) + 1


def _decoder_tokens(text: str) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return []
    tokens = [token for token in _DECODER_TOKEN_PATTERN.findall(raw) if token]
    return tokens if tokens else [token for token in raw.split() if token]
