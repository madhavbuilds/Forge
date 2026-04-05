"""tiktoken-based context trimming."""

from __future__ import annotations

import tiktoken


def get_encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    enc = get_encoding()
    return len(enc.encode(text))


def trim_to_budget(text: str, max_tokens: int) -> str:
    enc = get_encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens]) + "\n\n… [truncated]"
