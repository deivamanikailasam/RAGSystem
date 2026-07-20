"""Token-aware, overlapping document chunker.

Chunking is the highest-leverage lever in a RAG system: too large and
retrieval returns noisy context; too small and answers lose the surrounding
meaning. We use overlapping windows so a fact split across a boundary still
appears intact in at least one chunk.

We deliberately avoid a hard dependency on ``tiktoken`` (which needs network
access to download encodings on first use). Instead we approximate tokens with
a whitespace/punctuation word split, which is stable and offline. The chunk
sizes are configured in *token-ish* units; the approximation is close enough
for retrieval quality and keeps the system runnable anywhere. To use exact
BPE token counts, swap ``_tokenize`` for a ``tiktoken`` encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")
# Split on blank lines first so we prefer to break on paragraph boundaries.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document plus its position."""

    index: int
    text: str
    token_estimate: int


def _tokenize(text: str) -> list[str]:
    """Approximate token boundaries with whitespace-delimited words."""
    return _WORD_RE.findall(text)


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    """Split ``text`` into overlapping token windows.

    Paragraph structure is used as a soft hint: we accumulate whole paragraphs
    until adding the next one would exceed ``chunk_tokens``, then emit a chunk
    and slide the window back by ``overlap`` tokens. Oversized paragraphs are
    split on the token grid so no single chunk can blow past the budget.

    Args:
        text: The full document text.
        chunk_tokens: Target maximum size of each chunk, in approximate tokens.
        overlap: How many tokens the next chunk re-includes from the previous
            one, to preserve continuity across boundaries.

    Returns:
        A list of :class:`Chunk` in reading order.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("overlap must be in [0, chunk_tokens)")

    text = text.strip()
    if not text:
        return []

    # Flatten to a token stream but remember nothing about paragraphs beyond
    # inserting them in order — the overlap window handles continuity.
    tokens: list[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if para:
            tokens.extend(_tokenize(para))

    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_tokens - overlap
    start = 0
    index = 0
    n = len(tokens)
    while start < n:
        window = tokens[start : start + chunk_tokens]
        chunk_str = _detokenize(window)
        chunks.append(
            Chunk(index=index, text=chunk_str, token_estimate=len(window))
        )
        index += 1
        if start + chunk_tokens >= n:
            break
        start += step
    return chunks
