"""Answer generation: prompt assembly + LLM call, with guardrails + fallback.

Guardrails baked into the system prompt:
* answer **only** from the supplied context,
* say "I don't know" when the context is insufficient (hallucination control),
* cite sources by their bracketed index.

When no OpenAI key is configured, :class:`ExtractiveGenerator` returns the most
relevant retrieved snippet verbatim so the endpoint still yields a grounded,
citable answer offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise documentation assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. If the answer is not "
    "contained in the context, reply exactly: \"I don't know based on the "
    "available documents.\" Cite the passages you used with bracketed numbers "
    "like [1], [2]. Be concise and do not invent facts."
)


@dataclass
class Generation:
    answer: str
    model: str
    tokens: dict[str, int]


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, ch in enumerate(chunks, start=1):
        src = ch.record.source
        parts.append(f"[{i}] (source: {src})\n{ch.record.text}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context_block(chunks)
    user = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite passage numbers."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class ExtractiveGenerator:
    """Offline fallback: returns the top passage(s) as a grounded answer."""

    model = "local-extractive"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Generation:
        if not chunks:
            return Generation(
                answer="I don't know based on the available documents.",
                model=self.model,
                tokens={},
            )
        top = chunks[0]
        snippet = top.record.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + "…"
        answer = f"{snippet} [1]"
        return Generation(answer=answer, model=self.model, tokens={})


class OpenAIGenerator:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self.model = settings.generation_model
        self._temperature = settings.generation_temperature

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Generation:
        if not chunks:
            return Generation(
                answer="I don't know based on the available documents.",
                model=self.model,
                tokens={},
            )
        messages = build_messages(question, chunks)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
        )
        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens = (
            {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }
            if usage
            else {}
        )
        return Generation(answer=answer.strip(), model=self.model, tokens=tokens)


def build_generator(settings: Settings):
    if settings.use_openai:
        return OpenAIGenerator(settings)
    return ExtractiveGenerator()
