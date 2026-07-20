# 12. Multi-Turn Chatbot with Context Tracking

> **Goal of this doc:** turn the single-shot `/v1/query` RAG into a **stateful
> chatbot** that remembers the conversation, resolves follow-ups against earlier
> turns, and stays grounded in retrieved context — step by step.

---

## 12.1 The problem with single-shot RAG

`/v1/query` treats every question independently. But real conversations are
full of references:

```
User:  What is FAISS?
Bot:   FAISS is a similarity-search library… [1]
User:  how does it reduce latency?      ← "it" = FAISS; meaningless alone
```

If you send *"how does it reduce latency?"* straight to the retriever, it has no
idea what "it" is and retrieves poorly. A multi-turn chatbot needs two things:

1. **Memory** — persist the conversation so each turn can see the earlier ones.
2. **Context tracking** — rewrite the follow-up into a self-contained query
   *before* retrieval, and give the generator the recent turns so it can resolve
   references and keep tone.

## 12.2 The turn pipeline

Each call to `/v1/chat` runs (in `app/core/rag.py:RagEngine.chat`):

```
message + session_id
      │
      ▼
load recent history ──▶ CONDENSE follow-up → standalone question   [context tracking]
      │                          │
      │                          ▼
      │                 retrieve (hybrid) → rerank            (stages 1–2, docs 8–9)
      │                          │
      ▼                          ▼
   history messages ─────▶ GENERATE answer (grounded + history-aware)
                                 │
                                 ▼
              persist user + assistant turns  → response (answer, citations,
                                                 session_id, standalone_question)
```

Two new components do the conversational work; everything downstream
(retrieval, reranking, generation, guardrails, per-tenant isolation, quotas) is
reused unchanged.

## 12.3 Conversation memory (`app/core/conversation.py`)

`ConversationStore` persists chat history in its own SQLite file
(`{DATA_DIR}/conversations.db`), keyed by **`(tenant, session_id)`** so
conversations are tenant-isolated exactly like documents and vectors:

- `conversations` — one row per session (timestamps).
- `messages` — one row per message: `turn_index` (monotonic), `role`
  (`user`/`assistant`), `content`, and the JSON citations that grounded an
  assistant reply.

A session id is minted (`uuid4`) when the client doesn't supply one, and
returned in every response so the client can continue the thread. Because rows
are tenant-scoped, tenant B asking for tenant A's `session_id` sees **nothing**.

## 12.4 Context tracking: question condensing (`app/core/condenser.py`)

Before retrieval we rewrite the follow-up into a standalone question. Two
strategies behind one interface (`condense(history, question) -> str`):

| Strategy | How | When |
|----------|-----|------|
| `llm` | asks the model to rewrite the follow-up as standalone given the history | `OPENAI_API_KEY` set |
| `heuristic` | if the message looks like a follow-up, prepend the previous user question so its topic terms are in scope | offline (default) |
| `none` | pass the message through unchanged | `CHAT_CONDENSE_QUESTION=false` |

The offline **heuristic** flags a follow-up when the message is very short,
begins with a conjunction (`and`, `why`, `what about`…), or contains an early
referent (`it`, `its`, `that`…), then prepends the prior question. Example:

```
history: "How does FAISS reduce latency?"
message: "what about nprobe?"
standalone → "How does FAISS reduce latency? what about nprobe?"
```

Crude but it reliably pulls the referent into retrieval range with zero
dependencies; the LLM condenser produces cleaner rewrites when a key is present.
Either way the rewritten query is returned as `standalone_question` for
transparency.

## 12.5 History-aware generation

The recent turns are passed to the generator as prior chat messages
(`build_messages(..., history=...)` in `app/core/generator.py`), so the model
resolves references and maintains continuity — while the **grounding guardrails
and retrieved context still gate the answer** (answer only from context, cite
sources, say "I don't know" otherwise). `CHAT_HISTORY_TURNS` bounds how many
prior messages are included, so long conversations stay within a token budget.

## 12.6 API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat` | Send a turn. Body: `{message, session_id?, top_k?, filters?}`. Returns the answer, citations, `session_id`, and `standalone_question`. |
| GET | `/v1/chat/{session_id}` | Full message history for a session (404 if unknown). |
| DELETE | `/v1/chat/{session_id}` | Delete a conversation. |

All require the tenant bearer key (or are open in single-tenant mode); a chat
turn counts against the tenant's `max_queries_per_day` quota.

## 12.7 Step-by-step: hold a conversation

```bash
# Turn 1 — no session_id → a new one is created and returned
SID=$(curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
      -H 'Authorization: Bearer demo-key' \
      -d '{"message":"What is FAISS?"}' | jq -r .session_id)

# Turn 2 — a follow-up; "it" is resolved via the conversation
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -H 'Authorization: Bearer demo-key' \
  -d "{\"message\":\"how does it reduce latency?\",\"session_id\":\"$SID\"}" | jq \
  '{standalone_question, answer, citations: [.citations[].doc_id]}'

# Inspect the stored history
curl -s localhost:8000/v1/chat/$SID -H 'Authorization: Bearer demo-key' | jq '.messages[] | {role, content}'

# End the conversation
curl -s -X DELETE localhost:8000/v1/chat/$SID -H 'Authorization: Bearer demo-key'
```

Observed (offline): turn 2's `standalone_question` becomes
*"What is FAISS? how does it reduce latency?"*, so the FAISS document is
retrieved and the answer stays on-topic — the follow-up worked.

## 12.8 Configuration

```bash
CHAT_HISTORY_TURNS=8            # prior messages fed to the generator
CHAT_CONDENSE_QUESTION=true     # rewrite follow-ups before retrieval
```

## 12.9 Design notes & extensions

- **Grounding is preserved.** History helps interpret the question, but the
  answer is still constrained to retrieved context — no hallucinated memory.
- **Bounded context.** Only `CHAT_HISTORY_TURNS` messages enter the prompt; for
  very long chats, add rolling summarization (summarize older turns into one
  system note) behind the same interface.
- **Per-turn retrieval.** Every turn re-retrieves against the condensed query,
  so answers track the *current* focus rather than stale context.
- **Isolation & quotas** come for free from the tenant layer (docs 7, 10).
- **Streaming.** For a chat UX, stream tokens from `OpenAIGenerator` over
  SSE/WebSocket; the turn pipeline is unchanged.

## 12.10 Reference map

| Concern | Code |
|---------|------|
| Chat orchestration | `app/core/rag.py:RagEngine.chat` |
| Conversation persistence | `app/core/conversation.py` |
| Follow-up condensing | `app/core/condenser.py` |
| History-aware prompt | `app/core/generator.py:build_messages` (`history`) |
| Config | `app/config.py` (`chat_history_turns`, `chat_condense_question`) |
| Endpoints | `app/api/routes.py` (`/v1/chat`) |
| Request/response schemas | `app/models.py` (`ChatRequest`, `ChatResponse`, …) |

Back to the [docs index](README.md).
