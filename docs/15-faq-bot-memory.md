# 15. Context-Aware FAQ Bot with Memory

> **Goal of this doc:** build an FAQ bot that answers from a curated FAQ base
> first (falling back to RAG over documents), stays context-aware across a
> conversation, and **remembers facts about the user across sessions** — step by
> step.

---

## 15.1 What makes it more than "RAG chat"

Three additions over the chat of [doc 12](12-multi-turn-chat.md):

1. **FAQ-first.** A curated question→answer base is matched *before* open RAG.
   Curated answers are exact, authoritative, and free (no LLM call) — the right
   behavior for known questions ("how do I reset my password?").
2. **Long-term memory.** Facts about the *user* (name, plan, tools, topics
   asked) persist **across sessions**, not just within one conversation, and are
   fed back as context.
3. **Two-tier context.** Short-term memory = the conversation history
   (condensing follow-ups); long-term memory = the per-user store. Both inform
   the answer.

## 15.2 The turn pipeline (`app/core/faqbot.py`)

```
message (+ user_id, session_id)
   │
   ▼
1. condense follow-up  ← conversation history (short-term memory, doc 12)
2. extract & store new memories from the message   → memory store
3. recall relevant memories for this user          ← memory store
   │
   ▼
4. FAQ match on the standalone question
   ├─ score ≥ threshold  →  return curated FAQ answer         (source = "faq")
   └─ otherwise          →  5. RAG fallback, memory-augmented (source = "rag")
   │                            retrieve + rerank + generate,
   │                            memories injected into the prompt
   ▼
6. persist the turn (conversation) + remember the topic
```

Everything downstream — hybrid retrieval, reranking, grounding guardrails,
per-tenant isolation, quotas — is reused unchanged.

## 15.3 The FAQ base + matcher (`app/core/faq.py`)

`FAQStore` holds curated Q&A pairs per tenant. `FAQMatcher` scores an incoming
question against them with **two combined signals**:

- **semantic** — cosine between the query embedding and each FAQ-question
  embedding (catches paraphrases: *"change my password"* ≈ *"reset my
  password"*);
- **lexical** — Jaccard overlap of content words (catches exact keywords and
  keeps the offline fallback embedder honest).

`score = 0.5·cosine + 0.5·lexical`. If the best score clears
`FAQ_MATCH_THRESHOLD` (default 0.45) the curated answer wins. The per-tenant FAQ
embedding index is cached and rebuilt when FAQs change.

> **Offline caveat:** the fallback embedder is lexical, so a *purely* semantic
> match with no shared words (*"what time do you open?"* vs *"business hours"*)
> may fall below threshold and go to RAG. With real OpenAI embeddings it
> matches. The mechanism is identical; only the embedding quality differs.

## 15.4 Long-term memory (`app/core/memory.py`)

`MemoryStore` persists facts keyed by **`(tenant, user_id)`** — so memory follows
the *user*, across sessions — with de-duplication (re-stating a fact just
refreshes recency). Three kinds: `fact`, `topic`, `preference`.

**Extraction** (`extract_memories`) is a dependency-free heuristic that pulls
explicit facts from a message:

| Message | Remembered |
|---------|------------|
| "my name is Ada" | `name is Ada` |
| "I'm on the enterprise plan" | `plan: enterprise` |
| "I use Python" | `uses Python` |
| "my email is ada@x.com" | `email: ada@x.com` |

Swap in an LLM extractor for richer, fuzzier memories — same interface.

**Recall** ranks memories by relevance to the current query (content-word
overlap) then recency, and the top `MEMORY_RECALL_LIMIT` (default 5) are:
- injected into the generation system prompt on the RAG path
  (`"Known about the user: - name is Ada …"`), and
- returned as `memories_used` for transparency.

> On the offline extractive answerer, injected memory doesn't change the answer
> text (it doesn't call an LLM), but it is still recalled, surfaced, and stored —
> with a real model the answer becomes user-aware.

## 15.5 API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/faqs` | Add a curated FAQ (`question`, `answer`, `tags`). |
| GET | `/v1/faqs` | List the tenant's FAQs. |
| DELETE | `/v1/faqs/{faq_id}` | Remove an FAQ. |
| POST | `/v1/faq/ask` | Ask — `{message, user_id?, session_id?}`. Returns `source`, `answer`, `citations`, `memories_used`. |
| GET | `/v1/memory/{user_id}` | List a user's remembered facts. |
| DELETE | `/v1/memory/{user_id}` | Forget everything for a user. |

`user_id` scopes long-term memory; omit it and memory is scoped to the session
(pass a stable id for true cross-session memory).

## 15.6 Step-by-step walkthrough

```bash
H='-H Authorization:Bearer demo-key -H content-type:application/json'

# 1. Curate an FAQ
curl -s $H localhost:8000/v1/faqs \
  -d '{"question":"How do I reset my password?","answer":"Go to Settings > Security."}'

# 2. Session s1: the user states a fact, then asks a known question
curl -s $H localhost:8000/v1/faq/ask \
  -d '{"message":"Hi, my name is Ada.","user_id":"ada","session_id":"s1"}'
curl -s $H localhost:8000/v1/faq/ask \
  -d '{"message":"how can I reset my password?","user_id":"ada","session_id":"s1"}' | jq '{source, answer}'
#   -> { "source":"faq", "answer":"Go to Settings > Security." }   ← curated, no LLM

# 3. A NEW session s2, same user: the earlier fact is recalled
curl -s $H localhost:8000/v1/faq/ask \
  -d '{"message":"How does FAISS work?","user_id":"ada","session_id":"s2"}' | jq '{source, memories_used}'
#   -> { "source":"rag", "memories_used":["name is Ada", "asked about ...", ...] }

# 4. Inspect the user's persisted memory
curl -s $H localhost:8000/v1/memory/ada | jq '.memories[] | {kind, content}'
#   -> {"kind":"fact","content":"name is Ada"} ; {"kind":"topic","content":"asked about faiss"} ; …
```

Step 3 is the payoff: a fact stated in session `s1` is remembered and surfaced
in a *different* session `s2` — memory that outlives the conversation.

## 15.7 Configuration

```bash
FAQ_MATCH_THRESHOLD=0.45      # combined score to prefer a curated FAQ over RAG
MEMORY_ENABLED=true
MEMORY_RECALL_LIMIT=5         # memories injected/returned per turn
```

## 15.8 How it composes

- **Chat / condensing (doc 12):** the FAQ bot condenses follow-ups and persists
  to the same conversation store, so `session_id` history works identically.
- **Dialogue manager (doc 14):** route `question` intents to the FAQ bot to get
  FAQ-first + memory; keep canned replies for social intents.
- **Voice (doc 13):** the voice FSM's `transcript` can call the FAQ bot so spoken
  questions get curated answers and the caller is remembered across calls.
- **Tenancy (docs 7, 10):** FAQs and memory are tenant-isolated; `purge_tenant`
  clears both.

## 15.9 Extending

- **Analytics-driven FAQs:** mine the conversation logs / `topic` memories for
  frequent questions and promote them into curated FAQs (raising the FAQ hit
  rate and cutting LLM cost).
- **LLM extraction/summary:** replace `extract_memories` with an LLM that
  summarizes salient facts, and periodically compress old memories.
- **Memory decay / TTL:** expire stale memories using `last_seen`.
- **Confidence bands:** between "clear FAQ hit" and "no match", offer the FAQ as
  a suggestion ("Did you mean: …?") instead of answering outright.

## 15.10 Reference map

| Concern | Code |
|---------|------|
| FAQ store + semantic/lexical matcher | `app/core/faq.py` |
| Long-term memory store + extraction/recall | `app/core/memory.py` |
| Orchestration (FAQ-first, memory, RAG fallback) | `app/core/faqbot.py` |
| Engine wiring + FAQ management | `app/core/rag.py` (`faq_bot`, `add_faq`, …) |
| Endpoints | `app/api/routes.py` (`/v1/faqs`, `/v1/faq/ask`, `/v1/memory/*`) |
| Config | `app/config.py` (`faq_match_threshold`, `memory_enabled`, `memory_recall_limit`) |
| Schemas | `app/models.py` (`FAQCreate`, `FAQAskResponse`, `MemoryListResponse`, …) |

Back to the [docs index](README.md).
