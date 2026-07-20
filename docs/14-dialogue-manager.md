# 14. Dialogue Manager with Intent Persistence

> **Goal of this doc:** add a dialogue manager that classifies each turn's
> *intent*, decides an *action* via a policy, and **persists the intent history**
> and dialogue state across turns — step by step.

---

## 14.1 Why a dialogue manager?

The chat and voice layers (docs 12–13) answer questions, but a real assistant
handles more than questions: greetings, "what can you do?", "thanks, that's
all", "yes/no". Routing all of those through RAG produces awkward, ungrounded
answers ("I don't know based on the available documents." to *"hi"*).

A **dialogue manager** is the control layer that sits above chat:

```
user turn ─▶ 1. classify INTENT ─▶ 2. POLICY: intent → action ─▶ 3. ACT ─▶ 4. PERSIST
                                                                    │
              question ─────────────────────────────────────▶ RAG chat (grounded)
              greeting/goodbye/help/smalltalk ───────────────▶ canned reply (no RAG call)
```

It decides *what kind of turn this is* and *how to handle it*, then records the
decision so the conversation has a durable, inspectable **intent trail**.

## 14.2 Intent classification (`app/core/intents.py`)

Each message is classified into a small taxonomy:

| Intent | Examples | Action |
|--------|----------|--------|
| `question` | "What is FAISS?", "explain reranking" | answer via RAG |
| `greeting` | "hi", "hello there" | greet |
| `goodbye` | "bye", "thanks, that's all", "no more questions" | close |
| `help` | "what can you do?" | capabilities |
| `smalltalk` | "how are you?", "who are you?" | deflect |
| `affirm` / `deny` | "yes" / "no" | acknowledge |

Two interchangeable strategies (`INTENT_STRATEGY`), one interface
(`classify(message) -> IntentResult`):

- **`rule`** (default, offline) — keyword/pattern rules. Deterministic and
  dependency-free; the tests pin behavior to it. Question markers take
  precedence over stray social tokens, so *"thanks — how does hybrid work?"* is
  a `question`, not a `goodbye`.
- **`llm`** — the model classifies into the same taxonomy (better on
  paraphrases); needs OpenAI and falls back to rules on any error.

**Design bias:** this is a documentation assistant, so anything not clearly
social defaults to `question` and gets answered. Below
`INTENT_CONFIDENCE_THRESHOLD`, a non-question intent is also treated as a
question — a safe default.

### Slots
The classifier also extracts light **slots**. Today it detects a `doc_type`
from phrases like *"in the **policy** docs"*, which the manager passes to
retrieval as a metadata filter — so slots directly steer what gets retrieved.

## 14.3 The policy (`app/core/dialogue.py`)

A tiny, explicit table maps intent → action (`_POLICY`), and non-RAG actions
have canned replies (`_CANNED`). The policy is the one place to change behavior:

```python
_POLICY = {
    Intent.QUESTION: Action.ANSWER,     # -> RAG chat
    Intent.GREETING: Action.GREET,      # -> canned
    Intent.GOODBYE:  Action.CLOSE,
    Intent.HELP:     Action.HELP,
    Intent.SMALLTALK:Action.SMALLTALK,
    Intent.AFFIRM:   Action.ACKNOWLEDGE,
    Intent.DENY:     Action.ACKNOWLEDGE,
}
```

Only `ANSWER` calls the RAG chat (and thus spends a query quota); greetings and
goodbyes are answered instantly without retrieval.

## 14.4 Intent persistence (`app/core/dialogue_store.py`)

The "memory" of the manager — two tables keyed by `(tenant, session_id)`:

- **`dialogue_state`** — the running state: `current_intent`, `turn_count`, and
  accumulated `slots`.
- **`intent_events`** — an **append-only log** of every classified turn
  (message, intent, confidence, action, slots). This is the intent persistence:
  a durable trail you can inspect, audit, or later use to train a better
  classifier.

Sessions are tenant-isolated like everything else, and `purge_tenant` clears
them. The dialogue `session_id` doubles as the chat/voice session id, so RAG
answers keep full conversation memory and follow-up condensing (docs 12–13).

## 14.5 API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/dialogue` | Handle a turn: classify → act → persist. Returns `intent`, `action`, `answer`, `citations`, `slots`. |
| GET | `/v1/dialogue/{session_id}` | Dialogue state + the persisted intent history. |
| DELETE | `/v1/dialogue/{session_id}` | Clear the dialogue. |

## 14.6 Step-by-step walkthrough

```bash
say() { curl -s localhost:8000/v1/dialogue -H 'Authorization: Bearer demo-key' \
         -H 'content-type: application/json' -d "$1"; }

# A greeting → canned reply, no retrieval
say '{"message":"hi"}'
#  intent=greeting  action=greet  citations=[]

# A question → grounded RAG answer (reuse the returned session_id to continue)
say '{"message":"What is FAISS?","session_id":"S"}'
#  intent=question  action=answer  citations=[faiss]

# A slot steers retrieval: "in the policy docs" → doc_type=policy filter
say '{"message":"How are refunds handled in the policy docs?","session_id":"S"}'
#  intent=question  slots={"doc_type":"policy"}  citations=[policy]   ← filtered

# A farewell → close
say '{"message":"thanks, that'\''s all","session_id":"S"}'
#  intent=goodbye  action=close

# Inspect the persisted intent trail + state
curl -s localhost:8000/v1/dialogue/S -H 'Authorization: Bearer demo-key' | jq \
  '{current_intent, turn_count, slots, trail: [.intents[].intent]}'
#  { "current_intent":"goodbye", "turn_count":4, "slots":{"doc_type":"policy"},
#    "trail":["greeting","question","question","goodbye"] }
```

That last response *is* the intent persistence: every turn's classified intent,
the accumulated slots, and the current dialogue state, durably stored.

## 14.7 Configuration

```bash
INTENT_STRATEGY=rule               # rule | llm
INTENT_CONFIDENCE_THRESHOLD=0.5    # below this, non-question intents answer via RAG
```

## 14.8 How it composes with the other layers

- **Chat (doc 12):** `ANSWER` calls `RagEngine.chat`, so questions get
  conversation memory + follow-up condensing for free.
- **Voice (doc 13):** the voice FSM's `transcript` handling can route through
  the dialogue manager instead of straight to chat, so a spoken *"goodbye"*
  drives the FSM to `ended` — the manager provides the *intent*, the FSM
  provides the *phase*.
- **Tenancy / quotas (docs 7, 10):** inherited unchanged.

## 14.9 Extending

- **More intents:** add a taxonomy entry, a rule (or let the LLM handle it), and
  a `_POLICY` row — e.g. `command` ("summarize this doc"), `feedback`
  ("that was wrong").
- **Slot filling / forms:** add required slots per intent and a `clarifying`
  turn that asks for a missing slot before acting (the state already persists
  accumulated slots).
- **Analytics:** the `intent_events` log is ready-made for dashboards
  (intent mix, fallback rate) and for bootstrapping a trained classifier.

## 14.10 Reference map

| Concern | Code |
|---------|------|
| Intent taxonomy + classifiers | `app/core/intents.py` |
| Policy + manager (routing, canned replies) | `app/core/dialogue.py` |
| Intent + state persistence | `app/core/dialogue_store.py` |
| Engine wiring | `app/core/rag.py` (`self.dialogue`) |
| Endpoints | `app/api/routes.py` (`/v1/dialogue`) |
| Config | `app/config.py` (`intent_strategy`, `intent_confidence_threshold`) |
| Schemas | `app/models.py` (`DialogueRequest`, `DialogueResponse`, `DialogueStateResponse`) |

Back to the [docs index](README.md).
