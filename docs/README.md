# RAGSystem Documentation

Step-by-step documentation for the full product lifecycle of a FAISS + OpenAI
RAG document Q&A system. Each stage links to the code that implements it.

| # | Stage | What it covers |
|---|-------|----------------|
| 1 | [Ideation & requirements](01-ideation-and-requirements.md) | Framing the product, requirements, guardrails |
| 2 | [Architecture & data model](02-architecture.md) | Layers, data flow, the vectors-vs-metadata split |
| 3 | [Implementation](03-implementation.md) | Indexing pipeline + query/RAG pipeline, step by step |
| 4 | [Deployment & ops](04-deployment-and-ops.md) | Packaging, config, auth, secrets, error handling |
| 5 | [Evaluation & monitoring](05-evaluation-and-monitoring.md) | Offline/online evals, metrics, index maintenance |
| 6 | [Scaling & evolution](06-scaling-and-evolution.md) | FAISS scaling, caching, multi-tenancy, roadmap |
| 7 | [Deployment modes](07-deployment-modes.md) | **Single-tenant vs multi-tenant**, step by step |
| 8 | [Reranking stage](08-reranking.md) | **Two-stage retrieve-then-rerank**, strategies, tuning |
| 9 | [Hybrid retrieval](09-hybrid-retrieval.md) | **BM25 + vector search** fused (RRF / weighted) |
| 10 | [Tenant isolation](10-tenant-isolation.md) | **Index-per-tenant vs shared namespace** |
| 11 | [Retrieval evaluation](11-retrieval-evaluation.md) | **Precision / recall / MAP / nDCG** metrics, step by step |
| 12 | [Multi-turn chat](12-multi-turn-chat.md) | **Conversational RAG** with context tracking + condensing |
| 13 | [Voice session state machine](13-voice-session-state-machine.md) | **Voice assistant FSM**: listening/thinking/speaking, barge-in |
| 14 | [Dialogue manager](14-dialogue-manager.md) | **Intent classification + persistence**, policy routing |
| 15 | [FAQ bot with memory](15-faq-bot-memory.md) | **Context-aware FAQ bot**, curated-first + cross-session memory |

> Throughout, code references use the `path:symbol` form, e.g.
> `app/core/ingest.py:IngestionPipeline.ingest_document`.
