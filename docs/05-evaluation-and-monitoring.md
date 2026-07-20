# 5. Evaluation, Monitoring & Maintenance

> **Goal of this stage:** make the system *eval-driven*, not intuition-driven,
> and keep it healthy as the corpus and models change.

---

## 5.1 Offline evaluation

Implemented in `eval/run_eval.py` with a labeled set in `eval/dataset.jsonl` and
a fixed corpus in `eval/corpus.jsonl`.

```bash
python eval/run_eval.py
```

It reports, separating **retrieval** quality from **answer** quality:

| Metric | What it tells you | Why it matters |
|--------|-------------------|----------------|
| **Recall@k / hit-rate** | did the right doc appear in the citations? | isolates the retriever |
| **MRR** | how highly was it ranked? | ranking quality, not just presence |
| **Keyword coverage** | did the answer contain the expected facts? | cheap answer-relevance proxy |

The harness **fails CI** if `recall_at_k` drops below a floor (0.6), so a bad
chunking/embedding change is caught before it ships.

### How to build your eval set
1. Collect 20–200 real questions per domain.
2. Label each with the `expected_doc_id` (and optionally the exact passage).
3. Add `expected_keywords` — facts a correct answer must mention.
4. Keep the set in version control; grow it whenever a bug is found (every
   production failure becomes a regression test).

### Beyond keyword coverage
Keyword coverage is a stand-in that runs offline. For real answer grading, add:
- **LLM-as-judge:** ask a model to score `answer` vs `reference` for
  faithfulness/relevance (plug into `evaluate()` in `run_eval.py`).
- **Human labeling:** sample and score periodically; treat as ground truth.

## 5.2 Online evaluation & feedback loops

- **Explicit feedback:** capture thumbs up/down + a reason on each answer
  (extend `QueryResponse` and log it against `request_id`).
- **Traffic sampling:** send a % of real `(question, retrieved, answer)` triples
  to human review.
- **Faithfulness monitoring:** flag answers whose citations don't actually
  support the claims (secondary LLM check) — the leading indicator of drift.

## 5.3 Monitoring in production

From `/metrics` (`observability/metrics.py`) track and alert on:

- **Latency:** `retrieval_ms` vs `generation_ms` p95/p99 — tells you *which*
  half to optimize.
- **Throughput & errors:** `queries` rate, error rate.
- **Cost:** `tokens_total` per tenant.
- **Retrieval health:** fraction of queries returning zero hits or below
  `MIN_SCORE` (a spike means missing content or an index problem).

Export these to Prometheus/OTel and dashboard them; the in-process registry here
is the same shape, just not durable across restarts.

## 5.4 Model & prompt iteration

- **A/B test** prompt variants or model versions; compare on the offline set
  *and* online feedback before rolling out.
- **Justify upgrades with data:** move to a newer embedding/generation model
  only when it beats the incumbent on your eval set.
- **Guard against regressions:** the CI eval floor + a growing labeled set make
  silent quality drops visible.

> Changing the **embedding model changes the vector space** — you must re-embed
> the entire corpus and rebuild indices (see §5.5). Never mix vectors from two
> embedding models in one index.

## 5.5 Index & document maintenance

- **Incremental updates:** re-ingesting a changed doc removes its old vectors
  and adds new ones automatically (`ingest.py`) — no full rebuild needed for
  small changes.
- **Full rebuild:** when the embedding model changes or a large fraction of the
  corpus turns over, rebuild from the raw docs in S3 via
  `scripts/ingest_dir.py`. Build into a **new** index and swap it in so queries
  keep serving the old index until the new one is ready (versioned indices).
- **Deletion & permission revocation:** `DELETE /v1/documents/{id}` removes
  vectors (`remove_ids`) and metadata rows together, so revoked content stops
  being retrievable immediately.
- **Freshness:** store timestamps in chunk `metadata`; you can filter or weight
  by recency at query time.

## 5.6 Step-by-step checklist

1. [ ] Build and version a labeled eval set for each domain.
2. [ ] Wire `python eval/run_eval.py` into CI as a gate.
3. [ ] Add explicit user feedback capture keyed by `request_id`.
4. [ ] Dashboard `/metrics`; alert on latency, error, cost, zero-hit rate.
5. [ ] Define the re-embed/rebuild runbook (new index → swap).
6. [ ] Turn every production failure into a new eval case.

Proceed to [scaling & evolution](06-scaling-and-evolution.md).
