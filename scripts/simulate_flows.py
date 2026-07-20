#!/usr/bin/env python3
"""Simulate example conversation flows and emit a Markdown report with diagrams.

Runs a set of scripted flows (dialogue, FAQ, and voice) through the live stack
into a throwaway offline engine, then prints a Markdown report containing:

* the static voice-FSM state diagram and dialogue-policy diagram, and
* for each flow: a summary table, a User↔Bot sequence diagram, and the
  state/intent **path** actually taken — all as Mermaid.

Usage::

    python scripts/simulate_flows.py               # print report to stdout
    python scripts/simulate_flows.py --out FILE.md # also write it to FILE

Fully offline (no OPENAI_API_KEY needed).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.flow_diagram import (  # noqa: E402
    dialogue_policy_diagram,
    fsm_state_diagram,
    trace_report,
)
from app.core.rag import RagEngine  # noqa: E402
from app.core.simulator import Flow  # noqa: E402

TENANT = "sim"

FLOWS = [
    Flow(name="support-dialogue", channel="dialogue", turns=[
        "hi",
        "How does FAISS reduce latency?",
        "How are refunds handled in the policy docs?",
        "thanks, that's all",
    ]),
    Flow(name="faq-with-memory", channel="faq", user_id="ada", turns=[
        "Hi, my name is Ada.",
        "how can I reset my password?",
        "How does FAISS work?",
    ]),
    Flow(name="voice-call-with-barge-in", channel="voice", turns=[
        {"event": "start"},
        {"event": "transcript", "text": "How does FAISS reduce latency?"},
        {"event": "barge_in"},
        {"event": "speak_done"},   # illegal from listening -> rejected
        {"event": "transcript", "text": "what about nprobe?"},
        {"event": "speak_done"},
        {"event": "end"},
    ]),
]


def build_engine(tmp: Path) -> RagEngine:
    engine = RagEngine(Settings(deployment_mode="single_tenant",
                                openai_api_key=None, data_dir=tmp))
    engine.ingest(tenant=TENANT, doc_id="faiss", source="faiss.md",
                  text="FAISS is a similarity-search library. Its IVF index reduces "
                       "query latency by scanning only a few partitions via nprobe.")
    engine.ingest(tenant=TENANT, doc_id="policy", source="policy.md",
                  metadata={"doc_type": "policy"},
                  text="Refund policy: customers may request a refund within thirty days.")
    engine.add_faq(tenant=TENANT, question="How do I reset my password?",
                   answer="Go to Settings > Security > Reset password.")
    return engine


def build_report(engine: RagEngine) -> str:
    parts = ["# Simulated Conversation Flows\n",
             "## Voice assistant — state machine\n",
             "```mermaid\n" + fsm_state_diagram() + "\n```\n",
             "## Dialogue policy — intent → action\n",
             "```mermaid\n" + dialogue_policy_diagram() + "\n```\n",
             "## Flows\n"]
    for flow in FLOWS:
        trace = engine.simulator.run(tenant=TENANT, flow=flow)
        parts.append(trace_report(trace))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate conversation flows.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        engine = build_engine(Path(tmp))
        report = build_report(engine)

    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out} ({len(report)} chars).")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
