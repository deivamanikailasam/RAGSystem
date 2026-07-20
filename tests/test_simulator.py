"""Conversation-flow simulator + diagram generator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.flow_diagram import (
    dialogue_policy_diagram,
    flow_path_diagram,
    flow_sequence_diagram,
    fsm_state_diagram,
    trace_report,
)
from app.core.rag import RagEngine
from app.core.simulator import Flow


@pytest.fixture
def engine(tmp_path: Path) -> RagEngine:
    eng = RagEngine(Settings(deployment_mode="single_tenant", openai_api_key=None,
                             data_dir=tmp_path / "d"))
    eng.ingest(tenant="default", doc_id="faiss", source="faiss.md",
               text="FAISS reduces latency using an IVF index and nprobe over dense vectors.")
    eng.add_faq(tenant="default", question="How do I reset my password?",
                answer="Go to Settings > Security.")
    return eng


# --- static diagrams ------------------------------------------------------- #
def test_fsm_state_diagram_is_valid_mermaid():
    d = fsm_state_diagram()
    assert d.startswith("stateDiagram-v2")
    assert "idle --> listening: start" in d
    assert "speaking --> listening: barge_in" in d
    assert "ended --> [*]" in d


def test_dialogue_policy_diagram_namespaces_nodes():
    d = dialogue_policy_diagram()
    assert d.startswith("flowchart LR")
    # Intent and action nodes must have distinct ids (no self-merge on "help").
    assert "i_help([help]) --> a_help[[help]]" in d
    assert "i_question([question]) --> a_answer[[answer]]" in d


# --- dialogue flow --------------------------------------------------------- #
def test_dialogue_flow_records_intents(engine: RagEngine):
    flow = Flow(name="d1", channel="dialogue",
                turns=["hi", "How does FAISS reduce latency?", "bye"])
    trace = engine.simulator.run(tenant="default", flow=flow)
    assert trace.states == ["greeting", "question", "goodbye"]
    assert trace.steps[1].action == "answer"
    assert len(trace.steps) == 3


def test_dialogue_flow_path_diagram(engine: RagEngine):
    flow = Flow(name="d1", channel="dialogue", turns=["hi", "What is FAISS?"])
    trace = engine.simulator.run(tenant="default", flow=flow)
    d = flow_path_diagram(trace)
    assert d.startswith("flowchart LR")
    assert "greeting" in d and "question" in d


# --- faq flow -------------------------------------------------------------- #
def test_faq_flow_records_sources(engine: RagEngine):
    flow = Flow(name="f1", channel="faq", user_id="u1", turns=[
        "how can I reset my password?",     # -> faq
        "How does FAISS reduce latency?",   # -> rag
    ])
    trace = engine.simulator.run(tenant="default", flow=flow)
    assert trace.states == ["faq", "rag"]


# --- voice flow ------------------------------------------------------------ #
def test_voice_flow_walks_the_fsm(engine: RagEngine):
    flow = Flow(name="v1", channel="voice", turns=[
        {"event": "start"},
        {"event": "transcript", "text": "How does FAISS reduce latency?"},
        {"event": "speak_done"},
        {"event": "end"},
    ])
    trace = engine.simulator.run(tenant="default", flow=flow)
    assert trace.states == ["idle", "listening", "speaking", "listening", "ended"]
    # The transcript turn produced a spoken reply.
    assert trace.steps[1].response


def test_voice_flow_captures_rejected_transition(engine: RagEngine):
    flow = Flow(name="v2", channel="voice", turns=[
        {"event": "start"},
        {"event": "speak_done"},   # illegal from listening
        {"event": "transcript", "text": "How does FAISS work?"},
        {"event": "end"},
    ])
    trace = engine.simulator.run(tenant="default", flow=flow)
    rejected = trace.steps[1]
    assert rejected.detail.get("rejected") is True
    # A rejected event does not advance the state.
    assert rejected.state_from == rejected.state_to == "listening"


def test_voice_path_diagram_excludes_rejected(engine: RagEngine):
    flow = Flow(name="v3", channel="voice", turns=[
        {"event": "start"},
        {"event": "speak_done"},   # rejected
        {"event": "end"},
    ])
    trace = engine.simulator.run(tenant="default", flow=flow)
    d = flow_path_diagram(trace)
    assert "idle --> listening: start" in d
    assert "listening --> ended: end" in d
    assert "speak_done" not in d  # rejected transition not drawn


# --- report + sequence ----------------------------------------------------- #
def test_sequence_diagram_and_report(engine: RagEngine):
    flow = Flow(name="d1", channel="dialogue", turns=["hi", "What is FAISS?"])
    trace = engine.simulator.run(tenant="default", flow=flow)
    seq = flow_sequence_diagram(trace)
    assert seq.startswith("sequenceDiagram")
    assert "U->>B:" in seq and "B-->>U:" in seq
    report = trace_report(trace)
    assert "```mermaid" in report and "Path" in report


def test_unknown_channel_raises(engine: RagEngine):
    with pytest.raises(ValueError):
        engine.simulator.run(tenant="default",
                             flow=Flow(name="x", channel="telepathy", turns=[]))
