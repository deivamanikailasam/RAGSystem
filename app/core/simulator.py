"""Conversation-flow simulator.

Drives a *scripted* multi-turn conversation through the live stack and records
the trajectory, so you can exercise and visualize whole conversation flows —
regression-test dialogue paths, reproduce a bug, or generate documentation
diagrams — without a human at the keyboard.

Three channels, matching the layers built earlier:

* ``dialogue`` — each turn is a user message run through the dialogue manager
  (doc 14); the trace records the intent and action per turn.
* ``faq`` — each turn goes through the FAQ bot (doc 15); the trace records the
  answer source (``faq``/``rag``) and memories used.
* ``voice`` — each turn is a voice **event** driven through the FSM (doc 13);
  the trace records the ``state_from → state_to`` transition, so the path *is* a
  walk through the state machine. Illegal events are captured (not raised) so
  invalid flows can be simulated too.

The resulting :class:`FlowTrace` feeds :mod:`app.core.flow_diagram` to render
Mermaid state / sequence diagrams.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.voice_fsm import Event, InvalidTransition


@dataclass
class FlowStep:
    index: int
    kind: str                       # "dialogue" | "faq" | "voice"
    user: str | None = None         # user message or transcript text
    event: str | None = None        # voice event
    state_from: str | None = None
    state_to: str | None = None
    intent: str | None = None
    action: str | None = None
    source: str | None = None       # answer origin (faq/rag) or dialogue action
    response: str | None = None      # bot answer / spoken text (preview)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Flow:
    name: str
    channel: str                    # "dialogue" | "faq" | "voice"
    turns: list                     # dialogue/faq: list[str]; voice: list[dict]
    user_id: str | None = None
    session_id: str | None = None


@dataclass
class FlowTrace:
    name: str
    channel: str
    session_id: str
    steps: list[FlowStep]
    states: list[str]               # ordered path (states for voice, intents/sources otherwise)


def _preview(text: str | None, limit: int = 80) -> str | None:
    if not text:
        return text
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


class ConversationSimulator:
    def __init__(self, engine: Any) -> None:
        self._engine = engine  # RagEngine

    def run(self, *, tenant: str, flow: Flow) -> FlowTrace:
        if flow.channel == "dialogue":
            return self._run_dialogue(tenant, flow)
        if flow.channel == "faq":
            return self._run_faq(tenant, flow)
        if flow.channel == "voice":
            return self._run_voice(tenant, flow)
        raise ValueError(f"Unknown channel '{flow.channel}'.")

    # -- channels ---------------------------------------------------------- #
    def _run_dialogue(self, tenant: str, flow: Flow) -> FlowTrace:
        sid = flow.session_id or uuid.uuid4().hex
        steps: list[FlowStep] = []
        states: list[str] = []
        for i, message in enumerate(flow.turns):
            r = self._engine.dialogue.handle(
                tenant=tenant, message=message, session_id=sid
            )
            steps.append(FlowStep(
                index=i, kind="dialogue", user=message,
                intent=r.intent.value, action=r.action.value,
                source=r.action.value, response=_preview(r.answer),
                detail={"slots": r.slots, "citations": len(r.citations),
                        "confidence": r.confidence},
            ))
            states.append(r.intent.value)
        return FlowTrace(flow.name, "dialogue", sid, steps, states)

    def _run_faq(self, tenant: str, flow: Flow) -> FlowTrace:
        sid = flow.session_id or uuid.uuid4().hex
        steps: list[FlowStep] = []
        states: list[str] = []
        for i, message in enumerate(flow.turns):
            r = self._engine.faq_bot.ask(
                tenant=tenant, message=message,
                user_id=flow.user_id, session_id=sid,
            )
            steps.append(FlowStep(
                index=i, kind="faq", user=message, source=r.source,
                response=_preview(r.answer),
                detail={"memories_used": r.memories_used,
                        "faq_id": r.faq_id, "score": r.score},
            ))
            states.append(r.source)
        return FlowTrace(flow.name, "faq", sid, steps, states)

    def _run_voice(self, tenant: str, flow: Flow) -> FlowTrace:
        session = self._engine.voice.create_session(tenant, flow.session_id)
        sid = session.session_id
        steps: list[FlowStep] = []
        states: list[str] = [session.state.value]  # starts in idle
        for i, turn in enumerate(flow.turns):
            event = Event(turn["event"])
            text = turn.get("text")
            current = self._engine.voice.get_session(tenant, sid).state
            try:
                res = self._engine.voice.send_event(
                    tenant=tenant, session_id=sid, event=event, text=text
                )
                steps.append(FlowStep(
                    index=i, kind="voice", event=event.value, user=text,
                    state_from=res.previous_state.value, state_to=res.state.value,
                    response=_preview(res.say),
                ))
                states.append(res.state.value)
            except (InvalidTransition, ValueError) as exc:
                steps.append(FlowStep(
                    index=i, kind="voice", event=event.value, user=text,
                    state_from=current.value, state_to=current.value,
                    detail={"error": str(exc), "rejected": True},
                ))
        return FlowTrace(flow.name, "voice", sid, steps, states)
