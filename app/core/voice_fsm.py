"""Voice-assistant session state machine (pure logic).

A voice conversation is turn-taking with strict phases: the assistant is either
*listening* to the user, *thinking* about a reply, or *speaking* it — never two
at once. Modeling that as an explicit finite state machine makes the rules
enforceable (you can't start speaking while still listening), the behavior
testable, and barge-in / error handling first-class rather than ad-hoc flags.

This module is **pure**: no I/O, no persistence, no RAG. It defines the states,
the events, and the legal transitions between them. The
:mod:`app.core.voice_session` manager layers persistence and side effects
(running the RAG turn) on top.

States
------
* ``IDLE``      — session exists, not yet listening (e.g. awaiting a wake word).
* ``LISTENING`` — capturing the user's speech (VAD/ASR active).
* ``THINKING``  — transcript received; retrieving + generating a reply. Transient
                  on a synchronous server (entered and left within one request).
* ``SPEAKING``  — playing the reply via TTS.
* ``ENDED``     — terminal; the session is closed.
* ``ERROR``     — a recoverable fault; ``RECOVER`` returns to ``IDLE``.

Events
------
Client-driven: ``START``, ``TRANSCRIPT``, ``SPEAK_DONE``, ``BARGE_IN``,
``SILENCE_TIMEOUT``, ``END``, ``ERROR``, ``RECOVER``, ``WAKE``.
Server-internal: ``THINK_DONE`` (emitted once a reply is ready).

Transition diagram::

      ┌────── WAKE/START ──────┐
      ▼                        │
    IDLE ── START ─▶ LISTENING ─ TRANSCRIPT ─▶ THINKING ─ THINK_DONE ─▶ SPEAKING
      ▲                 ▲  │                                              │  │
      │  SILENCE_TIMEOUT┘  │                              SPEAK_DONE ─────┘  │
      │                    └───────────────── BARGE_IN ────────────────────┘
      │                                                            (SPEAKING→LISTENING)
   RECOVER                        END (from any active) ─▶ ENDED
      │
    ERROR ◀── ERROR (from any active)
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"
    ERROR = "error"


class Event(str, Enum):
    START = "start"
    WAKE = "wake"
    TRANSCRIPT = "transcript"
    THINK_DONE = "think_done"      # internal (server emits when reply is ready)
    SPEAK_DONE = "speak_done"
    BARGE_IN = "barge_in"
    SILENCE_TIMEOUT = "silence_timeout"
    END = "end"
    ERROR = "error"
    RECOVER = "recover"


# Events the server generates itself; clients cannot send these directly.
INTERNAL_EVENTS: frozenset[Event] = frozenset({Event.THINK_DONE})

# Terminal states accept no further events.
TERMINAL_STATES: frozenset[State] = frozenset({State.ENDED})


class InvalidTransition(Exception):
    """Raised when an event is not legal in the current state."""

    def __init__(self, state: State, event: Event) -> None:
        super().__init__(f"Event '{event.value}' is not allowed in state '{state.value}'.")
        self.state = state
        self.event = event


# The transition table is the single source of truth for the machine.
_TRANSITIONS: dict[State, dict[Event, State]] = {
    State.IDLE: {
        Event.START: State.LISTENING,
        Event.WAKE: State.LISTENING,
        Event.END: State.ENDED,
        Event.ERROR: State.ERROR,
    },
    State.LISTENING: {
        Event.TRANSCRIPT: State.THINKING,
        Event.SILENCE_TIMEOUT: State.IDLE,
        Event.END: State.ENDED,
        Event.ERROR: State.ERROR,
    },
    State.THINKING: {
        Event.THINK_DONE: State.SPEAKING,
        Event.END: State.ENDED,
        Event.ERROR: State.ERROR,
    },
    State.SPEAKING: {
        Event.SPEAK_DONE: State.LISTENING,
        Event.BARGE_IN: State.LISTENING,
        Event.END: State.ENDED,
        Event.ERROR: State.ERROR,
    },
    State.ENDED: {},
    State.ERROR: {
        Event.RECOVER: State.IDLE,
        Event.END: State.ENDED,
    },
}


class VoiceStateMachine:
    """Stateless helper that validates and applies transitions."""

    @staticmethod
    def next_state(state: State, event: Event) -> State:
        """Return the state reached by ``event`` from ``state``.

        Raises :class:`InvalidTransition` if the event is not legal there.
        """
        transitions = _TRANSITIONS.get(state, {})
        if event not in transitions:
            raise InvalidTransition(state, event)
        return transitions[event]

    @staticmethod
    def allowed_events(state: State, *, include_internal: bool = False) -> list[Event]:
        """Events legal from ``state`` (client-facing unless ``include_internal``)."""
        events = _TRANSITIONS.get(state, {}).keys()
        if include_internal:
            return list(events)
        return [e for e in events if e not in INTERNAL_EVENTS]

    @staticmethod
    def can(state: State, event: Event) -> bool:
        return event in _TRANSITIONS.get(state, {})

    @staticmethod
    def is_terminal(state: State) -> bool:
        return state in TERMINAL_STATES
