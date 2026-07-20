"""Pure state-machine tests for the voice assistant FSM."""

from __future__ import annotations

import pytest

from app.core.voice_fsm import (
    Event,
    InvalidTransition,
    State,
    VoiceStateMachine,
)

fsm = VoiceStateMachine()


def test_happy_path_transitions():
    assert fsm.next_state(State.IDLE, Event.START) == State.LISTENING
    assert fsm.next_state(State.LISTENING, Event.TRANSCRIPT) == State.THINKING
    assert fsm.next_state(State.THINKING, Event.THINK_DONE) == State.SPEAKING
    assert fsm.next_state(State.SPEAKING, Event.SPEAK_DONE) == State.LISTENING


def test_barge_in_interrupts_speaking():
    assert fsm.next_state(State.SPEAKING, Event.BARGE_IN) == State.LISTENING


def test_silence_timeout_returns_to_idle():
    assert fsm.next_state(State.LISTENING, Event.SILENCE_TIMEOUT) == State.IDLE


def test_wake_from_idle():
    assert fsm.next_state(State.IDLE, Event.WAKE) == State.LISTENING


def test_end_from_any_active_state():
    for s in (State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING):
        assert fsm.next_state(s, Event.END) == State.ENDED


def test_error_and_recover():
    assert fsm.next_state(State.LISTENING, Event.ERROR) == State.ERROR
    assert fsm.next_state(State.ERROR, Event.RECOVER) == State.IDLE


def test_illegal_transitions_raise():
    # Can't speak while listening, can't transcribe while speaking, etc.
    with pytest.raises(InvalidTransition):
        fsm.next_state(State.LISTENING, Event.SPEAK_DONE)
    with pytest.raises(InvalidTransition):
        fsm.next_state(State.SPEAKING, Event.TRANSCRIPT)
    with pytest.raises(InvalidTransition):
        fsm.next_state(State.IDLE, Event.TRANSCRIPT)


def test_terminal_state_accepts_nothing():
    assert fsm.is_terminal(State.ENDED)
    assert fsm.allowed_events(State.ENDED) == []
    with pytest.raises(InvalidTransition):
        fsm.next_state(State.ENDED, Event.START)


def test_allowed_events_excludes_internal():
    # THINK_DONE is server-internal and must not be offered to clients.
    client_events = fsm.allowed_events(State.THINKING)
    assert Event.THINK_DONE not in client_events
    assert Event.THINK_DONE in fsm.allowed_events(State.THINKING, include_internal=True)


def test_can_helper():
    assert fsm.can(State.IDLE, Event.START)
    assert not fsm.can(State.IDLE, Event.SPEAK_DONE)


def test_invalid_transition_carries_context():
    try:
        fsm.next_state(State.SPEAKING, Event.TRANSCRIPT)
    except InvalidTransition as exc:
        assert exc.state == State.SPEAKING
        assert exc.event == Event.TRANSCRIPT
