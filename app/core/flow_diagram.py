"""Render conversation state/flow diagrams as Mermaid text.

Two kinds of diagram:

* **Static** — the *definition* of the machines: the voice FSM state diagram
  (from its transition table) and the dialogue policy (intent → action). These
  document what *can* happen.
* **Dynamic** — the *trace* of a simulated conversation
  (:class:`app.core.simulator.FlowTrace`): a sequence diagram of the exchange
  and the state/intent **path** actually taken. These document what *did*
  happen.

Everything returns a Mermaid string (```mermaid ...```), which renders natively
on GitHub, in the docs, and in this project's Artifacts.
"""

from __future__ import annotations

from app.core.dialogue import _POLICY
from app.core.simulator import FlowTrace
from app.core.voice_fsm import _TRANSITIONS


def _san(text: str | None) -> str:
    """Make a label safe for Mermaid (no newlines / quotes / colons)."""
    if not text:
        return ""
    return " ".join(text.split()).replace('"', "'").replace(":", " -").replace(";", ",")


# --------------------------------------------------------------------------- #
# Static diagrams
# --------------------------------------------------------------------------- #
def fsm_state_diagram() -> str:
    """The voice-assistant FSM as a Mermaid stateDiagram."""
    lines = ["stateDiagram-v2", "    [*] --> idle"]
    for state, transitions in _TRANSITIONS.items():
        for event, target in transitions.items():
            lines.append(f"    {state.value} --> {target.value}: {event.value}")
    lines.append("    ended --> [*]")
    return "\n".join(lines)


def dialogue_policy_diagram() -> str:
    """The dialogue policy (intent → action) as a Mermaid flowchart."""
    # Namespace intent vs action node ids so shared names (e.g. "help",
    # "smalltalk") don't collapse into a single node.
    lines = ["flowchart LR"]
    for intent, action in _POLICY.items():
        lines.append(
            f"    i_{intent.value}([{intent.value}]) --> a_{action.value}[[{action.value}]]"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dynamic diagrams (from a simulated trace)
# --------------------------------------------------------------------------- #
def flow_sequence_diagram(trace: FlowTrace) -> str:
    """A Mermaid sequenceDiagram of the User↔Bot exchange."""
    lines = ["sequenceDiagram", "    participant U as User", "    participant B as Bot"]
    for step in trace.steps:
        if step.kind == "voice":
            label = step.event + (f" ({_san(step.user)})" if step.user else "")
            lines.append(f"    U->>B: {label}")
            if step.detail.get("rejected"):
                lines.append(f"    Note right of B: rejected in {step.state_from}")
            else:
                lines.append(f"    Note right of B: {step.state_from} ➜ {step.state_to}")
            if step.response:
                lines.append(f"    B-->>U: {_san(step.response)}")
        else:
            lines.append(f"    U->>B: {_san(step.user)}")
            tag = (f"intent={step.intent} action={step.action}"
                   if step.kind == "dialogue" else f"source={step.source}")
            lines.append(f"    Note right of B: {tag}")
            lines.append(f"    B-->>U: {_san(step.response)}")
    return "\n".join(lines)


def flow_path_diagram(trace: FlowTrace) -> str:
    """The path a flow actually took.

    Voice → a stateDiagram of the visited transitions (a walk through the FSM).
    Dialogue/FAQ → a flowchart of the intent/source sequence.
    """
    if trace.channel == "voice":
        lines = ["stateDiagram-v2", "    [*] --> idle"]
        seen: set[tuple[str, str, str]] = set()
        for step in trace.steps:
            if step.detail.get("rejected") or not step.state_to:
                continue
            edge = (step.state_from, step.state_to, step.event or "")
            if edge in seen:
                continue
            seen.add(edge)
            lines.append(f"    {step.state_from} --> {step.state_to}: {step.event}")
        return "\n".join(lines)

    # dialogue / faq: linear flowchart of the step sequence.
    lines = ["flowchart LR", "    start([start])"]
    prev = "start"
    for step in trace.steps:
        node = f"n{step.index}"
        label = step.intent if step.kind == "dialogue" else step.source
        lines.append(f'    {node}["{step.index}: {label}"]')
        lines.append(f"    {prev} --> {node}")
        prev = node
    return "\n".join(lines)


def trace_report(trace: FlowTrace) -> str:
    """A compact Markdown report: summary table + sequence + path diagrams."""
    header = f"### Flow: `{trace.name}`  (channel: {trace.channel})\n"
    rows = ["| # | turn | outcome |", "|---|------|---------|"]
    for s in trace.steps:
        if s.kind == "voice":
            turn = s.event + (f" ({_san(s.user)})" if s.user else "")
            outcome = ("rejected" if s.detail.get("rejected")
                       else f"{s.state_from} → {s.state_to}")
        else:
            turn = _san(s.user)
            outcome = (f"{s.intent} / {s.action}" if s.kind == "dialogue"
                       else f"source={s.source}")
        rows.append(f"| {s.index} | {turn} | {outcome} |")
    table = "\n".join(rows)
    seq = "```mermaid\n" + flow_sequence_diagram(trace) + "\n```"
    path = "```mermaid\n" + flow_path_diagram(trace) + "\n```"
    return f"{header}\n{table}\n\n**Exchange**\n\n{seq}\n\n**Path**\n\n{path}\n"
