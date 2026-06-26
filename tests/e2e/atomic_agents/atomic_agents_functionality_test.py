"""
AgentScribe x Atomic Agents — every capture surface, every output format.

WHAT ATOMIC AGENTS IS (in one line): a *structured-output* framework — you
define Pydantic input/output schemas and the agent returns a typed object. The
AgentScribe adapter turns those objects (and chat histories / log events) into
the canonical interaction shape.

WHY THIS TEST NEEDS NO FRAMEWORK AND NO API KEY: the adapter is duck-typed — it
reads fields off whatever object you hand it. An Atomic Agents schema is just a
Pydantic model, i.e. a class whose instance serializes to a dict. A plain
`@dataclass` serializes to the *same* dict shape, so feeding dataclasses
exercises the exact same adapter code paths while keeping the test tiny,
deterministic, and runnable anywhere.

ONE file, two modes:
  * `python atomic_agents_functionality_test.py`
        Writes out_atomic/<surface>_<format>.jsonl so you can eyeball real output.
  * `pytest tests/e2e/atomic_agents/atomic_agents_functionality_test.py`
        Asserts each surface is captured correctly AND every record conforms to
        its format (via agentscribe.core.validation). Writes nothing.

The five surfaces below mirror the adapter's five exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agentscribe.adapters.atomic_agents import (
    AtomicAgentsTraceCollector,
    from_agent_response,
    from_agent_run,
    from_chat_history,
    from_log_event,
)
from agentscribe.core.formatter import Formatter
from agentscribe.core.validation import validate_record

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT = SCRIPT_DIR / "out_atomic"
FORMATS = Formatter.SUPPORTED_FORMATS          # every shipped format, not just one


# --------------------------------------------------------------------------- #
# Atomic Agents-shaped data. Real schemas are Pydantic BaseIOSchema; these
# dataclasses produce the identical dict shape with zero extra dependencies.
# --------------------------------------------------------------------------- #
@dataclass
class QuestionInput:
    question: str


@dataclass
class AnswerOutput:
    answer: str
    confidence: float


def make_agent() -> dict:
    """A stand-in for an AtomicAgent. The adapter only reads these config fields
    (for provenance), so a plain dict is the clearest way to express it."""
    return {"config": {
        "system_prompt": "You are a helpful assistant. Answer with a short fact.",
        "model": "google/gemini-3.1-flash-lite",
        "input_schema": QuestionInput,
        "output_schema": AnswerOutput,
        "tools": [],
        "context_providers": [],
    }}


# --------------------------------------------------------------------------- #
# Capture surfaces — each returns one CanonicalInteraction.
# --------------------------------------------------------------------------- #
def capture_agent_run():
    """from_agent_run: an input schema + output schema -> a user/assistant pair."""
    return from_agent_run(
        QuestionInput(question="Give me one fact about the moon."),
        AnswerOutput(answer="The Moon is about 384,400 km from Earth.", confidence=0.99),
        agent=make_agent(),
    )


def capture_agent_response():
    """from_agent_response: a response object plus the prompt that produced it."""
    return from_agent_response(
        AnswerOutput(answer="Bonjour.", confidence=0.9),
        prompt=QuestionInput(question="Translate 'good morning' to French."),
        agent=make_agent(),
    )


def capture_chat_history():
    """from_chat_history: a multi-turn ChatHistory export."""
    history = {"messages": [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Name a primary color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Name another."},
        {"role": "assistant", "content": "Red."},
    ]}
    return from_chat_history(history, agent=make_agent())


def capture_log_event():
    """from_log_event: a hook/logging payload emitted around a run."""
    return from_log_event({
        "event_type": "agent.run",
        "input": "What is 2 + 2?",
        "output": "4.",
    })


def capture_collector():
    """AtomicAgentsTraceCollector: record one response, return its interaction."""
    collector = AtomicAgentsTraceCollector()
    collector.record_response(
        AnswerOutput(answer="Octopuses have three hearts.", confidence=0.95),
        prompt=QuestionInput(question="One fun fact about octopuses?"),
        agent=make_agent(),
    )
    return collector.interactions[-1]


SURFACES = [
    ("from_agent_run",      capture_agent_run),
    ("from_agent_response", capture_agent_response),
    ("from_chat_history",   capture_chat_history),
    ("from_log_event",      capture_log_event),
    ("collector",           capture_collector),
]


# --------------------------------------------------------------------------- #
# Correctness — does each surface capture what we expect?
# --------------------------------------------------------------------------- #
def test_agent_run_captures_user_then_assistant():
    interaction = capture_agent_run()
    assert [m.role for m in interaction.messages] == ["user", "assistant"]


def test_structured_output_is_preserved():
    # The typed answer object is kept verbatim under extra["structured_output"].
    interaction = capture_agent_run()
    assert interaction.extra["structured_output"] == {
        "answer": "The Moon is about 384,400 km from Earth.",
        "confidence": 0.99,
    }


def test_agent_metadata_is_captured():
    interaction = capture_agent_run()
    assert interaction.agent["system_prompt"].startswith("You are a helpful assistant")
    assert interaction.agent["input_schema"] == "QuestionInput"
    assert interaction.agent["output_schema"] == "AnswerOutput"


def test_chat_history_keeps_all_turns_in_order():
    interaction = capture_chat_history()
    roles = [m.role for m in interaction.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]


def test_log_event_records_span_and_io():
    interaction = capture_log_event()
    assert [m.content for m in interaction.messages] == ["What is 2 + 2?", "4."]
    assert interaction.spans[0]["kind"] == "atomic_agents.event"


# --------------------------------------------------------------------------- #
# Conformance — every surface, formatted into every format, must be valid.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("surface,capture", SURFACES, ids=[name for name, _ in SURFACES])
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_surface_conforms_to_every_format(surface, capture, fmt):
    record = Formatter(fmt).format_single(capture())
    issues = validate_record(record, fmt)
    assert issues == [], f"{surface} ({fmt}) is non-conformant: {issues}"


# --------------------------------------------------------------------------- #
# Script mode — write out_atomic/ for inspection (deterministic; no key needed).
# --------------------------------------------------------------------------- #
def _run_as_script():
    from agentscribe.storage import write_jsonl

    OUT.mkdir(exist_ok=True)
    for surface, capture in SURFACES:
        interaction = capture()
        for fmt in FORMATS:
            record = Formatter(fmt).format_single(interaction)
            write_jsonl(OUT / f"{surface}_{fmt}.jsonl", [record], mode="a")
            issues = validate_record(record, fmt)
            tagged = "OK" if not issues else "ISSUES"
            print(f"  [{tagged}] {surface} ({fmt})" + (f": {'; '.join(issues)}" if issues else ""))

    print("\nLocal outputs:")
    for p in sorted(OUT.glob("*.jsonl")):
        print(f"  {p.name}: {sum(1 for _ in p.open())} record(s)")


if __name__ == "__main__":
    _run_as_script()
