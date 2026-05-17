from __future__ import annotations

from agentscribe.adapters.agentops import from_events, from_trace


def test_from_trace_uses_nested_spans_and_merges_trace_metadata() -> None:
    trace = {
        "id": "trace-1",
        "metadata": {"tenant": "acme"},
        "data": {
            "spans": [
                {
                    "name": "llm",
                    "trace_id": "inner-trace",
                    "input.value": "hello",
                    "output.value": "world",
                }
            ]
        },
    }

    interaction = from_trace(trace, metadata={"source": "unit"})

    assert interaction.source_framework == "agentops"
    assert interaction.trace_id == "trace-1"
    assert interaction.metadata["source_shape"] == "trace"
    assert interaction.metadata["tenant"] == "acme"
    assert [message.role for message in interaction.messages] == ["user", "assistant"]


def test_from_trace_falls_back_to_trace_as_single_span_for_minimal_payload() -> None:
    interaction = from_trace({"input.value": "prompt"})

    assert len(interaction.spans) == 1
    assert interaction.messages[0].content == "prompt"


def test_from_events_adds_event_spans_without_duplicate_event_payloads() -> None:
    events = [{"event": "task.started", "input.value": "go"}, {"event": "task.done", "output.value": "done"}]

    interaction = from_events(events)

    assert interaction.metadata["source_shape"] == "events"
    assert [message.content for message in interaction.messages] == ["go", "done"]
    assert any(span.get("kind") == "agentops.event" for span in interaction.spans)


def test_from_events_accepts_empty_iterable_boundary() -> None:
    interaction = from_events([])

    assert interaction.source_framework == "agentops"
    assert interaction.messages == []
    assert interaction.spans == []
