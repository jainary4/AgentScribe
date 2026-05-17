from __future__ import annotations

from dataclasses import dataclass

from agentscribe.adapters.opentelemetry import from_spans, from_trace, messages_from_span, span_attributes


@dataclass
class SpanObject:
    name: str
    attributes: dict[str, object]
    trace_id: str = "trace-1"
    span_id: str = "span-1"
    events: list[dict[str, object]] | None = None


def test_span_attributes_merges_attributes_and_top_level_fields() -> None:
    attrs = span_attributes({"name": "span", "attributes": {"input.value": "Q"}, "trace_id": "trace"})

    assert attrs["input.value"] == "Q"
    assert attrs["name"] == "span"
    assert attrs["trace_id"] == "trace"
    assert "attributes" not in attrs


def test_messages_from_span_reads_flattened_jsonish_and_plain_values() -> None:
    span = {
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": "rules",
        "input.value": '[{"role":"user","content":"Q"}]',
        "output.value": '{"messages":[{"role":"assistant","content":"A"}]}',
    }

    messages = messages_from_span(span)

    assert [(message.role, message.content) for message in messages] == [
        ("system", "rules"),
        ("user", "Q"),
        ("assistant", "A"),
    ]


def test_from_spans_collects_messages_tool_spans_usage_model_provider_and_events() -> None:
    spans = [
        SpanObject(
            name="llm",
            attributes={
                "input.value": "Q",
                "output.value": "A",
                "gen_ai.usage.input_tokens": 1,
                "gen_ai.usage.output_tokens": 2,
                "gen_ai.request.model": "model-a",
                "gen_ai.system": "openai",
            },
            events=[{"name": "event"}],
        ),
        {
            "name": "tool",
            "span.kind": "tool",
            "tool.name": "lookup",
            "tool.call.id": "call-1",
            "tool.input": '{"q":"x"}',
            "tool.output": '{"answer":"y"}',
        },
    ]

    interaction = from_spans(spans, metadata={"case": "spans"})

    assert interaction.trace_id == "trace-1"
    assert interaction.metadata["case"] == "spans"
    assert interaction.model == "model-a"
    assert interaction.provider == "openai"
    assert interaction.token_usage == {"prompt_tokens": 1, "completion_tokens": 2}
    assert [message.role for message in interaction.messages] == ["user", "assistant", "tool_call", "tool_response"]
    assert interaction.messages[2].tool_args["q"] == "x"
    assert interaction.spans[0]["events"] == [{"name": "event"}]


def test_from_spans_accepts_empty_iterable_boundary() -> None:
    interaction = from_spans([])

    assert interaction.messages == []
    assert interaction.spans == []


def test_from_trace_uses_children_or_single_trace_fallback() -> None:
    child_trace = from_trace({"traceId": "trace-2", "children": [{"output.value": "child"}]})
    single_trace = from_trace({"trace_id": "trace-3", "input.value": "single"})

    assert child_trace.trace_id == "trace-2"
    assert child_trace.messages[0].content == "child"
    assert single_trace.trace_id == "trace-3"
    assert single_trace.messages[0].content == "single"
