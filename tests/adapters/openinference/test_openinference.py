from __future__ import annotations

from agentscribe.adapters.openinference import from_spans, from_trace, messages_from_span, span_attributes


def test_from_spans_defaults_to_openinference_source() -> None:
    interaction = from_spans([{"input.value": "Q"}])

    assert interaction.source_framework == "openinference"
    assert interaction.messages[0].content == "Q"


def test_from_trace_defaults_to_openinference_source_and_accepts_override() -> None:
    default = from_trace({"trace_id": "trace-1", "spans": [{"output.value": "A"}]})
    custom = from_trace({"spans": []}, source_framework="custom")

    assert default.source_framework == "openinference"
    assert default.trace_id == "trace-1"
    assert custom.source_framework == "custom"


def test_reexported_helpers_match_opentelemetry_behavior() -> None:
    span = {"attributes": {"input.value": "Q"}}

    assert span_attributes(span)["input.value"] == "Q"
    assert messages_from_span(span)[0].content == "Q"
