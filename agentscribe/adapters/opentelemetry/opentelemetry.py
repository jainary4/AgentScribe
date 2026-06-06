"""Generic OpenTelemetry and OpenInference trace adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

from ..utils import (
    append_unique_message,
    as_list,
    compact_dict,
    get_value,
    json_ready,
    message_to_canonical,
    object_to_dict,
    parse_jsonish,
    tool_call_message,
    tool_response_message,
)

MESSAGE_ATTR_RE = re.compile(r"(?:llm\.)?(input|output)_messages\.(\d+)\.message\.(role|content|name)$")


def span_attributes(span: Any) -> dict[str, Any]:
    """Return merged span attributes from dict-like or SDK-like span objects."""

    span_mapping = object_to_dict(span)
    attributes = object_to_dict(get_value(span, "attributes", default={}))
    for key, value in span_mapping.items():
        if key not in {"attributes", "events", "links", "resource"} and key not in attributes:
            attributes[key] = value
    return attributes


def _attr(attributes: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in attributes:
            return attributes[name]
    return default


def _span_name(span: Any, attributes: Mapping[str, Any]) -> str | None:
    return get_value(span, "name", default=_attr(attributes, "name", "span.name", default=None))


def _ids(span: Any, attributes: Mapping[str, Any]) -> dict[str, str]:
    return compact_dict(
        {
            "trace_id": get_value(span, "trace_id", "traceId", default=_attr(attributes, "trace_id", "traceId", default=None)),
            "span_id": get_value(span, "span_id", "spanId", default=_attr(attributes, "span_id", "spanId", default=None)),
        }
    )


def _message_from_value(value: Any, *, default_role: str) -> list[CanonicalMessage]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        return [message_to_canonical(item, default_role=default_role) for item in parsed]
    if isinstance(parsed, Mapping):
        if "messages" in parsed:
            return [message_to_canonical(item, default_role=default_role) for item in as_list(parsed["messages"])]
        if "role" in parsed or "content" in parsed:          # <-- NEW: a single message dict
            return [message_to_canonical(parsed, default_role=default_role)]
    return [message_to_canonical({"role": default_role, "content": parsed})]


def _messages_from_flattened_attrs(attributes: Mapping[str, Any]) -> list[CanonicalMessage]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for key, value in attributes.items():
        match = MESSAGE_ATTR_RE.search(str(key))
        if not match:
            continue
        direction, index, field = match.groups()
        grouped.setdefault((direction, int(index)), {})[field] = value

    messages = []
    for (_direction, _index), payload in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        messages.append(message_to_canonical(payload))
    return messages


def messages_from_span(span: Any) -> list[CanonicalMessage]:
    """Extract prompt/completion messages from GenAI/OpenInference span attrs."""

    attributes = span_attributes(span)
    messages = _messages_from_flattened_attrs(attributes)
    input_value = _attr(attributes, "input.value", "gen_ai.prompt", "prompt", "input", default=None)
    output_value = _attr(attributes, "output.value", "gen_ai.completion", "completion", "output", default=None)
    if input_value is not None:
        messages.extend(_message_from_value(input_value, default_role="user"))
    if output_value is not None:
        messages.extend(_message_from_value(output_value, default_role="assistant"))
    return messages


def _token_usage(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return compact_dict(
        {
            "prompt_tokens": _attr(attributes, "gen_ai.usage.input_tokens", "llm.token_count.prompt", "input_tokens", default=None),
            "completion_tokens": _attr(
                attributes,
                "gen_ai.usage.output_tokens",
                "llm.token_count.completion",
                "output_tokens",
                default=None,
            ),
            "total_tokens": _attr(attributes, "gen_ai.usage.total_tokens", "llm.token_count.total", "total_tokens", default=None),
        }
    )


def _tool_messages_from_span(span: Any, attributes: Mapping[str, Any]) -> list[CanonicalMessage]:
    span_kind = str(_attr(attributes, "openinference.span.kind", "span.kind", default="")).lower()
    tool_name = _attr(attributes, "gen_ai.tool.name", "tool.name", "mcp.tool.name", default=None)
    if "tool" not in span_kind and tool_name is None:
        return []
    call_id = _attr(attributes, "gen_ai.tool.call.id", "tool.call.id", "tool_call_id", default=None)
    input_value = _attr(attributes, "input.value", "tool.input", "gen_ai.tool.input", default=None)
    output_value = _attr(attributes, "output.value", "tool.output", "gen_ai.tool.output", default=None)
    metadata = {"span_name": _span_name(span, attributes), "span_kind": span_kind}
    messages: list[CanonicalMessage] = []
    if input_value is not None:
        messages.append(tool_call_message(str(tool_name) if tool_name is not None else None, parse_jsonish(input_value), tool_call_id=str(call_id) if call_id else None, metadata=metadata))
    if output_value is not None:
        messages.append(tool_response_message(str(tool_name) if tool_name is not None else None, parse_jsonish(output_value), tool_call_id=str(call_id) if call_id else None, metadata=metadata))
    return messages


def from_spans(
    spans: Iterable[Any],
    *,
    source_framework: str = "opentelemetry",
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Normalize OpenTelemetry/OpenInference spans into one interaction."""

    interaction = CanonicalInteraction(source_framework=source_framework, metadata=dict(metadata or {}))
    for span in spans:
        attributes = span_attributes(span)
        ids = _ids(span, attributes)
        if ids.get("trace_id") and not interaction.trace_id:
            interaction.trace_id = str(ids["trace_id"])
        span_name = _span_name(span, attributes)
        interaction.spans.append(
            {
                "kind": "otel.span",
                "name": span_name,
                "ids": ids,
                "attributes": json_ready(attributes),
                "events": json_ready(get_value(span, "events", default=[])),
            }
        )
        for message in messages_from_span(span):
            append_unique_message(interaction, message)
        for message in _tool_messages_from_span(span, attributes):
            append_unique_message(interaction, message)
        token_usage = _token_usage(attributes)
        if token_usage:
            interaction.token_usage.update(token_usage)
        model = _attr(attributes, "gen_ai.request.model", "gen_ai.response.model", "llm.model_name", "model", default=None)
        if model and not interaction.model:
            interaction.model = str(model)
        provider = _attr(attributes, "gen_ai.system", "llm.provider", "provider", default=None)
        if provider and not interaction.provider:
            interaction.provider = str(provider)
    return interaction


def from_trace(trace: Any, *, source_framework: str = "opentelemetry", metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize a trace object or mapping that contains spans."""

    spans = get_value(trace, "spans", "children", default=None)
    if spans is None:
        spans = [trace]
    interaction = from_spans(as_list(spans), source_framework=source_framework, metadata=metadata)
    trace_id = get_value(trace, "trace_id", "traceId", default=None)
    if trace_id is not None:
        interaction.trace_id = str(trace_id)
    return interaction


__all__ = [
    "from_spans",
    "from_trace",
    "messages_from_span",
    "span_attributes",
]
