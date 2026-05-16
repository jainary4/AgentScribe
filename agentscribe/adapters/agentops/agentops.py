"""AgentOps trace/export adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction

from ..utils import as_list, get_value, object_to_dict
from ..opentelemetry import from_spans


def from_trace(trace: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize an AgentOps trace/export payload."""

    spans = get_value(trace, "spans", "operations", "events", default=None)
    if spans is None:
        spans = get_value(get_value(trace, "data", default={}), "spans", default=[trace])
    interaction = from_spans(as_list(spans), source_framework="agentops", metadata={"source_shape": "trace", **dict(metadata or {})})
    trace_id = get_value(trace, "trace_id", "id", default=None)
    if trace_id is not None:
        interaction.trace_id = str(trace_id)
    metadata_payload = object_to_dict(get_value(trace, "metadata", default={}))
    if metadata_payload:
        interaction.metadata.update(metadata_payload)
    return interaction


def from_events(events: Iterable[Any], *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize AgentOps operation/task/tool event exports."""

    interaction = from_spans(events, source_framework="agentops", metadata={"source_shape": "events", **dict(metadata or {})})
    for event in events:
        event_payload = object_to_dict(event)
        if event_payload and not any(span.get("event") == event_payload for span in interaction.spans if isinstance(span, dict)):
            interaction.spans.append({"kind": "agentops.event", "event": event_payload})
    return interaction


__all__ = ["from_events", "from_trace"]