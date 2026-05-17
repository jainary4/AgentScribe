"""OpenInference compatibility adapter.

OpenInference traces are represented as OpenTelemetry spans with conventional
attributes, so the generic OpenTelemetry adapter performs the actual parsing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..opentelemetry import (
    from_spans as _from_spans,
    from_trace as _from_trace,
    messages_from_span,
    span_attributes,
)
from agentscribe.core.canonical import CanonicalInteraction


def from_spans(
    spans: Iterable[Any],
    *,
    source_framework: str = "openinference",
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Normalize OpenInference spans with the OpenInference source label."""

    return _from_spans(spans, source_framework=source_framework, metadata=metadata)


def from_trace(
    trace: Any,
    *,
    source_framework: str = "openinference",
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Normalize an OpenInference trace with the OpenInference source label."""

    return _from_trace(trace, source_framework=source_framework, metadata=metadata)


__all__ = ["from_spans", "from_trace", "messages_from_span", "span_attributes"]
