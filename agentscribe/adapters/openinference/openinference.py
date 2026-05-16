"""OpenInference compatibility adapter.

OpenInference traces are represented as OpenTelemetry spans with conventional
attributes, so the generic OpenTelemetry adapter performs the actual parsing.
"""

from __future__ import annotations

from ..opentelemetry import from_spans, from_trace, messages_from_span, span_attributes


__all__ = ["from_spans", "from_trace", "messages_from_span", "span_attributes"]
