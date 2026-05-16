"""OpenInference adapter package."""

from .openinference import from_spans, from_trace, messages_from_span, span_attributes

__all__ = ["from_spans", "from_trace", "messages_from_span", "span_attributes"]
