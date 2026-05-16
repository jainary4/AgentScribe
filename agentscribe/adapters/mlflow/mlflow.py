"""MLflow tracing adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction
from ..utils import as_list, get_nested, get_value, object_to_dict
from ..opentelemetry import from_spans


# ------------------------------------------------------------------
# Trace converters (MLflow-specific)
# ------------------------------------------------------------------
def from_trace(trace: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize MLflow GenAI trace objects or exported trace dictionaries.

    Parameters
    ----------
    trace : Any
        MLflow trace object, exported trace dictionary, or compatible mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction inferred from MLflow spans and trace metadata.

    Notes
    -----
    MLflow GenAI traces commonly expose span data under ``data.spans``.
    The adapter falls back to top-level ``spans`` or ``operations`` when needed.
    """

    spans = get_nested(trace, "data", "spans", default=None)
    if spans is None:
        spans = get_value(trace, "spans", "operations", default=[trace])
    interaction = from_spans(as_list(spans), source_framework="mlflow", metadata={"source_shape": "trace", **dict(metadata or {})})
    trace_id = get_nested(trace, "info", "trace_id", default=get_value(trace, "trace_id", "request_id", default=None))
    if trace_id is not None:
        interaction.trace_id = str(trace_id)
    request_metadata = object_to_dict(get_nested(trace, "info", "request_metadata", default={}))
    if request_metadata:
        interaction.metadata.update(request_metadata)
    return interaction


def from_trace_dict(trace: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize an MLflow JSON export dictionary.

    Parameters
    ----------
    trace : Mapping[str, Any]
        JSON-compatible MLflow trace export.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction inferred from the exported trace.
    """

    return from_trace(trace, metadata=metadata)


__all__ = ["from_trace", "from_trace_dict"]