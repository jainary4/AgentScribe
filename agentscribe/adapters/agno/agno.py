"""Agno adapter for AgentScribe.

Captures agent interactions using Agno's native post‑hooks and tool hooks.
No additional dependencies beyond ``agno`` are required.

Usage (post‑hooks — recommended):
    from agentscribe.adapters.agno import AgnoAdapter

    adapter = AgnoAdapter(format="openai_chat", output="./data.jsonl")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools(stock_price=True)],
        post_hooks=[adapter.post_hook],   # captures full message history
        tool_hooks=[adapter.tool_hook],   # captures individual tool calls
    )
    agent.print_response("What is the stock price of Apple?")
    adapter.flush()  # optional — also auto‑flushed on garbage collection

Alternative: MLflow autolog (requires ``mlflow>=3.3``)
    import mlflow
    mlflow.agno.autolog()
    # AgentScribe's CLI can later convert MLflow traces to training data:
    #   agentscribe convert agno-mlflow ./mlruns/ --format openai_chat --output ./data.jsonl
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.core.formatter import Formatter

from ..utils import (
    InteractionCollector,
    append_unique_message,
    as_list,
    build_metadata,
    compact_dict,
    get_value,
    interaction_from_messages,
    json_ready,
    message_to_canonical,
    object_to_dict,
    resolve_identifier,
    tool_call_message,
    tool_response_message,
)

_logger = logging.getLogger("agentscribe.agno")


def _run_session_id(run_output: Any, *, fallback: Any = None) -> str | None:
    """Resolve an Agno session id from a run-like object."""

    return resolve_identifier(run_output, "session_id", "session") or (str(fallback) if fallback is not None else None)


def _run_metadata(run_output: Any, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build compact Agno run metadata."""

    return {
        "source_shape": "run_output",
        **build_metadata(
            run_output,
            fields={
                "agent_name": ("agent_name", "agent", "name"),
                "agent_id": ("agent_id",),
                "run_id": ("run_id", "id"),
                "user_id": ("user_id",),
                "team_id": ("team_id",),
            },
        ),
        **dict(metadata or {}),
    }


def _tool_execution_messages(tool: Any) -> list[Any]:
    """Convert an Agno ToolExecution-like object into canonical tool messages."""

    tool_name = get_value(tool, "tool_name", "name", "function_name", default=None)
    tool_args = get_value(tool, "tool_args", "tool_input", "arguments", "args", "input", default=None)
    tool_result = get_value(tool, "tool_result", "result", "output", "content", default=None)
    tool_call_id = get_value(tool, "tool_call_id", "call_id", "id", default=None)
    metadata = compact_dict(
        {
            "tool_class": tool.__class__.__name__,
            "duration_ms": get_value(tool, "duration_ms", "duration", default=None),
            "status": get_value(tool, "status", default=None),
            "error": get_value(tool, "error", "exception", default=None),
        }
    )

    messages: list[Any] = []
    if tool_args is not None:
        messages.append(
            tool_call_message(
                str(tool_name) if tool_name is not None else None,
                tool_args,
                tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
                metadata=metadata,
            )
        )
    if tool_result is not None:
        messages.append(
            tool_response_message(
                str(tool_name) if tool_name is not None else None,
                tool_result,
                tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
                metadata=metadata,
            )
        )
    return messages


def _run_tools(run_output: Any) -> list[Any]:
    """Return tool executions from the common Agno run-output locations."""

    tools = get_value(run_output, "tools", "tool_calls", "tool_executions", default=[])
    return as_list(tools)


def _run_metrics(run_output: Any) -> dict[str, Any]:
    """Return JSON-ready Agno metrics or usage."""

    metrics = object_to_dict(get_value(run_output, "metrics", "token_usage", "usage", default={}))
    return compact_dict(metrics)


def _run_model(run_output: Any) -> str | None:
    """Resolve the model name from an Agno run output."""

    model = get_value(run_output, "model", "model_id", default=None)
    if model is None:
        model = get_value(get_value(run_output, "model_provider", default=None), "id", "name", default=None)
    return str(model) if model is not None else None


def from_run_output(run_output: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize an Agno ``RunOutput`` or compatible mapping.

    Parameters
    ----------
    run_output : Any
        Agno run output object, JSON export, or compatible mapping.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into the canonical interaction.

    Returns
    -------
    CanonicalInteraction
        Interaction containing messages, tool executions, metrics, and run
        provenance.
    """

    messages = get_value(run_output, "messages", "chat_history", "conversation", default=[]) or []
    run_id = get_value(run_output, "run_id", "id", default=None)
    interaction = interaction_from_messages(
        as_list(messages),
        source_framework="agno",
        session_id=_run_session_id(run_output, fallback=run_id),
        run_id=str(run_id) if run_id is not None else None,
        metadata=_run_metadata(run_output, metadata=metadata),
    )

    content = get_value(run_output, "content", "response", "output", default=None)
    if content is not None and not interaction.messages:
        interaction.add_message("assistant", content)

    for tool in _run_tools(run_output):
        for message in _tool_execution_messages(tool):
            append_unique_message(interaction, message)

    model = _run_model(run_output)
    if model is not None:
        interaction.model = model
    token_usage = _run_metrics(run_output)
    if token_usage:
        interaction.token_usage = token_usage

    interaction.instantiation = compact_dict(
        {
            "agent": build_metadata(
                get_value(run_output, "agent", default=run_output),
                fields={
                    "name": ("agent_name", "name"),
                    "id": ("agent_id", "id"),
                    "model": ("model", "model_id"),
                },
            ),
            "run": {
                "run_id": run_id,
                "session_id": get_value(run_output, "session_id", default=None),
            },
        }
    )
    return interaction


def from_session(session: Any, *, metadata: Mapping[str, Any] | None = None) -> list[CanonicalInteraction]:
    """Normalize an Agno session export into one interaction per run.

    Parameters
    ----------
    session : Any
        Session object or mapping containing ``runs``/``session_runs`` or a
        message list.
    metadata : Mapping[str, Any] | None
        Optional metadata to merge into each interaction.

    Returns
    -------
    list[CanonicalInteraction]
        Normalized session interactions.
    """

    session_id = get_value(session, "session_id", "id", default=None)
    session_metadata = {
        "source_shape": "session",
        **build_metadata(
            session,
            fields={
                "agent_name": ("agent_name", "agent", "name"),
                "agent_id": ("agent_id",),
                "user_id": ("user_id",),
            },
        ),
        **dict(metadata or {}),
    }
    runs = get_value(session, "runs", "session_runs", default=None)
    if runs is None:
        return [from_run_output(session, metadata=session_metadata)]

    interactions: list[CanonicalInteraction] = []
    for run in as_list(runs):
        if isinstance(run, Mapping):
            run_payload = {**run}
            if session_id is not None:
                run_payload.setdefault("session_id", session_id)
            for key in ("agent_name", "agent_id", "user_id"):
                value = get_value(session, key, default=None)
                if value is not None:
                    run_payload.setdefault(key, value)
        else:
            run_payload = run
        interactions.append(from_run_output(run_payload, metadata=session_metadata))
    return interactions


def _event_type(event: Any) -> str:
    return str(get_value(event, "event", "event_type", "type", default=event.__class__.__name__))


def _messages_from_event(event: Any) -> list[Any]:
    messages = get_value(event, "messages", "chat_history", default=None)
    if messages is not None:
        return as_list(messages)
    message = get_value(event, "message", default=None)
    return as_list(message) if message is not None else []


def from_event_stream(
    events: Iterable[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    include_message_chunks: bool = False,
) -> CanonicalInteraction:
    """Normalize Agno event-stream payloads into one interaction.

    Event payloads are preserved as spans. Complete messages and terminal
    output are promoted into canonical messages.
    """

    interaction = CanonicalInteraction(
        source_framework="agno",
        metadata={"source_shape": "event_stream", **dict(metadata or {})},
    )
    for event in events:
        event_type = _event_type(event)
        lower_type = event_type.lower()
        interaction.spans.append(
            {
                "kind": "agno.event",
                "event_type": event_type,
                "event": json_ready(object_to_dict(event) or str(event)),
            }
        )
        session_id = get_value(event, "session_id", default=None)
        if session_id is not None and interaction.session_id is None:
            interaction.session_id = str(session_id)
        run_id = get_value(event, "run_id", "id", default=None)
        if run_id is not None and not interaction.run_id:
            interaction.run_id = str(run_id)

        for message in _messages_from_event(event):
            message_metadata = {"event_type": event_type}
            if "chunk" in lower_type:
                message_metadata["stream_chunk"] = True
            canonical = message_to_canonical(message, metadata=message_metadata)
            if include_message_chunks or not canonical.metadata.get("stream_chunk"):
                append_unique_message(interaction, canonical)

        if "tool" in lower_type:
            for message in _tool_execution_messages(event):
                append_unique_message(interaction, message)
        elif "complete" in lower_type or "end" in lower_type:
            content = get_value(event, "content", "response", "output", default=None)
            if content is not None:
                append_unique_message(interaction, message_to_canonical({"role": "assistant", "content": content}))
    return interaction


def from_trace(trace: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize Agno AgentOS/MLflow/OpenTelemetry-style traces."""

    from ..opentelemetry import from_trace as from_otel_trace

    trace_payload = get_value(trace, "trace", default=trace)
    interaction = from_otel_trace(trace_payload, source_framework="agno", metadata={"source_shape": "trace", **dict(metadata or {})})
    trace_id = get_value(trace_payload, "trace_id", "traceId", "id", default=None)
    if trace_id is not None:
        interaction.trace_id = str(trace_id)
    return interaction


class AgnoTraceCollector(InteractionCollector):
    """Collector for Agno run outputs, sessions, event streams, and traces."""

    def __init__(self, *, format_name: str = "openai_chat", output_path: str | None = None) -> None:
        super().__init__(source_framework="agno", format_name=format_name, output_path=output_path)

    def record_run_output(self, run_output: Any) -> CanonicalInteraction:
        return self.record(from_run_output(run_output))

    def record_session(self, session: Any) -> list[CanonicalInteraction]:
        interactions = from_session(session)
        self.extend(interactions)
        return interactions

    def record_event_stream(self, events: Iterable[Any]) -> CanonicalInteraction:
        return self.record(from_event_stream(events))

    def record_trace(self, trace: Any) -> CanonicalInteraction:
        return self.record(from_trace(trace))


class AgnoAdapter(BaseAdapter):
    """Capture Agno agent interactions using post hooks and optional tool hooks."""

    def __init__(
        self,
        format: str = "openai_chat",
        output: str = "./agentscribe_data.jsonl",
        flush_interval: int = 10,
    ) -> None:
        super().__init__(format=format, output=output, flush_interval=flush_interval)
        self._pending_tool_messages: list[Any] = []

    def post_hook(
        self,
        run_output: Any,
        agent: Any,
        session: Any = None,
        run_context: Any = None,
    ) -> None:
        """Agno post hook. Pass this to ``Agent(post_hooks=[...])``."""

        try:
            metadata = compact_dict(
                {
                    "agent": object_to_dict(agent) or str(agent) if agent is not None else None,
                    "session": object_to_dict(session) or str(session) if session is not None else None,
                    "run_context": object_to_dict(run_context) or str(run_context) if run_context is not None else None,
                }
            )
            interaction = from_run_output(run_output, metadata=metadata)
            if agent is not None and not interaction.metadata.get("agent_name"):
                agent_name = get_value(agent, "name", default=None)
                if agent_name is not None:
                    interaction.metadata["agent_name"] = str(agent_name)
            for message in self._pending_tool_messages:
                append_unique_message(interaction, message)
            self._pending_tool_messages.clear()
            self._finalise_one(interaction)
        except Exception as exc:
            _logger.error("Error in AgentScribe Agno post hook: %s", exc)

    def tool_hook(
        self,
        function_name: str,
        function_call: Any,
        arguments: dict[str, Any],
    ) -> Any:
        """Agno tool hook. Pass this to ``Agent(tool_hooks=[...])``."""

        start_time = time.time()
        try:
            result = function_call(**arguments)
        except Exception as exc:
            duration_ms = int((time.time() - start_time) * 1000)
            self._pending_tool_messages.append(
                tool_call_message(function_name, arguments)
            )
            self._pending_tool_messages.append(
                tool_response_message(
                    function_name,
                    {"error": str(exc)},
                    metadata={"duration_ms": duration_ms, "error": exc.__class__.__name__},
                )
            )
            _logger.error("Agno tool %r failed after %sms: %s", function_name, duration_ms, exc)
            raise

        duration_ms = int((time.time() - start_time) * 1000)
        self._pending_tool_messages.append(tool_call_message(function_name, arguments))
        self._pending_tool_messages.append(
            tool_response_message(function_name, result, metadata={"duration_ms": duration_ms})
        )
        return result

    def _finalise_one(self, interaction: CanonicalInteraction) -> None:
        """Buffer one completed Agno interaction and flush when configured."""

        with self._lock:
            self._buffer.append(interaction)
            if self._flush_interval <= 0 or len(self._buffer) >= self._flush_interval:
                self._flush_buffer()


def parse_agno_run_output(run_output: Any, format_name: str = "openai_chat") -> list[dict[str, Any]]:
    """Backward-compatible formatted parser for saved Agno run outputs.

    New code should prefer :func:`from_run_output`, which returns a
    ``CanonicalInteraction`` like the rest of the adapter packages.
    """

    formatter = Formatter(format=format_name)
    return [formatter.format_single(from_run_output(run_output))]


AgnoHookAdapter = AgnoAdapter


__all__ = [
    "AgnoAdapter",
    "AgnoHookAdapter",
    "AgnoTraceCollector",
    "from_event_stream",
    "from_run_output",
    "from_session",
    "from_trace",
    "parse_agno_run_output",
]
