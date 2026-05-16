"""Adapter dispatch helpers for CLI and post-hoc conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import click

from agentscribe.core.canonical import CanonicalInteraction


AdapterRecord = Mapping[str, Any]
AdapterLoader = Callable[[AdapterRecord], list[CanonicalInteraction]]


def _langgraph_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..langgraph import from_state, from_stream_events

    if "events" in record or "stream" in record:
        return [from_stream_events(record.get("events", record.get("stream", [])), config=record.get("config"), metadata=record.get("metadata"))]
    return [from_state(record.get("state", record), config=record.get("config"), metadata=record.get("metadata"))]


def _crewai_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..crewai import from_event, from_kickoff_output, from_llm_call_context, from_tool_call_context

    if "event" in record:
        return [from_event(record.get("source"), record["event"], metadata=record.get("metadata"))]
    if "tool_name" in record or "tool_input" in record or "tool_result" in record:
        return [from_tool_call_context(record, metadata=record.get("metadata"))]
    if "kickoff_output" in record:
        return [from_kickoff_output(record["kickoff_output"], metadata=record.get("metadata"))]
    return [from_llm_call_context(record, metadata=record.get("metadata"))]


def _agno_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..agno import from_run_output, from_session, from_trace

    if "spans" in record or "trace" in record:
        return [from_trace(record.get("trace", record), metadata=record.get("metadata"))]
    if "runs" in record or "session_runs" in record:
        return from_session(record, metadata=record.get("metadata"))
    return [from_run_output(record, metadata=record.get("metadata"))]


def _autogen_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..autogen import from_chat_history, from_stream_events, from_task_result

    if "events" in record or "stream" in record:
        return [from_stream_events(record.get("events", record.get("stream", [])), metadata=record.get("metadata"))]
    if "chat_history" in record:
        return [from_chat_history(record, metadata=record.get("metadata"))]
    return [from_task_result(record, metadata=record.get("metadata"))]


def _atomic_agents_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..atomic_agents import from_agent_response, from_chat_history, from_log_event

    if "history" in record or "messages" in record:
        return [from_chat_history(record, metadata=record.get("metadata"))]
    if "event" in record or "event_type" in record:
        return [from_log_event(record, metadata=record.get("metadata"))]
    return [from_agent_response(record.get("response", record.get("output", record)), prompt=record.get("input", record.get("prompt")), metadata=record.get("metadata"))]


def _agentops_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..agentops import from_trace

    return [from_trace(record, metadata=record.get("metadata"))]


def _mlflow_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..mlflow import from_trace

    return [from_trace(record, metadata=record.get("metadata"))]


def _opentelemetry_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..opentelemetry import from_trace

    return [from_trace(record, source_framework="opentelemetry", metadata=record.get("metadata"))]


def _openinference_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..opentelemetry import from_trace

    return [from_trace(record, source_framework="openinference", metadata=record.get("metadata"))]


def _mcp_from_record(record: AdapterRecord) -> list[CanonicalInteraction]:
    from ..mcp import from_jsonrpc_messages, from_jsonrpc_pair

    if "messages" in record:
        return from_jsonrpc_messages(record["messages"], metadata=record.get("metadata"))
    if "request" in record:
        return [from_jsonrpc_pair(record["request"], record.get("response"), metadata=record.get("metadata"))]
    raise click.ClickException("MCP records must contain `messages` or `request`")


ADAPTER_RECORD_LOADERS: dict[str, AdapterLoader] = {
    "langgraph": _langgraph_from_record,
    "crewai": _crewai_from_record,
    "agno": _agno_from_record,
    "autogen": _autogen_from_record,
    "ag2": _autogen_from_record,
    "atomic": _atomic_agents_from_record,
    "atomic_agents": _atomic_agents_from_record,
    "agentops": _agentops_from_record,
    "mlflow": _mlflow_from_record,
    "opentelemetry": _opentelemetry_from_record,
    "openinference": _openinference_from_record,
    "mcp": _mcp_from_record,
}


def adapter_record_to_interactions(record: AdapterRecord, *, source: str) -> list[CanonicalInteraction]:
    """Dispatch one adapter record to the registered converter for that source."""

    loader = ADAPTER_RECORD_LOADERS.get(source)
    if loader is None:
        raise click.ClickException(f"No adapter parser is registered for `{source}`")
    return loader(record)


__all__ = ["ADAPTER_RECORD_LOADERS", "adapter_record_to_interactions"]