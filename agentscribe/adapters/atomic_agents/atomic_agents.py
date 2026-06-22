"""Atomic Agents adapter helpers."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from typing import Any

from agentscribe.adapters.base import BaseAdapter
from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

from ..utils import (
    InteractionCollector,
    as_list,
    build_metadata,
    compact_dict,
    get_nested,
    get_value,
    json_ready,
    object_to_dict,
    parse_jsonish,
    serialize_object_list,
)

_logger = logging.getLogger("agentscribe.atomic_agents")

# Known natural-language field names, specific (Atomic's own) -> generic.
_TEXT_FIELDS = (
    "chat_message", "message", "content", "text",
    "response", "answer", "final_answer", "reply", "output", "result",
)


# --------------------------------------------------------------------------- #
# Text / tool helpers
# --------------------------------------------------------------------------- #
def _extract_text(obj: Any) -> str:
    """Pull the natural-language text out of an Atomic schema or message content."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        parsed = parse_jsonish(obj)
        if not isinstance(parsed, Mapping):
            return obj
        obj = parsed
    mapping = dict(obj) if isinstance(obj, Mapping) else object_to_dict(obj)
    if not mapping:
        return str(obj)
    for key in _TEXT_FIELDS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value
    str_values = [v for v in mapping.values() if isinstance(v, str) and v.strip()]
    if len(str_values) == 1:
        return str_values[0]
    return json.dumps(mapping, ensure_ascii=False, default=str)

def _auto_tool_schemas(tool_calls: Any) -> list[dict[str, Any]]:
    """Synthesize minimal valid OpenAI function schemas from the executed calls.

    Atomic holds no tool registry, so when the developer doesn't pass explicit
    `tool_schemas` we build one schema per *unique* tool name (string-typed params
    inferred from the args keys). Enough to satisfy OpenAI fine-tuning validation,
    which requires a `tools` array on every line containing tool calls."""
    schemas: dict[str, dict[str, Any]] = {}
    for tc in tool_calls or []:
        if isinstance(tc, Mapping):
            name, args = tc.get("name"), tc.get("args")
        else:
            name, args, *_ = tc
        if not name or name in schemas:          # one schema per tool, not per call
            continue
        arg_map = _clean_tool_args(args)
        schemas[name] = {
            "type": "function",
            "function": {
                "name": str(name),
                "description": f"Auto-generated schema for the '{name}' tool.",
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in arg_map},
                    "required": list(arg_map.keys()),
                },
            },
        }
    return list(schemas.values())


def _clean_tool_args(args: Any) -> dict[str, Any]:
    parsed = parse_jsonish(args)
    if isinstance(parsed, Mapping):
        return {str(k): json_ready(v) for k, v in parsed.items()}
    return {} if parsed is None else {"value": json_ready(parsed)}


def _schema_name(schema: Any) -> str | None:
    if schema is None:
        return None
    return str(get_value(schema, "__name__", default=schema.__class__.__name__))


def _agent_metadata(agent: Any) -> dict[str, Any]:
    config = get_value(agent, "config", default=agent)
    return build_metadata(
        config,
        fields={
            "agent_class": lambda _v: agent.__class__.__name__ if agent is not None else None,
            "system_prompt": ("system_prompt", "system_message"),
            "model": lambda v: get_nested(v, "client", "model", default=get_value(v, "model", default=None)),
            "input_schema": lambda v: _schema_name(get_value(v, "input_schema", default=None)),
            "output_schema": lambda v: _schema_name(get_value(v, "output_schema", default=None)),
            "tools": lambda v: serialize_object_list(get_value(v, "tools", default=[]) or []),
            "context_providers": lambda v: serialize_object_list(get_value(v, "context_providers", default=[]) or []),
        },
    )


def _history_messages(history: Any) -> list[Any]:
    if history is None:
        return []
    if hasattr(history, "get_history"):              # Atomic ChatHistory
        try:
            return list(history.get_history())
        except Exception:
            pass
    return as_list(get_value(history, "messages", "history", default=history))


# --------------------------------------------------------------------------- #
# Converters (public API)
# --------------------------------------------------------------------------- #
def from_chat_history(history: Any, *, agent: Any = None, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize an Atomic ChatHistory (or message list) into a clean conversation.

    Tool results injected via ``agent.add_tool_result(...)`` appear under their
    stored role (user/system), because Atomic has no dedicated tool role."""
    interaction = CanonicalInteraction(
        source_framework="atomic_agents",
        metadata={"source_shape": "chat_history", **dict(metadata or {})},
    )
    for msg in _history_messages(history):
        role = get_value(msg, "role", default="user")
        text = _extract_text(get_value(msg, "content", default=msg))
        if text:
            interaction.add_message(role, text)
    interaction.agent = _agent_metadata(agent)
    return interaction


def from_agent_response(
    response: Any, *, prompt: Any = None, history: Any = None,
    agent: Any = None, metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Normalize an AtomicAgent run/run_async response (chat / structured agents).

    Pass ``history=agent.history`` to capture the full conversation; otherwise a
    single prompt -> response pair."""
    if history is not None:
        interaction = from_chat_history(history, agent=agent,
                                        metadata={"source_shape": "agent_response", **dict(metadata or {})})
        final_text = _extract_text(response)
        if final_text and (not interaction.messages or interaction.messages[-1].content != final_text):
            interaction.add_message("assistant", final_text)
    else:
        interaction = CanonicalInteraction(
            source_framework="atomic_agents",
            metadata={"source_shape": "agent_response", **dict(metadata or {})},
        )
        if prompt is not None:
            interaction.add_message("user", _extract_text(prompt))
        interaction.add_message("assistant", _extract_text(response))

    response_mapping = object_to_dict(response)
    if response_mapping:
        interaction.extra["structured_output"] = response_mapping
    interaction.agent = _agent_metadata(agent)
    interaction.instantiation = compact_dict({"agent": interaction.agent})
    return interaction


def from_agent_run(
    input_schema: Any, output_schema: Any, *, history: Any = None,
    agent: Any = None, metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Normalize a complete AtomicAgent input/output pair."""
    return from_agent_response(output_schema, prompt=input_schema, history=history, agent=agent, metadata=metadata)


def from_tool_interaction(
    user_message: Any, *, tool_calls: Any, final_answer: Any = None,
    system: Any = None, agent: Any = None, tool_schemas: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
    """Build ONE multi-format tool-use interaction from explicit pieces.

    Atomic Agents has no structured tool-call representation, so the developer
    (who executed the tools) declares them here. The result formats correctly to
    openai_chat (tool_calls), sharegpt (function_call/observation), AND alpaca/text.

    Parameters
    ----------
    user_message : the user's question (schema or text)
    tool_calls   : list of {"name","args","result"} dicts (or (name, args, result) tuples)
    final_answer : the agent's final answer (schema or text), optional
    system       : the system prompt, optional
    """
    interaction = CanonicalInteraction(
        source_framework="atomic_agents",
        metadata={"source_shape": "tool_interaction", **dict(metadata or {})},
    )
    if system:
        interaction.messages.append(CanonicalMessage(role="system", content=_extract_text(system)))
    interaction.messages.append(CanonicalMessage(role="user", content=_extract_text(user_message)))
    for tc in tool_calls or []:
        if isinstance(tc, Mapping):
            name, args, result = tc.get("name"), tc.get("args"), tc.get("result")
        else:
            name, args, result = tc
        cid = f"call_{uuid.uuid4().hex[:8]}"
        result_text = result if isinstance(result, str) else _extract_text(result)
        interaction.messages.append(CanonicalMessage(
            role="tool_call", content="",
            tool_name=str(name) if name is not None else None,
            tool_args=_clean_tool_args(args), tool_call_id=cid))
        interaction.messages.append(CanonicalMessage(
            role="tool_response", content=str(result_text),
            tool_name=str(name) if name is not None else None,
            tool_result=str(result_text), tool_call_id=cid))
    if final_answer is not None:
        interaction.messages.append(CanonicalMessage(role="assistant", content=_extract_text(final_answer)))

    schemas = list(tool_schemas) if tool_schemas else _auto_tool_schemas(tool_calls)
    if schemas:
        interaction.metadata["tool_schemas"] = schemas

    interaction.agent = _agent_metadata(agent)
    return interaction


def from_log_event(event: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
    """Normalize hook/logging/monitoring payloads emitted around AtomicAgent runs."""
    event_type = str(get_value(event, "type", "event", "event_type", default=event.__class__.__name__))
    interaction = CanonicalInteraction(
        source_framework="atomic_agents",
        metadata={"source_shape": "log_event", "event_type": event_type, **dict(metadata or {})},
    )
    interaction.spans.append({"kind": "atomic_agents.event", "event_type": event_type,
                              "event": object_to_dict(event) or str(event)})
    input_value = get_value(event, "input", "prompt", "input_schema", default=None)
    output_value = get_value(event, "output", "response", "output_schema", default=None)
    if input_value is not None:
        interaction.add_message("user", _extract_text(input_value))
    if output_value is not None:
        interaction.add_message("assistant", _extract_text(output_value))
    return interaction


# --------------------------------------------------------------------------- #
# Collector + live adapter
# --------------------------------------------------------------------------- #
class AtomicAgentsTraceCollector(InteractionCollector):
    """In-memory collector for Atomic Agents (manual, batch)."""

    def __init__(self, *, format_name: str = "openai_chat", output_path: str | None = None) -> None:
        super().__init__(source_framework="atomic_agents", format_name=format_name, output_path=output_path)

    def record_response(self, response: Any, *, prompt: Any = None, history: Any = None, agent: Any = None) -> CanonicalInteraction:
        return self.record(from_agent_response(response, prompt=prompt, history=history, agent=agent))

    def record_history(self, agent: Any) -> CanonicalInteraction:
        return self.record(from_chat_history(get_value(agent, "history", default=agent), agent=agent))

    def record_tool_interaction(self, user_message, *, tool_calls, final_answer=None,system=None, agent=None, tool_schemas=None):
        return self.record(from_tool_interaction(user_message, tool_calls=tool_calls, final_answer=final_answer,system=system, agent=agent, tool_schemas=tool_schemas))

    def on_log_event(self, event: Any) -> CanonicalInteraction:
        return self.record(from_log_event(event))


class AtomicAgentsAdapter(BaseAdapter):
    """Live capture via Atomic's completion hook. ``attach(agent)`` once, then flush()."""

    def attach(self, agent: Any) -> Any:
        try:
            agent.register_hook("completion:response", lambda *a, **k: self._snapshot(agent))
        except Exception as exc:
            _logger.error("AtomicAgentsAdapter.attach failed: %s", exc)
        return agent

    def capture(self, agent: Any) -> None:
        self._snapshot(agent)

    def _snapshot(self, agent: Any) -> None:
        with self._lock:
            self._pending[id(agent)] = from_chat_history(get_value(agent, "history", default=agent), agent=agent)

    def flush(self) -> int:
        with self._lock:
            for key in list(self._pending.keys()):
                self._buffer.append(self._pending.pop(key))
            return self._flush_buffer()


__all__ = [
    "AtomicAgentsAdapter",
    "AtomicAgentsTraceCollector",
    "from_agent_response",
    "from_agent_run",
    "from_chat_history",
    "from_log_event",
    "from_tool_interaction",
]