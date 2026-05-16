"""Atomic Agents adapter helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction

from ..utils import (
	InteractionCollector,
	append_unique_message,
	as_list,
	build_metadata,
	compact_dict,
	get_nested,
	get_value,
	interaction_from_messages,
	message_to_canonical,
	object_to_dict,
	serialize_object_list,
)


def _schema_name(schema: Any) -> str | None:
	if schema is None:
		return None
	return str(get_value(schema, "__name__", default=schema.__class__.__name__))


def _agent_metadata(agent: Any) -> dict[str, Any]:
	config = get_value(agent, "config", default=agent)
	return build_metadata(
		config,
		fields={
			"agent_class": lambda _value: agent.__class__.__name__ if agent is not None else None,
			"system_prompt": ("system_prompt", "system_message"),
			"model": lambda value: get_nested(value, "client", "model", default=get_value(value, "model", default=None)),
			"input_schema": lambda value: _schema_name(get_value(value, "input_schema", default=None)),
			"output_schema": lambda value: _schema_name(get_value(value, "output_schema", default=None)),
			"tools": lambda value: serialize_object_list(get_value(value, "tools", default=[]) or []),
			"context_providers": lambda value: serialize_object_list(get_value(value, "context_providers", default=[]) or []),
		},
	)


def from_chat_history(
	history: Any,
	*,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Normalize Atomic Agents ChatHistory or message-list exports."""

	messages = get_value(history, "messages", "history", default=history)
	interaction = interaction_from_messages(
		as_list(messages),
		source_framework="atomic_agents",
		metadata={"source_shape": "chat_history", **dict(metadata or {})},
	)
	interaction.agent = _agent_metadata(agent)
	return interaction


def from_agent_response(
	response: Any,
	*,
	prompt: Any = None,
	history: Any = None,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Normalize an AtomicAgent run/run_async response."""

	if history is not None:
		interaction = from_chat_history(history, agent=agent, metadata={"source_shape": "agent_response", **dict(metadata or {})})
	else:
		interaction = CanonicalInteraction(source_framework="atomic_agents", metadata={"source_shape": "agent_response", **dict(metadata or {})})
		if prompt is not None:
			interaction.add_message("user", prompt)

	append_unique_message(
		interaction,
		message_to_canonical(
			{"role": "assistant", "content": response, "type": response.__class__.__name__},
			metadata={"structured_output": bool(object_to_dict(response))},
		),
	)
	response_mapping = object_to_dict(response)
	if response_mapping:
		interaction.extra["structured_output"] = response_mapping
	interaction.agent = _agent_metadata(agent)
	interaction.instantiation = compact_dict({"agent": interaction.agent})
	return interaction


def from_agent_run(
	input_schema: Any,
	output_schema: Any,
	*,
	history: Any = None,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
) -> CanonicalInteraction:
	"""Normalize a complete AtomicAgent input/output pair."""

	return from_agent_response(output_schema, prompt=input_schema, history=history, agent=agent, metadata=metadata)


def from_log_event(event: Any, *, metadata: Mapping[str, Any] | None = None) -> CanonicalInteraction:
	"""Normalize hook/logging/monitoring payloads emitted around AtomicAgent runs."""

	event_type = str(get_value(event, "type", "event", "event_type", default=event.__class__.__name__))
	interaction = CanonicalInteraction(source_framework="atomic_agents", metadata={"source_shape": "log_event", "event_type": event_type, **dict(metadata or {})})
	interaction.spans.append({"kind": "atomic_agents.event", "event_type": event_type, "event": object_to_dict(event) or str(event)})
	input_value = get_value(event, "input", "prompt", "input_schema", default=None)
	output_value = get_value(event, "output", "response", "output_schema", default=None)
	if input_value is not None:
		interaction.add_message("user", input_value)
	if output_value is not None:
		interaction.add_message("assistant", output_value)
	return interaction


class AtomicAgentsTraceCollector(InteractionCollector):
	"""Collector for Atomic Agents run hooks."""

	def __init__(self, *, format_name: str = "openai_chat", output_path: str | None = None) -> None:
		super().__init__(source_framework="atomic_agents", format_name=format_name, output_path=output_path)

	def record_response(self, response: Any, *, prompt: Any = None, history: Any = None, agent: Any = None) -> CanonicalInteraction:
		return self.record(from_agent_response(response, prompt=prompt, history=history, agent=agent))

	def on_log_event(self, event: Any) -> CanonicalInteraction:
		return self.record(from_log_event(event))


__all__ = [
	"AtomicAgentsTraceCollector",
	"from_agent_response",
	"from_agent_run",
	"from_chat_history",
	"from_log_event",
]
