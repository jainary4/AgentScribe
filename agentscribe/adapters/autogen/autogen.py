"""AutoGen and AG2 adapter helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentscribe.core.canonical import CanonicalInteraction, CanonicalMessage

from ..utils import (
	append_unique_message,
	as_list,
	compact_dict,
	function_call_to_tool_call,
	get_value,
	interaction_from_messages,
	message_to_canonical,
	object_to_dict,
	parse_jsonish,
	tool_call_message,
	tool_response_message,
)


def _autogen_type(item: Any) -> str:
	return str(get_value(item, "type", "event_type", default=item.__class__.__name__))


def _usage(item: Any) -> dict[str, Any]:
	return object_to_dict(get_value(item, "models_usage", "usage", "token_usage", default={}))


def messages_from_autogen_item(item: Any, *, preserve_raw: bool = False) -> list[CanonicalMessage]:
	"""Normalize one AutoGen AgentChat message/event into canonical messages."""

	item_type = _autogen_type(item)
	lower_type = item_type.lower()
	content = get_value(item, "content", default=None)
	metadata = compact_dict(
		{
			"autogen_type": item_type,
			"source": get_value(item, "source", default=None),
			"metadata": object_to_dict(get_value(item, "metadata", default={})),
		}
	)

	if "toolcallrequest" in lower_type or "tool_call_request" in lower_type:
		calls = as_list(content or get_value(item, "tool_calls", "function_calls", default=[]))
		messages: list[CanonicalMessage] = []
		for call in calls:
			call_id = get_value(call, "id", "call_id", default=None)
			name = get_value(call, "name", "tool_name", default=None)
			arguments = parse_jsonish(get_value(call, "arguments", "args", default={}))
			message = tool_call_message(
				str(name) if name is not None else None,
				arguments,
				tool_call_id=str(call_id) if call_id is not None else None,
				metadata=metadata,
			)
			message.tool_calls = [function_call_to_tool_call(call)]
			messages.append(message)
		return messages or [message_to_canonical(item, default_role="tool_call", metadata=metadata, preserve_raw=preserve_raw)]

	if "toolcallexecution" in lower_type or "tool_call_execution" in lower_type:
		results = as_list(content or get_value(item, "results", "tool_results", default=[]))
		messages = []
		for result in results:
			call_id = get_value(result, "call_id", "id", "tool_call_id", default=None)
			name = get_value(result, "name", "tool_name", default=None)
			result_content = get_value(result, "content", "result", "output", default=result)
			result_metadata = {**metadata, "is_error": get_value(result, "is_error", default=None)}
			messages.append(
				tool_response_message(
					str(name) if name is not None else None,
					result_content,
					tool_call_id=str(call_id) if call_id is not None else None,
					metadata=result_metadata,
				)
			)
		return messages or [message_to_canonical(item, default_role="tool_response", metadata=metadata, preserve_raw=preserve_raw)]

	default_role = "user" if str(get_value(item, "source", default="")).lower() == "user" else "assistant"
	message = message_to_canonical(item, default_role=default_role, metadata=metadata, preserve_raw=preserve_raw)
	usage = _usage(item)
	if usage:
		message.token_usage = usage
	if "streamingchunk" in lower_type or "stream_chunk" in lower_type:
		message.metadata["stream_chunk"] = True
	return [message]


def from_task_result(
	result: Any,
	*,
	agent: Any = None,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize AutoGen AgentChat TaskResult or compatible mapping."""

	raw_messages = get_value(result, "messages", "chat_history", default=[])
	interaction = CanonicalInteraction(
		source_framework="autogen",
		metadata={
			"result_type": result.__class__.__name__,
			"stop_reason": get_value(result, "stop_reason", default=None),
			**dict(metadata or {}),
		},
	)
	interaction.agent = compact_dict(
		{
			"name": get_value(agent, "name", default=None),
			"description": get_value(agent, "description", default=None),
			"system_message": get_value(agent, "system_message", default=None),
		}
	)
	for item in raw_messages:
		for message in messages_from_autogen_item(item, preserve_raw=preserve_raw):
			append_unique_message(interaction, message)
	token_usage = object_to_dict(get_value(result, "models_usage", "usage", "token_usage", default={}))
	if token_usage:
		interaction.token_usage = token_usage
	if preserve_raw:
		interaction.extra["raw_result"] = object_to_dict(result) or str(result)
	return interaction


def from_chat_history(
	chat_history: Any,
	*,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize legacy AG2/autogen chat_history records."""

	messages = get_value(chat_history, "chat_history", "messages", default=chat_history)
	interaction = interaction_from_messages(
		as_list(messages),
		source_framework="autogen",
		metadata={"source_shape": "chat_history", **dict(metadata or {})},
	)
	if preserve_raw:
		interaction.extra["raw_chat_history"] = object_to_dict(chat_history) or chat_history
	return interaction


def from_stream_events(
	events: Iterable[Any],
	*,
	metadata: Mapping[str, Any] | None = None,
	preserve_raw: bool = False,
) -> CanonicalInteraction:
	"""Normalize events yielded by AssistantAgent.run_stream()."""

	interaction = CanonicalInteraction(source_framework="autogen", metadata={"source_shape": "run_stream", **dict(metadata or {})})
	for event in events:
		if get_value(event, "messages", default=None) is not None:
			final_result = from_task_result(event, metadata=metadata, preserve_raw=preserve_raw)
			for message in final_result.messages:
				append_unique_message(interaction, message)
			interaction.spans.extend(final_result.spans)
			continue
		interaction.spans.append({"kind": "autogen.stream", "event_type": _autogen_type(event), "event": object_to_dict(event) or str(event)})
		for message in messages_from_autogen_item(event, preserve_raw=preserve_raw):
			append_unique_message(interaction, message)
	return interaction


__all__ = [
	"from_chat_history",
	"from_stream_events",
	"from_task_result",
	"messages_from_autogen_item",
]
